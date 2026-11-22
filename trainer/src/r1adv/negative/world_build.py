"""Terrain bank -> mjlab terrain generator (runtime side; imports mjlab and r1wb.world, changes neither).

Columns of the bank are the generator's curriculum columns and its rows the levels (difficulty = row / 7), so every
(row, column) patch is one exact bank entry. The patch's random content (bumps, stones) comes from its own seed in
the seed tree, not from mjlab's generator rng. Environment origins are assigned explicitly per instance after the
scene is built (mjlab would draw rows with an unseeded torch.randint) and checked.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

from r1adv.negative import bank


def patch_heights(kind: str, value, x: np.ndarray, y: np.ndarray, rng: np.random.Generator, size: float) -> np.ndarray:
  from r1wb.world import heights
  if kind == "flat":
    return np.zeros_like(x)
  if kind in ("rough", "gravel"):
    return heights(f"{kind}{float(value)}", x, y, rng, size=size)
  if kind in ("tilt", "steep"):
    return np.tan(np.radians(float(value))) * y                       # full plane, no disc (see bank)
  if kind == "tilt_rough":
    degrees, centimetres = value
    return np.tan(np.radians(float(degrees))) * y + heights(f"rough{float(centimetres)}", x, y, rng, size=size)
  raise ValueError(kind)


@dataclass(kw_only=True)
class BankPatchCfg(SubTerrainCfg):
  kind: str
  levels: tuple
  bank_seed: int
  resolution: float = 0.1

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    row = int(round(difficulty * (bank.ROWS - 1)))
    local = np.random.default_rng(bank.patch_seed(self.bank_seed, self.kind, row))
    body = spec.body("terrain")
    columns, rows = int(self.size[0] / self.resolution), int(self.size[1] / self.resolution)
    x = np.linspace(-self.size[0] / 2, self.size[0] / 2, columns)[None, :].repeat(rows, 0)
    y = np.linspace(-self.size[1] / 2, self.size[1] / 2, rows)[:, None].repeat(columns, 1)
    z = patch_heights(self.kind, self.levels[row], x, y, local, self.size[0])
    low, high = float(z.min()), float(z.max())
    span = max(high - low, 1.0e-3)
    field = spec.add_hfield(name=f"bank_{self.kind}_{row}_{uuid.uuid4().hex[:8]}",
                            size=[self.size[0] / 2, self.size[1] / 2, span, 0.5], nrow=rows, ncol=columns,
                            userdata=((z - low) / span).astype(np.float32).flatten().tolist())
    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
                         pos=[self.size[0] / 2, self.size[1] / 2, low], rgba=[0.45, 0.47, 0.5, 1.0])
    return TerrainOutput(origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]),
                         geometries=[TerrainGeometry(geom=geom, hfield=field)])


@dataclass(kw_only=True)
class TiltedBoxCfg(SubTerrainCfg):
  """A plane through the patch centre rising along +y by the row's angle: one box rotated about x (box contacts,
  no height-field triangles under the feet)."""
  levels: tuple
  thickness: float = 1.0

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator) -> TerrainOutput:
    del rng
    row = int(round(difficulty * (bank.ROWS - 1)))
    angle = np.radians(float(self.levels[row]))
    body = spec.body("terrain")
    half = self.thickness / 2
    normal = np.array([0.0, -np.sin(angle), np.cos(angle)])
    centre = np.array([self.size[0] / 2, self.size[1] / 2, 0.0]) - half * normal
    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[self.size[0] / 2, self.size[1] / 2, half], pos=centre.tolist(),
                         quat=[float(np.cos(angle / 2)), float(np.sin(angle / 2)), 0.0, 0.0], rgba=[0.5, 0.5, 0.5, 1.0])
    return TerrainOutput(origin=np.array([self.size[0] / 2, self.size[1] / 2, 0.0]), geometries=[TerrainGeometry(geom=geom)])


def apply_bank(cfg, bank_seed: int) -> None:
  """The scene's floor becomes the whole bank; the ray sensors and physical fall of r1wb.world are installed."""
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.terrains import config as presets
  from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
  from r1wb import world
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in ("root_ground", "feet_ground_height")) + world.ray_sensors()
  cfg.terminations["fallen"] = TerminationTermCfg(func=world.fallen)
  terrain = cfg.scene.terrain
  terrain.terrain_type = "generator"
  terrain.max_init_terrain_level = None
  terrain.terrain_generator = generator_cfg(bank_seed)
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 250)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1600)


def generator_cfg(bank_seed: int):
  """The bank as an mjlab curriculum terrain generator (columns = kinds, rows = levels)."""
  from mjlab.terrains import config as presets
  from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
  size = (bank.PATCH, bank.PATCH)
  subs = {}
  for col in bank.COLUMNS:
    kind = col["kind"]
    if kind == "flat":
      subs[kind] = presets.flat(size=size, proportion=1.0)          # a box top: plane-like contacts (r1wb flat = plane)
    elif kind in ("tilt", "steep"):
      subs[kind] = TiltedBoxCfg(levels=tuple(col["levels"]), size=size, proportion=1.0)
    elif kind.startswith("stairs"):
      low, high = col["levels"][0] / 100.0, col["levels"][-1] / 100.0
      make = presets.pyramid_stairs_inv if kind == "stairs_up" else presets.pyramid_stairs
      subs[kind] = make(step_height_range=(low, high), step_width=col["step_width"], platform_width=col["platform"],
                        size=size, proportion=1.0)
    else:
      subs[kind] = BankPatchCfg(kind=kind, levels=tuple(col["levels"]), bank_seed=bank_seed,
                                resolution=col["resolution"], size=size, proportion=1.0)
  return TerrainGeneratorCfg(seed=int(bank_seed % (2**31)), size=size, border_width=2.0, num_rows=bank.ROWS,
                             num_cols=len(subs), curriculum=True, difficulty_range=(0.0, 1.0), color_scheme="none",
                             sub_terrains=subs)


def assign(env, rows: list[int], cols: list[int], grid: tuple[int, int] | None = None) -> torch.Tensor:
  """Place environment i on patch (rows[i], cols[i]) of the bank (or of `grid`); returns the origins (checked)."""
  terrain = env.scene.terrain
  r = torch.as_tensor(rows, dtype=torch.long, device=terrain.env_origins.device)
  c = torch.as_tensor(cols, dtype=torch.long, device=terrain.env_origins.device)
  if tuple(terrain.terrain_origins.shape[:2]) != (grid or (bank.ROWS, len(bank.COLUMNS))):
    raise ValueError(f"terrain grid {tuple(terrain.terrain_origins.shape[:2])} is not the bank")
  terrain.terrain_levels[:] = r
  terrain.terrain_types[:] = c
  terrain.env_origins[:] = terrain.terrain_origins[r, c]
  if not torch.equal(env.scene.env_origins, terrain.terrain_origins[r, c]):
    raise RuntimeError("scene origins do not follow the assigned patches")
  return env.scene.env_origins.clone()


def geometry_sha256(model) -> str:
  import hashlib
  digest = hashlib.sha256()
  for array in (model.geom_type, model.geom_pos, model.geom_quat, model.geom_size, model.hfield_size, model.hfield_data):
    value = np.ascontiguousarray(array)
    digest.update(str(value.dtype).encode() + str(value.shape).encode() + value.tobytes())
  return digest.hexdigest()
