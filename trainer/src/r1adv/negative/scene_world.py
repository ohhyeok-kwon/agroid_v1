"""Scene atlas (r1adv.negative.scenes) -> mjlab terrain: every 6 x 6 m patch is one floor height field (F) in a shared
grid, every environment spawns on its patch's spawn cell (MuJoCo-Warp shares geometry across worlds; per-world
variety comes from the patch an environment stands on and, later, from fixed slot bodies). The overhead field C is
kept for perception and planning; R1 (about 1.2 m) cannot pass under anything F does not already make solid.
"""
from __future__ import annotations

import math
import uuid
from dataclasses import dataclass

import mujoco
import numpy as np
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput

ROWS = 6
CELL = 0.05


def load_atlas(path):
  data = np.load(path, allow_pickle=False)
  import json
  return data["F"], data["C"], data["spawn"], json.loads(str(data["meta"]))


@dataclass(kw_only=True)
class AtlasPatchCfg(SubTerrainCfg):
  atlas: str
  indices: tuple

  def function(self, difficulty: float, spec: mujoco.MjSpec, rng) -> TerrainOutput:
    del rng
    row = int(round(difficulty * (ROWS - 1))) if len(self.indices) > 1 else 0
    index = self.indices[min(row, len(self.indices) - 1)]
    F, _, spawn, _ = load_atlas(self.atlas)
    field_xy = F[index].astype(np.float64)                     # [i along x, j along y]
    z = field_xy.T                                              # MuJoCo rows run along y
    low, high = float(z.min()), float(z.max())
    span = max(high - low, 1.0e-3)
    rows, columns = z.shape
    body = spec.body("terrain")
    hfield = spec.add_hfield(name=f"atlas_{index}_{uuid.uuid4().hex[:8]}", size=[self.size[0] / 2, self.size[1] / 2, span, 0.5],
                             nrow=rows, ncol=columns, userdata=((z - low) / span).astype(np.float32).flatten().tolist())
    geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hfield.name,
                         pos=[self.size[0] / 2, self.size[1] / 2, low], rgba=[0.55, 0.5, 0.45, 1.0])
    i, j = (int(v) for v in spawn[index])
    origin = np.array([(i + 0.5) * self.size[0] / field_xy.shape[0], (j + 0.5) * self.size[1] / field_xy.shape[1],
                       float(field_xy[i, j])])
    return TerrainOutput(origin=origin, geometries=[TerrainGeometry(geom=geom, hfield=hfield)])


def layout(count: int) -> tuple[int, int]:
  return ROWS, max(1, math.ceil(count / ROWS))


def apply_atlas(cfg, atlas: str) -> tuple[int, int]:
  """The scene's floor becomes every atlas patch (grid ROWS x columns); returns the grid shape."""
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
  from r1wb import world
  F, _, _, meta = load_atlas(atlas)
  rows, cols = layout(F.shape[0])
  cfg.scene.sensors = tuple(s for s in cfg.scene.sensors if s.name not in ("root_ground", "feet_ground_height")) + world.ray_sensors()
  cfg.terminations["fallen"] = TerminationTermCfg(func=world.fallen)
  size = (meta["patch_m"], meta["patch_m"])
  subs = {}
  for c in range(cols):
    indices = tuple((c * rows + r) % F.shape[0] for r in range(rows))
    subs[f"atlas{c}"] = AtlasPatchCfg(atlas=str(atlas), indices=indices, size=size, proportion=1.0)
  terrain = cfg.scene.terrain
  terrain.terrain_type = "generator"
  terrain.max_init_terrain_level = None
  terrain.terrain_generator = TerrainGeneratorCfg(seed=0, size=size, border_width=2.0, num_rows=rows, num_cols=cols,
                                                  curriculum=True, difficulty_range=(0.0, 1.0), color_scheme="none",
                                                  sub_terrains=subs)
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 250)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1600)
  return rows, cols


def patch_cell(index: int, count: int) -> tuple[int, int]:
  rows, _ = layout(count)
  return index % rows, index // rows


def bench_specs(atlas: str, count: int, flat: bool = False, round_name: str = "R00") -> list[dict]:
  """Throughput specs: stand 1 s, walk 0.2 m/s for 3 s, stand 2 s, spread over the atlas patches (or, with flat, the
  same missions on the bank's flat patch). Seeds come from the seed tree; the policy's behaviour is not scored."""
  from pathlib import Path
  from r1adv.negative import NAME, VERSION, generator as G
  from r1adv.negative.families import NEUTRAL_BODY, _terrain
  from r1adv.negative.seedtree import Draw, child_seed, r6, spec_hash
  F, _, _, meta = load_atlas(atlas)
  out = []
  for i in range(count):
    seed = child_seed(G.round_seed(round_name), f"scene-bench/{Path(atlas).stem}/{i}")
    d = Draw(seed)
    yaw = r6(d.uniform(-math.pi, math.pi))
    mission = dict(grammar="mission-v1", settle_s=G.SETTLE_S, duration_s=r6(G.SETTLE_S + 6.0), steps=[
      dict(type="Hold", duration_s=1.0), dict(type="Go", mode="velocity", shape="straight", vx=0.2, vy=0.0, wz=0.0, duration_s=3.0),
      dict(type="Hold", duration_s=2.0)])
    sense = dict(profile="uniform-n1shape-v1", scale=0.0, delay_steps=0, dropout_p=0.0, noise_seed=child_seed(seed, "sense/noise"))
    if flat:
      env = dict(terrain=_terrain("flat", 0, yaw), scene=None, obstacles=[], disturbances=[], sense=sense, body=dict(NEUTRAL_BODY))
      family = "baseline-v0"
    else:
      env = dict(terrain=None, scene=dict(source="atlas", atlas=Path(atlas).stem, index=i % F.shape[0], spawn_yaw_rad=yaw),
                 obstacles=[], disturbances=[], sense=sense, body=dict(NEUTRAL_BODY))
      family = "scene-bench-v0"
    spec = dict(schema="r1adv-scenario-v1", generator=dict(name=NAME, version=VERSION, family=family, source_sha256="bench"),
                seed=seed, seed_path=f"{round_name}/{family}/{i}", mission=mission, env=env, solvability_evidence=None)
    spec["spec_sha256"] = spec_hash(spec)
    out.append(spec)
  return out


if __name__ == "__main__":
  import argparse
  import json
  from pathlib import Path
  parser = argparse.ArgumentParser()
  parser.add_argument("--atlas", required=True)
  parser.add_argument("--count", type=int, required=True)
  parser.add_argument("--flat", action="store_true")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  from r1adv.negative.seedtree import canonical
  args.out.write_text("".join(canonical(s) + "\n" for s in bench_specs(args.atlas, args.count, args.flat)))
