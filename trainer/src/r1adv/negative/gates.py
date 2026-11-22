"""Admission gates for new environment code (coordinator 11-17 synthesis section 4), cheapest first.

1. determinism: the same (family, params, goal seed, episode) gives the same spec hash in two fresh processes.
2. physics probe: bodies (16 cm box, or a foot-like sole of three capsules) released with no force on every patch
   must come to rest, must not sink into the ground and must not gain energy; CPU MuJoCo and MuJoCo-Warp. A patch that
   keeps a free body moving would make its contact jitter look like policy tremor. PROVISIONAL: the probe compiles a
   bare MjSpec; with settings="env" (default since R1) it takes the runner env's mjlab SimulationCfg (timestep,
   integrator, cone, solver, iterations, warp options) and the R1 foot capsule's contact parameters. Still not
   validated against robot outcomes; a passive sole may legitimately rock on bumps.
3. analytic necessary conditions: family.feasible() (static slope, capture-point shift, step height, passage width).
4. witness: levels.LevelBuffer scores only levels whose witness calm-success lower bound reaches tau.
usage (server, GPU 3): python -m r1adv.negative.gates probe [--atlas atlas.npz] --out probe.json
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

BOX_HALF, BOX_MASS = 0.08, 2.0


def determinism(family: str, params: dict, goal_seed: int, episodes: int = 2) -> dict:
  code = ("import json,sys;from r1adv.negative.levels import Level,episode_spec,level_id;"
          "from r1adv.negative.families import family_source_sha256 as h;a=json.loads(sys.argv[1]);s=h();"
          "lv=Level(level_id(a['f'],a['p'],a['g'],s),a['f'],a['p'],a['g']);"
          "print(json.dumps([episode_spec(lv,e,'gate',s)['spec_sha256'] for e in range(a['n'])]))")
  arg = json.dumps(dict(f=family, p=params, g=goal_seed, n=episodes))
  runs = [subprocess.run([sys.executable, "-B", "-c", code, arg], capture_output=True, text=True, check=True).stdout for _ in range(2)]
  return dict(passed=runs[0] == runs[1] and bool(runs[0].strip()), hashes=json.loads(runs[0]))


def _patch_spec(kind: str, row: int, atlas=None):
  """MjSpec with one bank patch (or atlas patch index `row` when kind == 'atlas') under body 'terrain'."""
  import mujoco
  from r1adv.negative import bank as B
  from r1adv.negative.generator import BANK_SEED
  spec = mujoco.MjSpec()
  spec.option.timestep = 0.002
  spec.worldbody.add_body(name="terrain")
  if kind == "atlas":
    from r1adv.negative.scene_world import AtlasPatchCfg
    cfg = AtlasPatchCfg(atlas=str(atlas), indices=(row,), size=(6.0, 6.0))
    out = cfg.function(0.0, spec, None)
  else:
    from mjlab.terrains import config as presets
    from r1adv.negative.world_build import BankPatchCfg, TiltedBoxCfg
    col = B.COLUMNS[B.column(kind)]
    size = (B.PATCH, B.PATCH)
    if kind == "flat":
      cfg = presets.flat(size=size)
    elif kind in ("tilt", "steep"):
      cfg = TiltedBoxCfg(levels=tuple(col["levels"]), size=size)
    elif kind.startswith("stairs"):
      make = presets.pyramid_stairs_inv if kind == "stairs_up" else presets.pyramid_stairs
      cfg = make(step_height_range=(col["levels"][0] / 100, col["levels"][-1] / 100), step_width=col["step_width"],
                 platform_width=col["platform"], size=size)
    else:
      cfg = BankPatchCfg(kind=kind, levels=tuple(col["levels"]), bank_seed=BANK_SEED, resolution=col["resolution"], size=size)
    out = cfg.function(row / (B.ROWS - 1), spec, np.random.default_rng(0))
  return spec, np.asarray(out.origin, float)


def _surface(model, data, xy):
  """Height and unit normal of the ground (group 0) under xy."""
  import mujoco
  point = np.array([xy[0], xy[1], 30.0])
  geom = np.array([-1], dtype=np.int32)
  normal = np.zeros(3)
  distance = mujoco.mj_ray(model, data, point, np.array([0.0, 0.0, -1.0]), np.array([1, 0, 0, 0, 0, 0], np.uint8), 1, -1, geom, normal)
  if distance < 0 or normal[2] <= 0:
    return None, None
  return float(point[2] - distance), normal / np.linalg.norm(normal)


def _align(normal) -> np.ndarray:
  """Quaternion turning +z onto `normal` (the box rests on its face, not on an edge)."""
  axis = np.cross([0.0, 0.0, 1.0], normal)
  s, c = np.linalg.norm(axis), float(normal[2])
  if s < 1e-9:
    return np.array([1.0, 0.0, 0.0, 0.0])
  angle = math.atan2(s, c)
  axis = axis / s
  return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis])


def _drop_points(kind, row, origin, points, atlas):
  if kind != "atlas":
    return [(origin[0], origin[1])] + [(origin[0] + 1.2 * math.cos(k * 2 * math.pi / (points - 1)),
                                        origin[1] + 1.2 * math.sin(k * 2 * math.pi / (points - 1))) for k in range(points - 1)]
  from scipy import ndimage
  F = np.load(atlas)["F"][row]
  cell = 6.0 / F.shape[0]
  free = (F <= 0.06) & (ndimage.distance_transform_edt(F <= 0.06) * cell >= 0.3)
  cells = np.argwhere(free)
  chosen = [cells[np.argmin(((cells - np.array(F.shape) / 2) ** 2).sum(1))]]
  while len(chosen) < min(points, len(cells)):                    # farthest-point spread over the free floor
    gap = np.min([((cells - c) ** 2).sum(1) for c in chosen], axis=0)
    chosen.append(cells[int(np.argmax(gap))])
  return [((i + 0.5) * cell, (j + 0.5) * cell) for i, j in chosen]


def env_settings():
  """The rollouts' simulator settings (mjlab SimulationCfg of the runner's env) and the R1 foot capsule's contact
  parameters, so the probe asks the same simulator the policies stand in."""
  import mujoco
  from r1wb import robot
  from r1wb.student.stream import stream_env_cfg
  sim = stream_env_cfg(play=True, clean=True, plain_actions=True, matched=False).sim
  spec = robot.robot_cfg().spec_fn()
  model = spec.compile()
  foot = model.geom("left_foot4_collision")
  contact = dict(friction=[float(v) for v in foot.friction], condim=int(foot.condim[0]),
                 solref=[float(v) for v in foot.solref], solimp=[float(v) for v in foot.solimp],
                 margin=float(foot.margin[0]), gap=float(foot.gap[0]))
  return sim, contact


def probe(kind: str, row: int, *, points: int = 9, seconds: float = 3.0, backend: str = "cpu", atlas=None,
          body: str = "sole", settings: str = "env") -> dict:
  """Boxes (16 cm, 2 kg) placed 1 cm above the ground, aligned with it, released with no force. After 1 s they must
  rest (speed < 5 mm/s over the last 0.5 s), not sink (contact distance > -2 cm) and not gain energy (< 0.01 J)."""
  import mujoco
  spec, origin = _patch_spec(kind, row, atlas)
  sim, foot = env_settings() if settings == "env" else (None, None)
  for i in range(points):
    probe_body = spec.worldbody.add_body(name=f"probe{i}", pos=[0.0, 0.0, 50.0 + i])
    probe_body.add_freejoint()
    if body == "box":
      probe_body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[BOX_HALF] * 3, mass=BOX_MASS, group=3)
    else:                       # a foot-like sole: three parallel capsules (R1 feet are seven capsules each)
      for y in (-0.04, 0.0, 0.04):
        geom = probe_body.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.025, 0.09, 0], pos=[0.0, y, 0.0],
                                   quat=[0.7071068, 0.0, 0.7071068, 0.0], mass=BOX_MASS / 3, group=3)
        if foot is not None:                       # the R1 foot capsule's own contact parameters
          geom.friction, geom.condim = foot["friction"], foot["condim"]
          geom.solref, geom.solimp, geom.margin, geom.gap = foot["solref"], foot["solimp"], foot["margin"], foot["gap"]
  model = spec.compile()
  if sim is not None:
    sim.mujoco.apply(model)                        # timestep, integrator, cone, solver, iterations as the env
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  placed = 0
  for i, (x, y) in enumerate(_drop_points(kind, row, origin, points, atlas)):
    height, normal = _surface(model, data, (x, y))
    if height is None:
      continue
    adr = model.jnt_qposadr[model.body(f"probe{i}").jntadr[0]]
    data.qpos[adr:adr + 3] = np.array([x, y, height]) + normal * ((BOX_HALF if body == "box" else 0.025) + 0.01)
    data.qpos[adr + 3:adr + 7] = _align(normal)
    placed += 1
  mujoco.mj_forward(model, data)
  steps, settle = int(seconds / model.opt.timestep), int(1.0 / model.opt.timestep)
  tail_n = int(0.5 / model.opt.timestep)
  if backend == "cpu":
    energy, speed, depth = [], [], []
    for _ in range(steps):
      mujoco.mj_step(model, data)
      mujoco.mj_energyPos(model, data)
      mujoco.mj_energyVel(model, data)
      energy.append(float(data.energy.sum()))
      speed.append(float(np.abs(data.qvel).max()))
      depth.append(float(min([c.dist for c in data.contact[:data.ncon]], default=0.0)))
    energy, speed = np.asarray(energy), np.asarray(speed)
    gain = float(energy[settle:].max() - energy[settle])
    tail = speed[-tail_n:]
    return dict(kind=kind, row=row, backend="cpu", body=body, settings=settings, timestep=float(model.opt.timestep), placed=placed, settled=bool(tail.max() < 5e-3), tail_speed=float(tail.max()),
                min_contact_m=float(min(depth[settle:])), energy_gain_after_1s_J=gain,
                passed=bool(tail.max() < 5e-3 and min(depth[settle:]) > -0.02 and gain < 1e-2))
  import mujoco_warp as mjw
  import warp as wp
  m = mjw.put_model(model)
  if sim is not None:
    sim.apply_wp_opt(m.opt)
  d = mjw.put_data(model, data, nworld=1, nconmax=64 * points, njmax=600 * points)
  tail = []
  for k in range(steps):
    mjw.step(m, d)
    if k >= steps - tail_n:
      tail.append(float(np.abs(d.qvel.numpy()).max()))
  wp.synchronize()
  return dict(kind=kind, row=row, backend="warp", body=body, settings=settings, timestep=float(model.opt.timestep), placed=placed, settled=bool(max(tail) < 5e-3), tail_speed=float(max(tail)),
              passed=bool(max(tail) < 5e-3))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("what", choices=("probe",))
  parser.add_argument("--atlas", type=Path, default=None)
  parser.add_argument("--backends", nargs="+", default=["cpu", "warp"])
  parser.add_argument("--body", default="sole", choices=("box", "sole"))
  parser.add_argument("--settings", default="env", choices=("env", "default"))
  parser.add_argument("--rows", type=int, nargs="+", default=[0, 3, 7])
  parser.add_argument("--atlas-count", type=int, default=None, help="first N atlas patches only")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  from r1adv.negative import bank as B
  rows = []
  for col in B.COLUMNS:
    for row in args.rows:
      for backend in args.backends:
        rows.append(probe(col["kind"], row, backend=backend, body=args.body, settings=args.settings))
        print(json.dumps(rows[-1]), flush=True)
  if args.atlas is not None:
    count = int(np.load(args.atlas)["F"].shape[0]) if args.atlas_count is None else args.atlas_count
    for index in range(count):
      for backend in args.backends:
        rows.append(probe("atlas", index, backend=backend, atlas=args.atlas, body=args.body, settings=args.settings))
        print(json.dumps(rows[-1]), flush=True)
  args.out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
  main()
