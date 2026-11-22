"""Negative 28: level sets for the adversarial issuing loop (charter: n = 8 instances per level, Wilson lower bound,
issue by learnability p(1-p) with a solvability witness). Axes chosen by the coordinator (04:4x KST): loop delay, one
sensor-noise channel at a time, and pushes while walking; standing symmetry is read on every Hold.

A level = (axis, x). Its 8 instances come from the seed tree round -> "levels" -> axis -> x -> i. Missions and the
success criterion follow each axis' own failure mode (fixed 04:4x KST before any B'/B''r value), so a target's
mission-level weakness (e.g. forward speed) does not swamp the axis:
  delay, sdelay  8 random Go missions (start, 2-4 steps, stop);  success = no fall and every Hold after a Go has standing J
         (after its first second) <= max(20, 3 x the policy's own J at 0 ms) -> no delay-driven standing oscillation
  noise  8 Hold 20 s (gyro | ori | q | dq; realisations differ per seed); success = no fall and charter Hold G
         (end drift <= 0.10 m, heading <= 0.10 rad)
  push   forward 0.5 walk 6 s + stop, one push at a random time while walking / direction / height; success = no fall
  spush  Hold 20 s with one push of x N s at a random time / direction / height; success = no fall (08:2x KST)
  combo  every bundle-combo axis at severity s = x (friction, body, delay, 1-2 pushes, sense, roughness), Hold (even i)
         or Go (odd i); success = no fall (added 06:2x KST)
v2 (04:5x KST): instances are paired across levels (mission, push geometry and noise draws come from (axis, i) only;
x sets the magnitude alone) and delay / noise start at the settle end (env.onset_s), so the spawn transient is excluded.
Witness = manufacturer (evaluation only) under the same criterion. Spec content is r1adv-scenario-v1; the generator
hash covers this file, sonic_gen.py and seedtree.py.

  python -m r1adv.negative.sonic_levels --grid '{"delay": [5, 10]}' --out specs.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from r1adv.negative import sonic_gen as G
from r1adv.negative.seedtree import Draw, child_seed, r6, spec_hash

NAME, VERSION = "r1adv.negative.sonic_levels", "neg28-levels-v2"
N_INST = 8
_HERE = Path(__file__).resolve()
SENSE = {"gyro": "gyro_std", "ori": "ori_std", "q": "q_std", "dq": "dq_std", "sdelay": "delay_ms"}


def source_sha256() -> str:
  h = hashlib.sha256()
  for f in (_HERE, _HERE.parent / "sonic_gen.py", _HERE.parent / "seedtree.py"):
    h.update(f.read_bytes())
  return h.hexdigest()


def push_mission(D: Draw, J: float):
  m = dict(settle_s=G.SETTLE, steps=[dict(type="Go", vx=0.5, vy=0.0, wz=0.0, duration_s=6.0), dict(type="Hold", duration_s=3.0)])
  hit = [dict(t_s=r6(D.uniform(G.SETTLE + 1.5, G.SETTLE + 5.0)), J_Ns=r6(J), dir_rad=r6(D.uniform(0.0, 6.283185307)),
              z_off_m=r6(D.uniform(-0.15, 0.25)))]
  return m, hit


def make(axis: str, x: float, i: int, src: str):
  ax_seed = child_seed(child_seed(G.ROUND_SEED, "levels"), axis)
  seed = child_seed(ax_seed, f"{x}|i{i}")                 # spec identity
  inst = child_seed(ax_seed, f"i{i}")                     # paired across x: same mission / push geometry / noise draws
  D = Draw(inst)
  env = dict(body={}, hit=[], sense={}, floor={}, onset_s=G.SETTLE, noise_seed=child_seed(inst, "sense"))
  if axis == "push":
    mission, env["hit"] = push_mission(D, x)
  elif axis == "spush":                                   # static push: Hold 20 s, one push of x N s (08:2x KST)
    mission = G.hold_mission()
    env["hit"] = [dict(t_s=r6(D.uniform(G.SETTLE + 2.0, G.SETTLE + 14.0)), J_Ns=r6(x), dir_rad=r6(D.uniform(0.0, 6.283185307)),
                       z_off_m=r6(D.uniform(-0.15, 0.25)))]
  elif axis == "combo":                                   # all bundle-combo axes at one severity s = x (paired by i)
    mission = G.hold_mission() if i % 2 == 0 else G.go_mission(D)
    T = G.mission_T(mission)
    env["body"].update(foot_mu=r6(1.2 - 0.9 * x), cone=D.choice(["pyramidal", "elliptic"]))
    env["body"]["impratio"] = 10.0 if env["body"]["cone"] == "elliptic" else 1.0
    b = G.body_axis(D, x)
    b["kd_scale"] = list(b["kp_scale"])
    env["body"].update(b)
    env["hit"] = G.hits_axis(D, x, T)
    env["sense"] = G.sense_axis(D, x)
    env["floor"] = dict(rough_h_m=r6(0.001 + 0.004 * x), cell_m=0.05)
  else:
    mission = G.go_mission(D) if axis in ("delay", "sdelay") else G.hold_mission()
    if axis == "delay":
      env["body"]["action_delay_ms"] = r6(x)
    elif axis in SENSE:
      env["sense"][SENSE[axis]] = r6(x)
    else:
      raise ValueError(axis)
  spec = dict(schema="r1adv-scenario-v1", generator=dict(name=NAME, version=VERSION, source_sha256=src), seed=seed,
              seed_path=["r1adv-R04-neg28-sonic", "levels", axis, f"{x}|i{i}"], instance_seed=inst, family=f"L_{axis}",
              level=dict(axis=axis, x=x, i=i), mission=mission, env=env, solvability_evidence=None)
  spec["spec_sha256"] = spec_hash(spec)
  return spec


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--grid", required=True, help='JSON {axis: [levels]}; axes delay | gyro | ori | q | dq | sdelay | push | spush | combo')
  ap.add_argument("--out", required=True)
  ap.add_argument("--n", type=int, default=N_INST, help="instances per level (i = 0 .. n-1; pairing by i is kept)")
  a = ap.parse_args()
  grid = json.loads(a.grid)
  src = source_sha256()
  specs = [make(ax, float(x), i, src) for ax, xs in grid.items() for x in xs for i in range(a.n)]
  Path(a.out).write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in specs))
  print(json.dumps(dict(out=a.out, n=len(specs), source_sha256=src, grid=grid)))


if __name__ == "__main__":
  main()
