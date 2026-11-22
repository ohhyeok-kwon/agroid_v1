"""Negative 30: r1adv-scenario-v1 specs for the pass table (neg30_pass.py).

Seed tree: round = H(20261121, 'neg30-R04-pass') -> family = H(round, family) -> evalset = H(family, 'set' + A|B|C) ->
instance = H(evalset, 'i<k>') -> noise seed = H(instance, 'sense'). The loop cell is NOT in the instance seed, so the same
instance carries the same noise sequence in every cell (paired cells); every policy runs the same specs (paired policies).
Families (mission timing follows the P24 readouts the pass criteria were set with):
  walk_stop     settle 2 s, forward 0.5 m/s 13 s (step response [t0, t0+8], walking look [t0+3, t0+13]), stop, hold 3 s
  stand_push    hold 26 s (stand window [2, 20] s), push at 20.0 s: J in 12/15/20/25 N s x front/back/left/right, 0.1 m up
  moves         settle 2 s, forward 0.3 6 s, hold 2, left 0.3 6 s, hold 2, back 0.3 6 s, hold 2, turn 0.8 rad/s 5 s, hold 2
  stand_static  symmetric static reference (R1 default pose, no planner) hold 10 s (standing symmetry)
  cmd_t1        positive control: T1 cmd robot 0 / 3 timing (t0 = 1 s, 20 s), no noise
Cells: <ol|cl>_L<bridge latency x10>_d<action delay ms>; ol = open-loop replanning (P29 m1c), cl = closed loop (m1cctx).
D2 sensing: uniform at the D2 training bounds (gyro 0.35 rad/s, orientation 0.087 rad per axis, q 0.035 rad, dq 2.6 rad/s).

  python -I neg30_specs.py OUT_DIR
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

CAND = {"inward": 0.06, "min_sep": 0.09, "skip_side": True, "stride_vref": 0.45, "start_frac": 0.8}
D2 = {"dist": "uniform", "gyro": 0.35, "ori": 0.087, "q": 0.035, "dq": 2.6}
ROUND = 20261121
GEN = {"name": "neg30-pass", "version": "v1", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
DIRS = (("front", 0.0), ("back", math.pi), ("left", math.pi / 2), ("right", -math.pi / 2))
JS = (12.0, 15.0, 20.0, 25.0)


def H(parent: int, name: str) -> int:
  return int.from_bytes(hashlib.sha256(f"{int(parent)}|{name}".encode()).digest()[:8], "little") & ((1 << 63) - 1)


def finish(spec):
  spec["spec_sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
  return spec


def cell_env(cell):
  loop, L, dl = cell.split("_")
  return dict(loop=dict(planner=CAND, bridge="B2+M1", ctx_robot=(loop == "cl"), bridge_lat_s=int(L[1:]) / 10.0),
              body=dict(action_delay_ms=float(dl[1:])))


def mission(fam):
  if fam == "walk_stop":
    return dict(settle_s=2.0, steps=[dict(type="Go", vx=0.5, duration_s=13.0, label="fwd0.5"), dict(type="Hold", duration_s=3.0)])
  if fam == "stand_push":
    return dict(settle_s=0.0, steps=[dict(type="Hold", duration_s=26.0)])
  if fam == "moves":
    st = []
    for lab, vx, vy, wz, dur in (("fwd0.3", 0.3, 0.0, 0.0, 6.0), ("left0.3", 0.0, 0.3, 0.0, 6.0), ("back0.3", -0.3, 0.0, 0.0, 6.0),
                                 ("turn0.8", 0.0, 0.0, 0.8, 5.0)):
      st += [dict(type="Go", vx=vx, vy=vy, wz=wz, duration_s=dur, label=lab), dict(type="Hold", duration_s=2.0)]
    return dict(settle_s=2.0, steps=st)
  if fam == "stand_static":
    return dict(settle_s=0.0, ref="static", steps=[dict(type="Hold", duration_s=10.0)])
  raise KeyError(fam)


def make(fam, cell, evalset, i, noise=True):
  fs = H(H(ROUND, "neg30-R04-pass"), fam)
  inst = H(H(fs, "set" + evalset), f"i{i}")
  env = cell_env(cell)
  env.update(onset_s=1.0, sense=dict(D2) if noise else {}, noise_seed=H(inst, "sense"))
  if fam == "stand_push":
    J = JS[i % 4]
    dn, dr = DIRS[(i // 4) % 4]
    env["hit"] = [dict(t_s=20.0, J_Ns=J, dir_rad=dr, dir_name=dn, z_off_m=0.1)]
  return finish(dict(schema="r1adv-scenario-v1", generator=GEN, seed=inst, family=fam, cell=cell, evalset=evalset, i=i,
                     mission=mission(fam), env=env, solvability_evidence=None))


def positive_controls():
  out = []
  for loop in ("ol", "cl"):
    for L in (0, 4):
      cell = f"{loop}_L{L:02d}_d00"
      env = cell_env(cell)
      env.update(onset_s=0.0, sense={}, noise_seed=0)
      for kind, steps in (("fwd", [dict(type="Go", vx=0.5, duration_s=19.0)]),
                          ("stop", [dict(type="Go", vx=0.5, duration_s=8.0), dict(type="Hold", duration_s=11.0)])):
        out.append(finish(dict(schema="r1adv-scenario-v1", generator=GEN, seed=0, family="cmd_t1", cell=cell, evalset="pc",
                               i=0 if kind == "fwd" else 1, pc=kind, mission=dict(settle_s=1.0, steps=steps), env=env,
                               solvability_evidence=None)))
  return out


def main():
  od = Path(sys.argv[1])
  od.mkdir(parents=True, exist_ok=True)
  sets = {}
  sets["pc"] = positive_controls()
  # v1 primary: easiest cell (bridge 0.1 s, action delay 0, D2), open loop = main row, closed loop = tracking row
  sets["v1_A"] = ([make("walk_stop", c, "A", i) for c in ("ol_L01_d00", "cl_L01_d00") for i in range(32)]
                  + [make("stand_push", "ol_L01_d00", "A", i) for i in range(32)]
                  + [make("moves", c, "A", i) for c in ("ol_L01_d00", "cl_L01_d00") for i in range(16)]
                  + [make("stand_static", "ol_L01_d00", "A", i) for i in range(16)])
  # sweeps (one axis at a time from the easiest cell): bridge latency 0.2 / 0.4 s; action delay 20 / 40 ms (40 provisional)
  sets["sweep_A"] = ([make("walk_stop", c, "A", i) for c in ("ol_L02_d00", "ol_L04_d00", "ol_L01_d20", "ol_L01_d40") for i in range(32)]
                     + [make("stand_push", c, "A", i) for c in ("ol_L01_d20", "ol_L01_d40") for i in range(32)]
                     + [make("walk_stop", c, "A", i) for c in ("cl_L02_d00", "cl_L04_d00", "cl_L01_d20", "cl_L01_d40") for i in range(16)])
  # noise floor: evaluation seed sets B and C on the primary cell
  sets["floor_BC"] = [make(f, c, s, i) for s in ("B", "C") for f, c, n in (("walk_stop", "ol_L01_d00", 32), ("stand_push", "ol_L01_d00", 32),
                                                                           ("moves", "ol_L01_d00", 16), ("stand_static", "ol_L01_d00", 16))
                      for i in range(n)]
  for k, v in sets.items():
    (od / f"{k}.jsonl").write_text("".join(json.dumps(s) + "\n" for s in v))
    print(k, len(v))
  print("generator", GEN["source_sha256"][:16])


if __name__ == "__main__":
  main()
