"""Negative 28: r1adv-scenario-v1 generator for the SONIC -> R1 line (R04): Hold / Go (flat) x d_env
{Body, Hit, Sense, floor roughness}. Seed tree: round seed -> family seed -> instance seed (seedtree.child_seed);
draws are counter-based (seedtree.Draw), so a spec depends only on (generator version, source hash, seed path).

Each family draws a severity s in [0, 1] stratified over its instances (instance k: s = (k + u) / N) and scales its
axes with s, so every family is a 1-D difficulty axis for the boundary search:
  fric   foot mu = 1.2 - 0.9 s, cone pyramidal (impratio 1) or elliptic (impratio 10)
  body   mass x U[1 -+ 0.2 s] per group (torso / legs / arms), torso COM +- 0.03 s m, per-joint gain x U[1 -+ 0.15 s]
         (kp and kd together, realised torque-constant error), action delay U[0, 40 s] ms
  hit    1-2 impulses J = 5 + 25 s N s (second one 0.5-1x), any direction, height offset -0.15..0.25 m, any time
  sense  Gaussian gyro 0.2 s rad/s, orientation 0.05 s rad, q 0.02 s rad, dq 1.5 s rad/s, sensing delay U[0, 20 s] ms
  rough  1-5 mm random height field (h = 1 + 4 s mm, 5 cm cells)
  combo  all of the above at one s
  anchor nominal (s = 0)
Missions (coordinator's range), after a 2 s unscored settle: Hold 20 s, or Go = start, 2-4 of {forward 0.5 / 0.3, lateral +-0.3, turn +-0.8 rad/s,
short hold}, stop (final hold >= 2.5 s), <= 16 s. 'fricgrid' = the deterministic task-1 grid.

  python -m r1adv.negative.sonic_gen --set bundle200 --out specs.jsonl
  python -m r1adv.negative.sonic_gen --set fricgrid --out grid.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from r1adv.negative.seedtree import Draw, child_seed, r6, spec_hash

NAME, VERSION = "r1adv.negative.sonic_gen", "neg28-v1"
ROUND_SEED = child_seed(20261121, "r1adv-R04-neg28-sonic")
COUNTS = dict(anchor=12, fric=32, body=36, hit=48, sense=28, rough=24, combo=28)
_HERE = Path(__file__).resolve()


def source_sha256() -> str:
  h = hashlib.sha256()
  for f in (_HERE, _HERE.parent / "seedtree.py"):
    h.update(f.read_bytes())
  return h.hexdigest()


SETTLE = 2.0                             # spawn transient (default pose, empty history); unscored, G anchored at its end


def hold_mission():
  return dict(settle_s=SETTLE, steps=[dict(type="Hold", duration_s=20.0)])


def go_mission(D: Draw):
  pool = ["fwd05", "fwd03", "lat", "turn", "hold"]
  k = D.integer(2, 4)
  steps, last = [], None
  for i in range(k):
    opts = [p for p in pool if not (p == "hold" and (last in (None, "hold")))]
    p = D.choice(opts)
    if p == "fwd05":
      steps.append(dict(type="Go", vx=0.5, vy=0.0, wz=0.0, duration_s=r6(D.uniform(3.0, 5.0))))
    elif p == "fwd03":
      steps.append(dict(type="Go", vx=0.3, vy=0.0, wz=0.0, duration_s=r6(D.uniform(3.0, 5.0))))
    elif p == "lat":
      steps.append(dict(type="Go", vx=0.0, vy=0.3 * D.choice([1.0, -1.0]), wz=0.0, duration_s=r6(D.uniform(2.0, 3.0))))
    elif p == "turn":
      steps.append(dict(type="Go", vx=0.0, vy=0.0, wz=0.8 * D.choice([1.0, -1.0]), duration_s=r6(D.uniform(1.5, 2.5))))
    else:
      steps.append(dict(type="Hold", duration_s=r6(D.uniform(1.5, 2.5))))
    last = p
  if steps[-1]["type"] == "Hold":
    steps[-1]["duration_s"] = r6(max(steps[-1]["duration_s"], 2.5))
  else:
    steps.append(dict(type="Hold", duration_s=r6(D.uniform(2.5, 3.5))))
  total = sum(s["duration_s"] for s in steps)
  if total > 16.0:                       # keep episodes <= 18 s with the 2 s settle
    f = 16.0 / total
    for s in steps:
      s["duration_s"] = r6(s["duration_s"] * f)
  return dict(settle_s=SETTLE, steps=steps)


def mission_T(m):
  return m["settle_s"] + sum(s["duration_s"] for s in m["steps"])


def body_axis(D: Draw, s: float):
  return dict(mass_scale={g: r6(D.uniform(1 - 0.2 * s, 1 + 0.2 * s)) for g in ("torso", "legs", "arms")},
              torso_com=[r6(D.uniform(-0.03 * s, 0.03 * s)) for _ in range(3)],
              kp_scale=[r6(D.uniform(1 - 0.15 * s, 1 + 0.15 * s)) for _ in range(24)],
              action_delay_ms=r6(D.uniform(0.0, 40.0 * s)))


def hits_axis(D: Draw, s: float, T: float):
  n = D.integer(1, 2)
  out = []
  J0 = 5.0 + 25.0 * s
  for i in range(n):
    J = J0 if i == 0 else J0 * D.uniform(0.5, 1.0)
    out.append(dict(t_s=r6(D.uniform(1.5, T - 2.5)), J_Ns=r6(J), dir_rad=r6(D.uniform(0.0, 2 * math.pi)),
                    z_off_m=r6(D.uniform(-0.15, 0.25))))
  return sorted(out, key=lambda h: h["t_s"])


def sense_axis(D: Draw, s: float):
  return dict(gyro_std=r6(0.2 * s), ori_std=r6(0.05 * s), q_std=r6(0.02 * s), dq_std=r6(1.5 * s),
              delay_ms=r6(D.uniform(0.0, 20.0 * s)))


def make(family: str, k: int, n: int):
  fam_seed = child_seed(ROUND_SEED, family)
  seed = child_seed(fam_seed, f"i{k:03d}")
  D = Draw(seed)
  s = 0.0 if family == "anchor" else r6((k + D.u()) / n)
  hold = (k % 3 == 0) if family != "anchor" else (k < 4)
  mission = hold_mission() if hold else go_mission(D)
  T = mission_T(mission)
  env = dict(body={}, hit=[], sense={}, floor={})
  if family in ("fric", "combo"):
    env["body"].update(foot_mu=r6(1.2 - 0.9 * s), cone=D.choice(["pyramidal", "elliptic"]))
    env["body"]["impratio"] = 10.0 if env["body"]["cone"] == "elliptic" else 1.0
  if family in ("body", "combo"):
    b = body_axis(D, s)
    b["kd_scale"] = list(b["kp_scale"])
    env["body"].update(b)
  if family in ("hit", "combo"):
    env["hit"] = hits_axis(D, s, T)
  if family in ("sense", "combo"):
    env["sense"] = sense_axis(D, s)
  if family in ("rough", "combo"):
    env["floor"] = dict(rough_h_m=r6(0.001 + 0.004 * s), cell_m=0.05)
  spec = dict(schema="r1adv-scenario-v1", generator=dict(name=NAME, version=VERSION, source_sha256=source_sha256()),
              seed=seed, seed_path=["r1adv-R04-neg28-sonic", family, f"i{k:03d}"], family=family,
              level=dict(axis=family, s=s), mission=mission, env=env, solvability_evidence=None)
  spec["spec_sha256"] = spec_hash(spec)
  return spec


def fricgrid():
  out = []
  for mu in (0.3, 0.5, 0.7, 1.0):
    for cone, imp in (("pyramidal", 1.0), ("elliptic", 10.0)):
      for mname in ("hold20", "go05"):
        for start in ("spawn", "switch"):          # switch: spawn and settle on mu 1.0, foot mu drops at the settle end
          mission = hold_mission() if mname == "hold20" else dict(settle_s=SETTLE, steps=[
            dict(type="Go", vx=0.5, vy=0.0, wz=0.0, duration_s=15.0), dict(type="Hold", duration_s=4.0)])
          name = f"{mu}|{cone}|{mname}|{start}"
          seed = child_seed(child_seed(ROUND_SEED, "fricgrid"), name)
          body = dict(foot_mu=mu, cone=cone, impratio=imp)
          if start == "switch":
            body["mu_switch_s"] = SETTLE
          spec = dict(schema="r1adv-scenario-v1", generator=dict(name=NAME, version=VERSION, source_sha256=source_sha256()),
                      seed=seed, seed_path=["r1adv-R04-neg28-sonic", "fricgrid", name], family="fricgrid",
                      level=dict(axis="fric", mu=mu, cone=cone, mission=mname, start=start), mission=mission,
                      env=dict(body=body, hit=[], sense={}, floor={}), solvability_evidence=None)
          spec["spec_sha256"] = spec_hash(spec)
          out.append(spec)
  return out


def bundle():
  return [make(f, k, n) for f, n in COUNTS.items() for k in range(n)]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--set", choices=("bundle200", "fricgrid"), required=True)
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  specs = bundle() if a.set == "bundle200" else fricgrid()
  Path(a.out).write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in specs))
  print(json.dumps(dict(out=a.out, n=len(specs), generator=specs[0]["generator"], round_seed=ROUND_SEED,
                        families={f: sum(1 for s in specs if s["family"] == f) for f in sorted({s["family"] for s in specs})})))


if __name__ == "__main__":
  main()
