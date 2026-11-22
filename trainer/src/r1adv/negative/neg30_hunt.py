"""Negative 30 (coordinator 11:49 item 6): failure hunt per candidate. Seeded random adversarial specs (r1adv-scenario-v1,
seed tree round -> 'neg30-hunt' -> axis -> instance), one primary axis per spec at a drawn severity on top of the main loop
(open-loop m1c, bridge 0.1 s, D2 x 0.1 sensing, action delay 0), plus a share of two-axis combinations. Every spec is run for
the candidate and the manufacturer witness (evaluation only). Bank classes:
  task        candidate fails, manufacturer succeeds  -> the positive's to-do list (solvable)
  unsolvable  both fail                                -> kept apart, not a gate (charter: hard but solvable)
  pass        candidate succeeds
Axes and draws (value-before, 12:0x KST):
  push_stand  Hold 12 s, one push J ~ U[10, 35] N s, direction ~ U[0, 2 pi), time ~ U[3, 7] s, height ~ U[-0.10, 0.30] m
  push_walk   walk 0.5 m/s 10 s (+ stop 3 s), one push J ~ U[10, 35], any direction, time ~ U[t0 + 2, t0 + 8], height as above
  reversal    sudden command flips (no idle between): fwd 0.5 3 s -> back 0.3 3 s -> fwd 0.5 3 s -> stop | left 0.3 -> right 0.3
              -> left 0.3 -> stop | turn +0.8 -> -0.8 -> +0.8 rad/s -> stop | fwd 0.5 -> turn 0.8 in place -> fwd 0.5 -> stop
  friction    walk 0.5 or Hold + push 15 N s, foot mu ~ U[0.12, 0.6]
  mass        torso COM shift (x, y) ~ U[-0.06, 0.06]^2 m and torso / legs / arms mass scale ~ U[0.8, 1.3]; walk 0.5 + stop
  delay       action delay ~ U[0, 50] ms (5 ms grid); walk 0.5 + stop
  noise       sensing D2 x s, s ~ U[0.1, 1.0]; walk 0.5 + stop or Hold 12 s
  combo       two of the above axes drawn together (severity of each drawn as above)
Success (strict): no fall, no two-foot hop, and the mission G: walking segments of the commanded kind realised >= 0.7
(reversal segments >= 0.5, in the commanded direction); Hold without a push: drift <= 0.10 m (charter Hold G); any push
mission: still again (|v| < 0.10 m/s over the last 1 s).
  python -I neg30_hunt.py make OUT.jsonl N [ROUND_TAG]
  python -I neg30_hunt.py bank RUN_DIR SPECS.jsonl CANDIDATE OUT.md OUT_BANK.jsonl"""
from __future__ import annotations

import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/rl-agent/src/r1adv/negative")
import neg30_specs as G  # noqa: E402

AXES = ("push_stand", "push_walk", "reversal", "friction", "mass", "delay", "noise", "combo")
SHARE = (0.16, 0.16, 0.14, 0.10, 0.10, 0.10, 0.10, 0.14)
GEN = {"name": "neg30-hunt", "version": "v1", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
D2 = G.D2


def H(p, n):
  return G.H(p, n)


def base_env(inst):
  return dict(loop=dict(planner=G.CAND, bridge="B2+M1", ctx_robot=False, bridge_lat_s=0.1), body=dict(action_delay_ms=0.0),
              onset_s=1.0, sense=dict({k: (v * 0.1 if isinstance(v, float) else v) for k, v in D2.items()}, scale_of_D2=0.1),
              noise_seed=H(inst, "sense"), hit=[])


def walk_mission(dur=10.0, v=0.5):
  return dict(settle_s=2.0, steps=[dict(type="Go", vx=v, duration_s=dur, label="fwd"), dict(type="Hold", duration_s=3.0)])


def hold_mission(dur=12.0):
  return dict(settle_s=0.0, steps=[dict(type="Hold", duration_s=dur)])


def rev_mission(kind):
  segs = {0: [("fwd", 0.5, 0.0, 0.0), ("back", -0.3, 0.0, 0.0), ("fwd", 0.5, 0.0, 0.0)],
          1: [("left", 0.0, 0.3, 0.0), ("right", 0.0, -0.3, 0.0), ("left", 0.0, 0.3, 0.0)],
          2: [("turnL", 0.0, 0.0, 0.8), ("turnR", 0.0, 0.0, -0.8), ("turnL", 0.0, 0.0, 0.8)],
          3: [("fwd", 0.5, 0.0, 0.0), ("turnL", 0.0, 0.0, 0.8), ("fwd", 0.5, 0.0, 0.0)]}[kind]
  st = [dict(type="Go", vx=vx, vy=vy, wz=wz, duration_s=3.0, label=lab) for lab, vx, vy, wz in segs]
  return dict(settle_s=2.0, steps=st + [dict(type="Hold", duration_s=3.0)])


def apply_axis(axis, rng, env, mission):
  info = {}
  if axis == "push_stand":
    mission = hold_mission()
    J, d, t, z = rng.uniform(10, 35), rng.uniform(0, 2 * math.pi), rng.uniform(3, 7), rng.uniform(-0.1, 0.3)
    env["hit"] = [dict(t_s=round(t, 3), J_Ns=round(J, 2), dir_rad=round(d, 4), z_off_m=round(z, 3))]
    info = dict(J=J, dir=d, t=t, z=z)
  elif axis == "push_walk":
    mission = walk_mission()
    J, d, t, z = rng.uniform(10, 35), rng.uniform(0, 2 * math.pi), rng.uniform(4, 10), rng.uniform(-0.1, 0.3)
    env["hit"] = [dict(t_s=round(t, 3), J_Ns=round(J, 2), dir_rad=round(d, 4), z_off_m=round(z, 3))]
    info = dict(J=J, dir=d, t=t, z=z)
  elif axis == "reversal":
    k = int(rng.integers(0, 4))
    mission = rev_mission(k)
    info = dict(kind=k)
  elif axis == "friction":
    mu = rng.uniform(0.12, 0.6)
    env["body"]["foot_mu"] = round(mu, 3)
    if rng.random() < 0.5:
      mission = walk_mission()
    else:
      mission = hold_mission()
      env["hit"] = [dict(t_s=5.0, J_Ns=15.0, dir_rad=round(rng.uniform(0, 2 * math.pi), 4), z_off_m=0.1)]
    info = dict(mu=mu)
  elif axis == "mass":
    mission = walk_mission()
    env["body"]["torso_com"] = [round(rng.uniform(-0.06, 0.06), 3), round(rng.uniform(-0.06, 0.06), 3), 0.0]
    env["body"]["mass_scale"] = {g: round(rng.uniform(0.8, 1.3), 3) for g in ("torso", "legs", "arms")}
    info = dict(com=env["body"]["torso_com"], scale=env["body"]["mass_scale"])
  elif axis == "delay":
    mission = walk_mission()
    d = 5.0 * int(rng.integers(0, 11))
    env["body"]["action_delay_ms"] = d
    info = dict(delay_ms=d)
  elif axis == "noise":
    s = rng.uniform(0.1, 1.0)
    env["sense"] = dict({k: (v * s if isinstance(v, float) else v) for k, v in D2.items()}, scale_of_D2=round(s, 3))
    mission = walk_mission() if rng.random() < 0.5 else hold_mission()
    info = dict(scale=s)
  return mission, info


def make(out, n, tag="r1"):
  root = H(H(G.ROUND, "neg30-hunt"), tag)
  sp = []
  for i in range(n):
    inst = H(root, f"i{i}")
    rng = np.random.default_rng(inst)
    axis = AXES[int(rng.choice(len(AXES), p=SHARE))]
    env = base_env(inst)
    mission = walk_mission()
    infos = {}
    if axis == "combo":
      a1, a2 = rng.choice(["push_walk", "friction", "mass", "delay", "noise"], size=2, replace=False)
      mission, infos[a1] = apply_axis(str(a1), rng, env, mission)
      if a2 == "push_walk":
        mission, infos[a2] = apply_axis(str(a2), rng, env, mission)
      else:
        keep = mission
        _m, infos[a2] = apply_axis(str(a2), rng, env, mission)
        mission = keep if a2 in ("mass", "delay", "noise", "friction") else _m
    else:
      mission, infos[axis] = apply_axis(axis, rng, env, mission)
    s = dict(schema="r1adv-scenario-v1", generator=GEN, seed=inst, family="hunt", hunt=dict(axis=axis, draw=infos, tag=tag),
             cell="hunt_" + axis, evalset="hunt", i=i, mission=mission, env=env, solvability_evidence=None)
    sp.append(G.finish(json.loads(json.dumps(s, default=float))))
  Path(out).write_text("".join(json.dumps(s) + "\n" for s in sp))
  print(len(sp), "specs", {a: sum(s["hunt"]["axis"] == a for s in sp) for a in AXES})


def success(spec, r):
  """Strict success from a neg30_pass 'hunt' row (measure family 'hunt': per-segment realised + standing drift)."""
  if r.get("F") or r.get("hops", 0) > 0:
    return False
  segs = r.get("segments", [])
  rev = spec["hunt"]["axis"] == "reversal" or "reversal" in spec["hunt"].get("draw", {})
  for s_ in segs:
    if s_.get("realised") is None:
      return False
    if s_["realised"] < (0.5 if rev else 0.7):
      return False
  if r.get("hold_drift_m") is not None and r["hold_drift_m"] > 0.10:   # Hold without a push: charter Hold G
    return False
  if spec["env"].get("hit") and not r.get("still_end", False):          # after a push: standing still again by the end
    return False
  return True


def bank(run, specs_path, cand, out_md, out_bank):
  specs = {json.loads(l)["spec_sha256"]: json.loads(l) for l in open(specs_path) if l.strip()}
  rows = [json.loads(l) for l in (Path(run) / "rows.jsonl").read_text().splitlines() if l.strip()]
  by = {}
  for r in rows:
    if "error" in r or r["spec_sha256"] not in specs:
      continue
    by.setdefault(r["spec_sha256"], {})[r["policy"]] = r
  B, cnt = [], {}
  for h, d in by.items():
    if cand not in d or "mf" not in d:
      continue
    sp = specs[h]
    c, w = success(sp, d[cand]), success(sp, d["mf"])
    lateral = sp["hunt"]["axis"] == "reversal" and sp["hunt"]["draw"].get("reversal", {}).get("kind") == 1
    cls = "pass" if c else ("user_goal" if lateral else ("task" if w else "unsolvable"))   # coordinator 11:2x: side = user goal
    ax = sp["hunt"]["axis"]
    cnt.setdefault(ax, {"pass": 0, "task": 0, "unsolvable": 0, "user_goal": 0})[cls] += 1
    if cls != "pass":
      why = ("fall" if d[cand].get("F") else ("hop" if d[cand].get("hops", 0) > 0 else "G"))
      B.append(dict(spec_sha256=h, axis=ax, draw=sp["hunt"]["draw"], cls=cls, why=why, seed=sp["seed"],
                    cand_segments=d[cand].get("segments"), mf_F=d["mf"].get("F")))
  Path(out_bank).write_text("".join(json.dumps(b, default=float) + "\n" for b in B))
  L = [f"# 실패 사냥 — 후보 {cand} (제조사 증인; 성공 = 넘어짐·두 발 뛰기 없음 ∧ 미션 G)", "",
       "| 축 | 후보 통과 | 포지티브 과제(제조사는 풂) | 풀 수 없음(둘 다 실패) | 사용자 목표(옆, 증인 없음) |", "|---|---|---|---|---|"]
  for a in AXES:
    if a in cnt:
      c_ = cnt[a]
      L.append(f"| {a} | {c_['pass']} | {c_['task']} | {c_['unsolvable']} | {c_['user_goal']} |")
  L.append("")
  L.append("## 포지티브 과제 (시드로 재현: spec_sha256 → specs 파일)")
  for b in B:
    if b["cls"] == "task":
      L.append(f"- {b['axis']} {json.dumps(b['draw'], default=lambda x: round(float(x), 3))} → {b['why']} (spec {b['spec_sha256'][:16]})")
  Path(out_md).write_text("\n".join(L) + "\n")
  print("\n".join(L[:14]))


if __name__ == "__main__":
  if sys.argv[1] == "make":
    make(sys.argv[2], int(sys.argv[3]), sys.argv[4] if len(sys.argv) > 4 else "r1")
  else:
    bank(*sys.argv[2:7])
