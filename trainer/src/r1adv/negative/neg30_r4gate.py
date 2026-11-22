"""Negative 30 (coordinator 12:06): official decision line for R4 checkpoints from the jittered P29-style fast readout
(n30_fast.sh outputs fast30/<TAG>_k<i>/fast.json), against the B@0 noise band (fast30/B0band_k0..k7, n = 8).
Rules v2 (amended 12:2x KST, before any R4 line was read; v1 compared the candidate's k0 with the mean of the JITTERED B@0 band
and called B@0 itself KILL on symmetric S2, because the zero-jitter run k0 is not a draw from the jittered set):
  every cell: paired on the jitter index - d_i = candidate k_i - B@0 k_i, i = 0..n-1 (n = the candidate's runs, 4 for R4);
    changed when |mean d| > max(t_(n-1, 5 % two-sided) x SD(d) / sqrt(n), practical floor FLOOR[cell]); n = 1 falls back to
    2 x SD of the B@0 band. Floors: posture-card floors x 0.2, calm / command lines x 0.1 (FLOOR below).
  direction: 'better' = toward the manufacturer / pass line (lower for posture, roll, pelvis, t50, stop, recovery, W4 |ratio-1|,
    W5; higher for follow, displacement, side, turn, turn lift share)
  verdict:
    KILL  any posture cell worse (S1-S4 planner idle, S1-S4 symmetric idle, W1-W3) or push falls > 0 or fwd 0.5 follow < 0.85
    WIN   no KILL, and calm better (roll or pelvis), and fwd 0.3 follow / displacement and turn lift share not worse (paired)
    HOLD  otherwise
  absolute pass lines (pass table, manufacturer based) are listed for reference: roll <= 1 deg, pelvis p-p <= 0.06 m,
    t50 <= 0.5 s, stop <= 0.5 s, push recovery <= 1.03 s.
  python -I neg30_r4gate.py TAG OUT_MD [STEP]      (appends one block to OUT_MD; prints the block)"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

N = Path("/rl-agent/.agent/tmp/neg30/fast30")


def g(d, *path):
  for x in path:
    if not isinstance(d, dict):
      return None
    d = d.get(x)
  return d


def mx(a, b):
  v = [x for x in (a, b) if x is not None]
  return max(v) if v else None


M = {  # key: (label, fn, direction (+1 higher better / -1 lower better / 'one' |x-1| lower better), wide, posture)
  "f05": ("앞 0.5 따르기", lambda r: g(r, "w05", "follow"), 1, False, False),
  "f03": ("앞 0.3 따르기", lambda r: g(r, "w03", "follow"), 1, True, False),
  "d03": ("앞 0.3 이동 m", lambda r: g(r, "w03", "disp"), 1, True, False),
  "side": ("옆 0.3", lambda r: g(r, "side03"), 1, False, False),
  "turn": ("돌기 0.8", lambda r: g(r, "turn08"), 1, False, False),
  "lift": ("돌기 발 뗌", lambda r: g(r, "turn_lift_share"), 1, False, False),
  "t50": ("t50 s", lambda r: g(r, "t50"), -1, False, False),
  "stop": ("정지 s", lambda r: g(r, "stop"), -1, True, False),
  "roll": ("roll °", lambda r: g(r, "w05", "torso_roll_rms"), -1, False, False),
  "pelvis": ("골반 p-p m", lambda r: g(r, "w05", "pelvis_lat_pp"), -1, False, False),
  "W1": ("W1 °", lambda r: mx(g(r, "w05", "torso_roll_off"), g(r, "w05", "torso_pitch_off")), -1, False, True),
  "W2": ("W2 °", lambda r: mx(g(r, "w05", "roll_off"), g(r, "w05", "yaw_off")), -1, False, True),
  "W3": ("W3 rad", lambda r: g(r, "w05", "arm_dev_max"), -1, False, True),
  "W4t": ("W4 시간비", lambda r: g(r, "w05", "step_time_ratio"), "one", False, False),
  "W4L": ("W4 길이비", lambda r: g(r, "w05", "step_len_ratio"), "one", False, False),
  "W5": ("W5 °/m", lambda r: g(r, "w05", "heading_drift_deg_per_m"), -1, True, False),
  "S1": ("S1 °", lambda r: mx(g(r, "stand_m1c", "torso_roll_off"), g(r, "stand_m1c", "torso_pitch_off")), -1, False, True),
  "S2": ("S2 °", lambda r: mx(g(r, "stand_m1c", "roll_off"), g(r, "stand_m1c", "yaw_off")), -1, False, True),
  "S3": ("S3 °", lambda r: g(r, "stand_m1c", "leg_asym_max"), -1, False, True),
  "S4": ("S4 rad", lambda r: g(r, "stand_m1c", "arm_dev_max"), -1, False, True),
  "sS1": ("대칭 S1 °", lambda r: mx(g(r, "stand_m1cis", "torso_roll_off"), g(r, "stand_m1cis", "torso_pitch_off")), -1, False, True),
  "sS2": ("대칭 S2 °", lambda r: mx(g(r, "stand_m1cis", "roll_off"), g(r, "stand_m1cis", "yaw_off")), -1, False, True),
  "sS3": ("대칭 S3 °", lambda r: g(r, "stand_m1cis", "leg_asym_max"), -1, False, True),
  "sS4": ("대칭 S4 rad", lambda r: g(r, "stand_m1cis", "arm_dev_max"), -1, False, True),
  "pf": ("밂 20 넘어짐", lambda r: g(r, "push20", "falls"), -1, False, False),
  "prec": ("밂 20 회복 s", lambda r: g(r, "push20", "rec_med"), -1, True, False),
}
ABS = {"roll": 1.0, "pelvis": 0.06, "t50": 0.5, "stop": 0.5, "prec": 1.03}


def load(prefix):
  out = []
  for k in ["k0"] + [f"k{i}" for i in range(1, 8)]:
    p = N / f"{prefix}_{k}" / "fast.json"
    if p.exists():
      out.append(json.load(open(p)))
  return out


def val(x, d):
  if x is None:
    return None
  return abs(x - 1.0) if d == "one" else x


# practical floor per cell (amendment 12:2x KST): a change must also exceed this to count, so that a change far below what the
# posture card / pass table can resolve is not called (card floors x 0.2; calm / command lines x 0.1)
FLOOR = {"f05": 0.03, "f03": 0.05, "d03": 0.10, "side": 0.05, "turn": 0.05, "lift": 0.01, "t50": 0.05, "stop": 0.05, "roll": 0.1,
         "pelvis": 0.006, "W1": 0.1, "W2": 0.1, "W3": 0.03, "W4t": 0.01, "W4L": 0.01, "W5": 0.4, "S1": 0.1, "S2": 0.1, "S3": 0.2,
         "S4": 0.02, "sS1": 0.1, "sS2": 0.1, "sS3": 0.2, "sS4": 0.02, "pf": 0.5, "prec": 0.1}
TCRIT = {2: 12.71, 3: 4.30, 4: 3.18, 5: 2.78, 6: 2.57, 7: 2.45, 8: 2.36}


def main():
  tag, out_md = sys.argv[1], sys.argv[2]
  step = sys.argv[3] if len(sys.argv) > 3 else "?"
  base, cand = load("B0band"), load(tag)
  if not cand:
    print("no runs for", tag)
    return
  n = min(len(cand), len(base))
  rows, kill, better_calm, worse = [], [], False, []
  for key, (lab, fn, d, wide, post) in M.items():
    pairs = [(val(fn(base[i]), d), val(fn(cand[i]), d)) for i in range(n)]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    if not pairs:
      rows.append(f"| {lab} | — | — | — | — | — |")
      continue
    a = np.array([x[0] for x in pairs]); b = np.array([x[1] for x in pairs]); df = b - a
    dm = float(df.mean())
    if len(df) >= 2:
      se = float(df.std(ddof=1)) / math.sqrt(len(df))
      stat = TCRIT.get(len(df), 2.0) * se
    else:                                                      # single run: B@0 jitter SD as the spread (old rule)
      bv = [val(fn(r), d) for r in base]
      bv = [x for x in bv if x is not None]
      stat = 2 * float(np.std(bv, ddof=1)) if len(bv) > 1 else 0.0
    thr = max(stat, FLOOR[key])
    sgn = 1 if d == 1 else -1                                  # +1 when higher is better
    st = "="
    if abs(dm) > thr:
      st = "↑" if sgn * dm > 0 else "↓"
    if st == "↓":
      worse.append(key)
      if post or key == "pf":
        kill.append(f"{key} {dm:+.3f}")
    if st == "↑" and key in ("roll", "pelvis"):
      better_calm = True
    cm = float(b.mean())
    ab = ""
    if key in ABS:
      ab = " ✓" if cm <= ABS[key] else " ✗"
    rows.append(f"| {lab} | {cm:.3f}{ab} | {float(a.mean()):.3f} | {dm:+.3f} | {thr:.3f} | {st} |")
  f05 = M["f05"][1](cand[0])
  if f05 is not None and f05 < 0.85:
    kill.append("f05<0.85")
  if kill:
    verdict = "KILL (" + ", ".join(kill) + ")"
  elif better_calm and "f03" not in worse and "d03" not in worse and "lift" not in worse:
    verdict = "WIN"
  else:
    verdict = "HOLD"
  head = (f"\n### {tag} (step {step}, {time.strftime('%H:%M', time.gmtime(time.time() + 9 * 3600))} KST, 짝 흔들기 n = {n}, "
          f"규칙 v2 12:2x) → **{verdict}**")
  blk = [head, "| 지표 | 후보 평균 (✓/✗ = 절대 합격선) | B@0 같은 흔들기 평균 | 짝 차 평균 | 문턱 | 판 |", "|---|---|---|---|---|---|"] + rows
  with open(out_md, "a") as fo:
    fo.write("\n".join(blk) + "\n")
  print("\n".join(blk))


if __name__ == "__main__":
  main()
