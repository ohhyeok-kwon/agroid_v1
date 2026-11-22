"""Negative 30: noise floor of the pass-table readout (task 2). Same checkpoints (B''r@500, C3@1500) on evaluation seed sets
A, B, C (independent noise sequences, same missions); per metric the checkpoint statistic is computed per seed set:
  readout SD  = SD over the three seed sets (pooled over the two checkpoints: sqrt of the mean variance)
  ckpt spread = SD over the three checkpoints of a window (C3 1000 / 1250 / 1500, C2 likewise) on seed set A
  MDE         = 2 x readout SD (smallest difference between two checkpoints this readout can call)
  python -I neg30_floor.py OUT.md CELL RUN_DIR [RUN_DIR ...]      (CELL e.g. ol_L01_d00_lo)
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def load(dirs):
  out, seen = [], set()
  for d in dirs:
    p = Path(d) / "rows.jsonl"
    if p.exists():
      for l in p.read_text().splitlines():
        if l.strip():
          r = json.loads(l)
          if "error" not in r and (r["spec_sha256"], r["policy"]) not in seen:
            seen.add((r["spec_sha256"], r["policy"]))
            out.append(r)
  return out


def med(xs):
  xs = [x for x in xs if x is not None and math.isfinite(x)]
  return float(np.median(xs)) if xs else None


def seg(r, lab):
  s = next((x for x in r.get("segments", []) if x.get("label") == lab), None)
  return s.get("realised") if s and s.get("complete") else None


METRICS = {   # name -> (family, function over the episode rows -> statistic)
  "walk G 몫": ("walk_stop", lambda R: float(np.mean([bool(r.get("G_walk")) for r in R]))),
  "앞 0.5 정상 실현 중앙": ("walk_stop", lambda R: med([r.get("steady_ratio") for r in R])),
  "t50 중앙 s": ("walk_stop", lambda R: med([r.get("t50_s") for r in R])),
  "정지 중앙 s": ("walk_stop", lambda R: med([r.get("stop_s") for r in R])),
  "roll RMS 중앙 °": ("walk_stop", lambda R: med([r.get("roll_rms_deg") for r in R])),
  "팔 흔들기 중앙 rad": ("walk_stop", lambda R: med([r.get("arm_swing_rad") for r in R])),
  "골반 좌우 p-p 중앙 m": ("walk_stop", lambda R: med([r.get("pelvis_lat_pp_m") for r in R])),
  "걸음 빈도 중앙": ("walk_stop", lambda R: med([r.get("step_hz") for r in R])),
  "서기 떠밀림 중앙 cm": ("stand_push", lambda R: (lambda m: m * 100 if m is not None else None)(med([r.get("stand_drift_m") for r in R]))),
  "서기 디딤 0 몫": ("stand_push", lambda R: float(np.mean([r.get("stand_steps") == 0 for r in R]))),
  "밂 넘어짐 수": ("stand_push", lambda R: float(sum(bool(r.get("push_fell")) for r in R))),
  "밂 회복 중앙 s": ("stand_push", lambda R: med([r.get("push_recovery_s") for r in R if not r.get("push_fell")])),
  "앞 0.3 실현 중앙": ("moves", lambda R: med([seg(r, "fwd0.3") for r in R])),
  "옆 0.3 실현 중앙": ("moves", lambda R: med([seg(r, "left0.3") for r in R])),
  "대칭 서기 엉덩이 yaw L+R 중앙 °": ("stand_static", lambda R: med([r.get("sym_hip_yaw_LR") for r in R])),
}


def main():
  out, cell, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
  rows = [r for r in load(dirs) if r.get("cell") == cell]
  by = defaultdict(list)
  for r in rows:
    by[(r["policy"], r.get("evalset"), r["family"])].append(r)
  L = [f"# 잡음 바닥 (칸 {cell})", "", "| 지표 | B″r@500 A·B·C | C3@1500 A·B·C | 판독 SD | MDE = 2 SD | C3 창 SD (A) | C2 창 SD (A) | 판정 |",
       "|---|---|---|---|---|---|---|---|"]
  for name, (fam, fn) in METRICS.items():
    per, var = {}, []
    for pol in ("Bppr500", "C3s1500"):
      vals = [fn(by[(pol, s, fam)]) if by.get((pol, s, fam)) else None for s in ("A", "B", "C")]
      per[pol] = vals
      v = [x for x in vals if x is not None]
      if len(v) >= 2:
        var.append(float(np.var(v, ddof=1)))
    sd = math.sqrt(np.mean(var)) if var else None
    wsd = {}
    for w, members in (("C3", ("C3s1000", "C3s1250", "C3s1500")), ("C2", ("C2s1000", "C2s1250", "C2s1500"))):
      v = [fn(by[(p, "A", fam)]) for p in members if by.get((p, "A", fam))]
      v = [x for x in v if x is not None]
      wsd[w] = float(np.std(v, ddof=1)) if len(v) >= 2 else None
    fmt = lambda x: "—" if x is None else f"{x:.3f}"  # noqa: E731
    verdict = "—"
    if sd is not None and wsd.get("C3") is not None:
      verdict = "저장본 흔들림 > 판독 잡음" if wsd["C3"] > 2 * sd else ("판독 잡음 안" if wsd["C3"] <= sd else "경계")
    L.append(f"| {name} | {' · '.join(fmt(x) for x in per['Bppr500'])} | {' · '.join(fmt(x) for x in per['C3s1500'])} | {fmt(sd)} | "
             f"{fmt(2 * sd if sd is not None else None)} | {fmt(wsd['C3'])} | {fmt(wsd['C2'])} | {verdict} |")
  n = {k: len(v) for k, v in by.items() if k[0] in ("Bppr500", "C3s1500")}
  L.append("\n판 수: " + ", ".join(f"{k[0]}/{k[1]}/{k[2]} {v}" for k, v in sorted(n.items())))
  Path(out).write_text("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main()
