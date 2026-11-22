"""Negative 30 (coordinator 17:40): tables for the v4 real-time pass table and the push panel (neg30_v4.py rows).
Lines = 1.5 x manufacturer (manufacturer at action delay 0, same cells, evaluation only); trackers at 15 ms.
  python -I neg30_v4table.py OUT.md RUN_DIR [RUN_DIR ...]
Ruler (23:2x): NEG30_GEN (neg30_gen.py, default v4); tracker rows of run directories of another ruler are left out (mf kept)."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neg30_gen as RG  # noqa: E402
import neg30_body as NB  # noqa: E402  (NEG30_BODY: every run dir must be of this body)


def med(xs):
  xs = [x for x in xs if x is not None and (not isinstance(x, float) or math.isfinite(x))]
  return float(np.median(xs)) if xs else None


def f(x, nd=2):
  return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def cycle_terms(run, r):
  """Coordinator 17:58 (amendment recorded after the values were seen): overshoot = max over [t0, t0 + 8 s] of the forward
  speed averaged over one gait cycle (two steps; the run's own step rate) / command - 1 (the 0.5 s average stays beside it);
  steady realised against the REFERENCE = robot mean forward speed / reference mean forward speed over [t0 + 3, t0 + 8]
  (line 0.95-1.05; the manufacturer has no reference). Cells v4_walk_nf / v4_walk_lo only: t0 = 2.0 s, command 0.5 m/s."""
  f = run / "npz" / r["policy"] / f"{r['spec_sha256'][:16]}.npz"
  if r["F"] or not f.exists():
    return {}
  z = np.load(f)
  dt = 0.02
  v, q = z["v"].astype(float), z["quat"].astype(float)
  yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
  fwd = v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw)
  n = max(1, int(round(2.0 / (r.get("step_hz") or 3.0) / dt)))
  cyc = np.convolve(fwd, np.ones(n) / n, mode="same")
  a, b, st = int(2.0 / dt), int(10.0 / dt), slice(int(5.0 / dt), int(10.0 / dt))
  out = dict(ovs_cyc_pct=max(0.0, float(cyc[a:b].max()) / 0.5 - 1) * 100)
  rr = z["ref_rpos"].astype(float)
  if np.any(rr):
    out["steady_ref_ratio"] = float(fwd[st].mean() / np.gradient(rr[:, 0], dt)[st].mean())
  return out


def main():
  out, runs = sys.argv[1], [Path(x) for x in sys.argv[2:]]
  rows = []
  other = defaultdict(int)
  for run in runs:              # several run dirs (e.g. the baselines in v4/run + one checkpoint's dir); TAGs must not repeat
    NB.require(run)
    g = RG.dir_gen(run)
    for l in (run / "rows.jsonl").read_text().splitlines():
      if l.strip():
        r = json.loads(l)
        if g != RG.GEN and r.get("policy") != "mf":     # another ruler's tracker rows: not in this table
          other[f"{r.get('policy')} ({g})"] += 1
          continue
        r["_run"] = str(run)
        rows.append(r)
  err = [r for r in rows if "error" in r]
  rows = [r for r in rows if "error" not in r and r.get("cell") and not str(r.get("family", "")).startswith("scene_flow")]
  pols = ["mf"] + sorted({r["policy"] for r in rows} - {"mf"})
  by = defaultdict(list)
  for r in rows:
    by[(r["policy"], r["cell"].rsplit("_d", 1)[0])].append(r)
  W = lambda p: by[(p, "v4_walk_nf")] + by[(p, "v4_walk_lo")]  # noqa: E731
  for p in pols:
    for r in W(p):
      r.update(cycle_terms(Path(r["_run"]), r))
  REC = " — 기록만(품질 칸은 디스크리미네이터 카드)"     # coordinator 20:4x: one quantity, one line
  walk_keys = [("steady_ratio", "앞 0.5 실현", +1), ("arm_swing_rad", "팔 흔들기 rad" + REC, 0), ("pelvis_lat_pp_m", "골반 좌우 m" + REC, 0),
               ("roll_rms_deg", "roll °" + REC, 0), ("pitch_rms_deg", "pitch °" + REC, 0), ("ovs_cyc_pct", "넘침 % (한 주기 평균; 사전 선 15 %)", -1),
               ("overshoot_pct", "넘침 % (0.5 s 평균, 곁 값)", 0), ("steady_ref_ratio", "정상 실현 ÷ 참조 (0.95–1.05)", 2),
               ("t50_s", "t50 s", -1), ("stop_s", "정지 (앞 0.10 m/s) s", -1), ("stop2_s", "정지 (골반 0.03 m/s) s", -1),
               ("step_hz", "걸음/s", 0), ("jerk_walk_med", "걷기 jerk" + REC, 0)]
  mfW = W("mf")
  lines = {k: 1.5 * med([r.get(k) for r in mfW]) for k, _l, d in walk_keys if d == -1 and med([r.get(k) for r in mfW]) is not None}
  L = [f"# 몸 {NB.BODY} · {RG.GEN} 참조 · 실시간 경로 · 행동 지연 15 ms (제조사는 지연 0, 평가 전용) — 선 = 1.5 × 제조사", "",
       f"- 자: {RG.label()}" + (f"; 다른 자의 추종기 행은 뺐음: {', '.join(sorted(other))}" if other else ""), "",
       "## 1. 합격표 (앞 0.5: 잡음 없음 1 + D2 × 0.1 8 판의 중앙 [최소–최대]; 넘어짐·두 발 뛰기는 9 판 합)",
       "| 항목 (선) | " + " | ".join(pols) + " |", "|---|" + "---|" * len(pols)]
  for k, lab, d in walk_keys:
    cells = []
    for p in pols:
      v = [r.get(k) for r in W(p) if not r["F"]]
      v = [x for x in v if x is not None]
      if not v:
        cells.append("—")
        continue
      m = float(np.median(v))
      mark = "" if p == "mf" or k not in lines else (" ✓" if m <= lines[k] + 1e-9 else " ✗")   # a tie on the 0.02 s tick passes (23:5x)
      if k == "ovs_cyc_pct" and p != "mf":
        mark += f" (15 %: {'✓' if m <= 15.0 else '✗'})"
      if d == 2:
        mark = " ✓" if 0.95 <= m <= 1.05 else " ✗"
      cells.append(f"{m:.3f} [{min(v):.2f}–{max(v):.2f}]{mark}")
    L.append(f"| {lab}{f' (≤ {lines[k]:.3f})' if k in lines else ''} | " + " | ".join(cells) + " |")
  L.append("| 넘어짐 / 두 발 뛰기 (9 판) | " + " | ".join(f"{sum(r['F'] for r in W(p))} / {sum(r.get('hops', 0) for r in W(p))}" for p in pols) + " |")
  # moves
  for lab in ("fwd0.3", "left0.2", "back0.3", "turn0.8"):
    cells = []
    for p in pols:
      M = by[(p, "v4_moves_nf")]
      sg = [next((s for s in r.get("segments", []) if s.get("label") == lab), None) for r in M]
      sg = [s for s in sg if s]
      cells.append("—" if not sg else f"{f(sg[0].get('realised'))} (발 뗌 {f(sg[0].get('swing_share'))})")
    L.append(f"| {lab} 실현 (발 뗌) | " + " | ".join(cells) + " |")
  # standing + standard pushes
  mfS = by[("mf", "v4_stand_nf")]
  sline = None
  if mfS:
    byJ = defaultdict(list)
    for r in mfS:
      if not r["push_fell"] and r.get("push_recovery_s") is not None:
        byJ[r["push_J"]].append(r["push_recovery_s"])
    sline = 1.5 * max(float(np.median(v)) for v in byJ.values()) if byJ else None
  for lab, fn in (("서기 디딤 / 떠밀림 cm", lambda S: f"{med([r['stand_steps'] for r in S]):.0f} / {100 * med([r['stand_drift_m'] for r in S]):.1f}"),
                  ("밂 12–25 (4 방향) 넘어짐", lambda S: f"{sum(r['push_fell'] for r in S)}/{len(S)}"),
                  (f"밂 회복 J 별 중앙 최댓값 s{f' (≤ {sline:.2f})' if sline else ''}", None)):
    cells = []
    for p in pols:
      S_ = by[(p, "v4_stand_nf")]
      if not S_:
        cells.append("—")
        continue
      if fn is not None:
        cells.append(fn(S_))
      else:
        byJ = defaultdict(list)
        for r in S_:
          byJ[r["push_J"]].append(r["push_recovery_s"] if (not r["push_fell"] and r.get("push_recovery_s") is not None) else float("inf"))
        w = max(float(np.median(v)) for v in byJ.values())
        cells.append(("∞" if not math.isfinite(w) else f"{w:.2f}") + ("" if p == "mf" or sline is None else (" ✓" if w <= sline + 1e-9 else " ✗")))
    L.append(f"| {lab} | " + " | ".join(cells) + " |")
  # push panel
  L += ["", "## 2. 밂 판 (방향 8 × 세기 12·18·25 N·s; 걷기는 걸음 위상 2; 넘어짐 k/n · 회복 중앙/최대 s · 두 발 뛰기 · 팔 최대 rad · 몸통 최대 °)",
        "| 꼴 | J | " + " | ".join(pols) + " |", "|---|---|" + "---|" * len(pols)]
  for mode in ("stand", "walk03", "walk05"):
    for J in (12.0, 18.0, 25.0):
      cells = []
      for p in pols:
        R_ = [r for r in by[(p, f"v4push_{mode}")] if r.get("J") == J]
        if not R_:
          cells.append("—")
          continue
        ok = [r for r in R_ if not r["F"]]
        rec = [r["recovery_s"] for r in ok if r.get("recovery_s") is not None]
        nr = len(ok) - len(rec)
        cells.append(f"F {sum(r['F'] for r in R_)}/{len(R_)} · {f(med(rec))}/{f(max(rec) if rec else None)}{f' (회복 못 함 {nr})' if nr else ''}"
                     f" · 뜀 {sum(r.get('hops', 0) for r in R_)} · 팔 {f(med([r.get('arm_max') for r in ok]))} · 몸통 {f(med([r.get('torso_max_deg') for r in ok]), 1)}")
      L.append(f"| {mode} | {int(J)} | " + " | ".join(cells) + " |")
  L += ["", "### 정확한 앞·뒤 밂 (방향 0°·180°; 넘어짐 · 회복 s)", "| 꼴 | " + " | ".join(pols) + " |", "|---|" + "---|" * len(pols)]
  for mode in ("stand", "walk03", "walk05"):
    cells = []
    for p in pols:
      R_ = [r for r in by[(p, f"v4push_{mode}")] if r.get("dir_deg") in (0, 180)]
      if not R_:
        cells.append("—")
        continue
      cells.append(" ".join(f"{'앞' if r['dir_deg'] == 0 else '뒤'}{int(r['J'])}{'' if not r.get('phase') else '′'}:{'F' if r['F'] else f(r.get('recovery_s'), 1)}"
                            for r in sorted(R_, key=lambda r: (r["dir_deg"], r["J"], r.get("phase", 0)))))
    L.append(f"| {mode} | " + " | ".join(cells) + " |")
  L.append(f"\n오류 행 {len(err)}" + (f": {err[0].get('error')}" if err else ""))
  Path(out).write_text("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  main()
