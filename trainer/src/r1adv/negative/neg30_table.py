"""Negative 30: pass table (합격표) from neg30_pass.py rows.

Rule (fixed before values, 08_pass_table_design.md): an episode meets an item when its value is inside the pass line;
walking / command items also need the episode's mission G (no fall and steady forward ratio >= 0.7), so standing still
cannot pass them. A checkpoint passes an item when >= 90 % of its episodes meet it (charter 'solved' share); count items
(falls, two-foot hops) pass only at 0 / n. Push recovery: P24 statistic, every per-J median <= line. A 3-checkpoint window
is reported as the median of the per-checkpoint statistic with its [min, max]; the window passes when that median passes.
Tiers (coordinator 10:5x, tracker 47): main loop = noise-free (deterministic, n = 1 per mission, 16 pushes) and 'real-level'
noise D2 x 0.1 (provisional, n = 32); robustness tier = D2 noise, lines max(1.5 x manufacturer, line); an item the
manufacturer itself fails in a tier is marked unsolvable there (not a gate).
Coordinator 11:2x (tracker 54): push recovery line = max(1.0, manufacturer's value in the same loop); side / back 0.3 are
user-goal columns (manufacturer stands: no witness), not gates; a moves segment counts only with realised >= 0.7 AND a
foot-lift share (one sole > 1 cm) >= max(0.10, 0.5 x manufacturer's share in the same loop; fwd 0.3 value for side / back),
so dragging / pivoting on the soles does not count as walking.

  python -I neg30_table.py OUT.md OUT.json RUN_DIR [RUN_DIR ...]
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

WINDOWS = {"C3": ["C3s1000", "C3s1250", "C3s1500"], "C2": ["C2s1000", "C2s1250", "C2s1500"]}
SINGLE = ["mf", "B0", "Bppr500"]
LIM0 = {"stand_drift": 0.01, "arm": 0.15, "pelvis": 0.06, "roll": 1.0, "pitch": 1.0, "overshoot": 15.0, "t50": 0.5, "stop": 0.5,
        "push_recovery": 1.0, "mv": 0.7, "lift_fwd": 0.10, "lift_turn": 0.10}
VALKEY = {"arm": "arm_swing_rad", "pelvis": "pelvis_lat_pp_m", "roll": "roll_rms_deg", "pitch": "pitch_rms_deg",
          "overshoot": "overshoot_pct", "t50": "t50_s", "stop": "stop_s"}
SHARE_ITEMS = ("stand_steps", "stand_drift", "walk_G", "arm", "pelvis", "roll", "pitch", "overshoot", "t50", "stop",
               "mv_fwd0.3", "mv_left0.3", "mv_back0.3", "mv_turn0.8")
LABEL = {"stand_steps": "서기 디딤 0", "stand_drift": "서기 떠밀림 ≤ {stand_drift} m", "push_falls": "밂 12–25 넘어짐 0",
         "push_recovery": "밂 회복 ≤ {push_recovery} s (J 별 중앙 최댓값)", "walk_G": "[G] 앞 0.5 실현 ≥ 0.7·넘어짐 없음",
         "arm": "팔 ≤ {arm} rad", "pelvis": "골반 좌우 p-p ≤ {pelvis} m", "roll": "roll RMS ≤ {roll}°", "pitch": "pitch RMS ≤ {pitch}°",
         "walk_hops": "두 발 뛰기 0 (걷기 판)", "overshoot": "넘침 ≤ {overshoot} %", "t50": "50 % 도달 ≤ {t50} s", "stop": "정지 ≤ {stop} s",
         "mv_fwd0.3": "[지킴] 앞 0.3 실현 ≥ 0.7 ∧ 발 뗌 ≥ {lift_fwd}", "mv_left0.3": "[사용자 목표·관문 아님] 옆 0.3 실현 ≥ 0.7 ∧ 발 뗌 ≥ {lift_fwd}",
         "mv_back0.3": "[사용자 목표·관문 아님] 뒤 0.3 실현 ≥ 0.7 ∧ 발 뗌 ≥ {lift_fwd}", "mv_turn0.8": "[지킴] 제자리 돌기 실현 ≥ 0.7 ∧ 발 뗌 ≥ {lift_turn}",
         "push_hops": "덧: 밂 뒤 두 발 뛰기 수", "step_hz": "덧: 걸음 빈도 (착지/s)", "walk_jerk": "덧: 걷기 jerk 중앙 (rad/s³)",
         "stand_jerk": "덧: 서기 jerk 중앙", "slip_p95": "덧: 디딘 발 미끄럼 p95 (m/s)", "realised": "덧: 실현 앞 속도 (m/s)",
         "ref_g1": "덧: G1 계획기 참조 속도 / 명령", "walk_falls": "덧: 걷기 판 넘어짐", "replan_robot": "덧: 로봇 문맥 재계획 수 (판 중앙)",
         "sym_hip_yaw": "덧: 대칭 참조 서기 |엉덩이 yaw L+R| (°)", "sym_waist_roll": "덧: 대칭 참조 서기 |허리 roll| (°)",
         "sym_pelvis_roll": "덧: 대칭 참조 서기 |골반 roll| (°)"}
VIEWS = (  # title, walk cell, stand cell, moves cell, manufacturer cell, tier
  ("주 고리 A — 잡음 없음 · 열린 고리 m1c · 다리 0.1 s · 행동 지연 0 (결정론: 미션당 n = 1, 밂 16)", "ol_L01_d00_nf", "ol_L01_d00_nf", "ol_L01_d00_nf", "ol_L01_d00_nf", "main"),
  ("주 고리 B — 실기 수준 잡음 D2 × 0.1 (잠정, 실측 근거 없음) · 열린 고리 · 다리 0.1 s · 지연 0", "ol_L01_d00_lo", "ol_L01_d00_lo", "ol_L01_d00_lo", "ol_L01_d00_lo", "main"),
  ("추적 행 A — 잡음 없음 · 닫힌 고리 m1cctx · 다리 0.1 s · 지연 0", "cl_L01_d00_nf", None, "cl_L01_d00_nf", "ol_L01_d00_nf", "main"),
  ("추적 행 B — D2 × 0.1 · 닫힌 고리 m1cctx · 다리 0.1 s · 지연 0", "cl_L01_d00_lo", None, "cl_L01_d00_lo", "ol_L01_d00_lo", "main"),
  ("강건성 단계 — D2 잡음 · 열린 고리 · 다리 0.1 s · 지연 0 (선 = max(1.5 × 제조사, 선))", "ol_L01_d00", "ol_L01_d00", "ol_L01_d00", "ol_L01_d00", "robust"),
  ("강건성 단계 — D2 잡음 · 닫힌 고리 · 다리 0.1 s · 지연 0", "cl_L01_d00", None, "cl_L01_d00", "ol_L01_d00", "robust"))
ITEMS_STAND = ["stand_steps", "stand_drift", "push_falls", "push_recovery"]
ITEMS_WALK = ["walk_G", "arm", "pelvis", "roll", "pitch", "walk_hops", "overshoot", "t50", "stop", "mv_fwd0.3", "mv_left0.3",
              "mv_back0.3", "mv_turn0.8", "step_hz", "walk_jerk", "slip_p95", "realised", "ref_g1", "walk_falls", "replan_robot"]
ITEMS_EXTRA = ["push_hops", "stand_jerk", "sym_hip_yaw", "sym_waist_roll", "sym_pelvis_roll"]


def load(dirs):
  rows, errs = [], []
  for d in dirs:
    p = Path(d) / "rows.jsonl"
    if not p.exists():
      continue
    for l in p.read_text().splitlines():
      if l.strip():
        r = json.loads(l)
        (errs if "error" in r else rows).append(r)
  seen, out = set(), []
  for r in rows:
    k = (r["spec_sha256"], r["policy"])
    if k not in seen:
      seen.add(k)
      out.append(r)
  return out, errs


def share(xs):
  return (sum(xs) / len(xs)) if xs else None


def med(xs):
  xs = [x for x in xs if x is not None and math.isfinite(x)]
  return float(np.median(xs)) if xs else None


def pct(xs, q):
  xs = [x for x in xs if x is not None and math.isfinite(x)]
  return float(np.percentile(xs, q)) if xs else None


def f(x, nd=2):
  return "—" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def sel(rows, pol, fam, cell):
  return [r for r in rows if r["policy"] == pol and r["family"] == fam and r["cell"] == cell]


def items_for(rows, pol, cw, cs, cm, lim):
  R = {}
  W, P, M, Z = sel(rows, pol, "walk_stop", cw), sel(rows, pol, "stand_push", cs), sel(rows, pol, "moves", cm), sel(rows, pol, "stand_static", cs)
  if P:
    n = len(P)
    R["stand_steps"] = (share([r["stand_steps"] == 0 for r in P]), None, f"0 디딤 {sum(r['stand_steps'] == 0 for r in P)}/{n}, 최대 {max(r['stand_steps'] for r in P)}")
    dm, d90 = med([r["stand_drift_m"] for r in P]), pct([r["stand_drift_m"] for r in P], 90)
    R["stand_drift"] = (share([(r["stand_drift_m"] if r["stand_drift_m"] is not None else 9) <= lim["stand_drift"] for r in P]), None,
                        f"중앙 {f(dm * 100 if dm is not None else None, 1)} cm, p90 {f(d90 * 100 if d90 is not None else None, 1)} cm")
    nf = sum(bool(r["push_fell"]) for r in P)
    R["push_falls"] = (float(nf), nf == 0, f"{nf}/{n}")
    byJ = defaultdict(list)
    for r in P:
      byJ[r["push_J"]].append(r["push_recovery_s"] if (not r["push_fell"] and r.get("push_recovery_s") is not None) else float("inf"))
    meds = {J: float(np.median(v)) for J, v in byJ.items()}
    worst = max(meds.values()) if meds else None
    R["push_recovery"] = (worst, worst is not None and worst <= lim["push_recovery"],
                          " / ".join(f"{int(J)}: {f(meds[J]) if math.isfinite(meds[J]) else '∞'}" for J in sorted(meds)))
    R["push_hops"] = (float(sum(r.get("push_hops", 0) for r in P)), None, "")
    R["stand_jerk"] = (med([r.get("stand_jerk_med") for r in P]), None, "")
  if W:
    n = len(W)
    G = [bool(r.get("G_walk")) for r in W]
    R["walk_G"] = (share(G), None, f"{sum(G)}/{n}, 실현 중앙 {f(med([r.get('steady_ratio') for r in W]))}")
    for item, key in VALKEY.items():
      l_ = lim[item]
      ok = [bool(r.get("G_walk")) and r.get(key) is not None and r[key] <= l_ for r in W]
      nd = 3 if l_ < 1 else 2
      R[item] = (share(ok), None, f"G 판 중앙 {f(med([r.get(key) for r in W if r.get('G_walk')]), nd)} (전체 {f(med([r.get(key) for r in W]), nd)})")
    nh = sum(r.get("hops", 0) for r in W)
    R["walk_hops"] = (float(nh), nh == 0, f"{nh} 회 / {n} 판")
    R["step_hz"] = (med([r.get("step_hz") for r in W]), None, "")
    R["walk_jerk"] = (med([r.get("jerk_walk_med") for r in W]), None, "")
    R["slip_p95"] = (med([r.get("slip_p95_walk") for r in W]), None, "")
    R["ref_g1"] = (med([r.get("ref_g1_steady_ratio") for r in W]), None, "")
    R["realised"] = (med([r.get("realised_fwd_mps") for r in W]), None, "")
    R["walk_falls"] = (float(sum(bool(r["F"]) for r in W)), None, f"{sum(bool(r['F']) for r in W)}/{n}")
    R["replan_robot"] = (med([r.get("replan_robot") for r in W]), None, f"self {med([r.get('replan_self') for r in W])}")
  if M:
    for lab in ("fwd0.3", "left0.3", "back0.3", "turn0.8"):
      segs = [next((s for s in r["segments"] if s.get("label") == lab), None) for r in M]
      rz = [s.get("realised") if s and s.get("complete") else None for s in segs]
      sw = [s.get("swing_share") if s else None for s in segs]
      dist = [(s.get("dist_m") if "dist_m" in s else s.get("turn_rad")) if s else None for s in segs]
      thr_l = lim["lift_turn"] if lab == "turn0.8" else lim["lift_fwd"]
      R["mv_" + lab] = (share([x is not None and x >= lim["mv"] and w_ is not None and w_ >= thr_l for x, w_ in zip(rz, sw)]), None,
                        f"실현 {f(med(rz))}, {'거리 m' if lab != 'turn0.8' else '회전 rad'} {f(med(dist))}, 발 뗌 {f(med(sw))}")
  if Z:
    for it, key, nd in (("sym_hip_yaw", "sym_hip_yaw_LR", 1), ("sym_waist_roll", "sym_waist_roll", 2), ("sym_pelvis_roll", "sym_pelvis_roll", 2)):
      vals = [r[key] for r in Z if key in r]
      R[it] = (med([abs(v) for v in vals]), None, f"중앙 {f(med(vals), nd)}°")
  R["_n"] = (None, None, dict(walk=len(W), stand=len(P), moves=len(M), static=len(Z)))
  return R


def lift_lims(rows, cm, lim):
  M = sel(rows, "mf", "moves", cm)
  for lab, key in (("fwd0.3", "lift_fwd"), ("turn0.8", "lift_turn")):
    sw = [next((s_.get("swing_share") for s_ in r["segments"] if s_.get("label") == lab), None) for r in M]
    m = med(sw)
    if m is not None:
      lim[key] = round(max(0.10, 0.5 * m), 3)
  return lim


def main_lims(rows, cmf):
  """Main tier: the pre-fixed lines, with push recovery = max(1.0, manufacturer's worst per-J median in the same view)
  (coordinator 11:2x, tracker 54) and the foot-lift thresholds from the manufacturer (half its share)."""
  lim = dict(LIM0)
  P_ = sel(rows, "mf", "stand_push", cmf)
  byJ = defaultdict(list)
  for r in P_:
    byJ[r["push_J"]].append(r["push_recovery_s"] if (not r["push_fell"] and r.get("push_recovery_s") is not None) else float("inf"))
  if byJ:
    w_ = max(float(np.median(v)) for v in byJ.values())
    if math.isfinite(w_):
      lim["push_recovery"] = round(max(LIM0["push_recovery"], w_), 3)
  return lift_lims(rows, cmf, lim)


def robust_lims(rows, cw, cs, cm):
  """D2 tier: line' = max(1.5 x manufacturer median, line) per item (median over its episodes)."""
  lim = dict(LIM0)
  W, P = sel(rows, "mf", "walk_stop", cw), sel(rows, "mf", "stand_push", cs)
  for item, key in VALKEY.items():
    m = med([r.get(key) for r in W if r.get("G_walk")])
    if m is not None:
      lim[item] = max(LIM0[item], 1.5 * m)
  m = med([r.get("stand_drift_m") for r in P])
  if m is not None:
    lim["stand_drift"] = max(LIM0["stand_drift"], 1.5 * m)
  byJ = defaultdict(list)
  for r in P:
    if not r["push_fell"] and r.get("push_recovery_s") is not None:
      byJ[r["push_J"]].append(r["push_recovery_s"])
  if byJ:
    lim["push_recovery"] = max(LIM0["push_recovery"], 1.5 * max(float(np.median(v)) for v in byJ.values()))
  return lift_lims(rows, cm, lim)


def passes(st, item):
  stat, passed, _d = st
  if item in SHARE_ITEMS:
    return stat is not None and stat >= 0.9
  return passed


def cell_text(st, item):
  stat, passed, detail = st
  if stat is None and not detail:
    return "—"
  if item in SHARE_ITEMS:
    return f"{'✓' if stat is not None and stat >= 0.9 else '✗'} {f(stat * 100 if stat is not None else None, 0)} % · {detail}"
  if passed is not None:
    return f"{'✓' if passed else '✗'} {detail}"
  return f"{f(stat, 3 if stat is not None and abs(stat) < 1 else 2)}" + (f" ({detail})" if detail else "")


def window_text(per, item):
  vals = [p[item][0] for p in per if item in p and p[item][0] is not None]
  if not vals:
    return "—"
  m, lo, hi = float(np.median(vals)), min(vals), max(vals)
  if item in SHARE_ITEMS:
    return f"{'✓' if m >= 0.9 else '✗'} {m * 100:.0f} % [{lo * 100:.0f}–{hi * 100:.0f}]"
  if item in ("push_falls", "walk_hops"):
    return f"{'✓' if hi == 0 else '✗'} {m:.0f} [{lo:.0f}–{hi:.0f}]"
  if item == "push_recovery":                                # window rule = median passes = majority of the 3 passes
    pp = [bool(p[item][1]) for p in per if item in p]
    return f"{'✓' if 2 * sum(pp) > len(pp) else '✗'} {m:.2f} [{lo:.2f}–{hi:.2f}]"
  nd = 3 if abs(m) < 1 else 2
  return f"{m:.{nd}f} [{lo:.{nd}f}–{hi:.{nd}f}]"


def main():
  out_md, out_json, dirs = sys.argv[1], sys.argv[2], sys.argv[3:]
  rows, errs = load(dirs)
  pols = sorted({r["policy"] for r in rows})
  L, res = [], {}
  for title, cw, cs, cm, cmf, tier in VIEWS:
    lim = robust_lims(rows, cmf, cmf, cmf) if tier == "robust" else main_lims(rows, cmf)
    per = {p: items_for(rows, p, cmf if p == "mf" else cw, (cmf if p == "mf" else cs) or "-", cmf if p == "mf" else cm, lim) for p in pols}
    per = {p: d for p, d in per.items() if any(v for v in d["_n"][2].values())}
    if not per or all(p == "mf" for p in per):
      continue
    res[title] = dict(lim=lim, per=per)
    inwin = {p for w in WINDOWS.values() for p in w}
    singles = [c for c in SINGLE if c in per] + sorted(p for p in per if p not in SINGLE and p not in inwin)
    cols = singles + [w for w in WINDOWS if any(p in per for p in WINDOWS[w])]
    L += ["", f"### {title}", "| 항목 | " + " | ".join((c if c not in WINDOWS else f"{c} 창 1000·1250·1500") for c in cols) + " |",
          "|---|" + "---|" * len(cols)]
    items = (ITEMS_STAND if cs else []) + ITEMS_WALK + (ITEMS_EXTRA if cs else [])
    for it in items:
      cells = []
      for c in cols:
        if c not in WINDOWS:
          st = per.get(c, {}).get(it)
          cells.append(cell_text(st, it) if st else "—")
        else:
          cells.append(window_text([per[p] for p in WINDOWS[c] if p in per], it))
      if all(x == "—" for x in cells):
        continue
      lab = LABEL.get(it, it).format(**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in lim.items()})
      if tier == "robust" and "mf" in per and it in per["mf"] and it in SHARE_ITEMS + ("push_falls", "push_recovery", "walk_hops") \
         and per["mf"][it][0] is not None and not passes(per["mf"][it], it):
        lab += " **[제조사도 못 지킴 → 풀 수 없음, 관문 아님]**"
      L.append(f"| {lab} | " + " | ".join(cells) + " |")
    L.append("\nn (판 수): " + " · ".join(f"{p} {per[p]['_n'][2]}" for p in per))
    if tier == "main" and cs:
      L += ["", f"#### 저장본별 ({title.split(' — ')[0]})", "| 항목 | " + " | ".join(per) + " |", "|---|" + "---|" * len(per)]
      for it in ITEMS_STAND + ITEMS_WALK[:13] + ["step_hz", "realised"]:
        L.append(f"| {LABEL.get(it, it).format(**lim)} | " + " | ".join(cell_text(per[p][it], it) if it in per[p] else "—" for p in per) + " |")
  # latency sweeps (walk_stop): noise-free deterministic cells, then D2 x 0.1 cells
  for tag, title in (("_nf", "잡음 없음, 결정론"), ("_lo", "D2 × 0.1, 판 중앙")):
    cells = sorted({r["cell"] for r in rows if r["family"] == "walk_stop" and r["cell"].endswith(tag)})
    if not cells:
      continue
    L += ["", f"### 지연 훑기 ({title}): 앞 0.5 정상 실현 / t50 s / 정지 s / roll °",
          "| 칸 | " + " | ".join(pols) + " |", "|---|" + "---|" * len(pols)]
    for c in cells:
      cs_ = []
      for p in pols:
        W = sel(rows, p, "walk_stop", c)
        cs_.append("—" if not W else f"{f(med([r.get('steady_ratio') for r in W]))} / {f(med([r.get('t50_s') for r in W]))} / "
                   f"{f(med([r.get('stop_s') for r in W]))} / {f(med([r.get('roll_rms_deg') for r in W]), 1)}"
                   + (f" F{sum(bool(r['F']) for r in W)}" if any(r["F"] for r in W) else ""))
      L.append(f"| {c} | " + " | ".join(cs_) + " |")
  L.append(f"\n오류 행 {len(errs)} 개" + (f": {errs[0].get('error')}" if errs else ""))
  Path(out_md).write_text("\n".join(L) + "\n")
  json.dump(res, open(out_json, "w"), indent=1, default=str)
  print("\n".join(L))


if __name__ == "__main__":
  main()
