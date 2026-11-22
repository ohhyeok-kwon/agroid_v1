"""Positive 24 E3 h20 frozen readout (rounds/R04/positive/e3_h20/rules.md 4-5).

  python -m r1adv.positive_obj.p24_h20_readout <h20_rows.json> <out.json> <out.md>
Cell = (grip, contact, width, mass), 8 worlds. Cell held(T) = held share over start-valid worlds; '불가' = IK infeasible or
start-valid rate < 0.5 (c09 rule). A_T(grip, contact) = mean cell held(T) over the feasible cells (primary; 불가 = 0 also
reported). h20 reads A_20(C, K1); a01 reads the share of feasible C·K1 cells whose held(30) >= 0.5.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

CELLS = [("C", "K1"), ("C", "K3"), ("H", "K3"), ("H", "K2"), ("C", "K2")]
ORDER = ("K1", "K2", "K3")                       # cost order (c09 rules 7)
WIDTHS, MASSES = (0.2, 0.3), (0.2, 0.6, 1.22)
MARKS = ("5", "10", "15", "20", "25", "30")


def med(x):
  x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
  return float(np.median(x)) if x else float("nan")


def main(argv):
  data = json.load(open(argv[0]))
  rows = data["rows"]
  cells = {}
  for g, k in CELLS:
    for w in WIDTHS:
      for m in MASSES:
        rs = [r for r in rows if r["grip"] == g and r["contact"] == k and abs(r["w"] - w) < 1e-9 and abs(r["mass"] - m) < 1e-9]
        valid = [r for r in rs if r["start_valid"]]
        feas = all(r["feasible"] for r in rs) and len(valid) >= 0.5 * len(rs)
        cells[(g, k, w, m)] = dict(
          n=len(rs), n_valid=len(valid), feasible=feas,
          held20=(sum(r["held20"] for r in valid) / len(valid)) if (feas and valid) else float("nan"),
          held30=(sum(r["held30"] for r in valid) / len(valid)) if (feas and valid) else float("nan"),
          slip_med={t: med([r["slip_curve_mm"][t] for r in valid]) for t in MARKS},
          t50_med=med([r["t50_after_release_s"] for r in valid]),
          crossed_share=(sum(1 for r in valid if r["t50_after_release_s"] == r["t50_after_release_s"]) / len(valid)) if valid else float("nan"))
  panels = {}
  for g, k in CELLS:
    ks = [(g, k, w, m) for w in WIDTHS for m in MASSES]
    fe = [c for c in ks if cells[c]["feasible"]]
    for T in ("20", "30"):
      key = f"held{T}"
      panels[(g, k, T)] = dict(A=float(np.mean([cells[c][key] for c in fe])) if fe else float("nan"),
                               A0=float(np.mean([cells[c][key] if cells[c]["feasible"] else 0.0 for c in ks])),
                               n_infeasible=len(ks) - len(fe),
                               cells_held_ge_half=sum(1 for c in fe if cells[c][key] >= 0.5), n_feasible=len(fe))
  A20 = {k: panels[("C", k, "20")]["A"] for k in ORDER}
  ok = lambda x: not math.isnan(x)                                    # noqa: E731
  # h20 rule (20 s)
  a = A20["K1"]
  if ok(a) and a >= 0.8:
    h20 = dict(rule="① 유지 확인", text=f"20 s C·기본 {a:.3f} ≥ 0.8 → '① 유지 확인'")
  elif ok(a) and a < 0.5:
    cheapest = next((k for k in ORDER if ok(A20[k]) and A20[k] >= 0.8), None)
    h20 = dict(rule="② 로 바꿈", cheapest=cheapest,
               text=f"20 s C·기본 {a:.3f} < 0.5 → '② 로 바꿈': 20 s 에서 C ≥ 0.8 인 가장 싼 접촉 = {cheapest or '없음'}")
  else:
    h20 = dict(rule="보류", text=f"20 s C·기본 {a:.3f} (0.5–0.8) → 보류")
  # a01 rule (30 s): share of feasible pyramidal-base (C·K1) cells held at 30 s (cell held >= 0.5)
  p = panels[("C", "K1", "30")]
  share = p["cells_held_ge_half"] / p["n_feasible"] if p["n_feasible"] else float("nan")
  if ok(share) and share < 0.5:
    a01 = dict(rule="① 철회", share=share, text=f"30 s pyramidal(C·기본) 유지 칸 {p['cells_held_ge_half']}/{p['n_feasible']} < 1/2 → '① 철회(집기 계열 elliptic)'")
  elif ok(share) and share >= 0.75:
    a01 = dict(rule="① 유지", share=share, text=f"30 s pyramidal(C·기본) 유지 칸 {p['cells_held_ge_half']}/{p['n_feasible']} ≥ 3/4 → '① 유지'")
  else:
    a01 = dict(rule="밂 켜고 재측정", share=share, text=f"30 s pyramidal(C·기본) 유지 칸 {p['cells_held_ge_half']}/{p['n_feasible']} (1/2–3/4) → '밂 켜고 재측정'")
  agree = (h20["rule"] == "① 유지 확인" and a01["rule"] == "① 유지") or (h20["rule"] == "② 로 바꿈" and a01["rule"] == "① 철회")
  combined = dict(agree=agree, text=("두 읽기가 같은 쪽: " + h20["rule"] + " / " + a01["rule"]) if agree else
                  ("두 읽기가 어긋남 → 20 s 읽기를 따름(조율자 값 전 해석): " + h20["text"]))
  res = dict(meta=data["meta"], cells={"|".join(map(str, k)): v for k, v in cells.items()},
             panels={"|".join(map(str, k)): v for k, v in panels.items()}, A20=A20, h20=h20, a01=a01, combined=combined)
  json.dump(res, open(argv[1], "w"), indent=1, ensure_ascii=False, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x)
  L = ["| 쥠·접촉 | A20(가능 칸) | A20(불가=0) | A30(가능 칸) | A30(불가=0) | 불가 칸 | held30 ≥ 0.5 칸 |", "|---|---|---|---|---|---|---|"]
  for g, k in CELLS:
    p20, p30 = panels[(g, k, "20")], panels[(g, k, "30")]
    L.append(f"| {g}·{k} | {p20['A']:.3f} | {p20['A0']:.3f} | {p30['A']:.3f} | {p30['A0']:.3f} | {p20['n_infeasible']} | {p30['cells_held_ge_half']}/{p30['n_feasible']} |")
  L.append("\n| 쥠·접촉·폭·질량 | 유효 | held20 | held30 | 미끄럼 중앙 mm @5/10/15/20/25/30 s | 50 mm 넘은 몫 | 처음 넘은 시각 중앙 s |")
  L.append("|---|---|---|---|---|---|---|")
  for (g, k, w, m), c in cells.items():
    sl = "/".join("—" if math.isnan(c["slip_med"][t]) else f"{c['slip_med'][t]:.0f}" for t in MARKS)
    hd = lambda x: "불가" if not c["feasible"] else f"{x:.2f}"           # noqa: E731
    t50s = "—" if math.isnan(c["t50_med"]) else f"{c['t50_med']:.1f}"
    L.append(f"| {g}·{k}·{w}·{m} | {c['n_valid']}/{c['n']} | {hd(c['held20'])} | {hd(c['held30'])} | {sl} | "
             f"{c['crossed_share']:.2f} | {t50s} |")
  L.append(f"\n- h20(20 s): {h20['text']}\n- a01(30 s): {a01['text']}\n- 함께 읽기: {combined['text']}")
  open(argv[2], "w").write("\n".join(L) + "\n")
  print(json.dumps(dict(A20=A20, h20=h20, a01=a01, combined=combined), ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
  main(sys.argv[1:])
