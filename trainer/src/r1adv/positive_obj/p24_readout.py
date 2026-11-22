"""Positive 24 (E1 c09) frozen readout (rounds/R04/positive/e1_c09/rules.md 5-8).

  python -m r1adv.positive_obj.p24_readout <rows.json> <out.json> <out.md>
Cells = (grip, contact, shake, delay, width, mass), 8 worlds each. held fraction over start-valid worlds; a cell whose
start-valid rate is < 0.5, or a C cell whose hug pose is IK-infeasible, is '불가'. Rule statistic A(grip, contact) =
mean held fraction over the feasible cells with mass <= 1.22 kg (9 cells; 불가 cells excluded = primary, 불가 = 0 also
reported), main panel = shake 'trace', delay 0. H = primary hands-only grip (aligned start); H0 = record frame (legacy
start) used for rule 1's reproduction check; C = hug (edge hook, chest loaded); Cs = side arm (reported, not in rules).
Rule 4 statistic = per (grip, contact) median over start-valid worlds of the internal-force CV over the shake-window
steps where both hands touch the object (>= 0.5 s of such steps, else not measured); max over the six H / C panels.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np

MASSES, WIDTHS = (0.2, 0.6, 1.22, 2.0), (0.2, 0.3, 0.4)
RULE_MASSES = (0.2, 0.6, 1.22)
CONTACT_ORDER = ("K1", "K2", "K3")                  # cost order: base < pads (+3 % sim cost, positive 20) < elliptic (1.6-1.7x)
GRIPS = ("H", "C", "Cs", "H0")
SHAKES, DELAYS = ("trace", "sin493", "sin196"), (0, 12)
MAIN = ("trace", 0)


def nanmed(x):
  x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
  return float(np.median(x)) if x else float("nan")


def cell_table(rows):
  out = {}
  for r in rows:
    key = (r["grip"], r["contact"], r["shake"], r["delay_steps"], r["w"], r["mass"])
    out.setdefault(key, []).append(r)
  cells = {}
  for key, rs in out.items():
    n = len(rs)
    valid = [r for r in rs if r["start_valid"]]
    feas_ik = all(r["feasible"] for r in rs)
    feasible = feas_ik and len(valid) >= 0.5 * n
    held = [r for r in valid if r["held"]]
    cells[key] = dict(n=n, n_valid=len(valid), feasible=feasible, feasible_ik=feas_ik,
                      held=(len(held) / len(valid)) if (feasible and valid) else float("nan"),
                      held_pre=(sum(r["held_pre"] for r in valid) / len(valid)) if (feasible and valid) else float("nan"),
                      slip_med=nanmed([r["slip_main"] for r in held]), slip_shake_med=nanmed([r["slip_shake"] for r in held]),
                      cv_shake_med=nanmed([r["fint_cv_shake"] for r in held]), cv_still_med=nanmed([r["fint_cv_still"] for r in held]),
                      cv_shake_c_med=nanmed([r["fint_cv_shake_c"] for r in valid]), M_start_total_med=nanmed([r["M_start_total"] for r in valid]),
                      hf_shake_med=nanmed([r["fint_hf_shake"] for r in held]), M_start_med=nanmed([r["M_start"] for r in valid]),
                      t_loss_med=nanmed([r["t_loss"] for r in valid if not r["held"]]),
                      align_max=max((max(r["align"]) for r in valid), default=float("nan")),
                      nc_release_min=min((min(r["nc_release"]) for r in valid), default=-1),
                      one_point_med=nanmed([r["one_point_frac"] for r in valid]),
                      chest_frac_med=nanmed([r["chest_frac"] for r in valid]))
  return cells


def panel(cells, grip, ck, sh, dl, rows):
  ks = [(grip, ck, sh, dl, w, m) for w in WIDTHS for m in RULE_MASSES]
  ks = [k for k in ks if k in cells]
  if not ks:
    return None
  feas = [k for k in ks if cells[k]["feasible"]]
  A = float(np.mean([cells[k]["held"] for k in feas])) if feas else float("nan")
  A0 = float(np.mean([cells[k]["held"] if cells[k]["feasible"] else 0.0 for k in ks]))
  sel = [r for r in rows if r["grip"] == grip and r["contact"] == ck and r["shake"] == sh and r["delay_steps"] == dl
         and r["mass"] in RULE_MASSES and r["start_valid"]]
  held = [r for r in sel if r["held"]]
  return dict(A=A, A_infeasible_as_0=A0, n_cells=len(ks), n_infeasible=len(ks) - len(feas),
              infeasible_cells=[(k[4], k[5]) for k in ks if not cells[k]["feasible"]],
              n_worlds=len(sel), n_held=len(held), slip_med=nanmed([r["slip_main"] for r in held]),
              slip_shake_med=nanmed([r["slip_shake"] for r in held]), cv_shake_med=nanmed([r["fint_cv_shake"] for r in held]),
              cv_still_med=nanmed([r["fint_cv_still"] for r in held]), hf_shake_med=nanmed([r["fint_hf_shake"] for r in held]),
              cv_shake_c_med=nanmed([r["fint_cv_shake_c"] for r in sel]), cv_still_c_med=nanmed([r["fint_cv_still_c"] for r in sel]),
              n_cv_c=sum(1 for r in sel if not math.isnan(r["fint_cv_shake_c"])),
              M_start_total_med=nanmed([r["M_start_total"] for r in sel]),
              held_pre=float(np.mean([r["held_pre"] for r in sel])) if sel else float("nan"))


def rules(P) -> dict:
  """c09 decision rules 1-4, applied literally to the panel statistics P[(grip, contact)] (rules.md 7)."""
  def A(g, k):
    p = P.get((g, k))
    return p["A"] if p else float("nan")
  ok = lambda x: not math.isnan(x)                                     # noqa: E731
  out = {"inputs": {f"{g}|{k}": A(g, k) for g in ("H", "C", "H0") for k in CONTACT_ORDER}}
  c_vals = [A("C", k) for k in CONTACT_ORDER]
  if not any(ok(v) for v in c_vals):
    out["rule"] = "적용 불가"
    out["text"] = "C 의 가능 칸이 없음(모든 C 칸 불가) → 규칙 ①–③ 적용 불가"
  elif ok(A("C", "K1")) and A("C", "K1") >= 0.8:
    out["rule"] = "①"
    out["text"] = "C·기본 ≥ 0.8 → 껴안기를 집기 정의로 삼고 pyramidal 유지"
    h0 = A("H0", "K1")
    out["frame_check_H0_K1"] = h0
    if ok(h0) and h0 > 0.2:
      out["text"] += f"; H0·기본(기록 틀) {h0:.3f} > 0.2 → 기록이 재현되지 않음, 시험 틀 점검"
  else:
    cheapest = next((k for k in CONTACT_ORDER if ok(A("C", k)) and A("C", k) >= 0.8), None)
    if cheapest is not None:
      out["rule"] = "②"
      out["cheapest_contact"] = cheapest
      out["text"] = f"C ≥ 0.8 인 가장 싼 접촉 설정 = {cheapest} → 먼저 깔고 그 전에는 집기를 학습하지 않음"
      if ok(A("H", cheapest)) and A("H", cheapest) >= 0.8:
        out["text"] += f"; H 도 {A('H', cheapest):.3f} ≥ 0.8 → 쥐는 형태는 학습에 맡김"
    else:
      best = max(v for v in c_vals if ok(v))
      if best < 0.5:
        out["rule"] = "③-기각"
        out["text"] = f"모든 설정에서 C < 0.5 (최고 {best:.3f}) → 관점 기각, 집기는 밀기·끌기로"
      else:
        out["rule"] = "③-재시험"
        out["text"] = f"최고 C = {best:.3f} (0.5–0.8) → impratio 30·100 을 넣어 다시 시험"
  out["frame_check_H0_K1_value"] = A("H0", "K1")
  cvs = {f"{g}|{k}": P[(g, k)]["cv_shake_c_med"] for g in ("H", "C") for k in CONTACT_ORDER if P.get((g, k))}
  meas = [v for v in cvs.values() if ok(v)]
  out["rule4_inputs_cv_shake_pct"] = cvs
  if meas and max(meas) >= 20.0:
    out["rule4"] = f"흔들림 최대 {max(meas):.1f} % ≥ 20 % → 팔 강성 양 끝 시험 필수"
  elif meas and max(meas) < 5.0:
    out["rule4"] = f"흔들림 최대 {max(meas):.1f} % < 5 % → 떨림 주장 철회"
  elif meas:
    out["rule4"] = f"흔들림 최대 {max(meas):.1f} % (5–20 %) → 규칙 문구 없음"
  else:
    out["rule4"] = "흔들림 측정 불가(두 손 접촉 구간 없음)"
  return out


def predictions(P) -> dict:
  A = lambda g, k: P[(g, k)]["A"] if P.get((g, k)) else float("nan")      # noqa: E731
  cv = P[("H", "K1")]["cv_shake_c_med"] if P.get(("H", "K1")) else float("nan")
  cv0 = P[("H0", "K1")]["cv_shake_c_med"] if P.get(("H0", "K1")) else float("nan")
  return {"C·기본 ≥ 0.8": [A("C", "K1"), A("C", "K1") >= 0.8], "H·기본 ≤ 0.1": [A("H", "K1"), A("H", "K1") <= 0.1],
          "H0·기본 ≤ 0.1": [A("H0", "K1"), A("H0", "K1") <= 0.1], "패치가 H 를 ≥ 0.5 로": [A("H", "K2"), A("H", "K2") >= 0.5],
          "패치가 H0 를 ≥ 0.5 로": [A("H0", "K2"), A("H0", "K2") >= 0.5], "H 내부력 흔들림 ≥ 20 %": [cv, cv >= 20.0],
          "H0 내부력 흔들림 ≥ 20 %": [cv0, cv0 >= 20.0]}


def fmt(x, nd=2):
  return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def main(argv):
  data = json.load(open(argv[0]))
  rows = data["rows"]
  cells = cell_table(rows)
  panels = {}
  for g in GRIPS:
    for k in CONTACT_ORDER:
      for sh in SHAKES:
        for dl in DELAYS:
          p = panel(cells, g, k, sh, dl, rows)
          if p is not None:
            panels[(g, k, sh, dl)] = p
  res = dict(meta=data["meta"], panels={"|".join(map(str, k)): v for k, v in panels.items()},
             cells={"|".join(map(str, k)): v for k, v in cells.items()})
  by = {}
  for sh in SHAKES:
    for dl in DELAYS:
      P = {(g, k): panels.get((g, k, sh, dl)) for g in ("H", "C", "H0") for k in CONTACT_ORDER}
      by[f"{sh}|{dl}"] = dict(rules=rules(P), predictions=predictions(P))
  res["verdict_main"] = by[f"{MAIN[0]}|{MAIN[1]}"]
  res["rules_other_panels"] = {k: v for k, v in by.items() if k != f"{MAIN[0]}|{MAIN[1]}"}
  json.dump(res, open(argv[1], "w"), indent=1, ensure_ascii=False, default=float)
  # markdown tables
  L = []
  for sh in SHAKES:
    for dl in DELAYS:
      L.append(f"\n### 흔들림 {sh}, 지연 {dl * 5} ms" + ("  (주 판)" if (sh, dl) == MAIN else ""))
      L.append("| 쥠·접촉 | A(≤1.22, 가능 칸) | A(불가=0) | 불가 칸 | 미끄럼 중앙 mm | 내부력 CV % (흔들림, 접촉 구간) | CV % (정지, 접촉 구간) | >10 Hz % (유지 세계) | held_pre |")
      L.append("|---|---|---|---|---|---|---|---|---|")
      for g in GRIPS:
        for k in CONTACT_ORDER:
          p = panels.get((g, k, sh, dl))
          if p is None:
            continue
          L.append(f"| {g}·{k} | {fmt(p['A'], 3)} | {fmt(p['A_infeasible_as_0'], 3)} | {p['n_infeasible']} | {fmt(p['slip_med'], 1)} | "
                   f"{fmt(p['cv_shake_c_med'], 1)} | {fmt(p['cv_still_c_med'], 1)} | {fmt(p['hf_shake_med'], 1)} | {fmt(p['held_pre'], 3)} |")
  L.append("\n### 칸별 held (주 판: trace, 지연 0; 행 = 쥠·접촉·폭, 열 = 질량 kg)")
  L.append("| 쥠·접촉·폭 | " + " | ".join(str(m) for m in MASSES) + " |")
  L.append("|---|" + "---|" * len(MASSES))
  for g in GRIPS:
    for k in CONTACT_ORDER:
      for w in WIDTHS:
        vals = []
        for m in MASSES:
          c = cells.get((g, k, MAIN[0], MAIN[1], w, m))
          if c is None:
            break
          vals.append("불가" if not c["feasible"] else f"{c['held']:.2f}")
        if len(vals) == len(MASSES):
          L.append(f"| {g}·{k}·{w} | " + " | ".join(vals) + " |")
  open(argv[2], "w").write("\n".join(L) + "\n")
  print(json.dumps(res["verdict_main"], ensure_ascii=False, indent=1, default=float))


if __name__ == "__main__":
  main(sys.argv[1:])
