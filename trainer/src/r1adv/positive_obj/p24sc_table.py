"""Positive 24 E2 sc readout (rounds/R04/positive/e2_sc/rules.md 2-4): self-collision audit tables, the c08 hazard
table (B0) and the machine computations of the branch wording.

  python p24sc_table.py <out_dir> <result.json> <result.md>
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys

import numpy as np

DT = 0.02
WIN, EARLY = 50, 15                     # push response window 1.0 s, capture-point look 0.3 s (steps)
KINDS = ("hold", "locomotion", "crouch")
MIN_N = 30


def load(out_dir):
  cells = {}
  for f in sorted(glob.glob(os.path.join(out_dir, "*.json"))):
    tag = os.path.basename(f)[:-5]
    d = json.load(open(f))
    z = dict(np.load(f[:-5] + ".npz"))
    cells[tag] = dict(meta=d["meta"], events=d["events"], arr=z)
  return cells


def posture(e):
  if e["rms_jv_01"] < 0.5 and e["vxy"] < 0.15:
    return "quasi"
  if e["rms_jv_01"] >= 1.5 or e["vxy"] >= 0.5:
    return "fast"
  return "mid"


def pair(e):
  b = [n for n, f in e.get("bodies", []) if f > 10.0]
  if len(b) < 2:
    b = [n for n, f in e.get("bodies", [])][:2]
  return "|".join(sorted(b[:2])) if len(b) >= 2 else (b[0] if b else "?")


def push_state(arr):
  """per step x env: 0 none, 1 in, 2 out (rules 3)."""
  push, cpin, alive = arr["push"].astype(bool), arr["cp_in"].astype(bool), arr["alive"].astype(bool)
  S, N = push.shape
  st = np.zeros((S, N), np.int8)
  onset = push & ~np.vstack([np.zeros((1, N), bool), push[:-1]])
  for k0, i in zip(*np.nonzero(onset)):
    k1 = min(S, k0 + WIN)
    early = slice(k0, min(S, k0 + EARLY))
    out = bool((~cpin[early, i] & alive[early, i]).any())
    st[k0:k1, i] = 2 if out else 1
  return st


def audit(events):
  selfs = [e for e in events if e["type"] == "self"]
  phys = [e for e in events if e["type"] in ("self", "ground")]
  out = dict(n=len(events), counts=dict(collections.Counter(e["type"] for e in events)), n_self=len(selfs),
             self_share_of_physical=(len(selfs) / len(phys)) if phys else float("nan"))
  if not selfs:
    return out
  pairs = collections.Counter(pair(e) for e in selfs)
  out["pairs_top"] = [(p, c, round(c / len(selfs), 3)) for p, c in pairs.most_common(6)]
  post = collections.Counter(posture(e) for e in selfs)
  out["posture"] = dict(post)
  for cls in ("quasi", "mid", "fast"):
    sub = [e for e in selfs if posture(e) == cls]
    if sub:
      pc = collections.Counter(pair(e) for e in sub)
      out[f"pairs_{cls}"] = [(p, c, round(c / len(sub), 3)) for p, c in pc.most_common(3)]
  in_push = [e for e in selfs if e["push_age"] <= 1.0]
  out["self_in_push_window"] = len(in_push)
  out["self_in_push_window_share"] = round(len(in_push) / len(selfs), 3)
  out["self_push_fast"] = sum(1 for e in in_push if posture(e) == "fast")
  out["gait"] = dict(collections.Counter({0: "none", 1: "left", 2: "right", 3: "both"}[e["feet"]] for e in selfs))
  q = lambda key: [round(float(np.percentile([e[key] for e in selfs], p)), 3) for p in (10, 50, 90)]     # noqa: E731
  out["t_p10_50_90"] = q("t")
  out["rms_jv_01_p10_50_90"] = q("rms_jv_01")
  out["max_jv_p10_50_90"] = q("max_jv")
  out["vxy_p10_50_90"] = q("vxy")
  out["since_feet_change_p50"] = round(float(np.median([e["since_feet_change"] for e in selfs])), 3)
  out["self_t_le_1s"] = sum(1 for e in selfs if e["t"] <= 1.0)
  if "goal_kind" in selfs[0]:
    out["self_by_goal_kind"] = dict(collections.Counter(KINDS[e["goal_kind"]] for e in selfs))
    out["self_after_deadline"] = sum(1 for e in selfs if e["tau"] <= 0)
  return out


def hazard_table(cells_list):
  """c08 table over a list of B0 cells (pooled): exposure, segments and terminations per group."""
  expo = collections.Counter()
  segs = collections.Counter()
  term = {t: collections.Counter() for t in ("self", "ground")}
  single, alive_total = 0, 0
  for c in cells_list:
    a = c["arr"]
    alive = a["alive"].astype(bool)
    kind = a["kind"].astype(int)
    before = a["tau"].astype(np.float32) > 0
    ps = push_state(a)
    S, N = alive.shape
    g = kind * 6 + (~before) * 3 + ps                       # group id
    for gid, cnt in zip(*np.unique(g[alive], return_counts=True)):
      expo[int(gid)] += int(cnt)
    prev = np.vstack([np.full((1, N), -1), g[:-1]])
    prev_alive = np.vstack([np.zeros((1, N), bool), alive[:-1]])
    start = alive & (~prev_alive | (g != prev))
    for gid, cnt in zip(*np.unique(g[start], return_counts=True)):
      segs[int(gid)] += int(cnt)
    fl, fr = a["feet_l"].astype(bool), a["feet_r"].astype(bool)
    single += int(((fl ^ fr) & alive).sum())
    alive_total += int(alive.sum())
    for e in c["events"]:
      if e["type"] in ("self", "ground"):
        term[e["type"]][int(g[e["step"], e["env"]])] += 1
  rows = {}
  for k in range(3):
    for dl in range(2):
      for p in range(3):
        gid = k * 6 + dl * 3 + p
        ex = expo[gid] * DT
        ns, ng = term["self"][gid], term["ground"][gid]
        n = ns + ng
        rows[f"{KINDS[k]}|{'after' if dl else 'before'}|{('none', 'in', 'out')[p]}"] = dict(
          exposure_s=round(ex, 2), segments=segs[gid], self=ns, ground=ng, physical=n,
          hazard_phys_per_s=(n / ex) if ex > 0 else float("nan"), seg_prob_phys=(n / segs[gid]) if segs[gid] else float("nan"),
          hazard_self_per_s=(ns / ex) if ex > 0 else float("nan"), hazard_ground_per_s=(ng / ex) if ex > 0 else float("nan"),
          hold=n < MIN_N)
  return rows, dict(single_support_share=single / alive_total if alive_total else float("nan"), alive_s=alive_total * DT)


def merge(rows, keys):
  ex = sum(rows[k]["exposure_s"] for k in keys)
  sg = sum(rows[k]["segments"] for k in keys)
  ns = sum(rows[k]["self"] for k in keys)
  ng = sum(rows[k]["ground"] for k in keys)
  n = ns + ng
  return dict(exposure_s=round(ex, 2), segments=sg, self=ns, ground=ng, physical=n, hazard_phys_per_s=n / ex if ex else float("nan"),
              seg_prob_phys=n / sg if sg else float("nan"), hazard_self_per_s=ns / ex if ex else float("nan"), hold=n < MIN_N)


def branch_b0(rows, extra, audit_mean, audit_mean_push_on):
  out = {}
  # (ga) crouch vs locomotion self-collision hazard
  cr = merge(rows, [k for k in rows if k.startswith("crouch|")])
  lo = merge(rows, [k for k in rows if k.startswith("locomotion|")])
  ho = merge(rows, [k for k in rows if k.startswith("hold|")])
  ratio = cr["hazard_self_per_s"] / lo["hazard_self_per_s"] if lo["hazard_self_per_s"] > 0 else float("inf")
  ok = cr["self"] >= MIN_N and lo["self"] >= MIN_N
  out["ga_crouch_vs_loco"] = dict(crouch=cr, locomotion=lo, hold=ho, ratio=ratio,
                                  text=("보류(자기 충돌 < 30)" if not ok else ("≥ 2 → '종료부터 고침'" if ratio >= 2 else "< 2 → 문구 안 걸림")))
  # (na) rule 1: none+in, after / before
  bef = merge(rows, [k for k in rows if "|before|" in k and not k.endswith("|out")])
  aft = merge(rows, [k for k in rows if "|after|" in k and not k.endswith("|out")])
  r1 = aft["hazard_phys_per_s"] / bef["hazard_phys_per_s"] if bef["hazard_phys_per_s"] > 0 else float("inf")
  ok1 = bef["physical"] >= MIN_N and aft["physical"] >= MIN_N
  out["na_rule1"] = dict(before=bef, after=aft, ratio=r1,
                         text=("보류(종료 < 30)" if not ok1 else ("≥ 3 → 'S 재정의 + 목표 잇기'" if r1 >= 3 else "< 3 → 문구 안 걸림")))
  # (da) rule 3: before, none+in segment probability >= 10 %
  p3 = bef["seg_prob_phys"]
  out["da_rule3"] = dict(seg_prob=p3, n=bef["physical"], segments=bef["segments"],
                         text=("보류(종료 < 30)" if bef["physical"] < MIN_N else ("≥ 10 % → 'Φ 기울기 점검'(규칙 3 우선)" if p3 >= 0.10 else "< 10 % → 문구 안 걸림")))
  outc = merge(rows, [k for k in rows if k.endswith("|out")])
  inc = merge(rows, [k for k in rows if k.endswith("|in")])
  out["rule2_ref"] = dict(out_seg_prob=outc["seg_prob_phys"], out_n=outc["physical"], in_seg_prob=inc["seg_prob_phys"], in_n=inc["physical"],
                          single_support=extra["single_support_share"])
  out["la_geometry"] = la(audit_mean)
  out["ma_urgency"] = ma(audit_mean_push_on)
  return out


def la(selfs_events):
  q = [e for e in selfs_events if e["type"] == "self" and posture(e) == "quasi"]
  if len(q) < MIN_N:
    return dict(n_quasi=len(q), text="보류(저속·준정적 자기 충돌 < 30)",
                top=collections.Counter(pair(e) for e in q).most_common(3))
  top = collections.Counter(pair(e) for e in q).most_common(3)
  share = top[0][1] / len(q)
  return dict(n_quasi=len(q), top=top, share=share, text=("≥ 70 % → '충돌 형상 인공물(형상 고침)'" if share >= 0.7 else "< 70 % → 문구 안 걸림"))


def ma(events_push_on):
  s = [e for e in events_push_on if e["type"] == "self"]
  if len(s) < MIN_N:
    return dict(n_self=len(s), text="보류(자기 충돌 < 30)",
                n_push_fast=sum(1 for e in s if e["push_age"] <= 1.0 and posture(e) == "fast"))
  k = sum(1 for e in s if e["push_age"] <= 1.0 and posture(e) == "fast")
  share = k / len(s)
  return dict(n_self=len(s), n_push_fast=k, share=share, text=("≥ 50 % → '다급함(네거티브·판별자 표적)'" if share >= 0.5 else "< 50 % → 문구 안 걸림"))


def r1_push_split(cells_list):
  cnt = collections.Counter()
  expo = collections.Counter()
  for c in cells_list:
    a = c["arr"]
    alive = a["alive"].astype(bool)
    ps = push_state(a)
    for p in range(3):
      expo[p] += int((alive & (ps == p)).sum())
    for e in c["events"]:
      if e["type"] in ("self", "ground"):
        cnt[(e["type"], int(ps[e["step"], e["env"]]))] += 1
  return {("none", "in", "out")[p]: dict(exposure_s=round(expo[p] * DT, 1), self=cnt[("self", p)], ground=cnt[("ground", p)],
                                         hazard_self_per_s=(cnt[("self", p)] / (expo[p] * DT)) if expo[p] else float("nan"))
          for p in range(3)}


def main(argv):
  cells = load(argv[0])
  res = dict(cells={}, pooled={}, meta={k: dict(ck=v["meta"]["checkpoint"], sha=v["meta"]["checkpoint_sha256"][:16], wall=v["meta"]["wall_s"],
                                                steps=v["meta"]["steps_run"], std_mean=float(np.mean(v["meta"]["std"])) if v["meta"]["std"] else None)
                                          for k, v in cells.items()})
  for tag, c in cells.items():
    res["cells"][tag] = audit(c["events"])
  for run in ("r1", "b0"):
    for mode in ("mean", "noise"):
      for push in ("on", "off"):
        lst = [cells[t] for t in cells if t.startswith(run + "_") and t.endswith(f"_{mode}_{push}")]
        if lst:
          res["pooled"][f"{run}|{mode}|{push}"] = audit([e for c in lst for e in c["events"]])
    if run == "r1":
      lst = [cells[t] for t in cells if t.startswith("r1_") and t.endswith("_on")]
      if lst:
        res["r1_push_split_push_on"] = {m: r1_push_split([c for c in lst if f"_{m}_" in [k for k, v in cells.items() if v is c][0]]) for m in ("mean", "noise")}
  # c08 tables (B0)
  res["c08"] = {}
  for mode in ("noise", "mean"):
    for push in ("on", "off"):
      lst = [cells[t] for t in cells if t.startswith("b0_") and t.endswith(f"_{mode}_{push}")]
      if lst:
        rows, extra = hazard_table(lst)
        res["c08"][f"{mode}|{push}"] = dict(rows=rows, extra=extra)
  if "noise|on" in res["c08"]:
    mean_ev = [e for t in cells if t.startswith("b0_") and "_mean_" in t for e in cells[t]["events"]]
    mean_on = [e for t in cells if t.startswith("b0_") and t.endswith("_mean_on") for e in cells[t]["events"]]
    noise_ev = [e for t in cells if t.startswith("b0_") and "_noise_" in t for e in cells[t]["events"]]
    noise_on = [e for t in cells if t.startswith("b0_") and t.endswith("_noise_on") for e in cells[t]["events"]]
    P = res["c08"]["noise|on"]
    res["branch_main"] = branch_b0(P["rows"], P["extra"], mean_ev, mean_on)
    res["branch_ref"] = dict(la_noise=la(noise_ev), ma_noise=ma(noise_on))
    if "mean|on" in res["c08"]:
      Q = res["c08"]["mean|on"]
      ref = branch_b0(Q["rows"], Q["extra"], mean_ev, mean_on)
      res["branch_ref"]["c08_on_mean_actions"] = {k: ref[k] for k in ("ga_crouch_vs_loco", "na_rule1", "da_rule3", "rule2_ref")}
  for run in ("r1",):
    ev_mean = [e for t in cells if t.startswith("r1_") and "_mean_" in t for e in cells[t]["events"]]
    ev_on = [e for t in cells if t.startswith("r1_") and t.endswith("_mean_on") for e in cells[t]["events"]]
    res["r1_la_ma_mean"] = dict(la=la(ev_mean), ma=ma(ev_on))
    ev_n = [e for t in cells if t.startswith("r1_") and "_noise_" in t for e in cells[t]["events"]]
    ev_non = [e for t in cells if t.startswith("r1_") and t.endswith("_noise_on") for e in cells[t]["events"]]
    res["r1_la_ma_noise"] = dict(la=la(ev_n), ma=ma(ev_non))
  json.dump(res, open(argv[1], "w"), indent=1, ensure_ascii=False, default=lambda x: x if not isinstance(x, float) else (None if math.isnan(x) else x))
  # markdown
  L = ["| 칸 | 판 | self | ground | timeout | 자기 충돌 몫(물리 종료) | 흔한 몸 쌍(몫) | 밂 1 s 안 몫 | 자세 quasi/mid/fast | rms_jv_01 p50 | t p50 |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
  for key, a in res["pooled"].items():
    cnt = a["counts"]
    top = a.get("pairs_top", [["—", 0, 0]])[0]
    po = a.get("posture", {})
    L.append(f"| {key} | {a['n']} | {cnt.get('self', 0)} | {cnt.get('ground', 0)} | {cnt.get('timeout', 0)} | "
             f"{a['self_share_of_physical'] if isinstance(a['self_share_of_physical'], str) else round(a['self_share_of_physical'], 3) if a['self_share_of_physical'] == a['self_share_of_physical'] else '—'} | "
             f"{top[0]} ({top[2]}) | {a.get('self_in_push_window_share', '—')} | {po.get('quasi', 0)}/{po.get('mid', 0)}/{po.get('fast', 0)} | "
             f"{a.get('rms_jv_01_p10_50_90', ['—'] * 3)[1]} | {a.get('t_p10_50_90', ['—'] * 3)[1]} |")
  for key, tb in res["c08"].items():
    L.append(f"\n### c08 표 B0 ({key}, 두 시드 합침; 위험률 = 물리 종료/노출 초, 보류 = 종료 < 30)")
    L.append("| 목표·마감·밂 | 노출 s | 구간 | self | ground | 위험률 /s | 구간 확률 | 보류 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for g, r in tb["rows"].items():
      if r["exposure_s"] == 0 and r["physical"] == 0:
        continue
      hz = r["hazard_phys_per_s"]
      sp = r["seg_prob_phys"]
      L.append(f"| {g} | {r['exposure_s']} | {r['segments']} | {r['self']} | {r['ground']} | {hz:.4f} | {sp:.3f} | {'보류' if r['hold'] else ''} |"
               if hz == hz else f"| {g} | 0 | 0 | 0 | 0 | — | — | 보류 |")
    L.append(f"- 단일 지지 몫 {tb['extra']['single_support_share']:.3f}, 살아 있는 시간 {tb['extra']['alive_s']:.0f} s")
  open(argv[2], "w").write("\n".join(L) + "\n")
  print(json.dumps({k: res[k] for k in ("branch_main",) if k in res}, ensure_ascii=False, indent=1, default=str)[:4000])


if __name__ == "__main__":
  main(sys.argv[1:])
