"""Negative 28: scoreboard over sonic_run rows (no composite score). Per policy and per family x severity bin:
F (fall) rate, complete-success rate (charter: no F, every step's end G within tolerance, J <= locked J_ref), G_ok, J_ok,
G integral, path ratio, calm readouts (hold lift-offs per minute, hold drift, Go lateral p-p, torso roll RMS), and
learnability p(1-p). Boundary list: specs the target policy fails (fall / not complete) with a solvability witness = the
same spec or a same-family spec within |ds| <= 0.1 solved by another policy (or the target itself).

  python -I sonic_score.py --specs SPECS.jsonl --rows DIR/rows.jsonl [--rows ...] --target B0 --out PREFIX
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

BINS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0001))


def load_rows(paths):
  rows = {}
  for p in paths:
    for l in Path(p).read_text().splitlines():
      if not l.strip():
        continue
      r = json.loads(l)
      if "error" in r:
        continue
      rows[(r["spec_sha256"], r["policy"])] = r
  return rows


def med(xs):
  xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
  return float(np.median(xs)) if xs else None


def fmt(x, d=2):
  if x is None:
    return "—"
  if isinstance(x, bool):
    return str(int(x))
  if isinstance(x, (int,)) and not isinstance(x, bool):
    return str(x)
  return f"{x:.{d}f}"


def summary(rs):
  n = len(rs)
  if not n:
    return {}
  nohit = [r for r in rs if not r.get("pushes")]
  hold_nohit = [r for r in nohit if r["hold_only"]]
  go = [r for r in rs if not r["hold_only"]]
  return dict(
    n=n, F=sum(r["F"] for r in rs) / n, complete=sum(r["complete"] for r in rs) / n,
    G_ok=sum(r["G_ok"] for r in rs) / n, J_ok=sum(r["J_ok"] for r in rs) / n,
    G_int=med([r["G_int"] for r in rs]), path_ratio=med([r["path_ratio"] for r in go]),
    speed_ratio=med([r["speed_ratio"] for r in go]),
    lift_pm=med([r["hold_liftoffs_per_min"] for r in hold_nohit]), drift=med([r["hold_drift_m"] for r in hold_nohit]),
    lat_pp=med([r["lat_pp_m"] for r in go if not r.get("pushes")]), roll_go=med([r["roll_rms_go_deg"] for r in go if not r.get("pushes")]),
    roll_hold=med([r["roll_rms_hold_deg"] for r in hold_nohit]),
    slip95_go=med([r["slip_p95"] for r in go if not r.get("pushes")]),
    util95_go=med([r["util_p95"] for r in go if not r.get("pushes")]),
    J_hold=med([s_["J"] for r in hold_nohit for s_ in r["steps"] if s_.get("type") == "Hold" and not s_.get("settle") and s_.get("J") is not None]),
    learn=(lambda p: p * (1 - p))(sum(r["complete"] for r in rs) / n))


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--specs", required=True)
  ap.add_argument("--rows", action="append", required=True)
  ap.add_argument("--target", default="B0")
  ap.add_argument("--ref", default="mf")
  ap.add_argument("--out", required=True)
  a = ap.parse_args()
  specs = {json.loads(l)["spec_sha256"]: json.loads(l) for l in Path(a.specs).read_text().splitlines() if l.strip()}
  rows = load_rows(a.rows)
  pols = sorted({p for _, p in rows})
  lines = [f"# scoreboard ({len(specs)} specs, rows {len(rows)})", ""]
  hdr = ("| policy | n | F | complete | G_ok | J_ok | G_int med | path ratio | speed ratio | hold lift/min | hold drift m | hold J rad/s³ | "
         "Go lat p-p m | roll Go ° | roll hold ° | Go stance slip p95 m/s | Go util. friction p95 | p(1-p) |")
  lines += ["## all", "", hdr, "|" + "---|" * 18]
  out = dict(policies={}, families={}, boundary={})
  for p in pols:
    rs = [rows[(h, p)] for h in specs if (h, p) in rows]
    s = summary(rs)
    out["policies"][p] = s
    if s:
      lines.append(f"| {p} | {s['n']} | {fmt(s['F'])} | {fmt(s['complete'])} | {fmt(s['G_ok'])} | {fmt(s['J_ok'])} | {fmt(s['G_int'])} | "
                   f"{fmt(s['path_ratio'])} | {fmt(s['speed_ratio'])} | {fmt(s['lift_pm'], 1)} | {fmt(s['drift'], 3)} | {fmt(s['J_hold'], 1)} | "
                   f"{fmt(s['lat_pp'], 3)} | {fmt(s['roll_go'])} | {fmt(s['roll_hold'])} | {fmt(s['slip95_go'])} | {fmt(s['util95_go'])} | {fmt(s['learn'])} |")
  fams = sorted({sp["family"] for sp in specs.values()})
  lines += ["", "## per family (medians; calm readouts from no-hit specs)", "",
            "| family | policy | n | F | complete | G_ok | J_ok | G_int | speed ratio | hold lift/min | hold drift m | Go lat p-p m | roll Go ° | roll hold ° | slip p95 | util p95 |",
            "|" + "---|" * 16]
  for f in fams:
    for p in pols:
      rs = [rows[(h, p)] for h, sp in specs.items() if sp["family"] == f and (h, p) in rows]
      s = summary(rs)
      if not s:
        continue
      out["families"].setdefault(f, {})["_all_" + p] = s
      lines.append(f"| {f} | {p} | {s['n']} | {fmt(s['F'])} | {fmt(s['complete'])} | {fmt(s['G_ok'])} | {fmt(s['J_ok'])} | {fmt(s['G_int'])} | "
                   f"{fmt(s['speed_ratio'])} | {fmt(s['lift_pm'], 1)} | {fmt(s['drift'], 3)} | {fmt(s['lat_pp'], 3)} | {fmt(s['roll_go'])} | {fmt(s['roll_hold'])} | "
                   f"{fmt(s['slip95_go'])} | {fmt(s['util95_go'])} |")
  lines += ["", "## family x severity bin: F / complete (n)", "", "| family | s bin | " + " | ".join(pols) + " |", "|" + "---|" * (2 + len(pols))]
  for f in fams:
    for lo, hi in BINS:
      hs = [h for h, sp in specs.items() if sp["family"] == f and lo <= sp["level"].get("s", 0.0) < hi]
      if not hs:
        continue
      cells = []
      for p in pols:
        rs = [rows[(h, p)] for h in hs if (h, p) in rows]
        if rs:
          F = sum(r["F"] for r in rs) / len(rs)
          C = sum(r["complete"] for r in rs) / len(rs)
          cells.append(f"{F:.2f} / {C:.2f} ({len(rs)})")
          out["families"].setdefault(f, {}).setdefault(f"{lo:.2f}-{min(hi, 1):.2f}", {})[p] = dict(n=len(rs), F=F, complete=C, learn=C * (1 - C))
        else:
          cells.append("—")
      lines.append(f"| {f} | {lo:.2f}–{min(hi, 1):.2f} | " + " | ".join(cells) + " |")
  # boundary list for the target
  T = a.target
  bl = []
  for h, sp in specs.items():
    r = rows.get((h, T))
    if r is None:
      continue
    fail_F, fail_C = r["F"], not r["complete"]
    if not (fail_F or fail_C):
      continue
    s = sp["level"].get("s", 0.0)
    wit = []
    for h2, sp2 in specs.items():
      if sp2["family"] != sp["family"] or abs(sp2["level"].get("s", 0.0) - s) > 0.1:
        continue
      for p in pols:
        r2 = rows.get((h2, p))
        if r2 is None or (p == T and h2 == h):
          continue
        if fail_F and not r2["F"]:
          wit.append(dict(kind="survived", policy=p, spec=h2[:16], s=sp2["level"].get("s"), same_spec=h2 == h))
        if fail_C and r2["complete"]:
          wit.append(dict(kind="complete", policy=p, spec=h2[:16], s=sp2["level"].get("s"), same_spec=h2 == h))
    bl.append(dict(spec=h[:16], family=sp["family"], s=s, hold=r["hold_only"], F=r["F"], complete=r["complete"],
                   G_ok=r["G_ok"], J_ok=r["J_ok"], witnesses_F=[w for w in wit if w["kind"] == "survived"][:6],
                   witnesses_C=[w for w in wit if w["kind"] == "complete"][:6]))
  out["boundary"][T] = bl
  nF = [b for b in bl if b["F"] and b["witnesses_F"]]
  nC = [b for b in bl if not b["complete"] and b["witnesses_C"]]
  lines += ["", f"## boundary for {T}: fails with a solvability witness (same family, |ds| <= 0.1)", "",
            f"- falls with a survival witness: {len(nF)}; not complete with a complete-success witness: {len(nC)}", "",
            "| spec | family | s | mission | F | complete | G_ok | J_ok | witness (policy s same-spec) |", "|---|---|---|---|---|---|---|---|---|"]
  for b in sorted(bl, key=lambda b: (b["family"], b["s"])):
    w = (b["witnesses_F"] if b["F"] else b["witnesses_C"])[:3]
    if not w:
      continue
    ws = "; ".join(f"{x['policy']} {x['s']:.2f}{' =' if x['same_spec'] else ''}" for x in w)
    lines.append(f"| {b['spec']} | {b['family']} | {b['s']:.2f} | {'Hold' if b['hold'] else 'Go'} | {int(b['F'])} | {int(b['complete'])} | "
                 f"{int(b['G_ok'])} | {int(b['J_ok'])} | {ws} |")
  ref = a.ref
  same = []
  for h, sp in specs.items():
    r, w = rows.get((h, T)), rows.get((h, ref))
    if r is None or w is None:
      continue
    if (r["F"] and not w["F"]) or (not r["complete"] and w["complete"]):
      same.append((sp, r, w))
  out["boundary_same_spec"] = [dict(spec=sp["spec_sha256"][:16], family=sp["family"], s=sp["level"].get("s"), hold=r["hold_only"],
                                     target_F=r["F"], target_t_fall=r["t_fall"], target_complete=r["complete"], ref_F=w["F"],
                                     ref_complete=w["complete"], env=sp["env"]) for sp, r, w in same]
  lines += ["", f"## {T} fails, {ref} solves the same spec (fall level: {ref} stands; complete level: {ref} complete)", "",
            f"- fall level: {sum(1 for _, r, w in same if r['F'] and not w['F'])}; complete level: {sum(1 for _, r, w in same if not r['complete'] and w['complete'])}", "",
            "| spec | family | s | mission | kind | T fall t s | env summary |", "|---|---|---|---|---|---|---|"]
  for sp, r, w in sorted(same, key=lambda x: (x[0]["family"], x[0]["level"].get("s", 0))):
    e = sp["env"]
    b = e.get("body") or {}
    bits = []
    if b.get("foot_mu") is not None:
      bits.append(f"mu {b['foot_mu']:.2f} {b.get('cone', 'pyr')[:3]}")
    if b.get("mass_scale"):
      bits.append("mass " + "/".join(f"{v:.2f}" for v in b["mass_scale"].values()))
    if b.get("action_delay_ms") is not None:
      bits.append(f"act delay {b['action_delay_ms']:.0f} ms")
    if b.get("kp_scale"):
      bits.append(f"gain {min(b['kp_scale']):.2f}-{max(b['kp_scale']):.2f}")
    for hh in e.get("hit") or []:
      bits.append(f"hit {hh['J_Ns']:.0f} N s @{hh['t_s']:.1f} s z{hh['z_off_m']:+.2f}")
    se = e.get("sense") or {}
    if se:
      bits.append(f"noise s={sp['level'].get('s', 0):.2f} delay {se.get('delay_ms', 0):.0f} ms")
    if (e.get("floor") or {}).get("rough_h_m"):
      bits.append(f"rough {e['floor']['rough_h_m'] * 1000:.1f} mm")
    kind = "fall" if (r["F"] and not w["F"]) else "complete"
    lines.append(f"| {sp['spec_sha256'][:16]} | {sp['family']} | {sp['level'].get('s', 0):.2f} | {'Hold' if r['hold_only'] else 'Go'} | {kind} | "
                 f"{fmt(r['t_fall'], 1) if r['F'] else '—'} | {'; '.join(bits)} |")
  Path(a.out + ".md").write_text("\n".join(lines) + "\n")
  Path(a.out + ".json").write_text(json.dumps(out, indent=1))
  print("\n".join(lines[:12 + len(pols)]))


if __name__ == "__main__":
  main()
