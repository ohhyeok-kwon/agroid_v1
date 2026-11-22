"""Negative 28: issuing-loop readout over level sets (sonic_levels). Per axis level and policy: successes k / n under
the axis' own criterion (see sonic_levels docstring; delay axes v2 04:58 KST: post-stop standing J <= max(20, 3 x the
same policy's J on the same instance at x = 0), because the @500 dry run showed post-stop settling J 20-49 at 0 ms for
the SONIC targets, a mission-level property that is not delay), Wilson 95 % bounds, learnability p(1-p); a level is issuable when
the witness (manufacturer, evaluation only) succeeds at least once. Next levels per target: the issuable level with the
largest p(1-p), plus the midpoint between adjacent levels where the target's p crosses 0.5 (bisection).

  python -I sonic_loop.py --specs L.jsonl --rows rows.jsonl [--rows ...] --j0 headline_rows.jsonl --targets Bp500,Bppr500
         --witness mf --out PREFIX
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson(k, n, z=1.96):
  if n == 0:
    return 0.0, 1.0
  p = k / n
  den = 1 + z * z / n
  c = (p + z * z / (2 * n)) / den
  h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
  return max(0.0, c - h), min(1.0, c + h)


def j0_table(paths):
  """Standing J at 0 ms action delay per policy (axis_probe 'delay 0 hold')."""
  out = {}
  for p in paths:
    for l in Path(p).read_text().splitlines():
      r = json.loads(l)
      if "error" in r:
        continue
      out.setdefault(r["spec_sha256"], {})[r["policy"]] = r
  return out


def post_stop_j(r):
  """Largest standing J (after the first second) over the Holds that follow a Go."""
  out, prev = [], None
  for s in r["steps"]:
    if s.get("type") == "Hold" and prev == "Go" and s.get("J") is not None:
      out.append(s["J"])
    prev = s.get("type")
  return max(out) if out else None


FWD_MIN = 0.7      # strict (mission G) criterion for walking steps, coordinator 08:0x KST; see fwd_ok


def fwd_ok(r, thr=FWD_MIN):
  """Every forward Go step (vx > 0, vy = 0, measured after its first second) realises >= thr of the commanded speed.
  Rationale: the charter Go tolerance (0.10 m + 0.10 x commanded distance) taken literally needs >= 0.875-0.9 on 4-5 m
  walks, which the evaluation-only witness (0.81-0.85) itself fails; 0.7 keeps the witness passing with margin and
  fails near-standing policies (B3d@1000: 0.05-0.28). Lateral steps are not checked (the witness does not side-step)."""
  for s in r["steps"]:
    c = s.get("cmd")
    if s.get("speed_ratio") is not None and c and c[0] > 0 and abs(c[1]) < 1e-6 and s["speed_ratio"] < thr:
      return False
  return True


def success(axis, r, j0, base=None, strict=False):
  """base: the same policy's row on the same instance at x = 0 (delay axes; paired baseline, v2 04:58 KST).
  strict (08:0x KST): delay axes also need the mission G (fwd_ok); noise axes already use the charter Hold G."""
  if r["F"]:
    return False
  if axis in ("delay", "sdelay"):
    jb = post_stop_j(base) if base is not None and not base["F"] else None
    lim = max(20.0, 3.0 * (jb if jb is not None else (j0 or 0.0)))
    j = post_stop_j(r)
    return (j is None or j <= lim) and (fwd_ok(r) if strict else True)
  if axis == "push":                                    # walking push: strict also needs the forward ratio (08:2x KST)
    return fwd_ok(r) if strict else True
  if axis in ("spush", "combo"):
    return True
  return bool(r["G_ok"])                        # noise axes: no fall and charter Hold G (drift / heading)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--specs", required=True)
  ap.add_argument("--rows", action="append", required=True)
  ap.add_argument("--j0", required=True, help="axis_probe specs:rows, e.g. specs/axis_probe.jsonl:runs/axis_probe/rows.jsonl")
  ap.add_argument("--targets", required=True)
  ap.add_argument("--witness", default="mf")
  ap.add_argument("--out", required=True)
  ap.add_argument("--strict", action="store_true", help="success also needs mission G (delay axes: forward steps >= 0.7)")
  a = ap.parse_args()
  specs = {json.loads(l)["spec_sha256"]: json.loads(l) for l in Path(a.specs).read_text().splitlines() if l.strip()}
  rows = {}
  for p in a.rows:
    for l in Path(p).read_text().splitlines():
      r = json.loads(l)
      if "error" not in r and r["spec_sha256"] in specs:
        rows[(r["spec_sha256"], r["policy"])] = r
  ps, pr = a.j0.split(":")
  hs = [json.loads(l) for l in Path(ps).read_text().splitlines() if l.strip()]
  h0 = [s["spec_sha256"] for s in hs if s["level"].get("axis") == "delay" and s["level"].get("x") == 0 and s["level"].get("mission") == "hold"][0]
  j0 = {}
  for l in Path(pr).read_text().splitlines():
    r = json.loads(l)
    if r.get("spec_sha256") == h0 and "error" not in r:
      j = [s["J"] for s in r["steps"] if s.get("type") == "Hold" and not s.get("settle") and s.get("J") is not None]
      j0[r["policy"]] = j[0] if j else None
  targets = a.targets.split(",")
  pols = targets + [a.witness]
  agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
  zero = {}
  for (h, p), r in rows.items():
    lv = specs[h]["level"]
    if lv["x"] == 0:
      zero[(lv["axis"], lv["i"], p)] = r
  for (h, p), r in rows.items():
    lv = specs[h]["level"]
    if p not in pols:
      continue
    c = agg[(lv["axis"], lv["x"])][p]
    c[0] += success(lv["axis"], r, j0.get(p), zero.get((lv["axis"], lv["i"], p)), strict=a.strict)
    c[1] += 1
  axes = sorted({k[0] for k in agg})
  lines = [f"# issuing loop readout (targets {', '.join(targets)}; witness {a.witness}; criterion {'strict = v2 + mission G' if a.strict else 'v2'})", "",
           f"- J at 0 ms (standing, for the delay criterion): " + ", ".join(f"{p} {j0.get(p) if j0.get(p) is None else round(j0[p], 1)}" for p in pols), ""]
  nxt = {}
  for ax in axes:
    xs = sorted(x for (aa, x) in agg if aa == ax)
    lines += [f"## {ax}", "", "| x | " + " | ".join(f"{p} k/n (p(1-p))" for p in pols) + " | witness Wilson LB | issuable |", "|" + "---|" * (len(pols) + 3)]
    for x in xs:
      cells = []
      for p in pols:
        k, n = agg[(ax, x)][p]
        pp = k / n if n else None
        cells.append(f"{k}/{n}" + (f" ({pp * (1 - pp):.2f})" if pp is not None and p in targets else ""))
      kw, nw = agg[(ax, x)][a.witness]
      lb = wilson(kw, nw)[0]
      lines.append(f"| {x:g} | " + " | ".join(cells) + f" | {lb:.2f} | {'yes' if kw >= 1 else 'no'} |")
    for t in targets:
      cand = []
      for x in xs:
        k, n = agg[(ax, x)][t]
        kw, _ = agg[(ax, x)][a.witness]
        if n and kw >= 1:
          cand.append((k / n * (1 - k / n), x, k / n))
      best = max(cand) if cand else None
      mids = []
      for x0, x1 in zip(xs, xs[1:]):
        k0, n0 = agg[(ax, x0)][t]
        k1, n1 = agg[(ax, x1)][t]
        if n0 and n1 and (k0 / n0 - 0.5) * (k1 / n1 - 0.5) < 0:
          mids.append(round((x0 + x1) / 2, 4))
      nxt.setdefault(t, {})[ax] = dict(best_level=best[1] if best else None, best_learnability=round(best[0], 3) if best else None,
                                        best_p=round(best[2], 3) if best else None, bisect=mids)
    lines.append("")
  lines += ["## next levels per target (issuable max p(1-p); bisection midpoints where p crosses 0.5)", ""]
  for t, d in nxt.items():
    for ax, v in d.items():
      lines.append(f"- {t} / {ax}: best x = {v['best_level']} (p {v['best_p']}, p(1-p) {v['best_learnability']}); bisect {v['bisect']}")
  Path(a.out + ".md").write_text("\n".join(lines) + "\n")
  Path(a.out + ".json").write_text(json.dumps(dict(j0=j0, next=nxt, levels={f"{k[0]}|{k[1]}": {p: v for p, v in d.items()} for k, d in agg.items()}), indent=1))
  print("\n".join(lines))


if __name__ == "__main__":
  main()
