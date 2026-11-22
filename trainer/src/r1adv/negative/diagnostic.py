"""R0 diagnostic sweep checks on scored repeats (pre-registered in agents/negative/status.md, 11-17 01:00).

(a) known-defect lift: per policy, among instances without a fall (and without a spawn failure), the worst 10 % by
    the metric of the defect; lift = share of predicate instances there / share among all; hit = share of the
    predicate instances that land there. Pass lift >= 3, kill < 1.5.
(b) repeat stability: Spearman of per-instance G, J (time median), J deficit between two separate processes (the
    second with a permuted environment order); fall agreement. Pass >= 0.8.
(c) reproducibility: identical spec set hashes (checked by the caller on PC and server) and equal outcome
    distributions between repeats (complete-success rates with Wilson intervals, KS statistics of G and J).
usage: python -m r1adv.negative.diagnostic --specs specs.jsonl --rep A=<runs dir> --rep B=<runs dir> --out diag.json
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from r1adv.negative import score as SC

SETTLE_TICKS = 50
METRIC_OF = {"P_side": "J_deficit", "P_up": "G", "P_noise": "J_deficit", "P_noise_flat": "J_deficit"}
KNOWN = {"slope45": ("P_side",), "rad10x_student": ("P_up",), "deployable_D": ("P_up",), "v12h": ("P_noise", "P_noise_flat")}


def ranks(x: np.ndarray) -> np.ndarray:
  order = np.argsort(x, kind="mergesort")
  out = np.empty(len(x))
  sorted_x = x[order]
  i = 0
  while i < len(x):
    j = i
    while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
      j += 1
    out[order[i:j + 1]] = 0.5 * (i + j)
    i = j + 1
  return out


def spearman(a, b) -> float | None:
  a, b = np.asarray(a, float), np.asarray(b, float)
  if len(a) < 5:
    return None
  ra, rb = ranks(a), ranks(b)
  if ra.std() == 0 or rb.std() == 0:
    return None
  return float(np.corrcoef(ra, rb)[0, 1])


def spearman_ci(a, b, rounds: int = 400, seed: int = 0):
  a, b = np.asarray(a, float), np.asarray(b, float)
  rng = np.random.default_rng(seed)
  values = []
  for _ in range(rounds):
    pick = rng.integers(0, len(a), len(a))
    value = spearman(a[pick], b[pick])
    if value is not None:
      values.append(value)
  return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values else None


def ks(a, b) -> float:
  a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
  grid = np.concatenate([a, b])
  return float(np.abs(np.searchsorted(a, grid, side="right") / len(a) - np.searchsorted(b, grid, side="right") / len(b)).max())


def wilson(k: int, n: int, z: float = 1.96):
  if n == 0:
    return None
  p = k / n
  centre = (p + z * z / (2 * n)) / (1 + z * z / n)
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
  return [round(centre - half, 4), round(centre + half, 4)]


def valid(row) -> bool:
  return not (row["F"] and row["fell_step"] is not None and row["fell_step"] < SETTLE_TICKS)


def lift_table(rows: list[dict], preds: list[dict]) -> dict:
  """rows/preds of one policy (instance order)."""
  out = {}
  for name, metric in METRIC_OF.items():
    pool = [i for i, r in enumerate(rows) if valid(r) and not r["F"] and r.get(metric) is not None]
    if not pool:
      continue
    values = np.array([rows[i][metric] for i in pool])
    cut = max(1, int(round(0.1 * len(pool))))
    worst = set(np.asarray(pool)[np.argsort(-values, kind="mergesort")[:cut]].tolist())
    has = [i for i in pool if preds[i][name]]
    base = len(has) / len(pool)
    inside = sum(1 for i in worst if preds[i][name]) / len(worst)
    out[name] = dict(metric=metric, pool=len(pool), predicate_n=len(has), base_rate=round(base, 4),
                     worst_share=round(inside, 4), lift=round(inside / base, 2) if base > 0 else None,
                     hit_rate=round(sum(1 for i in has if i in worst) / len(has), 4) if has else None,
                     predicate_median=float(np.median([rows[i][metric] for i in has])) if has else None,
                     others_median=float(np.median([rows[i][metric] for i in pool if not preds[i][name]])))
  return out


def composition(rows: list[dict], specs: list[dict], key: str) -> dict:
  pool = [i for i, r in enumerate(rows) if valid(r) and not r["F"] and r.get(key) is not None]
  values = np.array([rows[i][key] for i in pool])
  worst = np.asarray(pool)[np.argsort(-values, kind="mergesort")[:max(1, int(round(0.1 * len(pool))))]]
  def tally(fn, idx):
    out = {}
    for i in idx:
      label = fn(specs[i])
      out[label] = out.get(label, 0) + 1
    return {k: round(v / len(idx), 3) for k, v in sorted(out.items(), key=lambda kv: -kv[1])}
  fns = dict(terrain=lambda s: s["env"]["terrain"]["kind"],
             hit=lambda s: "hit" if s["env"]["disturbances"] else "none",
             noise=lambda s: "N0" if s["env"]["sense"]["scale"] == 0 else ("N1" if s["env"]["sense"]["scale"] >= 0.75 else "mid"))
  return {name: dict(worst=tally(fn, worst), all=tally(fn, pool)) for name, fn in fns.items()}


def summarize(specs, scored: dict[str, list[dict]]) -> dict:
  out = {}
  for policy, rows in scored.items():
    preds = [SC.predicates(s, r) for s, r in zip(specs, rows)]
    ok = [r for r in rows if valid(r)]
    complete = sum(r["complete"] for r in ok)
    falls = sum(r["F"] for r in ok)
    G = [r["G"] for r in ok if r["G"] is not None and not r["F"]]
    J = [r["J_deficit"] for r in ok if r.get("J_deficit") is not None and not r["F"]]
    out[policy] = dict(n=len(rows), spawn_failures=len(rows) - len(ok), falls=falls, fall_rate=round(falls / len(ok), 4),
                       complete=complete, p=round(complete / len(ok), 4), p_ci=wilson(complete, len(ok)),
                       G_median=float(np.median(G)), G_p90=float(np.percentile(G, 90)), G_pass=round(float(np.mean(np.asarray(G) <= SC.TAU_G)), 4),
                       J_deficit_median=float(np.median(J)), J_deficit_p90=float(np.percentile(J, 90)),
                       J_pass=round(float(np.mean(np.asarray(J) <= 1.0)), 4),
                       lifts=lift_table(rows, preds),
                       worst_by_J=composition(rows, specs, "J_deficit"), worst_by_G=composition(rows, specs, "G"))
  return out


def repeat_check(specs, a: dict[str, list[dict]], b: dict[str, list[dict]]) -> dict:
  out = {}
  for policy in a:
    if policy not in b:
      continue
    ra, rb = a[policy], b[policy]
    both = [i for i in range(len(specs)) if valid(ra[i]) and valid(rb[i])]
    entry = dict(n=len(both))
    for key in ("G", "J_med", "J_deficit"):
      idx = [i for i in both if ra[i].get(key) is not None and rb[i].get(key) is not None and not ra[i]["F"] and not rb[i]["F"]]
      x, y = [ra[i][key] for i in idx], [rb[i][key] for i in idx]
      entry[key] = dict(n=len(idx), spearman=spearman(x, y), ci=spearman_ci(x, y), ks=ks(x, y),
                        median_abs_diff=float(np.median(np.abs(np.asarray(x) - np.asarray(y)))))
    fa = np.array([ra[i]["F"] for i in both]); fb = np.array([rb[i]["F"] for i in both])
    ca = np.array([ra[i]["complete"] for i in both]); cb = np.array([rb[i]["complete"] for i in both])
    def kappa(x, y):
      po = float(np.mean(x == y))
      pe = float(np.mean(x) * np.mean(y) + (1 - np.mean(x)) * (1 - np.mean(y)))
      return (po - pe) / (1 - pe) if pe < 1 else None
    entry.update(fall_agreement=float(np.mean(fa == fb)), fall_kappa=kappa(fa, fb), falls=[int(fa.sum()), int(fb.sum())],
                 complete_agreement=float(np.mean(ca == cb)), complete_kappa=kappa(ca, cb),
                 complete=[int(ca.sum()), int(cb.sum())], complete_ci=[wilson(int(ca.sum()), len(both)), wilson(int(cb.sum()), len(both))])
    out[policy] = entry
  return out


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--specs", type=Path, required=True)
  parser.add_argument("--rep", action="append", required=True, help="NAME=<dir holding <policy>/arrays.npz>")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--rows", type=Path, default=None, help="also write the first repeat's scored rows (JSONL)")
  args = parser.parse_args()
  specs = [json.loads(line) for line in args.specs.read_text().splitlines() if line.strip()]
  reps = {}
  for item in args.rep:
    name, path = item.split("=", 1)
    runs = {re.sub(r"_p\d+$", "", p.name): p for p in sorted(Path(path).iterdir()) if (p / "arrays.npz").exists()}
    reps[name] = SC.score_runs(specs, runs)
  names = list(reps)
  result = dict(schema="r1adv-negative-diagnostic-v1", specs=str(args.specs), count=len(specs),
                tau_G=SC.TAU_G, J_floor=SC.J_FLOOR, J_factor=SC.J_FACTOR,
                summary={name: summarize(specs, reps[name]) for name in names})
  if len(names) >= 2:
    result["repeat"] = {f"{names[0]}~{names[i]}": repeat_check(specs, reps[names[0]], reps[names[i]]) for i in range(1, len(names))}
  args.out.write_text(json.dumps(result, indent=1))
  if args.rows:
    with args.rows.open("w") as stream:
      for policy, rows in reps[names[0]].items():
        for spec, row in zip(specs, rows):
          slim = {k: v for k, v in row.items() if k != "steps"}
          slim["steps"] = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                            if k in ("k", "type", "cls", "G", "J", "J_ref", "J_ref_source", "J_deficit", "path_err_m",
                                     "progress_ratio", "height_err_m", "slope_deg_here", "heading_uphill_cos",
                                     "travel_uphill_cos", "fell_in", "out_of_patch")} for s in row["steps"]]
          stream.write(json.dumps(dict(policy=policy, seed_path=spec["seed_path"], **SC.predicates(spec, row), **slim)) + "\n")
  print(json.dumps({n: {p: {k: v for k, v in s.items() if k in ("p", "falls", "spawn_failures")} for p, s in result["summary"][n].items()}
                    for n in names}))


if __name__ == "__main__":
  main()
