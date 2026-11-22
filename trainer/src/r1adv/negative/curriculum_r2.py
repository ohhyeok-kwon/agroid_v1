"""R2 tick runner pieces (coordinator 12:30; prepared, not connected to training): from a training checkpoint, probe
the current levels deterministically, score them, and publish the next level bundle through level_feed.

  propose   python -m r1adv.negative.curriculum_r2 propose --state DIR --tick K [--count 64 --episodes 8]
            -> DIR/probe_tick<K>.jsonl (runner specs, start frames and hits from families_r2) and the buffer state
  score     python -m r1adv.negative.curriculum_r2 score --state DIR --tick K --runs RUNS --learner NAME
              [--witnesses NAME,...] --feed FEED_DIR [--level-share 0.3]
            -> calm success per episode (r1adv.negative.calm), learner p and witness successes per level,
               S = E_Beta[p(1-p)] * 1[witness Wilson lower >= 0.5] (+ coin filter delta 0.1), buffer update,
               next bundle feed_tick<K+1>.json (weights = PLR priority over the levels with S > 0)
The base training distribution (no hit + start spread) is the calm anchor: the bundle carries level_share (0.3 at
the start, capped at 0.5); the training env draws a level with that probability and the base otherwise.
Witnesses: our own population only (earlier learner snapshots, teachers, students; never the manufacturer).
"""
from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path

import numpy as np

from r1adv.negative import families_r2 as F2
from r1adv.negative import level_feed as LF
from r1adv.negative.calm import SWAY_RAD, TREMOR_RAD, VIB_RAD, calm_still, calm_success
from r1adv.negative.levels import LevelBuffer, beta_learnability, r6, wilson_lower
from r1adv.negative.seedtree import canonical

LEVEL_SHARE_START, LEVEL_SHARE_CAP, LEVEL_SHARE_STEP, COIN_DELTA, TAU = 0.3, 0.5, 0.1, 0.1, 0.5
ANCHOR_EPISODES, ANCHOR_CALM = 64, 0.9


def anchor_specs(start_default: bool = False) -> list[dict]:
  """The base anchor (no hit, the start spread: half nominal pose, half human frames): the same 64 episodes every
  tick (common random numbers across ticks), judged with the same calm success."""
  from r1adv.negative.levels import Level, episode_spec, level_id
  from r1adv.negative.seedtree import child_seed
  out = []
  for e in range(ANCHOR_EPISODES):
    params = dict(start="default" if (e % 2 == 0 or start_default) else "frame", frame=(e * 17) % 1132, hits=0)
    goal = child_seed(20261117, f"R02-curriculum/anchor/{e}")
    level = Level(level_id("r2-anchor", params, goal, F2.source_sha256()), "r2-hit-v1", dict(PARAM_ANCHOR, **params), goal)
    out.append(episode_spec(level, 0, "R02-curriculum-anchor", F2.source_sha256(), F2.FAMILIES, F2.mission_r2, version="r2c-v1"))
  return out


PARAM_ANCHOR = dict(j1=0.0, az1=0.0, h1=0.0, t1=2.5, j2=0.0, az2=0.0, h2=0.0, dt2=2.0)


def share_rule(state: Path, tick: int, anchor: dict, has_levels: bool) -> tuple[float, float]:
  """Level share (coordinator 13:00): start 0.3; +0.1 when the base anchor's calm success is >= 0.9 on two ticks in a
  row and its fall rate has not risen; cap 0.5. Reading used here (stated in curriculum_ready.md): a tick counts only
  when the interval it measures was trained with levels (the previous bundle was not empty), and after a raise the
  count restarts, so each new share is itself held for two good ticks before the next raise. Returns (rule share,
  effective share = 0 when this tick's bundle is empty). History in share.json (one entry per tick)."""
  path = state / "share.json"
  hist = json.loads(path.read_text()) if path.exists() else []
  hist = [h for h in hist if h["tick"] < tick]
  prev = hist[-1] if hist else None
  share = prev["share"] if prev else LEVEL_SHARE_START
  good = (anchor["calm"] is not None and anchor["calm"] >= ANCHOR_CALM and prev is not None and prev["effective"] > 0
          and (prev["anchor_fall"] is None or anchor["fall"] <= prev["anchor_fall"]))
  streak = (prev["streak"] + 1 if prev else 1) if good else 0
  if streak >= 2 and share < LEVEL_SHARE_CAP:
    share, streak = min(round(share + LEVEL_SHARE_STEP, 3), LEVEL_SHARE_CAP), 0
  effective = share if has_levels else 0.0
  hist.append(dict(tick=tick, share=share, effective=effective, streak=streak, anchor_calm=anchor["calm"],
                   anchor_fall=anchor["fall"]))
  path.write_text(json.dumps(hist, indent=0))
  return share, effective


def new_buffer(round_name: str = "R02-curriculum") -> LevelBuffer:
  return LevelBuffer(round_name, registry=F2.FAMILIES, source=F2.source_sha256(), mission_fn=F2.mission_r2, version="r2c-v1",
                     baseline_share=0.0, family_cap=1.0, tau=TAU)   # one family: no family cap; the base is the anchor


def load(state: Path, tick: int) -> LevelBuffer:
  from r1adv.negative.levels import Level
  buf = new_buffer()
  path = state / f"buffer_tick{tick}.json"
  if path.exists():
    data = json.loads(path.read_text())
    if data["source_sha256"] != buf.source:
      raise ValueError("family source changed since this buffer was written")
    buf.tick, buf.cfg, buf.log = data["tick"], data["cfg"], data["log"]
    buf.seen_signatures = set(data["seen_signatures"])
    buf.levels = {row["level_id"]: Level(**row) for row in data["levels"]}
  elif tick != 0:
    raise FileNotFoundError(path)
  return buf


def unique(levels):
  seen, out = set(), []
  for lv in levels:
    if lv.level_id not in seen:
      seen.add(lv.level_id)
      out.append(lv)
  return out


def propose(state: Path, tick: int, count: int, episodes: int, start_default: bool = False) -> Path:
  """start_default (R2e, coordinator 17:2x): every probe starts from the nominal pose (the positive env ignores the
  level's start frame), so the level's start gene is dropped from the probe specs; the anchor is all default too."""
  state.mkdir(parents=True, exist_ok=True)
  buf = load(state, tick)
  if not (state / f"buffer_tick{tick}.json").exists():
    buf.save(state / f"buffer_tick{tick}.json")             # the state before the proposal (score replays it)
  levels = unique(buf.propose(count))
  specs = [s for lv in levels for s in buf.episodes(lv, episodes)]
  if start_default:
    from r1adv.negative.seedtree import spec_hash
    for sp in specs:
      sp["env"]["start_pose"] = None
      sp["spec_sha256"] = spec_hash(sp)
  out = state / f"probe_tick{tick}.jsonl"
  out.write_text("".join(canonical(s) + "\n" for s in specs))
  anchor = anchor_specs(start_default)                      # the learner's probe = the tick's levels + the base anchor
  (state / f"probe_tick{tick}.learner.jsonl").write_text("".join(canonical(s) + "\n" for s in specs + anchor))
  (state / f"probe_tick{tick}.levels.json").write_text(json.dumps(dict(count=count, level_ids=[lv.level_id for lv in levels])))
  return out


def outcomes_of_run(run: Path, specs: list[dict]) -> dict:
  """spec sha256 -> calm-success record, for a runner output made on exactly these specs."""
  with np.load(run / "arrays.npz") as z:
    a = {k: z[k] for k in z.files}
  if list(a["spec_sha256"]) != [s["spec_sha256"] for s in specs]:
    raise ValueError(f"{run}: arrays do not follow the probe specs")
  return {s["spec_sha256"]: calm_success(a, i, s) for i, s in enumerate(specs)}


def cache_path(state: Path, name: str) -> Path:
  return state / "witness_cache" / f"{name}.jsonl"


def cached(state: Path, name: str) -> dict:
  path = cache_path(state, name)
  out = {}
  if path.exists():
    for line in path.read_text().splitlines():
      if line.strip():
        row = json.loads(line)
        out[row.pop("spec_sha256")] = row
  return out


def subset(state: Path, tick: int, name: str) -> Path:
  """The tick's level probes a fixed witness has not run yet (witness outcomes are cached per spec)."""
  have = cached(state, name)
  specs = [l for l in (state / f"probe_tick{tick}.jsonl").read_text().splitlines() if l.strip()]
  todo = [l for l in specs if json.loads(l)["spec_sha256"] not in have]
  out = state / f"probe_tick{tick}.{name}.todo.jsonl"
  out.write_text("".join(l + "\n" for l in todo))
  return out


def add_to_cache(state: Path, name: str, run: Path, specs_file: Path) -> int:
  specs = [json.loads(l) for l in specs_file.read_text().splitlines() if l.strip()]
  rows = outcomes_of_run(run, specs)
  path = cache_path(state, name)
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("a") as f:
    for sha, r in rows.items():
      f.write(json.dumps(dict(r, spec_sha256=sha), sort_keys=True) + "\n")
  return len(rows)


def score(state: Path, tick: int, learner_run: Path | None, witnesses: list[str], feed: Path, distribution: dict,
          coin_delta: float | str | None = COIN_DELTA, snapshots: tuple[str, ...] = (), bundle_temperature: float | None = None,
          fresh_only: bool = False) -> dict:
  """witnesses: fixed checkpoints whose cache must cover every probe episode of the tick; snapshots: earlier learner
  ticks (learner_tick<k>, cache only, no runs) that witness only the levels whose episodes they already played."""
  buf = load(state, tick)
  proposal = json.loads((state / f"probe_tick{tick}.levels.json").read_text())
  level_ids = [lv.level_id for lv in unique(buf.propose(proposal["count"]))]   # replay the tick's proposal
  if level_ids != proposal["level_ids"]:
    raise ValueError("the buffer does not replay this tick's proposal")
  specs = [json.loads(l) for l in (state / f"probe_tick{tick}.jsonl").read_text().splitlines() if l.strip()]
  index = collections.defaultdict(list)
  for i, s in enumerate(specs):
    index[s["generator"]["level_id"]].append(i)
  learner_specs = [json.loads(l) for l in (state / f"probe_tick{tick}.learner.jsonl").read_text().splitlines() if l.strip()]
  if learner_run is None:                                   # tick 0 before training: no learner yet, p = 0/8 everywhere
    nothing = dict(ok=False, fell=False, calm_before=False, recovered=False)
    learn = [nothing] * len(specs)
    anchor = dict(calm=None, fall=None, n=0)
  else:
    by_sha = outcomes_of_run(learner_run, learner_specs)
    learn = [by_sha[s["spec_sha256"]] for s in specs]
    anchor_rows = [by_sha[s["spec_sha256"]] for s in learner_specs[len(specs):]]
    tremor = [r.get("tremor_rad") for r in anchor_rows]            # coordinator 14:56: the anchor calm includes the
    have = [t for t in tremor if t is not None]                     # joint-tremor condition (<= 1 mrad); level p not
    anchor = dict(calm=round(float(np.mean([calm_still(r) for r in anchor_rows])), 4),
                  calm_pelvis_only=round(float(np.mean([r["ok"] for r in anchor_rows])), 4),
                  fall=round(float(np.mean([r["fell"] for r in anchor_rows])), 4), n=len(anchor_rows),
                  tremor_p50_mrad=round(float(np.median(have)) * 1e3, 3) if have else None,
                  tremor_max_mrad=round(float(np.max(have)) * 1e3, 3) if have else None,
                  tremor_over=int(sum(t > TREMOR_RAD for t in have)), tremor_missing=int(len(tremor) - len(have)))
    for key, limit in (("vib", VIB_RAD), ("sway", SWAY_RAD)):     # addendum 2: vibration (>= 2 Hz) and sway (0.2-2 Hz)
      vals = [r.get(f"{key}_rad") for r in anchor_rows if r.get(f"{key}_rad") is not None]
      anchor.update({f"{key}_p50_mrad": round(float(np.median(vals)) * 1e3, 3) if vals else None,
                     f"{key}_max_mrad": round(float(np.max(vals)) * 1e3, 3) if vals else None,
                     f"{key}_over": int(sum(v > limit for v in vals))})
  wit = {}
  for w in witnesses:
    have = cached(state, w)
    missing = [s["spec_sha256"] for s in specs if s["spec_sha256"] not in have]
    if missing:
      raise ValueError(f"witness {w}: {len(missing)} probe episodes not in its cache")
    wit[w] = [have[s["spec_sha256"]] for s in specs]
  for w in snapshots:
    have = cached(state, w)
    wit[w] = [have.get(s["spec_sha256"]) for s in specs]
  results, report = {}, []
  for lid in level_ids:
    idx = index[lid]
    lk = [learn[i]["ok"] for i in idx]
    usable = [w for w in wit if all(wit[w][i] is not None for i in idx)]
    best = max(usable, key=lambda w: sum(wit[w][i]["ok"] for i in idx)) if usable else None
    wk = [wit[best][i]["ok"] for i in idx] if best else []
    results[lid] = dict(learner=lk, witness=wk, signatures=[])
    report.append(dict(level=lid, learner_k=int(sum(lk)), witness=best, witness_k=int(sum(wk)), n=len(idx),
                       witness_lb=wilson_lower(int(sum(wk)), len(idx)) if wk else 0.0,
                       learnability=beta_learnability(int(sum(lk)), len(idx)),
                       learner_reasons=dict(collections.Counter(
                         "fell" if learn[i]["fell"] else "not calm before" if not learn[i]["calm_before"]
                         else "not recovered" if not learn[i]["recovered"] else "ok" for i in idx))))
  for lid in results:          # a replayed level reuses the same 8 deterministic episodes: keep this tick's counts only
    buf.levels[lid].learner, buf.levels[lid].witness = [0, 0], [0, 0]   # (no double counting of cached witnesses)
  buf.update(results)
  for lid in results if coin_delta == "self" else ():       # option (SFL-like): the learner's own partial success is
    level = buf.levels[lid]                                 # its feasibility proof; score = p(1-p) with p = k/n;
    k, n = level.learner                                    # 0/n keeps the witness-gated score; n/n is mastered (0)
    if n and k == n:
      level.score = 0.0
    elif n and k >= 1:
      level.score = r6(k / n * (1 - k / n))
  for lid in results if coin_delta == "mastered" else ():   # option: drop only levels the learner itself passes
    level = buf.levels[lid]                                 # (its Wilson lower >= tau: it would be its own witness)
    if level.score > 0 and wilson_lower(*level.learner) >= TAU:
      level.score = 0.0
      buf.log.append(dict(tick=buf.tick, event="mastered", level=lid))
  for lid in results if isinstance(coin_delta, float) else ():   # coin filter: the witness must beat the learner by delta
    level = buf.levels[lid]
    lb = wilson_lower(*level.witness)
    ub = 1.0 - wilson_lower(level.learner[1] - level.learner[0], level.learner[1]) if level.learner[1] else 1.0
    if level.score > 0 and lb < max(TAU, ub + coin_delta):
      level.score = 0.0
      buf.log.append(dict(tick=buf.tick, event="coin", level=lid))
  buf.save(state / f"buffer_tick{tick + 1}.json")
  live = sorted((lv for lv in buf.levels.values() if lv.score > 0 and (not fresh_only or lv.last_seen == tick)),
                key=lambda lv: -lv.score)                   # fresh_only: only levels re-scored at this tick
  temperature = bundle_temperature or buf.cfg["temperature"]   # bundle weights only (proposals keep the buffer's)
  rank = [(1.0 / (i + 1)) ** (1.0 / temperature) for i in range(len(live))]
  i = 0
  while i < len(live):                                      # equal scores share their rank weight equally
    j = i
    while j + 1 < len(live) and live[j + 1].score == live[i].score:
      j += 1
    mean = sum(rank[i:j + 1]) / (j + 1 - i)
    rank[i:j + 1] = [mean] * (j + 1 - i)
    i = j + 1
  stale = [max(buf.tick - lv.last_seen, 0) for lv in live]
  rho = buf.cfg["staleness"]                                # PLR replay mixture: rank priority + staleness
  ranks = [(1 - rho) * r / max(sum(rank), 1e-12) + rho * (st / sum(stale) if sum(stale) else 1.0 / len(live))
           for r, st in zip(rank, stale)]
  share, effective = share_rule(state, tick, anchor, has_levels=bool(live))
  bundle_levels = [dict(level_id=lv.level_id, family=lv.family, params=lv.params, weight=r,
                        env=dict(hits=F2.hits_of(lv.params), start=F2.start_of(lv.params)))
                   for lv, r in zip(live, ranks)]
  manifest = LF.write_tick(feed, tick + 1, bundle_levels, dict(distribution, level_share=share), source_sha256=F2.source_sha256())
  if learner_run is not None:                               # the learner's outcomes, kept as a snapshot witness cache
    path = cache_path(state, f"learner_tick{tick}")         # (coordinator: learner snapshots become witnesses later)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(dict(by_sha[s["spec_sha256"]], spec_sha256=s["spec_sha256"]), sort_keys=True) + "\n"
                            for s in specs))
  (state / f"report_tick{tick}.json").write_text(json.dumps(report, indent=0))
  return dict(tick=tick, levels=len(report), live=len(live), level_share=effective, level_share_rule=share, anchor=anchor,
              coin_delta=coin_delta, bundle_temperature=temperature, fresh_only=fresh_only, top_weight=round(max(ranks), 4) if ranks else None,
              snapshot_witnesses=snapshots, manifest=manifest,
              learner_ok=sum(r["learner_k"] for r in report), n=sum(r["n"] for r in report))


def main():
  ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = ap.add_subparsers(dest="cmd", required=True)
  p = sub.add_parser("propose")
  p.add_argument("--state", type=Path, required=True)
  p.add_argument("--tick", type=int, required=True)
  p.add_argument("--count", type=int, default=64)
  p.add_argument("--episodes", type=int, default=8)
  p.add_argument("--start-default", action="store_true", help="R2e: nominal-pose starts only")
  u = sub.add_parser("subset")
  u.add_argument("--state", type=Path, required=True)
  u.add_argument("--tick", type=int, required=True)
  u.add_argument("--name", required=True)
  c = sub.add_parser("cache")
  c.add_argument("--state", type=Path, required=True)
  c.add_argument("--name", required=True)
  c.add_argument("--run", type=Path, required=True)
  c.add_argument("--specs", type=Path, required=True)
  s = sub.add_parser("score")
  s.add_argument("--state", type=Path, required=True)
  s.add_argument("--tick", type=int, required=True)
  s.add_argument("--learner-run", default="none", help="the learner's probe run dir, or none (tick 0 before training)")
  s.add_argument("--witnesses", default="")
  s.add_argument("--snapshots", default="", help="learner_tick<k>,... (cache-only witnesses; coordinator: later)")
  s.add_argument("--coin-delta", default=str(COIN_DELTA),
                 help="the coin filter's delta (12:30 decision 0.1), off, or mastered (drop only levels the learner passes)")
  s.add_argument("--bundle-temperature", type=float, default=None, help="rank temperature of the bundle weights (buffer: 0.3)")
  s.add_argument("--fresh-only", action="store_true", help="bundle = levels re-scored at this tick only")
  s.add_argument("--feed", type=Path, required=True)
  s.add_argument("--distribution", default="shared/r2/r2_train_distribution_v1.json")
  s.add_argument("--distribution-sha256", default="9dff1729a8b0a69f0edd77b6b7d1033f5d9499fe8396cdb17ad701a029ba10b8")
  args = ap.parse_args()
  if args.cmd == "propose":
    print(propose(args.state, args.tick, args.count, args.episodes, args.start_default))
  elif args.cmd == "subset":
    out = subset(args.state, args.tick, args.name)
    print(out, sum(1 for l in out.read_text().splitlines() if l.strip()))
  elif args.cmd == "cache":
    print(add_to_cache(args.state, args.name, args.run, args.specs))
  else:
    print(json.dumps(score(args.state, args.tick, None if args.learner_run == "none" else Path(args.learner_run),
                           [w for w in args.witnesses.split(",") if w], args.feed,
                           dict(path=args.distribution, sha256=args.distribution_sha256),
                           None if args.coin_delta == "off" else args.coin_delta if args.coin_delta in ("mastered", "self")
                           else float(args.coin_delta),
                           tuple(w for w in args.snapshots.split(",") if w), args.bundle_temperature, args.fresh_only), indent=1))


if __name__ == "__main__":
  main()
