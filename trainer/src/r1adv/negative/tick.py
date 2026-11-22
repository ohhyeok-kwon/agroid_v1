"""One tick of the R1 level search: score a tick's episodes (criteria r1), pick witnesses (rule v1), update the level
buffer, propose the next tick and write it as the level stream (see rounds/R01/negative/level_stream.md).

  python -m r1adv.negative.tick --round R01 --tick 0 --runs <dir with <policy>/arrays.npz> --human human_j_v1.json \
      --learner positive_b1_prev_check300 --stream <stream dir> [--count 128 --episodes 8]

Witness rule v1.1 (coordinator r1a): a sensor-degraded level may be witnessed only by a policy that uses no more
information than the learner (information atoms below, privileged simulator inputs never); physical levels by the
calmest successful policy of our population. The manufacturer's runs, if present, are reported for comparison and
never enter p, the witness or S. Level scores use DETERMINISTIC actions (the policy mean; runner never samples);
stochastic rollouts belong to the training side only.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

from r1adv.negative import families_ext as FX
from r1adv.negative import families_v1 as F1
from r1adv.negative import score as SC
from r1adv.negative.levels import Level, LevelBuffer, beta_learnability, pick_witness, wilson_lower
from r1adv.negative.seedtree import canonical, r6


def wilson_upper(k: int, n: int) -> float:
  return 1.0 - wilson_lower(n - k, n) if n else 1.0

OBSERVATION = {"positive": "deploy810", "v12h": "teacher417", "slope45": "teacher417", "rad10x_student": "studentA1227",
               "deployable_D": "studentD1227", "manufacturer": "manufacturer"}


INFORMATION = {                                   # what each observation class can know (privileged = simulator only)
  "deploy570": {"imu", "encoders", "command", "history"},
  "deploy810": {"imu", "encoders", "command", "history", "previous_action"},
  "studentD1227": {"imu", "encoders", "command", "history", "previous_action"},       # privileged columns are zero
  "studentA1227": {"imu", "encoders", "command", "history", "previous_action", "privileged"},
  "teacher417": {"imu", "encoders", "command", "previous_action", "privileged"},
}


def may_witness_sensor_level(candidate: str, learner: str) -> bool:
  have, limit = INFORMATION.get(candidate), INFORMATION.get(learner)
  return have is not None and limit is not None and "privileged" not in have and have <= limit


def observation_of(name: str, run_dir: Path | None = None) -> str:
  """Observation class of a run: deploy-contract runs by the width their meta records (810 or 570)."""
  if run_dir is not None and (run_dir / "meta.json").exists():
    meta = json.loads((run_dir / "meta.json").read_text())
    if meta.get("identity", {}).get("kind") == "deploy":
      width = meta.get("deploy_width")
      if width is None and meta.get("checkpoint"):          # runs before the meta field: the actor's input width
        import torch
        state = torch.load(meta["checkpoint"], map_location="cpu", weights_only=False)["actor_state_dict"]
        first = min(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight"))
        width = int(state[f"mlp.{first}.weight"].shape[1])
      return f"deploy{width if width is not None else 'unknown'}"
  for prefix, kind in OBSERVATION.items():
    if name == prefix or name.startswith(prefix + "_"):
      return kind
  return "unknown"


def buffer_for(round_name: str, tau: float) -> LevelBuffer:
  return registry_buffer(round_name, tau, "v1")


def registries() -> dict:
  """Family sets of the Hold stream. v1: the R1 Hold set. v1.2 (coordinator r1b-1): v1 + thrown objects and body
  randomization (families_ext; the vibrating floor is held back until it has a criterion)."""
  return {
    "v1": dict(registry=F1.FAMILIES, families=tuple(F1.FAMILIES), source=F1.source_sha256(), version="v1"),
    "v1.2": dict(registry={**F1.FAMILIES, **FX.FAMILIES}, families=tuple(F1.FAMILIES) + ("hold-throw-v1", "hold-body-v1"),
                 source=FX.source_sha256(), version="v1.2"),
  }


def registry_buffer(round_name: str, tau: float, name: str) -> LevelBuffer:
  r = registries()[name]
  buffer = LevelBuffer(round_name, registry=r["registry"], families=r["families"], source=r["source"],
                       mission_fn=F1.mission_hold, version=r["version"], tau=tau)
  buffer.registry_name = name
  return buffer


def switch_registry(buffer: LevelBuffer, name: str) -> None:
  """Between ticks only: levels already in the buffer keep their ids; their next episodes carry the new source."""
  r = registries()[name]
  buffer.log.append(dict(tick=buffer.tick, event="registry", previous=getattr(buffer, "registry_name", None), now=name,
                         source=r["source"], families=list(r["families"])))
  buffer.registry, buffer.source, buffer.version = r["registry"], r["source"], r["version"]
  buffer.families = [f for f in r["families"] if r["registry"][f].runnable == "legacy"]
  buffer.registry_name = name


def load_buffer(path: Path, tau: float) -> LevelBuffer:
  """The buffer as saved before tick k's proposal (buffer_tick<k>.json); propose() then re-creates levels_tick<k>."""
  data = json.loads(path.read_text())
  names = [n for n, r in registries().items() if r["source"] == data["source_sha256"]]
  if not names:
    raise ValueError("family source changed since this buffer was written")
  buffer = registry_buffer(data["round"], tau, names[0])
  if list(buffer.families) != list(data["families"]):
    raise ValueError("the buffer's family list differs from its registry")
  buffer.tick, buffer.cfg, buffer.log = data["tick"], data["cfg"], data["log"]
  buffer.seen_signatures = set(data["seen_signatures"])
  buffer.levels = {row["level_id"]: Level(**row) for row in data["levels"]}
  return buffer


def unique(levels: list) -> list:
  out, ids = [], set()
  for level in levels:
    if level.level_id not in ids:
      ids.add(level.level_id)
      out.append(level)
  return out


def replay_to(buffer: LevelBuffer, tick: int, count: int) -> list:
  if tick != buffer.tick:
    raise ValueError(f"buffer is at tick {buffer.tick}, not {tick}")
  return buffer.propose(count)


def signature(row: dict) -> str:
  if row["F"]:
    return "F"
  bad = [r for r in row["steps"] if r.get("G_ratio_r1", 0) > 1 or r.get("J_ratio_r1", 0) > 1]
  if not bad:
    return "ok"
  r = max(bad, key=lambda r: max(r.get("G_ratio_r1", 0), r.get("J_ratio_r1", 0)))
  return ("G" if r.get("G_ratio_r1", 0) > 1 else "") + ("J" if r.get("J_ratio_r1", 0) > 1 else "") + ":" + r.get("J_type", r["cls"])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--round", default="R01")
  parser.add_argument("--tick", type=int, default=0)
  parser.add_argument("--specs", type=Path, required=True)
  parser.add_argument("--runs", type=Path, required=True)
  parser.add_argument("--human", type=Path, required=True)
  parser.add_argument("--learner", required=True)
  parser.add_argument("--tau", type=float, default=0.5)
  parser.add_argument("--count", type=int, default=128)
  parser.add_argument("--episodes", type=int, default=8)
  parser.add_argument("--stream", type=Path, required=True)
  parser.add_argument("--next-registry", choices=("v1", "v1.2"), default=None,
                      help="family set from the next proposal on (default: keep the buffer's)")
  parser.add_argument("--coin-delta", type=float, default=None,
                      help="R2 (off when absent, so published R1 ticks re-score as before; R2 ticks pass 0.1): a level is "
                           "solvable only if witness Wilson lower >= max(tau, learner Wilson upper + delta)")
  parser.add_argument("--freeze", default=None,
                      help="R2 difficulty freeze while the learner trembles: the evidence (text); the next tick re-issues "
                           "this tick's levels (same specs, common random numbers across ticks), no new levels or edits")
  args = parser.parse_args()
  specs = [json.loads(line) for line in args.specs.read_text().splitlines() if line.strip()]
  human = json.loads(args.human.read_text())
  runs = {p.name: p for p in sorted(args.runs.iterdir()) if (p / "arrays.npz").exists()}
  actions = {}
  for name, run in runs.items():                    # level scores come from deterministic actions only
    stated = json.loads((run / "meta.json").read_text()).get("actions")
    if stated is None:
      actions[name] = "deterministic (inferred: runner before the meta field; every policy path returns the mean)"
    elif not str(stated).startswith("deterministic"):
      raise SystemExit(f"{name}: level scores need deterministic actions, the run says {stated!r}")
    else:
      actions[name] = stated
  scored = SC.score_runs_r1(specs, runs, human)
  current = {}
  if args.tick == 0:
    buffer = buffer_for(args.round, args.tau)
  else:                                              # the manifest names the current buffer (a revision may supersede)
    current = json.loads((args.stream / "manifest.json").read_text()) if (args.stream / "manifest.json").exists() else {}
    name = current.get("buffer_file", f"buffer_tick{args.tick}.json") if current.get("latest_tick") == args.tick else f"buffer_tick{args.tick}.json"
    buffer = load_buffer(args.stream / name, args.tau)
  frozen = current.get("frozen") if current.get("latest_tick") == args.tick else None
  if frozen:                                         # a frozen tick re-issued the previous tick's levels (no proposal)
    levels = [buffer.levels[lid] for lid in frozen["level_ids"]]
  else:
    levels = replay_to(buffer, args.tick, args.count)
  have = [s["spec_sha256"] for s in specs]
  if have not in ([s["spec_sha256"] for lv in levels for s in buffer.episodes(lv, args.episodes)],
                  [s["spec_sha256"] for lv in unique(levels) for s in buffer.episodes(lv, args.episodes)]):
    raise ValueError("the evaluated specs are not this tick's proposal")
  # an ACCEL edit that moves no parameter and keeps the goal is its parent (same id): a level can be proposed twice.
  # Each level is scored once, on its distinct episodes (repeated identical specs are not new evidence).
  levels = unique(levels)
  index, seen = collections.defaultdict(list), set()
  for i, s in enumerate(specs):
    if s["spec_sha256"] not in seen:
      seen.add(s["spec_sha256"])
      index[s["generator"]["level_id"]].append(i)
  learner_obs = observation_of(args.learner, runs.get(args.learner))
  results, report = {}, []
  for level in levels:
    idx = index[level.level_id]
    rule = F1.witness_rule(specs[idx[0]]["env"])
    outcomes = {p: [(rows[i]["complete_r1"], rows[i]["steps"][0].get("J")) for i in idx] for p, rows in scored.items()}
    classes = {p: observation_of(p, runs.get(p)) for p in outcomes}
    # rule v1.1: observation levels admit every class with no more information than the learner's (pick_witness
    # compares equality, so the admissible classes are mapped onto the learner's class first)
    mapped = {p: (learner_obs if may_witness_sensor_level(c, learner_obs) else c) for p, c in classes.items()}
    witness, witness_ok = pick_witness(outcomes, rule, learner_obs, mapped)
    learner_ok = [scored[args.learner][i]["complete_r1"] for i in idx]
    results[level.level_id] = dict(learner=learner_ok, witness=witness_ok,
                                   signatures=[f"{level.family}/{signature(scored[args.learner][i])}" for i in idx])
    report.append(dict(level=level.level_id, family=level.family, params=level.params, rule=rule, witness=witness,
                       witness_k=int(sum(witness_ok)), learner_k=int(sum(learner_ok)), n=len(idx),
                       witness_lb=wilson_lower(int(sum(witness_ok)), len(idx)),
                       learnability=beta_learnability(int(sum(learner_ok)), len(idx)),
                       by_policy={p: int(sum(rows[i]["complete_r1"] for i in idx)) for p, rows in scored.items()}))
  buffer.update(results)
  coins = []
  for lid in (results if args.coin_delta is not None else ()):    # R2 coin filter: witness must beat learner by delta
    level = buffer.levels[lid]
    lb, ub = wilson_lower(*level.witness), wilson_upper(*level.learner)
    if level.score > 0 and lb < max(args.tau, ub + args.coin_delta):
      coins.append(lid)
      buffer.log.append(dict(tick=buffer.tick, event="coin", level=lid, witness_lb=r6(lb), learner_ub=r6(ub), delta=args.coin_delta))
      level.score = 0.0
  if args.freeze:
    buffer.log.append(dict(tick=buffer.tick, event="frozen", evidence=args.freeze, levels=len(levels)))
  if args.next_registry and args.next_registry != getattr(buffer, "registry_name", None):
    switch_registry(buffer, args.next_registry)
  args.stream.mkdir(parents=True, exist_ok=True)
  buffer.save(args.stream / f"buffer_tick{args.tick + 1}.json")
  if args.freeze:
    nxt = levels                                     # same levels, same specs: no new levels, no edits
  else:
    nxt = unique(buffer.propose(args.count))
  episodes = [s for lv in nxt for s in buffer.episodes(lv, args.episodes)]
  out = args.stream / f"levels_tick{args.tick + 1}.jsonl"
  out.write_text("".join(canonical(s) + "\n" for s in episodes))
  (args.stream / f"report_tick{args.tick}.json").write_text(json.dumps(report, indent=0))
  from r1adv.negative.runner import package_sha256, sha256
  manifest = dict(schema="r1adv-level-stream-v1", round=args.round, latest_tick=args.tick + 1, criteria="r1",
                  package_sha256=package_sha256(), actions=actions, witness_rule="v1.1",
                  human_j=dict(path=str(args.human), sha256=sha256(args.human), status=human.get("status", "draft"),
                               table_sha256=human.get("table_sha256"),
                               J_ref={k: v.get("J_ref") for k, v in human["types"].items()}),
                  runs={p: json.loads((d / "meta.json").read_text()).get("package_sha256") for p, d in runs.items()},
                  tau=args.tau, episodes_per_level=args.episodes, learner=args.learner,
                  family_source_sha256=buffer.source, registry=buffer.registry_name, families=list(buffer.families),
                  buffer_file=f"buffer_tick{args.tick + 1}.json", report_file=f"report_tick{args.tick}.json",
                  levels_file=out.name, coin_delta=args.coin_delta, coin_filtered=len(coins),
                  frozen=dict(evidence=args.freeze, level_ids=[lv.level_id for lv in nxt]) if args.freeze else None,
                  levels_sha256=__import__("hashlib").sha256(out.read_bytes()).hexdigest(),
                  live_levels=sum(1 for lv in buffer.levels.values() if lv.score > 0))
  (args.stream / "manifest.json").write_text(json.dumps(manifest, indent=1))
  by_family = collections.defaultdict(list)
  for row in report:
    by_family[row["family"]].append(row)
  print(f"tick {args.tick}: {len(report)} levels; S>0 {sum(1 for lv in buffer.levels.values() if lv.score > 0)}")
  for family, rows in sorted(by_family.items()):
    pols = sorted(rows[0]["by_policy"])
    print(f"  {family:20s} n {len(rows):3d} witness>=tau {sum(r['witness_lb'] >= args.tau for r in rows):3d} "
          + " ".join(f"{p[:14]}={np.mean([r['by_policy'][p] / r['n'] for r in rows]):.2f}" for p in pols))
  print("  next tick:", dict(collections.Counter(lv.family for lv in nxt)))


if __name__ == "__main__":
  main()
