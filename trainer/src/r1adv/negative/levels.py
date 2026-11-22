"""Lower layer of generator v0.2: the level sampler (PLR replay + ACCEL edits) and the level score.

level = (family, family source hash, params, goal seed); its mission comes from the coordinator's d_goal sampler
(generator.mission) and its environment from family.make(); every episode of a level differs only by nuisance
(sensor-noise seed, +-2 deg spawn yaw, +-0.04 s impulse time), so p is the learner's robustness on that level.
  S = E[p (1 - p)] under Beta(k + 1, n - k + 1)  x  1[witness calm-success lower bound >= tau]
p and the witness rate count complete (calm) successes as score.py judges them (F, G, J; the family never judges).
Proposals per round: a baseline share (against forgetting), PLR replay by rank of S mixed with staleness, ACCEL edits
of replayed levels, and new levels; a cap per family. Seeds: H(root, round, component, level id, episode), counter
draws, independent of batch size and GPU. The buffer state and every decision are written as JSON (append-only log).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from r1adv.negative import NAME, generator as G
from r1adv.negative.families import FAMILIES, family_source_sha256, mutate, sample
from r1adv.negative.seedtree import Draw, canonical, child_seed, r6, sha256_text, spec_hash


def beta_learnability(k: int, n: int) -> float:
  a, b = k + 1.0, n - k + 1.0
  return a * b / ((a + b) * (a + b + 1.0))


def wilson_lower(k: int, n: int, z: float = 1.645) -> float:
  if n == 0:
    return 0.0
  p = k / n
  centre = (p + z * z / (2 * n)) / (1 + z * z / n)
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
  return max(0.0, centre - half)


@dataclass
class Level:
  level_id: str
  family: str
  params: dict
  goal_seed: int
  parent: str | None = None
  born: str = ""
  learner: list = field(default_factory=lambda: [0, 0])        # complete successes, episodes
  witness: list = field(default_factory=lambda: [0, 0])
  signatures: dict = field(default_factory=dict)
  last_seen: int = -1
  score: float = 0.0
  infeasible: list = field(default_factory=list)


def level_id(family: str, params: dict, goal_seed: int, source: str) -> str:
  return sha256_text(canonical(dict(family=family, params=params, goal_seed=int(goal_seed), source=source)))[:24]


def episode_spec(level: Level, episode: int, round_name: str, source: str, registry: dict | None = None,
                 mission_fn=None, version: str = "v0.2") -> dict:
  family = (registry or FAMILIES)[level.family]
  goal = Draw(child_seed(level.goal_seed, "goal"))
  mission = mission_fn(goal) if mission_fn else G.mission(goal, "legacy-v0" if family.runnable == "legacy" else "full-v0")
  env = family.make(level.params, Draw(child_seed(level.goal_seed, "env")), mission)
  seed = child_seed(level.goal_seed, f"episode/{episode}")
  nuisance = Draw(child_seed(seed, "nuisance"))
  env["sense"]["noise_seed"] = child_seed(seed, "sense/noise")
  if env.get("terrain"):
    env["terrain"]["spawn_yaw_rad"] = r6(env["terrain"]["spawn_yaw_rad"] + math.radians(nuisance.uniform(-2.0, 2.0)))
  for item in env["disturbances"]:
    if "t_s" in item:
      item["t_s"] = G.ticks(max(G.SETTLE_S + 0.2, item["t_s"] + nuisance.uniform(-0.04, 0.04)))
  spec = dict(schema="r1adv-scenario-v1", generator=dict(name=NAME, version=version, family=level.family,
                                                         source_sha256=source, level_id=level.level_id),
              seed=seed, seed_path=f"{round_name}/{level.family}/{level.level_id}/{episode}", mission=mission, env=env,
              solvability_evidence=None)
  spec["spec_sha256"] = spec_hash(spec)
  return spec


class LevelBuffer:
  def __init__(self, round_name: str, *, capacity: int = 4096, baseline_share: float = 0.25, family_cap: float = 0.35,
               new_share: float = 0.3, edit_share: float = 0.3, staleness: float = 0.3, temperature: float = 0.3,
               tau: float = 0.7, novelty: float = 0.05, families: tuple | None = None, runnable: tuple = ("legacy",),
               registry: dict | None = None, source: str | None = None, mission_fn=None, version: str = "v0.2"):
    self.round, self.capacity = round_name, capacity
    self.registry, self.mission_fn, self.version = registry or FAMILIES, mission_fn, version
    self.cfg = dict(baseline_share=baseline_share, family_cap=family_cap, new_share=new_share, edit_share=edit_share,
                    staleness=staleness, temperature=temperature, tau=tau, novelty=novelty)
    self.families = [f for f in (families or tuple(self.registry)) if self.registry[f].runnable in runnable]
    self.levels: dict[str, Level] = {}
    self.source = source or family_source_sha256()
    self.tick = 0
    self.log: list[dict] = []
    self.seen_signatures: set = set()

  # -- proposals --------------------------------------------------------------------------------------------------
  def _new(self, d: Draw, family: str, parent: Level | None = None) -> Level:
    spec = self.registry[family].PARAM_SPACE
    params = mutate(parent.params, spec, d) if parent else sample(spec, d)
    goal = parent.goal_seed if parent and d.bernoulli(0.5) else child_seed(d.integer(0, (1 << 62)), "goal-seed")
    level = Level(level_id(family, params, goal, self.source), family, params, goal, parent.level_id if parent else None,
                  born=f"{self.round}/{self.tick}", infeasible=self.registry[family].feasible(params))
    return level

  def propose(self, count: int) -> list[Level]:
    """Levels to evaluate this tick (deterministic given round, tick and the buffer)."""
    d = Draw(child_seed(G.round_seed(self.round), f"sampler/tick/{self.tick}"))
    cfg, chosen = self.cfg, []
    per_family = {f: 0 for f in self.families}
    def room(family):
      return per_family.get(family, 0) < max(1, int(cfg["family_cap"] * count))
    def take(level):
      if level.infeasible:
        self.log.append(dict(tick=self.tick, event="rejected", level=level.level_id, family=level.family, why=level.infeasible))
        return False
      self.levels.setdefault(level.level_id, level)
      chosen.append(self.levels[level.level_id])
      per_family[level.family] = per_family.get(level.family, 0) + 1
      return True
    baseline = [f for f in self.families if self.registry[f].kind == "baseline"]
    for _ in range(int(cfg["baseline_share"] * count)):
      if baseline:
        take(self._new(d, d.choice(baseline)))
    scored = sorted((lv for lv in self.levels.values() if lv.score > 0), key=lambda lv: -lv.score)
    n_replay = int((1 - cfg["new_share"] - cfg["baseline_share"]) * count) if scored else 0
    if scored:
      ranks = [(1.0 / (i + 1)) ** (1.0 / cfg["temperature"]) for i in range(len(scored))]
      stale = [self.tick - lv.last_seen for lv in scored]
      total_r, total_s = sum(ranks), max(sum(stale), 1)
      weights = [(1 - cfg["staleness"]) * r / total_r + cfg["staleness"] * s / total_s for r, s in zip(ranks, stale)]
      for _ in range(n_replay):
        level = d.choice(scored, weights)
        if d.bernoulli(cfg["edit_share"]):
          child = self._new(d, level.family, parent=level)
          if room(child.family):
            take(child)
        elif room(level.family) and level not in chosen:
          take(level)
    attempts = 0
    while len(chosen) < count and attempts < 20 * count:
      attempts += 1
      family = d.choice([f for f in self.families if self.registry[f].kind != "baseline"])
      if room(family):
        take(self._new(d, family))
    self.log.append(dict(tick=self.tick, event="proposed", count=len(chosen),
                         by_family={f: n for f, n in per_family.items() if n}))
    return chosen

  # -- scores -----------------------------------------------------------------------------------------------------
  def update(self, results: dict) -> None:
    """results[level_id] = dict(learner=[bool...], witness=[bool...], signatures=[str...])."""
    for lid, res in results.items():
      level = self.levels[lid]
      level.learner = [level.learner[0] + sum(res["learner"]), level.learner[1] + len(res["learner"])]
      level.witness = [level.witness[0] + sum(res["witness"]), level.witness[1] + len(res["witness"])]
      new_sig = 0
      for sig in res.get("signatures", []):
        level.signatures[sig] = level.signatures.get(sig, 0) + 1
        if sig not in self.seen_signatures:
          self.seen_signatures.add(sig)
          new_sig += 1
      level.last_seen = self.tick
      solvable = wilson_lower(*level.witness) >= self.cfg["tau"]
      level.score = r6(beta_learnability(*level.learner) * solvable + self.cfg["novelty"] * min(new_sig, 1) * solvable)
      self.log.append(dict(tick=self.tick, event="scored", level=lid, family=level.family, learner=level.learner,
                           witness=level.witness, solvable=solvable, score=level.score, new_signatures=new_sig))
    if len(self.levels) > self.capacity:
      keep = sorted(self.levels.values(), key=lambda lv: (-lv.score, -lv.last_seen))[:self.capacity]
      self.levels = {lv.level_id: lv for lv in keep}
    self.tick += 1

  def episodes(self, level: Level, count: int) -> list[dict]:
    return [episode_spec(level, e, self.round, self.source, self.registry, self.mission_fn, self.version) for e in range(count)]

  def save(self, path: Path) -> None:
    Path(path).write_text(json.dumps(dict(schema="r1adv-level-buffer-v1", round=self.round, tick=self.tick, cfg=self.cfg,
                                          source_sha256=self.source, families=self.families,
                                          levels=[asdict(lv) for lv in self.levels.values()], log=self.log,
                                          seen_signatures=sorted(self.seen_signatures)), indent=0))


def pick_witness(outcomes: dict, rule: str, learner_obs: str, observation_of: dict) -> tuple[str | None, list]:
  """outcomes[policy] = list of (complete, J median) over the level's episodes. Physical rule: the policy with the most
  complete successes, ties to the lowest median J; observation rule: the same among policies that read the learner's
  observations. The manufacturer is never a candidate."""
  candidates = [p for p in outcomes if p != "manufacturer" and (rule == "physical" or observation_of.get(p) == learner_obs)]
  if not candidates:
    return None, []
  def key(p):
    rows = outcomes[p]
    ok = [j for c, j in rows if c and j is not None]
    return (-sum(c for c, _ in rows), sorted(ok)[len(ok) // 2] if ok else float("inf"))
  best = min(candidates, key=key)
  return best, [c for c, _ in outcomes[best]]
