"""Level feed between the negative's level search and a training environment (R3 design, implemented on the CPU side;
the training connection is decided by the coordinator after the positive's one-factor runs).

Files in a feed directory (the negative writes, a training env only reads):
  feed_tick<k>.json      one tick's level bundle (never rewritten): levels [{level_id, family, params, weight}], the
                         training distribution it sits on (path + sha256), frozen flag and evidence, the code hash
  feed_manifest.json     the only file that changes: current tick, bundle name and its sha256. Replaced atomically
                         (written to a temporary file in the same directory, then os.replace), so a reader always sees
                         either the old or the new complete manifest, and the bundle it names already exists.
Reader (training env side):
  feed = LevelFeed(directory); feed.refresh() -> True when a new tick was loaded (sha256 checked; a bad bundle is
  refused and the old one kept); feed.sample(seed, env_ids, episodes) -> one level per env, drawn by weight from a
  counter-based stream keyed by (training seed, env id, episode counter): the same run seed reproduces the same draws.
Deterministic probe (negative side): probe_specs(bundle, make_spec, episodes) -> the evaluation episodes of a bundle
  for runner (deterministic actions), so the learner's p per level comes from deterministic rollouts only.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path

from r1adv.negative.seedtree import Draw, canonical, child_seed

SCHEMA = "r1adv-level-feed-v1"


def sha256_bytes(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
  fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
  try:
    with os.fdopen(fd, "wb") as f:
      f.write(data)
      f.flush()
      os.fsync(f.fileno())
    os.replace(tmp, path)
  finally:
    if os.path.exists(tmp):
      os.unlink(tmp)


def write_tick(directory: Path, tick: int, levels: list[dict], distribution: dict, frozen: str | None = None,
               source_sha256: str = "") -> dict:
  """Write feed_tick<tick>.json (refuses to overwrite) and then point the manifest at it. levels: [{level_id, family,
  params, weight}] with weight > 0; the weights are normalised here. Returns the manifest."""
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  total = sum(float(lv["weight"]) for lv in levels)
  if levels and total <= 0:
    raise ValueError("level weights must be positive")
  # an empty bundle is allowed (no level admissible): its distribution.level_share is 0, so a reader that draws a
  # level with that probability (positive hold_r2c) never samples it; the rule's share is kept as level_share_rule
  effective = float(distribution.get("level_share", 0.0)) if levels else 0.0
  distribution = dict(distribution, level_share=effective, level_share_rule=float(distribution.get("level_share", 0.0)))
  bundle = dict(schema=SCHEMA, tick=int(tick), frozen=frozen, distribution=distribution, source_sha256=source_sha256,
                levels=[dict(dict(level_id=lv["level_id"], family=lv["family"], params=lv["params"],
                                  weight=float(lv["weight"]) / total), **({"env": lv["env"]} if "env" in lv else {}))
                        for lv in levels])
  bundle["level_share"] = effective
  data = (canonical(bundle) + "\n").encode()
  name = f"feed_tick{int(tick)}.json"
  path = directory / name
  if path.exists():
    if path.read_bytes() != data:
      raise FileExistsError(f"{name} exists with other content (bundles are never rewritten)")
  else:
    _atomic_write(path, data)
  manifest = dict(schema=SCHEMA, tick=int(tick), bundle=name, sha256=sha256_bytes(data), levels=len(levels),
                  level_share=bundle["level_share"])
  _atomic_write(directory / "feed_manifest.json", (json.dumps(manifest, sort_keys=True) + "\n").encode())
  return manifest


class LevelFeed:
  """Read side. Keeps the last good bundle; refresh() swaps only to a complete bundle whose sha256 matches."""

  def __init__(self, directory: Path):
    self.directory = Path(directory)
    self.tick, self.bundle, self.sha256, self.history = None, None, None, []
    self._cumulative = []

  def refresh(self) -> bool:
    try:
      manifest = json.loads((self.directory / "feed_manifest.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
      return False
    if manifest.get("schema") != SCHEMA or manifest["tick"] == self.tick:
      return False
    data = (self.directory / manifest["bundle"]).read_bytes()
    if sha256_bytes(data) != manifest["sha256"]:
      self.history.append(dict(tick=manifest["tick"], refused="sha256 mismatch"))
      return False
    bundle = json.loads(data)
    self.tick, self.bundle, self.sha256 = manifest["tick"], bundle, manifest["sha256"]
    acc, self._cumulative = 0.0, []
    for lv in bundle["levels"]:
      acc += lv["weight"]
      self._cumulative.append(acc)
    self.history.append(dict(tick=self.tick, sha256=self.sha256))
    return True

  def level_share(self) -> float:
    """The probability that a reset draws a level (0 without a bundle or with an empty one)."""
    return float(self.bundle.get("level_share", 0.0)) if self.bundle and self.bundle["levels"] else 0.0

  def sample(self, seed: int, env_ids, episodes) -> list[dict]:
    """One level per (env id, episode counter): a uniform draw from child_seed(seed, f'feed/{tick}/{env}/{episode}')
    mapped through the cumulative weights. Same seed, env and episode -> same level for a given tick."""
    if self.bundle is None:
      raise RuntimeError("no bundle loaded (call refresh first)")
    if not self.bundle["levels"]:
      raise RuntimeError("the current bundle is empty: draw from the base distribution (level_share is 0)")
    out = []
    for env, episode in zip(env_ids, episodes):
      u = Draw(child_seed(int(seed), f"feed/{self.tick}/{int(env)}/{int(episode)}")).u() * self._cumulative[-1]
      index = next(i for i, c in enumerate(self._cumulative) if u < c or i == len(self._cumulative) - 1)
      out.append(self.bundle["levels"][index])
    return out


def probe_specs(bundle: dict, make_spec, episodes: int) -> list[dict]:
  """The deterministic probe of a bundle: make_spec(level, episode) -> a runner spec (deterministic actions in the
  runner); returns episodes x levels specs in bundle order."""
  return [make_spec(lv, e) for lv in bundle["levels"] for e in range(episodes)]
