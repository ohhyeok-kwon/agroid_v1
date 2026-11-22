"""Seed tree, version-stable draws and canonical hashing.

round seed -> family seed -> instance seed: child = first 63 bits of SHA-256("parent|name"). Draws come from BLAKE2b
counters over (seed, k), so a spec depends only on (generator version, source hash, seed path), never on NumPy's
distribution code, the platform or the batch. Floats are rounded to 6 decimals before hashing.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MASK63 = (1 << 63) - 1
SPEC_FILES = ("__init__.py", "seedtree.py", "bank.py", "generator.py")   # files that decide a spec's content


def child_seed(parent: int, name: str) -> int:
  digest = hashlib.sha256(f"{int(parent)}|{name}".encode()).digest()
  return int.from_bytes(digest[:8], "little") & MASK63


def r6(value: float) -> float:
  value = float(value)
  if not math.isfinite(value):
    raise ValueError("non-finite value in a spec")
  value = round(value, 6)
  return 0.0 if value == 0.0 else value


class Draw:
  """Counter-based uniform draws: u_k = BLAKE2b(seed:k) >> 11, scaled to [0, 1)."""

  def __init__(self, seed: int):
    self.seed, self.k = int(seed), 0

  def u(self) -> float:
    digest = hashlib.blake2b(f"{self.seed}:{self.k}".encode(), digest_size=8).digest()
    self.k += 1
    return (int.from_bytes(digest, "little") >> 11) * (1.0 / (1 << 53))

  def uniform(self, lo: float, hi: float) -> float:
    return lo + (hi - lo) * self.u()

  def loguniform(self, lo: float, hi: float) -> float:
    return math.exp(self.uniform(math.log(lo), math.log(hi)))

  def integer(self, lo: int, hi: int) -> int:
    """Uniform integer in [lo, hi] (inclusive)."""
    return lo + min(int(self.u() * (hi - lo + 1)), hi - lo)

  def bernoulli(self, p: float) -> bool:
    return self.u() < p

  def choice(self, options, weights=None):
    options = list(options)
    if weights is None:
      return options[self.integer(0, len(options) - 1)]
    total = float(sum(weights))
    x, acc = self.u() * total, 0.0
    for option, weight in zip(options, weights):
      acc += weight
      if x < acc:
        return option
    return options[-1]


def canonical(value) -> str:
  return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
  return hashlib.sha256(text.encode()).hexdigest()


def spec_hash(spec: dict) -> str:
  return sha256_text(canonical({k: v for k, v in spec.items() if k != "spec_sha256"}))


def source_sha256() -> str:
  """Hash of the files that decide a spec (name and bytes, in a fixed order)."""
  here = Path(__file__).resolve().parent
  digest = hashlib.sha256()
  for name in SPEC_FILES:
    digest.update(name.encode() + b"\0" + (here / name).read_bytes() + b"\0")
  return digest.hexdigest()
