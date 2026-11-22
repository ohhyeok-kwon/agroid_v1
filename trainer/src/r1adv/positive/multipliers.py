"""Per-scenario constraint pressure (NOT part of the fixed objective; never adds anything to the reward).

Status 2026-11-17 01:40 KST: the coordinator fixed the objective as r = L + beta * D (objective.py, panel2 synthesis 1).
This module only keeps, per scenario bin xi and per measured quantity c (fall rate, normalised goal error, ...), a dual
variable mu_c(xi) <- clip(mu_c(xi) + lr_c * (mean_c(xi) - eps_c), 0, mu_max) as a difficulty signal for the negative
agent and the coordinator: a value that keeps rising marks a scenario the policy does not yet solve, zero marks a
solved one. Training-side bookkeeping only; nothing of this exists in the deployed network.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


class ScenarioMultipliers:
  def __init__(self, bins: int, limits: dict[str, float], lr: dict[str, float], mu_max: float = 100.0,
               init: float = 0.0, device: str = "cpu"):
    self.names = tuple(limits)
    self.limits = torch.tensor([limits[n] for n in self.names], device=device)
    self.lr = torch.tensor([lr[n] for n in self.names], device=device)
    self.mu = torch.full((bins, len(self.names)), float(init), device=device)
    self.mu_max = mu_max
    self.sum = torch.zeros_like(self.mu)
    self.count = torch.zeros(bins, device=device)
    self.history: list[list[list[float]]] = []

  def weights(self, bin_ids: torch.Tensor) -> torch.Tensor:
    """(N, C) multipliers of each environment's scenario bin."""
    return self.mu[bin_ids]

  def record(self, bin_ids: torch.Tensor, values: torch.Tensor, done: torch.Tensor) -> None:
    """Accumulate per-episode constraint values (N, C) for episodes that ended this step."""
    if not bool(done.any()):
      return
    ids, vals = bin_ids[done], values[done]
    self.sum.index_add_(0, ids, vals)
    self.count.index_add_(0, ids, torch.ones_like(ids, dtype=self.count.dtype))

  def update(self, min_episodes: int = 20) -> dict:
    """Dual ascent step for bins with enough finished episodes; returns a log row."""
    seen = self.count >= min_episodes
    mean = self.sum / self.count.clamp_min(1.0).unsqueeze(1)
    step = self.lr[None] * (mean - self.limits[None])
    self.mu = torch.where(seen.unsqueeze(1), (self.mu + step).clamp(0.0, self.mu_max), self.mu)
    row = dict(mu=self.mu.tolist(), mean=mean.tolist(), episodes=self.count.tolist())
    self.history.append(self.mu.tolist())
    self.sum.zero_()
    self.count.zero_()
    return row

  def state_dict(self) -> dict:
    return dict(names=self.names, limits=self.limits.tolist(), lr=self.lr.tolist(), mu=self.mu.tolist(),
                mu_max=self.mu_max, history=self.history)

  def save(self, path: Path) -> None:
    Path(path).write_text(json.dumps(self.state_dict()))
