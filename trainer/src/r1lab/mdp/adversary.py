"""Bounded contextual environment adversary, independent of deployment observations.

This is an empirical capability-frontier objective, not PAIRED's regret guarantee.
RARL: arxiv.org/abs/1703.02702; PAIRED: arxiv.org/abs/2012.02096;
ACCEL: arxiv.org/abs/2203.01302. The runner owns synchronization and updates.
"""
from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.distributions import Categorical

KINDS = ("clean", "ball", "force", "push")
CARDINALITIES = (4, 5, 8, 4, 5)  # kind, difficulty, body-relative direction, time, terrain


def challenge_score(success_rate, difficulty, valid=None):
  """Prefer feasible challenges; p must be an empirical group success rate.

  A permanently impossible group scores below a solved group. This does not
  prove solvability. Never substitute a single episode's binary success for p.
  Callers retain raw actor success/recovery scores separately.
  """
  p = success_rate.clamp(0., 1.)
  d = difficulty.float().clamp(0, 4) / 4
  score = 4*p*(1-p) + .25*d*p - .5*(1-p).pow(4)
  return score if valid is None else torch.where(valid, score, -torch.ones_like(score))


def decode_actions(action):
  """Physical budgets only; caller applies one kind, never all simultaneously.

  Forces and velocity jumps intentionally keep separate units/semantics. Ball
  momentum is NOT interpreted as transmitted contact impulse. Terrain levels
  address an existing bounded terrain bank, never arbitrary geometry edits.
  """
  if action.ndim != 2 or action.shape[1] != 5:
    raise ValueError("Expected [N,5] categorical actions")
  if action.dtype != torch.long:
    raise ValueError("Actions must be torch.long")
  for i, size in enumerate(CARDINALITIES):
    if ((action[:, i] < 0) | (action[:, i] >= size)).any():
      raise ValueError("Action outside physical catalogue")
  kind, level, direction, timing, terrain = action.unbind(-1)
  d = level.float()/4
  active = kind != 0
  return dict(kind=kind, difficulty=level, terrain_level=terrain,
              angle=direction.float()*(2*math.pi/8), delay_s=2.+timing.float(),
              ball_speed_mps=6.+12*d, ball_shots=1+(level >= 2).long()+(level >= 4).long(),
              ball_interval_s=torch.full_like(d, .3), ball_mass_kg=torch.ones_like(d),
              ball_radius_m=torch.full_like(d, .07),
              force_n=20.+40*d, force_duration_s=.05+.15*d,
              push_delta_v_mps=.1+.5*d, active=active)


class AdversaryController(nn.Module):
  """Small factored categorical policy; no simulation or hidden optimizer calls.

  sample() retains exact behavior log probabilities and masks. observe() only
  queues detached transitions. Every rank calls update() at the same runner
  barrier, even ranks with zero samples. reduce_sum(tensor) must return the SUM
  over ranks (or mutate in place and return None). Broadcast initial parameters
  and optimizer state before collecting samples. Save only from rank zero.
  """
  def __init__(self, context_dim=8, hidden=64, learning_rate=3e-4,
               exploration=.1, epochs=2, device="cpu"):
    super().__init__()
    if context_dim < 1 or not 0 <= exploration < 1 or epochs < 1:
      raise ValueError("Invalid adversary configuration")
    self.context_dim = context_dim
    self.exploration = exploration
    self.epochs = epochs
    self.net = nn.Sequential(nn.Linear(context_dim, hidden), nn.Tanh(),
                             nn.Linear(hidden, hidden), nn.Tanh())
    self.actor = nn.Linear(hidden, sum(CARDINALITIES))
    self.critic = nn.Linear(hidden, 1)
    # Begin close to uniform rather than with a random preferred attack.
    nn.init.zeros_(self.actor.weight)
    nn.init.zeros_(self.actor.bias)
    self.to(device)
    self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
    self.pending = []
    self.updates = 0

  @property
  def device(self):
    return next(self.parameters()).device

  def _distribution(self, context, mask):
    features = self.net(context)
    logits = self.actor(features)
    distributions = []
    for part, allowed in zip(logits.split(CARDINALITIES, -1), mask.split(CARDINALITIES, -1)):
      probs = part.masked_fill(~allowed, -torch.inf).softmax(-1)
      uniform = allowed.float()/allowed.sum(-1, keepdim=True)
      distributions.append(Categorical(probs=(1-self.exploration)*probs+self.exploration*uniform))
    return distributions, self.critic(features).squeeze(-1)

  @torch.no_grad()
  def sample(self, context, clean_mask=None, max_difficulty=4, max_terrain_level=4,
             deterministic=False):
    context = context.detach().to(self.device, dtype=torch.float32)
    if context.ndim != 2 or context.shape[1] != self.context_dim or not torch.isfinite(context).all():
      raise ValueError("Finite [N,context_dim] context required")
    if not 0 <= max_difficulty <= 4 or not 0 <= max_terrain_level <= 4:
      raise ValueError("Catalogue ceilings must lie in 0..4")
    n = len(context)
    clean = (torch.zeros(n, dtype=torch.bool, device=self.device) if clean_mask is None
             else clean_mask.to(self.device, dtype=torch.bool))
    if clean.shape != (n,):
      raise ValueError("clean_mask must have shape [N]")
    masks = [torch.ones(n, size, dtype=torch.bool, device=self.device) for size in CARDINALITIES]
    masks[1][:, max_difficulty+1:] = False
    masks[4][:, max_terrain_level+1:] = False
    # Reserved clean episodes are flat and attack-free and are not policy data.
    for m in masks:
      m[clean] = False
      m[clean, 0] = True
    mask = torch.cat(masks, -1)
    distributions, value = self._distribution(context, mask)
    action = torch.stack([dist.probs.argmax(-1) if deterministic else dist.sample()
                          for dist in distributions], -1)
    log_prob = sum(dist.log_prob(action[:, i]) for i, dist in enumerate(distributions))
    return dict(context=context.clone(), action=action, log_prob=log_prob, value=value,
                mask=mask, trainable=~clean)

  def observe(self, sample, score, valid=None, success=None):
    """Queue completed samples; no gradients/collectives. success is audit-only."""
    n = len(sample["context"])
    score = torch.as_tensor(score, device=self.device, dtype=torch.float32).reshape(-1)
    if score.shape != (n,):
      raise ValueError("One score per completed environment sample is required")
    keep = sample["trainable"].to(self.device) & torch.isfinite(score)
    if valid is not None:
      keep &= torch.as_tensor(valid, device=self.device, dtype=torch.bool)
    record = {key: sample[key].detach().to(self.device)[keep].clone()
              for key in ("context", "action", "log_prob", "value", "mask")}
    record["score"] = score[keep].detach().clone()
    record["success"] = (torch.full_like(score, float("nan")) if success is None else
                          torch.as_tensor(success, device=self.device, dtype=torch.float32))[keep].clone()
    self.pending.append(record)

  @staticmethod
  def _sum(value, reduce_sum):
    if reduce_sum is None:
      return value
    result = reduce_sum(value)
    return value if result is None else result

  def update(self, reduce_sum=None):
    """Full-buffer clipped contextual PPO, weighted by global sample count.

    A rank with zero completions still joins every collective. No rank-local
    early stops/minibatch counts; empty globally means no optimizer step.
    """
    keys = ("context", "action", "log_prob", "value", "mask", "score", "success")
    n = sum(len(batch["score"]) for batch in self.pending)
    count = self._sum(torch.tensor(float(n), device=self.device), reduce_sum)
    if count.item() == 0:
      self.pending.clear()
      return {"samples": 0, "updates": self.updates}
    data = {k: torch.cat([b[k] for b in self.pending], 0) for k in keys} if n else None
    stats = torch.zeros(5, device=self.device)
    if n:
      advantage = data["score"]-data["value"]
      success = data["success"]
      stats[:] = torch.stack([advantage.sum(), advantage.square().sum(), data["score"].sum(),
                              success.nan_to_num().sum(), torch.isfinite(success).float().sum()])
    stats = self._sum(stats, reduce_sum)
    mean = stats[0]/count
    std = (stats[1]/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    params = list(self.parameters())
    for _ in range(self.epochs):
      self.optimizer.zero_grad(set_to_none=True)
      if n:
        dists, value = self._distribution(data["context"], data["mask"])
        log_prob = sum(dist.log_prob(data["action"][:, i]) for i, dist in enumerate(dists))
        ratio = (log_prob-data["log_prob"]).clamp(-20, 20).exp()
        advantage = (data["score"]-data["value"]-mean)/std
        objective = torch.minimum(ratio*advantage, ratio.clamp(.8, 1.2)*advantage)
        entropy = sum(dist.entropy() for dist in dists)
        loss = (-objective + .5*(value-data["score"]).square() - .01*entropy).sum()
      else:
        loss = sum(parameter.sum()*0 for parameter in params)
      loss.backward()
      grads = torch.cat([p.grad.reshape(-1) for p in params])
      grads = self._sum(grads, reduce_sum)/count
      offset = 0
      for p in params:
        p.grad.copy_(grads[offset:offset+p.numel()].view_as(p))
        offset += p.numel()
      nn.utils.clip_grad_norm_(params, 1.)
      self.optimizer.step()
    self.pending.clear()
    self.updates += 1
    return {"samples": int(count.item()), "updates": self.updates,
            "score": float((stats[2]/count).item()),
            "actor_success": float((stats[3]/stats[4]).item()) if stats[4] > 0 else None}

  def checkpoint(self):
    return copy.deepcopy(dict(version=1, context_dim=self.context_dim,
        cardinalities=CARDINALITIES, exploration=self.exploration, epochs=self.epochs,
        model=self.state_dict(), optimizer=self.optimizer.state_dict(), updates=self.updates))

  def restore(self, state):
    if (state["version"] != 1 or state["context_dim"] != self.context_dim
        or tuple(state["cardinalities"]) != CARDINALITIES):
      raise ValueError("Incompatible adversary checkpoint")
    self.load_state_dict(state["model"])
    self.optimizer.load_state_dict(state["optimizer"])
    self.exploration, self.epochs = state["exploration"], state["epochs"]
    self.updates = state["updates"]
    self.pending.clear()
