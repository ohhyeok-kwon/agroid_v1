"""The S1 goal: one masked outcome on the pelvis with a deadline (objective.md sections 1, 2, 4).

Per environment the term holds the goal kind, the mask (planar position, heading, height), the world target, the
time left (deadline) and the hold that follows it. A goal lasts tau0 + hold; the base class counts that down in
`time_left` (dt = 0 on the reset step), so tau = time_left - hold is positive before the deadline and negative
after it. The goal frame is the pelvis pose at the first update after the draw (the state is fresh there: the
command manager runs after the simulator's forward()).

What the actor sees (goal_observation, 9 values): planar error in the heading frame / lambda_p (2), sin and cos of
the heading error (2), height error / lambda_h (1), mask (3), max(tau, 0) / TAU_SCALE_S (1); masked components read
as zero error (cos 1).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from r1s1 import numbers as N

KINDS = ("hold", "locomotion", "crouch")


def _yaw(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap(angle: torch.Tensor) -> torch.Tensor:
  return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def _uniform(count: int, low: float, high: float, device) -> torch.Tensor:
  return low + (high - low) * torch.rand(count, device=device)


class GoalCommand(CommandTerm):
  cfg: "GoalCommandCfg"

  def __init__(self, cfg: "GoalCommandCfg", env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    n, device = self.num_envs, self.device
    self.kind = torch.zeros(n, dtype=torch.long, device=device)
    self.mask = torch.zeros(n, 3, device=device)
    self.delta = torch.zeros(n, 3, device=device)        # dx, dy (goal frame), dpsi
    self.h_target = torch.full((n,), N.STAND_H, device=device)
    self.p_target = torch.zeros(n, 2, device=device)
    self.psi_target = torch.zeros(n, device=device)
    self.hold = torch.zeros(n, device=device)
    self.tau0 = torch.zeros(n, device=device)
    self.pending = torch.ones(n, dtype=torch.bool, device=device)
    self.fresh_episode = torch.ones(n, dtype=torch.bool, device=device)
    self.phi_prev = torch.zeros(n, device=device)
    self._goal_obs = torch.zeros(n, 9, device=device)
    self._crouch_rate = torch.full((n,), N.CROUCH_RATE_MPS[0], device=device)
    share = torch.tensor([cfg.share[k] for k in KINDS], device=device)
    self._share_cdf = torch.cumsum(share / share.sum(), dim=0)
    for name in ("success", "in_goal", "at_rest", "phi", "tau"):
      self.metrics[name] = torch.zeros(n, device=device)

  # -- the term's outputs ---------------------------------------------------------------------------------------
  @property
  def command(self) -> torch.Tensor:
    return self._goal_obs

  @property
  def tau(self) -> torch.Tensor:
    """Seconds to the deadline (negative after it)."""
    return self.time_left - self.hold

  def state(self):
    data = self.robot.data
    origin = self._env.scene.env_origins
    pos = data.root_link_pos_w - origin
    yaw = _yaw(data.root_link_quat_w)
    return pos, yaw, data.root_link_lin_vel_w, data.root_link_ang_vel_w

  def errors(self):
    """Signed errors in the current heading frame: planar (2), heading (target - current), height."""
    pos, yaw, _, _ = self.state()
    d = self.p_target - pos[:, :2]
    c, s = torch.cos(yaw), torch.sin(yaw)
    ex, ey = c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]
    epsi = _wrap(self.psi_target - yaw)
    eh = self.h_target - pos[:, 2]
    return torch.stack([ex, ey], dim=1), epsi, eh

  def potential(self) -> torch.Tensor:
    ep, epsi, eh = self.errors()
    lp, lpsi, lh = N.LAMBDA
    d2 = (self.mask[:, 0] * (ep.norm(dim=1) / lp) ** 2 + self.mask[:, 1] * (epsi / lpsi) ** 2
          + self.mask[:, 2] * (eh / lh) ** 2)
    return torch.exp(-torch.sqrt(d2 + 1e-12))

  def judge(self):
    """(success, in_goal, at_rest) at the current state (objective.md 2)."""
    ep, epsi, eh = self.errors()
    _, _, lin, ang = self.state()
    ok_p = (self.mask[:, 0] < 0.5) | (ep.norm(dim=1) <= N.TOL[0])
    ok_psi = (self.mask[:, 1] < 0.5) | (epsi.abs() <= N.TOL[1])
    ok_h = (self.mask[:, 2] < 0.5) | (eh.abs() <= N.TOL[2])
    in_goal = ok_p & ok_psi & ok_h
    at_rest = (lin.norm(dim=1) <= N.REST[0]) & (ang.norm(dim=1) <= N.REST[1])
    success = torch.where(self.tau > 0.0, torch.ones_like(in_goal), in_goal & at_rest)
    return success, in_goal, at_rest

  # -- CommandTerm interface --------------------------------------------------------------------------------------
  def _update_metrics(self) -> None:
    success, in_goal, at_rest = self.judge()
    dt = self._env.step_dt
    self.metrics["success"] += success.float() * dt
    self.metrics["in_goal"] += in_goal.float() * dt
    self.metrics["at_rest"] += at_rest.float() * dt
    self.metrics["phi"] += self.potential() * dt
    self.metrics["tau"] += self.tau.clamp_min(0.0) * dt

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count, device = len(env_ids), self.device
    draw = torch.rand(count, device=device)
    kind = torch.searchsorted(self._share_cdf, draw.contiguous()).clamp_max(len(KINDS) - 1)
    if self.cfg.fixed_kind is not None:
      kind = torch.full_like(kind, KINDS.index(self.cfg.fixed_kind))
    mask = torch.zeros(count, 3, device=device)
    delta = torch.zeros(count, 3, device=device)
    h = torch.full((count,), N.STAND_H, device=device)
    tau0 = torch.zeros(count, device=device)
    hold = _uniform(count, *N.HOLD_AFTER_S, device)
    # hold (balance): height only, deadline now
    is_hold = kind == 0
    mask[is_hold] = torch.tensor([0.0, 0.0, 1.0], device=device)
    hold = torch.where(is_hold, _uniform(count, *N.HOLD_ONLY_S, device), hold)
    # locomotion: planar position and heading
    loco = kind == 1
    dist = _uniform(count, *N.LOCO_DIST_M, device)
    direction = _uniform(count, -math.pi, math.pi, device)
    turn = _uniform(count, *N.LOCO_TURN_RAD, device)
    pace = _uniform(count, *N.PACE_MPS, device)
    rate = _uniform(count, *N.TURN_RATE_RPS, device)
    loco_delta = torch.stack([dist * torch.cos(direction), dist * torch.sin(direction), turn], dim=1)
    delta = torch.where(loco.unsqueeze(1), loco_delta, delta)
    mask[loco] = torch.tensor([1.0, 1.0, 0.0], device=device)
    tau0 = torch.where(loco, N.SETTLE_S + dist / pace + turn.abs() / rate, tau0)
    # crouch: height, in place, heading kept
    crouch = kind == 2
    h_new = _uniform(count, *N.CROUCH_H_M, device)
    mask[crouch] = torch.tensor([1.0, 1.0, 1.0], device=device)
    h = torch.where(crouch, h_new, h)
    self.kind[env_ids], self.mask[env_ids], self.delta[env_ids] = kind, mask, delta
    self.tau0[env_ids], self.hold[env_ids] = tau0, hold
    self.h_target[env_ids] = h
    self._crouch_rate[env_ids] = _uniform(count, *N.CROUCH_RATE_MPS, device)
    self.time_left[env_ids] = tau0 + hold
    self.pending[env_ids] = True

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    if self.pending.any():
      ids = self.pending.nonzero().squeeze(-1)
      pos, yaw, _, _ = self.state()
      c, s = torch.cos(yaw[ids]), torch.sin(yaw[ids])
      d = self.delta[ids]
      self.p_target[ids] = pos[ids, :2] + torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], dim=1)
      self.psi_target[ids] = _wrap(yaw[ids] + d[:, 2])
      kind = self.kind[ids]
      # hold keeps the height it is issued at; a crouch budget follows the height change
      self.h_target[ids] = torch.where(kind == 0, pos[ids, 2], self.h_target[ids])
      crouch = kind == 2
      if crouch.any():
        cid = ids[crouch]
        budget = N.SETTLE_S + (self.h_target[cid] - pos[cid, 2]).abs() / self._crouch_rate[cid]
        self.tau0[cid] = budget
        self.time_left[cid] = budget + self.hold[cid]
      self.pending[ids] = False
      fresh = ids[self.fresh_episode[ids]]
      if len(fresh) > 0:
        self.phi_prev[fresh] = self.potential()[fresh]
        self.fresh_episode[fresh] = False
    self._goal_obs = self._observe()

  def reset(self, env_ids):
    if isinstance(env_ids, torch.Tensor):
      self.fresh_episode[env_ids] = True
    return super().reset(env_ids)

  def _observe(self) -> torch.Tensor:
    ep, epsi, eh = self.errors()
    m = self.mask
    lp, _, lh = N.LAMBDA
    return torch.cat([
      ep / lp * m[:, :1],
      torch.stack([torch.sin(epsi) * m[:, 1], torch.cos(epsi) * m[:, 1] + (1.0 - m[:, 1])], dim=1),
      (eh / lh * m[:, 2]).unsqueeze(1),
      m,
      (self.tau.clamp_min(0.0) / N.TAU_SCALE_S).unsqueeze(1),
    ], dim=1)


@dataclass(kw_only=True)
class GoalCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)   # every draw sets its own duration
  share: dict = field(default_factory=lambda: dict(N.GOAL_SHARE))
  fixed_kind: str | None = None                                   # evaluation: one goal kind only

  def build(self, env) -> GoalCommand:
    return GoalCommand(self, env)
