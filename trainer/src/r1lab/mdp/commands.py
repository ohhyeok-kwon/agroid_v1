"""Crouch (protective squat) height command.

command = [h_cmd - H0, h_target - H0, rate]   (metres, metres, m/s)
h_cmd(t) = max(h_target, H0 - rate * max(0, t - delay)),  t = time since episode start.

The deploy runtime reproduces the same three numbers from its own clock, so the
policy can be driven slower/faster or stopped at any height.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg

from r1lab import contract as C

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class CrouchCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  resampling_time_range: tuple[float, float] = (1.0e9, 1.0e9)
  target_height_range: tuple[float, float] = (0.40, 0.52)
  rate_range: tuple[float, float] = (0.04, 0.12)
  delay_range: tuple[float, float] = (0.0, 1.0)
  stand_height: float = C.STAND_HEIGHT

  def build(self, env: ManagerBasedRlEnv) -> CrouchCommand:
    return CrouchCommand(self, env)


class CrouchCommand(CommandTerm):
  cfg: CrouchCommandCfg

  def __init__(self, cfg: CrouchCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]
    n = self.num_envs
    self.h_target = torch.full((n,), cfg.target_height_range[0], device=self.device)
    self.rate = torch.full((n,), cfg.rate_range[0], device=self.device)
    self.delay = torch.zeros(n, device=self.device)
    self.cmd = torch.zeros(n, 3, device=self.device)
    self.metrics["height_error"] = torch.zeros(n, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return self.cmd

  @property
  def h_cmd(self) -> torch.Tensor:
    return self.cmd[:, 0] + self.cfg.stand_height

  @property
  def progress(self) -> torch.Tensor:
    """0 at stand height, 1 when the commanded height reached the target."""
    span = (self.cfg.stand_height - self.h_target).clamp_min(1e-3)
    return ((self.cfg.stand_height - self.h_cmd) / span).clamp(0.0, 1.0)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    k = len(env_ids)
    u = lambda lo, hi: torch.rand(k, device=self.device) * (hi - lo) + lo  # noqa: E731
    self.h_target[env_ids] = u(*self.cfg.target_height_range)
    self.rate[env_ids] = u(*self.cfg.rate_range)
    self.delay[env_ids] = u(*self.cfg.delay_range)

  def _update_command(self, env_ids: torch.Tensor | None) -> None:
    del env_ids
    t = self._env.episode_length_buf.float() * self._env.step_dt
    h0 = self.cfg.stand_height
    h = h0 - self.rate * (t - self.delay).clamp_min(0.0)
    h = torch.maximum(h, self.h_target)
    self.cmd[:, 0] = h - h0
    self.cmd[:, 1] = self.h_target - h0
    self.cmd[:, 2] = self.rate

  def _update_metrics(self) -> None:
    site_ids, _ = self.robot.find_sites(("left_foot", "right_foot"), preserve_order=True)
    foot_z = self.robot.data.site_pos_w[:, site_ids, 2].min(dim=1).values
    h = self.robot.data.root_link_pos_w[:, 2] - foot_z
    self.metrics["height_error"] = (h - self.h_cmd).abs()
