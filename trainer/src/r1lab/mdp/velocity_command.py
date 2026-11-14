"""Velocity command with explicit 8-direction, pure-turn and omni modes.

Independent uniform sampling of (vx, vy, wz) almost never produces clean in-place turns
or clean single-axis moves, which are exactly what a joystick sends.  Modes per resample:
  omni      : uniform box (like UniformVelocityCommand)
  dir8      : one of 8 directions, speed uniform up to the per-axis limits, wz = 0
  turn      : vx = vy = 0, |wz| uniform in [0.15, wz_max]
  dir8_turn : dir8 plus a small yaw rate
Standing envs (rel_standing_envs) still get zero.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

DIRS = torch.tensor([[1, 0], [-1, 0], [0, 1], [0, -1],
                     [math.sqrt(0.5), math.sqrt(0.5)], [math.sqrt(0.5), -math.sqrt(0.5)],
                     [-math.sqrt(0.5), math.sqrt(0.5)], [-math.sqrt(0.5), -math.sqrt(0.5)]])


@dataclass(kw_only=True)
class MixedVelocityCommandCfg(UniformVelocityCommandCfg):
  mode_probs: tuple[float, float, float, float] = (0.25, 0.3, 0.3, 0.15)  # omni, dir8, turn, dir8_turn
  turn_slow_prob: float = 0.0  # fraction of pure turns drawn from [0.15, 0.55] rad/s

  def build(self, env) -> MixedVelocityCommand:
    return MixedVelocityCommand(self, env)


class MixedVelocityCommand(UniformVelocityCommand):
  cfg: MixedVelocityCommandCfg

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)  # omni sample + standing/forward flags
    n = len(env_ids)
    if n == 0:
      return
    dev = self.device
    r = self.cfg.ranges
    probs = torch.tensor(self.cfg.mode_probs, device=dev)
    mode = torch.multinomial(probs, n, replacement=True)
    d = DIRS.to(dev)[torch.randint(0, 8, (n,), device=dev)]
    # per-direction max speed from the (asymmetric) ranges
    vx_max = torch.where(d[:, 0] >= 0, torch.tensor(r.lin_vel_x[1], device=dev),
                         torch.tensor(-r.lin_vel_x[0], device=dev))
    vy_max = torch.full((n,), max(abs(r.lin_vel_y[0]), abs(r.lin_vel_y[1])), device=dev)
    s = torch.rand(n, device=dev) * 0.85 + 0.15
    vx = d[:, 0] * s * vx_max
    vy = d[:, 1] * s * vy_max
    wz_max = max(abs(r.ang_vel_z[0]), abs(r.ang_vel_z[1]))
    sign = torch.where(torch.rand(n, device=dev) < 0.5, -1.0, 1.0)
    wz_turn = sign * (torch.rand(n, device=dev) * (wz_max - 0.15) + 0.15)
    slow = torch.rand(n, device=dev) < self.cfg.turn_slow_prob
    wz_turn = torch.where(slow, sign * (torch.rand(n, device=dev) * 0.4 + 0.15), wz_turn)
    wz_small = (torch.rand(n, device=dev) * 2 - 1) * 0.4 * wz_max
    cmd = self.vel_command_b[env_ids]
    m_dir = mode == 1
    m_turn = mode == 2
    m_dt = mode == 3
    cmd[m_dir | m_dt, 0] = vx[m_dir | m_dt]
    cmd[m_dir | m_dt, 1] = vy[m_dir | m_dt]
    cmd[m_dir, 2] = 0.0
    cmd[m_dt, 2] = wz_small[m_dt]
    cmd[m_turn, 0] = 0.0
    cmd[m_turn, 1] = 0.0
    cmd[m_turn, 2] = wz_turn[m_turn]
    self.vel_command_b[env_ids] = cmd
    self.vel_command_w[env_ids] = cmd
