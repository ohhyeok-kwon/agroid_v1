"""S+ hook (OFF): a human-motion prior on abstract, body-size-normalised outcome features phi, scored by a
Wasserstein discriminator with gradient penalty (WASABI form). Not connected to any registered task.

The coordinator's G0 check (2026-11-19 19:0x) found the discriminator saturated even without the body-signature
channels (feet contact, hand contact, XcoM separate human from the parent robot at 0.91; the 1.5x positive control
only 0.72), so S+ is not launched. What stays here is the connection point a later S+ needs:

  phi(env)            the robot side of the features, computed by the same definitions as the human bank
                      (server /rl-agent/.agent/data/r1_caption_pilot/g0/bank/): pelvis height / L, pelvis velocity /
                      sqrt(gL) in the heading frame, feet and fists relative to the pelvis / L, feet contact, XcoM margin
  PriorRewardCfg      a reward term spec with weight beta = 0.0 that refuses to build without a discriminator file

Enabling it requires a registered beta, a discriminator checkpoint and a passed G0 (none exist for S1 v1).
"""
from __future__ import annotations

import math

import torch

from r1s1 import numbers as N

L = N.STAND_H


def phi(env) -> torch.Tensor:
  robot = env.scene["robot"]
  data = robot.data
  names = robot.body_names
  pelvis = data.root_link_pos_w
  quat = data.root_link_quat_w
  w, x, y, z = quat.unbind(-1)
  yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  c, s = torch.cos(yaw), torch.sin(yaw)

  def heading(v):
    return torch.stack([c * v[:, 0] + s * v[:, 1], -s * v[:, 0] + c * v[:, 1], v[:, 2]], dim=1)

  ends = [names.index(n) for n in ("left_ankle_roll_link", "right_ankle_roll_link",
                                   "left_wrist_roll_link", "right_wrist_roll_link")]
  rel = torch.cat([heading(data.body_link_pos_w[:, i] - pelvis) / L for i in ends], dim=1)
  vel = heading(data.root_link_lin_vel_w) / math.sqrt(9.81 * L)
  contact = (env.scene["s1_feet"].data.found.reshape(env.num_envs, -1) > 0).float()
  return torch.cat([(pelvis[:, 2:3] - env.scene.env_origins[:, 2:3]) / L, vel, rel, contact], dim=1)


class PriorRewardCfg:
  """Placeholder spec: beta and the discriminator file must be registered before use."""

  def __init__(self, beta: float = 0.0, discriminator: str | None = None):
    if beta != 0.0 and not discriminator:
      raise ValueError("S+ prior needs a discriminator checkpoint and a passed G0 before beta != 0")
    self.beta, self.discriminator = beta, discriminator
