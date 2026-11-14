"""Evaluation-only interventions on what a policy is given: questions about causes, never part of a deployed path."""
from __future__ import annotations

import torch


def lean_standing_plan(term, entity, lean: float, whole: bool = True) -> float:
  """The standing frames of the plan lean forward over their feet by `lean` rad (backward if negative).

  Why: the standing pose retargeted from people has the centre of mass about 2.6 cm behind the middle of the soles
  (heel side 6.5 cm of room, toe side 11.8 cm), and both the late steps after stopping from backward walking and the
  larger reaction to a push onto the heels show up in every checkpoint (gates 11-13 15:0x). The question is whether
  moving the plan's standing pose forward moves those numbers.

  How: the slower a library frame travels, the more it leans (a frame at rest fully, a frame travelling at 0.1 m/s
  or faster not at all). The two ankle pitch angles change, and with `whole` every body above the ankles is turned
  about the line through the ankles, so that the joint angles, the tilt and the body positions the plan states
  agree: the body then stands about 0.65 m x lean further forward. Without `whole` only the ankle angles change
  (measured 11-13 15:05: the robot then moves its centre of mass by a sixth of that). A probe, not a new library.
  Returns the share of library frames that were changed.
  """
  motion = term.motion
  names = list(entity.joint_names)
  ankles = [names.index("left_ankle_pitch_joint"), names.index("right_ankle_pitch_joint")]
  speed = torch.norm(motion.body_lin_vel_w[:, term._root, :2], dim=1)
  weight = (1.0 - speed / 0.1).clamp(0.0, 1.0)
  angle = lean * weight
  # Leaning the body forward over a foot that stays flat turns the foot backward against the shank.
  motion.joint_pos[:, ankles] -= angle.unsqueeze(1)
  if whole:
    bodies = list(term.cfg.body_names)                         # the bodies the plan states (a subset of the robot's)
    feet = [i for i, name in enumerate(bodies) if "ankle" in name]
    above = [i for i in range(len(bodies)) if i not in feet]
    assert len(feet) >= 2 and motion.body_pos_w.shape[1] == len(bodies), f"feet among the plan's bodies: {bodies}"
    pivot = motion.body_pos_w[:, feet].mean(dim=1, keepdim=True)                     # between the ankles
    w, x, y, z = motion.body_quat_w[:, term._root].unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    # A turn by `angle` about the sideways axis of the heading frame (forward pitch), as a quaternion.
    half = 0.5 * angle
    turn = torch.stack([torch.cos(half), -torch.sin(yaw) * torch.sin(half), torch.cos(yaw) * torch.sin(half),
                        torch.zeros_like(half)], dim=1)
    from mjlab.utils.lab_api.math import quat_apply, quat_mul
    count = len(above)
    spread = turn.unsqueeze(1).expand(-1, count, -1).reshape(-1, 4)
    arm = (motion.body_pos_w[:, above] - pivot).reshape(-1, 3)
    motion.body_pos_w[:, above] = pivot + quat_apply(spread, arm).reshape(-1, count, 3)
    motion.body_quat_w[:, above] = quat_mul(spread, motion.body_quat_w[:, above].reshape(-1, 4)).reshape(-1, count, 4)
  return float((weight > 0.0).float().mean())
