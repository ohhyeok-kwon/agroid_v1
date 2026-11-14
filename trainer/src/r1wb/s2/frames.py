"""Whole-body reference frames: what System 2 produces and System 1 reads.

One frame (FRAME_DIM = 30) at 50 Hz, independent of where the robot stands and which way it faces:
  joint_pos (24)   joint angles (entity order)
  height (1)       height of the anchor body (the pelvis) above the ground (m)
  tilt (2)         its roll and pitch with the heading removed (rad)
  step (3)         where it goes until the next frame: forward, left (m) and turn (rad), in this frame's
                   heading frame (zero when the reference stays on a frame)

`DenseReference` holds a run of such frames per environment and offers the quantities the tracking policy's
observation reads from a reference (r1wb.tracking.env_cfg.teacher_observation), so that a reference that is not a
run of library frames (decoded tokens, a model's output) can drive the same policy. Velocities are finite
differences of the frames (a System 2 does not emit them).
"""
from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat

FRAME_DIM = 30
JOINTS = slice(0, 24)
HEIGHT, ROLL, PITCH = 24, 25, 26
STEP = slice(27, 30)


def _axis_quat(angle: torch.Tensor, axis: int) -> torch.Tensor:
  quat = torch.zeros(*angle.shape, 4, device=angle.device, dtype=angle.dtype)
  quat[..., 0] = torch.cos(0.5 * angle)
  quat[..., 1 + axis] = torch.sin(0.5 * angle)
  return quat


def tilt_quat(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
  return quat_mul(_axis_quat(pitch, 1), _axis_quat(roll, 0))


def turn_quat(angle: torch.Tensor) -> torch.Tensor:
  return _axis_quat(angle, 2)


def library_frames(motion, anchor: int, clip_end_of_frame: torch.Tensor) -> torch.Tensor:
  """(frames, FRAME_DIM) for every library frame; the step leads to the next frame of the same clip."""
  total = motion.joint_pos.shape[0]
  index = torch.arange(total, device=motion.joint_pos.device)
  ahead = torch.minimum(index + 1, clip_end_of_frame)
  pos, quat = motion.body_pos_w[:, anchor], motion.body_quat_w[:, anchor]
  heading = yaw_quat(quat)
  w, x, y, z = quat_mul(quat_inv(heading), quat).unbind(-1)
  roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
  move = quat_apply_inverse(heading, pos[ahead] - pos)[:, :2]
  forward = torch.zeros(total, 3, device=pos.device)
  forward[:, 0] = 1.0
  ahead_forward = quat_apply_inverse(heading, quat_apply(yaw_quat(quat[ahead]), forward))
  turn = torch.atan2(ahead_forward[:, 1], ahead_forward[:, 0])
  return torch.cat([motion.joint_pos, pos[:, 2:3], roll.unsqueeze(1), pitch.unsqueeze(1), move, turn.unsqueeze(1)], dim=1)


class DenseReference:
  """frames: (envs, lead + length, FRAME_DIM): `lead` frames before the run (the reference System 1 used last, at
  least one), then the run. Index i below means the run's frame i. Velocities are central differences, as in the
  motion library."""

  def __init__(self, frames: torch.Tensor, step_dt: float, velocity_point: torch.Tensor | None = None, lead: int = 1):
    """velocity_point: the point of the torso (in its own frame) whose velocity `anchor_velocity` reports. The motion
    library stores each body's centre-of-mass velocity next to its frame position (MuJoCo's mjOBJ_BODY velocity),
    and the tracking policies were trained on that: pass the torso's centre of mass to match it."""
    self.frames, self.dt, self.velocity_point, self.lead = frames, step_dt, velocity_point, lead
    count, length, _ = frames.shape
    turn = frames[:, :, STEP][:, :, 2]
    self.heading = torch.cat([torch.zeros(count, 1, device=frames.device), torch.cumsum(turn, dim=1)[:, :-1]], dim=1)
    cos, sin = torch.cos(self.heading), torch.sin(self.heading)
    move = frames[:, :, STEP]
    world = torch.stack([cos * move[..., 0] - sin * move[..., 1], sin * move[..., 0] + cos * move[..., 1]], dim=-1)
    self.planar = torch.cat([torch.zeros(count, 1, 2, device=frames.device), torch.cumsum(world, dim=1)[:, :-1]], dim=1)
    self.last = length - 1                                    # position of the run's last frame in the buffer
    self._rows = torch.arange(count, device=frames.device)

  def _spot(self, index: torch.Tensor) -> torch.Tensor:
    return (index + self.lead).clamp(min=0, max=self.last)

  def _quat(self, spot: torch.Tensor) -> torch.Tensor:
    frame = self.frames[self._rows, spot]
    return quat_mul(turn_quat(self.heading[self._rows, spot]), tilt_quat(frame[:, ROLL], frame[:, PITCH]))

  def _around(self, index: torch.Tensor):
    spot = self._spot(index)
    before, after = (spot - 1).clamp(min=0), (spot + 1).clamp(max=self.last)
    span = (after - before).float().clamp_min(1.0).unsqueeze(1) * self.dt
    return spot, before, after, span

  def joint_pos(self, index: torch.Tensor) -> torch.Tensor:
    return self.frames[self._rows, self._spot(index)][:, JOINTS]

  def joint_vel(self, index: torch.Tensor) -> torch.Tensor:
    _, before, after, span = self._around(index)
    return (self.frames[self._rows, after][:, JOINTS] - self.frames[self._rows, before][:, JOINTS]) / span

  def anchor_height(self, index: torch.Tensor) -> torch.Tensor:
    return self.frames[self._rows, self._spot(index)][:, HEIGHT]

  def anchor_tilt(self, index: torch.Tensor) -> torch.Tensor:
    """Torso rotation without its heading."""
    frame = self.frames[self._rows, self._spot(index)]
    return tilt_quat(frame[:, ROLL], frame[:, PITCH])

  def anchor_velocity(self, index: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear and angular velocity of the torso in the frame's own heading frame (z up)."""
    spot, before, after, span = self._around(index)
    heading = turn_quat(self.heading[self._rows, spot])
    planar = self.planar[self._rows, after] - self.planar[self._rows, before]
    climb = self.frames[self._rows, after][:, HEIGHT] - self.frames[self._rows, before][:, HEIGHT]
    linear = quat_apply_inverse(heading, torch.cat([planar, climb.unsqueeze(1)], dim=1)) / span
    first = self._quat(before)
    relative = quat_mul(quat_inv(first), self._quat(after))
    relative = relative * torch.where(relative[:, 0:1] < 0.0, -1.0, 1.0)
    half = torch.norm(relative[:, 1:], dim=1, keepdim=True)
    angle = 2.0 * torch.atan2(half, relative[:, 0:1])
    turning = quat_apply_inverse(heading, quat_apply(first, relative[:, 1:] / half.clamp_min(1.0e-9) * angle) / span)
    if self.velocity_point is not None:
      arm = quat_apply(self.anchor_tilt(index), self.velocity_point.unsqueeze(0).expand(len(index), -1))
      linear = linear + torch.linalg.cross(turning, arm)
    return linear, turning

  def command(self, index: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
    """The tracking command vector of r1wb.tracking.command (current joint state, then the preview)."""
    spot = self._spot(index)
    parts = [self.joint_pos(index), self.joint_vel(index)]
    now_quat, now_heading = self._quat(spot), turn_quat(self.heading[self._rows, spot])
    now_planar, now_height = self.planar[self._rows, spot], self.frames[self._rows, spot][:, HEIGHT]
    for step in future:
      ahead = (spot + step).clamp(max=self.last)
      delta = torch.cat([self.planar[self._rows, ahead] - now_planar,
                         (self.frames[self._rows, ahead][:, HEIGHT] - now_height).unsqueeze(1)], dim=1)
      parts.append(self.frames[self._rows, ahead][:, JOINTS])
      parts.append(quat_apply_inverse(now_heading, delta))
      relative = matrix_from_quat(quat_mul(quat_inv(now_quat), self._quat(ahead)))
      parts.append(relative[..., :2].reshape(len(index), 6))
    return torch.cat(parts, dim=1)
