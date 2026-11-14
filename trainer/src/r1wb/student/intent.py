"""Arm intent: the arm joint targets System 1's arm correction is built around.

The same code runs in training and on the robot, and it reads only what System 1 has at run time (its own hip
angles and the commanded speed), never the reference motion: a reference-derived arm target would leak the clip's
phase into the student. Rest pose and swing gain come from the natural walking captures (see r1wb.library).
Hand goals (inverse kinematics) are added when the reach skill is built.
"""
from __future__ import annotations

import math

import torch

from r1wb import library, robot

REST = {"shoulder_pitch_joint": -0.006, "shoulder_roll_joint": 0.163, "shoulder_yaw_joint": 0.615,
        "elbow_joint": 1.361, "wrist_roll_joint": 0.033}


class ArmIntent:
  def __init__(self, num_envs: int, device: str, step_dt: float):
    joints = list(robot.JOINTS)
    self.hips = [joints.index("left_hip_pitch_joint"), joints.index("right_hip_pitch_joint")]
    rest = torch.zeros(len(robot.ARM_JOINTS), device=device)
    for index, name in enumerate(robot.ARM_JOINTS):
      side, joint = name.split("_", 1)
      sign = -1.0 if side == "right" and any(k in joint for k in ("roll", "yaw")) else 1.0
      rest[index] = sign * REST[joint]
    self.rest = rest
    self.pitch = [robot.ARM_JOINTS.index("left_shoulder_pitch_joint"), robot.ARM_JOINTS.index("right_shoulder_pitch_joint")]
    self.alpha = 1.0 - math.exp(-2.0 * math.pi * library.SWING_CUTOFF_HZ * step_dt)
    self.filtered = torch.zeros(num_envs, device=device)          # low-passed (left hip pitch - right hip pitch)

  def reset(self, env_ids, joint_pos: torch.Tensor) -> None:
    self.filtered[env_ids] = joint_pos[env_ids, self.hips[0]] - joint_pos[env_ids, self.hips[1]]

  def step(self, joint_pos: torch.Tensor, speed: torch.Tensor) -> torch.Tensor:
    """joint_pos: measured joint angles (N, 24, entity order); speed: commanded planar speed (N,). Returns (N, 10)."""
    difference = joint_pos[:, self.hips[0]] - joint_pos[:, self.hips[1]]
    self.filtered += self.alpha * (difference - self.filtered)
    low, high = library.SWING_SPEED
    gain = library.SWING_GAIN * ((speed - low) / (high - low)).clamp(0.0, 1.0)
    swing = (gain * self.filtered).clamp(-library.SWING_LIMIT, library.SWING_LIMIT)
    intent = self.rest.unsqueeze(0).repeat(joint_pos.shape[0], 1)
    intent[:, self.pitch[0]] += swing
    intent[:, self.pitch[1]] -= swing
    return intent
