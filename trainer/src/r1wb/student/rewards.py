"""Goal-following rewards of the System 1 student (the reference-tracking rewards come from the teacher task).

Goal vector layout (student.command.GOAL_DIM = 18):
  0:3 twist | 3 height | 4 pitch | 5 left active, 6:9 left hand, 9 time-to-go | 10 right active, 11:14, 14 |
  15:18 masks (twist, height, pitch)
A goal that is switched off returns the full reward, so it neither helps nor hurts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _goal(env, command_name):
  return env.command_manager.get_term(command_name).goal


def goal_twist(env: ManagerBasedRlEnv, command_name: str, std_linear: float = 0.15, std_yaw: float = 0.3):
  """Kernel on the 0.4 s mean of the own twist against the twist goal.

  The first version compared the instantaneous velocity with widths of 0.3 m/s and 0.6 rad/s. Standing still under
  a 0.2 m/s command then kept 64% of the term, and the student learned a dead zone: it walked backward at 0.35 m/s
  (55%) and turned at 1.2 rad/s (78%) but ignored 0.2 m/s and 0.6 rad/s (15%). The running mean removes the stride
  ripple, which is what allows the narrower widths."""
  term = env.command_manager.get_term(command_name)
  goal, mean = term.goal, term.twist_mean
  error = (torch.sum(torch.square(mean[:, :2] - goal[:, 0:2]), dim=1) / std_linear**2
           + torch.square(mean[:, 2] - goal[:, 2]) / std_yaw**2)
  return torch.exp(-error) * goal[:, 15] + (1.0 - goal[:, 15])


def goal_twist_l1(env: ManagerBasedRlEnv, command_name: str, yaw_weight: float = 0.3):
  """Linear twist error on the same running mean: it does not flatten out for small commands."""
  term = env.command_manager.get_term(command_name)
  goal, mean = term.goal, term.twist_mean
  return torch.norm(mean[:, :2] - goal[:, 0:2], dim=1) + yaw_weight * (mean[:, 2] - goal[:, 2]).abs()


def goal_height(env: ManagerBasedRlEnv, command_name: str, std: float = 0.05):
  from r1wb import world
  goal = _goal(env, command_name)
  height = world.standing_height(env)            # above the surface the robot stands on (flat floor: the start level)
  return torch.exp(-torch.square(height - goal[:, 3]) / std**2) * goal[:, 16] + (1.0 - goal[:, 16])


def goal_pitch(env: ManagerBasedRlEnv, command_name: str, std: float = 0.2):
  goal = _goal(env, command_name)
  w, x, y, z = env.scene["robot"].data.root_link_quat_w.unbind(-1)
  pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
  return torch.exp(-torch.square(pitch - goal[:, 4]) / std**2) * goal[:, 17] + (1.0 - goal[:, 17])


def goal_hands(env: ManagerBasedRlEnv, command_name: str, std: float = 0.1):
  term = env.command_manager.get_term(command_name)
  goal = term.goal
  data = env.scene["robot"].data
  heading = yaw_quat(data.root_link_quat_w)
  total = torch.zeros(env.num_envs, device=env.device)
  for wrist, flag, start in zip(term._wrists, (5, 10), (6, 11)):
    position = quat_apply_inverse(heading, term.robot_body_pos_w[:, wrist] - data.root_link_pos_w)
    error = torch.sum(torch.square(position - goal[:, start:start + 3]), dim=1)
    total += torch.exp(-error / std**2) * goal[:, flag] + (1.0 - goal[:, flag])
  return 0.5 * total


def arm_deviation_l1(env: ManagerBasedRlEnv, command_name: str):
  """Mean distance of the measured arm joints from the arm intent (rad). Priced on what the arms do, not on the
  correction command; it never switches off and does not saturate."""
  term = env.command_manager.get_term(command_name)
  arms = env.scene["robot"].data.joint_pos[:, term.arm_ids]
  return (arms - term.arm_intent_now).abs().mean(dim=1)


def standing_still(env: ManagerBasedRlEnv, command_name: str, linear: float = 0.03, yaw: float = 0.05):
  """Mean joint speed (rad/s) while the twist goal is zero. A robot that is asked to stand should not move, also
  when its sensors are noisy: S1-a at 2,500 iterations stood at 0.008 rad/s without noise and at 0.098 (arms) with
  the training noise, against 0.001 for the manufacturer's policy."""
  goal = env.command_manager.get_term(command_name).twist_goal()
  standing = ((torch.norm(goal[:, :2], dim=1) < linear) & (goal[:, 2].abs() < yaw)).float()
  return standing * env.scene["robot"].data.joint_vel.abs().mean(dim=1)


def goal_lag_l1(env: ManagerBasedRlEnv, command_name: str, yaw_weight: float = 0.3):
  """How far the robot has fallen behind (or run ahead of) its twist goal: |leaky integral of the planar velocity
  error| in metres plus `yaw_weight` times the same for the yaw rate in radians. It grows without saturating, so
  standing still while asked to walk is never a comfortable optimum."""
  lag = env.command_manager.get_term(command_name).goal_lag
  return torch.norm(lag[:, :2], dim=1) + yaw_weight * lag[:, 2].abs()


def goal_lag_exceeded(env: ManagerBasedRlEnv, command_name: str, limit_xy: float, limit_yaw: float, grace: float = 0.0):
  lag = env.command_manager.get_term(command_name).goal_lag
  exceeded = (torch.norm(lag[:, :2], dim=1) > limit_xy) | (lag[:, 2].abs() > limit_yaw)
  clock = getattr(env, "hit_clock", None)
  return exceeded if grace <= 0.0 or clock is None else exceeded & (clock > grace)
