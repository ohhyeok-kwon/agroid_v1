"""Arm shape tracking in each pelvis frame, without invented wrist orientations."""
from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import quat_apply_inverse

ARM_POINTS = (
    "left_elbow_link", "left_wrist_roll_link",
    "right_elbow_link", "right_wrist_roll_link",
)
POSITION_STD_M = 0.05
LOW_SIGNAL_SCORE = 0.02


def pelvis_relative_points(points, pelvis_position, pelvis_quaternion):
    """Express identical robot body origins in their own full pelvis frame."""
    quaternion = pelvis_quaternion[:, None, :].expand(*points.shape[:-1], 4)
    return quat_apply_inverse(quaternion, points - pelvis_position[:, None, :])


def arm_position_error(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    indexes = [command.cfg.body_names.index(name) for name in ARM_POINTS]
    actual = pelvis_relative_points(command.robot_body_pos_w[:, indexes],
                                    command.robot_anchor_pos_w, command.robot_anchor_quat_w)
    target = pelvis_relative_points(command.body_pos_w[:, indexes],
                                    command.anchor_pos_w, command.anchor_quat_w)
    return actual - target


def arm_point_scores(error, std=POSITION_STD_M):
    if std <= 0:
        raise ValueError("Arm position tracking standard deviation must be positive metres")
    return torch.exp(-error.square().sum(-1) / std**2)


def arm_position_tracking(env, command_name="motion", std=POSITION_STD_M):
    # Mean four independent bounded scores; one bad hand cannot saturate all arms.
    return arm_point_scores(arm_position_error(env, command_name), std).mean(-1)


def arm_point_metric(env, point=0, quantity="distance", command_name="motion"):
    if point not in range(len(ARM_POINTS)):
        raise ValueError("Arm point must index left/right elbow/wrist")
    error = arm_position_error(env, command_name)[:, point]
    if quantity == "distance":
        return torch.linalg.vector_norm(error, dim=-1)
    score = arm_point_scores(error)
    if quantity == "score":
        return score
    if quantity == "low_signal":
        return (score < LOW_SIGNAL_SCORE).float()
    raise ValueError(f"Unknown arm tracking metric {quantity!r}")
