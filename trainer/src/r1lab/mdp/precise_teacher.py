"""World-frame foot tracking and diagnostic progress for the precise teacher."""
from __future__ import annotations

import torch

from mjlab.managers.manager_base import ManagerTermBase
from r1lab.robot import FOOT_BODIES

XY_STD_M = 0.05
Z_STD_M = 0.02
LOW_SIGNAL_SCORE = 0.02


def foot_world_error(env, command_name="motion"):
    """Actual minus archived target, [env,left/right,xyz] metres.

    Both endpoints are ankle-roll body origins. The command already adds each
    environment origin; no current-robot XY or yaw realignment is applied.
    """
    command = env.command_manager.get_term(command_name)
    indexes = [command.cfg.body_names.index(name) for name in FOOT_BODIES]
    return command.robot_body_pos_w[:, indexes] - command.body_pos_w[:, indexes]


def foot_scores(error, xy_std=XY_STD_M, z_std=Z_STD_M):
    """One bounded exponential per foot, before averaging the two feet."""
    if xy_std <= 0 or z_std <= 0:
        raise ValueError("Foot tracking standard deviations must be positive metres")
    distance = error[..., :2].square().sum(-1) / xy_std**2
    distance = distance + error[..., 2].square() / z_std**2
    return torch.exp(-distance)


def precise_foot_trajectory(env, command_name="motion", xy_std=XY_STD_M, z_std=Z_STD_M):
    return foot_scores(foot_world_error(env, command_name), xy_std, z_std).mean(-1)


def foot_trajectory_metric(env, foot=0, quantity="xy", command_name="motion"):
    if foot not in (0, 1):
        raise ValueError("foot must be 0 (left) or 1 (right)")
    error = foot_world_error(env, command_name)[:, foot]
    if quantity == "xy":
        return torch.linalg.vector_norm(error[:, :2], dim=-1)
    if quantity == "z":
        return error[:, 2].abs()
    if quantity == "distance":
        return torch.linalg.vector_norm(error, dim=-1)
    score = foot_scores(error)
    if quantity == "score":
        return score
    if quantity == "low_signal":
        return (score < LOW_SIGNAL_SCORE).float()
    raise ValueError(f"Unknown foot trajectory metric {quantity!r}")


def reference_direction(command):
    """Fixed planar start-to-end direction of the complete archived clip."""
    index = command.motion_anchor_body_index
    delta = command.motion.body_pos_w[-1, index, :2] - command.motion.body_pos_w[0, index, :2]
    return delta / torch.linalg.vector_norm(delta).clamp(min=1e-8)


def root_directed_speed(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return (command.robot_anchor_lin_vel_w[:, :2] * reference_direction(command)).sum(-1)


class DirectedRootProgress(ManagerTermBase):
    """Signed net translation since first post-reset policy sample, not path length.

    Metrics reset before command state initialization, so the first subsequent
    physical sample supplies both origins. The first 20 ms are excluded. This
    diagnostic has no simulator writes and never supplies a reward or failure.
    """
    def __init__(self, cfg, env):
        super().__init__(env)
        self._pending = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._actual_origin = torch.zeros(self.num_envs, 2, device=self.device)
        self._reference_origin = torch.zeros_like(self._actual_origin)

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._pending[env_ids] = True

    def __call__(self, env, source="actual", command_name="motion"):
        command = env.command_manager.get_term(command_name)
        actual = command.robot_anchor_pos_w[:, :2]
        target = command.anchor_pos_w[:, :2]
        self._actual_origin[self._pending] = actual[self._pending]
        self._reference_origin[self._pending] = target[self._pending]
        self._pending.zero_()
        direction = reference_direction(command)
        actual_progress = ((actual - self._actual_origin) * direction).sum(-1)
        reference_progress = ((target - self._reference_origin) * direction).sum(-1)
        if source == "actual":
            return actual_progress
        if source == "reference":
            return reference_progress
        if source == "deficit":
            return reference_progress - actual_progress
        raise ValueError(f"Unknown progress source {source!r}")
