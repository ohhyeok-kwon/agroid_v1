"""Keep learned foot tracking and strengthen source-supported arm trajectories."""
from __future__ import annotations

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from r1lab.mdp import coordinated_teacher as arms
from r1lab.tasks.precise_teacher import precise_teacher_env_cfg, precise_teacher_rl_cfg

ARM_REWARD_CONTRACT = {
    "name": "pelvis-relative-arm-points-v1",
    "weights": {"motion_body_pos": 0.25, "precise_foot_trajectory": 0.5,
                "arm_position_tracking": 0.25},
    "points": list(arms.ARM_POINTS),
    "frame": "each actual/reference full pelvis frame; consistent robot body origins",
    "std_m": arms.POSITION_STD_M,
    "formula": "mean_points exp(-squared_3D_error/std_m^2)",
    "targets": "robot FK from retargeted elbow/hand motion; not measured human wrist orientation",
    "wrist_orientation_target": "none",
    "low_signal_threshold": arms.LOW_SIGNAL_SCORE,
    "unchanged": "foot reward, reference, observations, physics, PPO, resets and terminations",
}


def coordinated_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    cfg = precise_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                 reference_file=reference_file)
    cfg.rewards["motion_body_pos"].weight = 0.25
    cfg.rewards["arm_position_tracking"] = RewardTermCfg(
        func=arms.arm_position_tracking, weight=0.25,
        params={"command_name": "motion", "std": arms.POSITION_STD_M})
    for point, name in enumerate(arms.ARM_POINTS):
        label = name.removesuffix("_link")
        for quantity, suffix in (("distance", "error_m"), ("score", "score"),
                                 ("low_signal", "low_signal_fraction")):
            cfg.metrics[f"arm_{label}_{suffix}"] = MetricsTermCfg(
                func=arms.arm_point_metric, params={"point": point, "quantity": quantity})
    return cfg


def coordinated_teacher_rl_cfg():
    return precise_teacher_rl_cfg()
