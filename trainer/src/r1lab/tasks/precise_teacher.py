"""A separate teacher variant that redistributes body-position reward to feet."""
from __future__ import annotations

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.reward_manager import RewardTermCfg

from r1lab.mdp import precise_teacher as precise
from r1lab.tasks.motion_teacher import (
    ACTOR_DIM, CRITIC_DIM, OBSERVATION_CONTRACT, motion_teacher_env_cfg,
    motion_teacher_rl_cfg, reset_teacher_to_frame,
)

PRECISE_REWARD_CONTRACT = {
    "name": "precise-foot-world-v1",
    "native_body_position_weight": 0.5,
    "individual_foot_trajectory_weight": 0.5,
    "position_total_weight": 1.0,
    "xy_std_m": precise.XY_STD_M,
    "z_std_m": precise.Z_STD_M,
    "frame": "archived world FK plus fixed per-environment origins",
    "points": "left/right ankle-roll body origins, identical reference and actual points",
    "formula": "mean_LR exp(-((dx^2+dy^2)/0.05^2 + dz^2/0.02^2))",
    "contact_targets": "none; inferred human contact and GRF are not reward targets",
    "low_signal_diagnostic_threshold": precise.LOW_SIGNAL_SCORE,
    "progress": "signed net root XY along clip start-end direction; first policy sample origin",
    "progress_first_interval_excluded_s": 0.02,
}


def precise_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    cfg = motion_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                 reference_file=reference_file)
    cfg.rewards["motion_body_pos"].weight = 0.5
    cfg.rewards["precise_foot_trajectory"] = RewardTermCfg(
        func=precise.precise_foot_trajectory, weight=0.5,
        params={"command_name": "motion", "xy_std": precise.XY_STD_M, "z_std": precise.Z_STD_M})
    for foot, side in enumerate(("left", "right")):
        for quantity, suffix in (("xy", "xy_error_m"), ("z", "z_error_m"),
                                 ("distance", "world_error_m"), ("score", "score"),
                                 ("low_signal", "low_signal_fraction")):
            cfg.metrics[f"precise_{side}_foot_{suffix}"] = MetricsTermCfg(
                func=precise.foot_trajectory_metric,
                params={"foot": foot, "quantity": quantity})
    for source in ("actual", "reference", "deficit"):
        cfg.metrics[f"precise_{source}_net_progress_m"] = MetricsTermCfg(
            func=precise.DirectedRootProgress, params={"source": source}, reduce="last")
    cfg.metrics["precise_actual_directed_speed_mps"] = MetricsTermCfg(func=precise.root_directed_speed)
    return cfg


def precise_teacher_rl_cfg():
    return motion_teacher_rl_cfg()
