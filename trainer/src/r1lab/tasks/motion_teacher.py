"""Flat-ground full-body reference teacher, trained from random network weights."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from mjlab.envs import mdp as env_mdp
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

from r1lab.mdp import motion_teacher as teacher
from r1lab.mdp.actions import PolicyJointPositionActionCfg
from r1lab.mdp.motion_teacher import reset_teacher_to_frame
from r1lab.robot import get_r1_cfg
from r1lab.run_record import work_root
from r1lab.tasks import common

ACTOR_DIM = 233
CRITIC_DIM = 458
OBSERVATION_CONTRACT = {
    "name": "reference-teacher-v1", "input_dimension": ACTOR_DIM,
    "critic_dimension": CRITIC_DIM, "deployment": "simulation-only",
    "policy_hz": 50, "history_frames": 1, "normalization": "disabled",
    "reference_joint_order": "ENTITY_JOINTS", "action_order": "POLICY_JOINTS",
    "future_frames": list(teacher.FUTURE_FRAMES), "phase": "sin_cos_0_to_2pi",
    "joint_velocity_scale": teacher.JOINT_VELOCITY_SCALE,
    "base_linear_velocity_scale": 0.5, "base_angular_velocity_scale": 0.25,
    "actor_terms": ["command:48", "motion_anchor_pos_b:3", "motion_anchor_ori_b:6",
                    "base_lin_vel:3", "base_ang_vel:3", "joint_pos:24", "joint_vel:24",
                    "actions:24", "phase:2", "future_reference:96"],
    "critic_terms": ["command:48", "motion_anchor_pos_b:3", "motion_anchor_ori_b:6",
                     "body_pos:75", "body_ori:150", "base_lin_vel:3", "base_ang_vel:3",
                     "joint_pos:24", "joint_vel:24", "actions:24", "phase:2", "future_reference:96"],
    "reference_file": "data/tracking/cautious_teacher_v1.npz",
    "reference_body_names": list(teacher.BODY_NAMES),
    "orientation_reward_bodies": list(teacher.ORIENTATION_BODIES),
    "force_targets": "none; contact force telemetry is simulated robot force",
}


def motion_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """One teacher policy; heldout uses disjoint data and fails if absent."""
    del t0
    if reference_split not in ("train", "heldout"):
        raise ValueError("reference_split must be train or heldout")
    name = "cautious_teacher_v1.npz" if reference_split == "train" else "cautious_teacher_heldout_v1.npz"
    path = Path(reference_file) if reference_file is not None else work_root() / "data/tracking" / name
    cfg = make_tracking_env_cfg()
    cfg.scene.entities = {"robot": replace(get_r1_cfg(command_delay_max=0, self_collision=True),
                                            collisions=())}
    cfg.scene.sensors = tuple(replace(sensor, history_length=20)
                              if hasattr(sensor, "history_length") else sensor
                              for sensor in common.sensors(self_collision=True) if sensor.name != "foot_height")
    cfg.scene.env_spacing = 4.0
    cfg.scene.num_envs = 1 if play else 4096
    cfg.sim = common.simulation(rough=False)
    cfg.sim.nconmax, cfg.sim.njmax, cfg.sim.contact_sensor_maxmatch = 192, 1536, 384
    cfg.sim.mujoco.timestep = 0.001
    cfg.sim.mujoco.integrator = "euler"
    cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations = 100, 50
    cfg.sim.mujoco.ccd_iterations = 35
    cfg.decimation = 20
    cfg.episode_length_s = 10.0
    cfg.is_finite_horizon = True
    cfg.events = {}
    cfg.actions = {"joint_pos": PolicyJointPositionActionCfg(entity_name="robot")}
    cfg.commands = {"motion": teacher.TeacherMotionCommandCfg(
        entity_name="robot", motion_file=str(path), anchor_body_name="pelvis",
        body_names=teacher.BODY_NAMES, resampling_time_range=(1e9, 1e9), debug_vis=False,
        sampling_mode="start" if play else "uniform", minimum_remaining_frames=25,
        pose_range={} if play else {"x": (-.02, .02), "y": (-.02, .02),
                                    "roll": (-.02, .02), "pitch": (-.02, .02), "yaw": (-.05, .05)},
        velocity_range={}, joint_position_range=(0., 0.) if play else (-.005, .005))}
    for group in cfg.observations.values():
        group.enable_corruption = False
        for term in group.terms.values():
            term.noise = None
        group.terms["command"].func = teacher.current_reference
        group.terms["base_lin_vel"].scale = 0.5
        group.terms["base_ang_vel"].scale = 0.25
        group.terms["joint_pos"].params = {"biased": False}
        group.terms["joint_vel"].scale = teacher.JOINT_VELOCITY_SCALE
        group.terms["phase"] = ObservationTermCfg(func=teacher.phase_observation)
        group.terms["future_reference"] = ObservationTermCfg(func=teacher.future_reference)
    cfg.rewards["motion_body_ori"].params["body_names"] = teacher.ORIENTATION_BODIES
    cfg.rewards["motion_body_ang_vel"].params["body_names"] = teacher.ORIENTATION_BODIES
    cfg.rewards["self_collisions"].weight = -2.0  # Preserve native 4-substep penalty at 20 substeps.
    cfg.terminations["ee_body_pos"].params["body_names"] = (
        "left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link")
    cfg.terminations["reference_end"] = TerminationTermCfg(func=teacher.reference_end, time_out=True)
    cfg.terminations["illegal_contact"] = TerminationTermCfg(func=teacher.illegal_contact)
    cfg.terminations["nonfinite"] = TerminationTermCfg(func=env_mdp.nan_detection)
    cfg.metrics = {
        "reference_end": MetricsTermCfg(func=teacher.reference_end, reduce="last"),
        "joint_rmse_rad": MetricsTermCfg(func=teacher.joint_position_rmse),
        "joint_speed_max_rad_s": MetricsTermCfg(func=teacher.joint_speed_max, reduce="max"),
        "foot_slip_mps": MetricsTermCfg(func=teacher.contact_slip_speed),
        "left_foot_force_n": MetricsTermCfg(func=teacher.foot_force, params={"foot": 0}, reduce="max"),
        "right_foot_force_n": MetricsTermCfg(func=teacher.foot_force, params={"foot": 1}, reduce="max"),
        "left_foot_force_bw": MetricsTermCfg(func=teacher.foot_force, params={"foot": 0, "normalized": True}),
        "right_foot_force_bw": MetricsTermCfg(func=teacher.foot_force, params={"foot": 1, "normalized": True}),
    }
    cfg.viewer.body_name = "pelvis"
    if play:
        cfg.auto_reset = False
    return cfg


def motion_teacher_rl_cfg():
    cfg = common.ppo_runner_cfg("r1_careful_walk", iters=500, symmetric=False)
    cfg.save_interval = 50
    cfg.actor.obs_normalization = cfg.critic.obs_normalization = False
    cfg.actor.distribution_cfg["init_std"] = 0.5
    cfg.algorithm.learning_rate = 3e-4
    return cfg
