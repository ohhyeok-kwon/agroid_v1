"""R1 protective crouch: on a fault, descend very slowly into a balanced squat and hold it.

Starts from standing, disturbed and walking-like states; pushes/hits during descent;
weak motors (fault emulation), IMU/joint noise and latency.
Deploy contract: same 81x10 frame stack; command slot = [h_cmd-H0, h_target-H0, rate].
"""
from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains import config as tcfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.viewer import ViewerConfig

from r1lab.mdp import amp
from r1lab.mdp import curriculums as cur
from r1lab.mdp import rewards as rw
from r1lab.mdp.actions import PolicyJointPositionActionCfg
from r1lab.mdp.commands import CrouchCommandCfg
from r1lab.robot import get_r1_cfg
from r1lab.run_record import work_root
from r1lab.tasks import common

# Balanced squat from tools/crouch_pose.py (pelvis 0.462 m, CoM over feet, torso +0.35 rad).
CROUCH_POSE_HEIGHT = 0.462
CROUCH_POSE = {
  "left_hip_pitch_joint": -1.849, "left_hip_roll_joint": -0.559, "left_hip_yaw_joint": -0.422,
  "left_knee_joint": 2.047, "left_ankle_pitch_joint": -0.518, "left_ankle_roll_joint": -0.006,
  "right_hip_pitch_joint": -1.849, "right_hip_roll_joint": 0.56, "right_hip_yaw_joint": 0.423,
  "right_knee_joint": 2.047, "right_ankle_pitch_joint": -0.518, "right_ankle_roll_joint": 0.006,
  "waist_roll_joint": 0.0, "waist_yaw_joint": 0.0,
  "left_shoulder_pitch_joint": -0.683, "left_shoulder_roll_joint": 0.257,
  "left_shoulder_yaw_joint": 0.0, "left_elbow_joint": 1.285, "left_wrist_roll_joint": 0.0,
  "right_shoulder_pitch_joint": -0.683, "right_shoulder_roll_joint": -0.257,
  "right_shoulder_yaw_joint": 0.0, "right_elbow_joint": 1.285, "right_wrist_roll_joint": 0.0,
}
POSE_STD = {
  "left_hip_pitch_joint": 0.35, "right_hip_pitch_joint": 0.35,
  "left_hip_roll_joint": 0.35, "right_hip_roll_joint": 0.35,
  "left_hip_yaw_joint": 0.35, "right_hip_yaw_joint": 0.35,
  "left_knee_joint": 0.35, "right_knee_joint": 0.35,
  "left_ankle_pitch_joint": 0.25, "right_ankle_pitch_joint": 0.25,
  "left_ankle_roll_joint": 0.15, "right_ankle_roll_joint": 0.15,
  "waist_roll_joint": 0.1, "waist_yaw_joint": 0.15,
}


def terrain() -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(8.0, 8.0), border_width=10.0, num_rows=5, num_cols=6, curriculum=False,
    sub_terrains={
      "flat": tcfg.flat(proportion=0.4),
      "rough": tcfg.random_rough(proportion=0.3, noise_range=(0.0, 0.025), noise_step=0.005),
      "slope": tcfg.hf_pyramid_slope(proportion=0.15, slope_range=(0.0, 0.1)),
      "slope_inv": tcfg.hf_pyramid_slope_inv(proportion=0.15, slope_range=(0.0, 0.1)),
    },
    add_lights=True,
  )


AMP_DATA = work_root() / 'data/amp/crouch.npz'


def safe_crouch_env_cfg(play: bool = False, use_amp: bool = True) -> ManagerBasedRlEnvCfg:
  c = "crouch"
  cmd = CrouchCommandCfg(target_height_range=(0.42, 0.55), rate_range=(0.04, 0.12),
                         delay_range=(0.0, 1.0))
  events: dict[str, EventTermCfg] = {
    # Faults happen while standing, walking or being disturbed.
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform, mode="reset",
      params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.02),
                             "roll": (-0.08, 0.08), "pitch": (-0.08, 0.08),
                             "yaw": (-math.pi, math.pi)},
              "velocity_range": {"x": (-0.45, 0.45), "y": (-0.3, 0.3), "z": (-0.1, 0.1),
                                 "roll": (-0.4, 0.4), "pitch": (-0.4, 0.4), "yaw": (-0.6, 0.6)}}),
    "reset_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset, mode="reset",
      params={"position_range": (-0.15, 0.15), "velocity_range": (-0.6, 0.6),
              "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    **common.randomization_events(),
    **common.disturbance_events(push_vel=0.3, hit_force=45.0),
  }
  # Fault emulation: weaker motors than nominal.
  events["effort"] = EventTermCfg(
    mode="reset", func=dr.effort_limits,
    params={"asset_cfg": SceneEntityCfg("robot"), "effort_limit_range": (0.6, 1.05),
            "operation": "scale"})
  rewards = {
    "height": RewardTermCfg(func=rw.crouch_height_tracking, weight=3.0,
                            params={"command_name": c, "std": 0.08}),
    "height_err": RewardTermCfg(func=rw.crouch_height_error_l1, weight=-6.0,
                                params={"command_name": c}),
    "com": RewardTermCfg(func=rw.com_over_support, weight=2.0, params={"std": 0.05}),
    "upright": RewardTermCfg(func=vel_mdp.upright, weight=1.0,
                             params={"std": math.sqrt(0.25),
                                     "asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",))}),
    "posture": RewardTermCfg(func=rw.crouch_posture, weight=1.5,
                             params={"command_name": c, "crouch_pose": CROUCH_POSE, "std": POSE_STD,
                                     "pose_height": CROUCH_POSE_HEIGHT}),
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=1.0),
    "feet_planted": RewardTermCfg(func=rw.feet_planted, weight=-0.3, params={"sensor_name": "feet_contact"}),
    "descent_speed": RewardTermCfg(func=rw.descent_speed, weight=-50.0, params={"max_speed": 0.15}),
    "still": RewardTermCfg(func=rw.crouch_still, weight=-2.0, params={"command_name": c}),
    "ang_vel_xy": RewardTermCfg(func=rw.ang_vel_xy_l2, weight=-0.5),
    "torso_ang_vel": RewardTermCfg(func=vel_mdp.body_angular_velocity_penalty, weight=-0.2,
                                   params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))}),
    "soft_touchdown": RewardTermCfg(func=rw.soft_touchdown, weight=-3.0,
                                    params={"sensor_name": "feet_contact",
                                            "height_sensor_name": "foot_height", "near_height": 0.04}),
    "impact_peak": RewardTermCfg(func=rw.impact_peak, weight=-3.0,
                                 params={"sensor_name": "feet_contact", "threshold": 1.05}),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.1),
    "action_acc": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.05),
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-5e-7),
    "joint_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-3e-3),
    "power": RewardTermCfg(func=envs_mdp.electrical_power_cost, weight=-2e-3,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    "joint_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-5.0),
    "self_collision": RewardTermCfg(func=vel_mdp.self_collision_cost, weight=-1.0,
                                    params={"sensor_name": "self_collision", "force_threshold": 10.0}),
    "limb_contact": RewardTermCfg(func=_limb_contact, weight=-3.0),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-300.0),
  }
  if use_amp and not play:
    rewards["style"] = RewardTermCfg(func=amp.amp_style, weight=1.5,
                                     params={"data_file": str(AMP_DATA), "save_path": None})
  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=envs_mdp.bad_orientation,
                                    params={"limit_angle": common.TERMINATION_ANGLE}),
    "illegal_contact": TerminationTermCfg(func=vel_mdp.illegal_contact,
                                          params={"sensor_name": "illegal_contact",
                                                  "force_threshold": 5.0}),
    "out_of_bounds": TerminationTermCfg(func=vel_mdp.out_of_terrain_bounds, time_out=True),
  }
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="generator", terrain_generator=terrain()),
                   entities={"robot": get_r1_cfg(command_delay_max=4)},
                   sensors=common.sensors(), num_envs=1, extent=2.0),
    observations={"actor": common.actor_obs(c, noise=not play), "critic": common.critic_obs(c)},
    actions={"joint_pos": PolicyJointPositionActionCfg(entity_name="robot")},
    commands={c: cmd},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={} if play else {"robustness": CurriculumTermCfg(
      func=cur.robustness_ramp,
      params={"penalties": ["action_rate", "action_acc", "joint_acc", "joint_vel", "power",
                            "soft_touchdown", "impact_peak", "feet_planted", "ang_vel_xy",
                            "torso_ang_vel", "descent_speed", "still"],
              "events": ["push", "hit_torso", "hit_pelvis", "hit_hands", "reset_base", "reset_joints"],
              "start_frac": 0.25, "penalty_steps": 800 * 24, "disturb_steps": 700 * 24,
              "disturb_delay": 100 * 24})},
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
                        body_name="torso_link", distance=2.5, elevation=-10.0, azimuth=60.0),
    sim=common.simulation(rough=True),
    decimation=common.DECIMATION,
    episode_length_s=12.0,
  )
  if play:
    for k in ("push", "hit_torso", "hit_pelvis", "hit_hands"):
      cfg.events.pop(k, None)
  return cfg


def _limb_contact(env):
  import torch
  s = env.scene["limb_contact"]
  return torch.sum((s.data.found > 0).float(), dim=1)


def safe_crouch_rl_cfg():
  return common.ppo_runner_cfg("r1_safe_crouch", iters=3000)
