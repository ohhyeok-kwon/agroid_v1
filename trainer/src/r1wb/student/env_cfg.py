"""P2 environment: the P1 tracking scene with a reference-free actor.

The actor sees 15 frames of proprioception plus sparse goals and the arm intent. The reference stays visible to
the critic, the rewards and the frozen teacher that labels actions. The scene is the "hardened" one (actuator delay,
PD-gain and trunk-mass variation, sensor noise): the deployed network trains under them from the start, and its
teacher has been fine-tuned in the same conditions.

Nothing here depends on where the reference is in the plane or which way it faces: the actor only receives a twist
in its own heading frame and cannot observe that drift. The reference is used re-anchored to the robot (height and
tilt of the anchor, body poses relative to it), and progress is scored on the twist goal itself (`goal_lag`), which
also ends episodes that stop following it. Without that, removing the planar termination would reopen the
"step in place while the reference walks away" optimum of the first teacher run.
"""
from __future__ import annotations

from dataclasses import dataclass

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.tasks.tracking import mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from r1wb import robot
from r1wb.student import observations as obs
from r1wb.student import rewards as rw
from r1wb.student.actions import IntentResidualActionCfg
from r1wb.student.command import GoalMotionCommandCfg
from r1wb.tracking import mdp as wb
from r1wb.tracking.env_cfg import EXPERIMENT, tracking_env_cfg, use_local_anchor, use_local_terminations

HISTORY = 15


def student_env_cfg(play: bool = False, teacher: str | None = None, first_clip: int = 0,
                    subset: str = "core", clean: bool | None = None, masks_off: bool = False,
                    disturbance: str = "kick", joint_course: float = 0.0, hit_grace: float = 0.0) -> ManagerBasedRlEnvCfg:
  """disturbance / joint_course / hit_grace: as in the tracking environment (the evaluation's impulses, the uniform
  price on the joint distance to the course, tracking terminations suspended after a hit). With hit_grace the twist
  lag does not accumulate during that time either, and its termination is suspended."""
  clean = play if clean is None else clean
  cfg = tracking_env_cfg(play=play, first_clip=first_clip, subset=subset, hardened=True, clean=clean, slip_excess=True,
                         contact_match=True, disturbance=disturbance, joint_course=joint_course, hit_grace=hit_grace)
  use_local_anchor(cfg, groups=("critic",))
  use_local_terminations(cfg)
  old = cfg.commands["motion"]
  cfg.commands = {"motion": GoalMotionCommandCfg(
    entity_name=old.entity_name, resampling_time_range=old.resampling_time_range, debug_vis=old.debug_vis,
    motion_file=old.motion_file, anchor_body_name=old.anchor_body_name, body_names=old.body_names,
    pose_range=old.pose_range, velocity_range=old.velocity_range, joint_position_range=old.joint_position_range,
    sampling_mode=old.sampling_mode, pin_clips=old.pin_clips, first_clip=old.first_clip, masks_off=masks_off,
    lag_grace=hit_grace)}
  cfg.actions = {"joint_pos": IntentResidualActionCfg(entity_name="robot", actuator_names=(".*",),
                                                      scale=dict(robot.ACTION_SCALE), use_default_offset=True)}
  name = {"command_name": "motion"}
  proprio = {
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"},
                                       noise=Unoise(n_min=-0.2, n_max=0.2)),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, params={"biased": True}, noise=Unoise(n_min=-0.01, n_max=0.01)),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }
  goals = {
    "goal": ObservationTermCfg(func=obs.goal, params=name),
    "arm_intent": ObservationTermCfg(func=obs.arm_intent, params=name),
  }
  critic = dict(cfg.observations["critic"].terms)
  critic.update({"goal": ObservationTermCfg(func=obs.goal, params=name),
                 "arm_intent": ObservationTermCfg(func=obs.arm_intent, params=name),
                 "goal_lag": ObservationTermCfg(func=obs.goal_lag, params=name)})
  cfg.observations = {
    "actor": ObservationGroupCfg(terms=proprio, concatenate_terms=True, enable_corruption=not clean,
                                 history_length=HISTORY, flatten_history_dim=True),
    "goal": ObservationGroupCfg(terms=goals, concatenate_terms=True, enable_corruption=False),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
  }
  if teacher is not None:
    from r1wb import senses
    senses.add_sensor(cfg, senses.kind_of_checkpoint(str(teacher)))      # a System 1 that reads the ground gets its rays
    cfg.observations["bc"] = ObservationGroupCfg(
      terms={"label": ObservationTermCfg(func=obs.teacher_label, params={**name, "checkpoint": str(teacher)})},
      concatenate_terms=True, enable_corruption=False)
  # Reference terms, re-anchored: height and tilt of the anchor, body velocities turned to the robot's heading.
  cfg.rewards.pop("anchor_xy_l1")
  cfg.rewards["motion_global_root_pos"] = RewardTermCfg(func=wb.anchor_height_error_exp, weight=0.5,
                                                        params={**name, "std": 0.1})
  cfg.rewards["motion_global_root_ori"] = RewardTermCfg(func=wb.anchor_tilt_error_exp, weight=0.5,
                                                        params={**name, "std": 0.4})
  cfg.rewards["motion_body_lin_vel"].func = wb.local_body_linear_velocity_error_exp
  cfg.rewards["motion_body_ang_vel"].func = wb.local_body_angular_velocity_error_exp
  cfg.rewards.update({
    "goal_twist": RewardTermCfg(func=rw.goal_twist, weight=1.5, params=name),
    "goal_twist_l1": RewardTermCfg(func=rw.goal_twist_l1, weight=-2.0, params=name),
    "goal_lag": RewardTermCfg(func=rw.goal_lag_l1, weight=-2.0, params=name),
    "goal_height": RewardTermCfg(func=rw.goal_height, weight=0.5, params=name),
    "goal_pitch": RewardTermCfg(func=rw.goal_pitch, weight=0.25, params=name),
    "goal_hands": RewardTermCfg(func=rw.goal_hands, weight=0.5, params=name),
    "arm_deviation": RewardTermCfg(func=rw.arm_deviation_l1, weight=-1.0, params=name),
  })
  # Tried and dropped (E15, 250 iterations from E9 at 2,500): arm_deviation -2.0 plus rw.standing_still -1.0. Under
  # sensor noise the standing joint speed stayed where it was (arms 0.098 -> 0.089 rad/s) and following 0.2 m/s
  # forward fell from 0.92 to 0.58.
  # Limits at about five times what the references themselves reach (0.07 m, 0.21 rad). With the 3 s leak a
  # sustained error of 0.13 m/s or 0.33 rad/s is the largest that is never terminated; ignoring 0.2 m/s ends the
  # episode after 3.3 s. Limits of 0.3 m / 0.8 rad terminated the label policy itself once per 17 s on command
  # streams (smoke run, 11-11 09:22), which is too tight for what the labels can show.
  cfg.terminations["goal_lag"] = TerminationTermCfg(func=rw.goal_lag_exceeded,
                                                    params={**name, "limit_xy": 0.4, "limit_yaw": 1.0, "grace": hit_grace})
  return cfg


@dataclass
class ImitationPpoCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "r1wb.learning:ImitationPPO"
  bc_group: str = "bc"
  bc_coef: float = 1.0
  bc_floor: float = 0.3
  bc_decay_updates: int = 1500
  critic_warmup_updates: int = 50
  surrogate_ramp_updates: int = 100
  ppo_learning_rate: float | None = 2.0e-4      # after the warm-up, decaying linearly to final_learning_rate
  final_learning_rate: float | None = 1.0e-4
  lr_decay_updates: int = 4000
  ramp_learning_rate: bool = False              # multiply the rate by the surrogate ramp (needed when bc_coef is 0)


def student_rl_cfg() -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.3,
                                          "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(1024, 512, 256), activation="elu", obs_normalization=True),
    algorithm=ImitationPpoCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.003,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=5.0e-4, schedule="fixed",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    obs_groups={"actor": ("actor", "goal"), "critic": ("critic",)},
    experiment_name=EXPERIMENT, save_interval=250, num_steps_per_env=24, max_iterations=4000,
    logger="tensorboard", upload_model=False)
