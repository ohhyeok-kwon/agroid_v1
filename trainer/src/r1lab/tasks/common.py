"""Shared scene, sensor, randomisation and critic configuration for R1 tasks."""
from __future__ import annotations

import math

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import (ContactMatch, ContactSensorCfg, GridPatternCfg, ObjRef, RayCastSensorCfg,
                          RingPatternCfg, TerrainHeightSensorCfg)
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp

from r1lab.mdp import observations as obs
from r1lab.robot import FOOT_GEOMS, FOOT_SITES

DECIMATION = 4
PHYSICS_DT = 0.005


# Privileged terrain scan for the critic only (the deployed actor stays blind):
# yaw-aligned grid under the pelvis, x -0.6..0.6 m (13), y -0.4..0.4 m (9), row-major in y.
SCAN_SHAPE = (9, 13)


def terrain_scan() -> RayCastSensorCfg:
  return RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="pelvis", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.2, 0.8), resolution=0.1),
    max_distance=3.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )


def sensors(self_collision: bool = True, scan: bool = False) -> tuple:
  feet = ContactSensorCfg(
    name="feet_contact",
    primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                         entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
    history_length=DECIMATION,
  )
  foot_height = TerrainHeightSensorCfg(
    name="foot_height",
    frame=tuple(ObjRef(type="site", name=s, entity="robot") for s in FOOT_SITES),
    pattern=RingPatternCfg.single_ring(radius=0.03, num_samples=6),
    ray_alignment="yaw",
    max_distance=1.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),
  )
  self_coll = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=DECIMATION,
  )
  # Body parts that must never touch the ground (termination).
  illegal = ContactSensorCfg(
    name="illegal_contact",
    primary=ContactMatch(mode="body", pattern=r"^(pelvis|torso_link|waist_roll_link)$",
                         entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=DECIMATION,
  )
  # Limbs touching the ground (penalised): knees, elbows, hands.
  limbs = ContactSensorCfg(
    name="limb_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r"^(left|right)_(knee_link|elbow_link|wrist_roll_link|shoulder_yaw_link|hip_yaw_link)$",
      entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  base = (feet, foot_height, illegal, limbs) + ((terrain_scan(),) if scan else ())
  return base + ((self_coll,) if self_collision else ())


def simulation(rough: bool) -> SimulationCfg:
  return SimulationCfg(
    nconmax=128 if rough else 40,
    njmax=640 if rough else 250,
    contact_sensor_maxmatch=160 if rough else 64,
    mujoco=MujocoCfg(timestep=PHYSICS_DT, iterations=10, ls_iterations=20,
                     ccd_iterations=50),
  )


def actor_obs(command_name: str, noise: bool = True) -> ObservationGroupCfg:
  return ObservationGroupCfg(
    terms={"frames": ObservationTermCfg(func=obs.frame_stack,
                                        params={"command_name": command_name, "noise": noise})},
    concatenate_terms=True,
    enable_corruption=noise,
  )


CRITIC_LAYOUT = (("actor", 810), ("frame", 81), ("vec3", 3), ("scalar", 1), ("lr2", 2),
                 ("lr2", 2), ("lr_vec3", 6))
CRITIC_LAYOUT_SCAN = CRITIC_LAYOUT + (("ygrid", SCAN_SHAPE[0] * SCAN_SHAPE[1], SCAN_SHAPE),)


def critic_obs(command_name: str, scan: bool = False) -> ObservationGroupCfg:
  g = _critic_obs(command_name)
  if scan:
    g.terms["height_scan"] = ObservationTermCfg(func=envs_mdp.height_scan,
                                                params={"sensor_name": "terrain_scan"})
  return g


def _critic_obs(command_name: str) -> ObservationGroupCfg:
  return ObservationGroupCfg(
    terms={
      "frames": ObservationTermCfg(func=obs.frame_stack,
                                   params={"command_name": command_name, "noise": False}),
      "true_frame": ObservationTermCfg(func=obs.true_frame, params={"command_name": command_name}),
      "base_lin_vel": ObservationTermCfg(func=obs.base_lin_vel),
      "base_height": ObservationTermCfg(func=obs.base_height),
      "feet_contact": ObservationTermCfg(func=obs.feet_contact, params={"sensor_name": "feet_contact"}),
      "feet_force": ObservationTermCfg(func=_feet_force_norm, params={"sensor_name": "feet_contact"}),
      "feet_rel_pos": ObservationTermCfg(func=obs.feet_rel_pos),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )


def _feet_force_norm(env, sensor_name: str):
  import torch
  s = env.scene[sensor_name]
  return torch.norm(s.data.force, dim=-1) / (obs.ROBOT_MASS * obs.GRAVITY)


def randomization_events(level: str = "full") -> dict[str, EventTermCfg]:
  """Physical-parameter randomisation applied at every reset."""
  robot = SceneEntityCfg("robot")
  ev = {
    "foot_friction": EventTermCfg(
      mode="reset", func=dr.geom_friction,
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs",
              "ranges": (0.25, 1.4), "shared_random": True}),
    "pd_gains": EventTermCfg(
      mode="reset", func=dr.pd_gains,
      params={"asset_cfg": SceneEntityCfg("robot"),
              "kp_range": (0.85, 1.15), "kd_range": (0.8, 1.25), "operation": "scale"}),
    "effort": EventTermCfg(
      mode="reset", func=dr.effort_limits,
      params={"asset_cfg": SceneEntityCfg("robot"),
              "effort_limit_range": (0.8, 1.05), "operation": "scale"}),
    "armature": EventTermCfg(
      mode="reset", func=dr.joint_armature,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "operation": "scale",
              "ranges": (0.8, 1.25)}),
    "joint_friction": EventTermCfg(
      mode="reset", func=dr.joint_friction,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "operation": "abs",
              "ranges": (0.0, 0.35)}),
    "encoder_bias": EventTermCfg(
      mode="reset", func=dr.encoder_bias,
      params={"asset_cfg": robot, "bias_range": (-0.015, 0.015)}),
    "payload": EventTermCfg(
      mode="reset", func=dr.body_mass,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add",
              "ranges": (-0.5, 3.0)}),
    "hand_load": EventTermCfg(
      mode="reset", func=dr.body_mass,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*_wrist_roll_link",)),
              "operation": "add", "ranges": (0.0, 0.8)}),
    "torso_com": EventTermCfg(
      mode="reset", func=dr.body_com_offset,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add",
              "ranges": {0: (-0.04, 0.04), 1: (-0.03, 0.03), 2: (-0.04, 0.04)}}),
    "pelvis_com": EventTermCfg(
      mode="reset", func=dr.body_com_offset,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)), "operation": "add",
              "ranges": {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.02, 0.02)}}),
  }
  return ev


def disturbance_events(push_vel: float, hit_force: float,
                       hit_duration: tuple[float, float] = (0.05, 0.2)) -> dict[str, EventTermCfg]:
  """Pushes (velocity kicks) and hits (short force impulses on torso/pelvis/arms)."""
  return {
    "push": EventTermCfg(
      mode="interval", func=envs_mdp.push_by_setting_velocity, interval_range_s=(3.0, 7.0),
      params={"velocity_range": {"x": (-push_vel, push_vel), "y": (-push_vel, push_vel),
                                 "z": (-0.15, 0.15), "roll": (-0.25, 0.25), "pitch": (-0.25, 0.25),
                                 "yaw": (-0.4, 0.4)}}),
    "hit_torso": EventTermCfg(
      mode="step", func=envs_mdp.apply_body_impulse,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
              "force_range": (-hit_force, hit_force), "torque_range": (-hit_force / 12, hit_force / 12),
              "duration_s": hit_duration, "cooldown_s": (2.0, 6.0),
              "body_point_offset": (0.0, 0.0, 0.15)}),
    "hit_pelvis": EventTermCfg(
      mode="step", func=envs_mdp.apply_body_impulse,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),
              "force_range": (-0.6 * hit_force, 0.6 * hit_force),
              "torque_range": (-hit_force / 25, hit_force / 25),
              "duration_s": hit_duration, "cooldown_s": (3.0, 8.0)}),
    "hit_hands": EventTermCfg(
      mode="step", func=envs_mdp.apply_body_impulse,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=(".*_wrist_roll_link",)),
              "force_range": (-25.0, 25.0), "torque_range": (-1.0, 1.0),
              "duration_s": (0.1, 0.8), "cooldown_s": (2.0, 6.0)}),
  }


def ppo_runner_cfg(name: str, iters: int, symmetric: bool = True) -> RslRlOnPolicyRunnerCfg:
  del symmetric
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.8,
                                          "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=(512, 256, 128), activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name=name,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=iters,
    logger="tensorboard",
    upload_model=False,
  )


TERMINATION_ANGLE = math.radians(60.0)
_ = vel_mdp
