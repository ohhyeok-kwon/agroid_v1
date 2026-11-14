"""R1 careful walking: slow omnidirectional velocity tracking with quiet, soft footfalls,
heavy robustness randomisation (IMU/joint noise, latency, pushes, hits, debris, payload).

Deploy contract: identical to agroid walking_action_manager (drop-in for walk.onnx).
"""
from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from r1lab.mdp.velocity_command import MixedVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains import config as tcfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.viewer import ViewerConfig

from r1lab import contract as C
from r1lab.mdp import amp
from r1lab.mdp import curriculums as cur
from r1lab.mdp import rewards as rw
from r1lab.mdp.actions import PolicyJointPositionActionCfg
from r1lab.robot import FOOT_SITES, get_r1_cfg
from r1lab.run_record import work_root
from r1lab.tasks import common

# Careful speed envelope (m/s, m/s, rad/s) = final curriculum stage / deploy limits.
VX = (-0.40, 0.50)
VY = (-0.35, 0.35)
WZ = (-0.70, 0.70)
# Wider commands first: walking must be clearly better than standing still.
VX_WIDE, VY_WIDE, WZ_WIDE = (-0.6, 0.8), (-0.5, 0.5), (-1.0, 1.0)
WALK_HEIGHT = C.STAND_HEIGHT - 0.015
AMP_DATA = work_root() / 'data/amp/careful_walk.npz'


def terrain(play: bool = False, hard: bool = False) -> TerrainGeneratorCfg:
  """hard: more robots on debris/steps and 1-1.5 cm higher obstacles at the top level
  (margin over the evaluation terrain)."""
  w = (dict(flat=0.05, rough=0.15, debris=0.35, slope=0.075, steps=0.15) if hard else
       dict(flat=0.15, rough=0.2, debris=0.25, slope=0.1, steps=0.1))
  box_h = (0.015, 0.085) if hard else (0.015, 0.07)
  step_h = (0.01, 0.09) if hard else (0.01, 0.08)
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=12,
    curriculum=not play,
    sub_terrains={
      "flat": tcfg.flat(proportion=w["flat"]),
      "rough": tcfg.random_rough(proportion=w["rough"], noise_range=(0.005, 0.04), noise_step=0.005),
      "debris": tcfg.random_spread_boxes(proportion=w["debris"], num_boxes=70,
                                         box_width_range=(0.06, 0.35),
                                         box_length_range=(0.06, 0.5),
                                         box_height_range=box_h, platform_width=1.0),
      "slope": tcfg.hf_pyramid_slope(proportion=w["slope"], slope_range=(0.0, 0.25)),
      "slope_inv": tcfg.hf_pyramid_slope_inv(proportion=w["slope"], slope_range=(0.0, 0.25)),
      "steps": tcfg.pyramid_stairs(proportion=w["steps"], step_height_range=step_h, step_width=0.4,
                                   platform_width=2.5),
      "steps_inv": tcfg.pyramid_stairs_inv(proportion=w["steps"], step_height_range=step_h,
                                           step_width=0.4, platform_width=2.5),
    },
    add_lights=True,
  )


def careful_walk_env_cfg(play: bool = False, flat: bool = False, use_amp: bool = True,
                         stage: int = 2, t0: int = 0, slow_turn_focus: bool = False,
                         clearance: float | None = None, stumble: float = 0.0,
                         hard_terrain: bool = False, scan: bool = False) -> ManagerBasedRlEnvCfg:
  """stage 1: learn to walk (flat, no pushes/hits, no quiet/stepping penalties, wide commands).
  stage 2: robustness + careful style (rough terrain, disturbances, quiet contact, careful
  command range); curriculum thresholds start at iteration t0 (resumed step counter).
  stumble: weight of the foot-against-obstacle penalty; hard_terrain: see terrain();
  scan: privileged terrain scan in the critic (actor/deploy contract unchanged)."""
  if stage == 1:
    flat = True
  T = lambda it: (t0 + it) * 24  # noqa: E731
  cmd = MixedVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(4.0, 10.0),
    rel_standing_envs=0.15,
    rel_heading_envs=0.0,
    heading_command=False,
    rel_forward_envs=0.05,
    init_velocity_prob=0.3,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=VX if play else VX_WIDE,
                                            lin_vel_y=VY if play else VY_WIDE,
                                            ang_vel_z=WZ if play else WZ_WIDE, heading=None),
  )
  events: dict[str, EventTermCfg] = {
    "reset_base": EventTermCfg(
      func=envs_mdp.reset_root_state_uniform, mode="reset",
      params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (0.0, 0.02),
                             "yaw": (-math.pi, math.pi)},
              "velocity_range": {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "yaw": (-0.3, 0.3)}}),
    "reset_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset, mode="reset",
      params={"position_range": (-0.05, 0.05), "velocity_range": (-0.2, 0.2),
              "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    **common.randomization_events(),
    **common.disturbance_events(push_vel=0.4, hit_force=60.0),
  }
  t = "twist"
  rewards = {
    # task
    # Tracking on 0.4 s EMA velocities: natural gait sway is not punished, average motion is.
    "track_lin_vel": RewardTermCfg(func=rw.track_lin_vel_ema, weight=3.0,
                                   params={"command_name": t, "std": 0.2}),
    "track_ang_vel": RewardTermCfg(func=rw.track_yaw_ema, weight=3.0,
                                   params={"command_name": t, "std": 0.4}),
    "lin_vel_err": RewardTermCfg(func=rw.lin_vel_err_ema_l1, weight=-2.0, params={"command_name": t}),
    "yaw_rate_err": RewardTermCfg(func=rw.yaw_err_ema_l1, weight=-2.0, params={"command_name": t}),
    "alive": RewardTermCfg(func=envs_mdp.is_alive, weight=0.5),
    # stability
    "upright": RewardTermCfg(func=vel_mdp.upright, weight=1.0,
                             params={"std": math.sqrt(0.05),
                                     "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))}),
    "base_height": RewardTermCfg(func=rw.base_height_l2, weight=-20.0, params={"target": WALK_HEIGHT}),
    "lin_vel_z": RewardTermCfg(func=rw.lin_vel_z_l2, weight=-2.0),
    "ang_vel_xy": RewardTermCfg(func=rw.ang_vel_xy_l2, weight=-0.3),
    "torso_ang_vel": RewardTermCfg(func=vel_mdp.body_angular_velocity_penalty, weight=-0.15,
                                   params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",))}),
    "angular_momentum": RewardTermCfg(func=vel_mdp.angular_momentum_penalty, weight=-0.03,
                                      params={"sensor_name": "robot/root_angmom"}),
    "feet_distance": RewardTermCfg(func=rw.feet_distance, weight=-3.0,
                                   params={"min_dist": 0.17, "max_dist": 0.36}),
    "stand_still": RewardTermCfg(func=rw.stand_still, weight=-0.5, params={"command_name": t}),
    # gait shape
    "pose": RewardTermCfg(
      func=vel_mdp.variable_posture, weight=0.75,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)), "command_name": t,
              "std_standing": {".*": 0.06},
              "std_walking": {r".*hip_pitch.*": 0.3, r".*hip_roll.*": 0.12, r".*hip_yaw.*": 0.15,
                              r".*knee.*": 0.35, r".*ankle_pitch.*": 0.25, r".*ankle_roll.*": 0.1,
                              r"waist_yaw.*": 0.15, r"waist_roll.*": 0.06,
                              r".*shoulder_pitch.*": 0.2, r".*shoulder_roll.*": 0.12,
                              r".*shoulder_yaw.*": 0.12, r".*elbow.*": 0.2, r".*wrist.*": 0.2},
              "std_running": {".*": 0.3},
              "walking_threshold": 0.05, "running_threshold": 10.0}),
    "air_time": RewardTermCfg(func=vel_mdp.feet_air_time, weight=1.5,
                              params={"sensor_name": "feet_contact", "threshold_min": 0.1,
                                      "threshold_max": 0.6, "command_name": t,
                                      "command_threshold": 0.05}),
    "foot_clearance": RewardTermCfg(func=vel_mdp.feet_clearance, weight=-1.0,
                                    params={"target_height": 0.07, "height_sensor_name": "foot_height",
                                            "command_name": t, "command_threshold": 0.05,
                                            "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)}),
    "foot_swing_height": RewardTermCfg(func=vel_mdp.feet_swing_height, weight=-0.5,
                                       params={"sensor_name": "feet_contact",
                                               "height_sensor_name": "foot_height",
                                               "target_height": 0.07, "command_name": t,
                                               "command_threshold": 0.05}),
    "foot_slip": RewardTermCfg(func=vel_mdp.feet_slip, weight=-0.5,
                               params={"sensor_name": "feet_contact", "command_name": t,
                                       "command_threshold": 0.0,
                                       "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES)}),
    "foot_flat": RewardTermCfg(func=rw.foot_flat, weight=-1.0, params={"sensor_name": "feet_contact"}),
    # quiet / soft
    "soft_touchdown": RewardTermCfg(func=rw.soft_touchdown, weight=-20.0,
                                    params={"sensor_name": "feet_contact",
                                            "height_sensor_name": "foot_height", "near_height": 0.04}),
    "impact_peak": RewardTermCfg(func=rw.impact_peak, weight=-8.0,
                                 params={"sensor_name": "feet_contact", "threshold": 1.05}),
    "load_rate": RewardTermCfg(func=rw.load_rate, weight=-1.5, params={"sensor_name": "feet_contact"}),
    # smoothness & effort
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.05),
    "action_acc": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.02),
    "joint_acc": RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_vel": RewardTermCfg(func=envs_mdp.joint_vel_l2, weight=-1e-3),
    "power": RewardTermCfg(func=envs_mdp.electrical_power_cost, weight=-2e-3,
                           params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
    "joint_limits": RewardTermCfg(func=envs_mdp.joint_pos_limits, weight=-5.0),
    "limb_contact": RewardTermCfg(func=_limb_contact, weight=-2.0),
    "terminated": RewardTermCfg(func=envs_mdp.is_terminated, weight=-100.0),
  }
  if stumble:
    rewards["stumble"] = RewardTermCfg(func=rw.feet_stumble, weight=stumble,
                                       params={"sensor_name": "feet_contact"})
  if clearance is not None:
    rewards["foot_clearance"].params["target_height"] = clearance
    rewards["foot_swing_height"].params["target_height"] = clearance
  if slow_turn_focus:
    rewards["track_ang_vel"].params["std"] = 0.3
    rewards["yaw_rate_err"].weight = -3.0
    cmd.turn_slow_prob = 0.7
    cmd.mode_probs = (0.2, 0.3, 0.35, 0.15)
  if use_amp and not play:
    rewards["style"] = RewardTermCfg(func=amp.amp_style, weight=1.5,
                                     params={"data_file": str(AMP_DATA), "save_path": None})
    rewards["pose"].weight = 0.4
  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(func=envs_mdp.bad_orientation,
                                    params={"limit_angle": common.TERMINATION_ANGLE}),
    "illegal_contact": TerminationTermCfg(func=vel_mdp.illegal_contact,
                                          params={"sensor_name": "illegal_contact",
                                                  "force_threshold": 5.0}),
  }
  curriculum = {}
  if not play and stage == 1:
    for k in ("push", "hit_torso", "hit_pelvis", "hit_hands"):
      events.pop(k, None)
    for k in ("soft_touchdown", "impact_peak", "load_rate", "foot_flat", "foot_clearance",
              "foot_swing_height", "foot_slip"):
      rewards[k].weight = 0.0
    curriculum["smoothness"] = CurriculumTermCfg(
      func=cur.robustness_ramp,
      params={"penalties": ["action_rate", "action_acc", "joint_acc", "joint_vel", "power",
                            "stand_still", "ang_vel_xy", "torso_ang_vel", "angular_momentum",
                            "lin_vel_z"],
              "events": [], "start_frac": 0.2, "penalty_delay": T(0), "penalty_steps": 10_000 * 24})
  elif not play:
    curriculum["smoothness"] = CurriculumTermCfg(
      func=cur.robustness_ramp,
      params={"penalties": ["action_rate", "action_acc", "joint_acc", "joint_vel", "power",
                            "stand_still", "ang_vel_xy", "torso_ang_vel", "angular_momentum",
                            "lin_vel_z"],
              "events": [], "start_frac": 0.3, "penalty_delay": T(0), "penalty_steps": 1000 * 24})
    curriculum["quiet_and_disturb"] = CurriculumTermCfg(
      func=cur.robustness_ramp,
      params={"penalties": ["soft_touchdown", "impact_peak", "load_rate", "foot_flat"],
              "events": ["push", "hit_torso", "hit_pelvis", "hit_hands"],
              "start_frac": 0.0, "penalty_delay": T(200), "penalty_steps": 1000 * 24,
              "disturb_delay": T(0), "disturb_steps": 800 * 24})
    curriculum["command_vel"] = CurriculumTermCfg(
      func=vel_mdp.commands_vel,
      params={"command_name": t, "velocity_stages": [
        {"step": 0, "lin_vel_x": VX_WIDE, "lin_vel_y": VY_WIDE, "ang_vel_z": WZ_WIDE},
        {"step": T(600), "lin_vel_x": (-0.5, 0.65), "lin_vel_y": (-0.42, 0.42),
         "ang_vel_z": (-0.85, 0.85)},
        {"step": T(1200), "lin_vel_x": VX, "lin_vel_y": VY, "ang_vel_z": WZ}]})
    curriculum["stepping_taxes"] = CurriculumTermCfg(
      func=cur.robustness_ramp,
      params={"penalties": ["foot_clearance", "foot_swing_height", "foot_slip"], "events": [],
              "start_frac": 0.0, "penalty_delay": T(100), "penalty_steps": 600 * 24})
  if flat:
    terrain_cfg = TerrainEntityCfg(terrain_type="plane")
  else:
    terrain_cfg = TerrainEntityCfg(terrain_type="generator",
                                   terrain_generator=terrain(play, hard=hard_terrain and not play),
                                   max_init_terrain_level=3)
    terminations["out_of_bounds"] = TerminationTermCfg(func=vel_mdp.out_of_terrain_bounds,
                                                       time_out=True)
    if not play:
      curriculum["terrain_levels"] = CurriculumTermCfg(
        func=cur.terrain_levels_survival, params={"command_name": t, "track_tol": 0.45})

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=terrain_cfg, entities={"robot": get_r1_cfg(command_delay_max=4, self_collision=True)},
                   sensors=common.sensors(self_collision=True, scan=scan), num_envs=1, extent=2.0),
    observations={"actor": common.actor_obs(t, noise=not play), "critic": common.critic_obs(t, scan=scan)},
    actions={"joint_pos": PolicyJointPositionActionCfg(entity_name="robot")},
    commands={t: cmd},
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum=curriculum,
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot",
                        body_name="torso_link", distance=2.5, elevation=-10.0, azimuth=90.0),
    sim=common.simulation(rough=not flat),
    decimation=common.DECIMATION,
    episode_length_s=20.0,
  )
  if play:
    cfg.episode_length_s = 1e6
    for k in ("push", "hit_torso", "hit_pelvis", "hit_hands"):
      cfg.events.pop(k, None)
  return cfg


def projectile_walk_env_cfg(play=False, t0=0, min_scale=.25):
  """Physical balls and robot self-contact with the unchanged deployment actor."""
  from dataclasses import replace
  from r1lab.mdp.projectiles import ProjectileBurst, projectile_entities, projectile_sensors

  cfg = recovery_walk_env_cfg(play=play, t0=t0, variant="gated")
  # Preserve the XML's full contact model, including torsional/rolling friction.
  cfg.scene.entities["robot"] = replace(get_r1_cfg(command_delay_max=4, self_collision=True), collisions=())
  cfg.scene.entities.update(projectile_entities(count=8))
  cfg.scene.sensors += projectile_sensors(count=8, history_length=cfg.decimation)
  cfg.sim.nconmax, cfg.sim.njmax, cfg.sim.contact_sensor_maxmatch = 128, 1024, 256
  cfg.sim.mujoco.integrator = "euler"
  cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations = 100, 50
  cfg.sim.mujoco.ccd_iterations = 35
  cfg.commands["twist"].rel_standing_envs = .35
  for name in ("push", "hit_torso", "hit_pelvis", "hit_hands"):
    cfg.events.pop(name, None)
  for reward in cfg.rewards.values():
    if "min_scale" in reward.params:
      reward.params["min_scale"] = min_scale
  cfg.events["projectiles"] = EventTermCfg(
    func=ProjectileBurst, mode="step",
    params={"count": 8, "clean_fraction": 0. if play else .25,
            "speed_range": (6., 18.), "warmup_steps": 1 if play else 300*24,
            "spacing": (.3, .5), "cooldown": (3., 5.), "lifetime": 5.})
  return cfg


def projectile_walk_rl_cfg():
  cfg = common.ppo_runner_cfg("r1_careful_walk", iters=600)
  cfg.algorithm.learning_rate = 2e-4
  cfg.algorithm.schedule = "fixed"
  cfg.algorithm.entropy_coef = .002
  cfg.actor.distribution_cfg["init_std"] = .2
  return cfg


def _limb_contact(env):
  import torch
  s = env.scene["limb_contact"]
  return torch.sum((s.data.found > 0).float(), dim=1)


def careful_walk_rl_cfg():
  return common.ppo_runner_cfg("r1_careful_walk", iters=6000)


def recovery_walk_env_cfg(play=False, t0=0, variant="combined"):
  """Controlled flat-ground recovery ablation from a common walking checkpoint.

  Four variants share command distribution, physical randomization and actor
  contract. Only impulse protocol and state-dependent cost scaling differ.
  """
  from r1lab.mdp.recovery import ImpulseBurst, RecoveryScaledTerm, TrackedImpulse, tracked_velocity_push

  if variant not in ("control", "burst", "gated", "combined"):
    raise ValueError(f"Unknown recovery experiment: {variant}")
  cfg = careful_walk_env_cfg(play=play, flat=True, t0=t0)
  cfg.curriculum = {}
  cmd = cfg.commands["twist"]
  cmd.ranges.lin_vel_x, cmd.ranges.lin_vel_y, cmd.ranges.ang_vel_z = VX, VY, WZ
  # Preserve the original zero-command proportion in this first causal comparison.
  cmd.rel_standing_envs = .15
  for name in ("hit_torso", "hit_pelvis", "hit_hands"):
    if name in cfg.events:
      cfg.events[name].func = TrackedImpulse
  if "push" in cfg.events:
    cfg.events["push"].func = tracked_velocity_push
  # Flat-ground causal comparison: do not restart the earlier terrain/style ramps.
  if not play and variant in ("burst", "combined"):
    for name in ("hit_torso", "hit_pelvis", "hit_hands"):
      cfg.events.pop(name, None)
    cfg.events["impulse_burst"] = EventTermCfg(
      func=ImpulseBurst, mode="step",
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link", "pelvis")),
              # Exempt from bursts only; the legacy push event is shared by all variants.
              "clean_fraction": .4, "warmup_steps": 300 * 24,
              "impulse_range": (3., 12.), "interval_range": (.3, .5),
              "max_shots": 3, "vertical_fraction": (0., .25),
              "body_point_offset": (0., 0., 0.),
              "leading_fraction_range": (.6, .9), "point_offset_range": (.04, .04, .10)})
  if variant in ("gated", "combined"):
    for name in ("stand_still", "pose", "soft_touchdown", "impact_peak", "load_rate",
                 "foot_flat", "action_rate", "action_acc", "joint_acc", "joint_vel", "power"):
      term = cfg.rewards[name]
      term.params = {**term.params, "base_func": term.func, "min_scale": .25}
      if name == "pose":
        term.params["bounded_reward"] = True
      term.func = RecoveryScaledTerm
  return cfg
