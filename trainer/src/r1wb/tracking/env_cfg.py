"""P1 teacher environment: mjlab's BeyondMimic-style tracking task on the R1 + O6 motion library.

Second version (after the first run and two independent reviews): pelvis anchor, terminations that see planar drift,
heading and depth, L1 terms next to the kernels, a feet term, weaker and partly absent pushes, small reset noise,
the reference root motion and a 1 s preview in the observations, and a library whose arms are the intent arms.
"""
from __future__ import annotations

from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking import mdp
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from r1lab.run_record import work_root
from r1wb import robot
from r1wb.tracking import mdp as wb
from r1wb.tracking.command import LibraryMotionCommandCfg, clip_finished

EXPERIMENT = "r1_wholebody"
ANCHOR = "pelvis"
ORIENTATION_BODIES = tuple(b for b in robot.TRACKED_BODIES if "wrist" not in b and "hip" not in b and "shoulder" not in b)


def library_path(subset: str = "core") -> Path:
  return work_root() / f"data/wb/library_v2_{subset}.npz"


def sensors() -> tuple:
  feet = ContactSensorCfg(
    name="feet_ground", primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                                             entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1)
  trunk = ContactSensorCfg(
    name="trunk_ground", primary=ContactMatch(mode="body", pattern=r"^(pelvis|torso_link|waist_roll_link)$", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1)
  own = ContactSensorCfg(
    name="self_collision", primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"), reduce="none", num_slots=1, history_length=4)
  return (own, feet, trunk)


def tracking_env_cfg(play: bool = False, first_clip: int = 0, subset: str = "core", hardened: bool = False,
                     clean: bool | None = None, slip_excess: bool = False,
                     contact_match: bool = False, disturbance: str = "kick",
                     drift_limit: float = 0.5, joint_course: float = 0.0,
                     joint_course_cap: float | None = None, hit_grace: float = 0.0) -> ManagerBasedRlEnvCfg:
  """hardened: the conditions the deployed network trains under (actuator delay 0-10 ms, PD gains and trunk mass
  varied). The teacher is fine-tuned in them before it labels a student, so its labels are valid there.
  play: evaluation layout (environment i replays clip i from its first frame, no pushes, no reset noise).
  clean: no randomised dynamics and no delay; defaults to `play`. `play=True, clean=False` is the evaluation under
  the hardened conditions. slip_excess: charge stance-foot speed only beyond the reference's own (see mdp).
  contact_match: charge lifting a foot the reference keeps planted (see mdp.foot_lift_unprompted).
  disturbance: "kick" = root-velocity kicks of 0.3 m/s every 2 to 5 s (second version); "strong_kick" = the framework's
  stock kicks (0.5 m/s, 0.52 / 0.78 rad/s, every 1 to 3 s); "impulse" = force impulses of 5 to 25 N s at the torso,
  one policy step long, every 1 to 3 s (the evaluation's disturbance); "impulse_mix" = the same impulses spread over
  20 ms to 0.4 s, with the range opened over the first 12,000 steps (for training from scratch). drift_limit: planar drift that ends an episode.
  joint_course: price per rad of the mean absolute joint distance to the reference's joint angles (0 = off).
  hit_grace: seconds after an external hit during which the tracking terminations are not applied (a physical fall
  always ends the episode); 0 = the tracking terminations always apply."""
  clean = play if clean is None else clean
  cfg = make_tracking_env_cfg()
  cfg.scene.entities = {"robot": robot.robot_cfg(delay_steps=2 if hardened and not clean else 0)}
  cfg.scene.sensors = sensors()
  cfg.actions = {"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",),
                                                     scale=dict(robot.ACTION_SCALE), use_default_offset=True)}
  name = {"command_name": "motion"}
  cfg.commands = {"motion": LibraryMotionCommandCfg(
    entity_name="robot", resampling_time_range=(1.0e9, 1.0e9), debug_vis=not play,
    motion_file=str(library_path(subset)), anchor_body_name=ANCHOR, body_names=robot.TRACKED_BODIES,
    pose_range={} if play else {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.0, 0.02),
                                "roll": (-0.05, 0.05), "pitch": (-0.05, 0.05), "yaw": (-0.2, 0.2)},
    velocity_range={} if play else {"x": (-0.2, 0.2), "y": (-0.2, 0.2), "z": (-0.1, 0.1),
                                    "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.4, 0.4)},
    joint_position_range=(0.0, 0.0) if play else (-0.05, 0.05),
    sampling_mode="uniform" if play else "adaptive", pin_clips=play, first_clip=first_clip)}
  # Observations: the actor is privileged (never deployed) but must not rely on noise it cannot average.
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    terms["reference_velocity"] = ObservationTermCfg(func=wb_reference_velocity, params=name)
  cfg.observations["actor"].terms["motion_anchor_pos_b"].noise = Unoise(n_min=-0.02, n_max=0.02)
  cfg.observations["actor"].terms["base_lin_vel"].noise = Unoise(n_min=-0.05, n_max=0.05)
  # Rewards: stock six, orientation terms without the unmeasured wrist roll, plus feet, L1 anchor terms and slip.
  cfg.rewards["motion_body_ori"].params["body_names"] = ORIENTATION_BODIES
  cfg.rewards["motion_body_ang_vel"].params["body_names"] = ORIENTATION_BODIES
  cfg.rewards.update({
    "feet_pos": RewardTermCfg(func=mdp.motion_relative_body_position_error_exp, weight=1.0,
                              params={**name, "std": 0.1, "body_names": robot.FOOT_BODIES}),
    "anchor_xy_l1": RewardTermCfg(func=wb.anchor_xy_error_l1, weight=-1.0, params=name),
    "anchor_z_l1": RewardTermCfg(func=wb.anchor_z_error_l1, weight=-2.0, params=name),
    "stance_slip": RewardTermCfg(func=wb.stance_slip, weight=-2.0,
                                 params={"sensor_name": "feet_ground", "foot_bodies": robot.FOOT_BODIES,
                                         **(name if slip_excess else {})}),
  })
  if joint_course > 0.0:
    cfg.rewards["joint_course"] = RewardTermCfg(func=wb.joint_course_l1, weight=-joint_course,
                                                params={**name, **({} if joint_course_cap is None else {"cap": joint_course_cap})})
  if contact_match:
    cfg.rewards["foot_lift"] = RewardTermCfg(func=wb.foot_lift_unprompted, weight=-1.0,
                                             params={**name, "sensor_name": "feet_ground", "foot_bodies": robot.FOOT_BODIES})
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "clip_end": TerminationTermCfg(func=clip_finished, time_out=True, params=name),
    "anchor_z": TerminationTermCfg(func=mdp.bad_anchor_pos_z_only, params={**name, "threshold": 0.15}),
    "anchor_xy": TerminationTermCfg(func=wb.anchor_xy_error, params={**name, "threshold": drift_limit}),
    "anchor_rot": TerminationTermCfg(func=wb.anchor_rotation_error, params={**name, "threshold": 0.8}),
    "feet_z": TerminationTermCfg(func=mdp.bad_motion_body_pos_z_only,
                                 params={**name, "threshold": 0.15, "body_names": robot.FOOT_BODIES}),
    "hands_z": TerminationTermCfg(func=mdp.bad_motion_body_pos_z_only,
                                  params={**name, "threshold": 0.25,
                                          "body_names": ("left_wrist_roll_link", "right_wrist_roll_link")}),
    "trunk_ground": TerminationTermCfg(func=wb.body_on_ground, params={"sensor_name": "trunk_ground"}),
  }
  if hit_grace > 0.0:
    grace = {"grace": hit_grace}
    cfg.terminations.update({
      "anchor_z": TerminationTermCfg(func=wb.judged_anchor_z, params={**name, "threshold": 0.15, **grace}),
      "anchor_xy": TerminationTermCfg(func=wb.judged_anchor_xy, params={**name, "threshold": drift_limit, **grace}),
      "anchor_rot": TerminationTermCfg(func=wb.judged_anchor_rot, params={**name, "threshold": 0.8, **grace}),
      "feet_z": TerminationTermCfg(func=wb.judged_body_z, params={**name, "threshold": 0.15, "body_names": robot.FOOT_BODIES, **grace}),
      "hands_z": TerminationTermCfg(func=wb.judged_body_z, params={**name, "threshold": 0.25,
                                                                  "body_names": ("left_wrist_roll_link", "right_wrist_roll_link"), **grace}),
      "fallen": TerminationTermCfg(func=wb.fallen),
    })
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = robot.FOOT_GEOMS
  cfg.events["foot_friction"].params["ranges"] = (0.4, 1.2)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)
  cfg.events["hand_mass"] = EventTermCfg(
    mode="startup", func=dr.body_mass,
    params={"asset_cfg": SceneEntityCfg("robot", body_names=robot.HAND_BODIES), "operation": "add",
            "ranges": (-0.15, 0.6)})
  cfg.events["push_robot"] = EventTermCfg(
    func=wb.push_unless_calm, mode="interval", interval_range_s=(2.0, 5.0),
    params={**name, "calm_fraction": 0.2,
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.1, 0.1),
                               "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.4, 0.4)}})
  if disturbance == "strong_kick":
    cfg.events["push_robot"].interval_range_s = (1.0, 3.0)
    cfg.events["push_robot"].params["velocity_range"] = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
                                                         "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
  elif disturbance in ("impulse", "impulse_mix", "impulse_bursts", "impulse_events", "impulse_events_full",
                       "impulse_events_anywhere", "impulse_events_rows"):
    cfg.events.pop("push_robot")
    cfg.events["impulse"] = EventTermCfg(
      func=wb.impulse_unless_calm, mode="step",
      params={**name, "calm_fraction": 0.2, "impulse_range": (5.0, 25.0), "interval_s": (1.0, 3.0),
              "body": "torso_link", "height": 0.1})
    if disturbance == "impulse_mix":      # balls and shoves (20 ms to 0.4 s), opened over the first 500 iterations
      cfg.events["impulse"].params.update(duration_s=(0.02, 0.4), ramp_steps=12000)
    if disturbance == "impulse_bursts":   # half of the events are 4 to 12 balls of 8 to 16 N s, 0.08 to 0.25 s apart
      cfg.events["impulse"].params.update(burst_prob=0.5, burst_hits=(4, 12), burst_gap_s=(0.08, 0.25),
                                          burst_impulse=(8.0, 16.0))
    if disturbance.startswith("impulse_events"):
      # Separate events: the next one comes after the last has been dealt with (recovery measured 0.4 to 1.3 s, grace
      # 2 s), so most of the time is calm. With a hit every 1 to 3 s only 40% of the steps were outside the grace
      # (measured 11-12 06:48), and what is judged only while calm (progress on the command) was hardly judged.
      cfg.events["impulse"].params.update(interval_s=(4.0, 8.0))
    # What the evaluation throws: single balls and rows of balls, anywhere between the thighs and the head, also off
    # the centre line ("full"). The two halves alone: "anywhere" = single hits at any height / side, "rows" = rows
    # of balls at the trunk's centre (which half makes the reaction to a single trunk hit large: E77, E78).
    if disturbance in ("impulse_events_full", "impulse_events_rows"):
      cfg.events["impulse"].params.update(burst_prob=0.5, burst_hits=(4, 12), burst_gap_s=(0.08, 0.25),
                                          burst_impulse=(8.0, 16.0))
    if disturbance in ("impulse_events_full", "impulse_events_anywhere"):
      cfg.events["impulse"].params.update(height=(-0.35, 0.40), side=(-0.2, 0.2))
  else:
    assert disturbance == "kick", disturbance
  if hardened:
    cfg.events["pd_gains"] = EventTermCfg(
      mode="startup", func=dr.pd_gains,
      params={"asset_cfg": SceneEntityCfg("robot"), "kp_range": (0.9, 1.1), "kd_range": (0.9, 1.1), "operation": "scale"})
    cfg.events["trunk_mass"] = EventTermCfg(
      mode="startup", func=dr.body_mass,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add", "ranges": (-1.0, 2.0)})
  cfg.viewer.body_name = ANCHOR
  cfg.sim.nconmax, cfg.sim.njmax = 64, 400
  cfg.episode_length_s = 10.0
  if play:
    cfg.episode_length_s = 1.0e9
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.events.pop("impulse", None)
  if clean:
    for event in ("hand_mass", "base_com", "encoder_bias", "foot_friction", "pd_gains", "trunk_mass"):
      cfg.events.pop(event, None)
  return cfg


def use_local_anchor(cfg: ManagerBasedRlEnvCfg, groups: tuple[str, ...] = ("actor", "critic")) -> None:
  """Show the reference from where the robot is: no planar or heading error in the anchor terms of `groups`."""
  for group in groups:
    terms = cfg.observations[group].terms
    terms["motion_anchor_pos_b"].func = wb.local_anchor_pos_b
    terms["motion_anchor_ori_b"].func = wb.local_anchor_ori_b


def use_local_terminations(cfg: ManagerBasedRlEnvCfg) -> None:
  """Terminations that do not depend on planar or heading drift (height and tilt of the anchor stay)."""
  cfg.terminations.pop("anchor_xy", None)
  cfg.terminations["anchor_rot"] = TerminationTermCfg(func=wb.anchor_tilt_error,
                                                      params={"command_name": "motion", "threshold": 0.8})


def teacher_observation(env, command_name: str, last_action, local: bool = False, task: bool = False,
                        width: int | None = None):
  """The teacher actor's input, noise-free, in the order of the "actor" group above (single place that knows it).
  local: anchor terms without planar and heading error (labels for a student that cannot observe that drift).
  task: the teacher of r1wb.teacher_task, which also reads the goal and the goal lag (appended).
  width: the policy's input width instead of `task`: 273, 294 (task), or 294 plus the senses of r1wb.senses."""
  import torch
  kind = ""
  if width is not None:
    from r1wb import senses
    kind, task = senses.kind_of(width), width != 273
  anchor_pos = wb.local_anchor_pos_b if local else mdp.motion_anchor_pos_b
  anchor_ori = wb.local_anchor_ori_b if local else mdp.motion_anchor_ori_b
  term = env.command_manager.get_term(command_name)
  parts = [
    term.command,
    anchor_pos(env, command_name), anchor_ori(env, command_name),
    mdp.builtin_sensor(env, "robot/imu_lin_vel"), mdp.builtin_sensor(env, "robot/imu_ang_vel"),
    mdp.joint_pos_rel(env, biased=True), mdp.joint_vel_rel(env), last_action,
    wb_reference_velocity(env, command_name)]
  if task:
    parts += [term.goal, term.goal_lag]
  if kind:
    parts.append(senses.read(env, kind))
  return torch.cat(parts, dim=1)


def wb_reference_velocity(env, command_name: str):
  """Reference root (anchor) linear and angular velocity in the reference anchor's own frame."""
  import torch
  from mjlab.utils.lab_api.math import quat_apply_inverse
  term = env.command_manager.get_term(command_name)
  return torch.cat([quat_apply_inverse(term.anchor_quat_w, term.anchor_lin_vel_w),
                    quat_apply_inverse(term.anchor_quat_w, term.anchor_ang_vel_w)], dim=1)


def tracking_rl_cfg(hidden: tuple[int, ...] = (1024, 512, 256)) -> RslRlOnPolicyRunnerCfg:
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(hidden_dims=hidden, activation="elu", obs_normalization=True,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0,
                                          "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=hidden, activation="elu", obs_normalization=True),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=0.2, entropy_coef=0.005,
      num_learning_epochs=5, num_mini_batches=4, learning_rate=1.0e-3, schedule="adaptive",
      gamma=0.99, lam=0.95, desired_kl=0.01, max_grad_norm=1.0),
    experiment_name=EXPERIMENT, save_interval=250, num_steps_per_env=24, max_iterations=6000,
    logger="tensorboard", upload_model=False)
