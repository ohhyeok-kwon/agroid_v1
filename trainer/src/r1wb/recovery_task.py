"""Matched exposure/posture/prediction contrasts; every joint action remains available."""
import math

import torch
from mjlab.managers.reward_manager import RewardTermCfg

from r1wb import recovery, teacher_task


def env_cfg(mode: str, play: bool = False, *, hit_grace: float = 2.0, **kwargs):
  """Historical v6 configuration; positive grace is rejected by the training entry point."""
  if mode not in ("control", "pose", "predict"):
    raise ValueError("Unknown recovery contrast")
  cfg = teacher_task.teacher_task_env_cfg(
    play=play, disturbance="impulse_events_full", joint_course=135.0, failure_cost=3000.0,
    coherent_course=True, row_share=0.125, decisive_course=True, course_cap=0.12,
    ground=True, reset_impulses=True, hit_grace=hit_grace,
    recovery_cost=recovery.RISK_WEIGHT if mode == "predict" else 0.0,
    **kwargs)
  if "impulse" in cfg.events:
    cfg.events["impulse"].params["reset_wait_mode"] = "equilibrium"
  cfg.rewards["recovery_pose_only"] = RewardTermCfg(func=recovery.recovery_pose_cost,
                                                  weight=-recovery.RISK_WEIGHT if mode == "pose" else 0.0)
  return cfg


def no_hit_grace_env_cfg(mode: str, play: bool = False, **kwargs):
  """The matched v7 contrast: uniform state criteria before, during and after a hit."""
  from r1wb.reward_contract import require_no_hit_exemptions

  if "hit_grace" in kwargs:
    raise ValueError("The no-grace contrast fixes hit_grace=0")
  cfg = env_cfg(mode, play, hit_grace=0.0, **kwargs)
  if cfg.commands["motion"].steer:
    raise ValueError("The no-grace System 1 contrast does not load a historical steering policy")
  require_no_hit_exemptions(cfg)
  return cfg


def physical_ends_env_cfg(mode: str, play: bool = False, **kwargs):
  """v8 (Claude, 11-15): the v7 no-exemption task whose episodes end only on a physical failure (fall, trunk on the
  ground) or on time. Leaving the course's heights/tilt and the goal lag stay priced by their rewards, at every
  instant alike; nothing reads the hit clock. Tests whether v7's extra falls came from tracking endings that now
  fire during recoveries (review 20261115-0215 ★1, 20261115-0303 ★2)."""
  if "plan_ends" in kwargs:
    raise ValueError("The physical-ends contrast fixes plan_ends=False")
  cfg = no_hit_grace_env_cfg(mode, play, plan_ends=False, **kwargs)
  cfg.terminations.pop("goal_lag", None)
  missing = {"fallen", "trunk_ground", "time_out"} - set(cfg.terminations)
  if missing:
    raise ValueError(f"Physical endings missing: {sorted(missing)}")
  extra = set(cfg.terminations) - {"fallen", "trunk_ground", "time_out", "clip_end"}
  if extra:
    raise ValueError(f"Non-physical endings left: {sorted(extra)}")
  return cfg


def plan_free_env_cfg(mode: str, play: bool = False, **kwargs):
  """v8b (Claude, 11-15): bisect of v8 — the v7 no-exemption task without the course's height/tilt endings
  (plan_ends=False) but WITH the leaky goal-lag ending (no time exemption). Separates which removed ending carried
  v8's fall reduction and whether slow-command following rests on the goal-lag ending."""
  if "plan_ends" in kwargs:
    raise ValueError("The plan-free bisect fixes plan_ends=False")
  cfg = no_hit_grace_env_cfg(mode, play, plan_ends=False, **kwargs)
  expected = {"fallen", "trunk_ground", "time_out", "clip_end", "goal_lag"}
  if set(cfg.terminations) != expected:
    raise ValueError(f"Unexpected endings: {sorted(cfg.terminations)}")
  return cfg


ARM_ACTION_KEYS = ("shoulder", "elbow", "wrist")


def arm_action_rate_l2(env) -> torch.Tensor:
  """v9a (Claude, 11-15): squared change of the ARM joints' raw actions between control steps (shoulder, elbow,
  wrist), at every instant alike. mjlab's action_rate_l2 prices all 24 joints at -0.1; this prices the arms' share on
  its own. Sensor-noise diagnosis (review 20261115-0512): ~70% of the standing arms' action variance comes from the
  +-0.5 rad/s joint-velocity input noise the actor already trains with."""
  ids = getattr(env, "_r1wb_arm_action_ids", None)
  if ids is None:
    from r1wb import robot
    names = robot.JOINTS
    picked = [i for i, name in enumerate(names) if any(key in name for key in ARM_ACTION_KEYS)]
    if len(picked) != 10:
      raise ValueError(f"Expected 10 arm actions, found {len(picked)}: {names}")
    ids = torch.tensor(picked, dtype=torch.long, device=env.device)
    env._r1wb_arm_action_ids = ids
  delta = env.action_manager.action - env.action_manager.prev_action
  return torch.sum(torch.square(delta[:, ids]), dim=1)


def arm_smooth_env_cfg(mode: str, play: bool = False, *, arm_rate: float = 1.0, **kwargs):
  """v9a: the v8 physical-endings task plus the arm action-rate price (weight -arm_rate)."""
  if arm_rate <= 0.0:
    raise ValueError("arm_rate must be positive")
  cfg = physical_ends_env_cfg(mode, play, **kwargs)
  cfg.rewards["arm_action_rate_l2"] = RewardTermCfg(func=arm_action_rate_l2, weight=-arm_rate)
  return cfg


def heading_return_env_cfg(mode: str, play: bool = False, *, yaw_weight: float = 1.5, **kwargs):
  """v9c (Claude, 11-15): the v8 physical-endings task with the goal-lag reward's heading share raised
  (goal_lag_l1 yaw_weight 0.3 -> yaw_weight). Reward only, at every instant alike; no ending, no hit clock.
  v8 walks at the commanded speed after hits but returns less to the commanded heading (review 20261115-0430)."""
  if yaw_weight <= 0.0:
    raise ValueError("yaw_weight must be positive")
  cfg = physical_ends_env_cfg(mode, play, **kwargs)
  # The goal terms share one params dict object; replace (do not mutate) so only goal_lag changes.
  cfg.rewards["goal_lag"].params = {**cfg.rewards["goal_lag"].params, "yaw_weight": yaw_weight}
  return cfg


def heading_arm_smooth_env_cfg(mode: str, play: bool = False, *, yaw_weight: float = 1.5,
                               arm_rate: float = 1.0, **kwargs):
  """v9d: the v9c heading-return configuration plus v9a's continuous arm action-rate cost.

  Each component has a measured standalone contrast. Their interaction is a new
  experiment: no assumption of additive benefit, hit timer or action restriction.
  """
  if not math.isfinite(arm_rate) or arm_rate <= 0.0:
    raise ValueError("arm_rate must be finite and positive")
  if not math.isfinite(yaw_weight) or yaw_weight <= 0.0:
    raise ValueError("yaw_weight must be finite and positive")
  cfg = heading_return_env_cfg(mode, play, yaw_weight=yaw_weight, **kwargs)
  cfg.rewards["arm_action_rate_l2"] = RewardTermCfg(func=arm_action_rate_l2, weight=-arm_rate)
  return cfg


def upper_body_course_l1(env, command_name: str, cap: float | None = None, stream: str = "always", grace: float = 0.0):
  """v10a diagnostic: the same course error on the two waist and ten arm joints.

  Keep the original 24-joint denominator so each retained joint has its original weight. All leg actions remain
  available; all other tracking, stability and task rewards remain unchanged.
  """
  term = env.command_manager.get_term(command_name)
  ids = getattr(env, "_r1wb_upper_course_ids", None)
  if ids is None:
    from r1wb import robot
    names = robot.JOINTS
    picked = [i for i,name in enumerate(names) if not any(key in name for key in ("hip", "knee", "ankle"))]
    if len(picked) != 12:
      raise ValueError(f"Expected two waist and ten arm joints, found {len(picked)}")
    ids = torch.tensor(picked, dtype=torch.long, device=env.device)
    env._r1wb_upper_course_ids = ids
  distance = (term.robot_joint_pos[:,ids] - term.joint_pos[:,ids]).abs().sum(dim=1) / term.robot_joint_pos.shape[1]
  if cap is not None:
    distance = distance.clamp(max=cap)
  from r1wb.tracking.mdp import course_is_trusted
  return distance * course_is_trusted(env, command_name, stream, grace)


def upper_body_course_env_cfg(mode: str, play: bool = False, **kwargs):
  """v10a = v9c with only the joint_course joint subset changed."""
  cfg = heading_return_env_cfg(mode, play, **kwargs)
  old = cfg.rewards["joint_course"]
  cfg.rewards["joint_course"] = RewardTermCfg(func=upper_body_course_l1, weight=old.weight, params=dict(old.params))
  return cfg


def height_scan_env_cfg(mode: str, play: bool = False, **kwargs):
  """v10s = v9c with the existing 13x9 height scan as privileged actor input."""
  if "senses" in kwargs:
    raise ValueError("v10s fixes senses=scan")
  return heading_return_env_cfg(mode, play, senses="scan", **kwargs)


def upper_body_scan_env_cfg(mode: str, play: bool = False, **kwargs):
  """v11a = v10s with the 12 leg contributions removed from joint_course.

  Retain each of the 12 upper-body joint coefficients (the original denominator
  remains 24) and the original scalar reward cap. No action is disabled.
  """
  cfg = height_scan_env_cfg(mode, play, **kwargs)
  old = cfg.rewards["joint_course"]
  cfg.rewards["joint_course"] = RewardTermCfg(func=upper_body_course_l1,
                                            weight=old.weight, params=dict(old.params))
  return cfg


def noise_consistency_scan_env_cfg(mode: str, play: bool = False, **kwargs):
  """v11c = v10s + same-state independent sensor consistency auxiliary loss."""
  from r1wb.noise_consistency import add_consistency_observations
  return add_consistency_observations(height_scan_env_cfg(mode, play, **kwargs))


def noise_consistency_scan_rl_cfg():
  from r1wb.noise_consistency import consistency_rl_cfg
  actor = height_scan_env_cfg("control", play=False).observations["actor"]
  return consistency_rl_cfg(actor, 0.0011503888558512206)
