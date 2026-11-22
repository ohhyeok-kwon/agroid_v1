"""E1 bracket of the W1-J tracker (reference executor for the judge). Plan: .agent/adversarial/rounds/R01/wam/e1_plan.md

Built on r1wb's tracking task (P1c family; reused, not modified). Same for both arms: data (shared/motion_train v0,
captured human arms, intent arms only where arms_valid=False), observations (P1c actor + a denser preview:
1, 2, 3, 5, 10, 25, 50 steps ahead), network, PPO settings (init std 0.05 as SONIC), SONIC's reset perturbation,
nominal dynamics and noise-free observations (no randomisation, no pushes, no delay, no actor corruption), terminations, SONIC's six tracking kernels + end-effector
kernel (wrists, ankles; sigma 0.1, weight 2) + joint limit + self collision.
  A (SONIC as published): joint target = default pose + scale * a; absolute action-rate (-0.1), wrist 'anti-shake'
    (-5e-3 beyond 1.5 rad/s), ankle joint acceleration (-2.5e-6); plain references.
  B (identity): joint target = next reference angles + scale * a (the action-rate price then charges only the excess
    over the reference), joint-angle kernel (sigma 0.1 rad) and joint-velocity kernel (sigma 1.0 rad/s), stance slip
    beyond the reference's own and unprompted foot lift (r1wb P1c terms); library with one spectrally augmented copy
    per clip (2-20 Hz, 0.004-0.015 rad RMS).
"""
from __future__ import annotations

from pathlib import Path

from mjlab.envs import mdp as envs_mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.tracking import mdp

from r1lab.run_record import work_root
from r1wb import robot
from r1wb.tracking import env_cfg as base

from r1adv.wam import terms

FUTURE = (1, 2, 3, 5, 10, 25, 50)


def _code_record() -> dict:
  """sha256 of the sources this process actually imported (coordinator 09:54: hashes at import time; the batch runs
  from a code copy because trainer-sync mirrors PC edits into /rl-agent/src every 3 s)."""
  import hashlib, json, sys
  names = ("r1adv.wam.track_task", "r1adv.wam.terms", "r1adv.wam.library", "r1wb.tracking.env_cfg",
           "r1wb.tracking.command", "r1wb.tracking.mdp", "r1wb.robot", "r1lab.registry", "r1lab.tasks")
  out = {}
  for n in names:
    m = sys.modules.get(n)
    f = getattr(m, "__file__", None) if m else None
    out[n] = [f, hashlib.sha256(Path(f).read_bytes()).hexdigest() if f else None]
  print("[wam-code-at-import] " + json.dumps(out), flush=True)
  return out
SONIC_POSE = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01),
              "roll": (-0.1, 0.1), "pitch": (-0.1, 0.1), "yaw": (-0.2, 0.2)}
SONIC_VEL = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "z": (-0.2, 0.2),
             "roll": (-0.52, 0.52), "pitch": (-0.52, 0.52), "yaw": (-0.78, 0.78)}
WRISTS = ("left_wrist_roll_link", "right_wrist_roll_link")


def library_path(arm: str) -> Path:
  return work_root() / f"data/wam/e1_library_{arm}.npz"


def env_cfg(arm: str, play: bool = False, first_clip: int = 0, library: str | None = None, **kw):
  if arm not in ("A", "B"):
    raise ValueError(arm)
  _code_record()
  cfg = base.tracking_env_cfg(play, first_clip, hardened=False, clean=True, slip_excess=True,
                              contact_match=(arm == "B"), disturbance="kick", **kw)
  cfg.events.pop("push_robot", None)                       # nominal: no pushes in either arm
  cfg.observations["actor"].enable_corruption = False      # W1-J runs noise-free in sim: no observation noise
  mc = cfg.commands["motion"]
  mc.motion_file = str(library or library_path(arm))
  mc.future_steps = FUTURE
  if not play:
    mc.pose_range, mc.velocity_range, mc.joint_position_range = dict(SONIC_POSE), dict(SONIC_VEL), (-0.1, 0.1)
  r = cfg.rewards
  for name in ("feet_pos", "anchor_xy_l1", "anchor_z_l1"):  # P1c additions that SONIC does not have: out of both arms
    r.pop(name, None)
  r["ee_pos"] = RewardTermCfg(func=mdp.motion_relative_body_position_error_exp, weight=2.0,
                              params={"command_name": "motion", "std": 0.1, "body_names": robot.END_BODIES})
  if arm == "A":
    r.pop("stance_slip", None)
    r.pop("foot_lift", None)
    r["anti_shake"] = RewardTermCfg(func=terms.anti_shake_ang_vel, weight=-5e-3,
                                    params={"body_names": WRISTS, "threshold": 1.5})
    r["feet_acc"] = RewardTermCfg(func=envs_mdp.joint_acc_l2, weight=-2.5e-6,
                                  params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*_ankle_.*",))})
  else:
    r["joint_pos"] = RewardTermCfg(func=terms.joint_pos_error_exp, weight=1.0, params={"command_name": "motion", "std": 0.1})
    r["joint_vel"] = RewardTermCfg(func=terms.joint_vel_error_exp, weight=1.0, params={"command_name": "motion", "std": 1.0})
    cfg.actions = {"joint_pos": terms.ReferenceResidualJointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), scale=dict(robot.ACTION_SCALE), command_name="motion")}
  return cfg


def rl_cfg(init_std: float = 0.05):
  """init_std 0.05 = SONIC (E1 first launch, stalled: KL-adaptive steps scale with sigma, e1_progress 12:22);
  1.0 = the value P1 trained with (E1 relaunch candidate, both arms)."""
  cfg = base.tracking_rl_cfg()
  cfg.actor.distribution_cfg = {**cfg.actor.distribution_cfg, "init_std": init_std}
  cfg.experiment_name = "r1_r1adv_wam_track"
  cfg.max_iterations = 6000
  cfg.save_interval = 500
  return cfg
