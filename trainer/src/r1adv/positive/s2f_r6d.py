"""R6 diagnosis arm 'r6d' of the positive agent (positive session 8; coordinator 2026-11-18 10:5x KST,
.agent/adversarial/rounds/R04/04_r4_state.md 10:58 note): does a policy that already STEPS keep stepping when robustness
is trained on top of it? Diagnostic only: the parent's objective keeps its gait terms (air_time, foot_clearance,
foot_swing_height, ...), which are reward patches, so no cell here is a wave-4 candidate.

Parent: s2f/3300 = runs/r1_careful_walk/2026-11-08_08-13-55_s2f_quiet/model_3300.pt (sha256 debe54ed...), r1lab
careful_walk stage 2 with curriculum offset t0 = 1499 (its env.yaml: smoothness / quiet_and_disturb / command_vel /
stepping_taxes delays 35976, 40776, 50376, 64776, 38376 = (1499 + 0, 200, 600, 1200, 100) x 24), AMP style reward (data
careful_walk.npz 3ff0f61e..., discriminator resumed with --amp-disc). The parent's own objective, PPO settings,
observations, commands and push are kept; only the hit events change:

  ctl   careful_walk_env_cfg(stage=2, t0=1499) unchanged (= task careful_walk_s2 --t0 1499 with today's code; one physics
        difference from the parent's 11-08 run: robot self-collision is on, mandatory in r1lab.train)
  imp   ctl with the parent's hit events (hit_torso +-60 N per axis, hit_pelvis +-36 N, hit_hands +-25 N; 0.05-0.8 s,
        cooldown 2-8 s) replaced by the Hold hit stream R2.torso_hits with the pinned R2 training distribution
        (r2_train_distribution_v1.json 9dff1729...): one horizontal torso impulse U(4, 25) N s held for one 0.02 s control
        step, azimuth U(0, 2 pi) from the heading, height U(-0.1, 0.3) m above the torso CoM, first U(1.5, 5) s after the
        episode start, then every U(2, 6) s. The hit clock is episode time only: hits land whatever the command, standing
        (zero) commands included. The velocity push (event 'push', +-0.4 m/s every 3-7 s) is a velocity kick, not a hit,
        and stays.
  long  imp with option S of hold_w4s (positive 4, unchanged): 30 % of the hits stay the 0.02 s impulse, 70 % become a
        half-sine push of T ~ U(0.1, 0.8) s with the same impulse (with_human_push).
The curriculum quiet_and_disturb still names the removed hit terms: r1lab robustness_ramp skips names without a term, and
from the parent's restored step counter (79440) every ramp is complete (disturb scale 1), so the push keeps its full range;
the Hold hits are not ramped.
"""
from __future__ import annotations

import math

T0 = 1499
CELLS = ("ctl", "imp", "long")
HIT_EVENTS = ("hit_torso", "hit_pelvis", "hit_hands")
PARENT = dict(run="runs/r1_careful_walk/2026-11-08_08-13-55_s2f_quiet", checkpoint="model_3300.pt",
              checkpoint_sha256="debe54edf0a172015c4bf61ebc81c73daed044128972dba312f6e447a4e68e64",
              amp_disc_sha256="7364eaff140233db8573a65963022b7a1eae6164206e77ad2e77cb614f4f5726",
              amp_data_sha256="3ff0f61e6e34a34c31c88e99f724f143f1a20cc36b9f38e684834e94dd50819f",
              common_step_counter=79440, t0=T0)


def s2f_env_cfg(cell: str, play: bool = False, t0: int = T0):
  """The careful_walk stage-2 environment of s2f (offset t0) with the cell's hits (see the module docstring)."""
  from r1lab.tasks.careful_walk import careful_walk_env_cfg
  if cell not in CELLS:
    raise ValueError(f"unknown r6d cell {cell!r}")
  cfg = careful_walk_env_cfg(play=play, stage=2, t0=t0)
  if play or cell == "ctl":
    return cfg
  from mjlab.managers.event_manager import EventTermCfg
  from r1adv.positive import hold_r2 as R2
  missing = [k for k in HIT_EVENTS if k not in cfg.events]
  if missing or "push" not in cfg.events or "torso_hits" in cfg.events:
    raise ValueError(f"unexpected careful_walk events {sorted(cfg.events)}")
  for k in HIT_EVENTS:
    cfg.events.pop(k)
  hits = R2.distribution()["hits"]
  if abs(hits["duration_s"] - 0.02) > 1e-9:
    raise ValueError("hit duration of the distribution file is not one policy step")
  if abs(hits["azimuth_rad"][1] - hits["azimuth_rad"][0] - 2 * math.pi) > 1e-6:
    raise ValueError("hit azimuth range of the distribution file is not the full circle")
  cfg.events["torso_hits"] = EventTermCfg(func=R2.torso_hits, mode="step", params=dict(
    impulse_ns=list(hits["impulse_ns"]), height_m=list(hits["height_m"]), first_s=list(hits["first_s"]),
    gap_s=list(hits["gap_s"]), code_sha256=R2.CODE_SHA256, distribution_sha256=R2.R2_DIST["sha256"]))
  if cell == "long":
    from r1adv.positive.hold_w4s import with_human_push
    with_human_push(cfg)
  return cfg


def s2f_rl_cfg():
  """The parent's PPO recipe unchanged (careful_walk_rl_cfg: save every 100; --iters sets the length)."""
  from r1lab.tasks.careful_walk import careful_walk_rl_cfg
  return careful_walk_rl_cfg()
