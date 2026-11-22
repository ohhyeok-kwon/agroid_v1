"""S1 tasks for the whole-body runner (r1wb.train reads r1wb.tasks.TASKS; registered in r1lab.registry).

  wb_s1_s0_joule   S0 (no prior), effort = Joule heating
  wb_s1_s0_power   S0 (no prior), effort = mechanical power

Exactly one of the two is launched (negative 18's training-free check picks the effort). Both start from random
weights (no --resume, no --start-from).
"""
from __future__ import annotations

import functools


def _env(effort: str, play: bool = False, **kw):
  from r1s1.env_cfg import s1_env_cfg
  return s1_env_cfg(effort=effort, play=play, **kw)


def _rl():
  from r1s1.rl_cfg import s1_rl_cfg
  return s1_rl_cfg()


TASKS = {
  f"wb_s1_s0_{effort}": dict(env=functools.partial(_env, effort), rl=_rl, stage=f"S1-S0-{effort}", s1=True)
  for effort in ("joule", "power")
}

SPECS = {
  "wb_s1_s0_joule": {"action": "wholebody", "tags": "s1v1-s0-outcome-goal-joule-scratch", "runner": "r1wb"},
  "wb_s1_s0_power": {"action": "wholebody", "tags": "s1v1-s0-outcome-goal-power-scratch", "runner": "r1wb"},
}


# R12 sigma-direction measurement (coordinator 2026-11-20 02:5x; negative 18 addendum 23, freeze 42): S0's checkpoint
# continued 500 updates in three cells. Registered only in the measurement code copy (tools/s1_register.py --sigma).
def _rl_e0():
  from r1s1.rl_cfg import s1_rl_cfg
  return s1_rl_cfg(entropy=0.0)


TASKS.update({
  "wb_s1_s0_power_e0": dict(env=functools.partial(_env, "power"), rl=_rl_e0, stage="S1-S0-power-e0", s1=True),
  "wb_s1_v3_power": dict(env=functools.partial(_env, "power", success="v3"), rl=_rl, stage="S1-v3-power", s1=True),
})
SPECS.update({
  "wb_s1_s0_power_e0": {"action": "wholebody", "tags": "s1-sigma-b-entropy0", "runner": "r1wb"},
  "wb_s1_v3_power": {"action": "wholebody", "tags": "s1-sigma-c-objective-v3", "runner": "r1wb"},
})


# R12 S0b (coordinator 2026-11-20 04:1x-05:2x; negative 18 addenda 31-34, freezes 56-59): S0 unchanged (objective v2,
# entropy 0.001, rsl_rl's schedule: numbers.LR_FLOOR = 1e-5) run by r1s1.learning.S1PPO, which adds per-update records
# (KL, per-dimension KL share, per-joint std) and the in-run gates S1 / S2. The fix slot is open (not chosen yet).
def _rl_s0b():
  from r1s1.rl_cfg import s1_rl_cfg
  return s1_rl_cfg(algorithm="r1s1.learning:S1PPO")


TASKS["wb_s1_s0b_power"] = dict(env=functools.partial(_env, "power"), rl=_rl_s0b, stage="S1-S0b-power", s1=True)
SPECS["wb_s1_s0b_power"] = {"action": "wholebody", "tags": "s1-s0b-scratch", "runner": "r1wb"}


# R12 S0b sigma arms (coordinator 2026-11-20 06:1x; negative 18 addendum 37): prepared, not chosen. The same S0b run with
# the policy std mirror-tied (ㄱ), in log space (ㄷ) or both (r1s1.distribution, numbers.SIGMA_PARAMS).
def _rl_s0b_sigma(sigma):
  def rl():
    from r1s1.rl_cfg import s1_rl_cfg
    return s1_rl_cfg(algorithm="r1s1.learning:S1PPO", sigma=sigma)
  return rl


for _sigma in ("tied", "log", "tied_log"):
  TASKS[f"wb_s1_s0b_power_{_sigma}"] = dict(env=functools.partial(_env, "power"), rl=_rl_s0b_sigma(_sigma),
                                            stage=f"S1-S0b-power-{_sigma}", s1=True)
  SPECS[f"wb_s1_s0b_power_{_sigma}"] = {"action": "wholebody", "tags": f"s1-s0b-{_sigma.replace('_', '-')}-scratch",
                                        "runner": "r1wb"}



# R12 S0b-fixed (coordinator 2026-11-20 07:5x; negative 18 R-REPLAY 'split'): S0 with the policy std not learned, a constant
# per-joint vector from the records (numbers.FIXED_STD, r1s1.distribution.FixedGaussian), plus post-deadline records.
def _env_fixed(play=False, **kw):
  return _env("power", play=play, post_metrics=True, **kw)


def _rl_s0b_fixed():
  from r1s1.rl_cfg import s1_rl_cfg
  return s1_rl_cfg(algorithm="r1s1.learning:S1PPO", sigma="fixed")


TASKS["wb_s1_s0b_power_fixed"] = dict(env=_env_fixed, rl=_rl_s0b_fixed, stage="S1-S0b-power-fixed", s1=True)
SPECS["wb_s1_s0b_power_fixed"] = {"action": "wholebody", "tags": "s1-s0b-fixed-sigma-scratch", "runner": "r1wb"}


# R13 positive control (positive 23, coordinator 2026-11-20 09:2x-09:5x; negative 18 addenda 46-47, freezes 84-85).
# DIAGNOSTIC ONLY, NOT ADOPTED: S0b-fixed (wb_s1_s0b_power_fixed: same observations, randomisation, terminations, PPO,
# network and fixed std) with only the reward block swapped for the careful_walk stage-1 velocity-tracking recipe as
# recorded by its walking run; the velocity command is derived from the S1 goal (r1s1.posctl), never observed.
def _env_posctl(play=False, **kw):
  from r1s1.posctl import posctl_env_cfg
  return posctl_env_cfg(play=play, **kw)


TASKS["wb_s1_posctl_walk"] = dict(env=_env_posctl, rl=_rl_s0b_fixed, stage="S1-posctl-walk", s1=True)
SPECS["wb_s1_posctl_walk"] = {"action": "wholebody", "tags": "s1-posctl-velwalk-diagnostic-scratch", "runner": "r1wb"}


# R13 common-base candidate (positive 23, coordinator 2026-11-20 10:1x): DIAGNOSTIC, not adopted. S0b-fixed with objective
# v2 unchanged plus one dense tracking term per moving goal kind (planar velocity and turn rate for locomotion goals, pelvis
# height rate for crouch goals) at the goal spec's fixed rates (r1s1.posctl); no ingredient-table name, no debt, no human data.
def _env_posctl_dense(play=False, **kw):
  from r1s1.posctl import dense_env_cfg
  return dense_env_cfg(play=play, **kw)


TASKS["wb_s1_posctl_dense"] = dict(env=_env_posctl_dense, rl=_rl_s0b_fixed, stage="S1-posctl-dense", s1=True)
SPECS["wb_s1_posctl_dense"] = {"action": "wholebody", "tags": "s1-posctl-dense-walk-crouch-scratch", "runner": "r1wb"}
