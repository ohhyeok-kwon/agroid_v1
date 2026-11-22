"""R13 anchor (positive 23, coordinator 2026-11-20 14:4x KST): does careful_walk's own frame and recipe learn to walk
again in today's code, from random weights? DIAGNOSTIC ONLY, not adopted; no manufacturer policy or model is used
(the robot contract's gains / default pose / action scale are the hardware configuration every task shares).

The task is careful_walk stage 1 ("learn to walk": flat floor, no pushes or hits, quiet / stepping terms 0, wide
commands; 810 inputs with empirical normalisation, std learned from 0.8, adaptive learning rate, AMP style term, gait
terms) exactly as r1lab.tasks.careful_walk builds it today, except the four reward entries that were edited after the
walking run: they are restored to that run's recorded functions, weights and kernel widths
(r1_careful_walk/2026-11-08_05-01-55_s1, params/env.yaml sha256 9812463c...; the run tracked the instantaneous
pelvis velocity, today's code tracks a 0.4 s EMA). Differences that today's code imposes (self-collision is required
by r1lab.train, today's term bodies and command code, today's AMP file, a fresh discriminator, random weights at the
start of stage 1) are listed in rounds/R04/positive/r13_posctl/cwr/README.md, not changed here.
"""
from __future__ import annotations

SOURCE_RUN = "r1_careful_walk/2026-11-08_05-01-55_s1"
SOURCE_ENV_YAML_SHA256 = "9812463c8445c8d2cc3b638104497c2198ff30393cb717c8bc8222b52638acd6"
# name: (module, function, weight, params) as recorded by the walking run's params/env.yaml
RECORDED = {
  "track_lin_vel": ("mjlab.tasks.velocity.mdp.rewards", "track_linear_velocity", 3.0, {"command_name": "twist", "std": 0.15}),
  "track_ang_vel": ("mjlab.tasks.velocity.mdp.rewards", "track_angular_velocity", 1.5, {"command_name": "twist", "std": 0.3}),
  "lin_vel_err": ("r1lab.mdp.rewards", "lin_vel_error_l1", -3.0, {"command_name": "twist"}),
  "yaw_rate_err": ("r1lab.mdp.rewards", "yaw_rate_error_l1", -1.0, {"command_name": "twist"}),
}


def repro_env_cfg(play: bool = False, t0: int = 0):
  """careful_walk stage 1 with the walking run's recorded tracking terms; t0 = this stage's start (the smoothness
  ramp's delay, as the recorded run's t0 900 was its stage start)."""
  import importlib
  from r1lab.tasks import careful_walk
  cfg = careful_walk.careful_walk_env_cfg(play, stage=1, t0=t0)
  for name, (module, function, weight, params) in RECORDED.items():
    term = cfg.rewards[name]
    term.func, term.weight, term.params = getattr(importlib.import_module(module), function), weight, dict(params)
  return cfg
