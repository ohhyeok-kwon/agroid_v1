"""Positive session 18 (R11, 2026-11-19): object simulation foundation and the no-learning contact ladder.

  robot_fixed  fixed-base R1 with the O6 box hands (the training body without the floating base joint)
  scene        free objects (box / cylinder / capsule) and tables for mjlab scenes: specs, per-world randomisation,
               reset and observation helpers (pose, velocity, contacts) for later RL
  ik           mink IK for the two-hand squeeze path (hand separation x path position grid)
  ladder       the contact ladder protocol (scripted joint targets tracked by the deploy PD), CPU (C MuJoCo) and MJWarp
  readout      the frozen readout of rounds/R04/positive/r11_objects/rules.md
"""
