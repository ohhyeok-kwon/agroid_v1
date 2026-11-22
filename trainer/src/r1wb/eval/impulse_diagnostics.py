"""Opt-in impulse-probe interventions. Evaluation only; no training/deployment imports.

These are causal diagnostics of one action direction, not rules for choosing when an arm may move.
Copying a twin target is not a complete simulator/RNG state branch, especially after gait phases diverge.
"""
from __future__ import annotations

import math

import numpy as np
import torch


def validate_options(args) -> bool:
  enabled = args.goal_lag_grace is not None or args.arm_response_scale is not None
  if not enabled:
    return False
  if args.task not in ("stream_label", "clip_label"):
    raise ValueError("lag/arm diagnostics support teacher label policies only; manufacturer is excluded")
  if args.trace is None or not args.sync_twins or not args.tag:
    raise ValueError("lag/arm diagnostics require --trace, --sync-twins and a unique --tag")
  if args.world or args.throw:
    raise ValueError("lag/arm diagnostics currently require flat ground and directly applied forces, not throws")
  if (args.quiet != "none" or args.course or args.quiet_hits or args.graft or args.rate_feedback
      or args.rate_feedback_hip or args.leg_boost != 1.0 or args.reflex_gain or args.plan_lean):
    raise ValueError("lag/arm diagnostics cannot be combined with other action/course interventions")
  alpha = args.arm_response_scale
  if alpha is not None and (not math.isfinite(alpha) or not 0.0 <= alpha <= 1.0):
    raise ValueError("--arm-response-scale must be finite and in [0, 1]")
  return True


class AppliedImpulseClock:
  """Age of the most recently applied force interval, not a future hit reservation.

  Caller writes the force first, then calls applied() before physics. No actor is run in between; the command
  update after physics sees age zero for this interval. The next policy sees that completed interval's age.
  """
  def __init__(self, count, device, dt):
    self.dt = float(dt)
    self.last = torch.full((count,), -1, dtype=torch.long, device=device)
    self.current = torch.full((count,), 1000., device=device)

  def applied(self, step, active):
    self.last[active] = step
    self.current = torch.where(self.last >= 0, (step - self.last).float() * self.dt,
                               torch.full_like(self.current, 1000.))
    return self.current

  def reset(self, done):
    self.last[done] = -1
    self.current[done] = 1000.


def gate_s2_lag(previous, updated, clock, grace, dt, time_constant):
  """Keep the existing update exactly at grace=0; during grace retain leakage but omit new error."""
  if grace == 0.0:
    return updated
  return torch.where((clock < grace).unsqueeze(1), previous * math.exp(-dt / time_constant), updated)


class TargetIntervention:
  """Wrap only this evaluator's action term, after residual/intent mapping and before apply_actions()."""
  def __init__(self, term, twin_of, hit_step, alpha, trace):
    self.term, self.twin_of, self.hit_step, self.alpha, self.trace = term, twin_of, hit_step, alpha, trace
    self.step = None
    self.original = term.process_actions
    self.original_apply = term.apply_actions
    self.ids = None
    self.desired = None
    term.process_actions = self.process
    term.apply_actions = self.apply

  def process(self, actions):
    self.original(actions)
    if self.step is None:
      return
    processed = self.term._processed_actions
    bias = self.term._entity.data.encoder_bias[:, self.term.target_ids]
    target = processed - bias
    self.trace["pd_target_original"][self.step] = target
    self.ids = self.desired = None
    # alpha=1 does not even write the original target buffer (avoids interpolation rounding and version changes).
    if self.alpha is not None and self.alpha != 1.0:
      ids = (self.step >= self.hit_step).nonzero(as_tuple=False).flatten()
      if len(ids):
        twin = target[self.twin_of[ids], 14:24]
        replacement = twin if self.alpha == 0.0 else twin + self.alpha * (target[ids, 14:24] - twin)
        # Keep the term's previous plain-action representation consistent with the target intervention.
        processed[ids, 14:24] = replacement + bias[ids, 14:24]
        self.ids, self.desired = ids, replacement

  def apply(self):
    self.original_apply()
    if self.step is None:
      return
    if self.ids is not None:
      # Applying the exact value after bias subtraction avoids add/subtract roundoff at alpha=0.
      self.term._entity.set_joint_position_target(self.desired, joint_ids=self.term.target_ids[14:24], env_ids=self.ids)
    self.trace["pd_target_applied"][self.step] = self.term._entity.data.joint_pos_target[:, self.term.target_ids]

  def close(self):
    self.term.process_actions = self.original
    self.term.apply_actions = self.original_apply


def pre_hit_qa(raw, hit_step, twin_of, origins):
  """Pre-physics samples through the onset step, including the state immediately before its force.

  World-root offsets are removed from qpos for the meaningful comparison; raw world differences are also retained.
  These differences diagnose synchronization, not equality of hidden state or future random draws.
  """
  qa = []
  for index, (hit, twin) in enumerate(zip(hit_step, twin_of, strict=True)):
    stop = min(int(hit) + 1, len(raw["pre_qpos"]))
    row = {"samples": stop, "includes_pre_physics_onset": True}
    for key in ("pre_qpos", "pre_qvel", "actor_input", "pd_target_original", "pd_target_applied"):
      a, b = raw[key][:stop, index], raw[key][:stop, int(twin)]
      row[key + "_max_abs"] = float(np.abs(a - b).max()) if stop else None
    a = raw["pre_qpos"][:stop, index].copy()
    b = raw["pre_qpos"][:stop, int(twin)].copy()
    a[:, :3] -= origins[index]
    b[:, :3] -= origins[int(twin)]
    row["qpos_env_relative_max_abs"] = float(np.abs(a - b).max()) if stop else None
    qa.append(row)
  return qa
