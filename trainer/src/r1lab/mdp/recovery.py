"""Recovery gates and physical impulse bookkeeping for R1 policies.

Settling-cost gating is inspired by Safe-Stop (arXiv:2609.02358); its thresholds
are R1 experiment settings, not a reproduction or a stability guarantee.
The burst event is an impulse-matched surrogate for measured ball contacts.
"""
from __future__ import annotations

from dataclasses import replace
import inspect
import math

import torch

from mjlab.envs import mdp as envs_mdp
from mjlab.utils.lab_api.math import quat_apply


def _gate_state(env):
  if not hasattr(env, "_r1_recovery_gate"):
    zeros = torch.zeros(env.num_envs, device=env.device)
    env._r1_recovery_gate = dict(step=-1, level=zeros.clone(), remaining=zeros.clone(),
                                 settled=zeros.clone(), ready=torch.ones_like(zeros, dtype=torch.bool),
                                 external_until=torch.full_like(zeros, -1, dtype=torch.long),
                                 last_advanced_step=torch.full_like(zeros, -1, dtype=torch.long),
                                 last_evaluated_step=torch.full_like(zeros, -1, dtype=torch.long))
  return env._r1_recovery_gate


def mark_disturbance(env, env_ids=None):
  """Training reward bookkeeping only; never part of the actor observation."""
  state = _gate_state(env)
  ids = slice(None) if env_ids is None else env_ids
  state["external_until"][ids] = env.common_step_counter + round(1.5 / env.step_dt)
  state["last_evaluated_step"][ids] = -1
  state["step"] = -1


class TrackedImpulse(envs_mdp.apply_body_impulse):
  def __call__(self, env, env_ids, **params):
    super().__call__(env, env_ids, **params)
    mark_disturbance(env, self._active.nonzero(as_tuple=False).flatten())

  def reset(self, env_ids=None):
    ids = torch.arange(self._num_envs, device=self._device) if env_ids is None else env_ids
    if isinstance(ids, slice):
      ids = torch.arange(self._num_envs, device=self._device)[ids]
    self._active[ids] = False
    self._time_remaining[ids] = 0
    self._interval_time_left[ids] = self._sample_cooldown(len(ids))


def tracked_velocity_push(env, env_ids, **params):
  envs_mdp.push_by_setting_velocity(env, env_ids, **params)
  mark_disturbance(env, env_ids)


def recovery_level(env, command_name="twist"):
  """Advance each environment once per policy tick; marks may refresh risk only."""
  projectiles = getattr(env, "_r1_projectile_event", None)
  if projectiles is not None:
    projectiles.observe_contacts()
  state = _gate_state(env)
  # The scalar key is only a read-cache shortcut; elapsed time is per environment.
  if state["step"] == env.common_step_counter:
    return state["level"]
  advance = state["last_advanced_step"] != env.common_step_counter
  evaluate = state["last_evaluated_step"] != env.common_step_counter
  robot = env.scene["robot"].data
  command = env.command_manager.get_command(command_name)
  tilt = torch.linalg.vector_norm(robot.projected_gravity_b[:, :2], dim=-1) / math.sin(math.radians(8))
  spin = torch.linalg.vector_norm(robot.root_link_ang_vel_b[:, :2], dim=-1) / 1.
  velocity_error = torch.linalg.vector_norm(robot.root_link_lin_vel_b[:, :2] - command[:, :2], dim=-1) / .4
  risk = torch.maximum(torch.maximum(tilt, spin), velocity_error)
  recent = env.common_step_counter <= state["external_until"]
  dt = env.step_dt
  state["remaining"] = torch.where(advance, (state["remaining"] - dt).clamp_min(0), state["remaining"])
  settled = torch.where(risk < .5, state["settled"] + dt, 0.)
  state["settled"] = torch.where(advance, settled, state["settled"])
  state["ready"] |= advance & (state["settled"] >= .5)
  enter = (risk > 1.) & state["ready"] & recent & evaluate
  state["remaining"][enter] = 2.
  state["ready"][enter] = False
  # A permanently tilted policy cannot keep a permanent quiet-cost discount.
  target = ((risk - .5) / .5).clamp(0, 1) * (state["remaining"] > 0)
  decayed = torch.where(advance, state["level"] * math.exp(-dt / .12), state["level"])
  state["level"] = torch.where(evaluate, torch.maximum(target, decayed) * recent, state["level"])
  state["last_advanced_step"][advance] = env.common_step_counter
  state["last_evaluated_step"][evaluate] = env.common_step_counter
  state["step"] = env.common_step_counter
  env.extras["log"]["Metrics/recovery_fraction"] = (state["level"] > .5).float().mean()
  return state["level"]


def reset_recovery(env, env_ids=None):
  state = _gate_state(env)
  ids = slice(None) if env_ids is None else env_ids
  for key in ("level", "remaining", "settled"):
    state[key][ids] = 0
  state["ready"][ids] = True
  state["external_until"][ids] = -1
  # A reset state has just been created, so no time has elapsed in this tick.
  # Neighbours retain both their clock and their cached risk evaluation.
  state["last_advanced_step"][ids] = env.common_step_counter
  state["last_evaluated_step"][ids] = -1
  state["step"] = -1


class RecoveryScaledTerm:
  """Wrap a reward term while preserving its parameters and reset lifecycle."""
  def __init__(self, cfg, env):
    self.env = env
    if not 0. <= cfg.params.get("min_scale", .2) <= 1.:
      raise ValueError("Recovery reward scale must be in [0, 1].")
    params = {key: value for key, value in cfg.params.items()
              if key not in ("base_func", "min_scale", "bounded_reward")}
    func = cfg.params["base_func"]
    self.term = func(replace(cfg, func=func, params=params), env) if inspect.isclass(func) else func

  def reset(self, env_ids=None):
    reset_recovery(self.env, env_ids)
    if hasattr(self.term, "reset"):
      self.term.reset(env_ids)

  def __call__(self, env, base_func, min_scale=.2, bounded_reward=False, **params):
    del base_func
    value = self.term(env, **params)
    scale = 1. - (1. - min_scale) * recovery_level(env)
    # For [0, 1] bonuses soften the deviation cost (1-value), preserving the
    # same maximum reward. Multiplying the bonus itself would punish recovery.
    return 1. - scale * (1. - value) if bounded_reward else scale * value


class ImpulseBurst:
  """Three-tick pulse: a leading impulse followed by a two-tick tail.

  Horizontal impulse is specified in N*s, independently of the force amplitude.
  Repeated impulses share a direction and are separated by 0.3--0.5 seconds.
  This does not replace held-out CPU MuJoCo projectile-contact evaluation.
  """
  def __init__(self, cfg, env):
    self.env = env
    self.asset = env.scene[cfg.params["asset_cfg"].name]
    self.body_ids = cfg.params["asset_cfg"].body_ids
    self.num_bodies = len(self.body_ids)
    self.clean_fraction = cfg.params.get("clean_fraction", .4)
    if not 0. <= self.clean_fraction <= 1.:
      raise ValueError("clean_fraction must be in [0, 1].")
    self.wait = torch.zeros(env.num_envs, device=env.device)
    self.phase = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
    self.left = torch.zeros_like(self.phase)
    self.body = torch.zeros_like(self.phase)
    self.impulse = torch.zeros(env.num_envs, 3, device=env.device)
    self.interval = torch.zeros_like(self.wait)
    self.leading_fraction = torch.full_like(self.wait, .75)
    self.offset = torch.zeros_like(self.impulse)
    self.clean = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.origin = None
    self.reset()

  def reset(self, env_ids=None):
    ids = torch.arange(self.env.num_envs, device=self.env.device) if env_ids is None else env_ids
    if isinstance(ids, slice):
      ids = torch.arange(self.env.num_envs, device=self.env.device)[ids]
    n = len(ids)
    self.wait[ids] = 2. + 2. * torch.rand(n, device=self.env.device)
    self.phase[ids] = -1
    self.left[ids] = 0
    self.impulse[ids] = 0
    self.clean[ids] = torch.rand(n, device=self.env.device) < self.clean_fraction
    reset_recovery(self.env, ids)
    zeros = torch.zeros(n, self.num_bodies, 3, device=self.env.device)
    self.asset.write_external_wrench_to_sim(zeros, zeros, env_ids=ids, body_ids=self.body_ids)

  def __call__(self, env, env_ids, asset_cfg, clean_fraction=.4, warmup_steps=7200,
               impulse_range=(3., 12.), interval_range=(.3, .5), max_shots=3,
               vertical_fraction=(0., .25), body_point_offset=(0., 0., 0.),
               leading_fraction_range=(.6, .9), point_offset_range=(.04, .04, .10)):
    del env_ids, asset_cfg, clean_fraction
    dt = env.step_dt
    if abs(dt - .02) > 1e-8:
      raise ValueError("ImpulseBurst requires the 50 Hz policy clock.")
    if self.origin is None:
      self.origin = env.common_step_counter
    fraction = min(1., max(0., (env.common_step_counter - self.origin) / max(1, warmup_steps)))
    # First expose single lower-magnitude pulses, then double and triple bursts.
    shot_limit = min(max_shots, 1 + int(fraction * max_shots))
    impulse_max = impulse_range[0] + (impulse_range[1] - impulse_range[0]) * (.3 + .7 * fraction)
    idle = self.phase < 0
    self.wait[idle] -= dt
    finished = self.phase >= 3
    self.phase[finished] = -1
    gap = torch.where(self.left > 0, self.interval - 3 * dt,
                      2. + 2. * torch.rand(env.num_envs, device=env.device))
    self.wait[finished] = gap[finished]
    start = (self.phase < 0) & (self.wait <= 1e-6) & ~self.clean
    new_burst = start & (self.left == 0)
    ids = new_burst.nonzero(as_tuple=False).flatten()
    n = len(ids)
    if n:
      self.left[ids] = torch.randint(1, shot_limit + 1, (n,), device=env.device)
      self.body[ids] = torch.randint(0, self.num_bodies, (n,), device=env.device)
      self.interval[ids] = interval_range[0] + (interval_range[1] - interval_range[0]) * torch.rand(n, device=env.device)
      self.leading_fraction[ids] = (leading_fraction_range[0] +
          (leading_fraction_range[1] - leading_fraction_range[0]) * torch.rand(n, device=env.device))
      extent = torch.tensor(point_offset_range, device=env.device)
      self.offset[ids] = (2 * torch.rand(n, 3, device=env.device) - 1) * extent
      angle = torch.rand(n, device=env.device) * (2 * math.pi)
      magnitude = impulse_range[0] + (impulse_max - impulse_range[0]) * torch.rand(n, device=env.device)
      self.impulse[ids, 0] = magnitude * torch.cos(angle)
      self.impulse[ids, 1] = magnitude * torch.sin(angle)
      down = vertical_fraction[0] + (vertical_fraction[1] - vertical_fraction[0]) * torch.rand(n, device=env.device)
      self.impulse[ids, 2] = -magnitude * down * fraction
    self.phase[start] = 0
    self.left[start] -= 1
    active = self.phase >= 0
    pulse = torch.where(self.phase == 0, self.leading_fraction, (1 - self.leading_fraction) / 2) * active / dt
    forces = torch.zeros(env.num_envs, self.num_bodies, 3, device=env.device)
    batch = torch.arange(env.num_envs, device=env.device)
    forces[batch, self.body] = self.impulse * pulse[:, None]
    quat = self.asset.data.body_com_quat_w[:, self.body_ids]
    offset = (self.offset[:, None, :] + torch.tensor(body_point_offset, device=env.device)).expand_as(forces)
    offset = offset.reshape(-1, 3)
    offset_w = quat_apply(quat.reshape(-1, 4), offset).reshape_as(forces)
    torques = torch.cross(offset_w, forces, dim=-1)
    self.asset.write_external_wrench_to_sim(forces, torques, body_ids=self.body_ids)
    mark_disturbance(env, active.nonzero(as_tuple=False).flatten())
    self.phase[active] += 1
    env.extras["log"]["Metrics/impulse_active_fraction"] = active.float().mean()
    env.extras["log"]["Metrics/horizontal_impulse_max_ns"] = torch.tensor(impulse_max, device=env.device)
    env.extras["log"]["Metrics/impulse_shot_limit"] = torch.tensor(float(shot_limit), device=env.device)
