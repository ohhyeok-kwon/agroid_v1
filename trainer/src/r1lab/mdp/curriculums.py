"""Survival-based terrain curriculum (slow, careful walkers do not cover distance)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class robustness_ramp:
  """Linear ramps on env steps (iterations x num_steps_per_env):
    * penalty reward weights: start_frac -> 1.0 of their configured value over penalty_steps;
    * disturbance events (push velocity, impulse force/torque): 0 -> full over disturb_steps.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.penalties = {}
    for name in p["penalties"]:
      term = env.reward_manager.get_term_cfg(name)
      self.penalties[name] = (term, float(term.weight))
    self.events = {}
    for name in p["events"]:
      try:
        term = env.event_manager.get_term_cfg(name)
      except (KeyError, ValueError):
        continue
      keys = [k for k in ("velocity_range", "force_range", "torque_range", "position_range")
              if k in term.params and term.params[k]]
      self.events[name] = (term, {k: _copy(term.params[k]) for k in keys})
    self._last = None

  def __call__(self, env: ManagerBasedRlEnv, env_ids, penalties, events, start_frac=0.2,
               penalty_steps=36_000, disturb_steps=28_800, penalty_delay=0, disturb_delay=0):
    del env_ids, penalties, events
    t = env.common_step_counter
    a = min(1.0, max(0.0, (t - penalty_delay) / max(1, penalty_steps)))
    b = min(1.0, max(0.0, (t - disturb_delay) / max(1, disturb_steps)))
    key = (round(a, 3), round(b, 3))
    if key != self._last:
      for term, w in self.penalties.values():
        term.weight = w * (start_frac + (1.0 - start_frac) * a)
      for term, base in self.events.values():
        for k, v in base.items():
          term.params[k] = _scale(v, b)
      self._last = key
    return {"penalty_scale": torch.tensor(start_frac + (1 - start_frac) * a),
            "disturb_scale": torch.tensor(b)}


def _copy(v):
  return {k: tuple(x) for k, x in v.items()} if isinstance(v, dict) else tuple(v)


def _scale(v, s):
  if isinstance(v, dict):
    return {k: (x[0] * s, x[1] * s) for k, x in v.items()}
  return (v[0] * s, v[1] * s)


def terrain_levels_survival(env: ManagerBasedRlEnv, env_ids: torch.Tensor,
                            command_name: str = "twist", track_tol: float = 0.25):
  """Promote envs that survived the episode while tracking; demote envs that fell."""
  terrain = env.scene.terrain
  if terrain is None or terrain.cfg.terrain_generator is None or len(env_ids) == 0:
    return {}
  fell = env.termination_manager.terminated[env_ids]
  timed_out = env.termination_manager.time_outs[env_ids]
  term = env.command_manager.get_term(command_name)
  err = term.metrics.get("error_vel_xy")
  good = torch.ones_like(fell) if err is None else err[env_ids] < track_tol
  move_up = timed_out & ~fell & good
  move_down = fell
  if env.common_step_counter == 0:
    move_up = torch.zeros_like(move_up)
    move_down = torch.zeros_like(move_down)
  terrain.update_env_origins(env_ids, move_up, move_down)
  levels = terrain.terrain_levels.float()
  return {"mean": levels.mean(), "max": levels.max()}
