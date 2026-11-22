"""Evaluation-only standardized delivered commands; no training or app imports this adapter."""
from __future__ import annotations

import math

from r1wb.eval.command_contract import NativeCommand


class CommonCommand:
  """Advance once before action and expose the same native-filtered tensor to both systems.

  The h12 reference planner still has its intrinsic code latency. Only its explicit
  velocity goal/history/lag use this standardized command, without a second slew.
  """
  def __init__(self, initial, dt=0.02):
    self.native = NativeCommand(initial, dt)
    self.raw = initial.clone()
    self.delivered = initial.clone()
    self.tick = -1

  def advance(self, tick, raw):
    if tick != self.tick + 1:
      raise ValueError("common command must advance exactly once per control tick")
    self.raw.copy_(raw)
    self.delivered.copy_(self.native.step(raw))
    self.tick = tick
    return self.delivered

  def goal(self, template):
    result = template.clone()
    result[:, :3] = self.delivered
    return result

  def update_view(self, view):
    """S2View.update with its velocity slew replaced by this already filtered tape.

    Keep reference decisions, masks, height/pitch slew and history clock unchanged.
    This explicit version is sealed in evaluation provenance and never changes
    r1wb.s2.runtime.S2View or the training observation contract.
    """
    import torch
    from r1wb.s2.small import GRID_ORIGIN
    if self.tick != view._calls + 1:
      raise ValueError("S2 reference clock differs from common command clock")
    env, term = view._env, view._term
    view._calls += 1
    restarted = (env.episode_length_buf <= 1).nonzero(as_tuple=False).flatten()
    if len(restarted) and view._calls > 1:
      view._start(restarted)
    if view._calls >= GRID_ORIGIN and (view._calls - GRID_ORIGIN) % view._period == 0:
      previous_mask = getattr(view, "_waiting_mask", None)
      view._decide()
      if previous_mask is not None:
        view._mask[:, 1] = previous_mask
    dt = env.step_dt
    step = view._slew * dt
    view._stream[:, 3:] += torch.maximum(
      torch.minimum(view._target[:, 3:] - view._stream[:, 3:], step[3:]), -step[3:])
    view._stream[:, :3] = self.delivered
    view._twists = torch.roll(view._twists, 1, dims=1)
    view._twists[:, 0] = self.delivered
    error = term.own_twist() - view._twists[:, -1]
    view._lag = view._lag * math.exp(-dt / term.cfg.lag_time_constant) + dt * error

  @staticmethod
  def metadata():
    return dict(name="proper-r1-v1", velocity_filter="NativeCommand exactly once before action",
                s1_explicit_goal="same delivered tensor; second S2 velocity slew bypassed",
                reference_latency="h12 code/reference latency retained; standardized system comparison",
                score_target="unfiltered requested physical command")
