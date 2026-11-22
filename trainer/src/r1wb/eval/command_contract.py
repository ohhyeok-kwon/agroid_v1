"""Opt-in evaluation command contracts; no joystick remapping or training integration."""
from __future__ import annotations

import math


NATIVE_DT = 0.02
NATIVE_LIMITS = {"forward": 0.9, "reverse": 1.0, "lateral": 0.6, "moving_yaw": 1.0, "standing_yaw": 2.5}
NATIVE_DELTAS = {"forward_up": 0.015, "forward_down": 0.05, "reverse_up": 0.5,
                 "reverse_down": 0.03, "lateral_up": 0.10, "lateral_down": 0.05,
                 "yaw_up": 0.08, "yaw_down": 0.10}


def native_model(velocity):
  """Tensor of required native models: 0=walk, 1=run, 2=turn; turn has priority."""
  import torch
  turn = (velocity[..., 0].abs() <= 0.3) & (velocity[..., 1].abs() <= 0.3) & (velocity[..., 2].abs() >= 1.0)
  return torch.where(turn, 2, torch.where(velocity[..., 0] >= 1.5, 1, 0))


def native_slew(previous, target, dt: float = NATIVE_DT):
  """Physical (vx, vy, wz) -> native low-speed command, exactly once per 20 ms.

  Port of walking_action_manager/command.rs AFTER joystick target(). The native
  two sequential sign tests both apply at zero. Joystick deadzones and axis
  suppression are deliberately absent: these inputs already have physical units.
  """
  import torch
  if not math.isclose(dt, NATIVE_DT, rel_tol=0.0, abs_tol=1e-9):
    raise ValueError("native-r1 command contract requires a 0.02 s control step")
  if previous.shape != target.shape or target.shape[-1] != 3:
    raise ValueError("native commands must have matching (..., 3) shapes")
  out = target.clone()
  out[..., 0].clamp_(-NATIVE_LIMITS["reverse"], NATIVE_LIMITS["forward"])
  out[..., 1].clamp_(-NATIVE_LIMITS["lateral"], NATIVE_LIMITS["lateral"])
  for i, (up, down, reverse_up, reverse_down) in enumerate(((0.015, 0.05, 0.5, 0.03),
                                                          (0.10, 0.05, 0.10, 0.05),
                                                          (0.08, 0.10, 0.08, 0.10))):
    old, value = previous[..., i], out[..., i]
    value = torch.where(old >= 0, torch.maximum(torch.minimum(value, old + up), old - down), value)
    out[..., i] = torch.where(old <= 0, torch.maximum(torch.minimum(value, old + reverse_down), old - reverse_up), value)
  moving = (out[..., 0].abs() > 0.0001) | (out[..., 1].abs() > 0.0001)
  yaw = torch.where(moving, NATIVE_LIMITS["moving_yaw"], NATIVE_LIMITS["standing_yaw"])
  out[..., 2] = torch.maximum(torch.minimum(out[..., 2], yaw), -yaw)
  return out


class NativeCommand:
  """Explicit state for the evaluation-only native slew; never reads simulator state."""
  def __init__(self, initial, dt: float = NATIVE_DT):
    self.velocity = initial.clone()
    self.dt = dt

  def reset(self, ids=None):
    if ids is None:
      self.velocity.zero_()
    else:
      self.velocity[ids] = 0.0

  def step(self, target):
    self.velocity = native_slew(self.velocity, target, self.dt)
    return self.velocity


def comparison_support(script):
  """Support intersection with the currently available manufacturer WALK policy.

  Exclude whole scripts, not only unsupported frames, from comparative summaries.
  Raw outcomes remain available. This is independent of which policy is evaluated.
  """
  reasons = set()
  for row in script:
    _, vx, vy, wz, _, height_on, _, pitch_on = row
    if abs(vx) <= 0.3 and abs(vy) <= 0.3 and abs(wz) >= 1.0:
      reasons.add("manufacturer_turn_model_unavailable")
    if vx >= 1.5:
      reasons.add("manufacturer_run_model_unavailable")
    if height_on:
      reasons.add("manufacturer_height_input_unavailable")
    if pitch_on:
      reasons.add("manufacturer_pitch_input_unavailable")
    if vx > 0.9 or vx < -1.0 or abs(vy) > 0.6 or abs(wz) > (1.0 if vx or vy else 2.5):
      reasons.add("outside_native_low_speed_limits")
  return {"supported": not reasons, "reasons": sorted(reasons)}


def physical_suite(spoken, command_row, *, grid_origin: int = 1, dt: float = NATIVE_DT):
  """One physical suite for every policy, aligned to S2's absolute decision origin.

  S2 word_schedule already offsets its edges by GRID_ORIGIN; only the physical
  rows get that initial offset. No -.01 floating-point workaround is used.
  """
  suite = {name: [command_row(item, seconds) for item, seconds in script] for name, script in spoken.items()}
  for rows in suite.values():
    rows[0][0] += grid_origin * dt
  return suite


class PhysicalSchedule:
  """Integer-tick schedule, sampled BEFORE action; sample k applies on [k*dt,(k+1)*dt)."""
  def __init__(self, scripts, *, device="cpu", dt: float = NATIVE_DT):
    import torch
    longest = max(map(len, scripts))
    rows, edges = [], []
    for script in scripts:
      elapsed, ends = 0, []
      for row in script:
        ticks = round(row[0] / dt)
        if ticks <= 0 or not math.isclose(ticks * dt, row[0], rel_tol=0.0, abs_tol=1e-6):
          raise ValueError("comparison schedule durations must be positive whole control ticks")
        elapsed += ticks
        ends.append(elapsed)
      rows.append([list(row) for row in script] + [list(script[-1])] * (longest - len(script)))
      # Last segment stays in force, including when scripts have different lengths.
      ends[-1] = 2**60
      edges.append(ends + [2**60] * (longest - len(script)))
    self.rows = torch.tensor(rows, dtype=torch.float32, device=device)
    self.ends = torch.tensor(edges, dtype=torch.long, device=device)
    self.begins = torch.cat([torch.zeros_like(self.ends[:, :1]), self.ends[:, :-1]], dim=1)
    self.dt = dt
    self.envs = torch.arange(len(scripts), device=device)

  def sample(self, step: int):
    index = (step >= self.ends).sum(dim=1).clamp_max(self.rows.shape[1] - 1)
    row = self.rows[self.envs, index]
    age = (step - self.begins[self.envs, index]).float() * self.dt
    left = (self.ends[self.envs, index] - step).float() * self.dt
    return row, age, left
