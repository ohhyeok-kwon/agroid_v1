"""Reject privileged post-hit exemptions before a whole-body training environment starts."""
from __future__ import annotations

import inspect
import math

_ABSENT = object()


def _exact_zero(value, location, errors):
  try:
    valid = math.isfinite(float(value)) and float(value) == 0.0
  except (TypeError, ValueError, OverflowError):
    valid = False
  if not valid:
    errors.append(f"{location} must be finite zero, got {value!r}")


def _default(function, name):
  if inspect.isclass(function):
    function = function.__call__
  try:
    parameter = inspect.signature(function).parameters.get(name)
  except (TypeError, ValueError):
    return _ABSENT
  if parameter is None:
    return _ABSENT
  return None if parameter.default is inspect.Parameter.empty else parameter.default


def require_no_hit_exemptions(cfg):
  """Check active configuration, including defaults hidden behind stateful reward callables.

  The disturbance scheduler may retain its event clock for timing/logging. That
  clock must not exempt terminations, forgive command error, suspend prices, or
  switch reference selection. Legacy positive-grace helpers remain usable by
  explicit historical diagnostics, but this training entry point rejects them.
  """
  errors, checked = [], []
  for name, command in cfg.commands.items():
    for field in ("lag_grace", "hold_after_hit", "forgive", "relief_window"):
      if hasattr(command, field):
        location = f"commands.{name}.{field}"
        _exact_zero(getattr(command, field), location, errors)
        checked.append(location)
  for section in ("rewards", "terminations", "observations"):
    groups = getattr(cfg, section, {})
    if section == "observations":
      entries = [(f"{group}.{name}", term) for group, config in groups.items()
                 for name, term in config.terms.items() if term is not None]
    else:
      entries = [(name, term) for name, term in groups.items() if term is not None
                 and (section != "rewards" or term.weight != 0.0)]
    for name, term in entries:
      function = term.func
      module = getattr(function, "__module__", "")
      function_name = getattr(function, "__name__", type(function).__name__)
      params = getattr(term, "params", {})
      # These parameters are post-hit exemptions in r1wb, never episode ages.
      for field in ("grace", "lag_grace", "hold_after_hit", "forgive", "step_calm"):
        value = params.get(field, _default(function, field) if module.startswith("r1wb.") else _ABSENT)
        if value is not _ABSENT:
          location = f"{section}.{name}.{field}"
          _exact_zero(value, location, errors)
          checked.append(location)
      if module == "r1wb.tracking.mdp" and function_name == "step_unprompted":
        location = f"{section}.{name}.calm"
        _exact_zero(params.get("calm", _default(function, "calm")), location, errors)
        checked.append(location)
      if params.get("stream") in ("calm", "held"):
        errors.append(f"{section}.{name}.stream={params['stream']!r} is a legacy post-hit gate")
      if module == "r1wb.harden" and section == "observations":
        errors.append(f"{section}.{name} uses the legacy hit-window observation")
  if errors:
    raise ValueError("Post-hit exemptions are forbidden in training: " + "; ".join(errors))
  return {"version": "no-post-hit-exemptions-v1", "passed": True, "checked": checked,
          "disturbance_clock": "scheduler/diagnostic only", "exact_hit_frame_exempt": False}
