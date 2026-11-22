"""Opt-in, bounded audit of values actually evaluated by an existing RewardManager.

The manager's compute/reset implementations remain authoritative. A callable proxy
observes each returned raw tensor, without calling a term a second time. Statistics
and a small fixed trace are buffered on the original device; only flushes copy to
CPU. Nonfinite classifications retain *all* environment IDs, not just trace IDs.

This is diagnostic instrumentation, not a bitwise GPU-equivalence claim. In
particular, current applied-force buffers at reward time are not a substep force
integral, and manager-derived positions can be one physics substep old.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import hashlib
import inspect
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping
import zlib

import torch


STAT_FIELDS = ("finite_count", "finite_sum", "finite_mean", "finite_min", "finite_max",
               "finite_abs_sum", "nan_count", "posinf_count", "neginf_count")


@dataclass(frozen=True)
class PureVectorDiagnostic:
  """Caller-attested pure callback, once after native rewards and before reset.

  It must return an (environment, column) tensor on the manager's device. A
  caller may return an already computed read-only feature cache. This API never
  calls a reward function again and never substitutes values into the reward.
  """
  callback: Callable[[], torch.Tensor]
  columns: tuple[str, ...]
  compare_max: tuple[str, str] | None = None
  valid_column: str | None = None


@dataclass(frozen=True)
class LedgerOptions:
  sample_env_ids: tuple[int, ...] | None = None
  flush_every: int = 32
  max_steps: int = 4096
  max_pending_bytes: int = 32 * 1024**2
  max_file_bytes: int = 64 * 1024**2
  enabled: bool = True
  pure_diagnostics: Mapping[str, Callable[[], torch.Tensor]] | None = None
  pure_vector_diagnostics: Mapping[str, PureVectorDiagnostic] | None = None
  compression: str = "none"
  compact: bool = False


def _json_safe(value):
  if isinstance(value, float) and not math.isfinite(value):
    return "NaN" if math.isnan(value) else ("+Inf" if value > 0 else "-Inf")
  if isinstance(value, dict):
    return {str(k): _json_safe(v) for k, v in value.items()}
  if isinstance(value, (tuple, list)):
    return [_json_safe(v) for v in value]
  return value


def _stats(value: torch.Tensor) -> torch.Tensor:
  value = value.detach().reshape(-1).to(torch.float64)
  finite = torch.isfinite(value)
  count = finite.sum()
  clean = torch.where(finite, value, 0.)
  nan = torch.full((), float("nan"), device=value.device, dtype=torch.float64)
  if value.numel() == 0:
    return torch.stack((count, clean.sum(), nan, nan, nan, clean.abs().sum(),
                        torch.isnan(value).sum(), torch.isposinf(value).sum(), torch.isneginf(value).sum()))
  return torch.stack((count, clean.sum(), torch.where(count > 0, clean.sum() / count.clamp_min(1), nan),
                      torch.where(count > 0, torch.where(finite, value, float("inf")).amin(), nan),
                      torch.where(count > 0, torch.where(finite, value, -float("inf")).amax(), nan),
                      clean.abs().sum(), torch.isnan(value).sum(), torch.isposinf(value).sum(),
                      torch.isneginf(value).sum()))


def _bad_codes(value: torch.Tensor) -> torch.Tensor:
  value = value.detach()
  return (torch.isnan(value).to(torch.uint8) + 2 * torch.isposinf(value).to(torch.uint8)
          + 3 * torch.isneginf(value).to(torch.uint8))


class _ObservedTerm:
  def __init__(self, ledger, index, original):
    self.ledger, self.index, self.original = ledger, index, original

  def __call__(self, *args, **kwargs):
    value = self.original(*args, **kwargs)
    if self.ledger.active and self.ledger._current is not None:
      self.ledger._attempt(lambda: self.ledger._capture(self.index, value), "capture_raw")
    return value

  def __getattr__(self, name):
    # Stateful term reset/debug_vis and any existing public attributes are retained.
    return getattr(self.original, name)


class RewardLedger:
  def __init__(self, env, output_path: str | Path, options: LedgerOptions):
    self.env, self.manager, self.options = env, env.reward_manager, options
    manager = self.manager
    if getattr(manager, "_r1wb_reward_ledger", None) is not None:
      raise ValueError("a reward ledger is already attached to this manager")
    if options.flush_every < 1 or options.max_steps < 1 or options.max_pending_bytes < 1 or options.max_file_bytes < 8192:
      raise ValueError("ledger bounds must be positive; max_file_bytes must be at least8192")
    if options.compression not in ("none", "gzip"):
      raise ValueError("compression must be 'none' or 'gzip'")
    self.n = int(env.num_envs)
    ids = tuple(range(min(4, self.n))) if options.sample_env_ids is None else tuple(options.sample_env_ids)
    if len(ids) > 16 or len(set(ids)) != len(ids) or any(type(i) is not int or i < 0 or i >= self.n for i in ids):
      raise ValueError("sample_env_ids must contain at most16 distinct valid integer IDs")
    self.ids, self.names = ids, tuple(manager._term_names)
    self.pure_diagnostics = dict(options.pure_diagnostics or {})
    if any(name not in self.names or not callable(callback) for name, callback in self.pure_diagnostics.items()):
      raise ValueError("pure_diagnostics must map registered reward names to explicit pure callables")
    self.pure_vector_diagnostics = dict(options.pure_vector_diagnostics or {})
    for name, spec in self.pure_vector_diagnostics.items():
      if (name not in self.names or not isinstance(spec, PureVectorDiagnostic) or not callable(spec.callback)
          or not spec.columns or len(set(spec.columns)) != len(spec.columns)
          or any(not isinstance(c, str) or not c for c in spec.columns)
          or (spec.valid_column is not None and spec.valid_column not in spec.columns)
          or (spec.compare_max is not None and (len(spec.compare_max) != 2
              or any(c not in spec.columns for c in spec.compare_max)))):
        raise ValueError("pure_vector_diagnostics require registered names and valid named-column specifications")
    self.device = manager._reward_buf.device
    self._indices = torch.tensor(ids, dtype=torch.long, device=self.device)
    self._episodes = torch.zeros(self.n, dtype=torch.long, device=self.device)
    self._reset_counts = torch.zeros(self.n, dtype=torch.long, device=self.device)
    self._force_history, self._force_layouts = {}, {}
    self._field_schema, self._previous_metadata, self._previous_terms = {}, {}, {}
    self._previous_weights = None
    self._current, self._critic = None, None
    self._pending, self._pending_bytes = [], 0
    self.steps_seen, self.steps_written, self.bytes_written = 0, 0, 0
    self._quality = {"sum_exact_frames": 0, "force_available_frames": 0, "critic_supplied_frames": 0,
                     "vector_frames": {name: 0 for name in self.pure_vector_diagnostics},
                     "nonfinite_classifications": 0}
    self.active, self.closed, self.stop_reason = True, False, None
    self.errors: list[dict[str, str]] = []
    self._compute_original, self._reset_original = manager.compute, manager.reset
    self._compute_instance = manager.__dict__.get("compute")
    self._reset_instance = manager.__dict__.get("reset")
    self._compute_hook, self._reset_hook = self._compute, self._reset
    self._proxies = [_ObservedTerm(self, i, cfg.func) for i, cfg in enumerate(manager._term_cfgs)]
    self.path = Path(output_path)
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self._file = self.path.open("xb")
    self._compressor = zlib.compressobj(level=6, wbits=31) if options.compression == "gzip" else None
    source = inspect.getsourcefile(type(manager))
    termination = getattr(env, "termination_manager", None)
    self._termination_time_out_flags = {
      name: bool(cfg.time_out) for name, cfg in zip(getattr(termination, "_term_names", ()),
                                                   getattr(termination, "_term_cfgs", ()))}
    header = {"type": "header", "schema": "r1wb-reward-ledger-v2" if options.compact else "r1wb-reward-ledger-v1", "created_unix": time.time(),
              "num_envs": self.n, "trace_env_ids": ids,
              "options": {f.name: getattr(options, f.name) for f in fields(options)
                          if f.name not in ("pure_diagnostics", "pure_vector_diagnostics")}, "terms": self.names,
              "manager_source": source,
              "manager_source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None,
              "scale_by_dt": bool(manager._scale_by_dt), "stat_fields": STAT_FIELDS,
              "nonfinite_codes": {"1": "NaN", "2": "+Inf", "3": "-Inf"},
              "timing": "reward_compute_entry: post_physics/pre_event/pre_reset; derived quantities may lag one substep",
              "force_scope": "current xfrc_applied/qfrc_applied buffers, not per-substep integrated impulse",
              "force_events": {"new_force": "known finite inactive->active on consecutive samples in same episode",
                               "changed_active": "known active->active with any component changed; not necessarily a new scheduled hit",
                               "left_censored_active": "active without a known contiguous finite predecessor; never counted as a known onset",
                               "continuous_frames": "consecutive observed active samples in the current episode, including a possibly censored first sample",
                               "episode_active_frames": "observed active reward frames since attachment/reset, not a substep exposure integral",
                               "union": "any applied-force source; source switches during active union are not new union onsets"},
              "zero_weight_raw": "unknown/not_evaluated; never separately execute zero-weight terms",
              "pure_diagnostics": list(self.pure_diagnostics),
              "diagnostic_contract": "caller-attested pure/noRNG/read-only callbacks; only when the same registered reward has weight0; values are not native term evaluations",
              "pure_vector_diagnostics": {name: {"columns": s.columns, "compare_max": s.compare_max,
                                                  "valid_column": s.valid_column}
                                          for name, s in self.pure_vector_diagnostics.items()},
              "vector_diagnostic_contract": "caller-attested pure/noRNG/read-only; once after native compute, pre-reset; native reward is never re-evaluated",
              "max_winner_fields": ("first", "second", "tie", "invalid"),
              "termination_time_out_flags": self._termination_time_out_flags,
              "storage": {"byte_limit": "actual compressed bytes" if self._compressor else "UTF-8 bytes",
                          "compact": "field definitions appear in schema rows; values keys are field IDs; omitted nonfinite field means no nonfinite components ONLY if its schema present_if field exists in this step, otherwise unknown/not_evaluated; metadata_delta/terms_delta persist, null deletes a key; weights_at_entry persists until changed; missing other values means not evaluated",
                          "critic_validity": "compact: invalid IDs only, complement valid; legacy: valid IDs",
                          "abrupt_abort": "gzip uses sync-flushed chunks; a process abort can leave a missing gzip trailer; buffered reward frames can be lost"},
              "limits": "all-env finite aggregates and every nonfinite ID; finite raw trace only for fixed env IDs; max_pending_bytes excludes one current frame and flush stack workspace; abrupt process abort may lose unflushed frames"}
    self._write(header, reserve=False)
    for cfg, proxy in zip(manager._term_cfgs, self._proxies):
      cfg.func = proxy
    manager.compute, manager.reset = self._compute_hook, self._reset_hook
    manager._r1wb_reward_ledger = self

  def _write(self, value, *, reserve=True):
    payload = (json.dumps(_json_safe(value), ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
    # Trial the real compressed output before committing. This is a gzip stream,
    # never an uncompressed disk staging file; SYNC_FLUSH also supports live reads.
    trial = self._compressor.copy() if self._compressor else None
    encoded = trial.compress(payload) + trial.flush(zlib.Z_SYNC_FLUSH) if trial else payload
    trailer_reserve = 32 if trial else 0
    if self.bytes_written + len(encoded) > self.options.max_file_bytes - (4096 if reserve else 0) - trailer_reserve:
      self.active, self.stop_reason = False, "max_file_bytes"
      return False
    self._file.write(encoded)
    if trial:
      self._compressor = trial
    self.bytes_written += len(encoded)
    return True

  def _attempt(self, function, stage):
    try:
      return function()
    except Exception as exc:
      self.active, self.stop_reason = False, "instrumentation_error"
      self.errors.append({"stage": stage, "error": f"{type(exc).__name__}: {exc}"[:1000]})
      return None

  def _force_tensor(self, data, name):
    """Read the installed mjlab wrapper's existing zero-copy view, never coerce it."""
    wrapped = getattr(data, name, None)
    if isinstance(wrapped, torch.Tensor):
      value, adapter = wrapped, "torch.Tensor"
    else:
      from mjlab.sim.sim_data import TorchArray
      if not isinstance(wrapped, TorchArray):
        return None, {"status": "unavailable", "type": type(wrapped).__name__}
      value, adapter = wrapped._tensor, "mjlab.sim.sim_data.TorchArray._tensor"
    ndim = 3 if name == "xfrc_applied" else 2
    valid = (isinstance(value, torch.Tensor) and value.ndim == ndim and value.shape[0] == self.n
             and value.device == self.device and value.is_floating_point()
             and (name != "xfrc_applied" or value.shape[-1] == 6))
    if not valid:
      return None, {"status": "invalid_shape_world_axis_device_or_dtype", "adapter": adapter,
                    "shape": list(getattr(value, "shape", ())), "device": str(getattr(value, "device", None))}
    return value.detach(), {"status": "available", "adapter": adapter, "shape": list(value.shape),
                            "dtype": str(value.dtype), "device": str(value.device), "stride": list(value.stride()),
                            "world_axis": 0}

  def _exposure(self, name, active, finite, tensors, episode_length, *, value=None):
    prefix = f"physics/{name}/"
    previous = self._force_history.get(name)
    counter = getattr(self.env, "common_step_counter", None)
    if previous is None:
      previous = {"known": torch.zeros_like(active), "active": torch.zeros_like(active),
                  "continuous": torch.zeros(self.n, dtype=torch.long, device=self.device),
                  "episode_active_frames": torch.zeros(self.n, dtype=torch.long, device=self.device),
                  "episode_onsets": torch.zeros(self.n, dtype=torch.long, device=self.device),
                  "counter": None}
      self._force_history[name] = previous
    contiguous = previous["known"] & finite
    if isinstance(counter, int) and isinstance(previous["counter"], int) and counter != previous["counter"] + 1:
      contiguous = torch.zeros_like(contiguous)
    if episode_length is not None and "age" in previous:
      contiguous = contiguous & (episode_length == previous["age"] + 1)
    onset = active & contiguous & ~previous["active"]
    censored = active & ~contiguous
    continuous = torch.where(active, torch.where(contiguous & previous["active"], previous["continuous"] + 1, 1), 0)
    tensors[prefix + "active"] = active.clone()
    tensors[prefix + "invalid"] = (~finite).clone()
    tensors[prefix + "new_force"] = onset
    tensors[prefix + "left_censored_active"] = censored
    tensors[prefix + "continuous_frames"] = continuous.clone()
    if value is not None:
      changed = (value != previous["value"]).reshape(self.n, -1).any(dim=1) if "value" in previous else torch.zeros_like(active)
      tensors[prefix + "changed_active"] = changed & active & previous["active"] & contiguous
      previous["value"] = value.clone()
    previous["episode_active_frames"] += active.to(torch.long)
    previous["episode_onsets"] += onset.to(torch.long)
    for key in ("episode_active_frames", "episode_onsets"):
      tensors[prefix + key + "_stats"] = _stats(previous[key])
      tensors[prefix + key + "_trace"] = previous[key][self._indices].clone()
    previous.update(known=finite.clone(), active=active.clone(), continuous=continuous, counter=counter)
    if episode_length is not None:
      previous["age"] = episode_length.clone()

  def _forces(self, tensors, metadata, episode_length):
    data = getattr(getattr(self.env, "sim", None), "data", None)
    sources = []
    for name in ("xfrc_applied", "qfrc_applied"):
      value, layout = self._force_tensor(data, name)
      metadata[f"physics/{name}"] = layout
      if value is None:
        self._force_history.pop(name, None)
        continue
      self._force_layouts[name] = layout
      tensors[f"physics/{name}/trace"] = value[self._indices].clone()
      tensors[f"physics/{name}/stats"] = _stats(value)
      tensors[f"physics/{name}/nonfinite"] = _bad_codes(value)
      finite = torch.isfinite(value).reshape(self.n, -1).all(dim=1)
      # Unknown/nonfinite worlds stay explicit; 'active' means at least one
      # observed finite nonzero component, not a claim that invalid worlds are quiet.
      active = (torch.isfinite(value) & (value != 0)).reshape(self.n, -1).any(dim=1)
      self._exposure(name, active, finite, tensors, episode_length, value=value)
      if name == "xfrc_applied":
        for label, part in (("force_norm_N", value[..., :3]), ("torque_norm_Nm", value[..., 3:])):
          norm = torch.linalg.vector_norm(part, dim=-1)
          tensors[f"physics/{name}/{label}_trace"] = norm[self._indices].clone()
          tensors[f"physics/{name}/{label}_stats"] = _stats(norm)
      sources.append((active, finite))
    if len(sources) == 2:
      self._exposure("applied_union", sources[0][0] | sources[1][0], sources[0][1] & sources[1][1],
                     tensors, episode_length)
    else:
      self._force_history.pop("applied_union", None)
      metadata["physics/applied_union"] = "unavailable: both force sources must be validly shaped"

  def _begin(self, dt):
    if not self.active:
      self._critic = None
      return
    if self.steps_seen >= self.options.max_steps:
      self.active, self.stop_reason = False, "max_steps"
      self._critic = None
      self.flush()
      return
    self.steps_seen += 1
    tensors, metadata = {}, {}
    tensors["episode_serial"] = self._episodes[self._indices].clone()
    tensors["reset_counts"] = self._reset_counts.clone()
    self._reset_counts.zero_()
    episode_length = getattr(self.env, "episode_length_buf", None)
    if isinstance(episode_length, torch.Tensor) and episode_length.shape == (self.n,):
      tensors["episode_length_all"] = episode_length.detach().clone()
      tensors["episode_length_trace"] = episode_length[self._indices].clone()
      tensors["episode_length_stats"] = _stats(episode_length)
    else:
      episode_length = None
    termination = getattr(self.env, "termination_manager", None)
    terms = getattr(termination, "_term_dones", {})
    for name, value in terms.items():
      tensors[f"termination/{name}"] = value.detach().clone()
    for name in ("terminated", "time_outs"):
      value = getattr(termination, name, None)
      if isinstance(value, torch.Tensor):
        tensors[f"termination_union/{name}"] = value.detach().clone()
    if not self.options.compact:
      metadata["termination_time_out_flags"] = self._termination_time_out_flags
    if terms:
      counts = torch.stack([value.detach().to(torch.long) for value in terms.values()]).sum(dim=0)
      tensors["termination_overlap/reason_count"] = counts
      tensors["termination_overlap/multiple_reasons"] = counts > 1
    self._forces(tensors, metadata, episode_length)
    if self._critic is not None:
      values, valid, critic_meta = self._critic
      tensors["critic/trace"] = values[self._indices].clone()
      tensors["critic/invalid" if self.options.compact else "critic/valid"] = ~valid if self.options.compact else valid.clone()
      tensors["critic/stats"] = _stats(torch.where(valid, values, float("nan")))
      tensors["critic/nonfinite"] = _bad_codes(values)
      metadata["critic"] = {**critic_meta, "stats_mask_reset_invalid_with_nan": True}
    else:
      metadata["critic"] = "not_supplied"
    self._critic = None
    self._current = {"sequence": self.steps_seen, "dt": dt,
                     "scale": dt if self.manager._scale_by_dt else 1., "tensors": tensors,
                     "metadata": metadata, "terms": {}, "weights_at_entry": [float(c.weight) for c in self.manager._term_cfgs],
                     "common_step_counter": getattr(self.env, "common_step_counter", None),
                     "physics_step_counter": getattr(self.env, "_sim_step_counter", None),
                     "expected": torch.zeros_like(self.manager._reward_buf)}
    for name, callback in self.pure_diagnostics.items():
      cfg = self.manager._term_cfgs[self.names.index(name)]
      if cfg.weight != 0.:
        metadata[f"diagnostic/{name}"] = "not_called_active_reward_captured_natively"
        continue
      value = callback()
      if not isinstance(value, torch.Tensor) or value.shape != (self.n,):
        raise ValueError(f"pure diagnostic {name!r} must return shape({self.n},)")
      tensors[f"diagnostic/{name}/raw_stats"] = _stats(value)
      tensors[f"diagnostic/{name}/raw_trace"] = value.detach()[self._indices].clone()
      tensors[f"diagnostic/{name}/raw_nonfinite"] = _bad_codes(value)
      metadata[f"diagnostic/{name}"] = "explicit_pure_callback_once_before_native_compute; not_used_in_reward"

  def _capture(self, index, value):
    current = self._current
    if not isinstance(value, torch.Tensor) or value.shape != (self.n,):
      current["terms"][str(index)] = {"status": "invalid_return_shape", "shape": list(getattr(value, "shape", ())),
                                      "type": type(value).__name__}
      return  # Let the original manager raise its own exception.
    weight = float(self.manager._term_cfgs[index].weight)
    raw = value.detach()
    weighted = raw * weight * current["scale"]  # Same expression/order as the manager; never another term call.
    effective = torch.nan_to_num(weighted, nan=0., posinf=0., neginf=0.)
    tensors = current["tensors"]
    prefix = f"term/{index}/"
    for name, tensor in (("raw_stats", _stats(raw)), ("raw_trace", raw[self._indices]),
                         ("raw_nonfinite", _bad_codes(raw)), ("weighted_stats", _stats(weighted)),
                         ("weighted_trace", weighted[self._indices]), ("weighted_nonfinite", _bad_codes(weighted)),
                         ("effective_stats", _stats(effective)), ("effective_trace", effective[self._indices])):
      tensors[prefix + name] = tensor.clone()
    current["expected"] += effective
    current["terms"][str(index)] = {"status": "evaluated", "weight": weight,
                                    "raw_dtype": str(raw.dtype), "raw_device": str(raw.device)}

  def _vectors(self):
    if not self.active or self._current is None:
      return
    tensors = self._current["tensors"]
    for name, spec in self.pure_vector_diagnostics.items():
      value = spec.callback()
      if (not isinstance(value, torch.Tensor) or value.shape != (self.n, len(spec.columns))
          or value.device != self.device or not value.is_floating_point()):
        raise ValueError(f"vector diagnostic {name!r} must return floating shape({self.n},{len(spec.columns)}) on {self.device}")
      value = value.detach()
      prefix = f"vector_diagnostic/{name}/"
      tensors[prefix + "column_stats"] = torch.stack([_stats(value[:, i]) for i in range(value.shape[1])])
      tensors[prefix + "trace"] = value[self._indices].clone()
      tensors[prefix + "nonfinite"] = _bad_codes(value)
      if spec.compare_max is not None:
        left, right = (value[:, spec.columns.index(c)] for c in spec.compare_max)
        valid = torch.isfinite(left) & torch.isfinite(right)
        if spec.valid_column is not None:
          valid = valid & (value[:, spec.columns.index(spec.valid_column)] == 1)
        winner = torch.where(valid, torch.where(left > right, 1, torch.where(right > left, 2, 3)), 0)
        tensors[prefix + "max_winner_counts"] = torch.stack([(winner == i).sum() for i in (1, 2, 3, 0)])
        tensors[prefix + "max_winner_trace"] = winner[self._indices].clone()
        tensors[prefix + "invalid"] = ~valid

  def _finish(self, result=None, error=None):
    current, self._current = self._current, None
    if current is None:
      return
    expected = current.pop("expected")
    for index, weight in enumerate(current["weights_at_entry"]):
      current["terms"].setdefault(str(index), {"status": "not_evaluated", "raw": "unknown",
                                              "reason": ("zero_weight" if weight == 0 else
                                                         "instrumentation_error" if self.errors else "compute_did_not_reach_term"),
                                              "weight": weight})
    if result is not None:
      current["tensors"]["reward/actual_stats"] = _stats(result)
      current["tensors"]["reward/actual_trace"] = result.detach()[self._indices].clone()
      current["tensors"]["reward/nonfinite"] = _bad_codes(result)
      complete = all(t["status"] == "evaluated" or t.get("reason") == "zero_weight" for t in current["terms"].values())
      current["sum_check_status"] = "complete" if complete else "incomplete_instrumentation_or_term_error"
      if complete:
        current["tensors"]["reward/sum_abs_error"] = (result.detach() - expected).abs().amax().reshape(1)
        current["tensors"]["reward/sum_exact_equal"] = torch.eq(result.detach(), expected).all().reshape(1)
    current["compute_error"] = error
    size = sum(t.numel() * t.element_size() for t in current["tensors"].values())
    if size > self.options.max_pending_bytes:
      self.active, self.stop_reason = False, "single_frame_exceeds_max_pending_bytes"
      self.errors.append({"stage": "buffer", "error": f"unwritten sequence{current['sequence']}: {size}bytes"})
      return
    if self._pending_bytes + size > self.options.max_pending_bytes:
      self.flush()
    self._pending.append(current)
    self._pending_bytes += size
    if len(self._pending) >= self.options.flush_every or error is not None:
      self.flush()

  def _compute(self, *args, **kwargs):
    dt = kwargs.get("dt", args[0] if args else None)
    self._attempt(lambda: self._begin(dt), "begin")
    try:
      result = self._compute_original(*args, **kwargs)
    except BaseException as exc:
      self._attempt(lambda: self._finish(error=f"{type(exc).__name__}: {exc}"), "finish_exception")
      raise
    self._attempt(self._vectors, "pure_vector_diagnostics")
    self._attempt(lambda: self._finish(result), "finish")
    return result

  def _reset(self, *args, **kwargs):
    result = self._reset_original(*args, **kwargs)
    if not self.active:
      return result
    ids = kwargs.get("env_ids", args[0] if args else None)
    ids = slice(None) if ids is None else ids
    def record():
      self._episodes[ids] += 1
      self._reset_counts[ids] += 1
      for history in self._force_history.values():
        history["known"][ids] = False
        history["active"][ids] = False
        for key in ("continuous", "episode_active_frames", "episode_onsets"):
          history[key][ids] = 0
      if self._critic is not None:
        self._critic[1][ids] = False
    self._attempt(record, "reset")
    return result

  def record_critic(self, values: torch.Tensor, *, observation_step: int | None = None,
                    phase: str = "pre_action", metadata: Mapping[str, Any] | None = None):
    """Consume once at the next reward compute; caller must supply the action's critic.

    This API never runs the critic/observation pipeline. Resets after submission
    invalidate affected env values. Unused values do not carry to a later step.
    """
    if not self.active:
      return
    values = values.detach().reshape(-1)
    if values.shape != (self.n,) or values.device != self.device:
      raise ValueError("critic values must match manager's environment count and device")
    self._critic = (values.clone(), torch.ones(self.n, dtype=torch.bool, device=self.device),
                    {"observation_step": observation_step, "phase": phase, **dict(metadata or {})})

  def flush(self):
    if not self._pending:
      self._file.flush()
      return
    pending, self._pending = self._pending, []
    self._pending_bytes = 0
    # One stacked device-to-host transfer per field/buffer flush, not per term/env/step.
    keys = set().union(*(frame["tensors"] for frame in pending))
    cpu = {}
    for key in sorted(keys):
      present = [(i, f["tensors"][key]) for i, f in enumerate(pending) if key in f["tensors"]]
      packed = torch.stack([v for _, v in present]).detach().cpu()
      cpu[key] = {i: packed[j] for j, (i, _) in enumerate(present)}
    for i, frame in enumerate(pending):
      frame.pop("tensors")
      record = {"type": "step", **frame, "values": {}}
      schema = {}
      nonfinite_count = 0
      for key in sorted(keys):
        if i not in cpu[key]:
          continue
        if key == "episode_length_all":
          continue  # Kept only to align sparse force/termination/reset records.
        value = cpu[key][i]
        if self.options.compact and key not in self._field_schema:
          field_id = str(len(self._field_schema))
          self._field_schema[key] = field_id
          schema[field_id] = {"name": key, "shape": list(value.shape), "dtype": str(value.dtype)}
          if key.endswith("nonfinite"):
            companion = key.removesuffix("nonfinite") + "stats"
            if key == "reward/nonfinite":
              companion = "reward/actual_stats"
            elif key.startswith("vector_diagnostic/"):
              companion = key.removesuffix("nonfinite") + "column_stats"
            schema[field_id]["present_if"] = companion
        output_key = self._field_schema[key] if self.options.compact else key
        if key.endswith("nonfinite"):
          where = torch.nonzero(value, as_tuple=False)
          nonfinite_count += len(where)
          if not self.options.compact or len(where):
            record["values"][output_key] = {"shape": list(value.shape), "indices": where.tolist(),
                                             "codes": value[tuple(where.T)].tolist() if len(where) else []}
        elif (key.startswith("termination") or key in ("reset_counts", "critic/valid", "critic/invalid")
              or key.endswith(("/active", "/invalid", "/new_force", "/left_censored_active",
                               "/changed_active", "/continuous_frames"))):
          where = torch.nonzero(value, as_tuple=False).flatten()
          record["values"][output_key] = {"env_ids": where.tolist(), "values": value[where].tolist()}
          if i in cpu.get("episode_length_all", {}):
            record["values"][output_key]["episode_length_steps"] = cpu["episode_length_all"][i][where].tolist()
        else:
          record["values"][output_key] = value.tolist()
      if self.options.compact:
        for key, previous in (("metadata", self._previous_metadata), ("terms", self._previous_terms)):
          current = record.pop(key)
          delta = {k: v for k, v in current.items() if k not in previous or previous[k] != v}
          delta.update({k: None for k in previous if k not in current})
          previous.clear()
          previous.update(current)
          if delta:
            record[key + "_delta"] = delta
        weights = record["weights_at_entry"]
        if weights == self._previous_weights:
          record.pop("weights_at_entry")
        else:
          self._previous_weights = weights
        if record["compute_error"] is None:
          record.pop("compute_error")
        if schema and not self._write({"type": "schema", "fields": schema}):
          break
      if not self._write(record):
        break
      self.steps_written += 1
      self._quality["nonfinite_classifications"] += nonfinite_count
      exact = cpu.get("reward/sum_exact_equal", {}).get(i)
      self._quality["sum_exact_frames"] += int(exact is not None and bool(exact.all()))
      meta = frame["metadata"]
      self._quality["force_available_frames"] += int(all(
        isinstance(meta.get(f"physics/{name}"), dict) and meta[f"physics/{name}"].get("status") == "available"
        for name in ("xfrc_applied", "qfrc_applied")))
      self._quality["critic_supplied_frames"] += int(isinstance(meta.get("critic"), dict))
      for name in self.pure_vector_diagnostics:
        self._quality["vector_frames"][name] += int(i in cpu.get(f"vector_diagnostic/{name}/trace", {}))
    self._file.flush()

  def _finish_file(self):
    if self._compressor is not None:
      trailer = self._compressor.flush(zlib.Z_FINISH)
      self._file.write(trailer)
      self.bytes_written += len(trailer)
      self._compressor = None
    self._file.close()

  def summary(self):
    """Host-only run-metadata summary; call after close for a terminal verdict."""
    complete = (self.closed and self.steps_seen == self.steps_written == self.options.max_steps
                and not self.errors and self.stop_reason in (None, "max_steps")
                and self._quality["sum_exact_frames"] == self.steps_written
                and self._quality["force_available_frames"] == self.steps_written)
    return {"schema": "r1wb-reward-ledger-status-v2",
            "profile": "compact-gzip-v2" if self.options.compact and self.options.compression == "gzip" else "legacy",
            "closed": self.closed, "active": self.active, "stop_reason": self.stop_reason or ("closed" if self.closed else None),
            "expected_steps": self.options.max_steps, "steps_seen": self.steps_seen, "steps_written": self.steps_written,
            "bytes_written": self.bytes_written, "max_file_bytes": self.options.max_file_bytes,
            "errors": [dict(e) for e in self.errors],
            "quality": {**self._quality, "vector_frames": dict(self._quality["vector_frames"])},
            "final_success": complete}

  def close(self):
    if self.closed:
      return
    self._attempt(self.flush, "close_flush")
    manager = self.manager
    for name, hook, original_instance in (("compute", self._compute_hook, self._compute_instance),
                                          ("reset", self._reset_hook, self._reset_instance)):
      if manager.__dict__.get(name) is hook:
        if original_instance is None:
          delattr(manager, name)
        else:
          setattr(manager, name, original_instance)
    for cfg, proxy in zip(manager._term_cfgs, self._proxies):
      if cfg.func is proxy:
        cfg.func = proxy.original
    if getattr(manager, "_r1wb_reward_ledger", None) is self:
      delattr(manager, "_r1wb_reward_ledger")
    self._attempt(lambda: self._write({"type": "footer", "steps_seen": self.steps_seen,
                                      "steps_written": self.steps_written, "stop_reason": self.stop_reason or "closed",
                                      "errors": [{"stage": e["stage"], "error": e["error"][:300]} for e in self.errors[:8]],
                                      "expected_steps": self.options.max_steps, "quality": self._quality,
                                      "pending_critic_discarded": self._critic is not None},
                                     reserve=False), "footer")
    self._attempt(self._finish_file, "file_close")
    self._critic, self._current = None, None
    self.closed, self.active = True, False


def attach(env, output_path: str | Path, options: LedgerOptions | Mapping[str, Any] | None = None) -> RewardLedger | None:
  """Attach after managers exist. Disabled mode performs no env/file/tensor work."""
  opts = options if isinstance(options, LedgerOptions) else LedgerOptions(**dict(options or {}))
  if not opts.enabled:
    return None
  return RewardLedger(env, output_path, opts)
