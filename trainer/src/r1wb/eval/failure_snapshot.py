"""Opt-in, bounded evidence from immediately before each physical substep.

This is a physics-input snapshot, not an RL checkpoint or a claim of deterministic
replay. Keep every world: contact allocation and collision ordering are shared.
"""
from __future__ import annotations

from contextlib import nullcontext
import hashlib
from importlib import metadata
from pathlib import Path
import platform
import time

import torch


# MJWarp State.INTEGRATION, plus acceleration and sleeping-state diagnostics.
DATA_FIELDS = ("time", "qpos", "qvel", "act", "history", "qacc_warmstart", "ctrl",
               "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
               "userdata", "qacc", "act_dot", "tree_asleep")
_ABSENT = object()


def _tensor(value, *, shared_model=False):
  raw_array = getattr(value, "_wp_array", None)
  value = getattr(value, "_tensor", value)
  if not isinstance(value, torch.Tensor):
    raise TypeError("Expected a torch tensor or MJLab TorchArray")
  # Only the model bridge broadcasts shared arrays. Data arrays keep their
  # actual world axis, including empty arrays whose native strides can be zero.
  # Empty per-world model fields (e.g. tendons) must retain that axis as well.
  if (shared_model and value.numel() > 0 and value.ndim and value.shape[0] > 1 and
      value.stride(0) == 0 and (raw_array is None or raw_array.shape[0] == 1)):
    value = value[:1]
  return value


def _hash(path):
  digest = hashlib.sha256()
  with Path(path).open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def _layout(value, world_axis):
  return {"shape": list(value.shape), "dtype": str(value.dtype), "stride": list(value.stride()),
          "world_axis": world_axis, "bytes": value.numel() * value.element_size()}


def _signature(value):
  return value.data_ptr(), tuple(value.shape), value.dtype, tuple(value.stride())


def _option_scalars(sim):
  options = getattr(sim.wp_model, "opt", None)
  # IntEnum/IntFlag values are ints for isinstance(), but torch.save otherwise
  # pickles their class and weights_only=True correctly refuses that global.
  # Normalize subclasses here so snapshots only need tensors and builtins.
  result = {}
  for name, value in vars(options).items():
    for primitive in (bool, int, float, str):
      if isinstance(value, primitive):
        result[name] = primitive(value)
        break
  return result


def collision_execution_metadata(sim):
  """Read only scalar/layout evidence; never copy private scratch tensor values."""
  def layout(value):
    if isinstance(value, dict):
      return {str(name): layout(item) for name, item in value.items()}
    if isinstance(value, (tuple, list)):
      return [layout(item) for item in value]
    return {"shape": list(value.shape), "dtype": str(value.dtype)}
  batches = getattr(sim, "_r1wb_collision_workspaces", None)
  cells = getattr(sim, "_r1wb_cell_batch", None)
  result = {"use_cuda_graph": bool(sim.use_cuda_graph),
            "step_graph_present": getattr(sim, "step_graph", None) is not None,
            "batching_installed": batches is not None, "batching_workspaces": []}
  if batches is not None:
    result["batching_workspaces"] = [
      {"kernel": getattr(key[0], "key", str(type(key[0]))), "grouped_by_geom": len(values) >= 6,
       "arrays": layout(values)} for key, values in batches.items()]
    result["batching_workspaces"].sort(key=lambda item: (item["kernel"], str(item["arrays"])))
  if cells is not None:
    result["cells"] = {"budget": int(cells.budget), "max_pairs": int(cells.max_pairs),
                       "conditional": bool(cells.conditional), "builders_supported": cells.builders is not None,
                       "kernel_count": len(cells.kernels), "workspaces": []}
    for key, values in cells.workspaces.items():
      result["cells"]["workspaces"].append({"capacity": int(key[0]),
        "epa_layout": [[int(width), str(dtype)] for width, dtype in key[1]],
        "overflow_capacity": int(key[2]), "arrays": layout(values)})
    result["cells"]["workspaces"].sort(key=lambda item: (item["capacity"], str(item["epa_layout"])))
  result["source_sha256"] = {}
  for name in ("collision_batch.py", "collision_cells.py"):
    source = Path(__file__).parents[1] / name
    if source.is_file():
      result["source_sha256"][name] = _hash(source)
  return result


def _model_sources(sim):
  fields = set(sim._expanded_fields) | set(sim.default_model_fields)
  sources = {f"model.{name}": _tensor(getattr(sim.model, name), shared_model=True) for name in sorted(fields)}
  for name, value in vars(sim.wp_model.opt).items():
    if isinstance(value, torch.Tensor) or hasattr(value, "ptr"):
      sources[f"option.{name}"] = _tensor(getattr(sim.model.opt, name), shared_model=True)
  return sources


def _source(sim, name):
  prefix, field = name.split(".", 1)
  container = sim.data if prefix == "data" else sim.model.opt if prefix == "option" else sim.model
  return _tensor(getattr(container, field), shared_model=prefix != "data")


def restore_physics_inputs(sim, state, model):
  """Restore into an already configured, identically sized simulation, without stepping.

  The caller must recreate the saved MJB, capacities, options and expanded fields
  first. Validation happens before writes. This helper does not restore an agent,
  random generators, observation histories, external callbacks or sensor buffers.
  """
  entries = [(f"data.{name}", value) for name, value in state.items()] + list(model.items())
  targets = []
  for name, value in entries:
    target = _source(sim, name)
    if target.shape != value.shape or target.dtype != value.dtype:
      raise ValueError(f"Physics snapshot layout mismatch for {name}: saved shape={tuple(value.shape)}, "
                       f"dtype={value.dtype}; destination shape={tuple(target.shape)}, dtype={target.dtype}")
    targets.append((target, value))
  stream = getattr(getattr(sim.data, "qpos"), "_torch_stream", None)
  with torch.cuda.stream(stream) if stream is not None else nullcontext():
    for target, value in targets:
      target.copy_(value)


class FailureReplayBuffer:
  """Two model generations and decimation+1 full-world input frames on device.

  Construct only for the explicitly enabled diagnostic run. A broken capture
  disables itself and records why; it never replaces the simulation's exception.
  """

  def __init__(self, env, directory, rank, *, max_device_bytes=512 * 1024 * 1024,
               max_disk_bytes=1024 * 1024 * 1024):
    self.env, self.sim = env, env.sim
    self.directory = Path(directory) / f"failure_replay_rank{rank}"
    self.rank, self.max_device_bytes, self.max_disk_bytes = rank, max_device_bytes, max_disk_bytes
    self.errors, self.frames, self.models = [], [], []
    self.sequence, self.generation, self.last_control, self.last_physics = 0, -1, None, None
    self.substep, self.allocated_bytes, self.active = 0, 0, False
    self.static_model, self.stream = None, None
    self.native_step, self.previous_step = self.sim.step, self.sim.__dict__.get("step", _ABSENT)
    self.num_envs, self.decimation = int(env.num_envs), int(env.cfg.decimation)
    self.ring_size = self.decimation + 1
    self.metadata = {"schema_version": 1, "rank": rank, "num_envs": self.num_envs,
                     "decimation": self.decimation, "ring_size": self.ring_size,
                     "model_generations": 2, "all_worlds_preserved": True,
                     "max_device_bytes": max_device_bytes, "max_disk_bytes": max_disk_bytes,
                     "scope": "one physics step from recorded inputs; not policy/observation replay",
                     "limitations": ["Replay agreement must be measured, not inferred from a complete snapshot.",
                       "GPU atomics and collision ordering can remain nondeterministic.",
                       "Contact/solver scratch buffers are regenerated, not saved; stale-memory bugs may not replay.",
                       "Unexpanded model arrays are assumed immutable after construction.",
                       "The device budget covers retained tensors and the largest comparison, not allocator reservation.",
                       "Post-step events/reset/forward/sense and observation histories are not replayed."]}
    try:
      self._prepare()
      self.active = True
    except Exception as error:
      self._invalidate(f"prepare: {type(error).__name__}: {error}")

    def step(*args, **kwargs):
      if self.active:
        try:
          self.capture()
        except Exception as error:
          self._invalidate(f"capture: {type(error).__name__}: {error}")
      return self.native_step(*args, **kwargs)

    self.step = step
    self.sim.step = step

  def _invalidate(self, reason):
    self.errors.append(reason)
    self.active = False

  def _prepare(self):
    if self.decimation < 1 or self.num_envs < 1:
      raise ValueError("Positive decimation and world count are required")
    self.data_sources = {name: _tensor(getattr(self.sim.data, name)) for name in DATA_FIELDS}
    invalid = [f"data.{name}: shape={tuple(value.shape)}, stride={tuple(value.stride())}, dtype={value.dtype}"
               for name, value in self.data_sources.items() if value.ndim == 0 or value.shape[0] != self.num_envs]
    if invalid:
      raise ValueError(f"Every state field must retain the complete world axis (expected first axis={self.num_envs}): "
                       + "; ".join(invalid))
    self.model_sources = _model_sources(self.sim)
    self.expanded_fields = sorted(self.sim._expanded_fields)
    self.default_fields = sorted(self.sim.default_model_fields)
    self.signatures = {f"data.{name}": _signature(value) for name, value in self.data_sources.items()}
    self.signatures.update({name: _signature(value) for name, value in self.model_sources.items()})
    self.metadata["state_layout"] = {name: _layout(value, 0) for name, value in self.data_sources.items()}
    self.metadata["model_layout"] = {
      name: _layout(value, 0 if value.ndim and value.shape[0] == self.num_envs else None)
      for name, value in self.model_sources.items()}
    self.metadata["expanded_fields"] = self.expanded_fields
    self.metadata["default_model_layout"] = {
      name: _layout(value, 0 if name in getattr(self.sim, "per_world_default_fields", set()) else None)
      for name, value in self.sim.default_model_fields.items()}
    self.data_bytes = sum(value.numel() * value.element_size() for value in self.data_sources.values())
    self.model_bytes = sum(value.numel() * value.element_size() for value in self.model_sources.values())
    self.default_bytes = sum(value.numel() * value.element_size() for value in self.sim.default_model_fields.values())
    # Include the largest temporary bool comparison used to detect in-control
    # model writes; it is not safe to rely on Tensor._version for Warp writes.
    comparison_bytes = max((value.numel() for value in self.model_sources.values()), default=0)
    required = self.ring_size * self.data_bytes + 2 * self.model_bytes + comparison_bytes + 32
    self.metadata["required_device_bytes"] = required
    if required > self.max_device_bytes:
      raise ValueError(f"Full-world ring requires {required} bytes, budget is {self.max_device_bytes}; no world reduction")
    self.stream = getattr(getattr(self.sim.data, "qpos"), "_torch_stream", None)
    if next(iter(self.data_sources.values())).is_cuda and self.stream is None:
      raise ValueError("Cannot establish ordering with the simulator's CUDA stream")
    self.metadata["copy_stream"] = getattr(self.stream, "cuda_stream", None)
    self._physics_metadata()
    self.directory.mkdir(parents=True, exist_ok=True)
    self._save_static_model()
    expected_disk = self.ring_size * self.data_bytes + 2 * self.model_bytes + self.data_bytes + self.default_bytes
    if expected_disk + self.static_model["bytes"] + 4 * 1024 * 1024 > self.max_disk_bytes:
      raise ValueError("Full-world snapshot exceeds disk budget; no partial replay claim")
    self.metadata["expected_tensor_bytes"] = expected_disk
    with self._stream_context():
      self.frames = [{"state": {name: torch.empty_like(value) for name, value in self.data_sources.items()},
                      "sequence": None} for _ in range(self.ring_size)]
      self.models = [{"values": {name: torch.empty_like(value) for name, value in self.model_sources.items()},
                      "changed_within_control": torch.zeros((), dtype=torch.bool,
                                                            device=next(iter(self.data_sources.values())).device),
                      "generation": None} for _ in range(2)]
    self.allocated_bytes = self.ring_size * self.data_bytes + 2 * self.model_bytes + 2

  def _stream_context(self):
    return torch.cuda.stream(self.stream) if self.stream is not None else nullcontext()

  def _physics_metadata(self):
    import mujoco
    sim, model, data = self.sim, self.sim.wp_model, self.sim.wp_data
    self.metadata["capacities"] = {name: getattr(data, name, None) for name in
                                   ("nworld", "naconmax", "naccdmax", "njmax", "nvmax", "njmax_pad", "njmax_nnz")}
    self.metadata["simulation_config"] = {name: getattr(sim.cfg, name, None) for name in
                                          ("nconmax", "njmax", "contact_sensor_maxmatch", "broadphase", "broadphase_filter")}
    self.metadata["option_scalars_at_install"] = _option_scalars(sim)
    self.metadata["use_cuda_graph"] = sim.use_cuda_graph
    self.metadata["collision_execution"] = collision_execution_metadata(sim)
    self.metadata["seed"] = getattr(self.env.cfg, "seed", None)
    self.metadata["versions"] = {"python": platform.python_version(), "torch": torch.__version__,
                                  "torch_cuda": torch.version.cuda}
    for package in ("mjlab", "mujoco", "mujoco-warp", "warp-lang"):
      try:
        self.metadata["versions"][package] = metadata.version(package)
      except metadata.PackageNotFoundError:
        self.metadata["versions"][package] = "unknown"
    unsupported = []
    if int(model.opt.enableflags) & int(mujoco.mjtEnableBit.mjENBL_SLEEP):
      unsupported.append("sleeping physics retains additional derived state")
    if not model.opt.run_collision_detection:
      unsupported.append("user-populated contacts are not captured")
    if getattr(model, "nplugin", 0):
      unsupported.append("plugin state is not captured")
    if any(value is not None for value in vars(model.callback).values()):
      unsupported.append("custom physics callbacks can retain external state")
    self.metadata["unsupported_physics"] = unsupported
    self.metadata["source_artifacts"] = {}
    for path in (Path(__file__), Path(__file__).parents[1] / "train.py",
                 Path(__file__).parents[1] / "collision_batch.py", Path(__file__).parents[1] / "collision_cells.py",
                 self.directory.parent / "params" / "source_snapshot.json", self.directory.parent / "params" / "env.yaml",
                 self.directory.parent / "params" / "agent.yaml"):
      if path.is_file():
        self.metadata["source_artifacts"][str(path)] = _hash(path)

  def _save_static_model(self):
    import mujoco
    size = mujoco.mj_sizeModel(self.sim.mj_model)
    if size > self.max_disk_bytes:
      raise ValueError("Static MJB exceeds disk budget")
    path = self.directory / "model.mjb"
    temporary = path.with_suffix(".mjb.part")
    try:
      mujoco.mj_saveModel(self.sim.mj_model, str(temporary))
      temporary.replace(path)
    finally:
      temporary.unlink(missing_ok=True)
    self.static_model = {"path": str(path), "bytes": path.stat().st_size, "sha256": _hash(path)}

  def capture(self):
    control, physics = int(self.env.common_step_counter), int(self.env._sim_step_counter)
    if sorted(self.sim._expanded_fields) != self.expanded_fields or sorted(self.sim.default_model_fields) != self.default_fields:
      raise ValueError("Model field set changed after diagnostic installation")
    for name, signature in self.signatures.items():
      if _signature(_source(self.sim, name)) != signature:
        raise ValueError(f"Physics array pointer/layout changed: {name}")
    if self.last_physics is not None and physics != self.last_physics + 1:
      raise ValueError("Physics substep counter is not consecutive")
    new_control = self.last_control != control
    if self.last_control is not None and new_control and (control != self.last_control + 1 or self.substep != self.decimation):
      raise ValueError("Control/model generation boundary does not match decimation")
    if not new_control and self.substep >= self.decimation:
      raise ValueError("Too many physical substeps in a control/model generation")
    with self._stream_context():
      if new_control:
        self.generation += 1
        model = self.models[self.generation % 2]
        for name, value in self.model_sources.items():
          model["values"][name].copy_(value)
        model["changed_within_control"].zero_()
        model.update(generation=self.generation, control_step=control, option_scalars=_option_scalars(self.sim))
        self.substep = 0
      else:
        model = self.models[self.generation % 2]
        if _option_scalars(self.sim) != model["option_scalars"]:
          raise ValueError("Scalar physics option changed within a control/model generation")
        for name, value in self.model_sources.items():
          model["changed_within_control"].logical_or_(torch.any(value != model["values"][name]))
      frame = self.frames[self.sequence % self.ring_size]
      for name, value in self.data_sources.items():
        frame["state"][name].copy_(value)
      frame.update(sequence=self.sequence, model_generation=self.generation, control_step=control,
                   physics_step=physics, substep=self.substep)
    self.sequence += 1
    self.substep += 1
    self.last_control, self.last_physics = control, physics

  def status(self):
    return {"enabled": True, "active": self.active, "allocated_device_bytes": self.allocated_bytes,
            "required_device_bytes": self.metadata.get("required_device_bytes"), "errors": list(self.errors),
            "unsupported_physics": self.metadata.get("unsupported_physics", []),
            "replay_ready": False, "reason": "No exported failure frame has been validated yet"}

  def export(self, stem):
    from r1lab.run_record import atomic_json
    report = {**self.metadata, "created_at": time.time(), "static_model": self.static_model,
              "capture_errors": list(self.errors), "replay_ready": False, "last_finite_frame": None,
              "allocated_device_bytes": self.allocated_bytes}
    path = self.directory / f"{stem}.json"
    self.directory.mkdir(parents=True, exist_ok=True)
    atomic_json(path, report)
    try:
      if self.stream is not None:
        self.stream.synchronize()
      frames = sorted((frame for frame in self.frames if frame["sequence"] is not None), key=lambda frame: frame["sequence"])
      models = [model for model in self.models if model["generation"] is not None]
      payload = {"num_envs": self.num_envs, "frames": [], "models": [], "default_model_fields": {}}
      model_index = {}
      for model in models:
        copied = {key: value for key, value in model.items() if key not in ("values", "changed_within_control")}
        copied["values"] = {name: value.cpu().clone() for name, value in model["values"].items()}
        copied["changed_within_control"] = bool(model["changed_within_control"].item())
        copied["all_finite"] = all(bool(torch.isfinite(value).all()) for value in copied["values"].values())
        payload["models"].append(copied)
        model_index[copied["generation"]] = copied
      report["frames"] = []
      for frame in frames:
        copied = {key: value for key, value in frame.items() if key != "state"}
        copied["state"] = {name: value.cpu().clone() for name, value in frame["state"].items()}
        bad = {name: (~torch.isfinite(value).reshape(self.num_envs, -1).all(dim=1)).nonzero().flatten().tolist()
               for name, value in copied["state"].items()}
        model = model_index.get(frame["model_generation"])
        finite = not any(bad.values()) and model is not None and model["all_finite"]
        valid_model = model is not None and model["control_step"] == frame["control_step"] and not model["changed_within_control"]
        details = {**{key: value for key, value in frame.items() if key != "state"},
                   "all_finite": finite, "model_generation_matches": valid_model, "nonfinite_envs": bad}
        report["frames"].append(details)
        if finite and valid_model:
          report["last_finite_frame"] = frame["sequence"]
        if not valid_model:
          report["capture_errors"].append(f"Model generation missing or changed for frame {frame['sequence']}")
        payload["frames"].append(copied)
      if frames:
        # Exception-time state is explicitly separate; resets/events may already
        # have changed it. It must not be paired with an earlier model generation.
        payload["exception_state"] = {name: value.cpu().clone() for name, value in self.data_sources.items()}
        payload["default_model_fields"] = {name: value.cpu().clone() for name, value in self.sim.default_model_fields.items()}
      if frames:
        target = self.directory / f"{stem}.pt"
        temporary = target.with_suffix(".pt.part")
        try:
          torch.save(payload, temporary)
          if temporary.stat().st_size + (self.static_model or {}).get("bytes", 0) > self.max_disk_bytes:
            raise ValueError("Serialized snapshot exceeds disk budget")
          temporary.replace(target)
        finally:
          temporary.unlink(missing_ok=True)
        report["tensor_artifact"] = {"path": str(target), "bytes": target.stat().st_size, "sha256": _hash(target)}
      report["replay_ready"] = bool(frames and self.static_model and report["last_finite_frame"] is not None
                                    and not report["capture_errors"] and not report.get("unsupported_physics"))
    except Exception as error:
      report["capture_errors"].append(f"export: {type(error).__name__}: {error}")
      report["replay_ready"] = False
    atomic_json(path, report)
    return {"metadata_path": str(path), "replay_ready": report["replay_ready"],
            "last_finite_frame": report["last_finite_frame"], "capture_errors": report["capture_errors"],
            "unsupported_physics": report.get("unsupported_physics", [])}

  def close(self):
    if self.sim.step is self.step:
      if self.previous_step is _ABSENT:
        del self.sim.step
      else:
        self.sim.step = self.previous_step
    self.active = False


def install_failure_replay_buffer(env, directory, rank, *, enabled=False, **limits):
  """The disabled path does not inspect the environment, allocate or install hooks."""
  if not enabled:
    return None
  buffer = FailureReplayBuffer(env, directory, rank, **limits)
  env._failure_replay_buffer = buffer
  return buffer
