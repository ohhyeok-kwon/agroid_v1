"""Independent one-step physics replay of a full-world failure_snapshot bundle.

No policy, environment, reset, reward or observation replay. Every trial uses a
fresh subprocess/simulation, the saved complete world count and exact capacities.
Raw contact order is retained; geometry comparisons also match contacts by pair.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

import numpy as np


STATE_OUTPUTS = ("qpos", "qvel", "qacc", "qacc_warmstart")
CONTACT_FIELDS = ("worldid", "geom", "geomcollisionid", "dist", "pos", "frame", "dim", "type", "flex", "elem", "vert")
VARIANTS = ("native", "compacted", "optimized")
LIMITATIONS = [
  "One physical substep only; no policy/observation/sensor/history/optimizer replay.",
  "Contact/solver scratch is regenerated. Stale-memory failures may not recur.",
  "Post-step reset, forward, sense, actuator update and environment callbacks are omitted.",
  "The producer stores pre-step inputs, not an independent post-step truth.",
  "A next pre-step in the same control is only a conditional reference; callbacks can intervene.",
  "Native-vs-native differences measure observed replay noise; absence of differences is not a determinism proof.",
  "Matching finite outputs does not prove absence of an earlier physical or learning failure.",
  "Contact arrays describe the step's contact calculation, not a new forward pass at integrated qpos.",
]


def sha256(path):
  with Path(path).open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
  path = Path(path)
  temporary = path.with_suffix(path.suffix + ".part")
  temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
  temporary.replace(path)


def artifact_path(metadata_path, record):
  """Allow an adjacent relocated artifact, but always require its recorded digest."""
  original = Path(record["path"])
  choices = (Path(metadata_path).parent / original.name, original)
  for candidate in dict.fromkeys(choices):
    if candidate.is_file() and sha256(candidate) == record["sha256"]:
      return candidate.resolve()
  raise ValueError(f"Missing or mismatched snapshot artifact: {original.name}")


def load_snapshot(path):
  import torch
  from r1wb.eval.failure_snapshot import DATA_FIELDS
  path = Path(path).resolve()
  report = json.loads(path.read_text())
  if report.get("schema_version") != 1 or not report.get("all_worlds_preserved"):
    raise ValueError("Requires a version-1 full-world physics snapshot")
  if report.get("capture_errors") or report.get("unsupported_physics"):
    raise ValueError("Snapshot has capture errors or explicitly unsupported physics")
  tensor_path = artifact_path(path, report["tensor_artifact"])
  model_path = artifact_path(path, report["static_model"])
  payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
  worlds = int(report["num_envs"])
  if worlds < 1 or payload["num_envs"] != worlds or report["capacities"]["nworld"] != worlds:
    raise ValueError("World count mismatch; subsampling is forbidden")
  frames = payload["frames"]
  if not frames or len({frame["sequence"] for frame in frames}) != len(frames):
    raise ValueError("Missing or duplicate frame sequences")
  generations = {entry["generation"]: entry for entry in payload["models"]}
  if len(generations) != len(payload["models"]):
    raise ValueError("Duplicate model generations")
  for frame in frames:
    if set(frame["state"]) != set(DATA_FIELDS):
      raise ValueError("Incomplete integration state")
    for name, value in frame["state"].items():
      layout = report["state_layout"][name]
      if (value.ndim == 0 or value.shape[0] != worlds or list(value.shape) != layout["shape"] or
          str(value.dtype) != layout["dtype"] or layout["world_axis"] != 0):
        raise ValueError(f"Invalid complete-world state layout: {name}; tensor shape={tuple(value.shape)}, "
                         f"dtype={value.dtype}; recorded layout={layout}; required worlds={worlds}")
    generation = generations.get(frame["model_generation"])
    if generation is None or generation["control_step"] != frame["control_step"] or generation["changed_within_control"]:
      raise ValueError("Frame does not have an unchanged matching model generation")
  for generation in generations.values():
    if set(generation["values"]) != set(report["model_layout"]):
      raise ValueError("Model field set differs from snapshot metadata")
    for name, value in generation["values"].items():
      layout = report["model_layout"][name]
      if list(value.shape) != layout["shape"] or str(value.dtype) != layout["dtype"]:
        raise ValueError(f"Model layout mismatch: {name}")
      if layout["world_axis"] not in (None, 0) or (layout["world_axis"] == 0 and
          (value.ndim == 0 or value.shape[0] != worlds)):
        raise ValueError(f"Incomplete model world axis: {name}")
  if set(payload["default_model_fields"]) != set(report["default_model_layout"]):
    raise ValueError("DR default field set mismatch")
  for name, value in payload["default_model_fields"].items():
    layout = report["default_model_layout"][name]
    if list(value.shape) != layout["shape"] or str(value.dtype) != layout["dtype"]:
      raise ValueError(f"DR default layout mismatch: {name}")
    if layout["world_axis"] not in (None, 0) or (layout["world_axis"] == 0 and
        (value.ndim == 0 or value.shape[0] != worlds)):
      raise ValueError(f"Incomplete DR default world axis: {name}")
  return dict(path=path, report=report, payload=payload, frames={entry["sequence"]: entry for entry in frames},
              generations=generations, model_path=model_path, tensor_path=tensor_path)


def select_frame(snapshot, selector="last-finite"):
  import torch
  candidates = []
  for frame in snapshot["frames"].values():
    generation = snapshot["generations"][frame["model_generation"]]
    if all(bool(torch.isfinite(value).all()) for value in list(frame["state"].values()) + list(generation["values"].values())):
      candidates.append(frame["sequence"])
  if selector == "last-finite" and not candidates:
    raise ValueError("No finite physics input frame exists")
  sequence = max(candidates) if selector == "last-finite" else int(selector)
  if sequence not in candidates:
    raise ValueError("Selected frame is absent or contains nonfinite physics input")
  frame = snapshot["frames"][sequence]
  return frame, snapshot["generations"][frame["model_generation"]]


def next_pre_step_reference(snapshot, frame):
  next_frame = snapshot["frames"].get(frame["sequence"] + 1)
  if next_frame is None:
    return None
  if (next_frame["control_step"] != frame["control_step"] or
      next_frame["model_generation"] != frame["model_generation"] or
      next_frame["physics_step"] != frame["physics_step"] + 1 or next_frame["substep"] != frame["substep"] + 1):
    return None
  return next_frame


def array_comparison(left, right, *, atol=1e-6, rtol=1e-5):
  left, right = np.asarray(left), np.asarray(right)
  if left.shape != right.shape:
    return dict(shape_equal=False, left_shape=list(left.shape), right_shape=list(right.shape), finite_and_close=False)
  a, b = np.isfinite(left), np.isfinite(right)
  common = a & b
  delta = np.abs(left[common].astype(np.float64) - right[common].astype(np.float64))
  close = np.zeros(left.shape, dtype=bool)
  close[common] = delta <= atol + rtol * np.abs(right[common].astype(np.float64))
  flat = close.reshape((left.shape[0], math.prod(left.shape[1:]))) if left.ndim else close.reshape(1, 1)
  return dict(shape_equal=True, shape=list(left.shape), left_finite=bool(a.all()), right_finite=bool(b.all()),
              nonfinite_pattern_equal=bool(np.array_equal(a, b)), finite_and_close=bool(close.all()),
              max_abs_difference=float(delta.max()) if delta.size else None,
              rms_difference=float(np.sqrt(np.mean(delta * delta))) if delta.size else None,
              differing_worlds=np.flatnonzero(~flat.all(axis=1)).tolist(), atol=atol, rtol=rtol)


def finite_summary(value):
  value = np.asarray(value)
  flat = np.isfinite(value).reshape(value.shape[0], math.prod(value.shape[1:]))
  return dict(shape=list(value.shape), dtype=str(value.dtype), all_finite=bool(flat.all()),
              nonfinite_worlds=np.flatnonzero(~flat.all(axis=1)).tolist())


def minimum_assignment(cost):
  """Hungarian assignment, square finite matrix; used within a geom pair only."""
  cost = np.asarray(cost, dtype=np.float64)
  n = cost.shape[0]
  if cost.shape != (n, n) or not np.isfinite(cost).all():
    raise ValueError("Contact assignment requires a finite square cost matrix")
  u, v = np.zeros(n + 1), np.zeros(n + 1)
  p, way = np.zeros(n + 1, dtype=int), np.zeros(n + 1, dtype=int)
  for i in range(1, n + 1):
    p[0], j0 = i, 0
    minimum, used = np.full(n + 1, np.inf), np.zeros(n + 1, dtype=bool)
    while True:
      used[j0] = True
      i0, delta, j1 = p[j0], np.inf, 0
      for j in range(1, n + 1):
        if not used[j]:
          current = cost[i0 - 1, j - 1] - u[i0] - v[j]
          if current < minimum[j]:
            minimum[j], way[j] = current, j0
          if minimum[j] < delta:
            delta, j1 = minimum[j], j
      for j in range(n + 1):
        if used[j]:
          u[p[j]] += delta
          v[j] -= delta
        else:
          minimum[j] -= delta
      j0 = j1
      if p[j0] == 0:
        break
    while j0:
      j1 = way[j0]
      p[j0], j0 = p[j1], j1
  pairs = np.empty(n, dtype=int)
  for j in range(1, n + 1):
    pairs[p[j] - 1] = j - 1
  return pairs


def contact_comparison(left, right, *, position_atol=1e-5, normal_atol=1e-4):
  """Preserve raw ID differences; separately compare unordered geometric contacts.

  Per-pair matching canonicalizes geom order and flips its normal accordingly.
  It does not re-label a changed geomcollisionid as an identical contact ID.
  """
  def groups(data):
    result, ids = defaultdict(list), Counter()
    if np.asarray(data["dist"]).ndim != 1:
      raise ValueError(f"Contact dist must be one-dimensional, got {np.asarray(data['dist']).shape}")
    count = len(data["dist"])
    optional_status = {}
    def integer_field(name, shape, empty_default=None):
      value = data.get(name)
      if empty_default is not None and (value is None or np.asarray(value).size == 0):
        optional_status[name] = "absent" if value is None else "unallocated_empty"
        return np.full(shape, empty_default, dtype=np.int64)
      value = np.asarray(value)
      if value.shape != shape or not np.issubdtype(value.dtype, np.integer):
        raise ValueError(f"Contact {name} layout mismatch: shape={value.shape}, dtype={value.dtype}; "
                         f"expected integer shape={shape}")
      if empty_default is not None:
        optional_status[name] = "per_contact"
      return value
    geom = integer_field("geom", (count, 2))
    worldid = integer_field("worldid", (count,))
    dimensions = integer_field("dim", (count,))
    collisionid = integer_field("geomcollisionid", (count,))
    kinds = integer_field("type", (count,), empty_default=0)
    # MJWarp leaves flex/elem/vert unallocated in rigid-only scenes, even when
    # rigid contacts exist. Treat these empty optional arrays like omitted
    # fields; a populated array must still have one entity pair per contact.
    extras = [integer_field(name, (count, 2), empty_default=-1) for name in ("flex", "elem", "vert")]
    for name, shape in (("pos", (count, 3)), ("frame", (count, 3, 3))):
      if np.asarray(data[name]).shape != shape:
        raise ValueError(f"Contact {name} layout mismatch: shape={np.asarray(data[name]).shape}; expected={shape}")
    signs = np.ones(count, dtype=np.float64)
    for i in range(len(data["dist"])):
      g0, g1 = (int(x) for x in geom[i])
      world, dim = int(worldid[i]), int(dimensions[i])
      kind = int(kinds[i])
      first = (g0, *(int(value[i, 0]) for value in extras))
      second = (g1, *(int(value[i, 1]) for value in extras))
      if first > second:
        signs[i] = -1.
      key = (world, min(first, second), max(first, second), dim, kind)
      result[key].append(i)
      ids[(world, first, second, dim, kind, int(collisionid[i]))] += 1
    return result, ids, signs, optional_status
  ag, ai, asign, aoptional = groups(left)
  bg, bi, bsign, boptional = groups(right)
  unmatched, differences = [], []
  peak_position = peak_distance = peak_normal = 0.0
  matched = 0
  invalid = False
  for key in sorted(set(ag) | set(bg)):
    aa, bb = ag[key], bg[key]
    if len(aa) != len(bb):
      unmatched.append(dict(key=list(key), left=len(aa), right=len(bb)))
      continue
    if not aa:
      continue
    apos, bpos = left["pos"][aa].astype(np.float64), right["pos"][bb].astype(np.float64)
    adist, bdist = left["dist"][aa].astype(np.float64), right["dist"][bb].astype(np.float64)
    anormal = left["frame"][aa, 0].astype(np.float64) * asign[aa, None]
    bnormal = right["frame"][bb, 0].astype(np.float64) * bsign[bb, None]
    if not all(np.isfinite(x).all() for x in (apos, bpos, adist, bdist, anormal, bnormal)):
      invalid = True
      differences.append(dict(key=list(key), nonfinite=True))
      continue
    pd = np.max(np.abs(apos[:, None] - bpos[None]), axis=-1) / position_atol
    dd = np.abs(adist[:, None] - bdist[None]) / position_atol
    nd = np.max(np.abs(anormal[:, None] - bnormal[None]), axis=-1) / normal_atol
    distance = np.maximum.reduce([pd, dd, nd])
    # Prefer an all-within-tolerance matching if one exists. A plain squared
    # distance minimum can choose a worse maximum error even when it exists.
    cost = (distance > 1).astype(float) + (distance / (1 + distance)) / (len(aa) + 1)
    permutation = minimum_assignment(cost)
    position = float(np.max(np.abs(apos - bpos[permutation])))
    distance = float(np.max(np.abs(adist - bdist[permutation])))
    normal = float(np.max(np.abs(anormal - bnormal[permutation])))
    peak_position, peak_distance, peak_normal = max(peak_position, position), max(peak_distance, distance), max(peak_normal, normal)
    matched += len(aa)
    if position > position_atol or distance > position_atol or normal > normal_atol:
      differences.append(dict(key=list(key), position_max=position, distance_max=distance, normal_max=normal))
  return dict(left_contacts=len(left["dist"]), right_contacts=len(right["dist"]), matched_contacts=matched,
              ordered_geom_ids_equal=bool(np.array_equal(left["geom"], right["geom"])),
              id_multiset_equal=ai == bi, id_entries_only_left=sum((ai - bi).values()), id_entries_only_right=sum((bi - ai).values()),
              optional_field_status=dict(left=aoptional, right=boptional),
              geometry_within_tolerance=not unmatched and not differences and not invalid,
              unequal_pair_count=len(unmatched), unequal_pair_examples=unmatched[:30],
              geometric_difference_pairs=len(differences), geometric_difference_examples=differences[:30],
              max_position_difference=peak_position, max_distance_difference=peak_distance, max_normal_difference=peak_normal,
              position_atol=position_atol, normal_atol=normal_atol)


def current_versions():
  import torch
  result = {"python": platform.python_version(), "torch": str(torch.__version__), "torch_cuda": torch.version.cuda}
  for package in ("mjlab", "mujoco", "mujoco-warp", "warp-lang"):
    try:
      result[package] = metadata.version(package)
    except metadata.PackageNotFoundError:
      result[package] = "unknown"
  return result


def provenance(snapshot):
  versions = current_versions()
  differences = {name: dict(saved=value, current=versions.get(name)) for name, value in snapshot["report"]["versions"].items()
                 if versions.get(name) != value}
  source_dir = Path(__file__).resolve().parents[1]
  paths = [Path(__file__), source_dir / "collision_batch.py", source_dir / "collision_cells.py", source_dir / "train.py",
           source_dir / "eval/failure_snapshot.py"]
  saved = snapshot["report"].get("source_artifacts", {})
  sources = {}
  for path in paths:
    expected = next((digest for name, digest in saved.items() if Path(name).name == path.name), None)
    actual = sha256(path)
    sources[str(path)] = dict(sha256=actual, saved_sha256=expected, matches_saved=actual == expected if expected else None)
  critical_source = any(record["matches_saved"] is False for name, record in sources.items()
                        if Path(name).name in ("collision_batch.py", "collision_cells.py"))
  return dict(versions=versions, version_differences=differences, sources=sources,
              critical_mismatch=critical_source or bool(differences))


def collision_execution(sim):
  from r1wb.eval.failure_snapshot import collision_execution_metadata
  return collision_execution_metadata(sim)


def install_variant(sim, variant, saved_execution=None):
  """Select existing collision code; restore recorded scratch budget without changing math."""
  if variant == "native":
    sim.create_graph()
    return
  if not sim.use_cuda_graph:
    raise ValueError("Adapted collision kernels require CUDA graphs; native fallback is not this comparison")
  from r1wb.collision_batch import install_collision_batching
  recorded_cells = (saved_execution or {}).get("cells") if variant == "optimized" else None
  if recorded_cells is None:
    install_collision_batching(sim, group_by_geom=variant == "optimized", parallel_cells=variant == "optimized")
    return
  from r1wb import collision_cells
  native_constructor = collision_cells.HeightfieldCells
  def restored_constructor(instance):
    cells = native_constructor(instance, max_pairs=recorded_cells["max_pairs"])
    if cells.conditional != recorded_cells["conditional"] or (cells.builders is not None) != recorded_cells["builders_supported"]:
      raise ValueError("Conditional-graph/builder support differs from recorded optimized execution")
    cells.budget = recorded_cells["budget"]
    return cells
  collision_cells.HeightfieldCells = restored_constructor
  try:
    install_collision_batching(sim, group_by_geom=True, parallel_cells=True)
  finally:
    collision_cells.HeightfieldCells = native_constructor


def construct_simulation(snapshot, frame, generation, variant, device):
  import torch
  import mujoco
  import warp as wp
  from mjlab.sim import Simulation, SimulationCfg
  from r1wb.eval.failure_snapshot import _tensor, restore_physics_inputs
  report = snapshot["report"]
  cfg = SimulationCfg(**report["simulation_config"])
  # SimulationCfg normally overwrites MJB options with defaults. Retain the MJB
  # verbatim; install saved Warp-only scalars before put_data chooses capacities.
  class PreserveMjbOptions:
    def apply(self, model):
      return None
  cfg.mujoco = PreserveMjbOptions()
  base_apply = cfg.apply_wp_opt
  def apply_options(options):
    base_apply(options)
    for name, value in generation["option_scalars"].items():
      if not hasattr(options, name):
        raise ValueError(f"Unknown saved physics option: {name}")
      setattr(options, name, value)
  cfg.apply_wp_opt = apply_options
  sim = Simulation(report["num_envs"], cfg, model=mujoco.MjModel.from_binary_path(str(snapshot["model_path"])), device=device)
  if bool(sim.use_cuda_graph) != bool(report["use_cuda_graph"]):
    raise ValueError("CUDA graph mode differs from the original snapshot")
  for name, expected in report["capacities"].items():
    if getattr(sim.wp_data, name, None) != expected:
      raise ValueError(f"Physics capacity mismatch: {name}, saved={expected}, current={getattr(sim.wp_data, name, None)}")
  sim.expand_model_fields(tuple(report["expanded_fields"]))
  # Tensor-valued options can be per-world independently of expanded model fields.
  option_storage = []
  for name, value in generation["values"].items():
    if name.startswith("option."):
      field = name.split(".", 1)[1]
      old = getattr(sim.wp_model.opt, field)
      if _tensor(getattr(sim.model.opt, field), shared_model=True).shape != value.shape:
        storage = value.to(device=device).contiguous()
        setattr(sim.wp_model.opt, field, wp.from_torch(storage, dtype=old.dtype))
        option_storage.append(storage)
  sim._replay_option_storage = option_storage
  sim.model.clear_cache()
  sim._default_model_fields = {name: value.to(device=device).clone() for name, value in snapshot["payload"]["default_model_fields"].items()}
  sim._per_world_default_fields = {name for name, value in report["default_model_layout"].items() if value["world_axis"] == 0}
  install_variant(sim, variant, report.get("collision_execution"))
  # Restore AFTER every capture: neither initialization nor capture state is an input.
  restore_physics_inputs(sim, frame["state"], generation["values"])
  wp.synchronize_device(sim.wp_device)
  restored = {}
  for name, expected in frame["state"].items():
    actual = _tensor(getattr(sim.data, name)).cpu()
    if not torch.equal(actual, expected):
      raise ValueError(f"Input state was not restored exactly: {name}")
    restored[name] = list(actual.shape)
  for name, expected in generation["values"].items():
    prefix, field = name.split(".", 1)
    container = sim.model.opt if prefix == "option" else sim.model
    if not torch.equal(_tensor(getattr(container, field), shared_model=True).cpu(), expected):
      raise ValueError(f"Input model was not restored exactly: {name}")
  return sim, dict(state_shapes=restored, model_values_verified=True, no_forward_after_restore=True)


def run_worker(args):
  import torch
  import warp as wp
  from r1wb.eval.failure_snapshot import _tensor
  snapshot = load_snapshot(args.snapshot)
  frame, generation = select_frame(snapshot, args.frame)
  evidence = provenance(snapshot)
  if evidence["critical_mismatch"] and not args.allow_provenance_mismatch:
    raise ValueError("Source/library versions differ; see --inspect-only or explicitly allow a diagnostic mismatch")
  if not args.device.startswith("cuda"):
    raise ValueError("Physical collision comparison requires a CUDA device; CPU tests use mocks/--inspect-only")
  torch.set_num_threads(1)
  if torch.cuda.get_device_capability(args.device)[0] >= 10:
    wp.config.ptx_target_arch, wp.config.cuda_output = 90, "ptx"
  sim, restored = construct_simulation(snapshot, frame, generation, args.variant, args.device)
  before = collision_execution(sim)
  saved_execution = snapshot["report"].get("collision_execution")
  layout_matches = before == saved_execution if saved_execution is not None else None
  began = time.perf_counter()
  sim.step()
  wp.synchronize_device(sim.wp_device)
  elapsed = time.perf_counter() - began
  output = {name: _tensor(getattr(sim.data, name)).cpu().numpy().copy() for name in STATE_OUTPUTS}
  nacon = int(_tensor(sim.data.nacon).reshape(-1)[0].cpu())
  capacity = int(sim.wp_data.naconmax)
  count = max(0, min(nacon, capacity))
  for name in CONTACT_FIELDS:
    output["contact_" + name] = _tensor(getattr(sim.data.contact, name))[:count].cpu().numpy().copy()
  output["nacon"] = np.array(nacon)
  output["naconmax"] = np.array(capacity)
  out = Path(args.out)
  out.mkdir(parents=True, exist_ok=False)
  target = out / "output.npz"
  np.savez_compressed(target, **output)
  model = sim.mj_model
  flags = dict(disableflags=int(model.opt.disableflags), enableflags=int(model.opt.enableflags),
               warp_disableflags=int(sim.wp_model.opt.disableflags), warp_enableflags=int(sim.wp_model.opt.enableflags),
               run_collision_detection=bool(sim.wp_model.opt.run_collision_detection),
               geom_contype=model.geom_contype.tolist(), geom_conaffinity=model.geom_conaffinity.tolist(),
               geom_condim=model.geom_condim.tolist(), geom_names=[model.geom(i).name for i in range(model.ngeom)])
  report = dict(schema="r1wb-one-step-physics-replay-v1", variant=args.variant, device=args.device,
                input_snapshot_sha256=sha256(args.snapshot), input_tensor_sha256=sha256(snapshot["tensor_path"]),
                input_model_sha256=sha256(snapshot["model_path"]), sequence=frame["sequence"],
                model_generation=frame["model_generation"], num_envs=snapshot["report"]["num_envs"],
                capacities=snapshot["report"]["capacities"], restored_inputs=restored, execution=before,
                original_execution_metadata=snapshot["report"].get("collision_execution"),
                original_workspace_layout_known="collision_execution" in snapshot["report"],
                execution_layout_matches_original=layout_matches,
                collision_flags=flags, step_seconds=elapsed, step_timing_is_not_a_benchmark=True,
                finite={name: finite_summary(output[name]) for name in STATE_OUTPUTS},
                contacts=dict(reported=nacon, stored=count, capacity=capacity, overflow=nacon > capacity or nacon < 0),
                provenance=evidence, output=dict(path=str(target), sha256=sha256(target)), limitations=LIMITATIONS)
  write_json(out / "result.json", report)


def load_output(path):
  with np.load(path, allow_pickle=False) as data:
    return {name: data[name] for name in data.files}


def compare_outputs(left, right, args):
  result = {name: array_comparison(left[name], right[name], atol=args.state_atol, rtol=args.state_rtol) for name in STATE_OUTPUTS
            if name in left and name in right}
  if "contact_geom" in left and "contact_geom" in right:
    result["contacts"] = contact_comparison(
      {key.removeprefix("contact_"): value for key, value in left.items() if key.startswith("contact_")},
      {key.removeprefix("contact_"): value for key, value in right.items() if key.startswith("contact_")},
      position_atol=args.contact_atol, normal_atol=args.normal_atol)
    result["contacts"]["both_capacities_valid"] = all(0 <= int(data["nacon"]) <= int(data["naconmax"]) for data in (left, right))
  return result


def run_trial(args, sequence, variant, repeat, root):
  name = f"{variant}_{repeat}"
  output = root / name
  command = [sys.executable, "-B", "-m", "r1wb.eval.replay_physics", "--snapshot", str(Path(args.snapshot).resolve()),
             "--frame", str(sequence), "--out", str(output), "--device", args.device, "--worker", "--variant", variant]
  if args.allow_provenance_mismatch:
    command.append("--allow-provenance-mismatch")
  with (root / f"{name}.log").open("w") as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    try:
      code = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
      os.killpg(process.pid, signal.SIGTERM)
      try:
        process.wait(timeout=5)
      except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
      return dict(name=name, status="timeout", command=command, timeout_seconds=args.timeout)
  result_path = output / "result.json"
  if code or not result_path.is_file():
    return dict(name=name, status="failed", returncode=code, command=command, log=str(root / f"{name}.log"))
  return dict(name=name, status="completed", command=command, report=str(result_path),
              output=str(output / "output.npz"), output_sha256=sha256(output / "output.npz"))


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--snapshot", type=Path, required=True, help="failure snapshot .json; companion MJB/PT hashes are verified")
  parser.add_argument("--out", type=Path, required=True, help="new output directory; never overwrites a trial")
  parser.add_argument("--frame", default="last-finite", help="last-finite or saved sequence number")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--original-execution", choices=VARIANTS, default="optimized", help="asserted producer collision mode; validated first")
  parser.add_argument("--repeats", type=int, default=2, help="independent restores for every mode, minimum 2")
  parser.add_argument("--timeout", type=float, default=600, help="bounded seconds per isolated trial")
  parser.add_argument("--expected-output", type=Path, help="optional independently captured post-step NPZ with qpos/qvel/qacc")
  parser.add_argument("--state-atol", type=float, default=1e-6)
  parser.add_argument("--state-rtol", type=float, default=1e-5)
  parser.add_argument("--contact-atol", type=float, default=1e-5)
  parser.add_argument("--normal-atol", type=float, default=1e-4)
  parser.add_argument("--allow-provenance-mismatch", action="store_true", help="diagnostic only; disables original-transition validation claims")
  parser.add_argument("--inspect-only", action="store_true", help="CPU snapshot/layout/provenance inspection, no simulation construction")
  parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
  parser.add_argument("--variant", choices=VARIANTS, default="native", help=argparse.SUPPRESS)
  args = parser.parse_args(argv)
  if args.worker:
    run_worker(args)
    return
  numeric = (args.timeout, args.state_atol, args.state_rtol, args.contact_atol, args.normal_atol)
  if (not all(math.isfinite(value) for value in numeric) or args.repeats < 2 or args.timeout <= 0 or
      min(args.state_atol, args.contact_atol, args.normal_atol) <= 0 or args.state_rtol < 0):
    parser.error("Need at least 2 independent repeats, positive timeout/tolerances, nonnegative state rtol")
  snapshot = load_snapshot(args.snapshot)
  frame, generation = select_frame(snapshot, args.frame)
  evidence = provenance(snapshot)
  root = args.out.resolve()
  root.mkdir(parents=True, exist_ok=False)
  report = dict(schema="r1wb-physics-replay-comparison-v1", created_at=datetime.now(timezone.utc).isoformat(),
                snapshot=str(snapshot["path"]), snapshot_sha256=sha256(snapshot["path"]), sequence=frame["sequence"],
                model_generation=generation["generation"], num_envs=snapshot["report"]["num_envs"],
                selected_frame={name: frame[name] for name in ("control_step", "physics_step", "substep")},
                recorded_execution=snapshot["report"].get("collision_execution"),
                original_execution_asserted=args.original_execution, complete_world_count_preserved=True,
                provenance=evidence, limitations=LIMITATIONS, trials=[], comparisons={},
                original_transition_validated=False, conclusion="No physics trial has run")
  write_json(root / "summary.json", report)
  if args.inspect_only:
    report["conclusion"] = "CPU input inspection only; no physics replay"
    write_json(root / "summary.json", report)
    print(json.dumps(dict(inspected=True, sequence=frame["sequence"], num_envs=report["num_envs"], provenance=evidence), default=str))
    return
  if evidence["critical_mismatch"] and not args.allow_provenance_mismatch:
    raise ValueError("Snapshot/current provenance mismatch; see summary.json")
  reference = None
  if args.expected_output is not None:
    reference = load_output(args.expected_output)
    if not {"qpos", "qvel", "qacc"} <= reference.keys():
      raise ValueError("Expected post-step output needs qpos, qvel and qacc")
    report["reference"] = dict(kind="caller_supplied_post_step", path=str(args.expected_output), sha256=sha256(args.expected_output))
  else:
    next_frame = next_pre_step_reference(snapshot, frame)
    if next_frame is not None:
      reference = {name: next_frame["state"][name].numpy() for name in STATE_OUTPUTS}
      report["reference"] = dict(kind="conditional_next_pre_step", sequence=next_frame["sequence"],
                                 limitation="Not an independent post-step truth; interposed actuator/sensor/metric callbacks are omitted")
    else:
      report["reference"] = dict(kind="none", reason="No same-control consecutive pre-step; exception_state is not used as truth")
  order = [(args.original_execution, 0)] + [(variant, repeat) for variant in VARIANTS for repeat in range(args.repeats)
                                          if (variant, repeat) != (args.original_execution, 0)]
  outputs = {}
  for index, (variant, repeat) in enumerate(order):
    print(f"replay sequence={frame['sequence']} {variant} repeat={repeat}, worlds={report['num_envs']}", flush=True)
    trial = run_trial(args, frame["sequence"], variant, repeat, root)
    report["trials"].append(trial)
    if trial["status"] == "completed":
      outputs[trial["name"]] = load_output(trial["output"])
      worker = json.loads(Path(trial["report"]).read_text())
      trial["execution_layout_matches_original"] = worker["execution_layout_matches_original"]
      trial["contacts_overflow"] = worker["contacts"]["overflow"]
      # Preserve completed raw trial provenance even if later analysis rejects
      # an unexpected output layout. Analysis must never obscure a saved run.
      write_json(root / "summary.json", report)
      if index == 0 and reference is not None:
        report["original_vs_reference"] = compare_outputs(outputs[trial["name"]], reference, args)
        report["original_transition_validated"] = bool(
          report["reference"]["kind"] == "caller_supplied_post_step" and not evidence["critical_mismatch"] and
          not args.allow_provenance_mismatch and
          worker["execution_layout_matches_original"] is True and not worker["contacts"]["overflow"] and
          all(report["original_vs_reference"][name]["finite_and_close"] for name in ("qpos", "qvel", "qacc")))
    write_json(root / "summary.json", report)
    if index == 0 and trial["status"] != "completed":
      report["conclusion"] = "Original-mode replay failed to execute; alternative comparison was not started"
      write_json(root / "summary.json", report)
      return
  for left, right in [(f"{variant}_0", f"{variant}_1") for variant in VARIANTS] + [
      ("native_0", "compacted_0"), ("native_0", "optimized_0"), ("compacted_0", "optimized_0")]:
    if left in outputs and right in outputs:
      report["comparisons"][f"{left}_vs_{right}"] = compare_outputs(outputs[left], outputs[right], args)
  report["conclusion"] = ("Independent one-step differences and self-noise measured. "
                           "These results alone do not attribute the training failure to a collision algorithm.")
  write_json(root / "summary.json", report)
  print(str(root / "summary.json"))


if __name__ == "__main__":
  main()
