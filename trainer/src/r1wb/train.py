"""Training loop of the whole-body family. Launch and mandatory run registration stay in r1lab.train
(`scripts/01_train.sh <task>`), which dispatches here for tasks whose registry entry names this runner."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time


def start_policy(actor, checkpoint: str, device, noise: float | None = None, pad_inputs: bool = False) -> None:
  """The trained policy continues from a checkpoint's actor: weights, exploration noise and observation normaliser.
  The normaliser is frozen from here on, so that the policy and its frozen anchor keep reading the same inputs
  (hits would otherwise shift the statistics and with them the calm outputs). The critic starts new.
  noise: replace the checkpoint's exploration noise by this value on every joint.
  pad_inputs: the policy reads more inputs than the checkpoint (appended at the end): their first-layer weights
  start at zero and they pass the normaliser unchanged, so the policy starts as exactly the checkpoint's function."""
  import torch
  state = dict(torch.load(checkpoint, map_location=device, weights_only=False)["actor_state_dict"])
  own = actor.state_dict()
  first = min((key for key in state if key.startswith("mlp.") and key.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
  extra = own[first].shape[1] - state[first].shape[1]
  if extra > 0:
    if not pad_inputs:
      raise ValueError(f"The policy reads {extra} more inputs than the start checkpoint and the task does not allow padding")
    rows = state[first].shape[0]
    state[first] = torch.cat([state[first], torch.zeros(rows, extra, device=state[first].device)], dim=1)
    for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
      state[key] = torch.cat([state[key], torch.full((1, extra), fill, device=state[key].device)], dim=1)
  last = max((key for key in state if key.startswith("mlp.") and key.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
  rows = state[last].shape[0]
  if own[last].shape[0] == 2 * rows and "distribution.std_param" in state:
    # The policy outputs its exploration noise per state (second half of the last layer): the mean rows are the
    # checkpoint's, the noise rows start as the checkpoint's per-joint noise, independent of the state.
    noise = state.pop("distribution.std_param").clamp_min(1e-3)
    bias = last.replace("weight", "bias")
    start = noise.log() if actor.distribution.std_type == "log" else noise
    state[last] = torch.cat([state[last], torch.zeros_like(state[last])], dim=0)
    state[bias] = torch.cat([state[bias], start], dim=0)
  missing, unexpected = actor.load_state_dict(state, strict=False)
  if missing or unexpected:
    raise ValueError(f"The start checkpoint does not fit the policy network: missing {missing}, unexpected {unexpected}")
  actor.obs_normalizer.until = 0
  if noise is not None:
    with torch.no_grad():
      actor.distribution.std_param.fill_(noise)


def widened(checkpoint: str, runner, directory: Path, device) -> str:
  """--resume from a checkpoint whose networks read fewer inputs than this run's (inputs appended at the end): a copy
  in which the first layers have zero weights for the new inputs and the normalisers pass them unchanged, so that the
  policy AND the value function start as exactly the checkpoint's functions, with the optimiser's state kept (Adam's
  first moment for the new weights zero, its second moment that layer's mean: their first steps are of ordinary
  size). Returns the checkpoint itself when nothing has to be widened."""
  import os

  import torch
  loaded = torch.load(checkpoint, map_location=device, weights_only=False)
  changed = False
  for part, model in (("actor_state_dict", getattr(runner.alg, "_raw_actor", runner.alg.actor)),
                      ("critic_state_dict", getattr(runner.alg, "_raw_critic", runner.alg.critic))):
    state, own = loaded[part], model.state_dict()
    first = min((key for key in state if key.startswith("mlp.") and key.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
    extra = own[first].shape[1] - state[first].shape[1]
    if extra < 0:
      raise ValueError(f"The resumed checkpoint's {part} reads {-extra} more inputs than this task")
    if extra == 0:
      continue
    changed = True
    state[first] = torch.cat([state[first], torch.zeros(state[first].shape[0], extra, device=state[first].device)], dim=1)
    for key, fill in (("obs_normalizer._mean", 0.0), ("obs_normalizer._var", 1.0), ("obs_normalizer._std", 1.0)):
      state[key] = torch.cat([state[key], torch.full((1, extra), fill, device=state[key].device)], dim=1)
  if not changed:
    return checkpoint
  moments = loaded["optimizer_state_dict"]["state"]
  parameters = [parameter for group in runner.alg.optimizer.param_groups for parameter in group["params"]]
  for index, parameter in enumerate(parameters):
    state = moments.get(index)
    if state is None or state["exp_avg"].shape == parameter.shape:
      continue
    old = state["exp_avg"].shape
    if parameter.dim() != 2 or old[0] != parameter.shape[0] or old[1] > parameter.shape[1]:
      raise ValueError(f"The resumed optimiser state does not fit parameter {index}: {tuple(old)} against {tuple(parameter.shape)}")
    extra = parameter.shape[1] - old[1]
    state["exp_avg"] = torch.cat([state["exp_avg"], torch.zeros(old[0], extra, device=state["exp_avg"].device)], dim=1)
    state["exp_avg_sq"] = torch.cat([state["exp_avg_sq"], torch.full((old[0], extra), float(state["exp_avg_sq"].mean()),
                                                                      device=state["exp_avg_sq"].device)], dim=1)
  target = Path(directory) / f".resume_widened_{os.getpid()}.pt"
  torch.save(loaded, target)
  return str(target)


def resume_training(runner, checkpoint: str, directory: Path, device, *, pad_inputs: bool = False,
                    freeze_actor_normalizer: bool = False, next_iteration: bool = False, head_contract: bool = False) -> dict:
  """Load full training state; optionally continue after the iteration already saved in the checkpoint.

  Legacy runs reuse the saved iteration's label. Opting into the next label avoids repeating it, while the actual
  optimizer update count and learning-rate schedule always come from the saved algorithm state.
  """
  if head_contract:
    from r1wb.head_resume import prepare_resume
    source = prepare_resume(checkpoint, runner, directory, device)
  else:
    source = widened(checkpoint, runner, directory, device) if pad_inputs else checkpoint
  try:
    infos = runner.load(source, map_location=device)
  finally:
    if source != checkpoint:
      Path(source).unlink(missing_ok=True)
  saved_iteration = int(runner.current_learning_iteration)
  if next_iteration:
    runner.current_learning_iteration = saved_iteration + 1
  if freeze_actor_normalizer:
    runner.alg.actor.obs_normalizer.until = 0
  return {"resume_widened": source != checkpoint, "resume_checkpoint_iteration": saved_iteration,
          "resume_source_world_size": (infos or {}).get("world_size"),
          "resume_source_global_statistics": getattr(runner.alg, "loaded_global_statistics", False)}


def record_training_failure(error, record, env, runner, context, *, max_envs: int = 8,
                            max_tensor_bytes: int = 4 * 1024 * 1024) -> None:
  """Persist rank failure before cleanup; bounded evidence only, never a resumable checkpoint.

  Reads the failing traceback's observations (or the existing observation cache), without another environment
  step, observation computation or collective. CUDA failures may prevent tensor reads; the JSON header and rank
  status are written first. Model/Adam values stay on their device; only finite flags and selected envs are saved.
  """
  import os
  import math
  import traceback

  import torch
  from r1lab.run_record import atomic_json

  directory = Path(record.directory)
  stem = f"failure_rank{context.rank}_{time.time_ns()}"
  report_path, tensor_path = directory / f"{stem}.json", directory / f"{stem}.pt"
  errors = []

  def attempt(name, function):
    try:
      return function()
    except Exception as problem:
      errors.append({"stage": name, "error": f"{type(problem).__name__}: {problem}"})
      return None

  trace_values, trace_source, iteration = {}, None, None
  tb = error.__traceback__
  while tb is not None:
    values = tb.tb_frame.f_locals
    if "obs" in values:
      trace_values.update({key: values[key] for key in ("obs", "rewards", "dones", "actions") if key in values})
      trace_source = {"file": tb.tb_frame.f_code.co_filename, "line": tb.tb_lineno,
                      "function": tb.tb_frame.f_code.co_name}
    if tb.tb_frame.f_code.co_name == "learn" and isinstance(values.get("it"), int):
      iteration = values["it"]
    tb = tb.tb_next
  real = getattr(env, "unwrapped", env) if env is not None else None
  alg = getattr(runner, "alg", None)
  report = {"schema_version": 1, "rank": context.rank, "world_size": context.world_size, "pid": os.getpid(),
            "created_at": time.time(), "stage": "before_cleanup", "capture_complete": False,
            "error": f"{type(error).__name__}: {error}", "traceback": "".join(traceback.format_exception(error)),
            "iteration_in_failing_learn": iteration,
            "last_completed_iteration": getattr(runner, "current_learning_iteration", None),
            "update_count": getattr(alg, "update_count", None),
            "common_step_counter": getattr(real, "common_step_counter", None),
            "observation_source": trace_source, "capture_errors": errors,
            "limits": {"max_saved_envs": max_envs, "max_tensor_bytes": max_tensor_bytes,
                       "max_reported_env_ids_per_field": 4096},
            "run": {key: record.info.get(key) for key in ("id", "task", "seed", "num_envs", "num_envs_per_gpu",
                    "execution", "global_statistics", "failure_replay_buffer", "resume", "resume_sha256", "resume_next_iteration",
                    "checkpoint", "start_iteration", "total_iterations")},
            "limitations": ["Not a resumable checkpoint or proof of the root cause.",
                            "State is the current state at the exception, possibly after automatic reset.",
                            "Finite rewards do not rule out invalid term values masked by RewardManager.",
                            "Contact sensor match capacity and physical contact capacity are distinct."]}
  attempt("rank_status", lambda: record.update(state="failed", finished_at=time.time(), error=report["error"],
                                                failure_diagnostics=report_path.name, failure_iteration=iteration))
  attempt("initial_json", lambda: atomic_json(report_path, report))
  replay = getattr(real, "_failure_replay_buffer", None)
  if replay is not None:
    report["physics_replay"] = attempt("physics_replay", lambda: replay.export(stem))
  report["config_artifacts"] = {}
  for name in ("env.yaml", "agent.yaml", "source_snapshot.json", "library.json"):
    path = directory / "params" / name
    if path.is_file():
      report["config_artifacts"][name] = {"path": str(path), "sha256": attempt(name, lambda p=path: hashlib.sha256(p.read_bytes()).hexdigest())}
  if real is not None:
    report["rank_seed"] = getattr(getattr(real, "cfg", None), "seed", None)
    report["control_dt"] = getattr(real, "step_dt", None)
  finite = report["finite_flags"] = {}

  def finite_state(name, state):
    flags = {}
    for key, value in state.items():
      if isinstance(value, torch.Tensor):
        flags[str(key)] = bool(torch.isfinite(value.detach()).all().item())
    finite[name] = {"all_finite": all(flags.values()) if flags else None, "tensor_count": len(flags), "tensors": flags}

  if alg is not None:
    for name in ("actor", "critic"):
      model = getattr(alg, f"_raw_{name}", getattr(alg, name, None))
      if model is not None:
        attempt(f"finite_{name}", lambda n=name, m=model: finite_state(n, m.state_dict()))
    optimizer = getattr(alg, "optimizer", None)
    if optimizer is not None:
      attempt("finite_optimizer", lambda: finite_state("optimizer", {
        f"{index}.{key}": value for index, state in enumerate(optimizer.state.values()) for key, value in state.items()}))
      report["optimizer_learning_rates"] = [repr(group.get("lr")) for group in optimizer.param_groups]
      report["optimizer_learning_rates_finite"] = attempt("finite_learning_rates", lambda: [
        math.isfinite(float(group["lr"])) for group in optimizer.param_groups])
  manager = getattr(real, "observation_manager", None)
  obs = trace_values.get("obs")
  if obs is None and manager is not None:
    obs = getattr(manager, "_obs_buffer", None)  # Never compute fresh observations while diagnosing a failure.
    report["observation_source"] = "existing observation_manager._obs_buffer"
  report["observation_layout"] = {"terms": getattr(manager, "active_terms", None),
                                  "dimensions": getattr(manager, "group_obs_term_dim", None)}
  fields = {}
  if obs is not None and hasattr(obs, "items"):
    fields.update({f"observations.{key}": value for key, value in obs.items()})
  fields.update({name: value for name, value in trace_values.items() if name != "obs"})
  sim = getattr(real, "sim", None)
  sim_cfg = getattr(sim, "cfg", None)
  report["simulation_capacities"] = {
    "nconmax": getattr(sim_cfg, "nconmax", None), "njmax": getattr(sim_cfg, "njmax", None),
    "contact_sensor_maxmatch": getattr(sim_cfg, "contact_sensor_maxmatch",
                                      getattr(getattr(sim_cfg, "mujoco", None), "contact_sensor_maxmatch", None))}
  for source, names in (("data", ("qpos", "qvel", "qacc", "ctrl", "actuator_force", "qfrc_actuator", "xfrc_applied")),
                        ("model", ("body_mass", "body_inertia", "dof_damping", "actuator_gainprm", "actuator_biasprm"))):
    container = getattr(sim, source, None)
    if container is not None:
      for name in names:
        value = attempt(f"read_{source}.{name}", lambda c=container, n=name: getattr(c, n))
        if value is not None:
          fields[f"{source}.{name}"] = value
  for name in ("episode_length_buf", "reset_buf", "reset_terminated", "reset_time_outs"):
    value = getattr(real, name, None)
    if value is not None:
      fields[name] = value
  count = getattr(real, "num_envs", None)
  bad_ids, tensors = set(), {}
  report["nonfinite_envs"] = {}
  for name, value in fields.items():
    value = getattr(value, "_tensor", value)  # mjlab's zero-copy TorchArray bridge.
    if not isinstance(value, torch.Tensor):
      continue
    tensors[name] = value
    if value.ndim > 0 and value.shape[0] == count:
      def inspect_rows(v=value, n=name):
        ids = (~torch.isfinite(v.detach()).reshape(count, -1).all(dim=1)).nonzero().flatten()
        reported = ids[:4096].cpu().tolist()
        report["nonfinite_envs"][n] = {"count": ids.numel(), "ids": reported, "truncated": ids.numel() > 4096}
        bad_ids.update(reported)
      attempt(f"nonfinite_{name}", inspect_rows)
  selected = sorted(bad_ids)[:max_envs] if bad_ids else ([0] if count else [])
  report["saved_env_ids"] = selected
  report["selection"] = "first nonfinite envs" if bad_ids else "no nonfinite env identified; env0 diagnostic sample"
  payload, total_bytes = {}, 0
  report["tensor_fields"] = {}
  for name, value in tensors.items():
    def capture(v=value, n=name):
      nonlocal total_bytes
      per_env = v.ndim > 0 and v.shape[0] == count
      elements = (v[0].numel() * len(selected)) if per_env and count else v.numel()
      size = elements * v.element_size()
      if total_bytes + size > max_tensor_bytes:
        report["tensor_fields"][n] = {"skipped": "tensor byte budget", "bytes": size}
        return
      sample = (v.detach().index_select(0, torch.tensor(selected, device=v.device, dtype=torch.long))
                if per_env else v.detach())  # Shared model arrays have no environment dimension.
      payload[n] = sample.cpu().clone()
      total_bytes += size
      report["tensor_fields"][n] = {"shape": list(sample.shape), "dtype": str(sample.dtype), "bytes": size}
    attempt(f"capture_{name}", capture)
  if payload:
    temporary = tensor_path.with_suffix(".pt.part")
    def save_tensors():
      try:
        torch.save(payload, temporary)
        temporary.replace(tensor_path)
      finally:
        temporary.unlink(missing_ok=True)
    attempt("save_tensors", save_tensors)
    if tensor_path.is_file():
      report["tensor_artifact"] = {"path": tensor_path.name, "tensor_bytes": total_bytes,
                                   "sha256": attempt("tensor_sha256", lambda: hashlib.sha256(tensor_path.read_bytes()).hexdigest())}
  report["capture_complete"] = not errors
  report["capture_finished"] = True
  attempt("final_json", lambda: atomic_json(report_path, report))


def train(args, record):
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.runner import MjlabOnPolicyRunner
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends
  from r1lab.distributed import configure_local_device, distributed_context
  from r1lab.run_record import atomic_json
  from r1lab.train import ensure_robot_self_collision, snapshot_sources
  from r1wb.tasks import TASKS

  unsupported = [flag for flag in ("amp_disc", "actor_only", "freeze_actor", "freeze_normalizer", "t0")
                 if getattr(args, flag, None)]
  if unsupported:
    raise ValueError("Not used by the whole-body runner: " + ", ".join(unsupported))
  context = distributed_context()
  execution = getattr(args, "execution", "optimized")
  optimized = execution != "standard"
  device = configure_local_device(context)
  torch.backends.cuda.matmul.allow_tf32 = True
  configure_torch_backends()
  task = TASKS[args.task]
  if task.get("head_contract"):
    if not args.resume or getattr(args,"start_from",None):
      raise ValueError("Stage H requires --resume from the selected completed scan teacher or a head26 checkpoint")
    from r1wb.head_task import contract_record
    record.update(head_contract=contract_record(task["head_perception"],task["head_base"]))
  if task.get("require_reward_ledger") and not getattr(args, "reward_ledger", False):
    raise ValueError("The recovery contrast requires --reward-ledger in both arms")
  teacher = None
  if task.get("teacher"):
    if not args.teacher:
      raise ValueError("This task needs --teacher <tracking teacher model_N.pt>")
    teacher = Path(args.teacher).resolve()
    if not teacher.is_file():
      raise ValueError(f"Teacher checkpoint not found: {teacher}")
  elif args.teacher:
    raise ValueError("--teacher is only for tasks that imitate a teacher")
  start_from = None
  if getattr(args, "start_from", None):
    if not task.get("init_from_teacher") or args.resume:
      raise ValueError("--start-from is for tasks that start from a policy checkpoint, without --resume")
    start_from = Path(args.start_from).resolve()
    if not start_from.is_file():
      raise ValueError(f"Start checkpoint not found: {start_from}")
  env_cfg = task["env"](teacher=str(teacher)) if teacher else task["env"]()
  from r1wb.reward_contract import require_no_hit_exemptions
  hit_time_contract = require_no_hit_exemptions(env_cfg)
  record.update(hit_time_contract=hit_time_contract)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed + context.rank * 100003
  agent = task["rl"]()
  agent.seed = args.seed
  if getattr(args, "global_statistics", False):
    if not hasattr(agent.algorithm, "global_statistics"):
      raise ValueError("--global-statistics requires a whole-body ImitationPPO task")
    agent.algorithm.global_statistics = True
  global_statistics = bool(getattr(agent.algorithm, "global_statistics", False))
  resume_next_iteration = bool(getattr(args, "resume_next_iteration", False))
  failure_replay_buffer = bool(getattr(args, "failure_replay_buffer", False))
  if resume_next_iteration and not args.resume:
    raise ValueError("--resume-next-iteration requires --resume")
  if hasattr(agent.algorithm, "anchor_checkpoint"):
    agent.algorithm.anchor_checkpoint = str(teacher)
    if getattr(agent.algorithm, "repeat_weight", 0.0) > 0.0:
      source = args.resume or start_from
      if not source:
        raise ValueError("This task anchors the arms of repeated-hit states to the checkpoint it continues from: "
                         "give --resume or --start-from")
      agent.algorithm.repeat_checkpoint = str(Path(source).resolve())
  if agent.experiment_name != record.directory.parent.name:
    raise ValueError("학습 설정과 등록된 실험 경로가 다릅니다. TASK_SPECS를 확인하세요.")
  if args.iters is not None:
    agent.max_iterations = args.iters
  if getattr(args, "mini_batches", None):
    # Several GPUs, each with the single-GPU run's environments: an iteration holds world_size times the data. With
    # world_size times the minibatches a gradient step (the ranks' average) sees what it saw before, and a run of N
    # single-GPU iterations is N / world_size iterations here.
    agent.algorithm.num_mini_batches = int(args.mini_batches)
  log_dir = record.directory
  # The app's automatic evaluation only knows the 810-input walking contract; these runs are evaluated by
  # the whole-body tools instead.
  record.update(iters=agent.max_iterations, render=False, pipeline="r1wb", stage=task["stage"],
                mini_batches=int(agent.algorithm.num_mini_batches), ranks=context.world_size,
                execution=execution, global_statistics=global_statistics, resume_next_iteration=resume_next_iteration,
                failure_replay_buffer=failure_replay_buffer)
  motion = env_cfg.commands.get("motion")
  if context.is_primary:
    snapshot_sources(log_dir)
    if motion is not None:
      path = Path(motion.motion_file)
      manifest = json.loads(path.with_suffix(".manifest.json").read_text())
      library = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "clips": len(manifest["clips"]), "seconds": manifest["seconds"], "protocol": manifest["protocol"]}
      atomic_json(log_dir / "params" / "library.json", {**library, "manifest": manifest})
      record.update(library=library)
    if args.resume:
      record.update(resume_sha256=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest())
    if teacher is not None:
      record.update(teacher_checkpoint=str(teacher), teacher_sha256=hashlib.sha256(teacher.read_bytes()).hexdigest())
    if start_from is not None:
      record.update(start_checkpoint=str(start_from), start_sha256=hashlib.sha256(start_from.read_bytes()).hexdigest())
  print(f"[INFO] id={record.info['id']} task={args.task} rank={context.rank}/{context.world_size} device={device} "
        f"envs_per_rank={args.num_envs} seed={env_cfg.seed} iters={agent.max_iterations} log={log_dir}", flush=True)

  class Runner(MjlabOnPolicyRunner):
    def save(self, path: str, infos=None) -> None:
      if not context.is_primary:
        return
      target = Path(path)
      temporary = target.with_name("." + target.name + ".part")
      try:
        symmetric = {"symmetrized": True} if getattr(self.alg.get_policy(), "symmetrized", False) else {}
        MjlabOnPolicyRunner.save(self, str(temporary), {**(infos or {}), "world_size": context.world_size,
                                                      "resume_next_iteration": resume_next_iteration,
                                                      "failure_replay_buffer": failure_replay_buffer, **symmetric,
                                                      **({"head_contract": "r1wb-head26-reference24-scan-v1"} if task.get("head_contract") else {})})
        temporary.replace(target)
      finally:
        temporary.unlink(missing_ok=True)
      iteration = int(target.stem.split("_")[-1])
      policy, error = target.with_name(f"policy_{iteration}.onnx"), None
      try:
        self.export_policy_to_onnx(str(target.parent), policy.name)
      except Exception as problem:  # the checkpoint is what later stages need; report and continue
        error = str(problem)
        print(f"[WARN] ONNX export failed: {problem}", flush=True)
      term = self.env.unwrapped.command_manager.get_term("motion") if motion is not None else None
      if term is not None and hasattr(term, "clip_failures"):
        names = [clip["name"] for clip in manifest["clips"]]
        failures, completions = term.clip_failures.tolist(), term.clip_completions.tolist()
        atomic_json(target.with_name(f"clip_outcomes_{iteration}.json"),
                    {"iteration": iteration, "scope": "episodes of rank 0 since the previous checkpoint",
                     "clips": [{"name": n, "failed": int(f), "finished": int(c)}
                               for n, f, c in zip(names, failures, completions)]})
        term.clip_failures.zero_()
        term.clip_completions.zero_()
      record.update(iteration=iteration, checkpoint=target.name, policy=policy.name if policy.is_file() else None,
                    export_error=error)

  env, runner, failure, replay_buffer, ledger = None, None, None, None, None
  try:
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    if task.get("head_contract"):
      from r1wb.head_task import validate_env
      record.update(head_env_contract=validate_env(env))
    if optimized:
      from r1wb.performance import install_read_cache
      install_read_cache(env)
    if execution in ("compacted", "optimized"):
      from r1wb.performance import install_fast_resets, install_reference_tables
      from r1wb.collision_batch import install_collision_batching
      install_fast_resets(env)
      install_reference_tables(env)
      install_collision_batching(env.sim, group_by_geom=execution == "optimized",
                                parallel_cells=execution == "optimized")
    record.update(self_collision=ensure_robot_self_collision(env))
    if task.get("wholebody_calm"):
      from r1wb import wholebody_calm
      wholebody_calm.install(env)
      record.update(wholebody_calm_contract=wholebody_calm.contract_record(task["calm_cell"]))
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    agent_cfg = asdict(agent)
    if optimized:
      from rsl_rl.utils import resolve_callable
      from r1wb.learning import ImitationPPO
      if issubclass(resolve_callable(agent_cfg["algorithm"]["class_name"]), ImitationPPO):
        agent_cfg["algorithm"]["defer_metrics"] = True
        from r1wb.student.observations import no_label
        group = env_cfg.observations.get(agent_cfg["algorithm"].get("bc_group", "bc"))
        label = next(iter(group.terms.values())) if group is not None and len(group.terms) == 1 else None
        if (label is not None and label.func is no_label and not group.enable_corruption
            and label.clip is None and not getattr(label, "modifiers", None)):
          # This explicit observation contract always supplies zero driven flags.
          agent_cfg["algorithm"]["teacher_driven"] = False
    if context.is_primary:
      dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
    runner = Runner(env, agent_cfg, str(log_dir), device)
    if task.get("zero_init"):
      # The trained network is a correction on a frozen base policy: it starts as exactly zero.
      last = [module for module in runner.alg.actor.mlp.modules() if isinstance(module, torch.nn.Linear)][-1]
      rows = env.num_actions            # the mean; a state-dependent noise head (further rows) keeps its own start
      with torch.no_grad():
        last.weight[:rows].zero_()
        last.bias[:rows].zero_()
    if task.get("init_from_teacher") and not args.resume:
      # --start-from: the policy's start weights come from another checkpoint of the same network (the anchor stays
      # the --teacher checkpoint); the critic and the learning-rate schedule start new either way.
      start_policy(runner.alg.actor, str(start_from or teacher), device, noise=task.get("start_noise"),
                   pad_inputs=bool(task.get("pad_inputs")))
      if task.get("upper_noise_scale"):
        # Exploration noise of the waist and arm joints (action indices 12..23) scaled down: S1-a's own values there
        # (0.25-0.49 action units) are larger than the arms' reaction to a moderate hit (about 0.3).
        with torch.no_grad():
          runner.alg.actor.distribution.std_param[12:] *= float(task["upper_noise_scale"])
    if args.resume:
      resume_info = resume_training(runner, args.resume, log_dir, device, pad_inputs=bool(task.get("pad_inputs")),
                                    freeze_actor_normalizer=bool(task.get("init_from_teacher")),
                                    next_iteration=resume_next_iteration, head_contract=bool(task.get("head_contract")))
      record.update(**resume_info)
    start = int(runner.current_learning_iteration)
    record.update(state="running", start_iteration=start, total_iterations=start + agent.max_iterations,
                  actor_observation_dimension=int(runner.alg.get_policy().obs_dim), started_at=time.time())
    native_update = runner.alg.update

    def update():
      result = native_update()
      if ledger is not None and task.get("strict_reward_ledger"):
        status = ledger.summary()
        quality = status["quality"]
        if (not ledger.active or status["errors"] or quality["nonfinite_classifications"]
            or quality["force_available_frames"] != status["steps_written"]
            or quality["sum_exact_frames"] != status["steps_written"]):
          raise RuntimeError("Required recovery ledger failed or stopped before the training horizon: " + str(status))
      if context.is_primary and runner.current_learning_iteration % 10 == 0:
        record.update(iteration=int(runner.current_learning_iteration))
      return result

    runner.alg.update = update
    if context.is_primary:
      from r1wb.performance import record_iteration_times
      record_iteration_times(runner, log_dir / "performance.jsonl")
    if failure_replay_buffer:
      from r1wb.eval.failure_snapshot import install_failure_replay_buffer
      replay_buffer = install_failure_replay_buffer(env.unwrapped, log_dir, context.rank, enabled=True)
      replay_buffer.metadata["run"] = {key: record.info.get(key) for key in
                                        ("id", "task", "seed", "execution", "ranks", "resume", "resume_sha256")}
      record.update(failure_replay_status=replay_buffer.status())
    if getattr(args, "reward_ledger", False):
      from r1wb.reward_ledger import LedgerOptions, PureVectorDiagnostic, attach
      raw_env = env.unwrapped
      compact = task.get("ledger_profile") == "compact_v2"
      if compact:
        from r1wb.recovery import DIAGNOSTIC_COLUMNS, diagnostic_values
        suffix = ".jsonl.gz"
        options = LedgerOptions(
          flush_every=24, max_steps=agent.max_iterations * agent.num_steps_per_env,
          max_file_bytes=128 * 1024**2, compression="gzip", compact=True,
          pure_vector_diagnostics={"recovery_risk": PureVectorDiagnostic(
            callback=lambda: diagnostic_values(raw_env), columns=DIAGNOSTIC_COLUMNS,
            compare_max=("tilt", "predicted_tilt"), valid_column="valid")})
        if task.get("head_contract"):
          from r1wb import head_rewards
          options = __import__("dataclasses").replace(options, pure_vector_diagnostics={
            **options.pure_vector_diagnostics, "head_view": PureVectorDiagnostic(
              callback=lambda: head_rewards.diagnostic_values(raw_env), columns=head_rewards.DIAGNOSTIC_COLUMNS,
              valid_column="valid")})
        if task.get("wholebody_calm"):
          from r1wb import wholebody_calm
          options = __import__("dataclasses").replace(options, pure_vector_diagnostics={
            **options.pure_vector_diagnostics, "wholebody_calm": PureVectorDiagnostic(
              callback=lambda: wholebody_calm.diagnostic_values(raw_env),
              columns=wholebody_calm.DIAGNOSTIC_COLUMNS, valid_column="history_valid")})
      else:
        diagnostics = {}
        if "recovery_risk" in env_cfg.rewards:
          from r1wb.recovery import recovery_invalid, recovery_risk
          diagnostics["recovery_risk"] = lambda: recovery_risk(raw_env)
          diagnostics["recovery_invalid"] = lambda: recovery_invalid(raw_env)
        suffix = ".jsonl"
        options = LedgerOptions(flush_every=24, max_steps=agent.max_iterations * agent.num_steps_per_env,
                                max_file_bytes=256 * 1024**2, pure_diagnostics=diagnostics)
      ledger = attach(raw_env, log_dir / f"reward_ledger_rank{context.rank}{suffix}", options)
      native_act = runner.alg.act

      def audited_act(observations):
        actions = native_act(observations)
        ledger.record_critic(runner.alg.transition.values, observation_step=int(raw_env.common_step_counter),
                             phase="pre_action")
        return actions

      runner.alg.act = audited_act
      record.update(reward_ledger={"enabled": True, "path_pattern": "reward_ledger_rank*" + suffix,
                                  "max_bytes_per_rank": options.max_file_bytes,
                                  "max_steps_per_rank": agent.max_iterations * agent.num_steps_per_env,
                                  "trace_envs_per_rank": min(4, raw_env.num_envs),
                                  "compression": options.compression, "compact": options.compact})
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
    record.update(iteration=int(runner.current_learning_iteration))
    if context.is_distributed:
      torch.distributed.barrier()
  except Exception as error:
    failure = error
    try:
      record_training_failure(error, record, env, runner, context)
    except BaseException as diagnostic_error:
      try:
        print(f"[WARN] Failure evidence could not be completed: {diagnostic_error}", flush=True)
      except Exception:
        pass
    raise
  finally:
    for close in (((ledger.close,) if ledger is not None else ())
                  + ((replay_buffer.close,) if replay_buffer is not None else ())
                  + ((env.close,) if env is not None else ())
                  + (lambda: torch.distributed.destroy_process_group() if torch.distributed.is_initialized() else None,)):
      try:
        close()
      except BaseException as cleanup_error:
        if failure is None:
          raise
        try:
          print(f"[WARN] Cleanup failed after {type(failure).__name__}: {cleanup_error}", flush=True)
        except Exception:
          pass
    if ledger is not None:
      status = ledger.summary()
      record.update(reward_ledger_status=status)
      if failure is None and task.get("strict_reward_ledger"):
        expected = agent.max_iterations * agent.num_steps_per_env
        if (not status["final_success"] or status["quality"]["critic_supplied_frames"] != expected
            or status["quality"]["vector_frames"].get("recovery_risk", 0) != expected):
          raise RuntimeError("Recovery training finished without its complete required ledger: " + str(status))
