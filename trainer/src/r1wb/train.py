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
  device = configure_local_device(context)
  torch.backends.cuda.matmul.allow_tf32 = True
  configure_torch_backends()
  task = TASKS[args.task]
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
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed + context.rank * 100003
  agent = task["rl"]()
  agent.seed = args.seed
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
                mini_batches=int(agent.algorithm.num_mini_batches), ranks=context.world_size)
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
        MjlabOnPolicyRunner.save(self, str(temporary), {**(infos or {}), "world_size": context.world_size, **symmetric})
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

  env = None
  try:
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    record.update(self_collision=ensure_robot_self_collision(env))
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    agent_cfg = asdict(agent)
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
      source = widened(args.resume, runner, log_dir, device) if task.get("pad_inputs") else args.resume
      runner.load(source, map_location=device)
      if source != args.resume:
        Path(source).unlink(missing_ok=True)
        if context.is_primary:
          record.update(resume_widened=True)
      if task.get("init_from_teacher"):
        runner.alg.actor.obs_normalizer.until = 0
    start = int(runner.current_learning_iteration)
    record.update(state="running", start_iteration=start, total_iterations=start + agent.max_iterations,
                  actor_observation_dimension=int(runner.alg.get_policy().obs_dim), started_at=time.time())
    native_update = runner.alg.update

    def update():
      result = native_update()
      if context.is_primary and runner.current_learning_iteration % 10 == 0:
        record.update(iteration=int(runner.current_learning_iteration))
      return result

    runner.alg.update = update
    runner.learn(num_learning_iterations=agent.max_iterations, init_at_random_ep_len=True)
    record.update(iteration=int(runner.current_learning_iteration))
    if context.is_distributed:
      torch.distributed.barrier()
  finally:
    if env is not None:
      env.close()
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()
