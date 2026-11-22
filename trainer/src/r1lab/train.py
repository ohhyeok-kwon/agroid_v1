"""Registered R1 training entry point; both direct and script launches use this path."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

from r1lab.distributed import distributed_context
from r1lab.registry import TASK_SPECS
from r1lab.run_record import RunRecord, atomic_json, work_root


def validate_random_teacher_resume(checkpoint):
  """Verify every actual ancestor through a fresh, registered random teacher."""
  runs = (work_root()/"runs"/"r1_careful_walk").resolve()
  origin = "random_no_manufacturer_weights"
  visited = set()

  def local_path(value):
    try:
      path = Path(value)
    except (TypeError, ValueError) as error:
      raise ValueError("Teacher continuation must use a registered local teacher checkpoint") from error
    # These are the two recorded server spellings, not arbitrary outside roots.
    for prefix in (Path("/rl-agent/.agent/runs/r1_careful_walk"),
                   Path(".agent/runs/r1_careful_walk")):
      if path.is_relative_to(prefix):
        path = runs/path.relative_to(prefix)
        break
    path = path.resolve()
    if not path.is_relative_to(runs):
      raise ValueError("Teacher continuation must use a registered local teacher checkpoint")
    return path

  def read_record(path):
    path = local_path(path)
    try:
      value = json.loads(path.read_text())
    except (OSError, ValueError) as error:
      raise ValueError(f"Missing or invalid teacher lineage record: {path}") from error
    if not isinstance(value, dict):
      raise ValueError(f"Invalid teacher lineage record: {path}")
    return value

  def file_hash(path):
    path = local_path(path)
    try:
      return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as error:
      raise ValueError(f"Missing teacher lineage file: {path}") from error

  def verify(value):
    path = local_path(value)
    if path in visited or len(visited) >= 64:
      raise ValueError("Cyclic or excessive teacher lineage")
    visited.add(path)
    checkpoint_sha = file_hash(path)
    source = read_record(path.parent/"run_info.json")
    initial = read_record(path.parent/"params"/"initialization.json")
    spec = TASK_SPECS.get(source.get("task"), {})
    recorded = source.get("initialization", {})
    if (path.parent.parent != runs or source.get("id") != path.parent.name
        or spec.get("evaluation_profile") != "reference_teacher_v1"
        or source.get("state") != "completed" or path.name != source.get("checkpoint")
        or not path.name.startswith("model_") or not path.stem[6:].isdigit() or path.suffix != ".pt"
        or source.get("amp_disc") is not None
        or any(source.get(flag) for flag in ("actor_only", "freeze_actor", "freeze_normalizer"))
        or not isinstance(recorded, dict) or initial.get("mode") != recorded.get("mode")):
      raise ValueError("Teacher continuation requires a completed final reference-teacher checkpoint")
    if source.get("iteration", int(path.stem[6:])) != int(path.stem[6:]):
      raise ValueError("Teacher checkpoint is not the completed final iteration")
    reference_sha = file_hash(path.parent/"params"/"motion_reference.npz")
    contract = initial.get("observation_contract", {})
    reference_record = source.get("reference_teacher", {})
    if (not isinstance(contract, dict) or not isinstance(reference_record, dict)
        or reference_sha != contract.get("reference_sha256")
        or reference_record.get("sha256", reference_sha) != reference_sha):
      raise ValueError("Parent teacher reference provenance mismatch")
    mode = initial.get("mode")
    count = initial.get("optimizer_state_entries")
    if mode == "random":
      if (source.get("task") != "careful_walk_teacher" or source.get("resume") is not None
          or recorded.get("pretrained_weights") != [] or initial.get("pretrained_weight_files") != []
          or type(count) is not int or count != 0
          or recorded.get("parent") is not None or initial.get("parent") is not None):
        raise ValueError("Teacher lineage must end at a fresh random-only careful_walk_teacher")
      lineage = []
    elif mode == "continued_random_teacher" and spec.get("continue_random_teacher"):
      parents = (recorded.get("parent"), initial.get("parent"))
      if any(not isinstance(parent, dict) for parent in parents) or type(count) is not int or count <= 0:
        raise ValueError("Missing teacher continuation parent or restored optimizer provenance")
      parent_path = local_path(parents[0].get("checkpoint", ""))
      parent = verify(parent_path)
      for link in parents:
        if (local_path(link.get("checkpoint", "")) != parent_path
            or any(link.get(key) != parent[key] for key in
                   ("source_run", "checkpoint_sha256", "reference_sha256", "origin"))):
          raise ValueError("Teacher parent lineage evidence mismatch")
      if (local_path(source.get("resume") or "") != parent_path
          or source.get("resume_sha256") != parent["checkpoint_sha256"]):
        raise ValueError("Teacher resume or reference hash differs from its verified parent")
      if reference_sha != parent["reference_sha256"]:
        # Only registered reference-transition tasks may change the reference,
        # and both records must name exactly this parent->child transition.
        expected = {"parent_reference_sha256": parent["reference_sha256"],
                    "reference_sha256": reference_sha}
        if (not spec.get("reference_transition")
            or initial.get("reference_transition") != expected
            or recorded.get("reference_transition") != expected):
          raise ValueError("Teacher resume or reference hash differs from its verified parent")
      for files in (recorded.get("pretrained_weights"), initial.get("pretrained_weight_files")):
        if not isinstance(files, list) or len(files) != 1 or local_path(files[0]) != parent_path:
          raise ValueError("Teacher continuation must load only its verified parent checkpoint")
      parent_models = parent["lineage"][-1]["model_checksums"]
      if any(not parent_models.get(key) or initial.get("initial_model_checksums", {}).get(key)
             != parent_models[key] for key in ("actor", "critic")):
        raise ValueError("Teacher initial model checksums differ from its verified parent")
      lineage = parent["lineage"]
    else:
      raise ValueError("Teacher lineage has no verified random-only origin")
    evidence = {"source_run": source["id"], "task": source["task"], "checkpoint": str(path),
                "checkpoint_sha256": checkpoint_sha, "reference_sha256": reference_sha,
                "initialization": mode, "model_checksums": source.get("model_checksums", {}),
                "reference_transition": initial.get("reference_transition")}
    lineage = [*lineage, evidence]
    return {"source_run": source["id"], "checkpoint": str(path),
            "checkpoint_sha256": checkpoint_sha, "reference_sha256": reference_sha,
            "origin": origin, "root_source_run": lineage[0]["source_run"], "lineage": lineage}

  return verify(checkpoint)


def validate_distillation_teacher(args, task):
  """Frozen teacher of a distillation task: a completed, verified random-origin teacher of teacher_task."""
  forbidden = [flag for flag in ("resume", "amp_disc", "actor_only", "freeze_actor", "freeze_normalizer")
               if getattr(args, flag)]
  if forbidden:
    raise ValueError("Distillation starts a random student from a frozen teacher; remove --"
                     + forbidden[0].replace("_", "-"))
  if not args.teacher:
    raise ValueError("Distillation tasks require --teacher <completed final teacher model_N.pt>")
  teacher = validate_random_teacher_resume(args.teacher)
  if teacher["lineage"][-1]["task"] != task.get("teacher_task"):
    raise ValueError("The teacher checkpoint was not trained by the task's declared teacher_task")
  checkpoint = Path(teacher["checkpoint"])
  policy = checkpoint.with_name("policy_" + checkpoint.stem[len("model_"):] + ".onnx")
  if not policy.is_file():
    raise ValueError("The teacher run has no ONNX export of its final checkpoint")
  return {**teacher, "policy": str(policy), "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}


def validate_student_rl(args, task):
  """D6 fine-tune: --resume = completed final checkpoint of student_task, --teacher = the teacher it was
  distilled from (verified random-origin lineage of teacher_task, same checkpoint hash)."""
  forbidden = [flag for flag in ("amp_disc", "actor_only", "freeze_actor", "freeze_normalizer") if getattr(args, flag)]
  if forbidden:
    raise ValueError("Student RL fine-tunes the distilled actor with its frozen normalizer; remove --"
                     + forbidden[0].replace("_", "-"))
  self_anchor = bool(task.get("self_anchor"))
  if self_anchor and args.teacher:
    raise ValueError("Self-anchored student RL takes no --teacher: the initial student is its own anchor")
  if not args.resume or (not args.teacher and not self_anchor):
    raise ValueError("Student RL requires --resume <final distilled student model_N.pt> and --teacher <its teacher model_N.pt>")
  teacher = None
  if not self_anchor:
    teacher = validate_random_teacher_resume(args.teacher)
    if teacher["lineage"][-1]["task"] != task.get("teacher_task"):
      raise ValueError("The teacher checkpoint was not trained by the task's declared teacher_task")
    checkpoint = Path(teacher["checkpoint"])
    policy = checkpoint.with_name("policy_" + checkpoint.stem[len("model_"):] + ".onnx")
    if not policy.is_file():
      raise ValueError("The teacher run has no ONNX export of its final checkpoint")
    teacher = {**teacher, "policy": str(policy), "policy_sha256": hashlib.sha256(policy.read_bytes()).hexdigest()}
  runs = (work_root()/"runs"/"r1_careful_walk").resolve()
  student = Path(args.resume).resolve()
  try:
    info = json.loads((student.parent/"run_info.json").read_text())
  except (OSError, ValueError) as error:
    raise ValueError(f"Missing or invalid student run record: {student.parent}") from error
  spec = TASK_SPECS.get(info.get("task"), {})
  if (student.parent.parent != runs or info.get("id") != student.parent.name
      or info.get("task") != task.get("student_task") or not spec.get("distillation")
      or info.get("state") != "completed" or student.name != info.get("checkpoint")
      or not student.name.startswith("model_") or student.suffix != ".pt"):
    raise ValueError("Student RL starts from the completed final checkpoint of the declared student_task")
  distill = info.get("distillation") or {}
  if self_anchor:
    # No teacher acts or labels; the student's own distillation teacher is carried for evaluation (shadow
    # labels in the student scene) and provenance, after re-hashing its recorded files.
    work = work_root().resolve()
    required = ("teacher_checkpoint", "teacher_checkpoint_sha256", "teacher_policy", "teacher_policy_sha256",
                "teacher_lineage", "origin", "reference_sha256")
    if any(not distill.get(key) for key in required):
      raise ValueError("The initial student's run record lacks its distillation teacher")
    for key in ("teacher_checkpoint", "teacher_policy"):
      if hashlib.sha256((work/distill[key]).read_bytes()).hexdigest() != distill[key + "_sha256"]:
        raise ValueError("The initial student's recorded teacher files changed")
    teacher = {"source_run": distill["teacher_lineage"][-1]["source_run"],
               "checkpoint": str(work/distill["teacher_checkpoint"]),
               "checkpoint_sha256": distill["teacher_checkpoint_sha256"],
               "policy": str(work/distill["teacher_policy"]), "policy_sha256": distill["teacher_policy_sha256"],
               "reference_sha256": distill["reference_sha256"], "origin": distill["origin"],
               "lineage": distill["teacher_lineage"], "teacher_env": distill.get("teacher_env"),
               "iteration": distill.get("teacher_iteration")}
  elif distill.get("teacher_checkpoint_sha256") != teacher["checkpoint_sha256"]:
    raise ValueError("The student was distilled from a different teacher checkpoint")
  student_policy = student.parent/str(info.get("policy"))
  if not info.get("policy") or not student_policy.is_file():
    raise ValueError("The student run has no ONNX export of its final checkpoint")
  return {"teacher": teacher,
          "student": {"source_run": info["id"], "task": info["task"],
                      "checkpoint": str(student.relative_to(work_root().resolve())),
                      "checkpoint_sha256": hashlib.sha256(student.read_bytes()).hexdigest(),
                      "policy": str(student_policy.relative_to(work_root().resolve())),
                      "policy_sha256": hashlib.sha256(student_policy.read_bytes()).hexdigest(),
                      "teacher_checkpoint_sha256": distill.get("teacher_checkpoint_sha256")}}


def snapshot_sources(directory):
  """Preserve the actual source used, not just a dirty git revision."""
  root = Path(__file__).resolve().parents[2]
  paths = sorted([p for p in root.joinpath("src").rglob("*")
                  if p.is_file() and p.suffix in (".py", ".xml", ".json", ".yaml")]
                 + list(root.joinpath("scripts").glob("*.sh")))
  manifest = {}
  destination = directory / "params"
  destination.mkdir(exist_ok=True)
  with zipfile.ZipFile(destination / "source_snapshot.zip", "w", zipfile.ZIP_DEFLATED) as archive:
    for path in paths:
      content = path.read_bytes()
      relative = path.relative_to(root).as_posix()
      manifest[relative] = hashlib.sha256(content).hexdigest()
      archive.writestr(relative, content)
  atomic_json(destination / "source_snapshot.json", manifest)


def ensure_robot_self_collision(env):
  """Reject a training model whose R1 collision geoms cannot collide with each other."""
  model = env.sim.mj_model
  geoms = [model.geom(i) for i in range(model.ngeom)
           if model.geom(i).name.startswith("robot/") and model.geom(i).name.endswith("_collision")]
  disabled = [geom.name for geom in geoms
              if int(geom.contype[0]) & int(geom.conaffinity[0]) == 0]
  if not geoms or disabled:
    raise ValueError("R1 학습에는 자기충돌이 필수입니다. 충돌 마스크를 확인하세요: "
                     + ", ".join(disabled[:5]))
  return {"enabled": True, "collision_geoms": len(geoms)}


def configure_actor_training(policy, *, freeze_actor=False, freeze_normalizer=False):
  """Freeze actor parameters and/or its running statistics across train() calls.

  EmpiricalNormalization.until is a public sample-limit setting. Zero prevents
  updates even for an imported count=0 and survives recursive Module.train().
  It is not a state_dict field: reapply the requested setting after every load.
  The critic normalizer is deliberately unaffected.
  """
  if freeze_actor:
    for parameter in policy.parameters():
      parameter.requires_grad_(False)
  effective = bool(freeze_actor or freeze_normalizer)
  if effective:
    from rsl_rl.modules.normalization import EmpiricalNormalization
    normalizer = policy.obs_normalizer
    if not isinstance(normalizer, EmpiricalNormalization):
      raise TypeError("Actor normalization freeze requires EmpiricalNormalization.")
    normalizer.until = 0
  return effective


def extend_actor_for_terrain(source, target):
  """Preserve the 810-input actor and append 117 initially inactive inputs."""
  first = "mlp.0.weight"
  if source[first].shape[1] != 810 or target[first].shape[1] != 927:
    raise ValueError("Only the explicit 810 -> 927 terrain transfer is supported")
  if set(source) != set(target):
    raise ValueError("Actor parameter layout changed beyond terrain input")
  result = {}
  for key, destination in target.items():
    original = source[key].to(destination.device)
    if original.shape == destination.shape:
      result[key] = original.clone()
    elif key == first and original.shape[0] == destination.shape[0]:
      value = destination.new_zeros(destination.shape)
      value[:, :810] = original
      result[key] = value
    elif key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
      value = destination.clone()  # appended mean=0, std/variance=1
      value[..., :810] = original
      result[key] = value
    else:
      raise ValueError(f"Unexpected changed actor tensor: {key}")
  return result


def train(args, record):
  # Registration must succeed before importing or allocating GPU resources.
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.rl.runner import MjlabOnPolicyRunner
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends
  from r1lab.export import export_onnx
  from r1lab.distributed import configure_local_device, broadcast_module, assert_model_consistent
  from r1lab.mdp.symmetry import Mirror
  from r1lab.tasks import TASKS

  from r1lab.distillation import R1DistillationRunner

  context = distributed_context()
  device = configure_local_device(context)
  observation_contract = None
  student_metadata = {}

  def save_with_policy(runner, path, infos=None, metadata=None):
    if not context.is_primary:
      raise RuntimeError("Only the primary rank may write a shared checkpoint")
    target = Path(path)
    temporary = target.with_name("."+target.name+".part")
    try:
      MjlabOnPolicyRunner.save(runner, str(temporary), {**(infos or {}),
          "scores": getattr(runner.alg, "last_metrics", {}), "world_size": context.world_size})
      temporary.replace(target)
    finally:
      temporary.unlink(missing_ok=True)
    iteration = int(Path(path).stem.split("_")[-1])
    policy = Path(path).with_name(f"policy_{iteration}.onnx")
    temporary_policy = policy.with_name("."+policy.name+".part")
    export_error = None
    try:
      export_onnx(runner.alg.get_policy(), temporary_policy,
                  observation_contract=observation_contract, metadata=metadata)
      temporary_policy.replace(policy)
    except Exception as error:
      export_error = str(error)
      print(f"[WARN] ONNX export failed: {error}", flush=True)
    finally:
      temporary_policy.unlink(missing_ok=True)
    record.update(iteration=iteration, checkpoint=Path(path).name,
                  policy=policy.name if policy.is_file() else None, export_error=export_error,
                  atomic_checkpoints=export_error is None)

  class R1Distiller(R1DistillationRunner):
    def save(self, path: str, infos=None) -> None:
      save_with_policy(self, path, infos, metadata=student_metadata)

  class R1Runner(MjlabOnPolicyRunner):
    def load(self, path, load_cfg=None, strict=True, map_location=None):
      if advanced and load_cfg and load_cfg.get("actor") and not load_cfg.get("critic"):
        state = torch.load(path, map_location=map_location, weights_only=False)
        source = state.get("actor_state_dict", {})
        target = self.alg._raw_actor.state_dict()
        if source.get("mlp.0.weight") is not None and source["mlp.0.weight"].shape[1] == 810:
          self.alg._raw_actor.load_state_dict(extend_actor_for_terrain(source, target))
          if load_cfg.get("iteration"):
            self.current_learning_iteration = int(state.get("iter", 0))
          record.update(actor_input_transfer={"source_dimension":810,"target_dimension":927,
                        "new_weight_columns":"zero", "preserved":"weights and original normalizer"})
          return state.get("infos")
      return super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)

    def save(self, path: str, infos=None) -> None:
      save_with_policy(self, path, infos, metadata=student_metadata or None)

  torch.backends.cuda.matmul.allow_tf32 = True
  configure_torch_backends()
  task = TASKS[args.task]
  reference_teacher = task.get("evaluation_profile") == "reference_teacher_v1"
  distillation = bool(task.get("distillation"))
  student_rl = bool(task.get("student_rl"))
  if distillation and student_rl:
    raise ValueError("A task is either a distillation or a student RL fine-tune")
  distill_teacher = validate_distillation_teacher(args, task) if distillation else None
  rl_init = validate_student_rl(args, task) if student_rl else None
  if student_rl:
    distill_teacher = rl_init["teacher"]
  if args.teacher and not (distillation or student_rl):
    raise ValueError("--teacher is only for registered distillation or student RL tasks")
  teacher_parent = None
  if reference_teacher:
    if args.amp_disc or args.actor_only or args.freeze_actor or args.freeze_normalizer:
      raise ValueError("Reference teacher experiments require active actor/critic learning with their complete contract")
    if task.get("continue_random_teacher"):
      if not args.resume:
        raise ValueError("This controlled follow-up requires our completed random teacher checkpoint")
      teacher_parent = validate_random_teacher_resume(args.resume)
    elif args.resume:
      raise ValueError("The first reference teacher experiment requires fresh actor/critic weights")
  env_cfg = task["env"](t0=args.t0) if args.t0 is not None else task["env"]()
  if distillation or (student_rl and not task.get("self_anchor")):
    motion_cfg = env_cfg.commands["motion"]
    if hashlib.sha256(Path(motion_cfg.motion_file).read_bytes()).hexdigest() != distill_teacher["reference_sha256"]:
      raise ValueError("The student scene's library differs from the teacher's training reference")
    recorded = json.loads((Path(distill_teacher["checkpoint"]).parent/"params"/"motion_reference.json").read_text())
    mix = [list(m[:2]) + [list(r) for r in m[2:]] for m in motion_cfg.command_mix]
    if mix != recorded.get("matching", {}).get("command_mix"):
      raise ValueError("The student command mix differs from the teacher run's command mix")
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed + context.rank*100003
  agent = task["rl"]()
  agent.seed = args.seed
  if agent.experiment_name != record.directory.parent.name:
    raise ValueError("학습 설정과 등록된 실험 경로가 다릅니다. TASK_SPECS를 확인하세요.")
  if args.iters is not None:
    agent.max_iterations = args.iters
  record.update(iters=agent.max_iterations)
  log_dir = record.directory
  if context.is_primary:
    snapshot_sources(log_dir)
    if args.resume:
      record.update(resume_sha256=hashlib.sha256(Path(args.resume).read_bytes()).hexdigest())
  if reference_teacher:
    from r1lab.mdp import motion_teacher as teacher_mdp
    from r1lab.tasks.motion_teacher import OBSERVATION_CONTRACT
    if task.get("phase_mode") == "zero":
      from r1lab.tasks.phasefree_teacher import PHASEFREE_OBSERVATION_CONTRACT as OBSERVATION_CONTRACT
    expected_phase = (teacher_mdp.zero_phase_observation if task.get("phase_mode") == "zero"
                      else teacher_mdp.phase_observation)
    # The deployed actor's phase must match the declared mode; critics may keep the clip phase.
    if env_cfg.observations["actor"].terms["phase"].func is not expected_phase:
      raise ValueError("Teacher actor phase observation differs from the task's declared phase mode")
    reference = Path(env_cfg.commands["motion"].motion_file)
    reference_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    reference_transition = None
    if teacher_parent and reference_hash != teacher_parent["reference_sha256"]:
      if not task.get("reference_transition"):
        raise ValueError("The controlled teacher follow-up must preserve its parent's reference")
      reference_transition = {"parent_reference_sha256": teacher_parent["reference_sha256"],
                              "reference_sha256": reference_hash}
    observation_contract = {"name": OBSERVATION_CONTRACT["name"],
                            "dimension": OBSERVATION_CONTRACT["input_dimension"],
                            "reference_sha256": reference_hash,
                            "phase_mode": OBSERVATION_CONTRACT.get("phase_mode", "clip_fraction")}
    provenance = {"source": str(reference), "sha256": reference_hash,
                  "observation_contract": OBSERVATION_CONTRACT}
    manifest = reference.with_suffix(".manifest.json")
    if manifest.is_file():
      provenance["manifest"] = json.loads(manifest.read_text())
    # The 14-18 s file is disjoint only from the original 8-12 s training clip.
    heldout = reference.with_name("cautious_teacher_heldout_v1.npz")
    heldout_valid = reference.name == "cautious_teacher_v1.npz" and heldout.is_file()
    if not heldout_valid:
      provenance["heldout"] = "none: no disjoint held-out split exists for this reference"
    if task.get("reference_contract") == "full_capture":
      from r1lab.tasks.fullcapture_teacher import FULL_CAPTURE_CONTRACT
      provenance["full_capture"] = FULL_CAPTURE_CONTRACT
    elif task.get("reference_contract") == "matching":
      from r1lab.tasks.matching_teacher import MATCHING_CONTRACT
      motion_cfg = env_cfg.commands["motion"]
      # The static contract names the v3 defaults; record what this task actually uses.
      provenance["matching"] = {**MATCHING_CONTRACT, "library": Path(motion_cfg.motion_file).name,
                                "command_mix": [list(m[:2]) + [list(r) for r in m[2:]] for m in motion_cfg.command_mix]}
      if "stance_slip" in env_cfg.rewards:
        provenance["stance_slip_weight"] = env_cfg.rewards["stance_slip"].weight
      if motion_cfg.matcher.get("hold_settled_heading"):
        provenance["hold_settled_heading"] = "settled standing keeps its heading (idle-clip yaw drift corrected)"
      if getattr(motion_cfg, "disturbed_reanchor", None) is not None:
        provenance["disturbed_reanchor"] = {"thresholds_m_rad": list(motion_cfg.disturbed_reanchor),
                                            "window_s": 1.5, "marks": "ball contacts only"}
      if "landing_speed" in env_cfg.rewards:
        provenance["landing_speed"] = {"weight": env_cfg.rewards["landing_speed"].weight,
                                       **env_cfg.rewards["landing_speed"].params,
                                       "human_reference": "00230 landing vz mean .089, p95 .152 m/s (library v3)"}
      if "projectiles" in env_cfg.events:
        params = env_cfg.events["projectiles"].params
        provenance["projectiles"] = {"ball_mass_kg": 1.0, "ball_radius_m": .07, "aim": "torso COM, random azimuth",
                                     **{k: list(v) if isinstance(v, tuple) else v for k, v in params.items()}}
      if "push_robot" in env_cfg.events:
        provenance["pushes"] = {"velocity_range": env_cfg.events["push_robot"].params["velocity_range"],
                                "interval_s": list(env_cfg.events["push_robot"].interval_range_s),
                                "marked": getattr(env_cfg.events["push_robot"].func, "__name__", "") == "tracked_velocity_push"}
      provenance["heldout"] = "generated references; held-out behaviour is evaluated with scripted commands"
    elif task.get("reference_contract") == "library":
      from r1lab.tasks.library_teacher import LIBRARY_CONTRACT
      provenance["library"] = LIBRARY_CONTRACT
      provenance["heldout"] = ("library segments flagged segment_heldout: " + LIBRARY_CONTRACT["heldout"])
      if task.get("xy_termination_m") is not None:
        provenance["xy_termination_m"] = task["xy_termination_m"]
        if env_cfg.terminations["anchor_xy"].params["threshold"] != task["xy_termination_m"]:
          raise ValueError("Registered xy termination differs from the task environment")
    record.update(evaluation_profile=task["evaluation_profile"], reference_teacher=provenance,
                  initialization={"mode": "continued_random_teacher" if teacher_parent else "random",
                                  "pretrained_weights": [teacher_parent["checkpoint"]] if teacher_parent else [],
                                  "parent": teacher_parent,
                                  "reference_transition": reference_transition,
                                  "manufacturer_policy_role": "evaluation_only",
                                  "actor_and_critic_normalization": "disabled"})
    if context.is_primary:
      shutil.copyfile(reference, log_dir/"params"/"motion_reference.npz")
      if heldout_valid:
        shutil.copyfile(heldout, log_dir/"params"/"motion_reference_heldout.npz")
        provenance["heldout_sha256"] = hashlib.sha256(heldout.read_bytes()).hexdigest()
        heldout_manifest = heldout.with_suffix(".manifest.json")
        if heldout_manifest.is_file():
          provenance["heldout_manifest"] = json.loads(heldout_manifest.read_text())
      atomic_json(log_dir/"params"/"motion_reference.json", provenance)
    if task.get("precise_foot_tracking"):
      from r1lab.tasks.precise_teacher import PRECISE_REWARD_CONTRACT
      foot_contract = dict(PRECISE_REWARD_CONTRACT)
      foot_contract["native_body_position_weight"] = env_cfg.rewards["motion_body_pos"].weight
      foot_contract["position_total_weight"] = (
          env_cfg.rewards["motion_body_pos"].weight + env_cfg.rewards["precise_foot_trajectory"].weight)
      record.update(precise_foot_tracking=foot_contract)
      if context.is_primary:
        atomic_json(log_dir/"params"/"precise_foot_tracking.json", foot_contract)
    if task.get("arm_position_tracking"):
      from r1lab.tasks.coordinated_teacher import ARM_REWARD_CONTRACT
      record.update(arm_position_tracking=ARM_REWARD_CONTRACT)
      if context.is_primary:
        atomic_json(log_dir/"params"/"arm_position_tracking.json", ARM_REWARD_CONTRACT)
  print(f"[INFO] id={record.info['id']} task={args.task} rank={context.rank}/{context.world_size} "
        f"device={device} envs_per_rank={args.num_envs} seed={env_cfg.seed} "
        f"iters={agent.max_iterations} log={log_dir}", flush=True)
  # Select the protocol by the algorithm, including registered task variants.
  advanced = agent.algorithm.class_name == "r1lab.learning:RiskPPO"
  if advanced and args.amp_disc:
    raise ValueError("The full-body human prior resumes inside model.pt; legacy --amp-disc is incompatible")
  if "style" in env_cfg.rewards and not advanced:
    env_cfg.rewards["style"].params["save_path"] = str(log_dir / "amp_disc.pt")
    if args.amp_disc:
      env_cfg.rewards["style"].params["load_path"] = args.amp_disc

  env = None
  try:
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    record.update(self_collision=ensure_robot_self_collision(env))
    if getattr(args, "record_body_hash", False):    # option (07B 'one body definition'): robot body signature in run_info
      from r1lab.body_hash import env_signature
      record.update(body_signature=env_signature(env))
    env = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    agent_cfg = asdict(agent)
    if context.is_primary:
      dump_yaml(log_dir / "params" / "env.yaml", asdict(env_cfg))
      dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)
    if not args.no_symmetry and task.get("symmetry", True):
      agent_cfg["algorithm"]["symmetry_cfg"] = {
        "use_data_augmentation": True, "use_mirror_loss": False, "mirror_loss_coeff": 0.0,
        "data_augmentation_func": Mirror(task["cmd_sign"], task["critic_layout"]),
      }
    if args.freeze_actor:
      agent_cfg["algorithm"]["schedule"] = "fixed"
      agent_cfg["algorithm"]["learning_rate"] = 5e-4
    runner = (R1Distiller if distillation else R1Runner)(env, agent_cfg, str(log_dir), device)
    if distillation:
      from r1lab import contract as C
      from r1lab.tasks.matching_student import STUDENT_CONTRACT
      STUDENT_CONTRACT = task.get("student_contract") or STUDENT_CONTRACT  # task-specific command contract
      loaded_teacher = runner.load_teacher(distill_teacher["checkpoint"], map_location=device)
      if not runner.alg.teacher_loaded:
        raise RuntimeError("Distillation requires the strictly loaded teacher")
      widths = (int(runner.alg._raw_student.obs_dim), int(runner.alg._raw_teacher.obs_dim))
      if widths != (C.OBS_DIM, 233):
        raise RuntimeError(f"Student/teacher observation widths {widths} differ from 810/233")
      if runner.alg.optimizer.state:
        raise RuntimeError("A fresh student must start with empty optimizer state")
      import inspect
      teacher_env = inspect.signature(task["env"]).parameters.get("teacher_env")
      distill = {"teacher_task": task["teacher_task"], "teacher_iteration": loaded_teacher["iteration"],
                 "teacher_checkpoint": str(Path(distill_teacher["checkpoint"]).relative_to(work_root().resolve())),
                 "teacher_checkpoint_sha256": distill_teacher["checkpoint_sha256"],
                 "teacher_policy": str(Path(distill_teacher["policy"]).relative_to(work_root().resolve())),
                 "teacher_policy_sha256": distill_teacher["policy_sha256"],
                 "teacher_env": teacher_env.default.__name__ if teacher_env is not None else None,
                 "teacher_lineage": distill_teacher["lineage"], "origin": distill_teacher["origin"],
                 "reference_sha256": distill_teacher["reference_sha256"], "student_contract": STUDENT_CONTRACT,
                 "obs_groups": {k: list(v) for k, v in agent_cfg["obs_groups"].items()},
                 "algorithm": {k: v for k, v in agent_cfg["algorithm"].items() if k not in ("rnd_cfg", "symmetry_cfg")}}
      student_metadata.update({"r1lab.teacher_run": distill_teacher["source_run"],
                               "r1lab.teacher_checkpoint_sha256": distill_teacher["checkpoint_sha256"],
                               "r1lab.teacher_reference_sha256": distill_teacher["reference_sha256"],
                               "r1lab.command_contract": STUDENT_CONTRACT["name"]})
      record.update(evaluation_profile=task["evaluation_profile"], distillation=distill,
                    initialization={"mode": "random_student", "pretrained_weights": [],
                                    "frozen_teacher": distill["teacher_checkpoint"],
                                    "manufacturer_policy_role": "evaluation_only",
                                    "student_normalization": "global rollout moments once per update"})
      if context.is_primary:
        atomic_json(log_dir/"params"/"distillation.json", distill)
    if student_rl:
      import inspect
      from r1lab import contract as C
      STUDENT_CONTRACT = task["student_contract"]
      self_anchor = bool(task.get("self_anchor"))
      if self_anchor:
        loaded = runner.alg.load_self_anchor(str(work_root().resolve()/rl_init["student"]["checkpoint"]),
                                             map_location=device)
        loaded["teacher_iteration"] = distill_teacher.get("iteration")
      else:
        loaded = runner.alg.load_initialization(str(work_root().resolve()/rl_init["student"]["checkpoint"]),
                                                distill_teacher["checkpoint"], map_location=device)
      if not runner.alg.teacher_loaded:
        raise RuntimeError("Student RL requires the strictly loaded teacher")
      widths = (int(runner.alg._raw_actor.obs_dim), int(runner.alg._raw_teacher.obs_dim))
      if widths != (C.OBS_DIM, C.OBS_DIM if self_anchor else 233):
        raise RuntimeError(f"Student/anchor observation widths {widths} differ from the declared contract")
      if runner.alg.optimizer.state:
        raise RuntimeError("The fine-tune must start with empty optimizer state")
      teacher_env = inspect.signature(task["env"]).parameters.get("teacher_env")
      distill = {"teacher_task": task["teacher_task"], "teacher_iteration": loaded["teacher_iteration"],
                 "teacher_checkpoint": str(Path(distill_teacher["checkpoint"]).relative_to(work_root().resolve())),
                 "teacher_checkpoint_sha256": distill_teacher["checkpoint_sha256"],
                 "teacher_policy": str(Path(distill_teacher["policy"]).relative_to(work_root().resolve())),
                 "teacher_policy_sha256": distill_teacher["policy_sha256"],
                 "teacher_env": (distill_teacher.get("teacher_env") if self_anchor else
                                 teacher_env.default.__name__ if teacher_env is not None else None),
                 "teacher_lineage": distill_teacher["lineage"], "origin": distill_teacher["origin"],
                 "reference_sha256": distill_teacher["reference_sha256"], "student_contract": STUDENT_CONTRACT,
                 "obs_groups": {k: list(v) for k, v in agent_cfg["obs_groups"].items()},
                 "algorithm": {k: v for k, v in agent_cfg["algorithm"].items() if k not in ("rnd_cfg", "symmetry_cfg")}}
      fine_tune = {"student_task": task["student_task"], "self_anchor": self_anchor,
                   "initial_student": {**rl_init["student"], "iteration": loaded["student_iteration"]},
                   "bc_target": ("frozen initial student's mean action on the rollout states (self anchor)" if self_anchor
                                 else "frozen teacher mean action on the student's rollout states (DAgger loss)"),
                   "actor_normalizer": "frozen as distilled", "acting_std": "fixed 0.1 (DAgger acting std)",
                   "critic_initialization": ("fresh, value-only warm-up first" if self_anchor else
                                             "teacher run critic_state_dict, value-only warm-up first")}
      student_metadata.update({"r1lab.teacher_run": distill_teacher["source_run"],
                               "r1lab.teacher_checkpoint_sha256": distill_teacher["checkpoint_sha256"],
                               "r1lab.teacher_reference_sha256": distill_teacher["reference_sha256"],
                               "r1lab.command_contract": STUDENT_CONTRACT["name"],
                               "r1lab.initial_student_run": rl_init["student"]["source_run"],
                               "r1lab.initial_student_checkpoint_sha256": rl_init["student"]["checkpoint_sha256"]})
      record.update(evaluation_profile=task["evaluation_profile"], distillation=distill, student_rl=fine_tune,
                    initialization={"mode": "distilled_student_rl",
                                    "pretrained_weights": ([rl_init["student"]["checkpoint"]] if self_anchor else
                                                           [rl_init["student"]["checkpoint"], distill["teacher_checkpoint"]]),
                                    "frozen_teacher": distill["teacher_checkpoint"],
                                    "manufacturer_policy_role": "evaluation_only",
                                    "student_normalization": "frozen as distilled"})
      if context.is_primary:
        atomic_json(log_dir/"params"/"distillation.json", distill)
        atomic_json(log_dir/"params"/"student_rl.json", fine_tune)
    if reference_teacher:
      if runner.alg.optimizer.state:
        raise RuntimeError("A fresh teacher must start with empty optimizer state")
      if int(runner.alg._raw_actor.obs_dim) != observation_contract["dimension"]:
        raise RuntimeError("Teacher observation width differs from the declared contract")
    if args.resume and not student_rl:
      print(f"[INFO] resume from {args.resume}" + (" (actor only)" if args.actor_only else ""), flush=True)
      load_cfg = ({"actor": True, "critic": False, "optimizer": False, "iteration": True}
                  if args.actor_only else None)
      runner.load(args.resume, load_cfg=load_cfg, map_location=device)
      if teacher_parent:
        # Native adaptive PPO keeps this scalar separately from the Adam state.
        rates = {float(group["lr"]) for group in runner.alg.optimizer.param_groups}
        if len(rates) != 1:
          raise ValueError("Teacher continuation requires one consistent restored learning rate")
        runner.alg.learning_rate = rates.pop()
    # A resumed optimizer also carries its previous rate. Fixed-rate tasks use
    # their declared rate, including when resuming a faster critic warmup.
    if agent_cfg["algorithm"].get("schedule") == "fixed":
      rate = agent_cfg["algorithm"]["learning_rate"]
      runner.alg.learning_rate = rate
      for group in runner.alg.optimizer.param_groups:
        group["lr"] = rate
    record.update(learning_rate=float(runner.alg.learning_rate))
    frozen_normalizer = configure_actor_training(
      runner.alg.get_policy(), freeze_actor=args.freeze_actor,
      freeze_normalizer=args.freeze_normalizer or advanced or student_rl)
    record.update(actor_normalizer_frozen=frozen_normalizer)
    start = int(runner.current_learning_iteration)
    # All workers construct environments independently. Broadcast every learned
    # model before resampling episodes so behavior log-probabilities match the
    # shared opponent that will later be updated.
    modules = ({"student": runner.alg._raw_student, "teacher": runner.alg._raw_teacher} if distillation
               else {"actor": runner.alg._raw_actor, "critic": runner.alg._raw_critic})
    if student_rl:
      modules["teacher"] = runner.alg._raw_teacher
    prior = getattr(env.unwrapped, "_r1_human_prior", None)
    opponent = getattr(env.unwrapped, "_r1_adversary", None)
    if prior is not None:
      modules["human_discriminator"] = prior.disc
      record.update(human_prior=prior.provenance)
      if context.is_primary:
        atomic_json(log_dir/"params"/"human_reference.json", prior.reference_manifest)
    if opponent is not None:
      modules["environment_agent"] = opponent.controller
      if getattr(opponent, "restore_metadata", None):
        record.update(environment_restore=opponent.restore_metadata)
    if budget := getattr(runner.alg, "anchor_budget", None):
      shared_budget = torch.nn.Module()
      for name in ("baseline", "window", "ema", "limits", "integral", "multipliers",
                   "stall_episode_counts", "stall_ready"):
        shared_budget.register_buffer(name, getattr(budget, name))
      modules["fixed_anchor_budgets"] = shared_budget
      record.update(stall_protocol_version=budget.stall_protocol_version,
                    stall_budget_migration=budget.restore_metadata,
                    stall_budget_calibrated=budget.stall_ready.tolist())
    for module in modules.values():
      broadcast_module(module)
    audits = {name: assert_model_consistent(module, include_buffers=True, label=name)
              for name, module in modules.items()}
    record.update(model_checksums=audits, device=device, physics_seed=env_cfg.seed,
                  evaluation_profile=task.get("evaluation_profile"),
                  actor_observation_dimension=int(runner.alg.get_policy().obs_dim if distillation
                                                  else runner.alg._raw_actor.obs_dim))
    if distillation or student_rl:
      if context.is_primary:
        atomic_json(log_dir/"params"/"initialization.json", {
            "mode": "distilled_student_rl" if student_rl else "random_student",
            "pretrained_weight_files": (([rl_init["student"]["checkpoint"]] if task.get("self_anchor") else
                                         [rl_init["student"]["checkpoint"], distill["teacher_checkpoint"]])
                                        if student_rl else []),
            "initial_student": rl_init["student"] if student_rl else None, "frozen_teacher": distill,
            "optimizer_state_entries": len(runner.alg.optimizer.state), "seed": args.seed,
            "world_size": context.world_size, "initial_model_checksums": audits})
      native_student_update = runner.alg.update
      student_updates = 0
      student_begin = time.monotonic()

      def audited_student_update():
        nonlocal student_updates
        losses = native_student_update()  # imitation/* are already pooled over ranks
        student_updates += 1
        iteration = start + student_updates - 1
        from r1lab.distributed import all_reduce_mean
        values = {("" if key.startswith("imitation/") else "loss/") + key: float(value) for key, value in losses.items()}
        values["policy/action_std_mean"] = float(runner.alg.get_policy().output_std.mean())
        try:  # scenes without a motion-matching command (student terrain RL) have no tracking metrics
          command = env.unwrapped.command_manager.get_term("motion")
        except KeyError:
          command = None
        if command is not None:
          values.update({"tracking/" + key: float(value.float().mean()) for key, value in command.metrics.items()})
        keys = sorted(values)
        pooled = all_reduce_mean(torch.tensor([values[key] for key in keys], device=device))
        measured = dict(zip(keys, pooled.cpu().tolist()))
        measured["throughput/global_steps_per_second"] = (
            args.num_envs * context.world_size * agent.num_steps_per_env * student_updates
            / max(time.monotonic() - student_begin, .001))
        if student_updates == 1 or iteration % agent.save_interval == 0:
          checksums = {name: assert_model_consistent(module, include_buffers=True, label=name)
                       for name, module in modules.items()}
          record.update(iteration=iteration, model_checksums=checksums)
          if context.is_primary:
            atomic_json(log_dir/"params"/"distributed_audit.json", {
                "iteration": iteration, "world_size": context.world_size, "model_checksums": checksums})
        if context.is_primary:
          row = {"iteration": iteration, "time": time.time(), "metrics": measured}
          with (log_dir/"score_history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
          atomic_json(log_dir/"scores_latest.json", row)
        return losses

      runner.alg.update = audited_student_update
    if reference_teacher:
      if context.is_primary:
        atomic_json(log_dir/"params"/"initialization.json", {
            "mode": "continued_random_teacher" if teacher_parent else "random",
            "pretrained_weight_files": [teacher_parent["checkpoint"]] if teacher_parent else [],
            "parent": teacher_parent, "reference_transition": reference_transition,
            "optimizer_state_entries": len(runner.alg.optimizer.state),
            "seed": args.seed, "world_size": context.world_size,
            "initial_model_checksums": audits, "observation_contract": observation_contract})
      native_update = runner.alg.update
      teacher_updates = 0
      teacher_begin = time.monotonic()

      def audited_teacher_update():
        nonlocal teacher_updates
        losses = native_update()
        teacher_updates += 1
        iteration = start + teacher_updates - 1
        from r1lab.distributed import all_reduce_mean
        values = {"loss/" + key: float(value) for key, value in losses.items()}
        values["policy/action_std_mean"] = float(runner.alg.get_policy().output_std.mean())
        command = env.unwrapped.command_manager.get_term("motion")
        values.update({"tracking/" + key: float(value.float().mean())
                       for key, value in command.metrics.items()})
        if task.get("precise_foot_tracking"):
          from r1lab.mdp.precise_teacher import foot_world_error, foot_scores, LOW_SIGNAL_SCORE
          error = foot_world_error(env.unwrapped)
          scores = foot_scores(error)
          for foot, side in enumerate(("left", "right")):
            values[f"foot_update_state/{side}_xy_error_m"] = float(error[:, foot, :2].norm(dim=-1).mean())
            values[f"foot_update_state/{side}_z_error_m"] = float(error[:, foot, 2].abs().mean())
            values[f"foot_update_state/{side}_score"] = float(scores[:, foot].mean())
            values[f"foot_update_state/{side}_low_signal_fraction"] = float((scores[:, foot] < LOW_SIGNAL_SCORE).float().mean())
        if task.get("arm_position_tracking"):
          from r1lab.mdp.coordinated_teacher import arm_position_error, arm_point_scores, ARM_POINTS
          error = arm_position_error(env.unwrapped)
          scores = arm_point_scores(error)
          for point, name in enumerate(ARM_POINTS):
            label = name.removesuffix("_link")
            values[f"arm_update_state/{label}_error_m"] = float(error[:, point].norm(dim=-1).mean())
            values[f"arm_update_state/{label}_score"] = float(scores[:, point].mean())
            values[f"arm_update_state/{label}_low_signal_fraction"] = float((scores[:, point] < 0.02).float().mean())
        keys = sorted(values)
        pooled = all_reduce_mean(torch.tensor([values[key] for key in keys], device=device))
        measured = dict(zip(keys, pooled.cpu().tolist()))
        measured["throughput/global_steps_per_second"] = (
            args.num_envs * context.world_size * agent.num_steps_per_env * teacher_updates
            / max(time.monotonic() - teacher_begin, .001))
        if teacher_updates == 1 or iteration % agent.save_interval == 0:
          checksums = {name: assert_model_consistent(module, include_buffers=True, label=name)
                       for name, module in modules.items()}
          record.update(iteration=iteration, model_checksums=checksums)
          if context.is_primary:
            atomic_json(log_dir/"params"/"distributed_audit.json", {
                "iteration": iteration, "world_size": context.world_size, "model_checksums": checksums})
        if context.is_primary:
          row = {"iteration": iteration, "time": time.time(), "metrics": measured}
          with (log_dir/"score_history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
          atomic_json(log_dir/"scores_latest.json", row)
        return losses

      runner.alg.update = audited_teacher_update
    if advanced:
      env.reset()
      reference = getattr(env.unwrapped, '_r1_reference_initializer', None)
      if reference is not None:
        rsi = dict(provenance=reference.provenance,
                   sampler_seed=opponent.rsi_seed+100003*context.rank,
                   sampler_rank=context.rank,
                   rsi_environments=int(opponent.rsi_episode.sum()),
                   source_frame_count=int(reference.bank['source_frame'].numel()),
                   initial_frames=opponent.rsi_source_frame[opponent.rsi_episode].tolist(),
                   actor_contract='927 unchanged; reference states are reset-only',
                   anchor_calibration='RSI excluded; fixed standing starts preserved')
        record.update(reference_initialization=rsi)
        atomic_json(log_dir/'params'/f'reference_initialization_rank{context.rank}.json', rsi)
      record.update(metric_scope={
          "scores_latest.json": "global pooled constraints, auxiliary metrics and PPO losses",
          "native_train_reward_and_episode": "rank zero local episode summaries"})
      begin = time.monotonic()
      initial_update_index = runner.alg.update_index

      def update_scores(metrics, index):
        updates_done = index-initial_update_index
        iteration = start+updates_done-1
        measured = dict(metrics)
        measured["throughput/global_steps_per_second"] = (
            args.num_envs*context.world_size*agent.num_steps_per_env*updates_done /
            max(time.monotonic()-begin, .001))
        if updates_done == 1 or iteration % agent.save_interval == 0:
          checksums = {name: assert_model_consistent(module, include_buffers=True, label=name)
                       for name, module in modules.items()}
          record.update(iteration=iteration, model_checksums=checksums)
          if context.is_primary:
            atomic_json(log_dir/"params"/"distributed_audit.json",
                        {"iteration": iteration, "world_size": context.world_size,
                         "model_checksums": checksums})
            from r1lab.mdp.scene_program import program_json
            samples = [program_json(opponent.program, i) for i in range(min(4, args.num_envs))]
            atomic_json(log_dir/"params"/"scene_examples.json", {"iteration":iteration,
                         "physics_seed":env_cfg.seed, "programs": samples})
        if context.is_primary:
          row = {"iteration": iteration, "time": time.time(), "metrics": measured}
          with (log_dir/"score_history.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False)+"\n")
          atomic_json(log_dir/"scores_latest.json", row)
      runner.alg.on_update = update_scores
    record.update(state="running", start_iteration=start, total_iterations=start+agent.max_iterations)
    runner.learn(num_learning_iterations=agent.max_iterations,
                 init_at_random_ep_len=task.get("random_episode_start", not (reference_teacher or distillation)))
    audits = {name: assert_model_consistent(module, include_buffers=True, label=name)
              for name, module in modules.items()}
    record.update(iteration=int(runner.current_learning_iteration), model_checksums=audits)
    if context.is_distributed:
      torch.distributed.barrier()
  finally:
    if env is not None:
      env.close()
    if torch.distributed.is_initialized():
      torch.distributed.destroy_process_group()


def main(argv=None):
  ap = argparse.ArgumentParser()
  ap.add_argument("--task", required=True, choices=sorted(TASK_SPECS))
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--iters", type=int, default=None)
  ap.add_argument("--seed", type=int, default=1)
  ap.add_argument("--execution", choices=("standard", "cached", "compacted", "optimized"), default="optimized",
                  help="whole-body execution: native, read cache, type-compacted collisions, or grouped and parallel heightfield collisions")
  ap.add_argument("--global-statistics", action="store_true",
                  help="whole-body ImitationPPO: merge observation statistics and rollout advantages across all GPUs")
  ap.add_argument("--resume-next-iteration", action="store_true",
                  help="whole-body resume: start after the saved iteration label (optimizer state and update count are preserved)")
  ap.add_argument("--failure-replay-buffer", action="store_true",
                  help="whole-body diagnostics: retain a bounded full-world buffer of pre-substep physics inputs on failure")
  ap.add_argument("--reward-ledger", action="store_true",
                  help="whole-body diagnostics: record bounded reward-time raw costs, failures, wrench and critic values")
  ap.add_argument("--run-name", default="")
  ap.add_argument("--description", default="", help="human-readable training objective shown in Trainer")
  ap.add_argument("--resume", default=None, help="checkpoint .pt to continue from")
  ap.add_argument("--start-from", default=None,
                  help="whole-body anchored tasks: policy start weights from this checkpoint (new critic, new schedule)")
  ap.add_argument("--teacher", default=None,
                  help="distillation tasks: the frozen teacher, a completed final model_N.pt of teacher_task")
  ap.add_argument("--mini-batches", type=int, default=None,
                  help="whole-body tasks: minibatches per epoch instead of the task's (several GPUs with the single-GPU "
                       "environment count on each: N times the task's keeps the data one gradient step sees)")
  ap.add_argument("--no-symmetry", action="store_true")
  ap.add_argument("--record-body-hash", action="store_true",
                  help="record the robot body signature (r1lab.body_hash: per-component sha256) in run_info.json")
  ap.add_argument("--log-root", default=None, help="must be the canonical trainer work directory's runs folder")
  ap.add_argument("--amp-disc", default=None, help="discriminator checkpoint to resume")
  ap.add_argument("--t0", type=int, default=None, help="stage start iteration (curriculum offset)")
  ap.add_argument("--actor-only", action="store_true", help="load actor only; reset critic and optimizer")
  ap.add_argument("--freeze-actor", action="store_true", help="warm up the critic while keeping the actor fixed")
  ap.add_argument("--freeze-normalizer", action="store_true",
                  help="freeze actor observation statistics while allowing actor learning; critic statistics still update")
  args = ap.parse_args(argv)
  if args.num_envs <= 0 or (args.iters is not None and args.iters <= 0):
    ap.error("--num-envs와 --iters는 양수여야 합니다.")
  if args.mini_batches is not None and (args.mini_batches <= 0 or TASK_SPECS[args.task].get("runner") != "r1wb"):
    ap.error("--mini-batches는 전신 계열(r1wb) 과제에서 양수로만 씁니다.")
  if (args.global_statistics or args.resume_next_iteration or args.failure_replay_buffer or args.reward_ledger) and TASK_SPECS[args.task].get("runner") != "r1wb":
    ap.error("--global-statistics, --resume-next-iteration, --failure-replay-buffer, --reward-ledger는 전신 계열(r1wb) 과제에서만 씁니다.")
  if args.resume_next_iteration and not args.resume:
    ap.error("--resume-next-iteration에는 --resume이 필요합니다.")
  with RunRecord.create_distributed(args, TASK_SPECS[args.task]) as record:
    if TASK_SPECS[args.task].get("runner") == "r1wb":
      from r1wb.train import train as whole_body_train
      whole_body_train(args, record)
    else:
      train(args, record)


if __name__ == "__main__":
  main()
