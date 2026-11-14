"""Mandatory, atomic training registration and lifecycle records for Operation Center."""
from __future__ import annotations

from contextlib import AbstractContextManager
from copy import copy
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import socket
import sys
import tempfile
import time
import traceback
from zoneinfo import ZoneInfo

from r1lab.distributed import distributed_context


REGISTRATION_TIMEOUT = 120.0


def atomic_json(path: Path, value) -> None:
  text = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=1) + "\n"
  temporary = None
  try:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.stem}-", delete=False) as stream:
      temporary = Path(stream.name)
      stream.write(text)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    if temporary is not None:
      temporary.unlink(missing_ok=True)


def work_root() -> Path:
  trainer = Path(__file__).resolve().parents[2]
  return trainer / ".agent" if (trainer / ".agent").is_dir() else trainer.parent / ".agent/trainer"


def runs_root(args, work) -> Path:
  root = (Path(work) if work is not None else work_root()).resolve() / "runs"
  if args.log_root and Path(args.log_root).resolve() != root:
    raise ValueError(f"Trainer 조회 보장을 위해 --log-root는 {root}만 사용할 수 있습니다.")
  return root


def slug(value: str) -> str:
  if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@+-]*", value) or len(value) > 200:
    raise ValueError(f"학습 식별자에 사용할 수 없는 값입니다: {value!r}")
  return value


def source_tag(resume: str) -> str:
  checkpoint = Path(resume)
  if checkpoint.name == "manufacturer_seed.pt":
    return "manufacturer@0"
  iteration = re.sub(r"\D", "", checkpoint.stem) or "unknown"
  try:
    tag = json.loads((checkpoint.parent / "run_info.json").read_text())["tags"].split("-")[0]
  except (OSError, ValueError, KeyError):
    tag = checkpoint.parent.name.split("_", 2)[-1].replace("_", "-")
  return slug(f"{tag}@{iteration}")


class Tee:
  def __init__(self, console, log):
    self.console, self.log = console, log

  def write(self, text):
    self.log.write(text)
    self.log.flush()
    return self.console.write(text)

  def flush(self):
    self.log.flush()
    self.console.flush()

  def __getattr__(self, name):
    return getattr(self.console, name)


class RunRecord(AbstractContextManager):
  def __init__(self, directory, info):
    self.directory, self.info = directory, info

  @classmethod
  def create(cls, args, task, work=None):
    root = runs_root(args, work)
    stamp = os.environ.get("RL_STAMP") or datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M")
    parts = [slug(task["action"]), slug(stamp), slug(task["tags"] + ("-critic_warmup" if args.freeze_actor else "")), f"n{args.num_envs}"]
    if args.resume:
      parts.append("from-" + source_tag(args.resume))
    if getattr(args, "teacher", None):
      parts.append("teacher-" + source_tag(args.teacher))
    if args.run_name:
      parts.append(slug(args.run_name))
    rid = slug("_".join(parts))
    parent = root / slug(task["experiment"])
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1, 10000):
      directory = parent / (rid if index == 1 else f"{rid}-{index}")
      try:
        directory.mkdir()
        break
      except FileExistsError:
        continue
    else:
      raise RuntimeError("고유한 학습 ID를 만들 수 없습니다.")
    info = {"schema_version": 1, "id": directory.name, "action": task["action"],
            "task": args.task, "tags": task["tags"], "started": stamp,
            "num_envs": args.num_envs, "iters": args.iters, "seed": args.seed, "t0": args.t0,
            "resume": args.resume, "teacher": getattr(args, "teacher", None),
            "amp_disc": args.amp_disc, "actor_only": args.actor_only,
            "freeze_actor": args.freeze_actor,
            "freeze_normalizer": bool(getattr(args, "freeze_normalizer", False)),
            "actor_normalizer_frozen": bool(args.freeze_actor or getattr(args, "freeze_normalizer", False)),
            "render": not args.freeze_actor,
            "description": args.description,
            "state": "starting", "pid": os.getpid(), "host": socket.gethostname(),
            "created_at": time.time(), "log_file": "train.log", "iteration": None}
    record = cls(directory, info)
    try:
      record.update()
    except BaseException:
      directory.rmdir()  # Only the empty directory created by this attempt.
      raise
    return record

  @classmethod
  def create_distributed(cls, args, task, work=None):
    """Register one run before GPU imports and attach torchrun workers to it."""
    if int(os.environ.get("WORLD_SIZE", "1")) == 1:
      return cls.create(args, task, work=work)
    context = distributed_context()
    root = runs_root(args, work)
    launch_id = os.environ.get("RL_LAUNCH_ID")
    if not launch_id:
      raise ValueError("Distributed registration requires shared RL_LAUNCH_ID; use scripts/01_train.sh --gpus N")
    launch_id = slug(launch_id)
    handoff = root.parent / "tmp" / "distributed" / f"{launch_id}.json"
    handoff.parent.mkdir(parents=True, exist_ok=True)
    if context.is_primary:
      # A restarted or duplicated rank zero must not silently register another run.
      try:
        descriptor = os.open(handoff.with_suffix(".lock"), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
      except FileExistsError as error:
        raise RuntimeError(f"Distributed launch {launch_id} already registered; use a new RL_LAUNCH_ID") from error
      with os.fdopen(descriptor, "w") as stream:
        stream.write(f"{os.getpid()}\n")
      record = None
      try:
        registered_args = copy(args)
        registered_args.num_envs = args.num_envs * context.world_size
        record = cls.create(registered_args, task, work=work)
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        workers = []
        for rank in range(context.world_size):
          local_rank = rank % context.local_world_size
          workers.append({"rank": rank, "local_rank": local_rank, "device": f"cuda:{local_rank}",
                          "visible_gpu": visible[local_rank] if local_rank < len(visible) else None,
                          "pid": os.getpid() if rank == 0 else None,
                          "host": socket.gethostname() if rank == 0 else None, "state": "starting",
                          "log_file": "train.log" if rank == 0 else f"rank_{rank}.log",
                          "status_file": "run_info.json" if rank == 0 else f"rank_{rank}.json"})
        record.update(world_size=context.world_size, num_envs_per_gpu=args.num_envs, launcher_pid=os.getppid(),
                      launch_id=launch_id, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                      gpu_ranks=list(range(context.world_size)), workers=workers)
        atomic_json(handoff, {"schema_version": 1, "launch_id": launch_id,
                              "directory": str(record.directory), "info": record.info})
        return record
      except BaseException as error:
        detail = f"{type(error).__name__}: {error}"
        if record is not None:
          try:
            record.update(state="failed", finished_at=time.time(), error=detail)
          except BaseException:
            pass
        try:
          atomic_json(handoff, {"schema_version": 1, "launch_id": launch_id, "error": detail})
        except BaseException:
          pass
        raise
    deadline = time.monotonic() + REGISTRATION_TIMEOUT
    while True:
      try:
        payload = json.loads(handoff.read_text())
        break
      except FileNotFoundError:
        if time.monotonic() >= deadline:
          raise TimeoutError(f"Rank {context.rank} waited {REGISTRATION_TIMEOUT:g}s for rank zero registration: {handoff}")
        time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
      except (OSError, ValueError) as error:
        raise RuntimeError(f"Cannot read distributed registration {handoff}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != 1 or payload.get("launch_id") != launch_id:
      raise RuntimeError(f"Invalid distributed registration identity in {handoff}")
    if "error" in payload:
      raise RuntimeError(f"Rank zero failed to register launch {launch_id}: {payload['error']}")
    try:
      directory, info = Path(payload["directory"]).resolve(), payload["info"]
      valid = (directory.parent == root / slug(task["experiment"]) and info["id"] == directory.name
               and info["task"] == args.task and info["world_size"] == context.world_size
               and info["num_envs_per_gpu"] == args.num_envs
               and info["num_envs"] == args.num_envs * context.world_size
               and (directory / "run_info.json").is_file())
    except (KeyError, TypeError, ValueError) as error:
      raise RuntimeError(f"Invalid distributed run record in {handoff}: {error}") from error
    if not valid:
      raise RuntimeError(f"Distributed registration does not match this worker's task, environment count, or canonical path: {handoff}")
    try:
      current = json.loads((directory / "run_info.json").read_text())
    except (OSError, ValueError) as error:
      raise RuntimeError(f"Cannot read the registered primary run in {directory}: {error}") from error
    if (not isinstance(current, dict) or current.get("id") != info["id"]
        or current.get("launch_id") != launch_id or current.get("state") not in ("starting", "running")):
      raise RuntimeError(f"Distributed registration is no longer an active primary run: {handoff}")
    return WorkerRecord(directory, info, context.rank, context.local_rank)

  def update(self, **changes):
    info = {**self.info, **changes, "updated_at": time.time()}
    if info.get("world_size", 1) > 1:
      workers = [dict(worker) for worker in info["workers"]]
      for worker in workers:
        if worker["rank"] == 0:
          status = info
        else:
          try:
            status = json.loads((self.directory / worker["status_file"]).read_text())
          except (FileNotFoundError, ValueError):
            continue
          if not isinstance(status, dict) or status.get("id") != info["id"] or status.get("rank") != worker["rank"]:
            continue
        for key in ("pid", "host", "state", "iteration", "updated_at", "finished_at", "error"):
          if key in status:
            worker[key] = status[key]
      info["workers"] = workers
    atomic_json(self.directory / "run_info.json", info)
    self.info = info

  @property
  def log_path(self):
    return self.directory / self.info["log_file"]

  def __enter__(self):
    try:
      self.log = self.log_path.open("a", buffering=1)
    except OSError as error:
      self.update(state="failed", finished_at=time.time(), error=f"{type(error).__name__}: {error}")
      raise
    self.stdout, self.stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = Tee(self.stdout, self.log), Tee(self.stderr, self.log)
    self.signals = {}
    for sig in (signal.SIGTERM, signal.SIGINT):
      self.signals[sig] = signal.getsignal(sig)
      signal.signal(sig, self.interrupt)
    print(f"[INFO] id={self.info['id']} task={self.info['task']} log={self.directory}", flush=True)
    return self

  @staticmethod
  def interrupt(signum, frame):
    raise KeyboardInterrupt(f"signal {signum}")

  def __exit__(self, kind, error, tb):
    try:
      if error is None:
        self.update(state="completed", finished_at=time.time())
      else:
        traceback.print_exception(kind, error, tb, file=self.log)
        self.update(state="stopped" if issubclass(kind, (KeyboardInterrupt, SystemExit)) else "failed",
                    finished_at=time.time(), error=f"{kind.__name__}: {error}")
    finally:
      sys.stdout, sys.stderr = self.stdout, self.stderr
      for sig, handler in self.signals.items():
        signal.signal(sig, handler)
      self.log.close()
    return False


class WorkerRecord(RunRecord):
  """Rank-local logging/status; only rank zero may mutate run_info.json."""

  def __init__(self, directory, info, rank, local_rank):
    super().__init__(directory, dict(info))
    self.rank, self.local_rank = rank, local_rank

  @property
  def log_path(self):
    return self.directory / f"rank_{self.rank}.log"

  def update(self, **changes):
    self.info = {**self.info, **changes, "updated_at": time.time()}
    self._write_status()

  def _write_status(self):
    status = {**self.info, "rank": self.rank, "local_rank": self.local_rank,
              "pid": os.getpid(), "host": socket.gethostname(), "log_file": self.log_path.name}
    atomic_json(self.directory / f"rank_{self.rank}.json", status)

  def __enter__(self):
    self.update(state="starting")
    self._write_status()
    try:
      return super().__enter__()
    except BaseException as error:
      self.update(state="failed", finished_at=time.time(), error=f"{type(error).__name__}: {error}")
      self._write_status()
      raise

  def __exit__(self, kind, error, tb):
    try:
      return super().__exit__(kind, error, tb)
    finally:
      self._write_status()
