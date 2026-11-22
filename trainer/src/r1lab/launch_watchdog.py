"""Supervise only an official launch's child session, without importing GPU code."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time
import uuid

from r1lab.run_record import atomic_json, refreshed_workers, run_info_lock, slug, work_root


def _read(path):
  path = Path(path)
  if path.stat().st_size > 2 * 1024 * 1024:
    raise ValueError(f"Oversized launch record: {path}")
  value = json.loads(path.read_text())
  if not isinstance(value, dict):
    raise ValueError(f"Invalid launch record: {path}")
  return value


def _stat(pid):
  try:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return int(fields[3]), int(fields[19])  # session and starttime; comm may contain spaces.
  except (OSError, ValueError, IndexError):
    return None


class OwnedSession:
  """Pin identities with pidfds; never signal numeric PIDs from a run record."""

  def __init__(self, child, launch_id, token):
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
      raise RuntimeError("Official launch supervision requires Linux pidfds")
    self.child, self.launch_id, self.token = child, launch_id, token
    self.handles = {}
    self.refresh()

  def refresh(self):
    for entry in Path("/proc").iterdir():
      if not entry.name.isdigit():
        continue
      pid = int(entry.name)
      identity = _stat(pid)
      if identity is None or identity[0] != self.child.pid:
        continue
      if pid in self.handles:
        continue
      try:
        environment = (entry / "environ").read_bytes().split(b"\0")
        if (f"RL_LAUNCH_ID={self.launch_id}".encode() not in environment
            or f"RL_SUPERVISOR_TOKEN={self.token}".encode() not in environment):
          continue
        descriptor = os.pidfd_open(pid)
        if _stat(pid) != identity:
          os.close(descriptor)
          continue
        self.handles[pid] = descriptor
      except (OSError, ProcessLookupError):
        continue

  def alive(self):
    return [pid for pid, descriptor in self.handles.items() if not select.select([descriptor], [], [], 0)[0]]

  def send(self, signum):
    self.refresh()
    sent = []
    for pid in self.alive():
      try:
        signal.pidfd_send_signal(self.handles[pid], signum)
        sent.append(pid)
      except ProcessLookupError:
        pass
    return sent

  def close(self):
    for descriptor in self.handles.values():
      os.close(descriptor)


def _bind(work, launch_id, token, expected_world_size, expected_task):
  handoff = Path(work) / "tmp" / "distributed" / f"{launch_id}.json"
  try:
    payload = _read(handoff)
  except FileNotFoundError:
    return None
  if payload.get("schema_version") != 1 or payload.get("launch_id") != launch_id:
    raise ValueError("Launch handoff identity does not match this supervisor")
  if "error" in payload:
    raise RuntimeError(f"Rank zero registration failed: {payload['error']}")
  directory = Path(payload["directory"]).resolve()
  root = (Path(work) / "runs").resolve()
  if directory.parent.parent != root or not directory.is_relative_to(root):
    raise ValueError("Launch handoff is outside the canonical run root")
  current = _read(directory / "run_info.json")
  identity = current.get("supervision", {})
  if (current.get("id") != directory.name or current.get("launch_id") != launch_id
      or current.get("world_size") != expected_world_size or current.get("task") != expected_task
      or identity.get("pid") != os.getpid() or identity.get("token") != token
      or not current.get("supervised_launch")):
    raise ValueError("Registered run does not belong to this supervisor")
  return directory


def _snapshot(directory, launch_id, token):
  current = _read(directory / "run_info.json")
  if (current.get("launch_id") != launch_id or current.get("supervision", {}).get("token") != token
      or current.get("id") != directory.name):
    raise ValueError("Registered run identity changed")
  current["workers"] = refreshed_workers(directory, current)
  return current


def _publish(directory, launch_id, token, **changes):
  with run_info_lock(directory):
    current = _snapshot(directory, launch_id, token)
    supervision = {**current.get("supervision", {}), **changes, "updated_at": time.time()}
    current["supervision"] = supervision
    current["workers"] = refreshed_workers(directory, current)
    for name in ("state", "error", "finished_at", "launcher_exit_code", "supervisor_exit_code"):
      if name in supervision:
        current[name] = supervision[name]
    if changes.get("phase") == "finished":
      current["finalization_pending"] = False
    current["updated_at"] = time.time()
    atomic_json(directory / "run_info.json", current)


def _pending_forensics(directory, workers):
  pending = []
  for worker in workers:
    name = worker.get("failure_diagnostics")
    # The cleanup signal can change failed -> stopped while an advertised dump
    # is still incomplete. Keep that evidence pending after the state change.
    if not name:
      continue
    if Path(name).name != name:
      pending.append(worker["rank"])
      continue
    try:
      finished = _read(directory / name).get("capture_finished") is True
    except (OSError, ValueError):
      finished = False
    if not finished:
      pending.append(worker["rank"])
  return pending


def supervise(command, *, launch_id, expected_world_size, expected_task, work=None,
              forensic_grace=45.0, shutdown_grace=10.0, poll_interval=0.2, registration_timeout=120.0):
  """Return a shell exit code; child return code and terminal state are persisted.

  Failed ranks are published immediately. Only advertised incomplete forensic
  dumps receive the bounded grace; SIGINT/SIGTERM from a user propagate at once.
  """
  if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
    raise RuntimeError("Official launch supervision requires Linux pidfds")
  launch_id = slug(launch_id)
  work = Path(work or work_root()).resolve()
  handoff = work / "tmp" / "distributed" / f"{launch_id}.json"
  handoff.parent.mkdir(parents=True, exist_ok=True)
  if handoff.exists() or handoff.with_suffix(".lock").exists():
    raise RuntimeError("Refusing to attach a new child to an existing launch identity")
  lock = handoff.with_suffix(".supervisor.lock")
  descriptor = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
  os.close(descriptor)
  token = uuid.uuid4().hex
  environment = {**os.environ, "RL_LAUNCH_ID": launch_id, "RL_SUPERVISED_LAUNCH": "1",
                 "RL_SUPERVISOR_PID": str(os.getpid()), "RL_SUPERVISOR_TOKEN": token}
  requested = []
  previous = {}
  for signum in (signal.SIGINT, signal.SIGTERM):
    previous[signum] = signal.getsignal(signum)
    signal.signal(signum, lambda number, frame: requested.append(number))
  child, session, directory = None, None, None
  terminal, reason, grace_deadline, stop_started, killed_at = None, None, None, None, None
  final_code, child_code, workers, pending = 1, None, [], []
  summary = {"launch_id": launch_id, "supervisor_pid": os.getpid(), "started_at": time.time(),
             "forensic_grace_seconds": forensic_grace, "shutdown_grace_seconds": shutdown_grace,
             "phase": "starting"}
  summary_path = handoff.with_suffix(".supervisor.json")

  def publish(**changes):
    nonlocal directory
    summary.update(changes)
    atomic_json(summary_path, summary)
    if directory is not None:
      try:
        _publish(directory, launch_id, token, **changes)
      except (OSError, ValueError, KeyError) as error:
        summary.setdefault("publication_errors", []).append(str(error))
        directory = None
        atomic_json(summary_path, summary)

  try:
    child = subprocess.Popen(command, env=environment, start_new_session=True)
    session = OwnedSession(child, launch_id, token)
    started = time.monotonic()
    publish(launcher_pid=child.pid, phase="registering")
    while True:
      now = time.monotonic()
      session.refresh()
      try:
        if directory is None:
          directory = _bind(work, launch_id, token, expected_world_size, expected_task)
          if directory is not None:
            publish(run_directory=str(directory), phase="running")
        current = _snapshot(directory, launch_id, token) if directory is not None else {}
        workers = current.get("workers", [])
        pending = _pending_forensics(directory, workers) if directory is not None else []
      except (OSError, ValueError, RuntimeError, KeyError) as error:
        if terminal is None:
          terminal, reason = "failed", f"Launch registration/status error: {error}"
          grace_deadline = now
        # Never write through an unverified or changed run identity.
        directory = None
      child_code = child.poll()
      if requested and stop_started is None:
        if terminal is None:
          terminal, reason = "stopped", f"User signal {requested[0]} received by launch supervisor"
        publish(state=terminal, error=reason, phase="stopping", user_signal=requested[0])
        session.send(requested[0])
        stop_started = now
      failed = [worker for worker in workers if worker.get("state") == "failed"]
      stopped = [worker for worker in workers if worker.get("state") == "stopped"]
      if terminal is None and (failed or stopped):
        source = (failed or stopped)[0]
        terminal = "failed" if failed else "stopped"
        reason = f"Rank {source['rank']} {terminal}: {source.get('error', 'no error detail')}"
        grace_deadline = now + forensic_grace if failed else now
        publish(state=terminal, error=reason, phase="forensic_grace" if failed else "stopping",
                failure_detected_at=time.time(), failed_ranks=[worker["rank"] for worker in failed])
      if terminal is None and directory is None and now - started >= registration_timeout:
        terminal, reason, grace_deadline = "failed", "Launch registration timed out", now
        publish(state=terminal, error=reason, phase="stopping")
      if child_code is not None and terminal is None:
        all_complete = (len(workers) == expected_world_size
                        and {worker.get("rank") for worker in workers} == set(range(expected_world_size))
                        and all(worker.get("state") == "completed" for worker in workers))
        if child_code == 0 and directory is not None and all_complete and not session.alive():
          terminal, reason = "completed", None
        else:
          terminal = "failed"
          reason = (f"Launcher exited with code {child_code}" if child_code != 0
                    else "Launcher exited zero without every registered rank completed and exited")
        grace_deadline = now + forensic_grace if pending else now
        publish(state=terminal, error=reason, phase="forensic_grace" if pending else "stopping")
      if terminal is not None and terminal != "completed" and stop_started is None:
        if not pending or now >= (grace_deadline or now):
          publish(state=terminal, error=reason, phase="stopping",
                  forensic_timeout=bool(pending), pending_forensic_ranks=pending)
          session.send(signal.SIGTERM)
          stop_started = now
      alive = session.alive()
      if child_code is not None and not alive:
        break
      if stop_started is not None and now - stop_started >= shutdown_grace and killed_at is None:
        publish(phase="killing", remaining_owned_processes=alive, forced_termination=True)
        session.send(signal.SIGKILL)
        killed_at = now
      if killed_at is not None and now - killed_at >= 5.0:
        # A kernel task can resist even SIGKILL. Report residual identities;
        # never turn that into an unbounded wait or claim that GPUs are free.
        break
      time.sleep(poll_interval)
    child_code = child.poll()
    final_code = (128 + requested[0] if requested else
                  0 if terminal == "completed" else (128 - child_code if child_code is not None and child_code < 0
                                                     else child_code if child_code else 1))
    publish(state=terminal or "failed", error=reason, phase="finished", finished_at=time.time(),
            launcher_exit_code=child_code, supervisor_exit_code=final_code,
            remaining_owned_processes=session.alive(), pending_forensic_ranks=pending,
            forensic_incomplete=bool(pending))
    return final_code
  except BaseException as error:
    # Even an I/O/supervisor exception must not leave the child it created
    # running indefinitely. Do not broaden identity checks to make cleanup work.
    if session is not None:
      session.send(signal.SIGTERM)
      deadline = time.monotonic() + shutdown_grace
      while session.alive() and time.monotonic() < deadline:
        time.sleep(poll_interval)
      if session.alive():
        session.send(signal.SIGKILL)
    try:
      publish(state="failed", error=f"Supervisor error: {type(error).__name__}: {error}", phase="finished",
              finished_at=time.time(), launcher_exit_code=child.poll() if child is not None else None,
              supervisor_exit_code=1, remaining_owned_processes=session.alive() if session is not None else [])
    except Exception:
      pass
    raise
  finally:
    for signum, handler in previous.items():
      signal.signal(signum, handler)
    if session is not None:
      session.close()


def main(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--world-size", required=True, type=int)
  parser.add_argument("--task", required=True)
  parser.add_argument("command", nargs=argparse.REMAINDER)
  args = parser.parse_args(argv)
  command = args.command[1:] if args.command[:1] == ["--"] else args.command
  if not command or args.world_size < 1 or not os.environ.get("RL_LAUNCH_ID"):
    parser.error("A command, positive world size and RL_LAUNCH_ID are required")
  return supervise(command, launch_id=os.environ["RL_LAUNCH_ID"], expected_world_size=args.world_size,
                   expected_task=args.task)


if __name__ == "__main__":
  raise SystemExit(main())
