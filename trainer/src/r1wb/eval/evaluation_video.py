"""Local, app-owned, bounded-memory evaluation videos; physics stays with callers."""
from __future__ import annotations

from datetime import datetime
import hashlib
import io
import json
from importlib import metadata as package_metadata
import os
from pathlib import Path
import select
import shutil
import signal
import platform
import sys
import subprocess
import threading
import time
from urllib.parse import quote
import uuid
import tempfile
import zipfile

from r1wb.eval.paths import AGENT, ROOT

PROJECT = ROOT.parent
RESULT = AGENT / "result"
_BOUND = set()


def _json(path, value):
  path = Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".video-", delete=False) as stream:
    temporary = Path(stream.name)
    json.dump(value, stream, indent=2, allow_nan=False,
              default=lambda item: item.tolist() if hasattr(item, "tolist") else str(item))
    stream.write("\n")
  try:
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def configure_local_video():
  """Verify the PC repository layout before any rendering or output creation."""
  root = PROJECT.resolve()
  if (root == Path("/rl-agent") or Path("/rl-agent") in root.parents
      or not (root / "app/operation_center/simulation_worker.py").is_file()
      or not (root / "trainer/src/r1lab").is_dir()):
    raise ValueError("Evaluation videos are local-PC only; /rl-agent/server rendering is forbidden")
  os.environ.setdefault("MUJOCO_GL", "egl")
  return RESULT.resolve()


def _safe(value):
  result = "".join(c if c.isalnum() or c in "_-@." else "_" for c in str(value))
  if result in ("", ".", ".."):
    raise ValueError("A nonempty video identifier is required")
  return result


def infer_run_id(policy_path=None, explicit=None):
  selected = os.environ.get("RL_EVALUATION_RUN_ID") or explicit
  if selected:
    return _safe(selected)
  if policy_path is not None:
    policy = Path(policy_path).resolve()
    for directory in [policy.parent, *list(policy.parents)[1:5]]:
      record = directory / "run_info.json"
      if record.is_file():
        info = json.loads(record.read_text())
        if info.get("id"):
          return _safe(info["id"])
    if "_n" in policy.parent.name and "_20" in policy.parent.name:
      return _safe(policy.parent.name)
  return "evaluation_" + datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")


def new_video_path(run_id, case_id, policy):
  root = configure_local_video()
  stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S-%f")
  return root / "video" / _safe(run_id) / f"{stamp}_{_safe(policy)}_{_safe(case_id)}_{uuid.uuid4().hex[:8]}.mp4"


def _owner_from_ancestors():
  pid = os.getpid()
  for _ in range(24):
    try:
      base = Path("/proc") / str(pid)
      args = base.joinpath("cmdline").read_bytes().decode(errors="replace").split("\0")
      for flag in ("--app-pid", "--owner-pid"):
        if flag in args:
          return int(args[args.index(flag)+1])
      pid = int(base.joinpath("stat").read_text().split(") ", 1)[1].split()[1])
      if pid <= 1:
        break
    except (OSError, ValueError, IndexError):
      break
  return None


def _terminate_children(pid):
  try:
    children = (Path("/proc") / str(pid) / "task" / str(pid) / "children").read_text().split()
  except OSError:
    children = []
  for child in children:
    _terminate_children(int(child))
    try:
      os.kill(int(child), signal.SIGTERM)
    except ProcessLookupError:
      pass


def require_app_owner(*, bind=True):
  """Return the active owner PID; CLI binding stops its descendants on app exit."""
  value = os.environ.get("RL_EVALUATION_APP_PID")
  try:
    owner = int(value) if value else _owner_from_ancestors()
    if owner is None or owner <= 1:
      raise ValueError("No app owner: use with_app.py --app-pid or --app-pid/RL_EVALUATION_APP_PID")
    fd = os.pidfd_open(owner)
    if select.select([fd], [], [], 0)[0]:
      os.close(fd)
      raise ValueError("Evaluation app owner is already closed")
  except (OSError, TypeError) as error:
    raise ValueError(f"Evaluation app owner is unavailable: {error}") from error
  os.environ["RL_EVALUATION_APP_PID"] = str(owner)
  identity = (os.getpid(), owner)
  if not bind or identity in _BOUND:
    os.close(fd)
    return owner
  _BOUND.add(identity)
  def watch():
    try:
      select.select([fd], [], [])
      _terminate_children(os.getpid())
      os.kill(os.getpid(), signal.SIGTERM)
    finally:
      os.close(fd)
  threading.Thread(target=watch, daemon=True, name="evaluation-app-owner").start()
  return owner


class StreamingVideoWriter:
  """Stream RGB uint8 frames to fragmented MP4; never keep a frame collection."""
  def __init__(self, path, fps=30, width=960, height=600, *, label=None, min_free_bytes=512*1024**2):
    root = configure_local_video()
    self.path = Path(path).resolve()
    if not self.path.is_relative_to(root / "video") or self.path.suffix.lower() != ".mp4":
      raise ValueError("Video path must be a .mp4 below local .agent/trainer/result/video/")
    relative = self.path.relative_to(root / "video")
    if len(relative.parts) != 2:
      raise ValueError("Video path must use video/<runID>/<timestamp_filename>.mp4")
    if not (fps > 0 and width > 0 and height > 0 and width % 2 == height % 2 == 0):
      raise ValueError("Require positive fps and positive even dimensions")
    self.fps, self.width, self.height = float(fps), int(width), int(height)
    self.label, self.min_free_bytes = label, int(min_free_bytes)
    self.frames = 0
    self.started_at = time.time()
    self.process = self.log = None
    self.error_kind = self.reason = None
    self.closed = False
    self.owner = None
    self._stop = threading.Event()
    try:
      self.owner = require_app_owner(bind=False)
    except ValueError as error:
      self.fail("owner", str(error))

  def fail(self, kind, reason):
    if self.reason is None:
      self.error_kind, self.reason = kind, str(reason)
      self._save("partial" if self.frames else "unavailable")

  def _save(self, status):
    try:
      _json(self.path.with_suffix(".video.json"), self.metadata(status))
    except OSError:
      pass  # The caller still receives the reason when the filesystem is full.

  def _start(self):
    try:
      self.path.parent.mkdir(parents=True, exist_ok=True)
      if self.path.exists():
        self.fail("destination", "Existing video preserved; refusing overwrite")
        return
      if shutil.disk_usage(self.path.parent).free < self.min_free_bytes:
        self.fail("disk_space", "Free disk space is below the video reserve; no files deleted")
        return
      self.log = self.path.with_suffix(".encoder.log").open("wb")
      self._save("recording")
      self.process = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-n",
          "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{self.width}x{self.height}",
          "-r", str(self.fps), "-i", "-", "-an", "-c:v", "libx264", "-threads", "1",
          "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags",
          "+frag_keyframe+empty_moov+default_base_moof", "-frag_duration", "1000000", str(self.path)],
          stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self.log)
      fd = os.pidfd_open(self.owner)
      def watch():
        try:
          while not self._stop.is_set():
            if select.select([fd], [], [], .1)[0]:
              self.fail("owner", "App closed during video recording; partial stream preserved")
              if self.process.poll() is None:
                self.process.terminate()
              break
        finally:
          os.close(fd)
      threading.Thread(target=watch, daemon=True, name="video-encoder-owner").start()
    except (OSError, ValueError) as error:
      self.fail("encoder", error)

  def write(self, frame):
    if self.closed or self.reason:
      return False
    if self.process is None:
      self._start()
    if self.reason:
      return False
    try:
      import numpy as np
      array = np.asarray(frame)
      if array.shape != (self.height, self.width, 3) or array.dtype != np.uint8:
        raise ValueError("Expected RGB uint8 frame matching the configured dimensions")
      if self.frames % max(1, round(self.fps)) == 0 and shutil.disk_usage(self.path.parent).free < self.min_free_bytes:
        self.fail("disk_space", "Video stopped at the disk reserve; partial stream preserved, no files deleted")
        return False
      self.process.stdin.write(array.tobytes())
      self.frames += 1
      if self.frames % max(1, round(self.fps)) == 0:
        self._save("recording")
      return True
    except (OSError, ValueError) as error:
      self.fail("encoder", error)
      return False

  def close(self):
    if not self.closed:
      self.closed = True
      if self.process is not None:
        try:
          self.process.stdin.close()
        except OSError:
          pass
        try:
          code = self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
          self.process.terminate()
          try:
            self.process.wait(timeout=2)
          except subprocess.TimeoutExpired:
            self.process.kill(); self.process.wait(timeout=2)
          code = -1
        if code:
          self.fail("encoder", f"ffmpeg exited {code}; any partial stream is preserved")
      self._stop.set()
      if self.log is not None:
        self.log.close()
      if not self.frames and self.reason is None:
        self.fail("empty", "No video frames were captured")
    metadata = self.metadata()
    self._save(metadata["status"])
    return metadata

  def metadata(self, status=None):
    size = self.path.stat().st_size if self.path.exists() else 0
    status = status or ("written" if self.frames and self.reason is None else "partial" if self.frames and size else "unavailable")
    run_id = self.path.parent.name
    return dict(protocol="evaluation-video-v1", status=status, complete=status == "written",
        reason=self.reason, error_kind=self.error_kind, path=str(self.path),
        relative_path=self.path.relative_to(RESULT.resolve()).as_posix(),
        url=f"/trainer-artifact/{quote(run_id, safe='')}/video/{quote(self.path.name, safe='')}",
        frames=self.frames, fps=self.fps, width=self.width, height=self.height, bytes=size,
        app_pid=self.owner, started_at=self.started_at, sim_start_s=None, sim_end_s=None,
        frame_times_s=getattr(self, "frame_times_s", []),
        container="fragmented-mp4", encoder_log=str(self.path.with_suffix(".encoder.log")),
        sidecar=str(self.path.with_suffix(".video.json")),
        interrupted_interpretation="recording without a completed marker means interrupted; inspect partial stream")


class EvaluationVideoRecorder:
  """Render a copied CPU MjData at 30 Hz without stepping or changing source data."""
  def __init__(self, model, data, path, *, fps=30, width=960, height=600, label=None,
               root_position=None, min_free_bytes=512*1024**2, metadata=None, context=None, record_replay=True):
    self.model, self.data, self.root_position = model, data, root_position
    self.writer = StreamingVideoWriter(path, fps, width, height, label=label, min_free_bytes=min_free_bytes)
    self.renderer = self.render_data = self.camera = self.option = None
    self.first_time = self.last_time = None
    self.next_time = float(data.time)
    self.closed = False
    self.frame_times_s = self.writer.frame_times_s = []
    self.context = {**(metadata or {}), **(context or {})}
    self.replay = ReplayBundle(model, data, path, self.context, min_free_bytes) if record_replay else None

  def record_event(self, name, payload):
    if self.replay is not None:
      self.replay.record_event(name, payload)

  def capture(self, force=False):
    now = float(self.data.time)
    if not self.closed and self.replay is not None:
      self.replay.capture(force=force)
    if self.closed or self.writer.reason or (not force and now+1e-9 < self.next_time):
      return False
    if self.last_time is not None and abs(now-self.last_time) < 1e-9:
      return False
    try:
      import mujoco
      if self.renderer is None:
        self.render_data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=self.writer.height, width=self.writer.width)
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 135., -20., 4.8
        self.option = mujoco.MjvOption()
        self.option.geomgroup[:] = [1, 1, 1, 0, 0, 0]
      mujoco.mj_copyData(self.render_data, self.model, self.data)
      position = self.root_position() if callable(self.root_position) else self.root_position
      self.camera.lookat[:] = self.render_data.qpos[:3] if position is None else position
      self.renderer.update_scene(self.render_data, camera=self.camera, scene_option=self.option)
      frame = self.renderer.render()
      self.frame_times_s.append(now)
      if not self.writer.write(frame):
        self.frame_times_s.pop()
        return False
      self.first_time = now if self.first_time is None else self.first_time
      self.last_time = now
      self.next_time = self.first_time+self.writer.frames/self.writer.fps
      return True
    except Exception as error:
      self.writer.fail("renderer", f"{type(error).__name__}: {error}")
      return False

  def close(self):
    if not self.closed:
      self.capture(force=True)
      self.closed = True
      if self.renderer is not None:
        self.renderer.close()
    metadata = self.writer.close()
    metadata.update(sim_start_s=self.first_time, sim_end_s=self.last_time,
                    source_data="copied with mj_copyData; renderer never receives physics MjData",
                    replay=self.replay.close() if self.replay is not None else dict(
                        complete=False, status="source_archive", manifest=self.context.get("source_state_manifest"),
                        reason="Visual replay only; original state archive is authoritative. Synthetic controls/forces are not recorded."))
    try:_json(self.writer.path.with_suffix(".video.json"), metadata)
    except OSError:pass
    return metadata


class ReplayBundle:
  """50Hz state stream in atomic one-second ZIP chunks, plus shared MJB/code assets."""
  INPUT_FIELDS = ("qpos", "qvel", "act", "ctrl", "qfrc_applied", "xfrc_applied", "qacc_warmstart",
                  "mocap_pos", "mocap_quat", "eq_active", "userdata")
  DATA_FIELDS = INPUT_FIELDS + ("qfrc_constraint", "qfrc_actuator", "actuator_force")
  MODEL_FIELDS = ("geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_rgba", "geom_contype",
      "geom_conaffinity", "geom_group", "geom_priority", "geom_solref", "geom_solimp", "body_mass",
      "body_inertia", "body_gravcomp", "dof_damping", "dof_armature", "dof_frictionloss")

  def __init__(self, model, data, video_path, context=None, min_free_bytes=512*1024**2):
    self.model, self.data = model, data
    self.context = json.loads(json.dumps(dict(context or {}), allow_nan=False,
        default=lambda item:item.tolist() if hasattr(item,"tolist") else str(item)))
    self.video_path = Path(video_path).resolve()
    self.manifest = self.video_path.with_suffix(".replay.json")
    self.directory = self.video_path.with_suffix(".states")
    self.min_free_bytes = min_free_bytes
    self.assets, self.chunks = {}, []
    self.total_frames = self.committed_frames = self.chunk_frames = 0
    self.first_time = self.last_time = self.chunk_start = None
    self.next_time = float(data.time)
    self.archive = self.temporary = None
    self.model_hash = None
    self.reason = self.error_kind = None
    self.started = self.closed = False

  def _asset(self, body, suffix):
    digest = hashlib.sha256(body).hexdigest()
    path = self.video_path.parent / "assets" / (digest+suffix)
    if not path.exists():
      path.parent.mkdir(parents=True, exist_ok=True)
      if shutil.disk_usage(path.parent).free < self.min_free_bytes+len(body):
        raise OSError("Insufficient disk reserve for replay assets")
      with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".asset-", delete=False) as stream:
        tmp = Path(stream.name); stream.write(body)
      os.replace(tmp, path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
      raise ValueError("Content-addressed replay asset hash differs")
    return dict(path=str(path), sha256=digest, bytes=len(body), verified=True)

  def _start(self):
    import mujoco
    import numpy as np
    require_app_owner(bind=False)
    self.directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(self.directory).free < self.min_free_bytes:
      raise OSError("Insufficient disk reserve for replay state recording")
    buffer = np.empty(mujoco.mj_sizeModel(self.model), dtype=np.uint8)
    mujoco.mj_saveModel(self.model, buffer=buffer)
    self.assets["model"] = self._asset(buffer.tobytes(), ".mjb")
    policy = self.context.get("policy_path")
    if policy and Path(policy).is_file():
      self.assets["policy"] = self._asset(Path(policy).read_bytes(), Path(policy).suffix)
    source_files = [Path(__file__), PROJECT/"app/operation_center/simulation_worker.py",
                    ROOT/"src/r1lab/contract.py", ROOT/"src/r1lab/terrain_observation.py",
                    AGENT/"tools/eval_hard_cases.py", AGENT/"tools/_external_events.py"]
    source_files += [Path(path) for path in self.context.get("source_files", [])]
    self.assets["sources"] = {}
    for source in dict.fromkeys(source_files):
      if source.is_file():
        self.assets["sources"][str(source)] = self._asset(source.read_bytes(), source.suffix)
    self.physics = dict(mujoco_version=mujoco.__version__, timestep_s=float(self.model.opt.timestep),
        solver=int(self.model.opt.solver), integrator=int(self.model.opt.integrator),
        iterations=int(self.model.opt.iterations), tolerance=float(self.model.opt.tolerance),
        gravity=self.model.opt.gravity.tolist())
    versions={}
    for package in ("numpy","onnxruntime","mujoco","torch","mjlab","mujoco-warp"):
      try:versions[package]=package_metadata.version(package)
      except package_metadata.PackageNotFoundError:versions[package]=None
    self.physics.update(python_version=sys.version,platform=platform.platform(),package_versions=versions)
    self.started = True
    self._publish(False)

  def _publish(self, complete):
    _json(self.manifest, self.metadata(complete))

  def metadata(self, complete=False):
    return dict(protocol="evaluation-replay-v1", complete=bool(complete and not self.reason),
        status="written" if complete and not self.reason else "partial" if self.committed_frames else "recording" if self.started and not self.reason else "unavailable",
        reason=self.reason, error_kind=self.error_kind, path=str(self.manifest), manifest=str(self.manifest),
        video_path=str(self.video_path), state_hz=50., frames=self.committed_frames,
        captured_frames=self.total_frames, sim_start_s=self.first_time,
        sim_end_s=self.chunks[-1]["end_s"] if self.chunks else None,
        physics=getattr(self, "physics", None), context=self.context, assets=self.assets, chunks=self.chunks,
        model_change_sampling="Full model-field snapshots when changed, sampled at50Hz; explicit record_event timestamps supplement this",
        replay_capability="Stored-state visual replay. Exact deterministic resimulation additionally requires complete controller/history/RNG and substep inputs; not asserted.",
        force_sampling_limit="Forces/contact wrenches are50Hz observations; substep peaks and between-sample forces are not retained.",
        event_log=str(self.video_path.with_suffix(".events.jsonl")),
        interruption="Only atomically completed chunks are valid. Ignore any .zip.part current chunk; original files are never deleted.")

  def _commit(self):
    if self.archive is None:
      return
    self.archive.close()
    final = self.temporary.with_suffix("")
    os.replace(self.temporary, final)
    self.chunks.append(dict(path=str(final), frames=self.chunk_frames, start_frame=self.committed_frames,
        start_s=self.chunk_start, end_s=self.last_time, sha256=hashlib.sha256(final.read_bytes()).hexdigest()))
    self.committed_frames += self.chunk_frames
    self.archive = self.temporary = None
    self.chunk_frames = 0
    self._publish(False)

  def capture(self, force=False):
    now = float(self.data.time)
    if self.closed or self.reason or (not force and now+1e-9 < self.next_time):
      return False
    if self.last_time is not None and abs(now-self.last_time)<1e-9:
      return False
    try:
      import numpy as np
      import mujoco
      if self.last_time is not None and not 0<now-self.last_time<=.02100001:
        raise ValueError("State capture skipped50Hz samples; motion stream is incomplete")
      if not self.started:
        self._start()
      if shutil.disk_usage(self.directory).free < self.min_free_bytes:
        raise OSError("Disk reserve reached; completed replay chunks preserved")
      if self.archive is None:
        self.temporary = self.directory / f"{len(self.chunks):06d}.zip.part"
        self.archive = zipfile.ZipFile(self.temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1)
        self.chunk_start = now
      values = {name: np.asarray(getattr(self.data,name)) for name in self.DATA_FIELDS}
      values["time"] = np.asarray(now)
      values["terminal_sample"] = np.asarray(bool(force))
      count=int(self.data.ncon)
      values["contact_geom"] = np.asarray([[c.geom1,c.geom2] for c in self.data.contact],dtype=np.int32).reshape(count,2)
      values["contact_position"] = np.asarray([c.pos for c in self.data.contact],dtype=float).reshape(count,3)
      values["contact_frame"] = np.asarray([c.frame for c in self.data.contact],dtype=float).reshape(count,3,3)
      forces=np.zeros((count,6))
      for index in range(count):mujoco.mj_contactForce(self.model,self.data,index,forces[index])
      values["contact_force"] = forces
      digest = hashlib.sha256()
      for name in self.MODEL_FIELDS:
        digest.update(np.asarray(getattr(self.model,name)).tobytes())
      changed = digest.hexdigest()
      if changed != self.model_hash:
        values.update({"model__"+name:np.asarray(getattr(self.model,name)) for name in self.MODEL_FIELDS})
        self.model_hash = changed
      stream = io.BytesIO(); np.savez(stream, **values)
      self.archive.writestr(f"{self.total_frames:08d}.npz", stream.getvalue())
      self.first_time = now if self.first_time is None else self.first_time
      self.last_time = now
      self.total_frames += 1; self.chunk_frames += 1
      self.next_time = self.first_time+self.total_frames*.02
      if self.chunk_frames >= 50:
        self._commit()
      return True
    except Exception as error:
      self.reason = f"{type(error).__name__}: {error}"
      self.error_kind = "disk_space" if isinstance(error,OSError) else "replay"
      try:self._publish(False)
      except OSError:pass
      return False

  def record_event(self, name, payload):
    if not self.started:
      self.capture(force=True)
    if not self.started or self.reason:
      return
    try:
      with self.video_path.with_suffix(".events.jsonl").open("a") as stream:
        json.dump(dict(time_s=float(self.data.time),name=name,payload=payload),stream,
                  default=lambda item:item.tolist() if hasattr(item,"tolist") else str(item),allow_nan=False)
        stream.write("\n")
    except (OSError,ValueError) as error:
      self.reason=str(error);self.error_kind="events"

  def close(self):
    if not self.closed:
      self.capture(force=True)
      try:
        self._commit()
      except OSError as error:
        self.reason=str(error);self.error_kind="disk_space"
      self.closed=True
    metadata=self.metadata(complete=self.started and not self.reason)
    try:_json(self.manifest,metadata)
    except OSError:pass
    return metadata


def iter_replay_states(manifest_path):
  """Read only verified, atomically completed chunks, including interrupted runs."""
  import numpy as np
  manifest=json.loads(Path(manifest_path).read_text())
  for chunk in manifest["chunks"]:
    path=Path(chunk["path"])
    if hashlib.sha256(path.read_bytes()).hexdigest()!=chunk["sha256"]:
      raise ValueError("Replay chunk hash differs")
    with zipfile.ZipFile(path) as archive:
      for name in sorted(archive.namelist()):
        with np.load(io.BytesIO(archive.read(name)),allow_pickle=False) as arrays:
          yield {key:arrays[key].copy() for key in arrays.files}


def restore_replay_state(model, data, state):
  """Restore a stored sample into a separate replay model/data, then forward it."""
  import mujoco
  for name,value in state.items():
    if name.startswith("model__"):
      getattr(model,name[7:])[:] = value
    elif name in ReplayBundle.INPUT_FIELDS:
      getattr(data,name)[:] = value
  data.time=float(state["time"])
  mujoco.mj_forward(model,data)


def verify_replay(manifest_path):
  manifest=json.loads(Path(manifest_path).read_text())
  def verify_assets(value):
    if not isinstance(value,dict):return
    if "path" in value and "sha256" in value:
      if hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest()!=value["sha256"]:
        raise ValueError("Replay asset hash differs")
    else:
      for entry in value.values():verify_assets(entry)
  verify_assets(manifest["assets"])
  count=0;previous=None;maximum_gap=0.
  for state in iter_replay_states(manifest_path):
    now=float(state["time"])
    if previous is not None:
      delta=now-previous
      if not 0<delta<=.02100001:
        raise ValueError(f"Replay timeline is discontinuous: gap{delta}s")
      maximum_gap=max(maximum_gap,delta)
    previous=now;count+=1
  if count!=manifest["frames"]:
    raise ValueError("Replay frame count differs from completed chunks")
  return dict(valid=True,complete=manifest["complete"],frames=count,maximum_gap_s=maximum_gap,
              scope="completed chunks only; stored-state visual replay")
