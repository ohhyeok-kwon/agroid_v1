"""Evaluation-only shared sensor stress, injected into copies at the actor boundary.

N1_shared_sensor_v1 is NOT the historical training-channel N1. It has no delay,
no ground/depth noise and no claim of measured R1 sensor accuracy. Physics, plans,
rewards and truth used for metrics are never written. Manufacturer data is not used
to select the noise amplitudes. N0 returns the original actor input object.
"""
from __future__ import annotations

import atexit
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


PROFILES = ("N0", "N1_shared_sensor_v1")
CHANNELS = (("gyro", 3, .2), ("attitude", 3, .05), ("joint_pos", 24, .01), ("joint_vel", 24, .5))


def profile_metadata(profile: str) -> dict:
  if profile not in PROFILES:
    raise ValueError(f"Unknown observation profile: {profile}")
  return dict(
    name=profile, version=1, delay_control_steps=0,
    distribution="independent zero-mean uniform per named physical channel/axis/control frame",
    amplitudes={name: amplitude if profile != "N0" else 0. for name, _, amplitude in CHANNELS},
    attitude="pelvis-frame roll/pitch rotation vector; yaw error zero; transformed into anchor frame for teacher",
    previous_action="exact previously commanded action; no invented action sensor noise",
    history="manufacturer oldest-first 10 frames: initial sample repeated, then retain each sampled noisy frame",
    generator="NumPy Philox(SeedSequence([noise_seed, nominal_control_step, channel_index]))",
    scope="observation copies only; noise_seed independent of policy, environment batch index, global RNG and force clock",
    limitations=["research stress amplitudes, not measured R1 sensor specifications",
                 "not equivalent to historical N1 training-channel noise; no observation delay",
                 "no depth/terrain/height/localization noise; teacher true linear velocity and goal lag remain privileged",
                 "anchor frame conversion uses true relative FK; encoder noise does not rebuild that FK",
                 "equal sensor errors do not make different actor information sets equivalent"],
  )


def read_seeds(path, count: int) -> list[int]:
  values = json.loads(Path(path).read_text())
  if isinstance(values, dict):
    values = values.get("noise_seeds")
  if not isinstance(values, list) or len(values) != count:
    raise ValueError(f"sensor-seeds must contain exactly {count} environment noise_seeds")
  if any(type(v) is not int or v < 0 or v >= 2**63 for v in values):
    raise ValueError("noise seeds must be integers in [0, 2**63)")
  return values


def _matrix(quaternion):
  """Unit wxyz quaternion to rotation matrix, without importing simulation modules."""
  w, x, y, z = quaternion.unbind(-1)
  return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                      2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                      2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), dim=-1).reshape(-1, 3, 3)


def _rotation(vector):
  x, y, z = vector.unbind(-1)
  zero = torch.zeros_like(x)
  skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(-1, 3, 3)
  theta = torch.linalg.vector_norm(vector, dim=-1)
  eye = torch.eye(3, dtype=vector.dtype, device=vector.device).expand(len(vector), -1, -1)
  first = torch.sinc(theta / torch.pi)[:, None, None]
  second = (.5 * torch.sinc(theta / (2 * torch.pi)).square())[:, None, None]
  return eye + first * skew + second * (skew @ skew)


class SharedSensorStream:
  """A frame is indexed by nominal simulation step, not by number of reads or hits."""
  def __init__(self, seeds, profile="N0"):
    self.metadata = profile_metadata(profile)
    self.profile, self.seeds = profile, tuple(seeds)
    if not self.seeds or any(type(v) is not int or not 0 <= v < 2**63 for v in self.seeds):
      raise ValueError("Provide one valid sensor seed per environment")
    self._step, self._cached = None, None

  def sample(self, step: int) -> dict[str, np.ndarray]:
    if type(step) is not int or step < 0:
      raise ValueError("Sensor step must be a nonnegative integer")
    if self._step == step:
      return {key: value.copy() for key, value in self._cached.items()}
    sample = {}
    for channel, (name, width, amplitude) in enumerate(CHANNELS):
      values = np.zeros((len(self.seeds), width), dtype=np.float32)
      if self.profile != "N0":
        # Repeated seeds deliberately give actual and its own twin the same noise tape.
        unique = {seed: np.random.Generator(np.random.Philox(np.random.SeedSequence([seed, step, channel])))
                  .uniform(-amplitude, amplitude, width).astype(np.float32) for seed in set(self.seeds)}
        values[:] = np.stack([unique[seed] for seed in self.seeds])
      if name == "attitude":
        values[:, 2] = 0.  # No unmeasured absolute yaw sensor is invented.
      sample[name] = values
    self._step, self._cached = step, sample
    return {key: value.copy() for key, value in sample.items()}

  def tensors(self, step, like):
    return {key: torch.as_tensor(value, dtype=like.dtype, device=like.device)
            for key, value in self.sample(step).items()}

  def teacher(self, actor, step, command_width, *, root_quaternion=None, anchor_quaternion=None):
    if (actor.ndim != 2 or actor.shape[0] != len(self.seeds)
        or actor.shape[1] not in (294, 411) or command_width != 180):
      raise ValueError("Shared-sensor teacher contract requires verified 294/411-wide actor / 180 command fields")
    if self.profile == "N0":
      return actor
    noise = self.tensors(step, actor)
    rotation = _rotation(noise["attitude"])
    if (root_quaternion is None) != (anchor_quaternion is None):
      raise ValueError("Supply both root and anchor orientation, or neither for identical-frame CPU fixtures")
    if root_quaternion is not None:
      relative = _matrix(root_quaternion).transpose(1, 2) @ _matrix(anchor_quaternion)
      rotation = relative.transpose(1, 2) @ rotation @ relative
    inverse = rotation.transpose(1, 2)
    out = actor.clone()
    start = command_width
    out[:, start:start+3] = (inverse @ actor[:, start:start+3, None]).squeeze(-1)
    out[:, start+3:start+9] = (inverse @ actor[:, start+3:start+9].reshape(-1, 3, 2)).reshape(-1, 6)
    out[:, start+12:start+15] += noise["gyro"]
    out[:, start+15:start+39] += noise["joint_pos"]
    out[:, start+39:start+63] += noise["joint_vel"]
    # True site linear velocity, reference previews, goal/lag and prior command remain unchanged.
    return out

  def manufacturer_frame(self, frame, step, to_policy):
    if frame.shape != (len(self.seeds), 81):
      raise ValueError("Manufacturer frame must be [envs,81]")
    if self.profile == "N0":
      return frame
    noise = self.tensors(step, frame)
    out = frame.clone()
    out[:, :3] += noise["gyro"]
    out[:, 3:6] = (_rotation(noise["attitude"]).transpose(1, 2) @ frame[:, 3:6, None]).squeeze(-1)
    out[:, 9:33] += noise["joint_pos"][:, to_policy]
    out[:, 33:57] += noise["joint_vel"][:, to_policy]
    return out


class ActorInputTrace:
  """Bounded CPU chunks; committed chunks and partial manifest survive interruptions."""
  def __init__(self, directory, metadata, chunk_frames=32):
    self.directory = Path(directory).resolve()
    self.directory.mkdir(parents=True, exist_ok=False)
    self.manifest_path = self.directory / "manifest.json"
    self.manifest = dict(schema="r1wb-actor-input-trace-v1", state="recording", sampling="immediately before actor forward",
                         frames=0, chunks=[], metadata=metadata,
                         clean_meaning="same live trajectory, sensors unperturbed at actor boundary; not a noise-free counterfactual rollout")
    self.chunk_frames, self.pending = chunk_frames, []
    self._save()
    atexit.register(self.flush)

  def _save(self):
    temporary = self.directory / "manifest.tmp"
    temporary.write_text(json.dumps(self.manifest, indent=2))
    temporary.replace(self.manifest_path)

  def append(self, step, clean, actual, noise, context):
    def cpu(value):
      # sim.data exposes TorchArray proxies. np.asarray(proxy) can try converting
      # its CUDA tensor directly; read the existing tensor view explicitly first.
      tensor = value if isinstance(value, torch.Tensor) else getattr(value, "_tensor", None)
      result = tensor.detach().cpu().numpy().copy() if isinstance(tensor, torch.Tensor) else np.asarray(value).copy()
      if result.dtype.hasobject:
        raise TypeError("Actor trace contains a non-numeric object instead of a tensor/array")
      return result
    record = dict(step=np.int64(step), actor_clean=cpu(clean), actor_input=cpu(actual),
                  **{"noise_"+key: cpu(value) for key, value in noise.items()},
                  **{key: cpu(value) for key, value in context.items()})
    self.pending.append(record)
    if len(self.pending) >= self.chunk_frames:
      self.flush()

  def flush(self):
    if not self.pending:
      return
    filename = f"inputs_{len(self.manifest['chunks']):05d}.npz"
    target = self.directory / filename
    with target.open("xb") as stream:
      np.savez_compressed(stream, **{key: np.stack([row[key] for row in self.pending]) for key in self.pending[0]})
    self.manifest["chunks"].append(dict(path=filename, frames=len(self.pending),
                                         first_step=int(self.pending[0]["step"]), last_step=int(self.pending[-1]["step"]),
                                         sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size))
    self.manifest["frames"] += len(self.pending)
    self.pending.clear()
    self._save()

  def close(self, expected_steps):
    self.flush()
    if self.manifest["frames"] != expected_steps:
      raise ValueError("Actor input trace is incomplete")
    self.manifest.update(state="complete", expected_steps=expected_steps)
    self._save()
    atexit.unregister(self.flush)
    return dict(path=str(self.manifest_path), sha256=hashlib.sha256(self.manifest_path.read_bytes()).hexdigest(),
                frames=self.manifest["frames"], schema=self.manifest["schema"])


class ActorSensorHook:
  """Opt-in model pre-forward hook. Outside begin()/finish(), even auxiliary labels are untouched."""
  def __init__(self, model, stream, trace, *, kind, to_policy=None, command_width=None,
               orientation=None):
    if kind not in ("teacher", "manufacturer"):
      raise ValueError("Unsupported actor kind")
    self.stream, self.trace, self.kind = stream, trace, kind
    self.to_policy, self.command_width, self.orientation = to_policy, command_width, orientation
    self.step, self.context, self.calls = None, None, 0
    self.history = self.previous_episode = self.previous_step = None
    self.outcomes = []
    self.handle = model.register_forward_pre_hook(self._before)

  def begin(self, step, **context):
    if self.step is not None or (self.previous_step is not None and step != self.previous_step + 1):
      raise ValueError("Actor trace steps must be consecutive, exactly once per control step")
    self.step, self.context, self.calls = int(step), context, 0

  def _before(self, model, inputs):
    if self.step is None:
      return None
    self.calls += 1
    if self.calls != 1:
      raise RuntimeError("Expected exactly one actor forward in this control frame")
    observation = inputs[0]
    clean = observation["actor"]
    if self.kind == "teacher":
      poses = self.orientation() if self.orientation is not None else {}
      actual = self.stream.teacher(clean, self.step, self.command_width(), **poses)
    else:
      if clean.shape != (len(self.stream.seeds), 810):
        raise ValueError("Manufacturer actor must be [envs,810]")
      episode = self.context["episode_length_before"]
      restarted = (episode < self.previous_episode) if self.previous_episode is not None else None
      if self.stream.profile == "N0":
        actual = clean
      else:
        frame = self.stream.manufacturer_frame(clean.reshape(-1, 10, 81)[:, -1], self.step, self.to_policy)
        self.history = (frame[:, None].repeat(1, 10, 1) if self.history is None else
                        torch.cat((self.history[:, 1:], frame[:, None]), dim=1))
        if restarted is not None:
          self.history[restarted] = frame[restarted, None].expand(-1, 10, -1)
        actual = self.history.reshape(-1, 810)
      self.previous_episode = episode.clone()
    self.trace.append(self.step, clean, actual, self.stream.sample(self.step), self.context)
    if actual is clean:
      return None  # Exact N0: no replacement TensorDict, tensor or buffer write.
    changed = observation.clone()
    changed["actor"] = actual
    return (changed, *inputs[1:])

  def finish(self):
    if self.step is None or self.calls != 1:
      raise RuntimeError("Actor input was not observed exactly once before physics")
    self.previous_step, self.step, self.context = self.step, None, None

  def after_step(self, dones, terminated, alive_before):
    self.outcomes.append(dict(step=self.previous_step,
                             done=dones.detach().cpu().numpy().copy().astype(bool),
                             physical_terminated=terminated.detach().cpu().numpy().copy().astype(bool),
                             first_physical_fall=(alive_before & dones.bool() & terminated).detach().cpu().numpy().copy(),
                             alive_after=(alive_before & ~dones.bool()).detach().cpu().numpy().copy()))

  def close(self, expected_steps):
    self.handle.remove()
    if len(self.outcomes) != expected_steps:
      raise ValueError("Actor trace outcome coverage is incomplete")
    outcome_path = self.trace.directory / "outcomes.npz"
    with outcome_path.open("xb") as output:
      np.savez_compressed(output, **{key: np.asarray([r[key] for r in self.outcomes]) for key in self.outcomes[0]})
    self.trace.manifest["outcomes"] = dict(path=outcome_path.name, sha256=hashlib.sha256(outcome_path.read_bytes()).hexdigest(),
                                            frames=expected_steps,
                                            unique_physical_falls=np.asarray([r["first_physical_fall"] for r in self.outcomes]).sum(axis=0).tolist(),
                                            semantics="terminated & done & alive_before; reasons ORed; exclude every subsequent auto-reset episode")
    return self.trace.close(expected_steps)


def install_sensor_hook(env, model, args, archive, *, kind, command=None, to_policy=None):
  """The archive owns its complete actor trace so normal archive transport retains it."""
  seeds = read_seeds(args.sensor_seeds, env.num_envs)
  stream = SharedSensorStream(seeds, args.observation_profile)
  data = env.scene["robot"].data
  metadata = dict(profile=stream.metadata, noise_seeds=seeds, dt_s=float(env.step_dt), actor_kind=kind,
                  sensor_seed_file_sha256=hashlib.sha256(Path(args.sensor_seeds).read_bytes()).hexdigest(),
                  joint_names=list(env.scene["robot"].joint_names), body_names=list(env.scene["robot"].body_names),
                  body_ids=[int(i) for i in env.scene["robot"].indexing.body_ids],
                  env_origins=env.scene.env_origins.detach().cpu().tolist(),
                  anchor_body_name=command.cfg.anchor_body_name if command is not None else None,
                  time_axis="pre-action frame t=nominal_step*control_dt; outcomes refer to following physical interval",
                  actor_layer="raw input before checkpoint normalizer and symmetric actor transform",
                  snapshot_is_full_state_rng_branch=False,
                  requested_directory=str(args.actor_trace))
  directory = archive.directory / "actor_inputs" if archive is not None else Path(args.actor_trace)
  trace = ActorInputTrace(directory, metadata)
  orientation = ((lambda: dict(root_quaternion=data.root_link_quat_w, anchor_quaternion=command.robot_anchor_quat_w))
                 if kind == "teacher" else None)
  return ActorSensorHook(model, stream, trace, kind=kind, to_policy=to_policy,
                         command_width=(lambda: int(command.command.shape[1])) if command is not None else None,
                         orientation=orientation)


def trace_context(env, alive, **extra):
  """Ground truth is read only. No sensor noise is injected into these metric fields."""
  data = env.scene["robot"].data
  return dict(episode_length_before=env.episode_length_buf.clone(), alive_before=alive.clone(),
              pre_qpos=env.sim.data.qpos, pre_qvel=env.sim.data.qvel,
              joint_pos=data.joint_pos, joint_vel=data.joint_vel, actuator_torque=data.qfrc_actuator,
              body_pos_w=data.body_link_pos_w, body_quat_w=data.body_link_quat_w,
              body_ang_vel_w=data.body_link_ang_vel_w, root_pos_w=data.root_link_pos_w,
              root_quat_w=data.root_link_quat_w, root_lin_vel_b=data.root_link_lin_vel_b,
              root_ang_vel_b=data.root_link_ang_vel_b, projected_gravity_b=data.projected_gravity_b,
              foot_contact=env.scene["feet_ground"].data.found.reshape(env.num_envs, -1)[:, :2] > 0,
              **extra)


def validate_profile_options(args):
  """N0 legacy defaults are untouched; proper contract cannot silently omit evidence."""
  proper = args.comparison_contract == "proper-r1-v1"
  profile = getattr(args, "observation_profile", "N0")
  if profile not in PROFILES:
    raise ValueError("Unknown observation profile")
  if not proper and (profile != "N0" or getattr(args, "sensor_seeds", None) or getattr(args, "actor_trace", None)):
    raise ValueError("Shared sensor profiles/evidence require explicit proper-r1-v1; native-r1 stays frozen")
  if proper:
    if not getattr(args, "sensor_seeds", None) or not getattr(args, "actor_trace", None):
      raise ValueError("proper-r1-v1 requires --sensor-seeds and --actor-trace for every actor input")
    if Path(args.actor_trace).exists():
      raise ValueError("Actor trace destination already exists; partial evidence is never overwritten")
    if getattr(args, "steer", "") or getattr(args, "goal_lag_grace", None) is not None or getattr(args, "arm_response_scale", None) is not None:
      raise ValueError("proper-r1-v1 uses no learned steering or force-clock/arm interventions")
    for key in ("course", "quiet_hits", "graft", "rate_feedback", "rate_feedback_hip", "reflex_gain", "plan_lean",
                "hold_after_hit", "course_set", "goal_model", "backbone", "progress_feedback", "replay"):
      if getattr(args, key, None):
        raise ValueError(f"proper-r1-v1 forbids diagnostic intervention {key}")
    if getattr(args, "quiet", "none") != "none" or getattr(args, "leg_boost", 1.) != 1.:
      raise ValueError("proper-r1-v1 forbids quiet/leg target interventions")
  return proper
