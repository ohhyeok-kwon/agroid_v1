"""Train-only human reference-state resets; no motion clock or tracking reward.

Call apply() after reset managers, before the environment's final forward/sense
and observation computation. Only reset rows are written. These kinematic
initial states are not certified balanced states or human force measurements.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from r1lab import contract as C

SCHEMA = "r1-reference-initialization-v1"
FPS = 50.0


def _quat_mul(a, b):
  w = a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdims=True)
  xyz = a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
  return np.concatenate((w, xyz), axis=-1)


def reference_states(qpos: np.ndarray, fps: float = FPS):
  """Causal native-time q/dq/root13(world) and valid-pair mask; no integration."""
  poses = np.asarray(qpos, dtype=np.float64)
  if poses.ndim != 2 or poses.shape[1] != 31 or len(poses) < 2 or fps != FPS:
    raise ValueError("reference requires (N>=2,31) native 50 Hz qpos")
  finite = np.isfinite(poses).all(-1)
  norm = np.linalg.norm(poses[:, 3:7], axis=-1)
  good = finite & (np.abs(norm-1.) <= 1.e-4)
  quat = poses[:, 3:7] / np.maximum(norm[:, None], 1.e-12)
  conjugate = quat[:-1].copy(); conjugate[:, 1:] *= -1
  delta = _quat_mul(quat[1:], conjugate)
  delta *= np.where(delta[:, :1] < 0, -1., 1.)
  magnitude = np.linalg.norm(delta[:, 1:], axis=-1, keepdims=True)
  angle = 2*np.arctan2(magnitude, np.clip(delta[:, :1], 0., 1.))
  angular = delta[:, 1:] * np.divide(angle, magnitude, out=np.full_like(angle, 2.), where=magnitude > 1.e-10)*fps
  root = np.zeros((len(poses), 13), dtype=np.float64)
  root[:, :3], root[:, 3:7] = poses[:, :3], quat
  root[1:, 7:10] = np.diff(poses[:, :3], axis=0)*fps
  root[1:, 10:13] = angular
  velocity = np.zeros((len(poses), 24), dtype=np.float64)
  velocity[1:] = np.diff(poses[:, 7:], axis=0)*fps
  valid = np.zeros(len(poses), dtype=bool)
  valid[1:] = good[1:] & good[:-1]
  valid &= np.isfinite(root).all(-1) & np.isfinite(velocity).all(-1)
  return poses[:, 7:].copy(), velocity, root, valid


def align_root_states(root: torch.Tensor, yaw: torch.Tensor, xy: torch.Tensor,
                      floor_height: torch.Tensor | None = None) -> torch.Tensor:
  """Rigid yaw/XY registration, preserving tilt and rotating WORLD velocities."""
  result = root.clone()
  q = root[:, 3:7]
  old = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
  delta = yaw-old
  co, si = delta.cos(), delta.sin()
  for start in (7, 10):
    x, y = root[:, start], root[:, start+1]
    result[:, start] = co*x-si*y
    result[:, start+1] = si*x+co*y
  c, s = (delta/2).cos(), (delta/2).sin()
  w, x, y, z = q.unbind(-1)
  result[:, 3:7] = torch.stack((c*w-s*z, c*x-s*y, c*y+s*x, c*z+s*w), -1)
  result[:, :2] = xy
  if floor_height is not None:
    result[:, 2] += floor_height
  return result


def load_state_bank(path: Path | str) -> tuple[dict, dict[str, np.ndarray]]:
  path = Path(path)
  provenance = json.loads(path.with_suffix(".json").read_text())
  digest = hashlib.sha256(path.read_bytes()).hexdigest()
  if provenance.get("artifact_sha256") != digest:
    raise ValueError("reference initialization artifact SHA256 mismatch")
  with np.load(path, allow_pickle=False) as z:
    manifest = json.loads(str(z["manifest_json"].item()))
    arrays = {key: z[key].copy() for key in ("joint_pos", "joint_vel", "root_state", "source_frame")}
  if manifest != provenance.get("manifest") or manifest.get("schema") != SCHEMA:
    raise ValueError("reference initialization schema/provenance mismatch")
  if manifest.get("source_split") != "train" or manifest.get("fps") != FPS:
    raise ValueError("reference initialization requires native 50 Hz train source")
  if manifest.get("joint_names") != list(C.ENTITY_JOINTS):
    raise ValueError("reference initialization joint map mismatch")
  n = len(arrays["source_frame"])
  if n == 0 or any(arrays[key].shape != (n, width) for key, width in
                   (("joint_pos", 24), ("joint_vel", 24), ("root_state", 13))):
    raise ValueError("empty or malformed reference initialization bank")
  if any(not np.isfinite(arrays[key]).all() for key in ("joint_pos", "joint_vel", "root_state")):
    raise ValueError("non-finite reference initialization bank")
  if not np.allclose(np.linalg.norm(arrays["root_state"][:, 3:7], axis=-1), 1., atol=1.e-4):
    raise ValueError("reference initialization quaternion is not unit length")
  if arrays["source_frame"].dtype.kind not in "iu" or (np.diff(arrays["source_frame"]) <= 0).any():
    raise ValueError("source frame indices must be unique and increasing")
  manifest = dict(manifest, artifact_sha256=digest)
  return manifest, arrays


class ReferenceStateInitializer:
  """Explicit reset helper. Caller owns RSI cohort/teacher exclusion and commands."""

  def __init__(self, env, data_file: str | Path, seed: int = 0, action_name="joint_pos"):
    self.env, self.robot, self.action_name = env, env.scene["robot"], action_name
    if tuple(self.robot.joint_names) != C.ENTITY_JOINTS:
      raise ValueError("robot joint order differs from reference initialization bank")
    self.provenance, arrays = load_state_bank(data_file)
    self.data_file = str(Path(data_file).resolve())
    self.bank = {key: torch.as_tensor(value, device=env.device,
                  dtype=torch.long if key == "source_frame" else torch.float32)
                 for key, value in arrays.items()}
    self.rng = torch.Generator(device=env.device).manual_seed(seed)
    self.last_applied_step = torch.full((env.num_envs,), -1, device=env.device, dtype=torch.long)
    self.source_frames = torch.full_like(self.last_applied_step, -1)
    self.draws = 0
    env._r1_reference_initializer = self

  def _ids(self, ids):
    values = torch.arange(self.env.num_envs, device=self.env.device)
    ids = values if ids is None else values[ids] if isinstance(ids, slice) else torch.as_tensor(ids, device=self.env.device, dtype=torch.long)
    ids = ids.flatten()
    if len(ids) and ((ids < 0).any() or (ids >= self.env.num_envs).any() or ids.unique().numel() != ids.numel()):
      raise ValueError("invalid or duplicate reference reset ids")
    return ids

  def begin_reset(self, ids):
    """Rearm once at entry to a new logical reset, even at the same policy tick.

    This changes bookkeeping only. Callers must invoke it once per actual reset,
    never between two apply() calls within that same reset.
    """
    ids = self._ids(ids)
    self.last_applied_step[ids] = -1
    self.source_frames[ids] = -1

  @torch.no_grad()
  def apply(self, ids, *, yaw=None, xy=None, at_reset=False):
    """Write once at reset, before forward/sense/obs.

    at_reset=True is for a reset hook preceding episode-counter zeroing;
    callers must never use it to overwrite an ongoing physical trajectory.
    """
    ids = self._ids(ids)
    step = int(self.env.common_step_counter)
    if not len(ids):
      return {"env_ids": ids, "source_frame": self.source_frames[ids]}
    if ((not at_reset and (self.env.episode_length_buf[ids] != 0).any())
        or (self.last_applied_step[ids] == step).any()):
      raise ValueError("reference state may be applied only once at a reset boundary")
    indices = torch.randint(len(self.bank["source_frame"]), (len(ids),), device=self.env.device, generator=self.rng)
    current = self.robot.data
    # Reset writes precede forward(): xpos/xquat can still describe the terminal
    # pose. Read the already reset free-joint qpos rather than derived geometry.
    pose = current.data.qpos[ids[:, None], self.robot.indexing.free_joint_q_adr[None, :]]
    q = pose[:, 3:7]
    if yaw is None:
      yaw = torch.atan2(2*(q[:, 0]*q[:, 3]+q[:, 1]*q[:, 2]), 1-2*(q[:, 2]**2+q[:, 3]**2))
    if xy is None:
      xy = pose[:, :2]
    yaw = torch.as_tensor(yaw, device=self.env.device, dtype=torch.float32).expand(len(ids))
    xy = torch.as_tensor(xy, device=self.env.device, dtype=torch.float32).expand(len(ids), 2)
    origin_z = self.env.scene.env_origins[ids, 2]
    if not (torch.isfinite(yaw).all() and torch.isfinite(xy).all() and torch.isfinite(origin_z).all()):
      raise ValueError("non-finite reference reset placement")
    root = align_root_states(self.bank["root_state"][indices], yaw, xy, origin_z)
    position, velocity = self.bank["joint_pos"][indices], self.bank["joint_vel"][indices]
    self.robot.reset(env_ids=ids)
    self.robot.write_joint_state_to_sim(position, velocity, env_ids=ids)
    self.robot.write_root_state_to_sim(root, env_ids=ids)
    self.initialize_histories(ids, position=position)
    self.source_frames[ids] = self.bank["source_frame"][indices]
    self.last_applied_step[ids] = step
    self.draws += len(ids)
    return {"env_ids": ids.clone(), "source_frame": self.source_frames[ids].clone(),
            "root_state_world": root.clone(), "artifact_sha256": self.provenance["artifact_sha256"]}

  @torch.no_grad()
  def initialize_histories(self, ids, *, position=None):
    """Reset selected histories; ordinary env forward/sense fills fresh observations."""
    ids = self._ids(ids)
    if not len(ids):
      return
    position = self.robot.data.joint_pos[ids] if position is None else position
    bias = getattr(self.robot.data, "encoder_bias", torch.zeros_like(self.robot.data.joint_pos))[ids]
    perm = torch.tensor(C.POLICY_TO_ENTITY, device=self.env.device)
    offset = torch.tensor(C.DEFAULT_POS, device=self.env.device)
    scale = torch.tensor(C.ACTION_SCALE, device=self.env.device)
    action = ((position+bias)[:, perm]-offset)/scale
    manager = self.env.action_manager
    for name in ("action", "prev_action", "prev_prev_action"):
      getattr(manager, name)[ids] = action
    term = manager.get_term(self.action_name)
    term._raw_actions[ids] = action
    term._processed_actions[ids] = position+bias
    self.robot.set_joint_position_target(position, env_ids=ids)
    self.robot.set_joint_velocity_target(torch.zeros_like(position), env_ids=ids)
    self.robot.set_joint_effort_target(torch.zeros_like(position), env_ids=ids)
    for sensor in self.env.scene.sensors.values():
      sensor.reset(env_ids=ids)
    self.env.observation_manager.reset(env_ids=ids)
    prior = getattr(self.env, "_r1_human_prior", None)
    if prior is not None:
      prior.reset(ids)
    else:
      from r1lab.mdp.human_prior import reset_sim_features
      reset_sim_features(self.env, ids)
    # Existing reward EMA reinitializes rows at episode_length <= 1. Its global
    # tick cache must not be invalidated here (which could advance other rows).
    from r1lab.mdp.recovery import reset_recovery
    reset_recovery(self.env, ids)

  def state_dict(self):
    return {"schema": SCHEMA, "artifact_sha256": self.provenance["artifact_sha256"],
            "rng_state": self.rng.get_state(), "draws": self.draws}

  def load_state_dict(self, state):
    if state.get("schema") != SCHEMA or state.get("artifact_sha256") != self.provenance["artifact_sha256"]:
      raise ValueError("reference initialization checkpoint provenance mismatch")
    self.rng.set_state(state["rng_state"].cpu())
    self.draws = int(state["draws"])
    self.last_applied_step.fill_(-1)
    self.source_frames.fill_(-1)


ReferenceInitializer = ReferenceStateInitializer
