"""Full-body human-motion discriminator inspired by ASE's imitation objective.

Twenty-five 50 Hz frames contain all 24 joints, root motion, feet and palms.
This is the discriminator part of adversarial imitation, not ASE latent-skill
pretraining. Reward calls only collect data; the PPO runner owns update/checkpoint
timing. Classifier probabilities are diagnostics, not proof of natural movement.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from r1lab import contract as C

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

NJ = C.NUM_DOF
FEAT = 70
WINDOW = 25
FPS = 50.0
SCHEMA = "r1-human-fullbody-v3"
SITES = ("left_foot", "right_foot", "left_palm", "right_palm")
ReduceSum = Callable[[torch.Tensor], torch.Tensor | None]


def _copy_feature_cache(value, device=None):
  if isinstance(value, torch.Tensor):
    return value.to(device=device if device is not None else value.device).clone()
  if isinstance(value, dict):
    return {key: _copy_feature_cache(item, device) for key, item in value.items()}
  return value


def mirror_features(features: np.ndarray | torch.Tensor):
  """Reflect all joint/Cartesian features through the sagittal plane."""
  if features.shape[-1] != FEAT:
    raise ValueError(f"expected {FEAT} features, got {features.shape}")
  idx, sign = list(range(FEAT)), [1.] * FEAT
  for entity_i, policy_i in enumerate(C.ACTION_IDX):
    source_i = C.POLICY_TO_ENTITY[C.MIRROR_INDEX[policy_i]]
    for offset in (0, NJ):
      idx[offset + entity_i] = offset + source_i
      sign[offset + entity_i] = C.MIRROR_SIGN[policy_i]
  for offset, signs in ((49, (1, -1, 1)), (52, (1, -1, 1)), (55, (-1, 1, -1))):
    sign[offset:offset + 3] = signs
  for offset in (58, 64):
    idx[offset:offset + 6] = [offset + i for i in (3, 4, 5, 0, 1, 2)]
    sign[offset:offset + 6] = [1, -1, 1, 1, -1, 1]
  if isinstance(features, torch.Tensor):
    return features[..., idx] * features.new_tensor(sign)
  return features[..., idx] * np.asarray(sign, dtype=features.dtype)


def _inverse_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  xyz = quaternion[..., 1:]
  cross = torch.cross(xyz, vector, dim=-1)
  return vector + 2. * (-quaternion[..., :1] * cross + torch.cross(xyz, cross, dim=-1))


def tensor_state_features(joint_pos: torch.Tensor, root_pos: torch.Tensor, root_quat: torch.Tensor,
                          site_pos: torch.Tensor, previous: dict | None = None,
                          dt: float = 1. / FPS) -> tuple[torch.Tensor, dict, torch.Tensor]:
  """Common causal pose features for simulator, reference and offline evaluation.

  Quaternions use wxyz; sites are L/R foot then L/R palm. Returns features,
  detached state for the next policy tick, and a valid-velocity-pair mask.
  First/reset pairs are invalid. Set previous['valid'][ids]=False on a reset.
  All arithmetic uses float32; physical joint/root velocities are never inputs.
  """
  if not np.isfinite(dt) or dt <= 0.:
    raise ValueError("feature dt must be positive")
  device = joint_pos.device
  state = {"joint_pos": joint_pos.detach().to(dtype=torch.float32),
           "root_pos": root_pos.detach().to(device=device, dtype=torch.float32),
           "root_quat": root_quat.detach().to(device=device, dtype=torch.float32),
           "site_pos": site_pos.detach().to(device=device, dtype=torch.float32)}
  n = len(state["joint_pos"])
  for key, shape in (("joint_pos", (n, NJ)), ("root_pos", (n, 3)),
                     ("root_quat", (n, 4)), ("site_pos", (n, len(SITES), 3))):
    if state[key].shape != shape:
      raise ValueError(f"invalid human feature state shape for {key}: {state[key].shape}")
  finite = torch.ones(n, dtype=torch.bool, device=device)
  for value in state.values():
    finite &= torch.isfinite(value).flatten(1).all(-1)
  length = state["root_quat"].norm(dim=-1, keepdim=True)
  finite &= length[:, 0] > 1.e-6
  quaternion = state["root_quat"] / length.clamp_min(1.e-8)
  q, position, sites = state["joint_pos"], state["root_pos"], state["site_pos"]
  pair_valid = torch.zeros_like(finite)
  dq = torch.zeros_like(q)
  linear = torch.zeros_like(position)
  angular = torch.zeros_like(position)
  if previous is not None:
    pair_valid = finite & previous["valid"].to(device=device, dtype=torch.bool)
    previous_quat = previous["root_quat"].to(device=device, dtype=torch.float32)
    previous_quat = previous_quat / previous_quat.norm(dim=-1, keepdim=True).clamp_min(1.e-8)
    dq = (q - previous["joint_pos"].to(device=device, dtype=torch.float32)) / dt
    world_velocity = (position - previous["root_pos"].to(device=device, dtype=torch.float32)) / dt
    linear = _inverse_rotate(quaternion, world_velocity)
    # Shortest quaternion logarithm of previous^-1 * current. The finite-step
    # rotation axis has the same coordinates in the previous/current body frames.
    a, b = previous_quat, quaternion
    real = a[:, :1] * b[:, :1] + (a[:, 1:] * b[:, 1:]).sum(-1, keepdim=True)
    imaginary = a[:, :1] * b[:, 1:] - b[:, :1] * a[:, 1:] - torch.cross(a[:, 1:], b[:, 1:], dim=-1)
    sign = torch.where(real < 0., -1., 1.)
    real, imaginary = real * sign, imaginary * sign
    magnitude = imaginary.norm(dim=-1, keepdim=True)
    angle = 2. * torch.atan2(magnitude, real.clamp_min(0.))
    scale = torch.where(magnitude > 1.e-6, angle / magnitude.clamp_min(1.e-8), 2.)
    angular = imaginary * scale / dt
    dq = torch.where(pair_valid[:, None], dq, 0.)
    linear = torch.where(pair_valid[:, None], linear, 0.)
    angular = torch.where(pair_valid[:, None], angular, 0.)
  gravity = _inverse_rotate(quaternion, quaternion.new_tensor([0., 0., -1.]).expand(n, -1))
  height = position[:, 2] - sites[:, :2, 2].amin(dim=1)
  w, x, y, z = quaternion.unbind(-1)
  yaw = torch.atan2(2. * (w * z + x * y), 1. - 2. * (y.square() + z.square()))
  c, s = yaw.cos()[:, None], yaw.sin()[:, None]
  relative = sites - position[:, None]
  local = torch.stack((c * relative[..., 0] + s * relative[..., 1],
                       -s * relative[..., 0] + c * relative[..., 1], relative[..., 2]), dim=-1)
  features = torch.cat((q, dq, height[:, None], gravity, linear, angular, local.flatten(1)), dim=-1)
  state = {key: value.clone() for key, value in state.items()}
  state["valid"] = finite.clone()
  return features, state, pair_valid


def numpy_state_features(joint_pos: np.ndarray, root_pos: np.ndarray, root_quat: np.ndarray,
                         site_pos: np.ndarray, previous: dict | None = None,
                         dt: float = 1. / FPS) -> tuple[np.ndarray, dict, np.ndarray]:
  """CPU ndarray adapter with exactly the same arithmetic and state contract."""
  values = [torch.as_tensor(value) for value in (joint_pos, root_pos, root_quat, site_pos)]
  old = {key: torch.as_tensor(value) for key, value in previous.items()} if previous is not None else None
  features, state, valid = tensor_state_features(*values, previous=old, dt=dt)
  return features.numpy(), {key: value.numpy() for key, value in state.items()}, valid.numpy()


def ref_features(qpos: np.ndarray, fps: float, model=None) -> np.ndarray:
  """CPU FK plus the common backward-difference utility; first row is invalid."""
  import mujoco
  from r1lab.robot import R1_XML

  qpos = np.asarray(qpos, dtype=np.float64)
  if qpos.ndim != 2 or qpos.shape[1] != 7 + NJ or len(qpos) < 3:
    raise ValueError("reference qpos must be (frames >= 3, 31)")
  if not np.isfinite(qpos).all() or not np.isfinite(fps) or fps <= 0:
    raise ValueError("non-finite qpos or invalid frame rate")
  model = model if model is not None else mujoco.MjModel.from_xml_path(str(R1_XML))
  addresses = [int(model.joint(name).qposadr[0]) for name in C.ENTITY_JOINTS]
  site_ids = [model.site(name).id for name in SITES]
  data = mujoco.MjData(model)
  sites = np.empty((len(qpos), len(SITES), 3), dtype=np.float32)
  for index, pose in enumerate(qpos):
    data.qpos[:] = pose
    mujoco.mj_kinematics(model, data)
    sites[index] = data.site_xpos[site_ids]
  state = dict(joint_pos=qpos[:, addresses].astype(np.float32), root_pos=qpos[:, :3].astype(np.float32),
               root_quat=qpos[:, 3:7].astype(np.float32), site_pos=sites)
  previous = {key: np.concatenate((value[:1], value[:-1]), axis=0) for key, value in state.items()}
  previous["valid"] = np.arange(len(qpos)) > 0
  features, _, _ = numpy_state_features(**state, previous=previous, dt=1. / fps)
  return features


def make_windows(features: np.ndarray) -> np.ndarray:
  """Flatten chronological frames; caller must supply one contiguous clip."""
  features = np.asarray(features, dtype=np.float32)
  if features.ndim != 2 or features.shape[1] != FEAT:
    raise ValueError(f"features must have shape (frames, {FEAT})")
  if len(features) < WINDOW:
    return np.empty((0, WINDOW * FEAT), dtype=np.float32)
  return np.stack([features[i:i + WINDOW].reshape(-1)
                   for i in range(len(features) - WINDOW + 1)])


def reset_sim_features(env: ManagerBasedRlEnv, env_ids=None) -> None:
  """Invalidate predecessor poses; the next observed frame starts a new pair."""
  cache = getattr(env, "_r1_human_feature_cache", None)
  if cache is not None:
    ids = slice(None) if env_ids is None else env_ids
    cache["state"]["valid"][ids] = False
    cache["pair_valid"][ids] = False


def sim_features(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Named pose collection, causally differenced once per common policy tick."""
  dt = float(getattr(env, "step_dt", 1. / FPS))
  if not np.isclose(dt, 1. / FPS, atol=1.e-7):
    raise ValueError("human motion features require a 50 Hz policy clock")
  step = getattr(env, "common_step_counter", None)
  cache = getattr(env, "_r1_human_feature_cache", None)
  if cache is not None and step is not None and cache["step"] == int(step):
    return cache["features"]
  if cache is not None and step is not None and cache["step"] is not None and int(step) != cache["step"] + 1:
    reset_sim_features(env)
  asset = env.scene["robot"]
  indices = getattr(asset, "_human_prior_indices", None)
  if indices is None:
    indices = (asset.find_joints(C.ENTITY_JOINTS, preserve_order=True)[0],
               asset.find_sites(SITES, preserve_order=True)[0])
    if len(indices[0]) != NJ or len(indices[1]) != len(SITES):
      raise ValueError("human prior requires all 24 joints, both foot and palm sites")
    asset._human_prior_indices = indices
  joint_ids, site_ids = indices
  d = asset.data
  lengths = getattr(env, "episode_length_buf", None)
  if cache is not None and lengths is not None and cache["episode_length"] is not None:
    reset_sim_features(env, (lengths < cache["episode_length"]).nonzero(as_tuple=False).flatten())
  previous = cache["state"] if cache is not None else None
  features, state, valid = tensor_state_features(d.joint_pos[:, joint_ids], d.root_link_pos_w,
                                                d.root_link_quat_w, d.site_pos_w[:, site_ids], previous, dt)
  env._r1_human_feature_cache = {"step": int(step) if step is not None else None,
                               "features": features, "state": state, "pair_valid": valid,
                               "episode_length": lengths.clone() if lengths is not None else None}
  env._r1_human_feature_valid = valid
  return features


class Discriminator(nn.Module):
  def __init__(self, dims=(512, 256)):
    super().__init__()
    layers, width = [], WINDOW * FEAT
    for hidden in dims:
      layers.extend((nn.Linear(width, hidden), nn.ReLU()))
      width = hidden
    layers.append(nn.Linear(width, 1))
    self.net = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x).squeeze(-1)


def _load_reference(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
  with np.load(path, allow_pickle=False) as data:
    manifest = json.loads(str(data["manifest_json"].item()))
    arrays = {key: data[key].copy() for key in
              ("train_windows", "validation_windows", "train_weights",
               "validation_weights", "train_clip_ids", "validation_clip_ids")}
  if (manifest.get("schema") != SCHEMA or manifest.get("frame_features") != FEAT
      or manifest.get("window_frames") != WINDOW or manifest.get("fps") != FPS
      or manifest.get("joint_names") != list(C.ENTITY_JOINTS)
      or manifest.get("site_names") != list(SITES)):
    raise ValueError("human reference feature schema mismatch; rebuild the full-body dataset")
  train_ids, validation_ids = set(arrays["train_clip_ids"]), set(arrays["validation_clip_ids"])
  if train_ids & validation_ids:
    raise ValueError("human reference clips leak between train and validation")
  records = manifest.get("clips", [])
  train_hashes = {r["retarget_sha256"] for r in records if r["split"] == "train"}
  validation_hashes = {r["retarget_sha256"] for r in records if r["split"] == "validation"}
  if not records or not train_hashes or not validation_hashes or train_hashes & validation_hashes:
    raise ValueError("human reference requires disjoint source provenance for held-out clips")
  for split in ("train", "validation"):
    x, w, ids = (arrays[f"{split}_{suffix}"] for suffix in ("windows", "weights", "clip_ids"))
    if (x.ndim != 2 or x.shape[1] != FEAT * WINDOW or len(x) == 0
        or not np.isfinite(x).all() or w.shape != (len(x),) or ids.shape != (len(x),)
        or not np.isfinite(w).all() or np.any(w <= 0)):
      raise ValueError(f"invalid or empty human reference {split} arrays")
  return manifest, arrays


class human_motion_prior:
  """Collect on reward calls; update synchronously once after each PPO rollout.

  ``reduce_sum`` performs a SUM (never a mean). Every rank must call ``update``
  with the same hyperparameters, even when that rank collected zero valid windows.
  Synchronize initial ``disc`` parameters before the first rollout. A fixed
  schedule reduces counts, one packed gradient per optimizer step, then metrics.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self._env = env
    step_dt = getattr(env, "step_dt", 1. / FPS)
    if not np.isclose(float(step_dt), 1. / FPS, atol=1.e-7):
      raise ValueError(f"human prior expects {WINDOW} consecutive 50 Hz policy frames")
    self.data_file = Path(p["data_file"])
    self.manifest, arrays = _load_reference(self.data_file)
    self.data_sha256 = hashlib.sha256(self.data_file.read_bytes()).hexdigest()
    self.reference_manifest = self.manifest
    self.provenance = {"data_file": str(self.data_file), "data_sha256": self.data_sha256,
                       "manifest_file": str(self.data_file.with_suffix(".manifest.json")),
                       "schema": SCHEMA}
    self.device = torch.device(env.device)
    self.batch = int(p.get("batch", 2048))
    self.n_updates = int(p.get("num_updates", 4))
    self.gp = float(p.get("grad_penalty", 5.))
    self.max_grad_norm = float(p.get("max_grad_norm", 1.))
    self.eval_fraction = float(p.get("validation_fraction", .2))
    self.eval_batch = int(p.get("validation_batch", 2048))
    self.stable_locomotion_only = bool(p.get("stable_locomotion_only", False))
    self.command_name = str(p.get("command_name", "twist"))
    capacity = int(p.get("buffer", 65536))
    if (min(self.batch, self.n_updates, self.eval_batch, capacity) < 1
        or not 0. < self.eval_fraction < 1. or self.gp < 0):
      raise ValueError("invalid human discriminator batch/update/buffer/validation configuration")
    with torch.inference_mode(False):
      self.ref = torch.as_tensor(arrays["train_windows"], dtype=torch.float32, device=self.device)
      self.validation_ref = torch.as_tensor(arrays["validation_windows"], dtype=torch.float32, device=self.device)
      self.ref_weights = torch.as_tensor(arrays["train_weights"], dtype=torch.float32, device=self.device)
      self.validation_weights = torch.as_tensor(arrays["validation_weights"], dtype=torch.float32, device=self.device)
      weights = self.ref_weights / self.ref_weights.sum()
      self.mu = (self.ref * weights[:, None]).sum(dim=0)
      self.sd = ((self.ref - self.mu).square() * weights[:, None]).sum(dim=0).sqrt().clamp_min(.05)
      self.disc = Discriminator(tuple(p.get("hidden_dims", (512, 256)))).to(self.device)
      self.opt = torch.optim.Adam(self.disc.parameters(), lr=float(p.get("lr", 1.e-4)),
                                  weight_decay=float(p.get("weight_decay", 1.e-4)))
      self.buf = torch.zeros((capacity, FEAT * WINDOW), device=self.device)
      self.history = torch.zeros((env.num_envs, WINDOW, FEAT), device=self.device)
      self.history_length = torch.zeros(env.num_envs, dtype=torch.long, device=self.device)
      self._velocity_ready = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
      self._last_reward = torch.zeros(env.num_envs, device=self.device)
      self.careful_quality = torch.zeros(env.num_envs, dtype=torch.bool, device=self.device)
      self.careful_quality_valid = torch.zeros_like(self.careful_quality)
    self.rng = torch.Generator(device=self.device)
    self.rng.manual_seed(int(p.get("seed", 0)) + int(p.get("rank", 0)))
    self.buf_n = self.buf_i = self.steps = self.updates = self.optimizer_steps = 0
    self.collected = self.invalid_frames = 0
    self.stats: dict[str, float] = {}
    self._last_step: int | None = None
    self._last_episode_length: torch.Tensor | None = None
    env._r1_human_prior = self
    # Learned proxy only: consumers must check validity, and still assess safety
    # and motion quality independently. Standing/recovery never count as failure.
    env._r1_careful_quality = self.careful_quality
    env._r1_careful_quality_valid = self.careful_quality_valid

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.history_length[ids] = 0
    self._velocity_ready[ids] = False
    self.history[ids] = 0
    self._last_reward[ids] = 0
    self.careful_quality[ids] = False
    self.careful_quality_valid[ids] = False
    if self._last_episode_length is not None:
      self._last_episode_length[ids] = -1
    reset_sim_features(self._env, env_ids)

  def _norm(self, x: torch.Tensor) -> torch.Tensor:
    return ((x - self.mu) / self.sd).clamp(-20., 20.)

  def _push(self, x: torch.Tensor) -> None:
    if not len(x):
      return
    self.collected += len(x)
    x = x[-len(self.buf):]
    indices = (torch.arange(len(x), device=self.device) + self.buf_i) % len(self.buf)
    self.buf[indices] = x.detach()
    self.buf_i = (self.buf_i + len(x)) % len(self.buf)
    self.buf_n = min(len(self.buf), self.buf_n + len(x))

  def _eligible_context(self, env: ManagerBasedRlEnv) -> torch.Tensor:
    """Keep ordinary gait imitation separate from stillness and recovery."""
    if not self.stable_locomotion_only:
      return torch.ones(env.num_envs, dtype=torch.bool, device=self.device)
    from r1lab.mdp.recovery import recovery_level
    command = env.command_manager.get_command(self.command_name)
    eligible = torch.linalg.vector_norm(command[:, :3], dim=-1) > .025
    eligible &= recovery_level(env, self.command_name) < .01
    adversary = getattr(env, "_r1_adversary", None)
    if adversary is not None and hasattr(adversary, "recovering"):
      eligible &= ~adversary.recovering
    return eligible

  @torch.no_grad()
  def score_windows(self, windows: torch.Tensor) -> torch.Tensor:
    """Unmasked classifier diagnostics for separate pre/during/post-recovery audits.

    A low diagnostic during emergency recovery is not a style failure or a
    training penalty; the online reward/negative buffer use eligible contexts.
    """
    windows = windows.to(device=self.device, dtype=torch.float32)
    if windows.ndim == 3 and windows.shape[1:] == (WINDOW, FEAT):
      windows = windows.flatten(1)
    if windows.ndim != 2 or windows.shape[1] != WINDOW * FEAT or not torch.isfinite(windows).all():
      raise ValueError(f"diagnostic windows must be finite (N, {WINDOW * FEAT}) tensors")
    return self.disc(self._norm(windows)).sigmoid()

  @torch.no_grad()
  def __call__(self, env: ManagerBasedRlEnv, data_file: str, **kwargs) -> torch.Tensor:
    del data_file, kwargs
    step = getattr(env, "common_step_counter", None)
    if step is not None and self._last_step == int(step):
      return self._last_reward
    lengths = getattr(env, "episode_length_buf", None)
    if lengths is not None:
      if self._last_episode_length is not None:
        self.reset((lengths < self._last_episode_length).nonzero(as_tuple=False).flatten())
      self._last_episode_length = lengths.clone()
    features = sim_features(env).detach()
    finite = torch.isfinite(features).all(dim=-1)
    eligible = self._eligible_context(env)
    pair_valid = getattr(env, "_r1_human_feature_valid", self._velocity_ready).clone()
    pair_valid &= self._velocity_ready
    self._velocity_ready.copy_(finite & eligible)
    self.invalid_frames += int((~finite).sum())
    self.history = torch.roll(self.history, shifts=-1, dims=1)
    self.history[:, -1] = torch.nan_to_num(features)
    self.history_length = torch.where(finite & eligible & pair_valid,
                                      (self.history_length + 1).clamp_max(WINDOW), 0)
    ready = self.history_length == WINDOW
    windows = self.history[ready].reshape(-1, FEAT * WINDOW)
    reward = torch.zeros_like(self._last_reward)
    logits = reward[:0]
    if len(windows):
      logits = self.disc(self._norm(windows))
      reward[ready] = logits.sigmoid()
      self._push(windows)
    self.steps += 1
    self._last_step = int(step) if step is not None else None
    self._last_reward = reward
    quality_ready = ready & self.stable_locomotion_only
    self.careful_quality.copy_(quality_ready & (reward >= .35))
    calibrated = (self.updates >= 32 and self.stats.get("HumanPrior/validation_valid", 0.) > 0.
                  and self.stats.get("HumanPrior/reference_accuracy", 0.) >= .6)
    self.careful_quality_valid.copy_(quality_ready & calibrated)
    log = getattr(env, "extras", {}).get("log")
    if log is not None:
      log["HumanPrior/humanlikeness"] = reward.sum() / ready.sum().clamp_min(1)
      log["HumanPrior/live_probability"] = reward.sum() / ready.sum().clamp_min(1)
      log["HumanPrior/live_logit"] = logits.sum() / ready.sum().clamp_min(1)
      log["HumanPrior/valid_window_fraction"] = ready.float().mean()
      log["HumanPrior/eligible_context_fraction"] = eligible.float().mean()
      log["HumanPrior/careful_proxy_fraction"] = self.careful_quality.float().mean()
      log["HumanPrior/careful_proxy_valid_fraction"] = self.careful_quality_valid.float().mean()
    return reward

  @staticmethod
  def _sum(value: torch.Tensor, reduce_sum: ReduceSum | None) -> torch.Tensor:
    if reduce_sum is None:
      return value
    result = reduce_sum(value)
    return value if result is None else result

  def _indices(self, count: int, n: int) -> torch.Tensor:
    return torch.randint(count, (n,), device=self.device, generator=self.rng)

  def _reference_batch(self, validation: bool, n: int) -> torch.Tensor:
    weights = self.validation_weights if validation else self.ref_weights
    ref = self.validation_ref if validation else self.ref
    indices = torch.multinomial(weights, n, replacement=True, generator=self.rng)
    return ref[indices]

  @torch.no_grad()
  def _metric_sums(self, train_ids: torch.Tensor, validation_ids: torch.Tensor) -> torch.Tensor:
    # Counts/sums only: the final SUM gives sample-weighted metrics across ranks.
    n = min(self.eval_batch, len(validation_ids))
    values = torch.zeros(15 + 10 * 3, device=self.device)
    if n:
      pi = self.buf[validation_ids[:n]]
      ref = self._reference_batch(True, n)
      pi_logit, ref_logit = self.disc(self._norm(pi)), self.disc(self._norm(ref))
      pi_prob, ref_prob = pi_logit.sigmoid(), ref_logit.sigmoid()
      values[:10] = torch.stack((ref_prob.new_tensor(n), (ref_prob >= .5).sum(),
                                (pi_prob < .5).sum(), ref_prob.sum(), pi_prob.sum(),
                                ref_logit.sum(), pi_logit.sum(),
                                (1. - ref_prob).square().sum() + pi_prob.square().sum(),
                                F.softplus(-ref_logit).sum() + F.softplus(pi_logit).sum(),
                                (ref_prob - pi_prob).sum()))
      probability = torch.cat([ref_prob, pi_prob])
      target = torch.cat([torch.ones_like(ref_prob), torch.zeros_like(pi_prob)])
      bins = (probability * 10).long().clamp_max(9)
      for i in range(10):
        mask = bins == i
        values[15 + i * 3:18 + i * 3] = torch.stack(
          (mask.sum(), probability[mask].sum(), target[mask].sum()))
    train_n = min(self.eval_batch, len(train_ids))
    if train_n:
      ref_prob = self.disc(self._norm(self._reference_batch(False, train_n))).sigmoid()
      values[10:13] = torch.stack((ref_prob.new_tensor(train_n),
                                   (ref_prob >= .5).sum(), ref_prob.sum()))
    values[13] = self.buf_n
    values[14] = self.invalid_frames
    return values

  def update(self, reduce_sum: ReduceSum | None = None) -> dict[str, float]:
    """Run fixed-count synchronous discriminator steps outside the reward callback.

    Each local loss is a SUM divided by the global number of sampled windows;
    packed gradients are then SUM-reduced. Empty ranks contribute zeros but take
    the identical collective/optimizer schedule. Validation windows are excluded
    from this update and discarded afterwards, so they never enter later fits.
    """
    with torch.inference_mode(False), torch.enable_grad():
      order = torch.randperm(self.buf_n, device=self.device, generator=self.rng)
      validation_n = min(self.eval_batch, max(1, int(self.buf_n * self.eval_fraction))) if self.buf_n > 1 else 0
      validation_ids, train_ids = order[:validation_n], order[validation_n:]
      sample_n = min(self.batch, len(train_ids))
      counts = self._sum(torch.tensor([sample_n, len(train_ids), validation_n],
                                     dtype=torch.float64, device=self.device), reduce_sum)
      global_n = float(counts[0])
      losses = torch.zeros(3, device=self.device)
      params = tuple(self.disc.parameters())
      for _ in range(self.n_updates):
        self.opt.zero_grad(set_to_none=True)
        if sample_n:
          x_pi = self._norm(self.buf[train_ids[self._indices(len(train_ids), sample_n)]].clone())
          x_ref = self._norm(self._reference_batch(False, sample_n).clone()).requires_grad_(True)
          pi_logits, ref_logits = self.disc(x_pi), self.disc(x_ref)
          classifier = .5 * (F.softplus(pi_logits).sum() + F.softplus(-ref_logits).sum())
          grad = torch.autograd.grad(ref_logits.sum(), x_ref, create_graph=True)[0]
          penalty = grad.square().sum()
          loss = (classifier + self.gp * penalty) / max(global_n, 1.)
          loss.backward()
          losses += torch.stack([classifier.detach(), penalty.detach(), loss.detach() * global_n])
        packed = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                            for p in params])
        packed = self._sum(packed, reduce_sum)
        offset = 0
        for p in params:
          p.grad = packed[offset:offset + p.numel()].view_as(p).clone()
          offset += p.numel()
        if global_n:
          nn.utils.clip_grad_norm_(params, self.max_grad_norm)
          self.opt.step()
          self.optimizer_steps += 1
      metric_values = self._metric_sums(train_ids, validation_ids)
      reduced = self._sum(torch.cat([losses, metric_values]), reduce_sum)
    self.updates += 1
    loss_values, m = reduced[:3], reduced[3:]
    val_n, train_n = float(m[0]), float(m[10])
    def ratio(a, n):
      return float(a) / n if n else 0.
    ece = sum(abs(float(m[16 + i * 3] - m[17 + i * 3])) for i in range(10))
    stats = {
      "loss": ratio(loss_values[2], global_n * self.n_updates),
      "gradient_penalty": ratio(loss_values[1], global_n * self.n_updates),
      "train_reference_accuracy": ratio(m[11], train_n),
      "train_reference_probability": ratio(m[12], train_n),
      "reference_accuracy": ratio(m[1], val_n),
      "policy_accuracy": ratio(m[2], val_n),
      "balanced_accuracy": ratio(m[1] + m[2], 2 * val_n),
      "reference_probability": ratio(m[3], val_n),
      "policy_probability": ratio(m[4], val_n),
      "humanlikeness": ratio(m[4], val_n),
      "reference_logit": ratio(m[5], val_n),
      "policy_logit": ratio(m[6], val_n),
      "brier_score": ratio(m[7], 2 * val_n),
      "cross_entropy": ratio(m[8], 2 * val_n),
      "probability_gap": ratio(m[9], val_n),
      "expected_calibration_error": ratio(ece, 2 * val_n),
      "validation_valid": float(val_n > 0),
      "validation_pairs": val_n,
      "policy_train_windows": float(counts[1]),
      "policy_buffer_windows": float(m[13]),
      "invalid_frames": float(m[14]),
      "reference_train_windows": float(len(self.ref)),
      "reference_validation_windows": float(len(self.validation_ref)),
      "reference_valid": 1.,
      "reference_train_clips": float(sum(c["split"] == "train" for c in self.manifest["clips"])),
      "reference_validation_clips": float(sum(c["split"] == "validation" for c in self.manifest["clips"])),
      "updates": float(self.updates),
      "optimizer_steps": float(self.optimizer_steps),
    }
    self.stats = {f"HumanPrior/{key}": value for key, value in stats.items()}
    # Keep episode history, but no validation policy window becomes later training data.
    self.buf_n = self.buf_i = self.invalid_frames = 0
    return dict(self.stats)

  def state_dict(self, include_runtime: bool = False) -> dict:
    """Shared checkpoint by default; optional runtime is for same-env exact tests."""
    state = {"schema": SCHEMA, "data_sha256": self.data_sha256,
             "disc": self.disc.state_dict(), "optimizer": self.opt.state_dict(),
             "mu": self.mu.clone(), "sd": self.sd.clone(), "stats": dict(self.stats),
             "steps": self.steps, "updates": self.updates, "optimizer_steps": self.optimizer_steps,
             "collected": self.collected, "rng_state": self.rng.get_state()}
    if include_runtime:
      state["runtime"] = {"buf": self.buf.clone(), "buf_n": self.buf_n, "buf_i": self.buf_i,
                          "history": self.history.clone(), "history_length": self.history_length.clone(),
                          "velocity_ready": self._velocity_ready.clone(),
                          "feature_cache": _copy_feature_cache(getattr(self._env, "_r1_human_feature_cache", None)),
                          "last_reward": self._last_reward.clone(), "last_step": self._last_step,
                          "careful_quality": self.careful_quality.clone(),
                          "careful_quality_valid": self.careful_quality_valid.clone(),
                          "last_episode_length": (self._last_episode_length.clone()
                                                  if self._last_episode_length is not None else None),
                          "invalid_frames": self.invalid_frames}
    return state

  def load_state_dict(self, state: dict, restore_runtime: bool = False) -> None:
    if state.get("schema") != SCHEMA or state.get("data_sha256") != self.data_sha256:
      raise ValueError("human discriminator checkpoint uses a different reference dataset/schema")
    self.disc.load_state_dict(state["disc"])
    self.opt.load_state_dict(state["optimizer"])
    for name in ("mu", "sd"):
      value = state[name].to(self.device)
      if value.shape != (FEAT * WINDOW,) or not torch.isfinite(value).all():
        raise ValueError(f"invalid human discriminator normalizer {name}")
      setattr(self, name, value.clone())
    if bool((self.sd <= 0).any()):
      raise ValueError("human discriminator normalizer standard deviation must be positive")
    self.stats = dict(state["stats"])
    for name in ("steps", "updates", "optimizer_steps", "collected"):
      setattr(self, name, int(state[name]))
    self.rng.set_state(state["rng_state"].cpu())
    self.reset()
    self.buf_n = self.buf_i = self.invalid_frames = 0
    self._last_step = None
    self._last_episode_length = None
    if restore_runtime:
      runtime = state["runtime"]
      for name in ("buf", "history", "history_length"):
        destination = getattr(self, name)
        if runtime[name].shape != destination.shape:
          raise ValueError("cannot restore human prior runtime to a different environment shape")
        destination.copy_(runtime[name].to(self.device))
      self.buf_n, self.buf_i = int(runtime["buf_n"]), int(runtime["buf_i"])
      self._last_reward.copy_(runtime["last_reward"].to(self.device))
      self._velocity_ready.copy_(runtime["velocity_ready"].to(self.device))
      feature_cache = _copy_feature_cache(runtime["feature_cache"], self.device)
      if feature_cache is not None:
        self._env._r1_human_feature_cache = feature_cache
        self._env._r1_human_feature_valid = feature_cache["pair_valid"]
      self.careful_quality.copy_(runtime["careful_quality"].to(self.device))
      self.careful_quality_valid.copy_(runtime["careful_quality_valid"].to(self.device))
      self._last_step = runtime["last_step"]
      lengths = runtime["last_episode_length"]
      self._last_episode_length = lengths.to(self.device).clone() if lengths is not None else None
      self.invalid_frames = int(runtime["invalid_frames"])


HumanMotionPrior = human_motion_prior
