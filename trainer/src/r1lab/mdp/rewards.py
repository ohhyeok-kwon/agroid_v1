"""Reward terms for careful (quiet, stable) walking and protective crouching.

Quiet-walking terms follow QuietWalk (arXiv 2604.23702: penalise per-foot vertical
force), TactileStep (arXiv 2609.28959: penalise fast downward motion before contact,
touchdown force and load spikes) and aibo quiet walking (arXiv 2502.10983: foot
contact velocity tracks footstep loudness).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from r1lab import contract as C
from r1lab.mdp.observations import GRAVITY, ROBOT_MASS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

FEET = ("left_foot", "right_foot")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
MG = ROBOT_MASS * GRAVITY


def _robot(env) -> Entity:
  return env.scene["robot"]


def _cached(asset: Entity, key: str, fn):
  cache = asset.__dict__.setdefault("_r1lab_cache", {})
  if key not in cache:
    cache[key] = fn()
  return cache[key]


def _foot_sites(asset: Entity) -> list[int]:
  return _cached(asset, "foot_sites", lambda: asset.find_sites(FEET, preserve_order=True)[0])


def _foot_bodies(asset: Entity) -> list[int]:
  return _cached(asset, "foot_bodies",
                 lambda: asset.find_bodies(FOOT_BODIES, preserve_order=True)[0])


def rel_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Pelvis height above the lower foot sole."""
  a = _robot(env)
  foot_z = a.data.site_pos_w[:, _foot_sites(a), 2]
  return a.data.root_link_pos_w[:, 2] - foot_z.min(dim=1).values


def _cmd_norm(env, command_name: str) -> torch.Tensor:
  c = env.command_manager.get_command(command_name)
  return torch.norm(c[:, :2], dim=1) + c[:, 2].abs()


# --- tracking ----------------------------------------------------------------

def _vel_ema(env: ManagerBasedRlEnv, tau: float = 0.4) -> torch.Tensor:
  """Per-env EMA of body-frame [vx, vy, wz] (gait sway averages out). Updated once per step."""
  a = _robot(env)
  st = a.__dict__.setdefault("_r1lab_ema", {"step": -1, "v": None})
  if st["step"] != env.common_step_counter:
    v = torch.cat([a.data.root_link_lin_vel_b[:, :2], a.data.root_link_ang_vel_b[:, 2:3]], dim=-1)
    if st["v"] is None:
      st["v"] = v.clone()
    alpha = min(1.0, env.step_dt / tau)
    fresh = (env.episode_length_buf <= 1).unsqueeze(-1)
    st["v"] = torch.where(fresh, v, st["v"] + alpha * (v - st["v"]))
    st["step"] = env.common_step_counter
  return st["v"]


def track_lin_vel_ema(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  c = env.command_manager.get_command(command_name)
  e = torch.sum(torch.square(c[:, :2] - _vel_ema(env)[:, :2]), dim=1)
  return torch.exp(-e / std**2)


def track_yaw_ema(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  c = env.command_manager.get_command(command_name)
  return torch.exp(-torch.square(c[:, 2] - _vel_ema(env)[:, 2]) / std**2)


def lin_vel_err_ema_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  c = env.command_manager.get_command(command_name)
  return torch.norm(c[:, :2] - _vel_ema(env)[:, :2], dim=1)


def yaw_err_ema_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  c = env.command_manager.get_command(command_name)
  return (c[:, 2] - _vel_ema(env)[:, 2]).abs()


def lin_vel_error_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """|v_cmd - v|_xy (linear kernel: constant gradient away from standing still)."""
  c = env.command_manager.get_command(command_name)
  v = _robot(env).data.root_link_lin_vel_b[:, :2]
  return torch.norm(c[:, :2] - v, dim=1)


def yaw_rate_error_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  c = env.command_manager.get_command(command_name)
  return (c[:, 2] - _robot(env).data.root_link_ang_vel_b[:, 2]).abs()


# --- stability ---------------------------------------------------------------

def base_height_l2(env: ManagerBasedRlEnv, target: float) -> torch.Tensor:
  return torch.square(rel_height(env) - target)


def lin_vel_z_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.square(_robot(env).data.root_link_lin_vel_b[:, 2])


def ang_vel_xy_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
  return torch.sum(torch.square(_robot(env).data.root_link_ang_vel_b[:, :2]), dim=1)


def feet_distance(env: ManagerBasedRlEnv, min_dist: float, max_dist: float) -> torch.Tensor:
  """Lateral foot spacing in the yaw frame; also punishes crossed feet."""
  a = _robot(env)
  rel = a.data.site_pos_w[:, _foot_sites(a)] - a.data.root_link_pos_w[:, None]
  qy = yaw_quat(a.data.root_link_quat_w)[:, None].expand(-1, 2, -1)
  rel_b = quat_apply_inverse(qy, rel)
  dy = rel_b[:, 0, 1] - rel_b[:, 1, 1]
  return (min_dist - dy).clamp_min(0.0) + (dy - max_dist).clamp_min(0.0)


def foot_flat(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Foot tilt while in contact (sum over feet of |g_xy|^2 in foot frame)."""
  a = _robot(env)
  s: ContactSensor = env.scene[sensor_name]
  q = a.data.body_link_quat_w[:, _foot_bodies(a)]
  g = torch.tensor([0.0, 0.0, -1.0], device=env.device).expand(q.shape[0], 2, 3)
  gb = quat_apply_inverse(q, g)
  contact = (s.data.found > 0).float()
  return torch.sum(torch.sum(torch.square(gb[..., :2]), dim=-1) * contact, dim=1)


def stand_still(env: ManagerBasedRlEnv, command_name: str, threshold: float = 0.05) -> torch.Tensor:
  """With a zero command: keep joints near default and feet still."""
  a = _robot(env)
  still = (_cmd_norm(env, command_name) < threshold).float()
  dev = torch.sum(torch.square(a.data.joint_pos - a.data.default_joint_pos), dim=1)
  foot_v = torch.sum(torch.square(a.data.site_lin_vel_w[:, _foot_sites(a), :2]), dim=(1, 2))
  return still * (dev + foot_v)


# --- quiet / soft contact ----------------------------------------------------

class soft_touchdown:
  """Downward foot speed near the ground and at the moment of touchdown."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.prev_vz = torch.zeros(env.num_envs, 2, device=env.device)
    self.step_dt = env.step_dt

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self.prev_vz[env_ids] = 0.0

  def __call__(self, env: ManagerBasedRlEnv, sensor_name: str, height_sensor_name: str,
               near_height: float = 0.04) -> torch.Tensor:
    a = _robot(env)
    s: ContactSensor = env.scene[sensor_name]
    h = env.scene[height_sensor_name].data.heights  # [B, 2]
    vz = a.data.site_lin_vel_w[:, _foot_sites(a), 2]
    down = (-vz).clamp_min(0.0)
    near = (h < near_height).float() * (s.data.found == 0).float()
    approach = torch.sum(near * torch.square(down), dim=1)
    first = s.compute_first_contact(dt=self.step_dt).float()
    touch = torch.sum(first * torch.square((-self.prev_vz).clamp_min(0.0)), dim=1)
    env.extras["log"]["Metrics/touchdown_speed"] = (
      torch.sum(first * (-self.prev_vz).clamp_min(0.0)) / first.sum().clamp_min(1.0))
    self.prev_vz = vz.clone()
    return approach + touch


def impact_peak(env: ManagerBasedRlEnv, sensor_name: str, threshold: float = 1.1) -> torch.Tensor:
  """Per-foot peak ground force within the policy step above `threshold` x body weight."""
  s: ContactSensor = env.scene[sensor_name]
  fh = s.data.force_history  # [B, 2, H, 3]
  peak = torch.norm(fh, dim=-1).amax(dim=-1) / MG  # [B, 2]
  env.extras["log"]["Metrics/foot_force_peak_bw"] = peak.amax(dim=1).mean()
  return torch.sum(torch.square((peak - threshold).clamp_min(0.0)), dim=1)


def feet_stumble(env: ManagerBasedRlEnv, sensor_name: str, ratio: float = 2.0,
                 min_force: float = 0.1) -> torch.Tensor:
  """Feet pushing into vertical faces (debris edges, step risers): horizontal ground force
  above `ratio` x vertical force and `min_force` x body weight, counted per foot.
  Teaches the blind policy to lift its feet over obstacles instead of kicking them."""
  s: ContactSensor = env.scene[sensor_name]
  fh = s.data.force_history / MG  # [B, 2, H, 3] world frame (netforce)
  fxy = torch.norm(fh[..., :2], dim=-1)
  hit = (fxy > ratio * fh[..., 2].abs()) & (fxy > min_force)
  return torch.sum(hit.any(dim=-1).float(), dim=1)


class load_rate:
  """Squared change of per-foot ground force between policy steps (load spikes)."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.prev = torch.zeros(env.num_envs, 2, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self.prev[env_ids] = -1.0

  def __call__(self, env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
    s: ContactSensor = env.scene[sensor_name]
    f = torch.norm(s.data.force_history, dim=-1).amax(dim=-1) / MG
    valid = (self.prev >= 0.0).float()
    cost = torch.sum(valid * torch.square(f - self.prev.clamp_min(0.0)), dim=1)
    self.prev = f.clone()
    return cost


# --- crouch ------------------------------------------------------------------

def crouch_height_error_l1(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  return (rel_height(env) - term.h_cmd).abs()


def crouch_height_tracking(env: ManagerBasedRlEnv, command_name: str, std: float) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  return torch.exp(-torch.square(rel_height(env) - term.h_cmd) / std**2)


class com_over_support:
  """Whole-body CoM ground projection vs. the mid-point of the two feet."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    a = _robot(env)
    ids = a.indexing.body_ids
    self.body_ids = torch.as_tensor(ids, device=env.device, dtype=torch.long)
    m = env.sim.model.body_mass
    self._mass_static = None if m.ndim > 1 else m[self.body_ids].clone()

  def __call__(self, env: ManagerBasedRlEnv, std: float) -> torch.Tensor:
    a = _robot(env)
    m = env.sim.model.body_mass
    mass = self._mass_static if self._mass_static is not None else m[:, self.body_ids]
    com = a.data.body_com_pos_w  # [B, nb, 3]
    w = mass if mass.ndim == 2 else mass.unsqueeze(0)
    com_xy = torch.sum(com[..., :2] * w.unsqueeze(-1), dim=1) / w.sum(dim=1, keepdim=True)
    mid = a.data.site_pos_w[:, _foot_sites(a), :2].mean(dim=1)
    d2 = torch.sum(torch.square(com_xy - mid), dim=1)
    env.extras["log"]["Metrics/com_offset"] = d2.sqrt().mean()
    return torch.exp(-d2 / std**2)


def feet_planted(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  a = _robot(env)
  s: ContactSensor = env.scene[sensor_name]
  v = torch.sum(torch.square(a.data.site_lin_vel_w[:, _foot_sites(a), :2]), dim=(1, 2))
  air = torch.sum((s.data.found == 0).float(), dim=1)
  return v + air


class crouch_posture:
  """Joint posture interpolated from DEFAULT_POS (stand height) to a crouch pose
  (pose_height) by the commanded pelvis height."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    a = _robot(env)
    names = list(a.joint_names)
    pose = cfg.params["crouch_pose"]  # dict joint -> angle (entity names)
    std = cfg.params["std"]           # dict joint -> std
    self.q0 = a.data.default_joint_pos[0].clone()
    self.q1 = self.q0.clone()
    for k, v in pose.items():
      self.q1[names.index(k)] = v
    self.std = torch.tensor([std.get(n, 0.3) for n in names], device=env.device)

  def __call__(self, env: ManagerBasedRlEnv, command_name: str, crouch_pose: dict,
               std: dict, pose_height: float) -> torch.Tensor:
    del crouch_pose, std
    term = env.command_manager.get_term(command_name)
    h0 = term.cfg.stand_height
    p = ((h0 - term.h_cmd) / (h0 - pose_height)).clamp(0.0, 1.0).unsqueeze(-1)
    ref = self.q0 + p * (self.q1 - self.q0)
    err = torch.square(_robot(env).data.joint_pos - ref) / torch.square(self.std)
    return torch.exp(-err.mean(dim=1))


def crouch_still(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """After the descent ends: hold still (base velocity and joint velocity)."""
  term = env.command_manager.get_term(command_name)
  done = (term.progress >= 0.999).float()
  a = _robot(env)
  v = torch.sum(torch.square(a.data.root_link_lin_vel_w[:, :2]), dim=1)
  w = torch.sum(torch.square(a.data.root_link_ang_vel_b), dim=1)
  jv = torch.sum(torch.square(a.data.joint_vel), dim=1)
  return done * (v + 0.2 * w + 0.01 * jv)


def descent_speed(env: ManagerBasedRlEnv, max_speed: float = 0.15) -> torch.Tensor:
  """Penalise pelvis vertical speed above `max_speed` (slow, controlled descent)."""
  vz = _robot(env).data.root_link_lin_vel_w[:, 2].abs()
  return torch.square((vz - max_speed).clamp_min(0.0))


__all__ = [n for n in dir() if not n.startswith("_")]
_ = C  # contract is imported for side-effect free access in configs
