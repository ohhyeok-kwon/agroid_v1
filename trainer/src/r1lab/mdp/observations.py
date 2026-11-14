"""Observations: agroid frame-stacked actor input with a realistic sensor model,
plus privileged critic terms."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.sensor import ContactSensor

from r1lab import contract as C

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

GRAVITY = 9.81
ROBOT_MASS = 28.93


def _policy_perm(device) -> torch.Tensor:
  return torch.tensor(C.POLICY_TO_ENTITY, device=device, dtype=torch.long)


class frame_stack:
  """10 x 81 frame-major history, oldest first (agroid walking_action_manager layout).

  Sensor model (when ``noise=True``), sampled once per policy step and frozen into the
  history like on the robot:
    * gyro: per-episode bias and scale error, white noise, rare spikes;
    * gravity (IMU attitude): per-episode tilt bias with slow drift + white noise;
    * joints: encoder bias (mjlab dr.encoder_bias), white position/velocity noise;
    * latency: sensor part blended toward the previous sample by a per-episode
      fraction (0..1 policy step = 0..20 ms).
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    self.asset: Entity = env.scene["robot"]
    assert tuple(self.asset.joint_names) == C.ENTITY_JOINTS, self.asset.joint_names
    n, dev = env.num_envs, env.device
    self.n, self.dev = n, dev
    self.perm = _policy_perm(dev)
    self.default_e = self.asset.data.default_joint_pos[0].clone()
    self.hist = torch.zeros(n, C.HISTORY, C.FRAME_DIM, device=dev)
    self.prev_sensor = torch.zeros(n, 54, device=dev)
    self.sensor = torch.zeros(n, 54, device=dev)
    self.env_ids = torch.arange(n, device=dev)
    self.needs_fill = torch.ones(n, dtype=torch.bool, device=dev)
    self.last_step = -1
    self.noise = bool(p.get("noise", True))
    self.np = dict(
      gyro_std=0.08, gyro_bias=0.04, gyro_scale=0.04, spike_prob=0.002, spike_mag=1.0,
      tilt_bias=0.04, tilt_drift=0.0005, tilt_std=0.01,
      q_noise=0.01, dq_std=0.4, lag_prob=0.8,
    )
    self.np.update(p.get("noise_params", {}))
    self.gyro_bias = torch.zeros(n, 3, device=dev)
    self.gyro_scale = torch.ones(n, 3, device=dev)
    self.tilt_bias = torch.zeros(n, 2, device=dev)
    self.lag = torch.zeros(n, 1, device=dev)

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      env_ids = self.env_ids
    elif isinstance(env_ids, slice):
      env_ids = self.env_ids[env_ids]
    self.needs_fill[env_ids] = True
    if not self.noise or len(env_ids) == 0:
      return
    k, q = len(env_ids), self.np
    u = lambda *s: torch.rand(*s, device=self.dev) * 2.0 - 1.0  # noqa: E731
    self.gyro_bias[env_ids] = u(k, 3) * q["gyro_bias"]
    self.gyro_scale[env_ids] = 1.0 + u(k, 3) * q["gyro_scale"]
    self.tilt_bias[env_ids] = u(k, 2) * q["tilt_bias"]
    lag = torch.rand(k, 1, device=self.dev)
    on = torch.rand(k, 1, device=self.dev) < q["lag_prob"]
    self.lag[env_ids] = torch.where(on, lag, torch.zeros_like(lag))

  def _measure(self, env: ManagerBasedRlEnv, env_ids: torch.Tensor) -> torch.Tensor:
    d = self.asset.data
    gyro = d.root_link_ang_vel_b[env_ids]
    grav = d.projected_gravity_b[env_ids]
    q_e = (d.joint_pos_biased if self.noise else d.joint_pos)[env_ids] - self.default_e
    q = q_e[:, self.perm]
    dq = d.joint_vel[env_ids][:, self.perm]
    if self.noise:
      p = self.np
      n = env_ids.numel()
      gyro = gyro * self.gyro_scale[env_ids] + self.gyro_bias[env_ids] + torch.randn_like(gyro) * p["gyro_std"]
      spike = (torch.rand(n, 1, device=self.dev) < p["spike_prob"]).float()
      gyro = gyro + spike * (torch.rand_like(gyro) * 2 - 1) * p["spike_mag"]
      bias = self.tilt_bias[env_ids] + torch.randn(n, 2, device=self.dev) * p["tilt_drift"]
      self.tilt_bias[env_ids] = bias.clamp(-1.5 * p["tilt_bias"], 1.5 * p["tilt_bias"])
      rv = torch.zeros(n, 3, device=self.dev)
      rv[:, :2] = self.tilt_bias[env_ids] + torch.randn(n, 2, device=self.dev) * p["tilt_std"]
      grav = grav + torch.cross(rv, grav, dim=-1)
      grav = grav / grav.norm(dim=-1, keepdim=True).clamp_min(1e-6)
      q = q + (torch.rand_like(q) * 2 - 1) * p["q_noise"]
      dq = dq + torch.randn_like(dq) * p["dq_std"]
    return torch.cat([gyro, grav, q, dq], dim=-1)

  def __call__(self, env: ManagerBasedRlEnv, command_name: str, noise: bool = True,
               noise_params: dict | None = None) -> torch.Tensor:
    del noise, noise_params
    step = env.common_step_counter
    # Queries may repeat while commands are edited. Advance sensing/noise/latency
    # only once per policy tick, except for environments explicitly reset at it.
    ids = self.env_ids if step != self.last_step else self.needs_fill.nonzero(as_tuple=False).flatten()
    if ids.numel():
      measured = self._measure(env, ids)
      prev = torch.where(self.needs_fill[ids, None], measured, self.prev_sensor[ids])
      self.prev_sensor[ids] = measured
      self.sensor[ids] = ((1.0 - self.lag[ids]) * measured + self.lag[ids] * prev
                          if self.noise else measured)
    sensor = self.sensor
    cmd = env.command_manager.get_command(command_name)[:, :3]
    act = env.action_manager.action
    frame = torch.cat([sensor[:, :6], cmd, sensor[:, 6:30], sensor[:, 30:54], act], dim=-1)
    if step != self.last_step:
      self.hist = torch.roll(self.hist, shifts=-1, dims=1)
    self.hist[:, -1] = frame
    if self.needs_fill.any():
      ids = self.needs_fill.nonzero(as_tuple=False).flatten()
      self.hist[ids] = frame[ids].unsqueeze(1).expand(-1, C.HISTORY, -1)
      self.needs_fill[ids] = False
    self.last_step = step
    return self.hist.reshape(self.n, C.OBS_DIM)


def true_frame(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Clean current frame (critic)."""
  asset: Entity = env.scene["robot"]
  d = asset.data
  perm = _policy_perm(env.device)
  q = (d.joint_pos - d.default_joint_pos)[:, perm]
  dq = d.joint_vel[:, perm]
  cmd = env.command_manager.get_command(command_name)[:, :3]
  return torch.cat([d.root_link_ang_vel_b, d.projected_gravity_b, cmd, q, dq,
                    env.action_manager.action], dim=-1)


def base_lin_vel(env: ManagerBasedRlEnv) -> torch.Tensor:
  return env.scene["robot"].data.root_link_lin_vel_b


def base_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Pelvis height above the lower foot sole site."""
  asset: Entity = env.scene["robot"]
  site_ids, _ = asset.find_sites(("left_foot", "right_foot"), preserve_order=True)
  foot_z = asset.data.site_pos_w[:, site_ids, 2]
  return (asset.data.root_link_pos_w[:, 2] - foot_z.min(dim=1).values).unsqueeze(-1)


def feet_contact(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  s: ContactSensor = env.scene[sensor_name]
  return (s.data.found > 0).float()


def feet_force(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  s: ContactSensor = env.scene[sensor_name]
  return (s.data.force / (ROBOT_MASS * GRAVITY)).reshape(env.num_envs, -1)


def feet_rel_pos(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Foot site positions relative to pelvis, yaw-aligned body frame (L xyz, R xyz)."""
  from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
  asset: Entity = env.scene["robot"]
  site_ids, _ = asset.find_sites(("left_foot", "right_foot"), preserve_order=True)
  rel = asset.data.site_pos_w[:, site_ids] - asset.data.root_link_pos_w[:, None]
  qy = yaw_quat(asset.data.root_link_quat_w)[:, None].expand(-1, 2, -1)
  return quat_apply_inverse(qy, rel).reshape(env.num_envs, 6)
