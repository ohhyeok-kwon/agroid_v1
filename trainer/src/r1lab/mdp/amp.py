"""Adversarial Motion Prior (Peng et al. 2021) as a self-contained mjlab reward term.

Style features (44, lower body + trunk; arms are left to the posture reward because many
mocap clips hold props):
  q[legs+waist] (14) | dq[legs+waist] (14) | pelvis height over lower foot (1) |
  projected gravity (3) | pelvis lin vel body (3) | pelvis ang vel body (3) |
  feet positions rel. pelvis, yaw frame (6)
The discriminator sees (s_t, s_t+1) at 50 Hz, LSGAN targets +1 (mocap) / -1 (policy),
gradient penalty on mocap samples; style reward = max(0, 1 - 0.25 (D - 1)^2).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import nn

from mjlab.managers.reward_manager import RewardTermCfg

from r1lab.mdp.observations import feet_rel_pos
from r1lab.mdp.rewards import rel_height

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

NJ = 14  # legs (12) + waist (2) in entity order
FEAT = 44
# Mirror in feature space (entity order joints 0..13).
_J_IDX = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5, 12, 13]
_J_SGN = [1, -1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1]


def mirror_features(f: np.ndarray | torch.Tensor):
  lib = torch if isinstance(f, torch.Tensor) else np
  idx = list(range(FEAT))
  sgn = [1.0] * FEAT
  for i in range(NJ):
    idx[i], sgn[i] = _J_IDX[i], float(_J_SGN[i])
    idx[NJ + i], sgn[NJ + i] = NJ + _J_IDX[i], float(_J_SGN[i])
  o = 2 * NJ  # 28: height
  for k, s in zip(range(o + 1, o + 4), (1, -1, 1)):    # gravity
    sgn[k] = s
  for k, s in zip(range(o + 4, o + 7), (1, -1, 1)):    # lin vel
    sgn[k] = s
  for k, s in zip(range(o + 7, o + 10), (-1, 1, -1)):  # ang vel
    sgn[k] = s
  fo = o + 10  # feet L xyz, R xyz -> swap, flip y
  for j, (src, s) in enumerate(zip([3, 4, 5, 0, 1, 2], (1, -1, 1, 1, -1, 1))):
    idx[fo + j], sgn[fo + j] = fo + src, s
  if lib is torch:
    t_idx = torch.tensor(idx, device=f.device)
    return f[..., t_idx] * torch.tensor(sgn, device=f.device, dtype=f.dtype)
  return f[..., idx] * np.asarray(sgn, dtype=f.dtype)


def sim_features(env: ManagerBasedRlEnv) -> torch.Tensor:
  d = env.scene["robot"].data
  return torch.cat([
    d.joint_pos[:, :NJ], d.joint_vel[:, :NJ], rel_height(env).unsqueeze(-1),
    d.projected_gravity_b, d.root_link_lin_vel_b, d.root_link_ang_vel_b, feet_rel_pos(env),
  ], dim=-1)


def ref_features(qpos: np.ndarray, fps: float, model=None) -> np.ndarray:
  """Same features from a qpos trajectory (MuJoCo FK on CPU). Returns (T, 44)."""
  import mujoco
  from scipy.spatial.transform import Rotation as R

  from r1lab.robot import R1_XML
  m = model or mujoco.MjModel.from_xml_path(str(R1_XML))
  d = mujoco.MjData(m)
  T = len(qpos)
  dt = 1.0 / fps
  rot = R.from_quat(qpos[:, [4, 5, 6, 3]])
  Rm = rot.as_matrix()
  q = qpos[:, 7:7 + NJ]
  dq = np.gradient(q, dt, axis=0)
  v_w = np.gradient(qpos[:, :3], dt, axis=0)
  v_b = np.einsum("tji,tj->ti", Rm, v_w)
  rel = (rot[:-1].inv() * rot[1:]).as_rotvec() / dt
  w_b = np.concatenate([rel, rel[-1:]], 0)
  g_b = np.einsum("tji,j->ti", Rm, np.array([0.0, 0.0, -1.0]))
  sid = [m.site("left_foot").id, m.site("right_foot").id]
  feet = np.zeros((T, 6))
  h = np.zeros(T)
  yaw = np.arctan2(Rm[:, 1, 0], Rm[:, 0, 0])
  for t in range(T):
    d.qpos[:] = qpos[t]
    mujoco.mj_kinematics(m, d)
    p = d.site_xpos[sid] - qpos[t, :3]
    c, s = np.cos(yaw[t]), np.sin(yaw[t])
    rz = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    feet[t] = (p @ rz.T).reshape(-1)
    h[t] = qpos[t, 2] - d.site_xpos[sid, 2].min()
  return np.concatenate([q, dq, h[:, None], g_b, v_b, w_b, feet], -1).astype(np.float32)


class Discriminator(nn.Module):
  def __init__(self, dims=(1024, 512)):
    super().__init__()
    layers, i = [], 2 * FEAT
    for h in dims:
      layers += [nn.Linear(i, h), nn.ReLU()]
      i = h
    self.trunk = nn.Sequential(*layers)
    self.head = nn.Linear(i, 1)

  def forward(self, x):
    return self.head(self.trunk(x))


class amp_style:
  """Reward term: style reward from an online-trained AMP discriminator."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    dev = env.device
    data = np.load(p["data_file"])
    pairs = torch.from_numpy(data["pairs"]).to(dev)  # (M, 88), mirrored included
    self.ref = pairs
    self.ref_w = torch.from_numpy(data["weights"]).to(dev) if "weights" in data else None
    mu, sd = pairs.mean(0), pairs.std(0).clamp_min(1e-2)
    self.mu, self.sd = mu, sd
    self.disc = Discriminator().to(dev)
    self.opt = torch.optim.Adam(self.disc.parameters(), lr=p.get("lr", 1e-4), weight_decay=1e-4)
    self.buf = torch.zeros(p.get("buffer", 400_000), 2 * FEAT, device=dev)
    self.buf_n, self.buf_i = 0, 0
    self.prev = torch.zeros(env.num_envs, FEAT, device=dev)
    self.valid = torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
    self.steps = 0
    self.update_every = p.get("update_every", 24)
    self.n_updates = p.get("num_updates", 4)
    self.batch = p.get("batch", 4096)
    self.gp = p.get("grad_penalty", 5.0)
    self.ckpt = p.get("save_path")
    if p.get("load_path") and Path(p["load_path"]).exists():
      sd_ = torch.load(p["load_path"], map_location=dev)
      self.disc.load_state_dict(sd_["disc"])
      print(f"[AMP] loaded discriminator {p['load_path']}")

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    self.valid[env_ids] = False

  def _norm(self, x):
    return (x - self.mu) / self.sd

  def _push(self, x):
    n = x.shape[0]
    if n == 0:
      return
    cap = self.buf.shape[0]
    idx = (torch.arange(n, device=x.device) + self.buf_i) % cap
    self.buf[idx] = x
    self.buf_i = int((self.buf_i + n) % cap)
    self.buf_n = min(cap, self.buf_n + n)

  def _update(self, env):
    with torch.inference_mode(False), torch.enable_grad():
      for _ in range(self.n_updates):
        pi = torch.randint(0, self.buf_n, (self.batch,), device=self.ref.device)
        if self.ref_w is not None:
          ri = torch.multinomial(self.ref_w, self.batch, replacement=True)
        else:
          ri = torch.randint(0, self.ref.shape[0], (self.batch,), device=self.ref.device)
        x_pi = self._norm(self.buf[pi].clone())
        x_ref = self._norm(self.ref[ri].clone()).requires_grad_(True)
        d_pi = self.disc(x_pi)
        d_ref = self.disc(x_ref)
        loss = ((d_ref - 1) ** 2).mean() + ((d_pi + 1) ** 2).mean()
        g = torch.autograd.grad(d_ref.sum(), x_ref, create_graph=True)[0]
        loss = loss + self.gp * g.pow(2).sum(-1).mean()
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.disc.parameters(), 1.0)
        self.opt.step()
    env.extras["log"]["Metrics/amp_d_ref"] = d_ref.detach().mean()
    env.extras["log"]["Metrics/amp_d_policy"] = d_pi.detach().mean()

  def __call__(self, env: ManagerBasedRlEnv, data_file: str, **kwargs) -> torch.Tensor:
    del data_file, kwargs
    f = sim_features(env)
    pair = torch.cat([self.prev, f], dim=-1)
    with torch.no_grad():
      d = self.disc(self._norm(pair)).squeeze(-1)
    r = (1.0 - 0.25 * (d - 1.0) ** 2).clamp_min(0.0) * self.valid.float()
    self._push(pair[self.valid])
    self.prev = f.clone()
    self.valid[:] = True
    self.steps += 1
    if self.steps % self.update_every == 0 and self.buf_n >= self.batch:
      self._update(env)
      if self.ckpt and self.steps % (self.update_every * 100) == 0:
        Path(self.ckpt).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"disc": self.disc.state_dict()}, self.ckpt)
    env.extras["log"]["Metrics/amp_style"] = r.mean()
    return r
