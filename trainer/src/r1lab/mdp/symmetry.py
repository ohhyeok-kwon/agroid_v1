"""Left/right mirror augmentation for rsl_rl (policy-order contract tables)."""
from __future__ import annotations

import torch
from tensordict import TensorDict

from r1lab import contract as C


class Mirror:
  """Callable for rsl_rl `symmetry_cfg.data_augmentation_func`.

  critic_layout: tuple of (kind, dim) segments describing the critic vector, kinds:
    frame | vec3 (x,-y,z) | scalar | lr2 (swap 2) | lr_vec3 (swap 2x3 with y flip)
    ygrid (third item (ny, nx): y-row-major grid, rows reversed)
  """

  def __init__(self, cmd_sign=C.VEL_CMD_SIGN, critic_layout=()):
    self.cmd_sign = cmd_sign
    self.critic_layout = tuple(critic_layout)
    self._t = {}

  def _tables(self, device):
    if device not in self._t:
      idx = torch.tensor(C.MIRROR_INDEX, device=device)
      sgn = torch.tensor(C.MIRROR_SIGN, device=device)
      frame_idx = torch.cat([
        torch.arange(9, device=device), 9 + idx, 33 + idx, 57 + idx])
      frame_sign = torch.cat([
        torch.tensor(C.ANG_VEL_SIGN, device=device),
        torch.tensor(C.GRAVITY_SIGN, device=device),
        torch.tensor(self.cmd_sign, device=device, dtype=torch.float32),
        sgn, sgn, sgn])
      self._t[device] = (idx, sgn, frame_idx, frame_sign)
    return self._t[device]

  def frame(self, x: torch.Tensor) -> torch.Tensor:
    _, _, fi, fs = self._tables(x.device)
    return x[..., fi] * fs

  def actor(self, x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    return self.frame(x.view(b, C.HISTORY, C.FRAME_DIM)).reshape(b, -1)

  def critic(self, x: torch.Tensor) -> torch.Tensor:
    out, o = [], 0
    for kind, dim, *extra in self.critic_layout:
      seg = x[:, o:o + dim]
      if kind == "frame":
        seg = self.frame(seg)
      elif kind == "actor":
        seg = self.actor(seg)
      elif kind == "vec3":
        seg = seg * torch.tensor([1.0, -1.0, 1.0], device=x.device)
      elif kind == "lr2":
        seg = seg[:, [1, 0]]
      elif kind == "lr_vec3":
        seg = seg.view(-1, 2, 3)[:, [1, 0]] * torch.tensor([1.0, -1.0, 1.0], device=x.device)
        seg = seg.reshape(-1, 6)
      elif kind == "ygrid":
        ny, nx = extra[0]
        seg = seg.reshape(-1, ny, nx).flip(1).reshape(-1, dim)
      elif kind != "scalar":
        raise ValueError(kind)
      out.append(seg)
      o += dim
    assert o == x.shape[1], (o, x.shape)
    return torch.cat(out, dim=1)

  def act(self, a: torch.Tensor) -> torch.Tensor:
    idx, sgn, _, _ = self._tables(a.device)
    return a[:, idx] * sgn

  def __call__(self, env=None, obs: TensorDict | None = None, actions: torch.Tensor | None = None):
    obs_aug = None
    if obs is not None:
      b = obs.batch_size[0]
      data = {}
      for k, v in obs.items():
        m = self.actor(v) if k == "actor" else self.critic(v)
        data[k] = torch.cat([v, m], dim=0)
      obs_aug = TensorDict(data, batch_size=[2 * b], device=obs.device)
    act_aug = None
    if actions is not None:
      act_aug = torch.cat([actions, self.act(actions)], dim=0)
    return obs_aug, act_aug
