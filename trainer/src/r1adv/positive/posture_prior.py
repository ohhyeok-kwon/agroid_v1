"""Human neutral-standing support of the R3 wave-1 'distribution posture L' cells (L1, BL; coordinator 2026-11-17 21:41,
fixes 22:1x). The posture channel of the Hold likelihood L = 1 / (1 + 1/2 sum_k (e_k / sigma_k)^2) is, in the point cells
(C0, B1), P_point = (posture RMS to q* / 0.02)^2 = |q - q*|^2 / (24 * 0.02^2). In the distribution cells it is

    P(q) = g(q) * |q - q*|^2 / (24 * 0.02^2),     g = min(1, max(g_J, g_W, g_S)) in [0, 1]

  the point term with the SAME strength outside the human neutral-standing support and 0 inside it:
  g_J = (NLL*(q_U) - c)+ / T_J       joint support: NLL*(x) = min_k [(x - mu_k)^T S_k^-1 (x - mu_k) / 2 - log w_k] of a
                                     Gaussian mixture on the 22 joints without the wrist rolls, fitted to the stance
                                     gate's neutral quiet frames and mirror-averaged (left-right symmetric); c = the median
                                     held-out NLL* of those frames ('at least as likely as the median human frame');
                                     T_J = the NLL* rise of a 0.02 rad RMS step outward along the least-variable direction
  g_W = (width - w99)+ / ramp        feet width (left - right ankle_roll_link across the pelvis heading) beyond the human
                                     neutral p99 (the stance gate's own bound)
  g_S = (|stagger| - s99)+ / ramp    fore-aft stagger along the heading beyond the human neutral p99
All numbers live in the fitted file (.agent/adversarial/positive/tools/w1_posture_gmm.py, v3), pinned by sha256 in
hold_w1.POSTURE_GMM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def stance(feet_xy: torch.Tensor, root_quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
  """(width, |stagger|) of the feet: feet_xy (N, 2, 2) = left / right ankle_roll_link xy (world), root_quat (N, 4) wxyz;
  width across and stagger along the pelvis heading (the negative runner's / stance gate's definition)."""
  w, x, y, z = root_quat.unbind(-1)
  yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
  d = feet_xy[:, 0] - feet_xy[:, 1]
  fwd = d[:, 0] * torch.cos(yaw) + d[:, 1] * torch.sin(yaw)
  lat = -d[:, 0] * torch.sin(yaw) + d[:, 1] * torch.cos(yaw)
  return lat.abs(), fwd.abs()


class PosturePrior:
  def __init__(self, path: str | Path, device: str | torch.device = "cpu"):
    z = np.load(path)
    f = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)  # noqa: E731
    self.joint_names = [str(n) for n in z["joint_names"]]
    self.used = torch.as_tensor(z["used"], dtype=torch.long, device=device)
    if len(self.joint_names) != 24 or len(self.used) != 22:
      raise ValueError("the posture support must cover the 24 body joints (22 in the mixture)")
    self.mu, self.lam, self.V = f(z["mu"]), f(z["eigval"]), f(z["eigvec"])      # (K, 22), (K, 22), (K, 22, 22)
    self.logw = f(z["logw"])
    self.c, self.price = float(z["c"]), float(z["price_rad2"])
    self.T_J, self.w99, self.s99, self.ramp = (float(z[k]) for k in ("gate_T_J", "width_p99", "stagger_p99", "stance_ramp"))
    self.stance_bodies = [str(b) for b in z["stance_body"]]
    if abs(self.price - 24 * 0.02 ** 2) > 1e-12 or min(self.T_J, self.ramp) <= 0:
      raise ValueError("bad posture constants")

  def constants(self) -> dict:
    return dict(c=self.c, price_rad2=self.price, T_J=self.T_J, width_p99=self.w99, stagger_p99=self.s99, ramp=self.ramp,
                components=int(self.mu.shape[0]))

  def nll_max(self, q: torch.Tensor) -> torch.Tensor:
    """NLL* of the 22 joints (the joint support is {NLL* <= c})."""
    z = torch.einsum("kji,nkj->nki", self.V, q[:, self.used][:, None, :] - self.mu[None])
    return (0.5 * (z * z / self.lam[None]).sum(-1) - self.logw[None]).min(1).values

  def gate(self, q: torch.Tensor, width: torch.Tensor, stagger: torch.Tensor) -> torch.Tensor:
    g_j = (self.nll_max(q) - self.c).clamp_min(0.0) / self.T_J
    g_w = (width - self.w99).clamp_min(0.0) / self.ramp
    g_s = (stagger.abs() - self.s99).clamp_min(0.0) / self.ramp
    return torch.maximum(torch.maximum(g_j, g_w), g_s).clamp(max=1.0)

  def cost(self, q: torch.Tensor, q_star: torch.Tensor, width: torch.Tensor, stagger: torch.Tensor) -> torch.Tensor:
    """P = g * |q - q*|^2 / (24 * 0.02^2): the posture channel's term inside the likelihood's sum."""
    return self.gate(q, width, stagger) * (q - q_star).pow(2).sum(1) / self.price
