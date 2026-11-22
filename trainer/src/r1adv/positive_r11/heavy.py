"""Heavy-tailed versions of B0's human-motion density d_v0b (positive session 17, R11 / W8 analysis helper; the training term
lives in logprod.py and freezes ONE form).

B0's D (r1adv.positive.human_surprise.HumanSurprise.log_density) is a Gaussian mixture in the 99-dim standardised feature
space z: PCA y = z V (k 20), residual r = z - y V^T scored by one isotropic Gaussian (res_var) shared by every component,
mixture over y (K 10: means m, covariances C, weights pi). Because V has orthonormal columns this is exactly the mixture of
99-dim Gaussians N(z; V m_c, Sigma_c) with Sigma_c = V C_c V^T + res_var (I - V V^T), log|Sigma_c| = log|C_c| + 79 log res_var,
Mahalanobis delta_c^2 = (y - m_c)^T C_c^-1 (y - m_c) + |r|^2 / res_var.
Forms (same means, scale matrices, weights; the low-activity clamp f >= f_p5 and the z-space of B0 kept):
  gauss  log N_99 (B0's own log p, check)
  t1     multivariate Student-t, nu = 1, p = 99 (multivariate Cauchy):
         log Gamma((1+p)/2) - log Gamma(1/2) - p/2 log pi - 1/2 log|Sigma_c| - (1+p)/2 log(1 + delta_c^2)
  k1     the objective.py 'cauchy' kernel per component with the Gaussian peak kept:
         -p/2 log(2 pi) - 1/2 log|Sigma_c| - log(1 + delta_c^2 / 2)   (= the Gaussian component to second order at its mean)
log D~ = logsumexp_c(log pi_c + comp_c)."""
from __future__ import annotations

import math

import torch

P = 99


class HeavyMixture:
  """Holds the d_v0b parameters of a HumanSurprise instance and scores features under the forms above."""

  def __init__(self, hs):
    self.hs = hs
    k = hs.V.shape[1]
    self.k, self.d_res = k, P - k
    eye = torch.eye(k, dtype=hs.V.dtype, device=hs.V.device)
    err = float((hs.V.T @ hs.V - eye).abs().max())
    if err > 1e-4:
      raise ValueError(f"d_v0b V is not orthonormal (max |V^T V - I| {err})")
    self.ortho_err = err
    self.logdet_sigma = 2.0 * hs.logdet + self.d_res * math.log(hs.res_var)        # (K,)
    self.t1_const = math.lgamma((1 + P) / 2) - math.lgamma(0.5) - 0.5 * P * math.log(math.pi)
    self.g_const = -0.5 * P * math.log(2 * math.pi)

  def delta2(self, f: torch.Tensor) -> torch.Tensor:
    hs = self.hs
    if hs.clamp_low:
      f = torch.maximum(f, hs.f_p5)
    z = (f - hs.mu) / hs.sd - hs.zm
    y = z @ hs.V
    r = z - y @ hs.V.T
    rr = r.pow(2).sum(1) / hs.res_var
    diff = y[:, None, :] - hs.m[None]
    sol = torch.linalg.solve_triangular(hs.chol[None].expand(len(y), -1, -1, -1), diff.unsqueeze(-1), upper=False)
    return sol.squeeze(-1).pow(2).sum(-1) + rr[:, None]                               # (N, K)

  def log_density(self, f: torch.Tensor, form: str) -> torch.Tensor:
    d2 = self.delta2(f)
    base = self.hs.logpi[None] - 0.5 * self.logdet_sigma[None]
    if form == "gauss":
      comp = base + self.g_const - 0.5 * d2
    elif form == "t1":
      comp = base + self.t1_const - 0.5 * (1 + P) * torch.log1p(d2)
    elif form == "k1":
      comp = base + self.g_const - torch.log1p(0.5 * d2)
    else:
      raise ValueError(f"unknown form {form!r}")
    return torch.logsumexp(comp, dim=1)
