"""Surprise of the robot's recent motion under a fixed human-motion density (torch, batched over environments).

The density is the R0 G0 model (.agent/adversarial/positive/tools/proxy_g0.py, dynamics factor): windows of 1.0 s of
50 Hz joint angles (22 body joints; wrist roll is constant in every human retarget and carries no information),
features
  log RMS of joint velocity / acceleration / jerk                        (22 x 3, after a zero-phase 12 Hz low-pass)
  log band power of joint velocity per body group, 0.5-1.5 / 1.5-2.5 / 2.5-3.5 / 3.5-5 / 5-10 Hz   (5 groups x 5)
  log power above 12 Hz per body group (unfiltered: motion the human data never contain)            (5)
  log RMS root angular rate (3)
standardised -> PCA k + isotropic residual -> Gaussian mixture. S = -log p (nats per window).

Low-activity clamp (one-sided surprise): every feature is a log activity; it is raised to the 5th percentile of the human
windows before scoring, so moving LESS than the quietest human windows is never surprising. The user's rule "move only
what is needed" outranks looking like a person; the density must not pay for invented idle motion.

Reward use: D = objective.naturalness(S): 1 up to S_ref (median human surprise), heavy-tailed beyond (parameter file).
Parameters: an .npz written by .agent/adversarial/positive/tools/export_surprise.py from the G0 fit.
"""
from __future__ import annotations

import math

import numpy as np
import torch

DT = 0.02
WIN = 50
PAD = 50                      # extra history so the zero-phase filter edge does not reach the scored window
BUFFER = WIN + 3 + PAD
GROUPS = ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11), (12, 13), (14, 15, 16, 17), (19, 20, 21, 22))
USED = tuple(i for i in range(24) if i not in (18, 23))
BANDS = ((0.5, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 5.0), (5.0, 10.0))
HIGH = 12.0
LOWPASS_HZ = 12.0


def _butter_gain(freqs: torch.Tensor, fc: float, order: int = 4) -> torch.Tensor:
  """|H|^2 of a Butterworth low-pass: the magnitude a forward-backward (zero-phase) pass applies."""
  return 1.0 / (1.0 + (freqs / fc) ** (2 * order))


def zero_phase_lowpass(x: torch.Tensor, fc: float = LOWPASS_HZ) -> torch.Tensor:
  """x (N, T, J): zero-phase low-pass by FFT with mirrored extension (approximates scipy filtfilt)."""
  n = x.shape[1]
  ext = torch.cat([x.flip(1), x, x.flip(1)], dim=1)
  f = torch.fft.rfftfreq(ext.shape[1], DT).to(x.device)
  y = torch.fft.irfft(torch.fft.rfft(ext, dim=1) * _butter_gain(f, fc)[None, :, None], n=ext.shape[1], dim=1)
  return y[:, n:2 * n]


def body_rates(q: torch.Tensor) -> torch.Tensor:
  """q (N, T, 4) wxyz -> body-frame angular velocity (N, T-1, 3)."""
  q = q / q.norm(dim=-1, keepdim=True)
  inv = q[:, :-1] * torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device)
  w1, x1, y1, z1 = inv.unbind(-1)
  w2, x2, y2, z2 = q[:, 1:].unbind(-1)
  d = torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                   w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
  d = d * torch.sign(d[..., :1] + 1e-12)
  vec = d[..., 1:]
  ang = 2.0 * torch.atan2(vec.norm(dim=-1), d[..., 0])
  return vec / vec.norm(dim=-1, keepdim=True).clamp_min(1e-9) * (ang / DT).unsqueeze(-1)


def features(jp: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
  """Features of the LAST 1-s window. jp (N, T, 24) rad, quat (N, T, 4) wxyz, T >= WIN + 3."""
  x = zero_phase_lowpass(jp)
  v = torch.diff(x, dim=1) / DT
  a = torch.diff(v, dim=1) / DT
  j = torch.diff(a, dim=1) / DT
  vraw = torch.diff(jp, dim=1) / DT
  used = list(USED)
  vv, aa, jj, vr = v[:, -WIN:], a[:, -WIN:], j[:, -WIN:], vraw[:, -WIN:]
  out = [torch.log(vv[..., used].pow(2).mean(1).sqrt() + 1e-3), torch.log(aa[..., used].pow(2).mean(1).sqrt() + 1e-2),
         torch.log(jj[..., used].pow(2).mean(1).sqrt() + 1e-1)]
  hann = torch.hann_window(WIN, periodic=False, dtype=jp.dtype, device=jp.device)[None, :, None]
  fr = torch.fft.rfftfreq(WIN, DT).to(jp.device)
  P = torch.fft.rfft((vv - vv.mean(1, keepdim=True)) * hann, dim=1).abs().pow(2)
  Pr = torch.fft.rfft((vr - vr.mean(1, keepdim=True)) * hann, dim=1).abs().pow(2)
  bands = []
  for g in GROUPS:
    pg = P[..., list(g)].sum(-1)
    for lo, hi in BANDS:
      bands.append(torch.log(pg[:, (fr >= lo) & (fr < hi)].sum(1) + 1e-4))
  for g in GROUPS:
    bands.append(torch.log(Pr[:, fr >= HIGH][..., list(g)].sum((1, 2)) + 1e-4))
  out.append(torch.stack(bands, 1))
  w = body_rates(quat)[:, -WIN:]
  out.append(torch.log(w.pow(2).mean(1).sqrt() + 1e-3))
  return torch.cat(out, 1)


class HumanSurprise(torch.nn.Module):
  """-log p_human of the last second of motion, per environment."""

  def __init__(self, params: dict, num_envs: int, device: str = "cpu", clamp_low: bool = True):
    super().__init__()
    t = {k: torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device) for k, v in params.items()
         if k in ("mu", "sd", "zm", "V", "res_var", "m", "C", "pi", "f_p5", "s_ref", "s_scale")}
    self.mu, self.sd, self.zm, self.V = t["mu"], t["sd"], t["zm"], t["V"]
    self.res_var = float(t["res_var"])
    self.logpi = torch.log(t["pi"])
    chol = torch.linalg.cholesky(t["C"])
    self.m, self.chol = t["m"], chol
    self.logdet = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)
    self.f_p5 = t.get("f_p5")
    self.s_ref = float(t["s_ref"]) if "s_ref" in t else 0.0
    self.s_scale = float(t["s_scale"]) if "s_scale" in t else 50.0
    self.clamp_low = clamp_low and self.f_p5 is not None
    self.jp = torch.zeros(num_envs, BUFFER, 24, device=device)
    self.quat = torch.zeros(num_envs, BUFFER, 4, device=device)
    self.quat[..., 0] = 1.0
    self.count = torch.zeros(num_envs, dtype=torch.long, device=device)

  def reset(self, env_ids: torch.Tensor, jp: torch.Tensor, quat: torch.Tensor) -> None:
    """Fill the buffer of restarted environments with their first frame (a still start is not surprising)."""
    self.jp[env_ids] = jp[:, None, :24]
    self.quat[env_ids] = quat[:, None]
    self.count[env_ids] = 0

  def push(self, jp: torch.Tensor, quat: torch.Tensor) -> None:
    self.jp = torch.roll(self.jp, -1, dims=1)
    self.quat = torch.roll(self.quat, -1, dims=1)
    self.jp[:, -1] = jp[:, :24]
    self.quat[:, -1] = quat
    self.count += 1

  def log_density(self, f: torch.Tensor) -> torch.Tensor:
    if self.clamp_low:
      f = torch.maximum(f, self.f_p5)
    z = (f - self.mu) / self.sd - self.zm
    y = z @ self.V
    r = z - y @ self.V.T
    d_res = z.shape[1] - y.shape[1]
    l_res = -0.5 * r.pow(2).sum(1) / self.res_var - 0.5 * d_res * math.log(2 * math.pi * self.res_var)
    diff = y[:, None, :] - self.m[None]                                    # (N, K, k)
    sol = torch.linalg.solve_triangular(self.chol[None].expand(len(y), -1, -1, -1), diff.unsqueeze(-1), upper=False)
    maha = sol.squeeze(-1).pow(2).sum(-1)                                  # (N, K)
    comp = -0.5 * maha - self.logdet[None] - 0.5 * y.shape[1] * math.log(2 * math.pi) + self.logpi[None]
    return torch.logsumexp(comp, dim=1) + l_res

  def surprise(self) -> tuple[torch.Tensor, torch.Tensor]:
    """(S (N,), valid (N,)): valid once a full window has been seen since the last reset."""
    s = -self.log_density(features(self.jp, self.quat))
    return s, self.count >= WIN + 3

  def naturalness(self, form: str = "cauchy") -> tuple[torch.Tensor, torch.Tensor]:
    """(D in (0, 1], S): D = objective.naturalness(S). Also right after a reset: the buffer then holds the reset frame
    (held still) followed by what has happened since, so the score is never a free 1 (smoke test 11-17 02:04: masking
    the first 53 steps gave falling policies D ~ 0.8)."""
    from r1adv.positive.objective import naturalness
    s, _ = self.surprise()
    return naturalness(s, self.s_ref, self.s_scale, form), s
