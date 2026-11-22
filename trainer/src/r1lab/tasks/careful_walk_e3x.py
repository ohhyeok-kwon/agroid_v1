"""E3 second board (positive 23, coordinator 2026-11-20 22:5x / 23:0x: one seed per arm, four GPUs on four arms).
DIAGNOSTIC. Arms on top of B0 (r1lab.tasks.careful_walk_b0):
  cap     the policy std learned as in B0 (scalar per joint, from 0.8) but clipped at its starting value 0.8 (rsl_rl
          GaussianDistribution std_range (1e-6, 0.8): per joint, raw policy-output units, the clamp B0 has at 1e6)
  capa04  cap + the a04 push boundary d (r1lab.tasks.careful_walk_e3.a04_env_cfg)
  col     time-correlated (low-pass) exploration noise (R18-a r05): the sampled action is mean + std x e_t with e_t a
          stationary AR(1) / Ornstein-Uhlenbeck sequence per environment and joint, e_t = rho e_(t-1) + sqrt(1 - rho^2) xi_t,
          rho = exp(-dt / tau), dt 20 ms (policy step), tau 30 ms (inside the asked 20-40 ms), so each e_t is N(0, 1) and
          PPO's log-probability of the action is the usual N(mean, std) one (the noise state is not observed; colored-noise
          PPO as Hollenstein et al. 2022). std learned from 0.8 as in B0. The state continues across episode resets."""
from __future__ import annotations

import math

import torch
from rsl_rl.modules.distribution import GaussianDistribution

CAP = 0.8
TAU_S, DT_S = 0.03, 0.02


class ColoredGaussian(GaussianDistribution):
  def __init__(self, output_dim: int, init_std: float = 1.0, std_range: tuple[float, float] = (1e-6, 1e6),
               std_type: str = "scalar", learn_std: bool = True, tau_s: float = TAU_S, dt_s: float = DT_S) -> None:
    super().__init__(output_dim, init_std=init_std, std_range=std_range, std_type=std_type, learn_std=learn_std)
    self.rho = math.exp(-dt_s / tau_s)
    self._eps: torch.Tensor | None = None

  def sample(self) -> torch.Tensor:
    mean, std = self._distribution.mean, self._distribution.stddev
    xi = torch.randn_like(mean)
    if self._eps is None or self._eps.shape != mean.shape or self._eps.device != mean.device:
      self._eps = xi
    else:
      self._eps = self.rho * self._eps + math.sqrt(1.0 - self.rho ** 2) * xi
    return mean + std * self._eps


def _b0():
  from r1lab.tasks.careful_walk_b0 import b0_rl_cfg
  return b0_rl_cfg()


def cap_rl_cfg():
  cfg = _b0()
  cfg.actor.distribution_cfg = {"class_name": "GaussianDistribution", "init_std": CAP, "std_type": "scalar",
                                "std_range": [1e-6, CAP]}
  return cfg


def col_rl_cfg():
  cfg = _b0()
  cfg.actor.distribution_cfg = {"class_name": "r1lab.tasks.careful_walk_e3x:ColoredGaussian", "init_std": 0.8,
                                "std_type": "scalar", "tau_s": TAU_S, "dt_s": DT_S}
  return cfg
