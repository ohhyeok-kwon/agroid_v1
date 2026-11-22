"""S1 policy-std parametrisations for S0b (R12-G.16; coordinator 2026-11-20 06:1x; negative 18 addendum 37 arms).

Prepared, not chosen: the fix is picked after negative 18's verdict and the coordinator's decision. Off by default
(numbers.SIGMA_PARAM = "scalar" is rsl_rl's GaussianDistribution exactly as in S0).

  MirrorTiedGaussian   (arm (ㄱ)) one std parameter per orbit of the actor's own action mirror map
                       (r1s1.model.action_mirror: 11 left/right pairs share one value, the two waist joints keep their
                       own: 13 parameters), expanded to the 24 action dimensions; std_type "scalar" or "log" (arm (ㄷ)
                       on top). The orbit map is a persistent buffer, so a checkpoint carries it.
  (arm (ㄷ) alone is rsl_rl's GaussianDistribution with std_type "log": no class of its own.)
  FixedGaussian        (S0b-fixed, coordinator 07:5x / 08:0x) a constant per-joint std (numbers.FIXED_STD x
                       numbers.SIGMA_FIXED_SCALE) held as a buffer: not a parameter, not in the optimizer.

sigma_of(distribution) and sigma_from_state(actor_state_dict) give the 24 per-joint std of any of the four
parametrisations (the per-update records, the symmetry / std gates and the tools read them).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import GaussianDistribution
from torch.distributions import Normal

STD_RANGE = (1e-6, 1e6)                   # rsl_rl's default clamp (GaussianDistribution)


def mirror_orbits(width: int) -> torch.Tensor:
  """Index of each action dimension's orbit under the actor's mirror map (pairs share an index)."""
  from r1s1.model import action_mirror
  amap, _ = action_mirror("cpu")
  if len(amap) != width:
    raise ValueError(f"action mirror width {len(amap)} != distribution width {width}")
  rep = torch.minimum(torch.arange(width), amap.long())
  if not torch.equal(amap.long()[amap.long()], torch.arange(width)):
    raise ValueError("action mirror map is not an involution")
  keys = sorted(set(rep.tolist()))
  return torch.tensor([keys.index(r) for r in rep.tolist()], dtype=torch.long)


class MirrorTiedGaussian(GaussianDistribution):
  def __init__(self, output_dim: int, init_std: float = 1.0, std_range: tuple[float, float] = STD_RANGE,
               std_type: str = "scalar", learn_std: bool = True) -> None:
    super().__init__(output_dim, init_std=init_std, std_range=std_range, std_type=std_type, learn_std=learn_std)
    index = mirror_orbits(output_dim)
    self.register_buffer("orbit_index", index)
    k = int(index.max()) + 1
    if std_type == "scalar":
      del self.std_param
      self.std_param = nn.Parameter(init_std * torch.ones(k), requires_grad=learn_std)
    else:
      del self.log_std_param
      self.log_std_param = nn.Parameter(math.log(init_std) * torch.ones(k), requires_grad=learn_std)

  def tied_std(self) -> torch.Tensor:
    if self.std_type == "scalar":
      std = self.std_param.clamp(self.std_range[0], self.std_range[1])
    else:
      std = torch.exp(self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1]))
    return std[self.orbit_index]

  def update(self, mlp_output: torch.Tensor) -> None:
    self._distribution = Normal(mlp_output, self.tied_std())


class FixedGaussian(GaussianDistribution):
  """S0b-fixed (coordinator 07:5x): the policy std is a constant per-joint vector (numbers.FIXED_STD), held as a buffer,
  so it is in the actor's state dict ('distribution.std_param') but not among its parameters: the optimizer never sees
  it and the entropy term is a constant."""

  def __init__(self, output_dim: int, std_values: list | tuple = (), scale: float = 1.0,
               std_range: tuple[float, float] = STD_RANGE) -> None:
    super().__init__(output_dim, init_std=1.0, std_range=std_range, std_type="scalar", learn_std=False)
    values = torch.as_tensor(list(std_values), dtype=torch.float) * float(scale)
    if values.numel() != output_dim:
      raise ValueError(f"FixedGaussian needs {output_dim} std values, got {values.numel()}")
    del self.std_param
    self.register_buffer("std_param", values.clone())
    self.register_buffer("std_target", values.clone(), persistent=False)   # for the in-run 'still fixed' checks


def sigma_of(distribution) -> torch.Tensor:
  """The per-dimension std (24) of the actor's distribution, whatever its parametrisation."""
  if isinstance(distribution, MirrorTiedGaussian):
    return distribution.tied_std().detach()
  if getattr(distribution, "std_type", "scalar") == "log":
    return torch.exp(distribution.log_std_param.detach().clamp(*distribution.log_std_range))
  return distribution.std_param.detach().clamp(*distribution.std_range)


def sigma_from_state(state: dict) -> torch.Tensor | None:
  """The per-dimension std from an actor state dict (rsl_rl checkpoint 'actor_state_dict')."""
  if "distribution.std_param" in state:
    std = state["distribution.std_param"].float().clamp(*STD_RANGE)
  elif "distribution.log_std_param" in state:
    std = torch.exp(state["distribution.log_std_param"].float().clamp(math.log(STD_RANGE[0]), math.log(STD_RANGE[1])))
  else:
    return None
  if "distribution.orbit_index" in state:
    std = std[state["distribution.orbit_index"].long()]
  return std
