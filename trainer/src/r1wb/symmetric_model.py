"""A policy network that commutes with the left-right mirror by construction.

  output(o) = (mlp(o) + mirror(mlp(mirror(o)))) / 2          (r1wb.symmetry holds the maps)

The weights are an ordinary MLP's (a checkpoint of the plain model loads as it is and then IS its own symmetric
average); PPO trains this function directly, so the policy stays exactly symmetric at every update and both sides of
the body learn from every sample. Measured 11-13 15:20 without any training: E99 averaged with its mirror image earns
23% more reward in its own scene and falls half as often, while a mirror LOSS on the plain network destroyed the
policy on the way to symmetry. Exported policies contain the whole function (two passes).
"""
from __future__ import annotations

import torch

import rsl_rl.models.mlp_model as base
from rsl_rl.models import MLPModel

from r1wb.symmetry import ActorMirror


class _OnnxSymmetric(base._OnnxMLPModel):
  def __init__(self, model: MLPModel, verbose: bool) -> None:
    super().__init__(model, verbose)
    mirror = ActorMirror(model.obs_dim, "cpu")
    for name in ("obs_index", "obs_sign", "act_index", "act_sign"):
      self.register_buffer(name, getattr(mirror, name))

  def one(self, x: torch.Tensor) -> torch.Tensor:
    return self.deterministic_output(self.mlp(self.obs_normalizer(x)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    flipped = torch.index_select(x, 1, self.obs_index) * self.obs_sign
    return 0.5 * (self.one(x) + torch.index_select(self.one(flipped), 1, self.act_index) * self.act_sign)


class SymmetricMLPModel(MLPModel):
  symmetrized = True

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    # Buffers made here (not on first use): rollouts run in inference mode, and tensors created there cannot take
    # part in the update's backward pass. Not persistent: the checkpoint keeps the plain network's keys.
    mirror = ActorMirror(self.obs_dim, "cpu")
    for name in ("obs_index", "obs_sign", "act_index", "act_sign"):
      self.register_buffer("mirror_" + name, getattr(mirror, name), persistent=False)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    obs = base.unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    plain = self.mlp(self.obs_normalizer(raw))
    other = self.mlp(self.obs_normalizer(raw[..., self.mirror_obs_index] * self.mirror_obs_sign))
    if plain.shape[-1] != self.mirror_act_index.numel():
      raise ValueError("SymmetricMLPModel mirrors a mean per joint; a state-dependent noise head is not supported")
    output = 0.5 * (plain + other[..., self.mirror_act_index] * self.mirror_act_sign)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(output)
    return output

  def as_onnx(self, verbose: bool):
    return _OnnxSymmetric(self, verbose)
