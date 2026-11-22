"""The S1 actor: left-right symmetric by construction, output(o) = (mlp(o) + M mlp(M o)) / 2 (as r1wb.symmetric_model),
with the mirror map of the S1 actor observation:

  proprio history  HISTORY frames (oldest first) x [gyro (axial), gravity (polar), q - q0 (joints), q_dot (joints)]
  last_target      HISTORY frames x joints
  goal             planar error (x, -y), sin(-), cos(+), height error, mask (3), time left

Joint map, signs and vector rules are r1wb.symmetry's (verified on the model by forward kinematics); the in-simulator
check of the whole observation map is r1s1.gates `mirror`.
"""
from __future__ import annotations

import torch

import rsl_rl.models.mlp_model as base
from rsl_rl.models import MLPModel

from r1s1 import numbers as N
from r1wb.symmetry import AXIAL, JOINT_PERM, JOINT_SIGN, POLAR

FRAME = 54
GOAL = 9
WIDTH = N.HISTORY * (FRAME + 24) + GOAL


def actor_mirror(device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
  index: list[int] = []
  sign: list[float] = []

  def add(local, signs):
    start = len(index)
    index.extend(start + i for i in local)
    sign.extend(signs)

  for _ in range(N.HISTORY):
    add(range(3), AXIAL)
    add(range(3), POLAR)
    add(JOINT_PERM, JOINT_SIGN)
    add(JOINT_PERM, JOINT_SIGN)
  for _ in range(N.HISTORY):
    add(JOINT_PERM, JOINT_SIGN)
  add(range(GOAL), (1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0))
  assert sorted(index) == list(range(len(index))) and len(index) == WIDTH
  return torch.tensor(index, dtype=torch.long, device=device), torch.tensor(sign, device=device)


def action_mirror(device="cpu") -> tuple[torch.Tensor, torch.Tensor]:
  return torch.tensor(JOINT_PERM, dtype=torch.long, device=device), torch.tensor(JOINT_SIGN, device=device)


class _OnnxS1Symmetric(base._OnnxMLPModel):
  def __init__(self, model: MLPModel, verbose: bool) -> None:
    super().__init__(model, verbose)
    obs_index, obs_sign = actor_mirror()
    act_index, act_sign = action_mirror()
    for name, value in (("obs_index", obs_index), ("obs_sign", obs_sign), ("act_index", act_index),
                        ("act_sign", act_sign)):
      self.register_buffer(name, value)

  def one(self, x: torch.Tensor) -> torch.Tensor:
    return self.deterministic_output(self.mlp(self.obs_normalizer(x)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    flipped = torch.index_select(x, 1, self.obs_index) * self.obs_sign
    return 0.5 * (self.one(x) + torch.index_select(self.one(flipped), 1, self.act_index) * self.act_sign)


class S1SymmetricMLPModel(MLPModel):
  symmetrized = True

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    if self.obs_dim != WIDTH:
      raise ValueError(f"S1 actor observation must have {WIDTH} values, got {self.obs_dim}")
    obs_index, obs_sign = actor_mirror()
    act_index, act_sign = action_mirror()
    for name, value in (("obs_index", obs_index), ("obs_sign", obs_sign), ("act_index", act_index),
                        ("act_sign", act_sign)):
      self.register_buffer("mirror_" + name, value, persistent=False)

  def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False) -> torch.Tensor:
    obs = base.unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    raw = torch.cat([obs[group] for group in self.obs_groups], dim=-1)
    plain = self.mlp(self.obs_normalizer(raw))
    other = self.mlp(self.obs_normalizer(raw[..., self.mirror_obs_index] * self.mirror_obs_sign))
    output = 0.5 * (plain + other[..., self.mirror_act_index] * self.mirror_act_sign)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(output)
    return output

  def as_onnx(self, verbose: bool):
    return _OnnxS1Symmetric(self, verbose)
