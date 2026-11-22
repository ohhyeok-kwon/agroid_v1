"""Left-right symmetric actor for the R1 Hold task (symmetric by construction, as r1wb.symmetric_model):

    output(o) = (mlp(o) + M_a mlp(M_o o)) / 2

Observation layouts (frame-major, 10 frames, oldest first, joints in POLICY order as the agroid deploy contract):
  810  frame = [gyro 3, gravity 3, twist command 3, q - q0 24, dq 24, previous action 24]   (deploy contract)
  570  frame = [gyro 3, gravity 3, twist command 3, q - q0 24, dq 24]                      (no previous action)
  330  frame = [gyro 3, gravity 3, twist command 3, q - q0 24]                              (R2 arm V: no joint velocity)
Foot-lead widths (R3 wave 1, coordinator 21:4x): 811 / 571 / 331 = one of the layouts above followed by ONE extra input,
the foot-lead constant b in {-1, +1}, appended at the END (index 810 / 570 / 330). Training draws b uniformly per episode
at reset (r1adv.positive.hold_w1.foot_lead); at deployment b is a constant chosen once (e.g. +1; the negative runner's
--foot-lead). b is a pseudo-scalar: the mirror of (o, b) is (M_o o, -b), so output(o, b) = (mlp(o, b) + M_a mlp(M_o o, -b)) / 2
stays exactly equivariant in the extended input space, and the first-layer weights from b start at zero (the initial
network ignores b). The same class serves every width; the width alone selects the layout.
The mirror maps are derived here from physics and from the joint NAMES (left <-> right, roll and yaw change sign;
r1wb.head_contract), not taken from any manufacturer table: angular velocity (x, y, z) -> (-x, y, -z), gravity
(x, y, z) -> (x, -y, z), twist (vx, vy, wz) -> (vx, -vy, -wz).
"""
from __future__ import annotations

import json

import torch

import rsl_rl.models.mlp_model as base
from rsl_rl.models import MLPModel

from r1lab import contract as LC
from r1wb import head_contract as HC

FRAMES = 10
BASE_WIDTHS = (810, 570, 330)
LEAD_WIDTHS = tuple(w + 1 for w in BASE_WIDTHS)        # base layout + the foot-lead input b as the last value


def has_lead(width: int) -> bool:
  """True for a foot-lead width (base contract + b)."""
  return width in LEAD_WIDTHS


def joint_mirror_policy_order() -> tuple[list[int], list[float]]:
  """(index, sign) in policy order: mirrored[p] = sign[p] * x[index[p]]."""
  ent_perm, ent_sign = HC.MIRROR_PERM[:24], HC.MIRROR_SIGN[:24]
  e_of_p = list(LC.POLICY_TO_ENTITY)                       # entity index of policy index p
  p_of_e = {e: p for p, e in enumerate(e_of_p)}
  index = [p_of_e[ent_perm[e_of_p[p]]] for p in range(24)]
  sign = [float(ent_sign[e_of_p[p]]) for p in range(24)]
  return index, sign


def frame_mirror(with_action: bool, with_velocity: bool = True) -> tuple[list[int], list[float]]:
  ji, js = joint_mirror_policy_order()
  index, sign = [0, 1, 2], [-1.0, 1.0, -1.0]                # gyro
  index += [3, 4, 5]; sign += [1.0, -1.0, 1.0]              # gravity
  index += [6, 7, 8]; sign += [1.0, -1.0, -1.0]             # twist command
  blocks = 1 + int(with_velocity) + int(with_action)
  for b in range(blocks):                                   # q - q0, (dq), (previous action)
    off = 9 + 24 * b
    index += [off + i for i in ji]; sign += js
  return index, sign


def observation_mirror(width: int) -> tuple[torch.Tensor, torch.Tensor]:
  """(index, sign) with mirrored[i] = sign[i] * x[index[i]]. Foot-lead widths: the base map, then b -> -b (last slot)."""
  lead = has_lead(width)
  base = width - 1 if lead else width
  if base not in BASE_WIDTHS:
    raise ValueError(f"no Hold observation layout has {width} values")
  fi, fs = frame_mirror(base == 810, with_velocity=base != 330)
  d = len(fi)
  index = [f * d + i for f in range(FRAMES) for i in fi]
  sign = fs * FRAMES
  if lead:
    index, sign = index + [base], sign + [-1.0]
  return torch.tensor(index, dtype=torch.long), torch.tensor(sign, dtype=torch.float)


def action_mirror() -> tuple[torch.Tensor, torch.Tensor]:
  ji, js = joint_mirror_policy_order()
  return torch.tensor(ji, dtype=torch.long), torch.tensor(js, dtype=torch.float)


class _OnnxHoldSymmetric(base._OnnxMLPModel):
  def __init__(self, model: MLPModel, verbose: bool) -> None:
    super().__init__(model, verbose)
    oi, os_ = observation_mirror(model.obs_dim)
    ai, as_ = action_mirror()
    for name, value in (("obs_index", oi), ("obs_sign", os_), ("act_index", ai), ("act_sign", as_)):
      self.register_buffer(name, value)

  def one(self, x: torch.Tensor) -> torch.Tensor:
    return self.deterministic_output(self.mlp(self.obs_normalizer(x)))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    flipped = torch.index_select(x, 1, self.obs_index) * self.obs_sign
    return 0.5 * (self.one(x) + torch.index_select(self.one(flipped), 1, self.act_index) * self.act_sign)


LAYOUTS = {810: "r1adv-hold-810-deploy-contract-v1: 10 frames oldest first x [gyro 3, gravity 3, twist command 3, "
                "q - q0 24, dq 24, previous action 24], joints in agroid POLICY order (r1lab.contract), r1lab frame_stack",
           330: "r1adv-hold-330-novel-v1: 10 frames oldest first x [gyro 3, gravity 3, twist command 3, q - q0 24], "
                "joints in agroid POLICY order, r1adv.positive.hold_r2.frame_stack_novel (R2 arm V)",
           570: "r1adv-hold-570-noprev-v1: 10 frames oldest first x [gyro 3, gravity 3, twist command 3, q - q0 24, dq 24], "
                "joints in agroid POLICY order, r1adv.positive.hold_task.frame_stack_noprev"}
for _w in BASE_WIDTHS:
  LAYOUTS[_w + 1] = (f"r1adv-hold-{_w + 1}-lead-v1: the {_w} layout ({LAYOUTS[_w]}) followed by ONE foot-lead input b at "
                     f"index {_w} (the last value): b in {{-1, +1}}, drawn uniformly per episode at reset in training "
                     "(r1adv.positive.hold_w1.foot_lead), a constant chosen once at deployment (e.g. +1); mirror (M o, -b)")


def contract(obs_dim: int) -> str:
  function = ("output = (mlp(o, b) + M_a mlp(M_o o, -b)) / 2, mirror maps of this module; b = the last input (foot lead, "
              "fixed at deployment)" if has_lead(obs_dim) else "output = (mlp(o) + M_a mlp(M_o o)) / 2, mirror maps of this module")
  return json.dumps({"model_class": "r1adv.positive.hold_model:HoldSymmetricMLP",
                     "function": function,
                     "observation": LAYOUTS[obs_dim], "action": "24 joint position actions in POLICY order "
                     "(r1lab.mdp.actions.PolicyJointPositionAction: default pose + action * contract scale)"}, sort_keys=True)


def read_contract(actor_state_dict: dict) -> dict | None:
  """The contract stored in a checkpoint's actor_state_dict (None for checkpoints without it)."""
  raw = actor_state_dict.get("r1adv_contract")
  return None if raw is None else json.loads(bytes(raw.cpu().tolist()).decode())


class HoldSymmetricMLP(MLPModel):
  symmetrized = True

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    oi, os_ = observation_mirror(self.obs_dim)
    ai, as_ = action_mirror()
    for name, value in (("obs_index", oi), ("obs_sign", os_), ("act_index", ai), ("act_sign", as_)):
      self.register_buffer("mirror_" + name, value, persistent=False)
    # Persistent: the checkpoint carries its model class and observation contract, and a plain MLP loader with strict
    # loading (e.g. r1wb.student.observations.load_teacher) refuses it instead of silently computing another function.
    self.register_buffer("r1adv_contract", torch.tensor(list(contract(self.obs_dim).encode()), dtype=torch.uint8))
    if has_lead(self.obs_dim):                              # foot-lead input b: the initial network ignores it
      with torch.no_grad():
        self.mlp[0].weight[:, -1].zero_()

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

  def as_onnx(self, verbose: bool = False) -> torch.nn.Module:
    return _OnnxHoldSymmetric(self, verbose)
