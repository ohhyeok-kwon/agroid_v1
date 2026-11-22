"""Plain (non-symmetric) actor for the R4 wave-2 cell XB (coordinator 2026-11-18 01:4x KST, rounds/R04/04_r4_state.md
section 3, X-b): the C0 actor with the mirror average removed, i.e. exactly rsl_rl's MLPModel (same layers, same
initialisation, same normaliser) plus two persistent buffers that name the format:

  r1adv_contract  the observation / function contract (JSON bytes), model_class r1adv.positive.hold_plain:HoldPlainMLP,
                  function 'output = mlp(o)' (no mirror symmetry)
  r1adv_plain     one value (1): a format marker. A loader that builds HoldSymmetricMLP with strict loading (for example the
                  negative runner's load_deploy as of 2026-11-18) refuses an XB checkpoint ('Unexpected key(s) ...
                  r1adv_plain', and the contract size differs) instead of silently evaluating (mlp(o) + M mlp(M o)) / 2,
                  which is not the trained policy.

load_actor(state_dict) builds the right class (HoldPlainMLP or HoldSymmetricMLP) from a checkpoint's actor_state_dict:
the dispatch the negative runner needs for XB (see rounds/R04/positive/ for the proposed patch). Observation layouts are
those of hold_model (570 for XB: 10 frames x [gyro 3, gravity 3, twist command 3, q - q0 24, dq 24], joints in POLICY
order). Only rsl_rl, torch and hold_model are imported, so the module can be copied into an evaluation code copy alone.
"""
from __future__ import annotations

import json

import torch
from rsl_rl.models import MLPModel

from r1adv.positive import hold_model as HM

MODEL_CLASS = "r1adv.positive.hold_plain:HoldPlainMLP"


def contract(obs_dim: int) -> str:
  if obs_dim not in HM.LAYOUTS or HM.has_lead(obs_dim):
    raise ValueError(f"no plain Hold layout has {obs_dim} values (XB has no foot-lead input)")
  return json.dumps({"model_class": MODEL_CLASS,
                     "function": "output = mlp(o) (plain MLP, no mirror symmetry; R4 wave-2 cell XB)",
                     "observation": HM.LAYOUTS[obs_dim], "action": "24 joint position actions in POLICY order "
                     "(r1lab.mdp.actions.PolicyJointPositionAction: default pose + action * contract scale)"}, sort_keys=True)


class HoldPlainMLP(MLPModel):
  """rsl_rl MLPModel (forward, initialisation and normaliser unchanged) carrying the plain-format buffers."""

  symmetrized = False

  def __init__(self, *args, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.register_buffer("r1adv_contract", torch.tensor(list(contract(self.obs_dim).encode()), dtype=torch.uint8))
    self.register_buffer("r1adv_plain", torch.ones(1, dtype=torch.uint8))


def is_plain(actor_state_dict: dict) -> bool:
  """True for an XB (HoldPlainMLP) checkpoint: the format marker, or a contract naming this class."""
  if "r1adv_plain" in actor_state_dict:
    return True
  c = HM.read_contract(actor_state_dict)
  return bool(c and c.get("model_class") == MODEL_CLASS)


def load_actor(actor_state_dict: dict, device: str = "cpu"):
  """(model, width): the checkpoint's own actor class (HoldPlainMLP for XB, else HoldSymmetricMLP), built with the
  checkpoint's layer sizes, strict loading of every non-distribution weight, eval mode, no gradients."""
  from tensordict import TensorDict
  state = actor_state_dict
  layers = sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight"))
  hidden = [state[f"mlp.{i}.weight"].shape[0] for i in layers[:-1]]
  width, actions = state[f"mlp.{layers[0]}.weight"].shape[1], state[f"mlp.{layers[-1]}.weight"].shape[0]
  example = TensorDict({"actor": torch.zeros(1, width, device=device)}, batch_size=[1])
  cls = HoldPlainMLP if is_plain(state) else HM.HoldSymmetricMLP
  model = cls(example, {"actor": ["actor"]}, "actor", actions, hidden_dims=hidden, activation="elu",
              obs_normalization=True, distribution_cfg=None)
  weights = {k: v for k, v in state.items() if not k.startswith("distribution.")}
  if cls is HM.HoldSymmetricMLP and "r1adv_contract" not in weights:     # checkpoints saved before the contract buffer
    weights["r1adv_contract"] = model.state_dict()["r1adv_contract"]
  model.load_state_dict(weights)
  model.to(device).eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  return model, width
