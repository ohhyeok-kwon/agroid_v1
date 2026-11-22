"""Observation terms of the System 1 student: goals, arm intent and the frozen teacher's action label."""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from mjlab.managers.observation_manager import ObservationTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def goal(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return env.command_manager.get_term(command_name).goal


def arm_intent(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return env.command_manager.get_term(command_name).arm_intent


def goal_lag(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return env.command_manager.get_term(command_name).goal_lag


def load_teacher(checkpoint: str, device: str):
  """The tracking teacher's actor (normaliser included), frozen."""
  from rsl_rl.models import MLPModel
  loaded = torch.load(checkpoint, map_location=device, weights_only=False)
  state = loaded["actor_state_dict"]
  layers = sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight"))
  hidden = [state[f"mlp.{i}.weight"].shape[0] for i in layers[:-1]]
  width, actions = state[f"mlp.{layers[0]}.weight"].shape[1], state[f"mlp.{layers[-1]}.weight"].shape[0]
  example = TensorDict({"actor": torch.zeros(1, width, device=device)}, batch_size=[1])
  # Without the output distribution the model returns the action mean, which is all a frozen teacher needs.
  model = MLPModel(example, {"actor": ["actor"]}, "actor", actions, hidden_dims=hidden, activation="elu",
                   obs_normalization=True, distribution_cfg=None)
  model.load_state_dict({key: value for key, value in state.items() if not key.startswith("distribution.")})
  model.to(device).eval()
  for parameter in model.parameters():
    parameter.requires_grad_(False)
  # A checkpoint of r1wb.symmetrize, or of a run whose policy is r1wb.symmetric_model: the policy is the network
  # averaged with its own mirror image.
  if loaded.get("symmetrized") or (loaded.get("infos") or {}).get("symmetrized"):
    from r1wb.symmetry import Symmetrized
    model = Symmetrized(model)
  return model


def load_policy(checkpoint: str, device: str, action_dim: int | None = None):
  """A checkpoint's actor as a function of the concatenated observation: the action mean (a network with a
  state-dependent noise head outputs the mean in its first `action_dim` values)."""
  model = load_teacher(checkpoint, device)
  action_dim = (26 if model.obs_dim == 417 else 24) if action_dim is None else action_dim

  def policy(observation: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
      output = model(TensorDict({"actor": observation}, batch_size=[observation.shape[0]]))
    return output.reshape(observation.shape[0], -1)[:, :action_dim]

  return policy


def no_label(env: ManagerBasedRlEnv) -> torch.Tensor:
  """A label group that says nothing (weight 0): for runs of the imitation learner without a teacher."""
  return torch.zeros(env.num_envs, env.action_manager.total_action_dim + 2, device=env.device)


class teacher_label:
  """[teacher action in the student's action space (24), weight (1), driven (1)].

  `driven` marks environments whose current episode is executed by this label instead of the student's own action
  (teacher-driven share of the command streams); the learning algorithm reads it.

  The teacher sees its own observation (reference, base velocity, joints, the action actually applied) without
  noise, and the reference re-anchored to where the robot is: height and tilt error only. The student cannot
  observe planar or heading drift, so a label that says "catch up with the reference" would be noise to it.
  The student never sees this group, only the imitation loss does.
  """

  def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRlEnv):
    self.model = load_teacher(cfg.params["checkpoint"], env.device)

  def __call__(self, env: ManagerBasedRlEnv, command_name: str, checkpoint: str, stream: str = "always",
               grace: float = 0.0) -> torch.Tensor:
    """stream / grace: where a matched command stream's label counts (see tracking.mdp.course_is_trusted)."""
    del checkpoint
    from r1wb.tracking.env_cfg import teacher_observation
    action_term = env.action_manager.get_term("joint_pos")
    # the policy's width says what it reads: the tracking teacher (273), the task teacher with the goal (294), or
    # that plus the senses of r1wb.senses
    observation = teacher_observation(env, command_name, action_term.plain_action, local=True, width=self.model.obs_dim)
    if observation.shape[1] != self.model.obs_dim:
      raise ValueError(f"Teacher expects {self.model.obs_dim} inputs, the environment provides {observation.shape[1]}")
    with torch.no_grad():
      action = self.model(TensorDict({"actor": observation}, batch_size=[observation.shape[0]]))
    label = action_term.from_plain_action(action)
    from r1wb.tracking.mdp import course_is_trusted
    weight = course_is_trusted(env, command_name, stream, grace).unsqueeze(1)
    driven = getattr(env.command_manager.get_term(command_name), "driven", None)
    driven = torch.zeros_like(weight) if driven is None else driven.float().unsqueeze(1)
    return torch.cat([label, weight, driven], dim=1)
