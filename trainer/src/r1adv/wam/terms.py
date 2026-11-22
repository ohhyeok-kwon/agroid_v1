"""Action and reward terms of the W1-J tracker bracket (E1). Each term measures the robot against the reference only.

ReferenceResidualJointPositionAction (arm B): joint position target = reference joint angles of the next control step
+ scale * action. With this parameterization the stock action-rate price ||a_t - a_{t-1}||^2 charges only the change
of the target beyond the reference's own change (the excess), not the reference's motion itself.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg


class ReferenceResidualJointPositionAction(JointPositionAction):
  cfg: "ReferenceResidualJointPositionActionCfg"

  def process_actions(self, actions: torch.Tensor):
    term = self._env.command_manager.get_term(self.cfg.command_name)
    nxt = torch.minimum(term.time_steps + 1, term.clip_end[term.clip_ids])
    # library joint order == robot.JOINTS == entity order (r1wb.tracking); keep the action's target subset
    self._offset = term.motion.joint_pos[nxt][:, self._target_ids]
    super().process_actions(actions)


@dataclass(kw_only=True)
class ReferenceResidualJointPositionActionCfg(JointPositionActionCfg):
  command_name: str = "motion"
  use_default_offset: bool = False

  def build(self, env):
    return ReferenceResidualJointPositionAction(self, env)


def joint_pos_error_exp(env, command_name: str, std: float) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  err = term.joint_pos - env.scene["robot"].data.joint_pos
  return torch.exp(-(err ** 2).mean(dim=1) / std ** 2)


def joint_vel_error_exp(env, command_name: str, std: float) -> torch.Tensor:
  term = env.command_manager.get_term(command_name)
  err = term.joint_vel - env.scene["robot"].data.joint_vel
  return torch.exp(-(err ** 2).mean(dim=1) / std ** 2)


def anti_shake_ang_vel(env, body_names: tuple[str, ...], threshold: float) -> torch.Tensor:
  """SONIC's 'anti-shake': sum over bodies of ||omega||^2 where ||omega|| > threshold (rad/s)."""
  robot = env.scene["robot"]
  ids = getattr(env, "_wam_shake_ids", None)
  if ids is None:
    ids = torch.tensor([robot.body_names.index(b) for b in body_names], device=env.device)
    env._wam_shake_ids = ids
  w = robot.data.body_link_ang_vel_w[:, ids]
  n2 = (w ** 2).sum(-1)
  return (n2 * (n2 > threshold ** 2)).sum(-1)
