"""System 1 action: legs and waist as joint position targets around the default pose; arms as a bounded
correction around the arm intent (reference arm angles in training, IK / rest-pose output at run time)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from r1wb import robot

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class IntentResidualAction(JointPositionAction):
  cfg: IntentResidualActionCfg

  def __init__(self, cfg: IntentResidualActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    names = list(self._target_names)
    assert tuple(names) == tuple(robot.JOINTS), names
    self.arm = torch.tensor([names.index(n) for n in robot.ARM_JOINTS], device=self.device)
    self._command = env.command_manager.get_term(cfg.command_name)

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    target = actions * self._scale + self._offset
    target[:, self.arm] = (self._command.arm_intent_now
                           + self.cfg.residual_limit * actions[:, self.arm].clamp(-1.0, 1.0))
    self._processed_actions = target

  @property
  def arm_residual(self) -> torch.Tensor:
    """Applied arm correction as a fraction of its limit, in [-1, 1]."""
    return self._raw_actions[:, self.arm].clamp(-1.0, 1.0)

  @property
  def plain_action(self) -> torch.Tensor:
    """The applied targets expressed as a plain joint-position action (what the tracking teacher outputs)."""
    return (self._processed_actions - self._offset) / self._scale

  def from_plain_action(self, action: torch.Tensor) -> torch.Tensor:
    """A plain joint-position action expressed in this term's action space (arm part clipped to the limit)."""
    label = action.clone()
    target = action[:, self.arm] * self._scale[:, self.arm] + self._offset[:, self.arm]
    label[:, self.arm] = ((target - self._command.arm_intent_now) / self.cfg.residual_limit).clamp(-1.0, 1.0)
    return label


@dataclass(kw_only=True)
class IntentResidualActionCfg(JointPositionActionCfg):
  command_name: str = "motion"
  residual_limit: float = 0.5      # rad; the command itself is not priced (it has to cover gravity and payload sag)

  def build(self, env: ManagerBasedRlEnv) -> IntentResidualAction:
    return IntentResidualAction(self, env)
