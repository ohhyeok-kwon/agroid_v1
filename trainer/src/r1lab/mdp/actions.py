"""Joint position action in agroid POLICY (IsaacLab BFS) order."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

from r1lab import contract as C

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class PolicyJointPositionActionCfg(JointPositionActionCfg):
  """target(entity j) = DEFAULT_POS[p] + a[p] * ACTION_SCALE[p], p = ACTION_IDX[j]."""

  actuator_names: tuple[str, ...] = (".*",)
  use_default_offset: bool = False

  def build(self, env: ManagerBasedRlEnv) -> PolicyJointPositionAction:
    return PolicyJointPositionAction(self, env)


class PolicyJointPositionAction(JointPositionAction):
  def __init__(self, cfg: PolicyJointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    assert tuple(self._target_names) == C.ENTITY_JOINTS, self._target_names
    idx = torch.tensor(C.ACTION_IDX, device=self.device, dtype=torch.long)
    self._perm = idx  # entity j <- policy ACTION_IDX[j]
    self._scale_e = torch.tensor(C.ACTION_SCALE, device=self.device)[idx]
    self._offset_e = torch.tensor(C.DEFAULT_POS, device=self.device)[idx]

  def process_actions(self, actions: torch.Tensor):
    # raw actions stay in policy order (they are fed back as "previous action").
    self._raw_actions[:] = actions
    self._processed_actions = actions[:, self._perm] * self._scale_e + self._offset_e
