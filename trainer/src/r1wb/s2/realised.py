"""The robot's own motion in the form of System 2's plans.

A plan is a run of frames (r1wb.s2.frames: joint angles, the anchor body's height and tilt, its step to the next
frame). The same 30 numbers can be read off what the robot did. Measured 11-13 (gates): the course generator's plan
is not a motion - every 0.64 s its legs jump by 0.5 rad on average and the ground force it implies leaves the soles -
while what the robot does under it is continuous and obeys physics. `frames_of` turns recorded robot motion into
frames; `replay_brain` plays such a recording back as motion codes, for the question "does System 1 do again what it
did, when that is what it is asked for".
"""
from __future__ import annotations

import torch

from r1wb.s2 import frames as s2
from r1wb.s2.small import AHEAD, GRID_ORIGIN


def frames_of(joints: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
  """joints (steps, envs, 24) in entity order; anchor (steps, envs, 6): x, y, heading, height, roll, pitch of the
  anchor body -> frames (envs, steps, 30). The last frame's step is zero."""
  position, heading = anchor[..., :2], anchor[..., 2]
  move = position[1:] - position[:-1]
  cos, sin = torch.cos(heading[:-1]), torch.sin(heading[:-1])
  forward, left = cos * move[..., 0] + sin * move[..., 1], -sin * move[..., 0] + cos * move[..., 1]
  turn = heading[1:] - heading[:-1]
  turn = torch.atan2(torch.sin(turn), torch.cos(turn))
  step = torch.cat([torch.stack([forward, left, turn], dim=-1), torch.zeros_like(anchor[:1, :, :3])], dim=0)
  frames = torch.cat([joints, anchor[..., 3:6], step], dim=-1).float()
  assert frames.shape[-1] == s2.FRAME_DIM
  return frames.permute(1, 0, 2).contiguous()


def smoothed(frames: torch.Tensor, window: int) -> torch.Tensor:
  """frames (envs, steps, 30), centred moving average over `window` steps along time (odd; 1 = unchanged). A recording
  at 50 Hz carries the landing impacts as steps in the joint angles; mocap-derived references do not."""
  if window <= 1:
    return frames
  pad = window // 2
  rows = frames.permute(0, 2, 1)
  rows = torch.nn.functional.pad(rows, (pad, pad), mode="replicate")
  return torch.nn.functional.avg_pool1d(rows, window, stride=1).permute(0, 2, 1).contiguous()


def codes_of(frames: torch.Tensor, tokenizer) -> torch.Tensor:
  """frames (envs, steps, 30) -> codes (envs, n, dims) with values 0 .. levels-1; code k = steps GRID_ORIGIN + stride k ..."""
  length = ((frames.shape[1] - GRID_ORIGIN) // tokenizer.stride) * tokenizer.stride
  with torch.no_grad():
    codes = torch.cat([tokenizer.encode(part[:, GRID_ORIGIN:GRID_ORIGIN + length]) for part in frames.split(256)])
  return codes + tokenizer.levels // 2


def replay_brain(frames: torch.Tensor, tokenizer, state_goal):
  """A System 2 that writes, at decision m, the codes of the recording from code 2m on (the code that starts playing
  at that decision); the goal it states is `state_goal(decision)`."""
  codes = codes_of(frames, tokenizer)

  def brain(decision, history):
    first = 2 * decision
    index = (first + torch.arange(AHEAD, device=codes.device)).clamp(max=codes.shape[1] - 1)
    return codes[:, index], state_goal(decision)
  return brain


def raw_replay_view(env, term, frames: torch.Tensor, tokenizer, state_goal):
  """As replay_brain, without motion codes: the recorded frames themselves are the plan (what a perfect tokenizer
  would give). Separates "System 1 does not do it again" from "the codes cannot say what the robot did"."""
  from r1wb.s2.runtime import PAST, S2View

  class RawReplay(S2View):
    def _decide(self) -> None:
      stride = self.tokenizer.stride
      first = GRID_ORIGIN + self._period * self._decision - PAST * stride
      index = (first + torch.arange(self._frames.shape[1], device=frames.device)).clamp(0, frames.shape[1] - 1)
      run = frames[:, index].clone()
      stays = torch.cat([index[1:] == index[:-1], torch.ones(1, dtype=torch.bool, device=frames.device)])
      run[:, :, s2.STEP] = run[:, :, s2.STEP] * (~stays).float().view(1, -1, 1)
      self._frames = run
      self._base["born"][:] = self._calls
      self._reference = None
      goal = state_goal(self._decision)
      self._target = self._waiting
      self._waiting = torch.cat([goal[:, :4], torch.zeros_like(goal[:, :1])], dim=1)
      self._waiting_mask = (goal[:, 4] > 0.5).float()
      self._decision += 1

  return RawReplay(env, term, None, tokenizer)
