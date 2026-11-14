"""Reference command over a multi-clip library (extends mjlab's single-motion tracking command).

Differences from the single-motion command: failure-weighted sampling uses a failure rate per visited step
(counts alone feed back on themselves once a few clips cannot be tracked) mixed with a uniform share and capped,
an episode never crosses a clip boundary (the `clip_finished` time-out
ends it at the clip's last frame), resets start at least `min_remaining_frames` before a clip end, the command
also carries a short preview of the reference (future joint angles and anchor pose), and `pin_clips` (evaluation)
pins environment i to clip i and starts at its first frame.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat, quat_apply_inverse, quat_inv, quat_mul, yaw_quat

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class LibraryMotionCommand(MotionCommand):
  cfg: LibraryMotionCommandCfg

  def __init__(self, cfg: LibraryMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    table = np.load(cfg.motion_file, allow_pickle=False)
    start = torch.tensor(table["clip_start"], dtype=torch.long, device=self.device)
    length = torch.tensor(table["clip_length"], dtype=torch.long, device=self.device)
    assert int((start + length)[-1]) == self.motion.time_step_total
    self.clip_start, self.clip_end = start, start + length - 1
    self.frame_clip = torch.repeat_interleave(torch.arange(len(start), device=self.device), length)
    self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.clip_finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.future = torch.tensor(cfg.future_steps, dtype=torch.long, device=self.device)
    # Per-clip outcome counters for the feasibility report (reset by whoever reads them).
    self.clip_failures = torch.zeros(len(start), device=self.device)
    self.clip_completions = torch.zeros(len(start), device=self.device)
    # Failure-rate sampling state: smoothed visits and failures per one-second bin of the library timeline.
    self.bin_visits = torch.zeros(self.bin_count, device=self.device)
    self.bin_fails = torch.zeros(self.bin_count, device=self.device)
    self._step_fails = torch.zeros(self.bin_count, device=self.device)
    # Bins that may be sampled (subclasses restrict the clip set); a bin belongs to the clip of its first frame.
    self.bin_allowed = torch.ones(self.bin_count, device=self.device)

  # -- the reference on the surface the robot stands on ----------------------------------------------------
  # cfg.on_support: every height of the reference (anchor, bodies) is taken above the ground under the robot's feet
  # (r1wb.world.support_under) instead of above the start level; the same numbers on a flat floor. An episode starts
  # on the flat middle of its patch: while the robot is being placed the reference is the library's own height with
  # its anchor moved over the environment's origin (a clip's frames lie metres away from where the clip began, and
  # the ground there is not the start level). Nothing in these tasks reads the planar offset between the robot and
  # the reference (the anchor terms are local), so the reference keeps its own planar path afterwards.

  def _resample_command(self, env_ids: torch.Tensor):
    self._placing = True
    try:
      super()._resample_command(env_ids)
    finally:
      self._placing = False

  def _support(self) -> torch.Tensor | None:
    if not self.cfg.on_support or getattr(self, "_placing", False):
      return None
    from r1wb import world
    return world.support_under(self._env)

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    position = super().anchor_pos_w
    support = self._support()
    if support is not None:
      position[:, 2] += support
    return position

  @property
  def body_pos_w(self) -> torch.Tensor:
    position = super().body_pos_w
    if self.cfg.on_support and getattr(self, "_placing", False):
      position[..., :2] -= self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index, :2].unsqueeze(1)
      return position
    support = self._support()
    if support is not None:
      position[..., 2] += support.unsqueeze(1)
      if self.cfg.feet_on_ground:                 # each foot above the ground under that foot of the robot
        from r1wb import world
        if not hasattr(self, "_foot_bodies"):
          self._foot_bodies = [list(self.cfg.body_names).index(f"{side}_ankle_roll_link") for side in ("left", "right")]
        position[:, self._foot_bodies, 2] += world.feet_under(self._env) - support.unsqueeze(1)
    return position

  def allow_clips(self, allowed: torch.Tensor) -> None:
    """allowed: bool per clip. Resets are then drawn only from frames of allowed clips."""
    first = (torch.arange(self.bin_count, device=self.device) * (self.motion.time_step_total - 1) // self.bin_count)
    self.bin_allowed = allowed[self.frame_clip[first]].float()
    self.clip_allowed = allowed

  def _bins(self, steps: torch.Tensor) -> torch.Tensor:
    return torch.clamp(steps * self.bin_count // max(self.motion.time_step_total, 1), 0, self.bin_count - 1)

  # -- reference preview ------------------------------------------------------------------------------------

  @property
  def command(self) -> torch.Tensor:
    """Current reference joint state, then per preview step: joint angles, anchor displacement in the current
    reference anchor's yaw frame, and the relative anchor rotation (first two matrix columns)."""
    parts = [self.joint_pos, self.joint_vel]
    if len(self.future):
      anchor = self.motion_anchor_body_index
      now_pos = self.motion.body_pos_w[self.time_steps, anchor]
      now_quat = self.motion.body_quat_w[self.time_steps, anchor]
      heading = yaw_quat(now_quat)
      for step in self.future:
        index = torch.minimum(self.time_steps + step, self.clip_end[self.clip_ids])
        parts.append(self.motion.joint_pos[index])
        parts.append(quat_apply_inverse(heading, self.motion.body_pos_w[index, anchor] - now_pos))
        relative = matrix_from_quat(quat_mul(quat_inv(now_quat), self.motion.body_quat_w[index, anchor]))
        parts.append(relative[..., :2].reshape(self.num_envs, 6))
    return torch.cat(parts, dim=1)

  # -- sampling ---------------------------------------------------------------------------------------------

  def _place(self, env_ids: torch.Tensor) -> None:
    """Keep sampled frames inside one clip and far enough from its end."""
    clip = self.frame_clip[self.time_steps[env_ids]]
    allowed = getattr(self, "clip_allowed", None)
    if allowed is not None and not bool(allowed[clip].all()):
      # A bin can straddle two clips; frames that fell into a disallowed clip move to an allowed one.
      choices = allowed.nonzero(as_tuple=False).flatten()
      swap = ~allowed[clip]
      clip = torch.where(swap, choices[torch.randint(len(choices), (len(clip),), device=self.device)], clip)
      self.time_steps[env_ids] = torch.where(swap, self.clip_start[clip], self.time_steps[env_ids])
    latest = torch.maximum(self.clip_end[clip] - self.cfg.min_remaining_frames, self.clip_start[clip])
    steps = torch.minimum(self.time_steps[env_ids], latest)
    from_start = torch.rand(len(env_ids), device=self.device) < self.cfg.clip_start_fraction
    self.time_steps[env_ids] = torch.where(from_start, self.clip_start[clip], steps)
    self.clip_ids[env_ids] = clip
    self.clip_finished[env_ids] = False

  def _count_outcomes(self, env_ids: torch.Tensor) -> None:
    if not self.clip_start.numel():
      return
    failed = self._env.termination_manager.terminated[env_ids]
    clips = self.clip_ids[env_ids]
    self.clip_failures.scatter_add_(0, clips[failed], torch.ones_like(clips[failed], dtype=torch.float))
    done = self.clip_finished[env_ids] & ~failed
    self.clip_completions.scatter_add_(0, clips[done], torch.ones_like(clips[done], dtype=torch.float))

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    self._count_outcomes(env_ids)
    failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(failed):
      self._step_fails += torch.bincount(self._bins(self.time_steps[env_ids][failed]), minlength=self.bin_count)
    uniform = 1.0 / self.bin_count
    hazard = self.bin_fails / (self.bin_visits + 1e-3)
    total = hazard.sum()
    weight = hazard / total if total > 0 else torch.full_like(hazard, uniform)
    share = self.cfg.adaptive_uniform_share
    probability = (share * uniform + (1.0 - share) * weight).clamp(max=self.cfg.adaptive_cap * uniform)
    probability = probability * self.bin_allowed
    probability = probability / probability.sum()
    bins = torch.multinomial(probability, len(env_ids), replacement=True)
    offset = torch.rand(len(env_ids), device=self.device)
    self.time_steps[env_ids] = ((bins + offset) / self.bin_count * (self.motion.time_step_total - 1)).long()
    entropy = -(probability * (probability + 1e-12).log()).sum() / math.log(self.bin_count)
    top, where = probability.max(dim=0)
    self.metrics["sampling_entropy"][:] = entropy
    self.metrics["sampling_top1_prob"][:] = top
    self.metrics["sampling_top1_bin"][:] = where.float() / self.bin_count
    self._place(env_ids)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self._count_outcomes(env_ids)
    if self.cfg.pin_clips:
      # Evaluation: environment i replays clip (first_clip + i) from its first frame.
      if self.cfg.clip_list:
        table = torch.tensor(self.cfg.clip_list, dtype=torch.long, device=self.device)
        clip = table[env_ids % len(table)]
      else:
        clip = (self.cfg.first_clip + env_ids) % len(self.clip_start)
      self.time_steps[env_ids] = self.clip_start[clip]
      self.clip_ids[env_ids] = clip
      self.clip_finished[env_ids] = False
      return
    super()._uniform_sampling(env_ids)
    self._place(env_ids)

  # -- stepping ---------------------------------------------------------------------------------------------

  def _update_command(self, env_ids: torch.Tensor | None = None):
    hold = getattr(self, "hold", None)      # per-environment: the reference pauses on its current frame
    if self.cfg.frozen:
      pass                                  # goal-stream mode: the reference stays on the frame it was reset to
    elif env_ids is None:
      self.time_steps += 1 if hold is None else (~hold).long()
    else:
      self.time_steps[env_ids] += 1 if hold is None else (~hold[env_ids]).long()
    end = self.clip_end[self.clip_ids]
    self.clip_finished |= self.time_steps > end
    self.time_steps = torch.minimum(self.time_steps, end)
    if self._pending_forward:
      self._pending_forward = False
      self._env.sim.forward()
    self.update_relative_body_poses()
    if env_ids is None and self.cfg.sampling_mode == "adaptive":
      alpha = self.cfg.adaptive_alpha
      visits = torch.bincount(self._bins(self.time_steps), minlength=self.bin_count).float()
      self.bin_visits = (1 - alpha) * self.bin_visits + alpha * visits
      self.bin_fails = (1 - alpha) * self.bin_fails + alpha * self._step_fails
      self._step_fails.zero_()


@dataclass(kw_only=True)
class LibraryMotionCommandCfg(MotionCommandCfg):
  adaptive_uniform_share: float = 0.3   # part of the resets drawn uniformly over the library timeline
  adaptive_cap: float = 5.0             # no bin is drawn more often than this multiple of uniform
  min_remaining_frames: int = 50
  future_steps: tuple[int, ...] = (5, 10, 25, 50)   # 0.1, 0.2, 0.5 and 1.0 s ahead
  clip_start_fraction: float = 0.2                  # share of resets that begin at a clip's first frame
  pin_clips: bool = False        # with sampling_mode="uniform": environment i <-> clip first_clip + i, from frame 0
  first_clip: int = 0
  clip_list: tuple[int, ...] = ()   # with pin_clips: environment i replays clip_list[i % len] instead
  frozen: bool = False              # the reference does not advance (used when goals come from a command stream)
  on_support: bool = False          # heights of the reference above the ground under the robot's feet (r1wb.world)
  feet_on_ground: bool = False      # with on_support: each foot of the reference above the ground under that foot

  def build(self, env: ManagerBasedRlEnv) -> LibraryMotionCommand:
    return LibraryMotionCommand(self, env)


def clip_finished(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Time-out: the reference clip has reached its last frame."""
  return env.command_manager.get_term(command_name).clip_finished
