"""Sparse goals derived from the reference motion (what System 2 or skill code supplies at run time).

Per library frame f, computed once at start-up:
  twist   mean root velocity over the next `goal_window` frames, in the root's yaw frame at f (vx, vy, yaw rate)
  height  mean root height over the same window
  pitch   mean root pitch over the same window
  hands   wrist positions `hand_lead` frames ahead, relative to the root in its yaw frame at f, with a time-to-go;
          active only in clips whose hands have a task (pick-up), otherwise zero with the flag off
Arm intent is NOT read from the reference: it comes from r1wb.student.intent (the robot's own hip angles and the
commanded speed), the same code that runs on the robot.
Height and pitch are switched off at random per episode so the policy also works when only part of the goal is
given. "Off" means "default": it is drawn only in clips whose reference stays at the default height (pitch)
from start to end, so the teacher's label and the goal never disagree.
Mirrored copies of clips whose goals are left-right symmetric (straight walking, standing) are not sampled: they
would give the student identical goals with mirrored teacher actions.

`goal_lag` is the student's replacement for the planar and heading error of a tracker: a leaky integral of
(own twist in the own heading frame - goal twist). The policy cannot observe where the reference is in the world,
but it can observe whether it moves as commanded, so progress is scored and episodes are ended on this quantity.
The goal describes the mean over the coming `goal_window` frames, so the present twist is compared with the goal of
half a window ago (on the references themselves this keeps the lag below 0.07 m and 0.21 rad; without the delay
fast turns read 1.1 rad while tracking perfectly).
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import TYPE_CHECKING

import numpy as np
import torch

from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from r1wb import robot
from r1wb.student.intent import ArmIntent
from r1wb.tracking.command import LibraryMotionCommand, LibraryMotionCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

GOAL_DIM = 18
INTENT_DIM = 10


class GoalMotionCommand(LibraryMotionCommand):
  cfg: GoalMotionCommandCfg

  def __init__(self, cfg: GoalMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    names = list(cfg.body_names)
    self._root = names.index("pelvis")
    self._wrists = [names.index("left_wrist_roll_link"), names.index("right_wrist_roll_link")]
    joints = list(self.robot.joint_names)
    self.arm_ids = torch.tensor([joints.index(n) for n in robot.ARM_JOINTS], device=self.device)
    manifest = json.loads(str(np.load(cfg.motion_file, allow_pickle=False)["manifest"]))
    self._build_tables(manifest)
    self.goal_mask = torch.ones(self.num_envs, 3, device=self.device)
    self.goal_lag = torch.zeros(self.num_envs, 3, device=self.device)
    self.twist_mean = torch.zeros(self.num_envs, 3, device=self.device)     # own twist, 0.4 s running mean
    self.driven = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)    # episodes run by the label
    self._goal_history = torch.zeros(self.num_envs, cfg.goal_window // 2 + 1, 3, device=self.device)
    self.metrics["goal_lag_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_lag_yaw"] = torch.zeros(self.num_envs, device=self.device)
    self.intent = ArmIntent(self.num_envs, self.device, env.step_dt)
    self._intent_now = self.intent.rest.unsqueeze(0).repeat(self.num_envs, 1)
    # Symmetric goal streams keep only the original clip; asymmetric ones (sideways, turning) keep their mirror.
    lateral = torch.zeros(len(self.clip_start), device=self.device)
    turning = torch.zeros(len(self.clip_start), device=self.device)
    lateral.scatter_add_(0, self.frame_clip, self.table_twist[:, 1])
    turning.scatter_add_(0, self.frame_clip, self.table_twist[:, 2])
    length = (self.clip_end - self.clip_start + 1).float()
    asymmetric = ((lateral / length).abs() > 0.05) | ((turning / length).abs() * length * env.step_dt > 0.5)
    mirrored = torch.tensor([clip["mirrored"] for clip in manifest["clips"]], dtype=torch.bool, device=self.device)
    # Standing clips in which the feet are re-placed (idle fidgeting, a step into a pose) have a zero twist goal
    # and a stepping label: they would teach steps nobody asked for. Only standing clips whose feet stay put are
    # sampled, and more often than their share of the library (standing is the most common commanded state).
    feet = [names.index(name) for name in robot.FOOT_BODIES]
    standing = torch.tensor([clip["context"] == "stand" for clip in manifest["clips"]], dtype=torch.bool, device=self.device)
    foot_range = torch.zeros(len(self.clip_start), device=self.device)
    for index in standing.nonzero(as_tuple=False).flatten().tolist():
      planar = self.motion.body_pos_w[self.clip_start[index]:self.clip_end[index] + 1][:, feet, :2]
      foot_range[index] = torch.norm(planar.amax(dim=0) - planar.amin(dim=0), dim=-1).max()
    self.clip_quiet_stand = standing & (foot_range < cfg.quiet_stand_foot_range)
    self.clip_trained = (asymmetric | ~mirrored) & (~standing | self.clip_quiet_stand)     # evaluation reports it
    if not cfg.pin_clips:
      self._allow_trained()

  def _allow_trained(self) -> None:
    """Sample resets from the clips the student trains on, quiet standing frames `stand_boost` times as often."""
    self.allow_clips(self.clip_trained)
    first = (torch.arange(self.bin_count, device=self.device) * (self.motion.time_step_total - 1) // self.bin_count)
    self.bin_allowed = self.bin_allowed * torch.where(self.clip_quiet_stand[self.frame_clip[first]],
                                                      self.cfg.stand_boost, 1.0)

  def _build_tables(self, manifest: dict) -> None:
    m, total = self.motion, self.motion.time_step_total
    frames = torch.arange(total, device=self.device)
    end = self.clip_end[self.frame_clip]
    root_pos, root_quat = m.body_pos_w[:, self._root], m.body_quat_w[:, self._root]
    heading = yaw_quat(root_quat)
    linear, angular = m.body_lin_vel_w[:, self._root], m.body_ang_vel_w[:, self._root]
    w, x, y, z = root_quat.unbind(-1)
    pitch = torch.asin((2.0 * (w * y - z * x)).clamp(-1.0, 1.0))
    twist = torch.zeros(total, 3, device=self.device)
    height = torch.zeros(total, device=self.device)
    lean = torch.zeros(total, device=self.device)
    window = self.cfg.goal_window
    for step in range(window + 1):
      index = torch.minimum(frames + step, end)
      twist[:, :2] += quat_apply_inverse(heading, linear[index])[:, :2]
      twist[:, 2] += angular[index, 2]
      height += root_pos[index, 2]
      lean += pitch[index]
    self.table_twist, self.table_height, self.table_pitch = (v / (window + 1) for v in (twist, height, lean))
    ahead = torch.minimum(frames + self.cfg.hand_lead, end)
    self.table_hands = torch.stack(
      [quat_apply_inverse(heading, m.body_pos_w[ahead, wrist] - root_pos) for wrist in self._wrists], dim=1)
    active = torch.tensor([clip["context"] in self.cfg.hand_goal_contexts for clip in manifest["clips"]],
                          dtype=torch.float, device=self.device)
    self.table_hand_active = active[self.frame_clip]
    # Clips in which "goal off" is the truth: the reference never leaves the default height (pitch).
    clips = len(self.clip_start)
    off_height = torch.zeros(clips, device=self.device).scatter_reduce_(
      0, self.frame_clip, (root_pos[:, 2] - self.cfg.default_height).abs(), reduce="amax")
    off_pitch = torch.zeros(clips, device=self.device).scatter_reduce_(0, self.frame_clip, pitch.abs(), reduce="amax")
    self.clip_height_free = off_height < self.cfg.height_free_band
    self.clip_pitch_free = off_pitch < self.cfg.pitch_free_band

  def _place(self, env_ids: torch.Tensor) -> None:
    super()._place(env_ids)
    count = len(env_ids)
    clip = self.clip_ids[env_ids]
    drop_height = (torch.rand(count, device=self.device) > self.cfg.height_goal_prob) & self.clip_height_free[clip]
    drop_pitch = (torch.rand(count, device=self.device) > self.cfg.pitch_goal_prob) & self.clip_pitch_free[clip]
    self.goal_mask[env_ids, 0] = 1.0
    self.goal_mask[env_ids, 1] = (~drop_height).float()
    self.goal_mask[env_ids, 2] = (~drop_pitch).float()

  def twist_goal(self) -> torch.Tensor:
    return self.table_twist[self.time_steps] * self.goal_mask[:, 0:1]

  @property
  def goal(self) -> torch.Tensor:
    t, mask = self.time_steps, self.goal_mask
    active = self.table_hand_active[t].unsqueeze(1)
    hands = self.table_hands[t] * active.unsqueeze(1)
    time_to_go = active * (self.cfg.hand_lead * self._env.step_dt)
    return torch.cat([self.twist_goal(), self.table_height[t].unsqueeze(1) * mask[:, 1:2],
                      self.table_pitch[t].unsqueeze(1) * mask[:, 2:3],
                      active, hands[:, 0], time_to_go, active, hands[:, 1], time_to_go, mask], dim=1)

  def own_twist(self) -> torch.Tensor:
    """Root velocity in the robot's own heading frame (vx, vy, yaw rate)."""
    data = self.robot.data
    velocity = quat_apply_inverse(yaw_quat(data.root_link_quat_w), data.root_link_lin_vel_w)
    return torch.cat([velocity[:, :2], data.root_link_ang_vel_w[:, 2:3]], dim=1)

  def _update_command(self, env_ids: torch.Tensor | None = None):
    super()._update_command(env_ids)
    goal = self.twist_goal()
    self._intent_now = self.intent.step(self.robot.data.joint_pos, torch.norm(goal[:, :2], dim=1))
    if env_ids is None:
      dt = self._env.step_dt
      self._goal_history = torch.roll(self._goal_history, 1, dims=1)
      self._goal_history[:, 0] = goal
      own = self.own_twist()
      error = own - self._goal_history[:, -1]
      if self.cfg.lag_grace > 0.0:
        # Historical evaluation only: suppress new lag during this explicit post-hit window.
        # With lag_grace=0 every frame is integrated without reading the hit clock.
        clock = getattr(self._env, "hit_clock", None)
        if clock is not None:
          error = error * (clock >= self.cfg.lag_grace).float().unsqueeze(1)
      self.goal_lag = self.goal_lag * math.exp(-dt / self.cfg.lag_time_constant) + dt * error
      self.twist_mean = self.twist_mean + (own - self.twist_mean) * (dt / self.cfg.twist_mean_time)
      self.metrics["goal_lag_xy"][:] = torch.norm(self.goal_lag[:, :2], dim=1)
      self.metrics["goal_lag_yaw"][:] = self.goal_lag[:, 2].abs()

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    # The reference pose was just written to the simulator; start the swing filter from it.
    self.intent.reset(env_ids, self.motion.joint_pos[self.time_steps])
    self._intent_now[env_ids] = self.intent.rest
    if self.cfg.pin_clips:
      # Evaluation: goals on, or (masks_off) off wherever off is allowed for the clip.
      clip = self.clip_ids[env_ids]
      self.goal_mask[env_ids] = 1.0
      if self.cfg.masks_off:
        self.goal_mask[env_ids, 1] = (~self.clip_height_free[clip]).float()
        self.goal_mask[env_ids, 2] = (~self.clip_pitch_free[clip]).float()
    self.goal_lag[env_ids] = 0.0
    self.twist_mean[env_ids] = self.table_twist[self.time_steps[env_ids]]
    self._goal_history[env_ids] = (self.table_twist[self.time_steps[env_ids]] * self.goal_mask[env_ids, 0:1]).unsqueeze(1)

  @property
  def arm_intent_now(self) -> torch.Tensor:
    return self._intent_now

  @property
  def arm_intent(self) -> torch.Tensor:
    return self._intent_now


@dataclass(kw_only=True)
class GoalMotionCommandCfg(LibraryMotionCommandCfg):
  goal_window: int = 20            # frames (0.4 s) averaged for twist, height and pitch goals
  hand_lead: int = 10              # frames (0.2 s) by which the hand goal leads
  hand_goal_contexts: tuple[str, ...] = ("pickup",)
  height_goal_prob: float = 0.7    # among clips where "off" is allowed
  pitch_goal_prob: float = 0.5
  default_height: float = 0.731    # pelvis height of the standing clips (median); what "height off" means
  height_free_band: float = 0.04   # a clip may run with the height goal off if it stays this close to the default
  pitch_free_band: float = 0.15    # rad, same for pitch (default 0)
  # Leak of the twist-error integral. With 2 s and a 0.4 m limit a robot that ignored a 0.2 m/s command settled at
  # exactly 0.4 m and was never terminated (measured: lag peaks of 0.39-0.40 m on every ignored script); 3 s with
  # the limits in student.env_cfg ends such an episode after about 2 s.
  lag_time_constant: float = 3.0
  lag_grace: float = 0.0          # positive values are historical evaluation only; zero never reads hit timing
  twist_mean_time: float = 0.4     # s, running mean of the own twist used by the twist rewards
  masks_off: bool = False          # with pin_clips: evaluate with height / pitch goals off where allowed
  quiet_stand_foot_range: float = 0.05   # m; a standing clip is "quiet" if neither foot moves farther than this
  stand_boost: float = 5.0         # sampling weight of quiet standing frames relative to other frames

  def build(self, env: ManagerBasedRlEnv) -> GoalMotionCommand:
    return GoalMotionCommand(self, env)
