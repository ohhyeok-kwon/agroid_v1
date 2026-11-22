"""Goal-stream mode: System 1 driven by a command stream instead of a reference clip.

This is the regime the robot runs in: System 2 (or skill code) sends a twist, a height and a pitch that stay constant
for seconds and change in steps. Nothing here uses a reference motion: the robot starts from a standing frame of the
library, the stream is slew-limited the way the goal governor will be, and only a fall ends an episode.

One mode serves three purposes: the S2-like evaluation of every student checkpoint (scripted streams), the bed on
which the balance-urgency signal is calibrated on frozen policies, and the training scene of the hardening stage.

`MatchedStreamCommand` adds a reference to the stream without a clip script: every few tenths of a second it looks
up the library frame that best fits the commanded twist and height AND the robot's own foot configuration, and
continues the reference from there (motion matching inside the environment). The frozen tracking teacher then labels
"how a human would continue from this state toward this command". Because the reference is re-anchored to the robot
(height and tilt only) a switch causes no planar or heading jump, and because the match includes the robot's own
feet the reference follows the robot's gait phase instead of imposing one, also after an unplanned step.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.tasks.tracking import mdp
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

from r1wb import robot
from r1wb.student import observations as obs
from r1wb.student.command import GoalMotionCommand, GoalMotionCommandCfg  # noqa: F401
from r1wb.student.env_cfg import student_env_cfg
from r1wb.tracking import mdp as wb

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

SCRIPT_COLUMNS = ("duration_s", "vx", "vy", "yaw_rate", "height", "height_on", "pitch", "pitch_on")


def _no_hit_reference_hold(value: float) -> None:
  if not math.isfinite(value) or value != 0.0:
    raise ValueError("hold_after_hit must be 0: post-hit reference overrides are disabled")


class GoalStreamCommand(GoalMotionCommand):
  cfg: GoalStreamCommandCfg

  def __init__(self, cfg: GoalStreamCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    manifest = json.loads(Path(cfg.motion_file).with_suffix(".manifest.json").read_text())
    stand = torch.tensor([clip["context"] == "stand" and not clip["mirrored"] for clip in manifest["clips"]],
                         dtype=torch.bool, device=self.device)
    if not cfg.pin_clips:
      self.allow_clips(stand)
    count = self.num_envs
    self.stream = torch.zeros(count, 5, device=self.device)            # vx, vy, yaw rate, height, pitch (slewed)
    self.stream[:, 3] = cfg.default_height
    self.stream_target = self.stream.clone()
    # The instruction as it was given (the script's or the sampler's segment). `stream_target` is what the generator
    # plays; a System 2 that steers (r1wb.s2.steer) writes its own command there and reads the instruction here.
    self.told = self.stream.clone()
    self.segment_left = torch.zeros(count, device=self.device)
    self.segment_index = torch.zeros(count, dtype=torch.long, device=self.device)
    self.since_change = torch.zeros(count, device=self.device)        # seconds since the stream target last changed
    self.script: torch.Tensor | None = None
    self.slew = torch.tensor(cfg.slew, device=self.device)

  def set_script(self, script: torch.Tensor) -> None:
    """script: (num_envs, segments, 8) with the columns of SCRIPT_COLUMNS; the last segment repeats."""
    assert script.shape[0] == self.num_envs and script.shape[2] == len(SCRIPT_COLUMNS)
    self.script = script.to(self.device, torch.float)

  # -- stream ---------------------------------------------------------------------------------------------------

  def _uniform(self, count: int, low: float, high: float) -> torch.Tensor:
    return low + (high - low) * torch.rand(count, device=self.device)

  def _sign(self, count: int) -> torch.Tensor:
    return torch.where(torch.rand(count, device=self.device) < 0.5, -1.0, 1.0)

  def _sample_segment(self, ids: torch.Tensor) -> None:
    cfg, count = self.cfg, len(ids)
    share = torch.tensor(cfg.segment_shares, device=self.device)       # stand, forward, backward, lateral, turn
    kind = torch.multinomial(share / share.sum(), count, replacement=True)
    forward, backward, lateral, turn = kind == 1, kind == 2, kind == 3, kind == 4
    zero = torch.zeros(count, device=self.device)
    curve = forward & (torch.rand(count, device=self.device) < 0.3)
    target = torch.zeros(count, 5, device=self.device)
    target[:, 0] = torch.where(forward, self._uniform(count, cfg.min_forward, cfg.max_forward),
                               torch.where(backward, -self._uniform(count, 0.1, cfg.max_backward), zero))
    target[:, 1] = torch.where(lateral, self._sign(count) * self._uniform(count, 0.1, cfg.max_lateral), zero)
    target[:, 2] = torch.where(turn, self._sign(count) * self._uniform(count, cfg.min_yaw_rate, cfg.max_yaw_rate),
                               torch.where(curve, self._sign(count) * self._uniform(count, 0.1, 0.4), zero))
    low = torch.rand(count, device=self.device) < cfg.low_height_prob
    height = torch.where(low, self._uniform(count, cfg.min_height, cfg.default_height - 0.03),
                         torch.full((count,), cfg.default_height, device=self.device))
    target[:, :3] *= torch.where(low & (height < 0.62), 0.5, 1.0).unsqueeze(1)      # slower while clearly lowered
    lean = torch.rand(count, device=self.device) < cfg.pitch_prob
    target[:, 3] = height
    target[:, 4] = torch.where(lean, self._uniform(count, -0.05, 0.3), zero)
    self.stream_target[ids] = target
    self.goal_mask[ids, 1] = torch.where(low, 1.0, (torch.rand(count, device=self.device) < 0.5).float())
    self.goal_mask[ids, 2] = lean.float()
    self.segment_left[ids] = self._uniform(count, *cfg.segment_time)

  def _begin_segment(self, ids: torch.Tensor) -> None:
    if self.script is None:
      self._sample_segment(ids)
    else:
      row = self.script[ids, self.segment_index[ids].clamp(max=self.script.shape[1] - 1)]
      self.segment_left[ids] = row[:, 0]
      self.stream_target[ids] = row[:, [1, 2, 3, 4, 6]]
      self.goal_mask[ids, 1], self.goal_mask[ids, 2] = row[:, 5], row[:, 7]
    self.told[ids] = self.stream_target[ids]
    self.segment_index[ids] += 1
    self.since_change[ids] = 0.0

  def _advance_stream(self) -> None:
    dt = self._env.step_dt
    self.segment_left -= dt
    self.since_change += dt
    expired = (self.segment_left <= 0.0).nonzero(as_tuple=False).flatten()
    if len(expired):
      self._begin_segment(expired)
    step = self.slew * dt
    self.stream += torch.maximum(torch.minimum(self.stream_target - self.stream, step), -step)

  # -- goal interface ---------------------------------------------------------------------------------------------

  def twist_goal(self) -> torch.Tensor:
    return self.stream[:, :3]

  @property
  def goal(self) -> torch.Tensor:
    mask = self.goal_mask
    hands = torch.zeros(self.num_envs, 10, device=self.device)        # both hand goals inactive
    return torch.cat([self.stream[:, :3], self.stream[:, 3:4] * mask[:, 1:2], self.stream[:, 4:5] * mask[:, 2:3],
                      hands, mask], dim=1)

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None:
      self._advance_stream()
    super()._update_command(env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    count = len(env_ids)
    self.stream[env_ids] = 0.0
    self.stream[env_ids, 3] = self.cfg.default_height
    self.stream_target[env_ids] = self.stream[env_ids]
    self.told[env_ids] = self.stream[env_ids]
    self.goal_mask[env_ids] = 1.0
    self.goal_mask[env_ids, 2] = 0.0
    self.segment_index[env_ids] = 0
    self.since_change[env_ids] = 0.0
    # Scripts start with their first segment; random streams start with a short stand.
    self.segment_left[env_ids] = 0.0 if self.script is not None else self._uniform(count, 0.5, 1.5)
    self.goal_lag[env_ids] = 0.0
    self._goal_history[env_ids] = 0.0


@dataclass(kw_only=True)
class GoalStreamCommandCfg(GoalMotionCommandCfg):
  segment_time: tuple[float, float] = (2.0, 6.0)
  # Slew limits per second: vx, vy (m/s^2), yaw rate (rad/s^2), height (m/s), pitch (rad/s).
  slew: tuple[float, float, float, float, float] = (1.0, 1.0, 2.0, 0.15, 0.3)
  min_forward: float = 0.1
  max_forward: float = 0.6
  max_backward: float = 0.3
  max_lateral: float = 0.3
  min_yaw_rate: float = 0.3
  max_yaw_rate: float = 1.0
  min_height: float = 0.52
  low_height_prob: float = 0.3
  pitch_prob: float = 0.2
  segment_shares: tuple[float, ...] = (0.25, 0.30, 0.10, 0.15, 0.20)   # stand, forward, backward, lateral, turn in place

  def build(self, env: ManagerBasedRlEnv) -> GoalStreamCommand:
    return GoalStreamCommand(self, env)


class MatchedStreamCommand(GoalStreamCommand):
  cfg: MatchedStreamCommandCfg

  def __init__(self, cfg: MatchedStreamCommandCfg, env: ManagerBasedRlEnv):
    _no_hit_reference_hold(cfg.hold_after_hit)
    super().__init__(cfg, env)
    names = list(cfg.body_names)
    self._feet = [names.index(name) for name in robot.FOOT_BODIES]
    m = self.motion
    root_pos, root_quat = m.body_pos_w[:, self._root], m.body_quat_w[:, self._root]
    pose = self._pose_features(root_pos, root_quat, m.body_lin_vel_w[:, self._root],
                               m.body_pos_w[:, self._feet], m.body_lin_vel_w[:, self._feet])
    goal = torch.cat([self.table_twist, self.table_height.unsqueeze(1)], dim=1)
    # Travel: the mean twist over a whole stride (in the heading frame of the frame itself). The 0.4 s goal alone
    # cannot tell a sideways step from a sideways sway of the pelvis over planted feet: a 0.2 m/s lateral command
    # was matched to the weight shift at the end of a turning clip, and the robot stayed where it was.
    frames, end = torch.arange(m.time_step_total, device=self.device), self.clip_end[self.frame_clip]
    heading = yaw_quat(root_quat)
    travel = torch.zeros(m.time_step_total, 3, device=self.device)
    for step in range(0, cfg.travel_window + 1, 2):
      index = torch.minimum(frames + step, end)
      travel[:, :2] += quat_apply_inverse(heading, m.body_lin_vel_w[index, self._root])[:, :2]
      travel[:, 2] += m.body_ang_vel_w[index, self._root, 2]
    travel /= len(range(0, cfg.travel_window + 1, 2))
    self.table_travel = travel
    # Continuity: the root twist of the frame itself, compared with the robot's own twist now. Without it a robot
    # that stands and is told to walk backward was matched to frames in the middle of a backward walk (their 0.4 s
    # and 1.2 s means fit the command), so the label asked a standing robot to be mid-stride and the student never
    # started. With it the frames where a capture starts to move from standing win.
    current = torch.cat([quat_apply_inverse(heading, m.body_lin_vel_w[:, self._root])[:, :2],
                         m.body_ang_vel_w[:, self._root, 2:3]], dim=1)
    self.table_twist_now = current                                    # a frame's own root twist in its heading frame
    self.course_twist_mean = torch.zeros(self.num_envs, 3, device=self.device)   # ... of the course, 0.4 s running mean
    scale = torch.tensor(cfg.goal_scale + cfg.travel_scale + cfg.current_scale + cfg.pose_scale, device=self.device)
    weight = torch.tensor(cfg.goal_weight + cfg.travel_weight + cfg.current_weight + cfg.pose_weight,
                          device=self.device).sqrt()
    self._feature_scale = weight / scale
    table = torch.cat([goal, travel, current, pose], dim=1) * self._feature_scale
    # Candidates: frames of clips the student trains on (mirrors included: the robot's own feet pick the side),
    # with at least `match_min_remaining` frames left, thinned by `match_stride`.
    remaining = end - frames
    usable = (self.clip_trained | self._mirror_of_trained())[self.frame_clip]
    # Two candidate sets. While a motion is commanded a candidate needs room to play (`match_min_remaining`). When
    # the command is "stand", the last second of a clip is exactly what is wanted: captures end by coming to rest.
    # Frames from which the reference stands still for the next second (root and both feet at rest).
    busy = torch.zeros(m.time_step_total, device=self.device)
    for step in range(0, 51, 2):
      index = torch.minimum(frames + step, end)
      root = torch.norm(m.body_lin_vel_w[index, self._root, :2], dim=1) / 0.08
      turn = m.body_ang_vel_w[index, self._root, 2].abs() / 0.2
      feet = torch.norm(m.body_lin_vel_w[index][:, self._feet, :2], dim=-1).amax(dim=1) / 0.12
      busy = torch.maximum(busy, torch.maximum(torch.maximum(root, turn), feet))
    self.stand_frame = busy < 1.0
    # ... and frames whose clip reaches such a frame within the window a look-up commits to.
    arrives = torch.zeros(m.time_step_total, dtype=torch.bool, device=self.device)
    for step in range(0, cfg.travel_window + 1, 2):
      arrives |= self.stand_frame[torch.minimum(frames + step, end)]
    self._sets = []
    for which, least in enumerate((cfg.match_min_remaining, cfg.match_stop_remaining)):
      allowed = usable & (remaining >= least)
      if which == 1 and cfg.match_stop == "arrive":
        allowed = allowed & arrives
      index = allowed.nonzero(as_tuple=False).flatten()[::cfg.match_stride]
      self._sets.append((index, table[index], torch.sum(table[index] * table[index], dim=1)))
    self._all = table
    self.match_timer = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.hold = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
    self.match_switches = torch.zeros(self.num_envs, device=self.device)
    self.match_cost = torch.zeros(self.num_envs, device=self.device)
    self.metrics["match_cost"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["match_switch_rate"] = torch.zeros(self.num_envs, device=self.device)

  def _mirror_of_trained(self) -> torch.Tensor:
    """Clips left out of clip-mode training only because they are the mirror of a symmetric clip."""
    manifest = json.loads(Path(self.cfg.motion_file).with_suffix(".manifest.json").read_text())
    index = {clip["name"]: i for i, clip in enumerate(manifest["clips"])}
    out = torch.zeros(len(self.clip_start), dtype=torch.bool, device=self.device)
    for i, clip in enumerate(manifest["clips"]):
      if clip["mirrored"]:
        source = index.get(clip["name"][:-2]) if clip["name"].endswith("_M") else None
        out[i] = bool(source is not None and self.clip_trained[source])
    return out

  @staticmethod
  def _pose_features(root_pos, root_quat, root_vel, feet_pos, feet_vel) -> torch.Tensor:
    """Feet position and velocity relative to the pelvis in the pelvis heading frame (12), pelvis height (1)."""
    heading = yaw_quat(root_quat).unsqueeze(1).expand(-1, feet_pos.shape[1], -1)
    position = quat_apply_inverse(heading, feet_pos - root_pos.unsqueeze(1))
    velocity = quat_apply_inverse(heading, feet_vel - root_vel.unsqueeze(1))
    count = root_pos.shape[0]
    return torch.cat([position.reshape(count, -1), velocity.reshape(count, -1), root_pos[:, 2:3]], dim=1)

  def _query(self, ids: torch.Tensor) -> torch.Tensor:
    _no_hit_reference_hold(self.cfg.hold_after_hit)
    aimed = self.cfg.match_height == "target"
    counts = self.goal_mask[ids, 1] > 0
    height = torch.where(counts, (self.stream_target if aimed else self.stream)[ids, 3],
                         torch.full((len(ids),), self.cfg.default_height, device=self.device))
    origin = self._env.scene.env_origins[ids]
    pose = self._pose_features(self.robot_body_pos_w[ids, self._root] - origin, self.robot_body_quat_w[ids, self._root],
                               self.robot_body_lin_vel_w[ids, self._root],
                               self.robot_body_pos_w[ids][:, self._feet] - origin.unsqueeze(1),
                               self.robot_body_lin_vel_w[ids][:, self._feet])
    # Expected mean of the slew-limited command over the travel window.
    now, target, slew = self.stream[ids, :3], self.stream_target[ids, :3], self.slew[:3]
    horizon = self.cfg.travel_window * self._env.step_dt
    reach = ((target - now).abs() / slew).clamp(max=horizon)
    ramp_end = now + torch.sign(target - now) * slew * reach
    travel = (reach * 0.5 * (now + ramp_end) + (horizon - reach) * target) / horizon
    query = torch.cat([now, height.unsqueeze(1), travel, self.own_twist()[ids], pose], dim=1) * self._feature_scale
    # A course does not listen to a hit. With the robot's feet in the query, a hit while walking moved the reference
    # to another gait for good (measured 11-12 03:00), so what the robot did after the hit could not be priced.
    # match_pose "reference" always uses the course's own frame. There is no
    # time after a hit in which reference matching follows a different rule.
    if self.cfg.match_pose == "reference":
      width = len(self.cfg.pose_scale)
      course = self._all[self.time_steps[ids], -width:]
      query[:, -width:] = course
    if aimed:
      # The stance is compared as it will be at the commanded height (pelvis up by the difference, feet down by it
      # relative to the pelvis): the pose terms then judge the feet, not the height the command is about to change.
      # Without this a look-up closed 80% of a height change (goal weight 4 against pose weight 1) and the rest
      # followed one `travel_window` later.
      width = len(self.cfg.pose_scale)
      scale = self._feature_scale[-width:]
      shift = torch.where(counts, height - query[:, -1] / scale[-1], torch.zeros_like(height))
      query[:, -1] += shift * scale[-1]
      for foot_z in (2, 5):
        query[:, -width + foot_z] -= shift * scale[foot_z]
    return query

  def _rematch(self, ids: torch.Tensor, forced: torch.Tensor) -> None:
    """Move the reference of `ids` to the best-fitting library frame (forced: no hysteresis)."""
    query = self._query(ids)
    current = torch.sum(torch.square(self._all[self.time_steps[ids]] - query), dim=1)
    best_cost = torch.empty(len(ids), device=self.device)
    best = torch.empty(len(ids), dtype=torch.long, device=self.device)
    stopping = torch.norm(self.stream_target[ids, :3], dim=1) < 1e-6
    for which, (index, table, norm) in enumerate(self._sets):
      rows = (stopping == bool(which)).nonzero(as_tuple=False).flatten()
      for start in range(0, len(rows), self.cfg.match_chunk):
        part_rows = rows[start:start + self.cfg.match_chunk]
        part = query[part_rows]
        cost = norm.unsqueeze(0) - 2.0 * part @ table.T + torch.sum(part * part, dim=1, keepdim=True)
        # A little noise spreads equal-cost choices over the library instead of always taking the first frame.
        cost = cost * (1.0 + self.cfg.match_noise * torch.rand_like(cost))
        value, where = cost.min(dim=1)
        best_cost[part_rows], best[part_rows] = value, index[where]
    # Leave a reference that already fits; otherwise switch when the best frame is clearly better.
    switch = forced | ((current > self.cfg.match_good_enough) & (best_cost < self.cfg.match_hysteresis * current))
    chosen = ids[switch]
    frames = best[switch]
    self.time_steps[chosen] = frames
    self.clip_ids[chosen] = self.frame_clip[frames]
    self.clip_finished[chosen] = False
    self.match_cost[ids] = torch.where(switch, best_cost, current)
    self.match_switches[chosen] += 1.0
    self.match_timer[ids] = 0

  def _standing(self) -> torch.Tensor:
    """Asked to stand, and the reference already stands at the asked height: it pauses there (holding still is
    the label for "stand", not the idle fidgeting of a capture)."""
    height = torch.where(self.goal_mask[:, 1] > 0, self.stream_target[:, 3],
                         torch.full((self.num_envs,), self.cfg.default_height, device=self.device))
    frame = self.time_steps
    return (self._matched_envs() & (torch.norm(self.stream_target[:, :3], dim=1) < 1e-6)
            & (torch.norm(self.stream[:, :3], dim=1) < 0.02)
            & self.stand_frame[frame] & ((self.table_height[frame] - height).abs() < 0.03))

  def _matched_envs(self) -> torch.Tensor:
    """Environments whose reference is chosen by matching (all of them here; a subclass mixes in clip replay)."""
    return torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None:
      self.hold = self._standing()
    super()._update_command(env_ids)
    if env_ids is not None:
      return
    self.match_timer += 1
    # The course's own travel as the robot's is measured (twist_mean): a running mean over what was played.
    played = self.table_twist_now[self.time_steps] * (~self.hold).float().unsqueeze(1)
    self.course_twist_mean += (played - self.course_twist_mean) * (self._env.step_dt / self.cfg.twist_mean_time)
    ending = (self.clip_end[self.clip_ids] - self.time_steps <= self.cfg.match_end_margin) & ~self.hold
    # While the command is changing the reference has to follow it: look up twice as often. Hysteresis stays on
    # (forcing every look-up made the reference jump between clips and the robot stumble at gait initiation); the
    # growing goal mismatch of the current frame is what triggers the switch.
    changing = self.since_change < self.cfg.match_follow_time
    due = ((self.match_timer >= self.cfg.match_interval) | ending
           | (changing & (self.match_timer >= self.cfg.match_interval // 2))) & ~self.hold & self._matched_envs()
    if self.cfg.match_commit:
      # A course is a piece of captured motion that is played as it was chosen: a look-up scores the mean twist of
      # the next `travel_window` frames, so that window is played before the next look-up (and one happens when the
      # command changes or the clip ends). Looking up every 0.4 s re-chose whenever the frame's own twist drifted
      # from a constant command and strung together half-second pieces (turning, sideways, backward: 1.3 to 1.5
      # switches per second) whose root moved as commanded while the feet never finished a step: the same teacher
      # follows the turning and sideways clips themselves at 0.92 to 0.98 and those streams at 0.02 to 0.5.
      changed = self.since_change < 0.5 * self._env.step_dt
      due = ((self.match_timer >= self.cfg.travel_window) | ending | changed) & ~self.hold & self._matched_envs()
    ids = due.nonzero(as_tuple=False).flatten()
    if len(ids):
      self._rematch(ids, ending[ids])
      self.update_relative_body_poses()
    self.metrics["match_cost"][:] = self.match_cost
    self.metrics["match_switch_rate"][:] = self.match_switches / (self._env.episode_length_buf.float().clamp_min(1.0)
                                                                   * self._env.step_dt)

  def _resample_command(self, env_ids: torch.Tensor):
    super()._resample_command(env_ids)
    self.match_timer[env_ids] = 0
    self.hold[env_ids] = False
    self.course_twist_mean[env_ids] = self.twist_mean[env_ids]
    self.match_switches[env_ids] = 0.0
    self.match_cost[env_ids] = 0.0


@dataclass(kw_only=True)
class MatchedStreamCommandCfg(GoalStreamCommandCfg):
  frozen: bool = False                      # the matched reference advances
  match_interval: int = 20                  # steps (0.4 s) between look-ups
  match_hysteresis: float = 0.7             # switch only if the best frame costs less than this share of the current one
  match_good_enough: float = 3.0            # ... and the current frame costs more than this
  match_follow_time: float = 1.2            # s after a command change with look-ups at twice the rate
  match_min_remaining: int = 60             # a candidate must have this many frames left in its clip
  match_stop_remaining: int = 12            # ... or this many when the command is "stand"
  match_end_margin: int = 8                 # forced look-up this many frames before the clip ends
  match_stride: int = 2
  match_chunk: int = 1024
  match_noise: float = 0.05
  match_pose: str = "robot"                 # whose feet enter the look-up: "robot" (state-matched) or "reference"
  hold_after_hit: float = 0.0               # compatibility field; nonzero post-hit overrides are rejected
  match_commit: bool = False                # look up only when the scored window is played, the command changes or the clip ends
  # Which height a look-up asks for while the height command is still moving (slew 0.15 m/s): "stream" = its value
  # now, "target" = where it is going (as the twist terms do with the mean over `travel_window`). With "stream" the
  # look-up at a command change still asks for the old height, so the reference changes height one `travel_window`
  # (1.2 s) later, on a clock that nothing outside this class can see; a model imitating the course from the frames
  # and the goal then never leaves a held crouch (gates 11-13 03:05). "target": the change happens at the look-up
  # the command change itself triggers.
  match_height: str = "stream"
  # Which frames a look-up may choose when the command is "stand": "room" = any frame with `match_stop_remaining`
  # frames left (the look-up then often lands on motion that does not come to rest, and the reference stands only
  # after the next look-up: 24% of the recorded stops waited for that look-up at exactly 60 steps, again on the
  # unseen clock, and a model imitating the course kept moving for 3 to 6 s after "stand"; gates 11-13 04:15).
  # "arrive" = only frames whose own clip comes to rest within `travel_window`: a stop is one piece of captured
  # motion that ends standing.
  match_stop: str = "room"
  # Feature scales and weights: goal = vx, vy, yaw rate (0.4 s mean), height; travel = vx, vy, yaw rate (mean over
  # `travel_window`); current = vx, vy, yaw rate of the frame itself; pose = 2 x foot position (xyz), 2 x foot
  # velocity (xyz), pelvis height.
  # Values validated in closed loop (teacher driven over the 18 scripts: 0 falls, mean distance ratio 0.83).
  # A 2 s window with weights (1, 1, 1) / (16, 16, 8) looked better in an offline comparison of candidate costs and
  # was worse in closed loop (forward 0.2 m/s: 1.02 -> 0.01; ledger 11-11 08:37): judge matcher changes only by
  # driving the robot. Pure lateral walking is a data gap (1.2 s of sustained 0.15-0.25 m/s in the library).
  travel_window: int = 60
  goal_scale: tuple[float, ...] = (0.15, 0.15, 0.3, 0.04)
  goal_weight: tuple[float, ...] = (3.0, 3.0, 2.0, 4.0)
  travel_scale: tuple[float, ...] = (0.15, 0.15, 0.3)
  travel_weight: tuple[float, ...] = (8.0, 8.0, 5.0)
  # Root twist of the frame itself vs the robot's own (continuity). Off: with weights (6, 6, 3) the teacher-driven
  # suite gained on backward (0.78 -> 0.90) and lost on low walking (0.78 -> 0.44) and on average (0.83 -> 0.79),
  # and a standing student was still matched to mid-clip frames (ledger 11-11 08:55).
  current_scale: tuple[float, ...] = (0.15, 0.15, 0.3)
  current_weight: tuple[float, ...] = (0.0, 0.0, 0.0)
  pose_scale: tuple[float, ...] = (0.08,) * 6 + (0.5,) * 6 + (0.04,)
  pose_weight: tuple[float, ...] = (1.0,) * 6 + (0.5,) * 6 + (1.0,)

  def __post_init__(self) -> None:
    parent = getattr(super(), "__post_init__", None)
    if parent is not None:
      parent()
    _no_hit_reference_hold(self.hold_after_hit)

  def build(self, env: ManagerBasedRlEnv) -> MatchedStreamCommand:
    return MatchedStreamCommand(self, env)


class MixedGoalCommand(MatchedStreamCommand):
  """Clip replay and matched command streams side by side (training of the stream-capable student).

  Environments with `stream_env` receive a command stream and a state-matched reference; the others replay library
  clips with goals derived from the clip, exactly as GoalMotionCommand does. Both kinds reset on library frames
  (failure-weighted), so streams begin in every kind of motion, and a stream's command values are drawn from what
  the library demonstrates (the stride-mean twist and height of a random frame), held for a few seconds.
  """
  cfg: MixedGoalCommandCfg

  def __init__(self, cfg: MixedGoalCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    index = torch.arange(self.num_envs, device=self.device)
    self.stream_env = (index % cfg.stream_period) < cfg.stream_share
    if not cfg.pin_clips:
      self._allow_trained()
    self.metrics["stream_goal_lag_xy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["driven"] = torch.zeros(self.num_envs, device=self.device)
    # Command kinds, by the stride-mean twist of the candidate frames. Drawing frames at random gives commands in
    # proportion to the library: 1% backward, 3% turning in place. After 250 iterations of such streams the student
    # followed 0.2 m/s forward (0.89) but not backward or turning in place (0.0-0.08), so kinds are drawn evenly.
    candidates = self._sets[0][0]
    travel = self.table_travel[candidates]
    speed, vx, vy, wz = torch.norm(travel[:, :2], dim=1), travel[:, 0], travel[:, 1], travel[:, 2].abs()
    kinds = {
      "stand": (speed < 0.05) & (wz < 0.1),
      "forward": (vx > 0.1) & (vy.abs() < 0.12) & (wz < 0.25),
      "backward": vx < -0.08,
      "lateral": (vy.abs() > 0.1) & (vx.abs() < 0.12),
      "turn": (wz > 0.3) & (speed < 0.12),
      "arc": (vx > 0.1) & (wz >= 0.25),
    }
    rest = ~torch.stack(list(kinds.values())).any(dim=0)
    kinds["other"] = rest
    self._kind_names = list(kinds)
    self._kind_frames = [candidates[mask] for mask in kinds.values()]
    weight = torch.tensor([cfg.kind_weights[name] if len(frames) else 0.0
                           for name, frames in zip(self._kind_names, self._kind_frames)], device=self.device)
    self._kind_weight = weight / weight.sum()
    self.kind_count = {name: int(len(frames)) for name, frames in zip(self._kind_names, self._kind_frames)}
    self._steerer, self._steer_policy, self._steer_tick = None, None, 0

  def _steer_step(self) -> None:
    """Once per control step: the learned System 2 gives the streams their command at its own period."""
    if self._steerer is None:
      from r1wb.eval.paths import AGENT
      from r1wb.s2.steer import Steer, SteerPolicy
      self._steer_policy = SteerPolicy.load(AGENT / "data/s2" / f"steer_{self.cfg.steer}.pt", self.device)
      self._steerer = Steer(self._env, self, self._steer_policy.see, leak=self._steer_policy.leak,
                            remember=self._steer_policy.remember, turned=self._steer_policy.turned)
    else:
      self._steerer.progress.after_step()
    if self._steer_tick % self._steer_policy.period == 0:
      with torch.no_grad():
        self._steerer.act(self._steer_policy.best(self._steerer.observe()))
    self._steer_tick += 1

  def _matched_envs(self) -> torch.Tensor:
    return self.stream_env

  def twist_goal(self) -> torch.Tensor:
    clip = self.table_twist[self.time_steps] * self.goal_mask[:, 0:1]
    return torch.where(self.stream_env.unsqueeze(1), self.stream[:, :3], clip)

  @property
  def goal(self) -> torch.Tensor:
    return torch.where(self.stream_env.unsqueeze(1), GoalStreamCommand.goal.fget(self), GoalMotionCommand.goal.fget(self))

  def _sample_segment(self, ids: torch.Tensor) -> None:
    cfg, count = self.cfg, len(ids)
    kind = torch.multinomial(self._kind_weight, count, replacement=True)
    frame = torch.zeros(count, dtype=torch.long, device=self.device)
    for index, frames in enumerate(self._kind_frames):
      rows = (kind == index).nonzero(as_tuple=False).flatten()
      if len(rows):
        frame[rows] = frames[torch.randint(len(frames), (len(rows),), device=self.device)]
    # Down to 0.4 of the demonstrated travel: small sustained commands (0.2 m/s, 0.6 rad/s) are what System 2 sends
    # for approach and alignment, and they are rare in the captures.
    twist = self.table_travel[frame] * self._uniform(count, 0.4, 1.1).unsqueeze(1)
    height = self.table_height[frame].clone()
    stand = kind == self._kind_names.index("stand")
    twist = torch.where(stand.unsqueeze(1), torch.zeros_like(twist), twist)
    # No ambiguous crawl commands: below this the command means "stand".
    slow = (torch.norm(twist[:, :2], dim=1) < 0.08) & (twist[:, 2].abs() < 0.15)
    twist = torch.where(slow.unsqueeze(1), torch.zeros_like(twist), twist)
    upright = stand & (torch.rand(count, device=self.device) < 0.7)
    height = torch.where(upright, torch.full_like(height, cfg.default_height), height)
    lowered = (height - cfg.default_height).abs() > cfg.height_free_band
    target = torch.zeros(count, 5, device=self.device)
    target[:, :3], target[:, 3] = twist, torch.where(lowered, height, torch.full_like(height, cfg.default_height))
    self.stream_target[ids] = target
    self.goal_mask[ids, 1] = torch.where(lowered, 1.0, (torch.rand(count, device=self.device) < 0.5).float())
    self.goal_mask[ids, 2] = 0.0
    self.segment_left[ids] = self._uniform(count, *cfg.segment_time)

  def _update_command(self, env_ids: torch.Tensor | None = None):
    if env_ids is None and self.cfg.steer:
      self._steer_step()
    super()._update_command(env_ids)
    if env_ids is None:
      lag = torch.norm(self.goal_lag[:, :2], dim=1)
      self.metrics["stream_goal_lag_xy"][:] = torch.where(self.stream_env, lag, torch.zeros_like(lag))

  def _resample_command(self, env_ids: torch.Tensor):
    GoalMotionCommand._resample_command(self, env_ids)       # reset frame, clip-mode masks, intent, lag, goal history
    count, frame, mask = len(env_ids), self.time_steps[env_ids], self.goal_mask[env_ids]
    # A stream starts as the clip it was reset into and departs from it after its first segment.
    self.stream[env_ids, :3] = self.table_twist[frame]
    self.stream[env_ids, 3] = torch.where(mask[:, 1] > 0, self.table_height[frame],
                                          torch.full((count,), self.cfg.default_height, device=self.device))
    self.stream[env_ids, 4] = self.table_pitch[frame] * mask[:, 2]
    self.stream_target[env_ids] = self.stream[env_ids]
    self.told[env_ids] = self.stream[env_ids]
    self.segment_index[env_ids] = 0
    self.since_change[env_ids] = 0.0
    self.segment_left[env_ids] = torch.where(self.stream_env[env_ids], self._uniform(count, 0.5, 1.5),
                                             torch.full((count,), 1.0e9, device=self.device))
    self.match_timer[env_ids] = 0
    self.hold[env_ids] = False
    self.match_switches[env_ids] = 0.0
    self.match_cost[env_ids] = 0.0
    # Teacher-driven episodes (stream environments only): the label is executed instead of the student's action,
    # so the student also sees states along motions it cannot produce yet (see learning.ImitationPPO).
    self.driven[env_ids] = self.stream_env[env_ids] & (torch.rand(count, device=self.device) < self.cfg.driven_share)
    if self._steerer is not None:
      self._steerer.restart(env_ids)
    self.metrics["driven"][:] = self.driven.float()


@dataclass(kw_only=True)
class MixedGoalCommandCfg(MatchedStreamCommandCfg):
  driven_share: float = 0.3        # share of stream episodes in which the teacher's label drives the robot
  stream_period: int = 2           # environments i with (i mod stream_period) < stream_share run command streams
  stream_share: int = 1
  # Share of stream segments per command kind (kinds without library frames are skipped).
  kind_weights: dict = field(default_factory=lambda: {"stand": 0.2, "forward": 0.18, "backward": 0.14, "lateral": 0.1,
                                                      "turn": 0.16, "arc": 0.12, "other": 0.1})
  # A learned System 2 (r1wb.s2.steer, <work>/data/s2/steer_<name>.pt) steers the streams' commands while System 1
  # trains: System 1 then meets the courses it is given in deployment (with the loop the robot walks at the told
  # speed, about a third faster than the courses it was trained on, and needs 4 to 7 steps to stop where it needed
  # 1 or 2: gates 11-13 13:18).
  steer: str = ""

  def build(self, env: ManagerBasedRlEnv) -> MixedGoalCommand:
    return MixedGoalCommand(self, env)


def mixed_rl_cfg():
  """The student's learning settings with a higher imitation floor: new kinds of commands are first learned from
  teacher-driven episodes, and the run resumes past the point where the floor is reached."""
  from r1wb.student.env_cfg import student_rl_cfg
  cfg = student_rl_cfg()
  cfg.algorithm.bc_floor = 0.5
  return cfg


def mixed_env_cfg(play: bool = False, teacher: str | None = None, clean: bool | None = None,
                  subset: str = "core", disturbance: str = "kick", joint_course: float = 0.0, hit_grace: float = 0.0,
                  stream_course: str = "always", stream_hits: bool = True) -> ManagerBasedRlEnvCfg:
  """The student environment with half of the environments on matched command streams (P2b).
  disturbance / joint_course / hit_grace: as in the student environment. stream_course: where the stream
  environments' reference counts as a course for the labels and the joint-course price: "always", "never"
  (the streams are then learned from the goal rewards alone) or "plan" (always, and the reference is
  selected from the command and its own frame only, never from the robot). Other than "always": no episode is driven
  by the teacher. stream_hits=False: the impulses fall on the clip environments only (the streams stay calm, so
  their reference is a course at all times and the reaction is learned where it can be priced)."""
  if stream_course not in ("always", "never", "plan"):
    raise ValueError("stream_course must be always, never or plan; post-hit calm/held exemptions are disabled")
  cfg = student_env_cfg(play=play, teacher=teacher, clean=clean, subset=subset, disturbance=disturbance,
                        joint_course=joint_course, hit_grace=hit_grace)
  old = cfg.commands["motion"]
  values = {field.name: getattr(old, field.name) for field in fields(old)}
  cfg.commands = {"motion": MixedGoalCommandCfg(**values)}
  if stream_course == "plan":       # the course is the command's plan: it never listens to the robot (or a hit)
    cfg.commands["motion"].driven_share = 0.0
    cfg.commands["motion"].match_pose = "reference"
  elif stream_course != "always":
    cfg.commands["motion"].driven_share = 0.0
    trust = {"stream": stream_course, "grace": 0.0}  # "never" is unconditional, with no implicit hit window.
    if "joint_course" in cfg.rewards:
      cfg.rewards["joint_course"].params.update(trust)
    if "bc" in cfg.observations:
      cfg.observations["bc"].terms["label"].params.update(trust)
  if not stream_hits and "impulse" in cfg.events:
    cfg.events["impulse"].params["calm_streams"] = True
  return cfg


def fallen(env: ManagerBasedRlEnv, min_height: float = 0.35, max_tilt: float = 0.9) -> torch.Tensor:
  from r1wb import world
  data = env.scene["robot"].data
  height = world.standing_height(env)            # above the surface the robot stands on (flat floor: the start level)
  tilt = torch.acos((-data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
  return (height < min_height) | (tilt > max_tilt)


def root_height(env: ManagerBasedRlEnv) -> torch.Tensor:
  from r1wb import world
  return world.standing_height(env).unsqueeze(1)


def foot_forces(env: ManagerBasedRlEnv, sensor_name: str = "feet_ground", scale: float = 0.01) -> torch.Tensor:
  return env.scene[sensor_name].data.force.reshape(env.num_envs, -1)[:, :6] * scale


def stand_clips(motion_file: str) -> tuple[int, ...]:
  manifest = json.loads(Path(motion_file).with_suffix(".manifest.json").read_text())
  return tuple(i for i, clip in enumerate(manifest["clips"]) if clip["context"] == "stand" and not clip["mirrored"])


def stream_env_cfg(play: bool = False, clean: bool | None = None, subset: str = "core",
                   plain_actions: bool = False, matched: bool = False, teacher: str | None = None) -> ManagerBasedRlEnvCfg:
  """plain_actions: joint position targets around the default pose for all joints (comparison policies that do not
  use the arm-intent action). matched: keep a state-matched reference next to the stream (MatchedStreamCommand);
  with `teacher` the frozen tracker's label on that reference is available as the "bc" observation group."""
  cfg = student_env_cfg(play=play, clean=clean, subset=subset, teacher=teacher if matched else None)
  old = cfg.commands["motion"]
  values = {field.name: getattr(old, field.name) for field in fields(old)}
  values.update(frozen=not matched, sampling_mode="uniform")
  if play:
    values["clip_list"] = stand_clips(old.motion_file)
  cfg.commands = {"motion": (MatchedStreamCommandCfg if matched else GoalStreamCommandCfg)(**values)}
  if plain_actions:
    cfg.actions = {"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",),
                                                       scale=dict(robot.ACTION_SCALE), use_default_offset=True)}
  name = {"command_name": "motion"}
  keep = ("action_rate_l2", "joint_limit", "self_collisions", "goal_twist", "goal_twist_l1", "goal_lag", "goal_height",
          "goal_pitch", "arm_deviation")
  cfg.rewards = {key: value for key, value in cfg.rewards.items() if key in keep}
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "trunk_ground": TerminationTermCfg(func=wb.body_on_ground, params={"sensor_name": "trunk_ground"}),
    "fallen": TerminationTermCfg(func=fallen),
  }
  critic = {
    "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}),
    "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
    "goal": ObservationTermCfg(func=obs.goal, params=name),
    "arm_intent": ObservationTermCfg(func=obs.arm_intent, params=name),
    "goal_lag": ObservationTermCfg(func=obs.goal_lag, params=name),
    "root_height": ObservationTermCfg(func=root_height),
    "foot_forces": ObservationTermCfg(func=foot_forces),
  }
  cfg.observations["critic"] = ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False)
  if not matched:
    cfg.observations.pop("bc", None)
  cfg.episode_length_s = 1.0e9 if play else 20.0
  return cfg
