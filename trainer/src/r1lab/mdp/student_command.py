"""Student command for matching-teacher distillation (810-input student, 233-input teacher).

The user command is processed once per policy tick exactly like the Simulation careful_walk path
(app/operation_center/simulation_worker.py, Simulation.command): rate commands (vx, vy, wz) as
sampled; heading targets become wz = clip(3*wrap(target - pelvis yaw), +-1 rad/s), zero inside a
.025 rad dead band; then a per-tick slew of (.025 m/s, .025 m/s, .08 rad/s). This is the student's
command input (frame_stack reads the "twist" view). The motion matcher (label generator) receives
the same user command with the teacher task's own semantics: step changes, heading rate from its
facing, absolute heading target (Gate 0 against the teacher's scripted results; matcher_yaw=None
restores the single shared tensor of plan_d1, matcher_slew=True shares only the slew).

Resets follow the teacher (idle library frames or any library frame), plus an optional share at the
R1 default standing pose (DEFAULT_POS, random yaw) with the matcher started on the idle frame
nearest to that pose, facing = yaw - sway (as the Simulation's reference teacher does).
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from r1lab import contract as C
from r1lab.mdp import motion_matching as mm
from r1lab.mdp.matching_command import MatchingMotionCommand, MatchingMotionCommandCfg

SLEW = (0.025, 0.025, 0.08)   # per policy tick: vx, vy (m/s), wz (rad/s)
HEADING_GAIN = 3.0
HEADING_RATE_LIMIT = 1.0      # careful_walk heading turn rate (rad/s)
HEADING_DEADBAND = 0.025      # rad
START_LIBRARY, START_DEFAULT = 0, 1
DEFAULT_HEIGHT = C.STAND_HEIGHT + 0.01   # R1 init_state and Simulation.reset pelvis height


def process_rate_command(previous, desired, heading_mode, heading_target, yaw, *, heading_gain=HEADING_GAIN,
                         rate_limit=HEADING_RATE_LIMIT, deadband=HEADING_DEADBAND, slew=SLEW):
    """One Simulation.command tick for a batch.

    previous [n,3] processed command of the last tick; desired [n,3] user (vx, vy, wz); heading_mode
    [n] bool replaces wz by the heading-error rate of heading_target [n] against yaw [n] (rad).
    Returns previous + clip(target - previous, -slew, slew); slew=None returns the target (no ramp).
    """
    error = mm.wrap(heading_target - yaw)
    rate = (heading_gain * error).clamp(-rate_limit, rate_limit)
    rate = torch.where(error.abs() < deadband, torch.zeros_like(rate), rate)
    target = torch.cat((desired[:, :2], torch.where(heading_mode, rate, desired[:, 2])[:, None]), dim=-1)
    if slew is None:
        return target
    step = torch.as_tensor(slew, dtype=previous.dtype, device=previous.device)
    return previous + torch.clamp(target - previous, min=-step, max=step)


@dataclass(kw_only=True)
class StudentMatchingCommandCfg(MatchingMotionCommandCfg):
    slew: tuple[float, float, float] | None = SLEW  # None: step commands (teacher-env semantics)
    heading_deadband: float = HEADING_DEADBAND
    # Share of resets at the R1 default standing pose (random yaw); the others follow the teacher's
    # reset rule (stand_start_fraction idle frames, otherwise any library frame).
    default_start_fraction: float = 0.0
    # Yaw the student's heading error is measured on: "pelvis" (Simulation and robot) or "reference"
    # (matcher facing; diagnostic only, a deployed student has no reference).
    heading_yaw: str = "pelvis"
    # Command of the matcher (label generator). None: the student's processed tensor itself (plan_d1).
    # "reference" or "pelvis": processed separately, heading targets converted against that yaw.
    # Gate 0 (I8): with the pelvis yaw the human turning sway (pelvis up to +-23 deg off the facing)
    # reverses the heading rate mid-turn and a +90 deg turn peaks at 131 deg; with the reference facing
    # it peaks at 96 deg (teacher's own run 95).
    matcher_yaw: str | None = "reference"
    # Slew of a separately processed matcher command. False: step commands as in the teacher task
    # (Gate 0: the slewed matcher changed rate-turn clip choices, default-start turns -141 vs -133 deg
    # across command timings); the student still observes the slewed command.
    matcher_slew: bool = False
    # "rate": heading commands steer the matcher only through its processed wz. "target": the matcher
    # also keeps the absolute heading target (teacher-env semantics).
    matcher_heading: str = "target"
    # Heading hold on the student's command only (runtime-computable from the IMU yaw): while the user's
    # yaw rate is zero (rate commands, standing included), the heading at that moment becomes a heading
    # target, so drift reaches the student as a small wz (gain/dead band as heading commands). The target
    # follows the yaw for the first hold_settle_s (residual rotation after a turn), then stays fixed.
    # The matcher (label generator) keeps its own heading semantics. D1 drifted -22 deg in stand20 and up
    # to 13 deg per 10 s walking straight because the reference heading it was labelled from is hidden.
    hold_heading: bool = False
    hold_settle_s: float = 0.5

    def build(self, env):
        return StudentMatchingCommand(self, env)


class StudentMatchingCommand(MatchingMotionCommand):
    cfg: StudentMatchingCommandCfg

    def __init__(self, cfg, env):
        if cfg.matcher_heading not in ("rate", "target"):
            raise ValueError("matcher_heading must be 'rate' or 'target'")
        if cfg.heading_yaw not in ("pelvis", "reference") or cfg.matcher_yaw not in (None, "pelvis", "reference"):
            raise ValueError("heading_yaw must be 'pelvis' or 'reference'; matcher_yaw also None")
        if not 0.0 <= cfg.default_start_fraction <= 1.0:
            raise ValueError("default_start_fraction must be within [0, 1]")
        if cfg.disturbed_reanchor is not None:
            raise ValueError("The student scene has no ball disturbances to re-anchor after")
        super().__init__(cfg, env)
        n = self.num_envs
        self.processed = torch.zeros(n, 3, device=self.device)        # student input (twist view)
        self.matcher_command = torch.zeros(n, 3, device=self.device)  # label generator input
        self.since_change = torch.zeros(n, device=self.device)
        self.hold_target = torch.zeros(n, device=self.device)
        self.hold_age = torch.full((n,), -1.0, device=self.device)  # < 0: no hold
        self.start_mode = torch.zeros(n, dtype=torch.long, device=self.device)
        # Evaluation tools may pin each env's start: START_LIBRARY or START_DEFAULT per env.
        self.start_override: torch.Tensor | None = None
        self.kind_names = tuple(m[1] for m in cfg.command_mix)
        default = self.robot.data.default_joint_pos[0]
        distance = (self.matcher.db.joint_pos[self.idle_candidates] - default).square().sum(-1)
        self.default_frame = self.idle_candidates[int(distance.argmin())]
        self.metrics["start_default"] = torch.zeros(n, device=self.device)

    # ------------------------------------------------------------ commands
    def current_command(self):
        """The student's processed command of this tick (the twist view); no side effects."""
        return self.processed.clone()

    def heading_yaw(self, source=None):
        if (source or self.cfg.heading_yaw) == "reference":
            return self.matcher.facing
        return mm.yaw_of(self.robot_body_quat_w[:, 0])

    def _process(self, previous, source, slew, hold=False):
        mode, target = self.heading_mode, self.heading_target
        if hold:
            held = self.hold_age >= 0
            mode, target = mode | held, torch.where(held, self.hold_target, target)
        return process_rate_command(previous, self.user_command, mode, target,
                                    self.heading_yaw(source), heading_gain=self.cfg.heading_gain,
                                    rate_limit=self.cfg.heading_rate_limit, deadband=self.cfg.heading_deadband,
                                    slew=slew)

    def _update_hold(self, active, dt):
        """Arm, follow (settle window) and freeze the student's heading-hold target."""
        yaw = self.heading_yaw(None)
        wants = active & ~self.heading_mode & (self.user_command[:, 2].abs() < 1e-6)
        arm = wants & (self.hold_age < 0)
        self.hold_age = torch.where(wants, torch.where(arm, torch.zeros_like(self.hold_age), self.hold_age + dt),
                                    torch.where(active, torch.full_like(self.hold_age, -1.0), self.hold_age))
        settling = wants & (self.hold_age < self.cfg.hold_settle_s)
        self.hold_target = torch.where(settling, yaw, self.hold_target)

    def desired_command(self):
        """This tick's unslewed student target (heading targets already converted to a rate)."""
        return self._process(self.processed, None, None)

    def _sample_commands(self, env_ids):
        before = (self.user_command[env_ids].clone(), self.heading_mode[env_ids].clone(),
                  self.heading_target[env_ids].clone())
        super()._sample_commands(env_ids)
        changed = ((self.user_command[env_ids] - before[0]).abs().amax(-1) > 1e-6) | (
            self.heading_mode[env_ids] != before[1]) | (
            self.heading_mode[env_ids] & ((self.heading_target[env_ids] - before[2]).abs() > 1e-6))
        self.since_change[env_ids[changed]] = 0.0

    # ------------------------------------------------------------ resets
    def _resample_command(self, env_ids):
        n = len(env_ids)
        if self.start_override is not None:
            default = self.start_override[env_ids] == START_DEFAULT
        elif self.cfg.default_start_fraction >= 1.0:
            default = torch.ones(n, dtype=torch.bool, device=self.device)
        elif self.cfg.default_start_fraction > 0.0:
            default = torch.rand(n, device=self.device) < self.cfg.default_start_fraction
        else:  # no extra random draw: identical stream to the teacher task
            default = torch.zeros(n, dtype=torch.bool, device=self.device)
        if bool((~default).any()):
            super()._resample_command(env_ids[~default])
        if bool(default.any()):
            self._reset_default_pose(env_ids[default])
        self.start_mode[env_ids] = default.long()
        self.metrics["start_default"][env_ids] = default.float()
        self.processed[env_ids] = 0.0
        self.matcher_command[env_ids] = 0.0
        self.since_change[env_ids] = 0.0
        self.hold_age[env_ids] = -1.0  # re-arm on the new pose's heading

    def _reset_default_pose(self, env_ids):
        """R1 DEFAULT_POS standing start with a random yaw (the deployed entry state)."""
        from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform
        n = len(env_ids)
        origins = self._env.scene.env_origins[env_ids]
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch")],
                              device=self.device)
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (n, 5), device=self.device)
        yaw = (2.0 * torch.rand(n, device=self.device) - 1.0) * math.pi
        root_pos = origins + noise[:, :3]
        root_pos[:, 2] += DEFAULT_HEIGHT
        root_ori = quat_from_euler_xyz(noise[:, 3], noise[:, 4], yaw)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_pos += sample_uniform(self.cfg.joint_position_range[0], self.cfg.joint_position_range[1],
                                    joint_pos.shape, device=self.device)
        zeros = torch.zeros(n, 3, device=self.device)
        self._write_reference_state_to_sim(env_ids, root_pos, root_ori, zeros, zeros.clone(), joint_pos,
                                           torch.zeros_like(joint_pos))
        self._pending_forward = True
        db = self.matcher.db
        frames = self.default_frame.expand(n).clone()
        self.time_steps[env_ids] = frames
        facing = mm.wrap(mm.yaw_of(root_ori) - db.sway[frames])
        self.matcher.reset(env_ids, frames, root_pos[:, :2] - origins[:, :2], facing)
        self.yaw_rate_filtered[env_ids] = db.facing_rate[frames]
        self._sample_commands(env_ids)
        self.ref = self.matcher.reference()

    # ------------------------------------------------------------ step
    def _update_command(self, env_ids=None):
        """MatchingMotionCommand._update_command with the processed command (once per tick)."""
        dt = self._env.step_dt
        active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if env_ids is not None:
            active = torch.zeros_like(active)
            active[env_ids] = True
        if not bool(active.any()):
            self.update_relative_body_poses()
            return
        self.command_left = torch.where(active, self.command_left - dt, self.command_left)
        expired = torch.nonzero(active & (self.command_left <= 0)).flatten()
        if len(expired):
            self._sample_commands(expired)
        alpha = 1.0 - math.exp(-dt / max(self.cfg.yaw_rate_filter_s, 1e-3))
        filtered = self.yaw_rate_filtered + alpha * (self.robot_body_ang_vel_w[:, 0, 2] - self.yaw_rate_filtered)
        self.yaw_rate_filtered = torch.where(active, filtered, self.yaw_rate_filtered)
        if self.cfg.hold_heading:
            self._update_hold(active, dt)
        processed = self._process(self.processed, None, self.cfg.slew, hold=self.cfg.hold_heading)
        matched = processed if self.cfg.matcher_yaw is None else self._process(
            self.matcher_command, self.cfg.matcher_yaw, self.cfg.slew if self.cfg.matcher_slew else None)
        self.processed = torch.where(active[:, None], processed, self.processed)
        self.matcher_command = torch.where(active[:, None], matched, self.matcher_command)
        self.since_change = torch.where(active, self.since_change + dt, self.since_change)
        command = self.matcher_command
        heading = (dict(heading_target=self.heading_target, heading_mode=self.heading_mode)
                   if self.cfg.matcher_heading == "target" else {})
        self.ref = self.matcher.step(command, self.robot_features(), dt=dt, active=active, **heading)
        steps = self.matcher.steps.clamp(min=1)
        self.metrics["mm_reanchor_rate"][:] = self.matcher.reanchors / steps
        self.metrics["mm_switch_rate"][:] = self.matcher.switches / steps
        self.metrics["mm_command_vx"][:] = command[:, 0]
        self.metrics["mm_command_wz"][:] = command[:, 2]
        self.metrics["mm_heading_error"][:] = mm.wrap(self.matcher.command_heading - self.matcher.facing).abs()
        asked = command[:, :2].norm(dim=-1)
        moving = asked > 0.1
        self.metrics["mm_idle_while_moving"][:] = (self.idle_frame[self.matcher.play] & moving).float()
        speed = self.robot_body_lin_vel_w[:, 0, :2].norm(dim=-1)
        self.speed_ema = self.speed_ema + 0.02 * (speed - self.speed_ema)
        self.metrics["mm_realized_speed_ratio"][:] = torch.where(moving, self.speed_ema / asked.clamp(min=1e-3),
                                                                  torch.zeros_like(asked))
        self.update_relative_body_poses()


@dataclass(kw_only=True)
class StudentCommandViewCfg(CommandTermCfg):
    """Read-only (vx, vy, wz) view of the motion term's processed command for frame_stack."""
    source: str = "motion"
    resampling_time_range: tuple[float, float] = (1e9, 1e9)

    def build(self, env):
        return StudentCommandView(self, env)


class StudentCommandView(CommandTerm):
    """No state and no random draws: the source term processes the command once per tick."""
    cfg: StudentCommandViewCfg

    @property
    def command(self):
        return self._env.command_manager.get_term(self.cfg.source).processed

    def reset(self, env_ids=None):
        return {}

    def compute(self, dt, env_ids=None):
        pass

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids):
        pass

    def _update_command(self, env_ids=None):
        pass


def command_diagnostics(env, command_name="motion"):
    """[command kind index (COMMAND_MIX order), seconds since the command last changed]."""
    command = env.command_manager.get_term(command_name)
    return torch.stack((command.command_kind.float(), command.since_change), dim=-1)
