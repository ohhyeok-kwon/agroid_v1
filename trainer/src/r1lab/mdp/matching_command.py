"""Closed-loop motion-matching reference command for the teacher (mjlab).

The reference is produced every policy tick by MotionMatcher from a sampled
user command (vx, vy, wz) and the robot's actual feet/root state. Resets
place the robot on a library frame (a share on standing idle frames) and
start the matcher there. The reference never ends; episodes end on
tracking failure or time-out. Phase inputs are zero for actor and critic.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
import torch

from r1lab.mdp import motion_matching as mm
from r1lab.mdp.motion_teacher import JOINT_VELOCITY_SCALE, TeacherMotionCommand, TeacherMotionCommandCfg

# (probability, kind, vx range, wz range); vy stays zero (no free-arm lateral data).
COMMAND_MIX = (
    (0.25, "stand", (0.0, 0.0), (0.0, 0.0)),
    (0.35, "forward", (0.15, 0.30), (0.0, 0.0)),
    (0.15, "forward_turn", (0.15, 0.25), (0.2, 0.4)),
    (0.15, "heading", (0.0, 0.0), (math.pi / 2, math.pi)),
    (0.10, "backward", (-0.20, -0.10), (0.0, 0.0)),
)


def idle_candidates(db, clip_names, segment_clip):
    """Matching candidates on still standing-idle captures (StandIdle); shared with Simulation."""
    segment_clip = torch.as_tensor(segment_clip, device=db.candidates.device)
    idle_clips = [i for i, name in enumerate(clip_names) if "StandIdle" in name]
    idle_segments = torch.nonzero(torch.isin(segment_clip, torch.as_tensor(idle_clips, device=segment_clip.device))).flatten()
    frames = db.candidates[torch.isin(db.frame_segment[db.candidates], idle_segments)]
    if len(frames) == 0:
        raise ValueError("Matching teacher needs standing idle frames in the library")
    return frames


@dataclass(kw_only=True)
class MatchingMotionCommandCfg(TeacherMotionCommandCfg):
    command_mix: tuple = COMMAND_MIX
    command_duration_s: tuple[float, float] = (2.0, 6.0)
    stand_start_fraction: float = 0.5
    matcher: dict = field(default_factory=lambda: dict(weights=(1.0, 3.0, 0.5, 0.25, 0.25),
                                                      min_dwell_frames=30, max_heading_correction=0.35,
                                                      heading_gain=3.0, heading_filter_s=0.1))
    yaw_rate_filter_s: float = 0.5
    # (metres, radians) re-anchor thresholds for 1.5 s after a ball contact
    # (r1lab.mdp.projectiles.ball_contact_until); None keeps the matcher's thresholds. Pushes do
    # not count: they are recoverable inside the 0.5 m termination and are frequent enough that
    # counting them would suspend that termination for about half of training.
    disturbed_reanchor: tuple | None = None
    heading_gain: float = 3.0
    heading_rate_limit: float = 1.0

    def build(self, env):
        return MatchingMotionCommand(self, env)


class MatchingMotionCommand(TeacherMotionCommand):
    cfg: MatchingMotionCommandCfg

    def __init__(self, cfg, env):
        if cfg.start_frame_mixture:
            raise ValueError("Matching teachers sample their own start frames")
        super().__init__(cfg, env)
        self.matcher = mm.MotionMatcher(cfg.motion_file, self.num_envs, self.device, **cfg.matcher)
        db = self.matcher.db
        with np.load(cfg.motion_file, allow_pickle=False) as data:
            names = [str(n) for n in data["clip_names"]]
            clip = data["segment_clip"]
        self.idle_candidates = idle_candidates(db, names, clip)
        probs = torch.tensor([m[0] for m in cfg.command_mix], device=self.device)
        self.mix_prob = probs / probs.sum()
        self.mix_vx = torch.tensor([m[2] for m in cfg.command_mix], device=self.device)
        self.mix_wz = torch.tensor([m[3] for m in cfg.command_mix], device=self.device)
        self.mix_heading = torch.tensor([m[1] == "heading" for m in cfg.command_mix], device=self.device)
        # Optional 5th element: lateral speed range (sign drawn at random). Mixes without it keep
        # vy = 0 and draw no extra random numbers (same command stream as before).
        self.mix_vy = torch.tensor([m[4] if len(m) > 4 else (0.0, 0.0) for m in cfg.command_mix], device=self.device)
        self.has_lateral = bool((self.mix_vy != 0).any())
        self.user_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.heading_mode = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.command_kind = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command_left = torch.zeros(self.num_envs, device=self.device)
        # Gait-averaged pelvis yaw rate: the instantaneous rate carries the +-0.5 rad/s
        # pelvis sway of walking and would bias the search toward turning frames.
        self.yaw_rate_filtered = torch.zeros(self.num_envs, device=self.device)
        self.ref = self.matcher.reference()
        for name in ("reanchor_rate", "switch_rate", "command_vx", "command_wz", "heading_error",
                     "idle_while_moving", "realized_speed_ratio"):
            self.metrics["mm_" + name] = torch.zeros(self.num_envs, device=self.device)
        if cfg.disturbed_reanchor is not None:
            self.metrics["mm_disturbed"] = torch.zeros(self.num_envs, device=self.device)
        # Stationary-completion detectors: idle reference frames while asked to move, and
        # realized/commanded planar speed (EMA) for moving commands.
        self.idle_frame = torch.zeros(db.frames, dtype=torch.bool, device=self.device)
        self.idle_frame[self.idle_candidates] = True
        self.speed_ema = torch.zeros(self.num_envs, device=self.device)

    # ------------------------------------------------------------ properties
    @property
    def joint_pos(self):
        return self.ref["joint_pos"]

    @property
    def joint_vel(self):
        return self.ref["joint_vel"]

    @property
    def body_pos_w(self):
        return self.ref["body_pos_w"] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self):
        return self.ref["body_quat_w"]

    @property
    def body_lin_vel_w(self):
        return self.ref["body_lin_vel_w"]

    @property
    def body_ang_vel_w(self):
        return self.ref["body_ang_vel_w"]

    @property
    def anchor_pos_w(self):
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self):
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self):
        return self.body_lin_vel_w[:, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self):
        return self.body_ang_vel_w[:, self.motion_anchor_body_index]

    # ------------------------------------------------------------ commands
    def _sample_commands(self, env_ids):
        n = len(env_ids)
        kind = torch.multinomial(self.mix_prob, n, replacement=True)
        u = torch.rand(n, device=self.device)
        vx = self.mix_vx[kind, 0] + u * (self.mix_vx[kind, 1] - self.mix_vx[kind, 0])
        u = torch.rand(n, device=self.device)
        wz = self.mix_wz[kind, 0] + u * (self.mix_wz[kind, 1] - self.mix_wz[kind, 0])
        wz = wz * torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        heading = self.mix_heading[kind]
        self.command_kind[env_ids] = kind
        self.heading_mode[env_ids] = heading
        self.heading_target[env_ids] = mm.wrap(self.matcher.facing[env_ids] + torch.where(heading, wz, 0.0))
        self.user_command[env_ids, 0] = vx
        if self.has_lateral:
            u = torch.rand(n, device=self.device)
            vy = self.mix_vy[kind, 0] + u * (self.mix_vy[kind, 1] - self.mix_vy[kind, 0])
            self.user_command[env_ids, 1] = vy * torch.where(torch.rand(n, device=self.device) < 0.5, -1.0, 1.0)
        else:
            self.user_command[env_ids, 1] = 0.0
        self.user_command[env_ids, 2] = torch.where(heading, 0.0, wz)
        low, high = self.cfg.command_duration_s
        self.command_left[env_ids] = low + torch.rand(n, device=self.device) * (high - low)

    def current_command(self):
        """User command with heading targets converted to a limited yaw rate (UI semantics)."""
        command = self.user_command.clone()
        error = mm.wrap(self.heading_target - self.matcher.facing)
        rate = (self.cfg.heading_gain * error).clamp(-self.cfg.heading_rate_limit, self.cfg.heading_rate_limit)
        rate = torch.where(error.abs() < 0.025, torch.zeros_like(rate), rate)
        command[:, 2] = torch.where(self.heading_mode, rate, command[:, 2])
        return command

    # ------------------------------------------------------------ robot features
    def robot_features(self):
        origin = self._env.scene.env_origins[:, :2]
        facing = self.matcher.facing
        pos = self.robot_body_pos_w
        root = pos[:, 0, :2] - origin
        feet = mm.rotate_z(pos[:, list(mm.FEET), :2] - pos[:, :1, :2], -facing[:, None].expand(-1, 2))
        vel = mm.rotate_z(self.robot_body_lin_vel_w[:, list(mm.FEET), :2], -facing[:, None].expand(-1, 2))
        return {"feet_xy": feet, "feet_velocity_xy": vel,
                "root_velocity": mm.rotate_z(self.robot_body_lin_vel_w[:, 0, :2], -facing),
                "yaw_rate": self.yaw_rate_filtered, "root_xy": root,
                "pelvis_yaw": mm.yaw_of(self.robot_body_quat_w[:, 0])}

    # ------------------------------------------------------------ resets / steps
    def _resample_command(self, env_ids):
        n = len(env_ids)
        db = self.matcher.db
        stand = torch.rand(n, device=self.device) < self.cfg.stand_start_fraction
        idle = self.idle_candidates[torch.randint(len(self.idle_candidates), (n,), device=self.device)]
        anyf = db.candidates[torch.randint(len(db.candidates), (n,), device=self.device)]
        frames = torch.where(stand, idle, anyf)
        self.time_steps[env_ids] = frames
        # Robot state from the archived frame (MotionCommand's reset path, with pose noise).
        self._write_frame(env_ids, frames)
        self.matcher.reset(env_ids, frames, db.root_xy_world[frames], db.facing[frames])
        self.yaw_rate_filtered[env_ids] = db.facing_rate[frames]
        self._sample_commands(env_ids)
        self.ref = self.matcher.reference()

    def _write_frame(self, env_ids, frames):
        from mjlab.utils.lab_api.math import quat_from_euler_xyz, quat_mul, sample_uniform
        root_pos = self.motion.body_pos_w[frames, 0].clone() + self._env.scene.env_origins[env_ids]
        root_ori = self.motion.body_quat_w[frames, 0].clone()
        ranges = torch.tensor([self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
                              device=self.device)
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += noise[:, :3]
        root_ori = quat_mul(quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), root_ori)
        joint_pos = self.motion.joint_pos[frames].clone()
        joint_pos += sample_uniform(self.cfg.joint_position_range[0], self.cfg.joint_position_range[1],
                                    joint_pos.shape, device=self.device)
        self._write_reference_state_to_sim(env_ids, root_pos, root_ori, self.motion.body_lin_vel_w[frames, 0],
                                           self.motion.body_ang_vel_w[frames, 0], joint_pos,
                                           self.motion.joint_vel[frames])
        self._pending_forward = True

    def _update_command(self, env_ids=None):
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
        command = self.current_command()
        limits = {}
        if self.cfg.disturbed_reanchor is not None:
            from r1lab.mdp.projectiles import ball_contact_until
            projectiles = getattr(self._env, "_r1_projectile_event", None)
            if projectiles is not None:
                projectiles.observe_contacts()
            disturbed = self._env.common_step_counter <= ball_contact_until(self._env)
            limit_m, limit_rad = self.cfg.disturbed_reanchor
            limits = dict(reanchor_m=torch.where(disturbed, limit_m, self.matcher.reanchor_m),
                          reanchor_rad=torch.where(disturbed, limit_rad, self.matcher.reanchor_rad))
            self.metrics["mm_disturbed"][:] = disturbed.float()
        self.ref = self.matcher.step(command, self.robot_features(), dt=dt, active=active,
                                     heading_target=self.heading_target, heading_mode=self.heading_mode, **limits)
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


def matching_future_reference(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    rows = []
    for joint_pos, joint_vel in command.ref["future"]:
        rows.extend((joint_pos, joint_vel * JOINT_VELOCITY_SCALE))
    return torch.cat(rows, dim=-1)


def matching_reference_end(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return torch.zeros(command.num_envs, dtype=torch.bool, device=command.device)
