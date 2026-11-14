"""Batched motion matching over a reference library (torch, any batch size).

One implementation serves training (thousands of envs) and the Simulation
worker (batch 1). Every frame comes from real captures; transitions crossfade
joint angles and root-relative body poses while the reference root stays
continuous.

Frames: a smoothed facing yaw (1 s centered mean of the pelvis yaw inside each
segment) defines the heading frame, so the natural pelvis yaw sway of the gait
stays in the motion while steering acts on the facing direction only.

Contracts (from the design review):
- The reference root integrates the selected frames' facing-frame velocity;
  heading additionally follows the integral of the commanded yaw rate through
  a weak correction. Forward/lateral speed is whatever the frames carry.
- Search features combine the command trajectory (future facing-frame root
  positions and headings) with the robot's ACTUAL feet and root velocity, so
  a branch is a function of observable state.
- The reference never chases the robot; it is re-anchored only beyond
  reanchor_m / reanchor_rad and every re-anchor is counted.
"""
from __future__ import annotations

import math

import numpy as np
import torch

FPS = 50
FEET = (6, 12)  # left/right ankle_roll_link in the teacher BODY_NAMES order


def wrap(angle):
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


def yaw_of(quat):
    w, x, y, z = quat.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def yaw_quat(yaw):
    half = 0.5 * yaw
    zero = torch.zeros_like(half)
    return torch.stack((torch.cos(half), zero, zero, torch.sin(half)), dim=-1)


def quat_mul(a, b):
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw), dim=-1)


def rotate_z(vector, yaw):
    """Rotate the xy components of [..., 3] (or [..., 2]) vectors by yaw about z."""
    c, s = torch.cos(yaw), torch.sin(yaw)
    while c.dim() < vector.dim() - 1:
        c, s = c.unsqueeze(-1), s.unsqueeze(-1)
    x, y = vector[..., 0], vector[..., 1]
    rotated = torch.stack((c * x - s * y, s * x + c * y), dim=-1)
    if vector.shape[-1] == 2:
        return rotated
    return torch.cat((rotated, vector[..., 2:]), dim=-1)


class MotionDatabase:
    """Facing-frame library tables and normalized search features."""

    def __init__(self, path, device="cpu", include_heldout=False, horizons=(10, 20, 30),
                 facing_window=50, exclude_clips=(), still_clips=("StandIdle",)):
        with np.load(path, allow_pickle=False) as data:
            arrays = {k: torch.as_tensor(np.asarray(data[k]), device=device)
                      for k in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                                "body_lin_vel_w", "body_ang_vel_w", "segment_start",
                                "segment_end", "segment_heldout", "frame_segment")}
            # Original capture identity (a mirrored clip shares its source's id) and source
            # frame, so the no-rewind window also covers the mirrored twin of a clip.
            names = [str(n) for n in data["clip_names"]] if "clip_names" in data.files else []
            segment_clip = np.asarray(data["segment_clip"]) if "segment_clip" in data.files else None
            source_frame = np.asarray(data["source_frame"]) if "source_frame" in data.files else None
        self.device = device
        self.horizons = tuple(int(h) for h in horizons)
        f32 = torch.float32
        self.joint_pos = arrays["joint_pos"].to(f32)
        self.joint_vel = arrays["joint_vel"].to(f32)
        pos, quat = arrays["body_pos_w"].to(f32), arrays["body_quat_w"].to(f32)
        lin, ang = arrays["body_lin_vel_w"].to(f32), arrays["body_ang_vel_w"].to(f32)
        start, end = arrays["segment_start"].long(), arrays["segment_end"].long()
        self.segment_start, self.segment_end = start, end
        self.frame_segment = arrays["frame_segment"].long()
        self.frames = len(self.joint_pos)
        frame_end = end[self.frame_segment]
        self.frame_end = frame_end
        # Smoothed facing yaw inside each segment (centered window, clamped at ends).
        pelvis_yaw = yaw_of(quat[:, 0])
        facing = torch.empty_like(pelvis_yaw)
        half = facing_window // 2
        for a, b in zip(start.tolist(), end.tolist()):
            raw = torch.as_tensor(np.unwrap(pelvis_yaw[a:b].cpu().numpy()), device=device, dtype=f32)
            padded = torch.cat((raw[:1].repeat(half), raw, raw[-1:].repeat(half)))
            kernel = torch.ones(1, 1, 2 * half + 1, device=device) / (2 * half + 1)
            facing[a:b] = torch.nn.functional.conv1d(padded[None, None], kernel)[0, 0]
        self.facing = facing
        self.sway = wrap(pelvis_yaw - facing)
        root = pos[:, 0]
        inverse = -facing
        self.root_xy_world = root[:, :2].clone()
        self.root_height = root[:, 2]
        rel = pos - torch.cat((root[:, None, :2], torch.zeros_like(root[:, None, :1])), dim=-1)
        self.rel_pos = rotate_z(rel, inverse[:, None].expand(-1, pos.shape[1]))
        heading = yaw_quat(inverse)[:, None].expand(-1, pos.shape[1], 4)
        self.rel_quat = quat_mul(heading, quat)
        self.rel_lin_vel = rotate_z(lin, inverse[:, None].expand(-1, pos.shape[1]))
        self.rel_ang_vel = rotate_z(ang, inverse[:, None].expand(-1, pos.shape[1]))
        # Facing-frame root planar velocity and facing yaw rate (causal, within segments).
        self.root_velocity = rotate_z(lin[:, 0, :2], inverse)
        rate = torch.zeros_like(facing)
        rate[1:] = (facing[1:] - facing[:-1]) * FPS
        first = start.clamp(max=self.frames - 1)
        rate[first] = torch.where(first + 1 < end, rate[(first + 1).clamp(max=self.frames - 1)], rate[first])
        self.facing_rate = rate
        # Candidate frames: training segments with every horizon (and +10 obs) inside.
        heldout = arrays["segment_heldout"].bool()[self.frame_segment]
        lookahead = max(max(self.horizons), 10)
        index = torch.arange(self.frames, device=device)
        allowed = (index + lookahead < frame_end)
        if not include_heldout:
            allowed &= ~heldout
        if exclude_clips and names and segment_clip is not None:
            # e.g. "IdleWaiting": idle with weight shifts and small turns is not a stand target.
            banned = [i for i, n in enumerate(names) if any(x in n for x in exclude_clips)]
            if banned:
                frame_clip = torch.as_tensor(segment_clip, device=device)[self.frame_segment]
                allowed &= ~torch.isin(frame_clip, torch.as_tensor(banned, device=device))
        self.candidates = torch.nonzero(allowed).flatten()
        if len(self.candidates) == 0:
            raise ValueError("No motion-matching candidates")
        self.candidate_features = self.frame_features(self.candidates)
        self.candidate_segment = self.frame_segment[self.candidates]
        if names and segment_clip is not None and source_frame is not None:
            base = [n.replace("_M.npz", ".npz") for n in names]
            ids = {b: i for i, b in enumerate(dict.fromkeys(base))}
            clip_base = torch.as_tensor([ids[b] for b in base], device=device)
            self.frame_base = clip_base[torch.as_tensor(segment_clip, device=device)[self.frame_segment]]
            self.frame_source = torch.as_tensor(source_frame, device=device).long()
        else:  # fall back to segment identity and library frame index
            self.frame_base = self.frame_segment.clone()
            self.frame_source = torch.arange(self.frames, device=device)
        self.candidate_base = self.frame_base[self.candidates]
        self.candidate_source = self.frame_source[self.candidates]
        still = torch.zeros(len(self.candidates), dtype=torch.bool, device=device)
        if names and segment_clip is not None and still_clips:
            kept = [i for i, n in enumerate(names) if any(x in n for x in still_clips)]
            if kept:
                frame_clip = torch.as_tensor(segment_clip, device=device)[self.frame_segment]
                still = torch.isin(frame_clip[self.candidates], torch.as_tensor(kept, device=device))
        self.candidate_still = still

    def trajectory_features(self, frames):
        """Future facing-frame root offsets and heading changes for library frames."""
        offsets, headings = [], []
        for horizon in self.horizons:
            ahead = (frames + horizon).clamp(max=self.frames - 1)
            delta = self.world_root_xy(ahead) - self.world_root_xy(frames)
            offsets.append(rotate_z(delta, -self.facing[frames]))
            headings.append(wrap(self.facing[ahead] - self.facing[frames])[..., None])
        return torch.cat(offsets + headings, dim=-1)

    def world_root_xy(self, frames):
        return self.root_xy_world[frames]

    def pose_features(self, feet_xy, feet_velocity_xy, root_velocity, yaw_rate):
        return torch.cat((feet_xy.flatten(-2), feet_velocity_xy.flatten(-2), root_velocity,
                          yaw_rate[..., None]), dim=-1)

    def frame_features(self, frames):
        traj = self.trajectory_features(frames)
        pose = self.pose_features(self.rel_pos[frames][:, FEET, :2], self.rel_lin_vel[frames][:, FEET, :2],
                                  self.root_velocity[frames], self.facing_rate[frames])
        return torch.cat((traj, pose), dim=-1)


class MotionMatcher:
    """Per-env reference generator. Call reset() then step() once per policy tick."""

    # Fixed feature scales: [traj offset m, heading rad, feet pos m, feet vel m/s, root vel m/s, yaw rate rad/s]
    SCALES = (0.1, 0.3, 0.05, 0.2, 0.1, 0.3)

    def __init__(self, path, num_envs, device="cpu", *, include_heldout=False, search_interval=10,
                 blend_frames=10, horizons=(20, 40, 60), weights=(1.0, 1.0, 0.5, 0.25, 0.25), heading_gain=1.0,
                 heading_filter_s=1.0, reanchor_m=0.3, reanchor_rad=0.3, continuation_bonus=0.15,
                 command_time_constant=0.3, no_rewind_frames=60, chunk=4096, max_heading_correction=0.2,
                 min_dwell_frames=30, pose_source="reference", exclude_clips=("IdleWaiting",),
                 settle_s=0.5, settle_anchor_s=0.5, hold_settled_heading=False):
        self.db = MotionDatabase(path, device=device, include_heldout=include_heldout, horizons=horizons,
                                 exclude_clips=tuple(exclude_clips))
        self.num_envs, self.device = num_envs, device
        self.search_interval, self.blend_frames = int(search_interval), int(blend_frames)
        self.heading_gain = float(heading_gain)
        self.heading_alpha = 1.0 / max(1.0, heading_filter_s * FPS)
        self.reanchor_m, self.reanchor_rad = float(reanchor_m), float(reanchor_rad)
        self.continuation_bonus = float(continuation_bonus)
        self.command_tau = float(command_time_constant)
        self.no_rewind, self.chunk = int(no_rewind_frames), int(chunk)
        # Steering may only nudge the facing; large turns must come from turning frames.
        self.max_correction = float(max_heading_correction)
        # After a switch, keep playing at least this long unless the segment ends.
        self.min_dwell = int(min_dwell_frames)
        # Pose features for the search: "reference" (kinematic character, MM practice) or
        # "robot". Robot pose features made a lagging robot keep re-selecting slow start
        # frames (closed-loop start loop); the robot state is still used for re-anchoring.
        if pose_source not in ("reference", "robot"):
            raise ValueError("pose_source must be 'reference' or 'robot'")
        self.pose_source = pose_source
        # Standing rules: after settle_s of zero command with a stopped reference, search only
        # still idle frames and ease the reference root onto the robot (no creeping to close
        # a residual offset). Only while the command is zero, so walking lag is never hidden.
        self.settle_s, self.settle_anchor_s = float(settle_s), float(settle_anchor_s)
        # Settled standing keeps the heading it settled at (steering corrects the idle clips' slow
        # yaw drift, ~-4.7 deg per 20 s on library v3); off keeps the released-heading behaviour.
        self.hold_settled_heading = bool(hold_settled_heading)
        # Feature weights: [traj offsets, traj headings, feet pos, feet vel, root vel + yaw rate]
        h = len(self.db.horizons)
        sizes = [2 * h, h, 4, 4, 2, 1]
        group_weight = [weights[0], weights[1], weights[2], weights[3], weights[4], weights[4]]
        weight = torch.cat([torch.full((k,), w, device=device) for k, w in zip(sizes, group_weight)])
        scale = torch.cat([torch.full((k,), c, device=device) for k, c in zip(sizes, self.SCALES)])
        self.weight = weight / scale ** 2
        features = self.db.candidate_features
        self.candidate_norm = (features * features * self.weight).sum(-1)
        self.candidate_weighted = features * self.weight
        n = num_envs
        self.play = torch.zeros(n, dtype=torch.long, device=device)
        self.source = torch.zeros(n, dtype=torch.long, device=device)
        self.blend = torch.zeros(n, device=device)
        self.timer = torch.zeros(n, dtype=torch.long, device=device)
        self.root_xy = torch.zeros(n, 2, device=device)
        self.facing = torch.zeros(n, device=device)
        self.facing_filtered = torch.zeros(n, device=device)
        self.command_heading = torch.zeros(n, device=device)
        self.reanchors = torch.zeros(n, device=device)
        self.switches = torch.zeros(n, device=device)
        self.steps = torch.zeros(n, device=device)
        self.dwell = torch.zeros(n, dtype=torch.long, device=device)
        self.stand_time = torch.zeros(n, device=device)

    # ----------------------------------------------------------------- state
    def reset(self, env_ids, frames, root_xy, facing):
        """Start playback at library frames with the given root xy and facing yaw."""
        self.play[env_ids] = frames
        self.source[env_ids] = frames
        self.blend[env_ids] = 0.0
        # Staggered first searches spread the batch cost and decorrelate envs.
        count = len(env_ids) if torch.is_tensor(env_ids) else self.num_envs
        self.timer[env_ids] = torch.randint(0, self.search_interval, (count,), device=self.device)
        self.root_xy[env_ids] = root_xy
        self.facing[env_ids] = facing
        self.facing_filtered[env_ids] = facing
        self.command_heading[env_ids] = facing
        self.reanchors[env_ids] = 0.0
        self.switches[env_ids] = 0.0
        self.steps[env_ids] = 0.0
        self.dwell[env_ids] = 0
        self.stand_time[env_ids] = 0.0

    def weights_now(self, ahead=0):
        """Crossfade weight of the newly selected playback (1 = no blend)."""
        progress = (self.blend_frames - self.blend + ahead) / max(1, self.blend_frames)
        return torch.where(self.blend > 0, progress.clamp(0, 1), torch.ones_like(self.blend))

    def advance(self, frames, steps=1):
        return torch.minimum(frames + steps, self.db.frame_end[frames] - 1)

    # ----------------------------------------------------------------- search
    def query(self, command, robot, start_velocity=None, start_rate=None):
        """Desired command trajectory (facing frame) plus the robot's actual pose features.

        The trajectory spring starts from the REFERENCE's current facing-frame velocity
        and facing rate (kinematic state); starting it from the robot's state made robot
        deviations feed back into frame selection (heading drift while commanded straight).
        """
        n = command.shape[0]
        offsets, headings = [], []
        velocity = (robot["root_velocity"] if start_velocity is None else start_velocity).clone()
        rate = (robot["yaw_rate"] if start_rate is None else start_rate).clone()
        position = torch.zeros(n, 2, device=self.device)
        heading = torch.zeros(n, device=self.device)
        dt = 1.0 / FPS
        alpha = 1.0 - math.exp(-dt / max(self.command_tau, 1e-3))
        horizon_set = set(self.db.horizons)
        for step in range(1, max(self.db.horizons) + 1):
            velocity = velocity + alpha * (command[:, :2] - velocity)
            rate = rate + alpha * (command[:, 2] - rate)
            heading = heading + rate * dt
            position = position + rotate_z(velocity, heading) * dt
            if step in horizon_set:
                offsets.append(position.clone())
                headings.append(heading[:, None].clone())
        traj = torch.cat(offsets + headings, dim=-1)
        pose = self.db.pose_features(robot["feet_xy"], robot["feet_velocity_xy"],
                                     robot["root_velocity"], robot["yaw_rate"])
        return torch.cat((traj, pose), dim=-1)

    def search(self, env_ids, command, robot):
        """Pick the next frame for env_ids; no rewinds inside the playing segment."""
        switched = []
        w = self.weights_now()
        ref_velocity = (1 - w)[:, None] * self.db.root_velocity[self.source] + w[:, None] * self.db.root_velocity[self.play]
        ref_rate = (1 - w) * self.db.facing_rate[self.source] + w * self.db.facing_rate[self.play]
        if self.pose_source == "reference":
            robot = {"feet_xy": self._blend(self.db.rel_pos)[:, list(FEET), :2],
                     "feet_velocity_xy": self._blend(self.db.rel_lin_vel)[:, list(FEET), :2],
                     "root_velocity": ref_velocity, "yaw_rate": ref_rate}
        for block in torch.split(env_ids, self.chunk):
            q = self.query(command[block], {k: v[block] for k, v in robot.items()},
                           start_velocity=ref_velocity[block], start_rate=ref_rate[block])
            cost = (self.candidate_norm[None] - 2.0 * q @ self.candidate_weighted.T
                    + (q * q * self.weight).sum(-1, keepdim=True))
            current = self.play[block]
            candidates = self.db.candidates
            index = torch.searchsorted(candidates, current).clamp(max=len(candidates) - 1)
            valid = candidates[index] == current
            current_cost = torch.where(valid, cost.gather(1, index[:, None])[:, 0],
                                       torch.full_like(current, float("inf"), dtype=cost.dtype))
            same = self.db.candidate_base[None] == self.db.frame_base[current][:, None]
            source = self.db.frame_source[current][:, None]
            window = same & (self.db.candidate_source[None] >= source - self.no_rewind) & (
                self.db.candidate_source[None] <= source + 5)
            settled = (self.stand_time[block] >= self.settle_s)[:, None] & ~self.db.candidate_still[None]
            if bool(self.db.candidate_still.any()):
                window = window | settled
            best = cost.masked_fill(window, float("inf")).argmin(-1)
            best_cost = cost.gather(1, best[:, None])[:, 0]
            keep = valid & (current_cost - best_cost <= self.continuation_bonus * (1.0 + best_cost.abs()))
            switch = ~keep & torch.isfinite(best_cost)
            ids = block[switch]
            self.source[ids] = self.play[ids]
            self.play[ids] = candidates[best[switch]]
            self.blend[ids] = float(self.blend_frames)
            self.switches[ids] += 1
            self.dwell[ids] = self.min_dwell
            switched.append(switch)
        return torch.cat(switched) if switched else torch.zeros(0, dtype=torch.bool, device=self.device)

    # ----------------------------------------------------------------- step
    def step(self, command, robot, dt=1.0 / FPS, active=None, heading_target=None, heading_mode=None,
             reanchor_m=None, reanchor_rad=None):
        """Advance one policy tick for active envs (default all). command [n,3] (vx, vy, wz,
        facing frame); robot dict: feet_xy [n,2,2], feet_velocity_xy [n,2,2] (facing frame,
        relative to root), root_velocity [n,2], yaw_rate [n], root_xy [n,2], pelvis_yaw [n].
        reanchor_m / reanchor_rad: optional per-env thresholds [n] overriding the instance values."""
        if active is None:
            active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        if not bool(active.any()):
            return self.reference()
        a = active
        self.steps = self.steps + a.float()
        self.play = torch.where(a, self.advance(self.play), self.play)
        self.source = torch.where(a, self.advance(self.source), self.source)
        self.blend = torch.where(a, (self.blend - 1).clamp(min=0), self.blend)
        self.timer = torch.where(a, self.timer - 1, self.timer)
        self.dwell = torch.where(a, (self.dwell - 1).clamp(min=0), self.dwell)
        # Forced search only when the future-reference lookahead (+10) would leave the
        # segment; candidates keep >= max(horizons) frames, so playback runs well past this.
        near_end = self.play + 12 >= self.db.frame_end[self.play]
        # Overshoot guard: a playing turn that rotates away from the intended heading is
        # re-searched immediately (the dwell would otherwise carry the clip's turning momentum).
        heading_error = wrap(self.command_heading - self.facing)
        playing_rate = self.db.facing_rate[self.play]
        overshoot = (heading_error * playing_rate < 0) & (playing_rate.abs() > 0.2)
        due = torch.nonzero(a & ((((self.timer <= 0) & (self.dwell <= 0))) | near_end | (overshoot & (self.timer <= 0)))).flatten()
        if len(due):
            self.search(due, command, robot)
            self.timer[due] = self.search_interval
        # Root integration from the (crossfaded) frame velocity, facing-frame.
        w = self.weights_now()
        velocity = (1 - w)[:, None] * self.db.root_velocity[self.source] + w[:, None] * self.db.root_velocity[self.play]
        rate = (1 - w) * self.db.facing_rate[self.source] + w * self.db.facing_rate[self.play]
        command_heading = wrap(self.command_heading + command[:, 2] * dt)
        rate_mode = torch.ones_like(a) if heading_mode is None else ~heading_mode
        if heading_target is not None and heading_mode is not None:
            # Heading-target commands (UI right-click) define the intended heading absolutely;
            # integrating the derived rate lagged the target and caused rotation after it.
            command_heading = torch.where(heading_mode, wrap(heading_target), command_heading)
        # Rate commands released to zero: stop turning where the reference faces now instead of
        # catching up the integral of the commanded rate (that rotated the robot after stopping).
        released = rate_mode & (command[:, :2].norm(dim=-1) < 0.02) & (command[:, 2].abs() < 0.02)
        if self.hold_settled_heading:
            released = released & (self.stand_time < self.settle_s)
        command_heading = torch.where(released, self.facing_filtered, command_heading)
        filtered = wrap(self.facing_filtered + self.heading_alpha * wrap(self.facing - self.facing_filtered))
        correction = (self.heading_gain * wrap(command_heading - filtered)).clamp(
            -self.max_correction, self.max_correction)
        facing = wrap(self.facing + (rate + correction) * dt)
        root_xy = self.root_xy + rotate_z(velocity, facing) * dt
        standing = (command[:, :2].norm(dim=-1) < 0.02) & (command[:, 2].abs() < 0.02) & (velocity.norm(dim=-1) < 0.05)
        self.stand_time = torch.where(a, torch.where(standing, self.stand_time + dt, torch.zeros_like(self.stand_time)),
                                      self.stand_time)
        self.command_heading = torch.where(a, command_heading, self.command_heading)
        self.facing_filtered = torch.where(a, filtered, self.facing_filtered)
        self.facing = torch.where(a, facing, self.facing)
        self.root_xy = torch.where(a[:, None], root_xy, self.root_xy)
        if "root_xy" in robot:
            settle = a & (self.stand_time >= self.settle_s)
            if bool(settle.any()):
                k = 1.0 - math.exp(-dt / max(self.settle_anchor_s, 1e-3))
                eased = self.root_xy + k * (robot["root_xy"] - self.root_xy)
                self.root_xy = torch.where(settle[:, None], eased, self.root_xy)
        # Re-anchor only beyond the thresholds (counted).
        if "root_xy" in robot:
            pelvis_yaw = self.facing + self.sway()
            limit_m = self.reanchor_m if reanchor_m is None else reanchor_m
            limit_rad = self.reanchor_rad if reanchor_rad is None else reanchor_rad
            far = a & (((robot["root_xy"] - self.root_xy).norm(dim=-1) > limit_m) | (
                wrap(robot["pelvis_yaw"] - pelvis_yaw).abs() > limit_rad))
            if bool(far.any()):
                self.root_xy[far] = robot["root_xy"][far]
                self.facing[far] = wrap(robot["pelvis_yaw"][far] - self.sway()[far])
                self.facing_filtered[far] = self.facing[far]
                # The user's intended heading survives a re-anchor; steering brings the
                # reference back to it (resetting it here let robot drift become the goal).
                self.reanchors[far] += 1
        return self.reference()

    def sway(self):
        w = self.weights_now()
        return (1 - w) * self.db.sway[self.source] + w * self.db.sway[self.play]

    # ----------------------------------------------------------------- outputs
    def _blend(self, table, ahead=0):
        w = self.weights_now(ahead)
        a = table[self.advance(self.source, ahead) if ahead else self.source]
        b = table[self.advance(self.play, ahead) if ahead else self.play]
        while w.dim() < a.dim():
            w = w.unsqueeze(-1)
        return (1 - w) * a + w * b

    def reference(self):
        """World-frame reference (without env origins) for the current tick."""
        joint_pos, joint_vel = self._blend(self.db.joint_pos), self._blend(self.db.joint_vel)
        rel_pos = self._blend(self.db.rel_pos)
        rel_quat = self._blend(self.db.rel_quat)
        rel_quat = rel_quat / rel_quat.norm(dim=-1, keepdim=True)
        bodies = rel_pos.shape[1]
        facing = self.facing[:, None].expand(-1, bodies)
        pos = rotate_z(rel_pos, facing)
        pos = torch.cat((pos[..., :2] + self.root_xy[:, None], pos[..., 2:]), dim=-1)
        quat = quat_mul(yaw_quat(facing), rel_quat)
        lin = rotate_z(self._blend(self.db.rel_lin_vel), facing)
        ang = rotate_z(self._blend(self.db.rel_ang_vel), facing)
        future = [(self._blend(self.db.joint_pos, k), self._blend(self.db.joint_vel, k)) for k in (5, 10)]
        return {"joint_pos": joint_pos, "joint_vel": joint_vel, "body_pos_w": pos, "body_quat_w": quat,
                "body_lin_vel_w": lin, "body_ang_vel_w": ang, "future": future,
                "play": self.play.clone(), "blend": self.blend.clone()}

    def diagnostics(self):
        steps = self.steps.clamp(min=1)
        return {"reanchor_rate": float((self.reanchors / steps).mean()),
                "switch_rate": float((self.switches / steps).mean())}
