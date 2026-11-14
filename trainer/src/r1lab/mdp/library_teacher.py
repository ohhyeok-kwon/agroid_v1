"""Multi-segment reference library for the teacher: per-env segment bounds.

The library npz (tools/build_motion_library.py) concatenates valid runs of
several captures and stores segment tables. Every env tracks one segment:
its reference clock never crosses the segment end, the future targets clamp
inside the segment, and the segment end is the reference end. Held-out
segments are excluded from training resets unless explicitly requested.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from r1lab.mdp.motion_teacher import (
    FUTURE_FRAMES, JOINT_VELOCITY_SCALE, TeacherMotionCommand, TeacherMotionCommandCfg,
)


@dataclass(kw_only=True)
class LibraryMotionCommandCfg(TeacherMotionCommandCfg):
    """split: 'train' samples non-held-out segments, 'heldout' only held-out ones.

    segment_start_fraction: share of resets placed exactly at a segment's first
    frame (captures that begin standing exercise real starts there).
    """
    split: str = "train"
    segment_start_fraction: float = 0.0
    # Share of resets drawn in proportion to each segment's recent tracking-failure rate
    # (0 disables; the rest stays length-proportional). failure_ema smooths per reset.
    failure_weighting: float = 0.0
    failure_ema: float = 0.05

    def build(self, env):
        return LibraryMotionCommand(self, env)


class LibraryMotionCommand(TeacherMotionCommand):
    cfg: LibraryMotionCommandCfg

    def __init__(self, cfg, env):
        if cfg.start_frame_mixture:
            raise ValueError("Library teachers use segment_start_fraction, not absolute start frames")
        if cfg.split not in ("train", "heldout"):
            raise ValueError("Library split must be train or heldout")
        if not 0.0 <= cfg.segment_start_fraction < 1.0:
            raise ValueError("segment_start_fraction must be in [0, 1)")
        super().__init__(cfg, env)
        with np.load(cfg.motion_file, allow_pickle=False) as data:
            missing = {"segment_start", "segment_end", "segment_heldout", "frame_segment"} - set(data.files)
            if missing:
                raise ValueError(f"Library reference is missing {sorted(missing)}")
            start = torch.as_tensor(data["segment_start"], dtype=torch.long, device=self.device)
            end = torch.as_tensor(data["segment_end"], dtype=torch.long, device=self.device)
            heldout = torch.as_tensor(data["segment_heldout"], dtype=torch.bool, device=self.device)
            frame_segment = torch.as_tensor(data["frame_segment"], dtype=torch.long, device=self.device)
        total = self.motion.time_step_total
        if (start[0] != 0 or end[-1] != total or (start[1:] != end[:-1]).any()
                or (end - start <= self.cfg.minimum_remaining_frames + 1).any()
                or len(frame_segment) != total):
            raise ValueError("Library segments must tile the reference with enough frames each")
        self.segment_start, self.segment_end = start, end
        self.segment_heldout, self.frame_segment = heldout, frame_segment
        active = heldout if cfg.split == "heldout" else ~heldout
        if not bool(active.any()):
            raise ValueError(f"No {cfg.split} segments in the library")
        self.active_segments = torch.nonzero(active).flatten()
        # Weight by the number of admissible start frames (duration-proportional).
        span = (end - start - self.cfg.minimum_remaining_frames).clamp(min=1).float()
        self.segment_weight = span[self.active_segments] / span[self.active_segments].sum()
        self.env_segment = self.frame_segment[self.time_steps]
        self.metrics["segment_start_reset"] = torch.zeros(self.num_envs, device=self.device)
        if not 0.0 <= cfg.failure_weighting < 1.0:
            raise ValueError("failure_weighting must be in [0, 1)")
        self.segment_failure = torch.zeros(len(start), device=self.device)
        self.metrics["failure_weighted_reset"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def env_segment_start(self):
        return self.segment_start[self.env_segment]

    @property
    def env_last_frame(self):
        return self.segment_end[self.env_segment] - 1

    def _record_failures(self, env_ids):
        manager = getattr(self._env, "termination_manager", None)
        if manager is None or self.cfg.failure_weighting <= 0:
            return
        failed = (manager.terminated[env_ids] & ~manager.time_outs[env_ids]).float()
        segments = self.env_segment[env_ids]
        count = torch.zeros_like(self.segment_failure).index_add_(0, segments, torch.ones_like(failed))
        total = torch.zeros_like(self.segment_failure).index_add_(0, segments, failed)
        seen = count > 0
        rate = torch.where(seen, total / count.clamp(min=1), self.segment_failure)
        alpha = self.cfg.failure_ema
        self.segment_failure = torch.where(seen, (1 - alpha) * self.segment_failure + alpha * rate,
                                           self.segment_failure)

    def _uniform_sampling(self, env_ids):
        count = len(env_ids)
        self._record_failures(env_ids)
        choice = torch.multinomial(self.segment_weight, count, replacement=True)
        if self.cfg.failure_weighting > 0:
            failure = self.segment_failure[self.active_segments] * self.segment_weight
            if float(failure.sum()) > 0:
                weighted = torch.multinomial(failure / failure.sum(), count, replacement=True)
                use = torch.rand(count, device=self.device) < self.cfg.failure_weighting
                choice = torch.where(use, weighted, choice)
                self.metrics["failure_weighted_reset"][env_ids] = use.float()
        segment = self.active_segments[choice]
        start = self.segment_start[segment]
        span = self.segment_end[segment] - start - self.cfg.minimum_remaining_frames
        frames = start + (torch.rand(count, device=self.device) * span).long().clamp(max=span - 1)
        at_start = torch.rand(count, device=self.device) < self.cfg.segment_start_fraction
        frames = torch.where(at_start, start, frames)
        self.time_steps[env_ids] = frames
        self.env_segment[env_ids] = segment
        self.metrics["segment_start_reset"][env_ids] = at_start.float()
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = float(self.segment_weight.max())
        self.metrics["sampling_top1_bin"][:] = 0.5

    def _update_command(self, env_ids=None):
        last = self.env_last_frame
        if env_ids is None:
            self.time_steps.add_(1)
            self.time_steps.copy_(torch.minimum(self.time_steps, last))
        else:
            self.time_steps[env_ids] = torch.minimum(self.time_steps[env_ids] + 1, last[env_ids])
        self.update_relative_body_poses()

    def reset_to_frame(self, env_ids, frame):
        frame = int(frame)
        if not 0 <= frame < self.motion.time_step_total:
            raise ValueError(f"Reference frame {frame} is out of range")
        self.env_segment[env_ids] = self.frame_segment[frame]
        super().reset_to_frame(env_ids, frame)


def library_reference_end(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return command.time_steps >= command.env_last_frame


def library_future_reference(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    last = command.env_last_frame
    rows = []
    for offset in FUTURE_FRAMES:
        frame = torch.minimum(command.time_steps + offset, last)
        rows.extend((command.motion.joint_pos[frame],
                     command.motion.joint_vel[frame] * JOINT_VELOCITY_SCALE))
    return torch.cat(rows, dim=-1)


def library_segment_phase(env, command_name="motion"):
    """Privileged critic clock: fraction of the current segment (time-to-go)."""
    command = env.command_manager.get_term(command_name)
    start, last = command.env_segment_start, command.env_last_frame
    phase = 2.0 * torch.pi * (command.time_steps - start).float() / (last - start).clamp(min=1).float()
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)
