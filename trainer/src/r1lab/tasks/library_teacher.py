"""Phase-free precise-foot teacher on a multi-capture free-arm library.

Changes versus the phase-free T2 teacher: the reference becomes a library of
real captures (cautious forward walk with real standing start and stop,
backward walk, 90/180 degree turns, standing idle, each with its left-right
mirror); resets pick segments in proportion to length, a quarter at segment
starts; episodes are capped at 6 s and time-outs are bootstrapped so
long references cannot accumulate unobservable lag. Whole captures are held out.
Rewards, 233 observations (actor phase zero), physics and PPO are unchanged.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from r1lab.mdp import library_teacher as library
from r1lab.mdp import motion_teacher as teacher
from r1lab.run_record import work_root
from r1lab.tasks.precise_teacher import precise_teacher_env_cfg, precise_teacher_rl_cfg

REFERENCE_NAME = "careful_library_v2.npz"
TRAIN_EPISODE_S = 6.0
SEGMENT_START_FRACTION = 0.25

LIBRARY_CONTRACT = {
    "name": "free-arm-library-v1",
    "reference": REFERENCE_NAME,
    "segment_sampling": "proportional to admissible start frames; held-out segments excluded",
    "segment_start_fraction": SEGMENT_START_FRACTION,
    "train_episode_s": TRAIN_EPISODE_S,
    "time_outs": "bootstrapped (infinite-horizon value); reference end per segment",
    "actor_phase": "zero", "critic_phase": "segment fraction (privileged)",
    "heldout": "whole captures 01703, 01720, 01688 and mirrors; 00230 14-18 s and mirror",
}


def library_path(reference_file=None):
    return Path(reference_file) if reference_file is not None else work_root()/"data/tracking"/REFERENCE_NAME


def library_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    path = library_path(reference_file)
    cfg = precise_teacher_env_cfg(play=play, t0=t0, reference_split="train", reference_file=str(path))
    old = cfg.commands["motion"]
    values = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
    values.update(split=reference_split,
                  segment_start_fraction=0.0 if play else SEGMENT_START_FRACTION)
    cfg.commands["motion"] = library.LibraryMotionCommandCfg(**values)
    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(func=teacher.zero_phase_observation)
    cfg.observations["critic"].terms["phase"] = ObservationTermCfg(func=library.library_segment_phase)
    for group in cfg.observations.values():
        group.terms["future_reference"] = ObservationTermCfg(func=library.library_future_reference)
    cfg.terminations["reference_end"] = TerminationTermCfg(func=library.library_reference_end, time_out=True)
    cfg.metrics["reference_end"] = MetricsTermCfg(func=library.library_reference_end, reduce="last")
    # Start-to-end clip direction is meaningless across concatenated captures.
    for name in [k for k in cfg.metrics if "net_progress" in k or "directed_speed" in k]:
        cfg.metrics.pop(name)
    cfg.is_finite_horizon = False
    if play:
        with np.load(path, allow_pickle=False) as data:
            longest = int((data["segment_end"] - data["segment_start"]).max())
        cfg.episode_length_s = max(cfg.episode_length_s, longest / 50 + 1.0)
    else:
        cfg.episode_length_s = TRAIN_EPISODE_S
    return cfg


def library_teacher_rl_cfg():
    return precise_teacher_rl_cfg()


XY_TERMINATION_M = 0.5


def library_xy_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Library teacher plus a planar tracking termination (single change).

    With z-only terminations a standing balancer keeps collecting relative-pose
    rewards on motions it has not learned (I3: backward/turn segments completed
    by time with progress ~0). Ending episodes beyond 0.5 m pelvis-to-reference
    distance removes that optimum while normal 5-28 cm lag survives.
    """
    cfg = library_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                  reference_file=reference_file)
    cfg.terminations["anchor_xy"] = TerminationTermCfg(
        func=teacher.anchor_xy_deviation, params={"threshold": XY_TERMINATION_M})
    return cfg
