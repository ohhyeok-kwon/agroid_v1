"""T2 precise-foot teacher on the complete 00230 capture: stand, start, walk, slow down, stand.

Single concept versus its parent T2: the reference becomes the full capture and
resets mix in its real standing start and slow-down. Rewards, observations,
physics, PPO and terminations are the parent's.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from r1lab.run_record import work_root
from r1lab.tasks.precise_teacher import precise_teacher_env_cfg, precise_teacher_rl_cfg

REFERENCE_NAME = "cautious_full_v1.npz"
STANDING_START_FRAME = 0
SLOWDOWN_FRAME = 1220
START_FRAME_MIXTURE = ((STANDING_START_FRAME, 0.30), (SLOWDOWN_FRAME, 0.10))

FULL_CAPTURE_CONTRACT = {
    "name": "full-capture-start-stop-v1",
    "reference": REFERENCE_NAME,
    "start_frame_mixture": [list(entry) for entry in START_FRAME_MIXTURE],
    "uniform_remainder": 1.0 - sum(p for _, p in START_FRAME_MIXTURE),
    "frames": "0 human standing start; 1220 last steps before the 25.0-26.6 s slow-down",
    "heldout": "none; the former 14-18 s temporal holdout is training data",
    "unchanged": "T2 rewards (body position .5 + world foot trajectories .5), 233 observations, "
                 "physics, PPO, terminations, small reset noise",
}


def reference_path(reference_file=None):
    return Path(reference_file) if reference_file is not None else work_root()/"data/tracking"/REFERENCE_NAME


def fullcapture_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    if reference_split != "train":
        raise ValueError("The full-capture reference has no disjoint held-out split")
    path = reference_path(reference_file)
    cfg = precise_teacher_env_cfg(play=play, t0=t0, reference_split="train", reference_file=str(path))
    if play:
        with np.load(path, allow_pickle=False) as data:
            frames = len(data["joint_pos"])
        # Evaluate the whole capture, including the final standing stop.
        cfg.episode_length_s = max(cfg.episode_length_s, (frames - 1) / 50 + 1.0)
    else:
        cfg.commands["motion"].start_frame_mixture = START_FRAME_MIXTURE
    return cfg


def fullcapture_teacher_rl_cfg():
    return precise_teacher_rl_cfg()
