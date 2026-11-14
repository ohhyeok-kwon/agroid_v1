"""T2 precise-foot teacher with the clip-timer phase removed from the policy.

Single concept versus its parent T2: the actor's two phase inputs are constant
zeros, so gait timing must come from the current and future reference frames
rather than a memorized clip clock. The critic keeps the clip phase as
privileged training-only input (finite-horizon value needs time-to-go). Width
stays 233; reference, rewards, physics, PPO, resets and terminations are the parent's.
"""
from __future__ import annotations

from mjlab.managers.observation_manager import ObservationTermCfg

from r1lab.mdp import motion_teacher as teacher
from r1lab.tasks.motion_teacher import OBSERVATION_CONTRACT
from r1lab.tasks.precise_teacher import precise_teacher_env_cfg, precise_teacher_rl_cfg

PHASE_MODE = "zero"
PHASEFREE_OBSERVATION_CONTRACT = {
    **OBSERVATION_CONTRACT,
    "phase": "actor zero: two constant inputs kept for width, no clip timer; critic keeps clip phase",
    "phase_mode": PHASE_MODE,
}


def phasefree_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    cfg = precise_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                  reference_file=reference_file)
    cfg.observations["actor"].terms["phase"] = ObservationTermCfg(func=teacher.zero_phase_observation)
    return cfg


def phasefree_teacher_rl_cfg():
    return precise_teacher_rl_cfg()
