"""810-input careful-walk student distilled from a matching teacher (DAgger, phase 1).

Single concept (careful_walk_plan_20261108/student_design/plan_d1.txt): move a verified matching
teacher (233 inputs, motion-matching reference) into the deployable 810 -> 24 MLP contract in the
scene where the teacher passed its scripted commands. Base = the teacher task in play mode: no
pushes, balls, sensor noise or physics randomization; matcher re-anchor 0.3 m / 0.3 rad; rewards,
terminations (falls, ankle/wrist z, illegal contact, nan) and metrics unchanged; self-collision on.
Only three things change: the student acts, the student observes the agroid 10x81 frame stack, and
the command is processed like the Simulation careful_walk path (one tensor for the student input
and the matcher). Training resets add the teacher's reset noise and 25% R1 default-pose starts.
"""
from __future__ import annotations

import copy
import dataclasses

from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from r1lab import contract as C
from r1lab.mdp import student_command as sc
from r1lab.tasks import common
from r1lab.tasks.matching_teacher import (matching_arms_teacher_env_cfg, matching_lateral_teacher_env_cfg,
                                          matching_soft_teacher_env_cfg, matching_still_teacher_env_cfg)
from r1lab.tasks.motion_teacher import ACTOR_DIM

DEFAULT_START_FRACTION = 0.25
TRAIN_ENVS = 4096
DISTURBANCE_EVENTS = ("push_robot", "projectiles")

STUDENT_CONTRACT = {
    "name": "careful-walk-command-student-v1",
    "input_dimension": C.OBS_DIM, "output_dimension": C.NUM_DOF, "policy_hz": 50,
    "observation": "proprioception-v1: 10 x 81 frames, oldest first; frame = gyro(3), projected gravity(3), "
                   "command(3), q-q0(24), dq(24), previous action(24); POLICY order; normalizer inside the ONNX",
    "observation_noise": "none (phase 1)",
    "action": "POLICY order; joint target = DEFAULT_POS + action * ACTION_SCALE",
    "command": {
        "processing": "Simulation.command careful_walk (app/operation_center/simulation_worker.py)",
        "rate_commands": "(vx, vy, wz) as sampled",
        "heading_commands": "wz = clip(3 * wrap(target - pelvis yaw), +-1 rad/s); 0 when |error| < 0.025 rad",
        "slew_per_tick": list(sc.SLEW),
        "matcher_input": "the same user command with the teacher task's semantics (step changes, heading rate "
                         "from the matcher facing, absolute target); Gate 0: sharing the pelvis-yaw tensor "
                         "reversed the rate mid-turn (+90 deg turns peaked at 131 deg) and the slew changed "
                         "rate-turn clip choices (default-start turns -141 vs -133 deg)",
        "lateral": "vy = 0 (no free-arm side-step captures)",
        "simulation_command_limits": {"forward": 0.30, "reverse": "0.25 (library v4 teachers) / none (v3)",
                                      "lateral": 0.0, "yaw": 1.0},
    },
    "teacher": {"observation": "reference-teacher-v1", "input_dimension": ACTOR_DIM, "group": "actor",
                "reanchor_m_rad": [0.3, 0.3], "disturbances": "none"},
    "obs_groups": {"student": ["student"], "teacher": ["actor"]},
    "diagnostics": {"group": "diag", "terms": ["command_kind (COMMAND_MIX order)", "since_change_s"]},
    "reset": {"teacher_rule": "stand_start_fraction idle frames, otherwise any library frame (teacher noise)",
              "default_start_fraction": DEFAULT_START_FRACTION,
              "default_start": "R1 DEFAULT_POS at STAND_HEIGHT + 0.01 m, uniform yaw; matcher on the idle "
                               "frame nearest to DEFAULT_POS, facing = yaw - sway"},
    "dagger": {"beta": 0.0, "student_action_std": 0.1, "loss": "mse in action units"},
}


def matching_student_env_cfg(play=False, t0=0, teacher_env=matching_soft_teacher_env_cfg):
    """Teacher scene (play mode) with the student command, observation groups and resets."""
    del t0
    cfg = teacher_env(play=True)
    if any(name in cfg.events for name in DISTURBANCE_EVENTS):
        raise ValueError("The phase-1 student scene must not contain pushes or balls")
    old = cfg.commands["motion"]
    values = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
    values["matcher"] = dict(old.matcher)  # play matcher: re-anchor 0.3 m / 0.3 rad (MotionMatcher default)
    if not play:
        # The teacher's training reset noise (its play config has none); not its training matcher.
        reset = teacher_env(play=False).commands["motion"]
        values.update(pose_range=dict(reset.pose_range), velocity_range=dict(reset.velocity_range),
                      joint_position_range=tuple(reset.joint_position_range),
                      default_start_fraction=DEFAULT_START_FRACTION)
    cfg.commands = {"motion": sc.StudentMatchingCommandCfg(**values), "twist": sc.StudentCommandViewCfg()}
    cfg.observations = {
        "actor": cfg.observations["actor"],  # teacher 233, no noise (labels only)
        "student": common.actor_obs("twist", noise=False),
        "diag": ObservationGroupCfg(terms={"command": ObservationTermCfg(func=sc.command_diagnostics)},
                                    concatenate_terms=True, enable_corruption=False),
    }
    if not play:
        cfg.scene.num_envs = TRAIN_ENVS
        cfg.auto_reset = True
    return cfg


def matching_student_rl_cfg():
    from r1lab.distillation import R1DistillationRunnerCfg
    return R1DistillationRunnerCfg(experiment_name="r1_careful_walk", num_steps_per_env=24,
                                   max_iterations=600, save_interval=50)


def matching_student_crisis_env_cfg(play=False, t0=0, teacher_env=matching_still_teacher_env_cfg):
    """D3 student scene: the I11 student scene plus the teacher's training pushes and 1 kg balls (single change).

    D1 distilled calm walking only. The teacher lineage (I5-I11) was trained with root-velocity pushes
    every 2-5 s and bursts of 1-3 balls (6-25 m/s, ceiling ramped over 200 updates, 25% clean episodes),
    so it can label recovery actions in the states a disturbed student visits (DAgger, 12_train.md
    stage 6). Everything else is the D1 scene: play matcher (re-anchor 0.3 m / 0.3 rad), no sensor noise
    or physics randomization. Play configs stay calm, so command evaluations compare with D1 directly.
    teacher_env stays a parameter: train.py records its default name for the results service.
    """
    cfg = matching_student_env_cfg(play=play, t0=t0, teacher_env=teacher_env)
    if not play:
        from r1lab.mdp.projectiles import projectile_entities, projectile_sensors
        train = teacher_env(play=False)
        missing = [name for name in DISTURBANCE_EVENTS if name not in train.events]
        if missing:
            raise ValueError(f"The teacher's training scene lacks disturbance events: {missing}")
        count = train.events["projectiles"].params["count"]
        cfg.scene.entities = {**cfg.scene.entities, **projectile_entities(count=count)}
        cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(
            projectile_sensors(count=count, history_length=cfg.decimation))
        for name in DISTURBANCE_EVENTS:
            cfg.events[name] = train.events[name]
    return cfg


def matching_student_crisis_arms_env_cfg(play=False, t0=0, teacher_env=matching_arms_teacher_env_cfg):
    """D4: the D3 scene (calm student scene + teacher pushes and balls) distilled from the I12 arms teacher.

    The only difference from D3 is the teacher (I11 -> I12: elbow/wrist position tracking), so the student
    does not inherit I11's held-out left arm. teacher_env stays a parameter for the results service."""
    return matching_student_crisis_env_cfg(play=play, t0=t0, teacher_env=teacher_env)


STUDENT_CONTRACT_LATERAL = copy.deepcopy(STUDENT_CONTRACT)
STUDENT_CONTRACT_LATERAL["name"] = "careful-walk-command-student-v2-lateral"
STUDENT_CONTRACT_LATERAL["command"]["lateral"] = (
    "vy as sampled by the teacher mix: |vy| 0.18-0.28 m/s side steps (library v5: ChingMu G1 lateral "
    "captures retargeted to R1 by tools/retarget_g1_r1.py)")
STUDENT_CONTRACT_LATERAL["command"]["simulation_command_limits"] = {
    "forward": 0.30, "reverse": 0.25, "lateral": 0.28, "yaw": 1.0}
STUDENT_CONTRACT_LATERAL["command"]["hold_heading"] = True
STUDENT_CONTRACT_LATERAL["command"]["heading_hold"] = (
    "while the user's yaw rate is zero (standing included) the pelvis yaw at that moment (followed for "
    "0.5 s, then fixed) is a heading target: wz = clip(3*wrap(target - pelvis yaw), +-1), 0 inside .025 rad. "
    "Runtime-computable from the IMU; D1 without it drifted -22 deg in 20 s standing (eval-time hold: -1.4)")
STUDENT_CONTRACT_LATERAL["teacher"] = {
    **STUDENT_CONTRACT["teacher"],
    "disturbances": "teacher training pushes (2-5 s, +-0.5 m/s, +-0.3 rad/s) and 1 kg ball bursts (6-25 m/s)"}


def matching_student_crisis_lateral_env_cfg(play=False, t0=0, teacher_env=matching_lateral_teacher_env_cfg):
    """D5: the D3 scene (calm student scene + teacher pushes and balls) distilled from the I13 lateral teacher.

    Differences from D4: the teacher (I12 -> I13: library v5 side steps) and therefore the command mix
    (lateral kind, |vy| .18-.28 m/s) the student sees, and the student's heading hold (STUDENT_CONTRACT_LATERAL;
    training = evaluation = runtime command processing). teacher_env stays a parameter for the results service."""
    cfg = matching_student_crisis_env_cfg(play=play, t0=t0, teacher_env=teacher_env)
    cfg.commands["motion"].hold_heading = True
    return cfg


def matching_student_lateral_rl_cfg():
    """Same distillation; the per-kind imitation metrics also cover the lateral kind."""
    from r1lab.tasks.matching_teacher import COMMAND_MIX_V5
    cfg = matching_student_rl_cfg()
    cfg.algorithm.command_kinds = tuple(m[1] for m in COMMAND_MIX_V5)
    # D1 plateaued (walk MSE ~.195 from update ~80) at a fixed 1e-3 with synchronized episodes:
    # cosine decay to 1e-4 over a longer run (episode starts are spread by the task's random_episode_start).
    cfg.max_iterations = 800
    cfg.algorithm.lr_schedule = "cosine"
    cfg.algorithm.lr_final_ratio = 0.1
    cfg.algorithm.lr_total_updates = cfg.max_iterations
    return cfg


def matching_student_still_env_cfg(play=False, t0=0, teacher_env=matching_still_teacher_env_cfg):
    """Student scene of the I11 teacher (library v4, heading hold): the generator the Simulation runs.

    teacher_env stays a parameter: train.py records its default name and the results service
    rebuilds the same scene from it."""
    return matching_student_env_cfg(play=play, t0=t0, teacher_env=teacher_env)
