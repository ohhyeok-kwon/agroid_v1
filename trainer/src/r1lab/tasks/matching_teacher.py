"""Phase-free precise-foot teacher tracking closed-loop motion-matching references.

Single concept versus the library teacher: references are no longer fixed
library segments but are chained by the batched motion matcher from sampled
user commands (stand, cautious forward, walking turns, heading turns,
backward as captured) and the robot's actual state. Rewards, 233
observations (phase zero), physics and PPO are unchanged; the reference has
no end, so episodes run 20 s with bootstrapped time-outs.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg

from r1lab.mdp import matching_command as matching
from r1lab.mdp import motion_teacher as teacher
from r1lab.run_record import work_root
from r1lab.tasks.precise_teacher import precise_teacher_env_cfg, precise_teacher_rl_cfg

REFERENCE_NAME = "careful_library_v3.npz"
TRAIN_EPISODE_S = 20.0

MATCHING_CONTRACT = {
    "name": "closed-loop-motion-matching-v1",
    "library": REFERENCE_NAME,
    "command_mix": [list(m[:2]) + [list(m[2]), list(m[3])] for m in matching.COMMAND_MIX],
    "command_duration_s": [2.0, 6.0],
    "stand_start_fraction": 0.5,
    "search": "every 10 ticks after a 30-tick dwell; trajectory 0.4/0.8/1.2 s + robot feet/root state",
    "steering": "heading only, gain 3/s on the facing (0.1 s filter), |correction| <= 0.35 rad/s; speed from selected frames",
    "query": "trajectory spring seeded by the reference's kinematic velocity/facing rate; pose features from the reference (kinematic); robot state only for re-anchoring",
    "reanchor": "training 1.0 m / 0.6 rad (lag beyond 0.5 m terminates instead); evaluation and Simulation 0.3 m / 0.3 rad",
    "xy_termination_m": 0.5,
    "lateral": "none (no free-arm side-step captures)",
    "episode_s": TRAIN_EPISODE_S, "time_outs": "bootstrapped",
}


def matching_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    if reference_split != "train":
        raise ValueError("Matching references are generated; evaluate held-out behaviour with scripted commands")
    path = Path(reference_file) if reference_file is not None else work_root()/"data/tracking"/REFERENCE_NAME
    cfg = precise_teacher_env_cfg(play=play, t0=t0, reference_split="train", reference_file=str(path))
    old = cfg.commands["motion"]
    values = {f.name: getattr(old, f.name) for f in dataclasses.fields(old) if f.init}
    values["sampling_mode"] = "uniform"
    cfg.commands["motion"] = matching.MatchingMotionCommandCfg(**values)
    for group in cfg.observations.values():
        group.terms["phase"] = ObservationTermCfg(func=teacher.zero_phase_observation)
        group.terms["future_reference"] = ObservationTermCfg(func=matching.matching_future_reference)
    cfg.terminations.pop("reference_end", None)
    # Positional termination (I3'): a teacher that stands while the reference moves must end,
    # and training must not hide that lag by re-anchoring the reference onto the robot.
    cfg.terminations["anchor_xy"] = TerminationTermCfg(
        func=teacher.anchor_xy_deviation, params={"threshold": 0.5})
    if not play:
        command = cfg.commands["motion"]
        command.matcher = {**command.matcher, "reanchor_m": 1.0, "reanchor_rad": 0.6}
    cfg.metrics.pop("reference_end", None)
    for name in [k for k in cfg.metrics if "net_progress" in k or "directed_speed" in k]:
        cfg.metrics.pop(name)
    cfg.is_finite_horizon = False
    cfg.episode_length_s = TRAIN_EPISODE_S
    return cfg


def matching_teacher_rl_cfg():
    return precise_teacher_rl_cfg()


PUSH_VELOCITY_RANGE = {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-0.3, 0.3)}
PUSH_INTERVAL_S = (2.0, 5.0)


def matching_push_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Matching teacher plus random root-velocity pushes every 2-5 s (single change).

    12_train.md stage 6 after command following was confirmed (I4b). Standing pushes are
    followed by the matcher's settle rule (reference eased onto where the robot recovers);
    walking pushes must be recovered within the 0.5 m planar termination.
    """
    from mjlab.envs import mdp as env_mdp
    from mjlab.managers.event_manager import EventTermCfg
    cfg = matching_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                   reference_file=reference_file)
    if not play:
        cfg.events["push_robot"] = EventTermCfg(
            func=env_mdp.push_by_setting_velocity, mode="interval", interval_range_s=PUSH_INTERVAL_S,
            params={"velocity_range": PUSH_VELOCITY_RANGE})
    return cfg


STANCE_SLIP_WEIGHT = -2.0


def matching_quiet_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Push teacher plus a loaded-foot sole slip penalty (single change).

    Same-physics comparison (I4b vs manufacturer at .167 m/s): touchdown and loading were softer,
    but stance slip .042 vs .026 m/s. L1 speed of loaded sole sites, weight -2.
    """
    from mjlab.managers.reward_manager import RewardTermCfg
    cfg = matching_push_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                        reference_file=reference_file)
    cfg.rewards["stance_slip"] = RewardTermCfg(func=teacher.stance_slip_speed, weight=STANCE_SLIP_WEIGHT)
    return cfg


PROJECTILE_SPEED = (6.0, 25.0)
PROJECTILE_WARMUP_ITERATIONS = 200
PROJECTILE_PARAMS = {"count": 8, "clean_fraction": .25, "speed_range": PROJECTILE_SPEED,
                     "warmup_steps": PROJECTILE_WARMUP_ITERATIONS * 24,
                     "spacing": (.3, .5), "cooldown": (3., 5.), "lifetime": 5.}


def matching_projectile_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Quiet push teacher plus physical 1 kg balls (S7 ProjectileBurst), single change.

    Same-physics protocol v3 (12/18/25 m/s x3 at .3 s): the push teacher fell in 8/24 cases, all
    side hits (manufacturer 0/24). Bursts of 1-3 balls .3-.5 s apart from random directions at the
    torso; the speed ceiling ramps 9 -> 25 m/s over the first 200 updates; 25% clean episodes.
    Pushes and the slip penalty stay. Play configs keep the quiet scene (no balls).
    """
    from mjlab.managers.event_manager import EventTermCfg
    from r1lab.mdp.projectiles import ProjectileBurst, projectile_entities, projectile_sensors
    cfg = matching_quiet_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                         reference_file=reference_file)
    if not play:
        count = PROJECTILE_PARAMS["count"]
        cfg.scene.entities = {**cfg.scene.entities, **projectile_entities(count=count)}
        cfg.scene.sensors = tuple(cfg.scene.sensors) + tuple(
            projectile_sensors(count=count, history_length=cfg.decimation))
        cfg.events["projectiles"] = EventTermCfg(func=ProjectileBurst, mode="step",
                                                 params=dict(PROJECTILE_PARAMS))
    return cfg


LANDING_WEIGHT = -10.0
LANDING_PARAMS = {"deadband": 0.1, "near_height": 0.04, "calm_only": True}


def matching_soft_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Ball teacher plus a human-deadband landing speed penalty (single change).

    Same-physics eval_gait: I4b landed at .13-.18 m/s (human 00230 .089 mean, p95 .152), the push
    teachers I5/I6 at .29-.34 (manufacturer .27-.28). L1 excess of the downward sole speed over .1 m/s
    while an airborne foot is within 4 cm of the ground, plus the same excess at touchdown; weight -10
    (~4.6% of the positive reward for I6 with balls and pushes; the slip term is ~1.3%).
    12_train.md: crisis responses may use the force they need, so the charge is off for 1.5 s after
    a push or ball contact (pushes now use tracked_velocity_push: identical physics, plus the mark).
    """
    from mjlab.managers.reward_manager import RewardTermCfg
    from r1lab.mdp.recovery import tracked_velocity_push
    cfg = matching_projectile_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                              reference_file=reference_file)
    if "push_robot" in cfg.events:
        cfg.events["push_robot"].func = tracked_velocity_push
    cfg.rewards["landing_speed"] = RewardTermCfg(func=teacher.landing_speed, weight=LANDING_WEIGHT,
                                                 params=dict(LANDING_PARAMS))
    return cfg


LIBRARY_V4 = "careful_library_v4.npz"
BACKWARD_RANGE_V4 = (-0.25, -0.17)
COMMAND_MIX_V4 = tuple((m[0], m[1], BACKWARD_RANGE_V4 if m[1] == "backward" else m[2], *m[3:])
                       for m in matching.COMMAND_MIX)


def matching_v4_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Soft-landing ball teacher on library v4 (single concept: the matching data).

    v4 = v3 + 06039 (standing -> backward start; arm joints clamped into range by <= .078 rad,
    derive_clip.py) and its mirror, without pelvis-tilt > 12 deg frames (00739 leaning walk).
    Kinematic MM probes (lean_review/): .25/.3 m/s reference tilt p95 14.1/18.3 -> 6.7/3.8 deg;
    backward from standing starts for |v| >= .16 m/s (v3: never). The backward command range follows
    the captured backward speeds (.2-.26 m/s): -.25..-.17 instead of -.20..-.10.
    """
    path = Path(reference_file) if reference_file is not None else work_root()/"data/tracking"/LIBRARY_V4
    cfg = matching_soft_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                        reference_file=str(path))
    cfg.commands["motion"].command_mix = COMMAND_MIX_V4
    return cfg


DISTURBED_REANCHOR = (0.3, 0.3)


def matching_recover_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Library-v4 teacher whose reference follows the robot after pushes and ball hits (single change).

    I7: 85% of late failure terminations were the 0.5 m planar deviation after hits, because
    training re-anchored only at 1.0 m while evaluation and the Simulation re-anchor at 0.3 m.
    For 1.5 s after a ball contact the matcher re-anchors at 0.3 m / 0.3 rad (the evaluation
    thresholds) so recovery steps resume from where the robot is. Pushes do not count (I5/I6 kept
    anchor_xy terminations at .05-.08 under pushes; counting them suspended the 0.5 m termination
    for ~half of the time in a smoke test). Elsewhere the 1.0 m / 0.6 rad training thresholds and
    the 0.5 m termination still forbid standing still.
    """
    cfg = matching_v4_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                      reference_file=reference_file)
    if not play:
        cfg.commands["motion"].disturbed_reanchor = DISTURBED_REANCHOR
    return cfg


def matching_v4_ball_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Fallback: library v4 on the ball teacher without the landing term (if I8 must be dropped)."""
    path = Path(reference_file) if reference_file is not None else work_root()/"data/tracking"/LIBRARY_V4
    cfg = matching_projectile_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                              reference_file=str(path))
    cfg.commands["motion"].command_mix = COMMAND_MIX_V4
    return cfg


def matching_still_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """Recover teacher whose generator holds the heading once standing has settled (single change).

    Standing references followed the idle captures' slow yaw drift (-4.7 deg per 20 s), and the robot
    added its own (I8 stand20 -8.0 deg). Zero-training check on I8 with hold_settled_heading:
    stand20 -2.6 deg (reference 0.0), heading180 stop -0.8 (was -4.8). Training on the same generator
    the Simulation and distillation will run.
    """
    cfg = matching_recover_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                           reference_file=reference_file)
    command = cfg.commands["motion"]
    command.matcher = {**command.matcher, "hold_settled_heading": True}
    return cfg


LIBRARY_V5 = "careful_library_v5.npz"
LATERAL_RANGE_V5 = (0.18, 0.28)
COMMAND_MIX_V5 = COMMAND_MIX_V4 + ((0.15, "lateral", (0.0, 0.0), (0.0, 0.0), LATERAL_RANGE_V5),)


def matching_lateral_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """I12 arms teacher on library v5 with side-step commands (single concept: lateral walking data).

    v5 = v4 (byte-identical prefix) + 10 ChingMu G1 lateral-walking captures retargeted by
    tools/retarget_g1_r1.py (stance feet pinned; slip median .007-.008 m/s like the human clips; arms
    hanging): 4 left + 4 right for training, 1 + 1 held out. Kinematic MM probe (perfect tracking):
    vy +-.20/.25 plays the lateral clips 91-99% of the time and moves the reference .17-.26 m/s sideways
    with <= .03 m/s forward drift; vy .15 gives only ~.09 m/s, so the lateral command range follows the
    captured speeds (.18-.28 m/s, sign random). The new kind has probability .15 (mix renormalized).
    """
    path = Path(reference_file) if reference_file is not None else work_root()/"data/tracking"/LIBRARY_V5
    cfg = matching_arms_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                        reference_file=str(path))
    cfg.commands["motion"].command_mix = COMMAND_MIX_V5
    # Second change in this step (time budget, both evaluated separately): body sway beyond the human frame.
    # D1 student and I10 sway above the human careful walk (pelvis roll/pitch rate) and the manufacturer.
    from mjlab.managers.reward_manager import RewardTermCfg
    cfg.rewards["sway_excess"] = RewardTermCfg(func=teacher.sway_excess, weight=SWAY_WEIGHT,
                                               params=dict(SWAY_PARAMS))
    # Third change (same reason): the crouched stance. StandIdle reference .734 m / knees .35 rad, I11 .706 m / .69.
    cfg.rewards["anchor_height"] = RewardTermCfg(func=teacher.anchor_height_tracking, weight=HEIGHT_WEIGHT,
                                                 params=dict(HEIGHT_PARAMS))
    return cfg


HEIGHT_WEIGHT = 0.5
HEIGHT_PARAMS = {"std": 0.05, "calm_only": True}  # 2 cm left a 2.8 cm crouch in the flat tail (reward .14)


SWAY_WEIGHT = -0.5  # preflight raw excess ~.46 rad/s: -2 would be ~18% of the positive reward (anti-motion)
SWAY_PARAMS = {"body_names": ("pelvis", "torso_link"), "deadband": 0.05, "calm_only": True}


ARM_TRACKING_WEIGHT = 0.5


def matching_arms_teacher_env_cfg(play=False, t0=0, reference_split="train", reference_file=None):
    """I11 teacher plus pelvis-relative elbow/wrist position tracking (single change).

    Same-scene analysis 2026-11-10 (I11 acting in the student scene, careful-speed walking frames vs the
    nearest library v4 human frames): legs within 1-3 cm, but the left arm held out (wrist +7 cm lateral,
    elbow +6 cm forward; walking left shoulder roll +.43 rad vs human +.18, right -.20; elbows 1.5 vs 1.1),
    because only the all-body position kernel (std 0.3 m) constrained the arms. Adds T3's
    arm_position_tracking (elbow/wrist origins in each full pelvis frame, 5 cm kernel, no wrist-orientation
    target, coordinated_teacher.ARM_REWARD_CONTRACT) at weight 0.5 (~10% of the positive reward); feet,
    reference, generator, observations, physics, disturbances, PPO and terminations unchanged.
    """
    from mjlab.managers.metrics_manager import MetricsTermCfg
    from mjlab.managers.reward_manager import RewardTermCfg
    from r1lab.mdp import coordinated_teacher as arms
    cfg = matching_still_teacher_env_cfg(play=play, t0=t0, reference_split=reference_split,
                                         reference_file=reference_file)
    missing = [name for name in arms.ARM_POINTS if name not in cfg.commands["motion"].body_names]
    if missing:
        raise ValueError(f"The matching command does not track the arm points {missing}")
    cfg.rewards["arm_position_tracking"] = RewardTermCfg(
        func=arms.arm_position_tracking, weight=ARM_TRACKING_WEIGHT,
        params={"command_name": "motion", "std": arms.POSITION_STD_M})
    for point, name in enumerate(arms.ARM_POINTS):
        cfg.metrics[f"arm_{name.removesuffix('_link')}_error_m"] = MetricsTermCfg(
            func=arms.arm_point_metric, params={"point": point, "quantity": "distance"})
    return cfg
