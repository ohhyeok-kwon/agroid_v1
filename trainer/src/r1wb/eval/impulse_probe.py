"""Reaction to one impulse, measured against an undisturbed twin (mjlab physics, deterministic play).

Every case replays a library clip from its first frame and receives one horizontal impulse on the torso (force for one
policy step, applied 0.1 m above the torso's centre of mass) at a time spread over one second so that hits land in
different gait phases. One extra environment per clip is the twin: same clip, no impulse. Relative to the twin:
  fell                 the episode ended by a tracking termination within the record
  joints_moved         joints that leave the twin's angle by more than 0.1 rad in the 2 s after the hit (legs/waist/arms)
  arm_extra_speed      RMS arm joint speed difference to the twin over the same 2 s (rad/s)
  rejoin_s             time until every joint is within 0.1 rad of the twin again and stays there for 0.5 s
                       (phase-sensitive: retained as a legacy diagnostic, not a recovery verdict)
  settle_s             time until the 0.4 s EMA of CoM velocity is within 0.08 m/s of the twin's for 0.5 s;
                       also phase-sensitive, not a recovery verdict across different gaits
  urgent_s, d1_max     time with a positive one-step capture margin, and its peak (r1wb.dynamics, twin velocity as reference)
  outside_s, outside_max_m   primary signal: time and peak of the velocity-error capture point outside the hull of both soles
  extra_steps          foot touchdowns in the 3 s after the hit minus the twin's
Works for the tracking teacher and for the goal student (same clips, same hits), so the two can be compared.
`--task manufacturer` runs the manufacturer's walking policy (comparison, workstation only: r1wb.eval.manufacturer);
`--task stream_student` runs a System 1 student; `--task stream_residual --base
<S1-a model>` runs a hardening correction on its frozen base. These three share one environment (the command-stream
environment: constant `--twists` commands as vx vy wz triples, the same start pose, the same impulses, the same
physics and the same definition of a fall: pelvis below 0.35 m, tilt above 0.9 rad or trunk on the ground), so they
are the like-for-like comparison.
Every summary row also carries the anatomy of the reaction (medians over its cases, along the push direction):
  first_touch_s        time from the first hit to the first foot touchdown
  step_m               foot placement of the first three touchdowns: landing sole ahead of the other sole
  peak_speed           highest CoM speed along the push; com_travel_m: CoM travel along the push until the end or the fall
  peak_tilt            largest trunk tilt (rad)
  leg_saturation       share of leg actuator samples at 95% of the effort limit or more in the 0.6 s after the first hit
  leg_speed_peak       highest leg joint speed in the same 0.6 s (rad/s)
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import shutil
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.rl.runner import MjlabOnPolicyRunner  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402
from mjlab.utils.lab_api.math import quat_apply, yaw_quat  # noqa: E402

from r1wb import dynamics, robot  # noqa: E402
from r1wb.tasks import TASKS  # noqa: E402

from r1wb.eval.paths import AGENT as WORK, ROOT  # noqa: E402,F401
DIRECTIONS = {"front": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}   # push direction, body frame
_D = math.sqrt(0.5)
DIAGONALS = {"front_left": (_D, _D), "front_right": (_D, -_D), "back_left": (-_D, _D), "back_right": (-_D, -_D)}
class WalkerPolicy:
  """Stock velocity-task policy (r1wb.walker) run in the stream environment: its 84 inputs are rebuilt from the
  robot state (IMU linear and angular velocity, gravity, q - q0, dq, previous action, commanded twist)."""

  def __init__(self, env, twists: torch.Tensor, checkpoint: str):
    from r1wb.student.observations import load_teacher
    self.env, self.twists = env, twists
    self.model = load_teacher(checkpoint, env.device)
    self.previous = torch.zeros(env.num_envs, 24, device=env.device)

  def __call__(self, observation):
    from mjlab.envs import mdp
    from tensordict import TensorDict
    env = self.env
    frame = torch.cat([mdp.builtin_sensor(env, sensor_name="robot/imu_lin_vel"), mdp.builtin_sensor(env, sensor_name="robot/imu_ang_vel"),
                       mdp.projected_gravity(env), mdp.joint_pos_rel(env), mdp.joint_vel_rel(env), self.previous, self.twists], dim=1)
    with torch.no_grad():
      action = self.model(TensorDict({"actor": frame}, batch_size=[env.num_envs]))
    self.previous = action
    return action


DEFAULT_CLIPS = ("00227_A_P0_StandIdle_230213", "00230_A_P0_WalkCautious_230213", "00211_A_P0_WalkSlow_230213",
                 "00677_F_N_AisleWalkSideways_230216", "01690_A_N_WalkBackward_230425", "01850_A_N_CrouchIdle_230425")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--task", choices=sorted(TASKS) + ["manufacturer", "stream_student", "stream_residual", "walker", "stream_label", "clip_label", "clip_student"],
                      default="wb_track_teacher")
  parser.add_argument("--base", type=Path, default=None, help="stream_residual: the frozen S1-a checkpoint")
  parser.add_argument("--arm-base-scale", type=float, default=1.0, help="stream_residual: as in the training task")
  parser.add_argument("--harden-task", default=None,
                      help="stream_residual: build the environment of this registered task")
  parser.add_argument("--twists", nargs="+", type=float, default=[0.0, 0.0, 0.0, 0.3, 0.0, 0.0],
                      help="manufacturer only: velocity commands as vx vy wz triples")
  parser.add_argument("--clips", nargs="+", default=list(DEFAULT_CLIPS))
  parser.add_argument("--impulses", nargs="+", type=float, default=[12.0, 25.0], help="N s")
  parser.add_argument("--repeats", type=int, default=10, help="hit times per clip, direction and impulse")
  parser.add_argument("--first-hit", type=float, default=3.0)
  parser.add_argument("--seed", type=int, default=None, help="explicit environment and policy RNG seed")
  parser.add_argument("--comparison-contract", choices=["native-r1", "proper-r1-v1"], default=None,
                      help="shared corebal initialization and native manufacturer physical-command slew once")
  parser.add_argument("--observation-profile", choices=["N0", "N1_shared_sensor_v1"], default="N0")
  parser.add_argument("--sensor-seeds", type=Path, default=None, help="proper-r1-v1: JSON environment noise_seeds, paired by physical channel")
  parser.add_argument("--case-pack", type=Path, default=None, help="proper-r1-v1 frozen physical case manifest")
  parser.add_argument("--case-pack-smoke", action="store_true", help="allow a clearly excluded instrumentation subset")
  parser.add_argument("--actor-trace", type=Path, default=None, help="proper-r1-v1: new trace directory; with video, stored inside the archive")
  parser.add_argument("--manufacturer-bundle", type=Path, default=None,
                      help="explicit hash-checked manufacturer input bundle, evaluation only")
  parser.add_argument("--tag", default="", help="unique comparison label; avoids replacing historical results")
  parser.add_argument("--protocol", type=Path, default=None, help="frozen comparison protocol to preserve with the results")
  parser.add_argument("--ground-reference", choices=("", "feet", "pelvis"), default="",
                      help="use the same ground-relative reference as DynamicMode")
  parser.add_argument("--reflex-gain", type=float, default=0.0,
                      help="stream policies: add gain x (own mean twist - goal twist) to the twist goal the policy sees "
                           "(walk with the push); 0 = off")
  parser.add_argument("--reflex-dead-band", type=float, default=0.12, help="m/s of velocity error that is ignored")
  parser.add_argument("--reflex-limit", type=float, default=0.6, help="m/s, largest goal change")
  parser.add_argument("--quiet", choices=["none", "arms", "upper", "arms0", "upper0"], default="none",
                      help="diagnostic only (never deployed): from the first hit on, replace the policy's arm outputs by "
                           "those of the un-hit twin (the arms do what they would have done without the hit); 'upper' "
                           "does the same for the waist too. Answers: does the upper-body reaction keep the robot up, "
                           "and how much do the joints move without it? 'arms0'/'upper0' (first version): arm outputs "
                           "zero instead, which also moves the arms from their posture onto the arm intent.")
  parser.add_argument("--goal-lag-grace", type=float, choices=(0.0, 2.0), default=None,
                      help="evaluation-only: compare actual-force-aligned goal-lag accumulation grace (0 or 2 s), "
                           "including the separate S2View accumulator; requires trace, sync-twins and flat forces")
  parser.add_argument("--arm-response-scale", type=float, default=None,
                      help="evaluation-only: scale post-intent PD arm target deviation from the twin after onset; "
                           "0=twin target, 1=exact original target. Requires trace, sync-twins and flat forces")
  parser.add_argument("--course", nargs="+", default=[],
                      help="diagnostic only (never deployed): joint-name fragments (e.g. shoulder elbow wrist hip_yaw "
                           "waist_yaw); from the first hit on, the outputs of the matching joints are the un-hit twin's "
                           "(those joints do what they would have done without the hit). Answers: which parts of the "
                           "reaction are needed?")
  parser.add_argument("--rate-feedback", type=float, default=0.0,
                      help="diagnostic only (never deployed): for --rate-feedback-steps control steps after each hit, "
                           "add this many rad of ankle-pitch target per rad/s of measured pelvis pitch rate to the "
                           "policy's output (both ankles). Answers a physics question: does cancelling the rotation at "
                           "once with the stance ankles make the rest of the reaction smaller on this robot?")
  parser.add_argument("--rate-feedback-steps", type=int, default=2)
  parser.add_argument("--rate-feedback-hip", type=float, default=0.0, help="the same for both hip-pitch targets")
  parser.add_argument("--leg-boost", type=float, default=1.0,
                      help="diagnostic only (never deployed): for --leg-boost-window seconds after each hit, multiply "
                           "the policy's leg-output change against the un-hit twin's by this factor. Answers: is the "
                           "reaction large because the first leg response is too small?")
  parser.add_argument("--leg-boost-window", type=float, default=0.1)
  parser.add_argument("--quiet-hits", type=int, default=0,
                      help="with --quiet: keep the replacement only until this many hits have landed (0 = to the end). "
                           "1 = the arms stay on their course for the first hit of a burst and are the policy's own "
                           "from the second hit on")
  parser.add_argument("--graft", type=Path, default=None,
                      help="diagnostic only (never deployed): a second stream checkpoint. The outputs of the joints "
                           "named by --graft-joints are that checkpoint's on the same observation, for the whole run "
                           "(a hybrid of two policies). Answers: which outputs carry the difference between two checkpoints?")
  parser.add_argument("--graft-joints", nargs="+", default=[], help="joint-name fragments, or the words legs / waist / arms")
  parser.add_argument("--sync-twins", action="store_true",
                      help="stream modes: every environment starts from the same standing frame and (label modes) the "
                           "reference matching is noise-free, so that a case and its twin are identical until the hit. "
                           "Without it the start frame differs per environment (two to four joints sit more than "
                           "0.1 rad from the twin's before any hit) and matched references drift apart while walking.")
  parser.add_argument("--match-pose", choices=["robot", "reference"], default="robot",
                      help="label modes on a stream: whose feet enter the reference look-up (reference = the course is "
                           "driven by the command and its own state only)")
  parser.add_argument("--commit", action="store_true",
                      help="label modes on streams: the course plays the window a look-up scored (student.stream match_commit)")
  parser.add_argument("--hold-after-hit", type=float, default=0.0,
                      help="label modes on streams: for this long after the hit the course keeps its plan (its own "
                           "feet enter the look-up), then re-attaches to the robot (student.stream hold_after_hit)")
  parser.add_argument("--hit-body", default="torso_link", help="body that is hit (held-out probes: a shoulder, a thigh)")
  parser.add_argument("--hit-height", type=float, default=0.1, help="m above that body's centre of mass")
  parser.add_argument("--directions", nargs="+", default=list(DIRECTIONS), choices=list(DIRECTIONS) + list(DIAGONALS))
  parser.add_argument("--height", type=float, default=None, help="stream policies: commanded pelvis height (default: standing)")
  parser.add_argument("--hit-duration", type=float, default=0.02,
                      help="seconds each hit lasts, the impulse spread evenly (default: one control step, ball-like)")
  parser.add_argument("--burst", type=int, default=1, help="hits per case, all from the same side (1 = single impulse)")
  parser.add_argument("--burst-gap", type=float, default=0.15, help="seconds between the hits of a burst")
  parser.add_argument("--record", type=float, default=4.0, help="seconds recorded after the last hit time")
  parser.add_argument("--match-height", choices=["stream", "target"], default="stream",
                      help="label modes on streams: student.stream match_height")
  parser.add_argument("--match-stop", choices=["room", "arrive"], default="room",
                      help="label modes on streams: student.stream match_stop")
  parser.add_argument("--course-set", nargs="*", default=[],
                      help="label modes on streams: other course generator settings as name=value")
  parser.add_argument("--s2-head", default=None,
                      help="stream_label: System 2 writes the reference (r1wb.s2 code head small_<name>.pt, told the "
                           "goal of the commanded twist) instead of the course generator; the checkpoint is System 1")
  parser.add_argument("--steer", default="", help="stream_label without --s2-head: a learned System 2 (r1wb.s2.steer, "
                                                 "data/s2/steer_<name>.pt) gives the course generator its command every 0.32 s")
  parser.add_argument("--show-steer", action="store_true", help="--steer: print what System 2 asked for around the hit")
  parser.add_argument("--tokens", default="v2c", help="--s2-head: the motion tokenizer (tokenizer_<name>.pt)")
  parser.add_argument("--goal-model", default=None,
                      help="--s2-head: a vision-language goal model (folder under data/s2) reads the head camera and the "
                           "instruction (a held-out phrasing of the commanded twist / height) at every decision and states "
                           "the goal; without it the head is told the command's goal directly")
  parser.add_argument("--backbone", type=Path, default=None, help="--goal-model: the vision-language model's folder")
  parser.add_argument("--language", choices=["ko", "en"], default="ko", help="--goal-model: language of the instruction")
  parser.add_argument("--app-pid", type=int, required=True)
  parser.add_argument("--archive-compression", action="store_true",
                      help="lossless NPZ compression for server state transport; simulation and sampling are unchanged")
  parser.add_argument("--no-video", action="store_true")
  parser.add_argument("--trace", type=Path, default=None, help="also save the raw time series (npz) for offline analysis")
  parser.add_argument("--library", default="",
                      help="label runs: the motion library the course generator plays (library_v2_<name>.npz), e.g. "
                           "corebal = the rest poses given the largest margin their stance allows (r1wb.balance)")
  parser.add_argument("--plan-lean", type=float, default=0.0,
                      help="DIAGNOSTIC, stream_label without --s2-head: the plan's standing frames lean forward over "
                           "their feet by this many rad (r1wb.eval.diagnostics.lean_standing_plan)")
  parser.add_argument("--world", default="",
                      help="the ground (r1wb.world: rough<cm>, gravel<cm>, up<deg>, down<deg>, side<deg>); the fall "
                           "criterion then takes the pelvis height above the ground under it")
  parser.add_argument("--throw", choices=["ball", "box"], default=None,
                      help="the hits are bodies thrown at the hit point (r1wb.world: a 1 kg ball like the Operation "
                           "Center's, or a 5 kg box of 30 cm), one per hit; --impulses is then the body's momentum "
                           "(N s = mass x speed) and what the robot receives depends on the contact")
  parser.add_argument("--throw-distance", type=float, default=1.5, help="m from the hit point at which a body starts")
  parser.add_argument("--physics-dt", type=float, default=0.0,
                      help="s; physics time step for this run (the control period stays 0.02 s). The scenes use 0.005; "
                           "the Operation Center's simulation uses 0.001, which a body thrown at 25 m/s needs (at "
                           "0.005 it moves 12.5 cm per step and passes half through the trunk)")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  from r1wb.eval.impulse_diagnostics import validate_options
  try:
    diagnostics_enabled = validate_options(args)
  except ValueError as error:
    parser.error(str(error))
  from r1wb.eval.shared_sensor_observation import validate_profile_options
  try:
    proper = validate_profile_options(args)
  except ValueError as error:
    parser.error(str(error))
  case_pack = None
  if args.case_pack is not None:
    from r1wb.eval.case_pack import load_case_pack
    case_pack = load_case_pack(args.case_pack, args, allow_subset=args.case_pack_smoke)
  if proper and case_pack is None:
    parser.error("proper-r1-v1 requires a sealed --case-pack")
  comparison = args.comparison_contract in ("native-r1", "proper-r1-v1")
  if comparison:
    if args.task not in ("stream_label", "manufacturer") or args.seed is None or not args.tag or not args.protocol:
      parser.error("native-r1 impulse comparisons require stream_label/manufacturer, seed, unique tag and protocol")
    if args.task == "stream_label" and (args.s2_head != "h12" or args.tokens != "v2d"):
      parser.error("native-r1 fixes the app head h12 and tokenizer v2d")
    if args.library not in ("", "corebal") or args.ground_reference not in ("", "feet"):
      parser.error("native-r1 fixes corebal/feet")
    args.library, args.ground_reference = "corebal", "feet"
  if args.manufacturer_bundle and args.task != "manufacturer":
    parser.error("manufacturer bundle is evaluation-only and requires --task manufacturer")
  if args.trace is not None and args.trace.exists():
    parser.error(f"Trace already exists: {args.trace}")
  if args.tag and (Path(args.tag).name != args.tag or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-" for c in args.tag)):
    parser.error("--tag must contain only letters, digits, _, . and -")
  if args.seed is not None:
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
  os.environ["RL_EVALUATION_APP_PID"] = str(args.app_pid)
  from r1wb.eval.evaluation_video import require_app_owner
  require_app_owner(bind=True)
  run = args.run.resolve()
  manufacturer = args.task == "manufacturer"
  walker = args.task == "walker"                        # like the manufacturer's: plain joint targets, own inputs
  stream = args.task in ("stream_student", "stream_residual", "manufacturer", "walker", "stream_label")
  # Label modes (diagnostic): the frozen tracking teacher's action labels as the policy, with the anchor terms the
  # student's labels use (no planar or heading error). stream_label: on the state-matched stream reference (what
  # S1-a imitated); clip_label: on a time-indexed library clip.
  label = args.task in ("stream_label", "clip_label")
  if manufacturer:
    from r1wb.eval.manufacturer import load as load_manufacturer
    MANUFACTURER, ManufacturerPolicy = load_manufacturer(args.manufacturer_bundle, reset_history=comparison)
    checkpoint, iteration = MANUFACTURER, 0
  else:
    checkpoint = (args.checkpoint or max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))).resolve()
    iteration = int(checkpoint.stem.split("_")[1])
  if args.task == "stream_residual":
    from r1wb.harden import harden_env_cfg
    cfg = (TASKS[args.harden_task]["env"](play=True, teacher=str(args.base.resolve())) if args.harden_task else
           harden_env_cfg(play=True, teacher=str(args.base.resolve()), arm_base_scale=args.arm_base_scale))
    cfg.commands["motion"].calm_fraction = 1.0          # impulses come from this tool only
  elif stream:
    from r1wb.student.stream import stream_env_cfg
    cfg = (stream_env_cfg(play=True, matched=True, teacher=str(checkpoint)) if label else
           stream_env_cfg(play=True, plain_actions=manufacturer or walker))
  elif label:
    from r1wb.student.env_cfg import student_env_cfg
    cfg = student_env_cfg(play=True, teacher=str(checkpoint))
  elif args.task == "clip_student":
    from r1wb.student.env_cfg import student_env_cfg
    cfg = student_env_cfg(play=True)
  else:
    cfg = TASKS[args.task]["env"](play=True)
  if not stream:
    # One fall criterion for every mode: the clip environments also end an episode on TRACKING errors (pelvis or
    # foot height off the reference by 0.15 m, hands by 0.25 m, planar drift), which a recovery step from a strong
    # hit triggers without any fall. Until 11-12 02:55 those endings were counted as falls in the clip modes.
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.tasks.tracking import mdp as tracking_mdp
    from r1wb.student.stream import fallen
    from r1wb.tracking import mdp as wb_mdp
    cfg.terminations = {
      "time_out": TerminationTermCfg(func=tracking_mdp.time_out, time_out=True),
      "trunk_ground": TerminationTermCfg(func=wb_mdp.body_on_ground, params={"sensor_name": "trunk_ground"}),
      "fallen": TerminationTermCfg(func=fallen),
    }
  if comparison:
    from r1wb.tracking.env_cfg import library_path
    from r1wb.student.stream import stand_clips
    cfg.commands["motion"].motion_file = str(library_path("corebal"))
    cfg.commands["motion"].clip_list = stand_clips(cfg.commands["motion"].motion_file)
  if stream and label:
    cfg.commands["motion"].match_pose = args.match_pose
    cfg.commands["motion"].hold_after_hit = args.hold_after_hit
    cfg.commands["motion"].match_commit = args.commit
    cfg.commands["motion"].match_height, cfg.commands["motion"].match_stop = args.match_height, args.match_stop
    if args.library:
      from r1wb.tracking.env_cfg import library_path
      cfg.commands["motion"].motion_file = str(library_path(args.library))
    for item in args.course_set:
      key, value = item.split("=")
      setattr(cfg.commands["motion"], key, type(getattr(cfg.commands["motion"], key))(float(value)))
  if stream and args.sync_twins:
    cfg.commands["motion"].clip_list = tuple(cfg.commands["motion"].clip_list[:1])
    if hasattr(cfg.commands["motion"], "match_noise"):
      cfg.commands["motion"].match_noise = 0.0
  manifest = json.loads(Path(cfg.commands["motion"].motion_file).with_suffix(".manifest.json").read_text())
  names = [c["name"] for c in manifest["clips"]]
  if stream:
    triples = [tuple(args.twists[i:i + 3]) for i in range(0, len(args.twists), 3)]
    manifest = dict(manifest, clips=[dict(name=f"twist_{t[0]:g}_{t[1]:g}_{t[2]:g}", context="stand" if not any(t) else "walk",
                                          length=100000) for t in triples])
    names = [c["name"] for c in manifest["clips"]]
    clips = list(range(len(triples)))
  else:
    clips = [names.index(n) for n in args.clips]
  directions = {**DIRECTIONS, **DIAGONALS}
  cases = [(clip, direction, impulse, repeat) for clip in clips for direction in args.directions
           for impulse in args.impulses for repeat in range(args.repeats)]
  if case_pack is not None:
    cases, clips, triples = case_pack.cases, case_pack.clips, case_pack.triples
    manifest = dict(manifest, clips=[dict(name=r["block_id"], context=r["posture"], length=100000) for r in case_pack.rows])
    names = [c["name"] for c in manifest["clips"]]
  count = len(cases) + len(clips)                       # the last len(clips) environments are the twins
  twin_of = torch.tensor([len(cases) + clips.index(c[0]) for c in cases], device=args.device)
  cfg.scene.num_envs = count
  if args.seed is not None:
    cfg.seed = args.seed
  thrown = ()
  if args.physics_dt > 0.0:
    control = cfg.sim.mujoco.timestep * cfg.decimation
    cfg.sim.mujoco.timestep, cfg.decimation = args.physics_dt, round(control / args.physics_dt)
  if args.world or args.throw or args.ground_reference:
    from r1wb import world
    world.apply_ground(cfg, args.world)
    if args.throw:
      thrown = world.add_throwables(cfg, args.throw, max(args.burst, 1))
  if args.goal_model:                                         # the goal model looks through the head camera
    from r1wb.s2.collect import head_camera
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (head_camera(224),)
  if not stream:
    cfg.commands["motion"].clip_list = tuple(c[0] for c in cases) + tuple(clips)
  if args.goal_lag_grace is not None:
    cfg.commands["motion"].lag_grace = args.goal_lag_grace
  if case_pack is not None:
    case_pack.configure_cfg(cfg)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env)
  diagnostic_input = {"actor": None}
  if stream:
    order = [c[0] for c in cases] + clips
    script = torch.zeros(count, 1, 8)
    script[:, 0, 0] = 1.0e6
    script[:, 0, 1:4] = torch.tensor([triples[i] for i in order])
    script[:, 0, 4] = 0.731 if args.height is None else args.height
    script[:, 0, 5] = 0.0 if args.height is None else 1.0
    env.command_manager.get_term("motion").set_script(script)
    env.reset()
    command = env.command_manager.get_term("motion")
    if args.ground_reference:
      from r1wb import world as ground_world
      if not proper:
        ground_world.ground_reference(env, command, args.ground_reference)
  initialization_qa = case_pack.initialize(env, command) if case_pack is not None else None
  common_command = None
  if proper:
    from r1wb.eval.case_pack import apply_ground_reference
    from r1wb.eval.common_command import CommonCommand
    apply_ground_reference(env, command)
    common_command = CommonCommand(torch.zeros(count, 3, device=env.device), float(env.step_dt))
    requested_twists = torch.tensor([triples[i] for i in order], device=env.device)
    common_commands = []
  if label and args.s2_head:
    assert args.task == "stream_label", "--s2-head drives command streams"
    from tensordict import TensorDict

    from r1wb.s2.runtime import S2View, goal_brain, table_goal
    from r1wb.s2.small import SmallS2
    from r1wb.s2.tokenizer import MotionTokenizer
    from r1wb.student.observations import load_teacher
    from r1wb.tracking.env_cfg import teacher_observation
    tokenizer = MotionTokenizer.load(WORK / "data/s2" / f"tokenizer_{args.tokens}.pt", args.device)
    head = SmallS2.load(WORK / "data/s2" / f"small_{args.s2_head}.pt", args.device)
    goals = torch.cat([script[:, :, 1:4], script[:, :, 4:6]], dim=2).to(args.device)        # (envs, 1, 5): twist, height, counts
    state_goal = (lambda decision: common_command.goal(goals[:, 0])) if proper else table_goal(goals)
    if args.goal_model:
      import random

      from r1wb.s2 import instructions
      from r1wb.s2.runtime import vlm_goal
      from r1wb.s2.small import GOAL_SCALE, goal_of
      from r1wb.s2.vla import GoalVLA
      meanings = instructions.every_instruction()
      vocabulary = torch.tensor([goal_of(item) for item in meanings], device=args.device)
      scale = torch.tensor(GOAL_SCALE[:4], device=args.device)
      apart = ((((goals[:, 0, None, :4] - vocabulary[None, :, :4]) / scale) ** 2).sum(dim=-1)
               + 100.0 * (goals[:, 0, None, 4] - vocabulary[None, :, 4]).abs())
      assert float(apart.min(dim=1).values.max()) < 0.01, "a commanded twist or height is not an instruction of the vocabulary"
      chooser = random.Random(0)
      wording = [[instructions.phrase(meanings[int(m)], args.language, chooser, held_out=True)] for m in apart.argmin(dim=1)]
      reader = GoalVLA(args.backbone).to(args.device).eval()
      weights = torch.load(WORK / "data/s2" / args.goal_model / "goal_last.pt", map_location="cpu", weights_only=False)
      reader.load_state_dict(weights["state"], strict=False)
      del weights
      state_goal = vlm_goal(reader, env, wording, vocabulary=vocabulary)
    head_steerer = None
    if args.steer:                                             # the learned slow loop between the goal and the code head
      from r1wb.s2.steer import Steer, SteerPolicy, steered_goal
      steer_policy = SteerPolicy.load(WORK / "data/s2" / f"steer_{args.steer}.pt", args.device)
      head_steerer = Steer(env, command, steer_policy.see, generator=False, leak=steer_policy.leak, remember=steer_policy.remember, turned=steer_policy.turned)
      state_goal = steered_goal(state_goal, head_steerer, steer_policy)
    view = S2View(env, command, goal_brain(head, state_goal), tokenizer)
    if args.ground_reference:
      if proper:
        apply_ground_reference(env, view)
      else:
        ground_world.ground_reference(env, view, args.ground_reference)
    tracker = load_teacher(str(checkpoint), args.device)
    joint_action = env.action_manager.get_term("joint_pos")

    def policy(obs):
      if head_steerer is not None and view._calls >= 0:        # once after every control step
        head_steerer.progress.after_step()
      previous_lag = view.goal_lag.clone() if args.goal_lag_grace is not None else None
      common_command.update_view(view) if proper else view.update()
      if previous_lag is not None:
        from r1wb.eval.impulse_diagnostics import gate_s2_lag
        view._lag = gate_s2_lag(previous_lag, view.goal_lag, applied_clock.current, args.goal_lag_grace,
                                env.step_dt, command.cfg.lag_time_constant)
      seen = teacher_observation(view.env, "motion", joint_action.plain_action, local=True, width=tracker.obs_dim)
      if diagnostics_enabled:
        diagnostic_input["actor"] = seen.detach().clone()
      return joint_action.from_plain_action(tracker(TensorDict({"actor": seen}, batch_size=[seen.shape[0]])))
  elif label:
    def policy(obs):
      return obs["bc"][:, :24]
  elif args.task == "clip_student":
    from r1wb.student.observations import load_policy as load_student
    student_actor = load_student(str(checkpoint), args.device)

    def policy(obs):
      return student_actor(torch.cat([obs["actor"], obs["goal"]], dim=1))
  elif manufacturer:
    target_twists = torch.tensor([triples[i] for i in order], dtype=torch.float, device=args.device)
    if comparison:
      from r1wb.eval.command_contract import NativeCommand
      native_command = NativeCommand(torch.zeros_like(target_twists), float(env.step_dt))
      delivered_twists = torch.zeros_like(target_twists)
      manufacturer_policy = ManufacturerPolicy(env, delivered_twists)
      manufacturer_commands = []

      def policy(obs):
        delivered_twists.copy_(common_command.delivered if proper else native_command.step(target_twists))
        manufacturer_commands.append(delivered_twists.clone())
        return manufacturer_policy(obs)
    else:
      policy = ManufacturerPolicy(env, target_twists)
  elif walker:
    policy = WalkerPolicy(env, torch.tensor([triples[i] for i in order], dtype=torch.float, device=args.device), str(checkpoint))
  elif stream:
    from r1wb.student.observations import load_policy
    actor = load_policy(str(checkpoint), args.device)
    groups = {"legs": range(0, 12), "waist": range(12, 14), "arms": range(14, 24)}
    graft_ids = sorted({i for word in args.graft_joints for i in groups.get(word, [])} |
                       {i for i, name in enumerate(robot.JOINTS) if any(f in name for f in args.graft_joints if f not in groups)})
    grafted = load_policy(str(args.graft.resolve()), args.device) if args.graft else None
    assert bool(args.graft) == bool(graft_ids), "--graft needs --graft-joints (and the other way round)"

    def policy(obs):
      goal = obs["goal"]
      if args.reflex_gain > 0.0:
        # Goal-space reflex: when the body is carried away from the commanded twist, ask the walking policy to walk
        # along (it then steps under the centre of mass with its own gait).
        error = command.twist_mean[:, :2] - command.twist_goal()[:, :2]
        size = torch.norm(error, dim=1, keepdim=True)
        change = error * ((size - args.reflex_dead_band).clamp_min(0.0) / size.clamp_min(1e-6)) * args.reflex_gain
        change = change * (args.reflex_limit / torch.norm(change, dim=1, keepdim=True).clamp_min(args.reflex_limit))
        goal = goal.clone()
        goal[:, 0:2] += change
      joined = torch.cat([obs["actor"], goal], dim=1)
      action = actor(joined)
      if grafted is not None:
        action = action.clone()
        action[:, graft_ids] = grafted(joined)[:, graft_ids]
      return action
  else:
    runner = MjlabOnPolicyRunner(wrapped, asdict(TASKS[args.task]["rl"]()), None, args.device)
    runner.load(str(checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
  entity = env.scene["robot"]
  centroidal = dynamics.Centroidal(env)
  torso = entity.body_names.index(args.hit_body)
  dt = float(env.step_dt)
  hit_step = torch.tensor([round((args.first_hit + c[3] / args.repeats) / dt) for c in cases], device=env.device)
  impulse = torch.tensor([c[2] for c in cases], device=env.device)
  case_tensors = case_pack.tensors(env.device) if case_pack is not None else None
  if case_tensors is not None:
    hit_step = case_tensors["hit_step"]
  local = torch.tensor([directions[c[1]] + (0.0,) for c in cases], device=env.device)
  gap_steps = max(1, round(args.burst_gap / dt))
  hit_steps = max(1, round(args.hit_duration / dt))
  assert hit_steps <= gap_steps
  thrower = None
  if thrown:
    thrower = world.Thrower(env, thrown, args.throw, args.throw_distance)
    throw_speed = impulse / thrower.mass                  # m/s: the body carries the momentum the option names
    throw_lead = thrower.lead_steps(throw_speed, dt)      # policy steps of flight before the contact
    assert int(throw_lead.max()) < int(hit_step.min()), "the first hit comes before a body can arrive"
  horizon = int((args.first_hit + 1.0 + args.record + (args.burst - 1) * args.burst_gap) / dt)
  horizon = max(horizon, int(hit_step.max()) + (args.burst - 1) * gap_steps + round(args.record / dt) + 1)
  shortest = min(manifest["clips"][c]["length"] for c in clips)
  if horizon > shortest - 2:
    horizon = shortest - 2
  if case_pack is not None:
    horizon = case_pack.horizon
  out_dir = WORK / "result/render" / run.name
  out_dir.mkdir(parents=True, exist_ok=True)
  tag = ("impulse_probe_manufacturer" if manufacturer else
         f"impulse_probe_{args.task.replace('stream_', 'stream-')}_{iteration}" if stream else
         f"impulse_probe_clip-label_{iteration}" if label else
         f"impulse_probe_clip-student_{iteration}" if args.task == "clip_student" else f"impulse_probe_{iteration}")
  if args.sync_twins:
    tag += "_sync"
  if args.match_pose != "robot":
    tag += "_course-" + args.match_pose
  if args.hold_after_hit > 0.0:
    tag += f"_hold{args.hold_after_hit:g}"
  if args.commit:
    tag += "_commit"
  if args.match_height != "stream" or args.match_stop != "room" or args.course_set:
    tag += "_decisive"
  if args.s2_head:
    tag += f"_s2-{args.s2_head}" + (f"+{args.goal_model}-{args.language}" if args.goal_model else "")
  if args.steer:
    tag += f"_steer-{args.steer}"
  if args.plan_lean != 0.0:
    tag += f"_lean{args.plan_lean:+g}"
  if args.library:
    tag += f"_lib-{args.library}"
  if args.world:
    tag += f"_world-{args.world}"
  if args.throw:
    tag += f"_throw-{args.throw}"
  if args.physics_dt > 0.0:
    tag += f"_dt{args.physics_dt:g}"
  if [float(v) for v in args.impulses] != [12.0, 25.0]:      # a non-default impulse set gets its own result file
    tag += "_imp" + "-".join(f"{v:g}" for v in args.impulses)
  if args.burst > 1:
    tag += f"_burst{args.burst}x{args.burst_gap:g}s"
  if hit_steps > 1:
    tag += f"_dur{args.hit_duration:g}s"
  if args.hit_body != "torso_link" or args.hit_height != 0.1:
    tag += f"_at-{args.hit_body.replace('_link', '')}{args.hit_height:+g}"
  if list(args.directions) != list(DIRECTIONS):
    tag += "_dir" + str(len(args.directions))
  if args.height is not None:
    tag += f"_h{args.height:g}"
  if stream and [float(v) for v in args.twists] not in ([0.0, 0.0, 0.0, 0.35, 0.0, 0.0], [0.0, 0.0, 0.0, 0.3, 0.0, 0.0]):
    tag += "_tw" + "_".join(f"{v:g}" for v in args.twists)
  if args.reflex_gain > 0.0:
    tag += f"_reflex{args.reflex_gain:g}"
  if args.graft:
    assert stream and not manufacturer and not walker, "--graft is defined for the stream policies only"
    tag += f"_graft-{args.graft.parent.name.split('_')[-1]}@{args.graft.stem.split('_')[-1]}-" + "-".join(args.graft_joints)
  course_ids = [i for i, name in enumerate(robot.JOINTS) if any(fragment in name for fragment in args.course)]
  if args.course:
    assert stream and not manufacturer and not walker and course_ids, "--course is defined for the stream policies' action layout only"
    tag += "_course-" + "-".join(args.course) + (f"-first{args.quiet_hits}" if args.quiet_hits else "")
  if args.rate_feedback != 0.0 or args.rate_feedback_hip != 0.0:
    assert stream and not manufacturer and not walker, "--rate-feedback is defined for the stream policies' action layout only"
    tag += f"_ratefb{args.rate_feedback:g}-hip{args.rate_feedback_hip:g}x{args.rate_feedback_steps}"
  if args.leg_boost != 1.0:
    assert stream and not manufacturer and not walker, "--leg-boost is defined for the stream policies' action layout only"
    tag += f"_legboost{args.leg_boost:g}x{args.leg_boost_window:g}s"
  if args.quiet != "none":
    assert stream and not manufacturer and not walker, "--quiet is defined for the stream policies' action layout only"
    tag += f"_quiet-{args.quiet}" + (f"-first{args.quiet_hits}" if args.quiet_hits else "")
  if args.goal_lag_grace is not None:
    tag += f"_laggrace{args.goal_lag_grace:g}"
  if args.arm_response_scale is not None:
    tag += f"_armalpha{args.arm_response_scale:g}"
  if args.repeats != 10:
    tag += f"_n{args.repeats}"
  if args.ground_reference:
    tag += f"_gref-{args.ground_reference}"
  if args.seed is not None:
    tag += f"_seed{args.seed}"
  if args.tag:
    tag += f"_{args.tag}"
    if (out_dir / f"{tag}.json").exists():
      raise FileExistsError(f"Comparison result already exists: {tag}")
  comparison_metadata = None
  twin_metadata = dict(sync_requested=args.sync_twins, full_state_rng_snapshot=False,
                       scope="same initial clip and noise-free matching where supported; no simulator/RNG snapshot branch",
                       limitation="different terrain/DR, hidden state and later gait phase are not synchronized")
  diagnostics_metadata = None
  if diagnostics_enabled:
    diagnostics_metadata = dict(evaluation_only=True, goal_lag_grace_s=args.goal_lag_grace,
      arm_response_scale=args.arm_response_scale, target_space="joint position target after intent and encoder bias, before actuator dynamics",
      arm_scope="indices 14:24 from first scheduled force step onward; leg/waist and twin targets unchanged",
      clock="updated after wrench write for this physics interval, before step; post-physics command sees age 0 "
            "while force acts; next policy sees completed interval age; terminal resets clear clock",
      clock_changes_env=args.goal_lag_grace is not None,
      actor_lag_source="S2View._lag" if args.s2_head else "GoalMotionCommand.goal_lag",
      original_s2_lag="always accumulates; S2View.update does not read command.cfg.lag_grace",
      existing_quiet="IR residual replacement; different arm intents need not give equal final targets",
      actor_sampling="exact teacher input before policy forward; pre_qpos/qvel before the corresponding physics interval",
      target_sampling="original=processed target minus encoder bias; applied=joint_pos_target after apply_actions, "
                      "including terminal intervals; downstream actuator delay unchanged",
      pre_sample_offset_s=0.0, full_state_rng_snapshot=False)
  if initialization_qa is not None:
    twin_metadata = dict(scope="explicit one-to-one own twin", full_state_rng_snapshot=False, initialization=initialization_qa)
  if comparison:
    from r1wb.eval.command_contract import NATIVE_DELTAS, NATIVE_LIMITS, native_model
    requested = torch.tensor(triples)
    support = (native_model(requested) == 0).tolist()
    if not all(support):
      raise ValueError("This WALK-only impulse witness excludes turn/run commands; use the command suite's support table")
    initial_state = dict(qpos=env.sim.data.qpos.cpu().tolist(), qvel=env.sim.data.qvel.cpu().tolist())
    comparison_metadata = dict(name=args.comparison_contract, version=1, noise=args.observation_profile if proper else "N0; no added actor/sensor noise",
                               raw_twists=triples, native_limits=NATIVE_LIMITS, native_deltas_per_20ms=NATIVE_DELTAS,
                               command_layers="manufacturer native physical slew once; S2 h12/governor/optional steer unchanged",
                               hand_mass_kg_each=robot.HAND_MASS, initial_library="corebal", ground_reference="feet",
                               initial_state=initial_state,
                               initial_state_sha256=hashlib.sha256(json.dumps(initial_state, sort_keys=True).encode()).hexdigest(),
                               manufacturer_bundle=getattr(ManufacturerPolicy, "bundle_metadata", None) if manufacturer else None)
  if proper:
    comparison_metadata.update(command_adapter=common_command.metadata(), initialization=initialization_qa,
                               case_pack_sha256=case_pack.digest, instrumentation_only=args.case_pack_smoke)
  archive = None
  if not args.no_video:
    from r1wb.eval.vector_evaluation_video import VectorVideoArchive, destination
    labels = [f"{names[c[0]][:24]}_{c[1]}_{int(c[2])}Ns_{c[3]}" for c in cases] + [f"{names[c][:24]}_twin" for c in clips]
    archive = VectorVideoArchive(env, destination(run / f"policy_{iteration}.onnx", out_dir / f"{tag}.json"), labels,
                                 compressed=args.archive_compression)
    archive.manifest.update(evaluator=__file__,
                            evaluation_args=dict({k: str(v) for k, v in vars(args).items()}, run=str(run), checkpoint=str(checkpoint)),
                            result_json=str(out_dir / f"{tag}.json"), twin_initialization=twin_metadata)
    if diagnostics_metadata is not None:
      archive.manifest["impulse_diagnostics"] = diagnostics_metadata
    library_file = Path(cfg.commands["motion"].motion_file)
    assets = [checkpoint, library_file, library_file.with_suffix(".manifest.json")]
    if manufacturer:
      assets.append(Path(ManufacturerPolicy.wrapper_path))
      bundle_metadata = getattr(ManufacturerPolicy, "bundle_metadata", None)
      if bundle_metadata:
        assets += [Path(entry["source"]) for entry in bundle_metadata["assets"]]
        assets.append(Path(bundle_metadata["manifest"]))
    if args.protocol is not None:
      assets.append(args.protocol.resolve())
    if case_pack is not None:
      assets += [case_pack.path, args.sensor_seeds.resolve()]
    if args.s2_head:
      assets += [WORK / "data/s2" / f"small_{args.s2_head}.pt", WORK / "data/s2" / f"tokenizer_{args.tokens}.pt"]
    if args.steer:
      assets += [WORK / "data/s2" / f"steer_{args.steer}.pt"]
    shared = WORK / "result/video/_assets"
    shared.mkdir(parents=True, exist_ok=True)
    archive.manifest["model_assets"] = []
    for asset in assets:
      with asset.open("rb") as stream_file:
        digest = hashlib.file_digest(stream_file, "sha256").hexdigest()
      saved = shared / (digest + asset.suffix)
      if not saved.exists():
        shutil.copyfile(asset, saved)
      archive.manifest["model_assets"].append(dict(source=str(asset), sha256=digest, snapshot=str(saved)))
    archive.begin()
    if not archive.started:
      raise RuntimeError(archive.manifest.get("reason", "Cannot preserve evaluation"))
    archive.localize_assets()
    if comparison_metadata:
      archive.manifest["comparison_contract"] = comparison_metadata
    for source in (ROOT / "src/r1wb").rglob("*.py"):
      content = source.read_bytes()
      digest = hashlib.sha256(content).hexdigest()
      saved = archive.directory / "sources" / (digest + "_" + source.name)
      saved.write_bytes(content)
      archive.manifest["source_snapshot"][str(source)] = dict(sha256=digest, snapshot=str(saved))
  sensor_hook = None
  if proper:
    from r1wb.eval.shared_sensor_observation import install_sensor_hook, trace_context
    sensor_hook = install_sensor_hook(env, manufacturer_policy.model if manufacturer else tracker, args, archive,
                                       kind="manufacturer" if manufacturer else "teacher",
                                       command=None if manufacturer else view,
                                       to_policy=manufacturer_policy.to_policy if manufacturer else None)
    comparison_metadata["observation_contract"] = sensor_hook.stream.metadata
  joint_pos = torch.zeros(horizon, count, 24, device=env.device)
  course_term = env.command_manager.get_term("motion") if (label and not args.s2_head) else None
  if args.plan_lean != 0.0:
    from r1wb.eval.diagnostics import lean_standing_plan
    assert course_term is not None, "--plan-lean changes the library's plan (label runs without --s2-head)"
    print(f"plan lean {args.plan_lean:+g} rad on {100.0 * lean_standing_plan(course_term, entity, args.plan_lean):.1f}% of the library's frames")
  course_gap = torch.zeros(horizon, count, device=env.device)
  plan_joints = None                               # the joint angles the plan asks for now (generator's course or System 2's)
  if course_term is not None:
    plan_joints = lambda: course_term.joint_pos  # noqa: E731
  elif args.s2_head:
    def plan_joints():
      reference, index = view._dense()
      return reference.frames[torch.arange(count, device=env.device), reference._spot(index)][:, :24]
  joint_vel = torch.zeros(horizon, count, 24, device=env.device)
  margin = torch.zeros(horizon, len(cases), device=env.device)
  hull = torch.zeros(horizon, len(cases), device=env.device)
  contact = torch.zeros(horizon, count, 2, dtype=torch.bool, device=env.device)
  smooth = torch.zeros(horizon, count, 2, device=env.device)
  model = env.sim.mj_model
  ctrl = [int(i) for i in entity.indexing.ctrl_ids]
  leg_actuators = torch.tensor([any(k in model.actuator(i).name for k in ("hip", "knee", "ankle")) for i in ctrl], device=env.device)
  effort = torch.tensor(np.abs(model.actuator_forcerange[ctrl]).max(1), dtype=torch.float, device=env.device)
  trace = dict(com=torch.zeros(horizon, len(cases), 2, device=env.device), velocity=torch.zeros(horizon, len(cases), 2, device=env.device),
               sole=torch.zeros(horizon, len(cases), 2, 2, device=env.device), tilt=torch.zeros(horizon, len(cases), device=env.device),
               saturated=torch.zeros(horizon, len(cases), device=env.device), leg_speed=torch.zeros(horizon, len(cases), device=env.device))
  push_world = torch.zeros(len(cases), 2, device=env.device)
  raw = dict(target=torch.zeros(horizon, count, 24, device=env.device), gravity=torch.zeros(horizon, count, 3, device=env.device),
             gyro=torch.zeros(horizon, count, 3, device=env.device), height=torch.zeros(horizon, count, device=env.device),
             root_position=torch.zeros(horizon, count, 3, device=env.device),
             root_quaternion=torch.zeros(horizon, count, 4, device=env.device),
             joint_torque=torch.zeros(horizon, count, 24, device=env.device),
             force=torch.zeros(horizon, count, 24, device=env.device)) if args.trace else None
  target_intervention = label_hook = applied_clock = None
  if diagnostics_enabled:
    from r1wb.eval.impulse_diagnostics import AppliedImpulseClock, TargetIntervention
    assert raw is not None
    term = env.action_manager.get_term("joint_pos")
    if tuple(term.target_names) != tuple(robot.JOINTS):
      raise ValueError("arm diagnostic requires the documented 24-joint action target order")
    for key, width in (("pre_qpos", env.sim.data.qpos.shape[1]), ("pre_qvel", env.sim.data.qvel.shape[1]),
                       ("pd_target_original", 24), ("pd_target_applied", 24)):
      raw[key] = torch.zeros(horizon, count, width, device=env.device)
    for key in ("applied_clock_at_policy", "applied_clock_for_interval", "episode_length_before"):
      raw[key] = torch.zeros(horizon, count, device=env.device)
    raw["applied_force_active"] = torch.zeros(horizon, count, dtype=torch.bool, device=env.device)
    applied_clock = AppliedImpulseClock(count, env.device, dt)
    if args.goal_lag_grace is not None:
      env.hit_clock = applied_clock.current
    target_intervention = TargetIntervention(term, twin_of, hit_step, args.arm_response_scale, raw)
    if not args.s2_head:
      label_model = env.observation_manager.get_term_cfg("bc", "label").func.model
      def capture_teacher_input(model_, inputs):
        diagnostic_input["actor"] = inputs[0]["actor"].detach().clone()
      label_hook = label_model.register_forward_pre_hook(capture_teacher_input)
  reference_term = env.command_manager.get_term("motion") if (args.trace and label) else None
  if reference_term is not None:              # label modes: the course the teacher is asked to follow
    raw["reference"] = torch.zeros(horizon, count, 24, device=env.device)
  if args.steer and not args.s2_head:
    from r1wb.s2.steer import PERIOD as STEER_PERIOD, Steer, SteerPolicy
    assert label and stream, "--steer drives the course generator of a stream_label run"
    steer_policy = SteerPolicy.load(WORK / "data/s2" / f"steer_{args.steer}.pt", args.device)
    steerer = Steer(env, env.command_manager.get_term("motion"), steer_policy.see, leak=steer_policy.leak, remember=steer_policy.remember, turned=steer_policy.turned)
    course_policy, ticks = policy, [0]

    steer_log = []                                  # per decision: the command, the gap seen from the body, the heading gap

    def policy(obs):                                # called once before every control step
      if ticks[0] > 0:
        steerer.progress.after_step()
      if ticks[0] % steer_policy.period == 0:
        steerer.act(steer_policy.best(steerer.observe()))
        gap, heading = steerer.progress.error()
        steer_log.append(torch.cat([steerer.command, gap, heading.unsqueeze(1)], dim=1).cpu())
      ticks[0] += 1
      return course_policy(obs)
  running = None
  alive = torch.ones(count, dtype=torch.bool, device=env.device)
  fell_step = torch.full((count,), horizon, device=env.device)
  observation = wrapped.get_observations()
  started = time.time()
  if proper and raw is not None:
    for key in ("external_force_world", "external_torque_world", "external_force_body", "external_torque_body"):
      raw[key] = torch.zeros(horizon, count, 3, device=env.device)
  for step in range(horizon):
    if common_command is not None:
      common_command.advance(step, requested_twists)
      common_commands.append(common_command.delivered.clone())
    if sensor_hook is not None:
      sensor_hook.begin(step, **trace_context(env, alive, requested_twist=torch.tensor([triples[i] for i in order], device=env.device)))
    with torch.no_grad():
      action = policy(observation)
    if sensor_hook is not None:
      sensor_hook.finish()
    if diagnostics_enabled:
      seen = diagnostic_input["actor"]
      if seen is None:
        raise RuntimeError("Exact teacher actor input was not captured")
      if step == 0:
        if args.goal_lag_grace is not None and seen.shape[1] < 294:
          raise ValueError("goal-lag diagnostic requires a task teacher that consumes goal_lag")
        raw["actor_input"] = torch.zeros(horizon, *seen.shape, device=env.device)
        raw["actor_goal_lag"] = torch.zeros(horizon, count, 3 if seen.shape[1] >= 294 else 0, device=env.device)
        diagnostics_metadata["actor_goal_lag_slice"] = [291, 294] if seen.shape[1] >= 294 else None
      raw["actor_input"][step] = seen
      raw["actor_goal_lag"][step] = seen[:, 291:294] if seen.shape[1] >= 294 else seen[:, :0]
      raw["pre_qpos"][step], raw["pre_qvel"][step] = env.sim.data.qpos, env.sim.data.qvel
      raw["applied_clock_at_policy"][step] = applied_clock.current
      raw["episode_length_before"][step] = env.episode_length_buf
      target_intervention.step = step
    if course_ids:
      after = torch.zeros(count, 1, dtype=torch.bool, device=env.device)
      after[:len(cases), 0] = step >= hit_step
      if args.quiet_hits:
        after[:len(cases), 0] &= step < hit_step + args.quiet_hits * gap_steps
      source = torch.cat([twin_of, torch.arange(len(cases), count, device=env.device)])
      twin_action = action[source]
      action = action.clone()
      action[:, course_ids] = torch.where(after, twin_action[:, course_ids], action[:, course_ids])
    if args.rate_feedback != 0.0 or args.rate_feedback_hip != 0.0:
      since_hit = step - hit_step
      window = (since_hit >= 1) & (since_hit % gap_steps <= args.rate_feedback_steps) & (since_hit % gap_steps >= 1) & (since_hit < args.burst * gap_steps)
      rate = entity.data.root_link_ang_vel_b[:len(cases), 1] * window.float()
      action = action.clone()
      scale = torch.tensor([robot.ACTION_SCALE[name] for name in robot.JOINTS], device=env.device)
      for index in (4, 10):       # ankle pitch, left and right
        action[:len(cases), index] += args.rate_feedback * rate / scale[index]
      for index in (0, 6):        # hip pitch, left and right
        action[:len(cases), index] += args.rate_feedback_hip * rate / scale[index]
    if args.leg_boost != 1.0:
      since_hit = step - hit_step
      early = torch.zeros(count, 1, dtype=torch.bool, device=env.device)
      early[:len(cases), 0] = (since_hit >= 0) & (since_hit % gap_steps < round(args.leg_boost_window / dt)) & (since_hit < args.burst * gap_steps)
      source = torch.cat([twin_of, torch.arange(len(cases), count, device=env.device)])
      reference = action[source]
      action = action.clone()
      action[:, :12] = torch.where(early, reference[:, :12] + args.leg_boost * (action[:, :12] - reference[:, :12]), action[:, :12])
    if args.quiet != "none":
      after = torch.zeros(count, 1, dtype=torch.bool, device=env.device)
      after[:len(cases), 0] = step >= hit_step
      if args.quiet_hits:
        after[:len(cases), 0] &= step < hit_step + args.quiet_hits * gap_steps
      source = torch.cat([twin_of, torch.arange(len(cases), count, device=env.device)])
      twin_action = action[source]
      action = action.clone()
      arms = torch.zeros_like(action[:, 14:]) if args.quiet.endswith("0") else twin_action[:, 14:]
      action[:, 14:] = torch.where(after, arms, action[:, 14:])
      if args.quiet.startswith("upper"):
        action[:, 12:14] = torch.where(after, twin_action[:, 12:14], action[:, 12:14])
    since = step - hit_step
    hitting = case_pack.schedule(step, env.device) if case_pack is not None else ((since >= 0) & (since < args.burst * gap_steps) & (since % gap_steps < hit_steps))
    if args.hold_after_hit > 0.0:      # what the training event maintains: seconds since the last external hit
      last = since - torch.where(since >= 0, (since.clamp(max=args.burst * gap_steps - 1) // gap_steps) * gap_steps, torch.zeros_like(since))
      env.hit_clock = torch.full((count,), 1.0e3, device=env.device)
      env.hit_clock[:len(cases)] = torch.where(since >= 0, last.float() * dt, torch.full((len(cases),), 1.0e3, device=env.device))
    forces = torch.zeros(count, 1, 3, device=env.device)
    torques = torch.zeros(count, 1, 3, device=env.device)
    if thrower is not None:
      early = since + throw_lead                            # steps since the throw of the first body
      throwing = (early >= 0) & (early < args.burst * gap_steps) & (early % gap_steps == 0)
      if throwing.any():
        ids = throwing.nonzero().flatten()
        heading = yaw_quat(entity.data.root_link_quat_w[ids])
        along = quat_apply(heading, local[ids])
        along = along / torch.norm(along, dim=1, keepdim=True).clamp_min(1e-6)
        target = entity.data.body_com_pos_w[ids, torso] + torch.tensor([0.0, 0.0, args.hit_height], device=env.device)
        thrower.throw(ids, target, along, throw_speed[ids], entity.data.body_com_lin_vel_w[ids, torso])
        first = (early == 0).unsqueeze(1)
        heading_all = yaw_quat(entity.data.root_link_quat_w[:len(cases)])
        flight = quat_apply(heading_all, local)[:, :2]
        push_world = torch.where(first, flight / torch.norm(flight, dim=1, keepdim=True).clamp_min(1e-6), push_world)
      hitting = torch.zeros_like(hitting)                   # no applied force: the contact does it
    if hitting.any():
      heading = yaw_quat(entity.data.root_link_quat_w[:len(cases)])
      along_world = case_tensors["forces_world_unit"] if proper else quat_apply(heading, local)
      world = along_world * (impulse / (hit_steps * dt)).unsqueeze(1)
      arm = torch.tensor([0.0, 0.0, args.hit_height], device=env.device).expand(len(cases), 3)
      forces[:len(cases), 0] = torch.where(hitting.unsqueeze(1), world, torch.zeros_like(world))
      first = (since == 0).unsqueeze(1)
      along = along_world[:, :2]
      push_world = torch.where(first, along / torch.norm(along, dim=1, keepdim=True).clamp_min(1e-6), push_world)
      torques[:len(cases), 0] = torch.where(hitting.unsqueeze(1), torch.cross(arm, world, dim=1), torch.zeros_like(world))
    if proper and raw is not None:
      from mjlab.utils.lab_api.math import quat_apply_inverse
      body_quat = entity.data.body_link_quat_w[:, torso]
      raw["external_force_world"][step] = forces[:, 0]
      raw["external_torque_world"][step] = torques[:, 0]
      raw["external_force_body"][step] = quat_apply_inverse(body_quat, forces[:, 0])
      raw["external_torque_body"][step] = quat_apply_inverse(body_quat, torques[:, 0])
    entity.write_external_wrench_to_sim(forces, torques, body_ids=[torso])
    if diagnostics_enabled:
      active = (forces.abs().sum(dim=(1, 2)) + torques.abs().sum(dim=(1, 2))) > 0.0
      current_clock = applied_clock.applied(step, active)
      raw["applied_clock_for_interval"][step] = current_clock
      raw["applied_force_active"][step] = active
      if args.goal_lag_grace is not None:
        env.hit_clock = current_clock
    observation, _, dones, _ = wrapped.step(action)
    if sensor_hook is not None:
      sensor_hook.after_step(dones, env.termination_manager.terminated, alive)
    if diagnostics_enabled:
      applied_clock.reset(dones.bool())
    if thrower is not None:
      thrower.after_step(dt)
    if archive is not None:
      archive.after_step()
    state = centroidal.state()
    twin_velocity = state["velocity"][twin_of, :2]
    reference = torch.cat([twin_velocity, torch.zeros(len(clips), 2, device=env.device)])
    result = centroidal.capture_margin(reference)
    hull[step] = centroidal.support_margin(reference)["d"][:len(cases)]
    joint_pos[step], joint_vel[step] = entity.data.joint_pos, entity.data.joint_vel
    if plan_joints is not None:                # how far the joints are from the plan (what the posture price charges)
      course_gap[step] = (entity.data.joint_pos - plan_joints()).abs().mean(dim=1)
    margin[step] = result["d1"][:len(cases)]
    contact[step] = state["contact"]
    running = state["velocity"][:, :2].clone() if running is None else running + (state["velocity"][:, :2] - running) * (dt / 0.4)
    smooth[step] = running
    n = len(cases)
    trace["com"][step], trace["velocity"][step], trace["sole"][step] = state["com"][:n, :2], state["velocity"][:n, :2], state["sole"][:n]
    trace["tilt"][step] = torch.acos((-entity.data.projected_gravity_b[:n, 2]).clamp(-1.0, 1.0))
    force = entity.data.actuator_force[:n].abs()
    trace["saturated"][step] = ((force >= 0.95 * effort) & leg_actuators).float().sum(1) / leg_actuators.sum()
    trace["leg_speed"][step] = entity.data.joint_vel[:n, :12].abs().amax(1)
    if raw is not None:
      raw["target"][step], raw["gravity"][step] = entity.data.joint_pos_target, entity.data.projected_gravity_b
      raw["gyro"][step], raw["force"][step] = entity.data.root_link_ang_vel_b, entity.data.actuator_force
      raw["height"][step] = entity.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
      raw["root_position"][step] = entity.data.root_link_pos_w
      raw["root_quaternion"][step] = entity.data.root_link_quat_w
      raw["joint_torque"][step] = entity.data.qfrc_actuator
      if reference_term is not None:
        raw["reference"][step] = plan_joints() if plan_joints is not None else reference_term.joint_pos
    failed = alive & dones.bool() & env.termination_manager.terminated
    fell_step = torch.where(failed, torch.full_like(fell_step, step), fell_step)
    alive &= ~dones.bool()
  entity.write_external_wrench_to_sim(torch.zeros(count, 1, 3, device=env.device), torch.zeros(count, 1, 3, device=env.device),
                                      body_ids=[torso])
  if thrower is not None:                       # what the contacts made of the thrown momentum
    given = thrower.given[:, :len(cases)]
    print("thrown bodies: momentum the body lost along its flight (N s), median over the throws of a size")
    for size in sorted({c[2] for c in cases}):
      mine = torch.tensor([c[2] == size for c in cases], device=env.device)
      values = given[:, mine].flatten()
      values = values[~torch.isnan(values)]
      if len(values):
        print(f"  thrown {size:5.1f} N s ({thrower.kind}, {size / thrower.mass:4.1f} m/s): given {float(values.median()):5.1f}"
              f"  (10% {float(values.quantile(0.1)):5.1f}, 90% {float(values.quantile(0.9)):5.1f}, n {len(values)})")
  groups = {"legs": list(range(12)), "waist": [12, 13], "arms": list(range(14, 24))}
  window2, window3, hold = int(2.0 / dt), int(3.0 / dt), int(0.5 / dt)
  rows = []
  twin_falls = [int(fell_step[len(cases) + i]) < horizon for i in range(len(clips))]
  for index, (clip, direction, size, repeat) in enumerate(cases):
    twin = int(twin_of[index])
    start = int(hit_step[index])
    end = min(int(fell_step[index]), horizon)
    fell = end < horizon
    delta = (joint_pos[start:end, index] - joint_pos[start:end, twin]).abs()
    span = delta[:window2]
    moved = (span.amax(0) > 0.1) if len(span) else torch.zeros(24, dtype=torch.bool, device=env.device)
    speed = (joint_vel[start:end, index] - joint_vel[start:end, twin])[:window2]
    inside = (delta < 0.1).all(1).cpu().numpy()
    rejoin = None
    for k in range(len(inside) - hold):
      if inside[k:k + hold].all():
        rejoin = k * dt
        break
    calm = (torch.norm(smooth[start:end, index] - smooth[start:end, twin], dim=1) < 0.08).cpu().numpy()
    settle = None
    for k in range(len(calm) - hold):
      if calm[k:k + hold].all():
        settle = k * dt
        break
    positive = margin[start:end, index] > dynamics.DEAD_BAND
    touch = (contact[start + 1:end, index] & ~contact[start:end - 1, index])[:window3].sum()
    twin_touch = (contact[start + 1:end, twin] & ~contact[start:end - 1, twin])[:window3].sum()
    early = slice(start, min(end, start + int(0.6 / dt)))
    push = push_world[index]
    landings = (contact[start + 1:end, index] & ~contact[start:end - 1, index]).nonzero()      # (k, 2): step offset, foot
    placement = [round(float(((trace["sole"][start + 1 + int(k), index, int(f)] - trace["sole"][start + 1 + int(k), index, 1 - int(f)]) * push).sum()), 3)
                 for k, f in landings[:3]]
    anatomy = dict(first_touch_s=round((int(landings[0, 0]) + 1) * dt, 2) if len(landings) else None, step_m=placement,
                   peak_speed=round(float((trace["velocity"][start:end, index] * push).sum(1).max()), 2) if end > start else None,
                   com_travel_m=round(float(((trace["com"][end - 1, index] - trace["com"][start, index]) * push).sum()), 2) if end > start else None,
                   peak_tilt=round(float(trace["tilt"][start:end, index].max()), 2) if end > start else None,
                   leg_saturation=round(float(trace["saturated"][early, index].mean()), 3) if end > start else None,
                   leg_speed_peak=round(float(trace["leg_speed"][early, index].max()), 1) if end > start else None)
    rows.append(dict(clip=names[clip], context=manifest["clips"][clip]["context"], direction=direction, impulse_ns=size, **anatomy,
                     hit_s=round(start * dt, 2), fell=bool(fell), fell_after_s=round((end - start) * dt, 2) if fell else None,
                     joints_moved=int(moved.sum()), **{f"{g}_moved": int(moved[ids].sum()) for g, ids in groups.items()},
                     arm_extra_speed=round(float(speed[:, groups["arms"]].pow(2).mean().sqrt()), 3) if len(speed) else None,
                     leg_extra_speed=round(float(speed[:, groups["legs"]].pow(2).mean().sqrt()), 3) if len(speed) else None,
                     rejoin_s=None if rejoin is None else round(rejoin, 2),
                     settle_s=None if settle is None else round(settle, 2),
                     urgent_s=round(float(positive.sum()) * dt, 2), d1_max=round(float(margin[start:end, index].max()), 3) if end > start else None,
                     outside_s=round(float((hull[start:end, index] > dynamics.MARGIN_DEAD_BAND).sum()) * dt, 2),
                     outside_max_m=round(float(hull[start:end, index].max()), 3) if end > start else None,
                     extra_steps=int(touch - twin_touch)))
  if case_pack is not None:
    for index, (row, spec) in enumerate(zip(rows, case_pack.rows, strict=True)):
      row.update(block_id=spec["block_id"], frozen_case=spec, initialization=initialization_qa["rows"][index])
  summary = {}
  for row in rows:
    key = f'{row["context"]}|{row["clip"][:5]}|{row["direction"]}|{int(row["impulse_ns"])}'
    summary.setdefault(key, []).append(row)
  table = {}
  median = lambda values: None if not values else float(np.median(values))  # noqa: E731
  for key, items in summary.items():
    ok = [r for r in items if not r["fell"]]
    table[key] = dict(cases=len(items), falls=sum(r["fell"] for r in items),
                      joints_moved=median([r["joints_moved"] for r in ok]), arms_moved=median([r["arms_moved"] for r in ok]),
                      legs_moved=median([r["legs_moved"] for r in ok]),
                      arm_extra_speed=median([r["arm_extra_speed"] for r in ok]),
                      rejoin_s=median([r["rejoin_s"] for r in ok if r["rejoin_s"] is not None]),
                      rejoined=sum(r["rejoin_s"] is not None for r in ok),
                      settle_s=median([r["settle_s"] for r in ok if r["settle_s"] is not None]),
                      settled=sum(r["settle_s"] is not None for r in ok),
                      urgent_s=median([r["urgent_s"] for r in ok]), extra_steps=median([r["extra_steps"] for r in ok]),
                      outside_s=median([r["outside_s"] for r in ok]), outside_max_m=median([r["outside_max_m"] for r in ok]),
                      fell_after_s=median([r["fell_after_s"] for r in items if r["fell"]]),
                      first_touch_s=median([r["first_touch_s"] for r in items if r["first_touch_s"] is not None]),
                      step_m=[median([r["step_m"][k] for r in items if len(r["step_m"]) > k]) for k in range(3)],
                      **{k: median([r[k] for r in items if r[k] is not None])
                         for k in ("peak_speed", "com_travel_m", "peak_tilt", "leg_saturation", "leg_speed_peak")})
  result = dict(protocol="r1wb-impulse-probe-v2", definition=__doc__, task=args.task, run=run.name, iteration=iteration,
                checkpoint=str(checkpoint), impulse_point=f"{args.hit_body} CoM {args.hit_height:+g} m, {hit_steps} policy step(s)", record_s=horizon * dt,
                twin_falls=twin_falls, elapsed_s=round(time.time() - started, 1), summary=table, cases=rows)
  result["evaluation_args"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
  if comparison_metadata:
    if manufacturer or proper:
      commands_path = out_dir / f"{tag}.commands.npz"
      with commands_path.open("xb") as commands_file:
        np.savez_compressed(commands_file, delivered_twist=torch.stack(common_commands if proper else manufacturer_commands).cpu().numpy(),
                            raw_twist=(requested_twists if proper else target_twists).cpu().numpy(), dt=np.float64(dt), sampling="pre-action")
      comparison_metadata["commands"] = dict(path=str(commands_path), sha256=hashlib.sha256(commands_path.read_bytes()).hexdigest())
    result["comparison_contract"] = comparison_metadata
  result["effective_timing"] = dict(control_dt=dt, hit_duration_s=hit_steps * dt, burst_gap_s=gap_steps * dt,
                                    trace_sample_offset_s=dt)
  result["twin_initialization"] = twin_metadata
  if diagnostics_enabled:
    from r1wb.eval.impulse_diagnostics import pre_hit_qa
    qa_keys = ("pre_qpos", "pre_qvel", "actor_input", "pd_target_original", "pd_target_applied")
    qa = pre_hit_qa({key: raw[key].cpu().numpy() for key in qa_keys}, hit_step.cpu().numpy(),
                    twin_of.cpu().numpy(), env.scene.env_origins.cpu().numpy())
    for row, check in zip(rows, qa, strict=True):
      row["pre_hit_qa"] = check
    result["impulse_diagnostics"] = diagnostics_metadata
  if raw is not None:
    from r1wb.eval.reaction_metrics import pair_summary
    cpu = {"q": joint_pos.cpu().numpy(), "dq": joint_vel.cpu().numpy(), "contact": contact.cpu().numpy(),
           "torque": raw["joint_torque"].cpu().numpy(), "root": raw["root_position"].cpu().numpy()}
    for index, row in enumerate(rows):
      twin, start = int(twin_of[index]), int(hit_step[index])
      # Use identical windows and censor either member's first fall; a short survival must never earn low motion.
      onset = max(0, start - 1)  # post-step sample just before the first impulse; legacy fields keep their old indexing
      stop = min(onset + round(3.0 / dt) + 1, horizon, int(fell_step[index]), int(fell_step[twin]))
      sl = slice(onset, max(onset, stop))
      row["paired_motion"] = pair_summary(cpu["q"][sl, index], cpu["dq"][sl, index], cpu["contact"][sl, index],
                                           cpu["q"][sl, twin], cpu["dq"][sl, twin], cpu["contact"][sl, twin], dt,
                                           cpu["torque"][sl, index], cpu["torque"][sl, twin])
      row["paired_motion"]["complete_window"] = start > 0 and stop == onset + round(3.0 / dt) + 1
      if stop > onset + 1:
        row["paired_displacement_xy_m"] = {name: (cpu["root"][stop - 1, who, :2] - cpu["root"][onset, who, :2]).tolist()
                                             for name, who in (("actual", index), ("twin", twin))}
      bursts = int(case_tensors["burst_count"][index]) if proper else args.burst
      late_start = onset + (bursts - 1) * gap_steps + round(2.0 / dt)
      late_stop = late_start + round(2.0 / dt)
      row["late_motion"] = None
      if late_stop < min(horizon, int(fell_step[index]), int(fell_step[twin])):
        row["late_motion"] = {}
        for name, who in (("actual", index), ("twin", twin)):
          delta = cpu["root"][late_stop, who, :2] - cpu["root"][late_start, who, :2]
          row["late_motion"][name] = dict(mean_velocity_xy_m_s=(delta / 2.0).tolist())
      row["pre_hit_joint_max_difference_rad"] = float(np.abs(cpu["q"][:start, index] - cpu["q"][:start, twin]).max()) if start else None
  if raw is not None:
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.trace, joint_pos=joint_pos.cpu().numpy(), joint_vel=joint_vel.cpu().numpy(), contact=contact.cpu().numpy(),
                        hit_step=hit_step.cpu().numpy(), fell_step=fell_step.cpu().numpy(), twin_of=twin_of.cpu().numpy(),
                        push_world=push_world.cpu().numpy(), dt=dt, joints=np.array(entity.joint_names),
                        actuators=np.array([model.actuator(i).name for i in ctrl]), effort=effort.cpu().numpy(),
                        cases=np.array([f"{manifest['clips'][c[0]]['context']}|{c[1]}|{int(c[2])}|{c[3]}" for c in cases]),
                        support_margin=hull.cpu().numpy(), capture_margin=margin.cpu().numpy(), course_gap=course_gap.cpu().numpy(),
                        **{k: v.cpu().numpy() for k, v in trace.items()}, **{k: v.cpu().numpy() for k, v in raw.items()})
    with args.trace.open("rb") as trace_file:
      result["trace"] = dict(path=str(args.trace.resolve()), sha256=hashlib.file_digest(trace_file, "sha256").hexdigest())
    if archive is not None:
      archive.manifest["trace"] = result["trace"]
  if sensor_hook is not None:
    from r1wb.eval.reaction_metrics import summarize_actor_trace
    actor_trace = sensor_hook.close(horizon)
    result["proper_evaluation"] = dict(noise_profile=args.observation_profile, actor_trace=actor_trace,
                                       fixed_horizon_steps=horizon, observation_contract=sensor_hook.stream.metadata,
                                       motion_summary=summarize_actor_trace(actor_trace["path"]),
                                       interventions="none; no post-hit clock, arm scaling or learned steering")
    if archive is not None:
      archive.manifest["proper_evaluation"] = result["proper_evaluation"]
  if archive is not None:
    result["video"] = archive.close()
  (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1, default=str))
  print(f'{"context|clip|dir|Ns":34s}  n fall  joints arms legs  arm_spd rejoin_s(n) settle_s(n) urgent_s steps  outside_s/max')
  for key, v in table.items():
    f = lambda x, w=5, p=2: f'{x:{w}.{p}f}' if x is not None else ' ' * (w - 1) + '-'  # noqa: E731
    print(f'{key:34s} {v["cases"]:2d} {v["falls"]:3d}   {f(v["joints_moved"],5,1)} {f(v["arms_moved"],4,1)} {f(v["legs_moved"],4,1)}  {f(v["arm_extra_speed"],6,3)}  {f(v["rejoin_s"])}({v["rejoined"]:2d})  {f(v["settle_s"])}({v["settled"]:2d})  {f(v["urgent_s"])}  {f(v["extra_steps"],4,1)}  {f(v["outside_s"])}/{f(v["outside_max_m"],6,3)}')
  print(f'{"anatomy (all cases)":34s}  fell_after first_touch  step_m (first three)   peak_speed travel  tilt  leg_sat leg_speed')
  for key, v in table.items():
    f = lambda x, w=5, p=2: f'{x:{w}.{p}f}' if x is not None else ' ' * (w - 1) + '-'  # noqa: E731
    print(f'{key:34s}  {f(v["fell_after_s"])}      {f(v["first_touch_s"])}       {" ".join(f(x, 6) for x in v["step_m"])}   {f(v["peak_speed"])}  {f(v["com_travel_m"])} {f(v["peak_tilt"])}  {f(v["leg_saturation"],6,3)} {f(v["leg_speed_peak"],6,1)}')
  if True:
    # Reaction measured against the robot's own plan (no twin, so it also holds when System 2 re-plans after a hit),
    # and the arms' own speed: per context and impulse, mean over the 2 s after the first hit; twins = the same
    # environments' un-hit copies over the same time.
    arm_ids = [i for i, name in enumerate(robot.JOINTS) if any(k in name for k in ("shoulder", "elbow", "wrist"))]
    window2 = round(2.0 / dt)
    print("against its own plan, 2 s after the hit (hit / twin): mean joint distance to the plan (rad), mean arm joint speed (rad/s)")
    groups = {}
    for i, r in enumerate(rows):
      start, twin = int(hit_step[i]), int(twin_of[i])
      end = min(start + window2, horizon)
      groups.setdefault((r["context"], r["impulse_ns"]), []).append(
        [float(course_gap[start:end, i].mean()), float(course_gap[start:end, twin].mean()),
         float(joint_vel[start:end, i][:, arm_ids].abs().mean()), float(joint_vel[start:end, twin][:, arm_ids].abs().mean())])
    for (context, size), values in sorted(groups.items()):
      a = np.array(values).mean(0)
      plan = f"plan distance {a[0]:.3f} / {a[1]:.3f}" if plan_joints is not None else "plan distance     - /     -"
      print(f"  {context:6s} {size:5.1f} N s ({len(values):3d}): {plan}   arm speed {a[2]:.3f} / {a[3]:.3f}")
  if plan_joints is not None:
    # What staying up costs under the posture price: the joint distance to the course, summed over the 2.5 s after
    # the first hit (rad s; times the price = reward lost), for the cases that stayed up and, until they fell, the others.
    window = round(2.5 / dt)
    paid = torch.stack([course_gap[int(hit_step[i]):min(int(hit_step[i]) + window, int(fell_step[i]), horizon), i].sum() * dt for i in range(len(cases))])
    fell_case = torch.tensor([bool(r["fell"]) for r in rows], device=env.device)
    quiet = torch.stack([course_gap[max(int(hit_step[i]) - 50, 0):int(hit_step[i]), i].mean() for i in range(len(cases))])
    for label_, mask in (("stayed up", ~fell_case), ("fell", fell_case)):
      if mask.any():
        print(f"course gap, {label_} ({int(mask.sum())}): {float(paid[mask].mean()):.3f} rad s after the first hit "
              f"(largest {float(paid[mask].max()):.3f}); before the hit {float(quiet[mask].mean()):.3f} rad")
  if args.steer and not args.s2_head and args.show_steer:
    # What System 2 asked for around the hit, the hit environment next to its un-hit twin (first case of each row).
    log = torch.stack(steer_log)                                   # (decisions, envs, 6)
    shown = set()
    for i, case in enumerate(cases):
      key = (case[0], case[1], case[2])
      if key in shown:
        continue
      shown.add(key)
      first = int(hit_step[i]) // steer_policy.period
      twin = int(twin_of[i])
      print(f"System 2 around the hit, {names[case[0]][:20]} {case[1]} {case[2]} N s (decision: command vx vy wz | gap x y, heading; hit / twin)")
      for d in range(max(first - 1, 0), min(first + 9, len(log))):
        a, b = log[d, i], log[d, twin]
        print(f"  {d - first:+3d}: {a[0]:+.2f} {a[1]:+.2f} {a[2]:+.2f} | {a[3]:+.2f} {a[4]:+.2f} {a[5]:+.2f}   /   {b[0]:+.2f} {b[1]:+.2f} {b[2]:+.2f} | {b[3]:+.2f} {b[4]:+.2f} {b[5]:+.2f}")
  print("twin falls:", twin_falls, "| total falls:", sum(r["fell"] for r in rows), "/", len(rows))
  if target_intervention is not None:
    target_intervention.close()
  if label_hook is not None:
    label_hook.remove()
  env.close()
  _ = (math, robot)


if __name__ == "__main__":
  main()
