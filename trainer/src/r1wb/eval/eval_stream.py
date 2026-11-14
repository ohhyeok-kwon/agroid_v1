"""S2-like evaluation: drive a System 1 policy with scripted command streams (local GPU, mjlab physics).

No reference motion is involved. Every environment starts from a standing frame and follows one script of
piecewise-constant goals (twist, height), slew-limited as the goal governor will do. Measured per script:
falls, how well the twist and height goals are met, steps taken while asked to stand still, joint activity while
standing, and how often the balance-urgency signals are above zero in undisturbed motion.

  <work>/.venv/bin/python -m r1wb.eval.eval_stream --run <student run dir> --app-pid <owner pid>
  ... --policy manufacturer --run <any run dir for the output location>     (comparison, twist goals only)
  ... --policy label --run <teacher run dir> [--checkpoint model_N.pt]      (the tracking teacher on a
        state-matched reference: the labels a stream-trained student would imitate)
`--variant hardened` keeps the randomised dynamics and sensor noise of the training scene (no pushes).
`--matched` gives a student run the state-matched reference too (only to log how well a reference can be found).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
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
from mjlab.utils.lab_api.math import quat_apply, quat_apply_inverse, quat_inv, quat_mul, yaw_quat  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from r1wb import dynamics, robot  # noqa: E402
from r1wb.student.observations import load_teacher  # noqa: E402
from r1wb.student.stream import stream_env_cfg  # noqa: E402

from r1wb.eval.paths import AGENT as WORK, ROOT  # noqa: E402,F401
DEFAULT_HEIGHT = 0.731


def seg(duration, vx=0.0, vy=0.0, wz=0.0, height=None, pitch=None):
  return [duration, vx, vy, wz, DEFAULT_HEIGHT if height is None else height, 0.0 if height is None else 1.0,
          0.0 if pitch is None else pitch, 0.0 if pitch is None else 1.0]


# Every script lasts 20 s. Height goals are off unless a segment names one.
SUITE = {
  "stand": [seg(20)],
  "stand_height_on": [seg(20, height=DEFAULT_HEIGHT)],
  "forward_0.2": [seg(2), seg(10, vx=0.2), seg(8)],
  "forward_0.35": [seg(2), seg(10, vx=0.35), seg(8)],
  "forward_0.5": [seg(2), seg(10, vx=0.5), seg(8)],
  "backward_0.2": [seg(2), seg(10, vx=-0.2), seg(8)],
  "left_0.2": [seg(2), seg(10, vy=0.2), seg(8)],
  "right_0.2": [seg(2), seg(10, vy=-0.2), seg(8)],
  "turn_left_0.6": [seg(2), seg(10, wz=0.6), seg(8)],
  "turn_right_0.6": [seg(2), seg(10, wz=-0.6), seg(8)],
  "arc_left": [seg(2), seg(10, vx=0.35, wz=0.4), seg(8)],
  "arc_right": [seg(2), seg(10, vx=0.35, wz=-0.4), seg(8)],
  "diagonal": [seg(2), seg(10, vx=0.3, vy=0.15), seg(8)],
  "speed_steps": [seg(2), seg(4, vx=0.2), seg(4, vx=0.5), seg(4, vx=0.2), seg(6)],
  "reverse": [seg(2), seg(6, vx=0.3), seg(6, vx=-0.2), seg(6)],
  "stop_go": [seg(2), seg(2, vx=0.4), seg(2), seg(2, vx=0.4), seg(2), seg(2, vx=0.4), seg(8)],
  "lower_stand": [seg(2, height=DEFAULT_HEIGHT), seg(5, height=0.62), seg(5, height=0.54), seg(8, height=DEFAULT_HEIGHT)],
  "low_walk": [seg(2, height=DEFAULT_HEIGHT), seg(4, height=0.62), seg(8, vx=0.2, height=0.62), seg(6, height=DEFAULT_HEIGHT)],
}
USES_HEIGHT = ("stand_height_on", "lower_stand", "low_walk")

# Command floor: one command kind at increasing size, from standing. Read the smallest size at which the ratio of
# travelled to commanded motion reaches 0.8: below it System 2 sends "walk briefly, then stop" instead.
SWEEP = {}
for size in (0.1, 0.15, 0.2, 0.3):
  SWEEP[f"forward_{size}"] = [seg(2), seg(8, vx=size), seg(4)]
for size in (0.1, 0.15, 0.2, 0.3, 0.4):
  SWEEP[f"backward_{size}"] = [seg(2), seg(8, vx=-size), seg(4)]
for size in (0.1, 0.2, 0.3, 0.4):
  SWEEP[f"left_{size}"] = [seg(2), seg(8, vy=size), seg(4)]
  SWEEP[f"right_{size}"] = [seg(2), seg(8, vy=-size), seg(4)]
for size in (0.3, 0.6, 0.9, 1.2):
  SWEEP[f"turn_left_{size}"] = [seg(2), seg(8, wz=size), seg(4)]
  SWEEP[f"turn_right_{size}"] = [seg(2), seg(8, wz=-size), seg(4)]


# Time since an instruction to stand, in which the landings of a stop are counted (s): edges and their labels.
STOP_EDGES = (0.5, 1.0, 1.5, 2.0, 3.0)
STOP_LABELS = ("<0.5", "0.5-1", "1-1.5", "1.5-2", "2-3", ">3")

# Eight walking directions at two speeds, turning in place, standing and stop-and-go: "does it go where it is told".
# Every script: 2 s standing, 10 s of the command, 8 s standing (the quiet measures are taken there).
COMPASS = {"stand": [seg(20)]}
for speed in (0.2, 0.35):
  for name, (x, y) in {"forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0),
                       "forward_left": (0.7071, 0.7071), "forward_right": (0.7071, -0.7071),
                       "backward_left": (-0.7071, 0.7071), "backward_right": (-0.7071, -0.7071)}.items():
    COMPASS[f"{name}_{speed}"] = [seg(2), seg(10, vx=round(speed * x, 4), vy=round(speed * y, 4)), seg(8)]
for rate in (0.6, 1.0):
  COMPASS[f"turn_left_{rate}"] = [seg(2), seg(10, wz=rate), seg(8)]
  COMPASS[f"turn_right_{rate}"] = [seg(2), seg(10, wz=-rate), seg(8)]
COMPASS["stop_go"] = SUITE["stop_go"]
# Crouch and rise (the same times as the spoken suite of --policy s2).
for depth, height in (("slight", 0.62), ("deep", 0.54)):
  COMPASS[f"crouch_{depth}"] = [seg(1.92, height=None), seg(8.0, height=height), seg(9.92, height=DEFAULT_HEIGHT)]


def compass_instructions() -> dict:
  """The compass suite spoken as instructions (r1wb.s2.instructions); durations on System 2's 0.32 s grid."""
  stand = {"kind": "stand"}
  out = {"stand": [(stand, 19.84)]}
  for speed in ("slow", "normal"):
    for direction in ("forward", "backward", "left", "right", "forward_left", "forward_right", "backward_left",
                      "backward_right"):
      walk = {"kind": "walk", "direction": direction, "speed": speed}
      out[f"{direction}_{ {'slow': 0.2, 'normal': 0.35}[speed] }"] = [(stand, 1.92), (walk, 9.92), (stand, 8.0)]
  for rate, value in (("slow", 0.6), ("fast", 1.0)):
    for side in ("left", "right"):
      out[f"turn_{side}_{value}"] = [(stand, 1.92), ({"kind": "turn", "side": side, "rate": rate}, 9.92), (stand, 8.0)]
  go = {"kind": "walk", "direction": "forward", "speed": "normal"}
  out["stop_go"] = [(stand, 1.92), (go, 1.92), (stand, 1.92), (go, 1.92), (stand, 1.92), (go, 1.92), (stand, 8.32)]
  for depth, height in (("slight", 0.62), ("deep", 0.54)):
    out[f"crouch_{depth}"] = [(stand, 1.92), ({"kind": "crouch", "depth": depth}, 8.0), ({"kind": "rise"}, 9.92)]
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--policy", choices=["student", "manufacturer", "label", "residual", "s2"], default="student")
  parser.add_argument("--s2", default="", help="s2: name of the System 2 model in <work>/data/s2 (small_<name>.pt), or "
                                               "with --backbone the folder of a trained action model (vla_last.pt)")
  parser.add_argument("--backbone", type=Path, default=None, help="s2: the vision-language backbone (r1wb.s2.vla)")
  parser.add_argument("--head", default="", help="s2: motion-code head small_<name>.pt (goal-conditioned): the goal comes "
                                                 "from --goal-model (folder with goal_last.pt, needs --backbone) or, "
                                                 "without it, from the instruction itself")
  parser.add_argument("--goal-model", default="")
  parser.add_argument("--tokens", default="v1a", help="s2: motion tokenizer (tokenizer_<name>.pt)")
  parser.add_argument("--phrases", choices=["train", "heldout"], default="train", help="s2: wording of the instructions")
  parser.add_argument("--language", choices=["ko", "en"], default="ko")
  parser.add_argument("--base", type=Path, default=None, help="residual: the frozen S1-a checkpoint the correction is added to")
  parser.add_argument("--arm-base-scale", type=float, default=1.0, help="residual: as in the training task (wb_harden_restrained: 0.3)")
  parser.add_argument("--harden-task", default=None,
                      help="residual: build the environment of this registered task instead of the plain "
                           "correction with --arm-base-scale")
  parser.add_argument("--tag", default="", help="suffix of the result file name (diagnostic variants)")
  parser.add_argument("--anchor", type=Path, default=None,
                      help="student: also report the mean |action - action of this checkpoint on the same observation| "
                           "(how far a policy continued from that checkpoint has moved in undisturbed motion)")
  parser.add_argument("--matched", action="store_true")
  parser.add_argument("--match-pose", choices=["robot", "reference"], default="robot",
                      help="label policy: whose feet enter the reference look-up")
  parser.add_argument("--commit", action="store_true",
                      help="label: the course plays a chosen clip through (student.stream match_commit)")
  parser.add_argument("--match-height", choices=["stream", "target"], default="stream",
                      help="label: the height a look-up asks for while the command moves (student.stream match_height)")
  parser.add_argument("--course-set", nargs="*", default=[],
                      help="label: other settings of the course generator as name=value (student.stream "
                           "MatchedStreamCommandCfg), e.g. match_good_enough=1.0")
  parser.add_argument("--match-stop", choices=["room", "arrive"], default="room",
                      help="label: the frames a look-up may choose for \"stand\" (student.stream match_stop)")
  parser.add_argument("--chunk", default="",
                      help="label: the course reaches the policy as System 2 chunks (eval.chunk_view), e.g. "
                           "period=10,latency=5,horizon=50,noise=jump,sigma=0.03")
  parser.add_argument("--variant", choices=["train", "hardened"], default="train")
  parser.add_argument("--progress-feedback", type=float, default=0.0,
                      help="DIAGNOSTIC, label policy, evaluation only (never deployed, never trained on): every 0.32 s "
                           "the command given to the course generator is the instruction plus (instructed path - "
                           "realised path) / this many seconds, from the simulator's odometry. An upper bound for a "
                           "slow layer that sees the realised progress; the scores are still taken against the instruction.")
  parser.add_argument("--steer", default="", help="label policy: a learned System 2 (r1wb.s2.steer, data/s2/steer_<name>.pt) "
                                                 "gives the course generator its command every 0.32 s; the scores are "
                                                 "taken against the instruction")
  parser.add_argument("--stagger", type=float, default=0.0,
                      help="every repeat of a script holds each of its moving segments this much longer than the "
                           "repeat before it (s; a multiple of 0.32 with --policy s2), and the run is lengthened to "
                           "fit: the stops then come at different moments of the gait. Without it the repeats of "
                           "a generator-driven script are the same five trajectories over and over")
  parser.add_argument("--world", default="",
                      help="the ground (r1wb.world: rough<cm>, gravel<cm>, up<deg>, down<deg>, side<deg>); heights are "
                           "then taken above the ground under the pelvis")
  parser.add_argument("--ground-reference", nargs="?", const="pelvis", default="", choices=["pelvis", "feet"],
                      help="DIAGNOSTIC with --world: the plan's heights are raised by the ground under the robot "
                           "(pelvis) or under its feet (feet) (r1wb.world.ground_reference); the policy as trained "
                           "reads them against the start level")
  parser.add_argument("--library", default="",
                      help="label runs: the motion library the course generator plays (library_v2_<name>.npz), e.g. "
                           "corebal = the rest poses given the largest margin their stance allows (r1wb.balance)")
  parser.add_argument("--plan-lean", type=float, default=0.0,
                      help="DIAGNOSTIC, label policy: the plan's standing frames lean forward over their feet by this "
                           "many rad (r1wb.eval.diagnostics.lean_standing_plan)")
  parser.add_argument("--dump-plan", type=Path, default=None,
                      help="label / s2: save, per control step, the plan's frame (r1wb.s2.frames, 30 numbers: what the "
                           "generator or System 2 asks of System 1) next to the robot's pose, for the physics audit of "
                           "the plan (.agent/tests/r1wb/s2_plan_audit.py)")
  parser.add_argument("--replay", type=Path, default=None,
                      help="s2: the plan is a recorded motion of the robot itself (a --dump-plan file of the same suite "
                           "and repeats), written as motion codes (--tokens) and played back at the same times. Asks "
                           "whether System 1 does again what it did when that is what it is asked for.")
  parser.add_argument("--raw-replay", action="store_true", help="--replay without motion codes: the recorded frames as they are")
  parser.add_argument("--smooth-replay", type=int, default=1, help="--replay: moving average over this many steps first")
  parser.add_argument("--replay-what", choices=["robot", "plan"], default="robot",
                      help="--replay: the robot's recorded motion, or (control) the plan it was recorded under")
  parser.add_argument("--replay-goal", choices=["told", "achieved"], default="told",
                      help="--replay: the twist goal System 1 is given: the instruction's, or what the replayed motion does")
  parser.add_argument("--suite", choices=["standard", "sweep", "compass"], default="standard")
  parser.add_argument("--repeats", type=int, default=4)
  parser.add_argument("--seconds", type=float, default=20.0)
  parser.add_argument("--app-pid", type=int, required=True)
  parser.add_argument("--no-video", action="store_true", help="smoke runs only; real evaluations keep the archive")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  os.environ["RL_EVALUATION_APP_PID"] = str(args.app_pid)
  from r1wb.eval.evaluation_video import require_app_owner
  require_app_owner(bind=True)
  run = args.run.resolve()
  manufacturer = args.policy == "manufacturer"
  if manufacturer:
    from r1wb.eval.manufacturer import load as load_manufacturer
    MANUFACTURER, ManufacturerPolicy = load_manufacturer()
    checkpoint, iteration = MANUFACTURER, 0
  else:
    checkpoint = (args.checkpoint or max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))).resolve()
    iteration = int(checkpoint.stem.split("_")[1])
  suite = {"sweep": SWEEP, "compass": COMPASS}.get(args.suite, SUITE)
  spoken = None
  if args.policy == "s2":                         # the suite as instructions, on System 2's decision grid
    from r1wb.s2 import instructions
    assert args.suite == "compass", "System 2 is evaluated on the instruction suite"
    spoken = compass_instructions()
    suite = {name: [instructions.command_row(item, seconds - 0.01) for item, seconds in script]
             for name, script in spoken.items()}
  if args.suite == "sweep":
    args.seconds = min(args.seconds, 14.0)
  names = list(suite)
  count = len(names) * args.repeats
  moving = lambda row: any(abs(float(v)) > 1.0e-6 for v in row[1:4])  # noqa: E731
  if args.stagger > 0.0:
    args.seconds += args.stagger * (args.repeats - 1) * max(sum(moving(row) for row in script) for script in suite.values())
  label = args.policy in ("label", "s2")
  residual = args.policy == "residual"
  if residual:
    from r1wb.harden import harden_env_cfg
    if args.harden_task:
      from r1wb.tasks import TASKS
      cfg = TASKS[args.harden_task]["env"](play=True, teacher=str(args.base.resolve()), clean=args.variant != "hardened")
    else:
      cfg = harden_env_cfg(play=True, teacher=str(args.base.resolve()), clean=args.variant != "hardened",
                           arm_base_scale=args.arm_base_scale)
    cfg.commands["motion"].calm_fraction = 1.0             # no impulses: this is the undisturbed suite
  else:
    cfg = stream_env_cfg(play=True, clean=args.variant != "hardened", plain_actions=manufacturer,
                         matched=label or args.matched, teacher=str(checkpoint) if label else None)
  if args.policy == "s2" and args.backbone is not None:        # a vision-language model looks through the head camera
    from r1wb.s2.collect import head_camera
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (head_camera(224),)
  if label and not residual:
    cfg.commands["motion"].match_pose = args.match_pose
    cfg.commands["motion"].match_commit = args.commit
    cfg.commands["motion"].match_height = args.match_height
    cfg.commands["motion"].match_stop = args.match_stop
    if args.library:
      from r1wb.tracking.env_cfg import library_path
      cfg.commands["motion"].motion_file = str(library_path(args.library))
    for item in args.course_set:
      key, value = item.split("=")
      setattr(cfg.commands["motion"], key, type(getattr(cfg.commands["motion"], key))(float(value)))
  stand = cfg.commands["motion"].clip_list
  cfg.commands["motion"].clip_list = tuple(stand[(i % args.repeats) % len(stand)] for i in range(count))
  cfg.scene.num_envs = count
  if args.world:
    from r1wb import world
    world.apply_ground(cfg, args.world)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env)
  term = env.command_manager.get_term("motion")
  if args.ground_reference:
    assert args.world, "--ground-reference needs --world"
    world.ground_reference(env, term, args.ground_reference)   # (a System 2 view built below is raised too: the policy reads it)
  if args.plan_lean != 0.0:
    from r1wb.eval.diagnostics import lean_standing_plan
    assert args.policy == "label", "--plan-lean changes the library's plan (generator-driven runs)"
    print(f"plan lean {args.plan_lean:+g} rad on {100.0 * lean_standing_plan(term, env.scene['robot'], args.plan_lean):.1f}% of the library's frames")
  longest = max(len(script) for script in suite.values())
  table = torch.zeros(count, longest, 8)
  for index in range(count):
    script = [list(row) for row in suite[names[index // args.repeats]]]
    for row in script:
      if args.stagger > 0.0 and moving(row):
        row[0] += args.stagger * (index % args.repeats)
    rows = script + [seg(1.0e6, *script[-1][1:4])] * (longest - len(script))
    table[index] = torch.tensor(rows)
    table[index, len(script) - 1, 0] = 1.0e6                     # hold the last segment
  term.set_script(table)
  env.reset()
  observation = wrapped.get_observations()
  steerer = None
  if manufacturer:
    policy = ManufacturerPolicy(env, term.twist_goal())
  elif args.policy == "s2":
    import random as _random

    from r1wb.s2 import instructions
    from r1wb.s2.runtime import S2View, text_schedule
    from r1wb.s2.small import SmallS2
    from r1wb.s2.text import TextEncoder
    from r1wb.s2.tokenizer import MotionTokenizer
    from r1wb.tracking.env_cfg import teacher_observation
    tokenizer = MotionTokenizer.load(WORK / "data/s2" / f"tokenizer_{args.tokens}.pt", args.device)
    chooser = _random.Random(0)
    scripts = [[(item, seconds + (args.stagger * (i % args.repeats) if item["kind"] in ("walk", "turn") else 0.0))
                for item, seconds in spoken[names[i // args.repeats]]] for i in range(count)]
    wording = [[instructions.phrase(item, args.language, chooser, held_out=args.phrases == "heldout") for item, _ in script]
               for script in scripts]
    decisions = int(args.seconds / 0.32) + 2
    if args.replay is not None:                                # the robot's own recorded motion as the plan
      from r1wb.s2.realised import frames_of, replay_brain
      from r1wb.s2.runtime import table_goal, word_schedule
      from r1wb.s2.small import goal_of
      recorded = np.load(args.replay, allow_pickle=False)
      assert [str(n) for n in recorded["scripts"]] == [names[i // args.repeats] for i in range(count)], "another suite"
      meant = word_schedule(scripts, [[item for item, _ in script] for script in scripts], decisions, float(env.step_dt))
      goals = torch.tensor([[goal_of(item) for item in row] for row in meant], device=args.device)
      mind = None
      from r1wb.s2.realised import smoothed
      if args.replay_what == "plan":                           # control: the plan the recording itself was made under
        replayed = torch.as_tensor(recorded["plan"]).permute(1, 0, 2).contiguous().float().to(args.device)
      else:
        replayed = smoothed(frames_of(torch.as_tensor(recorded["joints"]), torch.as_tensor(recorded["anchor"])).to(args.device),
                            args.smooth_replay)
      if args.replay_goal == "achieved":                       # the goal System 1 is told = what the replayed motion does
        # ... one constant twist per instruction: the replayed motion's mean over the time that instruction is in force
        # (a goal that follows the motion within a stride sways with the pelvis; System 1 was never given such a goal)
        rate = replayed[:, :, 27:30] / float(env.step_dt)
        at = (9 + 16 * torch.arange(goals.shape[1], device=args.device)).clamp(max=replayed.shape[1] - 1)
        each = rate[:, at]                                                         # (envs, decisions, 3)
        achieved = torch.zeros_like(goals[:, :, :3])
        for row in range(goals.shape[0]):
          told_here = goals[row, :, :3]
          change = torch.cat([torch.ones(1, dtype=torch.bool, device=args.device), (told_here[1:] != told_here[:-1]).any(dim=1)])
          group = torch.cumsum(change.long(), dim=0) - 1
          for index in range(int(group.max()) + 1):
            mask = group == index
            if float(told_here[mask][0].abs().sum()) > 1.0e-6:
              inside = mask.nonzero().flatten()
              steady = inside[min(3, len(inside) - 1):]                            # after the first second
              first, last = int(at[steady[0]]), int(at[steady[-1]])
              achieved[row, mask] = rate[row, first:last + 1].mean(dim=0)
        goals = torch.cat([achieved, goals[:, :, 3:]], dim=2)
      brain = replay_brain(replayed, tokenizer, table_goal(goals))
    elif args.head:                                            # factorised: goal (model or table) -> code head
      from r1wb.s2.runtime import goal_brain, table_goal, vlm_goal, word_schedule
      from r1wb.s2.small import goal_of
      mind = SmallS2.load(WORK / "data/s2" / f"small_{args.head}.pt", args.device)
      spoken_words = word_schedule(scripts, wording, decisions, float(env.step_dt))
      if args.goal_model:
        from r1wb.s2.vla import GoalVLA
        reader = GoalVLA(args.backbone).to(args.device).eval()
        weights = torch.load(WORK / "data/s2" / args.goal_model / "goal_last.pt", map_location="cpu", weights_only=False)
        reader.load_state_dict(weights["state"], strict=False)
        vocabulary = torch.tensor([goal_of(item) for item in instructions.every_instruction()], device=args.device)
        state_goal = vlm_goal(reader, env, spoken_words, vocabulary=vocabulary)
      else:
        meant = word_schedule(scripts, [[item for item, _ in script] for script in scripts], decisions, float(env.step_dt))
        goals = torch.tensor([[goal_of(item) for item in row] for row in meant], device=args.device)
        state_goal = table_goal(goals)
      if args.steer:                                           # the learned slow loop between the goal and the code head
        from r1wb.s2.steer import Steer, SteerPolicy, steered_goal
        steer_policy = SteerPolicy.load(WORK / "data/s2" / f"steer_{args.steer}.pt", args.device)
        steerer = Steer(env, term, steer_policy.see, generator=False, leak=steer_policy.leak, remember=steer_policy.remember, turned=steer_policy.turned)
        state_goal = steered_goal(state_goal, steerer, steer_policy)
      brain = goal_brain(mind, state_goal)
    elif args.backbone is not None:                            # the vision-language action model
      from r1wb.s2.runtime import vla_brain, word_schedule
      from r1wb.s2.vla import MotionVLA
      mind = MotionVLA(args.backbone, tokenizer.dims, tokenizer.levels).to(args.device).eval()
      weights = torch.load(WORK / "data/s2" / args.s2 / "vla_last.pt", map_location="cpu", weights_only=False)
      mind.load_state_dict(weights["state"], strict=False)
      brain = vla_brain(mind, env, word_schedule(scripts, wording, decisions, float(env.step_dt)))
    else:
      from r1wb.s2.runtime import small_brain
      mind = SmallS2.load(WORK / "data/s2" / f"small_{args.s2}.pt", args.device)
      brain = small_brain(mind, text_schedule(scripts, wording, TextEncoder(WORK / "data/s2/text_cache.pt"), decisions,
                                              float(env.step_dt), args.device))
    if args.replay is not None and args.raw_replay:
      from r1wb.s2.realised import raw_replay_view
      view = raw_replay_view(env, term, replayed, tokenizer, table_goal(goals))
    else:
      view = S2View(env, term, brain, tokenizer)
    tracker = load_teacher(str(checkpoint), args.device)
    joint_action = env.action_manager.get_term("joint_pos")

    def policy(obs):
      view.update()
      seen = teacher_observation(view.env, "motion", joint_action.plain_action, local=True,
                                 width=tracker.obs_dim)
      return joint_action.from_plain_action(tracker(TensorDict({"actor": seen}, batch_size=[seen.shape[0]])))
  elif label and args.chunk:
    from r1wb.eval.chunk_view import ChunkView
    from r1wb.tracking.env_cfg import teacher_observation
    settings = dict(item.split("=") for item in args.chunk.split(","))
    transform, lead = None, 1
    if "tokens" in settings:                         # the chunk passes through a motion tokenizer and back
      from r1wb.s2.tokenizer import MotionTokenizer
      tokenizer = MotionTokenizer.load(WORK / "data/s2" / f"tokenizer_{settings['tokens']}.pt", args.device)
      transform, lead = tokenizer.roundtrip, 1 + 2 * tokenizer.stride * int(settings.get("context", 2))
    view = ChunkView(env, term, period=int(settings.get("period", 10)), latency=int(settings.get("latency", 5)),
                     horizon=int(settings.get("horizon", 50)), noise=settings.get("noise", "jump"),
                     sigma=float(settings.get("sigma", 0.0)), dense=settings.get("dense", "0") == "1",
                     transform=transform, lead=lead)
    tracker = load_teacher(str(checkpoint), args.device)
    joint_action = env.action_manager.get_term("joint_pos")

    def policy(obs):
      view.update()
      seen = teacher_observation(view.env, "motion", joint_action.plain_action, local=True,
                                 width=tracker.obs_dim)
      return joint_action.from_plain_action(tracker(TensorDict({"actor": seen}, batch_size=[seen.shape[0]])))
  elif label:
    def policy(obs):
      return obs["bc"][:, :24]
  else:
    from r1wb.student.observations import load_policy
    actor = load_policy(str(checkpoint), args.device)

    def policy(obs):
      return actor(torch.cat([obs["actor"], obs["goal"]], dim=1))

  reader = locals().get("view", term)                        # what System 1's observation reads the plan from
  if args.ground_reference and reader is not term:
    world.ground_reference(env, reader, args.ground_reference)
  if args.steer and args.policy == "label":                   # (with --policy s2 it stands in front of the code head)
    from r1wb.s2.steer import PERIOD as STEER_PERIOD, Steer, SteerPolicy
    steer_policy = SteerPolicy.load(WORK / "data/s2" / f"steer_{args.steer}.pt", args.device)
    steerer = Steer(env, term, steer_policy.see, leak=steer_policy.leak, remember=steer_policy.remember, turned=steer_policy.turned)
  anchor = None
  if args.anchor is not None:
    from r1wb.student.observations import load_policy as load_anchor
    anchor = load_anchor(str(args.anchor.resolve()), args.device)
  entity = env.scene["robot"]
  centroidal = dynamics.Centroidal(env)
  joints = list(entity.joint_names)
  arm_ids = torch.tensor([joints.index(n) for n in robot.ARM_JOINTS], device=env.device)
  leg_ids = torch.tensor([joints.index(n) for n in robot.LEG_JOINTS + robot.WAIST_JOINTS], device=env.device)
  # `actuator_force` is in actuator order, which is not the joint order: the limits are read in the same order.
  # (Until 11-12 06:40 the limits were taken in joint order, so earlier torque_ratio_max values are not comparable.)
  model = env.sim.mj_model
  ctrl = [int(i) for i in entity.indexing.ctrl_ids]
  effort = torch.tensor(np.abs(model.actuator_forcerange[ctrl]).max(1), dtype=torch.float, device=env.device)
  actuator_names = [model.actuator(i).name for i in ctrl]
  dt = float(env.step_dt)
  horizon = int(args.seconds / dt)
  out_dir = WORK / "result/render" / run.name
  out_dir.mkdir(parents=True, exist_ok=True)
  tag = f"stream_eval_{args.variant}_" + ("" if args.suite == "standard" else args.suite + "_") + ("manufacturer" if manufacturer else
                                           ("label_" if label else "residual_" if residual else "") + str(iteration))
  if args.match_pose != "robot":
    tag += "_course-" + args.match_pose
  if args.commit:
    tag += "_commit"
  if args.match_height != "stream":
    tag += "_height-" + args.match_height
  if args.match_stop != "room":
    tag += "_stop-" + args.match_stop
  for item in args.course_set:
    tag += "_" + item.replace("match_", "").replace("=", "")
  if args.chunk:
    tag += "_chunk-" + args.chunk.replace("=", "").replace(",", "-")
  if args.policy == "s2":
    who = args.s2 or (f"head-{args.head}" + (f"+{args.goal_model}" if args.goal_model else "+table"))
    tag = tag.replace("label_", "s2_") + f"_{who}-{args.tokens}-{args.language}-{args.phrases}".replace("/", "-")
    if args.replay is not None:
      tag += (("_rawreplay-" if args.raw_replay else "_replay-") + args.replay.stem + (f"-smooth{args.smooth_replay}" if args.smooth_replay > 1 else "")
              + ("-itsplan" if args.replay_what == "plan" else "") + ("-goalachieved" if args.replay_goal == "achieved" else ""))
  if args.steer:
    tag += "_steer-" + args.steer
  if args.plan_lean != 0.0:
    tag += f"_lean{args.plan_lean:+g}"
  if args.library:
    tag += f"_lib-{args.library}"
  if args.stagger > 0.0:
    tag += f"_stagger{args.stagger:g}"
  if args.world:
    tag += f"_world-{args.world}" + (f"-gref{args.ground_reference}" if args.ground_reference else "")
  if args.tag:
    tag += "_" + args.tag
  archive = None
  if not args.no_video:
    from r1wb.eval.vector_evaluation_video import VectorVideoArchive, destination
    labels = [f"{names[i // args.repeats]}_{i % args.repeats}" for i in range(count)]
    archive = VectorVideoArchive(env, destination(run / f"policy_{iteration}.onnx", out_dir / f"{tag}.json"), labels)
    archive.manifest.update(evaluator=__file__, result_json=str(out_dir / f"{tag}.json"),
                            evaluation_args=dict({k: str(v) for k, v in vars(args).items()}, checkpoint=str(checkpoint)),
                            checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                            scripts={name: suite[name] for name in names})
    archive.begin()
    if not archive.started:
      raise RuntimeError(archive.manifest.get("reason", "Cannot preserve evaluation"))
  zeros = lambda *shape: torch.zeros(*shape, device=env.device)  # noqa: E731
  # How human the motion is: at every step the robot's pose (leg and waist joint angles, pelvis height and tilt) is
  # compared with the closest pose of the retargeted human captures (the motion library), and the same joints'
  # velocities with that frame's. Smaller = closer to something a person did. The arms are left out: the library's
  # arms are this project's own arm model, not captured motion, so they would favour policies trained on it.
  # Height is put on the joints' scale (0.05 m counts like 0.1 rad). Reported while walking and while standing.
  human = None
  if hasattr(term, "motion"):
    from r1wb.s2 import frames as s2f
    table = s2f.library_frames(term.motion, term.motion_anchor_body_index, term.clip_end[term.frame_clip])
    plan_table = table
    weight = torch.ones(27, device=env.device)
    weight[14:24] = 0.0                                        # arms: not captured motion
    weight[24] = 2.0                                           # pelvis height: 0.05 m <-> 0.1 rad
    human = dict(pose=table[:, :27] * weight, weight=weight, velocity=term.motion.joint_vel,
                 norm=(table[:, :27] * weight).pow(2).sum(1))
  totals_human = {key: zeros(count) for key in ("walk_steps", "walk_pose", "walk_velocity", "stand_steps", "stand_pose")}
  # The gait itself, from the leg joints alone (the same numbers can be taken from a human capture): how often the
  # legs swap (sign changes of left minus right hip pitch = steps), and how far hip and knee swing (standard
  # deviation of the left hip pitch and left knee angle), while a twist is commanded.
  hip_left, hip_right, knee_left = joints.index("left_hip_pitch_joint"), joints.index("right_hip_pitch_joint"), joints.index("left_knee_joint")
  gait = {key: zeros(count) for key in ("swaps", "hip", "hip_sq", "knee", "knee_sq")}
  hip_sign = zeros(count)
  alive = torch.ones(count, dtype=torch.bool, device=env.device)
  fell_at = torch.full((count,), float("nan"), device=env.device)
  smooth = None
  totals = {key: zeros(count) for key in (
    "steps", "settled", "twist_error", "yaw_error", "height_steps", "height_error", "quiet_steps", "quiet_touchdowns",
    "touchdowns", "arm_speed", "leg_speed", "quiet_arm_speed", "quiet_leg_speed", "action_rate", "outside", "outside_band",
    "one_step", "quiet_outside")}
  peaks = {key: zeros(count) for key in ("torque_ratio", "lag_xy", "lag_yaw", "margin")}
  matched = hasattr(term, "match_cost")
  match_sum = zeros(count)
  residual_sum = zeros(count, 3)                                # mean |correction|: legs, waist, arms
  anchor_sum = zeros(count, 3)                                  # mean |action - anchor action|: legs, waist, arms
  odometry, commanded = zeros(count, 3), zeros(count, 3)
  # Where the robot ended up, seen from where and how it stood at the start (the odometry above is summed in the
  # robot's own turning frame, so it cannot show a walk that curves away), next to where the command leads.
  start_pos = entity.data.root_link_pos_w[:, :2].clone()
  start_heading = yaw_quat(entity.data.root_link_quat_w).clone()
  end_pos, end_heading = start_pos.clone(), start_heading.clone()
  commanded_path, commanded_yaw = zeros(count, 2), zeros(count)
  # The course by itself: the root twist of the reference frames the matcher plays (in their own heading frame),
  # integrated like the robot's odometry. It separates "the course does not follow the command" from "the course
  # does and the robot under it does not".
  course_odometry = zeros(count, 3)
  dumped = {key: [] for key in ("plan", "root", "joints", "contact", "told", "said", "anchor")} if args.dump_plan else None
  # Stops, one by one: from the step an instruction to stand follows an instruction to move until the next
  # instruction (or the end): the landings by their time since the instruction, the time of the last one, the last
  # time the body still travelled (planar centre-of-mass speed above 0.05 m/s) and how far it went meanwhile.
  # (With --policy s2 the instruction reaches System 2 one decision, 0.32 s, before the script's time: the zero here.)
  told_moving = torch.zeros(count, dtype=torch.bool, device=env.device)
  stopping = torch.zeros(count, dtype=torch.bool, device=env.device)
  stop_now = {key: zeros(count) for key in ("clock", "landings", "last_landing", "last_travel")}
  stop_bins, stop_from, stops = zeros(count, len(STOP_EDGES) + 1), zeros(count, 2), []
  stop_edges = torch.tensor(STOP_EDGES, device=env.device)

  def close_stops(mask):
    for i in mask.nonzero(as_tuple=False).flatten().tolist():
      stops.append(dict(env=i, script=names[i // args.repeats], lasted_s=round(float(stop_now["clock"][i]), 2),
                        landings=int(stop_now["landings"][i]), by_time=[int(v) for v in stop_bins[i].tolist()],
                        last_landing_s=round(float(stop_now["last_landing"][i]), 2),
                        last_travel_s=round(float(stop_now["last_travel"][i]), 2),
                        overshoot_m=round(float(torch.norm(end_pos[i] - stop_from[i])), 3), fell=bool(not alive[i])))
  feedback = args.progress_feedback > 0.0
  if feedback:
    assert args.policy == "label" and term.script is not None, "--progress-feedback is a probe of the course generator"
  shadow = zeros(count, 3)                                      # the instruction, slewed as the stream would be
  every = torch.arange(count, device=env.device)
  peak_actuator = torch.zeros(count, dtype=torch.long, device=env.device)
  air = zeros(count, 2)
  previous_action = None
  velocity_mean = None
  started = time.time()
  end_ground = torch.zeros(count, device=env.device)      # --world: ground under the robot above the start level
  peak_ground = torch.zeros(count, device=env.device)     # ... its largest size while the robot was up
  read_gap, read_steps = torch.zeros(count, device=env.device), torch.zeros(count, device=env.device)
  slope_quiet, slope_landings = torch.zeros(count, device=env.device), torch.zeros(count, device=env.device)
  for step in range(horizon):
    if manufacturer:
      policy.twists = term.twist_goal()
    with torch.no_grad():
      if steerer is not None and steerer.generator and step % steer_policy.period == 0:
        steerer.act(steer_policy.best(steerer.observe()))
      action = policy(observation)
      if anchor is not None:
        apart = (action - anchor(torch.cat([observation["actor"], observation["goal"]], dim=1))).abs()
        anchor_sum += torch.stack([apart[:, :12].mean(1), apart[:, 12:14].mean(1), apart[:, 14:].mean(1)], dim=1) * alive.float().unsqueeze(1)
    observation, _, dones, _ = wrapped.step(action)
    if steerer is not None:
      steerer.progress.after_step()
    if archive is not None:
      archive.after_step()
      if not archive.started:
        raise RuntimeError(archive.manifest.get("reason", "State preservation failed"))
    data = entity.data
    failed = alive & dones.bool() & env.termination_manager.terminated
    fell_at = torch.where(failed, torch.full_like(fell_at, step * dt), fell_at)
    alive &= ~dones.bool()
    live = alive.float()
    twist, goal = term.own_twist(), term.twist_goal()
    if steerer is not None and steerer.generator:               # score against the instruction, not System 2's command
      goal = steerer.progress.shadow.clone()
    if feedback:                                                # score against the instruction, not the corrected command
      told = term.script[every, (term.segment_index - 1).clamp(0, term.script.shape[1] - 1)][:, 1:4]
      limit = term.slew[:3] * dt
      shadow += torch.maximum(torch.minimum(told - shadow, limit), -limit)
      goal = shadow.clone()
    smooth = twist.clone() if smooth is None else smooth + (twist - smooth) * (dt / 0.4)
    settled = alive & (term.since_change > 1.0)
    # Asked to stand for 1.5 s already, and not within 0.5 s of the next change of the script: System 2 is given an
    # instruction one decision (0.32 s) before the script's time so that the motion begins on time, and the first
    # footfall of that motion is not "moving while asked to stand".
    quiet = (alive & (term.since_change > 1.5) & (torch.norm(term.stream_target[:, :3], dim=1) < 1e-6)
             & (term.segment_left > 0.5))
    totals["steps"] += live
    totals["settled"] += settled.float()
    totals["twist_error"] += torch.norm(smooth[:, :2] - goal[:, :2], dim=1) * settled.float()
    totals["yaw_error"] += (smooth[:, 2] - goal[:, 2]).abs() * settled.float()
    height_on = alive & (term.goal_mask[:, 1] > 0) & (term.since_change > 1.5)
    height = world.root_height(env) if args.world else data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    if args.world:
      under = world.ground_under(env)
      end_ground = torch.where(alive, under, end_ground)
      peak_ground = torch.where(alive & (under.abs() > peak_ground.abs()), under, peak_ground)
      away = alive & (under.abs() > 0.1)                       # the robot stands on the shape, not on the start disc
      if not manufacturer:                                     # the height error the policy reads: plan minus robot
        read_gap += (reader.anchor_pos_w[:, 2] - term.robot_anchor_pos_w[:, 2]) * away.float()
        read_steps += away.float()
    totals["height_steps"] += height_on.float()
    totals["height_error"] += (height - term.stream[:, 3]).abs() * height_on.float()
    odometry += twist * dt * live.unsqueeze(1)
    commanded += goal * dt * live.unsqueeze(1)
    end_pos = torch.where(alive.unsqueeze(1), data.root_link_pos_w[:, :2], end_pos)
    end_heading = torch.where(alive.unsqueeze(1), yaw_quat(data.root_link_quat_w), end_heading)
    cos, sin = torch.cos(commanded_yaw), torch.sin(commanded_yaw)
    commanded_path += torch.stack([cos * goal[:, 0] - sin * goal[:, 1], sin * goal[:, 0] + cos * goal[:, 1]], dim=1) * dt * live.unsqueeze(1)
    commanded_yaw += goal[:, 2] * dt * live
    if feedback and step % 16 == 15:
      tau = args.progress_feedback
      offset = torch.cat([end_pos - start_pos, zeros(count, 1)], dim=1)
      realised = quat_apply_inverse(start_heading, offset)[:, :2]                     # seen from the start pose
      turn = quat_mul(quat_inv(start_heading), end_heading)
      turned = 2.0 * torch.atan2(turn[:, 3], turn[:, 0])
      turned = torch.atan2(torch.sin(turned), torch.cos(turned))
      yaw_gap = commanded_yaw - turned
      yaw_gap = torch.atan2(torch.sin(yaw_gap), torch.cos(yaw_gap))
      walking_told = (torch.norm(told[:, :2], dim=1) > 1e-6).float().unsqueeze(1)
      cos, sin = torch.cos(commanded_yaw), torch.sin(commanded_yaw)
      want = torch.stack([cos * told[:, 0] - sin * told[:, 1], sin * told[:, 0] + cos * told[:, 1]], dim=1)
      want = want + ((commanded_path - realised) / tau).clamp(-0.25, 0.25) * walking_told
      cos, sin = torch.cos(turned), torch.sin(turned)
      body = torch.stack([cos * want[:, 0] + sin * want[:, 1], -sin * want[:, 0] + cos * want[:, 1]], dim=1)
      rate = told[:, 2] + (yaw_gap / tau).clamp(-0.6, 0.6)
      asked = torch.cat([body, rate.unsqueeze(1)], dim=1)
      moving_told = (torch.norm(told, dim=1) > 1e-6).unsqueeze(1)
      term.stream_target[:, :3] = torch.where(moving_told, asked, torch.zeros_like(asked))
    if matched:
      frame, motion = term.time_steps, term.motion
      heading = yaw_quat(motion.body_quat_w[frame, term._root])
      course = torch.cat([quat_apply_inverse(heading, motion.body_lin_vel_w[frame, term._root])[:, :2],
                          motion.body_ang_vel_w[frame, term._root, 2:3]], dim=1)
      course_odometry += course * dt * (live * (~term.hold).float()).unsqueeze(1)
    contact = env.scene["feet_ground"].data.found.reshape(count, -1)[:, :2] > 0
    if dumped is not None:
      if args.policy == "s2":
        reference, index = view._dense()
        plan_now = reference.frames[torch.arange(count, device=env.device), reference._spot(index)]
      else:
        plan_now = plan_table[term.time_steps].clone()
        plan_now[:, 27:30] *= (~term.hold).float().unsqueeze(1)
      dumped["plan"].append(plan_now.cpu())
      dumped["root"].append(torch.cat([data.root_link_pos_w - env.scene.env_origins, data.root_link_quat_w], dim=1).cpu())
      dumped["joints"].append(data.joint_pos.cpu()); dumped["contact"].append(contact.cpu()); dumped["told"].append(goal.cpu())
      dumped["said"].append(term.told[:, :3].cpu())             # the instruction itself (told: what the score is taken against)
      # The anchor body as the plan's frames describe it (r1wb.s2.frames): where it is, its heading, height and tilt.
      spot, facing = term.robot_anchor_pos_w, term.robot_anchor_quat_w
      aw, ax, ay, az = quat_mul(quat_inv(yaw_quat(facing)), facing).unbind(-1)
      dumped["anchor"].append(torch.stack([
        spot[:, 0] - env.scene.env_origins[:, 0], spot[:, 1] - env.scene.env_origins[:, 1],
        torch.atan2(2.0 * (facing[:, 0] * facing[:, 3] + facing[:, 1] * facing[:, 2]), 1.0 - 2.0 * (facing[:, 2] ** 2 + facing[:, 3] ** 2)),
        spot[:, 2] - env.scene.env_origins[:, 2], torch.atan2(2.0 * (aw * ax + ay * az), 1.0 - 2.0 * (ax * ax + ay * ay)),
        torch.asin((2.0 * (aw * ay - az * ax)).clamp(-1.0, 1.0))], dim=1).cpu())
    landing = contact & (air >= 3)
    totals["touchdowns"] += landing.float().sum(1) * live
    totals["quiet_touchdowns"] += landing.float().sum(1) * quiet.float()
    totals["quiet_steps"] += quiet.float()
    if args.world:                                             # the same, only where the ground has left the start level
      slope_landings += landing.float().sum(1) * (quiet & away).float()
      slope_quiet += (quiet & away).float()
    air = torch.where(contact, torch.zeros_like(air), air + 1)
    if human is not None:
      anchor_quat = term.robot_anchor_quat_w
      heading_free = quat_mul(quat_inv(yaw_quat(anchor_quat)), anchor_quat)
      qw, qx, qy, qz = heading_free.unbind(-1)
      roll = torch.atan2(2.0 * (qw * qx + qy * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
      pitch = torch.asin((2.0 * (qw * qy - qz * qx)).clamp(-1.0, 1.0))
      pelvis = term.robot_anchor_pos_w[:, 2] - env.scene.env_origins[:, 2]
      pose = torch.cat([data.joint_pos, pelvis.unsqueeze(1), roll.unsqueeze(1), pitch.unsqueeze(1)], dim=1) * human["weight"]
      distance = human["norm"].unsqueeze(0) - 2.0 * pose @ human["pose"].T + pose.pow(2).sum(1, keepdim=True)
      nearest_sq, nearest = distance.min(dim=1)
      pose_gap = (nearest_sq.clamp_min(0.0) / 17.0).sqrt()                                   # rad RMS per feature
      velocity_gap = (data.joint_vel - human["velocity"][nearest])[:, :14].pow(2).mean(1).sqrt()   # rad/s RMS, legs + waist
      walking = alive & (torch.norm(term.stream_target[:, :3], dim=1) > 1e-6) & (term.since_change > 1.5)
      standing = alive & (torch.norm(term.stream_target[:, :3], dim=1) < 1e-6) & (term.since_change > 1.5)
      totals_human["walk_steps"] += walking.float(); totals_human["walk_pose"] += pose_gap * walking.float()
      totals_human["walk_velocity"] += velocity_gap * walking.float()
      totals_human["stand_steps"] += standing.float(); totals_human["stand_pose"] += pose_gap * standing.float()
      split = data.joint_pos[:, hip_left] - data.joint_pos[:, hip_right]
      sign = torch.where(split.abs() > 0.03, torch.sign(split), hip_sign)                    # 0.03 rad dead band
      gait["swaps"] += ((sign * hip_sign) < 0).float() * walking.float()
      hip_sign = sign
      for key, index in (("hip", hip_left), ("knee", knee_left)):
        gait[key] += data.joint_pos[:, index] * walking.float()
        gait[key + "_sq"] += data.joint_pos[:, index].pow(2) * walking.float()
    arm_speed = data.joint_vel[:, arm_ids].abs().mean(1)
    leg_speed = data.joint_vel[:, leg_ids].abs().mean(1)
    totals["arm_speed"] += arm_speed * live
    totals["leg_speed"] += leg_speed * live
    totals["quiet_arm_speed"] += arm_speed * quiet.float()
    totals["quiet_leg_speed"] += leg_speed * quiet.float()
    if previous_action is not None:
      totals["action_rate"] += (action - previous_action).abs().mean(1) * live
    previous_action = action
    # Reference of the balance signals: the robot's own planar CoM velocity averaged over about 1 s (not the
    # command: a policy that ignores a small command would read as diverging).
    state = centroidal.state()
    own_velocity = state["velocity"][:, :2]
    velocity_mean = own_velocity.clone() if velocity_mean is None else velocity_mean + (own_velocity - velocity_mean) * dt
    margin = centroidal.support_margin(velocity_mean, state=state)["d"]
    one_step = centroidal.capture_margin(velocity_mean, state=state)["d1"]
    totals["outside"] += (margin > 0.0).float() * live
    totals["outside_band"] += (margin > dynamics.MARGIN_DEAD_BAND).float() * live
    totals["one_step"] += ((margin > 0.0) & (one_step > dynamics.DEAD_BAND)).float() * live
    totals["quiet_outside"] += (margin > 0.0).float() * quiet.float()
    moving_now = torch.norm(term.told[:, :3], dim=1) > 1e-6
    ended, begun = stopping & (moving_now | ~alive), alive & ~moving_now & told_moving
    if ended.any():
      close_stops(ended)
    for value in stop_now.values():
      value[begun] = 0.0
    stop_bins[begun] = 0.0
    stop_from[begun] = data.root_link_pos_w[begun, :2]
    stopping, told_moving = (stopping & ~ended) | begun, moving_now
    landed = landing.float().sum(1) * stopping.float()
    stop_now["landings"] += landed
    stop_now["last_landing"] = torch.where(landed > 0, stop_now["clock"], stop_now["last_landing"])
    stop_now["last_travel"] = torch.where(stopping & (torch.norm(own_velocity, dim=1) > 0.05), stop_now["clock"], stop_now["last_travel"])
    stop_bins[every, torch.bucketize(stop_now["clock"], stop_edges)] += landed
    stop_now["clock"] += dt * stopping.float()
    peaks["margin"] = torch.where(alive, torch.maximum(peaks["margin"], margin), peaks["margin"])
    ratio, which = (data.actuator_force.abs() / effort).max(dim=1)
    peak_actuator = torch.where(alive & (ratio > peaks["torque_ratio"]), which, peak_actuator)
    peaks["torque_ratio"] = torch.where(alive, torch.maximum(peaks["torque_ratio"], ratio), peaks["torque_ratio"])
    peaks["lag_xy"] = torch.where(alive, torch.maximum(peaks["lag_xy"], torch.norm(term.goal_lag[:, :2], dim=1)), peaks["lag_xy"])
    peaks["lag_yaw"] = torch.where(alive, torch.maximum(peaks["lag_yaw"], term.goal_lag[:, 2].abs()), peaks["lag_yaw"])
    if matched:
      match_sum += term.match_cost * live
    if residual:
      correction = env.action_manager.get_term("joint_pos").residual.abs()
      residual_sum += torch.stack([correction[:, :12].mean(1), correction[:, 12:14].mean(1), correction[:, 14:].mean(1)], dim=1) * live.unsqueeze(1)
    if not alive.any():
      break

  close_stops(stopping)

  def per(key, base):
    return (totals[key] / totals[base].clamp_min(1.0)).tolist()

  twist_error, yaw_error, height_error = per("twist_error", "settled"), per("yaw_error", "settled"), per("height_error", "height_steps")
  zero = torch.zeros(count, 1, device=env.device)
  travelled = quat_apply_inverse(start_heading, torch.cat([end_pos - start_pos, zero], dim=1))[:, :2]
  forward = quat_apply_inverse(start_heading, quat_apply(end_heading, torch.tensor([1.0, 0.0, 0.0], device=env.device).repeat(count, 1)))
  turned = torch.atan2(forward[:, 1], forward[:, 0])
  rows = []
  for i in range(count):
    name = names[i // args.repeats]
    asked_xy, asked_yaw = float(torch.norm(commanded[i, :2])), float(commanded[i, 2].abs())
    quiet_minutes = float(totals["quiet_steps"][i]) * dt / 60.0
    steps = max(float(totals["steps"][i]), 1.0)
    rows.append(dict(
      script=name, repeat=i % args.repeats, fell=bool(not np.isnan(float(fell_at[i]))),
      fell_at_s=None if np.isnan(float(fell_at[i])) else round(float(fell_at[i]), 2),
      twist_error_mps=round(twist_error[i], 4), yaw_rate_error_rps=round(yaw_error[i], 4),
      distance_ratio=round(float((odometry[i, :2] * commanded[i, :2]).sum()) / asked_xy**2, 3) if asked_xy > 0.5 else None,
      yaw_ratio=round(float(odometry[i, 2] * commanded[i, 2]) / asked_yaw**2, 3) if asked_yaw > 0.5 else None,
      height_error_m=round(height_error[i], 4) if float(totals["height_steps"][i]) > 0 else None,
      touchdowns=int(totals["touchdowns"][i]), quiet_touchdowns=int(totals["quiet_touchdowns"][i]),
      quiet_touchdowns_per_min=round(float(totals["quiet_touchdowns"][i]) / quiet_minutes, 2) if quiet_minutes > 0.02 else None,
      arm_speed_rps=round(float(totals["arm_speed"][i]) / steps, 4), leg_speed_rps=round(float(totals["leg_speed"][i]) / steps, 4),
      quiet_arm_speed_rps=round(float(totals["quiet_arm_speed"][i] / totals["quiet_steps"][i].clamp_min(1.0)), 4),
      quiet_leg_speed_rps=round(float(totals["quiet_leg_speed"][i] / totals["quiet_steps"][i].clamp_min(1.0)), 4),
      action_rate=round(float(totals["action_rate"][i]) / steps, 4),
      capture_point_outside_frac=round(float(totals["outside"][i]) / steps, 4),
      capture_point_outside_band_frac=round(float(totals["outside_band"][i]) / steps, 4),
      one_step_urgent_frac=round(float(totals["one_step"][i]) / steps, 4),
      quiet_outside_frac=round(float(totals["quiet_outside"][i] / totals["quiet_steps"][i].clamp_min(1.0)), 4),
      support_margin_peak_m=round(float(peaks["margin"][i]), 3), torque_ratio_max=round(float(peaks["torque_ratio"][i]), 2),
      goal_lag_peak_m=round(float(peaks["lag_xy"][i]), 3), goal_lag_peak_rad=round(float(peaks["lag_yaw"][i]), 3)))
    if args.world:
      rows[-1].update(ground_end_m=round(float(end_ground[i]), 3), ground_peak_m=round(float(peak_ground[i]), 3),
                      plan_minus_robot_height_m=round(float(read_gap[i] / read_steps[i]), 3) if float(read_steps[i]) > 0 else None,
                      away_quiet_s=round(float(slope_quiet[i]) * dt, 2), away_quiet_landings=int(slope_landings[i]))
    # Straight commands only (no turning asked): length and direction of the path actually walked, and how far the
    # heading drifted meanwhile. Position where the command leads minus where the robot stands, at the end.
    if asked_xy > 0.5 and asked_yaw < 0.1:
      path, asked = travelled[i], commanded_path[i]
      angle = float(torch.atan2(path[0] * asked[1] - path[1] * asked[0], (path * asked).sum()))
      rows[-1].update(path_ratio=round(float(torch.norm(path) / torch.norm(asked)), 3),
                      direction_error_deg=round(abs(math.degrees(angle)), 1) if float(torch.norm(path)) > 0.2 else None,
                      heading_drift_deg=round(abs(math.degrees(float(turned[i]))), 1),
                      end_miss_m=round(float(torch.norm(path - asked)), 3))
    else:
      rows[-1].update(path_ratio=None, direction_error_deg=None, heading_drift_deg=None, end_miss_m=None)
    if asked_xy < 0.05 and asked_yaw > 0.5:
      rows[-1]["turn_drift_m"] = round(float(torch.norm(travelled[i])), 3)           # turning in place: distance wandered
    else:
      rows[-1]["turn_drift_m"] = None
    if anchor is not None:
      rows[-1].update(anchor_gap_legs=round(float(anchor_sum[i, 0]) / steps, 4), anchor_gap_waist=round(float(anchor_sum[i, 1]) / steps, 4),
                      anchor_gap_arms=round(float(anchor_sum[i, 2]) / steps, 4))
    if residual:
      rows[-1].update(correction_legs=round(float(residual_sum[i, 0]) / steps, 4),
                      correction_waist=round(float(residual_sum[i, 1]) / steps, 4),
                      correction_arms=round(float(residual_sum[i, 2]) / steps, 4))
    rows[-1]["torque_peak_actuator"] = actuator_names[int(peak_actuator[i])]
    mine = [stop for stop in stops if stop["env"] == i]
    for key in ("landings", "last_landing_s", "last_travel_s", "overshoot_m"):
      rows[-1]["stop_" + key] = round(float(np.mean([stop[key] for stop in mine])), 3) if mine else None
    rows[-1]["stop_late_landings"] = round(float(np.mean([sum(stop["by_time"][3:]) for stop in mine])), 3) if mine else None
    walked, stood = float(totals_human["walk_steps"][i]), float(totals_human["stand_steps"][i])
    rows[-1].update(
      human_pose_gap_walking_rad=round(float(totals_human["walk_pose"][i]) / walked, 4) if walked > 25 else None,
      human_velocity_gap_walking_rps=round(float(totals_human["walk_velocity"][i]) / walked, 4) if walked > 25 else None,
      human_pose_gap_standing_rad=round(float(totals_human["stand_pose"][i]) / stood, 4) if stood > 25 else None,
      steps_per_s_walking=round(float(gait["swaps"][i]) / (walked * dt), 3) if walked > 25 else None,
      hip_swing_rad=round(float((gait["hip_sq"][i] / walked - (gait["hip"][i] / walked) ** 2).clamp_min(0.0).sqrt()), 4) if walked > 25 else None,
      knee_swing_rad=round(float((gait["knee_sq"][i] / walked - (gait["knee"][i] / walked) ** 2).clamp_min(0.0).sqrt()), 4) if walked > 25 else None)
    if matched:
      rows[-1].update(
        course_distance_ratio=round(float((course_odometry[i, :2] * commanded[i, :2]).sum()) / asked_xy**2, 3) if asked_xy > 0.5 else None,
        course_yaw_ratio=round(float(course_odometry[i, 2] * commanded[i, 2]) / asked_yaw**2, 3) if asked_yaw > 0.5 else None)
      rows[-1].update(match_cost=round(float(match_sum[i]) / steps, 3),
                      match_switches_per_s=round(float(term.match_switches[i]) / (steps * dt), 3) if rows[-1]["fell"] is False else None)

  def mean(values):
    values = [v for v in values if v is not None]
    return round(float(np.mean(values)), 4) if values else None

  by_script = {}
  for name in names:
    group = [r for r in rows if r["script"] == name]
    by_script[name] = {key: mean([r[key] for r in group]) for key in rows[0]
                       if key not in ("script", "repeat", "fell", "fell_at_s", "torque_peak_actuator")}
    by_script[name]["falls"] = sum(r["fell"] for r in group)
    worst = max(group, key=lambda r: r["torque_ratio_max"])
    by_script[name].update(torque_ratio_peak=worst["torque_ratio_max"], torque_peak_actuator=worst["torque_peak_actuator"])
  scored = [r for r in rows if not (manufacturer and r["script"] in USES_HEIGHT)]
  quiet_minutes = float(totals["quiet_steps"].sum()) * dt / 60.0
  summary = dict(
    episodes=count, falls=sum(r["fell"] for r in rows), fell_scripts=sorted({r["script"] for r in rows if r["fell"]}),
    twist_error_mps=mean([r["twist_error_mps"] for r in scored]), yaw_rate_error_rps=mean([r["yaw_rate_error_rps"] for r in scored]),
    distance_ratio=mean([r["distance_ratio"] for r in scored]), yaw_ratio=mean([r["yaw_ratio"] for r in scored]),
    distance_ratio_min=min([r["distance_ratio"] for r in scored if r["distance_ratio"] is not None], default=None),
    path_ratio=mean([r["path_ratio"] for r in scored]), direction_error_deg=mean([r["direction_error_deg"] for r in scored]),
    heading_drift_deg=mean([r["heading_drift_deg"] for r in scored]), end_miss_m=mean([r["end_miss_m"] for r in scored]),
    height_error_m=None if manufacturer else mean([r["height_error_m"] for r in rows]),
    quiet_touchdowns=int(totals["quiet_touchdowns"].sum()), quiet_minutes=round(quiet_minutes, 2),
    quiet_touchdowns_per_min=round(float(totals["quiet_touchdowns"].sum()) / max(quiet_minutes, 1e-6), 2),
    quiet_arm_speed_rps=mean([r["quiet_arm_speed_rps"] for r in rows]), quiet_leg_speed_rps=mean([r["quiet_leg_speed_rps"] for r in rows]),
    arm_speed_rps=mean([r["arm_speed_rps"] for r in rows]), action_rate=mean([r["action_rate"] for r in rows]),
    capture_point_outside_frac=mean([r["capture_point_outside_frac"] for r in rows]),
    capture_point_outside_band_frac=mean([r["capture_point_outside_band_frac"] for r in rows]),
    one_step_urgent_frac=mean([r["one_step_urgent_frac"] for r in rows]),
    torque_ratio_max=max(r["torque_ratio_max"] for r in rows),
    human_pose_gap_walking_rad=mean([r["human_pose_gap_walking_rad"] for r in rows]),
    human_velocity_gap_walking_rps=mean([r["human_velocity_gap_walking_rps"] for r in rows]),
    human_pose_gap_standing_rad=mean([r["human_pose_gap_standing_rad"] for r in rows]))
  if stops:                                                   # every stop of every script (a fall ends its stop early)
    last, travel = sorted(stop["last_landing_s"] for stop in stops), sorted(stop["last_travel_s"] for stop in stops)
    p95 = lambda values: values[int(0.95 * (len(values) - 1))]  # noqa: E731
    summary.update(
      stops=len(stops), stop_landings_per_stop=round(float(np.mean([stop["landings"] for stop in stops])), 3),
      stop_landings_by_time={label: int(sum(stop["by_time"][k] for stop in stops)) for k, label in enumerate(STOP_LABELS)},
      stops_with_landing_after_1p5s=sum(1 for stop in stops if sum(stop["by_time"][3:]) > 0),
      stops_with_landing_after_3s=sum(1 for stop in stops if stop["by_time"][5] > 0),
      stop_last_landing_s=dict(mean=round(float(np.mean(last)), 2), p95=p95(last), max=last[-1]),
      stop_last_travel_s=dict(mean=round(float(np.mean(travel)), 2), p95=p95(travel), max=travel[-1]),
      stop_overshoot_m=round(float(np.mean([stop["overshoot_m"] for stop in stops])), 3))
  if anchor is not None:
    summary.update(anchor_gap_legs=mean([r["anchor_gap_legs"] for r in rows]), anchor_gap_waist=mean([r["anchor_gap_waist"] for r in rows]),
                   anchor_gap_arms=mean([r["anchor_gap_arms"] for r in rows]))
  if residual:
    summary.update(correction_legs=mean([r["correction_legs"] for r in rows]), correction_waist=mean([r["correction_waist"] for r in rows]),
                   correction_arms=mean([r["correction_arms"] for r in rows]))
  if args.policy == "s2":                                     # how long System 2 takes to decide, in this loop
    spent = sorted(view.decision_seconds[2:])                  # the first calls include start-up work
    summary.update(s2_decision_ms_median=round(1000.0 * spent[len(spent) // 2], 1) if spent else None,
                   s2_decision_ms_p95=round(1000.0 * spent[int(0.95 * (len(spent) - 1))], 1) if spent else None,
                   s2_decision_envs=count,
                   gpu_peak_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None)
  result = dict(protocol="r1wb-stream-eval-v1", definition=__doc__, policy=args.policy, run=run.name, iteration=iteration,
                checkpoint=str(checkpoint), variant=args.variant, step_dt=dt, seconds=args.seconds, repeats=args.repeats,
                note="manufacturer: no height input; height scripts are run but not scored" if manufacturer else None,
                elapsed_s=round(time.time() - started, 1), summary=summary, by_script=by_script, episodes=rows, stops=stops)
  if archive is not None:
    result["video"] = archive.close()
  (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1, default=str))
  if dumped is not None:
    args.dump_plan.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dump_plan, **{key: torch.stack(value).numpy() for key, value in dumped.items()},
                        scripts=np.array([names[i // args.repeats] for i in range(count)]), joint_names=np.array(joints),
                        dt=np.float32(dt), model=str(archive.manifest.get("model_path", "")) if archive is not None else "",
                        what=tag)
    print("plan and robot saved to", args.dump_plan)
  print(json.dumps(summary, indent=1))
  print(f"{'script':18s} {'falls':>5s} {'twist':>6s} {'yaw':>6s} {'dist':>6s} {'yawr':>6s} {'height':>6s} {'quiet/min':>9s} {'q.arm':>6s} {'q.leg':>6s} {'out':>6s} {'1step':>6s} {'path':>6s} {'dir°':>6s} {'head°':>6s} {'miss m':>6s}")
  fmt = lambda v: "   -  " if v is None else f"{v:6.3f}"  # noqa: E731
  deg = lambda v: "   -  " if v is None else f"{v:6.1f}"  # noqa: E731
  for name, row in by_script.items():
    print(f"{name:18s} {row['falls']:5d} {fmt(row['twist_error_mps'])} {fmt(row['yaw_rate_error_rps'])} {fmt(row['distance_ratio'])} "
          f"{fmt(row['yaw_ratio'])} {fmt(row['height_error_m'])} {fmt(row['quiet_touchdowns_per_min']):>9s} {fmt(row['quiet_arm_speed_rps'])} "
          f"{fmt(row['quiet_leg_speed_rps'])} {fmt(row['capture_point_outside_frac'])} {fmt(row['one_step_urgent_frac'])} "
          f"{fmt(row['path_ratio'])} {deg(row['direction_error_deg'])} {deg(row['heading_drift_deg'])} {fmt(row['end_miss_m'])}")
  if args.world:
    # Only the runs that left the start level count: the others walked on the flat disc.
    print(f"\nground {args.world}{f' (plan raised by the ground under the {args.ground_reference})' if args.ground_reference else ''}: runs that "
          f"got more than 0.1 m above or below the start; falls and when (s); largest ground height reached (m); height "
          f"the policy read (plan - robot, m); landings while told to stand out there (per minute)")
    reached, fallen_out, quiet_s, quiet_n = 0, 0, 0.0, 0
    for name in names:
      group = [r for r in rows if r["script"] == name and abs(r["ground_peak_m"]) > 0.1]
      if not group:
        continue
      fell = [r["fell_at_s"] for r in group if r["fell"]]
      peak = [r["ground_peak_m"] for r in group]
      read = [r["plan_minus_robot_height_m"] for r in group if r["plan_minus_robot_height_m"] is not None]
      seconds, landings = sum(r["away_quiet_s"] for r in group), sum(r["away_quiet_landings"] for r in group)
      reached, fallen_out, quiet_s, quiet_n = reached + len(group), fallen_out + len(fell), quiet_s + seconds, quiet_n + landings
      print(f"  {name:18s} {len(fell):2d} of {len(group):2d}  {' '.join(f'{v:5.1f}' for v in fell):24s} ground {min(peak):+5.2f}..{max(peak):+5.2f}"
            f"  read {'' if not read else f'{sum(read) / len(read):+5.2f}':>5s}  standing {seconds:5.1f} s"
            + (f", {60.0 * landings / seconds:5.1f} /min" if seconds > 1.0 else ""))
    print(f"  on the shape: {fallen_out} falls of {reached} runs; told to stand there for {quiet_s:.0f} s"
          + (f", {60.0 * quiet_n / quiet_s:.1f} landings per minute" if quiet_s > 1.0 else ""))
  if stops:
    print("\nstops, one by one: landings by their time since the instruction to stand (s), all stops of a script together")
    print(f"{'stops of':18s} {'stops':>5s} {'land/stop':>9s} " + " ".join(f"{label:>6s}" for label in STOP_LABELS)
          + f" {'last s':>7s} {'max':>5s} {'travel s':>8s} {'max':>5s} {'went m':>7s}")
    groups = {name: [stop for stop in stops if stop["script"] == name] for name in names}
    groups["all"] = stops
    for name, group in groups.items():
      if group:
        print(f"{name:18s} {len(group):5d} {np.mean([g['landings'] for g in group]):9.2f} "
              + " ".join(f"{sum(g['by_time'][k] for g in group):6d}" for k in range(len(STOP_LABELS)))
              + f" {np.mean([g['last_landing_s'] for g in group]):7.2f} {max(g['last_landing_s'] for g in group):5.2f}"
              f" {np.mean([g['last_travel_s'] for g in group]):8.2f} {max(g['last_travel_s'] for g in group):5.2f}"
              f" {np.mean([g['overshoot_m'] for g in group]):7.3f}")
  env.close()


if __name__ == "__main__":
  main()
