"""Roll the reference generator over instruction scripts and keep what System 2 has to learn from.

Each environment follows a random script of instructions (r1wb.s2.instructions). The command stream's planner turns
the commands into a course of library frames (the plan: it follows the command and its own frame, never the robot),
and a System 1 policy drives the robot along it, so the course is recorded as it is played in closed loop.

  python -m r1wb.s2.collect --run <System 1 run dir> --checkpoint model_N.pt --out <file.npz> [--envs 2048 --seconds 30]

Saved: frame (steps, envs) int32 library frame of the course, hold (steps, envs) bool, stream (steps, envs, 5) the
slewed command, segment (steps, envs) int16 index into the environment's script, start (steps, envs) bool where an
episode (re)starts, scripts (json: per environment a list of [instruction, seconds]), library (path and sha256).
--camera N: the robot carries a head camera (N x N pixels, looking 25 degrees down from the torso); what it sees at
every System 2 decision (control steps 1 + 16 m) goes to <out>.images.npy, shape (decisions, envs, N, N, 3) uint8.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
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

from r1wb.s2 import instructions  # noqa: E402
from r1wb.student.stream import stream_env_cfg  # noqa: E402


def head_camera(size: int):
  """The camera System 2 looks through: on the torso at head height, facing forward, pitched 25 degrees down."""
  import math

  from mjlab.sensor.camera_sensor import CameraSensorCfg
  tilt = math.radians(25.0)
  w1, x1, y1, z1 = 0.5, 0.5, -0.5, -0.5                       # camera x = body right, y = up, looking along body +x
  w2, x2 = math.cos(-0.5 * tilt), math.sin(-0.5 * tilt)       # then pitched down about its own x axis
  quat = (w1 * w2 - x1 * x2, w1 * x2 + x1 * w2, y1 * w2 + z1 * x2, z1 * w2 - y1 * x2)
  return CameraSensorCfg(name="head_camera", parent_body="robot/torso_link", pos=(0.12, 0.0, 0.42), quat=quat,
                         fovy=70.0, width=size, height=size, data_types=("rgb",))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--checkpoint", default=None)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--envs", type=int, default=2048)
  parser.add_argument("--seconds", type=float, default=30.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--camera", type=int, default=0, help="image size; 0 = no camera")
  parser.add_argument("--match-height", choices=["stream", "target"], default="target",
                      help="the height a look-up asks for while the command moves (student.stream match_height)")
  parser.add_argument("--match-stop", choices=["room", "arrive"], default="arrive",
                      help="the frames a look-up may choose for \"stand\" (student.stream match_stop)")
  parser.add_argument("--segment-time", type=float, nargs=2, default=[2.56, 6.08],
                      help="an instruction lasts this long (s); short ones give more changes of instruction per hour")
  parser.add_argument("--course-set", nargs="*", default=[],
                      help="other settings of the course generator as name=value (student.stream MatchedStreamCommandCfg)")
  parser.add_argument("--match-noise", type=float, default=0.0,
                      help="random spread of a look-up's costs (student.stream match_noise; 0: the same state and "
                           "command always give the same course, which is what a model imitating it needs)")
  parser.add_argument("--library", default="",
                      help="the motion library the generator plays (library_v2_<name>.npz), e.g. corebal = the rest "
                           "poses given the largest margin their stance allows (r1wb.balance)")
  parser.add_argument("--steer", default="",
                      help="the generator's command is steered every 0.32 s as System 2 steers it (r1wb.s2.steer). "
                           "'random': the instruction seen from the body's heading plus a slowly varying random bounded "
                           "correction in six environments of ten, the instruction seen from the body's heading in two, "
                           "the instruction as it is in two (the whole of System 2's action space, for the code head to "
                           "learn from); <name>: the learned System 2 steer_<name>.pt, with a small random addition in "
                           "half of the environments. The command in force is saved as `target` and `counts`.")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  run = args.run.resolve()
  checkpoint = run / args.checkpoint if args.checkpoint else max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
  cfg = stream_env_cfg(play=True, clean=True, matched=True, teacher=str(checkpoint))
  cfg.commands["motion"].match_pose, cfg.commands["motion"].match_commit = "reference", True
  cfg.commands["motion"].match_height = args.match_height
  cfg.commands["motion"].match_stop = args.match_stop
  cfg.commands["motion"].match_noise = args.match_noise
  if args.library:
    from r1wb.tracking.env_cfg import library_path
    cfg.commands["motion"].motion_file = str(library_path(args.library))
  for item in args.course_set:
    key, value = item.split("=")
    setattr(cfg.commands["motion"], key, type(getattr(cfg.commands["motion"], key))(float(value)))
  stand = cfg.commands["motion"].clip_list
  cfg.commands["motion"].clip_list = tuple(stand[i % len(stand)] for i in range(args.envs))
  cfg.scene.num_envs = args.envs
  cfg.seed = args.seed
  if args.camera:
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (head_camera(args.camera),)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env)
  term = env.command_manager.get_term("motion")
  rng = random.Random(args.seed)
  scripts = [instructions.script(rng, args.seconds, tuple(args.segment_time)) for _ in range(args.envs)]
  longest = max(len(s) for s in scripts)
  table = torch.zeros(args.envs, longest, 8)
  for i, script in enumerate(scripts):
    # Half a control step is taken off every duration so that a segment ends on its last step whatever the rounding.
    rows = [instructions.command_row(item, seconds - 0.5 * env.step_dt) for item, seconds in script]
    rows += [[1.0e6] + rows[-1][1:]] * (longest - len(rows))
    table[i] = torch.tensor(rows)
    table[i, len(script) - 1, 0] = 1.0e6                       # the last instruction holds
  term.set_script(table)
  env.reset()
  observation = wrapped.get_observations()
  steps = int(args.seconds / env.step_dt)
  frame = np.zeros((steps, args.envs), np.int32)
  hold = np.zeros((steps, args.envs), bool)
  stream = np.zeros((steps, args.envs, 5), np.float32)
  segment = np.zeros((steps, args.envs), np.int16)
  start = np.zeros((steps, args.envs), bool)
  target = np.zeros((steps, args.envs, 5), np.float32)
  counts = np.zeros((steps, args.envs), np.float32)
  steerer = steer_policy = None
  if args.steer:
    from r1wb.s2.steer import PERIOD, Steer, SteerPolicy
    if args.steer != "random":
      steer_policy = SteerPolicy.load(args.out.parent / f"steer_{args.steer}.pt", args.device)
    steerer = Steer(env, term, steer_policy.see if steer_policy else "progress", leak=steer_policy.leak if steer_policy else 0.0, remember=steer_policy.remember if steer_policy else False, turned=steer_policy.turned if steer_policy else False)
    kind = torch.rand(args.envs, device=args.device)
    wander = torch.zeros(args.envs, 3, device=args.device)
  falls, began = 0, time.time()
  images = None
  if args.camera:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images = np.lib.format.open_memmap(args.out.with_suffix(".images.npy"), mode="w+", dtype=np.uint8,
                                       shape=((steps - 1) // 16 + 1, args.envs, args.camera, args.camera, 3))
  for step in range(steps):
    if images is not None and step >= 1 and (step - 1) % 16 == 0:
      images[(step - 1) // 16] = env.scene["head_camera"].data.rgb.cpu().numpy()
    frame[step], hold[step] = term.time_steps.cpu().numpy(), term.hold.cpu().numpy()
    stream[step] = term.stream.cpu().numpy()
    target[step], counts[step] = term.stream_target.cpu().numpy(), term.goal_mask[:, 1].cpu().numpy()
    segment[step] = (term.segment_index - 1).clamp(min=0).cpu().numpy()
    start[step] = (env.episode_length_buf <= 1).cpu().numpy()
    with torch.no_grad():
      if steerer is not None and step % (steer_policy.period if steer_policy else PERIOD) == 0:
        wander = 0.8 * wander + 0.42 * torch.randn_like(wander)                 # stationary spread 0.7
        told_moving = (torch.norm(steerer.told()[:, :3], dim=1, keepdim=True) > 1.0e-6).float()
        if steer_policy is None:      # moves when told to (one decision in twenty the other way round)
          flip = (torch.rand(args.envs, 1, device=args.device) < 0.05).float()
          action = torch.cat([told_moving + flip - 2.0 * told_moving * flip,
                              torch.where((kind < 0.6).unsqueeze(1), wander, torch.zeros_like(wander))], dim=1)
        else:
          action = steer_policy.best(steerer.observe())
          action[:, 1:] += torch.where((kind < 0.5).unsqueeze(1), 0.3 * wander, torch.zeros_like(wander))
        steerer.act(action)
        if steer_policy is None:                                                # two in ten: the instruction as it is
          plain = kind >= 0.8
          term.stream_target[plain, :3] = term.told[plain, :3]
      observation, _, dones, _ = wrapped.step(observation["bc"][:, :24])
      if steerer is not None:
        steerer.progress.after_step()
    falls += int((dones.bool() & env.termination_manager.terminated).sum())
  library = Path(cfg.commands["motion"].motion_file)
  args.out.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(
    args.out, frame=frame, hold=hold, stream=stream, segment=segment, start=start, step_dt=np.float32(env.step_dt),
    target=target, counts=counts,
    scripts=json.dumps([[[item, seconds] for item, seconds in script] for script in scripts]),
    meta=json.dumps(dict(protocol="r1wb-s2-courses-v0", library=str(library.name),
                         library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), system1=str(checkpoint),
                         envs=args.envs, seconds=args.seconds, seed=args.seed, falls=falls, camera=args.camera,
                         steer=args.steer,
                         course=f"plan (match_pose=reference, match_commit, match_height={args.match_height}, "
                                f"match_stop={args.match_stop}, match_noise={args.match_noise:g}"
                                + "".join(", " + item for item in args.course_set) + ")")))
  print(f"saved {args.out}: {steps} steps x {args.envs} environments = {steps * args.envs * env.step_dt / 3600:.1f} h of "
        f"course, {falls} falls, held {hold.mean():.3f} of the time, {time.time() - began:.0f} s")
  env.close()


if __name__ == "__main__":
  main()
