"""Does the steering reward prefer the behaviour we want? (lesson of 11-11: check before training on a reward)

Runs System 2's training scene (r1wb.s2.steer.SteerEnv) with fixed, hand-made actions and prints the mean reward per
decision and its ingredients. The proportional variant is a diagnostic only (never trained on, never deployed).
  as told        moves when told to move, stands when told to stand, no correction
  +feedback T    ... and correction = path gap / T, heading gap / T (the evaluation-only probe of gates 11-13 08:29)
  always moving  the feedback also while told to stand
  slow = stand   +feedback 1.5, but stands when told to move slower than 0.25 m/s and 0.7 rad/s (a dead zone)
  start policy   what the untrained policy does (moves with probability 0.82, random corrections)
  +noise S       as told, random corrections of spread S (what exploration costs)
usage: python -m r1wb.eval.steer_reward_check <System 1 checkpoint> <task> [--envs 512 --seconds 30 --period 16 --calm]
"""
import argparse
from pathlib import Path

import torch

from r1wb.s2.steer import SteerEnv

parser = argparse.ArgumentParser()
parser.add_argument("system1", type=Path)
parser.add_argument("task")
parser.add_argument("--envs", type=int, default=256)
parser.add_argument("--seconds", type=float, default=30.0)
parser.add_argument("--period", type=int, default=16)
parser.add_argument("--leak", type=float, default=0.0)
parser.add_argument("--forgive", type=float, default=0.0)
parser.add_argument("--calm", action="store_true")
parser.add_argument("--library", default="", help="the motion library of the scene (library_v2_<name>.npz), e.g. corebal")
args = parser.parse_args()
env = SteerEnv(args.task, args.system1, args.envs, "cuda:0", 0, "progress", args.calm, args.period, leak=args.leak, forgive=args.forgive, remember=True, library=args.library)
decisions = int(args.seconds / (args.period * 0.02))
warm = int(6.0 / (args.period * 0.02))
scale = env.steer.scale


def told_moving(seen):
  return (seen[:, :3].abs().sum(dim=1, keepdim=True) > 1.0e-6).float()


def feedback(seen, tau):
  gap, heading = seen[:, 10:12] * 0.3, seen[:, 12] * 0.5
  wanted = torch.cat([(gap / tau).clamp(-0.25, 0.25), (heading / tau).clamp(-0.6, 0.6).unsqueeze(1)], dim=1)
  return torch.atanh((wanted / scale).clamp(-0.999, 0.999))


def patient(seen, tau):
  """The feedback, but a gap that jumped since the last decision (a hit) is not answered: the smaller of the two."""
  now = torch.cat([seen[:, 10:12] * 0.3, seen[:, 12:13] * 0.5], dim=1)
  before = torch.cat([seen[:, 13:15] * 0.3, seen[:, 15:16] * 0.5], dim=1)
  slow = torch.where(now.abs() < before.abs(), now, before) * (torch.sign(now) == torch.sign(before)).float()
  wanted = torch.cat([(slow[:, :2] / tau).clamp(-0.25, 0.25), (slow[:, 2:3] / tau).clamp(-0.6, 0.6)], dim=1)
  return torch.atanh((wanted / scale).clamp(-0.999, 0.999))


zero = lambda: torch.zeros(args.envs, 3, device="cuda:0")  # noqa: E731
always = lambda: torch.ones(args.envs, 1, device="cuda:0")  # noqa: E731
variants = {"as told": lambda seen: torch.cat([told_moving(seen), zero()], dim=1),
            "+feedback 1.5": lambda seen: torch.cat([told_moving(seen), feedback(seen, 1.5)], dim=1),
            "+feedback 3.0": lambda seen: torch.cat([told_moving(seen), feedback(seen, 3.0)], dim=1),
            "+patient 1.5": lambda seen: torch.cat([told_moving(seen), patient(seen, 1.5)], dim=1),
            "always moving": lambda seen: torch.cat([always(), feedback(seen, 1.5)], dim=1),
            "slow = stand": lambda seen: torch.cat([((seen[:, :2] * 0.35).norm(dim=1, keepdim=True) > 0.25).float()
                                                    + ((seen[:, 2:3]).abs() > 0.7).float() > 0.5, feedback(seen, 1.5)], dim=1).float(),
            "start policy": lambda seen: torch.cat([(torch.rand(args.envs, 1, device="cuda:0") < 0.82).float(),
                                                    0.3 * torch.randn(args.envs, 3, device="cuda:0")], dim=1),
            "+noise 0.3": lambda seen: torch.cat([told_moving(seen), 0.3 * torch.randn(args.envs, 3, device="cuda:0")], dim=1)}
print(f"period {args.period} steps, progress forgotten after {args.leak} s, not owed for {args.forgive} s after a hit\n{'variant':14s} reward/decision  path gap m  heading deg  stand motion  |corr|   falls/100 env-s | follow  still  course  size  fall")
for name, act in variants.items():
  torch.manual_seed(0)
  seen, _ = env.reset()
  total = gap = heading = walked = stood = fell = size = 0.0
  parts = {}
  for index in range(decisions):
    (seen, _), reward, ended, info = env.step(act(seen))
    if index < warm:                    # let the episodes spread out first
      continue
    for key, value in env.parts.items():
      parts[key] = parts.get(key, 0.0) + float(value.mean())
    total += float(reward.mean()); gap += float(info["gap"].sum()); heading += float(info["heading"].sum())
    walked += float(info["walked"].sum()); stood += float(info["stood"].sum()); fell += float(info["fell"].sum())
    size += float(info["correction"].abs().mean())
  n = decisions - warm
  seconds = n * args.envs * args.period * 0.02
  print(f"{name:14s} {total / n:+.3f}          {gap / max(walked, 1):.3f}       {57.3 * heading / max(walked, 1):5.1f}        "
        f"{stood / (n * args.envs * args.period):.4f}     {size / n:.3f}    {100 * fell / seconds:.3f}           | "
        + "  ".join(f"{parts[key] / n:+.3f}" for key in ("follow", "still", "course", "size", "fall")), flush=True)
