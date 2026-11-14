"""DAgger for the motion-code head: it writes, the reference generator answers.

Trained only on recorded courses, the head has never seen what its own small mistakes lead to (for instance a held
crouch it keeps repeating while the instruction says "rise"; in the recordings a rise always starts on time). Here
the head drives the robot through System 1, and at every decision the simulator's reference generator says which
codes should come next: put on the library frame closest to what the head is playing, it keeps that clip while it
fits the command and looks up a new one when it does not (r1wb.s2.env.S2Env.expert_codes). Those answers, with the
head's own history, are added to the training set and the head is trained further on both.
(--expert robot: the generator runs by itself matched to the robot's state instead. Tried first, 11-13: its answers
do not continue what the head is playing, so the reference became less steady.)

  python -m r1wb.s2.dagger --system1 <checkpoint> --model <small_*.pt> --tokenizer <tokenizer.pt>
      --courses <courses.npz> --library <library.npz> --out <small_*.pt> [--rounds 2 --envs 1024]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--system1", type=Path, required=True)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--tokenizer", type=Path, required=True)
  parser.add_argument("--courses", type=Path, required=True)
  parser.add_argument("--library", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--rounds", type=int, default=2)
  parser.add_argument("--envs", type=int, default=1024)
  parser.add_argument("--decisions", type=int, default=124)
  parser.add_argument("--steps", type=int, default=1500, help="training steps per round")
  parser.add_argument("--batch", type=int, default=512)
  parser.add_argument("--expert", choices=["reference", "robot"], default="reference",
                      help="where the generator answers from: the reference System 2 is playing, or the robot's state")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  torch.manual_seed(0)
  from r1wb.s2 import instructions
  from r1wb.s2.env import NAMES, S2Env
  from r1wb.s2.small import AHEAD, GRID_ORIGIN, HISTORY, SmallS2, decisions, goal_of
  from r1wb.s2.tokenizer import MotionTokenizer, course_frames, library_table
  device = args.device
  tokenizer = MotionTokenizer.load(args.tokenizer, device)
  model = SmallS2.load(args.model, device)
  assert model.condition == "goal", "DAgger is written for the goal-conditioned head"
  scale = model.goal_scale
  # The recorded courses (as in r1wb.s2.small).
  courses = dict(np.load(args.courses, allow_pickle=False))
  scripts = json.loads(str(courses["scripts"]))
  frames = course_frames(courses, library_table(args.library, device))
  length = ((frames.shape[1] - GRID_ORIGIN) // tokenizer.stride) * tokenizer.stride
  with torch.no_grad():
    codes = torch.cat([tokenizer.encode(part[:, GRID_ORIGIN:GRID_ORIGIN + length]) for part in frames.split(256)])
  codes = codes + tokenizer.levels // 2
  del frames
  env_index, code_index, seen = decisions(courses, codes.cpu())
  meanings = instructions.every_instruction()
  goals = torch.tensor([goal_of(item) for item in meanings], device=device)
  meaning = torch.tensor([NAMES.index(instructions.identifier(scripts[e][s][0])) for e, s in zip(env_index.tolist(), seen.tolist())],
                         device=device)
  env_index, code_index = env_index.to(device), code_index.to(device)
  offsets = torch.arange(-HISTORY + 1, AHEAD + 1, device=device)

  def recorded(count: int):
    rows = torch.randint(len(meaning), (count,), device=device)
    window = codes[env_index[rows].unsqueeze(1), code_index[rows].unsqueeze(1) + offsets]
    return goals[meaning[rows]] / scale, window[:, :HISTORY], window[:, HISTORY:]

  env = S2Env(args.system1, tokenizer, args.envs, device, expert=args.expert == "robot")
  bank = {"goal": [], "history": [], "target": []}
  optimiser = torch.optim.AdamW(model.parameters(), lr=1.0e-4, weight_decay=0.01)
  log = []
  for round_index in range(args.rounds):
    # 1. the head drives; the generator answers
    model.eval()
    obs = env.reset()
    agree, kept, falls, reward_sum = 0.0, 0, 0, 0.0
    began = time.time()
    for _ in range(args.decisions):
      goal = env.goals[obs["meaning"]]
      written, _ = model.decide(goal / scale, obs["history"])
      answer = env.expert_codes()
      steady = env.steady()
      bank["goal"].append((goal / scale)[steady]); bank["history"].append(obs["history"][steady]); bank["target"].append(answer[steady])
      agree += float((written[steady] == answer[steady]).float().mean()) * int(steady.sum()); kept += int(steady.sum())
      obs, reward, fell, _, _ = env.step(written, goal)
      falls += int(fell.sum()); reward_sum += float(reward.mean())
    own = {key: torch.cat(value) for key, value in bank.items()}
    row = dict(round=round_index, decisions=kept, agreement_with_generator=round(agree / max(kept, 1), 4), falls=falls,
               reward_per_decision=round(reward_sum / args.decisions, 4), collect_s=round(time.time() - began))
    # 2. train on recorded courses and on the answers, half and half
    model.train()
    for step in range(args.steps):
      rows = torch.randint(len(own["goal"]), (args.batch // 2,), device=device)
      told, history, target = recorded(args.batch - len(rows))
      told = torch.cat([told, own["goal"][rows]]); history = torch.cat([history, own["history"][rows]])
      target = torch.cat([target, own["target"][rows]])
      logits, _ = model(told, history, target)
      loss = F.cross_entropy(logits.reshape(-1, model.levels), target.reshape(-1))
      optimiser.zero_grad(set_to_none=True)
      loss.backward()
      torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
      optimiser.step()
    row.update(loss=round(float(loss), 4), answers_total=len(own["goal"]), seconds=round(time.time() - began))
    log.append(row)
    print(json.dumps(row), flush=True)
  model.eval()
  model.save(args.out, **dict(model.meta, dagger=dict(source=args.model.name, system1=str(args.system1), rounds=log,
                                                       envs=args.envs, decisions=args.decisions)))
  print("saved", args.out)


if __name__ == "__main__":
  main()
