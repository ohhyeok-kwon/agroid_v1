"""Training the motion-code head on histories it wrote itself, against the recorded courses.

Trained on recorded histories only, the head has never read its own output: after a small mistake its history is one
no recording contains, and it can stay there (it repeated a held crouch while the instruction said "rise"). Here a
recorded course is entered at a random decision; from there the head writes on by itself for a few decisions (each
decision's first two codes become history, as at run time), under the course's own instructions; then it is asked for
the next codes and corrected with what the recorded course really did. The answer is always the recorded course, so
the head is taught to come back to it from wherever its own writing took it. No simulator is involved.

  python -m r1wb.s2.selfplay --model <small_*.pt (goal-conditioned)> --courses <courses.npz> --library <library.npz>
      --tokenizer <tokenizer.pt> --out <small_*.pt> [--steps 3000 --ahead-decisions 8]
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
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--courses", type=Path, required=True)
  parser.add_argument("--library", type=Path, required=True)
  parser.add_argument("--tokenizer", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=3000)
  parser.add_argument("--batch", type=int, default=512)
  parser.add_argument("--ahead-decisions", type=int, default=8, help="the head writes by itself for 0 .. this many decisions")
  parser.add_argument("--lr", type=float, default=1.0e-4)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  torch.manual_seed(0)
  from r1wb.s2 import instructions
  from r1wb.s2.small import AHEAD, DECISION_CODES, GRID_ORIGIN, HISTORY, SmallS2, goal_of
  from r1wb.s2.tokenizer import MotionTokenizer, course_frames, library_table
  device = args.device
  tokenizer = MotionTokenizer.load(args.tokenizer, device)
  model = SmallS2.load(args.model, device)
  assert model.condition == "goal"
  scale = model.goal_scale
  courses = dict(np.load(args.courses, allow_pickle=False))
  scripts = json.loads(str(courses["scripts"]))
  frames = course_frames(courses, library_table(args.library, device))
  length = ((frames.shape[1] - GRID_ORIGIN) // tokenizer.stride) * tokenizer.stride
  with torch.no_grad():
    codes = torch.cat([tokenizer.encode(part[:, GRID_ORIGIN:GRID_ORIGIN + length]) for part in frames.split(256)])
  codes = codes + tokenizer.levels // 2                                            # (envs, n, dims)
  del frames
  whole = ~torch.as_tensor(courses["start"][2:].any(axis=0))
  envs = whole.nonzero().flatten().to(device)
  # The goal read at every code position: that of the instruction in force one decision (DECISION_CODES codes) later.
  meanings = instructions.every_instruction()
  names = [instructions.identifier(item) for item in meanings]
  goals = torch.tensor([goal_of(item) for item in meanings], device=device)
  segment = torch.as_tensor(courses["segment"].T.astype(np.int64))[:, GRID_ORIGIN::tokenizer.stride][:, :codes.shape[1]]
  table = torch.tensor([[names.index(instructions.identifier(item)) for item, _ in script] + [0] * (64 - len(script))
                        for script in scripts])
  meaning_at = torch.gather(table, 1, segment.clamp(max=63)).to(device)             # (envs, n)
  count = codes.shape[1]
  reach = args.ahead_decisions * DECISION_CODES
  optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
  began = time.time()
  for step in range(args.steps):
    env = envs[torch.randint(len(envs), (args.batch,), device=device)]
    start = torch.randint(HISTORY - 1, count - AHEAD - DECISION_CODES - reach - 1, (args.batch,), device=device)
    start = start - start % DECISION_CODES + (HISTORY - 1) % DECISION_CODES          # on the decision grid
    alone = torch.randint(0, args.ahead_decisions + 1, (args.batch,), device=device)  # decisions written without help
    history = codes[env.unsqueeze(1), start.unsqueeze(1) + torch.arange(-HISTORY + 1, 1, device=device)]
    model.eval()
    with torch.no_grad():
      for k in range(args.ahead_decisions):
        at = start + k * DECISION_CODES
        told = goals[meaning_at[env, (at + DECISION_CODES).clamp(max=count - 1)]] / scale
        written, _ = model.decide(told, history)
        moved = torch.cat([history[:, DECISION_CODES:], written[:, :DECISION_CODES]], dim=1)
        history = torch.where((k < alone).reshape(-1, 1, 1), moved, history)
    model.train()
    at = start + alone * DECISION_CODES
    told = goals[meaning_at[env, (at + DECISION_CODES).clamp(max=count - 1)]] / scale
    target = codes[env.unsqueeze(1), at.unsqueeze(1) + torch.arange(1, AHEAD + 1, device=device)]
    logits, _ = model(told, history, target)
    loss = F.cross_entropy(logits.reshape(-1, model.levels), target.reshape(-1))
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()
    if step % 250 == 0 or step == args.steps - 1:
      with torch.no_grad():
        accuracy = float((logits.argmax(-1) == target).float().mean())
        late = alone >= args.ahead_decisions // 2
        accuracy_late = float((logits.argmax(-1) == target)[late].float().mean()) if late.any() else float("nan")
      print(f"step {step:5d} loss {float(loss):.4f} accuracy {accuracy:.3f} (after {args.ahead_decisions // 2}+ decisions alone: "
            f"{accuracy_late:.3f})  {time.time() - began:.0f} s", flush=True)
  model.eval()
  model.save(args.out, **dict(model.meta, selfplay=dict(source=args.model.name, steps=args.steps,
                                                         ahead_decisions=args.ahead_decisions, loss=round(float(loss), 4))))
  print("saved", args.out)


if __name__ == "__main__":
  main()
