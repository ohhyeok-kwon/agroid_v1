"""Supervised training of the vision-language goal model (r1wb.s2.vla.GoalVLA): camera image + instruction -> goal.

  one GPU       python -m r1wb.s2.train_goal --backbone <dir> --courses <courses.npz with .images.npy> --out <dir>
  two GPUs      torchrun --nproc_per_node 2 -m r1wb.s2.train_goal ... --shard     (all weights; fp32 masters, gradients
                and optimiser state sharded over the GPUs, forward in bf16: the 8B model needs both GPUs for these)

An example is one System 2 decision of a recorded course: the head-camera image at that moment and the instruction
in force one decision later, worded as a random training phrasing (Korean or English) -> that instruction's goal.
At the end the model is scored on phrasings it has never seen (r1wb.s2.instructions held-out sets): a goal counts as
right if twist, height and the height flag all match the instruction's.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch

from r1wb.s2 import instructions
from r1wb.s2.small import decisions, goal_of
from r1wb.s2.train_vla import save

TOLERANCE = torch.tensor([0.07, 0.07, 0.15, 0.04, 0.5])


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--backbone", type=Path, required=True)
  parser.add_argument("--courses", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=1500)
  parser.add_argument("--batch", type=int, default=16, help="per GPU")
  parser.add_argument("--lr", type=float, default=1.0e-5)
  parser.add_argument("--head-lr", type=float, default=2.0e-4)
  parser.add_argument("--languages", default="ko,en")
  parser.add_argument("--image-size", type=int, default=224)
  parser.add_argument("--shard", action="store_true")
  parser.add_argument("--freeze-backbone", action="store_true")
  parser.add_argument("--unfrozen", type=int, default=0)
  parser.add_argument("--log-every", type=int, default=25)
  parser.add_argument("--extra-phrases", type=Path, default=None, help="more training phrasings (r1wb.s2.paraphrase)")
  parser.add_argument("--look", type=Path, default=None,
                      help="recorded \"turn to face the <colour> box\" decisions (r1wb.s2.look collect): half of every "
                           "batch then comes from them (the picture decides the goal there)")
  parser.add_argument("--start-from", type=Path, default=None, help="a goal model's folder to continue from")
  args = parser.parse_args()
  rank = int(os.environ.get("LOCAL_RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  if world > 1:
    torch.distributed.init_process_group("nccl")
  device = f"cuda:{rank}"
  torch.cuda.set_device(device)
  torch.manual_seed(2000 + rank)
  rng = random.Random(2000 + rank)

  def say(*words):
    if rank == 0:
      print(*words, flush=True)

  courses = dict(np.load(args.courses, allow_pickle=False))
  scripts = json.loads(str(courses["scripts"]))
  images = np.load(args.courses.with_suffix(".images.npy"), mmap_mode="r")
  steps_recorded, envs = courses["frame"].shape
  codes_shape = torch.zeros(envs, (steps_recorded - 1) // 8, 1)
  env_index, code_index, seen = decisions(courses, codes_shape)
  usable = (code_index // 2) < images.shape[0]
  env_index, code_index, seen = env_index[usable], code_index[usable], seen[usable]
  meanings = instructions.every_instruction()
  names = [instructions.identifier(item) for item in meanings]
  languages = args.languages.split(",")
  added = instructions.load_extra(args.extra_phrases) if args.extra_phrases else 0
  wording = [[text for lang in languages for text in instructions.training_phrases(item, lang)] for item in meanings]
  meaning = torch.tensor([names.index(instructions.identifier(scripts[e][s][0])) for e, s in zip(env_index.tolist(), seen.tolist())])
  goals = torch.tensor([goal_of(item) for item in meanings], device=device)
  # Meanings are drawn evenly (the courses hold "stand" and "walk" far more often than "crouch deep").
  rows_of = [(meaning == m).nonzero().flatten() for m in range(len(names))]
  say(f"decisions with an image: {len(meaning)}; {len(names)} meanings, {sum(len(w) for w in wording)} training phrasings; ranks {world}")

  from r1wb.s2.vla import GoalVLA
  full = not args.freeze_backbone
  model = GoalVLA(args.backbone, dtype=torch.float32 if full else torch.bfloat16, image_size=args.image_size).to(device)
  if args.start_from is not None:                              # before sharding: the whole weights are still on this rank
    weights = torch.load(args.start_from / "goal_last.pt", map_location="cpu", weights_only=False)
    model.load_state_dict({key: value.to(next(model.parameters()).dtype) if value.is_floating_point() else value
                           for key, value in weights["state"].items()}, strict=False)
    del weights
  if not full:
    for p in model.lm.parameters():
      p.requires_grad_(False)
    layers = model.lm.model.language_model.layers
    for layer in layers[len(layers) - args.unfrozen:] if args.unfrozen else []:
      for p in layer.parameters():
        p.requires_grad_(True)
  if args.shard:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for block in list(model.lm.model.visual.blocks) + list(model.lm.model.language_model.layers):
      fully_shard(block, mp_policy=policy)
    fully_shard(model, mp_policy=policy)
  backbone = [p for n, p in model.named_parameters() if n.startswith("lm.") and p.requires_grad]
  new = [p for n, p in model.named_parameters() if not n.startswith("lm.")]
  optimiser = torch.optim.AdamW([{"params": backbone, "lr": args.lr}, {"params": new, "lr": args.head_lr}],
                                weight_decay=0.0, betas=(0.9, 0.95))
  warm = max(1, min(100, args.steps // 10))
  schedule = torch.optim.lr_scheduler.LambdaLR(
    optimiser, lambda step: min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * min(1.0, step / max(1, args.steps))))))
  say(f"model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B parameters, "
      f"{(sum(p.numel() for p in backbone) + sum(p.numel() for p in new)) / 1e9:.2f} B trained, "
      f"{'sharded over ' + str(world) + ' GPUs' if args.shard else 'one GPU'}; memory after load {torch.cuda.memory_allocated() / 2**30:.1f} GB")

  def picture(m: int):
    row = int(rows_of[m][rng.randrange(len(rows_of[m]))]) if len(rows_of[m]) else rng.randrange(len(meaning))
    return np.asarray(images[int(code_index[row]) // 2, int(env_index[row])])

  look = None
  if args.look is not None:                                    # the camera task: (picture, named colour) -> the chosen goal
    from r1wb.s2 import look as look_task
    recorded = np.load(args.look, allow_pickle=False)
    look = dict(images=np.load(str(args.look) + ".images.npy", mmap_mode="r"), target=recorded["target"], choice=recorded["choice"],
                goals=torch.tensor(recorded["goals"], device=device), colours=json.loads(str(recorded["meta"]))["colours"])
    # Choices are drawn evenly: most recorded decisions are "stand" (the box is already faced).
    look["rows_of"] = [np.argwhere(look["choice"] == c) for c in range(len(look["goals"]))]
    say(f"camera task: {look['choice'].size} decisions, per choice {[len(r) for r in look['rows_of']]}")

  def batch_of(count: int):
    if look is None:
      what = [rng.randrange(len(names)) for _ in range(count)]
      packed = model.pack([picture(m) for m in what], [rng.choice(wording[m]) for m in what]).to(device)
      return packed, goals[torch.tensor(what, device=device)]
    what = [rng.randrange(len(names)) for _ in range(count // 2)]
    pictures, texts, wanted = [picture(m) for m in what], [rng.choice(wording[m]) for m in what], [goals[m] for m in what]
    for _ in range(count - len(what)):
      rows = look["rows_of"][rng.randrange(len(look["rows_of"]))]
      step, env = rows[rng.randrange(len(rows))] if len(rows) else (rng.randrange(look["choice"].shape[0]), rng.randrange(look["choice"].shape[1]))
      pictures.append(np.asarray(look["images"][step, env]))
      texts.append(look_task.phrase(look["colours"][int(look["target"][step, env])], rng.choice(languages), rng))
      wanted.append(look["goals"][int(look["choice"][step, env])])
    return model.pack(pictures, texts).to(device), torch.stack(wanted)

  def score(held_out: bool, language: str):
    """Share of phrasings whose stated goal matches the instruction's (every phrasing of every meaning once)."""
    right = total = 0
    worst = []
    model.eval()
    for m, item in enumerate(meanings):
      texts = instructions.all_phrases(item, language, held_out)
      with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        stated = model.decide(model.pack([picture(m) for _ in texts], texts).to(device)).float()
      ok = ((stated - goals[m]).abs() <= TOLERANCE.to(device)).all(dim=1)
      right += int(ok.sum()); total += len(texts)
      worst += [(texts[i], names[m], [round(v, 2) for v in stated[i].tolist()]) for i in (~ok).nonzero().flatten().tolist()]
    model.train()
    return right, total, worst

  args.out.mkdir(parents=True, exist_ok=True)
  model.train()
  torch.cuda.reset_peak_memory_stats()
  began = last = time.time()
  for step in range(args.steps):
    packed, goal = batch_of(args.batch)
    with torch.autocast("cuda", dtype=torch.bfloat16):
      loss = model.loss(packed, goal)
    optimiser.zero_grad(set_to_none=True)
    loss.float().backward()
    torch.nn.utils.clip_grad_norm_(backbone + new, 1.0)
    optimiser.step()
    schedule.step()
    if step % args.log_every == 0 or step == args.steps - 1:
      torch.cuda.synchronize()
      now = time.time()
      say(f"step {step:5d} loss {float(loss):.5f}  {(now - last) / (args.log_every if step else 1):.2f} s/step  "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB on GPU {rank}  {now - began:.0f} s")
      last = now
  report = {}
  for language in languages:
    for held_out in (False, True):
      right, total, worst = score(held_out, language)
      report[f"{language}_{'heldout' if held_out else 'train'}"] = dict(right=right, total=total, misses=worst[:12])
      say(f"{language} {'held-out' if held_out else 'training'} phrasings: {right}/{total} goals right; misses: {worst[:5]}")
  meta = dict(kind="goal", backbone=str(args.backbone.name), courses=args.courses.name, steps=args.steps,
              batch=args.batch * world, ranks=world, sharded=args.shard, full=full, languages=languages,
              image_size=args.image_size, phrasing_scores=report, seconds=round(time.time() - began),
              extra_phrasings=added, unfrozen_layers=args.unfrozen,
              peak_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
  save(model, args.out / "goal_last.pt", rank, meta)
  say(json.dumps({k: v for k, v in meta.items() if k != "phrasing_scores"}, indent=1))


if __name__ == "__main__":
  main()
