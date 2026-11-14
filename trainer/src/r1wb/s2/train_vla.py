"""Supervised training of the action model (r1wb.s2.vla.MotionVLA) on recorded courses with camera images.

  one GPU       python -m r1wb.s2.train_vla --backbone <dir> --courses <courses.npz> --library <library.npz>
                    --tokenizer <tokenizer.pt> --out <dir> [--steps N --batch B]
  two GPUs      torchrun --nproc_per_node 2 -m r1wb.s2.train_vla ... --shard
                All weights are trained. Their fp32 masters, gradients and optimiser state are sharded over the GPUs
                (FSDP), the forward pass runs in bf16: the 8B model needs about 140 GB for these and so both GPUs.
  --freeze-backbone   train only the slots, the heads and the last `--unfrozen` language layers (small GPUs, tests)

Each example is one System 2 decision of a recorded course: the head-camera image at that moment, the instruction
in force one decision later (a training phrasing, Korean or English), the last HISTORY codes -> the next AHEAD codes
and the goal. Decisions and timing as in r1wb.s2.small.
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
from r1wb.s2.small import AHEAD, GRID_ORIGIN, HISTORY, decisions, goal_of


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--backbone", type=Path, required=True)
  parser.add_argument("--courses", type=Path, required=True)
  parser.add_argument("--library", type=Path, required=True)
  parser.add_argument("--tokenizer", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=2000)
  parser.add_argument("--batch", type=int, default=16, help="per GPU")
  parser.add_argument("--lr", type=float, default=2.0e-5)
  parser.add_argument("--head-lr", type=float, default=2.0e-4)
  parser.add_argument("--languages", default="ko,en")
  parser.add_argument("--image-size", type=int, default=224)
  parser.add_argument("--shard", action="store_true")
  parser.add_argument("--freeze-backbone", action="store_true")
  parser.add_argument("--unfrozen", type=int, default=0)
  parser.add_argument("--resume", type=Path, default=None)
  parser.add_argument("--save-every", type=int, default=0)
  parser.add_argument("--log-every", type=int, default=20)
  args = parser.parse_args()
  rank = int(os.environ.get("LOCAL_RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  if world > 1:
    torch.distributed.init_process_group("nccl")
  device = f"cuda:{rank}"
  torch.cuda.set_device(device)
  torch.manual_seed(1000 + rank)
  rng = random.Random(1000 + rank)

  def say(*words):
    if rank == 0:
      print(*words, flush=True)

  # -- data -------------------------------------------------------------------------------------------------------
  from r1wb.s2.tokenizer import MotionTokenizer, course_frames, library_table
  tokenizer = MotionTokenizer.load(args.tokenizer, device)
  courses = dict(np.load(args.courses, allow_pickle=False))
  scripts = json.loads(str(courses["scripts"]))
  frames = course_frames(courses, library_table(args.library, device))
  length = ((frames.shape[1] - GRID_ORIGIN) // tokenizer.stride) * tokenizer.stride
  with torch.no_grad():
    codes = torch.cat([tokenizer.encode(part[:, GRID_ORIGIN:GRID_ORIGIN + length]) for part in frames.split(256)])
  codes = codes + tokenizer.levels // 2
  del frames
  env_index, code_index, seen = decisions(courses, codes.cpu())
  image_file = args.courses.with_suffix(".images.npy")
  images = np.load(image_file, mmap_mode="r") if image_file.exists() else None
  if images is not None:                                   # image m was taken at the decision of code 2m
    usable = (code_index // 2) < images.shape[0]
    env_index, code_index, seen = env_index[usable], code_index[usable], seen[usable]
  meanings = instructions.every_instruction()
  names = [instructions.identifier(item) for item in meanings]
  languages = args.languages.split(",")
  wording = [[text for lang in languages for text in instructions.all_phrases(item, lang)] for item in meanings]
  meaning = torch.tensor([names.index(instructions.identifier(scripts[e][s][0])) for e, s in zip(env_index.tolist(), seen.tolist())])
  goals = torch.tensor([goal_of(item) for item in meanings], device=device)
  held = env_index >= int(codes.shape[0] * 0.95)
  train_rows, valid_rows = (~held).nonzero().flatten(), held.nonzero().flatten()
  offsets = torch.arange(-HISTORY + 1, AHEAD + 1, device=device)
  grey = np.full((args.image_size, args.image_size, 3), 127, np.uint8)
  say(f"decisions: {len(train_rows)} for training, {len(valid_rows)} held out; images: "
      f"{None if images is None else images.shape}; {len(names)} meanings; ranks {world}")

  # -- model ------------------------------------------------------------------------------------------------------
  from r1wb.s2.vla import MotionVLA
  full = not args.freeze_backbone
  model = MotionVLA(args.backbone, tokenizer.dims, tokenizer.levels, dtype=torch.float32 if full else torch.bfloat16,
                    image_size=args.image_size)
  if args.resume is not None:
    state = torch.load(args.resume, map_location="cpu", weights_only=False)["state"]
    missing = model.load_state_dict({k: v.to(next(model.parameters()).dtype) if v.is_floating_point() else v
                                     for k, v in state.items()}, strict=False)
    say("resumed", args.resume, "missing", len(missing.missing_keys), "unexpected", len(missing.unexpected_keys))
  model.to(device)
  new = [p for n, p in model.named_parameters() if not n.startswith("lm.")]
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
  warm = max(1, min(200, args.steps // 10))
  schedule = torch.optim.lr_scheduler.LambdaLR(
    optimiser, lambda step: min(1.0, (step + 1) / warm) * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * min(1.0, step / max(1, args.steps))))))
  trainable = sum(p.numel() for p in backbone) + sum(p.numel() for p in new)
  say(f"model: {sum(p.numel() for p in model.parameters()) / 1e9:.2f} B parameters, {trainable / 1e9:.2f} B trained, "
      f"{'sharded over ' + str(world) + ' GPUs' if args.shard else 'one GPU'}; memory after load {torch.cuda.memory_allocated() / 2**30:.1f} GB")

  def batch_of(rows):
    env, code, what = env_index[rows], code_index[rows], meaning[rows]
    window = codes[env.to(device).unsqueeze(1), code.to(device).unsqueeze(1) + offsets]
    pictures = [grey if images is None else np.asarray(images[int(c) // 2, int(e)]) for e, c in zip(env.tolist(), code.tolist())]
    texts = [rng.choice(wording[w]) for w in what.tolist()]
    packed = model.pack(pictures, texts).to(device)
    return packed, window[:, :HISTORY], window[:, HISTORY:], goals[what.to(device)]

  def check(rows):
    model.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
      packed, history, target, goal = batch_of(rows)
      logits, stated, _ = model(packed, history)
    model.train()
    return float((logits.argmax(-1) == target).float().mean()), float((stated.float() - goal).abs().mean(0).max())

  # -- training ---------------------------------------------------------------------------------------------------
  args.out.mkdir(parents=True, exist_ok=True)
  model.train()
  torch.cuda.reset_peak_memory_stats()
  began = last = time.time()
  for step in range(args.steps):
    packed, history, target, goal = batch_of(train_rows[torch.randint(len(train_rows), (args.batch,))])
    with torch.autocast("cuda", dtype=torch.bfloat16):
      loss_codes, loss_goal, logits = model.loss(packed, history, target, goal)
    optimiser.zero_grad(set_to_none=True)
    (loss_codes + loss_goal).float().backward()
    torch.nn.utils.clip_grad_norm_(backbone + new, 1.0)
    optimiser.step()
    schedule.step()
    if step % args.log_every == 0 or step == args.steps - 1:
      torch.cuda.synchronize()
      now = time.time()
      accuracy = float((logits.argmax(-1) == target).float().mean())
      say(f"step {step:5d} codes {float(loss_codes):.4f} (accuracy {accuracy:.3f}) goal {float(loss_goal):.5f}  "
          f"{(now - last) / (args.log_every if step else 1):.2f} s/step  peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
          f"on GPU {rank}  {now - began:.0f} s")
      last = now
    if args.save_every and step and step % args.save_every == 0:
      save(model, args.out / f"vla_{step}.pt", rank, dict(step=step))
  accuracy, gap = check(valid_rows[torch.randint(len(valid_rows), (min(64, args.batch * 2),))])
  meta = dict(backbone=str(args.backbone.name), courses=args.courses.name, tokenizer=args.tokenizer.name, steps=args.steps,
              batch=args.batch * world, ranks=world, sharded=args.shard, full=full, languages=languages,
              image_size=args.image_size, valid_code_accuracy=round(accuracy, 4), valid_goal_gap_max=round(gap, 4),
              seconds=round(time.time() - began), peak_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
  save(model, args.out / "vla_last.pt", rank, meta)
  say(json.dumps(meta, indent=1))


def save(model, path: Path, rank: int, meta: dict) -> None:
  """All ranks gather the weights (a collective call under sharding); rank 0 writes them in bf16. Weights that were
  not trained are left out (they are the backbone's own, loaded again from its folder), so a model whose backbone
  stayed frozen is a small file."""
  trained = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
  state = {}
  for name, value in model.state_dict().items():
    if name not in trained and name.startswith("lm."):
      continue
    value = value.full_tensor() if hasattr(value, "full_tensor") else value
    if rank == 0:
      state[name] = value.detach().to(torch.bfloat16).cpu() if value.is_floating_point() else value.detach().cpu()
  if rank == 0:
    torch.save(dict(state=state, meta=dict(meta, weights_saved=len(state))), path)


if __name__ == "__main__":
  main()
