"""Supervised start of the mission model (r1wb.s2.vla.MissionVLA): camera + mission + map so far -> next instruction.

  python -m r1wb.s2.train_mission --backbone <dir> --data <recordings *.npz ...> --held-out <recordings ...> --out <dir>
  two GPUs: torchrun --nproc_per_node 2 -m r1wb.s2.train_mission ... --shard          (all weights of a large backbone)

A recording is what many robots saw and were told while an expert led them through one house (one file per house:
maps (robots, decisions, 4, 64, 64) uint8, vector, action, alive, and next to it <file>.images.npy with the head
camera (robots, decisions, H, W, 3) uint8; a file with `label` was led by someone else, and `label` is what the
expert would have told there: that is what is learned). The
expert is a source of demonstrations only (a policy trained in the 2D proxy, or the wavefront walker on the true
map); the model that is deployed is this one, and it is trained further in the loop afterwards.
An example = one decision of one robot; the mission is worded as a random phrasing. Loss: cross-entropy on the
expert's instruction. Printed at the end, on the held-out houses: the share of decisions where the model's likeliest
instruction is the expert's, the mean probability it gives the expert's choice, and the same per instruction.
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
import torch.nn.functional as F

from r1wb.s2.train_vla import save

MISSIONS = {
  "map": {
    "ko": ("집 안 모든 공간을 돌아다니며 지도를 만들어", "모든 방을 둘러보고 지도를 완성해", "집 전체를 빠짐없이 살펴봐",
           "아직 안 본 곳을 찾아다니면서 집 구조를 파악해", "방마다 들어가서 전부 확인해"),
    "en": ("Walk through the whole house and map every room", "Explore every room and complete the map",
           "Look over the entire house without missing a space", "Find the places you have not seen yet and learn the layout",
           "Go into each room and check all of it"),
  },
}
HELD_OUT = {
  "map": {"ko": ("집을 구석구석 탐색해서 전체 지도를 그려", "빠진 방 없이 다 돌아봐"),
          "en": ("Survey the house room by room until the map is whole", "Cover every space in this home")},
}


def phrase(mission: str, language: str, rng: random.Random, held_out: bool = False) -> str:
  return rng.choice((HELD_OUT if held_out else MISSIONS)[mission][language])


class Recordings:
  """Decisions of several houses, read lazily (the images stay on disk until a batch asks for them)."""

  def __init__(self, files: list[Path]):
    self.parts, self.rows = [], []
    for index, file in enumerate(files):
      data = np.load(file, allow_pickle=False)
      part = {key: data[key] for key in ("maps", "vector", "action", "alive")}
      if "label" in data.files:                 # what the expert would have told (a run led by the model itself)
        part["action"] = data["label"]
      part["image"] = np.load(str(file) + ".images.npy", mmap_mode="r")          # (robots, decisions, H, W, 3), kept on disk
      self.parts.append(part)
      robots, decisions = part["action"].shape[:2]
      alive = np.asarray(part["alive"]).astype(bool)
      for robot, decision in zip(*np.nonzero(alive)):
        self.rows.append((index, int(robot), int(decision)))
      self.names = [str(n) for n in data["names"]]
    self.rows = np.array(self.rows)

  def __len__(self) -> int:
    return len(self.rows)

  def take(self, picks):
    images, maps, vectors, actions = [], [], [], []
    for index, robot, decision in self.rows[picks]:
      part = self.parts[index]
      images.append(np.asarray(part["image"][robot, decision]))
      maps.append(np.asarray(part["maps"][robot, decision]))
      vectors.append(np.asarray(part["vector"][robot, decision]))
      actions.append(int(part["action"][robot, decision]))
    return images, np.stack(maps).astype(np.float32) / 255.0, np.stack(vectors).astype(np.float32), np.array(actions)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--backbone", type=Path, required=True)
  parser.add_argument("--data", type=Path, nargs="+", required=True)
  parser.add_argument("--held-out", type=Path, nargs="*", default=[])
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=1500)
  parser.add_argument("--batch", type=int, default=16, help="per GPU")
  parser.add_argument("--lr", type=float, default=1.0e-5)
  parser.add_argument("--head-lr", type=float, default=2.0e-4)
  parser.add_argument("--languages", default="ko,en")
  parser.add_argument("--image-size", type=int, default=224)
  parser.add_argument("--shard", action="store_true")
  parser.add_argument("--freeze-backbone", action="store_true")
  parser.add_argument("--unfrozen", type=int, default=0, help="with --freeze-backbone: train the last layers of the language model")
  parser.add_argument("--log-every", type=int, default=25)
  parser.add_argument("--start-from", type=Path, default=None, help="a goal or mission model's folder to continue from")
  parser.add_argument("--no-memory", action="store_true",
                      help="CONTROL: the map tokens are zeroed (what the camera and the mission alone can tell)")
  args = parser.parse_args()
  rank = int(os.environ.get("LOCAL_RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  if world > 1:
    torch.distributed.init_process_group("nccl")
  device = f"cuda:{rank}"
  torch.cuda.set_device(device)
  torch.manual_seed(3000 + rank)
  rng = random.Random(3000 + rank)

  def say(*words):
    if rank == 0:
      print(*words, flush=True)

  train, held = Recordings(args.data), Recordings(args.held_out) if args.held_out else None
  languages = args.languages.split(",")
  width = train.take(np.array([0]))[2].shape[1]
  counts = np.bincount([int(train.parts[i]["action"][r, d]) for i, r, d in train.rows[:: max(1, len(train) // 5000)]], minlength=len(train.names))
  say(f"decisions: {len(train)} from {len(args.data)} houses" + (f", held out {len(held)} from {len(args.held_out)}" if held else "")
      + f"; expert's instructions (sample): " + ", ".join(f"{n} {c}" for n, c in zip(train.names, counts)))

  from r1wb.s2.vla import MissionVLA
  full = not args.freeze_backbone
  model = MissionVLA(args.backbone, choices=len(train.names), vector=width,
                     dtype=torch.float32 if full else torch.bfloat16, image_size=args.image_size).to(device)
  if args.start_from is not None:
    file = args.start_from / ("mission_last.pt" if (args.start_from / "mission_last.pt").exists() else "goal_last.pt")
    weights = torch.load(file, map_location="cpu", weights_only=False)
    missing = model.load_state_dict({key: value.to(next(model.parameters()).dtype) if value.is_floating_point() else value
                                     for key, value in weights["state"].items()}, strict=False)
    say(f"started from {file.name}: {len(missing.missing_keys)} new tensors")
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
      f"{(sum(p.numel() for p in backbone) + sum(p.numel() for p in new)) / 1e9:.3f} B trained; memory after load "
      f"{torch.cuda.memory_allocated() / 2**30:.1f} GB")

  def batch_of(source: Recordings, picks, held_out: bool = False):
    images, maps, vectors, actions = source.take(picks)
    texts = [phrase("map", rng.choice(languages), rng, held_out) for _ in images]
    packed = model.pack(images, texts).to(device)
    maps, vectors = torch.as_tensor(maps, device=device), torch.as_tensor(vectors, device=device)
    if args.no_memory:
      maps, vectors = torch.zeros_like(maps), torch.zeros_like(vectors)
    return packed, maps, vectors, torch.as_tensor(actions, device=device)

  def score(source: Recordings, examples: int = 512):
    model.eval()
    picks = np.random.default_rng(0).choice(len(source), size=min(examples, len(source)), replace=False)
    right, given, per = 0, 0.0, np.zeros((len(source.names), 3))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
      for start in range(0, len(picks), 32):
        packed, maps, vectors, actions = batch_of(source, picks[start:start + 32], held_out=True)
        chance = F.softmax(model(packed, maps, vectors)[0].float(), dim=1)
        top = chance.argmax(dim=1)
        right += int((top == actions).sum())
        given += float(chance.gather(1, actions[:, None]).sum())
        for a, t, p in zip(actions.tolist(), top.tolist(), chance.gather(1, actions[:, None])[:, 0].tolist()):
          per[a] += (1, a == t, p)
    model.train()
    return right / len(picks), given / len(picks), per

  args.out.mkdir(parents=True, exist_ok=True)
  model.train()
  torch.cuda.reset_peak_memory_stats()
  began = last = time.time()
  for step in range(args.steps):
    packed, maps, vectors, actions = batch_of(train, np.array([rng.randrange(len(train)) for _ in range(args.batch)]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
      loss = F.cross_entropy(model(packed, maps, vectors)[0].float(), actions)
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(backbone + new, 1.0)
    optimiser.step()
    schedule.step()
    if step % args.log_every == 0 or step == args.steps - 1:
      torch.cuda.synchronize()
      now = time.time()
      say(f"step {step:5d} loss {float(loss):.4f}  {(now - last) / (args.log_every if step else 1):.2f} s/step  "
          f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB on GPU {rank}  {now - began:.0f} s")
      last = now
  report = {}
  for name, source in (("train", train), ("held_out", held)):
    if source is None or rank != 0:
      continue
    agree, given, per = score(source)
    report[name] = dict(agree=round(agree, 3), probability=round(given, 3),
                        per_instruction={n: dict(count=int(c), agree=round(float(a / max(c, 1)), 2), probability=round(float(p / max(c, 1)), 2))
                                         for n, (c, a, p) in zip(source.names, per)})
    say(f"{name} houses: likeliest instruction = the expert's in {agree:.3f} of the decisions, mean probability of the "
        f"expert's choice {given:.3f} (uniform {1 / len(source.names):.3f}); per instruction "
        + ", ".join(f"{n} {int(c)}:{a / max(c, 1):.2f}" for n, (c, a, p) in zip(source.names, per) if c))
  meta = dict(kind="mission", backbone=str(args.backbone.name), data=[f.name for f in args.data], held_out=[f.name for f in args.held_out],
              steps=args.steps, batch=args.batch * world, ranks=world, sharded=args.shard, full=full, languages=languages,
              image_size=args.image_size, names=train.names, vector=int(width), no_memory=args.no_memory, scores=report,
              unfrozen_layers=args.unfrozen, seconds=round(time.time() - began),
              peak_memory_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
  save(model, args.out / "mission_last.pt", rank, meta)
  say(json.dumps({k: v for k, v in meta.items() if k != "scores"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
  main()
