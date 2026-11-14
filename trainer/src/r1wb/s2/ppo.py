"""Reinforcement learning of System 2 with System 1 in the loop (the small model; the large one goes through RLinf
on the same environment).

The policy is the motion-code writer. One environment step is one decision (r1wb.s2.env): the reward is how the body,
driven by System 1 along the decoded codes, met the instruction. Proximal policy optimisation with one probability
ratio per written number (a decision writes 80 numbers: the ratio of their joint probability leaves any clipping
range after the smallest change, so each number is clipped by itself and shares the decision's advantage), with a
penalty on the divergence from the supervised model (the codes must stay motions the library contains: System 1 was
trained to follow those).

  python -m r1wb.s2.ppo --system1 <checkpoint> --model <small_*.pt> --tokenizer <tokenizer.pt> --text-cache <file>
      --out <small_*.pt> [--envs 512 --iterations 60]
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--system1", type=Path, required=True)
  parser.add_argument("--model", type=Path, required=True)
  parser.add_argument("--tokenizer", type=Path, required=True)
  parser.add_argument("--text-cache", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--envs", type=int, default=512)
  parser.add_argument("--iterations", type=int, default=60)
  parser.add_argument("--horizon", type=int, default=16, help="decisions per rollout")
  parser.add_argument("--lr", type=float, default=2.0e-5)
  parser.add_argument("--gamma", type=float, default=0.97)
  parser.add_argument("--lam", type=float, default=0.9)
  parser.add_argument("--clip", type=float, default=0.2)
  parser.add_argument("--anchor", type=float, default=0.02, help="weight of the divergence from the supervised model, per code number")
  parser.add_argument("--temperature", type=float, default=1.0)
  parser.add_argument("--value-warmup", type=int, default=5)
  parser.add_argument("--languages", default="ko,en")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  torch.manual_seed(0)
  from r1wb.s2 import instructions
  from r1wb.s2.env import MEANINGS, NAMES, S2Env
  from r1wb.s2.small import SmallS2
  from r1wb.s2.text import TextEncoder
  from r1wb.s2.tokenizer import MotionTokenizer
  device = args.device
  tokenizer = MotionTokenizer.load(args.tokenizer, device)
  model = SmallS2.load(args.model, device)
  reference = copy.deepcopy(model).eval()
  for p in reference.parameters():
    p.requires_grad_(False)
  encoder = TextEncoder(args.text_cache)
  languages = args.languages.split(",")
  bank = [encoder([text for lang in languages for text in instructions.all_phrases(item, lang)]).to(device) for item in MEANINGS]

  def words(meaning: torch.Tensor) -> torch.Tensor:
    pick = [int(torch.randint(len(bank[m]), (1,))) for m in meaning.tolist()]
    return torch.stack([bank[m][k] for m, k in zip(meaning.tolist(), pick)])

  env = S2Env(args.system1, tokenizer, args.envs, device)
  policy = [p for n, p in model.named_parameters() if not n.startswith("value.")]
  optimiser = torch.optim.AdamW([{"params": policy, "lr": args.lr}, {"params": model.value.parameters(), "lr": 3.0e-4}],
                                weight_decay=0.0)
  obs = env.reset()
  log = []
  began = time.time()
  for iteration in range(args.iterations):
    model.eval()
    store = {key: [] for key in ("text", "history", "codes", "logp", "value", "reward", "ended", "fell", "kind")}
    for _ in range(args.horizon):
      text = words(obs["meaning"])
      codes, logp, value, goal = model.sample(text, obs["history"], args.temperature)
      kind = env.now.clone()
      next_obs, reward, fell, ended, _ = env.step(codes, None)
      for key, item in zip(store, (text, obs["history"], codes, logp, value, reward, ended, fell, kind)):
        store[key].append(item)
      obs = next_obs
    with torch.no_grad():
      _, _, last_value, _ = model.sample(words(obs["meaning"]), obs["history"], args.temperature)
    data = {key: torch.stack(value) for key, value in store.items()}             # (horizon, envs, ...)
    advantage = torch.zeros_like(data["reward"])
    running = torch.zeros(args.envs, device=device)
    following = last_value
    for t in reversed(range(args.horizon)):
      alive = (~data["ended"][t]).float()                    # an ended episode (fall or time limit) is not continued
      delta = data["reward"][t] + args.gamma * following * alive - data["value"][t]
      running = delta + args.gamma * args.lam * alive * running
      advantage[t] = running
      following = data["value"][t]
    returns = advantage + data["value"]
    flat = {key: value.reshape(-1, *value.shape[2:]) for key, value in data.items()}
    flat_advantage = advantage.reshape(-1)
    flat_advantage = (flat_advantage - flat_advantage.mean()) / (flat_advantage.std() + 1.0e-6)
    flat_returns = returns.reshape(-1)
    model.train()
    count = len(flat_returns)
    stats = dict(policy=0.0, value=0.0, away=0.0, ratio=0.0, n=0)
    for _ in range(3):
      order = torch.randperm(count, device=device)
      for rows in order.split(count // 4):
        logits, logp, entropy, value = model.evaluate(flat["text"][rows], flat["history"][rows], flat["codes"][rows],
                                                      args.temperature)
        with torch.no_grad():
          anchor, _, _, _ = reference.evaluate(flat["text"][rows], flat["history"][rows], flat["codes"][rows], args.temperature)
        away = (F.softmax(logits, -1) * (F.log_softmax(logits, -1) - F.log_softmax(anchor, -1))).sum(-1).mean()
        ratio = torch.exp(logp - flat["logp"][rows])                      # (batch, AHEAD, dims)
        gain = flat_advantage[rows].reshape(-1, 1, 1)
        loss_policy = -torch.min(ratio * gain, ratio.clamp(1.0 - args.clip, 1.0 + args.clip) * gain).mean()
        loss_value = F.mse_loss(value, flat_returns[rows])
        scale = 0.0 if iteration < args.value_warmup else 1.0          # the value is learned before the policy moves
        loss = scale * (loss_policy + args.anchor * away * model.dims * 10) + 0.5 * loss_value
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        stats["policy"] += float(loss_policy); stats["value"] += float(loss_value); stats["away"] += float(away)
        stats["ratio"] += float((ratio - 1.0).abs().mean()); stats["n"] += 1
    kinds = {}
    for name in ("stand", "walk", "turn", "crouch", "rise"):
      mask = torch.tensor([NAMES[k].split("_")[0] == name for k in flat["kind"].tolist()], device=device)
      kinds[name] = round(float(flat["reward"][mask].mean()), 4) if mask.any() else None
    row = dict(iteration=iteration, reward=round(float(data["reward"].mean()), 4), falls=int(data["fell"].sum()),
               by_kind=kinds, value_loss=round(stats["value"] / stats["n"], 4), away=round(stats["away"] / stats["n"], 5),
               ratio=round(stats["ratio"] / stats["n"], 4), seconds=round(time.time() - began))
    log.append(row)
    print(json.dumps(row), flush=True)
  model.eval()
  meta = dict(model.meta, ppo=dict(source=args.model.name, system1=str(args.system1), envs=args.envs,
                                   iterations=args.iterations, horizon=args.horizon, lr=args.lr, anchor=args.anchor,
                                   gamma=args.gamma, first=log[0], last=log[-1]))
  model.save(args.out, **meta)
  print("saved", args.out)


if __name__ == "__main__":
  main()
