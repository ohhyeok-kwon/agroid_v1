"""The small System 2: instruction + the motion codes it has emitted -> the next motion codes and the goal.

It is the stand-in that closes the loop (instruction -> System 2 -> tokens -> System 1) before the large model
exists, and the shape of the student the large model is distilled into. One decision every DECISION_CODES codes
(16 control steps): it reads the instruction as a sentence embedding (r1wb.s2.text) and the last HISTORY codes, and
writes the next AHEAD codes one after the other (each code's `dims` numbers at once), plus the goal the instruction
means for System 1 (twist, height and whether the height counts).

Time, in codes of `stride` control steps: at a decision the code being played (g) is fixed, codes g+1 .. g+AHEAD are
written. The instruction it reads takes effect one decision later (code g+2 onward), as in the training courses,
where instructions change on the decision grid.

  python -m r1wb.s2.small --courses <courses.npz> [more.npz ...] --library <library.npz> --tokenizer <tokenizer.pt>
      --text-cache <text_cache.pt> --out <model.pt> [--condition goal --joint --change-weight 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from r1wb.s2 import instructions
from r1wb.s2.text import WIDTH as TEXT_WIDTH

HISTORY, AHEAD, DECISION_CODES = 8, 10, 2
GOAL_DIM = 5                                    # vx, vy, yaw rate, height, height counts
GOAL_SCALE = (0.35, 0.35, 1.0, 0.1, 1.0)
GRID_ORIGIN = 1                                 # recorded step at which code 0 starts (courses change on 16k + 1)


class SmallS2(nn.Module):
  def __init__(self, dims: int = 8, levels: int = 7, width: int = 384, layers: int = 6, heads: int = 6,
               condition: str = "text", joint: bool = False):
    """condition: what the writer is told. "text": the instruction's sentence embedding (the stand-alone small
    System 2). "goal": the goal itself (twist, height, whether the height counts; divided by GOAL_SCALE): then this
    is the motion-code head of a vision-language model that reads the instruction and states the goal.
    joint: the numbers of a code are chosen one after the other, each knowing the ones before it. Chosen each on its
    own, a code can mix two recorded continuations where the recordings have more than one (12% of the walking and
    turning decisions): such a code occurs nowhere in the data and the codes after it drift (gates 11-13 04:35)."""
    super().__init__()
    self.dims, self.levels, self.width, self.condition, self.joint = dims, levels, width, condition, joint
    self.text = nn.Sequential(nn.Linear(TEXT_WIDTH if condition == "text" else GOAL_DIM, width), nn.SiLU(),
                              nn.Linear(width, width))
    self.code = nn.Embedding(dims * levels, width)
    self.position = nn.Parameter(torch.zeros(1 + HISTORY + AHEAD, width))
    nn.init.normal_(self.position, std=0.02)
    layer = nn.TransformerEncoderLayer(width, heads, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
    self.trunk = nn.TransformerEncoder(layer, layers)
    self.norm = nn.LayerNorm(width)
    if joint:
      self.given = nn.Embedding(dims * levels, width)         # a number already chosen, for the numbers after it
      self.place = nn.Parameter(torch.zeros(dims, width))
      nn.init.normal_(self.place, std=0.02)
      self.within = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, levels))
    else:
      self.head = nn.Linear(width, dims * levels)
    self.goal = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, GOAL_DIM))
    self.value = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))   # for policy-gradient training
    self.register_buffer("_offset", torch.arange(dims) * levels, persistent=False)
    self.register_buffer("goal_scale", torch.tensor(GOAL_SCALE), persistent=False)

  def _embed(self, codes: torch.Tensor) -> torch.Tensor:
    """codes (batch, n, dims) with values 0 .. levels-1 -> (batch, n, width)."""
    return self.code(codes + self._offset).sum(dim=2)

  def _run(self, text: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    sequence = torch.cat([self.text(text).unsqueeze(1), self._embed(codes)], dim=1)
    sequence = sequence + self.position[:sequence.shape[1]]
    mask = nn.Transformer.generate_square_subsequent_mask(sequence.shape[1], device=sequence.device)
    return self.norm(self.trunk(sequence, mask=mask, is_causal=True))

  def _logits(self, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
    """hidden (batch, n, width) at the positions that write `codes` (batch, n, dims) -> (batch, n, dims, levels).
    joint: every number is scored knowing the numbers of the same code before it (those of `codes`)."""
    if not self.joint:
      return self.head(hidden).reshape(*hidden.shape[:2], self.dims, self.levels)
    given = self.given(codes + self._offset)                              # (batch, n, dims, width)
    before = torch.cumsum(given, dim=2) - given
    return self.within(hidden.unsqueeze(2) + self.place + before)

  def _write(self, hidden: torch.Tensor, temperature: float | None = None):
    """hidden (batch, width) -> one code (batch, dims) and the log-probability of each of its numbers. Most likely
    values, or drawn at `temperature`."""
    if not self.joint:
      logits = self.head(hidden).reshape(len(hidden), self.dims, self.levels)
      if temperature is None:
        return logits.argmax(dim=-1), None
      choice = torch.distributions.Categorical(logits=logits / temperature)
      code = choice.sample()
      return code, choice.log_prob(code)
    state, numbers, logp = hidden, [], []
    for d in range(self.dims):
      logits = self.within(state + self.place[d])
      if temperature is None:
        number = logits.argmax(dim=-1)
      else:
        choice = torch.distributions.Categorical(logits=logits / temperature)
        number = choice.sample()
        logp.append(choice.log_prob(number))
      numbers.append(number)
      state = state + self.given(number + self._offset[d])
    return torch.stack(numbers, dim=1), (torch.stack(logp, dim=1) if logp else None)

  def forward(self, text: torch.Tensor, history: torch.Tensor, target: torch.Tensor):
    """Teacher forcing. Returns logits (batch, AHEAD, dims, levels) and the goal (batch, GOAL_DIM, physical units)."""
    hidden = self._run(text, torch.cat([history, target[:, :-1]], dim=1))
    return self._logits(hidden[:, HISTORY:], target), self.goal(hidden[:, HISTORY]) * self.goal_scale

  @torch.no_grad()
  def decide(self, text: torch.Tensor, history: torch.Tensor):
    """history (batch, HISTORY, dims) -> codes (batch, AHEAD, dims), goal (batch, GOAL_DIM). Most likely value each."""
    codes, goal = history, None
    for _ in range(AHEAD):
      hidden = self._run(text, codes)
      if goal is None:
        goal = self.goal(hidden[:, HISTORY]) * self.goal_scale
      step, _ = self._write(hidden[:, -1])
      codes = torch.cat([codes, step.unsqueeze(1)], dim=1)
    return codes[:, HISTORY:], goal

  @torch.no_grad()
  def sample(self, text: torch.Tensor, history: torch.Tensor, temperature: float = 1.0):
    """Codes drawn one after the other. Returns codes, the log-probability of every number (batch, AHEAD, dims),
    value (batch,), goal."""
    codes, logp, goal, value = history, [], None, None
    for _ in range(AHEAD):
      hidden = self._run(text, codes)
      if goal is None:
        goal = self.goal(hidden[:, HISTORY]) * self.goal_scale
        value = self.value(hidden[:, HISTORY].detach()).squeeze(-1)
      step, chance = self._write(hidden[:, -1], temperature)
      logp.append(chance)
      codes = torch.cat([codes, step.unsqueeze(1)], dim=1)
    return codes[:, HISTORY:], torch.stack(logp, dim=1), value, goal

  def evaluate(self, text: torch.Tensor, history: torch.Tensor, codes: torch.Tensor, temperature: float = 1.0):
    """For given codes: logits (batch, AHEAD, dims, levels) at that temperature, the log-probability and entropy of
    every number (batch, AHEAD, dims), value."""
    hidden = self._run(text, torch.cat([history, codes[:, :-1]], dim=1))
    logits = self._logits(hidden[:, HISTORY:], codes) / temperature
    choice = torch.distributions.Categorical(logits=logits)
    # The value reads the state without sending gradients into the code writer (its errors must not move the policy).
    value = self.value(hidden[:, HISTORY].detach()).squeeze(-1)
    return logits, choice.log_prob(codes), choice.entropy(), value

  def save(self, path: Path, **meta) -> None:
    torch.save(dict(state=self.state_dict(), config=dict(dims=self.dims, levels=self.levels, width=self.width,
                                                         layers=len(self.trunk.layers), condition=self.condition,
                                                         joint=self.joint),
                    meta=meta), path)

  @classmethod
  def load(cls, path: Path, device: str = "cpu") -> "SmallS2":
    blob = torch.load(path, map_location=device, weights_only=False)
    model = cls(**blob["config"]).to(device)
    model.load_state_dict(blob["state"], strict=False)       # older files have no value head
    model.meta = blob["meta"]
    return model.eval()


def goal_of(instruction: dict) -> list[float]:
  row = instructions.command_row(instruction, 0.0)
  return [row[1], row[2], row[3], row[4], row[5]]


def twin_frames(library: Path) -> np.ndarray:
  """For every library frame, the same moment of its clip's mirror image (every clip is stored with one)."""
  data = np.load(library, allow_pickle=False)
  clips = json.loads(str(data["manifest"]))["clips"]
  place = {clip["name"]: (int(clip["start"]), int(clip["length"])) for clip in clips}
  twin = np.arange(int(data["joint_pos"].shape[0]))
  for clip in clips:
    other = clip["name"][:-2] if clip["mirrored"] else clip["name"] + "_M"
    (start, length), (there, same) = place[clip["name"]], place[other]
    assert length == same, clip["name"]
    twin[start:start + length] = there + np.arange(length)
  return twin


def mirrored(courses: dict, twin: np.ndarray) -> dict:
  """The same recording seen in a mirror: every frame of the plan is that moment of the mirrored clip, sideways and
  turning commands change sign, left and right swap in the instructions. The body, System 1 and the generator are
  left-right symmetric, so this is the course the mirrored situation would have produced; a head trained on one
  side only answers mirrored commands differently (11-13 23:0x: stopping out of a walk to the left its plan kept
  stepping for 2 s, out of a walk to the right it froze at once)."""
  flip = np.array([1.0, -1.0, -1.0, 1.0, 1.0], np.float32)
  swap = lambda text: text.replace("left", "\0").replace("right", "left").replace("\0", "right")  # noqa: E731
  scripts = [[[{key: (swap(value) if key in ("direction", "side") else value) for key, value in item.items()}, seconds]
              for item, seconds in script] for script in json.loads(str(courses["scripts"]))]
  out = dict(courses)
  out.update(frame=twin[courses["frame"]], stream=courses["stream"] * flip, target=courses["target"] * flip,
             scripts=np.array(json.dumps(scripts)))
  return out


def decisions(courses: dict, codes: torch.Tensor, steady_only: bool = True):
  """Training examples from recorded courses. codes: (envs, n, dims), values 0 .. levels-1, code k = recorded steps
  GRID_ORIGIN + stride*k ...; returns (env index, code index g) of every usable decision and, per decision, the
  script position of the instruction it reads (the one in force one decision later).
  steady_only=False (steered recordings: the command changes at every decision): every decision is kept."""
  stride = (courses["frame"].shape[0] - GRID_ORIGIN) // codes.shape[1]
  segment = torch.as_tensor(courses["segment"].T.astype(np.int64))                 # (envs, steps)
  whole = ~torch.as_tensor(courses["start"][2:].any(axis=0))                        # environments without a restart
  at_code = segment[:, GRID_ORIGIN::stride][:, :codes.shape[1]]                     # instruction in force at each code
  grid = torch.arange(HISTORY - 1, codes.shape[1] - AHEAD - DECISION_CODES, DECISION_CODES)
  envs = whole.nonzero().flatten()
  env_index = envs.repeat_interleave(len(grid))
  code_index = grid.repeat(len(envs))
  seen = at_code[env_index, code_index + DECISION_CODES]
  # The instruction must stay in force over the codes that are written (a later change cannot be known now).
  steady = (at_code[env_index, code_index + AHEAD] == seen) | (not steady_only)
  return env_index[steady], code_index[steady], seen[steady]


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--courses", type=Path, nargs="+", required=True)
  parser.add_argument("--library", type=Path, required=True)
  parser.add_argument("--tokenizer", type=Path, required=True)
  parser.add_argument("--text-cache", type=Path, default=None, help="sentence embeddings (--condition text)")
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--steps", type=int, default=6000)
  parser.add_argument("--change-weight", type=float, default=1.0,
                      help="weight of the decisions whose first written code starts a new instruction (they are one "
                           "decision in fourteen and decide how every change of motion begins)")
  parser.add_argument("--batch", type=int, default=512)
  parser.add_argument("--languages", default="ko,en")
  parser.add_argument("--condition", choices=["text", "goal"], default="text")
  parser.add_argument("--joint", action="store_true", help="the numbers of a code are chosen one after the other")
  parser.add_argument("--mirror", action="store_true",
                      help="every recording is also learned from as its mirror image (small.mirrored): exact pairs")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  torch.manual_seed(0)
  rng = random.Random(0)
  from r1wb.s2.text import TextEncoder
  from r1wb.s2.tokenizer import MotionTokenizer, course_frames, library_table
  tokenizer = MotionTokenizer.load(args.tokenizer, args.device)
  table = library_table(args.library, args.device)
  scripts, parts, held_parts, picked, commands = [], [], [], [], []
  twin = twin_frames(args.library) if args.mirror else None
  recordings = []
  for path in args.courses:                                    # several recordings: environments one after the other
    recorded = dict(np.load(path, allow_pickle=False))
    recordings += [recorded] + ([mirrored(recorded, twin)] if args.mirror else [])
  for courses in recordings:
    steered = bool(json.loads(str(courses["meta"])).get("steer"))
    frames = course_frames(courses, table)
    length = ((frames.shape[1] - GRID_ORIGIN) // tokenizer.stride) * tokenizer.stride
    with torch.no_grad():
      codes = torch.cat([tokenizer.encode(part[:, GRID_ORIGIN:GRID_ORIGIN + length]) for part in frames.split(256)])
    codes = (codes + tokenizer.levels // 2).cpu()                                  # 0 .. levels-1
    env_index, code_index, seen = decisions(courses, codes, steady_only=not steered)
    if steered:
      # The command in force over the two codes this decision plays (written at recorded step 16k, visible at 16k + 1).
      at = GRID_ORIGIN + tokenizer.stride * (code_index + 1)
      told = torch.as_tensor(courses["target"])[at, env_index]
      commands.append(torch.cat([told[:, :4], torch.as_tensor(courses["counts"])[at, env_index].unsqueeze(1)], dim=1))
    else:
      commands.append(torch.full((len(env_index), GOAL_DIM), float("nan")))       # the instruction's own goal (below)
    changed = torch.as_tensor(courses["segment"].T.astype(np.int64))[:, GRID_ORIGIN::tokenizer.stride][:, :codes.shape[1]]
    changed = changed[env_index, code_index + 1] != changed[env_index, code_index]   # the first written code is new
    first = sum(len(part) for part in parts)
    picked.append((env_index + first, code_index, seen, changed,
                   env_index >= int(codes.shape[0] * 0.95)))                        # last environments of each: validation
    parts.append(codes)
    scripts += json.loads(str(courses["scripts"]))
    del frames
  assert len({part.shape[1] for part in parts}) == 1, "recordings of the same length only"
  codes = torch.cat(parts)
  env_index, code_index, seen, changes, held = (torch.cat([item[i] for item in picked]) for i in range(5))
  languages = args.languages.split(",")
  meanings = instructions.every_instruction()
  names = [instructions.identifier(item) for item in meanings]
  phrases = {(name, lang): instructions.all_phrases(item, lang) for item, name in zip(meanings, names) for lang in languages}
  # All training phrasings of a meaning (both languages) in one bank: [first, count) per meaning.
  if args.condition == "text":
    encoder = TextEncoder(args.text_cache, "cpu")
    bank = [encoder([text for lang in languages for text in phrases[(name, lang)]]) for name in names]
  else:                                                        # told the goal: no sentence is read
    bank = [torch.zeros(1, TEXT_WIDTH) for _ in names]
  bank_count = torch.tensor([len(part) for part in bank], device=args.device)
  bank_first = torch.cumsum(bank_count, 0) - bank_count
  bank = torch.cat(bank).to(args.device)
  # Per decision: which meaning it reads, and that meaning's goal.
  meaning = torch.tensor([names.index(instructions.identifier(scripts[e][s][0])) for e, s in zip(env_index.tolist(), seen.tolist())])
  goals = torch.tensor([goal_of(item) for item in meanings])
  # Per decision: the goal the writer is told. Steered recordings: the command that was in force; others: the
  # instruction's own goal.
  command = torch.cat(commands)
  command = torch.where(torch.isnan(command), goals[meaning], command)
  train_rows, valid_rows = (~held).nonzero().flatten(), held.nonzero().flatten()
  print(f"decisions: {len(train_rows)} for training, {len(valid_rows)} held out; codes {tuple(codes.shape)}; "
        f"{len(names)} meanings, {sum(len(v) for v in phrases.values())} training phrasings", flush=True)

  codes_gpu, goals_gpu, meaning_gpu = codes.to(args.device), goals.to(args.device), meaning.to(args.device)
  command_gpu = command.to(args.device)
  weight_gpu = torch.where(changes, args.change_weight, 1.0).to(args.device)
  env_gpu, code_gpu = env_index.to(args.device), code_index.to(args.device)
  offsets = torch.arange(-HISTORY + 1, AHEAD + 1, device=args.device)

  def batch_of(rows):
    rows = rows.to(args.device)
    what = meaning_gpu[rows]
    window = codes_gpu[env_gpu[rows].unsqueeze(1), code_gpu[rows].unsqueeze(1) + offsets]
    pick = bank_first[what] + (torch.rand(len(rows), device=args.device) * bank_count[what]).long()
    told = bank[pick] if args.condition == "text" else command_gpu[rows] / goal_scale
    return told, window[:, :HISTORY], window[:, HISTORY:], command_gpu[rows] if args.condition == "goal" else goals_gpu[what], weight_gpu[rows]

  model = SmallS2(tokenizer.dims, tokenizer.levels, condition=args.condition, joint=args.joint).to(args.device)
  goal_scale = model.goal_scale
  optimiser = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=0.01)
  schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 5.0e-4, total_steps=args.steps, pct_start=0.05)
  scale = model.goal_scale
  began = time.time()
  for step in range(args.steps):
    text, history, target, goal, weight = batch_of(train_rows[torch.randint(len(train_rows), (args.batch,))])
    logits, predicted = model(text, history, target)
    each = F.cross_entropy(logits.reshape(-1, model.levels), target.reshape(-1), reduction="none")
    loss_codes = (each.reshape(len(weight), -1).mean(dim=1) * weight).sum() / weight.sum()
    loss_goal = F.mse_loss(predicted / scale, goal / scale)
    loss = loss_codes + loss_goal
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()
    schedule.step()
    if step % 500 == 0 or step == args.steps - 1:
      print(f"step {step:5d} codes {float(loss_codes):.4f} goal {float(loss_goal):.5f}  {time.time() - began:.0f} s", flush=True)
  model.eval()
  with torch.no_grad():
    exact = same = count = 0
    goal_gap = 0.0
    begins = begins_right = 0
    valid_rows = valid_rows[torch.randperm(len(valid_rows), generator=torch.Generator().manual_seed(0))]
    for begin in range(0, min(len(valid_rows), 16384), 1024):
      text, history, target, goal, weight = batch_of(valid_rows[begin:begin + 1024])
      logits, predicted = model(text, history, target)
      same += float((logits.argmax(-1) == target).float().mean()) * len(text)        # next-code accuracy, teacher forced
      written, free_goal = model.decide(text, history)
      exact += float((written == target).float().mean()) * len(text)                  # all ten codes written freely
      goal_gap += float((free_goal - goal).abs().mean(0).max()) * len(text)
      count += len(text)
      new = weight != 1.0 if args.change_weight != 1.0 else changes[valid_rows[begin:begin + 1024]].to(args.device)
      begins += int(new.sum())
      begins_right += int((written[:, 0] == target[:, 0]).all(dim=1)[new].sum())       # a change of instruction begun as recorded
  meta = dict(courses=[path.name for path in args.courses], mirror=args.mirror, tokenizer=args.tokenizer.name, steps=args.steps, languages=languages,
              change_weight=args.change_weight, valid_changes_begun_as_recorded=round(begins_right / max(begins, 1), 4),
              condition=args.condition, joint=args.joint,
              decisions=int(len(train_rows)), history=HISTORY, ahead=AHEAD, decision_codes=DECISION_CODES,
              stride=tokenizer.stride, valid_code_accuracy_teacher_forced=round(same / count, 4),
              valid_code_accuracy_free=round(exact / count, 4), valid_goal_gap_max=round(goal_gap / count, 4),
              parameters=sum(p.numel() for p in model.parameters()))
  model.save(args.out, **meta)
  print(json.dumps(meta, indent=1))


if __name__ == "__main__":
  main()
