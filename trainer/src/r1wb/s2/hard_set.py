"""A harder phrasing test for the goal model, and its scoring.

The held-out phrasings of r1wb.s2.instructions are templates with other words; the extra training phrasings were
written by the 8B model itself, so they resemble them. These were written by hand on 2026-11-13 03:50 KST, before any
model was scored on them: colloquial and indirect wording, negation and contrast, missing spaces and typos, mixed
language, speeds given as numbers. None is used for training (five that coincided with a template or a generated
training phrasing were taken out before scoring).

  python -m r1wb.s2.hard_set --backbone <folder> --model <goal model folder> [--frames 4]
prints the score per language and group for the hard set, and for the training and held-out phrasings again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from r1wb.s2 import instructions

# (group, phrasing, meaning)
HARD = {
  "ko": [
    ("colloquial", "거기 서", "stand"), ("colloquial", "이제 그만 움직여", "stand"), ("colloquial", "스톱", "stand"),
    ("colloquial", "동작 그만", "stand"), ("colloquial", "아무것도 하지 말고 서 있어", "stand"),
    ("colloquial", "이제 일어나", "rise"), ("colloquial", "수그린 거 풀고 일어서", "rise"),
    ("colloquial", "다시 똑바로 서 줘", "rise"), ("colloquial", "몸 펴", "rise"),
    ("colloquial", "직진해", "walk_forward_normal"),
    ("colloquial", "앞쪽으로 걸어가 볼래?", "walk_forward_normal"), ("colloquial", "전진", "walk_forward_normal"),
    ("colloquial", "앞으로 살살 걸어", "walk_forward_slow"), ("colloquial", "조심조심 앞으로 가", "walk_forward_slow"),
    ("colloquial", "앞으로 빨리빨리 걸어", "walk_forward_fast"), ("colloquial", "속도 내서 직진해", "walk_forward_fast"),
    ("colloquial", "뒤로 물러나", "walk_backward_normal"), ("colloquial", "후진해", "walk_backward_normal"),
    ("colloquial", "뒷걸음질 쳐", "walk_backward_normal"), ("colloquial", "뒤로 살살 물러서", "walk_backward_slow"),
    ("colloquial", "천천히 뒷걸음질 해", "walk_backward_slow"),
    ("colloquial", "왼쪽으로 옆걸음 해", "walk_left_normal"), ("colloquial", "좌로 이동", "walk_left_normal"),
    ("colloquial", "왼쪽으로 살살 옆으로 가", "walk_left_slow"),
    ("colloquial", "오른쪽으로 옆걸음 해", "walk_right_normal"), ("colloquial", "우로 이동", "walk_right_normal"),
    ("colloquial", "오른쪽으로 살금살금 옆으로 가", "walk_right_slow"),
    ("colloquial", "왼쪽 앞으로 비스듬히 걸어", "walk_forward_left_normal"),
    ("colloquial", "오른쪽 앞으로 비스듬히 가", "walk_forward_right_normal"),
    ("colloquial", "왼쪽 뒤로 비스듬히 물러나", "walk_backward_left_normal"),
    ("colloquial", "오른쪽 뒤로 천천히 비스듬히 물러나", "walk_backward_right_slow"),
    ("colloquial", "왼쪽으로 살살 돌아봐", "turn_left_slow"), ("colloquial", "천천히 좌회전", "turn_left_slow"),
    ("colloquial", "왼쪽으로 휙 돌아", "turn_left_fast"), ("colloquial", "빨리 좌회전해", "turn_left_fast"),
    ("colloquial", "오른쪽으로 천천히 돌아봐", "turn_right_slow"), ("colloquial", "서서히 우회전", "turn_right_slow"),
    ("colloquial", "오른쪽으로 휙 돌아", "turn_right_fast"), ("colloquial", "재빨리 우회전", "turn_right_fast"),
    ("colloquial", "무릎을 살짝만 굽혀", "crouch_slight"), ("colloquial", "약간만 낮게 서", "crouch_slight"),
    ("colloquial", "푹 쪼그려", "crouch_deep"), ("colloquial", "최대한 낮게 웅크려 앉아", "crouch_deep"),
    ("number", "초속 0.2미터로 전진해", "walk_forward_slow"), ("number", "초속 0.5미터로 앞으로 걸어", "walk_forward_fast"),
    ("number", "초당 1라디안으로 왼쪽으로 돌아", "turn_left_fast"),
    ("negation", "앞으로 말고 뒤로 걸어", "walk_backward_normal"), ("negation", "돌지 말고 그냥 서 있어", "stand"),
    ("negation", "빠르게 말고 천천히 앞으로 걸어", "walk_forward_slow"),
    ("negation", "오른쪽 말고 왼쪽으로 천천히 돌아", "turn_left_slow"),
    ("negation", "깊게 말고 살짝만 수그려", "crouch_slight"),
    ("typo", "앞으로걸어", "walk_forward_normal"), ("typo", "왼쪽으로빠르게돌아", "turn_left_fast"),
    ("typo", "뒤로 천천이 걸어", "walk_backward_slow"), ("typo", "깊게 수구려", "crouch_deep"),
    ("mixed", "forward로 천천히 가", "walk_forward_slow"), ("mixed", "왼쪽으로 turn 빠르게", "turn_left_fast"),
    ("mixed", "stop 해", "stand"), ("mixed", "backward로 걸어", "walk_backward_normal"),
  ],
  "en": [
    ("colloquial", "hold it right there", "stand"),
    ("colloquial", "that's enough, stand still", "stand"), ("colloquial", "no more moving", "stand"),
    ("colloquial", "get back on your feet", "rise"), ("colloquial", "come back up", "rise"),
    ("colloquial", "stand tall again", "rise"),
    ("colloquial", "go ahead", "walk_forward_normal"),
    ("colloquial", "walk straight", "walk_forward_normal"), ("colloquial", "advance", "walk_forward_normal"),
    ("colloquial", "creep forward", "walk_forward_slow"), ("colloquial", "walk forward nice and slow", "walk_forward_slow"),
    ("colloquial", "hurry forward", "walk_forward_fast"), ("colloquial", "walk forward as fast as you can", "walk_forward_fast"),
    ("colloquial", "back up", "walk_backward_normal"), ("colloquial", "walk backwards", "walk_backward_normal"),
    ("colloquial", "reverse", "walk_backward_normal"), ("colloquial", "back up slowly", "walk_backward_slow"),
    ("colloquial", "ease backwards", "walk_backward_slow"),
    ("colloquial", "sidestep left", "walk_left_normal"), ("colloquial", "shuffle to your left", "walk_left_normal"),
    ("colloquial", "sidestep left slowly", "walk_left_slow"),
    ("colloquial", "sidestep right", "walk_right_normal"), ("colloquial", "strafe right", "walk_right_normal"),
    ("colloquial", "slowly shuffle to the right", "walk_right_slow"),
    ("colloquial", "walk forward and left at an angle", "walk_forward_left_normal"),
    ("colloquial", "go diagonally ahead to your right", "walk_forward_right_normal"),
    ("colloquial", "back away diagonally to the left", "walk_backward_left_normal"),
    ("colloquial", "slowly back away diagonally to the right", "walk_backward_right_slow"),
    ("colloquial", "slowly turn to face left", "turn_left_slow"), ("colloquial", "rotate counterclockwise slowly", "turn_left_slow"),
    ("colloquial", "quick spin to the left", "turn_left_fast"), ("colloquial", "whip around to your left", "turn_left_fast"),
    ("colloquial", "gently turn right on the spot", "turn_right_slow"), ("colloquial", "rotate clockwise slowly", "turn_right_slow"),
    ("colloquial", "spin right fast on the spot", "turn_right_fast"), ("colloquial", "turn clockwise quickly", "turn_right_fast"),
    ("colloquial", "bend your knees a little", "crouch_slight"), ("colloquial", "lower yourself just a bit", "crouch_slight"),
    ("colloquial", "squat all the way down", "crouch_deep"), ("colloquial", "get as low as you can", "crouch_deep"),
    ("number", "go forward at 0.2 m/s", "walk_forward_slow"), ("number", "walk forward at half a metre per second", "walk_forward_fast"),
    ("number", "turn left at one radian per second", "turn_left_fast"),
    ("negation", "not forward, walk backward", "walk_backward_normal"), ("negation", "don't turn, just stand", "stand"),
    ("negation", "walk forward, but slowly, not fast", "walk_forward_slow"),
    ("negation", "turn left, not right, and do it slowly", "turn_left_slow"),
    ("negation", "crouch, but only a little, not deeply", "crouch_slight"),
    ("typo", "wlak forward", "walk_forward_normal"), ("typo", "turn rigth quickly", "turn_right_fast"),
    ("typo", "walk bakcward slowly", "walk_backward_slow"), ("typo", "crouch deeeply", "crouch_deep"),
    ("mixed", "앞으로 walk slowly", "walk_forward_slow"), ("mixed", "turn 왼쪽 fast", "turn_left_fast"),
    ("mixed", "멈춰 please", "stand"), ("mixed", "뒤로 walk", "walk_backward_normal"),
  ],
}
TOLERANCE = (0.07, 0.07, 0.15, 0.04, 0.5)                    # as r1wb.s2.train_goal: twist, height, flag


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--backbone", type=Path, required=True)
  parser.add_argument("--model", type=Path, required=True, help="folder with goal_last.pt")
  parser.add_argument("--images", type=Path, default=None, help="camera frames (.npy, decisions x envs x H x W x 3)")
  parser.add_argument("--frames", type=int, default=4, help="camera frames each phrasing is read with")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  from r1wb.s2.small import goal_of
  from r1wb.s2.vla import GoalVLA
  model = GoalVLA(args.backbone).to(args.device).eval()
  weights = torch.load(args.model / "goal_last.pt", map_location="cpu", weights_only=False)
  model.load_state_dict(weights["state"], strict=False)
  rng = random.Random(0)
  if args.images is not None:
    images = np.load(args.images, mmap_mode="r")
    pictures = [np.asarray(images[rng.randrange(images.shape[0]), rng.randrange(images.shape[1])]) for _ in range(args.frames)]
  else:                                                       # no recorded frames at hand: plain grey pictures
    pictures = [np.full((224, 224, 3), 96 + 32 * i, dtype=np.uint8) for i in range(args.frames)]
  meanings = instructions.every_instruction()
  goal = {instructions.identifier(item): torch.tensor(goal_of(item)) for item in meanings}
  tolerance = torch.tensor(TOLERANCE)

  vocabulary = torch.stack(list(goal.values()))
  names = list(goal)

  def stated(texts: list[str], choose: bool = False) -> torch.Tensor:
    out = []
    for text in texts:
      batch = model.pack(pictures, [text] * len(pictures)).to(args.device)
      with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        out.append(model.decide(batch, vocabulary.to(args.device) if choose else None).float().cpu())
    return torch.stack(out)                                   # (texts, frames, 5)

  def chosen_right(rows: list) -> tuple[int, list]:
    """As the runtime reads the model (the nearest instruction of the vocabulary): right in every frame?"""
    said = stated([text for _, text, _ in rows], choose=True)
    right, misses = 0, []
    for (group, text, name), value in zip(rows, said):
      picked = {names[int(((vocabulary - frame).abs().sum(dim=1)).argmin())] for frame in value}
      if picked == {name}:
        right += 1
      else:
        misses.append((text, name, sorted(picked)))
    return right, misses

  report = {}
  for language in ("ko", "en"):
    rows = HARD[language]
    said = stated([text for _, text, _ in rows])
    want = torch.stack([goal[name] for _, _, name in rows]).unsqueeze(1)
    right = ((said - want).abs() <= tolerance).all(dim=2).all(dim=1)       # right in every frame
    by_group = {}
    for (group, text, name), ok, value in zip(rows, right.tolist(), said):
      cell = by_group.setdefault(group, [0, 0, []])
      cell[0] += int(ok); cell[1] += 1
      if not ok:
        cell[2].append((text, name, [round(float(v), 2) for v in value[0]]))
    report[language] = dict(right=int(right.sum()), total=len(rows),
                            groups={group: dict(right=cell[0], total=cell[1], misses=cell[2]) for group, cell in by_group.items()})
    print(f"hard set, {language}: {int(right.sum())}/{len(rows)}   " + "  ".join(f"{g} {c[0]}/{c[1]}" for g, c in by_group.items()))
    for group, cell in by_group.items():
      for text, name, value in cell[2]:
        print(f"    miss [{group}] {text!r} -> wanted {name}, stated {value}")
    count, misses = chosen_right(rows)
    report[language]["chosen_right"] = count
    print(f"hard set, {language}, read as the nearest instruction: {count}/{len(rows)}")
    for text, name, picked in misses:
      same_kind = all(p.split("_")[0] == name.split("_")[0] for p in picked)
      print(f"    chose {picked} for {text!r} (wanted {name}){'' if same_kind else '   <-- another kind of motion'}")
    for held in (False, True):
      texts, wants = [], []
      for item in meanings:
        for text in instructions.all_phrases(item, language, held):
          texts.append(text); wants.append(goal[instructions.identifier(item)])
      said = stated(texts)
      ok = ((said - torch.stack(wants).unsqueeze(1)).abs() <= tolerance).all(dim=2).all(dim=1)
      report[f"{language}_{'heldout' if held else 'train'}"] = dict(right=int(ok.sum()), total=len(texts))
      print(f"{'held-out' if held else 'training'} phrasings, {language}: {int(ok.sum())}/{len(texts)}")
  (args.model / "hard_set.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
  main()
