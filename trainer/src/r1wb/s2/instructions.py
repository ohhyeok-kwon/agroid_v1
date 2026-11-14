"""The instructions the action model is given, and what each one asks of the body.

An instruction is a kind with parameters. `command_row` turns it into the command the reference generator follows
(twist, height), `phrase` writes it out in Korean or English. Every kind has training phrasings and held-out
phrasings (other wording, never used for training): following the held-out ones is what counts as understanding an
instruction rather than looking it up.
"""
from __future__ import annotations

import math
import random

DEFAULT_HEIGHT = 0.731
DIRECTIONS = {            # name: unit vector (forward, left)
  "forward": (1.0, 0.0), "backward": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0),
  "forward_left": (0.7071, 0.7071), "forward_right": (0.7071, -0.7071),
  "backward_left": (-0.7071, 0.7071), "backward_right": (-0.7071, -0.7071),
}
SPEEDS = {"slow": 0.2, "normal": 0.35, "fast": 0.5}
FAST_DIRECTIONS = ("forward", "forward_left", "forward_right")        # the motion data has no fast walking elsewhere
TURN_RATES = {"slow": 0.6, "fast": 1.0}
CROUCH_HEIGHTS = {"slight": 0.62, "deep": 0.54}

KOREAN = {
  "direction": {"forward": "앞으로", "backward": "뒤로", "left": "왼쪽으로", "right": "오른쪽으로",
                "forward_left": "왼쪽 앞 대각선으로", "forward_right": "오른쪽 앞 대각선으로",
                "backward_left": "왼쪽 뒤 대각선으로", "backward_right": "오른쪽 뒤 대각선으로"},
  "direction_other": {"forward": "전방으로", "backward": "후방으로", "left": "좌측으로", "right": "우측으로",
                      "forward_left": "좌측 전방으로", "forward_right": "우측 전방으로",
                      "backward_left": "좌측 후방으로", "backward_right": "우측 후방으로"},
  "speed": {"slow": "천천히", "normal": "", "fast": "빠르게"},
  "speed_other": {"slow": "느리게", "normal": "보통 속도로", "fast": "빨리"},
}
ENGLISH = {
  "direction": {"forward": "forward", "backward": "backward", "left": "to the left", "right": "to the right",
                "forward_left": "diagonally forward to the left", "forward_right": "diagonally forward to the right",
                "backward_left": "diagonally backward to the left", "backward_right": "diagonally backward to the right"},
  "direction_other": {"forward": "straight ahead", "backward": "back", "left": "leftward", "right": "rightward",
                      "forward_left": "ahead and to the left", "forward_right": "ahead and to the right",
                      "backward_left": "back and to the left", "backward_right": "back and to the right"},
  "speed": {"slow": "slowly", "normal": "", "fast": "fast"},
  "speed_other": {"slow": "at a slow pace", "normal": "at a normal pace", "fast": "quickly"},
}
# kind -> language -> (training phrasings, held-out phrasings). {d} direction, {s} speed, {t} turn side.
PHRASES = {
  "stand": {
    "ko": (["멈춰", "가만히 서 있어", "그 자리에 서", "정지"], ["움직이지 마", "제자리에 멈춰 서 줘", "그대로 있어"]),
    "en": (["stop", "stand still", "halt"], ["do not move", "stay where you are", "hold your position"])},
  "walk": {
    "ko": (["{d} {s} 걸어", "{s} {d} 걸어가", "{d} {s} 이동해"], ["{d} {s} 걸어 줘", "{s} {d} 가", "{d} {s} 걸음을 옮겨"]),
    "en": (["walk {d} {s}", "go {d} {s}", "move {d} {s}"], ["please walk {d} {s}", "step {d} {s}", "head {d} {s}"])},
  "turn": {
    "ko": (["제자리에서 {t} {s} 돌아", "{t} {s} 회전해", "{s} {t} 돌아"], ["그 자리에서 {t} {s} 몸을 돌려", "{t} {s} 방향을 틀어"]),
    "en": (["turn {t} in place {s}", "rotate {t} {s}", "spin {t} {s}"], ["please turn around {t} {s}", "pivot {t} {s}"])},
  "crouch": {
    "ko": (["{c} 수그려", "{c} 몸을 낮춰", "{c} 웅크려"], ["{c} 자세를 낮춰 줘", "{c} 쪼그려 앉아", "{c} 숙여"]),
    "en": (["crouch {c}", "lower your body {c}", "get down {c}"], ["please squat {c}", "duck down {c}"])},
  "rise": {
    "ko": (["일어서", "똑바로 서", "몸을 일으켜"], ["일어나 줘", "허리를 펴고 서", "다시 서"]),
    "en": (["stand up", "rise", "stand up straight"], ["get up", "straighten up", "stand back up"])},
}
TURN_WORDS = {"ko": ({"left": "왼쪽으로", "right": "오른쪽으로"}, {"left": "좌측으로", "right": "우측으로"}),
              "en": ({"left": "left", "right": "right"}, {"left": "to your left", "right": "to your right"})}
TURN_SPEED = {"ko": ({"slow": "천천히", "fast": "빠르게"}, {"slow": "느리게", "fast": "빨리"}),
              "en": ({"slow": "slowly", "fast": "fast"}, {"slow": "at a slow pace", "fast": "quickly"})}
CROUCH_WORDS = {"ko": ({"slight": "살짝", "deep": "깊게"}, {"slight": "조금", "deep": "많이"}),
                "en": ({"slight": "a little", "deep": "deeply"}, {"slight": "slightly", "deep": "all the way"})}


def sample(rng: random.Random, crouched: bool = False) -> dict:
  """One instruction. While crouched the next instruction rises (walking low is not part of this set)."""
  if crouched:
    return {"kind": "rise"} if rng.random() < 0.8 else {"kind": "crouch", "depth": rng.choice(list(CROUCH_HEIGHTS))}
  kind = rng.choices(["stand", "walk", "turn", "crouch"], weights=[0.2, 0.5, 0.16, 0.14])[0]
  if kind == "walk":
    direction = rng.choice(list(DIRECTIONS))
    speeds = list(SPEEDS) if direction in FAST_DIRECTIONS else ["slow", "normal"]
    return {"kind": "walk", "direction": direction, "speed": rng.choice(speeds)}
  if kind == "turn":
    return {"kind": "turn", "side": rng.choice(["left", "right"]), "rate": rng.choice(list(TURN_RATES))}
  if kind == "crouch":
    return {"kind": "crouch", "depth": rng.choice(list(CROUCH_HEIGHTS))}
  return {"kind": "stand"}


def command_row(instruction: dict, duration: float) -> list[float]:
  """[duration_s, vx, vy, yaw_rate, height, height_on, pitch, pitch_on] (student.stream.SCRIPT_COLUMNS)."""
  kind = instruction["kind"]
  vx = vy = wz = 0.0
  height, height_on = DEFAULT_HEIGHT, 0.0
  if kind == "walk":
    x, y = DIRECTIONS[instruction["direction"]]
    speed = SPEEDS[instruction["speed"]]
    vx, vy = round(speed * x, 4), round(speed * y, 4)
  elif kind == "turn":
    wz = TURN_RATES[instruction["rate"]] * (1.0 if instruction["side"] == "left" else -1.0)
  elif kind == "crouch":
    height, height_on = CROUCH_HEIGHTS[instruction["depth"]], 1.0
  elif kind == "rise":
    height_on = 1.0
  return [duration, vx, vy, wz, height, height_on, 0.0, 0.0]


def phrase(instruction: dict, language: str, rng: random.Random, held_out: bool = False) -> str:
  kind = instruction["kind"]
  template = rng.choice(PHRASES[kind][language][1 if held_out else 0])
  words = KOREAN if language == "ko" else ENGLISH
  which = 1 if held_out and rng.random() < 0.5 else 0             # held-out phrasings also swap the parameter words
  fields = {}
  if kind == "walk":
    fields["d"] = words["direction_other" if which else "direction"][instruction["direction"]]
    fields["s"] = words["speed_other" if which else "speed"][instruction["speed"]]
  elif kind == "turn":
    fields["t"] = TURN_WORDS[language][which][instruction["side"]]
    fields["s"] = TURN_SPEED[language][which][instruction["rate"]]
  elif kind == "crouch":
    fields["c"] = CROUCH_WORDS[language][which][instruction["depth"]]
  return " ".join(template.format(**fields).split())


def all_phrases(instruction: dict, language: str, held_out: bool = False) -> list[str]:
  """Every phrasing `phrase` can produce for the instruction (training phrasings use the first word set only)."""
  kind = instruction["kind"]
  words = KOREAN if language == "ko" else ENGLISH
  out = []
  for template in PHRASES[kind][language][1 if held_out else 0]:
    for which in ((0, 1) if held_out else (0,)):
      fields = {}
      if kind == "walk":
        fields["d"] = words["direction_other" if which else "direction"][instruction["direction"]]
        fields["s"] = words["speed_other" if which else "speed"][instruction["speed"]]
      elif kind == "turn":
        fields["t"] = TURN_WORDS[language][which][instruction["side"]]
        fields["s"] = TURN_SPEED[language][which][instruction["rate"]]
      elif kind == "crouch":
        fields["c"] = CROUCH_WORDS[language][which][instruction["depth"]]
      text = " ".join(template.format(**fields).split())
      if text not in out:
        out.append(text)
  return out


_EXTRA: dict = {}


def load_extra(path) -> int:
  """More training phrasings per meaning and language (r1wb.s2.paraphrase): {meaning: {language: [phrasings]}}.
  Held-out phrasings are never taken in. Returns how many were added."""
  import json
  from pathlib import Path
  data = json.loads(Path(path).read_text())
  held = {text for item in every_instruction() for language in ("ko", "en") for text in all_phrases(item, language, True)}
  count = 0
  for name, by_language in data.items():
    for language, texts in by_language.items():
      fresh = [text for text in texts if text not in held]
      _EXTRA.setdefault((name, language), []).extend(fresh)
      count += len(fresh)
  return count


def training_phrases(instruction: dict, language: str) -> list[str]:
  """The hand-written training phrasings plus any loaded extra ones."""
  return all_phrases(instruction, language) + _EXTRA.get((identifier(instruction), language), [])


def identifier(instruction: dict) -> str:
  """A stable name of the instruction's meaning (for grouping results)."""
  kind = instruction["kind"]
  if kind == "walk":
    return f"walk_{instruction['direction']}_{instruction['speed']}"
  if kind == "turn":
    return f"turn_{instruction['side']}_{instruction['rate']}"
  if kind == "crouch":
    return f"crouch_{instruction['depth']}"
  return kind


def every_instruction() -> list[dict]:
  out = [{"kind": "stand"}, {"kind": "rise"}]
  for direction in DIRECTIONS:
    for speed in (SPEEDS if direction in FAST_DIRECTIONS else ("slow", "normal")):
      out.append({"kind": "walk", "direction": direction, "speed": speed})
  for side in ("left", "right"):
    for rate in TURN_RATES:
      out.append({"kind": "turn", "side": side, "rate": rate})
  for depth in CROUCH_HEIGHTS:
    out.append({"kind": "crouch", "depth": depth})
  return out


DECISION_S = 0.32            # System 2 decides every 16 control steps; instructions change on that grid


def script(rng: random.Random, seconds: float, segment_time: tuple[float, float] = (2.56, 6.08)) -> list[tuple[dict, float]]:
  """A sequence of (instruction, duration) filling `seconds`, starting from standing. Durations are multiples of
  DECISION_S, so every change of instruction falls on a System 2 decision."""
  def duration(low, high):
    return round(rng.uniform(low, high) / DECISION_S) * DECISION_S

  out, crouched = [({"kind": "stand"}, duration(0.64, 1.92))], False
  left = seconds - out[0][1]
  while left > 0.0:
    instruction = sample(rng, crouched)
    crouched = instruction["kind"] == "crouch" or (crouched and instruction["kind"] != "rise")
    out.append((instruction, duration(*segment_time)))
    left -= out[-1][1]
  return out


if __name__ == "__main__":
  generator = random.Random(0)
  for item in every_instruction():
    print(f"{identifier(item):28s} | {phrase(item, 'ko', generator):28s} | {phrase(item, 'ko', generator, True):30s} | "
          f"{phrase(item, 'en', generator):42s} | {phrase(item, 'en', generator, True)}")
  print(len(every_instruction()), "instructions;", math.prod([1]))
