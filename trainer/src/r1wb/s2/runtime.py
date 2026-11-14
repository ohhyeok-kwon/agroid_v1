"""System 2 in the loop: instruction -> motion codes -> decoded reference frames -> System 1's observation.

`S2View` stands where the command stream's planner stood: every 16 control steps the System 2 model reads the
instruction (a sentence embedding) and its own last codes and writes the next ones; the tokenizer decodes the codes
into reference frames; the tracking policy's observation is built from those frames and from the goal System 2
states (slew-limited as in training). Nothing of the environment's own planner reaches the policy.

Timing (as in the training courses): code k covers control steps 1 + 8k .. 8 + 8k; a decision is taken at control
steps 1 + 16m, when code 2m starts playing; the instruction read there takes effect one decision later.
"""
from __future__ import annotations

import math
import time

import torch

from r1wb.eval.chunk_view import ChunkView
from r1wb.s2 import frames as s2
from r1wb.s2.small import AHEAD, DECISION_CODES, GRID_ORIGIN, HISTORY

PAST = 2                      # decoded codes before the one that is playing (context for the decoder and velocities)


class S2View(ChunkView):
  def __init__(self, env, term, brain, tokenizer):
    """brain(decision index, history (envs, HISTORY, dims) with values 0..levels-1) -> (codes (envs, AHEAD, dims),
    goal (envs, 5)): the System 2 model with whatever it reads (see small_brain, vla_brain)."""
    stride = tokenizer.stride
    self._period = DECISION_CODES * stride
    super().__init__(env, term, period=self._period, latency=0, horizon=(1 + AHEAD) * stride, dense=True, lead=PAST * stride)
    self.brain, self.tokenizer = brain, tokenizer
    count, device = env.num_envs, env.device
    self._calls = -1
    self._decision = 0
    self._half = tokenizer.levels // 2
    self._history = torch.zeros(count, HISTORY, tokenizer.dims, dtype=torch.long, device=device)   # values 0..levels-1
    self._stream = torch.zeros(count, 5, device=device)
    self._target = torch.zeros(count, 5, device=device)
    self._waiting = torch.zeros(count, 5, device=device)
    self._mask = torch.zeros(count, 3, device=device)
    self._lag = torch.zeros(count, 3, device=device)
    self._twists = torch.zeros(count, term.cfg.goal_window // 2 + 1, 3, device=device)
    self._slew = term.slew
    self.code_log = []
    self.decision_seconds = []                                # wall time of every System 2 decision (all environments)
    self._start(torch.arange(count, device=device))

  def _start(self, ids: torch.Tensor) -> None:
    """An episode begins standing on the library frame the robot was reset to."""
    term, stride = self._term, self.tokenizer.stride
    frame = self._table[term.time_steps[ids]].clone()
    frame[:, s2.STEP] = 0.0
    run = frame.unsqueeze(1).expand(-1, 4 * stride, -1)
    with torch.no_grad():
      code = self.tokenizer.encode(run)[:, 1] + self._half
    self._history[ids] = code.unsqueeze(1).expand(-1, HISTORY, -1)
    self._frames[ids] = frame.unsqueeze(1).expand(-1, self._frames.shape[1], -1)
    self._base["born"][ids] = max(self._calls, 0)
    default = term.cfg.default_height
    self._stream[ids] = torch.tensor([0.0, 0.0, 0.0, default, 0.0], device=frame.device)
    self._target[ids], self._waiting[ids] = self._stream[ids], self._stream[ids]
    self._mask[ids] = torch.tensor([1.0, 0.0, 0.0], device=frame.device)
    self._lag[ids], self._twists[ids] = 0.0, 0.0
    self._reference = None

  def _decide(self) -> None:
    stride = self.tokenizer.stride
    began = time.perf_counter()
    codes, goal = self.brain(self._decision, self._history)
    if codes.is_cuda:
      torch.cuda.synchronize()
    self.decision_seconds.append(time.perf_counter() - began)
    # Decode the whole history with the new codes (a causal decoder reads the codes before a frame), keep
    # [PAST codes before the playing one, the playing one, the new ones]; frame 0 of the run = the playing code.
    window = torch.cat([self._history, codes], dim=1) - self._half
    with torch.no_grad():
      self._frames = self.tokenizer.decode(window)[:, (HISTORY - 1 - PAST) * stride:]
    self._base["born"][:] = self._calls
    self._reference = None
    self._history = torch.cat([self._history[:, DECISION_CODES:], codes[:, :DECISION_CODES]], dim=1)
    # The goal stated now is in force from the next decision on (when the motion it belongs to starts).
    self._target = self._waiting
    self._waiting = torch.cat([goal[:, :4], torch.zeros_like(goal[:, :1])], dim=1)
    self._waiting_mask = (goal[:, 4] > 0.5).float()
    self.code_log.append(codes[:, :DECISION_CODES].cpu())
    self._decision += 1

  def update(self) -> None:
    env, term = self._env, self._term
    self._calls += 1
    restarted = (env.episode_length_buf <= 1).nonzero(as_tuple=False).flatten()
    if len(restarted) and self._calls > 1:
      self._start(restarted)
    if self._calls >= GRID_ORIGIN and (self._calls - GRID_ORIGIN) % self._period == 0:
      previous_mask = getattr(self, "_waiting_mask", None)
      self._decide()
      if previous_mask is not None:
        self._mask[:, 1] = previous_mask
    dt = env.step_dt
    step = self._slew * dt
    self._stream += torch.maximum(torch.minimum(self._target - self._stream, step), -step)
    goal = self._stream[:, :3]
    self._twists = torch.roll(self._twists, 1, dims=1)
    self._twists[:, 0] = goal
    error = term.own_twist() - self._twists[:, -1]
    self._lag = self._lag * math.exp(-dt / term.cfg.lag_time_constant) + dt * error

  def _dense(self):
    if self._reference is None:
      self._reference = s2.DenseReference(self._frames, self._env.step_dt, velocity_point=self._torso_com, lead=self.lead)
    return self._reference, (self._calls - self._base["born"]).clamp(min=0, max=self.horizon - 1)

  @property
  def goal(self) -> torch.Tensor:
    mask, stream = self._mask, self._stream
    hands = torch.zeros(len(stream), 10, device=stream.device)
    return torch.cat([stream[:, :3], stream[:, 3:4] * mask[:, 1:2], stream[:, 4:5] * mask[:, 2:3], hands, mask], dim=1)

  @property
  def goal_lag(self) -> torch.Tensor:
    return self._lag


def small_brain(model, schedule: torch.Tensor):
  """The small model: it reads the sentence embedding of the instruction (schedule: (envs, decisions, width))."""
  def brain(decision, history):
    return model.decide(schedule[:, min(decision, schedule.shape[1] - 1)], history)
  return brain


def goal_brain(head, state_goal):
  """The factorised System 2: `state_goal(decision)` gives the goal (envs, 5) - a vision-language model reading the
  camera and the instruction (vlm_goal), or the instruction's own goal (table_goal) - and the motion-code head
  (SmallS2 with condition="goal") writes the codes from that goal and its history."""
  def brain(decision, history):
    goal = state_goal(decision)
    codes, _ = head.decide(goal / head.goal_scale, history)
    return codes, goal
  return brain


def vlm_goal(model, env, wording: list, camera: str = "head_camera", vocabulary: torch.Tensor | None = None):
  """wording[env][decision]: the instruction's words read at each decision. vocabulary: see GoalVLA.decide."""
  def state_goal(decision):
    images = env.scene[camera].data.rgb.cpu().numpy()
    texts = [row[min(decision, len(row) - 1)] for row in wording]
    packed = model.pack(list(images), texts).to(env.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
      return model.decide(packed, vocabulary).float()
  return state_goal


def table_goal(goals: torch.Tensor):
  """goals: (envs, decisions, 5) the goal of the instruction read at each decision (no model: the head alone)."""
  def state_goal(decision):
    return goals[:, min(decision, goals.shape[1] - 1)]
  return state_goal


def vla_brain(model, env, wording: list, camera: str = "head_camera"):
  """The vision-language model: it reads the head camera and the instruction's words (wording[env][decision])."""
  def brain(decision, history):
    images = env.scene[camera].data.rgb.cpu().numpy()
    texts = [row[min(decision, len(row) - 1)] for row in wording]
    packed = model.pack(list(images), texts).to(history.device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
      codes, goal, _, _ = model.decide(packed, history)
    return codes, goal.float()
  return brain


def word_schedule(scripts: list, phrases: list, decisions: int, step_dt: float) -> list:
  """scripts[i]: [(instruction, seconds), ...]; phrases[i][j]: the wording of segment j. Returns, per environment,
  the words read at every decision: those of the instruction in force one decision later."""
  wanted = []
  for script, words in zip(scripts, phrases):
    edges, clock = [], float(GRID_ORIGIN)
    for _, seconds in script:
      clock += seconds / step_dt
      edges.append(clock)
    row = []
    for m in range(decisions):
      at = GRID_ORIGIN + 16 * (m + 1)
      index = next((j for j, edge in enumerate(edges) if at < edge - 0.5), len(edges) - 1)
      row.append(words[index])
    wanted.append(row)
  return wanted


def text_schedule(scripts: list, phrases: list, encoder, decisions: int, step_dt: float, device: str) -> torch.Tensor:
  """The same as sentence embeddings (envs, decisions, width), for the small model."""
  wanted = word_schedule(scripts, phrases, decisions, step_dt)
  flat = encoder([p for row in wanted for p in row])
  return flat.reshape(len(scripts), decisions, -1).to(device)
