"""The large System 2: a vision-language model that writes motion codes.

Input: one camera image, the instruction (any language the backbone reads), and the last HISTORY motion codes it
wrote. Output: the next AHEAD codes (each `dims` numbers with `levels` values) and the goal for System 1.

How the language model is used: after the chat prompt (image + instruction) the sequence continues with HISTORY
slots that carry the past codes (one learned vector per code) and with one query slot per number to write. The
language model runs once over the whole sequence; a linear head reads each query's final state as a distribution
over the `levels` values. One more query at the end is the state: the goal (regression) and the value (for policy
gradient training) are read from it. Nothing is generated token by token.

The same class with a small backbone is the student.
"""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from r1wb.s2.small import AHEAD, GOAL_DIM, GOAL_SCALE, HISTORY

GOAL_REST = (0.0, 0.0, 0.0, 0.731, 0.0)

SLOT = "<|fim_pad|>"          # an existing special token of the Qwen tokenizer, used only to reserve positions


class MotionVLA(nn.Module):
  def __init__(self, backbone: str | Path, dims: int = 8, levels: int = 7, dtype=torch.bfloat16,
               image_size: int = 224, keep_language_head: bool = False):
    super().__init__()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    self.dims, self.levels, self.image_size = dims, levels, image_size
    self.processor = AutoProcessor.from_pretrained(backbone)
    self.processor.tokenizer.padding_side = "left"
    self.lm = AutoModelForImageTextToText.from_pretrained(backbone, dtype=dtype)
    if not keep_language_head:
      self.lm.lm_head = nn.Identity()                      # no text is generated
    width = self.lm.config.text_config.hidden_size
    self.width = width
    self.slot = self.processor.tokenizer.convert_tokens_to_ids(SLOT)
    assert isinstance(self.slot, int) and self.slot >= 0
    self.queries = AHEAD * dims + 1
    self.past = nn.Embedding(dims * levels, width)
    self.past_place = nn.Parameter(torch.zeros(HISTORY, width))
    self.query = nn.Parameter(torch.randn(self.queries, width) * 0.02)
    self.code_head = nn.Linear(width, levels)
    self.goal_head = nn.Sequential(nn.Linear(width, 512), nn.SiLU(), nn.Linear(512, GOAL_DIM))
    self.value_head = nn.Sequential(nn.Linear(width, 512), nn.SiLU(), nn.Linear(512, 1))
    self.register_buffer("_offset", torch.arange(dims) * levels, persistent=False)
    self.register_buffer("goal_scale", torch.tensor(GOAL_SCALE), persistent=False)
    self._tail = SLOT * (HISTORY + self.queries)
    # The slots' vectors are written into the language model's input embeddings as they are looked up.
    self._filling = None
    self.lm.get_input_embeddings().register_forward_hook(self._fill)

  def _fill(self, module, args, output):
    if self._filling is None or args[0].shape != self._filling[0].shape:
      return None
    slots, vectors = self._filling
    return output.masked_scatter(slots.unsqueeze(-1), vectors.to(output.dtype))

  # -- input ------------------------------------------------------------------------------------------------------

  def prompt(self, instruction: str) -> str:
    message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    return self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + self._tail

  def pack(self, images, instructions: list[str]) -> dict:
    """images: list of HxWx3 uint8 arrays or PIL images (resized to image_size)."""
    from PIL import Image
    pictures = []
    for image in images:
      picture = image if isinstance(image, Image.Image) else Image.fromarray(image)
      if picture.size != (self.image_size, self.image_size):
        picture = picture.resize((self.image_size, self.image_size))
      pictures.append(picture)
    return self.processor(text=[self.prompt(text) for text in instructions], images=pictures, padding=True,
                          return_tensors="pt", return_mm_token_type_ids=True)

  # -- one pass ---------------------------------------------------------------------------------------------------

  def states(self, batch: dict, history: torch.Tensor) -> torch.Tensor:
    """history (batch, HISTORY, dims), values 0 .. levels-1 -> the queries' final states (batch, queries, width)."""
    ids = batch["input_ids"]
    slots = ids == self.slot
    past = self.past(history + self._offset).sum(dim=2) + self.past_place
    self._filling = (slots, torch.cat([past, self.query.unsqueeze(0).expand(len(ids), -1, -1)], dim=1))
    try:
      out = self.lm.model(**{key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values",
                                                        "image_grid_thw", "mm_token_type_ids") if key in batch})
    finally:
      self._filling = None
    hidden = out.last_hidden_state[slots].reshape(len(ids), HISTORY + self.queries, -1)
    return hidden[:, HISTORY:].float()

  def forward(self, batch: dict, history: torch.Tensor):
    """logits (batch, AHEAD, dims, levels), goal (batch, GOAL_DIM), value (batch,)."""
    hidden = self.states(batch, history)
    logits = self.code_head(hidden[:, :-1]).reshape(len(hidden), AHEAD, self.dims, self.levels)
    # The value reads the state without sending gradients into the writer (its errors must not move the policy).
    return logits, self.goal_head(hidden[:, -1]) * self.goal_scale, self.value_head(hidden[:, -1].detach()).squeeze(-1)

  def loss(self, batch: dict, history: torch.Tensor, target: torch.Tensor, goal: torch.Tensor):
    logits, predicted, _ = self(batch, history)
    codes = F.cross_entropy(logits.reshape(-1, self.levels), target.reshape(-1))
    return codes, F.mse_loss(predicted / self.goal_scale, goal / self.goal_scale), logits

  @torch.no_grad()
  def decide(self, batch: dict, history: torch.Tensor, temperature: float = 0.0):
    """codes (batch, AHEAD, dims), goal, and for sampled codes their log-probability (batch,) and the value."""
    logits, goal, value = self(batch, history)
    if temperature <= 0.0:
      return logits.argmax(dim=-1), goal, None, value
    choice = torch.distributions.Categorical(logits=logits / temperature)
    codes = choice.sample()
    return codes, goal, choice.log_prob(codes).sum(dim=(1, 2)), value

  def evaluate(self, batch: dict, history: torch.Tensor, codes: torch.Tensor, temperature: float = 1.0):
    """For policy-gradient training: log-probability and entropy of `codes`, and the value."""
    logits, _, value = self(batch, history)
    choice = torch.distributions.Categorical(logits=logits / temperature)
    return choice.log_prob(codes).sum(dim=(1, 2)), choice.entropy().sum(dim=(1, 2)), value


class GoalVLA(nn.Module):
  """The vision-language part alone: camera image + instruction -> the goal (twist, height, whether the height
  counts). The motion codes are written by a small head from the goal and its own history (SmallS2 with
  condition="goal"): the language model is asked only what it is good at, so it needs few examples, and the head
  learns motion from as many recorded courses as there are.

  After the chat prompt one query slot follows; the goal and the value are read from its final state."""

  def __init__(self, backbone: str | Path, dtype=torch.bfloat16, image_size: int = 224):
    super().__init__()
    from transformers import AutoModelForImageTextToText, AutoProcessor
    self.image_size = image_size
    self.processor = AutoProcessor.from_pretrained(backbone)
    self.processor.tokenizer.padding_side = "left"
    self.lm = AutoModelForImageTextToText.from_pretrained(backbone, dtype=dtype)
    self.lm.lm_head = nn.Identity()
    width = self.lm.config.text_config.hidden_size
    self.slot = self.processor.tokenizer.convert_tokens_to_ids(SLOT)
    self.query = nn.Parameter(torch.randn(1, width) * 0.02)
    self.goal_head = nn.Sequential(nn.Linear(width, 1024), nn.SiLU(), nn.Linear(1024, GOAL_DIM))
    self.value_head = nn.Sequential(nn.Linear(width, 512), nn.SiLU(), nn.Linear(512, 1))
    self.register_buffer("goal_scale", torch.tensor(GOAL_SCALE), persistent=False)
    self.register_buffer("goal_rest", torch.tensor(GOAL_REST), persistent=False)     # standing: what "no goal" reads
    self._filling = None
    self.lm.get_input_embeddings().register_forward_hook(self._fill)

  def _fill(self, module, args, output):
    if self._filling is None or args[0].shape != self._filling.shape:
      return None
    slots = self._filling
    return output.masked_scatter(slots.unsqueeze(-1), self.query.to(output.dtype).expand(int(slots.sum()), -1))

  def prompt(self, instruction: str) -> str:
    message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    return self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + SLOT

  def pack(self, images, instructions: list[str]) -> dict:
    from PIL import Image
    pictures = []
    for image in images:
      picture = image if isinstance(image, Image.Image) else Image.fromarray(image)
      if picture.size != (self.image_size, self.image_size):
        picture = picture.resize((self.image_size, self.image_size))
      pictures.append(picture)
    return self.processor(text=[self.prompt(text) for text in instructions], images=pictures, padding=True,
                          return_tensors="pt", return_mm_token_type_ids=True)

  def state(self, batch: dict) -> torch.Tensor:
    ids = batch["input_ids"]
    self._filling = ids == self.slot
    try:
      out = self.lm.model(**{key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values",
                                                        "image_grid_thw", "mm_token_type_ids") if key in batch})
      return out.last_hidden_state[self._filling].float()
    finally:
      self._filling = None

  def forward(self, batch: dict):
    """raw (batch, GOAL_DIM): twist and height as (goal - GOAL_REST) / GOAL_SCALE, then the logit of "the height
    counts"; value (batch,)."""
    hidden = self.state(batch)
    return self.goal_head(hidden), self.value_head(hidden.detach()).squeeze(-1)

  def loss(self, batch: dict, goal: torch.Tensor) -> torch.Tensor:
    raw, _ = self(batch)
    target = (goal - self.goal_rest) / self.goal_scale
    return (F.smooth_l1_loss(raw[:, :4], target[:, :4], beta=0.05)
            + 0.2 * F.binary_cross_entropy_with_logits(raw[:, 4], goal[:, 4]))

  @torch.no_grad()
  def decide(self, batch: dict, vocabulary: torch.Tensor | None = None) -> torch.Tensor:
    """The goal in physical units (twist, height, 1 or 0 for "the height counts").
    vocabulary (entries, GOAL_DIM): the goal is read as the nearest of these (the model's output is then a choice
    among the instructions' goals: the motion-code head has only been told those, and a value between two of them
    makes it write motion that is in no recording; gates 11-13 06:25)."""
    raw, _ = self(batch)
    goal = torch.cat([raw[:, :4] * self.goal_scale[:4] + self.goal_rest[:4], (raw[:, 4:5] > 0.0).float()], dim=1)
    if vocabulary is None:
      return goal
    entries = vocabulary.to(goal)
    apart = (((goal[:, None, :4] - entries[None, :, :4]) / self.goal_scale[:4]) ** 2).sum(dim=-1)
    apart = apart + 1.0e3 * (goal[:, None, 4] - entries[None, :, 4]).abs()          # the height flag is not blended
    return entries[apart.argmin(dim=1)]


MAP_SLOTS = 4                 # tokens that carry the map the robot has built (MissionVLA)


class MissionVLA(GoalVLA):
  """The goal model with a memory: camera image + mission ("map every room of this house") + the map built so far
  -> which instruction to carry out next.

  A mission lasts minutes, far longer than what one image shows, so what the robot has already seen is kept outside
  the model as a map (four egocentric pictures: seen floor and where views ended, near and far, as the exploration
  proxy draws them) plus a short vector (ranges ahead, the instruction in force, time spent pushing, time used). A
  small encoder turns them into MAP_SLOTS input tokens that follow the chat prompt; then comes the query slot of
  GoalVLA. From the query's final state: the choice among `choices` instructions (a categorical head: the right next
  move is often one of several, which a regression would average away), the goal as before, and the value (for
  policy-gradient training in the loop)."""

  def __init__(self, backbone: str | Path, choices: int, vector: int, dtype=torch.bfloat16, image_size: int = 224):
    super().__init__(backbone, dtype=dtype, image_size=image_size)
    width = self.lm.config.text_config.hidden_size
    self.choices = choices
    self.eye = nn.Sequential(nn.Conv2d(4, 24, 5, 2, 2), nn.ELU(), nn.Conv2d(24, 48, 3, 2, 1), nn.ELU(),
                             nn.Conv2d(48, 64, 3, 2, 1), nn.ELU(), nn.Flatten(), nn.Linear(64 * 8 * 8, 256), nn.ELU())
    self.rest = nn.Sequential(nn.Linear(vector, 64), nn.ELU())
    self.to_tokens = nn.Linear(320, MAP_SLOTS * width)
    self.choice_head = nn.Sequential(nn.Linear(width, 1024), nn.SiLU(), nn.Linear(1024, choices))
    self._width = width

  def prompt(self, instruction: str) -> str:
    message = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    return self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + SLOT * (MAP_SLOTS + 1)

  def _fill(self, module, args, output):
    if self._filling is None or args[0].shape != self._filling[0].shape:
      return None
    slots, vectors = self._filling
    return output.masked_scatter(slots.unsqueeze(-1), vectors.to(output.dtype))

  def state(self, batch: dict, maps: torch.Tensor | None = None, vector: torch.Tensor | None = None) -> torch.Tensor:
    """maps (batch, 4, 64, 64) in 0..1, vector (batch, vector)."""
    ids = batch["input_ids"]
    slots = ids == self.slot
    memory = self.to_tokens(torch.cat([self.eye(maps.float()), self.rest(vector.float())], dim=1))
    memory = memory.reshape(len(ids), MAP_SLOTS, self._width)
    filled = torch.cat([memory, self.query.to(memory.dtype).unsqueeze(0).expand(len(ids), -1, -1)], dim=1)
    self._filling = (slots, filled.reshape(-1, self._width))
    try:
      out = self.lm.model(**{key: batch[key] for key in ("input_ids", "attention_mask", "pixel_values",
                                                        "image_grid_thw", "mm_token_type_ids") if key in batch})
      return out.last_hidden_state[slots].reshape(len(ids), MAP_SLOTS + 1, -1)[:, -1].float()
    finally:
      self._filling = None

  def forward(self, batch: dict, maps: torch.Tensor, vector: torch.Tensor):
    """logits over the instructions (batch, choices), value (batch,)."""
    hidden = self.state(batch, maps, vector)
    return self.choice_head(hidden), self.value_head(hidden.detach()).squeeze(-1)
