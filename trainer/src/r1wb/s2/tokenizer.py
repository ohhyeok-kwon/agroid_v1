"""Motion tokens: a run of reference frames (r1wb.s2.frames) <-> a short sequence of discrete codes.

Encoder: 1D convolutions over time that shorten the run by `stride` (8 frames = 0.16 s per code). A code is `dims`
numbers, each rounded to one of `levels` values (finite scalar quantisation: nothing to look up, a code is a small
integer vector; with levels 7 the values are -3..3). Decoder: the mirror image. A run that stays on one frame gives
one repeated code and decodes to a constant run, which is what "stand still" has to be for System 1.

causal: a code is made from the frames up to the end of its own 8 frames and from nothing later, and a frame is
decoded from the codes up to its own. Without it the convolutions look about a code ahead, so the code just before a
change of motion already tells that the change is coming. A model that continues a recorded code history then learns
to wait for that tell-tale, which its own history never contains when it runs (the code it wrote earlier could not
know of an instruction that had not arrived): it never left a held crouch (gates 11-13 03:30).

  python -m r1wb.s2.tokenizer --courses <courses.npz> --out <tokenizer.pt> [--stride 8 --dims 8 --levels 7 --causal]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from r1wb.s2.frames import FRAME_DIM, HEIGHT, JOINTS, PITCH, ROLL, STEP

GROUPS = {"joints": JOINTS, "height": slice(HEIGHT, HEIGHT + 1), "tilt": slice(ROLL, PITCH + 1), "step": STEP}
WEIGHT = {"joints": 1.0, "height": 2.0, "tilt": 2.0, "step": 3.0}


class _Conv(nn.Module):
  """A convolution over time: centred, or (causal) one whose output at t reads the inputs up to t only."""

  def __init__(self, inputs: int, outputs: int, kernel: int, causal: bool):
    super().__init__()
    self.left = kernel - 1 if causal else 0
    self.conv = nn.Conv1d(inputs, outputs, kernel, padding=0 if causal else kernel // 2)

  def forward(self, x):
    return self.conv(F.pad(x, (self.left, 0)) if self.left else x)


class _Down(nn.Module):
  """Half the length. Causal: output i reads the inputs up to 2i + 1 (the end of the pair it stands for)."""

  def __init__(self, width: int, causal: bool):
    super().__init__()
    self.causal = causal
    self.conv = nn.Conv1d(width, width, 4, stride=2, padding=0 if causal else 1)

  def forward(self, x):
    return self.conv(F.pad(x, (2, 0)) if self.causal else x)


class _Up(nn.Module):
  """Twice the length. Causal: every input is repeated and a causal convolution follows, so output j reads the
  inputs up to j // 2 and a constant input gives a constant output (a held code decodes to frames that do not move
  at all, which a transposed convolution does not guarantee)."""

  def __init__(self, width: int, causal: bool):
    super().__init__()
    self.causal = causal
    self.conv = _Conv(width, width, 3, True) if causal else nn.ConvTranspose1d(width, width, 4, stride=2, padding=1)

  def forward(self, x):
    return self.conv(torch.repeat_interleave(x, 2, dim=2)) if self.causal else self.conv(x)


class _Block(nn.Module):
  def __init__(self, width: int, causal: bool = False):
    super().__init__()
    self.net = nn.Sequential(nn.SiLU(), _Conv(width, width, 3, causal), nn.SiLU(), nn.Conv1d(width, width, 1))

  def forward(self, x):
    return x + self.net(x)


WARM = 12                                    # codes of lead-in for a causal model (its reach back is about 8 codes)


class MotionTokenizer(nn.Module):
  def __init__(self, stride: int = 8, dims: int = 8, levels: int = 7, width: int = 256, mean=None, scale=None,
               causal: bool = False):
    super().__init__()
    assert stride in (2, 4, 8, 16) and levels % 2 == 1
    self.stride, self.dims, self.levels, self.width, self.causal = stride, dims, levels, width, causal
    halvings = int(np.log2(stride))
    encoder = [_Conv(FRAME_DIM, width, 5, causal), _Block(width, causal)]
    for _ in range(halvings):
      encoder += [_Down(width, causal), _Block(width, causal)]
    self.encoder = nn.Sequential(*encoder, nn.SiLU(), _Conv(width, dims, 3, causal))
    decoder = [_Conv(dims, width, 3, causal), _Block(width, causal)]
    for _ in range(halvings):
      decoder += [_Up(width, causal), _Block(width, causal)]
    self.decoder = nn.Sequential(*decoder, nn.SiLU(), _Conv(width, FRAME_DIM, 5, causal))
    self.register_buffer("mean", torch.zeros(FRAME_DIM) if mean is None else torch.as_tensor(mean, dtype=torch.float))
    self.register_buffer("scale", torch.ones(FRAME_DIM) if scale is None else torch.as_tensor(scale, dtype=torch.float))

  # -- codes ------------------------------------------------------------------------------------------------------

  def _latent(self, frames: torch.Tensor) -> torch.Tensor:
    return torch.tanh(self.encoder(((frames - self.mean) / self.scale).transpose(1, 2))) * (self.levels // 2)

  def encode(self, frames: torch.Tensor) -> torch.Tensor:
    """frames (batch, time, FRAME_DIM), time a multiple of the stride -> codes (batch, time / stride, dims), integers.
    A causal model reads what came before: the run is taken to have stood on its first frame until it begins."""
    if not self.causal:
      return torch.round(self._latent(frames)).transpose(1, 2).long()
    lead = frames[:, :1].clone()
    lead[:, :, STEP] = 0.0
    frames = torch.cat([lead.expand(-1, WARM * self.stride, -1), frames], dim=1)
    return torch.round(self._latent(frames)).transpose(1, 2).long()[:, WARM:]

  def decode(self, codes: torch.Tensor) -> torch.Tensor:
    """codes (batch, n, dims) -> frames (batch, n * stride, FRAME_DIM). A causal model reads the codes before: the
    first code is taken to have been held until the run begins (pass earlier codes for an exact continuation)."""
    if not self.causal:
      return self.decoder(codes.float().transpose(1, 2)).transpose(1, 2) * self.scale + self.mean
    codes = torch.cat([codes[:, :1].expand(-1, WARM, -1), codes], dim=1)
    out = self.decoder(codes.float().transpose(1, 2)).transpose(1, 2) * self.scale + self.mean
    return out[:, WARM * self.stride:]

  def forward(self, frames: torch.Tensor) -> torch.Tensor:
    latent = self._latent(frames)
    quantised = latent + (torch.round(latent) - latent).detach()             # straight-through rounding
    return self.decoder(quantised).transpose(1, 2) * self.scale + self.mean

  def roundtrip(self, frames: torch.Tensor) -> torch.Tensor:
    """Any length: the run is extended by its last frame (standing on it) to a multiple of the stride."""
    extra = (-frames.shape[1]) % self.stride
    if extra:
      tail = frames[:, -1:].clone()
      tail[:, :, STEP] = 0.0
      frames = torch.cat([frames, tail.expand(-1, extra, -1)], dim=1)
    with torch.no_grad():
      out = self.decode(self.encode(frames))
    return out[:, :out.shape[1] - extra] if extra else out

  # -- files ------------------------------------------------------------------------------------------------------

  def save(self, path: Path, **meta) -> None:
    torch.save(dict(state=self.state_dict(), config=dict(stride=self.stride, dims=self.dims, levels=self.levels,
                                                         width=self.width, causal=self.causal), meta=meta), path)

  @classmethod
  def load(cls, path: Path, device: str = "cpu") -> "MotionTokenizer":
    blob = torch.load(path, map_location=device, weights_only=False)
    model = cls(**blob["config"]).to(device)
    state = blob["state"]
    if not any(key.endswith(".conv.weight") for key in state):       # files written before the layers were wrapped
      state = {key if key in ("mean", "scale") else _wrapped(key): value for key, value in state.items()}
    model.load_state_dict(state)
    return model.eval()


def _wrapped(key: str) -> str:
  """Name of a weight of the plain-convolution layout in the wrapped layout (convolutions sit in `.conv`)."""
  parts = key.split(".")                                             # encoder.<i>.weight | encoder.<i>.net.<j>.weight
  if len(parts) == 3:
    return f"{parts[0]}.{parts[1]}.conv.{parts[2]}"
  return ".".join(parts[:4] + ["conv", parts[4]]) if parts[3] == "1" else key


def course_frames(courses: dict, table: torch.Tensor) -> torch.Tensor:
  """(envs, steps, FRAME_DIM) reference frames of recorded courses; a frame that is stayed on does not travel."""
  index = torch.as_tensor(courses["frame"].T.astype(np.int64), device=table.device)
  frames = table[index].clone()
  stays = torch.cat([index[:, 1:] == index[:, :-1], torch.ones_like(index[:, :1], dtype=torch.bool)], dim=1)
  frames[:, :, STEP] = frames[:, :, STEP] * (~stays).float().unsqueeze(2)
  return frames


def library_table(library: Path, device: str) -> torch.Tensor:
  """Reference frames of every library frame, without a simulator (same numbers as frames.library_frames)."""
  from types import SimpleNamespace

  from r1wb.s2.frames import library_frames
  data = np.load(library, allow_pickle=False)
  manifest = json.loads(str(data["manifest"]))
  anchor = manifest["body_names"].index("pelvis")
  start, length = torch.as_tensor(data["clip_start"]), torch.as_tensor(data["clip_length"])
  end = torch.repeat_interleave(start + length - 1, length).to(device)
  motion = SimpleNamespace(joint_pos=torch.as_tensor(data["joint_pos"], device=device),
                           body_pos_w=torch.as_tensor(data["body_pos_w"], device=device),
                           body_quat_w=torch.as_tensor(data["body_quat_w"], device=device))
  return library_frames(motion, anchor, end)


def errors(truth: torch.Tensor, output: torch.Tensor) -> dict:
  out = {}
  for name, part in GROUPS.items():
    gap = (output[..., part] - truth[..., part]).abs()
    out[name] = (float(gap.pow(2).mean().sqrt()), float(torch.quantile(gap.flatten()[::max(1, gap.numel() // 200000)], 0.99)),
                 float(gap.max()))
  travel = (output[..., STEP] - truth[..., STEP]).cumsum(dim=1)[:, 49::50]                 # drift per second
  out["travel_per_s"] = (float(travel[..., :2].norm(dim=-1).mean()), float(travel[..., 2].abs().mean()))
  return out


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--courses", type=Path, nargs="+", required=True)
  parser.add_argument("--library", type=Path, required=True)
  parser.add_argument("--out", type=Path, required=True)
  parser.add_argument("--stride", type=int, default=8)
  parser.add_argument("--dims", type=int, default=8)
  parser.add_argument("--levels", type=int, default=7)
  parser.add_argument("--width", type=int, default=256)
  parser.add_argument("--steps", type=int, default=4000)
  parser.add_argument("--batch", type=int, default=256)
  parser.add_argument("--window", type=int, default=96)
  parser.add_argument("--causal", action="store_true", help="codes and frames read nothing later than themselves")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  torch.manual_seed(0)
  table = library_table(args.library, args.device)
  parts, holds = [], []
  for path in args.courses:                                                 # every file gives nine tenths to train on
    courses = dict(np.load(path, allow_pickle=False))
    parts.append(course_frames(courses, table))                             # (envs, steps, 30)
    holds.append(courses["hold"].T)
  steps = min(part.shape[1] for part in parts)
  cut = [int(len(part) * 0.9) for part in parts]
  held_out = torch.cat([part[c:, :steps] for part, c in zip(parts, cut)])
  train = torch.cat([part[:c, :steps] for part, c in zip(parts, cut)])
  held = np.concatenate([hold[c:, :steps] for hold, c in zip(holds, cut)])
  del parts
  flat = train.reshape(-1, FRAME_DIM)
  mean, scale = flat.mean(0), flat.std(0).clamp_min(1.0e-4)
  model = MotionTokenizer(args.stride, args.dims, args.levels, args.width, mean, scale, args.causal).to(args.device)
  if args.causal:
    args.window += WARM * args.stride                                      # the lead-in is not scored
  optimiser = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-4)
  schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 6.0e-4, total_steps=args.steps, pct_start=0.1)
  weight = torch.ones(FRAME_DIM, device=args.device)
  for name, part in GROUPS.items():
    weight[part] = WEIGHT[name]
  began = time.time()
  for step in range(args.steps):
    rows = torch.randint(len(train), (args.batch,), device=args.device)
    begin = torch.randint(steps - args.window, (args.batch,), device=args.device)
    index = begin.unsqueeze(1) + torch.arange(args.window, device=args.device).unsqueeze(0)
    batch = train[rows.unsqueeze(1), index]
    output = model(batch)
    if args.causal:                                                          # the first codes had nothing before them
      output, batch = output[:, WARM * args.stride:], batch[:, WARM * args.stride:]
    gap = (output - batch) / scale
    loss = (F.smooth_l1_loss(gap, torch.zeros_like(gap), beta=0.05, reduction="none") * weight).mean()
    loss = loss + ((gap[:, 1:] - gap[:, :-1]).abs() * weight).mean()         # frame-to-frame change: no added jitter
    optimiser.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimiser.step()
    schedule.step()
    if step % 500 == 0 or step == args.steps - 1:
      print(f"step {step:5d} loss {float(loss):.4f}  {time.time() - began:.0f} s", flush=True)
  model.eval()
  length = (steps // args.stride) * args.stride
  with torch.no_grad():
    output = torch.cat([model.decode(model.encode(part[:, :length])) for part in held_out.split(64)])
    codes = torch.cat([model.encode(part[:, :length]) for part in held_out.split(64)])
  report = errors(held_out[:, :length], output)
  still = torch.as_tensor(held[:, :length], device=args.device)
  inside = still.clone()                                                     # held for the whole surrounding second
  for shift in range(1, 26):
    inside[:, shift:] &= still[:, :-shift]
    inside[:, :-shift] &= still[:, shift:]
  change = (output[:, 1:, JOINTS] - output[:, :-1, JOINTS]).abs().amax(dim=-1)
  report["held_frame_to_frame_max_rad"] = float(change[inside[:, 1:]].max()) if inside.any() else None
  used = len(torch.unique(codes.reshape(-1, args.dims), dim=0))
  bits = args.dims * float(np.log2(args.levels))
  meta = dict(courses=[path.name for path in args.courses], library=str(args.library.name), steps=args.steps, report=report,
              codes_per_second=50 / args.stride, bits_per_code=round(bits, 1), distinct_codes_held_out=used,
              causal=args.causal,
              parameters=sum(p.numel() for p in model.parameters()))
  model.save(args.out, **meta)
  print(json.dumps(meta, indent=1))


if __name__ == "__main__":
  main()
