"""A fine-tuned vision-language checkpoint as its difference to the backbone it started from.

The link between the GPU server and the workstation carries about 1 MB/s, and a fully fine-tuned 2B model is 4.3 GB
of weights that differ from the public backbone (already on both machines) in their last bits only. The bit patterns
of the fine-tuned weights are XORed with the backbone's and compressed: exact, and a fraction of the size.

  python -m r1wb.s2.delta pack   --backbone <folder> --checkpoint <goal_last.pt> --out <goal_last.delta>
  python -m r1wb.s2.delta unpack --backbone <folder> --delta <goal_last.delta> --out <goal_last.pt>
"""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import time

import torch
import zstandard


def backbone_state(folder: Path) -> tuple[dict, str]:
  """The backbone's weights as the checkpoints name them ("lm." + name), in bf16, and the hash of its weight file."""
  from safetensors.torch import load_file
  files = sorted(folder.glob("*.safetensors"))
  digest = hashlib.sha256()
  state = {}
  for file in files:
    with open(file, "rb") as handle:
      for block in iter(lambda: handle.read(1 << 24), b""):
        digest.update(block)
    for name, value in load_file(str(file)).items():
      state["lm." + name] = value.to(torch.bfloat16) if value.is_floating_point() else value
  return state, digest.hexdigest()


def pack(backbone: Path, checkpoint: Path, out: Path) -> None:
  began = time.time()
  base, digest = backbone_state(backbone)
  blob = torch.load(checkpoint, map_location="cpu", weights_only=False)
  changed, plain, same = {}, {}, 0
  for name, value in blob["state"].items():
    origin = base.get(name)
    if origin is None or origin.shape != value.shape or origin.dtype != value.dtype or value.dtype != torch.bfloat16:
      plain[name] = value
      continue
    difference = value.contiguous().view(torch.int16) ^ origin.contiguous().view(torch.int16)
    if bool((difference == 0).all()):
      same += 1
      changed[name] = None                                   # present in the checkpoint, identical to the backbone
    else:
      changed[name] = difference
  buffer = io.BytesIO()
  torch.save(dict(changed=changed, plain=plain, meta=blob.get("meta"), backbone_sha256=digest), buffer)
  raw = buffer.getbuffer()
  packed = zstandard.ZstdCompressor(level=9, threads=-1).compress(raw)
  out.write_bytes(packed)
  print(f"{checkpoint.name}: {len(blob['state'])} weights ({len(plain)} not in the backbone, {same} unchanged) "
        f"{checkpoint.stat().st_size / 2**30:.2f} GB -> {len(packed) / 2**30:.2f} GB in {time.time() - began:.0f} s")


def unpack(backbone: Path, delta: Path, out: Path) -> None:
  began = time.time()
  base, digest = backbone_state(backbone)
  raw = zstandard.ZstdDecompressor().decompress(delta.read_bytes())
  blob = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
  if blob["backbone_sha256"] != digest:
    raise SystemExit(f"the backbone in {backbone} is not the one the difference was taken against")
  state = dict(blob["plain"])
  for name, difference in blob["changed"].items():
    origin = base[name].contiguous()
    state[name] = origin if difference is None else (origin.view(torch.int16) ^ difference).view(torch.bfloat16)
  out.parent.mkdir(parents=True, exist_ok=True)
  torch.save(dict(state=state, meta=blob["meta"]), out)
  print(f"{out}: {len(state)} weights restored in {time.time() - began:.0f} s")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("what", choices=["pack", "unpack"])
  parser.add_argument("--backbone", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--delta", type=Path)
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  if args.what == "pack":
    pack(args.backbone, args.checkpoint, args.out)
  else:
    unpack(args.backbone, args.delta, args.out)
