"""Reader of the positive's snapshot list (coordinator decision r1a, item 4). Read only: the positive writes it.

  .agent/adversarial/shared/snapshots/<ROUND>.jsonl, append-only, one JSON object per line:
  {"run_id", "server_checkpoint", "iteration", "sha256", "obs_spec", "model_class", "deterministic_eval": path|null,
   "created_kst"}

`entries` validates the fields and returns the lines in file order (a missing file is an empty list: the Hold ticks
then run with the existing population only; nobody waits for it). `verify` hashes the checkpoint where it lives (on
the server) and `runner_args` gives the runner's arguments for a verified line (--policy positive --checkpoint ...).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

FIELDS = ("run_id", "server_checkpoint", "iteration", "sha256", "obs_spec", "model_class", "deterministic_eval", "created_kst")
DEPLOY_OBS = ("deploy810", "deploy570")            # what runner --policy positive reads (r1lab deploy contract)


def entries(path: Path) -> list[dict]:
  if not path.exists():
    return []
  out = []
  for number, line in enumerate(path.read_text().splitlines(), 1):
    if not line.strip():
      continue
    entry = json.loads(line)
    missing = [k for k in FIELDS if k not in entry]
    if missing:
      raise ValueError(f"{path}:{number}: missing {missing}")
    out.append(dict(entry, line=number))
  return out


def verify(entry: dict) -> bool:
  digest = hashlib.sha256()
  with open(entry["server_checkpoint"], "rb") as f:
    for block in iter(lambda: f.read(1 << 20), b""):
      digest.update(block)
  return digest.hexdigest() == entry["sha256"]


def runner_args(entry: dict) -> list[str]:
  if not any(str(entry["obs_spec"]).startswith(kind) or kind in str(entry["obs_spec"]) for kind in DEPLOY_OBS):
    raise ValueError(f"{entry['run_id']}: obs_spec {entry['obs_spec']} is not a deploy contract the runner reads")
  return ["--policy", "positive", "--checkpoint", entry["server_checkpoint"]]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("list", type=Path)
  parser.add_argument("--verify", action="store_true", help="hash every checkpoint (run where the files are)")
  args = parser.parse_args()
  for entry in entries(args.list):
    ok = verify(entry) if args.verify else None
    print(json.dumps(dict(line=entry["line"], run_id=entry["run_id"], iteration=entry["iteration"], sha256_ok=ok,
                          args=runner_args(entry) if ok is not False else None)))


if __name__ == "__main__":
  main()
