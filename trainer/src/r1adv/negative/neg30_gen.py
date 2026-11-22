"""Negative 30 (coordinator 23:2x): the reference generator ('ruler') the official v4 tables drive the tracker with.
Chosen by the environment variable NEG30_GEN; default v43s since 23:4x (coordinator: Operation Center on v4.3s since 23:35),
v4 stays selectable (NEG30_GEN=v4, exactly as before):
  v4   = P34 calm4_causal.py   (reference stop 0.69 s, mean of the three stop phases)
  v43s = P38 calm43s_causal.py (v4.1 reachable steps + v4.2 braking last step + T5.3 start; adopted 11-21 23:1x;
         P38: reference stop 0.42 s, start t50 0.40 s)
Both run with the same CENTER parameters. A run directory that receives reference-driven rows gets ruler.json (generator,
file, sha256); a directory without one holds v4 rows (written before 23:2x). The runner refuses to add rows of another ruler,
or of a changed generator file, to a directory; tables leave out tracker rows of another ruler. The manufacturer reads the
command, not the reference: its rows serve every ruler. Spec files are shared by the rulers (their loop.planner label is not
read by the runner), so two rulers' runs differ only in the reference."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

P34 = "/rl-agent/.agent/tmp/p34"
N = "/rl-agent/.agent/tmp/neg30"
CENTER = dict(fps=50.0, cadence=3.0, width=0.20, swing_h=0.05, step_len_max=0.25)
FILES = {"v4": f"{P34}/calm4_causal.py", "v43s": "/rl-agent/.agent/tmp/p38/calm43s_causal.py"}
GEN = os.environ.get("NEG30_GEN") or "v43s"
if GEN not in FILES:
  raise SystemExit(f"NEG30_GEN={GEN!r}: unknown reference generator (known: {', '.join(FILES)})")
# names (ckpt_eval.sh / push_panel.sh): v4 keeps its names; another ruler G appends _G (v4/ck/<TAG>_<sha12>_G/, v4/pp/..._G/,
# report/ck_<TAG>_<sha12>_G*.md, report/push_panel_<TAG>_<sha12>_G.md); ruler.json in each directory is the record
# reference-only charter results on the n35 eval flows (tracker-independent; v4: the reference part of R8@400's ref_check)
REFCHECK = f"{N}/v4/flows_refcheck_R8s400.jsonl" if GEN == "v4" else f"{N}/v4/flows_refcheck_ref_{GEN}.jsonl"


def sha12(gen=None):
  return hashlib.sha256(Path(FILES[gen or GEN]).read_bytes()).hexdigest()[:12]


def ident(gen=None):
  gen = gen or GEN
  return dict(gen=gen, file=FILES[gen], sha256_12=sha12(gen))


def label(gen=None):
  i = ident(gen)
  return f"참조 생성기 {i['gen']} ({Path(i['file']).name}, sha256 {i['sha256_12']})"


def module(gen=None):
  gen = gen or GEN
  if P34 not in sys.path:
    sys.path.insert(0, P34)                      # calm2_synth, on which both generators build
  if gen == "v4":
    import calm4_causal                          # as before 23:2x
    return calm4_causal
  name = Path(FILES[gen]).stem
  if name not in sys.modules:                    # loaded from its file: nothing else in its folder can shadow a module
    import importlib.util
    sp = importlib.util.spec_from_file_location(name, FILES[gen])
    m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m
    sp.loader.exec_module(m)
  return sys.modules[name]


def stream(gen=None):
  return module(gen).Stream(**CENTER)


def dir_gen(d):
  f = Path(d) / "ruler.json"
  return json.loads(f.read_text())["gen"] if f.exists() else "v4"


def claim(out):
  """Before reference-driven rows go into OUT: refuse a directory of another ruler, else record this ruler there."""
  out, cur = Path(out), ident()
  f = out / "ruler.json"
  if f.exists():
    prev = json.loads(f.read_text())
    if (prev.get("gen"), prev.get("sha256_12")) != (cur["gen"], cur["sha256_12"]):
      raise SystemExit(f"{out}: holds rows of ruler {prev.get('gen')} ({prev.get('sha256_12')}); refusing {cur['gen']} "
                       f"({cur['sha256_12']}) rows - use another directory")
    return
  if cur["gen"] != "v4" and (out / "rows.jsonl").exists():
    raise SystemExit(f"{out}: holds v4 rows (written before ruler selection); refusing {cur['gen']} rows")
  out.mkdir(parents=True, exist_ok=True)
  f.write_text(json.dumps(dict(cur, written=time.strftime("%Y-%m-%d %H:%M:%S")), indent=1) + "\n")
