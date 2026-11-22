"""Negative 30 body (coordinator 11-22 04:19, todo 2w): which robot body the negative runners compile.
NEG30_BODY = r1wb (default; the old panel body: r1.xml + O6 hand boxes 0.43 kg each, r1.xml root passive terms)
           | b1   (panel body since 11-22 04:14, judge/v4/body/r1_b1.xml: manufacturer mass 29.138 kg, stock fist, root
                   passive terms 0, joint damping 0.05 / frictionloss 0.2; sha256 43a34b33a68d7e1f, body_hash ba5b52a27728a2e7)
Import this module before sonic_g1 (sonic_g1 reads R1_XML at import): it sets R1_BODY / R1_XML for this process and the
spawned workers and refuses a conflicting R1_BODY / R1_XML (e.g. NEG30_BODY=r1wb with R1_BODY=b1). NEG30_BODY unset: R1_BODY=b1 with R1_XML = the b1 file (judge/v4 scripts) counts as b1; any other
R1_BODY / R1_XML is 'ext' (recorded, not checked, refused in neg30 directories). Only directories under neg30 get body.json.
  SFX      '' | '_b1'          suffix of run / report names
  MF_RUN   v4/run | v4/b1/run  manufacturer rows (pass table, stop phases);  MF_PUSH  v4/pushb | v4/b1/pushb (push witness)
  GATE     clips | clips/b1    motion-range gate directory (GATE/run rows, refs / gate tables, tags.tsv)
  claim(d)    writes / checks d/body.json; a directory without body.json holds r1wb rows (refused under b1)
  require(d)  for readers: refuses a directory of another body
  patch(S)    wraps sonic_g1.SimR1.__init__: every compiled model is checked against the body (total mass, root dof damping /
              frictionloss / armature, O6 hand bodies, b1 also body_hash v2); the last check is S.NEG30_BODY_CHECK
CLI (positive control):  NEG30_BODY=b1 python -I neg30_body.py check     |   python -I neg30_body.py sh  (BODY BSFX MF_RUN MF_PUSH GATE)"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

B1_XML = "/rl-agent/.agent/tmp/judge/v4/body/r1_b1.xml"
B1_SHA16 = "43a34b33a68d7e1f"
B1_MASS = 29.13788877
B1_SIG16 = "ba5b52a27728a2e7"
HANDS = ("left_o6_hand", "right_o6_hand")
N = "/rl-agent/.agent/tmp/neg30"
_RB, _RX = os.environ.get("R1_BODY"), os.environ.get("R1_XML")
if "NEG30_BODY" in os.environ:
  BODY = os.environ["NEG30_BODY"]
  if BODY not in ("r1wb", "b1"):
    raise SystemExit(f"neg30_body: NEG30_BODY={BODY} (r1wb | b1)")
elif _RB in (None, "r1wb"):
  BODY = "r1wb"
elif _RB == "b1" and _RX == B1_XML:
  BODY = "b1"            # callers that set the b1 body themselves (judge/v4 scripts): the same as NEG30_BODY=b1
else:
  BODY = "ext"           # a body the caller chose (R1_BODY / R1_XML): recorded, not checked; refused in neg30 directories
SFX = {"r1wb": "", "b1": "_b1"}.get(BODY, "_ext")
MF_RUN = "v4/run" if BODY == "r1wb" else f"v4/{BODY}/run"
MF_PUSH = "v4/pushb" if BODY == "r1wb" else f"v4/{BODY}/pushb"
GATE = "clips" if BODY == "r1wb" else f"clips/{BODY}"


def _sha(fn):
  return hashlib.sha256(Path(fn).read_bytes()).hexdigest()


def _setup():
  rb, rx = _RB, _RX
  if BODY == "ext":
    return None
  if BODY == "r1wb":
    if rb not in (None, "r1wb"):
      raise SystemExit(f"neg30_body: R1_BODY={rb} with NEG30_BODY=r1wb (conflict)")
    os.environ["R1_BODY"] = "r1wb"
    return None
  if rb not in (None, "b1") or rx not in (None, B1_XML):
    raise SystemExit(f"neg30_body: NEG30_BODY=b1 but R1_BODY={rb} R1_XML={rx}")
  sha = _sha(B1_XML)
  if not sha.startswith(B1_SHA16):
    raise SystemExit(f"neg30_body: {B1_XML} sha256 {sha[:16]} != {B1_SHA16}")
  if "sonic_g1" in sys.modules and getattr(sys.modules["sonic_g1"], "R1_XML", None) != B1_XML:
    raise SystemExit("neg30_body: sonic_g1 was imported before neg30_body (its R1_XML is not the b1 file)")
  os.environ["R1_BODY"], os.environ["R1_XML"] = "b1", B1_XML
  return sha


XML_SHA = _setup()
_SIG = {}


def check(m) -> dict:
  """The compiled model against the body: raises on a mismatch, returns the record."""
  import numpy as np
  names = {m.body(i).name for i in range(m.nbody)}
  root = [i for i in range(m.njnt) if m.jnt_type[i] == 0]
  d0 = int(m.jnt_dofadr[root[0]]) if root else 0
  rec = dict(body=BODY, mass=round(float(np.sum(m.body_mass)), 6), nbody=int(m.nbody), nv=int(m.nv),
             root_damping=[float(x) for x in m.dof_damping[d0:d0 + 6]], root_frictionloss=[float(x) for x in m.dof_frictionloss[d0:d0 + 6]],
             root_armature=[float(x) for x in m.dof_armature[d0:d0 + 6]], o6_hands=sum(h in names for h in HANDS))
  if BODY == "b1":
    if id(m) not in _SIG:
      if "/rl-agent/src" not in sys.path:
        sys.path.insert(0, "/rl-agent/src")
      import r1lab.body_hash as BH
      _SIG.clear()
      _SIG[id(m)] = BH.signature(m, range(1, m.nbody))["total"][:16]
    rec["sig16"] = _SIG[id(m)]
    bad = []
    if abs(rec["mass"] - B1_MASS) > 1e-4:
      bad.append(f"mass {rec['mass']}")
    if any(rec[k] != [0.0] * 6 for k in ("root_damping", "root_frictionloss", "root_armature")):
      bad.append("root passive terms not 0")
    if rec["o6_hands"]:
      bad.append("O6 hand bodies present")
    if rec["sig16"] != B1_SIG16:
      bad.append(f"body_hash {rec['sig16']}")
    if bad:
      raise RuntimeError("neg30_body: model is not b1: " + "; ".join(bad))
  elif BODY == "r1wb" and rec["o6_hands"] != 2:
    raise RuntimeError("neg30_body: r1wb model without the two O6 hand bodies")
  return rec


def patch(S):
  """Wrap S.SimR1.__init__ (sonic_g1): the tracker, manufacturer and clip runners all compile their model there."""
  if getattr(S.SimR1, "_neg30_body", False):
    return
  if BODY == "b1" and getattr(S, "R1_XML", None) != B1_XML:
    raise SystemExit(f"neg30_body: sonic_g1.R1_XML = {getattr(S, 'R1_XML', None)} (imported before neg30_body?)")
  orig = S.SimR1.__init__

  def init(self, *a, **k):
    orig(self, *a, **k)
    S.NEG30_BODY_CHECK = check(self.m)

  S.SimR1.__init__ = init
  S.SimR1._neg30_body = True


def _inside(d) -> bool:
  return str(Path(d).resolve()).startswith(N + "/")


def dir_body(d) -> str:
  p = Path(d) / "body.json"                # relative = to the working directory, as the runners write it
  return json.loads(p.read_text())["body"] if p.exists() else "r1wb"


def require(d):
  """Readers: a neg30 directory of another body is refused (directories outside neg30 are the caller's)."""
  if not _inside(d):
    return
  b = dir_body(d)
  if b != BODY:
    raise SystemExit(f"neg30_body: {d} holds body {b} rows, NEG30_BODY = {BODY}")


def claim(d):
  """Run directory under neg30 = one body; a directory with rows but no body.json is an r1wb directory. Directories outside
  neg30 (e.g. judge/v4 runs) are left as the caller made them."""
  p = Path(d)
  if not _inside(p):
    return
  if BODY == "ext":
    raise SystemExit(f"neg30_body: R1_BODY={_RB} R1_XML={_RX} is neither r1wb nor b1; not written into {p}")
  p.mkdir(parents=True, exist_ok=True)
  f = p / "body.json"
  if f.exists():
    require(p)
    return
  if BODY != "r1wb" and (p / "rows.jsonl").exists():
    raise SystemExit(f"neg30_body: {p} has rows and no body.json (old body r1wb); NEG30_BODY = {BODY}")
  f.write_text(json.dumps(dict(body=BODY, xml=B1_XML if BODY != "r1wb" else "r1wb.robot.get_spec (r1.xml + O6 hand boxes)",
                               xml_sha256=XML_SHA, mass_expected=B1_MASS if BODY != "r1wb" else None,
                               body_hash16=B1_SIG16 if BODY != "r1wb" else None), indent=1) + "\n")


def row_tag(S) -> dict:
  c = getattr(S, "NEG30_BODY_CHECK", None) or {}
  return dict(body=BODY, body_mass=c.get("mass"), body_sig16=c.get("sig16"),
              body_root0=all(c.get(k) == [0.0] * 6 for k in ("root_damping", "root_frictionloss", "root_armature")) if c else None)


if __name__ == "__main__":
  if sys.argv[1:] == ["sh"]:
    print(f"BODY={BODY} BSFX={SFX} MF_RUN={MF_RUN} MF_PUSH={MF_PUSH} GATE={GATE}")
  elif sys.argv[1:] == ["check"]:
    for p in ("/rl-agent/src", "/rl-agent/.agent/tmp/t1_sonic/work"):
      sys.path.insert(0, p)
    import sonic_g1 as S
    patch(S)
    out = {}
    for g in ("r1", "g1"):
      S.SimR1(g)
      out[g] = S.NEG30_BODY_CHECK
    print(json.dumps(dict(R1_BODY=os.environ.get("R1_BODY"), R1_XML=os.environ.get("R1_XML"), sonic_g1_R1_XML=S.R1_XML,
                          xml_sha16=(XML_SHA or "")[:16], check=out["r1"], same_g1=out["g1"] == out["r1"])))
  else:
    raise SystemExit(__doc__)
