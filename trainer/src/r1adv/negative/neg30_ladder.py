"""Negative 30: adversarial loop with automatic difficulty (task 3). Axes are raised one level at a time, in the order
delay -> sensor noise -> push -> friction, and only on an axis where the positive candidate's strict success exceeds 0.9;
a raised level stays only with solvability evidence (charter: >= 1 success of the manufacturer witness, evaluation only, on
the same specs); a level without it is stepped back and the axis frozen. Loop = main loop (open-loop m1c, bridge 0.1 s); the non-raised axes stay at the base (delay 0, noise D2 x 0.1,
standing pushes 12 N s, foot mu as built). n = 16 paired instances per level (seed tree: round -> 'ladder' -> axis -> i;
the level sets the magnitude only). Formal tests on every round (coordinator 08:4x): static symmetric reference + exact
front / back pushes (20 / 25 / 30 N s, push time 4.0 / 4.5 / 5.0 / 5.5 s), noise-free and D2 x 0.1.
Strict success: walking missions = no fall, steady forward ratio >= 0.7, no two-foot hop; standing pushes = no fall and no
two-foot hop; static4 = no fall and no two-foot hop.

  python -I neg30_ladder.py make  STATE.json OUT_SPECS.jsonl         (specs for the current levels + formal tests)
  python -I neg30_ladder.py eval  STATE.json RUN_DIR CANDIDATE OUT.md  (success shares, witness, raise / hold, new state)
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, "/rl-agent/src/r1adv/negative")
import neg30_specs as G  # noqa: E402

LEVELS = {"delay": [0, 10, 20, 30, 40, 50, 60], "noise": [0.1, 0.25, 0.5, 0.75, 1.0], "push": [12, 15, 20, 25, 30, 35, 40],
          "friction": [None, 0.6, 0.4, 0.3, 0.25, 0.2, 0.15]}
ORDER = ("delay", "noise", "push", "friction")
N = 16


def H(p, n):
  return G.H(p, n)


def spec(axis, x, i):
  root = H(H(G.ROUND, "neg30-R04-ladder"), axis)
  inst = H(root, f"i{i}")
  noise = x if axis == "noise" else 0.1
  delay = x if axis == "delay" else 0
  fam = "stand_push" if axis == "push" else "walk_stop"
  s = G.make(fam, f"ol_L01_d{int(delay):02d}", "A", i, noise=True)
  s.pop("spec_sha256")
  s["env"]["sense"] = {k: (v * noise if isinstance(v, float) else v) for k, v in G.D2.items()}
  s["env"]["sense"]["scale_of_D2"] = noise
  s["env"]["noise_seed"] = H(inst, "sense")
  s["env"]["body"]["action_delay_ms"] = float(delay)
  if axis == "push":
    dn, dr = G.DIRS[i % 4]
    s["env"]["hit"] = [dict(t_s=20.0, J_Ns=float(x), dir_rad=dr, dir_name=dn, z_off_m=0.1)]
  if axis == "friction" and x is not None:
    s["env"]["body"]["foot_mu"] = float(x)
  s.update(seed=inst, family=fam, cell=f"lad_{axis}_{x}", evalset="lad", i=i, ladder=dict(axis=axis, x=x))
  return G.finish(s)


def static4(noise, i):
  J = (20.0, 25.0, 30.0)[i % 3]
  dn, dr = (("front", 0.0), ("back", math.pi))[(i // 3) % 2]
  tp = (4.0, 4.5, 5.0, 5.5)[(i // 6) % 4]
  s = G.make("stand_static", "ol_L01_d00", "A", i, noise=noise > 0)
  s.pop("spec_sha256")
  if noise > 0:
    s["env"]["sense"] = {k: (v * noise if isinstance(v, float) else v) for k, v in G.D2.items()}
    s["env"]["sense"]["scale_of_D2"] = noise
  s["env"]["hit"] = [dict(t_s=tp, J_Ns=J, dir_rad=dr, dir_name=dn, z_off_m=0.1)]
  s["mission"]["steps"][0]["duration_s"] = 10.0
  s.update(family="static4", cell=f"static4_{'nf' if noise == 0 else 'lo'}", evalset="lad", i=i)
  return G.finish(s)


def make(state_path, out):
  st = json.loads(Path(state_path).read_text()) if Path(state_path).exists() else {"level": {a: 0 for a in ORDER}, "history": []}
  sp = []
  for a in ORDER:
    x = LEVELS[a][st["level"][a]]
    sp += [spec(a, x, i) for i in range(N)]
  sp += [static4(0.0, i) for i in range(24)] + [static4(0.1, i) for i in range(24)]
  Path(out).write_text("".join(json.dumps(s) + "\n" for s in sp))
  Path(state_path).write_text(json.dumps(st, indent=1))
  print(len(sp), st["level"])


def ok(r):
  if r.get("F") or r.get("hops", 0) > 0:
    return False
  if r["family"] == "walk_stop":
    return bool(r.get("G_walk"))
  return True


def evaluate(state_path, run, cand, out_md):
  st = json.loads(Path(state_path).read_text())
  rows = [json.loads(l) for l in (Path(run) / "rows.jsonl").read_text().splitlines() if l.strip()]
  rows = [r for r in rows if "error" not in r]
  L = [f"# 적대 고리 (자동 난이도) — 후보 {cand}", "", "| 축 | 수준 | 후보 성공 | 제조사(증인) | 다음 |", "|---|---|---|---|---|"]
  new = dict(st["level"])
  for a in ORDER:
    x = LEVELS[a][st["level"][a]]
    cell = f"lad_{a}_{x}"
    c = [ok(r) for r in rows if r["policy"] == cand and r.get("cell") == cell]
    w = [ok(r) for r in rows if r["policy"] == "mf" and r.get("cell") == cell]
    sc, sw = (sum(c) / len(c) if c else None), (sum(w) / len(w) if w else None)
    nxt = "유지"
    frozen = st.setdefault("frozen", {})
    if w and sum(w) == 0 and st["level"][a] > 0:              # charter: a level needs >= 1 witness success (solvability)
      new[a] = st["level"][a] - 1
      frozen[a] = st["level"][a]
      nxt = f"증인 0/{len(w)} → 풀 수 있다는 증거 없음: {LEVELS[a][new[a]]} 로 내리고 이 축은 멈춤"
    elif a in frozen:
      nxt = f"멈춤(위 수준 {LEVELS[a][frozen[a]]} 은 증인 없음)"
    elif sc is not None and sc > 0.9 and st["level"][a] + 1 < len(LEVELS[a]):
      new[a] = st["level"][a] + 1
      nxt = f"올림 → {LEVELS[a][new[a]]} (증인은 다음 판에서 확인)"
    L.append(f"| {a} | {x} | {'—' if sc is None else f'{sum(c)}/{len(c)}'} | {'—' if sw is None else f'{sum(w)}/{len(w)}'} | {nxt} |")
  for cell in ("static4_nf", "static4_lo"):
    c = [r for r in rows if r["policy"] == cand and r.get("cell") == cell]
    w = [r for r in rows if r["policy"] == "mf" and r.get("cell") == cell]
    if c:
      nfc = sum(bool(r["F"]) for r in c)
      nhc = sum(r.get("hops", 0) > 0 for r in c)
      wit = "— (제조사는 참조 입력이 없음; neg28 static4 기록: 20·25·30 N·s 넘어짐 0·0·4/16)" if not w else \
        f"넘어짐 {sum(bool(r['F']) for r in w)}/{len(w)}"
      L.append(f"| 정식: 대칭 정적 참조 + 정확한 앞·뒤 밂 ({cell}) | 20·25·30 | 넘어짐 {nfc}/{len(c)}, 두 발 뛰기 {nhc} | {wit} | 정식 시험 |")
  st["history"].append(dict(candidate=cand, level=st["level"], next=new))
  st["level"] = new
  Path(state_path).write_text(json.dumps(st, indent=1))
  Path(out_md).write_text("\n".join(L) + "\n")
  print("\n".join(L))


if __name__ == "__main__":
  if sys.argv[1] == "make":
    make(sys.argv[2], sys.argv[3])
  else:
    evaluate(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
