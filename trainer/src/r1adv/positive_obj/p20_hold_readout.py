"""Frozen readout of the positive 20 fist hold test (rounds/R04/positive/r11_hold/rules.md, D2''(fist)).

  python -m r1adv.positive_obj.p20_hold_readout metrics <panel.npz> <out_metrics.npz>
  python -m r1adv.positive_obj.p20_hold_readout decide <metrics_dir> <engine> <out.json> [sel_seed conf_seed]
  python -m r1adv.positive_obj.p20_hold_readout cross <metrics_dir> <out.json>
metrics_dir holds <arm>_<engine>_s<seed>_<panel>.npz written by 'metrics'.

Windows: hold W = [1.2, 11.2) s (support removed at 1.2 s); start window [1.1, 1.2); slip and rule K on [2.2, 11.2]
(9 s, as the ladders' 6.5-15.5); onset slip [1.2, 2.2). held = six physical conditions (rules 5). Rule K code = copy of
negative 17 r11_khold.py d89a3576 (_bands, H_lo 1.514, H_hi 1.764 m/s^2) via p20_readout.
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

from r1adv.positive_obj import p20_readout as R

DT, NSTEP, NCTRL = 0.005, 2240, 560
T200 = (np.arange(NSTEP) + 1) * DT
T50 = (np.arange(NCTRL) + 1) * 0.02
T_REL, T_END, T_SLIP0 = 1.2, 11.2, 2.2
C_REL, C_SLIP0, C_END = 59, 109, 559                      # 50 Hz indices of t = 1.2, 2.2, 11.2
K_RELM1 = 239                                              # 200 Hz index of the last supported step (t = 1.2)
G = 9.81
TC_BINS = (0.010, 0.020, 0.030, 0.040)
PANELS = ("py3", "py4", "el3", "el4")
PANEL_COST_ORDER = {"py3": 0, "py4": 1, "el3": 2, "el4": 3}
MAIN_ARMS, SIDE_ARMS = ("fist", "fistfit"), ("fistpads",)
VERDICT, TASK, CREEP = 20, 21, 22
BASE = ("py3", 1.0, 1)
N_MIN, START_MIN, PASS, UNDEC = 30, 0.5, 0.70, 0.50
SLIP_MAX_MM, Q_MAX, ELLIPTIC_PREMIUM = 5.0, 1.0, 0.10
# fist_checks A / A' reading table (H1V shapes: 0 box, 1 upright cylinder, 2 upright capsule)
A2_CAPSULE_COUNT = {("fist", 0)}                           # capsule collider vs box: C and MJWarp the same count (<= 2, 1 at b5)
A2_POINT = {("fist", 1), ("fist", 2)}                      # crossed axes: point contact by geometry
A2_FAIL = {("fistfit", 2)}                                 # A': fistfit x upright capsule, MJWarp 1 contact at b1 (C 2)
M_START_MIN = 2.0


def mask200(a, b):
  return (T200 >= a - 1e-9) & (T200 < b - 1e-9)


def k_world(obj_v):
  m = mask200(T_SLIP0, T_END + 1e-6)
  a = np.diff(obj_v[m].astype(float), axis=0) / DT
  lo, hi = R._bands(a, 1.0 / DT, [(0.5, 8.0 - 1e-9), (8.0, 100.0)])
  return lo, hi, max(lo / R.H_LO, hi / R.H_HI)


def rel_disp(obj_p, hand_p, c0, c1):
  o0, R0 = R.pair_frame(hand_p, c0)
  o1, R1 = R.pair_frame(hand_p, c1)
  p0 = np.einsum("nji,nj->ni", R0, obj_p[:, c0].astype(np.float64) - o0)
  p1 = np.einsum("nji,nj->ni", R1, obj_p[:, c1].astype(np.float64) - o1)
  return np.linalg.norm(p1 - p0, axis=1) * 1000.0


def metrics(path: str) -> dict:
  z = dict(np.load(path))
  names = [str(s) for s in z["param_names"]]
  prm = z["params"]
  col = {n: prm[:, names.index(n)] for n in names}
  N = len(prm)
  obj_p, hand_p = z["obj_p"], z["hand_p"]
  ncon, fn, dmin = z["ncon"], z["fn_hand"].astype(np.float64), z["dist_min"]
  W = mask200(T_REL, T_END + 1e-6)
  slip = rel_disp(obj_p, hand_p, C_SLIP0, C_END)
  slip_onset = rel_disp(obj_p, hand_p, C_REL, C_SLIP0)
  slip_total = rel_disp(obj_p, hand_p, C_REL, C_END)
  mg = col["mass"] * G
  M = col["mu"] * fn[:, W].mean(1).sum(1) / mg
  M_start = col["mu"] * fn[:, mask200(T_REL - 0.1, T_REL)].mean(1).sum(1) / mg
  c1 = (ncon[:, W, 3] == 0).all(1) & (ncon[:, W, 4] == 0).all(1)
  c2 = obj_p[:, C_END, 2] >= obj_p[:, C_REL, 2] - 0.05
  c3 = slip_total <= 50.0
  both = (ncon[:, W, 1] > 0) & (ncon[:, W, 2] > 0)
  c4 = both.mean(1) >= 0.95
  body_frac = (ncon[:, W, 5] > 0).mean(1)
  c5 = body_frac <= 0.05
  pen = np.maximum(-dmin.min(1).min(1), 0.0)
  c6 = pen <= 0.005
  held = c1 & c2 & c3 & c4 & c5 & c6
  S = mask200(T_REL - 0.1, T_REL)
  start_both = ((ncon[:, S, 1] > 0) & (ncon[:, S, 2] > 0)).all(1)
  start_body = (ncon[:, S, 5] > 0).any(1)
  v_rel = np.linalg.norm(z["obj_v"][:, K_RELM1].astype(np.float64), axis=1)
  start_ok = start_both & ~start_body & (v_rel < 0.05)
  g = z["init_gate"]
  gate_ok = (g[:, 0] <= 0.002) & (g[:, 1] <= 0.001) & (g[:, 2] <= 0.001)
  obj_v_all = z["obj_v"]
  k = np.array([k_world(obj_v_all[i]) for i in range(N)])
  hand_con = ncon[:, :, 1].astype(np.int64) + ncon[:, :, 2]
  wrist_share = np.where(hand_con[:, W].sum(1) > 0, z["ncap"][:, W].sum((1, 2)) / np.maximum(hand_con[:, W].sum(1), 1), np.nan)
  tau = z["tau_ratio"]
  out = dict(params=prm, param_names=z["param_names"], slip=slip, slip_onset=slip_onset, slip_total=slip_total, M=M, M_start=M_start,
             c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, held=held, pen=pen, body_frac=body_frac, start_ok=start_ok,
             start_both=start_both, start_body=start_body, v_release=v_rel, gate_ok=gate_ok, rms_lo=k[:, 0], rms_hi=k[:, 1], q=k[:, 2],
             wrist_share=wrist_share, tau_peak=tau[:, C_REL:].max(-1).max(-1), ncon_max=ncon[..., 0].max(1),
             hand_con_max=hand_con.max(1), nefc_max=z["nefc_max"], drop=obj_p[:, C_REL, 2] - obj_p[:, C_END, 2])
  out["meta"] = str(z["meta"])
  return out


def _load(mdir, engine):
  out = {}
  for f in sorted(glob.glob(os.path.join(mdir, f"*_{engine}_s*_*.npz"))):
    arm, _, seed, panel = os.path.basename(f)[:-4].rsplit("_", 3)
    out[(arm, int(seed[1:]), panel)] = dict(np.load(f, allow_pickle=True))
  return out


def _col(mt, name):
  names = [str(n) for n in mt["param_names"]]
  return mt["params"][:, names.index(name)]


def _tcbin(tc):
  return np.clip(np.digitize(tc, TC_BINS[1:-1]), 0, 2)


def _settings():
  return [(p, imp, b) for p in PANELS for imp in (1.0, 10.0) for b in (0, 1, 2)]


def _cost_key(s):
  p, imp, b = s
  return (PANEL_COST_ORDER[p], imp, {1: 0, 0: 1, 2: 2}[b])


def cell(mt, setting, stratum, need_start=True):
  _, imp, b = setting
  sel = (_col(mt, "stratum") == stratum) & (_col(mt, "ik_ok") > 0.5) & mt["gate_ok"] & (_col(mt, "impratio") == imp) \
      & (_tcbin(_col(mt, "tc")) == b)
  if stratum != TASK:
    sel &= _col(mt, "forearm_clear") > 0.5
  base = sel.copy()
  if need_start:
    sel &= mt["start_ok"]
  return sel, base


def summary_at(mt, setting, stratum) -> dict:
  sel, base = cell(mt, setting, stratum)
  n = int(sel.sum())
  held = mt["held"][sel]
  hs = sel & mt["held"]
  good = mt["held"][sel] & (mt["slip"][sel] < SLIP_MAX_MM) & (mt["q"][sel] <= Q_MAX)
  return dict(n=n, n_base=int(base.sum()), start_rate=float(mt["start_ok"][base].mean()) if base.any() else float("nan"),
              held=float(held.mean()) if n else float("nan"), good=float(good.mean()) if n else float("nan"),
              slip_med=float(np.median(mt["slip"][hs])) if hs.any() else float("nan"),
              q_med=float(np.median(mt["q"][hs])) if hs.any() else float("nan"),
              q_le1=float((mt["q"][hs] <= Q_MAX).mean()) if hs.any() else float("nan"),
              M_med=float(np.median(mt["M"][sel])) if n else float("nan"),
              M_start_med=float(np.median(mt["M_start"][sel])) if n else float("nan"),
              fail_M_start_lt1=float((mt["M_start"][sel & ~mt["held"]] < 1).mean()) if (sel & ~mt["held"]).any() else float("nan"),
              wrist_share=float(np.nanmean(mt["wrist_share"][sel])) if n else float("nan"))


def select_setting(Ms, arm, seed) -> tuple:
  """rules 8.2: 'good hold' rate (held and slip < 5 mm and K q <= 1) on the selection seed (verdict stratum, n >= 30); best
  pyramidal unless the best elliptic beats it by >= 10 pp; ties -> cheaper / nearer the base (panel py3 < py4 < el3 < el4,
  impratio 1 < 10, tc bin 1 < 0 < 2). Panels that overflowed are absent (not written / not loaded)."""
  table = {}
  for s in _settings():
    mt = Ms.get((arm, seed, s[0]))
    if mt is None:
      continue
    if json.loads(str(mt["meta"])).get("overflow"):
      continue
    r = summary_at(mt, s, VERDICT)
    if r["n"] >= N_MIN:
      table[s] = r["good"]
  if not table:
    return None, {}
  best = {}
  for cone in ("py", "el"):
    cand = [s for s in table if s[0].startswith(cone)]
    if cand:
      best[cone] = sorted(cand, key=lambda s: (-table[s], _cost_key(s)))[0]
  if "py" in best and "el" in best:
    choice = best["el"] if table[best["el"]] >= table[best["py"]] + ELLIPTIC_PREMIUM - 1e-12 else best["py"]
  else:
    choice = best.get("py", best.get("el"))
  return choice, {f"{k[0]}|imp{int(k[1])}|tc{k[2]}": round(v, 4) for k, v in table.items()}


def label(r) -> str:
  if r["n"] < N_MIN:
    return "미결(셈 수)"
  if not r["start_rate"] >= START_MIN:
    return "미결(시작 못 섬)"
  if r["held"] >= PASS:
    return "통과" if (r["slip_med"] < SLIP_MAX_MM and r["q_med"] <= Q_MAX) else "미결(유지되나 미끄럼·떨림)"
  if r["held"] >= UNDEC:
    return "미결"
  return "실패"


def decide(mdir, engine, sel_seed=31, conf_seed=32) -> dict:
  Ms = {k: v for k, v in _load(mdir, engine).items()}
  out = dict(engine=engine, sel_seed=sel_seed, conf_seed=conf_seed, arms={})
  for arm in MAIN_ARMS + SIDE_ARMS:
    S, table = select_setting(Ms, arm, sel_seed)
    res = dict(selection_table=table, setting=None if S is None else dict(panel=S[0], impratio=S[1], tc_bin=S[2]))
    if S is not None and (arm, conf_seed, S[0]) in Ms:
      mt = Ms[(arm, conf_seed, S[0])]
      r = summary_at(mt, S, VERDICT)
      res["verdict"] = r
      res["label"] = "미결(넘침)" if json.loads(str(mt["meta"])).get("overflow") else label(r)
      if res["label"] != "통과":                            # rules 8.3: failure reading
        res["failure_reading"] = ("끼움 힘 부족(예산 전제 깨짐)" if not r["M_start_med"] >= M_START_MIN
                                  else "엔진·접촉 설정 한계(주먹 몸, 판정 범위 안)")
      res["selection_seed_at_S"] = summary_at(Ms[(arm, sel_seed, S[0])], S, VERDICT)
      shp = {}
      sel, _ = cell(mt, S, VERDICT)
      for k, name in enumerate(("box", "cylinder", "capsule")):
        m2 = sel & (_col(mt, "shape") == k)
        h = float(mt["held"][m2].mean()) if m2.any() else float("nan")
        reading = None
        if m2.sum() >= 10 and h < UNDEC:
          reading = ("기하상 점 접촉" if (arm, k) in A2_POINT else "캡슐 충돌체 접촉 수(두 엔진 같음)" if (arm, k) in A2_CAPSULE_COUNT
                     else "MJWarp 접촉 수 한계" if (arm, k) in A2_FAIL else "접촉 설정(마찰·연성) 한계")
        shp[name] = dict(n=int(m2.sum()), held=h, reading=reading)
      res["by_shape"] = shp
      task = {}
      for imp_all in (True,):
        selT = (_col(mt, "stratum") == TASK) & (_col(mt, "ik_ok") > 0.5) & mt["gate_ok"] & mt["start_ok"]
        selT &= (_col(mt, "impratio") == S[1]) & (_tcbin(_col(mt, "tc")) == S[2])
        bud = _col(mt, "budget")
        for nm, bm in (("all", np.ones_like(bud, bool)), ("budget<1", bud < 1), ("1<=budget<2", (bud >= 1) & (bud < 2)), ("budget>=2", bud >= 2)):
          mm = selT & bm
          task[nm] = dict(n=int(mm.sum()), held=float(mt["held"][mm].mean()) if mm.any() else float("nan"))
      res["task_H1A"] = task
      res["creep_HC1"] = summary_at(mt, S, CREEP)
    if (arm, conf_seed, BASE[0]) in Ms:
      res["base_side"] = summary_at(Ms[(arm, conf_seed, BASE[0])], BASE, VERDICT)
    out["arms"][arm] = res
  passed = [a for a in MAIN_ARMS if out["arms"].get(a, {}).get("label") == "통과"]
  out["line_verdict"] = "통과" if passed else "미결"
  out["passed_main_arms"] = passed
  out["reading"] = ("통과: 그간의 실패는 대본 탓, 접근은 RL 몫, 대본은 더 고치지 않는다" if passed
                    else "미결: 이 엔진·설정(주먹)의 한계이거나 판정이 서지 않음 — 여기서 멈추고 조율자가 다음 가지로")
  return out


def cross(mdir) -> dict:
  """C vs MJWarp per world on the C subset (rules 8): held agreement, slip difference."""
  C, Wm = _load(mdir, "cpu"), _load(mdir, "mjw")
  out = {}
  for (arm, seed, panel), mc in C.items():
    mw = Wm.get((arm, seed, panel))
    if mw is None:
      continue
    n = len(mc["held"])
    hw = mw["held"][:n]
    ok = mc["gate_ok"] & mw["gate_ok"][:n] & (_col(mc, "ik_ok") > 0.5)
    out[f"{arm}|s{seed}|{panel}"] = dict(n=int(ok.sum()), agree=float((mc["held"][ok] == hw[ok]).mean()) if ok.any() else float("nan"),
                                         held_c=float(mc["held"][ok].mean()) if ok.any() else float("nan"),
                                         held_w=float(hw[ok].mean()) if ok.any() else float("nan"),
                                         slip_med_abs_diff=float(np.median(np.abs(mc["slip"][ok] - mw["slip"][:n][ok]))) if ok.any() else float("nan"))
  return out


def main(argv):
  if argv[0] == "metrics":
    np.savez(argv[2], **metrics(argv[1]))
  elif argv[0] == "decide":
    sel, conf = (int(argv[4]), int(argv[5])) if len(argv) > 5 else (31, 32)
    res = decide(argv[1], argv[2], sel, conf)
    json.dump(res, open(argv[3], "w"), indent=1, default=float)
    print("written", argv[3])
  elif argv[0] == "cross":
    json.dump(cross(argv[1]), open(argv[2], "w"), indent=1)
    print("written", argv[2])


if __name__ == "__main__":
  main(sys.argv[1:])
