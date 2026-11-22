"""Frozen readout of the positive 20 two-arm ladder (rounds/R04/positive/r11_contact2/rules.md c394a1ac + rules_amend.md A1, A2).

  python -m r1adv.positive_obj.p20_readout metrics <panel.npz> <out_metrics.npz>
  python -m r1adv.positive_obj.p20_readout decide <metrics_dir> <engine> <out.json>
metrics_dir holds <arm>_<engine>_s<seed>_<panel>.npz written by 'metrics'.

Definitions (rules 4-6): slip in the hand-pair frame (G1) or the left O6 body frame (G2) between t 5.5 and 15.5 s; friction
margin M = mu * sum_k mean fn_k / (m g) with fn_k the summed contact-frame normal force of hand k; held = the six physical
conditions of positive 18 (G2: condition 4 on the left hand only); rule K q on the object acceleration (code below copies
negative 17 r11_khold.py d89a3576 _bands / k2_world, H_lo 1.514, H_hi 1.764 m/s^2); failure classes F0 (ladder IK
infeasible), F2-F6. F2's 'self-contact pair absent at t = 1.0 s' is read from the per-step self-contact count: a step in
[1, 22] s with more robot self-contacts than at t = 1.0 s.
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

DT, NSTEP, NCTRL = 0.005, 4800, 1200
T200 = (np.arange(NSTEP) + 1) * DT
T50 = (np.arange(NCTRL) + 1) * 0.02
PH = dict(P0=(0.0, 1.0), P1=(1.0, 2.5), P2=(2.5, 3.5), P3=(3.5, 5.5), P4=(5.5, 15.5), P5=(15.5, 18.5),
          P6=(18.5, 20.5), P7=(20.5, 22.0), P8=(22.0, 24.0))
G = 9.81
C_P4_START, C_P4_END = 274, 774
TC_BINS = (0.010, 0.020, 0.030, 0.040)
H_LO, H_HI = 1.514, 1.764
G1_ARMS = ("box", "boxpads", "o6straight")
G2_ARMS = ("o6", "o6pads", "o6full")
VERDICT_STRATA = {a: dict(slip=(1,), k=(0, 1), task=(0,)) for a in G1_ARMS}
VERDICT_STRATA.update({a: dict(slip=(6,), k=(5, 6), task=(5,)) for a in G2_ARMS})


def mask200(a, b):
  return (T200 >= a - 1e-9) & (T200 < b - 1e-9)


def mask50(a, b):
  return (T50 >= a - 1e-9) & (T50 < b - 1e-9)


def quat_to_mat(q):
  q = np.where((np.abs(q).sum(-1, keepdims=True) < 1e-12), np.array([1.0, 0.0, 0.0, 0.0]), q)
  w, x, y, z = (q[..., i] / np.linalg.norm(q, axis=-1) for i in range(4))
  R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
  return R.reshape(q.shape[:-1] + (3, 3))


def pair_frame(hand_p, c):
  pl, pr = hand_p[:, c, 0].astype(np.float64), hand_p[:, c, 1].astype(np.float64)
  o = (pl + pr) / 2
  y = pl - pr
  y /= np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)
  z = np.array([0.0, 0.0, 1.0]) - y[:, 2:3] * y
  z /= np.linalg.norm(z, axis=1, keepdims=True)
  return o, np.stack([np.cross(y, z), y, z], axis=-1)


# --- copy of negative 17 r11_khold.py (d89a3576) _bands / k2_world -------------------------------------------------
def _bands(x, f, edges):
  if x.shape[0] % 2:
    x = x[1:]
  N = x.shape[0]
  t = np.arange(N)
  A = np.stack([np.ones(N), t - t.mean()], axis=1)
  coef, *_ = np.linalg.lstsq(A, x, rcond=None)
  x = x - A @ coef
  w = np.hanning(N)
  X = np.fft.rfft(x * w[:, None], axis=0)
  fr = np.fft.rfftfreq(N, 1.0 / f)
  c = np.full(len(fr), 2.0)
  c[0] = 1.0
  c[-1] = 1.0
  P = c[:, None] * np.abs(X) ** 2 / (N * (w ** 2).sum())
  return [float(np.sqrt(P[(fr >= lo) & (fr <= hi)].sum())) for lo, hi in edges]


def k2_world(obj_v):
  m = mask200(6.5, 15.5)
  a = np.diff(obj_v[m].astype(float), axis=0) / DT
  lo, hi = _bands(a, 1.0 / DT, [(0.5, 8.0 - 1e-9), (8.0, 100.0)])
  return lo, hi, max(lo / H_LO, hi / H_HI)


def k3_world(f_hand):
  m = mask200(6.5, 15.5)
  out = []
  for k in range(f_hand.shape[1]):
    mag = np.linalg.norm(f_hand[m, k].astype(float), axis=1)
    mean = float(mag.mean())
    out.append(_bands(mag[:, None], 1.0 / DT, [(0.5, 100.0)])[0] / mean if mean > 0 else np.nan)
  return out
# ---------------------------------------------------------------------------------------------------------------------


def metrics(path: str) -> dict:
  z = dict(np.load(path))                                   # load every array once
  names = [str(s) for s in z["param_names"]]
  prm = z["params"]
  col = {n: prm[:, names.index(n)] for n in names}
  N = len(prm)
  g2 = col["grasp"] > 0.5
  obj_p, hand_p, hand_q = z["obj_p"], z["hand_p"], z["hand_q"]
  ncon, fn, dmin = z["ncon"], z["fn_hand"].astype(np.float64), z["dist_min"]
  # slip
  o0, R0 = pair_frame(hand_p, C_P4_START)
  o1, R1 = pair_frame(hand_p, C_P4_END)
  oL0, RL0 = hand_p[:, C_P4_START, 0].astype(np.float64), quat_to_mat(hand_q[:, C_P4_START, 0].astype(np.float64))
  oL1, RL1 = hand_p[:, C_P4_END, 0].astype(np.float64), quat_to_mat(hand_q[:, C_P4_END, 0].astype(np.float64))
  R0 = np.where(g2[:, None, None], RL0, R0)
  R1 = np.where(g2[:, None, None], RL1, R1)
  o0 = np.where(g2[:, None], oL0, o0)
  o1 = np.where(g2[:, None], oL1, o1)
  prel0 = np.einsum("nji,nj->ni", R0, obj_p[:, C_P4_START].astype(np.float64) - o0)
  prel1 = np.einsum("nji,nj->ni", R1, obj_p[:, C_P4_END].astype(np.float64) - o1)
  slip = np.linalg.norm(prel1 - prel0, axis=1) * 1000.0
  mP2, mP4 = mask200(*PH["P2"]), mask200(*PH["P4"])
  mg = col["mass"] * G
  M_P2 = col["mu"] * fn[:, mP2].mean(1).sum(1) / mg
  M_P4 = col["mu"] * fn[:, mP4].mean(1).sum(1) / mg
  nrm = np.repeat(quat_to_mat(hand_q.astype(np.float64))[..., :, 2], 4, axis=1)
  N18 = np.abs((z["f_hand"].astype(np.float64) * nrm).sum(-1))
  M18_P2 = col["mu"] * N18[:, mP2].mean(1).sum(1) / mg
  M18_P4 = col["mu"] * N18[:, mP4].mean(1).sum(1) / mg
  c1 = (ncon[:, mP4, 3] == 0).all(1) & (ncon[:, mP4, 4] == 0).all(1)
  c2 = obj_p[:, C_P4_END, 2] >= obj_p[:, C_P4_START, 2] - 0.05
  c3 = slip <= 50.0
  both = (ncon[:, mP4, 1] > 0) & (ncon[:, mP4, 2] > 0)
  left = ncon[:, mP4, 1] > 0
  c4 = np.where(g2, left.mean(1) >= 0.95, both.mean(1) >= 0.95)
  body_frac = (ncon[:, mP4, 5] > 0).mean(1)
  c5 = body_frac <= 0.05
  mG = mask200(1.0, 22.0)
  pen = np.maximum(-dmin[:, mG].min(1).min(1), 0.0)
  c6 = pen <= 0.005
  held5 = c1 & c2 & c3 & c4 & c5
  held = held5 & c6
  obj_v_all, f_hand_all = z["obj_v"], z["f_hand"]          # read each array once (NpzFile re-reads on every access)
  k = np.array([k2_world(obj_v_all[i]) for i in range(N)])
  k3 = np.array([k3_world(f_hand_all[i]) for i in range(N)])
  tau = z["tau_ratio"]
  tau_peak_p34 = tau[:, mask50(3.5, 15.5)].max(-1).max(-1)
  sat = (tau[:, mask50(*PH["P4"])].max(-1) >= 0.95).mean(1)
  rel = np.linalg.norm(z["obj_v"].astype(np.float64) - z["hand_v"].astype(np.float64)[:, :, 0 if False else slice(None)].mean(2), axis=-1)
  relL = np.linalg.norm(z["obj_v"].astype(np.float64) - z["hand_v"][:, :, 0].astype(np.float64), axis=-1)
  rel = np.where(g2[:, None], relL, rel)
  k0 = int(np.searchsorted(T200, 5.5 - 1e-9))
  q = rel[:, k0:] < 0.01
  cs = np.concatenate([np.zeros((N, 1)), np.cumsum(q, axis=1)], axis=1)
  full = (cs[:, 100:] - cs[:, :-100]) == 100
  t_first = T200[k0 + np.argmax(full, axis=1)] - 5.5
  settle = np.where(full.any(1) & (t_first < 10.0), t_first, 10.0)
  robot_table = (ncon[:, mG, 6] > 0).any(1)
  k10 = int(np.searchsorted(T200, 1.0 - 1e-9))
  self_new = (ncon[:, mG, 7] > ncon[:, k10:k10 + 1, 7]).any(1)
  hand_con = ncon[:, :, 1].astype(np.int64) + ncon[:, :, 2]
  mPP = mP2 | mP4
  wrist_share = np.where(hand_con[:, mPP].sum(1) > 0, z["ncap"][:, mPP].sum((1, 2)) / np.maximum(hand_con[:, mPP].sum(1), 1), np.nan)
  out = dict(params=prm, param_names=z["param_names"], slip=slip, M_P2=M_P2, M_P4=M_P4, M18_P2=M18_P2, M18_P4=M18_P4,
             c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, held5=held5, held=held, pen=pen, body_frac=body_frac,
             rms_lo=k[:, 0], rms_hi=k[:, 1], q=k[:, 2], k3_L=k3[:, 0], k3_R=k3[:, 1], tau_peak_p34=tau_peak_p34, sat=sat,
             settle=settle, robot_table=robot_table, self_new=self_new, wrist_share=wrist_share,
             ncon_max=ncon[..., 0].max(1), hand_con_max=hand_con.max(1), nefc_max=z["nefc_max"],
             obj_lift=obj_p[:, C_P4_START, 2] - obj_p[:, 0, 2])
  out["meta"] = str(z["meta"])
  return out


def failure_class(m) -> np.ndarray:
  col = {str(n): i for i, n in enumerate(m["param_names"])}
  ik_ok = m["params"][:, col["ik_ok"]] > 0.5
  F2 = m["robot_table"] | (m["body_frac"] > 0.05) | m["self_new"]
  F3 = (m["tau_peak_p34"] >= 0.99) | (m["sat"] > 0.2)
  F4 = m["held5"] & ~m["c6"]
  F5 = m["M_P2"] < 1.0
  cls = np.full(len(ik_ok), 6)
  for k, F in ((5, F5), (4, F4), (3, F3), (2, F2)):
    cls = np.where(F, k, cls)
  cls = np.where(m["held"], 0, cls)
  return np.where(~ik_ok, -1, cls)                      # -1 = F0 ladder geometry outside (denominator out)


def _load(mdir, engine):
  out = {}
  for f in sorted(glob.glob(os.path.join(mdir, f"*_{engine}_s*_*.npz"))):
    base = os.path.basename(f)[:-4]
    arm, eng, seed, panel = base.rsplit("_", 3)
    out[(arm, int(seed[1:]), panel)] = dict(np.load(f, allow_pickle=True))
  return out


def decide(mdir: str, engine: str) -> dict:
  M = _load(mdir, engine)
  arms = sorted({a for a, _, _ in M})
  res = {"engine": engine, "arms": {}, "rules": "c394a1ac + rules_amend A1"}
  frames = {}
  for arm in arms:
    cols = {}
    for (a, s, panel), m in M.items():
      if a != arm:
        continue
      col = {str(n): i for i, n in enumerate(m["param_names"])}
      prm = m["params"]
      n = len(prm)
      stratum = prm[:, col["stratum"]].astype(int)
      rank = np.zeros(n, int)
      for st in np.unique(stratum):
        ids = np.nonzero(stratum == st)[0]
        rank[ids] = np.arange(len(ids))
      vals = dict(seed=np.full(n, s), panel=np.array([panel] * n), cone=np.array([panel[:2]] * n),
                  condim=np.full(n, int(panel[2])), stratum=stratum, rank=rank, shape=prm[:, col["shape"]].astype(int),
                  mass=prm[:, col["mass"]], mu=prm[:, col["mu"]], imp=prm[:, col["impratio"]],
                  tcbin=np.searchsorted(np.array(TC_BINS[1:3]), prm[:, col["tc"]], side="right"),
                  ik_ok=prm[:, col["ik_ok"]] > 0.5, held=m["held"].astype(bool), slip=m["slip"], M_P4=m["M_P4"],
                  M_P2=m["M_P2"], q=m["q"], rms_lo=m["rms_lo"], rms_hi=m["rms_hi"], cls=failure_class(m),
                  pen=m["pen"], wrist=m["wrist_share"], settle=m["settle"], w=prm[:, col["dy"]])
      for k, v in vals.items():
        cols.setdefault(k, []).append(v)
    df = {k: np.concatenate(v) for k, v in cols.items()}
    frames[arm] = df
    vs = VERDICT_STRATA[arm]
    isin = lambda st: np.isin(df["stratum"], st)          # noqa: E731

    def stats(msk):
      a = msk & isin(vs["slip"]) & df["held"] & (df["M_P4"] >= 2)
      b = msk & isin(vs["k"]) & df["held"]
      t = msk & isin(vs["task"]) & df["ik_ok"]
      st = dict(n_slip=int(a.sum()), slip_med=float(np.median(df["slip"][a])) if a.any() else None,
                n_k=int(b.sum()), q_med=float(np.median(df["q"][b])) if b.any() else None,
                k_pass_share=float(np.mean(df["q"][b] <= 1.0)) if b.any() else None,
                held_task=float(df["held"][t].mean()) if t.any() else None, n_task=int(t.sum()))
      st["slip_pass"] = None if st["n_slip"] < 30 else bool(st["slip_med"] < 5.0)
      st["k_pass"] = None if st["n_k"] < 30 else bool(st["q_med"] <= 1.0)
      st["pass"] = (bool(st["slip_pass"] and st["k_pass"]) if st["slip_pass"] is not None and st["k_pass"] is not None else None)
      return st

    settings = []
    for cone in ("py", "el"):
      for condim in (3, 4):
        for imp in (1.0, 10.0):
          for tb in range(3):
            msk = (df["cone"] == cone) & (df["condim"] == condim) & (df["imp"] == imp) & (df["tcbin"] == tb)
            if not msk.any():
              continue
            st = stats(msk)
            st.update(cone=cone, condim=condim, imp=imp, tcbin=tb,
                      per_seed={int(s): stats(msk & (df["seed"] == s)) for s in np.unique(df["seed"])})
            settings.append(st)
    passing = [s for s in settings if s["pass"]]
    py_pass = [s for s in passing if s["cone"] == "py"]
    chosen = None
    if py_pass:
      best = min(s["slip_med"] for s in py_pass)
      near = sorted([s for s in py_pass if s["slip_med"] <= best + 1.0], key=lambda s: (s["condim"] != 3, s["imp"] != 1.0, -s["tcbin"]))
      chosen = near[0]
      el_ok = [s for s in passing if s["cone"] == "el" and s["held_task"] is not None and chosen["held_task"] is not None
               and s["held_task"] >= chosen["held_task"] + 0.10]
      if el_ok:
        chosen = max(el_ok, key=lambda s: s["held_task"])
    elif passing:                                          # rules_amend A2-1: same order as step 1
      best = min(s["slip_med"] for s in passing)
      near = sorted([s for s in passing if s["slip_med"] <= best + 1.0], key=lambda s: (s["condim"] != 3, s["imp"] != 1.0, -s["tcbin"]))
      chosen = near[0]
    if chosen is None:
      base = next((s for s in settings if s["cone"] == "py" and s["condim"] == 3 and s["imp"] == 1.0 and s["tcbin"] == 1),
                  settings[0])                       # the base setting always exists in a complete four-panel read
      d1 = dict(verdict="RL 전에 접촉 모델부터 고친다", chosen=None, n_undecidable=sum(1 for s in settings if s["pass"] is None),
                n_slip_fail=sum(1 for s in settings if s["slip_pass"] is False), n_k_fail=sum(1 for s in settings if s["k_pass"] is False))
      sel = base
    else:
      d1 = dict(verdict="설정 있음", chosen={k: chosen[k] for k in ("cone", "condim", "imp", "tcbin")}, n_passing=len(passing))
      sel = chosen
    smask = (df["cone"] == sel["cone"]) & (df["condim"] == sel["condim"]) & (df["imp"] == sel["imp"]) & (df["tcbin"] == sel["tcbin"])

    def d2(msk):
      den = msk & df["ik_ok"]
      return dict(n=int(den.sum()), held=float(df["held"][den].mean()) if den.any() else None, n_geom_out=int((msk & ~df["ik_ok"]).sum()),
                  cls={int(k): int(v) for k, v in zip(*np.unique(df["cls"][msk], return_counts=True))})

    out = dict(D1=d1, setting={k: sel[k] for k in ("cone", "condim", "imp", "tcbin")},
               settings=[{k: s[k] for k in ("cone", "condim", "imp", "tcbin", "n_slip", "slip_med", "n_k", "q_med", "k_pass_share",
                                           "held_task", "n_task", "slip_pass", "k_pass", "pass")} for s in settings])
    mu05 = np.abs(df["mu"] - 0.5) < 1e-9
    if arm in ("box", "boxpads"):
      r = d2(smask & (df["stratum"] == 4))
      r["verdict"] = None if r["held"] is None else ("24 차원 1 단계를 크고 가벼운 물체로 먼저 시작할 수 있다" if r["held"] >= 0.5 else "못 한다")
      r["per_seed"] = {int(s): d2(smask & (df["stratum"] == 4) & (df["seed"] == s)) for s in np.unique(df["seed"])}
      out["D2_B"] = r
    if arm in G2_ARMS:
      r = d2(smask & (df["stratum"] == 5) & mu05)
      h = r["held"]
      r["verdict"] = None if h is None else ("O6 를 넣는다, 26–28 차원" if h >= 0.7 else ("O6 를 넣는다, 36 차원(T1)" if h >= 0.5 else "묶음 O6 는 1 단계에 넣지 않는다"))
      r["per_seed"] = {int(s): d2(smask & (df["stratum"] == 5) & mu05 & (df["seed"] == s)) for s in np.unique(df["seed"])}
      out["D2_O"] = r
    # reported side values at the selected setting
    out["side"] = dict(
      held_by_stratum={int(st): d2(smask & (df["stratum"] == st)) for st in np.unique(df["stratum"])},
      held_by_shape={int(sh): d2(smask & isin(vs["task"]) & (df["shape"] == sh)) for sh in np.unique(df["shape"])},
      held_by_mass={f"{lo}-{hi}": d2(smask & isin(vs["task"]) & (df["mass"] >= lo) & (df["mass"] < hi)) for lo, hi in ((0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.01))},
      wrist_share_med=float(np.nanmedian(df["wrist"][smask])) if smask.any() else None,
      pen_p99=float(np.quantile(df["pen"][smask], 0.99)) if smask.any() else None)
    res["arms"][arm] = out
  # G1 side verdict: A1 mu 0.5, worlds IK-feasible for both box and o6 (each at its own selected setting)
  def held_at(arm, sel, strat, mu05_only=True):
    df = frames[arm]
    msk = ((df["cone"] == sel["cone"]) & (df["condim"] == sel["condim"]) & (df["imp"] == sel["imp"]) & (df["tcbin"] == sel["tcbin"])
           & (df["stratum"] == strat) & df["ik_ok"])
    if mu05_only:
      msk &= np.abs(df["mu"] - 0.5) < 1e-9
    return dict(n=int(msk.sum()), held=float(df["held"][msk].mean()) if msk.any() else None)
  if "o6full" in frames and "o6" in frames:                 # rules_amend A2-3: o6full at o6's D2' setting (side value)
    res["arms"]["o6full"]["D2_O_at_o6_setting"] = held_at("o6full", res["arms"]["o6"]["setting"], 5)
    a, b = res["arms"]["o6"].get("D2_O", {}).get("verdict"), res["arms"]["o6full"].get("D2_O", {}).get("verdict")
    res["calibration_split"] = None if (a is None or b is None) else bool(a != b)
  if "box" in frames and "o6" in frames:
    def g1(arm, sel=None):                                 # rules_amend A2-4: both arms at o6's D2' setting
      df = frames[arm]
      sel = sel or res["arms"]["o6"]["setting"]
      msk = ((df["cone"] == sel["cone"]) & (df["condim"] == sel["condim"]) & (df["imp"] == sel["imp"]) & (df["tcbin"] == sel["tcbin"])
             & (df["stratum"] == 0) & (np.abs(df["mu"] - 0.5) < 1e-9))
      return {(int(s), int(r)): (bool(ok), bool(h)) for s, r, ok, h in zip(df["seed"][msk], df["rank"][msk], df["ik_ok"][msk], df["held"][msk])}
    gb, go, gbo = g1("box"), g1("o6"), g1("box", res["arms"]["box"]["setting"])
    common = [k for k in gb if k in go and gb[k][0] and go[k][0]]
    hb = float(np.mean([gb[k][1] for k in common])) if common else None
    ho = float(np.mean([go[k][1] for k in common])) if common else None
    common_o = [k for k in gbo if k in go and gbo[k][0] and go[k][0]]
    hbo = float(np.mean([gbo[k][1] for k in common_o])) if common_o else None
    res["G1_side"] = dict(setting=res["arms"]["o6"]["setting"], n=len(common), held_box=hb, held_o6=ho,
                          diff=None if hb is None else ho - hb, o6_covers_large=None if ho is None else bool(ho >= 0.5),
                          held_box_own_setting=hbo, n_own=len(common_o))
  return res


def main(argv):
  if argv[0] == "metrics":
    out = metrics(argv[1])
    np.savez(argv[2], **out)
  elif argv[0] == "decide":
    res = decide(argv[1], argv[2])
    with open(argv[3], "w") as f:
      json.dump(res, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    for arm, r in res["arms"].items():
      print(arm, json.dumps({k: r[k] for k in ("D1", "D2_B", "D2_O") if k in r}, ensure_ascii=False))
    if "G1_side" in res:
      print("G1_side", res["G1_side"])


if __name__ == "__main__":
  main(sys.argv[1:])
