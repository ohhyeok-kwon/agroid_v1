"""Frozen readout of the R11 contact ladder (rules.md 9b788f10 + rules_amend.md A1/A2). Positive 18.

  python -m r1adv.positive_obj.readout metrics <panel.npz> <out_metrics.npz>      per-world quantities (rules 4)
  python -m r1adv.positive_obj.readout decide <metrics_dir> <engine> <out.json> [cost.json]   D1-D3 (rules 5, 6)

metrics_dir holds <engine>_s<seed>_<panel>.npz files written by 'metrics'.
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
C_P4_START, C_P4_END = 274, 774                        # 50 Hz samples at t = 5.5 s and 15.5 s
X_G, YAW_AXIS_X = 0.40, 0.0325
RHO = X_G - YAW_AXIS_X
PSI_PLACE = 0.30 / RHO / 2
TC_BINS = (0.010, 0.020, 0.030, 0.040)


def mask200(a: float, b: float) -> np.ndarray:
  return (T200 >= a - 1e-9) & (T200 < b - 1e-9)


def mask50(a: float, b: float) -> np.ndarray:
  return (T50 >= a - 1e-9) & (T50 < b - 1e-9)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
  """(..., 4) wxyz -> (..., 3, 3)."""
  q = np.where((np.abs(q).sum(-1, keepdims=True) < 1e-12), np.array([1.0, 0.0, 0.0, 0.0]), q)   # empty world: no object
  w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
  n = np.sqrt(w * w + x * x + y * y + z * z)
  w, x, y, z = w / n, x / n, y / n, z / n
  R = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1)
  return R.reshape(q.shape[:-1] + (3, 3))


def hand_frame(hand_p: np.ndarray, c: int) -> tuple[np.ndarray, np.ndarray]:
  """Origin (N, 3) and rotation (N, 3, 3) (columns x_H, y_H, z_H) of the hand-pair frame at 50 Hz sample c."""
  pl, pr = hand_p[:, c, 0].astype(np.float64), hand_p[:, c, 1].astype(np.float64)
  o = (pl + pr) / 2
  y = pl - pr
  y /= np.linalg.norm(y, axis=1, keepdims=True)
  z = np.array([0.0, 0.0, 1.0]) - y[:, 2:3] * y
  z /= np.linalg.norm(z, axis=1, keepdims=True)
  x = np.cross(y, z)
  return o, np.stack([x, y, z], axis=-1)


def hf_rms(v: np.ndarray) -> np.ndarray:
  """RMS of the >= 10 Hz part of the acceleration of 200 Hz velocity v (N, T, 3) (rules 4: per window mean and linear
  trend removed, FFT bins below 10 Hz zeroed)."""
  a = np.diff(v.astype(np.float64), axis=1) / DT
  n = a.shape[1]
  t = np.arange(n, dtype=np.float64)
  tc = t - t.mean()
  slope = (a * tc[None, :, None]).sum(1, keepdims=True) / (tc ** 2).sum()
  a = a - a.mean(1, keepdims=True) - slope * tc[None, :, None]
  F = np.fft.rfft(a, axis=1)
  freq = np.fft.rfftfreq(n, DT)
  F[:, freq < 10.0, :] = 0.0
  ahf = np.fft.irfft(F, n=n, axis=1)
  return np.sqrt((ahf ** 2).sum(-1).mean(1))


def metrics(path: str) -> dict:
  z = np.load(path)
  names = [str(s) for s in z["param_names"]]
  prm = z["params"]
  col = {n: prm[:, names.index(n)] for n in names}
  N = len(prm)
  out = {"params": prm, "param_names": z["param_names"]}
  obj_p, hand_p, hand_q = z["obj_p"], z["hand_p"], z["hand_q"]
  ncon, f_hand, dmin = z["ncon"], z["f_hand"], z["dist_min"]
  # hand-pair frame, slip
  o0, R0 = hand_frame(hand_p, C_P4_START)
  o1, R1 = hand_frame(hand_p, C_P4_END)
  prel0 = np.einsum("nji,nj->ni", R0, obj_p[:, C_P4_START].astype(np.float64) - o0)
  prel1 = np.einsum("nji,nj->ni", R1, obj_p[:, C_P4_END].astype(np.float64) - o1)
  slip = np.linalg.norm(prel1 - prel0, axis=1) * 1000.0
  # normal forces: projection on the O6 body z axis of the matching 50 Hz sample
  nrm = quat_to_mat(hand_q.astype(np.float64))[..., :, 2]                     # (N, T50, 2, 3)
  nrm200 = np.repeat(nrm, 4, axis=1)                                          # step k -> control period k // 4
  Nf = np.abs((f_hand.astype(np.float64) * nrm200).sum(-1))                    # (N, T, 2)
  mP2, mP4 = mask200(*PH["P2"]), mask200(*PH["P4"])
  NP2, NP4 = Nf[:, mP2].mean(1), Nf[:, mP4].mean(1)
  mass, mu = col["mass"], col["mu"]
  M_P2 = mu * NP2.sum(1) / (mass * G)
  M_P4 = mu * NP4.sum(1) / (mass * G)
  # held
  c1 = (ncon[:, mP4, 3] == 0).all(1) & (ncon[:, mP4, 4] == 0).all(1)
  c2 = obj_p[:, C_P4_END, 2] >= obj_p[:, C_P4_START, 2] - 0.05
  c3 = slip <= 50.0
  both = (ncon[:, mP4, 1] > 0) & (ncon[:, mP4, 2] > 0)
  c4 = both.mean(1) >= 0.95
  body_frac = (ncon[:, mP4, 5] > 0).mean(1)
  c5 = body_frac <= 0.05
  mG = mask200(1.0, 22.0)
  pen = np.maximum(-dmin[:, mG].min(1).min(1), 0.0)
  c6 = pen <= 0.005
  held5 = c1 & c2 & c3 & c4 & c5
  held = held5 & c6
  # chatter windows
  mW = mask200(6.5, 15.5)
  hf_hand = np.stack([hf_rms(z["hand_v"][:, mW, h]) for h in (0, 1)], 1).mean(1)
  hf_obj = hf_rms(z["obj_v"][:, mW])
  # torque
  m50 = mask50(1.0, 22.0)
  tau = z["tau_ratio"]
  tau_peak = tau[:, m50].max(-1).max(-1)
  tau_peak_p34 = tau[:, mask50(3.5, 15.5)].max(-1).max(-1)
  sat = (tau[:, mask50(*PH["P4"])].max(-1) >= 0.95).mean(1)
  # settling
  rel = np.linalg.norm(z["obj_v"].astype(np.float64) - z["hand_v"].astype(np.float64).mean(2), axis=-1)   # (N, T)
  quiet = rel < 0.01
  k0 = int(np.searchsorted(T200, 5.5 - 1e-9))
  settle = np.full(N, 10.0)
  win = 100
  q = quiet[:, k0:]
  cs = np.concatenate([np.zeros((N, 1)), np.cumsum(q, axis=1)], axis=1)
  full = (cs[:, win:] - cs[:, :-win]) == win                                # window starting at j fully quiet
  has = full.any(1)
  first = np.argmax(full, axis=1)
  t_first = T200[k0 + first] - 5.5
  settle = np.where(has & (t_first < 10.0), t_first, 10.0)
  # contacts
  ncon_max = ncon[..., 0].max(1)
  hand_con_max = (ncon[..., 1].astype(int) + ncon[..., 2]).max(1)
  robot_table = (ncon[:, mG, 6] > 0).any(1)
  robot_table_phase = {p: (ncon[:, mask200(*PH[p]), 6] > 0).any(1) for p in PH}
  self_con = (ncon[:, mG, 7] > 0).any(1)
  cap_frac = (z["ncap"][:, mP4] > 0).any(-1).mean(1)
  # placement (reference)
  place = np.array([YAW_AXIS_X + RHO * math.cos(PSI_PLACE), RHO * math.sin(PSI_PLACE)])
  place_err = np.linalg.norm(obj_p[:, -1, :2].astype(np.float64) - place, axis=1)
  zax = quat_to_mat(z["obj_q"][:, -1].astype(np.float64))[..., :, 2]
  tilt = np.degrees(np.arccos(np.clip(np.abs(zax[:, 2]), 0, 1)))
  still = (np.linalg.norm(z["obj_v"][:, mask200(*PH["P8"])], axis=-1) < 0.01).mean(1)
  out.update(slip=slip, NP2=NP2, NP4=NP4, M_P2=M_P2, M_P4=M_P4, c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, held5=held5,
             held=held, pen=pen, both_frac=both.mean(1), body_frac=body_frac, hf_hand=hf_hand, hf_obj=hf_obj,
             tau_peak=tau_peak, tau_peak_p34=tau_peak_p34, sat=sat, settle=settle, ncon_max=ncon_max,
             hand_con_max=hand_con_max, robot_table=robot_table, self_con=self_con, cap_frac=cap_frac,
             place_err=place_err, tilt=tilt, still=still, nefc_max=z["nefc_max"],
             **{f"robot_table_{p}": v for p, v in robot_table_phase.items()})
  out["meta"] = str(z["meta"])
  return out


def failure_class(m: dict, twin_self: np.ndarray | None) -> np.ndarray:
  """rules 6: 0 = held, 1..6 = F1..F6 (first match)."""
  col = {str(n): i for i, n in enumerate(m["param_names"])}
  ik_ok = m["params"][:, col["ik_ok"]] > 0.5
  f2_self = m["self_con"] & (~twin_self if twin_self is not None else np.zeros_like(m["self_con"]))
  F1 = ~ik_ok
  F2 = m["robot_table"] | (m["body_frac"] > 0.05) | f2_self
  F3 = (m["tau_peak_p34"] >= 0.99) | (m["sat"] > 0.2)
  F4 = m["held5"] & ~m["c6"]
  F5 = m["M_P2"] < 1.0
  cls = np.full(len(ik_ok), 6)
  for k, F in reversed(list(enumerate((F1, F2, F3, F4, F5), start=1))):
    cls = np.where(F, k, cls)
  return np.where(m["held"], 0, cls)


def _load_metrics(mdir: str, engine: str) -> dict:
  out = {}
  for f in sorted(glob.glob(os.path.join(mdir, f"{engine}_s*_*.npz"))):
    base = os.path.basename(f)[len(engine) + 2:-4]
    seed, panel = base.split("_", 1)
    out[(int(seed), panel)] = dict(np.load(f, allow_pickle=True))
  return out


def decide(mdir: str, engine: str, cost: dict | None = None) -> dict:
  M = _load_metrics(mdir, engine)
  seeds = sorted({s for s, _ in M})
  res = {"engine": engine, "seeds": seeds, "settings": [], "panels": {}}
  # twin chatter lookup per seed: original world index -> HF_hand, self-contact
  twin_hf, twin_self = {}, {}
  for s in seeds:
    e = M.get((s, "empty"))
    if e is None:
      continue
    col = {str(n): i for i, n in enumerate(e["param_names"])}
    idx = e["params"][:, col["twin_of"]].astype(int)
    twin_hf[s] = dict(zip(idx.tolist(), e["hf_hand"].tolist()))
    twin_self[s] = dict(zip(idx.tolist(), e["self_con"].tolist()))
  cols: dict[str, list] = {}
  def put(**kv):
    for k, v in kv.items():
      cols.setdefault(k, []).append(v)
  for (s, panel), m in M.items():
    if panel == "empty":
      continue
    col = {str(n): i for i, n in enumerate(m["param_names"])}
    n = len(m["params"])
    widx = np.arange(n)                                 # registered panels hold every world in sample_worlds order (A, C, B)
    t_hf = np.array([twin_hf.get(s, {}).get(int(i), np.nan) for i in widx])
    t_self = np.array([twin_self.get(s, {}).get(int(i), False) for i in widx], dtype=bool)
    prm = m["params"]
    stratum = prm[:, col["stratum"]]
    cls = failure_class(m, np.where(stratum <= 1, t_self, False))
    put(seed=np.full(n, s), panel=np.array([panel] * n), nc=np.full(n, panel.endswith("nc")),
        cone=np.array([panel[:2]] * n), condim=np.full(n, int(panel[2])), stratum=stratum.astype(int),
        shape=prm[:, col["shape"]].astype(int), w=prm[:, col["dy"]], mass=prm[:, col["mass"]], mu=prm[:, col["mu"]],
        imp=prm[:, col["impratio"]], tcbin=np.searchsorted(np.array(TC_BINS[1:3]), prm[:, col["tc"]], side="right"),
        held=m["held"].astype(bool), slip=m["slip"], M_P4=m["M_P4"], M_P2=m["M_P2"], r=m["hf_hand"] / t_hf,
        hf_obj=m["hf_obj"], hf_hand=m["hf_hand"], cls=cls, pen=m["pen"], settle=m["settle"], tau_peak=m["tau_peak"],
        ncon_max=m["ncon_max"], hand_con_max=m["hand_con_max"], cap_frac=m["cap_frac"],
        robot_table=m["robot_table"].astype(bool), nefc_max=m["nefc_max"])
  df = {k: np.concatenate(v) for k, v in cols.items()}
  res["n_worlds"] = int(len(df["seed"]))

  def where(base=None, **conds) -> np.ndarray:
    msk = np.ones(len(df["seed"]), bool) if base is None else base.copy()
    for k, v in conds.items():
      msk &= (df[k] == v)
    return msk

  def med(x):
    return float(np.median(x)) if len(x) else None

  def frac(x):
    return float(np.mean(x)) if len(x) else None

  def counts(x):
    u, c = np.unique(x, return_counts=True)
    return {int(a): int(b) for a, b in zip(u, c)}

  mu05 = np.abs(df["mu"] - 0.5) < 1e-9

  def setting_stats(msk: np.ndarray) -> dict:
    a = msk & (df["stratum"] == 1) & df["held"] & (df["M_P4"] >= 2)
    b = msk & (df["stratum"] <= 1) & df["held"] & np.isfinite(df["r"])
    d2 = msk & (df["stratum"] == 0) & mu05
    st = dict(n_slip=int(a.sum()), slip_med=med(df["slip"][a]), n_chat=int(b.sum()), r_med=med(df["r"][b]),
              d2_n=int(d2.sum()), d2_held=frac(df["held"][d2]))
    st["slip_pass"] = None if a.sum() < 30 else bool(st["slip_med"] < 5.0)
    st["chat_pass"] = None if b.sum() < 30 else bool(st["r_med"] < 2.0)
    st["pass"] = (bool(st["slip_pass"] and st["chat_pass"])
                  if (st["slip_pass"] is not None and st["chat_pass"] is not None) else None)
    return st

  for nc in (False, True):
    for cone in ("py", "el"):
      for condim in (3, 4):
        for imp in (1.0, 10.0):
          for tb in range(3):
            msk = where(nc=nc, cone=cone, condim=condim, imp=imp, tcbin=tb)
            if not msk.any():
              continue
            st = setting_stats(msk)
            st.update(nc=nc, cone=cone, condim=condim, imp=imp, tcbin=tb,
                      per_seed={int(s): setting_stats(msk & (df["seed"] == s)) for s in seeds})
            res["settings"].append(st)
  prim = [s for s in res["settings"] if not s["nc"]]
  passing = [s for s in prim if s["pass"]]
  if passing:
    best = min(s["slip_med"] for s in passing)
    near = [s for s in passing if s["slip_med"] <= best + 1.0]
    near.sort(key=lambda s: (s["cone"] != "py", s["condim"] != 3, s["imp"] != 1.0, -s["tcbin"]))
    chosen = near[0]
    res["D1"] = dict(verdict="설정 있음", chosen={k: chosen[k] for k in ("cone", "condim", "imp", "tcbin")},
                     n_passing=len(passing))
  else:
    chosen = next(s for s in prim if s["cone"] == "py" and s["condim"] == 3 and s["imp"] == 1.0 and s["tcbin"] == 1)
    res["D1"] = dict(verdict="RL 전에 접촉 모델부터 고친다", chosen=None,
                     default={"cone": "py", "condim": 3, "imp": 1.0, "tcbin": 1},
                     n_undecidable=sum(1 for s in prim if s["pass"] is None),
                     n_slip_fail=sum(1 for s in prim if s["slip_pass"] is False),
                     n_chat_fail=sum(1 for s in prim if s["chat_pass"] is False))
  d2 = chosen["d2_held"]
  res["D2"] = dict(setting={k: chosen[k] for k in ("cone", "condim", "imp", "tcbin")}, n=chosen["d2_n"], held=d2,
                   verdict=None if d2 is None else ("24 차원 + 큰 물체" if d2 >= 0.5 else "O6 를 1 단계에"))
  pass_d2 = [s["d2_held"] for s in passing if s["d2_held"] is not None]
  if passing and len({v >= 0.5 for v in pass_d2}) > 1:
    res["D2"]["note"] = "설정에 따라 갈림"

  def side(nc: bool) -> dict:
    base = where(nc=nc, cone=chosen["cone"], condim=chosen["condim"], imp=chosen["imp"], tcbin=chosen["tcbin"])
    if not base.any():
      return {"n_A": 0}
    A = base & (df["stratum"] == 0)
    A5 = A & mu05
    A5w = A5 & (df["w"] >= 0.2)
    Bm = base & (df["stratum"] == 2)
    hb = base & df["held"]
    return dict(n_A=int(A.sum()), held_A=frac(df["held"][A]), held_A_mu05=frac(df["held"][A5]),
                held_A_mu05_wide=frac(df["held"][A5w]), n_A_mu05_wide=int(A5w.sum()),
                cls_A_mu05=counts(df["cls"][A5]), cls_A=counts(df["cls"][A]),
                held_A_mass={f"{lo}-{hi}": frac(df["held"][A & (df["mass"] >= lo) & (df["mass"] < hi)])
                             for lo, hi in ((0.2, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.01))},
                held_A_shape={int(k): frac(df["held"][A & (df["shape"] == k)]) for k in range(3)},
                held_C=frac(df["held"][base & (df["stratum"] == 1)]), held_B=frac(df["held"][Bm]),
                held_B_mu05=frac(df["held"][Bm & mu05]), settle_med_held=med(df["settle"][hb]),
                pen_p99=float(np.quantile(df["pen"][base], 0.99)), tau_peak_max=float(df["tau_peak"][base].max()),
                hf_obj_med_held=med(df["hf_obj"][hb]), hf_hand_med_held=med(df["hf_hand"][hb]))
  res["chosen_side"] = side(False)
  if where(nc=True).any():
    res["chosen_side_nc"] = side(True)
  res["d2_all"] = [{k: s[k] for k in ("nc", "cone", "condim", "imp", "tcbin", "d2_n", "d2_held", "slip_med", "n_slip",
                                      "r_med", "n_chat", "pass")} for s in res["settings"]]
  res["contacts"] = dict(ncon_max=int(df["ncon_max"].max()), hand_con_max=int(df["hand_con_max"].max()),
                         nefc_max=int(df["nefc_max"].max()), ncon_p99=float(np.quantile(df["ncon_max"], 0.99)))
  res["by_panel"] = {}
  for p in np.unique(df["panel"]):
    g = where(panel=p)
    gA, gC, gB = g & (df["stratum"] == 0), g & (df["stratum"] == 1), g & (df["stratum"] == 2)
    res["by_panel"][str(p)] = dict(
      held_A=frac(df["held"][gA]), held_C=frac(df["held"][gC]), held_B=frac(df["held"][gB]),
      held_A_mu05=frac(df["held"][gA & mu05]), slip_med_C_held=med(df["slip"][gC & df["held"]]),
      r_med=med(df["r"][(gA | gC) & df["held"] & np.isfinite(df["r"])]), cls_A=counts(df["cls"][gA]),
      cap_frac_med=med(df["cap_frac"][g]), robot_table_frac=frac(df["robot_table"][g]))
  if cost:
    cone_name = {"py": "pyramidal", "el": "elliptic"}[chosen["cone"]]
    key = f"{cone_name}_{chosen['condim']}"
    r_ = cost.get("R4096", {}).get(key, {})
    res["D3"] = {sc: dict(R=r_.get(sc), verdict=None if r_.get(sc) is None else ("모든 env 에 물체" if r_[sc] <= 1.5 else "줄인다"))
                 for sc in ("one", "three", "park")}
    res["D3"]["key"] = key
  return res


def main(argv: list[str]) -> None:
  if argv[0] == "metrics":
    out = metrics(argv[1])
    np.savez(argv[2], **out)
  elif argv[0] == "decide":
    cost = json.load(open(argv[4])) if len(argv) > 4 else None
    res = decide(argv[1], argv[2], cost)
    with open(argv[3], "w") as f:
      json.dump(res, f, indent=1, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps({k: res[k] for k in ("D1", "D2") if k in res}, ensure_ascii=False, indent=1))
    if "D3" in res:
      print(json.dumps(res["D3"], ensure_ascii=False))


if __name__ == "__main__":
  main(sys.argv[1:])
