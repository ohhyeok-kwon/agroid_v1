"""Positive 20 slip-cause reproduction (rounds/R04/positive/r11_hold/slip_cause_rules.md): C MuJoCo on the CPU.
Scene M = two PD capsule fists on x/y/z slides + a box; scene R = the hold test's stock-fist R1 body + the same box.
The box is weightless (gravcomp / weight force) while the squeeze settles, then a downward force ramps 0 -> 1.5 m g over
3 s. One factor at a time (B, N2, PI10, E1, E10, D99, TC2, TC8, MOC, CD4, IT4).

  python -m r1adv.positive_obj.p20_slip_repro <out_dir> [M|R|MR] [--workers N]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import mujoco
import numpy as np

DT = 0.005
SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
MU, TC0 = 0.5, 0.02
KP, MF = 300.0, 0.325
KV = 2.0 * math.sqrt(KP * MF)
R_CAP, HL_CAP = 0.03, 0.035
DEPTH, HEIGHT = 0.125, 0.25
D_M, D_R = 0.05, 0.045                       # squeeze target inside the face (M: 300 N/m x 0.05 = 15 N; R: ~15 N measured)
OBJECTS = [(w, m) for w in (0.2, 0.4) for m in (0.2, 1.22)]
VARIANTS = ("B", "N2", "PI10", "E1", "E10", "D99", "TC2", "TC8", "MOC", "CD4", "IT4")
T_SETTLE = {"M": 1.0, "R": 1.5}
T_RAMP, L_MAX = 3.0, 1.5
QUAD = int(mujoco.mjtConstraintState.mjCNSTRSTATE_QUADRATIC)


def vparams(v: str) -> dict:
  p = dict(cone="pyramidal", imp=1.0, solimp=SOLIMP, tc=TC0, condim=3, it=10, ls=20, depth_mult=1.0, mocap=False)
  p.update({"N2": dict(depth_mult=2.0), "PI10": dict(imp=10.0), "E1": dict(cone="elliptic"), "E10": dict(cone="elliptic", imp=10.0),
            "D99": dict(solimp=(0.9, 0.99, 0.001, 0.5, 2.0)), "TC2": dict(tc=0.010), "TC8": dict(tc=0.040), "MOC": dict(mocap=True),
            "CD4": dict(condim=4), "IT4": dict(it=40, ls=80)}.get(v, {}))
  return p


def set_opt(o, p):
  o.timestep = DT
  o.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
  o.cone = mujoco.mjtCone.mjCONE_ELLIPTIC if p["cone"] == "elliptic" else mujoco.mjtCone.mjCONE_PYRAMIDAL
  o.impratio = p["imp"]
  o.iterations, o.ls_iterations = p["it"], p["ls"]
  o.tolerance, o.ls_tolerance, o.ccd_iterations = 1e-8, 0.01, 50


# ------------------------------------------------------------------------------------------------ scene M
def build_M(w, mass, p, delta=None):
  spec = mujoco.MjSpec()
  set_opt(spec.option, p)
  for side in (1, -1):
    if p["mocap"]:
      y = side * (w / 2 + R_CAP - delta)
      b = spec.worldbody.add_body(name=f"fist{side}", mocap=True, pos=[0, y, 0])
    else:
      b = spec.worldbody.add_body(name=f"fist{side}", pos=[0, side * (w / 2 + R_CAP + 0.001), 0], gravcomp=1.0)
      for ax, nm in (((1, 0, 0), "x"), ((0, 1, 0), "y"), ((0, 0, 1), "z")):
        b.add_joint(name=f"f{side}_{nm}", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=list(ax))
    b.add_geom(name=f"fist{side}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[-HL_CAP, 0, 0, HL_CAP, 0, 0],
               size=[R_CAP, 0, 0], mass=MF, contype=1, conaffinity=1, condim=1)
  ob = spec.worldbody.add_body(name="obj", pos=[0, 0, 0], gravcomp=1.0)
  ob.add_freejoint(name="objfree")
  ob.add_geom(name="obj", type=mujoco.mjtGeom.mjGEOM_BOX, size=[DEPTH / 2, w / 2, HEIGHT / 2], mass=mass, priority=2,
              condim=p["condim"], friction=[MU, 0.005, 0.0001], solref=[p["tc"], 1.0], solimp=list(p["solimp"]),
              contype=1, conaffinity=1)
  if not p["mocap"]:
    for side in (1, -1):
      for nm in ("x", "y", "z"):
        a = spec.add_actuator(name=f"a{side}_{nm}", target=f"f{side}_{nm}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.gaintype, a.biastype = mujoco.mjtGain.mjGAIN_FIXED, mujoco.mjtBias.mjBIAS_AFFINE
        a.gainprm = [KP] + [0.0] * 9
        a.biasprm = [0.0, -KP, -KV] + [0.0] * 7
  m = spec.compile()
  ctrl = np.zeros(m.nu)
  if not p["mocap"]:
    d_in = D_M * p["depth_mult"]
    for side in (1, -1):
      ctrl[m.actuator(f"a{side}_y").id] = -side * (0.001 + d_in)
  return m, ctrl


class Rec:
  def __init__(self, n):
    z = lambda *s: np.full(s, np.nan)                          # noqa: E731
    self.t, self.F, self.fn, self.ftz = z(n), z(n), z(n, 2), z(n, 2)
    self.vc, self.vcz, self.vpred, self.edge = z(n), z(n), z(n), z(n)
    self.vb, self.vbz, self.om, self.cpos = z(n, 3), z(n), z(n, 3), z(n, 2, 3)
    self.ncon, self.niter = np.zeros((n, 2), int), np.zeros(n, int)


def record(rec, k, m, d, og, ob, hand_sets, fist_vel, F, t):
  """Contacts of the object geom og with the two hands' geom sets; v_c from efc_vel (material slip at the contact point)."""
  pyr = m.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL
  f6 = np.zeros(6)
  vc, vcz, vp, ed = [], [], [], []
  fn, ftz, cp, nc = np.zeros(2), np.zeros(2), [[], []], np.zeros(2, int)
  R_o = d.xmat[ob].reshape(3, 3)
  for i in range(d.ncon):
    c = d.contact[i]
    g1, g2 = int(c.geom1), int(c.geom2)
    if og not in (g1, g2):
      continue
    o = g2 if g1 == og else g1
    h = 0 if o in hand_sets[0] else (1 if o in hand_sets[1] else -1)
    if h < 0 or c.efc_address < 0:
      continue
    mujoco.mj_contactForce(m, d, i, f6)
    fr = c.frame.reshape(3, 3)
    sgn = 1.0 if g2 == og else -1.0
    fw = sgn * (fr.T @ f6[:3])
    fn[h] += f6[0]
    ftz[h] += fw[2]
    nc[h] += 1
    cp[h].append(R_o.T @ (c.pos - d.xpos[ob]))
    adr, dim = c.efc_address, c.dim
    vt = np.zeros(2)
    vpk = np.zeros(2)
    zt = np.array([fr[1] @ [0, 0, 1.0], fr[2] @ [0, 0, 1.0]])
    if pyr:
      for kk in range(min(dim - 1, 2)):
        r0, r1 = adr + 2 * kk, adr + 2 * kk + 1
        muk = c.friction[kk]
        vt[kk] = (d.efc_vel[r0] - d.efc_vel[r1]) / (2 * muk)
        vpk[kk] = d.efc_R[r0] * (d.efc_force[r0] - d.efc_force[r1]) / (2 * d.efc_KBIP[r0, 1] * muk)
      kz = int(np.argmax(np.abs(zt)))
      r0, r1 = adr + 2 * kz, adr + 2 * kz + 1
      ed.append(float(d.efc_state[r0] == QUAD and d.efc_state[r1] == QUAD))
    else:
      for kk in range(min(dim - 1, 2)):
        r = adr + 1 + kk
        vt[kk] = d.efc_vel[r]
        vpk[kk] = d.efc_R[r] * d.efc_force[r] / d.efc_KBIP[r, 1]
    vc.append(float(np.hypot(*vt)))
    vcz.append(float(sgn * (vt @ zt)))
    vp.append(float(np.hypot(*vpk)))
  rec.t[k], rec.F[k] = t, F
  rec.fn[k], rec.ftz[k], rec.ncon[k] = fn, ftz, nc
  if vc:
    rec.vc[k], rec.vcz[k], rec.vpred[k] = np.mean(vc), np.mean(vcz), np.mean(vp)
    rec.edge[k] = np.mean(ed) if ed else np.nan
  for h in range(2):
    if cp[h]:
      rec.cpos[k, h] = np.mean(cp[h], axis=0)
  res = np.zeros(6)
  mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, ob, res, 0)
  rec.om[k] = res[:3]
  rec.vb[k] = res[3:] - fist_vel
  rec.vbz[k] = rec.vb[k, 2]
  rec.niter[k] = int(d.solver_niter[0])


def run_M(w, mass, v, delta=None):
  p = vparams(v)
  m, ctrl = build_M(w, mass, p, delta)
  d = mujoco.MjData(m)
  mujoco.mj_forward(m, d)
  og, ob = m.geom("obj").id, m.body("obj").id
  hands = ({m.geom("fist1").id}, {m.geom("fist-1").id})
  fb = [m.body("fist1").id, m.body("fist-1").id]
  ts = T_SETTLE["M"]
  n = int(round((ts + T_RAMP) / DT))
  rec = Rec(n)
  d.ctrl[:] = ctrl
  res = np.zeros(6)
  for k in range(n):
    t = k * DT
    F = 0.0 if t < ts else L_MAX * mass * 9.81 * (t - ts) / T_RAMP
    d.xfrc_applied[ob, 2] = -F
    mujoco.mj_step(m, d)
    fv = np.zeros(3)
    for b in fb:
      mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, res, 0)
      fv += res[3:] / 2
    record(rec, k, m, d, og, ob, hands, fv, F, t + DT)
  return rec, dict(scene="M", w=w, mass=mass, variant=v, delta=delta, iterations=p["it"])


def mocap_delta(w, mass, fn_target):
  """Penetration of the mocap fists that gives the base normal force per hand (weightless box, 0.5 s settle)."""
  lo, hi = 0.0, 0.006
  for _ in range(14):
    mid = (lo + hi) / 2
    p = vparams("MOC")
    m, _ = build_M(w, mass, p, mid)
    d = mujoco.MjData(m)
    og, ob = m.geom("obj").id, m.body("obj").id
    for _ in range(100):
      mujoco.mj_step(m, d)
    f6, fn = np.zeros(6), 0.0
    for i in range(d.ncon):
      c = d.contact[i]
      if og in (c.geom1, c.geom2) and m.geom_bodyid[c.geom1 if c.geom2 == og else c.geom2] == m.body("fist1").id:
        mujoco.mj_contactForce(m, d, i, f6)
        fn += f6[0]
    lo, hi = (mid, hi) if fn < fn_target else (lo, mid)
  return (lo + hi) / 2


# ------------------------------------------------------------------------------------------------ scene R
def run_R(w, mass, v):
  from r1adv.positive_obj import p20_hold as H
  from r1adv.positive_obj import p20_ladder as L
  from r1adv.positive_obj import p20_scene as PS
  p = vparams(v)
  ik = "/rl-agent/.agent/tmp/positive_r11p20/fist/ik/fist_g1.npz"
  grid = H.FistGrid(ik)
  m = H.build_model("fist", p["cone"], p["condim"])
  set_opt(m.opt, p)
  idx = PS.Index(m)
  lay = L.Layout(m, grid.names)
  base = [dict(stratum=20, shape=0, dims=(DEPTH, w, HEIGHT), mass=mass, mu=MU, tc=p["tc"], imp=p["imp"],
               d_in=D_R * p["depth_mult"], orient=0)]
  row = H.arm_params("fist", base, grid)[0]
  active = H.set_fields_c(m, idx, row, p["condim"])
  m.opt.impratio = p["imp"]
  og, ob = idx.obj_geom[active], idx.obj_body[active]
  m.geom_solimp[og] = p["solimp"]
  d = mujoco.MjData(m)
  q0, tgt = H.initial_q(row, grid, lay)
  H.init_state_c(m, d, idx, lay, tgt[0], q0, active, 0)
  oq, od = idx.obj_qadr[active], idx.obj_dadr[active]
  qfix = d.qpos[oq:oq + 7].copy()
  hands = (set(idx.hand_geoms[0]), set(idx.hand_geoms[1]))
  ts = T_SETTLE["R"]
  n = int(round((ts + T_RAMP) / DT))
  rec = Rec(n)
  res = np.zeros(6)
  for k in range(n):
    t = k * DT
    if k % H.DECIM == 0:
      d.ctrl[:] = tgt[min(k // H.DECIM, len(tgt) - 1)]
    F = 0.0 if t < ts else L_MAX * mass * 9.81 * (t - ts) / T_RAMP
    sup = H.support_c(m, d, ob, oq, od, qfix, t < 0.8)
    sup[2] -= F
    d.xfrc_applied[ob] = sup
    mujoco.mj_step(m, d)
    fv = np.zeros(3)
    for hb in idx.hand_bodies:
      mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, hb, res, 0)
      fv += res[3:] / 2
    record(rec, k, m, d, og, ob, hands, fv, F, t + DT)
  return rec, dict(scene="R", w=w, mass=mass, variant=v, iterations=p["it"], d_in=float(row[H.P["d_in"]]),
                   ik_ok=float(row[H.P["ik_ok"]]))


# ------------------------------------------------------------------------------------------------ measures (rules 5)
def measures(rec, meta) -> dict:
  ts = T_SETTLE[meta["scene"]]
  mass = meta["mass"]
  t = rec.t
  L_ = rec.F / (mass * 9.81)
  ramp = t > ts + 1e-9
  fn_sum = rec.fn.sum(1)
  lam = np.where(fn_sum > 0, rec.F / (MU * np.maximum(fn_sum, 1e-9)), np.inf)
  W = ramp & (L_ >= 0.4) & (L_ <= 1.0)
  lo, hi = ramp & (L_ >= 0.4) & (L_ <= 0.6), ramp & (L_ >= 0.8) & (L_ <= 1.0)
  med = lambda x, msk: float(np.nanmedian(x[msk])) if np.isfinite(x[msk]).any() else float("nan")   # noqa: E731
  v_in = med(rec.vc, W) * 1000
  rho = med(rec.vc, hi) / med(rec.vc, lo) if med(rec.vc, lo) > 0 else float("nan")
  kappa = med(rec.vc, W) / med(rec.vpred, W) if med(rec.vpred, W) > 0 else float("nan")
  cum = np.cumsum(np.where(ramp, np.nan_to_num(rec.vc), 0.0)) * DT
  k5 = np.argmax(cum >= 0.005) if (cum >= 0.005).any() else -1
  lam_star = float(lam[k5]) if k5 >= 0 else float("nan")
  acc = None
  if k5 >= 0 and k5 + 20 < len(t):
    acc = bool(np.nanmean(rec.vc[k5 + 1:k5 + 21]) > rec.vc[k5])
  vb_abs = np.linalg.norm(rec.vb, axis=1)
  upto1 = ramp & (L_ <= 1.0)
  slip_ramp = float(np.nansum(np.abs(rec.vbz[upto1])) * DT * 1000)
  slip_all = float(np.nansum(np.abs(rec.vbz[ramp])) * DT * 1000)
  rot = float(np.mean((rec.vc[W] < 0.3 * vb_abs[W]) & (vb_abs[W] > 1e-3))) if W.any() else float("nan")
  cp = rec.cpos.mean(1)
  jump = float(np.nanmax(np.linalg.norm(np.diff(cp[W], axis=0), axis=1)) * 1000) if W.sum() > 1 else float("nan")
  niter_cap = float(np.mean(rec.niter >= meta["iterations"]))
  settle = float(np.nansum(vb_abs[~ramp]) * DT * 1000)
  brace = float(np.nanmedian(np.abs(rec.ftz[W]).sum(1) / np.maximum(np.abs(rec.ftz[W].sum(1)), 1e-9))) if W.any() else float("nan")
  return dict(v_in_mm_s=v_in, rho=rho, edge=med(rec.edge, W), kappa=kappa, lam_star=lam_star, accel_after=acc,
              slip_ramp_mm=slip_ramp, slip_all_mm=slip_all, rot=rot, jump_mm=jump, niter_cap=niter_cap, settle_mm=settle,
              brace=brace, fn_hand_ramp0=[float(x) for x in np.nanmedian(rec.fn[ramp & (L_ < 0.05)], axis=0)] if (ramp & (L_ < 0.05)).any() else None,
              lam_max=float(np.nanmax(lam[ramp])) if ramp.any() else float("nan"),
              ncon_med=[float(x) for x in np.median(rec.ncon[W], axis=0)] if W.any() else None)


def _job(args):
  scene, w, mass, v, delta = args
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  rec, meta = run_M(w, mass, v, delta) if scene == "M" else run_R(w, mass, v)
  return meta, measures(rec, meta), rec


def main(argv):
  out, scenes = argv[0], (argv[1] if len(argv) > 1 and not argv[1].startswith("--") else "MR")
  workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 16
  os.makedirs(out, exist_ok=True)
  from concurrent.futures import ProcessPoolExecutor
  t0 = time.time()
  rows = []
  with ProcessPoolExecutor(max_workers=workers) as ex:
    if "M" in scenes:                                       # base first (its normal force sets the mocap penetration)
      base = {o: r for o, r in zip(OBJECTS, ex.map(_job, [("M", w, ms, "B", None) for w, ms in OBJECTS]))}
      deltas = list(ex.map(_delta, [(w, ms, float(np.mean(base[(w, ms)][1]["fn_hand_ramp0"]))) for w, ms in OBJECTS]))
      jobs = [("M", w, ms, v, deltas[i] if v == "MOC" else None) for i, (w, ms) in enumerate(OBJECTS) for v in VARIANTS if v != "B"]
      done = list(base.values()) + list(ex.map(_job, jobs))
      for meta, mm, rec in done:
        rows.append(dict(meta, **mm))
        np.savez_compressed(os.path.join(out, f"M_w{meta['w']}_m{meta['mass']}_{meta['variant']}.npz"), **vars(rec))
    if "R" in scenes:
      jobs = [("R", w, ms, v, None) for w, ms in OBJECTS for v in VARIANTS if v != "MOC"]
      for meta, mm, rec in ex.map(_job, jobs):
        rows.append(dict(meta, **mm))
        np.savez_compressed(os.path.join(out, f"R_w{meta['w']}_m{meta['mass']}_{meta['variant']}.npz"), **vars(rec))
  json.dump(dict(rules="rounds/R04/positive/r11_hold/slip_cause_rules.md", wall_s=time.time() - t0, rows=rows),
            open(os.path.join(out, "measures.json"), "w"), indent=1, default=float)
  print("done", len(rows), f"{time.time() - t0:.0f} s")


def _delta(args):
  w, mass, fn_target = args
  return mocap_delta(w, mass, fn_target)


if __name__ == "__main__":
  main(sys.argv[1:])
