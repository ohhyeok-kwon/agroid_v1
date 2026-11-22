"""Positive 24 (E1 c09) fist hold grid: grip shape x contact representation x mass x width, base shake after 3 s still
(rounds/R04/positive/e1_c09/rules.md). C MuJoCo on the CPU. Built on positive 20 p20_hold (timeline, support, legacy
script) and p20_gate (aligned squeeze: contact-point pose + joint offsets K_p^-1 J_p^T F, weight feed-forward at release).

Grips
  H   hands only (primary H), the hold test's latest start (p20_gate aligned squeeze, IK grid fist_g1 at x 0.36, z 0.76)
  C   hug against the chest (main C): back face on the torso capsule, each fist hooks a front vertical edge with a
      diagonal normal, so the fists' normal forces load the chest (p24_hug_ik mode 'edge')
  Cs  side arm: box at the chest, fists squeeze the side faces, chest not loaded by design (p24_hug_ik mode 'side')
  H0  side arm (record reproduction; rule 1 frame check): p20_hold legacy deep-joint-target squeeze (d_in, no feed-forward)
Contacts (object geom carries priority 3: its condim / friction / solref / solimp are used)
  K1 base   stock fist capsule, pyramidal, condim 3, impratio 1
  K2 pads   positive 20 fistpads body (fist box + 3 x 3 sphere pads), pyramidal, condim 3, impratio 1
  K3 ell10  stock fist capsule, elliptic, condim 3, impratio 10
Timeline (dt 5 ms, control 50 Hz): squeeze ramp [0, 0.5) s, soft support (weight + 4 Hz spring) [0, 0.8), weight only
[0.8, 1.2), release 1.2 (weight feed-forward per hand: H, Cs m g / 2; C m g / (2 + sqrt2)), still [1.2, 4.2), base shake [4.2, 7.2) (p24_shake:
gravity modulation g_eff = -(9.81 + a(t)) z), end 7.2 s. Delay variant: PD targets applied 60 ms (12 steps) late.

  python -m r1adv.positive_obj.p24_hold run <out_dir> <shake.npz> <hug_ik_dir> [--workers N] [--limit N] [--tend T]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import mujoco
import numpy as np

from r1adv.positive_obj import p20_fist as F
from r1adv.positive_obj import p20_gate as GT
from r1adv.positive_obj import p20_hold as H
from r1adv.positive_obj import p20_ladder as L
from r1adv.positive_obj import p20_readout as R
from r1adv.positive_obj import p20_scene as PS
from r1adv.positive_obj import p24_hug_ik as HK

RULES = "rounds/R04/positive/e1_c09/rules.md"
IK_H = "/rl-agent/.agent/tmp/positive_r11p20/fist/ik/fist_g1.npz"
DT, DECIM, G0 = 0.005, 4, 9.81
T_SQ, T_A, T_REL, T_SH0, T_END = 0.5, 0.8, 1.2, 4.2, 7.2
NSTEP = int(round(T_END / DT))                                  # 1440
K_A, K_REL, K_SH0 = int(round(T_A / DT)), int(round(T_REL / DT)), int(round(T_SH0 / DT))     # 160, 240, 840
N50 = NSTEP // DECIM                                            # 360 (50 Hz samples, state after step 4c+3)
C_REL, C_22, C_SH0, C_END = 59, 109, 209, 359                   # 50 Hz indices of t = 1.2, 2.2, 4.2, 7.2
DEPTH, HEIGHT, MU, BUDGET, TC = 0.125, 0.25, 0.5, 2.0, 0.02
SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)
CONTACTS = {"K1": ("fist", "pyramidal", 1.0), "K2": ("fistpads", "pyramidal", 1.0), "K3": ("fist", "elliptic", 10.0)}
GRIPS = ("H", "C", "Cs", "H0")
HUG_MODE = {"C": "edge", "Cs": "side"}
MASSES = (0.2, 0.6, 1.22, 2.0)
WIDTHS = (0.2, 0.3, 0.4)
N_WORLD, SEED = 8, 2411
SHAKES = ("trace", "sin493", "sin196")
DELAYS = (0, 12)                                                # physics steps (0, 60 ms)
JIT = dict(dx=0.003, dy=0.003, dz=0.010)                        # uniform half-ranges; world 0 of every cell = nominal
SQ2 = float(np.sqrt(2.0))
CHEST_LOAD_MIN = 0.25                                           # C start: mean chest normal force >= 25 % of the design sqrt2 f_n


def jitter_table(seed: int = SEED) -> dict:
  rng = np.random.Generator(np.random.PCG64(seed))
  out = {}
  for m in MASSES:
    for w in WIDTHS:
      rows = [np.zeros(3)]
      for _ in range(N_WORLD - 1):
        rows.append(np.array([rng.uniform(-JIT["dx"], JIT["dx"]), rng.uniform(-JIT["dy"], JIT["dy"]), rng.uniform(-JIT["dz"], JIT["dz"])]))
      out[(m, w)] = np.array(rows)
  return out


def job_list() -> list[tuple]:
  jobs = []
  for grip in GRIPS:
    for ck in CONTACTS:
      for w in WIDTHS:
        for m in MASSES:
          for j in range(N_WORLD):
            for sh in SHAKES:
              for dl in DELAYS:
                jobs.append((grip, ck, w, m, j, sh, dl))
  return jobs


# ------------------------------------------------------------------------------------------------ per-process cache
_CACHE: dict = {}


def model_for(arm: str, cone: str):
  key = (arm, cone)
  if key not in _CACHE:
    m = H.build_model(arm, cone, 3)
    idx = PS.Index(m)
    grid = H.FistGrid(IK_H)
    lay = L.Layout(m, grid.names)
    _CACHE[key] = (m, idx, lay, grid, mujoco.MjData(m))
  return _CACHE[key]


def hug_pose(ik_dir: str, mode: str, arm: str, w: float):
  key = ("hug", mode, arm)
  if key not in _CACHE:
    _CACHE[key] = dict(np.load(os.path.join(ik_dir, f"hug_{mode}_{arm}.npz")))
  z = _CACHE[key]
  i = int(np.argmin(np.abs(z["widths"] - w)))
  names = [str(n) for n in z["joint_names"]]
  return z["q_start"][i], z["q_contact"][i], bool(z["feasible"][i]), names


def act_vector(q_named: np.ndarray, names: list[str], lay) -> np.ndarray:
  """Joint vector in the IK model's joint order -> actuator order of the scene model."""
  out = np.zeros(lay.nu)
  for a, jn in enumerate(lay.act_joint):
    out[a] = q_named[names.index(jn)]
  return out


def gravity_offsets(m, lay, q_act) -> np.ndarray:
  d = mujoco.MjData(m)
  d.qpos[lay.act_qadr] = q_act
  mujoco.mj_forward(m, d)
  kp = m.actuator_gainprm[:, 0]
  dof = np.array([int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in range(m.nu)])
  return d.qfrc_bias[dof] / kp


def patch_gap_offsets(m, lay, q_act, feature, nout):
  """Joint offsets that put each patch site on its contact feature (signed gap along the feature's outward normal removed,
  least squares on the arm dofs) -- so the squeeze force equals the design force for every world."""
  d = mujoco.MjData(m)
  d.qpos[lay.act_qadr] = q_act
  mujoco.mj_forward(m, d)
  dq = np.zeros(m.nu)
  gaps = []
  for side in ("left", "right"):
    sid = m.site(f"robot/{side}_patch").id
    gap = float((d.site_xpos[sid] - feature[side]) @ nout[side])          # > 0: patch outside the feature
    gaps.append(gap)
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac, None, sid)
    acts, dofs = GT.arm_dofs(m, lay, side)
    dq[acts] = np.linalg.lstsq(jac[:, dofs], -gap * nout[side], rcond=None)[0]
  return dq, gaps


def force_offsets(m, lay, q_act, f_n, n_in, ff_up):
  """p20_gate.offsets for arbitrary push directions: K_p^-1 J_p^T (f_n n_in) per hand, weight feed-forward K_p^-1 J_p^T
  (ff_up z), arm capability F_max per hand along n_in."""
  d = mujoco.MjData(m)
  d.qpos[lay.act_qadr] = q_act
  mujoco.mj_forward(m, d)
  dq_sq, dq_w, fmax = np.zeros(m.nu), np.zeros(m.nu), []
  for side in ("left", "right"):
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac, None, m.site(f"robot/{side}_patch").id)
    acts, dofs = GT.arm_dofs(m, lay, side)
    kp = m.actuator_gainprm[acts, 0]
    tau_n = jac[:, dofs].T @ n_in[side]
    dq_sq[acts] = f_n * tau_n / kp
    dq_w[acts] = (jac[:, dofs].T @ np.array([0.0, 0.0, ff_up])) / kp
    eff = lay.effort[acts]
    grav = np.abs(d.qfrc_bias[dofs])
    ok = np.abs(tau_n) > 1e-6
    fmax.append(float(np.min((eff[ok] - grav[ok]) / np.abs(tau_n[ok]))))
  return dq_sq, dq_w, fmax


# ------------------------------------------------------------------------------------------------ one world
def setup(job, shake_prof, ik_dir):
  grip, ck, w, mass, j, sh, dl = job
  arm, cone, imp = CONTACTS[ck]
  m, idx, lay, grid, d = model_for(arm, cone)
  jit = jitter_table()[(mass, w)][j]
  d_in = 0.0
  if grip == "H0":
    d_req = mass * G0 / (MU * H.K_PD)
    d_in = float(np.clip(max(d_req, 0.01), 0.01, max(H.d_max(w), 0.01)))
  row = H.arm_params(arm, [dict(stratum=20, shape=0, dims=(DEPTH, w, HEIGHT), mass=mass, mu=MU, tc=TC, imp=imp, d_in=d_in, orient=0)], grid)[0]
  active = H.set_fields_c(m, idx, row, 3)
  m.opt.impratio = imp
  og, ob = idx.obj_geom[active], idx.obj_body[active]
  m.geom_solimp[og] = SOLIMP
  m.geom_solref[og] = (TC, 1.0)
  m.opt.gravity[:] = (0.0, 0.0, -G0)
  f_n = BUDGET * mass * G0 / (MU * (2 + SQ2)) if grip == "C" else BUDGET * mass * G0 / (2 * MU)
  ff_up = mass * G0 / (2 + SQ2) if grip == "C" else mass * G0 / 2
  info = dict(grip=grip, contact=ck, arm=arm, cone=cone, impratio=imp, w=w, mass=mass, world=j, shake=sh, delay_steps=dl,
              jitter=jit.tolist(), f_n_design=f_n if grip != "H0" else float(H.K_PD * d_in), d_in=d_in, feasible=True,
              chest_n_design=SQ2 * f_n if grip == "C" else 0.0, ff_up=ff_up if grip != "H0" else 0.0)
  if grip in ("H", "H0"):
    info["feasible"] = bool(row[H.P["ik_ok"]] > 0.5)           # IK grid residual <= 5 mm in its s range (p20_hold)
  if grip == "H0":
    q0, tgt = H.initial_q(row, grid, lay)
    H.init_state_c(m, d, idx, lay, tgt[0], q0, active, 0)
    sched = dict(kind="legacy", tgt=tgt)
    centre = d.qpos[idx.obj_qadr[active]:idx.obj_qadr[active] + 3].copy()
  else:
    if grip == "H":
      s_c, s_0 = row[H.P["s_sq"]], row[H.P["s_start"]]
      Q0 = np.zeros(lay.nu)
      Q1 = np.zeros(lay.nu)
      Q0[lay.ik_act] = grid.joints(np.array([s_0]), np.zeros(1), np.zeros(1), "Q")[0][lay.ik_from]
      Q1[lay.ik_act] = grid.joints(np.array([s_c]), np.zeros(1), np.zeros(1), "Q")[0][lay.ik_from]
      T0 = np.zeros(lay.nu)
      T1 = np.zeros(lay.nu)
      T0[lay.ik_act] = grid.joints(np.array([s_0]), np.zeros(1), np.zeros(1), "T")[0][lay.ik_from]
      T1[lay.ik_act] = grid.joints(np.array([s_c]), np.zeros(1), np.zeros(1), "T")[0][lay.ik_from]
      H.init_state_c(m, d, idx, lay, T0, Q0, active, 0)
      centre = d.qpos[idx.obj_qadr[active]:idx.obj_qadr[active] + 3].copy()      # patch midpoint at the start pose
      nout = {"left": np.array([0.0, 1.0, 0.0]), "right": np.array([0.0, -1.0, 0.0])}
      feature = {sd: centre + (w / 2) * nout[sd] for sd in ("left", "right")}
    else:
      mode = HUG_MODE[grip]
      qs, qc, feas, names = hug_pose(ik_dir, mode, arm, w)
      info["feasible"] = feas
      Q0, Q1 = act_vector(qs, names, lay), act_vector(qc, names, lay)
      T0 = Q0 + gravity_offsets(m, lay, Q0)
      T1 = Q1 + gravity_offsets(m, lay, Q1)
      H.init_state_c(m, d, idx, lay, T0, Q0, active, 0)
      centre = HK.box_centre(w).copy()
      nout = {sd: HK.n_out(mode, sd) for sd in ("left", "right")}
      feature = {sd: HK.contact_point(mode, w, sd) for sd in ("left", "right")}
    dq_gap, gaps = patch_gap_offsets(m, lay, Q1, feature, nout)
    dq_sq, dq_w, fmax = force_offsets(m, lay, Q1, f_n, {sd: -nout[sd] for sd in ("left", "right")}, ff_up)
    info.update(gap_contact_pose=gaps, fmax=fmax)
    sched = dict(kind="aligned", T0=T0, T1=T1, dq=dq_gap + dq_sq, dq_w=dq_w)
  # object start pose: nominal centre + jitter (C: no x jitter -- the back face stays 1 mm off the chest)
  oq = idx.obj_qadr[active]
  jj = jit.copy()
  if grip in ("C", "Cs"):
    jj[0] = 0.0
  d.qpos[oq:oq + 3] = centre + jj
  d.qpos[oq + 3:oq + 7] = (1.0, 0.0, 0.0, 0.0)
  d.qvel[:] = 0.0
  mujoco.mj_forward(m, d)
  return m, d, idx, lay, active, sched, info


def target_at(sched, c: int) -> np.ndarray:
  if sched["kind"] == "legacy":
    return sched["tgt"][min(max(c, 0), len(sched["tgt"]) - 1)]
  t = max(c, 0) * DT * DECIM
  a = L.ramp(np.array([t]), 0.0, T_SQ)[0]
  out = sched["T0"] + (sched["T1"] - sched["T0"]) * a + sched["dq"] * a
  if t >= T_REL - 1e-9:
    out = out + sched["dq_w"]
  return out


def run_world(job, shake_prof, ik_dir, tend=None):
  grip, ck, w, mass, j, sh, dl = job
  m, d, idx, lay, active, sched, info = setup(job, shake_prof, ik_dir)
  og, ob = idx.obj_geom[active], idx.obj_body[active]
  oq, od = idx.obj_qadr[active], idx.obj_dadr[active]
  gate = H.init_gate(m, d, idx, active)
  qfix = d.qpos[oq:oq + 7].copy()
  torso = m.geom("robot/torso_collision").id
  hand_of = {g: k for k, gs in idx.hand_geoms.items() for g in gs}
  robot = set(idx.robot_geoms)
  floor = set(idx.floor_geoms)
  caps = [m.geom(f"robot/{s}_hand_collision").id for s in ("left", "right")]
  nstep = NSTEP if tend is None else int(round(tend / DT))
  prof = shake_prof[sh]
  fn = np.zeros((nstep, 2), np.float32)
  nc = np.zeros((nstep, 2), np.int16)
  ch_n = np.zeros(nstep, np.int16)
  ch_f = np.zeros(nstep, np.float32)
  other = np.zeros(nstep, np.int16)
  flo = np.zeros(nstep, np.int16)
  dmin = np.ones((nstep, 2), np.float32)
  n50 = nstep // DECIM
  obj_p, obj_q, hand_p = np.zeros((n50, 3)), np.zeros((n50, 4)), np.zeros((n50, 2, 3))
  fr = np.zeros(6)
  align = None
  v_rel = None
  for k in range(nstep):
    if k % DECIM == 0:
      d.ctrl[:] = target_at(sched, (k - dl) // DECIM if k >= dl else 0)
    m.opt.gravity[2] = -(G0 + (prof[k - K_SH0] if K_SH0 <= k < K_SH0 + len(prof) else 0.0))
    d.xfrc_applied[ob] = H.support_c(m, d, ob, oq, od, qfix, k < K_A) if k < K_REL else 0.0
    mujoco.mj_step(m, d)
    for i in range(d.ncon):
      con = d.contact[i]
      g1, g2 = int(con.geom1), int(con.geom2)
      if og not in (g1, g2):
        continue
      o = g2 if g1 == og else g1
      mujoco.mj_contactForce(m, d, i, fr)
      if o in hand_of:
        h = hand_of[o]
        nc[k, h] += 1
        fn[k, h] += abs(fr[0])
        dmin[k, h] = min(dmin[k, h], float(con.dist))
      elif o == torso:
        ch_n[k] += 1
        ch_f[k] += abs(fr[0])
      elif o in robot:
        other[k] += 1
      elif o in floor:
        flo[k] += 1
    if k == K_REL - 1:
      v_rel = float(np.linalg.norm(d.qvel[od:od + 3]))
      cap = [float(np.degrees(np.arcsin(min(1.0, abs(d.geom_xmat[g].reshape(3, 3)[1, 2]))))) for g in caps]
      face = []
      for hb, s_in in zip(idx.hand_bodies, (-1.0, 1.0)):
        zf = d.xmat[hb].reshape(3, 3)[:, 2]
        face.append(float(np.degrees(np.arccos(np.clip(zf @ np.array([0.0, s_in, 0.0]), -1.0, 1.0)))))
      align = cap if info["arm"] == "fist" else face
      if grip == "C":                                          # edge hook: patch normal vs the designed diagonal
        align = [float(np.degrees(np.arccos(np.clip(d.site_xmat[m.site(f"robot/{sd}_patch").id].reshape(3, 3)[:, 2]
                                                    @ -HK.n_out("edge", sd), -1.0, 1.0)))) for sd in ("left", "right")]
    if k % DECIM == DECIM - 1:
      c = k // DECIM
      obj_p[c], obj_q[c] = d.xpos[ob], d.xquat[ob]
      hand_p[c] = [d.xpos[b] for b in idx.hand_bodies]
  m.opt.gravity[:] = (0.0, 0.0, -G0)
  tr = dict(fn=fn, nc=nc, ch_n=ch_n, ch_f=ch_f, other=other, flo=flo, dmin=dmin, obj_p=obj_p, obj_q=obj_q, hand_p=hand_p)
  info.update(init_gate=list(gate), v_rel=v_rel, align=align)
  return info, tr


# ------------------------------------------------------------------------------------------------ frozen metrics
def _mask(nstep, a, b):
  S = (np.arange(nstep) + 1) * DT
  return (S >= a - 1e-9) & (S < b - 1e-9)


def rel(obj_p, hand_p, c0, c1) -> float:
  return float(H_rel(obj_p[None], hand_p[None], c0, c1)[0])


def H_rel(op, hp, c0, c1):
  o0, R0 = R.pair_frame(hp, c0)
  o1, R1 = R.pair_frame(hp, c1)
  p0 = np.einsum("nji,nj->ni", R0, op[:, c0] - o0)
  p1 = np.einsum("nji,nj->ni", R1, op[:, c1] - o1)
  return np.linalg.norm(p1 - p0, axis=1) * 1000.0


def cv(x) -> float:
  mu = float(np.mean(x))
  return float(np.std(x) / mu * 100.0) if mu > 1e-9 else float("inf")


def metrics(info, tr) -> dict:
  nstep = len(tr["fn"])
  grip = info["grip"]
  W = _mask(nstep, T_REL, T_END + 1e-6)
  Wpre = _mask(nstep, T_REL, T_SH0)
  S = _mask(nstep, T_REL - 0.1, T_REL)
  nc, fn = tr["nc"], tr["fn"].astype(np.float64)
  both = (nc[:, 0] > 0) & (nc[:, 1] > 0)
  op, hp = tr["obj_p"], tr["hand_p"]
  slips = dict(slip_onset=rel(op, hp, C_REL, C_22), slip_main=rel(op, hp, C_22, C_END), slip_still=rel(op, hp, C_22, C_SH0),
               slip_shake=rel(op, hp, C_SH0, C_END), slip_total=rel(op, hp, C_REL, C_END), slip_pre=rel(op, hp, C_REL, C_SH0))
  pen = float(max(-tr["dmin"].min(), 0.0))
  c1 = bool((tr["flo"][W] == 0).all())
  c2 = bool(op[C_END, 2] >= op[C_REL, 2] - 0.05)
  c3 = slips["slip_total"] <= 50.0
  c4 = bool(both[W].mean() >= 0.95)
  if grip == "C":
    c5 = bool((tr["ch_n"][W] > 0).mean() >= 0.95)          # the hug keeps the chest contact (rules 5)
  elif grip == "Cs":
    c5 = True                                                # side arm: body support allowed, not a hug
  else:
    c5 = bool(((tr["other"][W] + tr["ch_n"][W]) > 0).mean() <= 0.05)
  c6 = pen <= 0.005
  held = c1 and c2 and c3 and c4 and c5 and c6
  # held over the pre-shake part only (same conditions on [1.2, 4.2))
  held_pre = bool((tr["flo"][Wpre] == 0).all() and op[C_SH0, 2] >= op[C_REL, 2] - 0.05 and slips["slip_pre"] <= 50.0
                  and both[Wpre].mean() >= 0.95 and pen <= 0.005
                  and ((tr["ch_n"][Wpre] > 0).mean() >= 0.95 if grip == "C" else True if grip == "Cs"
                       else (((tr["other"][Wpre] + tr["ch_n"][Wpre]) > 0).mean() <= 0.05)))
  start_both = bool(both[S].all())
  start_chest = bool((tr["ch_n"][S] > 0).all() and tr["ch_f"][S].mean() >= CHEST_LOAD_MIN * info["chest_n_design"]) if grip == "C" else True
  g = info["init_gate"]
  gate_ok = g[0] <= 0.002 and g[1] <= 0.001 and g[2] <= 0.001
  start_valid = bool(start_both and start_chest and info["v_rel"] < 0.05 and gate_ok and info["feasible"])
  fint = (fn[:, 0] + fn[:, 1]) / 2
  Ws = _mask(nstep, 2.2, T_SH0)
  Wk = _mask(nstep, T_SH0, T_END + 1e-6)
  Wsc, Wkc = Ws & both, Wk & both                              # rule 4: span where both hands touch the object
  hf = R._bands(fint[Wk][:, None], 1.0 / DT, [(10.0, 100.0)])[0] / max(fint[Wk].mean(), 1e-9)
  M_start = MU * fn[S].sum(1).mean() / (info["mass"] * G0)
  M_start_total = MU * (fn[S].sum(1) + tr["ch_f"][S]).mean() / (info["mass"] * G0)
  # first time (50 Hz) the pair-frame displacement from release exceeds 50 mm
  disp = np.array([rel(op, hp, C_REL, c) for c in range(C_REL, C_END + 1)])
  over = np.nonzero(disp > 50.0)[0]
  t_loss = float((C_REL + over[0] + 1) * DT * DECIM) if len(over) else float("nan")
  out = dict(info)
  out.update(slips)
  out.update(c1=c1, c2=c2, c3=c3, c4=c4, c5=c5, c6=c6, held=held, held_pre=held_pre, start_valid=start_valid,
             start_both=start_both, start_chest=start_chest, gate_ok=gate_ok, pen_max_mm=pen * 1000.0,
             nc_release=[int(x) for x in nc[K_REL - 1]], nc_min_still=[int(x) for x in nc[Wpre].min(0)],
             one_point_frac=float(((nc[W] == 1).any(1)).mean()), chest_release=int(tr["ch_n"][K_REL - 1]),
             chest_frac=float((tr["ch_n"][W] > 0).mean()), other_frac=float((tr["other"][W] > 0).mean()),
             chest_f_release=float(tr["ch_f"][K_REL - 1]), fn_release=[float(x) for x in fn[K_REL - 1]],
             M_start=float(M_start), M_start_total=float(M_start_total), fint_mean_still=float(fint[Ws].mean()), fint_mean_shake=float(fint[Wk].mean()),
             fint_cv_still=cv(fint[Ws]), fint_cv_shake=cv(fint[Wk]), fint_hf_shake=float(hf * 100.0), t_loss=t_loss,
             fint_cv_shake_c=cv(fint[Wkc]) if Wkc.sum() >= 100 else float("nan"),
             fint_cv_still_c=cv(fint[Wsc]) if Wsc.sum() >= 100 else float("nan"),
             chest_f_mean_still=float(tr["ch_f"][Ws].mean()),
             drop_mm=float((op[C_REL, 2] - op[C_END, 2]) * 1000.0),
             tilt_end_deg=float(np.degrees(2 * np.arccos(np.clip(abs(tr["obj_q"][C_END, 0]), -1, 1)))))
  return out


def start_metrics(info, tr) -> dict:
  """Dev check (rules 9): start-phase mechanics only (t <= 1.2 s), no hold outcome."""
  k = K_REL - 1
  S = _mask(len(tr["fn"]), T_REL - 0.1, T_REL)
  fn = tr["fn"].astype(np.float64)
  out = dict(info)
  out.update(nc_release=[int(x) for x in tr["nc"][k]], chest_release=int(tr["ch_n"][k]), other_release=int(tr["other"][k]),
             fn_release=[float(x) for x in fn[k]], M_start=float(MU * fn[S].sum(1).mean() / (info["mass"] * G0)),
             start_both=bool(((tr["nc"][S] > 0).all(1)).all()), chest_f_start=float(tr["ch_f"][S].mean()),
             M_start_total=float(MU * (fn[S].sum(1) + tr["ch_f"][S]).mean() / (info["mass"] * G0)),
             chest_all=bool((tr["ch_n"][S] > 0).all()))
  return out


# ------------------------------------------------------------------------------------------------ driver
def _job(args):
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  job, shake_path, ik_dir, tend = args
  prof = _CACHE.get("shake")
  if prof is None:
    prof = {k: v for k, v in np.load(shake_path).items()}
    _CACHE["shake"] = prof
  info, tr = run_world(job, prof, ik_dir, tend)
  row = metrics(info, tr) if tend is None else start_metrics(info, tr)
  keep = tr if job[4] == 0 else None
  return row, keep


def main(argv):
  cmd, out_dir, shake_path, ik_dir = argv[0], argv[1], argv[2], argv[3]
  rest = argv[4:]
  workers = int(rest[rest.index("--workers") + 1]) if "--workers" in rest else 24
  limit = int(rest[rest.index("--limit") + 1]) if "--limit" in rest else None
  tend = float(rest[rest.index("--tend") + 1]) if "--tend" in rest else None
  os.makedirs(out_dir, exist_ok=True)
  jobs = job_list()
  if "--dev" in rest:                                   # rules 9 dev check: nominal worlds, one shake / delay, start phase only
    jobs = [jb for jb in jobs if jb[4] == 0 and jb[5] == "trace" and jb[6] == 0]
  if limit is not None:
    jobs = jobs[:limit]
  from concurrent.futures import ProcessPoolExecutor
  t0 = time.time()
  rows, raw = [], {}
  with ProcessPoolExecutor(max_workers=workers) as ex:
    for job, (row, keep) in zip(jobs, ex.map(_job, [(jb, shake_path, ik_dir, tend) for jb in jobs], chunksize=8)):
      rows.append(row)
      if keep is not None:
        raw["|".join(str(x) for x in job)] = keep
  meta = dict(cmd=cmd, rules=RULES, n_jobs=len(jobs), wall_s=time.time() - t0, mujoco=mujoco.__version__, opt=L.SIM_OPT,
              shake=shake_path, shake_sha256=hashlib.sha256(open(shake_path, "rb").read()).hexdigest(), ik_dir=ik_dir,
              code_sha256={f: hashlib.sha256(open(os.path.join(os.path.dirname(__file__), f), "rb").read()).hexdigest()
                           for f in ("p24_hold.py", "p24_hug_ik.py", "p24_shake.py", "p20_hold.py", "p20_gate.py", "p20_fist.py",
                                     "p20_scene.py", "p20_ladder.py", "p20_readout.py")})
  with open(os.path.join(out_dir, "rows.json"), "w") as f:
    json.dump(dict(meta=meta, rows=rows), f, default=float)
  flat = {}
  for key, tr in raw.items():
    for k, v in tr.items():
      flat[f"{key}|{k}"] = v
  np.savez_compressed(os.path.join(out_dir, "raw_nominal.npz"), **flat)
  print(f"done {len(rows)} worlds in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
  main(sys.argv[1:])
