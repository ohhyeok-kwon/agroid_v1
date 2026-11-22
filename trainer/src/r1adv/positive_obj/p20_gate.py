"""Positive 20 contact gate test (rounds/R04/positive/r11_hold/gate_rules.md): stock R1 fist, aligned squeeze (contact-point
IK pose + joint offsets K_p^-1 J_p^T F for the squeeze force, weight feed-forward at release), pair contact parameters x
pyramidal impratio 1 / 10. C MuJoCo on the CPU.

  python -m r1adv.positive_obj.p20_gate <out_dir> [--side] [--arm fist|fistpads|fistfit] [--workers N]
(gate_rules.md A1: the same gates and grid with the collider changed; face alignment for box / pad fists = angle between the
fist face normal (hand-frame z) and the object face normal.)
Force convention: F is the force the hand applies to the object; the PD target offset is K_p^-1 J_p^T F (squeeze: f_n into
the object; weight feed-forward: +m g / 2 upward, i.e. the reaction -m g / 2 z on the hand is cancelled).
"""
from __future__ import annotations

import json
import os
import sys
import time

import mujoco
import numpy as np

from r1adv.positive_obj import p20_hold as H
from r1adv.positive_obj import p20_hold_readout as HR
from r1adv.positive_obj import p20_ladder as L
from r1adv.positive_obj import p20_readout as R
from r1adv.positive_obj import p20_scene as PS

IK = "/rl-agent/.agent/tmp/positive_r11p20/fist/ik/fist_g1.npz"
DT, DECIM = H.DT, H.DECIM
DEPTH, HEIGHT, MU, G = 0.125, 0.25, 0.5, 9.81
BASE_IMP = (0.9, 0.95, 0.001, 0.5, 2.0)
SETTINGS = [("pyr1_base", "pyramidal", 1.0, BASE_IMP, 0.02),
            ("pyr1_d95", "pyramidal", 1.0, (0.95, 0.95, 0.001, 0.5, 2.0), 0.010),
            ("pyr1_d99", "pyramidal", 1.0, (0.99, 0.99, 0.001, 0.5, 2.0), 0.010),
            ("pyr10_base", "pyramidal", 10.0, BASE_IMP, 0.02),
            ("pyr10_d95", "pyramidal", 10.0, (0.95, 0.95, 0.001, 0.5, 2.0), 0.010),
            ("pyr10_d99", "pyramidal", 10.0, (0.99, 0.99, 0.001, 0.5, 2.0), 0.010)]
SIDE = ("ell10_base", "elliptic", 10.0, BASE_IMP, 0.02)
WIDTHS = (0.20, 0.30, 0.40)
WORLDS = [("G1", w, 1.22, 2.0) for w in WIDTHS] + [("ref", w, m, 2.0) for w in WIDTHS for m in (0.2, 0.6)] + \
         [("G2", w, 1.22, 0.7) for w in WIDTHS]
C_REL, C_03, C_END = 59, 74, 559                # 50 Hz indices of t = 1.2, 1.5, 11.2


def arm_dofs(m, lay, side):
  acts = [a for a in lay.arm_act if lay.act_joint[a].startswith(side)]
  return acts, [int(m.jnt_dofadr[m.actuator_trnid[a, 0]]) for a in acts]


def offsets(m, lay, q_pose, mass, f_n):
  """Joint offsets for the squeeze force and the weight feed-forward, and the arm capability F_max per hand."""
  d = mujoco.MjData(m)
  d.qpos[lay.act_qadr] = q_pose
  mujoco.mj_forward(m, d)                                        # kinematics + bias forces (static: qvel 0)
  dq_sq, dq_w, fmax = np.zeros(m.nu), np.zeros(m.nu), []
  for side, s_in in (("left", -1.0), ("right", 1.0)):
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSite(m, d, jac, None, m.site(f"robot/{side}_patch").id)
    acts, dofs = arm_dofs(m, lay, side)
    kp = m.actuator_gainprm[acts, 0]
    n_in = np.array([0.0, s_in, 0.0])
    tau_n = jac[:, dofs].T @ n_in
    tau_w = jac[:, dofs].T @ np.array([0.0, 0.0, mass * G / 2])
    dq_sq[acts] = f_n * tau_n / kp
    dq_w[acts] = tau_w / kp
    eff = lay.effort[acts]
    grav = np.abs(d.qfrc_bias[dofs])
    ok = np.abs(tau_n) > 1e-6
    fmax.append(float(np.min((eff[ok] - grav[ok]) / np.abs(tau_n[ok]))))
  return dq_sq, dq_w, fmax


def run_world(setting, world, arm="fist"):
  name, cone, imp, solimp, tc = setting
  kind, w, mass, budget = world
  grid = H.FistGrid(IK)
  m = H.build_model(arm, cone, 3)
  idx = PS.Index(m)
  lay = L.Layout(m, grid.names)
  row = H.arm_params(arm, [dict(stratum=20, shape=0, dims=(DEPTH, w, HEIGHT), mass=mass, mu=MU, tc=tc, imp=imp, d_in=0.0,
                                orient=0)], grid)[0]
  active = H.set_fields_c(m, idx, row, 3)
  m.opt.impratio = imp
  og, ob = idx.obj_geom[active], idx.obj_body[active]
  m.geom_solimp[og] = solimp
  m.geom_solref[og] = (tc, 1.0)
  f_n = budget * mass * G / (2 * MU)
  s_c, s_0 = row[H.P["s_sq"]], row[H.P["s_start"]]
  qQ = grid.joints(np.array([s_c]), np.zeros(1), np.zeros(1), "Q")[0]
  qT_c = grid.joints(np.array([s_c]), np.zeros(1), np.zeros(1), "T")[0]
  qT_0 = grid.joints(np.array([s_0]), np.zeros(1), np.zeros(1), "T")[0]
  tq_c, tq_0 = np.zeros(m.nu), np.zeros(m.nu)
  tq_c[lay.ik_act], tq_0[lay.ik_act] = qT_c[lay.ik_from], qT_0[lay.ik_from]
  qa = np.zeros(m.nu)
  qa[lay.ik_act] = qQ[lay.ik_from]
  dq_sq, dq_w, fmax = offsets(m, lay, qa, mass, f_n)
  d = mujoco.MjData(m)
  q0 = np.zeros(m.nu)
  q0[lay.ik_act] = grid.joints(np.array([s_0]), np.zeros(1), np.zeros(1), "Q")[0][lay.ik_from]
  H.init_state_c(m, d, idx, lay, tq_0, q0, active, 0)
  gate = H.init_gate(m, d, idx, active)
  oq, od = idx.obj_qadr[active], idx.obj_dadr[active]
  qfix = d.qpos[oq:oq + 7].copy()
  caps = [m.geom(f"robot/{s}_hand_collision").id for s in ("left", "right")]
  n = H.NSTEP
  fn = np.zeros((n, 2))
  nc = np.zeros((n, 2), int)
  body = np.zeros(n, int)
  obj_p, hand_p = np.zeros((n // DECIM, 3)), np.zeros((n // DECIM, 2, 3))
  v_rel, align = None, None
  for k in range(n):
    t = k * DT
    if k % DECIM == 0:
      a = L.ramp(np.array([t]), 0.0, H.T_SQ)[0]
      d.ctrl[:] = tq_0 + (tq_c - tq_0) * a + dq_sq * a + (dq_w if t >= H.T_REL - 1e-9 else 0.0)
    d.xfrc_applied[ob] = H.support_c(m, d, ob, oq, od, qfix, k < H.K_A) if k < H.K_REL else 0.0
    mujoco.mj_step(m, d)
    counts, fh, fnh, dmin, ncap = PS.classify_contacts_c(m, d, idx, og)
    fn[k], nc[k], body[k] = fnh, counts[1:3], counts[5]
    if k == H.K_REL - 1:
      v_rel = float(np.linalg.norm(d.qvel[od:od + 3]))
      cap_align = [float(np.degrees(np.arcsin(min(1.0, abs(d.geom_xmat[g].reshape(3, 3)[1, 2]))))) for g in caps]
      face_align = []
      for hb, s_in in zip(idx.hand_bodies, (-1.0, 1.0)):
        zf = d.xmat[hb].reshape(3, 3)[:, 2]
        face_align.append(float(np.degrees(np.arccos(np.clip(zf @ np.array([0.0, s_in, 0.0]), -1.0, 1.0)))))
      align = cap_align if arm == "fist" else face_align
    if k % DECIM == DECIM - 1:
      c = k // DECIM
      obj_p[c] = d.xpos[ob]
      hand_p[c] = [d.xpos[b] for b in idx.hand_bodies]
  return dict(setting=name, arm=arm, kind=kind, w=w, mass=mass, budget=budget, f_n=f_n, fmax=fmax, gate=list(gate),
              ik_ok=float(row[H.P["ik_ok"]]), v_rel=v_rel, align=align, cap_align=cap_align, face_align=face_align), \
      dict(fn=fn, nc=nc, body=body, obj_p=obj_p, hand_p=hand_p)


def measures(meta, tr) -> dict:
  S = (np.arange(H.NSTEP) + 1) * DT
  sw = (S >= H.T_REL - 0.1 - 1e-9) & (S < H.T_REL - 1e-9)
  start_ok = bool((tr["nc"][sw] >= 2).all() and (tr["body"][sw] == 0).all() and meta["v_rel"] < 0.05 and max(meta["align"]) <= 1.0)
  op, hp = tr["obj_p"][None], tr["hand_p"][None]
  slip10 = float(HR.rel_disp(op, hp, C_REL, C_END)[0])
  slip03 = float(HR.rel_disp(op, hp, C_REL, C_03)[0])
  W = (S >= 2.2 - 1e-9) & (S <= 11.2 + 1e-9)
  hf = []
  for h in range(2):
    x = tr["fn"][W, h]
    mean = float(x.mean())
    hf.append(R._bands(x[:, None], 1.0 / DT, [(10.0, 100.0)])[0] / mean if mean > 0 else float("inf"))
  Wh = (S >= H.T_REL - 1e-9)
  zero = int((tr["nc"][Wh] == 0).any(1).sum())
  change = float((np.diff(tr["nc"][Wh], axis=0) != 0).any(1).mean())
  fn_ratio = [meta["f_n"] / f for f in meta["fmax"]]
  return dict(start_ok=start_ok, slip10_mm=slip10, slip03_mm=slip03, hf_ratio=hf, zero_contact_steps=zero, count_change=change,
              fn_over_fmax=fn_ratio, fn_release=[float(x) for x in tr["fn"][H.K_REL - 1]], nc_release=[int(x) for x in tr["nc"][H.K_REL - 1]])


def judge(rows) -> dict:
  out = {}
  for name in sorted({r["setting"] for r in rows}, key=lambda s: [x[0] for x in SETTINGS + [SIDE]].index(s)):
    rs = [r for r in rows if r["setting"] == name]
    g1w = [r for r in rs if r["kind"] == "G1"]
    g2w = [r for r in rs if r["kind"] == "G2"]
    G1 = all(r["start_ok"] and r["slip10_mm"] < 5.0 and max(r["fn_over_fmax"]) <= 0.6 for r in g1w)
    G2 = all(r["slip03_mm"] >= 5.0 for r in g2w)
    G3 = all(max(r["hf_ratio"]) < 0.05 and r["zero_contact_steps"] == 0 and r["count_change"] < 0.01 for r in g1w)
    out[name] = dict(G1=G1, G2=G2, G3=G3, passed=G1 and G2 and G3)
  order = [s[0] for s in SETTINGS]
  passing = [s for s in order if out.get(s, {}).get("passed")]
  out["selected"] = passing[0] if passing else None
  out["verdict"] = "통과" if passing else "미결"
  return out


def _job(args):
  os.environ.setdefault("OMP_NUM_THREADS", "1")
  setting, world, arm = args
  meta, tr = run_world(setting, world, arm)
  return dict(meta, **measures(meta, tr)), tr


def main(argv):
  out = argv[0]
  side = "--side" in argv
  workers = int(argv[argv.index("--workers") + 1]) if "--workers" in argv else 16
  arm = argv[argv.index("--arm") + 1] if "--arm" in argv else "fist"
  os.makedirs(out, exist_ok=True)
  settings = [SIDE] if side else SETTINGS
  from concurrent.futures import ProcessPoolExecutor
  t0 = time.time()
  jobs = [(s, w, arm) for s in settings for w in WORLDS]
  rows = []
  with ProcessPoolExecutor(max_workers=workers) as ex:
    for (s, w, _), (row, tr) in zip(jobs, ex.map(_job, jobs)):
      rows.append(row)
      np.savez_compressed(os.path.join(out, f"{arm}_{s[0]}_{w[0]}_w{w[1]}_m{w[2]}.npz"), **tr)
  res = dict(rules="rounds/R04/positive/r11_hold/gate_rules.md", wall_s=time.time() - t0, rows=rows, judge=judge(rows))
  fn = ("gate_side.json" if side else "gate_c.json") if arm == "fist" else f"gate_c_{arm}.json"
  json.dump(res, open(os.path.join(out, fn), "w"), indent=1, default=float)
  print(json.dumps(res["judge"], ensure_ascii=False))
  print("done", len(rows), f"{time.time() - t0:.0f} s")


if __name__ == "__main__":
  main(sys.argv[1:])
