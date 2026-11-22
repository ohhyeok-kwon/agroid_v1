"""MJWarp runner of the positive 20 fist hold test (rounds/R04/positive/r11_hold/rules.md): same scene, per-world fields,
initial state (computed with C kinematics per world), PD targets, support phases and trace format as p20_hold.py."""
from __future__ import annotations

import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from r1adv.positive_obj import p20_hold as H
from r1adv.positive_obj import p20_ladder as L
from r1adv.positive_obj import p20_ladder_mjw as W
from r1adv.positive_obj import p20_scene as PS

BUFFERS = {"fist": (64, 512), "fistfit": (96, 768), "fistpads": (192, 1536)}     # per world (nconmax, njmax)


def _t(a):
  return wp.to_torch(a)


def build_sim(arm, panel, n, device="cuda:0", buf_scale=1.0):
  from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
  cone, condim = L.PANELS[panel]
  m = H.build_model(arm, cone, condim)
  idx = PS.Index(m)
  for shape, qa in idx.obj_qadr.items():
    px, py = PS.PARK_POS[shape]
    m.qpos0[qa:qa + 7] = (px, py, PS.NOMINAL[shape][2] / 2 + 0.002, 1.0, 0.0, 0.0, 0.0)
  nconmax, njmax = (int(v * buf_scale) for v in BUFFERS[arm])
  cfg = SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=MujocoCfg(cone=cone, **L.SIM_OPT))
  sim = Simulation(num_envs=n, cfg=cfg, model=m, device=device)
  sim.expand_model_fields(W.FIELDS)
  return sim, m, idx, (nconmax, njmax)


def run_panel(arm, panel, params, ik_file, device="cuda:0", buf_scale=1.0, tend=None):
  t0 = time.time()
  n = len(params)
  cone, condim = L.PANELS[panel]
  grid = H.FistGrid(ik_file)
  sim, m, idx, (nconmax, njmax) = build_sim(arm, panel, n, device, buf_scale)
  lay = L.Layout(m, grid.names)
  with wp.ScopedDevice(device):
    active_geom = W.set_world_fields(sim, m, idx, params, condim, device)
  T = H.NSTEP if tend is None else int(round(tend / H.DT))
  T50 = T // H.DECIM
  tgt = np.zeros((n, H.NCTRL, m.nu), np.float32)
  q0 = np.zeros((n, m.nq))
  gate = np.zeros((n, 3), np.float32)
  d = mujoco.MjData(m)
  for w, row in enumerate(params):
    active = H.set_fields_c(m, idx, row, condim)
    qa, tt = H.initial_q(row, grid, lay)
    tgt[w] = tt
    H.init_state_c(m, d, idx, lay, tt[0], qa, active, int(row[H.P["orient"]]))
    gate[w] = H.init_gate(m, d, idx, active)
    q0[w] = d.qpos
  dev = torch.device(device)
  D = sim.wp_data
  qpos, qvel, ctrl = _t(D.qpos), _t(D.qvel), _t(D.ctrl)
  qpos.copy_(torch.as_tensor(q0, device=dev, dtype=qpos.dtype))
  qvel.zero_()
  tgt_t = torch.as_tensor(tgt, device=dev)
  ctrl.copy_(tgt_t[:, 0])
  sim.forward()
  ar = torch.arange(n, device=dev)
  shapes = [PS.SHAPES[int(r[H.P["shape"]])] for r in params]
  oq = torch.as_tensor([idx.obj_qadr[s] for s in shapes], device=dev)
  od = torch.as_tensor([idx.obj_dadr[s] for s in shapes], device=dev)
  off3, off4, off6, off7 = (torch.arange(k, device=dev) for k in (3, 4, 6, 7))
  xfrc = _t(D.xfrc_applied)
  ob = torch.as_tensor([idx.obj_body[s] for s in shapes], device=dev)
  mass_w = _t(sim.wp_model.body_mass)[ar, ob].double()            # per-world (expanded) object mass and inertia
  inert_w = _t(sim.wp_model.body_inertia)[ar, ob].double().max(-1).values
  qfix = qpos[ar[:, None], oq[:, None] + off7].double().clone()
  kp, kd = mass_w * H.SUP_W ** 2, 2 * mass_w * H.SUP_W
  kr, kdr = inert_w * H.SUP_W ** 2, 2 * inert_w * H.SUP_W
  conj = torch.tensor([1.0, -1.0, -1.0, -1.0], device=dev, dtype=torch.float64)

  def qmul(a, b):
    w1, x1, y1, z1 = a.unbind(-1)
    w2, x2, y2, z2 = b.unbind(-1)
    return torch.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)

  def qrot(q, v):
    u, w0 = q[..., 1:], q[..., :1]
    return v + 2 * torch.cross(u, torch.cross(u, v, dim=-1) + w0 * v, dim=-1)

  def support(spring: bool):                                     # same as p20_hold.support_c
    f = torch.zeros((n, 6), device=dev, dtype=torch.float64)
    f[:, 2] = mass_w * 9.81
    if spring:
      pq = qpos[ar[:, None], oq[:, None] + off7].double()
      vw = qvel[ar[:, None], od[:, None] + off6].double()
      p, q = pq[:, :3], pq[:, 3:7] / pq[:, 3:7].norm(dim=-1, keepdim=True)
      w = qrot(q, vw[:, 3:])
      qe = qmul(q, qfix[:, 3:7] * conj)
      e = 2.0 * torch.where(qe[:, :1] >= 0, 1.0, -1.0) * qe[:, 1:]
      f[:, :3] += -kp[:, None] * (p - qfix[:, :3]) - kd[:, None] * vw[:, :3]
      f[:, 3:] = -kr[:, None] * e - kdr[:, None] * w
    return f.to(xfrc.dtype)

  ag = torch.as_tensor(active_geom, device=dev)
  hb = torch.as_tensor(idx.hand_bodies, device=dev)
  root = torch.as_tensor(m.body_rootid[idx.hand_bodies], device=dev)
  cat = torch.as_tensor(idx.category, device=dev, dtype=torch.long)
  is_obj = torch.as_tensor(idx.is_obj, device=dev)
  is_cap = torch.as_tensor(idx.is_capsule, device=dev)
  arm_act = torch.as_tensor(lay.arm_act, device=dev)
  eff = torch.as_tensor(lay.effort[lay.arm_act], device=dev, dtype=torch.float32)
  yaw_q = lay.waist_yaw_qadr
  naconmax = D.contact.geom.shape[0]
  cid = wp.array(np.arange(naconmax, dtype=np.int32), dtype=int, device=device)
  fout = wp.zeros(naconmax, dtype=wp.spatial_vector, device=device)
  fout_t = _t(fout)
  cg, cdist, cw, cframe = _t(D.contact.geom), _t(D.contact.dist), _t(D.contact.worldid), _t(D.contact.frame)
  nacon_t, nefc_t = _t(D.nacon), _t(D.nefc)
  xpos, xquat, cvel, scom = _t(D.xpos), _t(D.xquat), _t(D.cvel), _t(D.subtree_com)
  afrc = _t(D.actuator_force)
  z = lambda *s, dt=torch.float32: torch.zeros(s, device=dev, dtype=dt)     # noqa: E731
  out_t = dict(obj_v=z(n, T, 3), hand_v=z(n, T, 2, 3), f_hand=z(n, T, 2, 3), fn_hand=z(n, T, 2),
               dist_min=torch.ones((n, T, 2), device=dev), ncon=z(n, T, 8, dt=torch.int16), ncap=z(n, T, 2, dt=torch.int16),
               obj_p=z(n, T50, 3), obj_q=z(n, T50, 4), hand_p=z(n, T50, 2, 3), hand_q=z(n, T50, 2, 4),
               tau_ratio=z(n, T50, 10), waist_yaw=z(n, T50))
  nefc_max = torch.zeros(n, device=dev, dtype=torch.int32)
  nacon_max = torch.zeros((), device=dev, dtype=torch.int64)
  tmax = torch.zeros((n, 10), device=dev)
  rng_c = torch.arange(naconmax, device=dev)
  xfrc_seen = {}
  for k in range(T):
    if k % H.DECIM == 0:
      ctrl.copy_(tgt_t[:, k // H.DECIM])
      tmax.zero_()
    if k < H.K_REL:
      xfrc[ar, ob] = support(k < H.K_A)
    elif k == H.K_REL:
      xfrc[ar, ob] = 0.0
    sim.step()
    if k in (H.K_REL - 1, H.K_REL + 1):                         # mechanics check: support on before, zero after release
      xfrc_seen[k] = float(xfrc[ar, ob].abs().sum().item())
    out_t["obj_v"][:, k] = qvel[ar[:, None], od[:, None] + off3].float()
    cv = cvel[:, hb]
    lever = xpos[:, hb] - scom[ar[:, None], root[None, :]]
    out_t["hand_v"][:, k] = (cv[..., 3:] + torch.cross(cv[..., :3], lever, dim=-1)).float()
    mjwarp.contact_force(sim.wp_model, D, cid, True, fout)
    na = nacon_t[0]
    valid = rng_c < na
    g1, g2 = cg[:, 0].long(), cg[:, 1].long()
    wid = cw.long().clamp(0, n - 1)
    c = cat[g1.clamp(min=0), g2.clamp(min=0)]
    act_ok = (g1 == ag[wid]) | (g2 == ag[wid])
    c = torch.where((c <= 5) & ~act_ok, torch.zeros_like(c), c)
    c = torch.where(valid, c, torch.zeros_like(c))
    tot = torch.bincount(wid[valid], minlength=n)
    cnt = torch.bincount(wid * 8 + c, weights=(c > 0).float(), minlength=n * 8).view(n, 8)
    cnt[:, 0] = tot
    out_t["ncon"][:, k] = cnt.to(torch.int16)
    hand = (c == 1) | (c == 2)
    sign = torch.where(is_obj[g2.clamp(min=0)], 1.0, -1.0)
    fw = fout_t[:, :3] * sign[:, None]
    slot = (wid * 2 + (c - 1).clamp(min=0))[hand]
    out_t["f_hand"][:, k] = torch.zeros((n * 2, 3), device=dev).index_add_(0, slot, fw[hand].float()).view(n, 2, 3)
    fnorm = (fout_t[:, :3] * cframe[:, 0, :]).sum(-1).abs()
    out_t["fn_hand"][:, k] = torch.zeros(n * 2, device=dev).index_add_(0, slot, fnorm[hand].float()).view(n, 2)
    dm = torch.ones(n * 2, device=dev).scatter_reduce(0, slot, cdist[hand].float(), reduce="amin", include_self=True)
    out_t["dist_min"][:, k] = dm.view(n, 2)
    capm = hand & (is_cap[g1.clamp(min=0)] | is_cap[g2.clamp(min=0)])
    out_t["ncap"][:, k] = torch.bincount((wid * 2 + (c - 1).clamp(min=0))[capm], minlength=n * 2).view(n, 2).to(torch.int16)
    nefc_max = torch.maximum(nefc_max, nefc_t.int())
    nacon_max = torch.maximum(nacon_max, na.long())
    tmax = torch.maximum(tmax, (afrc[:, arm_act].abs() / eff).float())
    if k % H.DECIM == H.DECIM - 1:
      ci = k // H.DECIM
      out_t["obj_p"][:, ci] = qpos[ar[:, None], oq[:, None] + off3].float()
      out_t["obj_q"][:, ci] = qpos[ar[:, None], oq[:, None] + 3 + off4].float()
      out_t["hand_p"][:, ci] = xpos[:, hb].float()
      out_t["hand_q"][:, ci] = xquat[:, hb].float()
      out_t["tau_ratio"][:, ci] = tmax
      out_t["waist_yaw"][:, ci] = qpos[:, yaw_q].float()
  if dev.type == "cuda":
    torch.cuda.synchronize()
  wall = time.time() - t0
  out = {k: v.cpu().numpy() for k, v in out_t.items()}
  out["nefc_max"] = nefc_max.cpu().numpy()
  out["init_gate"] = gate
  extra = dict(nconmax=nconmax, njmax=njmax, naconmax=int(naconmax), nacon_max_seen=int(nacon_max.item()),
               nefc_max_seen=int(out["nefc_max"].max()), ncon_world_max_seen=int(np.max(out["ncon"][..., 0])),
               overflow=bool(nacon_max.item() >= naconmax or out["nefc_max"].max() >= njmax),
               mujoco_warp=getattr(mjwarp, "__version__", "3.11.0"),
               xfrc_abs_sum_before_after_release=[xfrc_seen.get(H.K_REL - 1), xfrc_seen.get(H.K_REL + 1)])
  return out, wall, extra
