"""MJWarp runner of the positive 20 ladder (rules.md 1: the judging engine). Same compiled mjlab scene, per-world fields
(object size / mass / contact parameters, mesh geom_dataid + geom_pos / geom_quat), initial state, PD targets and trace
format as the C runner in p20_ladder.py; contacts are classified on the GPU with p20_scene.Index's category table and
mujoco_warp.contact_force (world frame; normal component = force . contact frame row 0).
"""
from __future__ import annotations

import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from r1adv.positive_obj import p20_ladder as L
from r1adv.positive_obj import p20_scene as PS

FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_invweight0", "body_subtreemass",
          "dof_invweight0", "geom_size", "geom_aabb", "geom_rbound", "geom_friction", "geom_solref", "geom_pos",
          "geom_quat", "geom_dataid")
# per-world buffers (rules 1: >= 2 x the C dev maximum; rules_amend records the measured maxima)
BUFFERS = {"box_wristfix": (96, 768), "o6lite": (192, 1536)}
BUFFERS_PADS = {"box_wristfix": (160, 1280), "o6lite": (256, 2048)}


def _t(a) -> torch.Tensor:
  return wp.to_torch(a)


def buffers(arm: str, scale: float = 1.0) -> tuple[int, int]:
  a = L.ARMS[arm]
  nc, nj = (BUFFERS_PADS if a.get("pads") else BUFFERS)[a["body"]]
  return int(nc * scale), int(nj * scale)


def build_sim(arm: str, panel: str, n: int, device: str = "cuda:0", buf_scale: float = 1.0):
  from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
  cone, condim = L.PANELS[panel]
  m = L.build_model(arm, cone, condim)
  idx = PS.Index(m)
  for shape, qa in idx.obj_qadr.items():
    px, py = PS.PARK_POS[shape]
    m.qpos0[qa:qa + 7] = (px, py, PS.NOMINAL[shape][2] / 2 + 0.002, 1.0, 0.0, 0.0, 0.0)
  nconmax, njmax = buffers(arm, buf_scale)
  cfg = SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=MujocoCfg(cone=cone, **L.SIM_OPT))
  sim = Simulation(num_envs=n, cfg=cfg, model=m, device=device)
  sim.expand_model_fields(FIELDS)
  return sim, m, idx, (nconmax, njmax)


def set_world_fields(sim, m, idx: PS.Index, params: np.ndarray, condim: int, device: str) -> np.ndarray:
  P = L.P
  mod = sim.wp_model
  arr = {f: _t(getattr(mod, f)) for f in FIELDS}
  host = {f: arr[f].cpu().numpy().copy() for f in FIELDS}
  active_geom = np.full(len(params), -1, np.int64)
  for w, row in enumerate(params):
    active = PS.SHAPES[int(row[P["shape"]])]
    for shape in PS.SHAPES:
      if shape == active:
        dims, mass, mu, tc = row[[P["dx"], P["dy"], P["dz"]]], row[P["mass"]], row[P["mu"]], row[P["tc"]]
        active_geom[w] = idx.obj_geom[shape]
      else:
        dims, mass, mu, tc = PS.NOMINAL[shape], 1.0, 0.7, 0.02
      f = PS.object_fields(shape, dims, mass, condim)
      b, g, dadr = idx.obj_body[shape], idx.obj_geom[shape], idx.obj_dadr[shape]
      old = host["body_mass"][w, b]
      host["body_mass"][w, b] = f["body_mass"]
      host["body_inertia"][w, b] = f["body_inertia"]
      host["body_ipos"][w, b] = f["body_ipos"]
      host["body_iquat"][w, b] = f["body_iquat"]
      host["body_invweight0"][w, b] = f["body_invweight0"]
      host["body_subtreemass"][w, b] = f["body_mass"]
      host["body_subtreemass"][w, 0] += f["body_mass"] - old
      host["dof_invweight0"][w, dadr:dadr + 6] = f["dof_invweight0"]
      host["geom_size"][w, g] = f["geom_size"]
      host["geom_aabb"][w, g] = f["geom_aabb"].reshape(2, 3)
      host["geom_rbound"][w, g] = f["geom_rbound"]
      host["geom_pos"][w, g] = f["geom_pos"]
      host["geom_quat"][w, g] = f["geom_quat"]
      if shape != "box":
        host["geom_dataid"][w, g] = idx.mesh_id[shape][PS.mesh_index(shape, PS.snap_dims(shape, dims))]
      host["geom_friction"][w, g] = [mu, 0.005, 0.0001]
      host["geom_solref"][w, g] = [tc, 1.0]
  for f in FIELDS:
    arr[f].copy_(torch.as_tensor(host[f], device=arr[f].device, dtype=arr[f].dtype))
  imp = np.asarray(params[:, P["impratio"]], dtype=np.float32)
  mod.opt.impratio_invsqrt = wp.array(1.0 / np.sqrt(np.maximum(imp, mujoco.mjMINVAL)), dtype=float, device=device)
  sim.create_graph()
  return active_geom


def run_panel(arm: str, panel: str, params: np.ndarray, ik_dir: str, device: str = "cuda:0",
              buf_scale: float = 1.0) -> tuple[dict, float, dict]:
  t0 = time.time()
  n = len(params)
  cone, condim = L.PANELS[panel]
  grids = L.load_grids(ik_dir)
  sim, m, idx, (nconmax, njmax) = build_sim(arm, panel, n, device, buf_scale)
  lay = L.Layout(m, grids[(L.ARMS[arm]["body"], 0)].names)
  with wp.ScopedDevice(device):
    active_geom = set_world_fields(sim, m, idx, params, condim, device)
  P = L.P
  tgt = np.zeros((n, L.NCTRL, m.nu), np.float32)
  hq = np.zeros((n, L.NCTRL, 2, 2), np.float32)
  q0 = np.zeros((n, m.nq))
  mpos = np.repeat(sim.mj_data.mocap_pos[None].copy(), n, 0) if m.nmocap else None
  mquat = np.repeat(sim.mj_data.mocap_quat[None].copy(), n, 0) if m.nmocap else None
  d = mujoco.MjData(m)
  for w, row in enumerate(params):
    active = L.set_fields_c(m, idx, row, condim)                 # host copy only to build the initial state
    qa, tt = L.initial_q(arm, row, grids, lay)
    tgt[w] = tt
    _, hq[w] = L.world_targets(arm, row, grids, lay)
    L.init_state_c(arm, m, d, idx, lay, row, tt[0], qa, active)
    q0[w] = d.qpos
    if m.nmocap:
      mpos[w], mquat[w] = d.mocap_pos, d.mocap_quat
  dev = torch.device(device)
  D = sim.wp_data
  qpos, qvel, ctrl = _t(D.qpos), _t(D.qvel), _t(D.ctrl)
  qpos.copy_(torch.as_tensor(q0, device=dev, dtype=qpos.dtype))
  qvel.zero_()
  if m.nmocap:
    _t(D.mocap_pos).copy_(torch.as_tensor(mpos, device=dev, dtype=torch.float32))
    _t(D.mocap_quat).copy_(torch.as_tensor(mquat, device=dev, dtype=torch.float32))
  tgt_t = torch.as_tensor(tgt, device=dev)
  ctrl.copy_(tgt_t[:, 0])
  sim.forward()
  ar = torch.arange(n, device=dev)
  shapes = [PS.SHAPES[int(r[P["shape"]])] for r in params]
  oq = torch.as_tensor([idx.obj_qadr[s] for s in shapes], device=dev)
  od = torch.as_tensor([idx.obj_dadr[s] for s in shapes], device=dev)
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
  T = L.NSTEP
  z = lambda *s, dt=torch.float32: torch.zeros(s, device=dev, dtype=dt)     # noqa: E731
  out_t = dict(obj_v=z(n, T, 3), hand_v=z(n, T, 2, 3), f_hand=z(n, T, 2, 3), fn_hand=z(n, T, 2),
               dist_min=torch.ones((n, T, 2), device=dev), ncon=z(n, T, 8, dt=torch.int16), ncap=z(n, T, 2, dt=torch.int16),
               obj_p=z(n, L.NCTRL, 3), obj_q=z(n, L.NCTRL, 4), hand_p=z(n, L.NCTRL, 2, 3), hand_q=z(n, L.NCTRL, 2, 4),
               tau_ratio=z(n, L.NCTRL, 10), waist_yaw=z(n, L.NCTRL))
  nefc_max = torch.zeros(n, device=dev, dtype=torch.int32)
  nacon_max = torch.zeros((), device=dev, dtype=torch.int64)
  tmax = torch.zeros((n, 10), device=dev)
  off3 = torch.arange(3, device=dev)
  off4 = torch.arange(4, device=dev)
  rng_c = torch.arange(naconmax, device=dev)
  for k in range(T):
    if k % L.DECIM == 0:
      ctrl.copy_(tgt_t[:, k // L.DECIM])
      tmax.zero_()
    sim.step()
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
    if k % L.DECIM == L.DECIM - 1:
      ci = k // L.DECIM
      out_t["obj_p"][:, ci] = qpos[ar[:, None], oq[:, None] + off3].float()
      out_t["obj_q"][:, ci] = qpos[ar[:, None], oq[:, None] + 3 + off4].float()
      out_t["hand_p"][:, ci] = xpos[:, hb].float()
      out_t["hand_q"][:, ci] = xquat[:, hb].float()
      out_t["tau_ratio"][:, ci] = tmax
      out_t["waist_yaw"][:, ci] = qpos[:, yaw_q].float()
  torch.cuda.synchronize() if dev.type == "cuda" else None
  wall = time.time() - t0
  out = {k: v.cpu().numpy() for k, v in out_t.items()}
  out["nefc_max"] = nefc_max.cpu().numpy()
  out["hand_q_ctrl"] = hq
  per_world_con = int(np.max(out["ncon"][..., 0]))
  extra = dict(nconmax=nconmax, njmax=njmax, naconmax=int(naconmax), nacon_max_seen=int(nacon_max.item()),
               nefc_max_seen=int(out["nefc_max"].max()), ncon_world_max_seen=per_world_con,
               overflow=bool(nacon_max.item() >= naconmax or out["nefc_max"].max() >= njmax),
               mujoco_warp=getattr(mjwarp, "__version__", "3.11.0"))
  return out, wall, extra
