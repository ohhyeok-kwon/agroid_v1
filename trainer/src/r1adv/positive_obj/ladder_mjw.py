"""MJWarp runner of the contact ladder (positive 18, R11; rules.md section 1: the judging engine). Same compiled mjlab scene,
per-world fields, initial state, PD targets and trace format as the C runner in ladder.py; contacts are classified on the
GPU from the contact arrays with the same category table (scene.SceneIndex) and mujoco_warp.contact_force.
"""
from __future__ import annotations

import time

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from r1adv.positive_obj import ladder as L
from r1adv.positive_obj import scene as S

FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_invweight0", "body_subtreemass",
          "dof_invweight0", "geom_size", "geom_aabb", "geom_rbound", "geom_friction", "geom_solref")
NCONMAX, NJMAX = 64, 512          # per world; C runs: ncon max 31, nefc max 210 (dev and seed 1)


def _t(a) -> torch.Tensor:
  return wp.to_torch(a)


def build_sim(panel: str, n: int, device: str = "cuda:0"):
  from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
  cone, condim = L.PANELS[panel]
  m = L.build_model(cone, condim, nocap=panel.endswith("nc"))
  # put_data checks the contact count at qpos0, where every free object sits at the origin: park them there instead
  # (the run sets qpos per world explicitly, as the C runner does).
  idx = S.SceneIndex(m)
  for shape, qa in idx.obj_qadr.items():
    px, py = S.PARK_POS[shape]
    m.qpos0[qa:qa + 7] = (px, py, S.NOMINAL_DIMS[2] / 2 + 0.002, 1.0, 0.0, 0.0, 0.0)
  cfg = SimulationCfg(nconmax=NCONMAX, njmax=NJMAX, mujoco=MujocoCfg(cone=cone, **L.SIM_OPT))
  sim = Simulation(num_envs=n, cfg=cfg, model=m, device=device)
  sim.expand_model_fields(FIELDS)
  return sim, m


def set_world_fields(sim, m, idx: S.SceneIndex, params: np.ndarray, condim: int, device: str) -> np.ndarray:
  """Per-world object fields (same values as write_world_fields_c) and per-world impratio. Returns active geom ids."""
  P = L.P
  mod = sim.wp_model
  arr = {f: _t(getattr(mod, f)) for f in FIELDS}
  host = {f: arr[f].cpu().numpy().copy() for f in FIELDS}
  active_geom = np.full(len(params), -1, np.int64)
  for w, row in enumerate(params):
    active = S.SHAPES[int(row[P["shape"]])] if row[P["stratum"]] != 3 else None
    for shape in idx.obj_body:
      if shape == active:
        dims, mass, mu, tc = row[[P["dx"], P["dy"], P["dz"]]], row[P["mass"]], row[P["mu"]], row[P["tc"]]
        active_geom[w] = idx.obj_geom[shape]
      else:
        dims, mass, mu, tc = S.NOMINAL_DIMS, 1.0, 0.7, 0.02
      f = S.object_body_fields(shape, dims, mass, condim)
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
      host["geom_friction"][w, g] = [mu, S.TORSION, S.ROLLING]
      host["geom_solref"][w, g] = [tc, 1.0]
  for f in FIELDS:
    arr[f].copy_(torch.as_tensor(host[f], device=arr[f].device, dtype=arr[f].dtype))
  imp = np.asarray(params[:, P["impratio"]], dtype=np.float32)
  mod.opt.impratio_invsqrt = wp.array(1.0 / np.sqrt(np.maximum(imp, mujoco.mjMINVAL)), dtype=float, device=device)
  sim.create_graph()
  return active_geom


def run_panel(panel: str, params: np.ndarray, ik_path: str, device: str = "cuda:0") -> tuple[dict, float, dict]:
  t0 = time.time()
  n = len(params)
  cone, condim = L.PANELS[panel]
  grid = L.IKGrid(ik_path)
  sim, m = build_sim(panel, n, device)
  idx = S.SceneIndex(m)
  lay = L.Layout(m, grid)
  with wp.ScopedDevice(device):
    active_geom = set_world_fields(sim, m, idx, params, condim, device)
  P = L.P
  # targets and initial state on the host (identical to the C runner)
  tgt = np.zeros((n, L.NCTRL, m.nu), np.float32)
  q0 = np.zeros((n, m.nq))
  mpos = np.repeat(sim.mj_data.mocap_pos[None].copy(), n, 0) if m.nmocap else None
  mquat = np.repeat(sim.mj_data.mocap_quat[None].copy(), n, 0) if m.nmocap else None
  d = mujoco.MjData(m)
  act_names = []
  for w, row in enumerate(params):
    tgt[w] = L.world_targets(row, grid, lay)
    active = S.SHAPES[int(row[P["shape"]])] if row[P["stratum"]] != 3 else None
    act_names.append(active)
    L.init_state_c(m, d, idx, lay, row, tgt[w, 0], active, q0=L.world_targets(row, grid, lay, "Q")[0])
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
  # gather indices
  ar = torch.arange(n, device=dev)
  has_obj = torch.as_tensor([a is not None for a in act_names], device=dev)
  oq = torch.as_tensor([idx.obj_qadr[a] if a else 0 for a in act_names], device=dev)
  od = torch.as_tensor([idx.obj_dadr[a] if a else 0 for a in act_names], device=dev)
  ag = torch.as_tensor(active_geom, device=dev)
  hb = torch.as_tensor(idx.hand_bodies, device=dev)
  root = torch.as_tensor(m.body_rootid[idx.hand_bodies], device=dev)
  cat = torch.as_tensor(idx.category, device=dev, dtype=torch.long)
  is_obj = torch.as_tensor(idx.is_obj, device=dev)
  is_cap = torch.as_tensor(idx.is_capsule, device=dev)
  arm = torch.as_tensor(lay.arm_act, device=dev)
  eff = torch.as_tensor(lay.effort[lay.arm_act], device=dev, dtype=torch.float32)
  yaw_q = lay.waist_yaw_qadr
  naconmax = D.contact.geom.shape[0]
  cid = wp.array(np.arange(naconmax, dtype=np.int32), dtype=int, device=device)
  fout = wp.zeros(naconmax, dtype=wp.spatial_vector, device=device)
  fout_t = _t(fout)
  cg, cdist, cw = _t(D.contact.geom), _t(D.contact.dist), _t(D.contact.worldid)
  nacon_t, nefc_t = _t(D.nacon), _t(D.nefc)
  xpos, xquat, cvel, scom = _t(D.xpos), _t(D.xquat), _t(D.cvel), _t(D.subtree_com)
  afrc = _t(D.actuator_force)
  T = L.NSTEP
  out_t = dict(obj_v=torch.zeros((n, T, 3), device=dev), hand_v=torch.zeros((n, T, 2, 3), device=dev),
               f_hand=torch.zeros((n, T, 2, 3), device=dev), dist_min=torch.ones((n, T, 2), device=dev),
               ncon=torch.zeros((n, T, 8), device=dev, dtype=torch.int16),
               ncap=torch.zeros((n, T, 2), device=dev, dtype=torch.int16),
               obj_p=torch.zeros((n, L.NCTRL, 3), device=dev), obj_q=torch.zeros((n, L.NCTRL, 4), device=dev),
               hand_p=torch.zeros((n, L.NCTRL, 2, 3), device=dev), hand_q=torch.zeros((n, L.NCTRL, 2, 4), device=dev),
               tau_ratio=torch.zeros((n, L.NCTRL, 10), device=dev), waist_yaw=torch.zeros((n, L.NCTRL), device=dev))
  nefc_max = torch.zeros(n, device=dev, dtype=torch.int32)
  nacon_max = 0
  tmax = torch.zeros((n, 10), device=dev)
  off3 = torch.arange(3, device=dev)
  for k in range(T):
    if k % L.DECIM == 0:
      ctrl.copy_(tgt_t[:, k // L.DECIM])
      tmax.zero_()
    sim.step()
    out_t["obj_v"][:, k] = torch.where(has_obj[:, None], qvel[ar[:, None], od[:, None] + off3], 0.0).float()
    cv = cvel[:, hb]                                                     # (n, 2, 6) rot, lin at the root subtree com
    lever = xpos[:, hb] - scom[ar[:, None], root[None, :]]
    out_t["hand_v"][:, k] = (cv[..., 3:] + torch.cross(cv[..., :3], lever, dim=-1)).float()
    na = int(nacon_t[0].item()) if k % 50 == 0 else None
    mjwarp.contact_force(sim.wp_model, D, cid, True, fout)
    valid = torch.arange(naconmax, device=dev) < nacon_t[0]
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
    fh = torch.zeros((n * 2, 3), device=dev).index_add_(0, slot, fw[hand].float())
    out_t["f_hand"][:, k] = fh.view(n, 2, 3)
    dm = torch.ones(n * 2, device=dev).scatter_reduce(0, slot, cdist[hand].float(), reduce="amin", include_self=True)
    out_t["dist_min"][:, k] = dm.view(n, 2)
    capm = hand & (is_cap[g1.clamp(min=0)] | is_cap[g2.clamp(min=0)])
    nc = torch.bincount((wid * 2 + (c - 1).clamp(min=0))[capm], minlength=n * 2).view(n, 2)
    out_t["ncap"][:, k] = nc.to(torch.int16)
    nefc_max = torch.maximum(nefc_max, nefc_t.int())
    if na is not None:
      nacon_max = max(nacon_max, na)
    tmax = torch.maximum(tmax, (afrc[:, arm].abs() / eff).float())
    if k % L.DECIM == L.DECIM - 1:
      ci = k // L.DECIM
      out_t["obj_p"][:, ci] = torch.where(has_obj[:, None], qpos[ar[:, None], oq[:, None] + off3], 0.0).float()
      out_t["obj_q"][:, ci] = torch.where(has_obj[:, None], qpos[ar[:, None], oq[:, None] + 3 + torch.arange(4, device=dev)], 0.0).float()
      out_t["hand_p"][:, ci] = xpos[:, hb].float()
      out_t["hand_q"][:, ci] = xquat[:, hb].float()
      out_t["tau_ratio"][:, ci] = tmax
      out_t["waist_yaw"][:, ci] = qpos[:, yaw_q].float()
  torch.cuda.synchronize()
  wall = time.time() - t0
  out = {k: v.cpu().numpy() for k, v in out_t.items()}
  out["nefc_max"] = nefc_max.cpu().numpy()
  extra = dict(nconmax=NCONMAX, njmax=NJMAX, naconmax=int(naconmax), nacon_max_seen=int(nacon_max),
               nefc_max_seen=int(out["nefc_max"].max()), overflow=bool(nacon_max >= naconmax or out["nefc_max"].max() >= NJMAX),
               mujoco_warp=getattr(mjwarp, "__version__", "3.11.0"))
  return out, wall, extra
