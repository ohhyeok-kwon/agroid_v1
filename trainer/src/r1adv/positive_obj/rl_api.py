"""Object worlds for RL on an mjlab Simulation (MuJoCo Warp): per-world randomisation, reset and observation
(positive 18, R11). Built from the pieces the contact ladder verified (scene.object_body_fields, SceneIndex category
table, mujoco_warp.contact_force); the frozen ladder files are not changed.

  sim = Simulation(num_envs=N, cfg=SimulationCfg(nconmax=64, njmax=512, mujoco=MujocoCfg(cone=..., ...)), model=m)
  ow = ObjectWorlds(sim, m, condim=3)
  ow.randomize(np.random.default_rng(0))                 # shape, size, mass, friction, solref, impratio per world
  ow.reset(env_ids, table_xyz, obj_xy_on_table, obj_yaw)  # table pose, active object on the table, others parked
  obs = ow.observe()                                      # object pose / velocity in the robot yaw frame, contacts

Scenes come from ladder.build_model(cone, condim, free_robot=True|False, shapes=SHAPES, table=True). One body per shape
is carried because geom_type is not per-world in MuJoCo Warp; the unused bodies stay parked behind the robot.
"""
from __future__ import annotations

import math

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import torch
import warp as wp

from r1adv.positive_obj import scene as S

FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_invweight0", "body_subtreemass",
          "dof_invweight0", "geom_size", "geom_aabb", "geom_rbound", "geom_friction", "geom_solref")
CATEGORIES = ("total", "hand_L_obj", "hand_R_obj", "obj_table", "obj_floor", "obj_robot_other", "robot_table", "robot_self")


def _t(a) -> torch.Tensor:
  return wp.to_torch(a)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


class ObjectWorlds:
  def __init__(self, sim, m: mujoco.MjModel, condim: int = 3, robot_root: str = "robot/pelvis"):
    self.sim, self.m, self.condim = sim, m, condim
    self.idx = S.SceneIndex(m)
    self.n = sim.num_envs
    self.dev = torch.device(sim.device)
    missing = [f for f in FIELDS if f not in sim.expanded_fields]
    if missing:
      sim.expand_model_fields(tuple(missing))
    imp = wp.array(np.ones(self.n, np.float32), dtype=float, device=str(sim.device))
    sim.wp_model.opt.impratio_invsqrt = imp            # per-world impratio, written in place from now on
    sim.create_graph()
    self.imp_t = _t(sim.wp_model.opt.impratio_invsqrt)
    self.root = m.body(robot_root).id
    self.shapes = list(self.idx.obj_body)
    self.active = torch.full((self.n,), -1, dtype=torch.long, device=self.dev)      # index into self.shapes, -1 none
    self.cat = torch.as_tensor(self.idx.category, device=self.dev, dtype=torch.long)
    self.is_obj = torch.as_tensor(self.idx.is_obj, device=self.dev)
    self.is_cap = torch.as_tensor(self.idx.is_capsule, device=self.dev)
    self.obj_geom = torch.as_tensor([self.idx.obj_geom[s] for s in self.shapes], device=self.dev)
    self.obj_qadr = torch.as_tensor([self.idx.obj_qadr[s] for s in self.shapes], device=self.dev)
    self.obj_dadr = torch.as_tensor([self.idx.obj_dadr[s] for s in self.shapes], device=self.dev)
    D = sim.wp_data
    naconmax = D.contact.geom.shape[0]
    self._cid = wp.array(np.arange(naconmax, dtype=np.int32), dtype=int, device=str(sim.device))
    self._fout = wp.zeros(naconmax, dtype=wp.spatial_vector, device=str(sim.device))
    self.params = {}

  # -- randomisation -------------------------------------------------------------------------------------------
  def sample(self, rng: np.random.Generator, n: int) -> dict:
    """Per-world draws in the ranges of scene.py (task spec): shape, full extents, mass, friction, solref tc, impratio."""
    u = rng.uniform(*S.SIZE_RANGE, size=(n, 3))
    shape = rng.integers(len(self.shapes), size=n)
    dims = u.copy()
    for i in range(n):
      name = self.shapes[shape[i]]
      if name == "cylinder":
        dims[i] = (u[i, 1], u[i, 1], u[i, 2])
      elif name == "capsule":
        dims[i] = (rng.uniform(u[i, 1], S.SIZE_RANGE[1]), u[i, 1], u[i, 1])
    return dict(shape=shape, dims=dims, mass=rng.uniform(*S.MASS_RANGE, size=n), mu=rng.uniform(*S.FRICTION_RANGE, size=n),
                tc=rng.uniform(*S.SOLREF_TC_RANGE, size=n), impratio=np.where(rng.random(n) < 0.5, 10.0, 1.0))

  def write(self, env_ids: np.ndarray, p: dict) -> None:
    """Write the drawn object of each env (and nominal values for its other bodies) into the per-world model fields."""
    mod = self.sim.wp_model
    arr = {f: _t(getattr(mod, f)) for f in FIELDS}
    host = {f: arr[f][torch.as_tensor(env_ids, device=self.dev)].cpu().numpy() for f in FIELDS}
    for j, w in enumerate(env_ids):
      for k, shape in enumerate(self.shapes):
        if k == p["shape"][j]:
          dims, mass, mu, tc = p["dims"][j], p["mass"][j], p["mu"][j], p["tc"][j]
        else:
          dims, mass, mu, tc = S.NOMINAL_DIMS, 1.0, 0.7, 0.02
        f = S.object_body_fields(shape, dims, mass, self.condim)
        b, g, dadr = self.idx.obj_body[shape], self.idx.obj_geom[shape], self.idx.obj_dadr[shape]
        old = host["body_mass"][j, b]
        host["body_mass"][j, b] = f["body_mass"]
        host["body_inertia"][j, b] = f["body_inertia"]
        host["body_ipos"][j, b] = f["body_ipos"]
        host["body_iquat"][j, b] = f["body_iquat"]
        host["body_invweight0"][j, b] = f["body_invweight0"]
        host["body_subtreemass"][j, b] = f["body_mass"]
        host["body_subtreemass"][j, 0] += f["body_mass"] - old
        host["dof_invweight0"][j, dadr:dadr + 6] = f["dof_invweight0"]
        host["geom_size"][j, g] = f["geom_size"]
        host["geom_aabb"][j, g] = f["geom_aabb"].reshape(2, 3)
        host["geom_rbound"][j, g] = f["geom_rbound"]
        host["geom_friction"][j, g] = [mu, S.TORSION, S.ROLLING]
        host["geom_solref"][j, g] = [tc, 1.0]
    ids = torch.as_tensor(env_ids, device=self.dev)
    for f in FIELDS:
      arr[f][ids] = torch.as_tensor(host[f], device=self.dev, dtype=arr[f].dtype)
    self.imp_t[ids] = torch.as_tensor(1.0 / np.sqrt(np.asarray(p["impratio"], dtype=np.float32)), device=self.dev)
    self.active[ids] = torch.as_tensor(np.asarray(p["shape"]), device=self.dev, dtype=torch.long)
    for k, v in p.items():
      self.params.setdefault(k, np.zeros((self.n,) + np.asarray(v).shape[1:]))[np.asarray(env_ids)] = v

  def randomize(self, rng: np.random.Generator, env_ids: np.ndarray | None = None) -> dict:
    env_ids = np.arange(self.n) if env_ids is None else np.asarray(env_ids)
    p = self.sample(rng, len(env_ids))
    self.write(env_ids, p)
    return p

  # -- reset ---------------------------------------------------------------------------------------------------
  def sample_table(self, rng: np.random.Generator, n: int, near_edge_x=(0.45, 0.80), top_z=S.TABLE_TOP_RANGE,
                   yaw_of_robot: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Table near-edge centre (world, robot at the origin facing +x unless yaw_of_robot) with the top height drawn in
    TABLE_TOP_RANGE (0.30-0.85 m), and an object spot 0.10-0.25 m behind the near edge, |y| <= 0.3 m."""
    x = rng.uniform(*near_edge_x, size=n)
    z = rng.uniform(*top_z, size=n)
    spot = np.stack([x + rng.uniform(0.10, 0.25, size=n), rng.uniform(-0.3, 0.3, size=n)], 1)
    table = np.stack([x, np.zeros(n), z], 1)
    if yaw_of_robot is not None:
      c, s = np.cos(yaw_of_robot), np.sin(yaw_of_robot)
      rot = lambda v: np.stack([c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1]], 1)
      table[:, :2] = rot(table[:, :2])
      spot = rot(spot)
    return table, spot

  def reset(self, env_ids: np.ndarray, table_xyz: np.ndarray, obj_xy: np.ndarray, obj_yaw: np.ndarray) -> None:
    """Table near-edge centre of the top at table_xyz (world, per env); the env's active object resting on the table
    top at obj_xy with yaw obj_yaw; other bodies parked; object velocities zero. Call before sim.forward()."""
    D = self.sim.wp_data
    qpos, qvel = _t(D.qpos), _t(D.qvel)
    ids = torch.as_tensor(env_ids, device=self.dev)
    mp, mq = _t(D.mocap_pos), _t(D.mocap_quat)
    if self.idx.table_mocap >= 0:
      mp[ids, self.idx.table_mocap] = torch.as_tensor(table_xyz, device=self.dev, dtype=mp.dtype)
      mq[ids, self.idx.table_mocap] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev, dtype=mq.dtype)
    for k, shape in enumerate(self.shapes):
      qa, da = self.idx.obj_qadr[shape], self.idx.obj_dadr[shape]
      px, py = S.PARK_POS[shape]
      pose = np.tile([px, py, S.NOMINAL_DIMS[2] / 2 + 0.002, 1.0, 0.0, 0.0, 0.0], (len(env_ids), 1))
      act = self.active[ids].cpu().numpy() == k
      if act.any():
        h = self.params["dims"][np.asarray(env_ids)[act], 2]
        pose[act, 0:2] = obj_xy[act]
        pose[act, 2] = np.asarray(table_xyz)[act, 2] + h / 2 + 0.001
        pose[act, 3:7] = np.stack([np.cos(obj_yaw[act] / 2), 0 * obj_yaw[act], 0 * obj_yaw[act], np.sin(obj_yaw[act] / 2)], 1)
      qpos[ids, qa:qa + 7] = torch.as_tensor(pose, device=self.dev, dtype=qpos.dtype)
      qvel[ids, da:da + 6] = 0.0

  # -- observation ---------------------------------------------------------------------------------------------
  def observe(self, contacts: bool = True) -> dict:
    """Active object pose / velocity in the robot root yaw frame (zeros for envs without an active object) and, if
    contacts, the per-env contact summary (counts over CATEGORIES, force on the object from each hand in the yaw frame,
    min hand-object distance, wrist-capsule contact counts)."""
    D = self.sim.wp_data
    qpos, qvel = _t(D.qpos), _t(D.qvel)
    xpos, xquat = _t(D.xpos), _t(D.xquat)
    n = self.n
    ar = torch.arange(n, device=self.dev)
    has = self.active >= 0
    k = self.active.clamp(min=0)
    qa, da = self.obj_qadr[k], self.obj_dadr[k]
    off3, off4 = torch.arange(3, device=self.dev), torch.arange(4, device=self.dev)
    pos = qpos[ar[:, None], qa[:, None] + off3]
    quat = qpos[ar[:, None], qa[:, None] + 3 + off4]
    lin = qvel[ar[:, None], da[:, None] + off3]
    ang_b = qvel[ar[:, None], da[:, None] + 3 + off3]                 # free joint: angular velocity in the body frame
    rq = xquat[:, self.root]
    yaw = torch.atan2(2 * (rq[:, 0] * rq[:, 3] + rq[:, 1] * rq[:, 2]), 1 - 2 * (rq[:, 2] ** 2 + rq[:, 3] ** 2))
    c, s = torch.cos(yaw), torch.sin(yaw)
    Ryt = torch.stack([torch.stack([c, s, torch.zeros_like(c)], -1), torch.stack([-s, c, torch.zeros_like(c)], -1),
                       torch.stack([torch.zeros_like(c), torch.zeros_like(c), torch.ones_like(c)], -1)], -2)   # world -> yaw
    rel = pos - xpos[:, self.root]
    yq = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), -torch.sin(yaw / 2)], -1)
    out = dict(active=self.active.clone(),
               obj_pos_b=torch.where(has[:, None], torch.einsum("nij,nj->ni", Ryt, rel), 0.0),
               obj_quat_b=torch.where(has[:, None], _quat_mul(yq, quat), 0.0),
               obj_linvel_b=torch.where(has[:, None], torch.einsum("nij,nj->ni", Ryt, lin), 0.0),
               obj_angvel_body=torch.where(has[:, None], ang_b, 0.0))
    if contacts:
      cnt, fh, dmin, ncap = self.contacts()
      out.update(contacts=cnt, f_hand_b=torch.einsum("nij,nhj->nhi", Ryt, fh), dist_min=dmin, ncap=ncap)
    return out

  def contacts(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    D = self.sim.wp_data
    n = self.n
    mjwarp.contact_force(self.sim.wp_model, D, self._cid, True, self._fout)
    cg, cdist, cw = _t(D.contact.geom), _t(D.contact.dist), _t(D.contact.worldid)
    nacon = _t(D.nacon)
    f = _t(self._fout)
    naconmax = cg.shape[0]
    valid = torch.arange(naconmax, device=self.dev) < nacon[0]
    g1, g2 = cg[:, 0].long(), cg[:, 1].long()
    wid = cw.long().clamp(0, n - 1)
    ag = torch.where(self.active >= 0, self.obj_geom[self.active.clamp(min=0)], -1)
    c = self.cat[g1.clamp(min=0), g2.clamp(min=0)]
    c = torch.where((c <= 5) & ~((g1 == ag[wid]) | (g2 == ag[wid])), torch.zeros_like(c), c)
    c = torch.where(valid, c, torch.zeros_like(c))
    cnt = torch.bincount(wid * 8 + c, weights=(c > 0).float(), minlength=n * 8).view(n, 8)
    cnt[:, 0] = torch.bincount(wid[valid], minlength=n).float()
    hand = (c == 1) | (c == 2)
    sign = torch.where(self.is_obj[g2.clamp(min=0)], 1.0, -1.0)
    slot = (wid * 2 + (c - 1).clamp(min=0))[hand]
    fh = torch.zeros((n * 2, 3), device=self.dev).index_add_(0, slot, (f[:, :3] * sign[:, None])[hand].float()).view(n, 2, 3)
    dmin = torch.ones(n * 2, device=self.dev).scatter_reduce(0, slot, cdist[hand].float(), reduce="amin",
                                                             include_self=True).view(n, 2)
    capm = hand & (self.is_cap[g1.clamp(min=0)] | self.is_cap[g2.clamp(min=0)])
    ncap = torch.bincount((wid * 2 + (c - 1).clamp(min=0))[capm], minlength=n * 2).view(n, 2)
    return cnt, fh, dmin, ncap
