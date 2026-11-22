"""Positive 20 scene parts (rounds/R04/positive/r11_contact2/rules.md 2.2, 4): objects box (primitive) / upright convex-mesh
cylinder (16 sides) / lying convex-mesh capsule (16 sides, ring-approximated ends), the per-world size of a mesh object by
a discrete mesh grid + per-world geom_dataid, and the contact classification with the extended hand geom sets (palm,
fingers, thumb, wrist capsule).

MuJoCo moves a mesh to its inertial frame at compile time and folds that offset into geom_pos / geom_quat, so a per-world
mesh swap writes geom_pos / geom_quat too (taken from a compile of the object alone with that mesh), with the body inertial
fields. check_world_fields() compares a swapped world with a native compile (fields and a 1 s trajectory).
"""
from __future__ import annotations

import functools
import math

import mujoco
import numpy as np

from r1adv.positive_obj import p20_check1b as K1B
from r1adv.positive_obj import scene as S

SHAPES = ("box", "cylinder", "capsule")
ENTITY_OF = {"box": "obj_box", "cylinder": "obj_cyl", "capsule": "obj_cap"}
N_SIDES = 16
D_GRID = tuple(round(0.05 + 0.025 * i, 4) for i in range(15))            # 0.05 .. 0.40
H_GRID = tuple(round(0.10 + 0.05 * i, 4) for i in range(7))              # 0.10 .. 0.40
L_GRID = H_GRID
PARK_POS = S.PARK_POS
# the nominal (compile-time) mesh of each mesh shape is one whose frame flags are the general ones (body_sameframe NONE,
# geom_sameframe INERTIA): tall cylinders compile to BODYROT shortcuts that would ignore a swapped mesh frame (14:4x)
NOMINAL = {"box": (0.2, 0.2, 0.2), "cylinder": (0.4, 0.4, 0.1), "capsule": (0.3, 0.2, 0.2)}
MESH_OFFSET = 1e-4


def snap(v: float, grid) -> float:
  g = np.asarray(grid)
  return float(g[np.argmin(np.abs(g - v))])


def cyl_mesh_list() -> list[tuple[float, float]]:
  return [(d, h) for d in D_GRID for h in H_GRID]


def cap_mesh_list() -> list[tuple[float, float]]:
  return [(d, L) for d in D_GRID for L in L_GRID if L >= d - 1e-9]


def cyl_verts(d: float, h: float) -> np.ndarray:
  """Upright 16-gon prism, circumradius d/2, a face normal on +-x and +-y (phase pi/16)."""
  a = math.pi / N_SIDES + 2 * np.pi * np.arange(N_SIDES) / N_SIDES
  ring = np.stack([d / 2 * np.cos(a), d / 2 * np.sin(a)], 1)
  return np.concatenate([np.c_[ring, -h / 2 * np.ones(N_SIDES)], np.c_[ring, h / 2 * np.ones(N_SIDES)]])


def cap_verts(d: float, L: float) -> np.ndarray:
  """Lying capsule along x (check 1 A1b shape), a flat face down (phase pi/16)."""
  return K1B.mcap_verts(d / 2, L, math.pi / N_SIDES)


def mesh_name(shape: str, i: int) -> str:
  return f"{'cyl' if shape == 'cylinder' else 'cap'}_{i:03d}"


def mesh_index(shape: str, dims) -> int:
  """Index of the grid mesh for the snapped dims (x, y, z): cylinder (d, d, h), capsule (L, d, d)."""
  if shape == "cylinder":
    return cyl_mesh_list().index((float(dims[1]), float(dims[2])))
  return cap_mesh_list().index((float(dims[1]), float(dims[0])))


def snap_dims(shape: str, dims) -> tuple:
  x, y, z = (float(v) for v in dims)
  if shape == "box":
    return (x, y, z)
  if shape == "cylinder":
    d, h = snap(y, D_GRID), snap(z, H_GRID)
    return (d, d, h)
  d = snap(y, D_GRID)
  L = max(snap(x, L_GRID), min(v for v in L_GRID if v >= d - 1e-9))
  return (L, d, d)


def _object_geom(body, shape: str, condim: int, mass: float, mu: float, tc: float, mesh: str | None, dims=None):
  kw = dict(name="shape", mass=mass, priority=S.OBJ_PRIORITY, condim=condim, friction=[mu, S.TORSION, S.ROLLING],
            solref=[tc, 1.0], solimp=list(S.SOLIMP), contype=1, conaffinity=1, group=2, rgba=[0.85, 0.55, 0.25, 1.0])
  if shape == "box":
    x, y, z = dims
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[x / 2, y / 2, z / 2], **kw)
  else:
    # MESH_OFFSET: a 0.1 mm geom offset makes every mesh object body 'non-simple' with the geom on the inertial frame
    # (geom_sameframe 2) whatever the mesh's principal-axis order, so a per-world mesh swap never meets a stale
    # body_simple / geom_sameframe shortcut of the nominal mesh (found by p20_check_fields, 14:4x)
    body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, pos=[0.0, 0.0, MESH_OFFSET], **kw)


def object_spec(shape: str, condim: int = 3, dims=None, mass: float = 1.0, mu: float = 0.7, tc: float = 0.02,
                all_meshes: bool = True) -> mujoco.MjSpec:
  """One free object body. Mesh shapes carry every grid mesh (all_meshes) so that a world can switch geom_dataid; the geom
  uses the mesh of dims (or the nominal one)."""
  spec = mujoco.MjSpec()
  dims = snap_dims(shape, dims if dims is not None else NOMINAL[shape])
  mesh = None
  if shape != "box":
    lst = cyl_mesh_list() if shape == "cylinder" else cap_mesh_list()
    want = mesh_index(shape, dims)
    for i, (a, b) in enumerate(lst):
      if all_meshes or i == want:
        v = cyl_verts(a, b) if shape == "cylinder" else cap_verts(a, b)
        spec.add_mesh(name=mesh_name(shape, i), uservert=v.reshape(-1).tolist())
    mesh = mesh_name(shape, want)
  body = spec.worldbody.add_body(name="body")
  body.add_freejoint(name="free")
  _object_geom(body, shape, condim, mass, mu, tc, mesh, dims)
  return spec


def object_entity_cfg(shape: str, condim: int = 3):
  from mjlab.entity import EntityCfg
  px, py = PARK_POS[shape]
  z = NOMINAL[shape][2] / 2 + 0.002
  return EntityCfg(spec_fn=functools.partial(object_spec, shape, condim),
                   init_state=EntityCfg.InitialStateCfg(pos=(px, py, z)))


@functools.lru_cache(maxsize=4096)
def _fields(shape: str, dims: tuple, mass: float, condim: int) -> dict:
  m = object_spec(shape, condim, dims, mass, all_meshes=False).compile()
  b, g = m.body("body").id, m.geom("shape").id
  return dict(body_mass=float(m.body_mass[b]), body_inertia=m.body_inertia[b].copy(), body_ipos=m.body_ipos[b].copy(),
              body_iquat=m.body_iquat[b].copy(), body_invweight0=m.body_invweight0[b].copy(),
              dof_invweight0=m.dof_invweight0[:6].copy(), dof_M0=m.dof_M0[:6].copy(), geom_size=m.geom_size[g].copy(),
              geom_aabb=m.geom_aabb[g].copy(), geom_rbound=float(m.geom_rbound[g]), geom_pos=m.geom_pos[g].copy(),
              geom_quat=m.geom_quat[g].copy(), bvh_aabb=m.bvh_aabb[m.body_bvhadr[b]:m.body_bvhadr[b] + m.body_bvhnum[b]].copy())


def object_fields(shape: str, dims, mass: float, condim: int = 3) -> dict:
  return _fields(shape, tuple(float(v) for v in snap_dims(shape, dims)), float(mass), int(condim))


class Index:
  """Ids and the contact category table of a compiled P20 ladder scene (entity prefixes robot/, table/, obj_*/)."""

  def __init__(self, m: mujoco.MjModel):
    self.m = m
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""          # noqa: E731
    G, B = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
    self.obj_body = {s: m.body(f"{e}/body").id for s, e in ENTITY_OF.items()}
    self.obj_geom = {s: m.geom(f"{e}/shape").id for s, e in ENTITY_OF.items()}
    self.obj_jnt = {s: m.joint(f"{e}/free").id for s, e in ENTITY_OF.items()}
    self.obj_qadr = {s: int(m.jnt_qposadr[j]) for s, j in self.obj_jnt.items()}
    self.obj_dadr = {s: int(m.jnt_dofadr[j]) for s, j in self.obj_jnt.items()}
    self.mesh_id = {}
    for s in ("cylinder", "capsule"):
      lst = cyl_mesh_list() if s == "cylinder" else cap_mesh_list()
      self.mesh_id[s] = [m.mesh(f"{ENTITY_OF[s]}/{mesh_name(s, i)}").id for i in range(len(lst))]
    self.table_geoms = [g for g in range(m.ngeom) if name(G, g).startswith("table/")]
    tb = mujoco.mj_name2id(m, B, "table/table")
    self.table_mocap = int(m.body_mocapid[tb]) if tb >= 0 else -1
    if self.table_mocap < 0 and mujoco.mj_name2id(m, B, "table/mocap_base") >= 0:
      self.table_mocap = int(m.body_mocapid[m.body("table/mocap_base").id])
    robot_bodies = {b for b in range(m.nbody) if name(B, b).startswith("robot/")}
    self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies and (m.geom_contype[g] | m.geom_conaffinity[g])]
    self.hand_bodies = [m.body(f"robot/{s}_o6_hand").id for s in ("left", "right")]
    # hand geoms: every collision geom in the subtree of robot/<side>_o6_hand, plus the wrist capsule
    self.hand_geoms = {}
    for k, s in enumerate(("left", "right")):
      root = m.body(f"robot/{s}_o6_hand").id
      sub = {b for b in range(m.nbody) if self._in_subtree(b, root)}
      gs = [g for g in self.robot_geoms if m.geom_bodyid[g] in sub]
      gs.append(m.geom(f"robot/{s}_hand_collision").id)
      self.hand_geoms[k] = gs
    self.capsule_geoms = {k: m.geom(f"robot/{s}_hand_collision").id for k, s in enumerate(("left", "right"))}
    self.floor_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]
    cat = np.zeros((m.ngeom, m.ngeom), np.int8)
    objs = set(self.obj_geom.values())
    hand = {g: k for k, gs in self.hand_geoms.items() for g in gs}
    robot, table, floor = set(self.robot_geoms), set(self.table_geoms), set(self.floor_geoms)
    for a in range(m.ngeom):
      for b in range(m.ngeom):
        if a in objs or b in objs:
          x = b if a in objs else a
          if x in hand:
            cat[a, b] = 1 + hand[x]
          elif x in table:
            cat[a, b] = 3
          elif x in floor:
            cat[a, b] = 4
          elif x in robot:
            cat[a, b] = 5
        elif (a in robot and b in table) or (b in robot and a in table):
          cat[a, b] = 6
        elif a in robot and b in robot:
          cat[a, b] = 7
    self.category = cat
    self.is_obj = np.zeros(m.ngeom, bool)
    self.is_obj[list(objs)] = True
    self.is_capsule = np.zeros(m.ngeom, bool)
    self.is_capsule[list(self.capsule_geoms.values())] = True

  def _in_subtree(self, b: int, root: int) -> bool:
    while b > 0:
      if b == root:
        return True
      b = int(self.m.body_parentid[b])
    return False


def write_world_fields_c(m, idx: Index, shape: str, dims, mass, condim, mu, tc) -> None:
  f = object_fields(shape, dims, mass, condim)
  b, g, dadr = idx.obj_body[shape], idx.obj_geom[shape], idx.obj_dadr[shape]
  old = float(m.body_mass[b])
  m.body_mass[b] = f["body_mass"]
  m.body_inertia[b] = f["body_inertia"]
  m.body_ipos[b] = f["body_ipos"]
  m.body_iquat[b] = f["body_iquat"]
  m.body_invweight0[b] = f["body_invweight0"]
  m.dof_invweight0[dadr:dadr + 6] = f["dof_invweight0"]
  m.dof_M0[dadr:dadr + 6] = f["dof_M0"]
  m.body_subtreemass[b] = f["body_mass"]
  m.body_subtreemass[0] += f["body_mass"] - old
  m.geom_size[g] = f["geom_size"]
  m.geom_aabb[g] = f["geom_aabb"]
  m.geom_rbound[g] = f["geom_rbound"]
  m.geom_pos[g] = f["geom_pos"]
  m.geom_quat[g] = f["geom_quat"]
  if shape != "box":
    m.geom_dataid[g] = idx.mesh_id[shape][mesh_index(shape, snap_dims(shape, dims))]
  adr, num = m.body_bvhadr[b], m.body_bvhnum[b]
  assert num == len(f["bvh_aabb"]), (num, len(f["bvh_aabb"]))
  m.bvh_aabb[adr:adr + num] = f["bvh_aabb"]
  m.geom_friction[g] = [mu, S.TORSION, S.ROLLING]
  m.geom_solref[g] = [tc, 1.0]


def classify_contacts_c(m, d, idx: Index, active_geom: int = -1):
  """counts (8), net force on the object from each hand (2, 3, world), sum of contact normal forces per hand (2),
  min hand-object distance (2), wrist-capsule hand-object contacts (2)."""
  counts = np.zeros(8, np.int16)
  f_hand = np.zeros((2, 3))
  fn = np.zeros(2)
  dmin = np.ones(2)
  ncap = np.zeros(2, np.int16)
  counts[0] = d.ncon
  fr = np.zeros(6)
  for i in range(d.ncon):
    con = d.contact[i]
    g1, g2 = int(con.geom1), int(con.geom2)
    c = int(idx.category[g1, g2])
    if c == 0 or (c <= 5 and g1 != active_geom and g2 != active_geom):
      continue
    counts[c] += 1
    if c in (1, 2):
      k = c - 1
      mujoco.mj_contactForce(m, d, i, fr)
      frame = con.frame.reshape(3, 3)
      fw = frame.T @ fr[:3]
      f_hand[k] += fw if idx.is_obj[g2] else -fw
      fn[k] += abs(fr[0])
      dmin[k] = min(dmin[k], float(con.dist))
      if idx.is_capsule[g1] or idx.is_capsule[g2]:
        ncap[k] += 1
  return counts, f_hand, fn, dmin, ncap
