"""Free objects and tables for R1 manipulation scenes (positive 18, R11): specs, per-world randomisation, reset and
observation helpers. Used by the contact ladder (ladder.py) and meant to be reused unchanged by RL tasks.

Objects
  One mjlab entity per shape ("obj_box", "obj_cyl", "obj_cap"), each a free body "body" with one geom "shape".
  geom_type is not batched in MuJoCo Warp, so a scene that needs a different shape per world carries one body per shape
  and parks the unused ones (PARK_POS). Box: full extents (x, y, z). Cylinder: upright, (diameter, diameter, height).
  Capsule: lying along the body x axis, (length, diameter, diameter).
  Contact parameters live on the object geom with priority OBJ_PRIORITY = 2: hand-object and object-table contacts take
  the object's condim, friction, solref and solimp (the same rule in MuJoCo and MJWarp collision_core). The robot body and
  its signature are untouched.
Table
  A fixed entity "table" (mjlab wraps it in a mocap body, so its pose is set per world by the mocap pose): top slab
  0.6 x 1.2 x 0.04 m and four legs. Frame origin = centre of the near edge of the top surface, +x away from the robot.
  Mocap bodies have weldid 0, so the table never generates contacts with the floor.
Per-world size / mass
  object_body_fields() compiles the object alone and returns every derived field MuJoCo would compute for it (mass,
  inertia, ipos, iquat, invweight0, dof_invweight0, dof_M0, geom size, aabb, rbound, body BVH); write_world_fields_c() / the MJWarp writer in
  ladder.py copy them into the shared model, so a world is identical to a native compile with that object (except
  stat.meaninertia, a solver-tolerance scale that MJWarp holds once for all worlds: the base model's value is kept).
"""
from __future__ import annotations

import functools
import math

import mujoco
import numpy as np

SHAPES = ("box", "cylinder", "capsule")
ENTITY_OF = {"box": "obj_box", "cylinder": "obj_cyl", "capsule": "obj_cap"}
OBJ_PRIORITY = 2
TORSION, ROLLING = 0.005, 0.0001
SOLIMP = (0.9, 0.95, 0.001, 0.5, 2.0)                 # MuJoCo default
NOMINAL_DIMS = (0.2, 0.2, 0.2)
SIZE_RANGE = (0.10, 0.40)
MASS_RANGE = (0.2, 3.0)
FRICTION_RANGE = (0.4, 1.0)
TABLE_TOP_RANGE = (0.30, 0.85)
SOLREF_TC_RANGE = (0.010, 0.040)                      # >= 2 x physics dt (5 ms)
TABLE_HALF = (0.30, 0.60, 0.02)
LEG_HALF = (0.02, 0.02, 0.43)
PARK_POS = {"box": (-1.5, 0.6), "cylinder": (-1.5, 0.0), "capsule": (-1.5, -0.6)}
CAPSULE_QUAT = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)   # geom z axis -> body x axis


def shape_geom(shape: str, dims) -> tuple[int, list[float], list[float]]:
  """(geom type, size, quat) of a shape with full extents dims = (x, y, z) in the body frame."""
  x, y, z = (float(v) for v in dims)
  if shape == "box":
    return mujoco.mjtGeom.mjGEOM_BOX, [x / 2, y / 2, z / 2], [1.0, 0.0, 0.0, 0.0]
  if shape == "cylinder":
    return mujoco.mjtGeom.mjGEOM_CYLINDER, [y / 2, z / 2, 0.0], [1.0, 0.0, 0.0, 0.0]
  if shape == "capsule":
    r = y / 2
    return mujoco.mjtGeom.mjGEOM_CAPSULE, [r, max(x / 2 - r, 1e-4), 0.0], list(CAPSULE_QUAT)
  raise KeyError(shape)


def object_spec(shape: str, condim: int = 3, dims=NOMINAL_DIMS, mass: float = 1.0, mu: float = 0.7,
                tc: float = 0.02) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="body")
  body.add_freejoint(name="free")
  gtype, size, quat = shape_geom(shape, dims)
  body.add_geom(name="shape", type=gtype, size=size, quat=quat, mass=mass, priority=OBJ_PRIORITY, condim=condim,
                friction=[mu, TORSION, ROLLING], solref=[tc, 1.0], solimp=list(SOLIMP), contype=1, conaffinity=1,
                group=2, rgba=[0.85, 0.55, 0.25, 1.0])
  return spec


def table_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table")
  tx, ty, tz = TABLE_HALF
  body.add_geom(name="top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[tx, ty, tz], pos=[tx, 0.0, -tz], condim=3,
                friction=[1.0, TORSION, ROLLING], contype=1, conaffinity=1, group=2, rgba=[0.55, 0.45, 0.35, 1.0])
  lx, ly, lz = LEG_HALF
  for i, (px, py) in enumerate(((lx + 0.01, ty - ly - 0.01), (lx + 0.01, -ty + ly + 0.01),
                                 (2 * tx - lx - 0.01, ty - ly - 0.01), (2 * tx - lx - 0.01, -ty + ly + 0.01))):
    body.add_geom(name=f"leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[lx, ly, lz], pos=[px, py, -2 * tz - lz],
                  condim=3, contype=1, conaffinity=1, group=2, rgba=[0.45, 0.35, 0.25, 1.0])
  return spec


def object_entity_cfg(shape: str, condim: int = 3):
  from mjlab.entity import EntityCfg
  px, py = PARK_POS[shape]
  return EntityCfg(spec_fn=functools.partial(object_spec, shape, condim),
                   init_state=EntityCfg.InitialStateCfg(pos=(px, py, NOMINAL_DIMS[2] / 2 + 0.002)))


def table_entity_cfg():
  from mjlab.entity import EntityCfg
  return EntityCfg(spec_fn=table_spec, init_state=EntityCfg.InitialStateCfg(pos=(0.5, 0.0, 0.75)))


@functools.lru_cache(maxsize=65536)
def _object_body_fields(shape: str, dims: tuple, mass: float, condim: int) -> dict:
  m = object_spec(shape, condim, dims, mass).compile()
  b, g = m.body("body").id, m.geom("shape").id
  return dict(body_mass=float(m.body_mass[b]), body_inertia=m.body_inertia[b].copy(), body_ipos=m.body_ipos[b].copy(),
              body_iquat=m.body_iquat[b].copy(), body_invweight0=m.body_invweight0[b].copy(),
              dof_invweight0=m.dof_invweight0[:6].copy(), dof_M0=m.dof_M0[:6].copy(), geom_size=m.geom_size[g].copy(),
              geom_aabb=m.geom_aabb[g].copy(), geom_rbound=float(m.geom_rbound[g]),
              bvh_aabb=m.bvh_aabb[m.body_bvhadr[b]:m.body_bvhadr[b] + m.body_bvhnum[b]].copy())


def object_body_fields(shape: str, dims, mass: float, condim: int = 3) -> dict:
  return _object_body_fields(shape, tuple(float(v) for v in dims), float(mass), int(condim))


class SceneIndex:
  """Ids of the scene parts in a compiled mjlab scene (entity prefixes 'robot/', 'table/', 'obj_*/')."""

  def __init__(self, m: mujoco.MjModel):
    self.m = m
    name = lambda t, i: mujoco.mj_id2name(m, t, i) or ""
    G, B = mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY
    self.obj_body = {s: m.body(f"{e}/body").id for s, e in ENTITY_OF.items() if _has(m, B, f"{e}/body")}
    self.obj_geom = {s: m.geom(f"{ENTITY_OF[s]}/shape").id for s in self.obj_body}
    self.obj_jnt = {s: m.joint(f"{ENTITY_OF[s]}/free").id for s in self.obj_body}
    self.obj_qadr = {s: int(m.jnt_qposadr[j]) for s, j in self.obj_jnt.items()}
    self.obj_dadr = {s: int(m.jnt_dofadr[j]) for s, j in self.obj_jnt.items()}
    self.table_geoms = [g for g in range(m.ngeom) if name(G, g).startswith("table/")]
    self.table_mocap = int(m.body_mocapid[m.body("table/table").id]) if _has(m, B, "table/table") else -1
    if self.table_mocap < 0 and _has(m, B, "table/mocap_base"):
      self.table_mocap = int(m.body_mocapid[m.body("table/mocap_base").id])
    robot_bodies = {b for b in range(m.nbody) if name(B, b).startswith("robot/")}
    self.robot_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in robot_bodies and m.geom_contype[g] | m.geom_conaffinity[g]]
    self.hand_geoms = {k: [m.geom(f"robot/{s}_o6_collision").id, m.geom(f"robot/{s}_hand_collision").id]
                       for k, s in enumerate(("left", "right"))}
    self.capsule_geoms = {k: m.geom(f"robot/{s}_hand_collision").id for k, s in enumerate(("left", "right"))}
    self.hand_bodies = [m.body(f"robot/{s}_o6_hand").id for s in ("left", "right")]
    self.floor_geoms = [g for g in range(m.ngeom) if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE]   # terrain body
    # contact category table over geom pairs (symmetric): 0 none, 1 hand L-obj, 2 hand R-obj, 3 obj-table, 4 obj-floor,
    # 5 obj-robot other, 6 robot-table, 7 robot-robot; plus a sign for 'force on the object' (+1 if geom2 is the object).
    cat = np.zeros((m.ngeom, m.ngeom), np.int8)
    objs = set(self.obj_geom.values())
    hand = {g: k for k, gs in self.hand_geoms.items() for g in gs}
    robot, table, floor = set(self.robot_geoms), set(self.table_geoms), set(self.floor_geoms)
    for a in range(m.ngeom):
      for b in range(m.ngeom):
        if a in objs or b in objs:
          o, x = (a, b) if a in objs else (b, a)
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


def _has(m, objtype, name) -> bool:
  return mujoco.mj_name2id(m, objtype, name) >= 0


def write_world_fields_c(m: mujoco.MjModel, idx: SceneIndex, shape: str, dims, mass: float, condim: int, mu: float,
                         tc: float) -> None:
  """Write one object's size / mass / contact parameters into a C model (one world = one model copy)."""
  f = object_body_fields(shape, dims, mass, condim)
  b, g, dadr = idx.obj_body[shape], idx.obj_geom[shape], idx.obj_dadr[shape]
  old = float(m.body_mass[b])
  m.body_mass[b] = f["body_mass"]
  m.body_inertia[b] = f["body_inertia"]
  m.body_ipos[b] = f["body_ipos"]
  m.body_iquat[b] = f["body_iquat"]
  m.body_invweight0[b] = f["body_invweight0"]
  m.dof_invweight0[dadr:dadr + 6] = f["dof_invweight0"]
  m.dof_M0[dadr:dadr + 6] = f["dof_M0"]              # C MuJoCo caches the diagonal M of 'simple' bodies here
  m.body_subtreemass[b] = f["body_mass"]
  m.body_subtreemass[0] += f["body_mass"] - old
  m.geom_size[g] = f["geom_size"]
  m.geom_aabb[g] = f["geom_aabb"]
  m.geom_rbound[g] = f["geom_rbound"]
  adr, num = m.body_bvhadr[b], m.body_bvhnum[b]
  assert num == len(f["bvh_aabb"]), (num, len(f["bvh_aabb"]))
  m.bvh_aabb[adr:adr + num] = f["bvh_aabb"]       # C broadphase uses the body BVH, not only geom_aabb
  m.geom_friction[g] = [mu, TORSION, ROLLING]
  m.geom_solref[g] = [tc, 1.0]


def footprint_xmin(shape: str, dims, cx: float, yaw: float) -> float:
  """Smallest world x of the object's footprint (centre x cx, yaw about z)."""
  x, y, _ = (float(v) for v in dims)
  c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
  if shape == "box":
    return cx - (x / 2 * c + y / 2 * s)
  if shape == "cylinder":
    return cx - y / 2
  r = y / 2
  return cx - ((x / 2 - r) * c + r)


def yaw_quat(yaw: float) -> np.ndarray:
  return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def classify_contacts_c(m: mujoco.MjModel, d: mujoco.MjData, idx: SceneIndex, active_geom: int = -1) -> tuple:
  """C engine: per-category counts (8), force on the object from each hand (2, 3, world), min hand-object distance (2),
  wrist-capsule hand-object counts (2). Object categories (1-5) count only the active object's geom (parked bodies and the
  empty twin's objects are ignored); the total (index 0) counts every contact."""
  counts = np.zeros(8, np.int16)
  f_hand = np.zeros((2, 3))
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
      fw = frame.T @ fr[:3]                       # force on geom2 from geom1, world frame
      f_hand[k] += fw if idx.is_obj[g2] else -fw
      dmin[k] = min(dmin[k], float(con.dist))
      if idx.is_capsule[g1] or idx.is_capsule[g2]:
        ncap[k] += 1
  return counts, f_hand, dmin, ncap
