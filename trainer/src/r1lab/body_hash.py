"""Robot body signature (coordinator 2026-11-19 00:2x; 07B common rule 'one body definition, hash-frozen'): sha256 per
component of the compiled MuJoCo model, restricted to the robot entity, plus one over all components. Names lose the
'robot/' prefix, so an entity compiled alone (CPU) and the same entity inside an env give the same signature.
  geometry   robot geoms: name, type, size, pos, quat, body
  contact    robot geoms: contype, conaffinity, condim, priority, margin, gap; candidate self-contact pairs (bitmask test,
             parent-child filter, explicit excludes); explicit pairs touching the robot
  friction   robot geoms: friction, solref, solimp, solmix
  mass       robot bodies: name, parent, mass, inertia, ipos, iquat, pos, quat; for a body carrying a free joint (the root,
             pelvis) pos and quat are left out (coordinator 2026-11-19 05:0x, rounds/R04/r9_readouts.md R-body (b)): they are
             only the model's qpos0 (the build pose, e.g. r1wb.world.raise_build_pose lifts the pelvis 8 m) and do not
             enter the dynamics, since a free body's pose comes from qpos
  actuators  robot actuators: name, joint, gainprm, biasprm, forcerange, ctrlrange; dofs: armature, frictionloss, damping;
             joints: range. Command delays live in Python (mjlab actuator buffers), not in the model: not covered.
entity_signature(EntityCfg) builds the entity on the CPU; env_signature(env) reads a built env (run start, evaluation runner).
Signature v2 (default, since 2026-11-19 09:4x, positive 16): free-joint root pos / quat excluded. Every call also returns the
old v1 values under 'v1' (mass, total). legacy=True returns the v1 signature itself (compatibility: 15446e6f = the O6 / r1wb
training body, 39a0b446 = the same body in the build pose of the evaluation runners).
CLI: python -m r1lab.body_hash [OUT.json] compares the r1lab stock / r1lab O6 / r1wb bodies."""
from __future__ import annotations

import hashlib
import json

import numpy as np

COMPONENTS = ("geometry", "contact", "friction", "mass", "actuators")


def _name(m, kind, i: int) -> str:
  import mujoco
  n = mujoco.mj_id2name(m, kind, int(i)) or ""
  return n.split("/", 1)[1] if n.startswith("robot/") else n


def _rows(m, body_ids, root_qpos0: bool = False) -> dict:
  import mujoco
  B = sorted(int(b) for b in body_ids)
  bset = set(B)
  G = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bset]
  coll = [g for g in G if m.geom_contype[g] or m.geom_conaffinity[g]]
  r = lambda a: np.round(np.asarray(a, dtype=np.float64), 9).tolist()   # noqa: E731
  gname = lambda g: _name(m, mujoco.mjtObj.mjOBJ_GEOM, g)   # noqa: E731
  bname = lambda b: _name(m, mujoco.mjtObj.mjOBJ_BODY, b)   # noqa: E731
  rows = {k: [] for k in COMPONENTS}
  for g in G:
    rows["geometry"].append([gname(g), int(m.geom_type[g]), r(m.geom_size[g]), r(m.geom_pos[g]), r(m.geom_quat[g]),
                             bname(m.geom_bodyid[g])])
    rows["contact"].append([gname(g), int(m.geom_contype[g]), int(m.geom_conaffinity[g]), int(m.geom_condim[g]),
                            int(m.geom_priority[g]), r(m.geom_margin[g]), r(m.geom_gap[g])])
    rows["friction"].append([gname(g), r(m.geom_friction[g]), r(m.geom_solref[g]), r(m.geom_solimp[g]), r(m.geom_solmix[g])])
  excl = {tuple(sorted((int(m.exclude_signature[e]) >> 16, int(m.exclude_signature[e]) & 0xFFFF))) for e in range(m.nexclude)}
  filterparent = not (int(m.opt.disableflags) & int(mujoco.mjtDisableBit.mjDSBL_FILTERPARENT))
  pairs = []
  for i, a in enumerate(coll):
    for b in coll[i + 1:]:
      ba, bb = int(m.geom_bodyid[a]), int(m.geom_bodyid[b])
      if ba == bb:
        continue
      if not ((m.geom_contype[a] & m.geom_conaffinity[b]) or (m.geom_contype[b] & m.geom_conaffinity[a])):
        continue
      if filterparent and (int(m.body_parentid[ba]) == bb or int(m.body_parentid[bb]) == ba):
        continue
      if tuple(sorted((ba, bb))) in excl:
        continue
      pairs.append(sorted([gname(a), gname(b)]))
  rows["contact"].append(["self_pairs", sorted(pairs)])
  rows["contact"].append(["explicit_pairs", sorted(sorted([gname(m.pair_geom1[p]), gname(m.pair_geom2[p])]) for p in range(m.npair)
                                                    if int(m.geom_bodyid[m.pair_geom1[p]]) in bset or int(m.geom_bodyid[m.pair_geom2[p]]) in bset)])
  free = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == int(mujoco.mjtJoint.mjJNT_FREE)}
  for b in B:
    pose = ["free_root"] if (b in free and not root_qpos0) else [r(m.body_pos[b]), r(m.body_quat[b])]
    rows["mass"].append([bname(b), bname(m.body_parentid[b]), r(m.body_mass[b]), r(m.body_inertia[b]), r(m.body_ipos[b]),
                         r(m.body_iquat[b])] + pose)
  J = [j for j in range(m.njnt) if int(m.jnt_bodyid[j]) in bset]
  for a in range(m.nu):
    j = int(m.actuator_trnid[a, 0])
    if j in J:
      rows["actuators"].append([_name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, a), _name(m, mujoco.mjtObj.mjOBJ_JOINT, j),
                                r(m.actuator_gainprm[a, :3]), r(m.actuator_biasprm[a, :3]), r(m.actuator_forcerange[a]),
                                r(m.actuator_ctrlrange[a])])
  for j in J:
    d = int(m.jnt_dofadr[j])
    rows["actuators"].append([_name(m, mujoco.mjtObj.mjOBJ_JOINT, j), r(m.dof_armature[d]), r(m.dof_frictionloss[d]),
                              r(m.dof_damping[d]), r(m.jnt_range[j])])
  rows["_counts"] = dict(bodies=len(B), geoms=len(G), collision_geoms=len(coll), self_pairs=len(pairs), joints=len(J),
                         free_roots=len(free & bset))
  return rows


def _hash(x) -> str:
  return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def _sig(rows: dict) -> dict:
  out = {k: _hash(rows[k]) for k in COMPONENTS}
  out["total"] = _hash([out[k] for k in COMPONENTS])
  return out


def signature(m, body_ids, legacy: bool = False) -> dict:
  """v2 (default): components + total without the free-joint root's qpos0, plus 'v1' = {mass, total} of the old rule.
  legacy=True: the v1 dict exactly as before (components, total, counts)."""
  rows_v1 = _rows(m, body_ids, root_qpos0=True)
  counts = {k: v for k, v in rows_v1["_counts"].items() if k != "free_roots"}
  v1 = _sig(rows_v1)
  if legacy:
    return dict(v1, counts=counts)
  rows = _rows(m, body_ids, root_qpos0=False)
  out = _sig(rows)
  out["counts"] = dict(counts, free_roots=rows["_counts"]["free_roots"])
  out["version"] = "v2-no-root-qpos0"
  out["v1"] = {"mass": v1["mass"], "total": v1["total"]}
  return out


def entity_signature(cfg, with_rows: bool = False, legacy: bool = False):
  from mjlab.entity import Entity
  m = Entity(cfg).compile()
  ids = range(1, m.nbody)                                       # every body but the world
  sig = signature(m, ids, legacy=legacy)
  return (sig, _rows(m, ids, root_qpos0=legacy)) if with_rows else sig


def env_signature(env, legacy: bool = False) -> dict:
  robot = env.scene["robot"]
  return signature(env.sim.mj_model, robot.indexing.body_ids.tolist(), legacy=legacy)


def diff(rows_a: dict, rows_b: dict, limit: int = 6) -> dict:
  """Per component: rows only in a / only in b (first `limit`, by name)."""
  out = {}
  for k in COMPONENTS:
    sa = {json.dumps(x, sort_keys=True) for x in rows_a[k]}
    sb = {json.dumps(x, sort_keys=True) for x in rows_b[k]}
    if sa != sb:
      out[k] = dict(only_a=[json.loads(x) for x in sorted(sa - sb)][:limit], only_b=[json.loads(x) for x in sorted(sb - sa)][:limit],
                    n_only_a=len(sa - sb), n_only_b=len(sb - sa))
  return out


def main(argv=None):
  import sys
  from r1lab.robot import get_r1_cfg
  from r1wb import robot as WB
  bodies = {"r1lab_stock": get_r1_cfg(command_delay_max=4, self_collision=True),
            "r1lab_o6": get_r1_cfg(command_delay_max=4, self_collision=True, hand="o6"),
            "r1wb": WB.robot_cfg(delay_steps=0)}
  sig, rows = {}, {}
  for k, cfg in bodies.items():
    sig[k], rows[k] = entity_signature(cfg, with_rows=True)
  res = dict(signatures=sig, diff_r1lab_o6_vs_r1wb=diff(rows["r1lab_o6"], rows["r1wb"]),
             diff_r1lab_stock_vs_r1lab_o6=diff(rows["r1lab_stock"], rows["r1lab_o6"]))
  text = json.dumps(res, indent=1)
  argv = sys.argv[1:] if argv is None else argv
  if argv:
    open(argv[0], "w").write(text)
  print(text)


if __name__ == "__main__":
  main()
