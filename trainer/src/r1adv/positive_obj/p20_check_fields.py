"""Per-world object field swap vs a native compile (advisor gate before the second freeze; positive 18 check_fields.py
style): for chosen worlds, the swapped C model (p20_ladder.set_fields_c) and a scene compiled natively with that object
size must have the same object fields and the same 1 s trajectory (robot PD targets of the ladder).

  python -m r1adv.positive_obj.p20_check_fields <ik_dir> <arm> <panel> <world,...>
"""
from __future__ import annotations

import dataclasses
import functools
import sys

import mujoco
import numpy as np

from r1adv.positive_obj import p20_ladder as L
from r1adv.positive_obj import p20_scene as PS

FLAGS = ("body_simple", "body_sameframe", "geom_sameframe", "dof_simplenum", "M_rowadr", "mapM2D")
FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_invweight0", "dof_invweight0", "dof_M0", "geom_size",
          "geom_aabb", "geom_rbound", "geom_pos", "geom_quat", "geom_friction", "geom_solref", "bvh_aabb")


def native_model(arm, cone, condim, row):
  shape = PS.SHAPES[int(row[L.P["shape"]])]
  dims = tuple(row[[L.P["dx"], L.P["dy"], L.P["dz"]]])
  orig = PS.object_entity_cfg

  def patched(sh, cd):
    cfg = orig(sh, cd)
    if sh != shape:
      return cfg
    return dataclasses.replace(cfg, spec_fn=functools.partial(PS.object_spec, sh, cd, dims, float(row[L.P["mass"]]),
                                                               float(row[L.P["mu"]]), float(row[L.P["tc"]])))
  PS.object_entity_cfg = patched
  try:
    m = L.build_model(arm, cone, condim)
  finally:
    PS.object_entity_cfg = orig
  m.opt.impratio = row[L.P["impratio"]]
  return m


def main(argv):
  ik_dir, arm, panel, worlds = argv[0], argv[1], argv[2], [int(x) for x in argv[3].split(",")]
  cone, condim = L.PANELS[panel]
  grids = L.load_grids(ik_dir)
  params = L.arm_params(arm, L.sample_base(0), grids)
  worst = 0.0
  for wi in worlds:
    row = params[wi]
    ms = L.build_model(arm, cone, condim)
    idx = PS.Index(ms)
    active = L.set_fields_c(ms, idx, row, condim)
    mn = native_model(arm, cone, condim, row)
    mn.stat.meaninertia = ms.stat.meaninertia
    g_s, g_n = idx.obj_geom[active], PS.Index(mn).obj_geom[active]
    bad = []
    for f in FIELDS:
      a, b = getattr(ms, f), getattr(mn, f)
      dd = float(np.abs(a - b).max())
      worst = max(worst, dd)
      if not np.allclose(a, b, rtol=1e-6, atol=1e-9):
        bad.append((f, dd))
    for f in FLAGS:
      if not np.array_equal(getattr(ms, f), getattr(mn, f)):
        bad.append((f, "structure"))
    mesh_ok = (ms.geom_dataid[g_s] == -1) or np.allclose(
      ms.mesh_vert[ms.mesh_vertadr[ms.geom_dataid[g_s]]:ms.mesh_vertadr[ms.geom_dataid[g_s]] + ms.mesh_vertnum[ms.geom_dataid[g_s]]],
      mn.mesh_vert[mn.mesh_vertadr[mn.geom_dataid[g_n]]:mn.mesh_vertadr[mn.geom_dataid[g_n]] + mn.mesh_vertnum[mn.geom_dataid[g_n]]])
    trajs = []
    for mm in (ms, mn):
      ix = PS.Index(mm)
      lay = L.Layout(mm, grids[(L.ARMS[arm]["body"], 0)].names)
      d = mujoco.MjData(mm)
      q0, tgt = L.initial_q(arm, row, grids, lay)
      L.init_state_c(arm, mm, d, ix, lay, row, tgt[0], q0, active)
      for k in range(200):
        if k % 4 == 0:
          d.ctrl[:] = tgt[k // 4]
        mujoco.mj_step(mm, d)
      trajs.append(d.qpos.copy())
    dq = np.abs(trajs[0] - trajs[1])
    print(f"w{wi} {active} dims {np.round(row[[L.P['dx'], L.P['dy'], L.P['dz']]], 3)}: field mismatches {bad} mesh verts equal {mesh_ok} "
          f"| qpos max diff after 1 s {dq.max():.3g}")
  print("worst field diff", worst)


if __name__ == "__main__":
  main(sys.argv[1:])
