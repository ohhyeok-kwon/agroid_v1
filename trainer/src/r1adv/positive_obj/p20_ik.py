"""Positive 20 IK grids (rules.md 2.6): positive 18 ik.solve_grid / gravity_offsets on the P20 ladder bodies (fixed base,
pelvis at robot_fixed.PELVIS_POS), hand frame = body robot/<side>_o6_hand (box centre; O6-lite palm centre).

  python -m r1adv.positive_obj.p20_ik <out.npz> <body: box_wristfix|o6lite> <x_g> [z_g]
"""
from __future__ import annotations

import sys
import time

import mujoco
import numpy as np

from r1adv.positive_obj import ik as IK
from r1adv.positive_obj import p20_bodies as B
from r1adv.positive_obj.robot_fixed import PELVIS_POS


def fixed_spec(body: str) -> mujoco.MjSpec:
  spec = B.spec_box_wristfix() if body == "box_wristfix" else B.spec_o6lite()
  for j in [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]:
    spec.delete(j)
  spec.body("pelvis").pos = list(PELVIS_POS)
  return spec


def solve(body: str, x_g: float, z_g: float = IK.Z_G, ori_cost=None) -> dict:
  """ori_cost: box-frame orientation cost (roll, yaw, pitch about the palm normal); default positive 18 (0.5, 0.5, 0.02).
  The G2 grid uses (0.5, 0.5, 0.5) (rules_amend A1): a level forearm keeps the thumb sweep in the plane across the forearm."""
  model = fixed_spec(body).compile()
  orig, orig_cost = IK.build_model, IK.ORI_COST
  IK.build_model = lambda: model                          # ik.py is reused unchanged (module-level lookup)
  if ori_cost is not None:
    IK.ORI_COST = tuple(ori_cost)
  try:
    res = IK.solve_grid(x_g, z_g)
    res["psi_grid"] = IK.psi_grid(IK.X_G)                  # the carry turn is set by the object arc at x_g 0.40
    res["G"] = IK.gravity_offsets(res["Q"], res["psi_grid"])
  finally:
    IK.build_model, IK.ORI_COST = orig, orig_cost
  res["body"] = body
  res["ori_cost"] = np.array(ori_cost if ori_cost is not None else orig_cost)
  return res


if __name__ == "__main__":
  out, body, xg = sys.argv[1], sys.argv[2], float(sys.argv[3])
  zg = float(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else IK.Z_G
  oc = tuple(float(v) for v in sys.argv[5].split(",")) if len(sys.argv) > 5 else None
  t0 = time.time()
  r = solve(body, xg, zg, oc)
  np.savez_compressed(out, **r)
  print(IK.summary(r))
  print(f"{body} x_g {xg}: {time.time() - t0:.0f} s")
