"""Positive 20 fist bodies for the hold test (rounds/R04/positive/r11_hold/; coordinator 16:0x, user: 'assume the stock
Unitree R1 fist, no finger joints').

Visual fist of r1.xml <side>_wrist_roll_link (mesh vertices with x >= 0.08 in the wrist_roll_link frame, measured
15:5x, fist_geom.json): x 0.0801..0.1427, y -0.0402..0.0209 (left; right mirrored), z -0.0311..0.0344.
Stock collision: <side>_hand_collision capsule fromto (0.05,0,0)-(0.12,0,0), r 0.03 -> x 0.02..0.15, y,z +-0.03.

  fist      the stock R1 hand unchanged (r1lab hand='stock').
  fistfit   the stock capsule cut back to the bare wrist (fromto 0.02..0.05, r 0.03: ends at x 0.08, as the ladder
            bodies' wrist fix) plus <side>_o6_collision = box equal to the axis-aligned bounding box of the visual fist.
  fistpads  fistfit + 3 x 3 sphere pads (r 5 mm, 1.5 mm proud) on the fist box +z face.
Every arm gets a jointless, massless frame body <side>_o6_hand at FIST_FRAME on the wrist_roll_link (IK / trace frame:
x = forearm, z = toward the object in the squeeze; ladder convention) and a site <side>_patch at the centre of the arm's
contact patch (capsule: the capsule line at z = r, middle of its straight part; box: the +z face centre). The fist's mass
stays the wrist_roll_link inertial of r1.xml (no mass added; pads 0.1 g each).
Collision: the training cfg (FULL_COLLISION) with the pads on bits 2/2 (they touch objects only; objects 3/3).
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
from mjlab.utils.spec_config import CollisionCfg

from r1lab import robot as RB

SIDES = ("left", "right")
FIST_LO = (0.0801, -0.0402, -0.0311)                 # left hand; right: y mirrored
FIST_HI = (0.1427, 0.0209, 0.0344)
FIST_C = tuple((a + b) / 2 for a, b in zip(FIST_LO, FIST_HI))     # (0.1114, -0.00965, 0.00165)
FIST_HALF = tuple((b - a) / 2 for a, b in zip(FIST_LO, FIST_HI))  # (0.0313, 0.03055, 0.03275)
FIST_FRAME = (FIST_C[0], 0.0, 0.0)
STOCK_CAP = ((0.05, 0.0, 0.0, 0.12, 0.0, 0.0), 0.03)
WRIST_CAP = ((0.02, 0.0, 0.0, 0.05, 0.0, 0.0), 0.03)
PAD_R, PAD_PROUD, PAD_MASS = 0.005, 0.0015, 1e-4
PADS = [(x, y) for x in (-0.022, 0.0, 0.022) for y in (-0.021, 0.0, 0.021)]   # on the fist box +z face, box frame
ARMS = ("fist", "fistfit", "fistpads")
# contact patch per arm in the <side>_o6_hand frame (left hand; right: y mirrored): centre and face offset r_face (the
# distance from the frame's x axis to the outermost contact surface along +z)
R_FACE = {"fist": STOCK_CAP[1], "fistfit": FIST_C[2] + FIST_HALF[2], "fistpads": FIST_C[2] + FIST_HALF[2] + PAD_PROUD}
PATCH = {"fist": ((STOCK_CAP[0][0] + STOCK_CAP[0][3]) / 2 - FIST_FRAME[0], 0.0),          # (dx, dy) in the frame
         "fistfit": (0.0, FIST_C[1]), "fistpads": (0.0, FIST_C[1])}
PAD_RE = r"^(left|right)_pad[0-9]+_collision$"
FOOT_RE = r"^(left|right)_foot[1-7]_collision$"


def collision_cfg() -> CollisionCfg:
  """r1lab FULL_COLLISION with the pads on bits 2/2."""
  return CollisionCfg(geom_names_expr=(".*_collision",),
                      contype={PAD_RE: 2, ".*": 1}, conaffinity={PAD_RE: 2, ".*": 1},
                      condim={FOOT_RE: 3, ".*_collision": 1}, priority={FOOT_RE: 1, ".*": 0},
                      friction={FOOT_RE: (0.8,)})


def spec_fist(arm: str) -> mujoco.MjSpec:
  if arm not in ARMS:
    raise KeyError(arm)
  spec = RB.get_spec()
  for side in SIDES:
    sy = 1.0 if side == "left" else -1.0
    wrist = spec.body(f"{side}_wrist_roll_link")
    frame = wrist.add_body(name=f"{side}_o6_hand", pos=list(FIST_FRAME))
    dx, dy = PATCH[arm]
    frame.add_site(name=f"{side}_patch", pos=[dx, sy * dy, R_FACE[arm]])
    frame.add_site(name=f"{side}_palm_box", pos=[0.0, 0.0, 0.0])        # check-2 frame name (p20_check2.Solver)
    if arm == "fist":
      continue
    g = spec.geom(f"{side}_hand_collision")
    g.fromto = list(WRIST_CAP[0])
    g.size = [WRIST_CAP[1], 0.0, 0.0]
    frame.add_geom(name=f"{side}_o6_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(FIST_HALF),
                   pos=[0.0, sy * FIST_C[1], FIST_C[2]], mass=0.0, rgba=[0.2, 0.6, 0.2, 0.4], group=3, condim=1)
    if arm == "fistpads":
      zc = FIST_C[2] + FIST_HALF[2] + PAD_PROUD - PAD_R
      for i, (x, y) in enumerate(PADS):
        frame.add_geom(name=f"{side}_pad{i}_collision", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[PAD_R, 0, 0],
                       pos=[x, sy * (FIST_C[1] + y), zc], mass=PAD_MASS, rgba=[0.9, 0.2, 0.2, 0.8], group=3, condim=1)
  return spec


def robot_cfg(arm: str, fixed: bool = False, pelvis_pos=None):
  """EntityCfg: training body r1lab get_r1_cfg(hand='stock') (deploy PD, 24 joints) with the fist arm's spec.
  fixed=True: free joint removed (mjlab wraps the entity in a mocap body)."""
  base = RB.get_r1_cfg(command_delay_max=0, self_collision=True, hand="stock")

  def spec_fn():
    spec = spec_fist(arm)
    if fixed:
      for j in [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]:
        spec.delete(j)
      spec.body("pelvis").pos = [0.0, 0.0, 0.0]
    return spec
  init = base.init_state if pelvis_pos is None else dataclasses.replace(base.init_state, pos=tuple(pelvis_pos))
  return dataclasses.replace(base, spec_fn=spec_fn, init_state=init, collisions=(collision_cfg(),))


def fixed_spec(arm: str) -> mujoco.MjSpec:
  from r1adv.positive_obj.robot_fixed import PELVIS_POS
  spec = spec_fist(arm)
  for j in [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]:
    spec.delete(j)
  spec.body("pelvis").pos = list(PELVIS_POS)
  return spec


def body_signature(arm: str) -> dict:
  from r1lab import body_hash as BH
  return BH.entity_signature(robot_cfg(arm))


def solve_ik_grid(x_g: float, z_g: float) -> dict:
  """positive 18 ik.solve_grid + gravity_offsets on the fistfit fixed body (same kinematics and frame for all three arms;
  the fist box stands in for the hand geom in the IK's collision avoidance and hand_low)."""
  from r1adv.positive_obj import ik as IK
  model = fixed_spec("fistfit").compile()
  orig = IK.build_model
  IK.build_model = lambda: model                       # noqa: E731
  try:
    res = IK.solve_grid(x_g, z_g)
    res["psi_grid"] = IK.psi_grid(IK.X_G)
    res["G"] = IK.gravity_offsets(res["Q"], res["psi_grid"])
  finally:
    IK.build_model = orig
  res["body"] = "fist"
  return res


if __name__ == "__main__":
  import json
  import sys
  if sys.argv[1] == "sig":
    out = {a: body_signature(a) for a in ARMS}
    out["training_stock_reference"] = __import__("r1lab.body_hash", fromlist=["x"]).entity_signature(
      RB.get_r1_cfg(command_delay_max=0, self_collision=True, hand="stock"))
    open(sys.argv[2], "w").write(json.dumps(out, indent=1))
    print({k: (v["total"][:12], v.get("counts")) for k, v in out.items()})
  elif sys.argv[1] == "ik":
    import time
    t0 = time.time()
    xg, zg = float(sys.argv[3]), float(sys.argv[4])
    r = solve_ik_grid(xg, zg)
    np.savez_compressed(sys.argv[2], **r)
    from r1adv.positive_obj import ik as IK
    print(IK.summary(r))
    print(f"fist IK x_g {xg} z_g {zg}: {time.time() - t0:.0f} s")
