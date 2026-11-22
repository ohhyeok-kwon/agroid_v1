"""Positive 20 ladder-only bodies (rounds/R04/positive/r11_contact2, not the training body; nothing shared is changed).

box_wristfix  the O6 training body (r1lab.robot.get_spec_o6, O6 box 14 x 9 x 4 cm, 0.43 kg) with the wrist collider fitted
              to the bare wrist: `*_hand_collision` capsule r 0.030, fromto x 0.020 -> 0.050 in the wrist-roll frame (extent
              x -0.010 .. 0.080 = the flange where the O6 box starts). Evidence (checks_rules.md 3, mesh slices of
              *_wrist_roll_link.STL): housing x 0.05-0.08 radius p99 0.0306 / max 0.0315 m about the forearm axis; x 0.08-0.143
              is the stock hand (radius p99 0.045), removed when the O6 is mounted. The stock capsule spanned x 0.02-0.15 with
              r 0.03 and stood 1 cm proud of the O6 box face over x 0.08-0.15. Policy action 24.
o6lite        the same wrist fix; the O6 box replaced by an O6-lite hand from the Linker O6 manual (PC copy v1.0) dimensions:
              palm box 103 x 76.5 x 40 mm (flange x 0.08 to the MCP line, palm centre at wrist x 0.1315, body name
              `*_o6_hand` kept, palm normal = +z of that body as in the box hand), four fingers of two box links
              (37 + 37 mm, 12 mm wide, 14 mm thick; finger shape = box from check 1), finger pitch 19 mm, the palm-side finger
              face flush with the palm face; thumb of three box links (45 + 32 + 28 mm) on the thumb-side palm edge, 16.5 mm
              proximal of the palm centre.
              Joints per hand: 4 MCP (0-1.6 rad), 4 DIP (0-1.43, = 0.89 x MCP by joint equality), thumb yaw (0-1.36, 0 = in
              the palm plane pointing out of the thumb side, + toward the palm normal), thumb pitch (0-0.58, + toward the
              fingers), thumb IP (0-1.08, = 1.86 x pitch). Coupling ratios and ranges from the vendor URDF/MJCF as recorded in
              .agent/analysis/unified_policy_20261111/research/arm_ownership_and_hand.md 4.1 (no file downloaded).
              Actuators per hand: 6 position actuators (4 MCP, thumb pitch, thumb yaw), kp 600, kv 4.5, force limits
              fingers 2.55, thumb pitch 4, thumb yaw 1 N m (official MJCF values as recorded there), armature HAND_ARMATURE
              (not in the spec: chosen for dt 5 ms stability, an assumption), times GRIP_FORCE_SCALE (calibration, see the
              ladder rules). Policy-level hand action per hand: grip (-> the 4 MCP and thumb pitch targets, scaled to their
              ranges) and thumb yaw: 24 + 2 x 2 = 28. Mass: palm 0.326 + fingers 4 x 0.016 + thumb 0.040 = 0.43 kg (O6 0.37
              + adapter 0.06, as the box hand). Contacts inside one hand (palm, fingers, thumb, wrist link) are excluded.
"""
from __future__ import annotations

import dataclasses

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from r1lab import contract as C
from r1lab import robot as RB

SIDES = ("left", "right")
WRIST_CAP_FROMTO = (0.020, 0.0, 0.0, 0.050, 0.0, 0.0)
WRIST_CAP_R = 0.030
# O6-lite geometry (m, wrist-roll / palm frame)
FLANGE_X = 0.08
PALM_HALF = (0.0515, 0.03825, 0.02)
PALM_X = FLANGE_X + PALM_HALF[0]                       # 0.1315
FINGER_Y = (0.0285, 0.0095, -0.0095, -0.0285)          # index .. little for the left hand (thumb side = +y); right: mirrored
FINGER_Z = PALM_HALF[2] - 0.007                         # palm-side finger face flush with the palm face
FIN_HALF = (0.0185, 0.006, 0.007)
THUMB_ROOT = (-0.0165, 0.030, 0.010)                   # left hand; right hand mirrors y
THUMB_LINKS = (0.045, 0.032, 0.028)
THUMB_HALF_W = 0.008
M_PALM, M_PROX, M_DIST, M_TH = 0.326, 0.010, 0.006, (0.020, 0.012, 0.008)
MCP_RANGE, DIP_RANGE = (0.0, 1.6), (0.0, 1.43)
TH_YAW_RANGE, TH_PITCH_RANGE, TH_IP_RANGE = (0.0, 1.36), (0.0, 0.58), (0.0, 1.08)
DIP_RATIO, IP_RATIO = 0.89, 1.86
HAND_KP, HAND_KV = 600.0, 4.5
FORCE = {"mcp": 2.55, "th_pitch": 4.0, "th_yaw": 1.0}
HAND_ARMATURE = 0.01                                   # free-air close/open test at dt 5 ms (README 2): stable in C and MJWarp for 5e-4..0.03; 0.01 settles to 0.005 rad/s; assumption
GRIP_FORCE_SCALE = 0.44                                # finger-side power-grasp force on a 5-8 cm mesh cylinder 54-76 N (mean 64) ~ 70 N of the brief (README 2); 1.0 = official limits (158-176 N)
HAND_ACT_JOINTS = {s: tuple(f"{s}_f{i}_mcp" for i in range(4)) + (f"{s}_th_pitch", f"{s}_th_yaw") for s in SIDES}
# contact pads (rules_amend A1, coordinator 14:3x): small spheres 1.5 mm proud of the contact faces, so that every pad makes
# its own (single, stable) contact whatever the face misalignment; the faces themselves stay.
PAD_R, PAD_PROUD, PAD_MASS = 0.005, 0.0015, 1e-4
BOX_PADS = [(x, y) for x in (-0.05, 0.0, 0.05) for y in (-0.03, 0.0, 0.03)]       # O6 box +z face (palm side)
PALM_PADS = [(x, y) for x in (-0.035, 0.0, 0.035) for y in (-0.025, 0.0, 0.025)]  # O6-lite palm +z face


def _wrist_fix(spec: mujoco.MjSpec) -> None:
  for side in SIDES:
    g = spec.geom(f"{side}_hand_collision")
    g.fromto = list(WRIST_CAP_FROMTO)
    g.size = [WRIST_CAP_R, 0.0, 0.0]


def _pad(body, name, pos):
  return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[PAD_R, 0, 0], pos=list(pos), mass=PAD_MASS,
                       rgba=[0.9, 0.2, 0.2, 0.8], group=3, condim=1)


def spec_box_wristfix(pads: bool = False) -> mujoco.MjSpec:
  spec = RB.get_spec_o6()
  _wrist_fix(spec)
  if pads:
    from r1wb import robot as WB
    zc = WB.HAND_HALF[2] + PAD_PROUD - PAD_R
    for side in SIDES:
      hb = spec.body(f"{side}_o6_hand")
      for i, (x, y) in enumerate(BOX_PADS):
        _pad(hb, f"{side}_pad{i}_collision", (x, y, zc))
  return spec


def _box(body, name, half, pos, mass):
  return body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(half), pos=list(pos), mass=mass,
                       rgba=[0.2, 0.2, 0.6, 0.6], group=3, condim=1)


def _hinge(body, name, axis, rng, armature):
  return body.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis), range=list(rng), limited=True,
                        armature=armature)


def spec_o6lite(armature: float | None = None, pads: bool = False) -> mujoco.MjSpec:
  arm = HAND_ARMATURE if armature is None else armature
  spec = RB.get_spec()                                  # stock R1 (no O6 box); the O6-lite hand is added here
  _wrist_fix(spec)
  for side in SIDES:
    sy = 1.0 if side == "left" else -1.0
    wrist = spec.body(f"{side}_wrist_roll_link")
    palm = wrist.add_body(name=f"{side}_o6_hand", pos=[PALM_X, 0.0, 0.0])
    _box(palm, f"{side}_o6_collision", PALM_HALF, (0, 0, 0), M_PALM)
    if pads:
      for i, (x, y) in enumerate(PALM_PADS):
        _pad(palm, f"{side}_pad{i}_collision", (x, sy * y, PALM_HALF[2] + PAD_PROUD - PAD_R))
    for i, fy in enumerate(FINGER_Y):
      prox = palm.add_body(name=f"{side}_f{i}_prox", pos=[PALM_HALF[0], sy * fy, FINGER_Z])
      _hinge(prox, f"{side}_f{i}_mcp", (0, -1, 0), MCP_RANGE, arm)
      _box(prox, f"{side}_f{i}_prox_collision", FIN_HALF, (FIN_HALF[0], 0, 0), M_PROX)
      if pads:
        _pad(prox, f"{side}_f{i}_proxpad_collision", (FIN_HALF[0], 0, FIN_HALF[2] + PAD_PROUD - PAD_R))
      dist = prox.add_body(name=f"{side}_f{i}_dist", pos=[2 * FIN_HALF[0], 0.0, 0.0])
      _hinge(dist, f"{side}_f{i}_dip", (0, -1, 0), DIP_RANGE, arm)
      _box(dist, f"{side}_f{i}_dist_collision", FIN_HALF, (FIN_HALF[0], 0, 0), M_DIST)
      if pads:
        _pad(dist, f"{side}_f{i}_distpad_collision", (FIN_HALF[0], 0, FIN_HALF[2] + PAD_PROUD - PAD_R))
      spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f"{side}_f{i}_dip", name2=f"{side}_f{i}_mcp",
                        data=[0.0, DIP_RATIO, 0.0, 0.0, 0.0] + [0.0] * 6)
    tx, ty, tz = THUMB_ROOT
    base = palm.add_body(name=f"{side}_th_base", pos=[tx, sy * ty, tz])
    _hinge(base, f"{side}_th_yaw", (sy, 0, 0), TH_YAW_RANGE, arm)
    L0, L1, L2 = THUMB_LINKS
    w = THUMB_HALF_W
    _box(base, f"{side}_th_base_collision", (w, L0 / 2, w), (0, sy * L0 / 2, 0), M_TH[0])
    if pads:
      _pad(base, f"{side}_th_basepad_collision", (w + PAD_PROUD - PAD_R, sy * L0 / 2, 0))
    prox = base.add_body(name=f"{side}_th_prox", pos=[0.0, sy * L0, 0.0])
    _hinge(prox, f"{side}_th_pitch", (0, 0, -sy), TH_PITCH_RANGE, arm)
    _box(prox, f"{side}_th_prox_collision", (w, L1 / 2, w), (0, sy * L1 / 2, 0), M_TH[1])
    if pads:
      _pad(prox, f"{side}_th_proxpad_collision", (w + PAD_PROUD - PAD_R, sy * L1 / 2, 0))
    dist = prox.add_body(name=f"{side}_th_dist", pos=[0.0, sy * L1, 0.0])
    _hinge(dist, f"{side}_th_ip", (0, 0, -sy), TH_IP_RANGE, arm)
    _box(dist, f"{side}_th_dist_collision", (w, L2 / 2, w), (0, sy * L2 / 2, 0), M_TH[2])
    if pads:
      _pad(dist, f"{side}_th_distpad_collision", (w + PAD_PROUD - PAD_R, sy * L2 / 2, 0))
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1=f"{side}_th_ip", name2=f"{side}_th_pitch",
                      data=[0.0, IP_RATIO, 0.0, 0.0, 0.0] + [0.0] * 6)
    # no contacts inside one hand (palm, fingers, thumb) nor with the wrist link
    hand_bodies = [f"{side}_wrist_roll_link", f"{side}_o6_hand"] + [f"{side}_f{i}_{p}" for i in range(4) for p in ("prox", "dist")] \
      + [f"{side}_th_{p}" for p in ("base", "prox", "dist")]
    for a in range(len(hand_bodies)):
      for b in range(a + 1, len(hand_bodies)):
        spec.add_exclude(bodyname1=hand_bodies[a], bodyname2=hand_bodies[b])
  return spec


def hand_actuators(force_scale: float | None = None, armature: float | None = None) -> tuple:
  k = GRIP_FORCE_SCALE if force_scale is None else force_scale
  arm = HAND_ARMATURE if armature is None else armature
  common = dict(stiffness=HAND_KP, damping=HAND_KV, armature=arm, frictionloss=0.0, viscous_damping=0.0)
  return (BuiltinPositionActuatorCfg(target_names_expr=(r"(left|right)_f[0-3]_mcp",), effort_limit=FORCE["mcp"] * k, **common),
          BuiltinPositionActuatorCfg(target_names_expr=(r"(left|right)_th_pitch",), effort_limit=FORCE["th_pitch"] * k, **common),
          BuiltinPositionActuatorCfg(target_names_expr=(r"(left|right)_th_yaw",), effort_limit=FORCE["th_yaw"] * k, **common))


def robot_cfg(body: str, fixed: bool = False, pelvis_pos=None, force_scale: float | None = None,
              armature: float | None = None, pads: bool = False) -> EntityCfg:
  """EntityCfg of a ladder body ('box_wristfix' | 'o6lite'), training collision cfg and deploy PD for the 24 joints.
  fixed=True: the free joint removed (mjlab wraps the entity in a mocap body; pelvis pose = init pos or mocap)."""
  base = RB.get_r1_cfg(command_delay_max=0, self_collision=True, hand="o6")
  if body == "box_wristfix":
    spec_fn, acts = (lambda: spec_box_wristfix(pads)), base.articulation.actuators          # noqa: E731
  elif body == "o6lite":
    spec_fn = lambda: spec_o6lite(armature, pads)               # noqa: E731
    acts = tuple(base.articulation.actuators) + hand_actuators(force_scale, armature)
  else:
    raise KeyError(body)
  if fixed:
    inner = spec_fn

    def spec_fn():                                              # noqa: F811
      spec = inner()
      for j in [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]:
        spec.delete(j)
      spec.body("pelvis").pos = [0.0, 0.0, 0.0]
      return spec
  init = base.init_state
  if pelvis_pos is not None:
    init = dataclasses.replace(init, pos=tuple(pelvis_pos))
  art = dataclasses.replace(base.articulation, actuators=tuple(acts))
  return dataclasses.replace(base, spec_fn=spec_fn, init_state=init, articulation=art)


def grip_targets(side: str, grip, thumb_yaw) -> np.ndarray:
  """Per-hand position targets (4 MCP, thumb pitch, thumb yaw) from the policy-level hand action:
  grip in [0, 1] -> MCP = grip x 1.6 rad, thumb pitch = grip x 0.58 rad; thumb_yaw in [0, 1] -> x 1.36 rad."""
  g = np.clip(np.asarray(grip, float), 0.0, 1.0)
  y = np.clip(np.asarray(thumb_yaw, float), 0.0, 1.0)
  return np.stack([g * MCP_RANGE[1]] * 4 + [g * TH_PITCH_RANGE[1], y * TH_YAW_RANGE[1]], axis=-1)


def body_signature(body: str, **kw) -> dict:
  from r1lab import body_hash as BH
  return BH.entity_signature(robot_cfg(body, **kw))


if __name__ == "__main__":
  import json
  import sys
  out = {b: body_signature(b) for b in ("box_wristfix", "o6lite")}
  out["training_o6_reference"] = __import__("r1lab.body_hash", fromlist=["x"]).entity_signature(
    RB.get_r1_cfg(command_delay_max=0, self_collision=True, hand="o6"))
  txt = json.dumps(out, indent=1)
  if len(sys.argv) > 1:
    open(sys.argv[1], "w").write(txt)
  print(txt)
