"""Unitree R1 with Linker O6 hands as an mjlab entity.

The hand is a rigid payload on each wrist (fingers are driven outside the whole-body policy): one extra body per
wrist with a box collision geom. Its mass is a bracket, not a measured value; training randomises it.
"""
from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

import r1lab
from r1lab import contract as C

R1_XML = Path(r1lab.__file__).parent / "assets" / "r1" / "r1.xml"
JOINTS = C.ENTITY_JOINTS                       # MJCF order: legs (12), waist (2), arms (10)
LEG_JOINTS, WAIST_JOINTS, ARM_JOINTS = JOINTS[:12], JOINTS[12:14], JOINTS[14:]
DEFAULT_POSE = dict(C.DEFAULT_POS_BY_NAME)
STAND_HEIGHT = C.STAND_HEIGHT
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_GEOMS = r"^(left|right)_foot[1-7]_collision$"
HAND_BODIES = ("left_o6_hand", "right_o6_hand")
# Linker O6 (0.37 kg per the manual and the vendor URDF) + adapter as one rigid body beyond the stock wrist end:
# about 0.43 kg mounted. Training randomises around it (see tracking.env_cfg); geometry is a palm-sized box.
HAND_MASS = 0.43
HAND_POS = (0.15, 0.0, 0.0)                    # wrist frame: +x points along the forearm
HAND_HALF = (0.07, 0.045, 0.02)
# Tracked bodies (reference poses are compared on these) and the anchor the comparison is relative to.
ANCHOR_BODY = "torso_link"
TRACKED_BODIES = (
  "pelvis",
  "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
  "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link", "left_elbow_link", "left_wrist_roll_link",
  "right_shoulder_roll_link", "right_elbow_link", "right_wrist_roll_link",
)
END_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link")
# Joint position target = default pose + action * scale (0.25 * effort limit / kp, as in the deploy contract).
ACTION_SCALE = {name: C.ACTION_SCALE[C.POLICY_JOINTS.index(name)] for name in JOINTS}


def get_spec(hand_mass: float = HAND_MASS) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(R1_XML))
  for side in ("left", "right"):
    wrist = spec.body(f"{side}_wrist_roll_link")
    hand = wrist.add_body(name=f"{side}_o6_hand", pos=list(HAND_POS))
    hand.add_geom(name=f"{side}_o6_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(HAND_HALF),
                  mass=hand_mass, rgba=[0.2, 0.2, 0.6, 0.5], group=3, condim=1)
  return spec


def write_xml(path: Path, hand_mass: float = HAND_MASS) -> Path:
  """The same model as an XML next to the stock meshes (CPU evaluation loads it directly)."""
  spec = get_spec(hand_mass)
  spec.compile()
  path.write_text(spec.to_xml())
  return path


def actuators(delay_steps: int = 0) -> tuple[BuiltinPositionActuatorCfg, ...]:
  """Deploy PD gains; optional command delay in physics steps (5 ms each)."""
  return tuple(
    BuiltinPositionActuatorCfg(
      target_names_expr=group["joints"], stiffness=group["kp"], damping=group["kd"],
      effort_limit=group["effort"], armature=group["armature"], frictionloss=0.1, viscous_damping=0.05,
      delay_min_lag=0, delay_max_lag=delay_steps, delay_hold_prob=0.9 if delay_steps > 0 else 0.0)
    for group in C.GROUPS.values())


COLLISIONS = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1, conaffinity=1,
  condim={FOOT_GEOMS: 3, ".*_collision": 1},
  priority={FOOT_GEOMS: 1, ".*": 0},
  friction={FOOT_GEOMS: (0.8,)},
)


def robot_cfg(delay_steps: int = 0) -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, STAND_HEIGHT + 0.01), joint_pos=DEFAULT_POSE,
                                         joint_vel={".*": 0.0}),
    collisions=(COLLISIONS,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(actuators=actuators(delay_steps), soft_joint_pos_limit_factor=0.95),
  )
