"""agroid deploy contract for R1 joint-space policies.

Matches src/motion_manager/walking_action_manager (agroid_v1) and the manufacturer
AmpLocomotion files:
  * ONNX input "obs" [1, 810] = 10 frames x 81, oldest frame first (frame-major);
    frame = [gyro(3), projected_gravity(3), command(3), q - q0 (24), dq (24),
             previous action (24)], all in POLICY order, no extra scaling.
  * ONNX output "actions" [1, 24] in POLICY order;
    joint target = default_joint_pos + action * action_scale (no clipping).
  * POLICY order is PhysX breadth-first order (IsaacLab).  ENTITY order (MJCF
    depth-first) equals the agroid/SDK "dof" order, so ACTION_IDX below is
    exactly `action_idx` from amp_locomotion.json.
"""
from __future__ import annotations

import math

NUM_DOF = 24
FRAME_DIM = 81
HISTORY = 10
OBS_DIM = FRAME_DIM * HISTORY

# MJCF / mjlab entity order == agroid SDK dof order (without unused slots).
ENTITY_JOINTS = (
  "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
  "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
  "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
  "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
  "waist_roll_joint", "waist_yaw_joint",
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
  "left_elbow_joint", "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
  "right_elbow_joint", "right_wrist_roll_joint",
)

# Policy (IsaacLab BFS) order.
POLICY_JOINTS = (
  "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_roll_joint",
  "left_hip_roll_joint", "right_hip_roll_joint", "waist_yaw_joint",
  "left_hip_yaw_joint", "right_hip_yaw_joint",
  "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
  "left_knee_joint", "right_knee_joint",
  "left_shoulder_roll_joint", "right_shoulder_roll_joint",
  "left_ankle_pitch_joint", "right_ankle_pitch_joint",
  "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
  "left_ankle_roll_joint", "right_ankle_roll_joint",
  "left_elbow_joint", "right_elbow_joint",
  "left_wrist_roll_joint", "right_wrist_roll_joint",
)

# entity index i -> policy index  (agroid amp_locomotion.json "action_idx")
ACTION_IDX = tuple(POLICY_JOINTS.index(n) for n in ENTITY_JOINTS)
assert ACTION_IDX == (0, 3, 6, 10, 14, 18, 1, 4, 7, 11, 15, 19, 2, 5, 8, 12, 16, 20, 22,
                      9, 13, 17, 21, 23)
# policy index p -> entity index
POLICY_TO_ENTITY = tuple(ENTITY_JOINTS.index(n) for n in POLICY_JOINTS)

# Deploy constants (policy order), identical to amp_locomotion.json.
DEFAULT_POS = (-0.305, -0.305, 0.0, 0.04, -0.04, 0.0, -0.14, 0.14, 0.2, 0.2, 0.41, 0.41,
               0.2, -0.2, -0.137, -0.137, 0.0, 0.0, -0.06, 0.06, 0.8, 0.8, 0.0, 0.0)
DEFAULT_POS_BY_NAME = dict(zip(POLICY_JOINTS, DEFAULT_POS))

# Joint groups: (kp, kd, effort limit, armature). kp = I w^2, kd = 2*zeta*I*w,
# w = 2*pi*10 Hz, zeta = 2 (manufacturer convention, see review notes).
_W = 2.0 * math.pi * 10.0
GROUPS = {
  "big": dict(
    joints=(r".*_hip_pitch_joint", r".*_hip_roll_joint", r".*_hip_yaw_joint", r".*_knee_joint",
            r"waist_roll_joint", r"waist_yaw_joint", r".*_shoulder_pitch_joint",
            r".*_shoulder_roll_joint"),
    kp=84.335, kd=5.369, effort=60.0, armature=84.335 / _W**2),
  "ankle": dict(
    joints=(r".*_ankle_pitch_joint", r".*_ankle_roll_joint"),
    kp=31.413, kd=2.0, effort=50.0, armature=31.413 / _W**2),
  "small": dict(
    joints=(r".*_shoulder_yaw_joint", r".*_elbow_joint", r".*_wrist_roll_joint"),
    kp=15.706, kd=1.0, effort=33.0, armature=15.706 / _W**2),
}


def _group_of(name: str) -> str:
  import re
  for g, spec in GROUPS.items():
    if any(re.fullmatch(p, name) for p in spec["joints"]):
      return g
  raise KeyError(name)


ACTION_SCALE = tuple(0.25 * GROUPS[_group_of(n)]["effort"] / GROUPS[_group_of(n)]["kp"]
                     for n in POLICY_JOINTS)
KP = tuple(GROUPS[_group_of(n)]["kp"] for n in POLICY_JOINTS)
KD = tuple(GROUPS[_group_of(n)]["kd"] for n in POLICY_JOINTS)
assert [round(s, 3) for s in ACTION_SCALE[:14]] == [0.178] * 14

# Mirror (left/right) tables in policy order, from amp_locomotion.json.
MIRROR_INDEX = (1, 0, 2, 4, 3, 5, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14, 17, 16, 19, 18, 21, 20,
                23, 22)
MIRROR_SIGN = (1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
               1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0)
ANG_VEL_SIGN = (-1.0, 1.0, -1.0)
GRAVITY_SIGN = (1.0, -1.0, 1.0)
VEL_CMD_SIGN = (1.0, -1.0, -1.0)   # (vx, vy, wz)
LIN_VEL_SIGN = (1.0, -1.0, 1.0)

# Standing pelvis height of DEFAULT_POS with flat feet (MJCF FK), metres.
STAND_HEIGHT = 0.7313
