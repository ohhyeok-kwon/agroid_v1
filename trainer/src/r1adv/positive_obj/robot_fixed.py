"""Fixed-base R1 with the O6 box hands (positive 18, R11 contact ladder).

The training body r1lab.robot.get_r1_cfg(hand="o6") (collision cfg FULL_COLLISION, deploy PD actuators) with the floating
base joint removed. mjlab wraps a fixed-base entity in a mocap body, so the pelvis is welded to that body (pose = entity
init_state.pos, or per world via the mocap pose) and balance is taken out of the test. Kinematics, collision geoms,
masses and actuators are those of the training body.
"""
from __future__ import annotations

import dataclasses

import mujoco
from mjlab.entity import EntityCfg

from r1lab import contract as C
from r1lab import robot as RB

# Pelvis 1 cm above the flat-feet standing height: in the default pose the feet hang 1 cm above the floor (no floor contact).
PELVIS_POS = (0.0, 0.0, C.STAND_HEIGHT + 0.01)
HAND_BODIES = RB.O6_HAND_BODIES                       # ("left_o6_hand", "right_o6_hand")
HAND_GEOMS = ("left_o6_collision", "right_o6_collision")
WRIST_CAPSULES = ("left_hand_collision", "right_hand_collision")     # stock wrist capsule, still on the O6 body


def get_spec_o6_fixed() -> mujoco.MjSpec:
  spec = RB.get_spec_o6()
  for joint in [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]:
    spec.delete(joint)
  spec.body("pelvis").pos = [0.0, 0.0, 0.0]             # the wrapping mocap body carries the pelvis pose
  return spec


def fixed_robot_cfg(pelvis_pos: tuple[float, float, float] = PELVIS_POS) -> EntityCfg:
  cfg = RB.get_r1_cfg(command_delay_max=0, self_collision=True, hand="o6")
  init = dataclasses.replace(cfg.init_state, pos=tuple(pelvis_pos))
  return dataclasses.replace(cfg, spec_fn=get_spec_o6_fixed, init_state=init)
