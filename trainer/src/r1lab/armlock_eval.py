"""Evaluation helper for ARMLOCK policies (positive session 15): put the arm lock of r1lab.armlock (sha 2ea974a5, the
training body of the r9-ARMLOCK runs) on another environment's O6 robot, e.g. an evaluation runner's env cfg. A policy
trained on the locked body never learned its ten arm outputs; on a body with free arms they would move the arms at
random, so every ARMLOCK evaluation must run on the locked body.
  lock_env_cfg(cfg)   cfg.scene.entities['robot']: spec_fn wrapped with the ten joint welds at LOCK_POSE (same constants,
                      solref / solimp as r1lab.armlock), every BuiltinPositionActuatorCfg split into its body joints
                      (unchanged) and its arm joints (kp = kd = 0, no force limit); ArmLockReset appended after the other
                      events (it checks the built model and refuses to run without the lock)
  check(env)          {'body': r1lab.body_hash total, 'lock': lock signature}; the training body is
                      9e3282b9afe4582e... / cca95cf85ee4cf2d... (an evaluation body with other actuator delays or masses
                      may differ in the body hash; the lock signature must be cca95cf8)
"""
from __future__ import annotations

import copy
import dataclasses
import re

import mujoco

from r1lab import armlock as AL

TRAIN_BODY = "9e3282b9afe4582ea24de8c556c4274996c910fb658ae2407042ced58f49d476"
TRAIN_LOCK = "cca95cf85ee4cf2d0803245527d3337abedcfe6e720b87accc578f1699bc08fc"


def add_arm_welds(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """the ten joint equalities of r1lab.armlock.get_spec_o6_armlock on any spec with the R1 arm joints."""
  for name in AL.ARM_JOINTS:
    joint = spec.joint(name)
    if joint is None or joint.type != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError(f"arm joint {name} missing or not a hinge")
    eq = spec.add_equality()
    eq.name = AL.EQ_PREFIX + name
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
    eq.name1 = name
    eq.name2 = ""
    data = [0.0] * len(eq.data)
    data[0] = AL.LOCK_POSE[name] - float(joint.ref)
    eq.data = data
    eq.solref = list(AL.LOCK_SOLREF)
    eq.solimp = list(AL.LOCK_SOLIMP)
  return spec


def _wrap(fn):
  def spec_fn():
    return add_arm_welds(fn())
  spec_fn.__name__ = f"armlocked_{getattr(fn, '__name__', 'spec')}"
  spec_fn.__qualname__ = spec_fn.__name__
  return spec_fn


def lock_env_cfg(cfg, entity: str = "robot") -> None:
  from mjlab.actuator import BuiltinPositionActuatorCfg
  from mjlab.managers.event_manager import EventTermCfg
  arm = re.compile("shoulder|elbow|wrist")
  old = cfg.scene.entities[entity]
  new = copy.deepcopy(old)
  new.spec_fn = _wrap(old.spec_fn)
  acts = []
  for a in old.articulation.actuators:
    if not isinstance(a, BuiltinPositionActuatorCfg):
      raise TypeError(f"actuator {type(a).__name__} not supported (BuiltinPositionActuatorCfg expected)")
    body = tuple(p for p in a.target_names_expr if not arm.search(p))
    arms = tuple(p for p in a.target_names_expr if arm.search(p))
    if body:
      acts.append(dataclasses.replace(a, target_names_expr=body))
    if arms:
      acts.append(dataclasses.replace(a, target_names_expr=arms, stiffness=0.0, damping=0.0, effort_limit=None))
  new.articulation.actuators = tuple(acts)
  cfg.scene.entities[entity] = new
  if "arm_lock_reset" in cfg.events:
    raise ValueError("arm_lock_reset already present")
  cfg.events["arm_lock_reset"] = EventTermCfg(func=AL.ArmLockReset, mode="reset", params={})


def check(env) -> dict:
  from r1lab.body_hash import env_signature
  rep = AL.model_lock_report(env.sim.mj_model)
  return dict(body=env_signature(env)["total"], lock=rep["lock_signature"], lock_ok=rep["ok"], problems=rep["problems"],
              lock_is_training_lock=rep["lock_signature"] == TRAIN_LOCK)
