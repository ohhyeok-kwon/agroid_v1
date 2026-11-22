"""Arm-locked R1 O6 body for the diagnosis cell ARMLOCK (positive session 15; coordinator 2026-11-19 02:4x KST,
.agent/adversarial/rounds/R04/09B_r9_synthesis.md 4.3, panel 09A physics test B / biomechanics candidate B).
DIAGNOSIS ONLY, never a deployment body (indoor hand support needs the arms).

Question: continued from the r6d long parents with the B0 recipe (A-GMM-O6: Hold objective L + D, standing commands, flat
floor, option-S hits, entropy 0.005, warm-up 50 -> main 752) on a body whose arms cannot move, does the policy hold the
90 / 180 deg 20 N s hits?  It separates 'the arms are needed' from 'arm use is a lineage habit' by training, which the
replay tests (frozen arms on a policy trained with arms) could not.

The lock (the only change from careful_walk_r7a_gmm_o6; observations, actions, rewards, hits, PPO untouched):
  weld    the ten arm joints (shoulder pitch / roll / yaw, elbow, wrist roll, both sides) are held at LOCK_POSE by MuJoCo
          joint equality constraints (mjEQ_JOINT with one joint: q - qpos0 = a0), solved implicitly by the constraint
          solver (mujoco_warp 3.11 has the kernel: constraint.py _equality_joint) with the hard settings LOCK_SOLREF /
          LOCK_SOLIMP: the default (0.02, 1) / (0.9, 0.95, 0.001) let the elbows give 0.094 rad and the wrists 0.061 rad
          under the 4-25 N s hits and hand-hip contacts (GPU smoke, p1/4000, 1,024 env x 750 steps; tolerance written
          before it: 0.01 rad). Time constant 0.01 s = 2 x the 5 ms physics step (MuJoCo's stability floor)
  ignore  the arm actuators keep their joints, armature, friction loss and viscous damping but get kp = kd = 0 (and no
          force limit, the force being identically zero), so the policy's ten arm outputs produce no torque (the 24-dim
          action of the deploy contract stays, so the parents' weights load; the outputs are still fed back as
          'previous action')
  start   ArmLockReset, appended after every other reset event, writes the arm joints to LOCK_POSE with zero velocity:
          reset_joints places them at the deploy default pose +-0.05 rad, +-0.2 rad/s, and without this term the weld
          would snap them by up to 0.24 rad (shoulder roll) at every episode start. The term also checks the built model
          (fail closed): ten joint equalities on the right joints with the right constants, arm actuators with zero gains.
LOCK_POSE = the arm entries of L's own target q*_r7a (hold_target_r7a_v1.json, sha256 41a5f5ec: the r6d parents'
quiet-standing arms, mirror-averaged), NOT the deploy default pose: L's posture channel is the RMS over all 24 joints with
sigma 0.02 rad, and the default-pose arms sit 0.076 rad RMS (24-joint) from q* (shoulder roll 0.242 rad), which would cap
L at about 0.12 and cut its gradient for every other joint about 70-fold. At q* the arm residual is zero, so L's landscape
for legs and waist is the one B0 sees when its arms rest. Observations keep q - q0 with the deploy q0.
Expected side effects (recorded for the readers, not corrected): (1) the ten arm action dimensions get the entropy bonus
with no counter-pressure, so their std can rise (log leg / waist and arm std separately); (2) the hands stay near the
thighs (shoulder roll -+0.042 rad), so hand-leg contact during side steps can confound 90 deg falls.
"""
from __future__ import annotations

import hashlib
import json
import re

import mujoco
import torch

from r1lab import contract as C

ARM_JOINTS = (
  "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
  "left_wrist_roll_joint",
  "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
  "right_wrist_roll_joint",
)
assert set(ARM_JOINTS) == {n for n in C.ENTITY_JOINTS if re.search("shoulder|elbow|wrist", n)}
LOCK_TARGET = dict(file="hold_target_r7a_v1.json",
                   sha256="41a5f5ec3e56e587ccd60f170a036a65cae75a98247dad23a493a378b15bbb9c")
# Arm entries of hold_target_r7a_v1.json 'target_rad' (= its 'arm_target_rad'), full precision; checked against the pinned
# file whenever an env cfg is built (lock_pose_check).
LOCK_POSE = {
  "left_shoulder_pitch_joint": 0.1862843023145771, "left_shoulder_roll_joint": -0.04198904926834594,
  "left_shoulder_yaw_joint": -0.06202415083350518, "left_elbow_joint": 0.7256022518015067,
  "left_wrist_roll_joint": -0.043259174844206424,
  "right_shoulder_pitch_joint": 0.1862843023145771, "right_shoulder_roll_joint": 0.04198904926834594,
  "right_shoulder_yaw_joint": 0.06202415083350518, "right_elbow_joint": 0.7256022518015067,
  "right_wrist_roll_joint": 0.043259174844206424,
}
EQ_PREFIX = "armlock_"
# Hard weld (see the module docstring): solref (time constant s, damping ratio), solimp (dmin, dmax, width, midpoint, power).
LOCK_SOLREF = (0.01, 1.0)
LOCK_SOLIMP = (0.99, 0.999, 0.001, 0.5, 2.0)


def lock_pose_check() -> None:
  """LOCK_POSE must equal the arm entries of the pinned q*_r7a file (read through hold_r2.pinned, sha256 checked)."""
  from r1adv.positive import hold_r2 as R2
  target = json.loads(R2.pinned(LOCK_TARGET).read_text())
  q = dict(zip(target["joint_names"], target["target_rad"]))
  bad = {n: (q.get(n), v) for n, v in LOCK_POSE.items() if q.get(n) != v}
  if bad:
    raise ValueError(f"LOCK_POSE differs from {LOCK_TARGET['file']}: {bad}")


def get_spec_o6_armlock() -> mujoco.MjSpec:
  """r1lab.robot.get_spec_o6 plus one joint equality per arm joint holding it at LOCK_POSE."""
  from r1lab.robot import get_spec_o6
  spec = get_spec_o6()
  for name in ARM_JOINTS:
    joint = spec.joint(name)
    if joint is None or joint.type != mujoco.mjtJoint.mjJNT_HINGE:
      raise ValueError(f"arm joint {name} missing or not a hinge")
    eq = spec.add_equality()
    eq.name = EQ_PREFIX + name
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
    eq.name1 = name
    eq.name2 = ""
    data = [0.0] * len(eq.data)
    data[0] = LOCK_POSE[name] - float(joint.ref)          # q - qpos0 = a0, qpos0 = joint ref
    eq.data = data
    eq.solref = list(LOCK_SOLREF)
    eq.solimp = list(LOCK_SOLIMP)
  return spec


def make_actuators_armlock(command_delay_max: int = 0):
  """r1lab.robot.make_actuators with every group split into its body joints (unchanged) and its arm joints (kp = kd = 0;
  armature, friction loss, viscous damping and delay settings as the group; no force limit: the force is identically
  zero, and mjlab's informational ctrlrange divides the effort limit by kp)."""
  from mjlab.actuator import BuiltinPositionActuatorCfg
  arm_pattern = re.compile("shoulder|elbow|wrist")
  acts = []
  for spec in C.GROUPS.values():
    body = tuple(p for p in spec["joints"] if not arm_pattern.search(p))
    arm = tuple(p for p in spec["joints"] if arm_pattern.search(p))
    for patterns, kp, kd, effort in ((body, spec["kp"], spec["kd"], spec["effort"]), (arm, 0.0, 0.0, None)):
      if patterns:
        acts.append(BuiltinPositionActuatorCfg(
          target_names_expr=patterns, stiffness=kp, damping=kd, effort_limit=effort,
          armature=spec["armature"], frictionloss=0.1, viscous_damping=0.05, delay_min_lag=0,
          delay_max_lag=command_delay_max, delay_hold_prob=0.9 if command_delay_max > 0 else 0.0))
  return tuple(acts)


class ArmLockReset:
  """Reset event: arm joints to LOCK_POSE, zero velocity (after reset_joints). At construction it checks the built model
  (fail closed): exactly the ten armlock joint equalities, on the arm joints, constant LOCK_POSE - qpos0, active; arm
  actuators with zero gains, every other actuator with its deploy gains."""

  def __init__(self, cfg, env):
    del cfg
    self.asset = env.scene["robot"]
    ids, names = self.asset.find_joints(ARM_JOINTS, preserve_order=True)
    if tuple(names) != ARM_JOINTS:
      raise ValueError(f"arm joints resolved as {names}")
    self.joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    self.pose = torch.tensor([LOCK_POSE[n] for n in ARM_JOINTS], device=env.device, dtype=torch.float32)
    self.report = model_lock_report(env.sim.mj_model)
    if not self.report["ok"]:
      raise RuntimeError(f"arm lock not in the built model: {self.report['problems']}")

  def reset(self, env_ids=None):
    del env_ids

  def __call__(self, env, env_ids):
    if env_ids is None:
      env_ids = torch.arange(env.num_envs, device=env.device)
    n = len(env_ids)
    self.asset.write_joint_state_to_sim(self.pose.expand(n, -1).clone(),
                                        torch.zeros(n, len(ARM_JOINTS), device=env.device),
                                        joint_ids=self.joint_ids, env_ids=env_ids)


def model_lock_report(m) -> dict:
  """The arm lock as found in a compiled MjModel (entity alone or whole scene, names with or without 'robot/')."""
  strip = lambda s: s.split("/", 1)[1] if s and s.startswith("robot/") else s   # noqa: E731
  jname = lambda j: strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, int(j)) or "")   # noqa: E731
  problems, eqs = [], []
  for e in range(m.neq):
    name = strip(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_EQUALITY, e) or "")
    if not name.startswith(EQ_PREFIX):
      continue
    j1, j2 = int(m.eq_obj1id[e]), int(m.eq_obj2id[e])
    adr = int(m.jnt_qposadr[j1]) if j1 >= 0 else -1
    row = dict(name=name, type=int(m.eq_type[e]), joint1=jname(j1) if j1 >= 0 else None, joint2=j2,
               a0=float(m.eq_data[e, 0]), poly_rest=[float(x) for x in m.eq_data[e, 1:5]],
               qpos0=float(m.qpos0[adr]) if adr >= 0 else None, active=bool(m.eq_active0[e]),
               solref=[float(x) for x in m.eq_solref[e]], solimp=[float(x) for x in m.eq_solimp[e]])
    row["lock_rad"] = row["a0"] + (row["qpos0"] or 0.0)
    eqs.append(row)
  got = {r["joint1"]: r for r in eqs}
  if len(eqs) != len(ARM_JOINTS) or set(got) != set(ARM_JOINTS):
    problems.append(f"{len(eqs)} armlock equalities on {sorted(got)}")
  for n, r in got.items():
    if n not in LOCK_POSE:
      continue
    if r["type"] != int(mujoco.mjtEq.mjEQ_JOINT) or r["joint2"] != -1 or not r["active"] or any(r["poly_rest"]):
      problems.append(f"{n}: not a single-joint constant equality ({r})")
    if abs(r["lock_rad"] - LOCK_POSE[n]) > 1e-6:
      problems.append(f"{n}: locked at {r['lock_rad']} instead of {LOCK_POSE[n]}")
    if any(abs(a - b) > 1e-9 for a, b in zip(r["solref"], LOCK_SOLREF)) or any(
        abs(a - b) > 1e-9 for a, b in zip(r["solimp"], LOCK_SOLIMP)):
      problems.append(f"{n}: solref {r['solref']} solimp {r['solimp']} differ from the hard weld")
  acts = []
  for a in range(m.nu):
    j = int(m.actuator_trnid[a, 0])
    n = jname(j)
    kp, kd = float(m.actuator_gainprm[a, 0]), -float(m.actuator_biasprm[a, 2])
    acts.append(dict(joint=n, kp=kp, kd=kd))
    if n in LOCK_POSE:
      if kp != 0.0 or kd != 0.0 or float(m.actuator_biasprm[a, 1]) != 0.0:
        problems.append(f"{n}: arm actuator gains kp {kp} kd {kd}")
    elif n in C.ENTITY_JOINTS:
      g = C.GROUPS[C._group_of(n)]
      if abs(kp - g["kp"]) > 1e-9 or abs(kd - g["kd"]) > 1e-9:
        problems.append(f"{n}: body actuator gains kp {kp} kd {kd} differ from the deploy {g['kp']} {g['kd']}")
  arm_acts = [a for a in acts if a["joint"] in LOCK_POSE]
  if len(arm_acts) != len(ARM_JOINTS):
    problems.append(f"{len(arm_acts)} arm actuators")
  h = hashlib.sha256(json.dumps([[r["name"], r["type"], r["joint1"], r["joint2"], round(r["lock_rad"], 9), r["active"],
                                  [round(x, 9) for x in r["solref"]], [round(x, 9) for x in r["solimp"]]]
                                 for r in sorted(eqs, key=lambda r: r["name"])], sort_keys=True).encode()).hexdigest()
  return dict(ok=not problems, problems=problems, n_equalities=len(eqs), equalities=eqs, lock_signature=h,
              arm_actuators=arm_acts)


def use_o6_armlock(cfg) -> None:
  """A careful_walk-family env cfg already on the O6 body (r1lab.robot.use_o6_hands): the robot entity cfg is copied with
  only spec_fn (arm welds) and the actuators (arm gains 0) replaced, and ArmLockReset is appended after the other events.
  The copied entity keeps the initial state, collisions, soft joint limits and command-delay settings of the O6 entity;
  its body actuators are checked to equal r1lab.robot.make_actuators with the same delay. Everything else unchanged."""
  import copy
  from mjlab.managers.event_manager import EventTermCfg
  from r1lab.robot import get_spec_o6, make_actuators
  lock_pose_check()
  old = cfg.scene.entities["robot"]
  if old.spec_fn is not get_spec_o6:
    raise ValueError(f"use_o6_armlock expects the O6 body (r1lab.robot.get_spec_o6), found spec_fn {old.spec_fn}")
  delays = {a.delay_max_lag for a in old.articulation.actuators}
  if len(delays) != 1:
    raise ValueError(f"mixed command delays {delays}")
  delay = delays.pop()
  if tuple(old.articulation.actuators) != make_actuators(delay):
    raise ValueError("the O6 entity's actuators are not r1lab.robot.make_actuators")
  new = copy.deepcopy(old)
  new.spec_fn = get_spec_o6_armlock
  new.articulation.actuators = make_actuators_armlock(delay)
  cfg.scene.entities["robot"] = new
  if "arm_lock_reset" in cfg.events:
    raise ValueError("arm_lock_reset already present")
  cfg.events["arm_lock_reset"] = EventTermCfg(func=ArmLockReset, mode="reset", params={})
