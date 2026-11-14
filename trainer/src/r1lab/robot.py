"""Unitree R1 (agroid) entity for mjlab with the agroid deploy actuator contract."""
from __future__ import annotations

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from r1lab import contract as C

R1_XML = Path(__file__).parent / "assets" / "r1" / "r1.xml"

FOOT_GEOMS = tuple(f"{s}_foot{i}_collision" for s in ("left", "right") for i in range(1, 8))
FOOT_SITES = ("left_foot", "right_foot")
FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(R1_XML))


def make_actuators(command_delay_max: int = 0) -> tuple[BuiltinPositionActuatorCfg, ...]:
  """Deploy PD gains; command delay in physics steps (5 ms each)."""
  acts = []
  for spec in C.GROUPS.values():
    acts.append(
      BuiltinPositionActuatorCfg(
        target_names_expr=spec["joints"],
        stiffness=spec["kp"],
        damping=spec["kd"],
        effort_limit=spec["effort"],
        armature=spec["armature"],
        frictionloss=0.1,
        viscous_damping=0.05,
        delay_min_lag=0,
        delay_max_lag=command_delay_max,
        delay_hold_prob=0.9 if command_delay_max > 0 else 0.0,
      )
    )
  return tuple(acts)


FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=1,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.8,)},
)


# Robot geoms collide with the terrain only (contype 0): much cheaper, used for walking.
NO_SELF_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1, ".*": 0},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.8,)},
)


def get_r1_cfg(command_delay_max: int = 0, init_height: float | None = None,
               self_collision: bool = True) -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, (init_height or C.STAND_HEIGHT) + 0.01),
      joint_pos=dict(C.DEFAULT_POS_BY_NAME),
      joint_vel={".*": 0.0},
    ),
    collisions=(FULL_COLLISION if self_collision else NO_SELF_COLLISION,),
    spec_fn=get_spec,
    articulation=EntityArticulationInfoCfg(
      actuators=make_actuators(command_delay_max),
      soft_joint_pos_limit_factor=0.95,
    ),
  )
