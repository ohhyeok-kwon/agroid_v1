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


# Option hand="o6" (coordinator 2026-11-19 00:2x, rounds/R04/positive/r8_scene/README.md 12: one body for training,
# evaluation and scenes): the Linker O6 hand body of r1wb.robot (same constants, imported lazily: r1wb imports r1lab).
HANDS = ("stock", "o6")
O6_HAND_BODIES = ("left_o6_hand", "right_o6_hand")
# Hand-load randomisation with O6 (careful_walk event 'hand_load', added to the O6 bodies): -0.15 .. +0.37 kg, i.e. an O6
# mass of 0.28-0.80 kg. Lower end as r1wb training (-0.15); upper end keeps the stock lineage's largest wrist-end load
# (stock wrist + 0.8 kg = stock wrist + O6 0.43 + 0.37).
O6_HAND_LOAD_KG = (-0.15, 0.37)


def get_spec_o6() -> mujoco.MjSpec:
  from r1wb import robot as WB
  spec = get_spec()
  for side in ("left", "right"):
    hand = spec.body(f"{side}_wrist_roll_link").add_body(name=f"{side}_o6_hand", pos=list(WB.HAND_POS))
    hand.add_geom(name=f"{side}_o6_collision", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(WB.HAND_HALF),
                  mass=WB.HAND_MASS, rgba=[0.2, 0.2, 0.6, 0.5], group=3, condim=1)
  return spec


def use_o6_hands(cfg, command_delay_max: int = 4, self_collision: bool = True) -> None:
  """A careful_walk-family env cfg on the O6 body: robot entity get_r1_cfg(hand="o6") and the 'hand_load' event moved
  from the wrist links onto the O6 bodies with O6_HAND_LOAD_KG. Everything else unchanged."""
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  cfg.scene.entities["robot"] = get_r1_cfg(command_delay_max=command_delay_max, self_collision=self_collision, hand="o6")
  ev = cfg.events.get("hand_load")
  if ev is not None:
    ev.params["asset_cfg"] = SceneEntityCfg("robot", body_names=O6_HAND_BODIES)
    ev.params["operation"] = "add"
    ev.params["ranges"] = O6_HAND_LOAD_KG


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
               self_collision: bool = True, hand: str = "stock") -> EntityCfg:
  return EntityCfg(
    init_state=EntityCfg.InitialStateCfg(
      pos=(0.0, 0.0, (init_height or C.STAND_HEIGHT) + 0.01),
      joint_pos=dict(C.DEFAULT_POS_BY_NAME),
      joint_vel={".*": 0.0},
    ),
    collisions=(FULL_COLLISION if self_collision else NO_SELF_COLLISION,),
    spec_fn={"stock": get_spec, "o6": get_spec_o6}[hand],
    articulation=EntityArticulationInfoCfg(
      actuators=make_actuators(command_delay_max),
      soft_joint_pos_limit_factor=0.95,
    ),
  )
