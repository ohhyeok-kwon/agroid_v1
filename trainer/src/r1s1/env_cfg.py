"""S1 environment: flat floor, stock-fist R1 (24 actions), one masked outcome goal, physical failures only."""
from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as base
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.viewer import ViewerConfig

from r1lab import robot as stock
from r1s1 import mdp
from r1s1 import numbers as N
from r1s1.goal import GoalCommandCfg

EFFORTS = ("joule", "power")
EXPERIMENT = "r1_wholebody"            # the whole-body runs folder (r1wb runner checks it against the registry)
FOOT_GEOMS = stock.FOOT_GEOMS


def sensors() -> tuple:
  feet = ContactSensorCfg(
    name="s1_feet", primary=ContactMatch(mode="subtree", pattern=r"^(left_ankle_roll_link|right_ankle_roll_link)$",
                                         entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="netforce", num_slots=1)
  body = ContactSensorCfg(
    name="s1_body_ground", primary=ContactMatch(mode="body", pattern=r".*", entity="robot", exclude=mdp.FOOT_BODIES),
    secondary=ContactMatch(mode="body", pattern="terrain"), fields=("found", "force"), reduce="maxforce", num_slots=1)
  own = ContactSensorCfg(
    name="s1_self", primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found", "force"), reduce="maxforce", num_slots=1, history_length=4)
  return (feet, body, own)


def s1_env_cfg(effort: str = "joule", play: bool = False, fixed_kind: str | None = None,
               clean: bool | None = None, success: str = "v2", post_metrics: bool = False) -> ManagerBasedRlEnvCfg:
  """effort: which single effort price is on ("joule" | "power"). play: evaluation layout (no pushes, no
  randomised dynamics unless clean=False). fixed_kind: evaluation with one goal kind only."""
  if effort not in EFFORTS:
    raise ValueError(f"effort must be one of {EFFORTS}")
  clean = play if clean is None else clean
  robot = stock.get_r1_cfg(command_delay_max=0 if clean else N.ACT_LAG_STEPS, hand="stock")
  goal = {"command_name": "goal"}
  actor = ObservationGroupCfg(
    terms={
      "proprio": ObservationTermCfg(func=mdp.proprio, params={"noise": not clean}, history_length=N.HISTORY,
                                    flatten_history_dim=True,
                                    delay_min_lag=0, delay_max_lag=0 if clean else N.OBS_LAG_STEPS,
                                    delay_hold_prob=N.OBS_HOLD),
      "last_target": ObservationTermCfg(func=mdp.last_target, history_length=N.HISTORY, flatten_history_dim=True),
      "goal": ObservationTermCfg(func=mdp.goal_observation, params=goal),
    },
    concatenate_terms=True, enable_corruption=False)
  critic = ObservationGroupCfg(
    terms={
      "proprio": ObservationTermCfg(func=mdp.proprio, params={"noise": False}),
      "last_target": ObservationTermCfg(func=mdp.last_target),
      "goal": ObservationTermCfg(func=mdp.goal_observation, params=goal),
      "privileged": ObservationTermCfg(func=mdp.privileged, params=goal),
    },
    concatenate_terms=True, enable_corruption=False)
  coefficient = N.C_JOULE if effort == "joule" else N.C_POWER
  rewards = {
    "s1_success": RewardTermCfg(func=mdp.s1_success if success == "v2" else mdp.s1_success_v3, weight=N.W_SUCCESS,
                                params=goal),
    "s1_progress": RewardTermCfg(func=mdp.s1_progress, weight=N.W_PROGRESS, params={**goal, "gamma": N.GAMMA}),
    f"s1_effort_{effort}": RewardTermCfg(func=getattr(mdp, f"s1_effort_{effort}"), weight=-coefficient),
  }
  terminations = {
    "time_out": TerminationTermCfg(func=base.time_out, time_out=True),
    "s1_body_ground": TerminationTermCfg(func=mdp.s1_body_ground),
    "s1_self_collision": TerminationTermCfg(func=mdp.s1_self_collision),
  }
  events = {
    "reset_base": EventTermCfg(func=base.reset_root_state_uniform, mode="reset", params={
      "pose_range": {"yaw": (-3.14159, 3.14159)},
      "velocity_range": {"x": (-N.RESET_VEL, N.RESET_VEL), "y": (-N.RESET_VEL, N.RESET_VEL)}}),
    "reset_joints": EventTermCfg(func=base.reset_joints_by_offset, mode="reset", params={
      "position_range": (-N.RESET_JOINT_RAD, N.RESET_JOINT_RAD), "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))}),
  }
  if not clean:
    events.update({
      "foot_friction": EventTermCfg(mode="startup", func=dr.geom_friction, params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS), "operation": "abs", "ranges": N.FOOT_MU,
        "shared_random": True}),
      "torso_com": EventTermCfg(mode="startup", func=dr.body_com_offset, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add",
        "ranges": dict(N.TORSO_COM_M)}),
      "torso_mass": EventTermCfg(mode="startup", func=dr.body_mass, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)), "operation": "add",
        "ranges": N.TORSO_MASS_ADD_KG}),
      "hand_load": EventTermCfg(mode="startup", func=dr.body_mass, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=mdp.HAND_BODIES), "operation": "add",
        "ranges": N.HAND_LOAD_KG}),
      "hand_load_com": EventTermCfg(mode="startup", func=dr.body_com_offset, params={
        "asset_cfg": SceneEntityCfg("robot", body_names=mdp.HAND_BODIES), "operation": "add",
        "ranges": {0: (-N.HAND_COM_M, N.HAND_COM_M), 1: (-N.HAND_COM_M, N.HAND_COM_M),
                   2: (-N.HAND_COM_M, N.HAND_COM_M)}}),
      "pd_gains": EventTermCfg(mode="startup", func=dr.pd_gains, params={
        "asset_cfg": SceneEntityCfg("robot"), "kp_range": N.PD_SCALE, "kd_range": N.PD_SCALE, "operation": "scale"}),
      "encoder_bias": EventTermCfg(mode="startup", func=dr.encoder_bias, params={
        "asset_cfg": SceneEntityCfg("robot"), "bias_range": (-N.ENCODER_BIAS_RAD, N.ENCODER_BIAS_RAD)}),
    })
  if not play:
    events["s1_impulse"] = EventTermCfg(func=mdp.s1_impulse, mode="step")
  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1, entities={"robot": robot},
                   sensors=sensors()),
    observations={"actor": actor, "critic": critic},
    actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",),
                                                 scale={n: s for n, s in zip(mdp.C.ENTITY_JOINTS, mdp.ACTION_SCALE)},
                                                 use_default_offset=True)},
    commands={"goal": GoalCommandCfg(fixed_kind=fixed_kind)},
    events=events, rewards=rewards, terminations=terminations,
    viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="pelvis",
                        distance=2.8, fovy=55.0, elevation=-5.0, azimuth=120.0),
    sim=SimulationCfg(nconmax=64, njmax=400, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
    decimation=4, episode_length_s=N.EPISODE_S if not play else 1.0e9,
  )
  if post_metrics:            # R12 S0b-fixed records (coordinator 07:5x); off by default, so S0's config is unchanged
    from mjlab.managers.metrics_manager import MetricsTermCfg
    cfg.metrics = {name: MetricsTermCfg(func=getattr(mdp, name), params=goal, reduce="mean")
                   for name in ("s1_post_deadline", "s1_post_in_goal", "s1_post_at_rest",
                                "s1_post_deadline_hold", "s1_post_in_goal_hold", "s1_post_at_rest_hold",
                                "s1_post_deadline_loco", "s1_post_in_goal_loco", "s1_post_at_rest_loco")}
  return cfg
