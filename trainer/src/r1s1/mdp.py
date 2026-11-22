"""S1 observation, reward, termination and event terms (objective.md; numbers in r1s1.numbers)."""
from __future__ import annotations

import math

import torch

from r1lab import contract as C
from r1s1 import numbers as N

FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
HAND_BODIES = ("left_wrist_roll_link", "right_wrist_roll_link")
ACTION_SCALE = tuple(C.ACTION_SCALE[C.POLICY_JOINTS.index(name)] for name in C.ENTITY_JOINTS)


def _goal(env, command_name: str = "goal"):
  return env.command_manager.get_term(command_name)


def _constants(env) -> dict:
  """Per-joint scales in the robot's joint order (checked against the contract's entity order once)."""
  cache = env.__dict__.get("_r1s1_constants")
  if cache is not None:
    return cache
  robot = env.scene["robot"]
  names = tuple(robot.joint_names)
  if names != tuple(C.ENTITY_JOINTS):
    raise ValueError(f"S1 expects the contract's entity joint order, got {names}")
  limits = robot.data.default_joint_pos_limits[0]          # (24, 2) from the model
  half = ((limits[:, 1] - limits[:, 0]) * 0.5).clamp_min(1e-3)
  device = env.device
  model = env.sim.mj_model
  import mujoco

  def body(name):
    for candidate in ("robot/" + name, name):
      index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate)
      if index >= 0:
        return index
    raise KeyError(name)

  def geom(name):
    for candidate in ("robot/" + name, name):
      index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, candidate)
      if index >= 0:
        return index
    raise KeyError(name)

  hands = torch.tensor([body(n) for n in HAND_BODIES], device=device)
  torso = torch.tensor([body("torso_link")], device=device)
  cache = dict(
    q0=robot.data.default_joint_pos[0].clone(), half=half,
    qdot_max=torch.tensor(N.QDOT_MAX_T, device=device), tau_max=torch.tensor(N.TAU_MAX, device=device),
    action_scale=torch.tensor(ACTION_SCALE, device=device),
    hands=hands, torso=torso, foot_geom=geom("left_foot1_collision"),
    hand_mass0=torch.tensor(model.body_mass[hands.cpu().numpy()], device=device, dtype=torch.float32),
    hand_ipos0=torch.tensor(model.body_ipos[hands.cpu().numpy()], device=device, dtype=torch.float32),
    torso_mass0=float(model.body_mass[int(torso[0])]),
    torso_ipos0=torch.tensor(model.body_ipos[int(torso[0])], device=device, dtype=torch.float32),
    gain0=torch.tensor(model.actuator_gainprm[:, 0], device=device, dtype=torch.float32),
    damp0=torch.tensor(-model.actuator_biasprm[:, 2], device=device, dtype=torch.float32),
    weight=float(sum(model.body_mass)) * 9.81,
  )
  env.__dict__["_r1s1_constants"] = cache
  return cache


# -- observations -------------------------------------------------------------------------------------------------
def proprio(env, noise: bool = True) -> torch.Tensor:
  """One sensor frame (54): gyro / sqrt(g/L) (3), gravity in the pelvis frame (3), (q - q0) / half range (24),
  q_dot / q_dot_max (24). noise: the sensors' uniform noise (actor only; drawn before the delay buffer)."""
  k = _constants(env)
  robot = env.scene["robot"]
  gyro = env.scene["robot/imu_ang_vel"].data
  gravity = robot.data.projected_gravity_b
  q = robot.data.joint_pos_biased if noise else robot.data.joint_pos
  qd = robot.data.joint_vel
  if noise:
    def u(x, a):
      return x + (torch.rand_like(x) * 2.0 - 1.0) * a
    gyro, gravity = u(gyro, N.NOISE["gyro"]), u(gravity, N.NOISE["gravity"])
    q, qd = u(q, N.NOISE["joint_pos"]), u(qd, N.NOISE["joint_vel"])
  return torch.cat([gyro / N.GYRO_SCALE, gravity, (q - k["q0"]) / k["half"], qd / k["qdot_max"]], dim=1)


def last_target(env) -> torch.Tensor:
  """The previous action as the joint-target offset it commanded, in half-range units (24)."""
  k = _constants(env)
  return env.action_manager.action * k["action_scale"] / k["half"]


def goal_observation(env, command_name: str = "goal") -> torch.Tensor:
  return _goal(env, command_name).command


def _lag_reads(env) -> tuple[torch.Tensor, torch.Tensor]:
  """Current actuation lag (physics steps) and observation lag (policy steps) per env, zeros if not readable."""
  zeros = torch.zeros(env.num_envs, device=env.device)
  act, obs = zeros, zeros
  group = getattr(env.scene["robot"], "_builtin_group", None)
  for fused in getattr(group, "_delayed_groups", []) or []:
    buffer = getattr(fused, "delay_buffer", None)
    if buffer is not None:
      act = buffer.current_lags.float()
      break
  manager = env.__dict__.get("observation_manager")
  buffers = getattr(manager, "_group_obs_term_delay_buffer", {}).get("actor", {}) if manager is not None else {}
  if "proprio" in buffers:
    obs = buffers["proprio"].current_lags.float()
  return act, obs


def privileged(env, command_name: str = "goal") -> torch.Tensor:
  """Critic-only physical state (34): pelvis linear velocity in the heading frame (3), pelvis height - standing
  height (1), feet vertical forces / weight (2), foot friction (1), trunk mass change (1), trunk CoM offset (3),
  hand mass changes (2), hand CoM offsets (6), kp scale and kd scale of the three actuator groups (6), actuation lag
  (1), observation lag (1), success / in goal / at rest (3), potential (1), external force on the trunk / weight (3)."""
  k = _constants(env)
  robot = env.scene["robot"]
  term = _goal(env, command_name)
  pos, yaw, lin, _ = term.state()
  c, s = torch.cos(yaw), torch.sin(yaw)
  vx, vy = c * lin[:, 0] + s * lin[:, 1], -s * lin[:, 0] + c * lin[:, 1]
  feet = env.scene["s1_feet"].data.force.reshape(env.num_envs, -1, 3)[:, :, 2].abs() / k["weight"]
  model = env.sim.model

  def rows(field):
    return field if field.shape[0] == env.num_envs else field.expand(env.num_envs, *field.shape[1:])

  mu = rows(model.geom_friction)[:, k["foot_geom"], 0:1]
  torso_dm = rows(model.body_mass)[:, k["torso"]] - k["torso_mass0"]
  torso_com = (rows(model.body_ipos)[:, k["torso"][0]] - k["torso_ipos0"]) / 0.05
  hand_dm = rows(model.body_mass)[:, k["hands"]] - k["hand_mass0"]
  hand_com = ((rows(model.body_ipos)[:, k["hands"]] - k["hand_ipos0"]) / N.HAND_COM_M).reshape(env.num_envs, 6)
  ids = robot.indexing.ctrl_ids
  gain = rows(model.actuator_gainprm)[:, ids, 0] / k["gain0"][ids]
  damp = -rows(model.actuator_biasprm)[:, ids, 2] / k["damp0"][ids].clamp_min(1e-6)
  pick = torch.tensor([0, 4, 16], device=env.device)          # a hip (big), an ankle, a shoulder yaw (small) actuator
  act_lag, obs_lag = _lag_reads(env)
  success, in_goal, at_rest = term.judge()
  push = getattr(env, "_r1s1_push_force", torch.zeros(env.num_envs, 3, device=env.device)) / k["weight"]
  return torch.cat([
    torch.stack([vx, vy, lin[:, 2]], dim=1), (pos[:, 2:3] - N.STAND_H) / 0.15, feet, mu,
    (torso_dm / 2.0).reshape(env.num_envs, 1), torso_com, hand_dm / 3.0, hand_com,
    gain[:, pick], damp[:, pick], (act_lag / max(N.ACT_LAG_STEPS, 1)).unsqueeze(1), obs_lag.unsqueeze(1),
    torch.stack([success.float(), in_goal.float(), at_rest.float()], dim=1), term.potential().unsqueeze(1), push,
  ], dim=1)


# -- rewards ------------------------------------------------------------------------------------------------------
def s1_success(env, command_name: str = "goal") -> torch.Tensor:
  return _goal(env, command_name).judge()[0].float()


def s1_progress(env, command_name: str = "goal", gamma: float = N.GAMMA) -> torch.Tensor:
  """(gamma * Phi(s', g) - Phi_prev) / dt: the potential difference (the reward manager multiplies by dt). Phi of a
  state ended by a physical failure is 0; a time-out keeps its true value. Phi_prev is the value this term computed
  one step earlier (the goal of that step), so the sum telescopes exactly across goal switches."""
  term = _goal(env, command_name)
  phi = term.potential()
  failed = env.termination_manager.terminated
  phi = torch.where(failed, torch.zeros_like(phi), phi)
  value = (gamma * phi - term.phi_prev) / env.step_dt
  term.phi_prev = phi.clone()
  return value


def _joint_torque(env) -> torch.Tensor:
  return env.scene["robot"].data.qfrc_actuator


def s1_effort_joule(env) -> torch.Tensor:
  """Sum over the 24 joints of (tau / tau_max)^2, tau = the PD torque the actuators applied (0 .. 24)."""
  k = _constants(env)
  return ((_joint_torque(env) / k["tau_max"]) ** 2).sum(dim=1)


def s1_effort_power(env) -> torch.Tensor:
  """Sum over the 24 joints of |tau * q_dot| in W."""
  return (_joint_torque(env) * env.scene["robot"].data.joint_vel).abs().sum(dim=1)


# -- terminations -------------------------------------------------------------------------------------------------
def s1_body_ground(env, sensor_name: str = "s1_body_ground", force_threshold: float = N.GROUND_FORCE_N) -> torch.Tensor:
  """Any robot body other than the two feet touches the floor."""
  force = env.scene[sensor_name].data.force
  return torch.norm(force.reshape(env.num_envs, -1, 3), dim=-1).amax(dim=1) > force_threshold


def s1_self_collision(env, sensor_name: str = "s1_self", force_threshold: float = N.SELF_FORCE_N) -> torch.Tensor:
  """A robot-robot contact above the threshold in any physics substep of the step."""
  data = env.scene[sensor_name].data
  if data.force_history is not None:
    return (torch.norm(data.force_history, dim=-1) > force_threshold).flatten(1).any(dim=1)
  return torch.norm(data.force.reshape(env.num_envs, -1, 3), dim=-1).amax(dim=1) > force_threshold


# -- events -------------------------------------------------------------------------------------------------------
class s1_impulse:
  """Force impulses on the trunk: one policy step (20 ms) long, magnitude U(IMPULSE_NS), horizontal direction
  uniform, acting at a point drawn up/down the trunk and across the line of force, every U(IMPULSE_EVERY_S) seconds,
  in every environment (no calm share, no height rule). mode="step"."""

  def __init__(self, cfg, env):
    self.robot = env.scene["robot"]
    self.body = [self.robot.body_names.index("torso_link")]
    n, device = env.num_envs, env.device
    low, high = N.IMPULSE_EVERY_S
    self.wait = low + (high - low) * torch.rand(n, device=device)
    self.active = torch.zeros(n, dtype=torch.bool, device=device)
    self.force = torch.zeros(n, 3, device=device)
    self.torque = torch.zeros(n, 3, device=device)
    self.written = False
    env._r1s1_push_force = torch.zeros(n, 3, device=device)

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      return
    low, high = N.IMPULSE_EVERY_S
    self.wait[env_ids] = low + (high - low) * torch.rand(len(env_ids), device=self.wait.device)
    self.active[env_ids] = False
    self.force[env_ids] = 0.0
    self.torque[env_ids] = 0.0

  def __call__(self, env, env_ids) -> None:
    del env_ids
    device, dt = env.device, env.step_dt
    self.wait -= dt
    start = (self.wait <= 0.0) & ~self.active
    self.active[:] = False
    if start.any():
      ids = start.nonzero().squeeze(-1)
      count = len(ids)
      low, high = N.IMPULSE_NS
      impulse = low + (high - low) * torch.rand(count, device=device)
      angle = (torch.rand(count, device=device) * 2.0 - 1.0) * math.pi
      force = torch.stack([angle.cos(), angle.sin(), torch.zeros_like(angle)], dim=1) * (impulse / dt).unsqueeze(1)
      arm = torch.zeros(count, 3, device=device)
      arm[:, 2] = N.IMPULSE_HEIGHT_M[0] + (N.IMPULSE_HEIGHT_M[1] - N.IMPULSE_HEIGHT_M[0]) * torch.rand(count, device=device)
      side = N.IMPULSE_SIDE_M[0] + (N.IMPULSE_SIDE_M[1] - N.IMPULSE_SIDE_M[0]) * torch.rand(count, device=device)
      arm[:, 0], arm[:, 1] = -angle.sin() * side, angle.cos() * side
      self.force[ids], self.torque[ids] = force, torch.cross(arm, force, dim=1)
      self.active[ids] = True
      low, high = N.IMPULSE_EVERY_S
      self.wait[ids] = low + (high - low) * torch.rand(count, device=device)
    if not (self.active.any() or self.written):
      env._r1s1_push_force.zero_()
      return
    on = self.active.float().unsqueeze(1)
    self.robot.write_external_wrench_to_sim((self.force * on).unsqueeze(1), (self.torque * on).unsqueeze(1),
                                            body_ids=self.body)
    env._r1s1_push_force = self.force * on
    self.written = bool(self.active.any())


# -- R12 measurement / v3 (positive 22, 2026-11-20): exact whole-body kinetic energy and the graded success ---------
def _quat_to_mat(q: torch.Tensor) -> torch.Tensor:
  w, x, y, z = q.unbind(-1)
  return torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                      2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                      2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def kinetic_energy(env) -> torch.Tensor:
  """KE = 1/2 qdot^T M(q) qdot of the robot (MuJoCo mj_energyVel convention): per body 1/2 m |v_com|^2 +
  1/2 w^T R I R^T w, plus the rotor (armature) energy 1/2 sum_d armature_d qdot_d^2 over ALL robot dofs including the
  floating base's six (negative 18 R-KE: those were missing in the first version)."""
  robot = env.scene["robot"]
  model = env.sim.model

  def rows(field):
    return field if field.shape[0] == env.num_envs else field.expand(env.num_envs, *field.shape[1:])

  ids = robot.indexing.body_ids.long()
  mass, inertia = rows(model.body_mass)[:, ids], rows(model.body_inertia)[:, ids]
  v, w = robot.data.body_com_lin_vel_w, robot.data.body_com_ang_vel_w
  wl = (_quat_to_mat(robot.data.body_com_quat_w).transpose(-1, -2) @ w.unsqueeze(-1)).squeeze(-1)
  bodies = (0.5 * mass * (v ** 2).sum(-1) + 0.5 * (inertia * wl ** 2).sum(-1)).sum(1)
  dofs = torch.cat([robot.indexing.free_joint_v_adr.reshape(-1), robot.indexing.joint_v_adr.reshape(-1)]).long()
  qvel = env.sim.data.qvel[:, dofs]
  armature = rows(model.dof_armature)[:, dofs]
  return bodies + 0.5 * (armature * qvel ** 2).sum(1)


def s1_success_v3(env, command_name: str = "goal", k0: float = 4.5, floor: float = 0.5) -> torch.Tensor:
  """objective.md v3: 1 before the deadline; after it Phi * (floor + (1 - floor) * k0 / (k0 + KE))."""
  term = _goal(env, command_name)
  rho = k0 / (k0 + kinetic_energy(env))
  post = term.potential() * (floor + (1.0 - floor) * rho)
  return torch.where(term.tau > 0.0, torch.ones_like(post), post)


# --- R12 S0b-fixed records (coordinator 07:5x): per-step indicators for mjlab metrics terms (reduce "mean"), so the
# logger reports per finished episode the share of steps after the deadline, and after the deadline in the goal / at
# rest (share of post-deadline steps = post_in_goal / post_deadline). Records only: no reward, no termination.
def s1_post_deadline(env, command_name: str = "goal") -> torch.Tensor:
  return (_goal(env, command_name).tau <= 0.0).float()


def s1_post_in_goal(env, command_name: str = "goal") -> torch.Tensor:
  term = _goal(env, command_name)
  return ((term.tau <= 0.0) & term.judge()[1]).float()


def s1_post_at_rest(env, command_name: str = "goal") -> torch.Tensor:
  term = _goal(env, command_name)
  return ((term.tau <= 0.0) & term.judge()[2]).float()


# per goal kind (coordinator 08:0x: the rest share for hold goals and for locomotion goals apart; negative 18 freeze 79)
def _post_kind(env, kind: int, command_name: str = "goal"):
  term = _goal(env, command_name)
  return term, (term.tau <= 0.0) & (term.kind == kind)


def s1_post_deadline_hold(env, command_name: str = "goal") -> torch.Tensor:
  return _post_kind(env, 0, command_name)[1].float()


def s1_post_at_rest_hold(env, command_name: str = "goal") -> torch.Tensor:
  term, post = _post_kind(env, 0, command_name)
  return (post & term.judge()[2]).float()


def s1_post_in_goal_hold(env, command_name: str = "goal") -> torch.Tensor:
  term, post = _post_kind(env, 0, command_name)
  return (post & term.judge()[1]).float()


def s1_post_deadline_loco(env, command_name: str = "goal") -> torch.Tensor:
  return _post_kind(env, 1, command_name)[1].float()


def s1_post_at_rest_loco(env, command_name: str = "goal") -> torch.Tensor:
  term, post = _post_kind(env, 1, command_name)
  return (post & term.judge()[2]).float()


def s1_post_in_goal_loco(env, command_name: str = "goal") -> torch.Tensor:
  term, post = _post_kind(env, 1, command_name)
  return (post & term.judge()[1]).float()
