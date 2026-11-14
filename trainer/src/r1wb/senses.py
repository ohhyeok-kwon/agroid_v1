"""What System 1 can be given beyond the plan and its own state at this moment: the last moments of its own body
and the ground around it.

  history   the last 10 control steps (0.2 s) of: joint angles, joint velocities, the action applied, the trunk's
            angular velocity, gravity in the trunk frame (78 values a step, oldest first)
  scan      the height of the ground on a 13 x 9 grid of 10 cm around the pelvis (0.6 m ahead and behind, 0.4 m to
            each side, in the robot's heading frame), above the surface the robot stands on (r1wb.world.support_under)
  both      history, then scan

They are appended to the actor's observation (teacher_task senses=...), so a policy that did not have them continues
as exactly the same function (r1wb.train widened). One object per environment holds them (Senses.of): the training
observation and every evaluation path (tracking.env_cfg.teacher_observation) read the same instance, and the
history advances once per control step whoever asks. The scan is a simulator quantity (the teacher is privileged);
what a robot's camera gives instead is the student's question.
"""
from __future__ import annotations

import torch

HISTORY = 10                                  # control steps
FRAME = 78                                    # joints 24 + joint velocities 24 + action 24 + angular velocity 3 + gravity 3
VELOCITY_SCALE, TURN_SCALE = 0.1, 0.25        # joint velocities (rad/s) and angular velocity (rad/s) as they enter
SCAN_SIZE, SCAN_STEP = (1.2, 0.8), 0.1        # m: along x (forward), along y (left); spacing
SCAN_GRID = (13, 9)                           # samples along x and along y; x runs fastest (mjlab GridPatternCfg)
SCAN = SCAN_GRID[0] * SCAN_GRID[1]
SCAN_CLIP, SCAN_SCALE = 0.5, 4.0              # m kept either way; factor as it enters (0.25 m reads as 1)
WIDTH = {"": 0, "history": HISTORY * FRAME, "scan": SCAN, "both": HISTORY * FRAME + SCAN}
BASE = 294                                    # the task teacher's observation they are appended to


def kind_of(width: int) -> str:
  """Which senses an actor observation of `width` values carries ("" for the plain 273 / 294)."""
  if width in (273, BASE):
    return ""
  for kind, extra in WIDTH.items():
    if extra and width == BASE + extra:
      return kind
  raise ValueError(f"no actor observation has {width} values (known: 273, {', '.join(str(BASE + w) for w in WIDTH.values())})")


def kind_of_checkpoint(checkpoint: str) -> str:
  state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
  first = min((key for key in state if key.startswith("mlp.") and key.endswith(".weight")), key=lambda k: int(k.split(".")[1]))
  return kind_of(int(state[first].shape[1]))


def add_sensor(cfg, kind: str) -> None:
  """The scene gets the rays of the scan when `kind` needs them (once)."""
  if kind not in ("scan", "both") or any(sensor.name == "ground_scan" for sensor in cfg.scene.sensors):
    return
  from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (RayCastSensorCfg(
    name="ground_scan", frame=ObjRef(type="body", name="pelvis", entity="robot"),
    pattern=GridPatternCfg(size=SCAN_SIZE, resolution=SCAN_STEP), ray_alignment="yaw", max_distance=3.0,
    exclude_parent_body=True, include_geom_groups=(0,), debug_vis=False),)


class Senses:
  def __init__(self, env):
    self.env = env
    self.frames = torch.zeros(env.num_envs, HISTORY, FRAME, device=env.device)
    self.scan = torch.zeros(env.num_envs, SCAN, device=env.device)
    self.stamp = None
    self.has_scan = "ground_scan" in env.scene.sensors

  @staticmethod
  def of(env) -> "Senses":
    real = getattr(env, "_env", env)                  # eval.chunk_view._EnvView stands in for the environment
    senses = real.__dict__.get("_r1wb_senses")
    if senses is None:
      senses = real.__dict__["_r1wb_senses"] = Senses(real)
    return senses

  def _frame(self) -> torch.Tensor:
    env = self.env
    data = env.scene["robot"].data
    term = env.action_manager.get_term("joint_pos")
    action = term.plain_action if hasattr(term, "plain_action") else env.action_manager.action
    return torch.cat([data.joint_pos_biased - data.default_joint_pos,
                      VELOCITY_SCALE * (data.joint_vel - data.default_joint_vel), action,
                      TURN_SCALE * env.scene["robot/imu_ang_vel"].data, data.projected_gravity_b], dim=1)

  def _scan(self, started: torch.Tensor) -> torch.Tensor:
    if not self.has_scan:
      return self.scan
    from r1wb import world
    env = self.env
    data = env.scene["ground_scan"].data
    level = env.scene.env_origins[:, 2] + world.support_under(env)           # the surface the robot stands on
    scan = (data.hit_pos_w[..., 2] - level.unsqueeze(1)).clamp(-SCAN_CLIP, SCAN_CLIP) * SCAN_SCALE
    scan = torch.where(data.distances < 0.0, torch.zeros_like(scan), scan)    # nothing below: reads as level ground
    # on the step an episode starts the rays are still those of the episode before; the robot starts on a flat spot
    return torch.where(started.unsqueeze(1), torch.zeros_like(scan), scan)

  def read(self) -> tuple[torch.Tensor, torch.Tensor]:
    """(history (n, HISTORY * FRAME) oldest first, scan (n, SCAN)) of this control step."""
    env = self.env
    stamp = int(env.common_step_counter)
    started = env.episode_length_buf < 1
    first = self.stamp is None
    if first or stamp != self.stamp or bool(started.any()):
      frame = self._frame()
      if first or stamp != self.stamp:
        self.frames = torch.roll(self.frames, -1, dims=1)
        self.frames[:, -1] = frame
        self.stamp = stamp
      fill = torch.ones_like(started) if first else started          # an episode's first step: every slot is this frame
      self.frames[fill] = frame[fill].unsqueeze(1)
      self.scan = self._scan(started)
    return self.frames.flatten(1), self.scan


def read(env, kind: str = "both") -> torch.Tensor:
  """Observation term (and the evaluation paths' reader): the senses of `kind` for this control step."""
  history, scan = Senses.of(env).read()
  if kind == "history":
    return history
  if kind == "scan":
    return scan
  if kind == "both":
    return torch.cat([history, scan], dim=1)
  raise ValueError(f"unknown senses: {kind!r}")
