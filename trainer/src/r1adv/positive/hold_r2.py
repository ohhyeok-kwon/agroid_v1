"""R2 Hold mission of the positive agent (coordinator lock 2026-11-17 10:12 KST, .agent/adversarial/rounds/R02/
coordinator_lock_r2.md). The R1 Hold task (hold_task; objective lock c672a53b unchanged: r = L + beta D with beta = 1,
target v0d, physical falls only, sensors = the nominal deploy model x 1, command delay 0-40 ms, R1 body randomisation)
with the R2 training distribution of the canonical file r2_train_distribution_v1.json (owner: the negative agent; used
unchanged, its sha256 is pinned in START_FRAMES / R2_DIST below once it exists):
  starts   50 % the R1 reset (deploy default pose, joints +-0.1 rad, base velocity +-0.1) / 50 % human standing frames
           (train side of shared/motion_train only, corebal included): joints, pelvis tilt and height from the frame,
           yaw uniform, zero velocity (event `human_start`, applied after the R1 reset terms)
  hits     one torso impulse at a time, held for one policy step: force = impulse / 0.02 s, horizontal, at the torso
           CoM + (0, 0, h) (torque h x F), azimuth relative to the robot heading -- the verdict runner's application
           (r1adv.negative.runner); impulse U(4, 25) N s, azimuth U(0, 2 pi), h U(-0.1, 0.3) m, first hit U(1.5, 5) s
           after the start, then every U(2, 6) s (event `torso_hits`, mode "step")
Teachers of wave 1: C0 = the R1 recipe (actor reads the noisy 570 frames, white exploration); T = the same with the
actor reading the noise-free 570 frames (same contract; a privileged-observation teacher, never deployed).
CODE_SHA256 is the hash of this package's sources taken at import; it is written into the event parameters, so every
run's env.yaml carries the hash of the code that was imported (evaluation protocol revision 1, section 6)."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from r1adv.positive import hold_task as HT

PACKAGE = Path(__file__).resolve().parent


def _code_sha256() -> str:
  h = hashlib.sha256()
  for f in sorted(PACKAGE.glob("*.py")):
    h.update(f.name.encode()); h.update(b"\0"); h.update(f.read_bytes()); h.update(b"\0")
  return h.hexdigest()


CODE_SHA256 = _code_sha256()
EXPERIMENT = HT.EXPERIMENT                  # the registry ties the run folder to the action (r1adv_hold)
# The R2 training distribution (coordinator lock, item 1). START_FRAMES names the frames file derived from the canonical
# distribution file (r2_frames.py); both sha256 values are checked when the environment is built.
R2_DIST = dict(file="r2_train_distribution_v1.json", sha256="9dff1729a8b0a69f0edd77b6b7d1033f5d9499fe8396cdb17ad701a029ba10b8")
# Frames of start.human_frame with the pelvis placed so that the lowest sole point is 2 mm above the floor (the verdict
# runner's spawn rule), precomputed by forward kinematics (.agent/adversarial/positive/tools/r2_frames.py).
START_FRAMES = dict(file="r2_start_frames_v1.npz", sha256="c2667c8b362e24174a0080a4ab65b6d40be50375caa16b00010d653f21343cb1")


def distribution() -> dict:
  """The canonical R2 training distribution (pinned sha256), and the start/hit constants read from it unchanged."""
  d = json.loads(pinned(R2_DIST).read_text())
  h = d["hits"]
  hits = dict(impulse_ns=tuple(h["magnitude_ns"]["uniform"]), height_m=tuple(h["height_m"]["uniform"]),
              first_s=tuple(h["first_s"]["uniform"]), gap_s=tuple(h["interval_s"]["uniform"]),
              azimuth_rad=tuple(h["azimuth_rad"]["uniform"]), duration_s=float(h["application"]["duration_s"]))
  starts = dict(human_fraction=float(d["start"]["mix"]["human_frame"]), clearance_m=0.0)
  return dict(raw=d, hits=hits, starts=starts)


def pinned(entry: dict) -> Path:
  """The first copy of entry["file"] in the Hold input directories (server inputs, then the PC models folder); its
  sha256 must equal entry["sha256"] when that is set."""
  for d in HT.INPUT_DIRS:
    path = d / entry["file"]
    if path.is_file():
      if entry.get("sha256"):
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != entry["sha256"]:
          raise ValueError(f"{path} sha256 {got} != pinned {entry['sha256']}")
      return path
  raise FileNotFoundError(f"{entry['file']} not found in {HT.INPUT_DIRS}")


class human_start:
  """Reset event: a share of the resetting environments starts from a human standing frame (joints, pelvis tilt and
  height of the frame; yaw uniform; zero velocity). Runs after the R1 reset terms, which it overrides for those envs."""

  def __init__(self, cfg, env):
    p = cfg.params
    data = np.load(p["frames_file"])
    dev = env.device
    self.q = torch.tensor(data["joint_pos"], dtype=torch.float, device=dev)          # (F, 24), entity order
    self.tilt = torch.tensor(data["tilt_quat"], dtype=torch.float, device=dev)       # (F, 4) wxyz, yaw removed
    self.z = torch.tensor(data["pelvis_z"], dtype=torch.float, device=dev)           # (F,) pelvis height on flat ground
    self.asset = env.scene["robot"]
    self.g = torch.Generator(device=dev)
    self.g.manual_seed(int(env.cfg.seed) * 7919 + 101)

  def reset(self, env_ids=None):
    pass

  def __call__(self, env, env_ids, frames_file: str, fraction: float, clearance_m: float, sha256: str = ""):
    if env_ids is None or len(env_ids) == 0:
      return
    dev = env.device
    pick = torch.rand(len(env_ids), device=dev, generator=self.g) < fraction
    ids = env_ids[pick]
    if len(ids) == 0:
      return
    k = torch.randint(0, len(self.q), (len(ids),), device=dev, generator=self.g)
    yaw = (torch.rand(len(ids), device=dev, generator=self.g) * 2.0 - 1.0) * math.pi
    turn = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], 1)
    quat = _quat_mul(turn, self.tilt[k])
    state = torch.zeros(len(ids), 13, device=dev)
    state[:, :3] = env.scene.env_origins[ids]
    state[:, 2] += self.z[k] + clearance_m
    state[:, 3:7] = quat / quat.norm(dim=1, keepdim=True)
    self.asset.write_root_state_to_sim(state, env_ids=ids)
    self.asset.write_joint_state_to_sim(self.q[k], torch.zeros_like(self.q[k]), env_ids=ids)


class torso_hits:
  """Step event: one horizontal torso impulse at a time, held for one policy step (see the module docstring)."""

  def __init__(self, cfg, env):
    p = cfg.params
    self.asset = env.scene["robot"]
    self.torso = self.asset.body_names.index("torso_link")
    n, dev = env.num_envs, env.device
    self.dt = env.step_dt
    if abs(self.dt - 0.02) > 1e-9:
      raise ValueError(f"hits are held for one policy step of 0.02 s; env.step_dt is {self.dt}")
    self.next = torch.zeros(n, device=dev)
    self.active = torch.zeros(n, dtype=torch.bool, device=dev)
    self.count = torch.zeros(n, dtype=torch.long, device=dev)
    self.last = torch.zeros(n, 4, device=dev)                  # last fired hit per env: time, impulse, azimuth, height
    self.g = torch.Generator(device=dev)
    self.g.manual_seed(int(env.cfg.seed) * 7919 + 202)
    self.first, self.gap = tuple(p["first_s"]), tuple(p["gap_s"])
    self.reset(None)

  def _u(self, lo, hi, n):
    return lo + (hi - lo) * torch.rand(n, device=self.next.device, generator=self.g)

  def reset(self, env_ids=None):
    ids = torch.arange(len(self.next), device=self.next.device) if env_ids is None else env_ids
    if len(ids) == 0:
      return
    self.next[ids] = self._u(*self.first, len(ids))
    self.count[ids] = 0
    was = ids[self.active[ids]]
    if len(was):
      z = torch.zeros(len(was), 1, 3, device=self.next.device)
      self.asset.write_external_wrench_to_sim(z, z, env_ids=was, body_ids=[self.torso])
    self.active[ids] = False

  def __call__(self, env, env_ids, impulse_ns, height_m, first_s, gap_s, code_sha256: str = "", distribution_sha256: str = ""):
    t = env.episode_length_buf.float() * self.dt
    fire = t >= self.next
    clear = self.active & ~fire
    if not (fire.any() or clear.any()):
      return
    dev = self.next.device
    upd = (fire | clear).nonzero().flatten()
    forces = torch.zeros(len(upd), 1, 3, device=dev)
    torques = torch.zeros(len(upd), 1, 3, device=dev)
    sel = fire[upd]
    ids = upd[sel]
    if len(ids):
      imp = self._u(*impulse_ns, len(ids))
      azimuth = self._u(0.0, 2.0 * math.pi, len(ids))
      h = self._u(*height_m, len(ids))
      q = self.asset.data.root_link_quat_w[ids]
      heading = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
      ang = heading + azimuth
      f = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], 1) * (imp / self.dt)[:, None]
      arm = torch.zeros_like(f)
      arm[:, 2] = h
      forces[sel, 0] = f
      torques[sel, 0] = torch.cross(arm, f, dim=1)
      self.next[ids] = t[ids] + self._u(*self.gap, len(ids))
      self.count[ids] += 1
      self.last[ids] = torch.stack([t[ids], imp, azimuth, h], 1)
    self.asset.write_external_wrench_to_sim(forces, torques, env_ids=upd, body_ids=[self.torso])
    self.active = fire.clone()


class frame_stack_novel(HT.frame_stack_noprev):
  """R2 arm V: the 570 frame stack without the joint-velocity block (33 values a frame, 330 in all). The sensor model is
  unchanged (the joint velocities are still measured with their noise, then left out of the frame)."""

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.hist = torch.zeros(self.n, HT.LC.HISTORY, 33, device=self.dev)

  def __call__(self, env, command_name: str, noise: bool = True, noise_params: dict | None = None) -> torch.Tensor:
    del noise, noise_params
    step = env.common_step_counter
    ids = self.env_ids if step != self.last_step else self.needs_fill.nonzero(as_tuple=False).flatten()
    if ids.numel():
      measured = self._measure(env, ids)
      prev = torch.where(self.needs_fill[ids, None], measured, self.prev_sensor[ids])
      self.prev_sensor[ids] = measured
      self.sensor[ids] = ((1.0 - self.lag[ids]) * measured + self.lag[ids] * prev if self.noise else measured)
    sensor = self.sensor
    cmd = env.command_manager.get_command(command_name)[:, :3]
    frame = torch.cat([sensor[:, :6], cmd, sensor[:, 6:30]], dim=-1)
    if step != self.last_step:
      self.hist = torch.roll(self.hist, shifts=-1, dims=1)
    self.hist[:, -1] = frame
    if self.needs_fill.any():
      fill = self.needs_fill.nonzero(as_tuple=False).flatten()
      self.hist[fill] = frame[fill].unsqueeze(1).expand(-1, HT.LC.HISTORY, -1)
      self.needs_fill[fill] = False
    self.last_step = step
    return self.hist.reshape(self.n, HT.LC.HISTORY * 33)


class frame_stack_lu(HT.frame_stack_noprev):
  """R2 arm C1-LU (coordinator 11:41, decision 3): the 570 frame stack whose sensor-noise size is drawn per episode and
  per channel type from a log-uniform range [LU_RANGE] times the nominal deploy model -- gyro (white, bias, scale error,
  spikes), attitude (bias, drift, white), joint position (white), joint velocity (white). The latency blend and the
  encoder bias (a body randomisation of the canonical distribution) are unchanged."""

  LU_RANGE = (0.003, 1.0)

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.lu = torch.ones(self.n, 4, device=self.dev)             # gyro, attitude, q, dq
    self.g_lu = torch.Generator(device=self.dev)
    self.g_lu.manual_seed(int(env.cfg.seed) * 7919 + 303)

  def reset(self, env_ids):
    super().reset(env_ids)
    if not self.noise:
      return
    ids = self.env_ids if env_ids is None else (self.env_ids[env_ids] if isinstance(env_ids, slice) else env_ids)
    if len(ids) == 0:
      return
    lo, hi = math.log(self.LU_RANGE[0]), math.log(self.LU_RANGE[1])
    self.lu[ids] = torch.exp(lo + (hi - lo) * torch.rand(len(ids), 4, device=self.dev, generator=self.g_lu))
    self.gyro_bias[ids] *= self.lu[ids, 0:1]
    self.gyro_scale[ids] = 1.0 + (self.gyro_scale[ids] - 1.0) * self.lu[ids, 0:1]
    self.tilt_bias[ids] *= self.lu[ids, 1:2]

  def _measure(self, env, env_ids):
    d = self.asset.data
    gyro = d.root_link_ang_vel_b[env_ids]
    grav = d.projected_gravity_b[env_ids]
    q_e = (d.joint_pos_biased if self.noise else d.joint_pos)[env_ids] - self.default_e
    q = q_e[:, self.perm]
    dq = d.joint_vel[env_ids][:, self.perm]
    if self.noise:
      p = self.np
      n = env_ids.numel()
      s = self.lu[env_ids]
      gyro = gyro * self.gyro_scale[env_ids] + self.gyro_bias[env_ids] + torch.randn_like(gyro) * p["gyro_std"] * s[:, 0:1]
      spike = (torch.rand(n, 1, device=self.dev) < p["spike_prob"]).float()
      gyro = gyro + spike * (torch.rand_like(gyro) * 2 - 1) * p["spike_mag"] * s[:, 0:1]
      bias = self.tilt_bias[env_ids] + torch.randn(n, 2, device=self.dev) * p["tilt_drift"] * s[:, 1:2]
      lim = 1.5 * p["tilt_bias"] * s[:, 1:2]
      self.tilt_bias[env_ids] = torch.maximum(torch.minimum(bias, lim), -lim)
      rv = torch.zeros(n, 3, device=self.dev)
      rv[:, :2] = self.tilt_bias[env_ids] + torch.randn(n, 2, device=self.dev) * p["tilt_std"] * s[:, 1:2]
      grav = grav + torch.cross(rv, grav, dim=-1)
      grav = grav / grav.norm(dim=-1, keepdim=True).clamp_min(1e-6)
      q = q + (torch.rand_like(q) * 2 - 1) * p["q_noise"] * s[:, 2:3]
      dq = dq + torch.randn_like(dq) * p["dq_std"] * s[:, 3:4]
    return torch.cat([gyro, grav, q, dq], dim=-1)


def _quat_mul(a, b):
  aw, ax, ay, az = a.unbind(-1)
  bw, bx, by, bz = b.unbind(-1)
  return torch.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                      aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], -1)


def hold_r2_env_cfg(actor_clean: bool = False, play: bool = False, velocity_input: bool = True, sense_lu: bool = False,
                    human_starts: bool = True, hit_impulse="dist"):
  """R2 Hold environment: the R1 570 Hold task (beta = 1) + the R2 starts and hits (training only). actor_clean = the
  T teacher (actor reads the noise-free frames); velocity_input=False = arm V (actor frames without joint velocity,
  330 values; the critic keeps its privileged inputs). One-factor runs (coordinator 12:18, wave-1 collapse):
  human_starts=False drops the human-frame starts (every reset is the R1 default reset); hit_impulse=None drops the
  hits, "dist" keeps the distribution's impulse range, (lo, hi) replaces only the impulse range (N s)."""
  from mjlab.managers.event_manager import EventTermCfg
  cfg = HT.hold_env_cfg(prev_action=False, beta=1.0, play=play)
  if actor_clean:
    grp = cfg.observations["actor"]
    grp.terms["frames"].params["noise"] = False
    grp.enable_corruption = False
  if not velocity_input:
    cfg.observations["actor"].terms["frames"].func = frame_stack_novel
  if sense_lu:                                                # arm C1-LU: per-episode, per-channel noise size
    cfg.observations["actor"].terms["frames"].func = frame_stack_lu
  if not play:
    dist = distribution()
    hits, starts = dist["hits"], dist["starts"]
    if abs(hits["duration_s"] - 0.02) > 1e-9:                 # one policy step (torso_hits checks env.step_dt too)
      raise ValueError("hit duration of the distribution file is not one policy step")
    if abs(hits["azimuth_rad"][1] - hits["azimuth_rad"][0] - 2 * math.pi) > 1e-6:
      raise ValueError("hit azimuth range of the distribution file is not the full circle")
    frames = pinned(START_FRAMES)
    if str(np.load(frames)["distribution_sha256"]) != R2_DIST["sha256"]:
      raise ValueError(f"{frames} was not derived from the pinned distribution file")
    if human_starts:
      cfg.events["human_start"] = EventTermCfg(func=human_start, mode="reset", params=dict(
        frames_file=str(frames), fraction=starts["human_fraction"], clearance_m=starts["clearance_m"],
        sha256=START_FRAMES["sha256"]))
    if hit_impulse is not None:
      impulse = list(hits["impulse_ns"]) if hit_impulse == "dist" else [float(hit_impulse[0]), float(hit_impulse[1])]
      if not 0.0 < impulse[0] <= impulse[1]:
        raise ValueError(f"bad hit impulse range {impulse}")
      cfg.events["torso_hits"] = EventTermCfg(func=torso_hits, mode="step", params=dict(
        impulse_ns=impulse, height_m=list(hits["height_m"]), first_s=list(hits["first_s"]),
        gap_s=list(hits["gap_s"]), code_sha256=CODE_SHA256, distribution_sha256=R2_DIST["sha256"]))
  return cfg


def hold_r2_rl_cfg(max_iterations: int = 1500, arm: str = "c0", entropy_coef: float | None = None):
  """C0 / T / V: the R1 PPO recipe (white Gaussian exploration). A: residual AR(1) exploration with the exact
  likelihood (r1adv.positive.ar1), everything else the same. entropy_coef replaces only the entropy bonus
  (one-factor arm P-e0)."""
  cfg = HT.hold_rl_cfg(max_iterations)
  if entropy_coef is not None:
    cfg.algorithm.entropy_coef = float(entropy_coef)
  if arm == "a":
    cfg.actor.class_name = "r1adv.positive.ar1:AR1HoldSymmetricMLP"
    cfg.actor.distribution_cfg = dict(cfg.actor.distribution_cfg, class_name="r1adv.positive.ar1:ScaledGaussian")
    cfg.algorithm.class_name = "r1adv.positive.ar1:AR1PPO"
  return cfg


def record() -> dict:
  d = distribution()
  return dict(code_sha256=CODE_SHA256, r2_distribution=dict(R2_DIST), start_frames=dict(START_FRAMES),
              starts=d["starts"], hits={k: (list(v) if isinstance(v, tuple) else v) for k, v in d["hits"].items()})


if __name__ == "__main__":
  print(json.dumps(record(), indent=1))
