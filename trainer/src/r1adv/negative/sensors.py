"""Per-instance sensor degradation at the actor boundary (evaluation and level search; physics untouched).

From the spec's env.sense (all draws from its noise_seed, so a function of the spec only):
  scale                 uniform noise of the N1 shape (gyro 0.2 rad/s, attitude 0.05 rad, joint 0.01 rad, joint
                        speed 0.5 rad/s) times scale, fresh at every control step
  gyro_bias_radps       constant gyro offset (3)
  attitude_bias_rad     constant IMU mounting tilt (roll, pitch)
  delay_steps           observation latency in control steps (20 ms each)
  delay_jitter_steps    extra latency 0..J drawn every step; delivery stays in time order (never older than the
                        previous delivery)
Delay applies to measured channels only (IMU, encoders and what is derived from them); plans, commands and the
previous action are the controller's own and arrive at once. Noise is the one of the measurement's own time.

profile "r1lab-deploy-v1" (R2 evaluation protocol, nominal sensor model = r1lab.mdp.observations.frame_stack, the
positive's deploy contract): gyro per-episode bias U(+-0.04) and scale error 1+U(+-0.04), white N(0, 0.08), spikes
(p 0.002, U(+-1.0)); attitude tilt bias U(+-0.04) with a random walk N(0, 0.0005) per step clamped at +-0.06 plus white
N(0, 0.01); encoder bias U(+-0.015) plus white U(+-0.01); joint speed white N(0, 0.4); sensor lag: with p 0.8 the
sensor part is blended toward the previous sample by a per-episode fraction U(0, 1) (0..20 ms). `scale` k multiplies
every amplitude (k = 0: clean, no lag blend). All draws come from the spec's noise_seed whatever k is, so k = 0/1/2
share the same random numbers. joint_vel_model "quant_lp" replaces the white joint-speed noise by a velocity estimated
from quantized joint positions (runner VelocityEstimator: difference at the physics rate, first-order low-pass).

profile "r1-floor-v1" (coordinator 14:56: the measured-floor cell, evaluation only, rounds/R02/negative/
r1_teleop_noise.md): encoder position = quantize(q + white N(0, 9 urad)) on the 15.14 urad lattice of the IM6014 rotor
encoder through the 3:38 reduction (the teleop data's measured quantum); joint speed = the quantized-difference
estimator (quant_lp, 50 Hz) fed by the same white noise at the physics rate (sense.quant.white_sd_rad); IMU = the
r1lab nominal amplitudes x 0.1 (no IMU measurement exists; an assumption); encoder static bias 0 (not measurable from
the teleop data; calibrated); no sensor-lag blend (the measured encoder values are raw lattice samples; the cell's
command-delay axis carries timing). The IMU draws reuse the r1lab stream (common random numbers with the k cells);
the encoder white comes from its own stream (seed, 12), the physics-rate white of the speed estimator from (seed, 13).
sense.quant.imu_scale (default 0.1) brackets the IMU assumption: cells "floor" use 0.1 (conservative until the IMU is
measured), cells "floor_imu001" 0.01 (a typical MEMS level; coordinator 15:3x).
"""
from __future__ import annotations

import numpy as np
import torch

CHANNELS = (("gyro", 3, 0.2), ("attitude", 3, 0.05), ("joint_pos", 24, 0.01), ("joint_vel", 24, 0.5))
WIDTH = sum(w for _, w, _ in CHANNELS)
DEPTH = 8                                  # control steps of history kept for delay (max delay + jitter < DEPTH)


R1LAB = dict(gyro_std=0.08, gyro_bias=0.04, gyro_scale=0.04, spike_prob=0.002, spike_mag=1.0, tilt_bias=0.04,
             tilt_drift=0.0005, tilt_std=0.01, tilt_clamp=1.5, q_noise=0.01, encoder_bias=0.015, dq_std=0.4, lag_prob=0.8)
FLOOR = dict(imu_scale=0.1, q_white_sd=9e-6, lsb=2.0 * np.pi / (32768 * 38.0 / 3.0))


def sense_of(spec: dict) -> dict:
  sense = spec["env"]["sense"]
  return dict(profile=sense.get("profile", "uniform-n1shape-v1"), joint_vel_model=sense.get("joint_vel_model", "white"),
              quant=sense.get("quant"), channels=sense.get("channels"), scale=float(sense.get("scale", 0.0)),
              noise_seed=int(sense["noise_seed"]),
              gyro_bias=list(sense.get("gyro_bias_radps", [0.0, 0.0, 0.0])),
              attitude_bias=list(sense.get("attitude_bias_rad", [0.0, 0.0])),
              delay=int(sense.get("delay_steps", 0)), jitter=int(sense.get("delay_jitter_steps", 0)),
              dropout=float(sense.get("dropout_p", 0.0)))


def r1lab_tapes(s: dict, horizon: int) -> tuple[np.ndarray, dict]:
  """The r1lab deploy-contract sensor model as an additive tape plus per-episode extras (gyro scale, lag fraction)."""
  p, k = R1LAB, s["scale"]
  rng = np.random.Generator(np.random.Philox(np.random.SeedSequence([s["noise_seed"], 11])))
  u = lambda *shape: 2.0 * rng.random(shape) - 1.0  # noqa: E731
  gyro_bias, gyro_scale, tilt0, encoder = u(3) * p["gyro_bias"], u(3) * p["gyro_scale"], u(2) * p["tilt_bias"], u(24) * p["encoder_bias"]
  lag_fraction, lag_on = rng.random(), rng.random() < p["lag_prob"]
  white_gyro = rng.standard_normal((horizon, 3)) * p["gyro_std"]
  spikes = (rng.random(horizon) < p["spike_prob"])[:, None] * u(horizon, 3) * p["spike_mag"]
  drift, tilt_white = rng.standard_normal((horizon, 2)) * p["tilt_drift"], rng.standard_normal((horizon, 2)) * p["tilt_std"]
  q_white, dq_white = u(horizon, 24) * p["q_noise"], rng.standard_normal((horizon, 24)) * p["dq_std"]
  noise = np.zeros((horizon, WIDTH), np.float64)
  extras = dict(gyro_scale=np.ones(3), lag=0.0)
  on = set(s["channels"] or ("gyro", "attitude", "encoder", "joint_vel", "lag"))      # channel ablation (diagnostic)
  if k > 0.0:
    noise[:, 0:3] = k * (gyro_bias + white_gyro + spikes) * ("gyro" in on)
    bias, limit = k * tilt0, p["tilt_clamp"] * k * p["tilt_bias"]
    for t in range(horizon):
      bias = np.clip(bias + k * drift[t], -limit, limit)
      noise[t, 3:5] = (bias + k * tilt_white[t]) * ("attitude" in on)
    noise[:, 6:30] = k * (encoder + q_white) * ("encoder" in on)
    if s["joint_vel_model"] == "white":
      noise[:, 30:54] = k * dq_white * ("joint_vel" in on)
    extras = dict(gyro_scale=1.0 + k * gyro_scale * ("gyro" in on),
                  lag=float(lag_fraction) if lag_on and "lag" in on else 0.0)
  return noise, extras


def floor_tapes(s: dict, horizon: int) -> tuple[np.ndarray, dict]:
  """The measured-floor sensor model (profile r1-floor-v1) as an additive tape plus extras (pos_lsb: quantize the joint
  position of the actor frame on this lattice)."""
  p, f = R1LAB, FLOOR
  rng = np.random.Generator(np.random.Philox(np.random.SeedSequence([s["noise_seed"], 11])))
  u = lambda *shape: 2.0 * rng.random(shape) - 1.0  # noqa: E731
  gyro_bias, gyro_scale, tilt0, _encoder = u(3) * p["gyro_bias"], u(3) * p["gyro_scale"], u(2) * p["tilt_bias"], u(24) * p["encoder_bias"]
  _lag_fraction, _lag_on = rng.random(), rng.random() < p["lag_prob"]            # drawn to keep the r1lab draw order
  white_gyro = rng.standard_normal((horizon, 3)) * p["gyro_std"]
  spikes = (rng.random(horizon) < p["spike_prob"])[:, None] * u(horizon, 3) * p["spike_mag"]
  drift, tilt_white = rng.standard_normal((horizon, 2)) * p["tilt_drift"], rng.standard_normal((horizon, 2)) * p["tilt_std"]
  enc = np.random.Generator(np.random.Philox(np.random.SeedSequence([s["noise_seed"], 12])))
  quant = s.get("quant") or {}                                   # the spec carries the lattice and the white SD
  lsb, white = float(quant.get("resolution_rad", f["lsb"])), float(quant.get("white_sd_rad", f["q_white_sd"]))
  imu = quant.get("imu", True)                                   # diagnostic switch (addendum 2: cause isolation):
  scale = float(quant.get("imu_scale", f["imu_scale"]))          # IMU x nominal: 0.1 (cells "floor"), 0.01 ("floor_imu001")
  mg = scale * (imu is True or imu == "gyro")                    # True = both IMU parts, "gyro" / "attitude" = one
  ma = scale * (imu is True or imu == "attitude")
  m = mg
  noise = np.zeros((horizon, WIDTH), np.float64)
  noise[:, 0:3] = mg * (gyro_bias + white_gyro + spikes)
  bias, limit = ma * tilt0, p["tilt_clamp"] * ma * p["tilt_bias"]
  for t in range(horizon):
    bias = np.clip(bias + ma * drift[t], -limit, limit)
    noise[t, 3:5] = bias + ma * tilt_white[t]
  pos_on = bool(quant.get("position", True))                     # diagnostic switch (addendum 2: cause isolation)
  noise[:, 6:30] = enc.standard_normal((horizon, 24)) * white * pos_on
  return noise, dict(gyro_scale=1.0 + m * gyro_scale, lag=0.0, pos_lsb=lsb if pos_on else 0.0)


def tapes(spec: dict, horizon: int) -> tuple[np.ndarray, np.ndarray]:
  """(noise (horizon, 54) incl. constant bias / mounting tilt, latency (horizon,) in steps) for one instance."""
  noise, latency, _ = tapes_full(spec, horizon)
  return noise, latency


def tapes_full(spec: dict, horizon: int) -> tuple[np.ndarray, np.ndarray, dict]:
  s = sense_of(spec)
  if s["profile"] == "r1-floor-v1":
    noise, extras = floor_tapes(s, horizon)
    latency = np.full(horizon, s["delay"], np.int64)
    if s["jitter"] > 0:
      latency = latency + Draws(s["noise_seed"]).integers(horizon, s["jitter"])
    return noise.astype(np.float32), latency, extras
  if s["profile"] == "r1lab-deploy-v1":
    noise, extras = r1lab_tapes(s, horizon)
    latency = np.full(horizon, s["delay"], np.int64)
    if s["jitter"] > 0:
      latency = latency + Draws(s["noise_seed"]).integers(horizon, s["jitter"])
    return noise.astype(np.float32), latency, extras
  rng = np.random.Generator(np.random.Philox(np.random.SeedSequence(s["noise_seed"])))
  noise = np.zeros((horizon, WIDTH), np.float64)
  if s["scale"] > 0.0:
    amplitude = np.concatenate([np.full(w, a) for _, w, a in CHANNELS])
    noise = (2.0 * rng.random((horizon, WIDTH)) - 1.0) * amplitude * s["scale"]
    noise[:, 5] = 0.0                                          # no absolute yaw sensor is invented
  noise[:, 0:3] += np.asarray(s["gyro_bias"])
  noise[:, 3:5] += np.asarray(s["attitude_bias"])
  latency = np.full(horizon, s["delay"], np.int64)
  if s["jitter"] > 0:
    latency = latency + Draws(s["noise_seed"]).integers(horizon, s["jitter"])
  if s["delay"] + s["jitter"] >= DEPTH:
    raise ValueError("latency beyond the kept history")
  return noise.astype(np.float32), latency, dict(gyro_scale=np.ones(3), lag=0.0)


class Draws:
  """Integer draws for the jitter, separate from the noise stream (version-stable Philox raw bits)."""
  def __init__(self, seed: int):
    self.bits = np.random.Philox(np.random.SeedSequence([int(seed), 7]))

  def integers(self, count: int, high: int) -> np.ndarray:
    raw = self.bits.random_raw(count).astype(np.uint64)
    return (raw % np.uint64(high + 1)).astype(np.int64)


class SensorModel:
  """All instances of a batch. Use: begin(step) once per control step, then degrade(...) for the actor input."""
  profile = "N1_shared_sensor_v1"                              # SharedSensorStream interface (transforms reused)

  def __init__(self, specs_env: list[dict], horizon: int, device):
    pairs = [tapes_full(s, horizon) for s in specs_env]
    self.noise = torch.as_tensor(np.stack([p[0] for p in pairs]), device=device)          # (envs, T, 54)
    self.latency = torch.as_tensor(np.stack([p[1] for p in pairs]), device=device)        # (envs, T)
    self.gyro_scale = torch.as_tensor(np.stack([p[2]["gyro_scale"] for p in pairs]), device=device, dtype=torch.float)
    self.lag = torch.as_tensor([[p[2]["lag"]] for p in pairs], device=device, dtype=torch.float)
    self.pos_lsb = torch.as_tensor([[p[2].get("pos_lsb", 0.0)] for p in pairs], device=device, dtype=torch.float)
    self.noise_seeds = [int(sense_of(s)["noise_seed"]) for s in specs_env]
    self.horizon = horizon
    senses = [sense_of(s) for s in specs_env]
    self.quant = torch.tensor([x["joint_vel_model"] == "quant_lp" for x in senses], device=device)
    self.quant_params = [x["quant"] for x in senses]
    self.clean = torch.tensor([sense_of(s)["scale"] == 0.0 and not any(sense_of(s)["gyro_bias"])
                               and not any(sense_of(s)["attitude_bias"]) for s in specs_env], device=device)
    self.delayed = (self.latency.max(dim=1).values > 0)
    self.seeds = list(range(len(specs_env)))
    self.rows = torch.arange(len(specs_env), device=device)
    self.used = torch.full((len(specs_env),), -1, dtype=torch.long, device=device)
    self.step = 0
    self.measured = torch.zeros(len(specs_env), dtype=torch.long, device=device)

  def begin(self, step: int) -> None:
    """Which measurement each environment receives now: max(previous, step - latency), clipped at 0."""
    self.step = step
    target = (step - self.latency[:, min(step, self.latency.shape[1] - 1)]).clamp_min(0)
    self.used = torch.maximum(self.used, target)
    self.measured = self.used.clone()

  def tensors(self, step, like):
    """Noise of each environment's measurement time (SharedSensorStream.tensors interface; `step` ignored)."""
    row = self.noise[self.rows, self.measured.clamp_max(self.noise.shape[1] - 1)].to(like.dtype)
    out, start = {}, 0
    for name, width, _ in CHANNELS:
      out[name] = row[:, start:start + width]
      start += width
    return out


class DelayLine:
  """Ring buffer of recent clean measurements (envs x DEPTH x width): deliver the one measured at `measured`."""
  def __init__(self, count: int, width: int, device):
    self.ring = torch.zeros(DEPTH, count, width, device=device)
    self.rows = torch.arange(count, device=device)
    self.filled = False

  def push(self, step: int, value: torch.Tensor) -> None:
    if not self.filled:
      self.ring[:] = value.unsqueeze(0)
      self.filled = True
    self.ring[step % DEPTH] = value

  def get(self, measured: torch.Tensor) -> torch.Tensor:
    return self.ring[measured % DEPTH, self.rows]

  def restart(self, envs: torch.Tensor, value: torch.Tensor) -> None:
    """A reset environment has no older measurement than its first."""
    if envs.numel():
      self.ring[:, envs] = value[envs].unsqueeze(0)
