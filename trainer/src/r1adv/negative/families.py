"""Upper layer of generator v0.2: environment families (coordinator 11-17 synthesis, section 4).

A family = PARAM_SPACE (typed ranges) + make(params, rng, mission) -> d_env, a one-line real situation, the kind of
regret it targets ("failure": the robot falls or misses the goal; "trap": staying calm is the answer and over-reaction
fails J or G; "baseline": ordinary levels kept against forgetting) and its witness rule ("teacher": privileged
physics; "student": the best student with the same observations, for sensor degradation). make() may use only the
injected rng (seedtree.Draw) and the mission (coordinator's d_goal, read-only). A family never judges success.
runnable: "legacy" = r1adv.negative.runner can run it with the existing r1wb policies; "future" = spec only.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from r1adv.negative import bank as B
from r1adv.negative.generator import BANK_SEED, DT, SETTLE_S, ticks
from r1adv.negative.seedtree import Draw, r6

ROBOT_MASS_KG = 30.0          # R1 with hands, rounded up (r1wb.robot + O6 hands); used by the analytic capture check
COM_HEIGHT_M = 0.55
STEP_REACH_M = 0.25
FOOT_FRICTION = 1.0           # MuJoCo default sliding friction of the R1 foot capsules


def sample(space: dict, d: Draw) -> dict:
  out = {}
  for name, spec in space.items():
    kind = spec[0]
    if kind == "float":
      out[name] = r6(d.uniform(spec[1], spec[2]))
    elif kind == "logfloat":
      out[name] = r6(d.loguniform(spec[1], spec[2]))
    elif kind == "int":
      out[name] = d.integer(spec[1], spec[2])
    elif kind == "choice":
      out[name] = d.choice(spec[1])
    else:
      raise ValueError(kind)
  return out


def mutate(params: dict, space: dict, d: Draw, rate: float = 0.35, scale: float = 0.12) -> dict:
  """ACCEL-style small edit: each parameter moves with probability `rate` by about `scale` of its range."""
  out = dict(params)
  for name, spec in space.items():
    if not d.bernoulli(rate):
      continue
    kind = spec[0]
    if kind in ("float", "logfloat"):
      lo, hi = spec[1], spec[2]
      gauss = math.sqrt(-2.0 * math.log(max(d.u(), 1e-12))) * math.cos(2 * math.pi * d.u())
      if kind == "float":
        out[name] = r6(min(max(out[name] + gauss * scale * (hi - lo), lo), hi))
      else:
        out[name] = r6(min(max(math.exp(math.log(out[name]) + gauss * scale * math.log(hi / lo)), lo), hi))
    elif kind == "int":
      out[name] = min(max(out[name] + d.choice((-1, 1)), spec[1]), spec[2])
    else:
      out[name] = d.choice([o for o in spec[1] if o != out[name]] or list(spec[1]))
  return out


def _terrain(kind: str, row: int, yaw: float) -> dict:
  value = B.level(kind, row)
  return dict(bank=B.NAME, bank_seed=BANK_SEED, bank_sha256=B.bank_sha256(BANK_SEED), kind=kind, row=int(row),
              level=list(value) if isinstance(value, tuple) else value, slope_deg=r6(B.slope_deg(kind, row)),
              spawn_yaw_rad=r6(math.atan2(math.sin(yaw), math.cos(yaw))))


def _nearest_row(kind: str, value: float) -> int:
  levels = [v[0] if isinstance(v, tuple) else v for v in B.COLUMNS[B.column(kind)]["levels"]]
  return min(range(B.ROWS), key=lambda r: abs(levels[r] - value))


def _slope_patch(slope_deg: float, rough_cm: float) -> tuple[str, int]:
  if rough_cm >= 1.0:
    levels = B.COLUMNS[B.column("tilt_rough")]["levels"]
    return "tilt_rough", min(range(B.ROWS), key=lambda r: abs(levels[r][0] - slope_deg) + abs(levels[r][1] - rough_cm))
  kind = "tilt" if slope_deg <= 16.0 else "steep"
  return kind, _nearest_row(kind, slope_deg)


def _first_travel(mission: dict) -> float:
  for step in mission["steps"]:
    if step["type"] == "Go" and step.get("mode") == "velocity" and math.hypot(step["vx"], step["vy"]) > 1e-6:
      return math.atan2(step["vy"], step["vx"])
  return 0.0


def _neutral_sense(d: Draw, scale: float = 0.0) -> dict:
  from r1adv.negative.seedtree import child_seed
  return dict(profile="uniform-n1shape-v1", scale=r6(scale), delay_steps=0, dropout_p=0.0,
              noise_seed=child_seed(int(d.u() * (1 << 53)), "sense/noise"))


NEUTRAL_BODY = dict(mass_scale=1.0, payload_kg=0.0, friction_scale=1.0, kp_scale=1.0, kd_scale=1.0)


def _impulses(params: dict, mission: dict) -> list[dict]:
  start = SETTLE_S + params["phase"] * max(mission["duration_s"] - SETTLE_S - 1.5, 0.5)
  return [dict(kind="impulse", t_s=ticks(start), body="torso_link", impulse_ns=params["impulse_ns"],
               burst_count=params["burst"], burst_gap_s=r6(params["gap_ticks"] * DT), duration_s=DT,
               azimuth_rad=r6(math.radians(params["azimuth_deg"])), height_m=params["height_m"])]


@dataclass
class Family:
  name: str
  kind: str
  runnable: str
  witness: str
  situation: str
  PARAM_SPACE: dict
  build: Callable
  limits: Callable = field(default=lambda params: [])

  def make(self, params: dict, rng: Draw, mission: dict) -> dict:
    return self.build(params, rng, mission)

  def feasible(self, params: dict) -> list[str]:
    """Violated analytic necessary conditions (empty = admissible); the witness decides the rest."""
    return self.limits(params)


def _slope_limits(params):
  out = []
  if math.tan(math.radians(params["slope_deg"])) >= 0.8 * FOOT_FRICTION:
    out.append("static slope beyond 0.8 of the friction cone")
  return out


def _push_limits(params):
  velocity = params["impulse_ns"] * params["burst"] / ROBOT_MASS_KG
  shift = velocity * math.sqrt(COM_HEIGHT_M / 9.81)
  return [f"capture point shift {shift:.2f} m beyond three steps"] if shift > 3 * STEP_REACH_M else []


def _baseline(p, d, m):
  kind = p["kind"]
  row = 0 if kind == "flat" else min(p["row"], 1)
  return dict(terrain=_terrain(kind, row, d.uniform(-math.pi, math.pi)), scene=None, obstacles=[], disturbances=[],
              sense=_neutral_sense(d, p["noise_scale"]), body=dict(NEUTRAL_BODY))


def _slope_hold(p, d, m):
  kind, row = _slope_patch(p["slope_deg"], p["rough_cm"])
  yaw = math.pi / 2 + math.radians(p["heading_deg"])                  # heading relative to the uphill direction (+y)
  return dict(terrain=_terrain(kind, row, yaw), scene=None, obstacles=[], disturbances=[],
              sense=_neutral_sense(d, p["noise_scale"]), body=dict(NEUTRAL_BODY))


def _slope_travel(sign):
  def build(p, d, m):
    kind, row = _slope_patch(p["slope_deg"], p["rough_cm"])
    # the mission's first walking direction (body frame) is turned to point `travel_deg` off uphill (or downhill)
    yaw = (math.pi / 2 if sign > 0 else -math.pi / 2) + math.radians(p["travel_deg"]) - _first_travel(m)
    return dict(terrain=_terrain(kind, row, yaw), scene=None, obstacles=[], disturbances=[],
                sense=_neutral_sense(d, p["noise_scale"]), body=dict(NEUTRAL_BODY))
  return build


def _push(p, d, m):
  kind = p["ground"]
  row = 0 if kind == "flat" else min(p["ground_row"], 3)
  return dict(terrain=_terrain(kind, row, d.uniform(-math.pi, math.pi)), scene=None, obstacles=[],
              disturbances=_impulses(p, m), sense=_neutral_sense(d, p["noise_scale"]), body=dict(NEUTRAL_BODY))


def _slope_push(p, d, m):
  env = _slope_hold(p, d, m)
  env["disturbances"] = _impulses(p, m)
  return env


def _ground(kind_param, value_param):
  def build(p, d, m):
    kind = p[kind_param]
    return dict(terrain=_terrain(kind, _nearest_row(kind, p[value_param]), d.uniform(-math.pi, math.pi)), scene=None,
                obstacles=[], disturbances=[], sense=_neutral_sense(d, p["noise_scale"]), body=dict(NEUTRAL_BODY))
  return build


def _sensor(p, d, m):
  return dict(terrain=_terrain("flat", 0, d.uniform(-math.pi, math.pi)), scene=None, obstacles=[], disturbances=[],
              sense=dict(_neutral_sense(d, p["noise_scale"]), gyro_bias_radps=[r6(d.uniform(-1, 1) * p["gyro_bias"]) for _ in range(3)],
                         attitude_bias_rad=[r6(d.uniform(-1, 1) * p["attitude_bias"]) for _ in range(2)],
                         delay_steps=p["delay_steps"], delay_jitter_steps=p["delay_jitter"], dropout_p=p["dropout_p"]),
              body=dict(NEUTRAL_BODY))


def _vibration(p, d, m):
  return dict(terrain=_terrain("flat", 0, d.uniform(-math.pi, math.pi)), scene=None, obstacles=[], disturbances=[
    dict(kind="floor_vibration", amplitude_m=p["amplitude_mm"] / 1000.0, frequency_hz=p["frequency_hz"],
         axis=p["axis"], t_s=SETTLE_S)], sense=_neutral_sense(d, 0.0), body=dict(NEUTRAL_BODY))


def _indoor(p, d, m):
  slots = [dict(slot=i, kind="box", half_m=r6(d.uniform(0.1, 0.3)), distance_m=r6(d.uniform(0.6, 2.0)),
                bearing_rad=r6(d.uniform(-math.pi, math.pi))) for i in range(p["boxes"])]
  return dict(terrain=None, scene=dict(source="atlas", atlas="atlas_hm3d3_v0", tag=p["tag"], index=d.integer(0, 1 << 20)),
              obstacles=slots, disturbances=[], sense=_neutral_sense(d, p["noise_scale"]), body=dict(NEUTRAL_BODY))


def _throw(p, d, m):
  env = _baseline(dict(kind="flat", row=0, noise_scale=0.0), d, m)
  env["disturbances"] = [dict(kind="throw", body=p["body"], t_s=ticks(SETTLE_S + p["phase"] * max(m["duration_s"] - SETTLE_S - 1.5, 0.5)),
                              speed_mps=p["speed_mps"], azimuth_rad=r6(math.radians(p["azimuth_deg"])), height_m=p["height_m"])]
  return env


def _body(p, d, m):
  env = _baseline(dict(kind="flat", row=0, noise_scale=0.0), d, m)
  env["body"] = dict(mass_scale=p["mass_scale"], payload_kg=p["payload_kg"], friction_scale=p["friction_scale"],
                     kp_scale=p["kp_scale"], kd_scale=p["kd_scale"])
  return env


PUSH = {"impulse_ns": ("logfloat", 4.0, 35.0), "azimuth_deg": ("float", -180.0, 180.0), "height_m": ("float", -0.15, 0.25),
        "burst": ("int", 1, 4), "gap_ticks": ("int", 8, 20), "phase": ("float", 0.05, 0.9)}
SLOPE = {"slope_deg": ("float", 4.0, 25.0), "rough_cm": ("choice", (0.0, 0.0, 3.0, 6.0))}
NOISE = {"noise_scale": ("choice", (0.0, 0.0, 1.0))}

FAMILIES = {f.name: f for f in (
  Family("baseline-v0", "baseline", "legacy", "teacher", "ordinary floors, standing and walking",
         {"kind": ("choice", ("flat", "rough", "gravel")), "row": ("int", 0, 1), "noise_scale": ("float", 0.0, 0.5)}, _baseline),
  Family("slope-hold-v0", "trap", "legacy", "teacher", "stop and stand on a ramp or a grassy bank",
         {**SLOPE, "heading_deg": ("float", -180.0, 180.0), **NOISE}, _slope_hold, _slope_limits),
  Family("uphill-v0", "failure", "legacy", "teacher", "walk up a ramp",
         {**SLOPE, "travel_deg": ("float", -60.0, 60.0), **NOISE}, _slope_travel(+1), _slope_limits),
  Family("downhill-v0", "failure", "legacy", "teacher", "walk down a ramp",
         {**SLOPE, "travel_deg": ("float", -60.0, 60.0), **NOISE}, _slope_travel(-1), _slope_limits),
  Family("push-v0", "failure", "legacy", "teacher", "bumped by a person or a cart",
         {**PUSH, "ground": ("choice", ("flat", "rough", "gravel")), "ground_row": ("int", 0, 3), **NOISE}, _push, _push_limits),
  Family("slope-push-v0", "failure", "legacy", "teacher", "bumped while on a ramp",
         {**PUSH, **SLOPE, "heading_deg": ("float", -180.0, 180.0), **NOISE}, _slope_push,
         lambda p: _slope_limits(p) + _push_limits(p)),
  Family("stairs-v0", "failure", "legacy", "teacher", "low steps and stairs",
         {"kind": ("choice", ("stairs_up", "stairs_down")), "step_cm": ("float", 2.0, 9.0), **NOISE}, _ground("kind", "step_cm"),
         lambda p: ["step beyond 12 cm"] if p["step_cm"] > 12 else []),
  Family("rough-v0", "failure", "legacy", "teacher", "uneven tiles, gravel, a rug edge",
         {"kind": ("choice", ("rough", "gravel")), "height_cm": ("float", 1.0, 12.5), **NOISE}, _ground("kind", "height_cm")),
  Family("sensor-noise-v0", "trap", "legacy", "student", "a noisy IMU and encoders while standing and walking",
         {"noise_scale": ("float", 0.5, 1.5)}, lambda p, d, m: _baseline(dict(kind="flat", row=0, noise_scale=p["noise_scale"]), d, m)),
  Family("sensor-bias-delay-v0", "trap", "future", "student", "a biased, late and dropping sensor stream",
         {"noise_scale": ("float", 0.0, 1.0), "gyro_bias": ("float", 0.0, 0.1), "attitude_bias": ("float", 0.0, 0.05),
          "delay_steps": ("int", 0, 3), "delay_jitter": ("int", 0, 2), "dropout_p": ("float", 0.0, 0.1)}, _sensor),
  Family("vibration-floor-v0", "trap", "future", "teacher", "a humming floor (machine room, bus deck)",
         {"amplitude_mm": ("float", 0.3, 3.0), "frequency_hz": ("float", 5.0, 30.0), "axis": ("choice", ("z", "x", "y"))}, _vibration),
  Family("indoor-v0", "failure", "future", "teacher", "rooms with doorways, thresholds and clutter",
         {"tag": ("choice", ("doorway", "corridor", "threshold", "clutter", "open")), "boxes": ("int", 0, 4), **NOISE}, _indoor),
  Family("throw-v0", "failure", "future", "teacher", "a ball or a box thrown at the robot",
         {"body": ("choice", ("ball", "box")), "speed_mps": ("float", 2.0, 20.0), "azimuth_deg": ("float", -180.0, 180.0),
          "height_m": ("float", -0.2, 0.3), "phase": ("float", 0.05, 0.9)}, _throw),
  Family("body-v0", "failure", "future", "teacher", "a heavier robot, a backpack, slippery soles, weak motors",
         {"mass_scale": ("float", 0.9, 1.1), "payload_kg": ("float", 0.0, 3.0), "friction_scale": ("logfloat", 0.4, 1.4),
          "kp_scale": ("float", 0.85, 1.15), "kd_scale": ("float", 0.85, 1.15)}, _body),
)}


def family_source_sha256() -> str:
  here = Path(__file__).resolve().parent
  digest = hashlib.sha256()
  for name in ("families.py", "levels.py", "generator.py", "bank.py", "seedtree.py", "__init__.py"):
    digest.update(name.encode() + b"\0" + (here / name).read_bytes() + b"\0")
  return digest.hexdigest()
