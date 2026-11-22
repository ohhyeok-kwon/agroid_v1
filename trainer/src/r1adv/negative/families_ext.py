"""Environment families of the runner extensions (3)-(5): Hold under thrown objects, body errors and a humming floor.

Same contract as families_v1 (PARAM_SPACE + make(params, rng, mission), mission = families_v1.mission_hold: settle
1 s, Hold 8 s). They are physical levels (clean sensors), so the witness rule is "physical". They are kept out of
the R1 Hold stream until the coordinator admits them; the runner accepts them (runnable "legacy").

  hold-throw-v1      a ball (1 kg) or a box (5 kg) thrown at the torso; its momentum is the difficulty axis (the
                     impulse families' unit), speed = momentum / mass, flying straight (r1wb.world.Thrower)
  hold-body-v1       a heavier or lighter robot (uniform density), a torso payload, slippery or sticky soles, weaker or
                     stiffer joint servos (kp, kd)
  hold-vibration-v1  stand on a shaker plate (2 x 2 m, top 2 cm above the floor) vibrating along x, y or z
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

from r1adv.negative.families import FOOT_FRICTION, Family, _terrain
from r1adv.negative.families_v1 import HOLD_S, _env, _sense
from r1adv.negative.generator import SETTLE_S, ticks
from r1adv.negative.seedtree import r6

MASS = {"ball": 1.0, "box": 5.0}           # r1wb.world BALL / BOX
SPEED_MAX = {"ball": 25.0, "box": 8.0}


def _flat(d) -> dict:
  return _terrain("flat", 0, d.uniform(-math.pi, math.pi))


def _throw(p, d, m):
  env = _env(_flat(d), _sense(d))
  start = SETTLE_S + p["phase"] * (HOLD_S - 2.0)               # at least 2 s of Hold remain after the hit
  env["disturbances"] = [dict(kind="throw", body=p["body"], t_s=ticks(start),
                              speed_mps=r6(p["momentum_ns"] / MASS[p["body"]]), azimuth_rad=r6(math.radians(p["azimuth_deg"])),
                              height_m=r6(p["height_m"]))]
  return env


def _throw_limits(p):
  return [f"{p['body']} faster than {SPEED_MAX[p['body']]} m/s"] if p["momentum_ns"] / MASS[p["body"]] > SPEED_MAX[p["body"]] else []


def _body(p, d, m):
  env = _env(_flat(d), _sense(d))
  env["body"] = dict(mass_scale=r6(p["mass_scale"]), payload_kg=r6(p["payload_kg"]), friction_scale=r6(p["friction_scale"]),
                     kp_scale=r6(p["kp_scale"]), kd_scale=r6(p["kd_scale"]))
  return env


def _vibration(p, d, m):
  env = _env(_flat(d), _sense(d))
  env["disturbances"] = [dict(kind="floor_vibration", amplitude_m=r6(p["amplitude_mm"] / 1000.0), frequency_hz=r6(p["frequency_hz"]),
                              axis=p["axis"], t_s=ticks(SETTLE_S))]
  return env


def _vibration_limits(p):
  # vertical: a plate accelerating down faster than g leaves the feet (no standing solution); horizontal: beyond 0.8 of
  # the friction cone the soles slide on the plate
  peak = (2.0 * math.pi * p["frequency_hz"]) ** 2 * p["amplitude_mm"] / 1000.0
  if p["axis"] == "z":
    return ["vertical plate acceleration beyond g"] if peak >= 9.81 else []
  return ["horizontal plate acceleration beyond 0.8 of the friction cone"] if peak >= 0.8 * FOOT_FRICTION * 9.81 else []


FAMILIES = {f.name: f for f in (
  Family("hold-throw-v1", "trap", "legacy", "physical", "stand still while a ball or a box hits the torso",
         {"body": ("choice", ("ball", "box")), "momentum_ns": ("logfloat", 4.0, 30.0), "azimuth_deg": ("float", -180.0, 180.0),
          "height_m": ("float", -0.2, 0.3), "phase": ("float", 0.05, 0.9)}, _throw, _throw_limits),
  Family("hold-body-v1", "trap", "legacy", "physical", "stand still with a different body: mass, payload, soles, servos",
         {"mass_scale": ("float", 0.9, 1.1), "payload_kg": ("float", 0.0, 5.0), "friction_scale": ("logfloat", 0.4, 1.4),
          "kp_scale": ("float", 0.85, 1.15), "kd_scale": ("float", 0.85, 1.15)}, _body),
  Family("hold-vibration-v1", "trap", "legacy", "physical", "stand still on a humming floor (machine room, bus deck)",
         {"amplitude_mm": ("float", 0.3, 3.0), "frequency_hz": ("float", 5.0, 30.0), "axis": ("choice", ("x", "y", "z"))},
         _vibration, _vibration_limits),
)}


def source_sha256() -> str:
  here = Path(__file__).resolve().parent
  digest = hashlib.sha256()
  for name in ("families_ext.py", "families_v1.py", "families.py", "levels.py", "generator.py", "bank.py", "seedtree.py", "__init__.py"):
    digest.update(name.encode() + b"\0" + (here / name).read_bytes() + b"\0")
  return digest.hexdigest()
