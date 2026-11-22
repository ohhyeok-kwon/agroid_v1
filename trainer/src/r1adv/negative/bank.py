"""Terrain bank v0: a fixed grid of patches built from the r1wb ground family (pure description, no simulator).

Every patch is 12 m wide and every instance starts at its centre. Rough and gravel patches keep the flat start disc
of r1wb.world (radius 0.75 m, 0.30 m blend); stairs keep mjlab's 1.5 m platform. A sloped patch is one plane through
the centre, uphill along +y, without a flat disc (a disc inside a tilted plane needs a lip: r1wb "plane" makes a
30-60 deg rim when walked radially uphill, measured 11-17), and the robot is placed on it by foot geometry as the
W protocol did. The instance's spawn yaw alone sets the slope direction: heading +x walks across it, +y climbs it.
Flat and plain sloped patches are boxes (a height field under a standing robot added joint jerk 30-50 rad/s^3 after a
walk, against 0 on a plane, 11-17 positive control); bumps, stones and tilt_rough stay height fields.
Many worlds share one patch (MuJoCo Warp keeps the worlds apart), so the bank is fixed and instances only choose.
"""
from __future__ import annotations

import math

from r1adv.negative.seedtree import canonical, child_seed, sha256_text

NAME = "neg-bank-v0.2"
PATCH = 12.0          # m, side of a patch
ROWS = 8              # levels per kind
START, BLEND = 0.75, 0.30   # r1wb.world start disc and blend (m)

# kind -> unit, levels (one per row), height-field resolution (m); stairs are mjlab box pyramids (as in r1wb training)
COLUMNS = (
  dict(kind="flat", unit="-", levels=(0, 0, 0, 0, 0, 0, 0, 0), resolution=0.1),
  dict(kind="rough", unit="cm", levels=(2.0, 3.5, 5.0, 6.5, 8.0, 9.5, 11.0, 12.5), resolution=0.1),
  dict(kind="gravel", unit="cm", levels=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0), resolution=0.05),
  dict(kind="tilt", unit="deg", levels=(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0), resolution=0.1),
  dict(kind="steep", unit="deg", levels=(17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 25.0), resolution=0.1),
  dict(kind="stairs_up", unit="cm", levels=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), step_width=0.4, platform=1.5),
  dict(kind="stairs_down", unit="cm", levels=(2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0), step_width=0.4, platform=1.5),
  dict(kind="tilt_rough", unit="deg+cm", levels=((4.0, 2.0), (6.0, 3.0), (8.0, 4.0), (10.0, 5.0), (12.0, 6.0),
                                                 (14.0, 7.0), (16.0, 8.0), (18.0, 9.0)), resolution=0.1),
)
KINDS = tuple(c["kind"] for c in COLUMNS)
SLOPED = ("tilt", "steep", "tilt_rough")


def column(kind: str) -> int:
  return KINDS.index(kind)


def level(kind: str, row: int):
  return COLUMNS[column(kind)]["levels"][row]


def slope_deg(kind: str, row: int) -> float:
  value = level(kind, row)
  if kind in ("tilt", "steep"):
    return float(value)
  if kind == "tilt_rough":
    return float(value[0])
  return 0.0


def description(bank_seed: int) -> dict:
  return dict(name=NAME, patch_m=PATCH, rows=ROWS, start_disc_m=START, blend_m=BLEND, seed=int(bank_seed),
              slopes="full plane through the centre, uphill +y, placement start",
              geometry="flat: mjlab box; tilt/steep: one rotated box; rough/gravel/tilt_rough: height fields; stairs: mjlab boxes",
              columns=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()} for c in COLUMNS])


def bank_sha256(bank_seed: int) -> str:
  return sha256_text(canonical(description(bank_seed)))


def patch_seed(bank_seed: int, kind: str, row: int) -> int:
  return child_seed(bank_seed, f"patch/{kind}/{row}")


def plane_height(kind: str, row: int, y):
  """Height of the patch's plane above its centre at lateral coordinate y (sloped kinds; 0 otherwise)."""
  return math.tan(math.radians(slope_deg(kind, row))) * y


def slope_at(kind: str, row: int, x: float, y: float) -> tuple[float, tuple[float, float]]:
  """Slope angle (rad) of the patch's mean surface at (x, y) from its centre, and the uphill unit vector (world)."""
  angle = math.radians(slope_deg(kind, row))
  if angle == 0.0:
    return 0.0, (0.0, 0.0)
  return angle, (0.0, 1.0)
