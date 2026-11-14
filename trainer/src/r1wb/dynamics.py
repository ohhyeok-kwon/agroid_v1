"""Centroidal quantities and the balance-urgency signal (privileged: rewards, critic, evaluation).

All signals use the capture point of the velocity error,
  xi = CoM_xy + (v_xy - v_ref_xy) / w0 + rot(L_xy) / (m z w0),
so that steady travel at the commanded velocity v_ref counts as no divergence (L: angular momentum about the CoM).

`support_margin`  d  = signed distance of xi to the convex hull of BOTH soles' ground projections, loaded or not
                       (< 0 inside). No assumed step time or reach.
`capture_margin`  d1 = if xi is outside the soles in contact: put the centre of pressure at the nearest point of the
                       stance sole, let xi run for STEP_TIME, and measure how far it then lies outside the area the
                       other foot can reach (grown by the sole size); 0 if no change of support is needed.
`urgency_margin`     = d1 where d > 0, else 0: "one ordinary step is not enough". This conjunction is the adopted
                       gate (second training review, section 4). On 280 recorded rollouts of four policies it was
                       above its dead band in at most 0.03% of calm frames for every reach box tried, whereas d
                       alone fires on walking hits that the manufacturer's gait absorbs (10/10) and d1 alone reads
                       35 cm in ordinary wide side steps. Callers add the hold (about 0.3 s) and the
                       "an external impulse was applied" condition; both are proposals that still need the
                       calibration on a frozen System 1 policy (review 4.4), as does the reach box below.
The model is a linear inverted pendulum with a flywheel term and an assumed step time and reach: it ranks states, it
is not ground truth. The deployed risk output is a separate learned estimate, never this gate.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

GRAVITY = 9.81
SOLE_HALF = (0.09, 0.035)                       # half length / half width of a sole (m)
# Where the swing foot can land, stance-foot frame, y outward. The middle box of the review (0.30 forward, 0.25
# back, 0.40 to the side); the largest step the robot could take (0.45 / 0.35 / 0.50) reads half of the 12 N s
# front hits on a standing robot as "one step suffices". To be calibrated with the hold on frozen S1-a.
REACH_X, REACH_Y = (-0.25, 0.30), (0.10, 0.40)
STEP_TIME, DEAD_BAND, SCALE = 0.25, 0.05, 0.15
MARGIN_DEAD_BAND, MARGIN_SCALE = 0.03, 0.10      # d0, d1 of the primary signal (to be calibrated on calm rollouts)
_DIRECTIONS = 64      # fan of test directions; with the soles' own edge normals the hull distance is within 1 mm


class Centroidal:
  """Whole-body CoM state, capture point and one-step capture margin for every environment."""

  def __init__(self, env: ManagerBasedRlEnv, contact_sensor: str = "feet_ground",
               foot_bodies: tuple[str, str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
               footprint: tuple[tuple[float, float], tuple[float, float]] | None = None):
    """footprint: ((x_min, x_max), (y_min, y_max)) of the sole in the foot body's frame, to use instead of the
    constants above. Measured from the model's seven foot capsules 11-13 14:33: x -0.050 .. +0.133, y +-0.035, so
    the default (centre = mean capsule position +0.051, half length 0.09) puts the support 1 cm too far forward.
    Evaluation keeps the default so that its columns stay comparable with earlier records."""
    self.env, self.sensor = env, contact_sensor
    self.entity = env.scene["robot"]
    device = env.device
    self.body_ids = torch.as_tensor(self.entity.indexing.body_ids, device=device, dtype=torch.long)
    self.feet = [self.entity.body_names.index(name) for name in foot_bodies]
    model = env.sim.mj_model
    # Sole centre in each foot body's frame: mean position of its collision capsules.
    centres = []
    for name in foot_bodies:
      body = model.body("robot/" + name).id if _has(model, "robot/" + name) else model.body(name).id
      geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body and model.geom(g).name.endswith("_collision")]
      centres.append(torch.tensor(model.geom_pos[geoms].mean(0), dtype=torch.float, device=device))
    self.sole_offset = torch.stack(centres)                                   # (2, 3)
    angles = torch.arange(_DIRECTIONS, device=device) * (2 * math.pi / _DIRECTIONS)
    self.directions = torch.stack([angles.cos(), angles.sin()], dim=1)          # (K, 2)
    half = torch.tensor(SOLE_HALF, device=device)
    if footprint is not None:
      (x0, x1), (y0, y1) = footprint
      self.sole_offset[:, 0], self.sole_offset[:, 1] = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
      half = torch.tensor((0.5 * (x1 - x0), 0.5 * (y1 - y0)), device=device)
    signs = torch.tensor([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=torch.float, device=device)
    self.corners = signs * half                                                 # (4, 2) in the sole frame
    self.half = half

  def masses(self) -> torch.Tensor:
    mass = self.env.sim.model.body_mass
    return mass[self.body_ids].unsqueeze(0) if mass.ndim == 1 else mass[:, self.body_ids]

  def state(self) -> dict[str, torch.Tensor]:
    data = self.entity.data
    mass = self.masses()
    total = mass.sum(dim=1, keepdim=True)
    com = (data.body_com_pos_w * mass.unsqueeze(-1)).sum(1) / total
    velocity = (data.body_com_lin_vel_w * mass.unsqueeze(-1)).sum(1) / total
    angular = self.env.scene["robot/root_angmom"].data
    quat = data.body_link_quat_w[:, self.feet]                                   # (N, 2, 4) w x y z
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))            # (N, 2)
    cos, sin = yaw.cos(), yaw.sin()
    position = data.body_link_pos_w[:, self.feet]
    offset = self.sole_offset.unsqueeze(0)
    sole = position[..., :2] + torch.stack([cos * offset[..., 0] - sin * offset[..., 1],
                                            sin * offset[..., 0] + cos * offset[..., 1]], dim=-1)
    contact = self.env.scene[self.sensor].data.found.reshape(self.env.num_envs, -1)[:, :2] > 0
    ground = self.env.scene.env_origins[:, 2]
    return dict(com=com, velocity=velocity, angular=angular, total_mass=total.squeeze(1), sole=sole, yaw=yaw,
                contact=contact, height=(com[:, 2] - ground).clamp_min(0.2))

  def support_margin(self, reference_velocity: torch.Tensor | None = None, state: dict | None = None) -> dict[str, torch.Tensor]:
    """Velocity-error capture point against the hull of both soles (loaded or not). reference_velocity: (N, 2) world.
    `state`: a result of state() to reuse."""
    s = self.state() if state is None else state
    return self._support(s, self._capture_point(s, reference_velocity))

  def _capture_point(self, s: dict, reference_velocity: torch.Tensor | None) -> torch.Tensor:
    w0 = torch.sqrt(GRAVITY / s["height"])
    velocity = s["velocity"][:, :2] if reference_velocity is None else s["velocity"][:, :2] - reference_velocity
    flywheel = torch.stack([s["angular"][:, 1], -s["angular"][:, 0]], dim=1) / (s["total_mass"] * s["height"] * w0).unsqueeze(1)
    return s["com"][:, :2] + velocity / w0.unsqueeze(1) + flywheel

  def _support(self, s: dict, xi: torch.Tensor) -> dict[str, torch.Tensor]:
    cos, sin = s["yaw"].cos(), s["yaw"].sin()
    cx, cy = self.corners[:, 0], self.corners[:, 1]
    corners = torch.stack([s["sole"][..., 0:1] + cos.unsqueeze(-1) * cx - sin.unsqueeze(-1) * cy,
                           s["sole"][..., 1:2] + sin.unsqueeze(-1) * cx + cos.unsqueeze(-1) * cy], dim=-1)
    axes = torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([-sin, cos], dim=-1)], dim=2)
    edges = torch.cat([axes, -axes], dim=2).reshape(xi.shape[0], 8, 2)
    directions = torch.cat([self.directions.unsqueeze(0).expand(xi.shape[0], -1, -1), edges], dim=1)
    reach = torch.einsum("nkd,nfcd->nfck", directions, corners).amax(dim=2).amax(dim=1)      # both feet, always
    d = (torch.einsum("nkd,nd->nk", directions, xi) - reach).amax(dim=1)
    urgency = ((d - MARGIN_DEAD_BAND) / MARGIN_SCALE).clamp(0.0, 1.0)
    return dict(d=d, urgency=urgency, xi=xi, **s)

  def urgency_margin(self, reference_velocity: torch.Tensor | None = None, state: dict | None = None) -> dict[str, torch.Tensor]:
    """The adopted gate: the one-step margin, counted only where the capture point is outside both soles.

    reference_velocity is what "no divergence" means. Use the robot's own planar CoM velocity averaged over about a
    second (what the validation on recordings used), not the commanded velocity: with the command, a policy that
    does not follow a 0.2 m/s goal reads as 5 cm of divergence although its balance is not in question."""
    s = self.state() if state is None else state
    xi = self._capture_point(s, reference_velocity)
    outside = self._support(s, xi)["d"]
    step = self._one_step(s, xi)
    d = torch.where(outside > 0.0, step["d1"], torch.zeros_like(outside))
    return dict(d=d, urgency=((d - DEAD_BAND) / SCALE).clamp(0.0, 1.0), support_d=outside, d0=step["d0"], d1=step["d1"],
                xi=xi, **s)

  def capture_margin(self, reference_velocity: torch.Tensor | None = None, state: dict | None = None) -> dict[str, torch.Tensor]:
    """reference_velocity: planar reference velocity in the world frame, (N, 2); None = zero."""
    s = self.state() if state is None else state
    return dict(**self._one_step(s, self._capture_point(s, reference_velocity)), **s)

  def _one_step(self, s: dict, xi: torch.Tensor) -> dict[str, torch.Tensor]:
    w0 = torch.sqrt(GRAVITY / s["height"])
    # With no foot loaded (a hop, one frame of contact chatter) both soles' projections stand in for the stance:
    # a constant "airborne" value would saturate the signal for a frame and, with a hold, pay for hopping.
    stance = s["contact"] | ~s["contact"].any(dim=1, keepdim=True)
    cos, sin = s["yaw"].cos(), s["yaw"].sin()                                    # (N, 2)
    # Sole corners in the world, (N, 2 feet, 4, 2).
    cx, cy = self.corners[:, 0], self.corners[:, 1]
    corners = torch.stack([s["sole"][..., 0:1] + cos.unsqueeze(-1) * cx - sin.unsqueeze(-1) * cy,
                           s["sole"][..., 1:2] + sin.unsqueeze(-1) * cx + cos.unsqueeze(-1) * cy], dim=-1)
    # Support function: xi is inside the hull iff n.xi <= max n.corner for every direction n. Directions are a
    # fixed fan plus the edge normals of both soles (exact for single support, within about 2% for double).
    axes = torch.stack([torch.stack([cos, sin], dim=-1), torch.stack([-sin, cos], dim=-1)], dim=2)   # (N, 2, 2, 2)
    edges = torch.cat([axes, -axes], dim=2).reshape(xi.shape[0], 8, 2)
    directions = torch.cat([self.directions.unsqueeze(0).expand(xi.shape[0], -1, -1), edges], dim=1)  # (N, K + 8, 2)
    reach = torch.einsum("nkd,nfcd->nfck", directions, corners).amax(dim=2)       # (N, 2, K + 8)
    reach = reach.masked_fill(~stance.unsqueeze(-1), -1e6).amax(dim=1)             # over the stance feet
    d0 = (torch.einsum("nkd,nd->nk", directions, xi) - reach).amax(dim=1)
    # One-step capture, each foot in contact taken as the stance foot.
    delta = xi.unsqueeze(1) - s["sole"]                                           # (N, 2, 2)
    local = torch.stack([cos * delta[..., 0] + sin * delta[..., 1], -sin * delta[..., 0] + cos * delta[..., 1]], dim=-1)
    nearest = torch.maximum(torch.minimum(local, self.half), -self.half)
    ahead = nearest + (local - nearest) * torch.exp(w0 * STEP_TIME).view(-1, 1, 1)
    outward = torch.tensor([-1.0, 1.0], device=xi.device)                         # left stance: right foot lands at -y
    x, y = ahead[..., 0], ahead[..., 1] * outward
    zero = torch.zeros_like(x)
    dx = torch.maximum(torch.maximum(REACH_X[0] - self.half[0] - x, x - REACH_X[1] - self.half[0]), zero)
    dy = torch.maximum(torch.maximum(REACH_Y[0] - self.half[1] - y, y - REACH_Y[1] - self.half[1]), zero)
    option = torch.hypot(dx, dy).masked_fill(~stance, 1e6).amin(dim=1)
    d1 = torch.where(d0 > 0, option, torch.zeros_like(option))
    urgency = ((d1 - DEAD_BAND) / SCALE).clamp(0.0, 1.0)
    return dict(d0=d0, d1=d1, urgency=urgency, xi=xi)


def _has(model, name: str) -> bool:
  try:
    model.body(name)
    return True
  except KeyError:
    return False
