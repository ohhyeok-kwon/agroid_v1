"""Bounded, task-relative recovery cost. Privileged reward diagnostics, never an action constraint.

The gravitational time sqrt(CoM height / g) supplies a short prediction scale. Constant angular velocities predict
the actual and requested trunk gravity vectors. Their tilt error prices present and impending sway, while retaining
all 24 actions. This is a local posture-risk proxy, not a contact-feasibility or minimal-reaction oracle.
"""
from __future__ import annotations

import math

import torch

GRAVITY = 9.81
TILT_SCALE = 0.35                 # rad; fixed before the first contrast, not fitted to manufacturer trajectories
HEIGHT_RANGE = (0.15, 1.5)        # m; the reduced model's time scale, not a desired robot height
RISK_WEIGHT = 2.0                 # reward/s, at most 0.04/step at dt=.02, discounted sum <4 at gamma=.99


def inverse_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
  """World vector in a unit quaternion's local frame, wxyz; differentiable and sign invariant."""
  q = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
  cross = 2.0 * torch.cross(q[..., 1:], vector, dim=-1)
  return vector - q[..., :1] * cross + torch.cross(q[..., 1:], cross, dim=-1)


def predict_gravity(gravity: torch.Tensor, angular_velocity_b: torch.Tensor,
                    seconds: torch.Tensor) -> torch.Tensor:
  """Rotate the gravity direction forward at constant body angular velocity (Rodrigues, stable at zero rate)."""
  rotation = angular_velocity_b * seconds.unsqueeze(-1)
  angle = rotation.norm(dim=-1, keepdim=True)
  first = torch.cross(rotation, gravity, dim=-1)
  return (gravity - torch.sinc(angle / math.pi) * first
          + 0.5 * torch.sinc(angle / (2.0 * math.pi)).square() * torch.cross(rotation, first, dim=-1))


def gravity_risk(actual_quat: torch.Tensor, actual_angular_w: torch.Tensor, desired_quat: torch.Tensor,
                 desired_angular_w: torch.Tensor, height: torch.Tensor, tilt_scale: float = TILT_SCALE,
                 desired_held: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
  """Pure state cost: no hit clock, hit count, action clamping, arm rules, RNG or temporal cache.

  Holding a reference means zero requested angular velocity, not the original clip's stored derivative. Finite but
  nonpositive/invalid quaternion inputs receive maximal cost and invalid diagnostics; nonfinite physics returns NaN
  for the required ledger to preserve before RewardManager sanitization. Ledger limits remain explicit; a NaN is
  invalid evidence, never quiet success. This term does not independently stop a simulator with corrupt state.
  """
  if not math.isfinite(tilt_scale) or tilt_scale <= 0.0:
    raise ValueError("tilt_scale must be finite and positive")
  if desired_held is not None:
    desired_angular_w = torch.where(desired_held.unsqueeze(-1), torch.zeros_like(desired_angular_w), desired_angular_w)
  finite = (torch.isfinite(actual_quat).all(-1) & torch.isfinite(desired_quat).all(-1)
            & torch.isfinite(actual_angular_w).all(-1) & torch.isfinite(desired_angular_w).all(-1)
            & torch.isfinite(height))
  valid = finite & (height > 0.0) & (actual_quat.norm(dim=-1) > 1.0e-8) & (desired_quat.norm(dim=-1) > 1.0e-8)
  gravity = torch.zeros_like(actual_angular_w)
  gravity[..., 2] = -1.0
  actual = inverse_rotate(actual_quat, gravity)
  desired = inverse_rotate(desired_quat, gravity)
  seconds = torch.sqrt(height.clamp(*HEIGHT_RANGE) / GRAVITY)
  future_actual = predict_gravity(actual, inverse_rotate(actual_quat, actual_angular_w), seconds)
  future_desired = predict_gravity(desired, inverse_rotate(desired_quat, desired_angular_w), seconds)
  # 2 atan2(sin(theta/2), cos(theta/2)) avoids acos loss of precision near the desired posture.
  tilt = 2.0 * torch.atan2((actual - desired).norm(dim=-1), (actual + desired).norm(dim=-1))
  predicted_tilt = 2.0 * torch.atan2((future_actual - future_desired).norm(dim=-1),
                                   (future_actual + future_desired).norm(dim=-1))
  risk = -torch.expm1(-torch.maximum(tilt, predicted_tilt).square() / tilt_scale**2)
  risk = torch.where(valid, risk, torch.ones_like(risk))
  risk = torch.where(finite, risk, torch.full_like(risk, float("nan")))
  return dict(risk=risk, valid=valid, tilt=tilt, predicted_tilt=predicted_tilt, seconds=seconds, height=height)


def features(env, command_name: str = "motion", tilt_scale: float = TILT_SCALE) -> dict[str, torch.Tensor]:
  """Torso features at reward time; a ray's position and clearance always come from the same measurement.

  Ray locations can be one control interval old. This adapter does not pretend that a delayed ray is a fresh terrain
  query. Missing rays are explicitly invalid and receive maximal cost, not a zero-cost fallback.
  """
  term = env.command_manager.get_term(command_name)
  entity = env.scene["robot"]
  if not hasattr(term, "_recovery_body_ids"):
    term._recovery_body_ids = torch.as_tensor(entity.indexing.body_ids, device=env.device, dtype=torch.long)
    term._recovery_torso_index = list(term.cfg.body_names).index("torso_link")
  mass = env.sim.model.body_mass
  mass = mass[term._recovery_body_ids].unsqueeze(0) if mass.ndim == 1 else mass[:, term._recovery_body_ids]
  com_z = (entity.data.body_com_pos_w[..., 2] * mass).sum(-1) / mass.sum(-1)
  if "root_ground" in env.scene.sensors:
    ray = env.scene["root_ground"].data
    ground_z = ray.frame_pos_w[:, 0, 2] - ray.heights.reshape(env.num_envs, -1)[:, 0]
    distances = ray.distances.reshape(env.num_envs, -1)
    ground_valid = (torch.isfinite(distances) & (distances >= 0.0)).all(-1) & torch.isfinite(ground_z)
  else:
    ground_z = env.scene.env_origins[:, 2]
    ground_valid = torch.ones_like(ground_z, dtype=torch.bool)
  torso = term._recovery_torso_index
  result = gravity_risk(term.robot_body_quat_w[:, torso], term.robot_body_ang_vel_w[:, torso],
                        term.body_quat_w[:, torso], term.body_ang_vel_w[:, torso], com_z - ground_z,
                        tilt_scale=tilt_scale, desired_held=getattr(term, "hold", None))
  result["ground_valid"] = ground_valid
  result["valid"] = result["valid"] & ground_valid
  result["risk"] = torch.where(ground_valid | ~torch.isfinite(result["risk"]), result["risk"],
                               torch.ones_like(result["risk"]))
  return result


def recovery_risk(env, command_name: str = "motion", tilt_scale: float = TILT_SCALE) -> torch.Tensor:
  """The reward manager integrates this [0,1] cost; all motions stay available throughout recovery."""
  return features(env, command_name, tilt_scale)["risk"]


def recovery_invalid(env, command_name: str = "motion") -> torch.Tensor:
  """Pure diagnostic, registered at weight zero and explicitly sampled by the required ledger in both arms."""
  return (~features(env, command_name)["valid"]).float()


def _pose_cost(state: dict[str, torch.Tensor], tilt_scale: float = TILT_SCALE) -> torch.Tensor:
  """Current-tilt ablation with the predictive cost's same scale, bound and invalid handling."""
  pose = -torch.expm1(-state["tilt"].square() / tilt_scale**2)
  pose = torch.where(state["valid"], pose, torch.ones_like(pose))
  return torch.where(torch.isfinite(state["risk"]), pose, state["risk"])


def recovery_pose_cost(env, command_name: str = "motion", tilt_scale: float = TILT_SCALE) -> torch.Tensor:
  """Evaluation of an extra posture price; no claim of predictive stabilization from this ablation."""
  return _pose_cost(features(env, command_name, tilt_scale), tilt_scale)


DIAGNOSTIC_COLUMNS = ("tilt", "predicted_tilt", "valid", "risk", "pose_cost", "predictive_increment",
                      "height", "seconds")


def diagnostic_values(env, command_name: str = "motion") -> torch.Tensor:
  """One read-only post-reward/pre-reset vector; separates posture cost from the prediction increment."""
  state = features(env, command_name)
  pose = _pose_cost(state)
  valid = state["valid"] & torch.isfinite(state["risk"]) & torch.isfinite(state["predicted_tilt"])
  return torch.stack((state["tilt"], state["predicted_tilt"], valid.float(), state["risk"], pose,
                      state["risk"] - pose, state["height"], state["seconds"]), dim=-1)
