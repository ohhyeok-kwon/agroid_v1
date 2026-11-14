"""Reference teacher primitives; reference state writes occur only on reset."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mjlab.tasks.tracking.mdp.commands import MotionCommand, MotionCommandCfg
from r1lab import contract as C

REFERENCE_FPS = 50
FUTURE_FRAMES = (5, 10)
JOINT_VELOCITY_SCALE = 0.05
BODY_NAMES = (
    "pelvis", "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link",
    "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
    "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link",
    "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    "waist_roll_link", "torso_link", "left_shoulder_pitch_link",
    "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link",
    "left_wrist_roll_link", "right_shoulder_pitch_link", "right_shoulder_roll_link",
    "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link",
)
ORIENTATION_BODIES = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")


def validate_reference_file(path, *, body_names=BODY_NAMES, joint_names=C.ENTITY_JOINTS):
    """Validate native MotionLoader order, units and finite shapes before loading."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        required = {"fps", "joint_names", "body_names", "joint_pos", "joint_vel",
                    "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"}
        if missing := required.difference(data.files):
            raise ValueError(f"Reference {path} is missing {sorted(missing)}")
        if data["fps"].size != 1 or float(data["fps"].item()) != REFERENCE_FPS:
            raise ValueError("Reference must be native 50 Hz; no implicit resampling")
        if tuple(data["joint_names"].tolist()) != tuple(joint_names):
            raise ValueError("Reference joint order must equal the complete R1 ENTITY_JOINTS")
        if tuple(data["body_names"].tolist()) != tuple(body_names):
            raise ValueError("Reference body order must equal complete R1 entity bodies (no world)")
        frames = len(data["joint_pos"])
        if frames < 26:
            raise ValueError("Reference must contain at least 26 consecutive frames")
        shapes = {"joint_pos": (frames, 24), "joint_vel": (frames, 24),
                  "body_pos_w": (frames, len(body_names), 3),
                  "body_quat_w": (frames, len(body_names), 4),
                  "body_lin_vel_w": (frames, len(body_names), 3),
                  "body_ang_vel_w": (frames, len(body_names), 3)}
        for key, shape in shapes.items():
            value = data[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"Invalid {key}: expected finite {shape}, got {value.shape}")
        norms = np.linalg.norm(data["body_quat_w"], axis=-1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("Reference quaternions must be normalized wxyz")
    return {"path": str(path), "frames": frames, "fps": REFERENCE_FPS,
            "duration_s": (frames - 1) / REFERENCE_FPS}


@dataclass(kw_only=True)
class TeacherMotionCommandCfg(MotionCommandCfg):
    """Uniform reset phases with sufficient remaining motion; no adaptive bins.

    start_frame_mixture: ((frame, probability), ...) reset frames drawn before
    the uniform remainder, e.g. a real standing start and a real slow-down.
    """
    minimum_remaining_frames: int = 25
    start_frame_mixture: tuple[tuple[int, float], ...] = ()

    def build(self, env):
        return TeacherMotionCommand(self, env)


class TeacherMotionCommand(MotionCommand):
    cfg: TeacherMotionCommandCfg

    def __init__(self, cfg, env):
        validate_reference_file(cfg.motion_file)
        if abs(float(env.step_dt) - 1.0 / REFERENCE_FPS) > 1e-9:
            raise ValueError("Teacher policy step must equal reference step: 0.02 seconds")
        if tuple(env.scene[cfg.entity_name].body_names) != BODY_NAMES:
            raise ValueError("Teacher asset body order differs from reference contract")
        super().__init__(cfg, env)
        limits = self.robot.data.soft_joint_pos_limits[0]
        outside = ((self.motion.joint_pos < limits[:, 0] - 1e-6)
                   | (self.motion.joint_pos > limits[:, 1] + 1e-6)).any()
        if bool(outside):
            raise ValueError("Reference exceeds R1 soft joint limits; refusing silent target clipping")
        last_start = max(0, self.motion.time_step_total - 1 - self.cfg.minimum_remaining_frames)
        mixture = tuple((int(frame), float(probability))
                        for frame, probability in self.cfg.start_frame_mixture)
        if any(not 0 <= frame <= last_start or not 0 < probability < 1
               for frame, probability in mixture) or sum(p for _, p in mixture) >= 1:
            raise ValueError("start_frame_mixture needs frames with remaining motion and total probability < 1")
        self._start_mixture = mixture
        # Realized reset counts per mode (0 = uniform, i = mixture entry i).
        self.start_mode_counts = torch.zeros(len(mixture) + 1, device=self.device)
        for index in range(len(mixture)):
            self.metrics[f"start_fixed_{index}"] = torch.zeros(self.num_envs, device=self.device)
            self.metrics[f"start_fixed_{index}_realized_fraction"] = torch.zeros(
                self.num_envs, device=self.device)

    def _uniform_sampling(self, env_ids):
        last_start = max(0, self.motion.time_step_total - 1 - self.cfg.minimum_remaining_frames)
        frames = torch.randint(last_start + 1, (len(env_ids),), device=self.device)
        modes = torch.zeros(len(env_ids), dtype=torch.long, device=self.device)
        mixture = getattr(self, "_start_mixture", ())
        if mixture:
            draw = torch.rand(len(env_ids), device=self.device)
            lower = 0.0
            for index, (frame, probability) in enumerate(mixture, start=1):
                chosen = (draw >= lower) & (draw < lower + probability)
                frames[chosen] = frame
                modes[chosen] = index
                lower += probability
            self.start_mode_counts += torch.bincount(modes, minlength=len(mixture) + 1)
            total = self.start_mode_counts.sum().clamp(min=1)
            for index in range(len(mixture)):
                self.metrics[f"start_fixed_{index}"][env_ids] = (modes == index + 1).float()
                self.metrics[f"start_fixed_{index}_realized_fraction"][:] = (
                    self.start_mode_counts[index + 1] / total)
        self.time_steps[env_ids] = frames
        self.metrics["sampling_entropy"][:] = 1.0
        self.metrics["sampling_top1_prob"][:] = 1.0 / (last_start + 1)
        self.metrics["sampling_top1_bin"][:] = 0.5

    def compute(self, dt, env_ids=None):
        # The native compute calls _update_command even at dt=0. A reset must
        # remain on its selected frame; mixed auto-reset tensors preserve that.
        self._update_metrics()
        if isinstance(dt, torch.Tensor):
            if env_ids is not None:
                raise ValueError("Per-env dt cannot also specify env_ids")
            advance_ids = (dt > 0).nonzero(as_tuple=False).flatten()
        elif float(dt) > 0:
            advance_ids = env_ids
        else:
            advance_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._update_command(advance_ids)

    def _update_command(self, env_ids=None):
        # Never call _resample_command from the step path: it writes qpos/qvel.
        if env_ids is None:
            self.time_steps.add_(1).clamp_(max=self.motion.time_step_total - 1)
        else:
            self.time_steps[env_ids] = (self.time_steps[env_ids] + 1).clamp(
                max=self.motion.time_step_total - 1)
        self.update_relative_body_poses()

    def reset_to_frame(self, env_ids, frame):
        frame = int(frame)
        if not 0 <= frame < self.motion.time_step_total:
            raise ValueError(f"Reference frame {frame} is out of range")
        super().reset_to_frame(env_ids, frame)
        self._pending_forward = False


def current_reference(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return torch.cat((command.joint_pos, command.joint_vel * JOINT_VELOCITY_SCALE), dim=-1)


def phase_observation(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    phase = 2.0 * torch.pi * command.time_steps / max(1, command.motion.time_step_total - 1)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=-1)


def zero_phase_observation(env, command_name="motion"):
    """Phase-free teachers keep the two inputs for width but receive no clip timer."""
    command = env.command_manager.get_term(command_name)
    return torch.zeros(command.time_steps.shape[0], 2, device=command.time_steps.device)


def future_reference(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    rows = []
    for offset in FUTURE_FRAMES:
        frame = (command.time_steps + offset).clamp(max=command.motion.time_step_total - 1)
        rows.extend((command.motion.joint_pos[frame],
                     command.motion.joint_vel[frame] * JOINT_VELOCITY_SCALE))
    return torch.cat(rows, dim=-1)


def reference_end(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return command.time_steps >= command.motion.time_step_total - 1


def stance_slip_speed(env, threshold_n=5.0, site_names=("left_foot", "right_foot")):
    """Sum over loaded feet of the sole-site horizontal speed (m/s); same definition as eval_gait slip."""
    robot = env.scene["robot"]
    site_ids = robot.find_sites(site_names, preserve_order=True)[0]
    speed = torch.linalg.vector_norm(robot.data.site_lin_vel_w[:, site_ids, :2], dim=-1)
    contact = torch.linalg.vector_norm(env.scene["feet_contact"].data.force, dim=-1) > threshold_n
    return (speed * contact).sum(-1)


def sway_excess(env, command_name="motion", body_names=("pelvis", "torso_link"), deadband=0.05, calm_only=True):
    """Horizontal-axis (roll/pitch) angular speed of pelvis and torso above the reference's own rate (L1, rad/s).

    Reference-relative so captured motions that do sway (human side steps: pelvis ~.44 rad/s) are not charged;
    only the robot's extra tilt rate beyond the human frame it tracks plus a dead band. Human careful walk
    (library v5, .12-.32 m/s): pelvis ~.16-.17 rad/s; I10 .18-.24, D1 student .19-.35, manufacturer .19-.21
    (eval_gait). calm_only: no charge within 1.5 s after a marked disturbance (pushes, ball contacts), as
    landing_speed, so recoveries may rotate the body as much as they need. Logs Metrics/sway_excess.
    """
    command = env.command_manager.get_term(command_name)
    idx = [command.cfg.body_names.index(name) for name in body_names]
    robot = torch.linalg.vector_norm(command.robot_body_ang_vel_w[:, idx, :2], dim=-1)
    reference = torch.linalg.vector_norm(command.body_ang_vel_w[:, idx, :2], dim=-1)
    value = (robot - reference - deadband).clamp_min(0.0).sum(-1)
    log = env.extras.setdefault("log", {})
    log["Metrics/sway_excess"] = value.mean()
    if calm_only:
        from r1lab.mdp.recovery import _gate_state
        projectiles = getattr(env, "_r1_projectile_event", None)
        if projectiles is not None:
            projectiles.observe_contacts()
        value = value * (env.common_step_counter > _gate_state(env)["external_until"]).float()
    return value


def anchor_height_tracking(env, command_name="motion", std=0.02, calm_only=True):
    """exp(-(anchor z - reference anchor z)^2 / std^2): stand and walk at the captured (R1-retargeted) height.

    The matcher keeps the library frame's height (it only eases the root in xy), so the reference height is the
    human's. I11 stood at pelvis .706 m with knees .69 rad while its StandIdle reference is .734 m / .35 rad
    (crouched 2.8 cm; the 3D root kernel std .3 m barely charged it). calm_only: zero within 1.5 s after a marked
    disturbance, so recoveries may lower the body. Logs Metrics/anchor_height_error_m.
    """
    command = env.command_manager.get_term(command_name)
    error = command.robot_anchor_pos_w[:, 2] - command.anchor_pos_w[:, 2]
    value = torch.exp(-error.square() / std ** 2)
    log = env.extras.setdefault("log", {})
    log["Metrics/anchor_height_error_m"] = error.abs().mean()
    if calm_only:
        from r1lab.mdp.recovery import _gate_state
        projectiles = getattr(env, "_r1_projectile_event", None)
        if projectiles is not None:
            projectiles.observe_contacts()
        value = value * (env.common_step_counter > _gate_state(env)["external_until"]).float()
    return value


class landing_speed:
    """Downward sole speed above a human deadband while an airborne foot is near the ground (L1).

    Human careful walk (00230 in library v3, retargeted): landing vz mean .089, p95 .152 m/s; the push
    teachers land at ~.23 (training physics, last policy step) / .30 m/s (Simulation, 1 kHz). Near:
    airborne sole site below near_height above the z=0 ground plane (flat-ground teachers only).
    The previous step's excess is also charged for feet that landed within this step, so a foot
    dropped from above near_height inside one policy step is not free.
    calm_only: no charge within 1.5 s after a marked disturbance (tracked pushes, ball contacts;
    r1lab.mdp.recovery.mark_disturbance), so recovery steps may land as hard as they need to.
    Logs the downward speed at the last step before each touchdown as Metrics/landing_speed.
    """

    def __init__(self, cfg, env):
        self.prev_vz = torch.zeros(env.num_envs, 2, device=env.device)
        self.step_dt = env.step_dt

    def reset(self, env_ids=None):
        self.prev_vz[env_ids] = 0.0

    def __call__(self, env, deadband=0.1, near_height=0.04, threshold_n=5.0,
                 site_names=("left_foot", "right_foot"), calm_only=False):
        robot = env.scene["robot"]
        site_ids = robot.find_sites(site_names, preserve_order=True)[0]
        vz = robot.data.site_lin_vel_w[:, site_ids, 2]
        height = robot.data.site_pos_w[:, site_ids, 2]
        sensor = env.scene["feet_contact"]
        contact = torch.linalg.vector_norm(sensor.data.force, dim=-1) > threshold_n
        near = (~contact & (height < near_height)).float()
        approach = (near * ((-vz) - deadband).clamp_min(0.0)).sum(-1)
        first = sensor.compute_first_contact(dt=self.step_dt).float()
        landing = (-self.prev_vz).clamp_min(0.0)
        touch = (first * (landing - deadband).clamp_min(0.0)).sum(-1)
        log = env.extras.setdefault("log", {})
        log["Metrics/landing_speed"] = (first * landing).sum() / first.sum().clamp_min(1.0)
        self.prev_vz = vz.clone()
        value = approach + touch
        if calm_only:
            from r1lab.mdp.recovery import _gate_state
            projectiles = getattr(env, "_r1_projectile_event", None)
            if projectiles is not None:
                projectiles.observe_contacts()
            calm = env.common_step_counter > _gate_state(env)["external_until"]
            log["Metrics/landing_calm_fraction"] = calm.float().mean()
            value = value * calm.float()
        return value


def anchor_xy_deviation(env, command_name="motion", threshold=0.5):
    """Planar pelvis-to-reference distance beyond threshold metres (positional tracking failure)."""
    command = env.command_manager.get_term(command_name)
    error = command.anchor_pos_w[:, :2] - command.robot_anchor_pos_w[:, :2]
    return torch.linalg.vector_norm(error, dim=-1) > threshold


def illegal_contact(env, sensor_name="illegal_contact", threshold=1.0):
    sensor = env.scene[sensor_name].data
    force = sensor.force_history if sensor.force_history is not None else sensor.force
    return (torch.linalg.vector_norm(force, dim=-1) > threshold).flatten(1).any(dim=-1)


def foot_force(env, foot=0, normalized=False):
    """Measured robot contact resultant in N or body-weight units, not human GRF."""
    force = env.scene["feet_contact"].data.force[:, foot]
    magnitude = torch.linalg.vector_norm(force, dim=-1)
    if normalized:
        magnitude = magnitude / (float(env.sim.mj_model.body_mass.sum()) * 9.81)
    return magnitude


def contact_slip_speed(env):
    robot = env.scene["robot"]
    ids = robot.find_bodies(ORIENTATION_BODIES[2:], preserve_order=True)[0]
    speed = torch.linalg.vector_norm(robot.data.body_link_lin_vel_w[:, ids, :2], dim=-1)
    contact = torch.linalg.vector_norm(env.scene["feet_contact"].data.force, dim=-1) > 5.0
    return (speed * contact).sum(-1) / contact.sum(-1).clamp(min=1)


def joint_speed_max(env):
    return env.scene["robot"].data.joint_vel.abs().amax(-1)


def joint_position_rmse(env, command_name="motion"):
    command = env.command_manager.get_term(command_name)
    return ((command.robot.data.joint_pos - command.joint_pos).square().mean(-1)).sqrt()


def reset_teacher_to_frame(env, frame, env_ids=None):
    """Explicit evaluation initialization; call after env.reset, never per step."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env.action_manager.reset(env_ids)
    env.observation_manager.reset(env_ids)
    command = env.command_manager.get_term("motion")
    command.reset_to_frame(env_ids, frame)
    env.scene.write_data_to_sim()
    env.sim.forward()
    command.update_relative_body_poses()
    env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True, env_ids=env_ids)
    return env.obs_buf
