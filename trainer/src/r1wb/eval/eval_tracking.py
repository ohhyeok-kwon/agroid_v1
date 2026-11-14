"""P1 / P2 evaluation: replay every library clip once with a checkpoint (local GPU, mjlab physics).
`--task wb_track_teacher` (default) evaluates the tracking teacher, `--task wb_goal_student` the reference-free
student (it sees goals and arm intent only; the reference is used for scoring).

Deterministic pass (no pushes, no noise, from each clip's first frame): which clips the robot can physically
execute, tracking errors, torque margins, footfalls against the reference, and a physics-filtered record of the
executed motion (joint states, actions, torques, foot forces, CoM, angular momentum) for the later stages.
`--variant app` repeats it with 1 kHz physics and the XML's own contact model (the Simulation tab's settings) as a
physics-sensitivity check. `--variant hardened` keeps the randomised dynamics of the hardened scene (actuator
delay, PD gains, masses, friction; sensor noise for the student) but no pushes: the conditions the student trains
in. `--local-anchor` drives the teacher with the reference re-anchored to the robot (no planar or heading error
in its input, no planar or heading termination): the input it gets when it labels the student.
Every rollout is stored through the project's vector video archive.

  <work>/.venv/bin/python -m r1wb.eval.eval_tracking --run <run dir> --app-pid <owner pid>
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from mjlab.rl.runner import MjlabOnPolicyRunner  # noqa: E402
from mjlab.sensor import ContactMatch, ContactSensorCfg  # noqa: E402

from r1wb import robot  # noqa: E402
from r1wb.student import env_cfg as student_cfg  # noqa: E402
from r1wb.tasks import TASKS  # noqa: E402
from r1wb.tracking import env_cfg as tracking_cfg  # noqa: E402

STAND_HEIGHT = 0.731

from r1wb.eval.paths import AGENT as WORK, ROOT  # noqa: E402,F401


def latest_checkpoint(run: Path) -> Path:
  return max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run", type=Path, required=True)
  parser.add_argument("--checkpoint", type=Path, default=None)
  parser.add_argument("--task", choices=sorted(TASKS), default="wb_track_teacher")
  parser.add_argument("--variant", choices=["train", "app", "hardened"], default="train")
  parser.add_argument("--local-anchor", action="store_true", help="teacher: re-anchored reference input")
  parser.add_argument("--masks-off", action="store_true", help="student: height / pitch goals off where allowed")
  parser.add_argument("--label", action="store_true",
                      help="with --task wb_goal_student: the checkpoint is a teacher (tracking or task teacher) and its "
                           "label drives the robot in the student's scene, judged like a student (own twist against "
                           "the clip's goal): the teacher through the bridge the student learns from")
  parser.add_argument("--app-pid", type=int, required=True)
  parser.add_argument("--no-video", action="store_true", help="smoke runs only; real evaluations keep the archive")
  parser.add_argument("--limit", type=int, default=None, help="first N clips only (smoke run)")
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  os.environ["RL_EVALUATION_APP_PID"] = str(args.app_pid)
  from r1wb.eval.evaluation_video import require_app_owner
  require_app_owner(bind=True)
  run = args.run.resolve()
  checkpoint = (args.checkpoint or latest_checkpoint(run)).resolve()
  iteration = int(checkpoint.stem.split("_")[1])
  is_student = args.task == "wb_goal_student"
  hardened = args.variant == "hardened"
  if args.label and not is_student:
    raise SystemExit("--label needs --task wb_goal_student")
  if is_student:
    cfg = student_cfg.student_env_cfg(play=True, clean=not hardened, masks_off=args.masks_off,
                                      teacher=str(checkpoint) if args.label else None)
  else:
    cfg = tracking_cfg.tracking_env_cfg(play=True, hardened=hardened, clean=not hardened)
    if args.local_anchor:
      tracking_cfg.use_local_anchor(cfg, groups=("actor",))
      tracking_cfg.use_local_terminations(cfg)
  manifest = json.loads(Path(cfg.commands["motion"].motion_file).with_suffix(".manifest.json").read_text())
  clips = manifest["clips"][:args.limit] if args.limit else manifest["clips"]
  cfg.scene.num_envs = len(clips)
  if args.variant == "app":
    cfg.scene.entities["robot"] = replace(cfg.scene.entities["robot"], collisions=())
    cfg.sim.mujoco.timestep, cfg.decimation = 0.001, 20
    cfg.sim.nconmax, cfg.sim.njmax = 192, 1536
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  wrapped = RslRlVecEnvWrapper(env)
  if args.label:
    def policy(observation):
      return observation["bc"][:, :24]
  else:
    runner = MjlabOnPolicyRunner(wrapped, asdict(TASKS[args.task]["rl"]()), None, args.device)
    runner.load(str(checkpoint), map_location=args.device)
    policy = runner.get_inference_policy(device=args.device)
  term = env.command_manager.get_term("motion")
  entity = env.scene["robot"]
  count = len(clips)
  lengths = torch.tensor([c["length"] for c in clips], device=env.device)
  horizon = int(lengths.max()) + 2
  out_dir = WORK / "result/render" / run.name
  out_dir.mkdir(parents=True, exist_ok=True)
  tag = (f"clip_eval_{args.variant}" + ("_local" if args.local_anchor else "") + ("_masksoff" if args.masks_off else "")
         + ("_label" if args.label else "") + f"_{iteration}")
  archive = None
  if not args.no_video:
    from r1wb.eval.vector_evaluation_video import VectorVideoArchive, destination
    directory = destination(run / f"policy_{iteration}.onnx", out_dir / f"{tag}.json")
    archive = VectorVideoArchive(env, directory, [c["name"] for c in clips])
    archive.manifest.update(evaluator=__file__, evaluation_args=dict(run=str(run), checkpoint=str(checkpoint),
                                                                      variant=args.variant, onnx=str(run / f"policy_{iteration}.onnx")),
                            result_json=str(out_dir / f"{tag}.json"),
                            checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                            library_sha256=hashlib.sha256(Path(cfg.commands["motion"].motion_file).read_bytes()).hexdigest())
    archive.begin()
    if not archive.started:
      raise RuntimeError(archive.manifest.get("reason", "Cannot preserve evaluation"))
  effort = torch.tensor([robot.C.GROUPS[robot.C._group_of(n)]["effort"] for n in entity.joint_names], device=env.device)
  masses = env.sim.model.body_mass
  body_ids = torch.as_tensor(entity.indexing.body_ids, device=env.device, dtype=torch.long)
  mass = masses[body_ids] if masses.ndim == 1 else masses[:, body_ids]
  keys = dict(joint_pos=24, joint_vel=24, action=24, torque=24, root_pos=3, root_quat=4, root_lin_vel=3, root_ang_vel=3,
              com=3, angmom=3, foot_force=6, foot_contact=2, ref_joint_pos=24)
  student = hasattr(term, "goal")
  goal_sums = {k: torch.zeros(count, device=env.device) for k in ("twist", "height", "residual")}
  velocity_mean = None
  odometry = torch.zeros(count, 3, device=env.device)        # integrals of own twist and of the twist goal
  commanded = torch.zeros(count, 3, device=env.device)
  lag_peak = torch.zeros(count, 2, device=env.device)
  arm_sum = torch.zeros(count, device=env.device)             # realised |arm joints - arm intent|, student only
  unprompted = torch.zeros(count, 2, device=env.device)       # touchdowns while the reference foot stayed put
  liftoff_xy = torch.zeros(count, 2, 2, device=env.device)
  liftoff_yaw = torch.zeros(count, 2, device=env.device)
  reference_moved = torch.zeros(count, 2, dtype=torch.bool, device=env.device)
  record = {k: np.zeros((horizon, count, n), np.float32) for k, n in keys.items()}
  valid = np.zeros((horizon, count), bool)
  active = torch.ones(count, dtype=torch.bool, device=env.device)
  completed = torch.zeros(count, dtype=torch.bool, device=env.device)
  steps = torch.zeros(count, device=env.device)
  sums = {k: torch.zeros(count, device=env.device) for k in ("error_body_pos", "error_anchor_pos", "error_joint_pos",
                                                              "error_anchor_rot")}
  peaks = {k: torch.zeros(count, device=env.device) for k in ("error_body_pos", "torque_ratio", "self_contact")}
  touchdowns = torch.zeros(count, 2, device=env.device)
  ref_touchdowns = torch.zeros(count, 2, device=env.device)
  air = torch.zeros(count, 2, device=env.device)          # consecutive steps without contact (robot / reference)
  ref_air = torch.zeros(count, 2, device=env.device)
  planar = torch.full((horizon, count), float("nan"), device=env.device)
  vertical = torch.full((horizon, count), float("nan"), device=env.device)
  start_robot = entity.data.root_link_pos_w[:, :2].clone()
  start_reference = term.anchor_pos_w[:, :2].clone()
  end_robot, end_reference = start_robot.clone(), start_reference.clone()
  lowest_robot = torch.full((count,), 10.0, device=env.device)
  lowest_reference = torch.full((count,), 10.0, device=env.device)
  previous_contact = None
  foot_index = [list(robot.TRACKED_BODIES).index(n) for n in robot.FOOT_BODIES]
  previous_ref_low = None
  observation = wrapped.get_observations()
  started = time.time()
  for step in range(horizon):
    with torch.no_grad():
      action = policy(observation)
    data = entity.data
    feet = env.scene["feet_ground"].data
    contact = (feet.found.reshape(count, -1)[:, :2] > 0)
    if active.any():
      row = active.cpu().numpy()
      weights = mass if mass.ndim == 2 else mass.unsqueeze(0)
      com = (data.body_com_pos_w * weights.unsqueeze(-1)).sum(1) / weights.sum(1, keepdim=True)
      values = dict(joint_pos=data.joint_pos, joint_vel=data.joint_vel, action=action, torque=data.actuator_force,
                    root_pos=data.root_link_pos_w - env.scene.env_origins, root_quat=data.root_link_quat_w,
                    root_lin_vel=data.root_link_lin_vel_w, root_ang_vel=data.root_link_ang_vel_w, com=com - env.scene.env_origins,
                    angmom=env.scene["robot/root_angmom"].data, foot_force=feet.force.reshape(count, -1)[:, :6],
                    foot_contact=contact.float(), ref_joint_pos=term.joint_pos)
      for key, value in values.items():
        record[key][step, row] = value.detach().cpu().numpy()[row]
      valid[step] = row
      for key in sums:
        sums[key] += torch.where(active, term.metrics[key], torch.zeros_like(steps))
      peaks["error_body_pos"] = torch.where(active, torch.maximum(peaks["error_body_pos"], term.metrics["error_body_pos"]),
                                            peaks["error_body_pos"])
      ratio = (data.actuator_force.abs() / effort).amax(1)
      peaks["torque_ratio"] = torch.where(active, torch.maximum(peaks["torque_ratio"], ratio), peaks["torque_ratio"])
      hit = (env.scene["self_collision"].data.found.reshape(count, -1) > 0).any(1).float()
      peaks["self_contact"] += torch.where(active, hit, torch.zeros_like(hit))
      # Footfalls: a touchdown counts only after at least 3 steps (0.06 s) off the ground. Robot: contact sensor;
      # reference: foot body within 2 cm of the lower foot.
      ref_low = term.body_pos_w[:, foot_index, 2] < (term.body_pos_w[:, foot_index, 2].amin(1, keepdim=True) + 0.02)
      # A touchdown is "unprompted" if the reference foot neither left the ground nor moved (3 cm, 0.1 rad) while
      # the robot's foot was in the air: a step the motion did not ask for.
      ref_xy = term.body_pos_w[:, foot_index, :2]
      qw, qx, qy, qz = term.body_quat_w[:, foot_index].unbind(-1)
      ref_yaw = torch.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
      lifting = ~contact & (air == 0)
      liftoff_xy = torch.where(lifting.unsqueeze(-1), ref_xy, liftoff_xy)
      liftoff_yaw = torch.where(lifting, ref_yaw, liftoff_yaw)
      reference_moved = torch.where(lifting, ~ref_low, reference_moved | (~contact & ~ref_low))
      landing = active.unsqueeze(1) & contact & (air >= 3)
      turned = torch.atan2(torch.sin(ref_yaw - liftoff_yaw), torch.cos(ref_yaw - liftoff_yaw)).abs() > 0.1
      shifted = torch.norm(ref_xy - liftoff_xy, dim=-1) > 0.03
      unprompted += (landing & ~(reference_moved | turned | shifted)).float()
      touchdowns += landing.float()
      ref_touchdowns += (active.unsqueeze(1) & ref_low & (ref_air >= 3)).float()
      air = torch.where(contact, torch.zeros_like(air), air + 1)
      ref_air = torch.where(ref_low, torch.zeros_like(ref_air), ref_air + 1)
      previous_contact, previous_ref_low = contact, ref_low
      error = term.anchor_pos_w - term.robot_anchor_pos_w
      planar[step] = torch.where(active, torch.norm(error[:, :2], dim=1), planar[step])
      vertical[step] = torch.where(active, error[:, 2].abs(), vertical[step])
      end_robot = torch.where(active.unsqueeze(1), entity.data.root_link_pos_w[:, :2], end_robot)
      end_reference = torch.where(active.unsqueeze(1), term.anchor_pos_w[:, :2], end_reference)
      lowest_robot = torch.where(active, torch.minimum(lowest_robot, term.robot_anchor_pos_w[:, 2]), lowest_robot)
      lowest_reference = torch.where(active, torch.minimum(lowest_reference, term.anchor_pos_w[:, 2]), lowest_reference)
      if student:
        from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat
        goal_now = term.goal
        own = quat_apply_inverse(yaw_quat(data.root_link_quat_w), data.root_link_lin_vel_w)[:, :2]
        velocity_mean = own.clone() if velocity_mean is None else velocity_mean + (own - velocity_mean) * 0.05
        goal_sums["twist"] += torch.where(active, torch.norm(velocity_mean - goal_now[:, 0:2], dim=1), torch.zeros_like(steps))
        live = active.unsqueeze(1).float()
        odometry += term.own_twist() * env.step_dt * live
        commanded += term.twist_goal() * env.step_dt * live
        arm_sum += torch.where(active, (data.joint_pos[:, term.arm_ids] - term.arm_intent_now).abs().mean(1), torch.zeros_like(steps))
        lag_now = torch.stack([torch.norm(term.goal_lag[:, :2], dim=1), term.goal_lag[:, 2].abs()], dim=1)
        lag_peak = torch.where(active.unsqueeze(1), torch.maximum(lag_peak, lag_now), lag_peak)
        height_error = (data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2] - goal_now[:, 3]).abs()
        goal_sums["height"] += torch.where(active, height_error, torch.zeros_like(steps))
        goal_sums["residual"] += torch.where(active, env.action_manager.get_term("joint_pos").arm_residual.abs().mean(1),
                                             torch.zeros_like(steps))
      steps += active.float()
    observation, _, dones, _ = wrapped.step(action)
    if archive is not None:
      archive.after_step()
      if not archive.started:
        raise RuntimeError(archive.manifest.get("reason", "State preservation failed"))
    finished = active & dones.bool()
    completed |= finished & ~env.termination_manager.terminated
    active &= ~dones.bool()
    if not active.any():
      break
  rows = []
  planar_p95 = torch.nanquantile(planar, 0.95, dim=0)
  vertical_p95 = torch.nanquantile(vertical, 0.95, dim=0)
  travel = end_reference - start_reference
  distance = torch.norm(travel, dim=1)
  progress = ((end_robot - start_robot) * travel).sum(1) / distance.clamp_min(1e-6) ** 2
  origin_z = env.scene.env_origins[:, 2]
  for i, clip in enumerate(clips):
    n = max(float(steps[i]), 1.0)
    rows.append(dict(clip=clip["name"], context=clip["context"], arms=clip["arms"], seconds=round(clip["length"] / 50, 2),
                     completed=bool(completed[i]), survived_s=round(float(steps[i]) / 50, 2),
                     body_pos_error_mean_m=round(float(sums["error_body_pos"][i]) / n, 4),
                     body_pos_error_max_m=round(float(peaks["error_body_pos"][i]), 4),
                     anchor_pos_error_mean_m=round(float(sums["error_anchor_pos"][i]) / n, 4),
                     anchor_rot_error_mean_rad=round(float(sums["error_anchor_rot"][i]) / n, 4),
                     joint_pos_error_mean_rad=round(float(sums["error_joint_pos"][i]) / n / 24 ** 0.5, 4),
                     torque_ratio_max=round(float(peaks["torque_ratio"][i]), 3),
                     self_contact_frac=round(float(peaks["self_contact"][i]) / n, 4),
                     footfalls=int(touchdowns[i].sum()), reference_footfalls=int(ref_touchdowns[i].sum()),
                     unprompted_footfalls=int(unprompted[i].sum()),
                     planar_error_p95_m=round(float(planar_p95[i]), 4), height_error_p95_m=round(float(vertical_p95[i]), 4),
                     reference_travel_m=round(float(distance[i]), 3),
                     progress_ratio=round(float(progress[i]), 3) if float(distance[i]) > 0.5 else None,
                     lowest_anchor_m=round(float(lowest_robot[i] - origin_z[i]), 3),
                     reference_lowest_anchor_m=round(float(lowest_reference[i] - origin_z[i]), 3)))
    drop = STAND_HEIGHT - float(lowest_reference[i] - origin_z[i])
    rows[-1]["depth_ratio"] = round((STAND_HEIGHT - float(lowest_robot[i] - origin_z[i])) / drop, 3) if drop > 0.08 else None
    if student:
      asked_xy, asked_yaw = float(torch.norm(commanded[i, :2])), float(commanded[i, 2].abs())
      rows[-1].update(trained=bool(term.clip_trained[i if not args.limit else i]), arm_deviation_mean_rad=round(float(arm_sum[i]) / n, 4),
                      twist_error_mean_mps=round(float(goal_sums["twist"][i]) / n, 4),
                      height_error_mean_m=round(float(goal_sums["height"][i]) / n, 4),
                      arm_residual_mean_frac=round(float(goal_sums["residual"][i]) / n, 4),
                      distance_ratio=round(float((odometry[i, :2] * commanded[i, :2]).sum()) / asked_xy**2, 3)
                      if asked_xy > 0.5 else None,
                      yaw_ratio=round(float(odometry[i, 2] * commanded[i, 2]) / asked_yaw**2, 3) if asked_yaw > 0.5 else None,
                      commanded_distance_m=round(asked_xy, 3), commanded_yaw_rad=round(asked_yaw, 3),
                      goal_lag_peak_m=round(float(lag_peak[i, 0]), 3), goal_lag_peak_rad=round(float(lag_peak[i, 1]), 3))
  done = [r for r in rows if r["completed"]]
  summary = dict(clips=len(rows), completed=len(done), completed_frac=round(len(done) / len(rows), 3),
                 body_pos_error_mean_m_completed=round(float(np.mean([r["body_pos_error_mean_m"] for r in done])), 4) if done else None,
                 by_context={})
  if student or args.local_anchor:
    # No planar anchoring: judged on what the policy can observe (its own twist against the goal) and on height.
    def within(value):
      return value is None or 0.85 <= value <= 1.15
    strict = [r for r in done if r["height_error_p95_m"] <= 0.04
              and (within(r.get("distance_ratio")) and within(r.get("yaw_ratio")) if student
                   else r["progress_ratio"] is None or r["progress_ratio"] >= 0.85)]
    definition = ("completed, height error p95 <= 0.04 m, "
                  + ("distance and yaw ratio (own twist integral / goal integral) within 0.85-1.15" if student
                     else "progress >= 0.85 (no planar criterion: the reference is re-anchored)"))
  else:
    strict = [r for r in done if r["planar_error_p95_m"] <= 0.15 and r["height_error_p95_m"] <= 0.04
              and (r["progress_ratio"] is None or r["progress_ratio"] >= 0.9)]
    definition = "completed, planar anchor error p95 <= 0.15 m, height error p95 <= 0.04 m, progress >= 0.9"
  summary.update(passed_strict=len(strict), passed_strict_frac=round(len(strict) / len(rows), 3),
                 strict_definition=definition)
  for context in sorted({r["context"] for r in rows}):
    group = [r for r in rows if r["context"] == context]
    moving = [r["progress_ratio"] for r in group if r["progress_ratio"] is not None]
    summary["by_context"][context] = dict(
      clips=len(group), completed=sum(r["completed"] for r in group), passed_strict=sum(r in strict for r in group),
      body_pos_error_mean_m=round(float(np.mean([r["body_pos_error_mean_m"] for r in group])), 4),
      planar_error_p95_m=round(float(np.mean([r["planar_error_p95_m"] for r in group])), 4),
      height_error_p95_m=round(float(np.mean([r["height_error_p95_m"] for r in group])), 4),
      progress_ratio=round(float(np.mean(moving)), 3) if moving else None,
      footfalls=int(sum(r["footfalls"] for r in group)), reference_footfalls=int(sum(r["reference_footfalls"] for r in group)),
      unprompted_footfalls=int(sum(r["unprompted_footfalls"] for r in group)),
      depth_ratio=round(float(np.mean([r["depth_ratio"] for r in group if r["depth_ratio"] is not None])), 3)
      if any(r["depth_ratio"] is not None for r in group) else None)
    if student:
      for key in ("distance_ratio", "yaw_ratio", "twist_error_mean_mps"):
        values = [r[key] for r in group if r.get(key) is not None]
        summary["by_context"][context][key] = round(float(np.mean(values)), 3) if values else None
  if student:
    trained = [r for r in rows if r["trained"]]
    summary["trained_clips"] = dict(
      clips=len(trained), completed=sum(r["completed"] for r in trained), passed_strict=sum(r in strict for r in trained),
      unprompted_footfalls=int(sum(r["unprompted_footfalls"] for r in trained)),
      footfalls=int(sum(r["footfalls"] for r in trained)), reference_footfalls=int(sum(r["reference_footfalls"] for r in trained)))
  if student and done:
    for key in ("twist_error_mean_mps", "height_error_mean_m", "arm_residual_mean_frac", "arm_deviation_mean_rad"):
      summary[key + "_completed"] = round(float(np.mean([r[key] for r in done])), 4)
  result = dict(protocol="r1wb-clip-eval-v2", definition=__doc__, task=args.task, run=run.name, iteration=iteration,
                checkpoint=str(checkpoint), variant=args.variant, physics_dt=float(env.physics_dt), step_dt=float(env.step_dt),
                elapsed_s=round(time.time() - started, 1), summary=summary, clips=rows)
  if archive is not None:
    result["video"] = archive.close()
  (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=1, default=str))
  if args.variant == "train" and not args.limit and not student and not args.local_anchor:
    target = WORK / "data/wb" / f"dphys_{run.name.split('_')[1]}_{iteration}.npz"
    np.savez_compressed(target, valid=valid, completed=completed.cpu().numpy(), clip_names=np.array([c["name"] for c in clips]),
                        **record)
    print("physics-filtered record:", target)
  print(json.dumps(summary, indent=1))
  failed = [r for r in rows if not r["completed"]]
  print("not completed:", [(r["clip"][:34], r["survived_s"], r["seconds"]) for r in failed][:40])
  env.close()


if __name__ == "__main__":
  main()
