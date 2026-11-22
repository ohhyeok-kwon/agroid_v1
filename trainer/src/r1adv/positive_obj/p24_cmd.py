"""Positive 24 E3 calm reference, part 2 (rounds/R04/positive/e3_calm_ref/rules_cmd.md): command step response and
walking appearance on the same r1wb stream env (play, clean, flat, time-out only). Workstation CPU.

Robots (20 s each, command from t0 = 1.0 s; before: stand / zero command):
  0  forward 0.5 m/s           (step response; walking appearance at 0.5, window [4, 14] s)
  1  lateral 0.3 m/s (left)    (step response)
  2  turn 0.8 rad/s (left)     (step response)
  3  forward 0.5 m/s until 9.0 s, then zero   (stop)
  4  forward 0.3 m/s           (step response; walking appearance at 0.3)
Subjects:
  manufacturer  evaluation-only wrapper, twist = the command (plain joint-position action term)
  e114direct    E114 System 1 (dynamic_mode.json system1) tracking the stream command's own reference, the command
                written as the stream script (duration, vx, vy, yaw_rate, height 0.731, ...): no System 2 / steer
  e114dm        DynamicCore (dynamic_mode.json): core.tell((vx, vy, yaw_rate, 0.731, 0)) every step = the W-key path from
                tell on (steer -> System 2 head -> tokens -> reference -> System 1)
  python p24_cmd.py --subject manufacturer|e114direct|e114dm --out OUT.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
T0, T_STOP, SECONDS, H = 1.0, 9.0, 20.0, 0.731
CMDS = [(0.5, 0.0, 0.0), (0.0, 0.3, 0.0), (0.0, 0.0, 0.8), (0.5, 0.0, 0.0), (0.3, 0.0, 0.0)]
SOLE = [(x, y, -0.055) for x in (-0.05, 0.133) for y in (-0.035, 0.035)]


def command_at(i: int, t: float):
  if t < T0 - 1e-9:
    return (0.0, 0.0, 0.0)
  if i == 3 and t >= T_STOP - 1e-9:
    return (0.0, 0.0, 0.0)
  return CMDS[i]


def selection():
  sel_path = ROOT / ".agent/operation_center/dynamic_mode.json"
  sel = json.loads(sel_path.read_text())
  for key in ("system1", "head", "tokens", "steer", "library"):
    sel[key]["path"] = str((ROOT / sel[key]["path"]).resolve())
  return sel, sel_path


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--subject", choices=("manufacturer", "e114direct", "e114dm"), required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--device", default="cpu")
  args = ap.parse_args()
  import torch
  t_start = time.time()
  n, dev = len(CMDS), args.device
  meta = dict(subject=args.subject, cmds=CMDS, t0=T0, t_stop=T_STOP, argv=sys.argv,
              driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs import mdp as base_mdp
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.rl import RslRlVecEnvWrapper
  from tensordict import TensorDict
  if args.subject == "e114dm":
    sys.path.insert(0, str(ROOT / "app/operation_center"))
    from dynamic_core import DynamicCore
    sel, sel_path = selection()
    core = DynamicCore(sel, environment="flat", envs=n, device=dev)
    env = core.env
    st = core.steerer
    meta.update(selection_sha256=hashlib.sha256(sel_path.read_bytes()).hexdigest(),
                steer=dict(see=getattr(st, "see", None), leak=getattr(st.progress, "leak", None), remember=getattr(st, "remember", None),
                           turned=getattr(st, "turned", None)),
                s2_period_steps=getattr(core.view, "_period", None))
    goal = torch.zeros(n, 5, device=dev)
    goal[:, 3] = H

    def step(t):
      for i in range(n):
        goal[i, :3] = torch.tensor(command_at(i, t), device=dev)
      core.tell(goal)
      core.step()
  elif args.subject == "e114direct":
    from r1wb.student.observations import load_teacher
    from r1wb.student.stream import stream_env_cfg
    from r1wb.tracking.env_cfg import teacher_observation
    sel, sel_path = selection()
    checkpoint = sel["system1"]["path"]
    cfg = stream_env_cfg(play=True, clean=True, matched=True, teacher=checkpoint)       # as DynamicCore
    motion = cfg.commands["motion"]
    motion.match_pose, motion.match_commit, motion.match_height, motion.match_stop = "reference", True, "stream", "room"
    motion.motion_file = sel["library"]["path"]
    stand = motion.clip_list
    motion.clip_list = tuple(stand[i % len(stand)] for i in range(n))
    cfg.scene.num_envs = n
    cfg.terminations = {"time_out": cfg.terminations["time_out"]}
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    wrapped = RslRlVecEnvWrapper(env)
    term = env.command_manager.get_term("motion")
    table = torch.zeros(n, 3, 8)
    for i in range(n):
      vx, vy, wz = CMDS[i]
      table[i, 0] = torch.tensor([T0, 0.0, 0.0, 0.0, H, 0.0, 0.0, 0.0])
      if i == 3:
        table[i, 1] = torch.tensor([T_STOP - T0, vx, vy, wz, H, 0.0, 0.0, 0.0])
        table[i, 2] = torch.tensor([1.0e6, 0.0, 0.0, 0.0, H, 0.0, 0.0, 0.0])
      else:
        table[i, 1] = torch.tensor([1.0e6, vx, vy, wz, H, 0.0, 0.0, 0.0])
        table[i, 2] = table[i, 1]
    term.set_script(table)
    env.reset()
    wrapped.get_observations()
    tracker = load_teacher(checkpoint, dev)
    joint_action = env.action_manager.get_term("joint_pos")
    meta.update(selection_sha256=hashlib.sha256(sel_path.read_bytes()).hexdigest(), script=table.tolist())

    def step(t):
      with torch.no_grad():
        seen = teacher_observation(env, "motion", joint_action.plain_action, local=True, width=tracker.obs_dim)
        action = joint_action.from_plain_action(tracker(TensorDict({"actor": seen}, batch_size=[n])))
        wrapped.step(action)
  else:
    from r1wb.student.stream import stream_env_cfg
    from r1wb.eval.manufacturer import load
    cfg = stream_env_cfg(play=True, clean=True, plain_actions=True)
    cfg.scene.num_envs = n
    cfg.terminations = {"time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True)}
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    wrapped = RslRlVecEnvWrapper(env)
    env.reset()
    obs = wrapped.get_observations()
    weights, Pol = load()
    twists = torch.zeros(n, 3, device=dev)
    pol = Pol(env, twists)
    meta.update(weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest())

    def step(t):
      nonlocal obs
      for i in range(n):
        twists[i] = torch.tensor(command_at(i, t), device=dev)
      with torch.no_grad():
        a = pol(obs)
      obs = wrapped.step(a)[0]
  rob = env.scene["robot"]
  names = list(rob.body_names)
  feet = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
  torso = names.index("torso_link")
  dt = float(env.step_dt)
  steps = int(round(SECONDS / dt))
  sole = torch.tensor(SOLE, device=dev)
  rec = {k: [] for k in ("q", "dq", "pos", "quat", "v", "w", "torso_quat", "sole_z", "cmd")}
  for k in range(steps):
    t = k * dt
    step(t)
    d = rob.data
    rec["q"].append(d.joint_pos.clone())
    rec["dq"].append(d.joint_vel.clone())
    rec["pos"].append((d.root_link_pos_w - env.scene.env_origins).clone())
    rec["quat"].append(d.root_link_quat_w.clone())
    rec["v"].append(d.root_link_lin_vel_w.clone())
    rec["w"].append(d.root_link_ang_vel_w.clone())
    rec["torso_quat"].append(d.body_link_quat_w[:, torso].clone())
    rec["cmd"].append(torch.tensor([command_at(i, t) for i in range(n)], device=dev))
    zs = []
    for b in feet:
      p, qq = d.body_link_pos_w[:, b], d.body_link_quat_w[:, b]
      u = qq[:, 1:]
      for corner in sole:
        v = corner.expand(n, 3)
        tt = 2.0 * torch.cross(u, v, dim=1)
        zs.append((p + v + qq[:, :1] * tt + torch.cross(u, tt, dim=1))[:, 2])
    rec["sole_z"].append(torch.stack(zs, 1).reshape(n, 2, 4).amin(-1))
  arrs = {k: torch.stack(v, 0).cpu().numpy().astype(np.float32) for k, v in rec.items()}
  meta.update(dt=dt, steps=steps, joint_names=list(rob.joint_names), wall_s=round(time.time() - t_start, 1))
  np.savez_compressed(args.out, meta=json.dumps(meta, default=str), **arrs)
  print(json.dumps(dict(out=args.out, wall_s=meta["wall_s"])), flush=True)


if __name__ == "__main__":
  main()
