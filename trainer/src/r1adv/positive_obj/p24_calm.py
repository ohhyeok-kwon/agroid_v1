"""Positive 24 E3 calm reference (rounds/R04/positive/e3_calm_ref/rules.md): manufacturer (evaluation only), B0 and E114
on the same panel. Workstation only (the manufacturer wrapper and weights are not synchronised).

Scene: r1wb stream env (play, clean, flat), terminations = time-out only (a fall is shown, not reset), 17 robots:
  robot 0       stand 20 s, zero command
  robots 1..16  push J in (12, 15, 20, 25) N s x (front, back, left, right) of the pelvis heading at 4.0 s: one control
                step (20 ms) of force J / 0.02 s on torso_link, 0.1 m above it (torque (0, 0, 0.1) x F), then 16 s more
Policies:
  manufacturer  r1wb.eval.manufacturer.load() wrapper (deploy contract, twist 0), plain joint-position action term
  b0            B0 seed 1 model_2400 actor (normaliser + mlp, mean): anchor frames with the S1 hold goal (mask (0,0,1),
                h* = the pelvis height after reset), policy-order deploy action -> entity order, plain action term
  e114          app/operation_center DynamicCore with .agent/operation_center/dynamic_mode.json, told STAND
Record (every 0.02 s): joint pos / vel / targets (24), qfrc_actuator (24 joint dofs), pelvis pos / quat / lin vel /
ang vel, foot sole minimum height (4 sole corners of each ankle_roll link, x -0.05..0.133, y +-0.035, z -0.055).

  python p24_calm.py --policy manufacturer|b0|e114 --out OUT.npz [--device cpu|cuda:0]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
J_SET = (12.0, 15.0, 20.0, 25.0)
DIRS = {"front": (1.0, 0.0), "back": (-1.0, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}
PUSH_T = 4.0
SECONDS = 20.0
SOLE = [(x, y, -0.055) for x in (-0.05, 0.133) for y in (-0.035, 0.035)]
B0_CK = "/rl-agent/.agent/runs/r1_careful_walk/careful_walk_20261120-1829_b0-s1objective-anchorrepr_n4096_p23b0_s1/model_2400.pt"


def rows():
  out = [("stand", 0.0, "none")]
  for J in J_SET:
    for d in DIRS:
      out.append(("push", J, d))
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", choices=("manufacturer", "b0", "e114"), required=True)
  ap.add_argument("--out", required=True)
  ap.add_argument("--device", default="cpu")
  ap.add_argument("--b0", default=None, help="local copy of the B0 seed 1 model_2400.pt")
  args = ap.parse_args()
  import torch
  t0 = time.time()
  R = rows()
  n = len(R)
  dev = args.device
  meta = dict(policy=args.policy, rows=R, device=dev, argv=sys.argv,
              driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  if args.policy == "e114":
    sys.path.insert(0, str(ROOT / "app/operation_center"))
    from dynamic_core import DynamicCore, STAND
    sel_path = ROOT / ".agent/operation_center/dynamic_mode.json"
    selection = json.loads(sel_path.read_text())
    for key in ("system1", "head", "tokens", "steer", "library"):
      selection[key]["path"] = str((ROOT / selection[key]["path"]).resolve())
    core = DynamicCore(selection, environment="flat", envs=n, device=dev)
    core.tell(STAND)
    env = core.env
    meta.update(selection=str(sel_path), selection_sha256=hashlib.sha256(sel_path.read_bytes()).hexdigest(),
                calm_fraction=getattr(core.term.cfg, "calm_fraction", None))

    def step():
      core.step()
  else:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.managers.termination_manager import TerminationTermCfg
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.envs import mdp as base_mdp
    from r1wb.student.stream import stream_env_cfg
    cfg = stream_env_cfg(play=True, clean=True, plain_actions=True)
    cfg.scene.num_envs = n
    cfg.terminations = {"time_out": TerminationTermCfg(func=base_mdp.time_out, time_out=True)}
    if hasattr(cfg.commands["motion"], "calm_fraction"):
      cfg.commands["motion"].calm_fraction = 1.0
    meta["events"] = list(cfg.events)
    env = ManagerBasedRlEnv(cfg=cfg, device=dev)
    wrapped = RslRlVecEnvWrapper(env)
    env.reset()
    obs = wrapped.get_observations()
    if args.policy == "manufacturer":
      from r1wb.eval.manufacturer import load
      weights, Pol = load()
      pol = Pol(env, torch.zeros(n, 3, device=dev))
      meta.update(weights=str(weights), weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                  wrapper=str(Pol.wrapper_path))

      def step():
        nonlocal obs
        with torch.no_grad():
          a = pol(obs)
        obs = wrapped.step(a)[0]
    else:
      from rsl_rl.modules import MLP
      from rsl_rl.modules.normalization import EmpiricalNormalization
      from r1lab import contract as C
      from r1s1 import numbers as N
      ck_path = args.b0 or B0_CK
      sd = torch.load(ck_path, map_location=dev, weights_only=False)["actor_state_dict"]
      layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
      dims = [sd[f"mlp.{i}.weight"].shape for i in layers]
      mlp = MLP(dims[0][1], dims[-1][0], [x[0] for x in dims[:-1]], "elu").to(dev)
      mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
      norm = EmpiricalNormalization(shape=[dims[0][1]], until=0).to(dev)
      norm.load_state_dict({k[len("obs_normalizer."):]: v for k, v in sd.items() if k.startswith("obs_normalizer.")})
      mlp.eval()
      norm.eval()
      to_policy = torch.tensor(C.POLICY_TO_ENTITY, device=dev)
      to_entity = torch.tensor(C.ACTION_IDX, device=dev)
      default = torch.tensor(C.DEFAULT_POS, device=dev)
      rob = env.scene["robot"]
      h_star = rob.data.root_link_pos_w[:, 2].clone()
      prev = torch.zeros(n, 24, device=dev)
      hist = None
      meta.update(b0=ck_path, b0_sha256=hashlib.sha256(Path(ck_path).read_bytes()).hexdigest(), lambda_h=N.LAMBDA[2])

      def step():
        nonlocal prev, hist
        d = rob.data
        goal = torch.zeros(n, 9, device=dev)
        goal[:, 3] = 1.0
        goal[:, 4] = (h_star - d.root_link_pos_w[:, 2]) / N.LAMBDA[2]
        goal[:, 7] = 1.0
        frame = torch.cat([d.root_link_ang_vel_b, d.projected_gravity_b, goal, d.joint_pos[:, to_policy] - default,
                           d.joint_vel[:, to_policy], prev], dim=1)
        hist = frame.unsqueeze(1).repeat(1, 10, 1) if hist is None else torch.cat([hist[:, 1:], frame.unsqueeze(1)], dim=1)
        with torch.no_grad():
          a = mlp(norm(hist.reshape(n, -1)))
        prev = a
        wrapped.step(a[:, to_entity])
  rob = env.scene["robot"]
  names = list(rob.body_names)
  feet = [names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")]
  torso = names.index("torso_link")
  dt = float(env.step_dt)
  steps = int(round(SECONDS / dt))
  k_push = int(round(PUSH_T / dt))
  sole = torch.tensor(SOLE, device=dev)
  pushed = torch.tensor([r[0] == "push" for r in R], device=dev)
  Jv = torch.tensor([r[1] for r in R], device=dev)
  Dv = torch.tensor([DIRS.get(r[2], (0.0, 0.0)) for r in R], device=dev)
  rec = {k: [] for k in ("q", "dq", "qt", "tau", "pos", "quat", "v", "w", "sole_z")}
  zero3 = torch.zeros(n, 1, 3, device=dev)
  for k in range(steps):
    if k == k_push:
      q = rob.data.root_link_quat_w
      yaw = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
      c, s = torch.cos(yaw), torch.sin(yaw)
      f = torch.stack([(c * Dv[:, 0] - s * Dv[:, 1]), (s * Dv[:, 0] + c * Dv[:, 1]), torch.zeros(n, device=dev)], 1) * (Jv / dt)[:, None]
      f = f * pushed[:, None]
      arm = torch.tensor([0.0, 0.0, 0.1], device=dev).expand(n, 3)
      rob.write_external_wrench_to_sim(f.unsqueeze(1), torch.cross(arm, f, dim=1).unsqueeze(1), body_ids=[torso])
    step()
    if k == k_push:
      rob.write_external_wrench_to_sim(zero3, zero3, body_ids=[torso])
    d = rob.data
    rec["q"].append(d.joint_pos.clone())
    rec["dq"].append(d.joint_vel.clone())
    rec["qt"].append(d.joint_pos_target.clone())
    rec["tau"].append(d.qfrc_actuator[:, -24:].clone())
    rec["pos"].append((d.root_link_pos_w - env.scene.env_origins).clone())
    rec["quat"].append(d.root_link_quat_w.clone())
    rec["v"].append(d.root_link_lin_vel_w.clone())
    rec["w"].append(d.root_link_ang_vel_w.clone())
    zs = []
    for b in feet:
      p, qq = d.body_link_pos_w[:, b], d.body_link_quat_w[:, b]
      u = qq[:, 1:]
      for corner in sole:
        v = corner.expand(n, 3)
        t = 2.0 * torch.cross(u, v, dim=1)
        zs.append((p + v + qq[:, :1] * t + torch.cross(u, t, dim=1))[:, 2])
    zz = torch.stack(zs, 1).reshape(n, 2, 4).amin(-1)
    rec["sole_z"].append(zz)
  arrs = {k: torch.stack(v, 0).cpu().numpy().astype(np.float32) for k, v in rec.items()}
  meta.update(dt=dt, steps=steps, k_push=k_push, joint_names=list(rob.joint_names), default=rob.data.default_joint_pos[0].cpu().tolist(),
              wall_s=round(time.time() - t0, 1))
  np.savez_compressed(args.out, meta=json.dumps(meta), **arrs)
  print(json.dumps(dict(out=args.out, wall_s=meta["wall_s"], steps=steps)), flush=True)


if __name__ == "__main__":
  main()
