"""Positive 24 E2 sc: self-collision audit + c08 hazard records (rounds/R04/positive/e2_sc/rules.md).

One cell = (run r1 | b0, checkpoint, action mode mean | noise, push on | off, eval seed): the run's TRAINING env config
(play=False: S1 startup randomisation, sensor noise, delays, terminations, 20 s time-out; r1 = careful_walk_c13 R1 env,
b0 = careful_walk_b0 env = S1 objective v2), only the S1 trunk push event removed for push off. N envs, each env's FIRST
episode is one episode (N = 200 per cell). Added, read-only: a per-body robot-robot contact sensor (primary = each robot
body, secondary = the pelvis subtree, maxforce, 4-substep history, as s1_self) and a no-op termination term run after
S1's terms (before the reset) that stores the pre-reset state of the step.
Policy: the checkpoint's anchor actor (empirical normaliser + mlp); mean = mlp output, noise = mean + std x N(0, 1)
(the learned std of the checkpoint, torch generator seeded with the eval seed) - the training sampling.

  python p24sc_drive.py --run r1|b0 --checkpoint C --mode mean|noise --push on|off --out OUT [--seed S] [--envs N]
writes OUT.npz (per-step arrays of the first episodes) and OUT.json (events, meta)
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

FOOT_RECT = ((-0.05, 0.133), (-0.035, 0.035))      # ankle_roll frame: foot capsules x -0.04..0.123, y +-0.025, r 0.01
FOOT_CONTACT_N = 5.0
N_DIR = 36


def probe(env):
  """No-op termination term (always False): stores the pre-reset state of this step on env._p24_stash."""
  import torch
  from r1s1 import mdp as SM
  st = env._p24
  rob = env.scene["robot"]
  d = rob.data
  dev = env.device
  selfc = SM.s1_self_collision(env)
  ground = SM.s1_body_ground(env)
  sb = env.scene["sc_body"].data
  if sb.force_history is not None:
    bf = torch.norm(sb.force_history, dim=-1).flatten(2).amax(dim=-1)            # (B, P)
  else:
    bf = torch.norm(sb.force, dim=-1)
  feet = torch.norm(env.scene["feet_contact"].data.force, dim=-1)                # (B, 2): the anchor sensor (both runs)
  com = d.data.subtree_com[:, d.indexing.root_body_id]
  m = st["mass"]
  comv = (d.body_com_lin_vel_w * m[None, :, None]).sum(1) / m.sum()
  w0 = torch.sqrt(9.81 / com[:, 2].clamp_min(0.2))
  cp = com[:, :2] + comv[:, :2] / w0[:, None]
  # support polygon: corners of the contacting feet's sole rectangles
  pts, msk = [], []
  for k, b in enumerate(st["feet_ids"]):
    p = d.body_link_pos_w[:, b]
    q = d.body_link_quat_w[:, b]
    for x in FOOT_RECT[0]:
      for y in FOOT_RECT[1]:
        v = torch.tensor([x, y, 0.0], device=dev).expand(env.num_envs, 3)
        pts.append((p + quat_rotate(q, v))[:, :2])
        msk.append(feet[:, k] > FOOT_CONTACT_N)
  P_ = torch.stack(pts, 1)                                                       # (B, 8, 2)
  M_ = torch.stack(msk, 1)                                                       # (B, 8)
  rel = P_ - cp[:, None, :]
  proj = torch.einsum("bpi,di->bdp", rel, st["dirs"])                            # (B, D, 8)
  proj = torch.where(M_[:, None, :], proj, torch.full_like(proj, -1e9))
  inside = (proj.amax(-1).amin(-1) >= 0.0) & M_.any(1)
  push = getattr(env, "_r1s1_push_force", None)
  push_on = (torch.norm(push, dim=-1) > 0) if push is not None else torch.zeros(env.num_envs, dtype=torch.bool, device=dev)
  jv = d.joint_vel
  stash = dict(selfc=selfc, ground=ground, bf=bf, feet=feet, cp_in=inside, push=push_on, jv=jv,
               lin=d.root_link_lin_vel_w, ang=d.root_link_ang_vel_w, z=d.root_link_pos_w[:, 2], com_z=com[:, 2])
  if st["run"] == "b0":
    g = env.command_manager.get_term("goal")
    stash.update(kind=g.kind, tau=g.tau)
  else:
    stash.update(cmd=env.command_manager.get_command("twist"))
  env._p24_stash = {k: v.detach().clone() for k, v in stash.items()}
  return torch.zeros(env.num_envs, dtype=torch.bool, device=dev)


def quat_rotate(q, v):
  w, x, y, z = q.unbind(-1)
  u = q[..., 1:]
  t = 2.0 * torch_cross(u, v)
  return v + w.unsqueeze(-1) * t + torch_cross(u, t)


def torch_cross(a, b):
  import torch
  return torch.cross(a, b, dim=-1)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", choices=("r1", "b0"), required=True)
  ap.add_argument("--checkpoint", required=True)
  ap.add_argument("--mode", choices=("mean", "noise"), required=True)
  ap.add_argument("--push", choices=("on", "off"), required=True)
  ap.add_argument("--seed", type=int, default=20261120)
  ap.add_argument("--envs", type=int, default=200)
  ap.add_argument("--steps", type=int, default=None)
  ap.add_argument("--device", default="cuda:0")
  ap.add_argument("--out", required=True)
  args = ap.parse_args()
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.termination_manager import TerminationTermCfg
  from mjlab.sensor import ContactMatch, ContactSensorCfg
  t0 = time.time()
  if args.run == "r1":
    from r1lab.tasks.careful_walk_c13 import r1_env_cfg
    cfg = r1_env_cfg(play=False)
  else:
    from r1lab.tasks.careful_walk_b0 import b0_env_cfg
    cfg = b0_env_cfg(play=False)
  cfg.scene.num_envs = args.envs
  cfg.seed = args.seed
  cfg.events = dict(cfg.events)
  if args.push == "off":
    cfg.events.pop("s1_impulse", None)
  sc = ContactSensorCfg(name="sc_body", primary=ContactMatch(mode="body", pattern=r".*", entity="robot"),
                        secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
                        fields=("found", "force"), reduce="maxforce", num_slots=1, history_length=4)
  cfg.scene.sensors = tuple(cfg.scene.sensors) + (sc,)
  cfg.terminations = dict(cfg.terminations)
  cfg.terminations["sc_probe"] = TerminationTermCfg(func=probe)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  dev = env.device
  rob = env.scene["robot"]
  names = list(rob.body_names)
  bid = rob.data.indexing.body_ids
  mass = torch.as_tensor(env.sim.mj_model.body_mass[bid.cpu().numpy()], device=dev, dtype=torch.float32)
  ang = torch.arange(N_DIR, device=dev) * (2 * math.pi / N_DIR)
  env._p24 = dict(run=args.run, mass=mass, feet_ids=[names.index("left_ankle_roll_link"), names.index("right_ankle_roll_link")],
                  dirs=torch.stack([ang.cos(), ang.sin()], 1))
  prim = list(env.scene["sc_body"].primary_names)
  joint_names = list(rob.joint_names)
  dt = float(env.step_dt)
  steps = args.steps or min(int(round(env.max_episode_length)) + 1, 1600)
  # policy
  ck = torch.load(args.checkpoint, map_location=dev, weights_only=False)
  sd = ck["actor_state_dict"]
  from rsl_rl.modules import MLP
  from rsl_rl.modules.normalization import EmpiricalNormalization
  layers = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.") and k.endswith(".weight")})
  dims = [sd[f"mlp.{i}.weight"].shape for i in layers]
  mlp = MLP(dims[0][1], dims[-1][0], [x[0] for x in dims[:-1]], "elu").to(dev)
  mlp.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("mlp.")})
  norm = EmpiricalNormalization(shape=[dims[0][1]], until=0).to(dev)
  norm.load_state_dict({k[len("obs_normalizer."):]: v for k, v in sd.items() if k.startswith("obs_normalizer.")})
  mlp.eval()
  norm.eval()
  std_keys = [k for k in sd if not k.startswith(("mlp.", "obs_normalizer.")) and "std" in k]
  std = None
  if std_keys:
    s = sd[std_keys[0]].float().to(dev)
    std = s.exp() if "log" in std_keys[0] else s
  if args.mode == "noise" and std is None:
    raise SystemExit(f"no std in checkpoint keys {list(sd)[:20]}")
  gen = torch.Generator(device=dev)
  gen.manual_seed(args.seed)
  n = args.envs
  obs, _ = env.reset()
  first = torch.ones(n, dtype=torch.bool, device=dev)
  rec = {k: [] for k in ("alive", "push", "cp_in", "feet_l", "feet_r", "rms_jv", "vxy", "kind", "tau", "cmd")}
  events = []
  last_push = torch.full((n,), -1e9, device=dev)
  last_feet = torch.zeros(n, dtype=torch.long, device=dev)
  last_change = torch.zeros(n, device=dev)
  rms_hist = []
  for k in range(steps):
    with torch.no_grad():
      a = mlp(norm(obs["actor"]))
      if args.mode == "noise":
        a = a + std * torch.randn(a.shape, device=dev, generator=gen)
    out = env.step(a)
    obs = out[0]
    term, trunc = out[2].bool().reshape(-1), out[3].bool().reshape(-1)
    s = env._p24_stash
    t = (k + 1) * dt
    rms = torch.sqrt((s["jv"] ** 2).mean(1))
    rms_hist.append(rms)
    rms01 = torch.stack(rms_hist[-5:], 0).mean(0)
    last_push = torch.where(s["push"], torch.full_like(last_push, t), last_push)
    fstate = (s["feet"][:, 0] > FOOT_CONTACT_N).long() + 2 * (s["feet"][:, 1] > FOOT_CONTACT_N).long()
    last_change = torch.where(fstate != last_feet, torch.full_like(last_change, t), last_change)
    last_feet = fstate
    vxy = torch.norm(s["lin"][:, :2], dim=1)
    rec["alive"].append(first.clone())
    rec["push"].append(s["push"])
    rec["cp_in"].append(s["cp_in"])
    rec["feet_l"].append(s["feet"][:, 0] > FOOT_CONTACT_N)
    rec["feet_r"].append(s["feet"][:, 1] > FOOT_CONTACT_N)
    rec["rms_jv"].append(rms)
    rec["vxy"].append(vxy)
    if args.run == "b0":
      rec["kind"].append(s["kind"].to(torch.int8))
      rec["tau"].append(s["tau"])
    else:
      rec["cmd"].append(torch.norm(s["cmd"][:, :2], dim=1))
    done = (term | trunc) & first
    if done.any():
      for i in done.nonzero().flatten().tolist():
        kind = "self" if bool(s["selfc"][i]) else ("ground" if bool(s["ground"][i]) else ("timeout" if bool(trunc[i]) else "other"))
        e = dict(env=i, step=k, t=round(t, 3), type=kind, push_age=round(float(t - last_push[i]), 3),
                 feet=int(fstate[i]), since_feet_change=round(float(t - last_change[i]), 3),
                 rms_jv=float(rms[i]), rms_jv_01=float(rms01[i]), max_jv=float(s["jv"][i].abs().max()), vxy=float(vxy[i]),
                 ang=float(torch.norm(s["ang"][i])), z=float(s["z"][i]), com_z=float(s["com_z"][i]), cp_in=bool(s["cp_in"][i]))
        if kind in ("self", "ground"):
          bfi = s["bf"][i]
          order = torch.argsort(bfi, descending=True)[:4].tolist()
          e["bodies"] = [(prim[j], round(float(bfi[j]), 2)) for j in order if float(bfi[j]) > 0.0]
          e["jv"] = [round(float(x), 3) for x in s["jv"][i].tolist()]
        if args.run == "b0":
          e["goal_kind"] = int(s["kind"][i])
          e["tau"] = round(float(s["tau"][i]), 3)
        else:
          e["cmd"] = [round(float(x), 3) for x in s["cmd"][i].tolist()]
        events.append(e)
      first &= ~done
    if not first.any():
      break
  arrs = {}
  for key, v in rec.items():
    if v:
      x = torch.stack(v, 0).cpu().numpy()
      arrs[key] = x.astype(np.float16) if x.dtype == np.float32 else x
  np.savez_compressed(args.out + ".npz", **arrs)
  meta = dict(argv=sys.argv, run=args.run, checkpoint=args.checkpoint,
              checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(), mode=args.mode, push=args.push,
              seed=args.seed, envs=n, steps_run=len(rec["alive"]), dt=dt, max_episode_length=float(env.max_episode_length),
              events_cfg=list(cfg.events), terminations_cfg=list(cfg.terminations), primaries=prim, joint_names=joint_names,
              std=None if std is None else [round(float(x), 4) for x in std.tolist()], std_key=std_keys[:1],
              wall_s=round(time.time() - t0, 1), driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
  Path(args.out + ".json").write_text(json.dumps(dict(meta=meta, events=events)))
  counts = {}
  for e in events:
    counts[e["type"]] = counts.get(e["type"], 0) + 1
  print(json.dumps(dict(out=args.out, counts=counts, steps=len(rec["alive"]), wall_s=meta["wall_s"])), flush=True)
  env.close()


if __name__ == "__main__":
  main()
