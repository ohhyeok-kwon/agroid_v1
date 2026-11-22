"""R2 wave 2, C1 / C1T (positive agent; coordinator lock 2026-11-17 10:12 KST, item 4): the clean-label DAgger student
of a teacher that passed the teacher gate. Not started before the gate result and the coordinator's go.

Teacher: the teacher's final checkpoint reading the noise-free 570 frames (the critic group's frame block; for T that is
its own actor input). Student: HoldSymmetricMLP 570 reading the noisy frames (0.2-s history), initialised from the
teacher, observation normaliser frozen. DAgger in the R2 training distribution (hold_r2_env_cfg(actor_clean=False):
starts, hits, nominal sensors, command delays, body randomisation): the first TEACHER_ITERS iterations act with the
teacher's clean action, then with the student's action; MSE between the student's mean action and the teacher's label;
buffer of the last BUFFER iterations. No fine-tune. The run folder follows the trainer's layout (run_info.json,
model_<k>.pt, distill.log) under the runs root's Hold experiment; the code hash at import (hold_r2.CODE_SHA256) and the
distribution sha256 are written into run_info.json.
usage: python -m r1adv.positive.distill_r2 --teacher RUN_DIR:IT --seed S [--iters 800] [--num-envs 4096] [--arm c1]"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import torch
from tensordict import TensorDict

TEACHER_ITERS = 5
BUFFER = 8
EPOCHS = 2
MINIBATCH = 16384
LR = 2.0e-4
HORIZON = 24


def load_actor(path: Path, obs, device: str):
  from r1adv.positive import hold_model as HM
  state = torch.load(path, map_location=device, weights_only=False)["actor_state_dict"]
  layers = sorted(int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight"))
  hidden = [state[f"mlp.{i}.weight"].shape[0] for i in layers[:-1]]
  model = HM.HoldSymmetricMLP(obs, {"actor": ["actor"]}, "actor", 24, hidden_dims=hidden, activation="elu",
                              obs_normalization=True, distribution_cfg=None).to(device)
  missing, unexpected = model.load_state_dict({k: v for k, v in state.items() if not k.startswith("distribution.")}, strict=False)
  if unexpected or [m for m in missing if m != "r1adv_contract"]:
    raise ValueError(f"checkpoint does not fit HoldSymmetricMLP: missing {missing}, unexpected {unexpected}")
  model.eval()
  return model, state.get("distribution.std_param")


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--teacher", required=True, help="RUN_DIR:ITERATION of a teacher that passed the gate")
  ap.add_argument("--seed", type=int, required=True)
  ap.add_argument("--iters", type=int, default=800)
  ap.add_argument("--num-envs", type=int, default=4096)
  ap.add_argument("--arm", default="c1", help="c1 (teacher C0) or c1t (teacher T): only names the folder")
  ap.add_argument("--save-every", type=int, default=100)
  ap.add_argument("--lu", action="store_true", help="arm C1-LU: student sensor noise size log-uniform 0.003-1 per channel type")
  a = ap.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from r1lab.run_record import work_root
  from r1adv.positive import hold_r2 as R2

  run, it = a.teacher.rsplit(":", 1)
  run = Path(run)
  tinfo = json.loads((run / "run_info.json").read_text())
  cfg = R2.hold_r2_env_cfg(actor_clean=False, play=False, sense_lu=a.lu)   # the student's training distribution
  cfg.scene.num_envs = a.num_envs
  cfg.seed = a.seed
  env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
  wrapped = RslRlVecEnvWrapper(env)
  obs = wrapped.get_observations()
  width = obs["actor"].shape[1]
  if width != 570:
    raise ValueError(f"the student contract is 570; the env gives {width}")
  teacher, std = load_actor(run / f"model_{it}.pt", obs, env.device)
  student = copy.deepcopy(teacher)
  student.train()
  student.obs_normalizer.eval()
  for p in student.obs_normalizer.parameters():
    p.requires_grad_(False)
  opt = torch.optim.Adam([p for n_, p in student.named_parameters() if n_.startswith("mlp.")], lr=LR)

  stamp = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d-%H%M")
  root = work_root().resolve() / "runs" / "r1_r1adv_hold"
  rid = f"r1adv_hold_{stamp}_r2-{a.arm}-from-{tinfo['id'].split('_r2-')[-1]}_n{a.num_envs}_s{a.seed}"
  out = root / rid
  out.mkdir(parents=True, exist_ok=False)
  info = dict(id=rid, task="adv_hold_r2_c0", kind="r1adv_distill", arm=a.arm, teacher=f"{tinfo['id']}:{it}",
              teacher_task=tinfo.get("task"), seed=a.seed, iterations=a.iters, num_envs=a.num_envs, sense_lu=bool(a.lu),
              code_sha256=R2.CODE_SHA256, distribution_sha256=R2.R2_DIST["sha256"], start_frames_sha256=R2.START_FRAMES["sha256"],
              hyper=dict(teacher_iters=TEACHER_ITERS, buffer=BUFFER, epochs=EPOCHS, minibatch=MINIBATCH, lr=LR, horizon=HORIZON),
              host=os.uname().nodename, started_at=time.time(), state="running")
  (out / "run_info.json").write_text(json.dumps(info, indent=1))
  log = open(out / "distill.log", "a")
  buf_o, buf_y = [], []
  t0 = time.time()
  for itn in range(a.iters):
    with torch.no_grad():
      for _ in range(HORIZON):
        y = teacher(TensorDict({"actor": obs["critic"][:, :width]}, batch_size=[a.num_envs]))
        act = student(obs) if itn >= TEACHER_ITERS else y
        buf_o.append(obs["actor"].clone()); buf_y.append(y.clone())
        obs, _, _, _ = wrapped.step(act)
    buf_o, buf_y = buf_o[-BUFFER * HORIZON:], buf_y[-BUFFER * HORIZON:]
    X, Y = torch.cat(buf_o), torch.cat(buf_y)
    losses = []
    for _ in range(EPOCHS):
      perm = torch.randperm(len(X), device=X.device)
      for k in range(0, len(X), MINIBATCH):
        ids = perm[k:k + MINIBATCH]
        loss = (student(TensorDict({"actor": X[ids]}, batch_size=[len(ids)])) - Y[ids]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    row = dict(iter=itn, loss=sum(losses) / len(losses), seconds=round(time.time() - t0, 1))
    log.write(json.dumps(row) + "\n"); log.flush()
    if (itn + 1) % a.save_every == 0 or itn == a.iters - 1:
      state = dict(student.state_dict())
      state["distribution.std_param"] = std.clone()
      torch.save({"actor_state_dict": state, "iter": itn + 1, "teacher": info["teacher"], "kind": "r1adv_distill"},
                 out / f"model_{itn + 1}.pt")
  info.update(state="completed", finished_at=time.time())
  (out / "run_info.json").write_text(json.dumps(info, indent=1))
  print(json.dumps(dict(run=str(out), last_loss=row["loss"])))
  env.close()


if __name__ == "__main__":
  main()
