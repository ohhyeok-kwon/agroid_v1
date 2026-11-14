"""A few learner updates of a mirror-loss task from a checkpoint, without a run folder: does the loss run, what is
its size at the start, and how fast does it fall while only the critic and the mirror term act (warm-up)?
usage: python -m r1wb.eval.mirror_smoke <task> <checkpoint> [iterations] [envs]"""
import os
import sys
from dataclasses import asdict

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from r1wb.tasks import TASKS
from r1wb.train import start_policy

task, checkpoint = sys.argv[1], sys.argv[2]
iterations = int(sys.argv[3]) if len(sys.argv) > 3 else 6
count = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
torch.manual_seed(0)
cfg = TASKS[task]["env"]()
cfg.scene.num_envs = count
env = RslRlVecEnvWrapper(ManagerBasedRlEnv(cfg=cfg, device="cuda:0"))
agent = asdict(TASKS[task]["rl"]())
if len(sys.argv) > 5:                                    # shortened warm-up and ramp, to see the whole schedule quickly
  agent["algorithm"]["critic_warmup_updates"], agent["algorithm"]["surrogate_ramp_updates"] = int(sys.argv[5]), int(sys.argv[6])
runner = MjlabOnPolicyRunner(env, agent, None, "cuda:0")
start_policy(runner.alg.actor, checkpoint, "cuda:0", pad_inputs=True)
native = runner.alg.update
seen = []


def update():
  result = native()
  seen.append(result)
  print(f"update {len(seen):2d}: " + "  ".join(f"{k} {v:.4f}" for k, v in result.items() if k in ("mirror", "value", "surrogate", "learning_rate", "surrogate_ramp")), flush=True)
  return result


runner.alg.update = update
runner.learn(num_learning_iterations=iterations)
print("learner", type(runner.alg).__name__, "| policy", type(runner.alg.get_policy()).__name__,
      "| mirror loss first -> last:", seen[0].get("mirror"), "->", seen[-1].get("mirror"))
