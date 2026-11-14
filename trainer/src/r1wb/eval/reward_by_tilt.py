"""Is staying up always worth more than going down?  (local, nothing is trained)
Step reward of a teacher-task scene, driven by a teacher, binned by the trunk's tilt and by whether a hit landed within
the last 0.3 s. A bin whose mean reward is negative is a state the learner is paid to leave by ending the episode.
usage: reward_by_tilt.py <task> <checkpoint> [seconds]"""
import os, sys
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import warp as wp
if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from tensordict import TensorDict
from r1wb.student.observations import load_teacher
from r1wb.tasks import TASKS
task, checkpoint = sys.argv[1], sys.argv[2]
seconds = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
count = 4096
torch.manual_seed(0)
cfg = TASKS[task]["env"]()
cfg.scene.num_envs = count
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
wrapped = RslRlVecEnvWrapper(env)
model = load_teacher(checkpoint, "cuda:0")
names = list(env.reward_manager.active_terms)
edges = torch.tensor([0.15, 0.3, 0.45, 0.6], device="cuda:0")
labels = ["tilt < 0.15", "0.15 - 0.30", "0.30 - 0.45", "0.45 - 0.60", "> 0.60 rad"]
B = len(labels)
sums = torch.zeros(B, 2, len(names), device="cuda:0"); steps = torch.zeros(B, 2, device="cuda:0")
ended = torch.zeros(B, 2, device="cuda:0")                      # steps after which the episode ended on a fall / tilt
left = torch.zeros(B, 2, device="cuda:0"); left_n = torch.zeros(B, 2, device="cuda:0")   # steps the episode still lasted
obs = wrapped.get_observations()
entity = env.scene["robot"]
history = []
for _ in range(int(seconds / env.step_dt)):
  tilt = torch.acos((-entity.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
  which = torch.bucketize(tilt, edges)
  recent = (env.hit_clock < 0.3).long()
  flat = which * 2 + recent
  with torch.no_grad():
    action = model(TensorDict({"actor": obs["actor"][:, :model.obs_dim]}, batch_size=[count]))
  obs, _, dones, _ = wrapped.step(action)
  reward = env.reward_manager._step_reward
  sums.view(B * 2, -1).index_add_(0, flat, reward)
  steps.view(-1).index_add_(0, flat, torch.ones(count, device="cuda:0"))
  manager = env.termination_manager
  down = torch.zeros(count, dtype=torch.bool, device="cuda:0")
  for name in ("fallen", "trunk_ground", "anchor_rot", "anchor_z"):
    if name in manager.active_terms:
      down |= manager.get_term(name).bool()
  ended.view(-1).index_add_(0, flat, down.float())
print(f"task {task}   driver {Path(checkpoint).parent.name[-30:]}@{Path(checkpoint).stem}")
order = sorted(range(len(names)), key=lambda i: float(sums[-1].sum(0)[i] / steps[-1].sum().clamp_min(1)))
worst = [names[i] for i in order[:6]]
print(f"{'state':14s} {'hit < 0.3 s ago':16s} {'share':>7s} {'reward/step':>11s} {'ends on a fall or tilt per s':>28s} | " + " ".join(f"{n[:14]:>14s}" for n in worst))
total_steps = float(steps.sum())
for b in range(B):
  for r in range(2):
    n = float(steps[b, r])
    if n < 50: continue
    row = sums[b, r] / n
    print(f"{labels[b]:14s} {('yes' if r else 'no'):16s} {n / total_steps:7.4f} {float(row.sum()):11.2f} {float(ended[b, r]) / (n * env.step_dt):28.2f} | "
          + " ".join(f"{float(row[names.index(k)]):14.2f}" for k in worst))
env.close()
