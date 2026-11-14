"""What a training scene pays a given driver, per kind of command.  (local, nothing is trained)

For a mixed student scene (half clips, half command streams) this splits the step reward of the stream environments by
the kind of the commanded segment (stand / slow forward / forward / curve / backward / lateral / turn in place) and by
"calm" vs "within the grace of a hit", and reports next to it what the driver actually did there: how much of the
commanded twist it travelled, how often it fell, how often the lag limit ended the episode, and on which share of
the steps the course (labels, price) and the lag accounting were switched on.

Use: before a run, drive the candidate scene with (a) the teacher's labels, (b) the policies whose behaviour the
labels compete with. Where the labels earn less than a competing behaviour, imitation and reward pull apart there.

usage: stream_reward_by_command.py <task> <checkpoint> student|label|teacher [seconds] [--teacher <ckpt>]
  student: the checkpoint is a reference-free student (--teacher only adds the label group to the scene)
  label:   the checkpoint is a tracking teacher; its label drives the robot (student scenes)
  teacher: the task is a teacher-task scene (r1wb.teacher_task) and the checkpoint a teacher that acts in it"""
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import torch
import warp as wp

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from r1wb.student.observations import load_policy
from r1wb.tasks import TASKS
from r1wb.tracking import mdp as wb

args = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] not in ("--teacher", "--average")]
task, checkpoint, kind = args[0], args[1], args[2]
seconds = float(args[3]) if len(args) > 3 else 20.0
teacher = sys.argv[sys.argv.index("--teacher") + 1] if "--teacher" in sys.argv else None
count = 4096
torch.manual_seed(0)
cfg = TASKS[task]["env"]() if kind == "teacher" else TASKS[task]["env"](teacher=checkpoint if kind == "label" else teacher)
cfg.scene.num_envs = count
grace = float(cfg.commands["motion"].lag_grace) or 2.0
trust = {k: v for k, v in cfg.rewards["joint_course"].params.items() if k in ("stream", "grace")} if "joint_course" in cfg.rewards else None
env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
wrapped = RslRlVecEnvWrapper(env)
if kind == "label":
  policy = lambda o: o["bc"][:, :24]  # noqa: E731
elif kind == "teacher":                     # a teacher-task scene driven by a teacher (an older one reads its own inputs)
  from tensordict import TensorDict
  from r1wb.student.observations import load_teacher
  teacher_model = load_teacher(checkpoint, "cuda:0")
  if "--symmetrize" in sys.argv:            # the policy averaged with its own mirror image (r1wb.symmetry)
    from r1wb.symmetry import Symmetrized
    teacher_model = Symmetrized(teacher_model)
  if "--average" in sys.argv:               # ... or with another checkpoint's policy (is the gain symmetry or averaging?)
    first, second = teacher_model, load_teacher(sys.argv[sys.argv.index("--average") + 1], "cuda:0")

    class Averaged:
      obs_dim = first.obs_dim

      def __call__(self, observations):
        return 0.5 * (first(observations) + second(observations))
    teacher_model = Averaged()
  policy = lambda o: teacher_model(TensorDict({"actor": o["actor"][:, :teacher_model.obs_dim]}, batch_size=[count]))  # noqa: E731
else:
  model = load_policy(checkpoint, "cuda:0")
  policy = lambda o: model(torch.cat([o["actor"], o["goal"]], dim=1))  # noqa: E731
names = list(env.reward_manager.active_terms)
term = env.command_manager.get_term("motion")
picked = {"twist": [names.index(n) for n in ("goal_twist", "goal_twist_l1", "goal_lag") if n in names],
          "price": [names.index(n) for n in ("joint_course",) if n in names],
          "step": [names.index(n) for n in ("step",) if n in names]}
step_weight = abs(float(cfg.rewards["step"].weight)) if "step" in cfg.rewards else 0.0
CLASSES = ("clip envs", "stand", "slow forward", "forward", "curve", "backward", "lateral", "turn in place")
default_height = float(cfg.commands["motion"].default_height)


def classes() -> torch.Tensor:
  """Index into CLASSES for every environment, from the stream's current target. The mixed scene draws a segment's
  twist from the stride-mean twist of a library frame (scaled 0.4 to 1.1), so the kinds are told apart by the
  proportions of the twist, as the scene's own sampler does before scaling."""
  target = term.stream_target
  vx, vy, wz = target[:, 0], target[:, 1], target[:, 2].abs()
  speed = torch.norm(target[:, :2], dim=1)
  moving = (speed > 1e-6) | (wz > 1e-6)
  turn = moving & (wz > 2.5 * speed)
  lateral = moving & ~turn & (vy.abs() > vx.abs())
  backward = moving & ~turn & ~lateral & (vx < 0)
  ahead = moving & ~turn & ~lateral & (vx > 0)
  curve = ahead & (wz > 0.8 * vx)
  out = torch.full((count,), 1, dtype=torch.long, device=env.device)                    # stand
  out = torch.where(ahead & ~curve, torch.where(speed <= 0.25, 2, 3), out)
  out = torch.where(curve, 4, out)
  out = torch.where(backward, 5, out)
  out = torch.where(lateral, 6, out)
  out = torch.where(turn, 7, out)
  return torch.where(term.stream_env, out, torch.zeros_like(out))


def zeros(*shape):
  return torch.zeros(*shape, device=env.device)


C = len(CLASSES)
steps = zeros(C, 2)                                   # [class, calm / recent hit]
total, twist, price = zeros(C, 2), zeros(C, 2), zeros(C, 2)
stepped = zeros(C, 2)                                 # unprompted touchdowns (the scene's step term, if it has one)
travelled, asked = zeros(C, 2), zeros(C, 2)           # along the commanded twist (slewed goal), settled segments only
trusted, lag_on = zeros(C), zeros(C)
falls, lag_ends, low_steps = zeros(C), zeros(C), zeros(C)
obs = wrapped.get_observations()
dt = float(env.step_dt)
for _ in range(int(seconds / dt)):
  which = classes()
  clock = getattr(env, "hit_clock", torch.full((count,), 1.0e3, device=env.device))
  recent = (clock < grace).long()
  goal = term.twist_goal().clone()
  settled = (term.since_change > 1.0) | ~term.stream_env
  low = term.stream_target[:, 3] < default_height - 0.03
  with torch.no_grad():
    action = policy(obs)
  obs, _, dones, _ = wrapped.step(action)
  reward = env.reward_manager._step_reward
  own = term.own_twist()
  flat = which * 2 + recent
  one = torch.ones(count, device=env.device)
  steps.view(-1).index_add_(0, flat, one)
  total.view(-1).index_add_(0, flat, reward.sum(1))
  twist.view(-1).index_add_(0, flat, reward[:, picked["twist"]].sum(1))
  if picked["price"]:
    price.view(-1).index_add_(0, flat, reward[:, picked["price"]].sum(1))
  if picked["step"]:
    stepped.view(-1).index_add_(0, flat, -reward[:, picked["step"]].sum(1) / step_weight)
  size = torch.sum(goal * goal * torch.tensor([1.0, 1.0, 0.25], device=env.device), dim=1)
  along = torch.sum(own * goal * torch.tensor([1.0, 1.0, 0.25], device=env.device), dim=1)
  moving = (settled & (size > 1e-6)).float()
  travelled.view(-1).index_add_(0, flat, along * moving)
  asked.view(-1).index_add_(0, flat, size * moving)
  if trust is not None:
    trusted.index_add_(0, which, wb.course_is_trusted(env, "motion", **trust).float())
  lag_on.index_add_(0, which, (clock >= grace).float())
  low_steps.index_add_(0, which, low.float())
  manager = env.termination_manager
  fell = torch.zeros(count, dtype=torch.bool, device=env.device)
  for name in ("fallen", "trunk_ground"):
    if name in manager.active_terms:
      fell |= manager.get_term(name).bool()
  falls.index_add_(0, which, fell.float())
  if "goal_lag" in manager.active_terms:
    lag_ends.index_add_(0, which, manager.get_term("goal_lag").bool().float())

per = lambda a, b: a / b.clamp_min(1.0)  # noqa: E731
print(f"task {task}   driver {kind} {Path(checkpoint).parent.name[-30:]}@{Path(checkpoint).stem}   {seconds:g} s x {count} envs   grace {grace:g} s")
print(f"{'command':14s} {'share':>6s} {'recent':>6s} | reward/step calm: {'total':>6s} {'twist':>6s} {'price':>6s} | after a hit: {'total':>6s} {'twist':>6s} {'price':>6s} |"
      f" travelled/asked {'calm':>5s} {'hit':>5s} | falls, lag ends /100 env-s | course on, lag on, low | unprompted steps /100 env-s calm, after a hit")
all_steps = float(steps.sum())
for i, name in enumerate(CLASSES):
  n = steps[i].sum()
  if float(n) < 1:
    continue
  env_s = float(n) * dt / 100.0
  print(f"{name:14s} {float(n) / all_steps:6.3f} {float(steps[i, 1] / n):6.2f} |                   "
        f"{float(per(total[i, 0], steps[i, 0])):6.2f} {float(per(twist[i, 0], steps[i, 0])):6.2f} {float(per(price[i, 0], steps[i, 0])):6.2f} |              "
        f"{float(per(total[i, 1], steps[i, 1])):6.2f} {float(per(twist[i, 1], steps[i, 1])):6.2f} {float(per(price[i, 1], steps[i, 1])):6.2f} |"
        f"                 {float(travelled[i, 0] / asked[i, 0].clamp_min(1e-9)):5.2f} {float(travelled[i, 1] / asked[i, 1].clamp_min(1e-9)):5.2f} |"
        f"        {float(falls[i]) / env_s:6.2f} {float(lag_ends[i]) / env_s:6.2f}          |"
        f"   {float(trusted[i] / n):5.2f} {float(lag_on[i] / n):5.2f} {float(low_steps[i] / n):5.2f} |"
        f"   {float(stepped[i, 0] / (steps[i, 0].clamp_min(1.0) * dt)) * 100.0:7.1f} {float(stepped[i, 1] / (steps[i, 1].clamp_min(1.0) * dt)) * 100.0:7.1f}")
stream = steps[1:].sum(0)
print(f"{'streams, all':14s} {float(stream.sum()) / all_steps:6.3f} {float(stream[1] / stream.sum()):6.2f} |                   "
      f"{float(total[1:, 0].sum() / stream[0]):6.2f} {float(twist[1:, 0].sum() / stream[0]):6.2f} {float(price[1:, 0].sum() / stream[0]):6.2f} |              "
      f"{float(total[1:, 1].sum() / stream[1]):6.2f} {float(twist[1:, 1].sum() / stream[1]):6.2f} {float(price[1:, 1].sum() / stream[1]):6.2f} |"
      f"                 {float(travelled[1:, 0].sum() / asked[1:, 0].sum().clamp_min(1e-9)):5.2f} {float(travelled[1:, 1].sum() / asked[1:, 1].sum().clamp_min(1e-9)):5.2f} |"
      f"        {float(falls[1:].sum()) / (float(stream.sum()) * dt / 100.0):6.2f} {float(lag_ends[1:].sum()) / (float(stream.sum()) * dt / 100.0):6.2f}          |"
      f"   {float(trusted[1:].sum() / stream.sum()):5.2f} {float(lag_on[1:].sum() / stream.sum()):5.2f} {float(low_steps[1:].sum() / stream.sum()):5.2f}")
env.close()
