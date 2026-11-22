"""S1 gates (prints one JSON line; exit code 1 if a gate fails).

  config <task> [env.yaml]   term names equal the S1 whitelist; no ingredient-table name of the old recipe appears
                             in the dumped configuration (12_ingredients.md rows) — also on a run's params/env.yaml
  body <task>                robot signature (r1lab.body_hash) of the task's entity = the stock fist c2f65e1a
  mirror <task>              in the simulator: pairs of environments in mirror-image states with mirror-image goals
                             and actions; actor observation of the twin = mirror map of the observation (history too)
  init <task> [envs] [steps] random-init look on the CPU: actions N(0, 1) (the initial policy's noise), termination
                             counts per term, episode length, reward terms, mean effort
  std <run_dir>              learned exploration noise of every checkpoint (rise = saturation signal, recorded)
  sym <checkpoint>           network equivariance of the saved actor on random inputs (by construction ~0)
"""
from __future__ import annotations

from dataclasses import asdict
import json
import re
import sys
from pathlib import Path

WHITELIST = {
  "rewards": {"s1_success", "s1_progress", "s1_effort_joule|s1_effort_power"},
  "terminations": {"time_out", "s1_body_ground", "s1_self_collision"},
  "commands": {"goal"},
  "events": {"reset_base", "reset_joints", "foot_friction", "torso_com", "torso_mass", "hand_load", "hand_load_com",
             "pd_gains", "encoder_bias", "s1_impulse"},
  "actor": {"proprio", "last_target", "goal"},
  "critic": {"proprio", "last_target", "goal", "privileged"},
}
# names of the parent recipe's ingredients (12_ingredients.md rows 1-68) that must not occur in an S1 configuration
FORBIDDEN = (
  "motion_global_root_pos", "motion_global_root_ori", "motion_body_pos", "motion_body_ori", "motion_body_lin_vel",
  "motion_body_ang_vel", "feet_pos", "anchor_z_l1", "joint_course", "stance_slip", "foot_lift", "goal_twist",
  "goal_lag", "goal_height", "goal_pitch", "goal_hands", "arm_deviation", "action_rate", "joint_limit",
  "self_collisions", "is_terminated", "clip_end", "clip_finished", "fallen", "trunk_ground", "anchor_z", "anchor_rot",
  "anchor_xy", "feet_z", "hands_z", "grace", "lag_grace", "hit_clock", "motion_file", "library_v2", "intent",
  "stream_course", "match_commit", "travel_window", "preview", "impulse_unless_calm", "calm_fraction", "burst_prob",
  "low_scale", "push_unless_calm", "step_unprompted", "rest_room", "course_travel", "recovery_risk", "LibraryMotion",
  "MixedGoal", "teacher_observation", "no_label", "bc_coef",
)


def _task(name):
  from r1wb.tasks import TASKS
  if name in TASKS:
    return TASKS[name]
  from r1s1.tasks import TASKS as OWN      # before the registration lines land
  return OWN[name]


def _out(record: dict) -> int:
  print(json.dumps(record))
  return 0 if record.get("ok") else 1


def config(name: str, yaml_path: str | None = None) -> int:
  from mjlab.utils.os import dump_yaml
  cfg = _task(name)["env"]()
  agent = _task(name)["rl"]()
  bad = []
  def keys(section):
    return set(section.keys())
  checks = {"rewards": keys(cfg.rewards), "terminations": keys(cfg.terminations), "commands": keys(cfg.commands),
            "events": keys(cfg.events), "actor": keys(cfg.observations["actor"].terms),
            "critic": keys(cfg.observations["critic"].terms)}
  for section, found in checks.items():
    allowed = set()
    for entry in WHITELIST[section]:
      allowed |= set(entry.split("|"))
    extra = found - allowed
    if extra:
      bad.append(f"{section}: not in whitelist {sorted(extra)}")
  efforts = [k for k in cfg.rewards if k.startswith("s1_effort_")]
  if len(efforts) != 1:
    bad.append(f"exactly one effort term expected, got {efforts}")
  import tempfile
  with tempfile.TemporaryDirectory() as tmp:
    dump_yaml(Path(tmp) / "env.yaml", asdict(cfg))
    dump_yaml(Path(tmp) / "agent.yaml", asdict(agent))
    texts = {"env.yaml(cfg)": (Path(tmp) / "env.yaml").read_text(), "agent.yaml(cfg)": (Path(tmp) / "agent.yaml").read_text()}
  if yaml_path:
    texts["run:" + yaml_path] = Path(yaml_path).read_text()
  hits = {}
  for label, text in texts.items():
    for word in FORBIDDEN:
      if re.search(re.escape(word), text):
        hits.setdefault(label, []).append(word)
  if hits:
    bad.append(f"forbidden ingredient names found: {hits}")
  agent_text = texts["agent.yaml(cfg)"]
  return _out({"gate": "config", "task": name, "ok": not bad, "failed": bad, "terms": {k: sorted(v) for k, v in checks.items()},
               "effort": efforts, "forbidden_checked": len(FORBIDDEN),
               "agent": {"S1SymmetricMLPModel": "S1SymmetricMLPModel" in agent_text,
                         "obs_normalization_false": agent_text.count("obs_normalization: false") == 2}})


def body(name: str) -> int:
  from r1lab.body_hash import entity_signature
  cfg = _task(name)["env"]()
  sig = entity_signature(cfg.scene.entities["robot"])
  total = sig["total"]
  return _out({"gate": "body", "task": name, "total": total, "ok": total.startswith("c2f65e1a"), "signature": sig})


def _cpu_env(name, envs, play=True, clean=True, **kw):
  import torch
  torch.set_num_threads(2)
  from mjlab.envs import ManagerBasedRlEnv
  cfg = _task(name)["env"](play=play, clean=clean, **kw)
  cfg.scene.num_envs = envs
  cfg.seed = 7
  return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def mirror(name: str, pairs: int = 4, steps: int = 30) -> int:
  import torch
  from r1s1.model import action_mirror, actor_mirror
  from r1wb.symmetry import JOINT_PERM, JOINT_SIGN
  env = _cpu_env(name, 2 * pairs)
  env.reset()
  robot = env.scene["robot"]
  n = env.num_envs
  even, odd = torch.arange(0, n, 2), torch.arange(1, n, 2)
  perm, jsign = torch.tensor(JOINT_PERM), torch.tensor(JOINT_SIGN)
  g = torch.Generator().manual_seed(3)
  q0 = robot.data.default_joint_pos.clone()
  q = q0.clone()
  q[even] = q0[even] + (torch.rand(pairs, 24, generator=g) * 2 - 1) * 0.15
  q[odd] = q0[odd] + ((q[even] - q0[even])[:, perm] * jsign)
  dq = torch.zeros(n, 24)
  dq[even] = (torch.rand(pairs, 24, generator=g) * 2 - 1) * 0.5
  dq[odd] = dq[even][:, perm] * jsign
  robot.write_joint_state_to_sim(q, dq)
  root = robot.data.default_root_state.clone()
  origin = env.scene.env_origins
  yaw = (torch.rand(pairs, generator=g) * 2 - 1) * 3.0
  local = torch.zeros(pairs, 3)
  local[:, 0], local[:, 1] = (torch.rand(pairs, generator=g) - 0.5) * 0.4, (torch.rand(pairs, generator=g) - 0.5) * 0.4
  local[:, 2] = root[even, 2]
  root[even, :3] = origin[even] + local
  mloc = local * torch.tensor([1.0, -1.0, 1.0])
  root[odd, :3] = origin[odd] + mloc
  half = yaw / 2
  quat = torch.stack([torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)], dim=1)
  root[even, 3:7] = quat
  root[odd, 3:7] = quat * torch.tensor([1.0, -1.0, 1.0, -1.0])
  lin = (torch.rand(pairs, 3, generator=g) - 0.5) * 0.2
  ang = (torch.rand(pairs, 3, generator=g) - 0.5) * 0.4
  root[even, 7:10], root[even, 10:13] = lin, ang
  root[odd, 7:10], root[odd, 10:13] = lin * torch.tensor([1.0, -1.0, 1.0]), ang * torch.tensor([-1.0, 1.0, -1.0])
  robot.write_root_state_to_sim(root)
  env.sim.forward()
  term = env.command_manager.get_term("goal")
  for field_name in ("kind", "mask", "h_target", "hold", "tau0"):
    getattr(term, field_name)[odd] = getattr(term, field_name)[even]
  term.delta[odd] = term.delta[even] * torch.tensor([1.0, -1.0, -1.0])
  term.hold[:] = 100.0
  term.time_left[odd] = term.time_left[even] = term.tau0[even] + 100.0
  term.pending[:] = True
  term._update_command(None)
  amap, asign = action_mirror()
  for _ in range(steps):
    a = torch.zeros(n, 24)
    a[even] = (torch.rand(pairs, 24, generator=g) * 2 - 1) * 0.3
    a[odd] = a[even][:, amap] * asign
    obs, *_ = env.step(a)
  actor = obs["actor"]
  index, sign = actor_mirror()
  mirrored = actor[even][:, index] * sign
  diff = (actor[odd] - mirrored).abs()
  control = (actor[odd] - actor[even]).abs()
  frame = 54
  newest = slice((25 - 1) * frame, 25 * frame)
  record = {"gate": "mirror", "task": name, "pairs": pairs, "steps": steps, "max_abs": float(diff.max()),
            "mean_abs": float(diff.mean()), "max_abs_newest_frame": float(diff[:, newest].max()),
            "max_abs_goal": float(diff[:, -9:].max()), "control_unmirrored_max": float(control.max()),
            "control_unmirrored_mean": float(control.mean())}
  record["ok"] = record["max_abs"] < 0.02 and record["control_unmirrored_mean"] > 20 * record["mean_abs"]
  env.close()
  return _out(record)


def init(name: str, envs: int = 64, steps: int = 250) -> int:
  import torch
  env = _cpu_env(name, envs, play=False, clean=False)
  env.reset()
  counts = {k: 0 for k in env.termination_manager.active_terms}
  lengths, rewards = [], {k: 0.0 for k in env.reward_manager.active_terms}
  effort_name = [k for k in rewards if k.startswith("s1_effort_")][0]
  effort_values = []
  g = torch.Generator().manual_seed(11)
  for _ in range(steps):
    a = torch.randn(envs, 24, generator=g)
    _, _, terminated, truncated, _ = env.step(a)
    for k in counts:
      counts[k] += int(env.termination_manager.get_term(k).sum())
    for k in rewards:
      rewards[k] += float(env.reward_manager._step_reward[:, env.reward_manager.active_terms.index(k)].mean())
    func = env.reward_manager.get_term_cfg(effort_name).func
    effort_values.append(float(func(env).mean()))
  record = {"gate": "init", "task": name, "envs": envs, "steps": steps, "terminations": counts,
            "episodes_ended": sum(v for k, v in counts.items() if k != "time_out"),
            "reward_terms_mean_weighted_per_s": {k: v / steps for k, v in rewards.items()},
            "effort_mean_raw": sum(effort_values) / len(effort_values), "ok": True}
  env.close()
  return _out(record)


def std(run_dir: str) -> int:
  import torch
  rows = []
  for path in sorted(Path(run_dir).glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
    state = torch.load(path, map_location="cpu", weights_only=False)["actor_state_dict"]
    from r1s1.distribution import sigma_from_state
    value = sigma_from_state(state)            # scalar / log / mirror-tied std (r1s1.distribution)
    if value is None:
      continue
    rows.append({"iteration": int(path.stem.split("_")[1]), "mean": float(value.mean()), "min": float(value.min()),
                 "max": float(value.max())})
  low = min((r["mean"] for r in rows), default=None)
  rising = bool(rows and low and rows[-1]["mean"] > 1.5 * low)
  return _out({"gate": "std", "run": run_dir, "checkpoints": rows, "min_mean": low,
               "saturation_signal_rise_over_1p5x_min": rising, "ok": bool(rows)})


def sym(checkpoint: str) -> int:
  import torch
  from r1s1.model import WIDTH, action_mirror, actor_mirror
  from rsl_rl.modules import MLP
  state = torch.load(checkpoint, map_location="cpu", weights_only=False)["actor_state_dict"]
  layers = sorted({int(k.split(".")[1]) for k in state if k.startswith("mlp.") and k.endswith(".weight")})
  mlp_state = {k[4:]: v for k, v in state.items() if k.startswith("mlp.")}
  dims = [state[f"mlp.{i}.weight"].shape for i in layers]
  hidden = [d[0] for d in dims[:-1]]
  mlp = MLP(dims[0][1], dims[-1][0], hidden, "elu")
  mlp.load_state_dict(mlp_state)
  index, sign = actor_mirror()
  amap, asign = action_mirror()
  x = torch.randn(4096, WIDTH)

  def policy(o):
    return 0.5 * (mlp(o) + mlp(o[:, index] * sign)[:, amap] * asign)

  with torch.no_grad():
    error = (policy(x[:, index] * sign) - policy(x)[:, amap] * asign).abs().max()
    plain = (mlp(x[:, index] * sign) - mlp(x)[:, amap] * asign).abs().max()
  # the policy's std is learned per joint and not tied across the mirror (R12-G.5): the gate also reads the largest
  # left/right pair ratio of std (coordinator 05:2x), threshold numbers.SIGMA_PAIR_RATIO_MAX
  from r1lab.contract import ENTITY_JOINTS
  from r1s1 import numbers as N
  record = {"gate": "sym", "checkpoint": checkpoint, "width": int(dims[0][1]), "equivariance_max_abs": float(error),
            "plain_network_asymmetry_max_abs": float(plain), "mean_ok": float(error) < 1e-4}
  from r1s1.distribution import sigma_from_state
  std = sigma_from_state(state)              # scalar / log / mirror-tied std (r1s1.distribution)
  if std is not None:
    std = std.float()
    pair = torch.maximum(std / std[amap], std[amap] / std)
    j = int(pair.argmax())
    record.update({"sigma_pair_ratio_max": float(pair.max()),
                   "sigma_pair": [ENTITY_JOINTS[j], ENTITY_JOINTS[int(amap[j])]] if len(std) == len(ENTITY_JOINTS) else [j, int(amap[j])],
                   "sigma_pair_values": [float(std[j]), float(std[int(amap[j])])], "sigma_min": float(std.min()),
                   "sigma_ok": float(pair.max()) <= N.SIGMA_PAIR_RATIO_MAX, "sigma_pair_ratio_limit": N.SIGMA_PAIR_RATIO_MAX,
                   "sigma": [round(float(v), 6) for v in std], "sigma_collapse_ratio": float(std.min() / std.quantile(0.5)),
                   "sigma_collapse": float(std.min() / std.quantile(0.5)) < N.SIGMA_COLLAPSE})
  record["ok"] = record["mean_ok"] and record.get("sigma_ok", True)
  return _out(record)


def main(argv=None) -> int:
  argv = list(sys.argv[1:] if argv is None else argv)
  command, rest = argv[0], argv[1:]
  if command == "config":
    return config(*rest)
  if command == "body":
    return body(*rest)
  if command == "mirror":
    return mirror(rest[0], *(int(v) for v in rest[1:]))
  if command == "init":
    return init(rest[0], *(int(v) for v in rest[1:]))
  if command == "std":
    return std(*rest)
  if command == "sym":
    return sym(*rest)
  raise SystemExit(__doc__)


if __name__ == "__main__":
  raise SystemExit(main())
