"""E3 (positive 23, coordinator 2026-11-20 22:0x; rounds/R04/16_rounds.md 'R18-b feedback' and 'E3 card'). DIAGNOSTIC.
L    B0 (r1lab.tasks.careful_walk_b0) + the E1 R2 leaf L1 only: the policy std fixed at S0b-fixed's sigma_500 x 0.5
     (r1s1.numbers.FIXED_STD x SIGMA_FIXED_SCALE, mean 0.34; r1s1.distribution.FixedGaussian), the per-joint vector put
     from S1's entity order into B0's policy order. Everything else is B0's.
     Units: B0's learned std and this one are both in raw policy-output units (joint target = q0 + a x ACTION_SCALE[joint];
     B0's anchor action term and S1's entity-order term have equal per-joint scales), so 0.34 and B0's 5.1 compare directly.
a04  B0 + a push magnitude boundary d: the S1 trunk impulses become d x U(5, 25) N s (timing, direction and point as
     r1s1.mdp.s1_impulse). d starts at 0; every 50 updates (1,200 env steps) it moves +0.1 when the training hold success
     of those steps is > 0.6 and -0.1 when < 0.4, kept in [0, 1]. Hold success = post-deadline hold-goal steps in the goal
     and at rest / post-deadline hold-goal steps (r1s1 GoalCommand.judge; hold is one of the training goal kinds).
     Logged each step as Metrics/a04_d, Metrics/a04_hold_success and Metrics/a04_post_success (post-deadline in goal and
     at rest over all goal kinds; both of the last finished window) and printed each window ('[A04] ...')."""
from __future__ import annotations

import math

import torch

WINDOW_STEPS = 50 * 24      # 50 updates of 24 env steps (careful_walk_rl_cfg num_steps_per_env)


def fixed_std_policy_order() -> list[float]:
  from r1lab import contract as C
  from r1s1 import numbers as N
  ent = list(N.FIXED_STD)
  perm = list(C.POLICY_TO_ENTITY)
  assert all(perm[C.ACTION_IDX[j]] == j for j in range(len(ent))), "policy / entity maps disagree"
  pol = [ent[perm[p]] for p in range(len(ent))]
  assert all(abs(pol[C.MIRROR_INDEX[p]] - pol[p]) < 1e-9 for p in range(len(pol))), "std not mirror symmetric"
  return pol


def l_rl_cfg():
  from r1lab.tasks.careful_walk_b0 import b0_rl_cfg
  from r1s1 import numbers as N
  cfg = b0_rl_cfg()
  cfg.actor.distribution_cfg = {"class_name": "r1s1.distribution:FixedGaussian", "std_values": fixed_std_policy_order(),
                                "scale": N.SIGMA_FIXED_SCALE}
  return cfg


class a04_impulse:
  """r1s1.mdp.s1_impulse with the impulse magnitude scaled by the boundary d, and the d controller. mode="step"."""

  def __init__(self, cfg, env):
    from r1s1 import numbers as N
    self.N = N
    self.robot = env.scene["robot"]
    self.body = [self.robot.body_names.index("torso_link")]
    n, device = env.num_envs, env.device
    low, high = N.IMPULSE_EVERY_S
    self.wait = low + (high - low) * torch.rand(n, device=device)
    self.active = torch.zeros(n, dtype=torch.bool, device=device)
    self.force = torch.zeros(n, 3, device=device)
    self.torque = torch.zeros(n, 3, device=device)
    self.written = False
    env._r1s1_push_force = torch.zeros(n, 3, device=device)
    self.d = float(cfg.params.get("d0", 0.0))
    self.sums = torch.zeros(4, device=device)        # hold post, hold ok, all post, all ok
    self.next = WINDOW_STEPS
    self.last = (float("nan"), float("nan"))

  def reset(self, env_ids=None) -> None:
    if env_ids is None:
      return
    low, high = self.N.IMPULSE_EVERY_S
    self.wait[env_ids] = low + (high - low) * torch.rand(len(env_ids), device=self.wait.device)
    self.active[env_ids] = False
    self.force[env_ids] = 0.0
    self.torque[env_ids] = 0.0

  def _control(self, env) -> None:
    term = env.command_manager.get_term("goal")
    _, in_goal, at_rest = term.judge()
    post = term.tau <= 0.0
    hold = post & (term.kind == 0)
    ok = in_goal & at_rest
    self.sums += torch.stack([hold.sum(), (hold & ok).sum(), post.sum(), (post & ok).sum()]).float()
    step = int(env.common_step_counter)
    if step >= self.next:
      s = self.sums.tolist()
      hold_success = s[1] / s[0] if s[0] > 0 else float("nan")
      post_success = s[3] / s[2] if s[2] > 0 else float("nan")
      d_before = self.d
      if s[0] > 0 and hold_success > 0.6:
        self.d = min(1.0, round(self.d + 0.1, 6))
      elif s[0] > 0 and hold_success < 0.4:
        self.d = max(0.0, round(self.d - 0.1, 6))
      print(f"[A04] step={step} update={step // 24} hold_success={hold_success:.4f} post_success={post_success:.4f} "
            f"hold_post_steps={int(s[0])} d {d_before:.1f} -> {self.d:.1f}", flush=True)
      self.last = (hold_success, post_success)
      self.sums.zero_()
      self.next += WINDOW_STEPS
    log = env.extras.setdefault("log", {})
    log["Metrics/a04_d"] = torch.tensor(self.d)
    log["Metrics/a04_hold_success"] = torch.tensor(self.last[0])
    log["Metrics/a04_post_success"] = torch.tensor(self.last[1])

  def __call__(self, env, env_ids, **params) -> None:
    del env_ids, params                              # params: d0 (read once in __init__)
    N = self.N
    self._control(env)
    device, dt = env.device, env.step_dt
    self.wait -= dt
    start = (self.wait <= 0.0) & ~self.active
    self.active[:] = False
    if start.any():
      ids = start.nonzero().squeeze(-1)
      count = len(ids)
      low, high = N.IMPULSE_NS
      impulse = self.d * (low + (high - low) * torch.rand(count, device=device))
      angle = (torch.rand(count, device=device) * 2.0 - 1.0) * math.pi
      force = torch.stack([angle.cos(), angle.sin(), torch.zeros_like(angle)], dim=1) * (impulse / dt).unsqueeze(1)
      arm = torch.zeros(count, 3, device=device)
      arm[:, 2] = N.IMPULSE_HEIGHT_M[0] + (N.IMPULSE_HEIGHT_M[1] - N.IMPULSE_HEIGHT_M[0]) * torch.rand(count, device=device)
      side = N.IMPULSE_SIDE_M[0] + (N.IMPULSE_SIDE_M[1] - N.IMPULSE_SIDE_M[0]) * torch.rand(count, device=device)
      arm[:, 0], arm[:, 1] = -angle.sin() * side, angle.cos() * side
      self.force[ids], self.torque[ids] = force, torch.cross(arm, force, dim=1)
      self.active[ids] = True
      low, high = N.IMPULSE_EVERY_S
      self.wait[ids] = low + (high - low) * torch.rand(count, device=device)
    if not (self.active.any() or self.written):
      env._r1s1_push_force.zero_()
      return
    on = self.active.float().unsqueeze(1)
    self.robot.write_external_wrench_to_sim((self.force * on).unsqueeze(1), (self.torque * on).unsqueeze(1),
                                            body_ids=self.body)
    env._r1s1_push_force = self.force * on
    self.written = bool(self.active.any())


def a04_env_cfg(play: bool = False, t0: int = 0):
  import dataclasses
  from r1lab.tasks.careful_walk_b0 import b0_env_cfg
  cfg = b0_env_cfg(play=play, t0=t0)
  if "s1_impulse" in cfg.events:
    t = cfg.events["s1_impulse"]
    cfg.events["s1_impulse"] = dataclasses.replace(t, func=a04_impulse, params={**t.params, "d0": 0.0})
  return cfg
