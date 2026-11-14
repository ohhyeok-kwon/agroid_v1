"""System 2 as a learned slow loop: it sees what the body did with the instruction, and steers.

Measured 11-13 (gates): a quiet System 1 follows the posture of its course and loses the course's travel and heading
(sideways 0.35 m/s: 0.71 to 0.77 of the distance, heading 34 degrees off). It cannot see that drift: its reference
is re-anchored to itself. Pricing travel inside System 1 made it chase the lost distance with the whole body after a
hit (E96). A hand-written feedback on the realised path, used in evaluation only, closed the gap (all sixteen walking
cells within 0.97 to 1.03 of the distance, 1 degree of direction) with the quiet numbers unchanged: global progress
is the slow layer's job. Here that layer is learned.

One decision every 16 control steps (0.32 s):
  sees      the instruction (twist, height), how long it has been in force, its own last correction, and - by `see` -
            "none": nothing of the body; "progress": where the body is and faces against where the instruction would
            have taken it since it began; "full": also the body's velocity, tilt and turning rates
  does      decides whether the body moves at all, and if so gives the motion generator its command: the
            instruction, seen from where the body now faces, plus a bounded correction
  is paid   for the body being where the instruction leads (path and heading), for being still when told to stand,
            for small corrections, for a course System 1 can keep; a fall costs
System 1 (a frozen tracking policy) follows the generator's course at 50 Hz under its own training disturbances, so
System 2 learns what to ask of it after a hit too. Reinforcement learning (PPO), privileged critic.

  train:  python -m r1wb.s2.steer train --system1 <run>/model_N.pt --task <System 1's task> --see progress --out steer_x.pt
  use:    r1wb.eval.eval_stream --steer x, r1wb.eval.impulse_probe --steer x
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402
from torch import nn  # noqa: E402
import warp as wp  # noqa: E402

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

PERIOD = 16
CORRECTION = (0.25, 0.25, 0.6)          # the most one decision adds to the instruction: m/s, m/s, rad/s
FASTEST = (0.6, 1.3)
TWIST_SCALE = (0.35, 0.35, 1.0)
SEE = {"none": 10, "progress": 13, "full": 22}
ACTIONS = 4                             # move or stand, correction (3)
DEFAULT_HEIGHT = 0.731


def yaw_of(quat: torch.Tensor) -> torch.Tensor:
  w, x, y, z = quat.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(angle: torch.Tensor) -> torch.Tensor:
  return torch.atan2(torch.sin(angle), torch.cos(angle))


def turn(vector: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
  cos, sin = torch.cos(angle), torch.sin(angle)
  return torch.stack([cos * vector[:, 0] - sin * vector[:, 1], sin * vector[:, 0] + cos * vector[:, 1]], dim=1)


class Progress:
  """How far the body is behind the instruction: where the instruction would have taken it minus where it went, and
  the same for its heading, summed since the instruction began (reads the simulator's root pose: the upper bound of
  what an odometry estimate can give).

  leak (seconds, 0 = none): what was lost longer ago than this is forgotten. Without it System 2 walks the body
  back onto the instruction's path after a hit, which is what nobody asked for (measured 11-13 11:37 with the first
  learned loop: 1.5 to 6.5 extra steps after one 12 N s hit while walking, against 0 without the loop, and 31
  against 21 falls in rows of eight). A steady shortfall or drift stays visible either way."""

  def __init__(self, env, term, told, leak: float = 0.0):
    """told() -> (envs, 5): the instruction now (twist, height, whether the height counts)."""
    self.env, self.term, self.robot, self.told, self.leak = env, term, env.scene["robot"], told, leak
    count, device = env.num_envs, env.device
    self.gap, self.turn_gap = torch.zeros(count, 2, device=device), torch.zeros(count, device=device)     # world frame
    # What the body owes the instruction (the reward's measure, training only): the same sums, but nothing is added
    # while the body is being hit or recovering (`after_step(calm)`): distance lost to a push is not an error of
    # following, and a System 2 that is paid for winning it back walks into the recovery (gates 11-13 11:37).
    self.owed, self.turn_owed = torch.zeros(count, 2, device=device), torch.zeros(count, device=device)
    self.last_pos, self.last_yaw = torch.zeros(count, 2, device=device), torch.zeros(count, device=device)
    self.shadow = torch.zeros(count, 3, device=device)         # the instruction, slewed as the stream slews
    self.age = torch.zeros(count, device=device)
    self.last = told()[:, :3].clone()
    self.begin(torch.arange(count, device=device))

  def begin(self, ids: torch.Tensor) -> None:
    data = self.robot.data
    self.last_pos[ids] = data.root_link_pos_w[ids, :2]
    self.last_yaw[ids] = yaw_of(data.root_link_quat_w[ids])
    self.gap[ids], self.turn_gap[ids], self.age[ids] = 0.0, 0.0, 0.0
    self.owed[ids], self.turn_owed[ids] = 0.0, 0.0

  def after_step(self, calm: torch.Tensor | None = None) -> None:
    """After every control step. An instruction begins when what is told changes, and when an episode restarts
    (the slewed instruction then starts from the restarted stream's own value)."""
    dt = float(self.env.step_dt)
    told = self.told()[:, :3]
    restarted = self.env.episode_length_buf <= 1
    new = (((told - self.last).abs().max(dim=1).values > 1.0e-4) | restarted).nonzero(as_tuple=False).flatten()
    if len(new):
      self.begin(new)
    again = restarted.nonzero(as_tuple=False).flatten()
    if len(again):
      self.shadow[again] = self.term.stream[again, :3]
    self.last = told.clone()
    limit = self.term.slew[:3] * dt
    self.shadow += torch.maximum(torch.minimum(told - self.shadow, limit), -limit)
    data = self.robot.data
    position, yaw = data.root_link_pos_w[:, :2], yaw_of(data.root_link_quat_w)
    # The instruction's heading is the body's plus what the body still has to turn.
    moved, turned = position - self.last_pos, wrap(yaw - self.last_yaw)
    counts = torch.ones_like(self.age) if calm is None else calm.float()
    self.owed += (turn(self.shadow[:, :2], self.last_yaw + self.turn_owed) * dt - moved) * counts.unsqueeze(1)
    self.turn_owed += (self.shadow[:, 2] * dt - turned) * counts
    self.gap += turn(self.shadow[:, :2], self.last_yaw + self.turn_gap) * dt - moved
    self.turn_gap += self.shadow[:, 2] * dt - turned
    if self.leak > 0.0:
      keep = 1.0 - dt / self.leak
      self.gap *= keep
      self.turn_gap *= keep
      self.owed *= keep
      self.turn_owed *= keep
    self.last_pos, self.last_yaw = position.clone(), yaw.clone()
    self.age += dt

  def error(self) -> tuple[torch.Tensor, torch.Tensor]:
    """(gap seen from the body (m), heading the body still has to turn (rad))."""
    return turn(self.gap, -yaw_of(self.robot.data.root_link_quat_w)), wrap(self.turn_gap)

  def owed_error(self) -> tuple[torch.Tensor, torch.Tensor]:
    return turn(self.owed, -yaw_of(self.robot.data.root_link_quat_w)), wrap(self.turn_owed)


class Steer:
  """What System 2 sees and how its action becomes the generator's command (shared by training and evaluation)."""

  def __init__(self, env, term, see: str = "progress", generator: bool = True, leak: float = 0.0, remember: bool = False,
               turned: bool = False):
    """generator: the command is written into the course generator's stream (training, generator-driven runs).
    Otherwise the instruction comes through `tell` and the command is read from `command` (a code head plays it)."""
    assert see in SEE
    self.env, self.term, self.see, self.generator, self.remember, self.turned = env, term, see, generator, remember, turned
    self.seen_gap = torch.zeros(env.num_envs, 3, device=env.device)      # the gap at the decision before (scaled)
    self._told = None
    self.progress = Progress(env, term, self.told, leak)
    self.scale = torch.tensor(CORRECTION, device=env.device)
    self.twist_scale = torch.tensor(TWIST_SCALE, device=env.device)
    self.previous = torch.zeros(env.num_envs, 3, device=env.device)
    self.moved = torch.zeros(env.num_envs, 1, device=env.device)
    self.command = torch.zeros(env.num_envs, 3, device=env.device)

  def told(self) -> torch.Tensor:
    if self._told is not None:
      return self._told
    return torch.cat([self.term.told[:, :4], self.term.goal_mask[:, 1:2]], dim=1)

  def tell(self, goal: torch.Tensor) -> None:
    """goal (envs, 5): twist, height, whether the height counts - what a vision-language model or a script says."""
    self._told = goal.clone()

  def restart(self, ids: torch.Tensor) -> None:
    self.previous[ids], self.moved[ids], self.seen_gap[ids] = 0.0, 0.0, 0.0

  def observe(self, see: str | None = None) -> torch.Tensor:
    see = see or self.see
    term, progress = self.term, self.progress
    told = self.told()
    parts = [told[:, :3] / self.twist_scale, (told[:, 3:4] - DEFAULT_HEIGHT) / 0.1, told[:, 4:5],
             (progress.age / 3.0).clamp(max=1.0).unsqueeze(1), self.moved, self.previous / self.scale]
    if see != "none":
      gap, heading = progress.error()
      parts += [(gap / 0.3).clamp(-3.0, 3.0), (heading / 0.5).clamp(-3.0, 3.0).unsqueeze(1)]
      if self.remember:                   # ... and what it was one decision ago: a jump is not a drift
        parts += [self.seen_gap.clone()]
    if see == "full":
      data = self.env.scene["robot"].data
      parts += [term.own_twist() / self.twist_scale, data.projected_gravity_b, data.root_link_ang_vel_b]
    return torch.cat(parts, dim=1)

  def act(self, action: torch.Tensor) -> torch.Tensor:
    """action (envs, 4): move (> 0.5) or stand, and the correction (3, unbounded). Writes the generator's command and
    returns the correction in force (zero where the body is told to stand)."""
    term = self.term
    if self.see != "none":
      gap, heading = self.progress.error()
      self.seen_gap = torch.cat([(gap / 0.3).clamp(-3.0, 3.0), (heading / 0.5).clamp(-3.0, 3.0).unsqueeze(1)], dim=1)
    move = action[:, :1] > 0.5
    correction = torch.tanh(action[:, 1:]) * self.scale * move.float()
    told = self.told()[:, :3]
    if self.see == "none" or not self.turned:
      base = told
    else:
      # turned: the instruction is first seen from where the body now faces. With this alone (no learned correction)
      # one 12 N s hit while walking is followed by 4 extra steps where there are none without it (11-13 12:06): the
      # body's heading jumps with a hit and sways with every stride, and the command follows it at once. Kept only
      # to read the first checkpoints; the correction has to learn the turning itself.
      _, heading = self.progress.error()
      base = torch.cat([turn(told[:, :2], heading), told[:, 2:3]], dim=1)
    command = base + correction
    speed = torch.norm(command[:, :2], dim=1, keepdim=True)
    command = torch.cat([command[:, :2] * (FASTEST[0] / speed.clamp_min(FASTEST[0])), command[:, 2:3].clamp(-FASTEST[1], FASTEST[1])], dim=1)
    command = torch.where(move, command, torch.zeros_like(command))
    if self.generator:
      term.stream_target[:, :3] = command
    self.previous, self.moved, self.command = correction, move.float(), command
    return correction


def steered_goal(state_goal, steerer: "Steer", policy: "SteerPolicy"):
  """System 2's slow loop in front of the code head (r1wb.s2.runtime.goal_brain): `state_goal(decision)` is what it is
  told (a vision-language model's goal, or the instruction's own); what it passes on is the command it steers with."""
  every = max(policy.period // PERIOD, 1)                 # a slower System 2 holds its command over the head's decisions

  def goal(decision):
    told = state_goal(decision)
    steerer.tell(told)
    if decision % every == 0:
      with torch.no_grad():
        steerer.act(policy.best(steerer.observe()))
    return torch.cat([steerer.command, told[:, 3:]], dim=1)
  return goal


class SteerPolicy(nn.Module):
  def __init__(self, see: str = "progress", width: int = 128, spread: float = 0.0, gate: str = "told", remember: bool = False,
               mirror: bool = False):
    """spread: the corrections' exploration noise, fixed (0: learned per state). Learned, the first two attempts
    bought their reward by shrinking it (a correction that is never tried is never found useful) and, with
    corrections near zero, "move" while told to stand cost almost nothing, so the stand decision faded too.
    gate: "told" = the body moves when the instruction is to move and stands when it is to stand; System 2 chooses
    how it moves, not whether (a gate drawn anew at every 0.32 s decision stops a walk every few decisions whatever
    its probability, so nothing it tried while "moving" could be told apart: the fourth attempt's gate sank from
    0.81 to 0.57 under a reward that clearly prefers moving). "learned": the gate is part of the action.
    mirror: the correction is left-right symmetric by construction, like System 1 (r1wb.symmetric_model): it is the
    mean of the network's answer and the mirrored answer to the mirrored situation (sideways and turning numbers
    change sign). The body, System 1 and the scene are symmetric, so the best loop is; a plain network is not
    (11-13 22:37: with a symmetric System 1 the stops out of walking left needed 5 late steps, out of walking
    right 1)."""
    super().__init__()
    self.mirrored = mirror
    assert not mirror or (spread > 0.0 and gate == "told"), "the mirrored loop has a fixed spread and no learned gate"
    twist, plane = [1.0, -1.0, -1.0], [1.0, -1.0, -1.0]               # (forward, sideways, turn); (ahead, aside, heading)
    seen = twist + [1.0, 1.0, 1.0, 1.0] + twist                         # told, height, height flag, age, moved, previous
    seen += (plane if see != "none" else []) + (plane if (remember and see != "none") else [])
    seen += (twist + [1.0, -1.0, 1.0] + [-1.0, 1.0, -1.0]) if see == "full" else []   # own twist, gravity, turning rates
    full = twist + [1.0, 1.0, 1.0, 1.0] + twist + plane + (plane if remember else [])
    full += twist + [1.0, -1.0, 1.0] + [-1.0, 1.0, -1.0] + plane + [1.0]              # ... owed progress, time since a hit
    self.register_buffer("seen_sign", torch.tensor(seen), persistent=False)
    self.register_buffer("full_sign", torch.tensor(full), persistent=False)
    self.register_buffer("mean_sign", torch.tensor(twist), persistent=False)
    self.see, self.fixed, self.period, self.gate, self.leak, self.remember, self.turned = see, spread, PERIOD, gate, 0.0, remember, False
    extra = 3 if (remember and see != "none") else 0
    self.body = nn.Sequential(nn.Linear(SEE[see] + extra, width), nn.ELU(), nn.Linear(width, width), nn.ELU())
    self.mean, self.spread, self.move = nn.Linear(width, 3), nn.Linear(width, 3), nn.Linear(width, 1)
    self.judge = nn.Sequential(nn.Linear(SEE["full"] + (3 if remember else 0) + 4, 256), nn.ELU(), nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1))
    nn.init.zeros_(self.mean.weight); nn.init.zeros_(self.mean.bias)       # starts with no correction
    nn.init.zeros_(self.spread.weight); nn.init.constant_(self.spread.bias, math.log(0.3))
    nn.init.zeros_(self.move.weight); nn.init.constant_(self.move.bias, 1.5)

  def forward(self, seen: torch.Tensor):
    """-> (whether to move: Bernoulli, the correction: Normal)."""
    hidden = self.body(seen)
    mean = self.mean(hidden)
    if self.mirrored:
      mean = 0.5 * (mean + self.mean(self.body(seen * self.seen_sign)) * self.mean_sign)
    spread = torch.full_like(mean, self.fixed) if self.fixed > 0.0 else self.spread(hidden).clamp(math.log(0.02), math.log(1.0)).exp()
    if self.gate == "told":
      go = 40.0 * (seen[:, :3].abs().sum(dim=1) > 1.0e-6).float() - 20.0
    else:
      go = self.move(hidden).squeeze(-1)
    return torch.distributions.Bernoulli(logits=go), torch.distributions.Normal(mean, spread)

  def sample(self, seen: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    go, shift = self(seen)
    move, correction = go.sample(), shift.sample()
    # A correction that is not played (stand) is not part of the action taken.
    return torch.cat([move.unsqueeze(1), correction], dim=1), go.log_prob(move) + move * shift.log_prob(correction).sum(dim=1)

  def evaluate(self, seen: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    go, shift = self(seen)
    move = action[:, 0]
    return (go.log_prob(move) + move * shift.log_prob(action[:, 1:]).sum(dim=1),
            go.entropy() + go.probs * shift.entropy().sum(dim=1))

  @torch.no_grad()
  def best(self, seen: torch.Tensor) -> torch.Tensor:
    go, shift = self(seen)
    return torch.cat([(go.probs > 0.5).float().unsqueeze(1), shift.mean], dim=1)

  def value(self, everything: torch.Tensor) -> torch.Tensor:
    if self.mirrored:
      return 0.5 * (self.judge(everything) + self.judge(everything * self.full_sign)).squeeze(-1)
    return self.judge(everything).squeeze(-1)

  @staticmethod
  def load(path, device: str) -> "SteerPolicy":
    saved = torch.load(path, map_location=device, weights_only=False)
    mirror = bool(saved["meta"].get("mirror", False))
    policy = SteerPolicy(saved["meta"]["see"], gate=saved["meta"].get("gate", "learned"), spread=0.3 if mirror else 0.0,
                         remember=bool(saved["meta"].get("remember", False)), mirror=mirror).to(device).eval()
    policy.load_state_dict({key: value for key, value in saved["state"].items() if not key.startswith("judge.")}, strict=False)
    policy.period = int(saved["meta"].get("period", PERIOD))
    policy.leak = float(saved["meta"].get("leak", 0.0))
    policy.turned = bool(saved["meta"].get("turned", True))        # checkpoints before 11-13 12:10 were trained with it
    return policy


class SteerEnv:
  """One step = one decision. System 1's own training scene (its disturbances, randomisation and terminations),
  every environment on a command stream, System 1 frozen."""

  def __init__(self, task: str, system1: Path, envs: int, device: str = "cuda:0", seed: int = 0, see: str = "progress",
               calm: bool = False, period: int = PERIOD, fall_cost: float = 3.0, calm_share: float = 0.5,
               size_cost: float = 0.03, leak: float = 0.0, forgive: float = 0.0, remember: bool = False,
               change_cost: float = 0.0, library: str = ""):
    """forgive (seconds): for this long after a hit, lost progress is not counted in the reward's measure.
    library: the motion library the scene's course generator plays (library_v2_<name>.npz; "" = the task's own)."""
    """period: control steps per decision. The course generator looks its library up once per piece (32 steps) and
    whenever the instruction changes: a command that is replaced before the next look-up never reaches the course."""
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from r1wb.student.observations import load_teacher
    from r1wb.tasks import TASKS
    cfg = TASKS[task]["env"](play=False)
    cfg.commands["motion"].stream_period, cfg.commands["motion"].stream_share = 1, 1
    if library:
      from r1wb.tracking.env_cfg import library_path
      cfg.commands["motion"].motion_file = str(library_path(library))
    if calm:
      cfg.events.pop("impulse", None)
    elif "impulse" in cfg.events:          # System 1 trained with four environments in five under hits
      cfg.events["impulse"].params["calm_fraction"] = calm_share
    cfg.scene.num_envs, cfg.seed = envs, seed
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.wrapped = RslRlVecEnvWrapper(self.env)
    self.term = self.env.command_manager.get_term("motion")
    self.device, self.count = device, envs
    self.tracker = load_teacher(str(system1), device)
    self.steer = Steer(self.env, self.term, see, leak=leak, remember=remember)
    self.period, self.fall_cost, self.size_cost, self.forgive, self.change_cost = period, fall_cost, size_cost, forgive, change_cost
    self.air = torch.zeros(envs, 2, device=device)
    self.seen = None

  def since_hit(self) -> torch.Tensor:
    clock = getattr(self.env, "hit_clock", None)
    return torch.full((self.count,), 1.0e3, device=self.device) if clock is None else clock

  def observation(self) -> tuple[torch.Tensor, torch.Tensor]:
    """(what System 2 sees, what the critic sees: everything, the owed progress and the time since a hit)."""
    owed, turn_owed = self.steer.progress.owed_error()
    extra = torch.cat([(owed / 0.3).clamp(-3.0, 3.0), (turn_owed / 0.5).clamp(-3.0, 3.0).unsqueeze(1),
                       (self.since_hit() / 3.0).clamp(max=1.0).unsqueeze(1)], dim=1)
    return self.steer.observe(), torch.cat([self.steer.observe("full"), extra], dim=1)

  def reset(self):
    self.env.reset()
    self.seen = self.wrapped.get_observations()
    everyone = torch.arange(self.count, device=self.device)
    self.steer.progress.begin(everyone)
    self.steer.progress.shadow[:] = self.term.stream[:, :3]
    return self.observation()

  def step(self, action: torch.Tensor):
    """Returns (observation, everything), reward, ended, info."""
    from tensordict import TensorDict
    env, term, progress = self.env, self.term, self.steer.progress
    before = self.steer.previous.clone()
    correction = self.steer.act(action)
    zeros = lambda: torch.zeros(self.count, device=self.device)  # noqa: E731
    follow, still, apart, gap_sum, turn_sum, walked, stood = zeros(), zeros(), zeros(), zeros(), zeros(), zeros(), zeros()
    ended = torch.zeros(self.count, dtype=torch.bool, device=self.device)
    fell = ended.clone()
    for _ in range(self.period):
      with torch.no_grad():
        target = self.tracker(TensorDict({"actor": self.seen["actor"]}, batch_size=[self.count]))
        self.seen, _, dones, _ = self.wrapped.step(target)
      dones = dones.bool()
      fell |= dones & env.termination_manager.terminated & ~ended
      ended |= dones
      progress.after_step(self.since_hit() >= self.forgive if self.forgive > 0.0 else None)
      alive = (~ended).float()
      gap, heading = progress.owed_error()
      distance, heading = torch.norm(gap, dim=1), heading.abs()
      moving = torch.norm(self.steer.told()[:, :3], dim=1) > 1.0e-6
      # Facing the right way is worth something only where the body is (a body that stays put keeps its heading:
      # paid for by itself, that made standing the better answer to every slow instruction - the third attempt
      # moved on 0.75 of the "move" decisions and not at all at 0.2 m/s).
      follow += alive * moving.float() * (torch.exp(-distance / 0.2) * (0.6 + 0.4 * torch.exp(-heading / 0.3))
                                           - distance.clamp(max=2.0) - 0.25 * heading)
      standing = ~moving & (progress.age > 1.0)
      contact = env.scene["feet_ground"].data.found.reshape(self.count, -1)[:, :2] > 0
      landing = (contact & (self.air >= 3)).float().sum(1)
      self.air = torch.where(contact, torch.zeros_like(self.air), self.air + 1)
      speed = env.scene["robot"].data.joint_vel.abs().mean(dim=1)
      still += alive * standing.float() * (4.0 * speed + 0.5 * landing)
      apart += alive * (term.robot_joint_pos - term.joint_pos).abs().mean(dim=1)
      gap_sum += alive * moving.float() * distance
      turn_sum += alive * moving.float() * heading
      walked += alive * moving.float()
      stood += alive * standing.float() * (speed + landing)
    self.parts = dict(follow=follow / self.period, still=-still / self.period, course=-2.0 * apart / self.period,
                      size=-self.size_cost * (correction / self.steer.scale).pow(2).sum(dim=1)
                           - self.change_cost * ((correction - before) / self.steer.scale).pow(2).sum(dim=1),
                      fall=-self.fall_cost * fell.float())
    reward = 1.0 + sum(self.parts.values())
    restarted = ended.nonzero(as_tuple=False).flatten()
    if len(restarted):
      self.steer.restart(restarted)
      self.air[restarted] = 0.0
    info = dict(fell=fell, gap=gap_sum, heading=turn_sum, walked=walked, stood=stood, correction=correction)
    return self.observation(), reward, ended, info


def train(args) -> None:
  env = SteerEnv(args.task, args.system1, args.envs, args.device, args.seed, args.see, args.calm, args.period, args.fall_cost,
                 args.calm_share, args.size_cost, args.leak, args.forgive, args.remember, args.change_cost, args.library)
  policy = SteerPolicy(args.see, spread=args.spread, gate=args.gate, remember=args.remember, mirror=args.mirror).to(args.device)
  if args.start is not None:                                   # goes on from a saved loop (its critic too)
    policy.load_state_dict(torch.load(args.start, map_location=args.device, weights_only=False)["state"])
  optimiser = torch.optim.Adam(policy.parameters(), lr=args.rate)
  seen, everything = env.reset()
  gamma, lam, horizon = 0.97, 0.95, args.horizon
  began = time.time()
  for iteration in range(args.iterations):
    store = {key: [] for key in ("seen", "everything", "action", "logp", "value", "reward", "ended")}
    sums = {key: 0.0 for key in ("fell", "gap", "heading", "walked", "stood", "correction", "moved", "told_move", "told_stand",
                                 "moved_told_move", "moved_told_stand")}
    with torch.no_grad():
      for _ in range(horizon):
        action, logp = policy.sample(seen)
        store["seen"].append(seen); store["everything"].append(everything); store["action"].append(action)
        store["logp"].append(logp); store["value"].append(policy.value(everything))
        (seen, everything), reward, ended, info = env.step(action)
        store["reward"].append(reward); store["ended"].append(ended.float())
        for key in ("fell", "gap", "heading", "walked", "stood"):
          sums[key] += float(info[key].float().sum())
        sums["correction"] += float(info["correction"].abs().mean())
        if args.see != "none":                 # does the mean correction follow the gap? (the hand-made probe: 0.67 per second)
          asked = torch.tanh(policy(store["seen"][-1])[1].mean) * env.steer.scale
          gap = torch.cat([store["seen"][-1][:, 10:12] * 0.3, store["seen"][-1][:, 12:13] * 0.5], dim=1)
          walking = (store["seen"][-1][:, :3].abs().sum(dim=1, keepdim=True) > 1.0e-6).float()
          sums["slope_top"] = sums.get("slope_top", 0.0) + (asked * gap * walking).sum(dim=0)
          sums["slope_bottom"] = sums.get("slope_bottom", 0.0) + (gap * gap * walking).sum(dim=0)
        told_move = torch.norm(env.steer.told()[:, :3], dim=1) > 1.0e-6
        sums["told_move"] += float(told_move.sum()); sums["told_stand"] += float((~told_move).sum())
        sums["moved_told_move"] += float((action[:, 0] * told_move).sum()); sums["moved_told_stand"] += float((action[:, 0] * ~told_move).sum())
      last = policy.value(everything)
    advantage, running = [], torch.zeros(env.count, device=args.device)
    for t in reversed(range(horizon)):
      following = last if t == horizon - 1 else store["value"][t + 1]
      alive = 1.0 - store["ended"][t]
      delta = store["reward"][t] + gamma * following * alive - store["value"][t]
      running = delta + gamma * lam * alive * running
      advantage.insert(0, running)
    flat = {key: torch.stack(value).reshape(-1, *value[0].shape[1:]) for key, value in store.items()}
    gain = torch.stack(advantage).reshape(-1)
    goal = gain + flat["value"]
    gain = (gain - gain.mean()) / (gain.std() + 1.0e-6)
    for _ in range(args.epochs):
      for rows in torch.randperm(len(gain), device=args.device).split(len(gain) // args.batches):
        logp, entropy = policy.evaluate(flat["seen"][rows], flat["action"][rows])
        ratio = torch.exp(logp - flat["logp"][rows])
        loss = (-torch.minimum(ratio * gain[rows], ratio.clamp(0.8, 1.2) * gain[rows]).mean()
                + 0.5 * (policy.value(flat["everything"][rows]) - goal[rows]).pow(2).mean()
                - args.entropy * entropy.mean())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimiser.step()
    if iteration % args.report == 0 or iteration == args.iterations - 1:
      seconds = horizon * env.count * env.period * float(env.env.step_dt)
      steps = max(sums["walked"], 1.0)
      print(f"iteration {iteration:4d} reward/decision {float(flat['reward'].mean()):+.3f}  while told to move: path gap "
            f"{sums['gap'] / steps:.3f} m heading {math.degrees(sums['heading'] / steps):.1f} deg  told to stand: motion "
            f"{sums['stood'] / max(horizon * env.count * env.period, 1):.4f}  |correction| {sums['correction'] / horizon:.3f}  "
            f"moves when told to move {sums['moved_told_move'] / max(sums['told_move'], 1):.2f} / to stand "
            f"{sums['moved_told_stand'] / max(sums['told_stand'], 1):.2f}  "
            + (f"correction per gap (x, y, yaw) {[round(float(v), 2) for v in (sums['slope_top'] / sums['slope_bottom'].clamp_min(1e-6))]}  "
               if "slope_top" in sums else "") +
            f"falls/100 env-s {100.0 * sums['fell'] / seconds:.3f}  {time.time() - began:.0f} s", flush=True)
    if (iteration + 1) % args.save_every == 0 or iteration == args.iterations - 1:
      torch.save(dict(state=policy.state_dict(), meta=dict(see=args.see, task=args.task, system1=str(args.system1), library=args.library,
                                                           mirror=args.mirror,
                                                           iterations=iteration + 1, envs=args.envs, calm=args.calm,
                                                           period=args.period, spread=args.spread, gate=args.gate,
                                                           leak=args.leak, forgive=args.forgive, remember=args.remember,
                                                           turned=False, change_cost=args.change_cost)), args.out)
  print("saved", args.out)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("mode", choices=["train"])
  parser.add_argument("--system1", type=Path, required=True)
  parser.add_argument("--task", required=True, help="the task System 1 was trained on (its scene and disturbances)")
  parser.add_argument("--see", choices=list(SEE), default="progress")
  parser.add_argument("--calm", action="store_true", help="no impulses")
  parser.add_argument("--period", type=int, default=PERIOD, help="control steps per decision")
  parser.add_argument("--fall-cost", type=float, default=3.0)
  parser.add_argument("--size-cost", type=float, default=0.03)
  parser.add_argument("--calm-share", type=float, default=0.5, help="share of the environments that are never hit")
  parser.add_argument("--envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=300)
  parser.add_argument("--horizon", type=int, default=24)
  parser.add_argument("--rate", type=float, default=1.0e-3)
  parser.add_argument("--epochs", type=int, default=8)
  parser.add_argument("--batches", type=int, default=8)
  parser.add_argument("--spread", type=float, default=0.3, help="fixed exploration noise of the corrections (0: learned)")
  parser.add_argument("--gate", choices=["told", "learned"], default="told")
  parser.add_argument("--leak", type=float, default=0.0, help="seconds after which lost progress is forgotten (0: never)")
  parser.add_argument("--forgive", type=float, default=0.0,
                      help="training reward only: seconds after a hit during which lost progress is not owed (0: off)")
  parser.add_argument("--remember", action="store_true", help="System 2 also sees the gap of the decision before")
  parser.add_argument("--change-cost", type=float, default=0.0, help="price on changing the correction from one decision to the next")
  parser.add_argument("--entropy", type=float, default=0.002)
  parser.add_argument("--report", type=int, default=5)
  parser.add_argument("--save-every", type=int, default=50)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--library", default="", help="the motion library of the scene (library_v2_<name>.npz), e.g. corebal")
  parser.add_argument("--mirror", action="store_true", help="the loop is left-right symmetric by construction (SteerPolicy mirror)")
  parser.add_argument("--start", type=Path, default=None, help="a saved loop of the same settings to go on from")
  parser.add_argument("--out", type=Path, required=True)
  args = parser.parse_args()
  train(args)


if __name__ == "__main__":
  main()
