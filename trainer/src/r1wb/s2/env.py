"""System 2's environment: one step is one decision.

Inside one step: the motion codes System 2 just wrote are decoded into reference frames, System 1 (a frozen
tracking policy) follows them for 16 control steps in the simulator, and the step's reward is what the body did
with the instruction during those steps. Every environment follows its own endless script of instructions that
change on the decision grid.

  observation   meaning (envs,) index into `MEANINGS` of the instruction read now (in force one decision later),
                history (envs, HISTORY, dims) the codes written last, image (envs, H, W, 3) uint8 if a camera is set
  action        codes (envs, AHEAD, dims), values 0 .. levels-1; goal (envs, 5) or None (then the instruction's own)
  reward        sum over the 16 control steps of the scene's reward (how well the commanded twist and height are
                met, joint limits, self collision), minus `still_cost` x joint speed while the instruction says
                stand, minus `fall_cost` when the robot falls
"""
from __future__ import annotations

import os
from pathlib import Path
import random

os.environ.setdefault("MUJOCO_GL", "egl")

import torch  # noqa: E402
import warp as wp  # noqa: E402

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

from mjlab.envs import ManagerBasedRlEnv  # noqa: E402
from mjlab.rl import RslRlVecEnvWrapper  # noqa: E402
from tensordict import TensorDict  # noqa: E402

from r1wb.s2 import instructions  # noqa: E402
from r1wb.s2.runtime import S2View  # noqa: E402
from r1wb.s2.small import goal_of  # noqa: E402
from r1wb.student.observations import load_teacher  # noqa: E402
from r1wb.student.stream import stream_env_cfg  # noqa: E402
from r1wb.tracking.env_cfg import teacher_observation  # noqa: E402

MEANINGS = instructions.every_instruction()
NAMES = [instructions.identifier(item) for item in MEANINGS]
PERIOD = 16


class S2Env:
  def __init__(self, system1: Path, tokenizer, envs: int, device: str = "cuda:0", seed: int = 0, camera: int = 0,
               still_cost: float = 0.5, fall_cost: float = 20.0, segments: tuple[int, int] = (8, 19),
               expert: bool = False):
    """segments: an instruction lasts this many decisions (2.56 to 6.08 s).
    expert: the simulator's own reference generator runs next to System 2, matched to the robot's state (it looks
    up the library from where the robot is, every 1.2 s, on a new command and at a clip's end): `expert_codes` then
    gives the codes of its continuation, the answer to "what should be written from here" (for DAgger)."""
    cfg = stream_env_cfg(play=False, clean=True, matched=True, teacher=None)
    cfg.commands["motion"].match_pose = "robot" if expert else "reference"
    cfg.commands["motion"].match_commit = True
    cfg.commands["motion"].match_height = "target"       # height changes where the command changes (no hidden clock)
    cfg.commands["motion"].match_stop = "arrive"         # ... and a stop is one piece of motion that ends standing
    cfg.events.pop("push_robot", None)                       # System 1 handles disturbances; none while System 2 learns
    cfg.scene.num_envs, cfg.seed = envs, seed
    if camera:
      from r1wb.s2.collect import head_camera
      cfg.scene.sensors = tuple(cfg.scene.sensors) + (head_camera(camera),)
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.wrapped = RslRlVecEnvWrapper(self.env)
    self.term = self.env.command_manager.get_term("motion")
    self.device, self.count, self.camera, self.expert = device, envs, camera, expert
    self.tokenizer = tokenizer
    self.tracker = load_teacher(str(system1), device)
    self.joint_action = self.env.action_manager.get_term("joint_pos")
    self.still_cost, self.fall_cost, self.segments = still_cost, fall_cost, segments
    self.rng = random.Random(seed)
    self.rows = torch.tensor([instructions.command_row(item, 1.0) for item in MEANINGS], device=device)   # (meanings, 8)
    self.goals = torch.tensor([goal_of(item) for item in MEANINGS], device=device)
    self.stand = NAMES.index("stand")
    # The script as it unfolds: the instruction in force now, the one that follows, and when (decision index).
    self.now = torch.full((envs,), self.stand, dtype=torch.long, device=device)
    self.following = self.now.clone()
    self.change_at = torch.zeros(envs, dtype=torch.long, device=device)
    self.crouched = torch.zeros(envs, dtype=torch.bool, device=device)
    self.decision = 0
    self._codes = self._goal = None
    self.view = None

  # -- instructions -----------------------------------------------------------------------------------------------

  def _draw(self, ids: torch.Tensor, start: int, span: tuple[int, int] | None = None) -> None:
    """Choose what follows the instruction now in force, and the decision at which it takes effect."""
    for i in ids.tolist():
      item = instructions.sample(self.rng, bool(self.crouched[i]))
      self.following[i] = NAMES.index(instructions.identifier(item))
      self.change_at[i] = start + self.rng.randint(*(span or self.segments))

  def _apply(self, decision: int) -> None:
    """Instructions that take effect at `decision` enter the simulator's command stream. Called before the last
    control step of the decision before, so the change shows from that decision's first control step on (as in
    the recorded courses)."""
    due = (self.change_at <= decision).nonzero(as_tuple=False).flatten()
    if len(due) == 0:
      return
    self.now[due] = self.following[due]
    row = self.rows[self.now[due]]
    term = self.term
    term.stream_target[due] = row[:, [1, 2, 3, 4, 6]]
    term.goal_mask[due, 1], term.goal_mask[due, 2] = row[:, 5], row[:, 7]
    term.since_change[due] = -self.env.step_dt                # reads 0 after the coming control step
    kind = self.now[due]
    crouch = torch.tensor([NAMES[k].startswith("crouch") for k in kind.tolist()], device=self.device)
    rise = torch.tensor([NAMES[k] == "rise" for k in kind.tolist()], device=self.device)
    self.crouched[due] = torch.where(crouch, True, torch.where(rise, False, self.crouched[due]))
    self._draw(due, decision)

  def _brain(self, decision, history):
    return self._codes, self._goal

  def expert_codes(self) -> torch.Tensor:
    """(envs, AHEAD, dims), values 0..levels-1: what the reference generator would play from here, as codes for
    the windows after the one that is playing.

    expert=True at construction: the generator runs by itself, matched to the robot (its own clips and timing).
    Otherwise ("from the reference"): the generator is put on the library frame closest to the reference System 2
    is playing right now and asked as it asks itself: keep playing that clip, unless 1.2 s have passed on it, it
    ends, or the command is "stand" and it stands - then look up the library for the command from that frame's
    pose. Its answer therefore continues what System 2 is doing when that fits the command and leaves it when it
    does not (a held crouch under "rise")."""
    from r1wb.s2 import frames as s2f
    from r1wb.s2.small import AHEAD
    term, stride = self.term, self.tokenizer.stride
    if not self.expert:
      reference, index = self.view._dense()
      now = reference.frames[torch.arange(self.count, device=self.device), (index + reference.lead).clamp(max=reference.last)]
      weight = self._project_weight
      distance = (self._project_norm.unsqueeze(0) - 2.0 * (now * weight) @ self._project_table.T)
      spot = self._project_index[distance.argmin(dim=1)]
      clip = term.frame_clip[spot]
      same = (clip == self._last_clip) & (spot >= self._last_spot) & (spot - self._last_spot <= 2 * PERIOD)
      self._on_clip = torch.where(same, self._on_clip + PERIOD, torch.zeros_like(self._on_clip))
      term.time_steps[:], term.clip_ids[:] = spot, clip
      hold = term._standing()
      ending = (term.clip_end[clip] - spot) <= term.cfg.match_end_margin
      due = ((self._on_clip >= term.cfg.travel_window) | ending) & ~hold
      ids = due.nonzero(as_tuple=False).flatten()
      if len(ids):
        term._rematch(ids, ending[ids])
        moved = ids[term.time_steps[ids] != spot[ids]]
        self._on_clip[moved] = 0
      term.hold = hold
      self._last_spot, self._last_clip = term.time_steps.clone(), term.clip_ids.clone()
    frame, last = term.time_steps, term.clip_end[term.clip_ids]
    first = term.clip_start[term.clip_ids]
    ahead = torch.arange(-2 * stride, (AHEAD + 3) * stride, device=self.device).unsqueeze(0)
    ahead = torch.where(term.hold.unsqueeze(1), torch.zeros_like(ahead), ahead)
    index = torch.maximum(torch.minimum(frame.unsqueeze(1) + ahead, last.unsqueeze(1)), first.unsqueeze(1))
    run = self.view._table[index].clone()
    stays = torch.cat([index[:, 1:] == index[:, :-1], torch.ones(self.count, 1, dtype=torch.bool, device=self.device)], dim=1)
    run[:, :, s2f.STEP] = run[:, :, s2f.STEP] * (~stays).float().unsqueeze(2)
    with torch.no_grad():
      codes = self.tokenizer.encode(run) + self.tokenizer.levels // 2
    return codes[:, 3:3 + AHEAD]

  def steady(self) -> torch.Tensor:
    """Environments whose instruction read now is the one already in force (no change within the next decision)."""
    return self.change_at > self.decision + 1

  def _restart(self, ids: torch.Tensor) -> None:
    """A fallen or timed-out environment begins again standing; its next instruction comes on the grid."""
    self.now[ids], self.crouched[ids] = self.stand, False
    self.term.segment_left[ids] = 1.0e9                      # the script table is not used: _apply drives the stream
    self._draw(ids, self.decision, (2, 6))

  # -- interface --------------------------------------------------------------------------------------------------

  def observation(self) -> dict:
    out = {"meaning": torch.where(self.change_at <= self.decision + 1, self.following, self.now),
           "history": self.view._history.clone()}
    if self.camera:
      out["image"] = self.env.scene["head_camera"].data.rgb.clone()
    return out

  def _control_step(self):
    self.view.update()
    with torch.no_grad():
      seen = teacher_observation(self.view.env, "motion", self.joint_action.plain_action, local=True,
                                 width=self.tracker.obs_dim)
      action = self.joint_action.from_plain_action(self.tracker(TensorDict({"actor": seen}, batch_size=[self.count])))
      _, reward, dones, extras = self.wrapped.step(action)
    return reward, dones.bool(), extras

  def reset(self) -> dict:
    self.env.reset()
    self.wrapped.get_observations()
    self.term.segment_left[:] = 1.0e9
    self.term.stream_target[:] = self.rows[self.stand][[1, 2, 3, 4, 6]]
    self.decision = 0
    everyone = torch.arange(self.count, device=self.device)
    self.now[:], self.crouched[:] = self.stand, False
    self._draw(everyone, 0, (2, 6))
    self.view = S2View(self.env, self.term, self._brain, self.tokenizer)
    # For expert_codes "from the reference": the library frames a reference frame is compared with (every second
    # frame of the clips the generator may choose from), joints, height and tilt on one scale, the step ten-fold.
    table = self.view._table
    weight = torch.ones(table.shape[1], device=self.device)
    weight[24] = 2.0
    weight[27:30] = 10.0
    self._project_weight = weight
    self._project_index = self.term._sets[0][0]
    self._project_table = table[self._project_index] * weight
    self._project_norm = self._project_table.pow(2).sum(1)
    self._last_spot = torch.zeros(self.count, dtype=torch.long, device=self.device)
    self._last_clip = torch.full((self.count,), -1, dtype=torch.long, device=self.device)
    self._on_clip = torch.zeros(self.count, dtype=torch.long, device=self.device)
    self._control_step()                                     # control step 0: before the first decision
    return self.observation()

  def step(self, codes: torch.Tensor, goal: torch.Tensor | None = None):
    """Returns observation, reward (envs,), fell (envs,), ended (envs,) (fell or timed out), info."""
    self._codes = codes
    self._goal = self.goals[self.observation()["meaning"]] if goal is None else goal
    reward = torch.zeros(self.count, device=self.device)
    fell = torch.zeros(self.count, dtype=torch.bool, device=self.device)
    ended = fell.clone()
    speed = torch.zeros(self.count, device=self.device)
    for index in range(PERIOD):
      if index == PERIOD - 1:
        self._apply(self.decision + 1)
      step_reward, dones, _ = self._control_step()
      alive = (~ended).float()
      reward += step_reward * alive
      quiet = (torch.norm(self.term.stream_target[:, :3], dim=1) < 1.0e-6) & (self.term.since_change > 1.0)
      speed += self.env.scene["robot"].data.joint_vel.abs().mean(dim=1) * quiet.float() * alive
      fell |= dones & self.env.termination_manager.terminated & ~ended
      ended |= dones
    reward = reward - self.still_cost * speed * self.env.step_dt - self.fall_cost * fell.float()
    self.decision += 1
    restarted = ended.nonzero(as_tuple=False).flatten()
    if len(restarted):
      self._restart(restarted)
      self.view._start(restarted)
    return self.observation(), reward, fell, ended, {"quiet_speed": speed / PERIOD}
