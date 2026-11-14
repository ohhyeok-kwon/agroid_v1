"""A task that needs the camera: "turn to face the <colour> box", learned by reinforcement at the goal level.

Three coloured boxes stand around the robot (r1wb.s2.scene). Every 0.32 s a small policy chooses one of the
instruction vocabulary's goals (stand, turn left / right, slowly / fast); the motion-code head turns the goal into a
reference and System 1 follows it, as in every other closed-loop run. The policy is not written by hand: it is
trained with PPO in that loop. What it reads is taken from the head camera's own picture (how much of the picture
has the named colour, and where), so everything it knows the vision-language model can also see; its choices are
then the answers the vision-language model is taught (r1wb.s2.train_goal --look).

  python -m r1wb.s2.look train   --system1 <model.pt> --head <small_*.pt> --tokenizer <tokenizer_*.pt> --out <look.pt>
  python -m r1wb.s2.look collect --system1 ... --head ... --tokenizer ... --policy <look.pt> --out <look_data.npz>
  python -m r1wb.s2.look score   --system1 ... --head ... --tokenizer ... --policy <look.pt>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
import warp as wp  # noqa: E402

if torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 10:
  wp.config.ptx_target_arch = 90
  wp.config.cuda_output = "ptx"

from r1wb.s2 import instructions, scene  # noqa: E402
from r1wb.s2.small import SmallS2, goal_of  # noqa: E402

CHOICES = ({"kind": "stand"}, {"kind": "turn", "side": "left", "rate": "slow"}, {"kind": "turn", "side": "left", "rate": "fast"},
           {"kind": "turn", "side": "right", "rate": "slow"}, {"kind": "turn", "side": "right", "rate": "fast"})
FACING = math.radians(15.0)                    # "faces the box": its centre within this of straight ahead (the robot
                                               # turns in steps of 15 to 20 degrees; 8 was finer than the motion)
EPISODE = 40                                   # decisions (12.8 s) with one target
WORDS = {"ko": {"red": "빨간", "green": "초록", "blue": "파란"}, "en": {"red": "red", "green": "green", "blue": "blue"}}
PHRASES = {"ko": (["{c} 상자를 봐", "{c} 상자 쪽으로 돌아서", "{c} 상자를 향해 서"], ["{c} 상자를 바라봐 줘", "{c} 상자 쪽을 봐"]),
           "en": (["turn to face the {c} box", "look at the {c} box", "face the {c} box"], ["turn towards the {c} box", "point yourself at the {c} box"])}


def phrase(colour: str, language: str, rng, held_out: bool = False) -> str:
  return rng.choice(PHRASES[language][1 if held_out else 0]).format(c=WORDS[language][colour])


def colour_features(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  """image (envs, H, W, 3) uint8, target (envs,) colour index (red, green, blue) -> (envs, 3): share of the picture
  that has the target's colour (x 10), mean column of those pixels (-1 left .. 1 right, 0 if none), 1 if any."""
  pixels = image.to(torch.int32)
  channel = torch.gather(pixels, 3, target.view(-1, 1, 1, 1).expand(-1, pixels.shape[1], pixels.shape[2], 1)).squeeze(3)
  largest_other = torch.where(torch.arange(3, device=image.device).view(1, 1, 1, 3) == target.view(-1, 1, 1, 1),
                              torch.zeros_like(pixels), pixels).amax(dim=3)
  # Clearly the target's colour: more than twice either other channel. ("Brighter than the other two together" also
  # took the bluish floor for a blue box: a policy trained on it stood still believing it saw the box, 27% of targets.)
  mask = (channel > 90) & (channel > 2 * largest_other)
  share = mask.float().mean(dim=(1, 2))
  columns = torch.linspace(-1.0, 1.0, image.shape[2], device=image.device)
  where = (mask.float().sum(dim=1) * columns).sum(dim=1) / mask.float().sum(dim=(1, 2)).clamp_min(1.0)
  seen = (share > 0.002).float()
  return torch.stack([10.0 * share, where * seen, seen], dim=1)


class LookEnv:
  def __init__(self, system1: Path, head: SmallS2, tokenizer, envs: int, device: str = "cuda:0", seed: int = 0, size: int = 224):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper

    from r1wb.s2.collect import head_camera
    from r1wb.student.observations import load_teacher
    from r1wb.student.stream import stream_env_cfg
    cfg = stream_env_cfg(play=True, clean=True, matched=True, teacher=str(system1))
    cfg.commands["motion"].match_pose, cfg.commands["motion"].match_commit = "reference", True
    stand = cfg.commands["motion"].clip_list
    cfg.commands["motion"].clip_list = tuple(stand[i % len(stand)] for i in range(envs))
    cfg.scene.num_envs, cfg.seed = envs, seed
    cfg.episode_length_s = 1.0e5                                 # targets change every EPISODE decisions; only a fall restarts
    cfg.scene.sensors = tuple(cfg.scene.sensors) + (head_camera(size),)
    cfg.scene.entities.update(scene.boxes())
    self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
    self.wrapped = RslRlVecEnvWrapper(self.env)
    self.term = self.env.command_manager.get_term("motion")
    self.device, self.count, self.head, self.tokenizer = device, envs, head, tokenizer
    self.tracker = load_teacher(str(system1), device)
    self.joint_action = self.env.action_manager.get_term("joint_pos")
    self.goals = torch.tensor([goal_of(item) for item in CHOICES], device=device)
    self.rows = torch.tensor([instructions.command_row(item, 1.0) for item in CHOICES], device=device)
    self.generator = torch.Generator(device=device).manual_seed(seed)
    self.target = torch.zeros(envs, dtype=torch.long, device=device)
    self.age = torch.zeros(envs, dtype=torch.long, device=device)
    self.choice = torch.zeros(envs, dtype=torch.long, device=device)
    self.positions = None
    self.view = None

  def _new_task(self, ids: torch.Tensor) -> None:
    placed = scene.place(self.env, self.generator, ids)
    if self.positions is None:
      self.positions = placed
    else:
      self.positions[ids] = placed
    self.target[ids] = torch.randint(len(scene.NAMES), (len(ids),), generator=self.generator, device=self.device)
    self.age[ids] = 0

  def bearing(self) -> torch.Tensor:
    _, bearing, _ = scene.seen_from(self.env, self.positions)
    return bearing[torch.arange(self.count, device=self.device), self.target]

  def observe(self) -> dict:
    image = self.env.scene["head_camera"].data.rgb
    return {"image": image, "target": self.target.clone(), "features": colour_features(image, self.target)}

  def _control_step(self):
    from tensordict import TensorDict

    from r1wb.tracking.env_cfg import teacher_observation
    self.view.update()
    with torch.no_grad():
      seen = teacher_observation(self.view.env, "motion", self.joint_action.plain_action, local=True,
                                 width=self.tracker.obs_dim)
      action = self.joint_action.from_plain_action(self.tracker(TensorDict({"actor": seen}, batch_size=[self.count])))
      _, _, dones, _ = self.wrapped.step(action)
    return dones.bool()

  def reset(self) -> dict:
    from r1wb.s2.runtime import S2View, goal_brain
    self.env.reset()
    self.wrapped.get_observations()
    self.term.segment_left[:] = 1.0e9
    self.choice[:] = 0
    self._new_task(torch.arange(self.count, device=self.device))
    self.view = S2View(self.env, self.term, goal_brain(self.head, lambda decision: self.goals[self.choice]), self.tokenizer)
    self._control_step()
    return self.observe()

  def step(self, choice: torch.Tensor):
    """choice (envs,) index into CHOICES. Returns observation, reward, ended (a new target was set), info."""
    self.choice = choice
    row = self.rows[choice]
    self.term.stream_target[:] = row[:, [1, 2, 3, 4, 6]]
    fell = torch.zeros(self.count, dtype=torch.bool, device=self.device)
    for _ in range(16):
      fell |= self._control_step() & self.env.termination_manager.terminated
    bearing = self.bearing()
    facing = bearing.abs() < FACING
    standing = choice == 0
    reward = -0.5 * bearing.abs() / math.pi - 0.02 * (~standing).float() + 1.0 * (facing & standing).float() - 5.0 * fell.float()
    self.age += 1
    ended = fell | (self.age >= EPISODE)
    info = {"bearing": bearing, "success": facing & standing, "fell": fell, "final": ended & ~fell}
    ids = ended.nonzero(as_tuple=False).flatten()
    if len(ids):
      restarted = fell.nonzero(as_tuple=False).flatten()
      if len(restarted):
        self.view._start(restarted)
      self._new_task(ids)
      self.choice[ids] = 0
    return self.observe(), reward, ended, info


class LookPolicy(nn.Module):
  def __init__(self, inputs: int = 3, width: int = 64):
    """Reads the picture's features only: what it chooses must be a function of what the vision-language model
    also sees (one picture and the words), or the same picture would have several right answers for that model."""
    super().__init__()
    self.body = nn.Sequential(nn.Linear(inputs, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
    self.choose, self.value = nn.Linear(width, len(CHOICES)), nn.Linear(width, 1)

  def forward(self, features: torch.Tensor):
    hidden = self.body(features)
    return torch.distributions.Categorical(logits=self.choose(hidden)), self.value(hidden).squeeze(-1)


def build(args):
  from r1wb.s2.tokenizer import MotionTokenizer
  tokenizer = MotionTokenizer.load(args.tokenizer, args.device)
  head = SmallS2.load(args.head, args.device)
  return LookEnv(args.system1, head, tokenizer, args.envs, args.device, args.seed)


def train(args):
  env = build(args)
  policy = LookPolicy().to(args.device)
  optimiser = torch.optim.Adam(policy.parameters(), lr=1.0e-3)
  observation = env.reset()
  gamma, lam, horizon = 0.95, 0.9, 20
  began = time.time()
  for iteration in range(args.iterations):
    store = {key: [] for key in ("features", "choice", "logp", "value", "reward", "ended")}
    success = finals = falls = 0.0
    with torch.no_grad():
      for _ in range(horizon):
        chance, value = policy(observation["features"])
        choice = chance.sample()
        store["features"].append(observation["features"]); store["choice"].append(choice)
        store["logp"].append(chance.log_prob(choice)); store["value"].append(value)
        # While learning, the world often repeats the previous choice instead (a choice that changes at random
        # every 0.32 s moves the robot nowhere: a turn needs about a second to get going, and the first run without
        # this saw 0.02 success after 100,000 decisions). The trained policy runs without it.
        repeat = torch.rand(env.count, device=args.device) < args.sticky * max(0.0, 1.0 - iteration / (0.7 * args.iterations))
        observation, reward, ended, info = env.step(torch.where(repeat, env.choice, choice))
        store["reward"].append(reward); store["ended"].append(ended.float())
        success += float((info["success"] & info["final"]).sum()); finals += float(info["final"].sum()); falls += float(info["fell"].sum())
      _, last = policy(observation["features"])
    advantage, running = [], torch.zeros(env.count, device=args.device)
    for t in reversed(range(horizon)):
      following = last if t == horizon - 1 else store["value"][t + 1]
      alive = 1.0 - store["ended"][t]
      delta = store["reward"][t] + gamma * following * alive - store["value"][t]
      running = delta + gamma * lam * alive * running
      advantage.insert(0, running)
    flat = {key: torch.stack(value).reshape(-1, *value[0].shape[1:]) for key, value in store.items()}
    gain = torch.stack(advantage).reshape(-1)
    target = gain + flat["value"]
    gain = (gain - gain.mean()) / (gain.std() + 1.0e-6)
    for _ in range(4):
      for rows in torch.randperm(len(gain), device=args.device).split(len(gain) // 4):
        chance, value = policy(flat["features"][rows])
        ratio = torch.exp(chance.log_prob(flat["choice"][rows]) - flat["logp"][rows])
        loss = (-torch.minimum(ratio * gain[rows], ratio.clamp(0.8, 1.2) * gain[rows]).mean()
                + 0.5 * (value - target[rows]).pow(2).mean() - 0.01 * chance.entropy().mean())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimiser.step()
    if iteration % 5 == 0 or iteration == args.iterations - 1:
      print(f"iteration {iteration:3d} reward/decision {float(flat['reward'].mean()):+.3f} facing and standing at the end "
            f"{success / max(finals, 1):.2f} ({int(finals)} targets) falls {int(falls)}  {time.time() - began:.0f} s", flush=True)
  torch.save(dict(state=policy.state_dict(), meta=dict(iterations=args.iterations, envs=args.envs, system1=str(args.system1),
                                                       head=str(args.head), choices=[instructions.identifier(c) for c in CHOICES])), args.out)
  print("saved", args.out)


def roll(args, decide, record: bool):
  """decide(observation) -> choice. Returns the score and (record) the pictures, targets and choices."""
  env = build(args)
  observation = env.reset()
  kept = {"image": [], "target": [], "choice": [], "bearing": []}
  success = finals = falls = 0.0
  turned = []
  for step in range(args.decisions):
    with torch.no_grad():
      choice = decide(observation)
    if record:
      kept["image"].append(observation["image"].cpu().numpy()); kept["target"].append(observation["target"].cpu().numpy())
      kept["choice"].append(choice.cpu().numpy()); kept["bearing"].append(env.bearing().cpu().numpy())
    observation, _, _, info = env.step(choice)
    success += float((info["success"] & info["final"]).sum()); finals += float(info["final"].sum()); falls += float(info["fell"].sum())
    turned.append(info["bearing"].abs()[info["final"]])
  miss = torch.cat(turned) if turned else torch.zeros(1)
  score = dict(targets=int(finals), facing_and_standing_at_the_end=round(success / max(finals, 1), 3), falls=int(falls),
               mean_final_bearing_deg=round(float(miss.mean()) * 57.3, 1), within_10_deg=round(float((miss < math.radians(10)).float().mean()), 3))
  return score, kept


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("what", choices=["train", "collect", "score"])
  parser.add_argument("--system1", type=Path, required=True)
  parser.add_argument("--head", type=Path, required=True)
  parser.add_argument("--tokenizer", type=Path, required=True)
  parser.add_argument("--policy", type=Path, default=None)
  parser.add_argument("--out", type=Path, default=None)
  parser.add_argument("--envs", type=int, default=256)
  parser.add_argument("--iterations", type=int, default=60)
  parser.add_argument("--sticky", type=float, default=0.75,
                      help="train: chance that the previous choice is repeated at the start (falls to 0 over the first 70%% of "
                           "the iterations: exploring needs it, stopping on time is spoiled by it)")
  parser.add_argument("--decisions", type=int, default=3 * EPISODE)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  torch.manual_seed(args.seed)
  if args.what == "train":
    return train(args)
  policy = LookPolicy().to(args.device)
  policy.load_state_dict(torch.load(args.policy, map_location=args.device, weights_only=False)["state"])
  policy.eval()
  score, kept = roll(args, lambda observation: policy(observation["features"])[0].probs.argmax(dim=1), record=args.what == "collect")
  print(json.dumps(score))
  if args.what == "collect":
    np.save(str(args.out) + ".images.npy", np.stack(kept["image"]))
    np.savez_compressed(args.out, target=np.stack(kept["target"]), choice=np.stack(kept["choice"]), bearing=np.stack(kept["bearing"]),
                        goals=np.array([goal_of(item) for item in CHOICES], dtype=np.float32),
                        meta=json.dumps(dict(protocol="r1wb-s2-look-v0", colours=list(scene.NAMES), score=score,
                                             choices=[instructions.identifier(c) for c in CHOICES])))
    print("saved", args.out)


if __name__ == "__main__":
  main()
