"""R2c Hold teacher with the adversarial hit curriculum (coordinator 2026-11-17 13:0x KST, after the one-factor runs;
rounds/R02/positive/one_factor.md). T form: the actor reads the noise-free 570 frames; the locked objective (c672a53b,
beta = 1) and the R1 PPO recipe (entropy 0.005) are unchanged.

Every episode is either
  anchor   (probability 1 - level_share): no hit; start spread of the canonical training distribution (the R1 default
           reset, or with the file's human_fraction a uniformly drawn human standing frame, yaw uniform)
  level    (probability level_share): one level drawn by weight from the negative's current level bundle
           (r1adv.negative.level_feed, family r2-hit-v1): its start ("default" or a frame index of the same 1,132
           frames) and its 1-2 torso impulses (time from the reset, impulse, azimuth relative to the heading, height
           above the torso CoM; one policy step, the verdict runner's application)
The bundle is read from the feed directory (environment variable R1ADV_FEED_DIR at configuration time; the path is
written into the event parameters, so env.yaml carries it together with the manifest sha256 seen at that time). The
reset event looks at the manifest every CHECK_EVERY_STEPS environment steps; a new tick is loaded only when its
sha256 matches (LevelFeed.refresh), so a bundle is swapped atomically between resets. level_share comes from the
bundle (capped at 0.5); before the first bundle every episode is an anchor episode. Each swap is printed to the
training log ([r2c] feed tick ...), and every step logs Curriculum/tick, Curriculum/level_share and
Curriculum/level_episodes (share of environments whose current episode is a level episode)."""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

from r1adv.positive import hold_r2 as R2

FEED_ENV = "R1ADV_FEED_DIR"
CHECK_EVERY_STEPS = 240          # 10 PPO updates of 24 steps
LEVEL_SHARE_CAP = 0.5
MAX_HITS = 5                     # level episodes use up to 2; R2e base episodes up to 5 (10 s, first >= 1.5 s, gaps >= 2 s)
_STATE: dict[int, "_Curriculum"] = {}


class _Curriculum:
  """State shared by the reset event (writes) and the step event (reads) of one environment."""

  def __init__(self, env, feed_dir: str):
    from r1adv.negative.level_feed import LevelFeed
    n, dev = env.num_envs, env.device
    self.hits = torch.zeros(n, MAX_HITS, 4, device=dev)          # t_s, impulse N s, azimuth (rel. heading), height m
    self.nhits = torch.zeros(n, dtype=torch.long, device=dev)
    self.fired = torch.zeros(n, MAX_HITS, dtype=torch.bool, device=dev)
    self.level = torch.zeros(n, dtype=torch.bool, device=dev)
    self.episode = torch.zeros(n, dtype=torch.long)             # per-env episode counter (the feed's draw key)
    self.feed = LevelFeed(Path(feed_dir))
    self.seed = int(env.cfg.seed)
    self.share, self.tick, self.sha, self.last_check = 0.0, -1, "", -10 ** 9


class curriculum_start:
  """Reset event (after the R1 reset terms): anchor or level per resetting env, start placement, hit schedule."""

  def __init__(self, cfg, env):
    p = cfg.params
    data = np.load(p["frames_file"])
    dev = env.device
    self.q = torch.tensor(data["joint_pos"], dtype=torch.float, device=dev)
    self.tilt = torch.tensor(data["tilt_quat"], dtype=torch.float, device=dev)
    self.z = torch.tensor(data["pelvis_z"], dtype=torch.float, device=dev)
    self.asset = env.scene["robot"]
    self.g = torch.Generator(device=dev)
    self.g.manual_seed(int(env.cfg.seed) * 7919 + 404)
    self.state = _Curriculum(env, p["feed_dir"])
    _STATE[id(env)] = self.state

  def reset(self, env_ids=None):
    pass

  def _refresh(self, env):
    st = self.state
    step = int(env.common_step_counter)
    if step - st.last_check < CHECK_EVERY_STEPS:
      return
    st.last_check = step
    if st.feed.refresh():
      st.tick, st.sha = int(st.feed.tick), str(st.feed.sha256)
      st.share = min(float(st.feed.bundle.get("distribution", {}).get("level_share", 0.0)), LEVEL_SHARE_CAP)
      print(f"[r2c] feed tick {st.tick} sha256 {st.sha} levels {len(st.feed.bundle['levels'])} level_share {st.share} "
            f"env_step {step}", flush=True)

  def _place(self, env, ids, k):
    dev = env.device
    yaw = (torch.rand(len(ids), device=dev, generator=self.g) * 2.0 - 1.0) * math.pi
    turn = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], 1)
    quat = R2._quat_mul(turn, self.tilt[k])
    state = torch.zeros(len(ids), 13, device=dev)
    state[:, :3] = env.scene.env_origins[ids]
    state[:, 2] += self.z[k] + self.clearance
    state[:, 3:7] = quat / quat.norm(dim=1, keepdim=True)
    self.asset.write_root_state_to_sim(state, env_ids=ids)
    self.asset.write_joint_state_to_sim(self.q[k], torch.zeros_like(self.q[k]), env_ids=ids)

  def __call__(self, env, env_ids, frames_file: str, feed_dir: str, anchor_human_fraction: float, clearance_m: float,
               manifest_sha256: str = "", code_sha256: str = "", distribution_sha256: str = "", frames_sha256: str = "",
               base_hits: dict | None = None, share_fixed: float | None = None, level_start_default: bool = False):
    if env_ids is None or len(env_ids) == 0:
      return
    self.clearance = float(clearance_m)
    self._refresh(env)
    st, dev = self.state, env.device
    ids = env_ids if torch.is_tensor(env_ids) else torch.as_tensor(env_ids, device=dev)
    n = len(ids)
    st.hits[ids] = 0.0
    st.nhits[ids] = 0
    st.fired[ids] = False
    is_level = torch.zeros(n, dtype=torch.bool, device=dev)
    share = st.share if share_fixed is None else (float(share_fixed) if (st.feed.bundle is not None and st.feed.bundle["levels"]) else 0.0)
    if st.feed.bundle is not None and share > 0:
      is_level = torch.rand(n, device=dev, generator=self.g) < share
    if base_hits is not None:                                   # R2e: base episodes take the R2 distribution's hits
      base = (~is_level).nonzero().flatten()
      if len(base):
        k = len(base)
        u = lambda lo, hi, *shape: lo + (hi - lo) * torch.rand(*shape, device=dev, generator=self.g)
        t = u(*base_hits["first_s"], k)
        sched = torch.zeros(k, MAX_HITS, 4, device=dev)
        for i in range(MAX_HITS):
          sched[:, i, 0] = t
          sched[:, i, 1] = u(*base_hits["impulse_ns"], k)
          sched[:, i, 2] = u(0.0, 2.0 * math.pi, k)
          sched[:, i, 3] = u(*base_hits["height_m"], k)
          t = t + u(*base_hits["gap_s"], k)
        st.hits[ids[base]] = sched
        st.nhits[ids[base]] = MAX_HITS                          # slots beyond the episode end never fire
    frame = torch.full((n,), -1, dtype=torch.long, device=dev)
    anchor_human = ~is_level & (torch.rand(n, device=dev, generator=self.g) < float(anchor_human_fraction))
    if anchor_human.any():
      frame[anchor_human] = torch.randint(0, len(self.q), (int(anchor_human.sum()),), device=dev, generator=self.g)
    pos = is_level.nonzero().flatten()
    if len(pos):
      lv_ids = ids[pos].tolist()
      ep = st.episode[ids[pos].cpu()].tolist()
      levels = st.feed.sample(st.seed, lv_ids, ep)
      arr = np.zeros((len(pos), MAX_HITS, 4), dtype=np.float32)
      nh = np.zeros(len(pos), dtype=np.int64)
      fr = np.full(len(pos), -1, dtype=np.int64)
      for j, lv in enumerate(levels):
        hl = lv["env"]["hits"][:MAX_HITS]
        for i, h in enumerate(hl):
          if h.get("kind", "impulse") != "impulse" or int(h.get("burst_count", 1)) != 1 or abs(float(h.get("duration_s", 0.02)) - 0.02) > 1e-9:
            raise ValueError(f"R2c applies single one-step impulses only; level {lv['level_id']} has {h}")
          arr[j, i] = (float(h["t_s"]), float(h["impulse_ns"]), float(h["azimuth_rad"]), float(h["height_m"]))
        nh[j] = len(hl)
        if lv["params"].get("start") == "frame" and not level_start_default:
          fr[j] = int(lv["params"]["frame"]) % len(self.q)
      st.hits[ids[pos]] = torch.as_tensor(arr, device=dev)
      st.nhits[ids[pos]] = torch.as_tensor(nh, device=dev)
      frame[pos] = torch.as_tensor(fr, device=dev)
    st.level[ids] = is_level
    st.episode[ids.cpu()] += 1
    sel = frame >= 0
    if sel.any():
      self._place(env, ids[sel], frame[sel])


class curriculum_hits:
  """Step event: fires the scheduled impulses of level episodes (one policy step each), clears them the step after."""

  def __init__(self, cfg, env):
    self.asset = env.scene["robot"]
    self.torso = self.asset.body_names.index("torso_link")
    self.dt = env.step_dt
    if abs(self.dt - 0.02) > 1e-9:
      raise ValueError(f"hits are held for one policy step of 0.02 s; env.step_dt is {self.dt}")
    self.active = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    self.slot = torch.arange(MAX_HITS, device=env.device)[None]

  def reset(self, env_ids=None):
    ids = torch.arange(len(self.active), device=self.active.device) if env_ids is None else env_ids
    if len(ids) == 0:
      return
    was = ids[self.active[ids]]
    if len(was):
      z = torch.zeros(len(was), 1, 3, device=self.active.device)
      self.asset.write_external_wrench_to_sim(z, z, env_ids=was, body_ids=[self.torso])
    self.active[ids] = False

  def __call__(self, env, env_ids, code_sha256: str = ""):
    st = _STATE[id(env)]
    log = env.extras.setdefault("log", {})
    log["Curriculum/tick"] = torch.tensor(float(st.tick), device=self.active.device)
    log["Curriculum/level_share"] = torch.tensor(float(st.share), device=self.active.device)
    log["Curriculum/level_episodes"] = st.level.float().mean()
    t = env.episode_length_buf.float() * self.dt
    due = (~st.fired) & (self.slot < st.nhits[:, None]) & (t[:, None] >= st.hits[:, :, 0])
    fire = due.any(1)
    clear = self.active & ~fire
    if not (fire.any() or clear.any()):
      return
    dev = self.active.device
    upd = (fire | clear).nonzero().flatten()
    forces = torch.zeros(len(upd), 1, 3, device=dev)
    torques = torch.zeros(len(upd), 1, 3, device=dev)
    sel = fire[upd]
    ids = upd[sel]
    if len(ids):
      which = due[ids].float().argmax(1)
      h = st.hits[ids, which]
      q = self.asset.data.root_link_quat_w[ids]
      heading = torch.atan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
      ang = heading + h[:, 2]
      f = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], 1) * (h[:, 1] / self.dt)[:, None]
      arm = torch.zeros_like(f)
      arm[:, 2] = h[:, 3]
      forces[sel, 0] = f
      torques[sel, 0] = torch.cross(arm, f, dim=1)
      st.fired[ids, which] = True
    self.asset.write_external_wrench_to_sim(forces, torques, env_ids=upd, body_ids=[self.torso])
    self.active = fire.clone()


PD_KP_RANGE, PD_KD_RANGE = (0.8, 1.25), (0.5, 2.0)
R2E_SHARE = 0.3                  # R2e: fixed level share (not tied to tremor; the bundle's level_share is ignored)


def hold_r2c_env_cfg(play: bool = False, feed_dir: str | None = None, pd_random: bool = False, velocity_input: bool = True,
                     r2e: bool = False):
  """T-form R2 Hold env with the curriculum events (training only; play = the plain R2 T env without events).
  R2d arms (coordinator 15:3x, coordinator_decisions_r2d.md): pd_random = per-episode actuator stiffness x
  log-uniform PD_KP_RANGE and damping x log-uniform PD_KD_RANGE on every joint (mjlab dr.pd_gains, scale of the
  deploy gains; training only, the play env keeps the deploy gains); velocity_input=False = the actor reads the 330
  frames without joint velocity (frame_stack_novel; the critic keeps its privileged inputs)."""
  from mjlab.envs.mdp import dr
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.scene_entity_config import SceneEntityCfg
  cfg = R2.hold_r2_env_cfg(actor_clean=True, play=play, human_starts=False, hit_impulse=None, velocity_input=velocity_input)
  if play:
    return cfg
  if pd_random:
    cfg.events["pd_gains"] = EventTermCfg(mode="reset", func=dr.pd_gains, params=dict(
      asset_cfg=SceneEntityCfg("robot"), kp_range=PD_KP_RANGE, kd_range=PD_KD_RANGE, distribution="log_uniform",
      operation="scale"))
  feed = feed_dir or os.environ.get(FEED_ENV, "")
  if not feed:
    raise ValueError(f"R2c needs the level-feed directory (set {FEED_ENV})")
  manifest = Path(feed) / "feed_manifest.json"
  seen = json.loads(manifest.read_text()).get("sha256", "") if manifest.exists() else ""
  dist = R2.distribution()
  starts = dist["starts"]
  extra = {}
  if r2e:                                                       # R2e curriculum arm (coordinator 17:17, decision 4)
    h = dist["hits"]
    extra = dict(base_hits=dict(impulse_ns=list(h["impulse_ns"]), height_m=list(h["height_m"]), first_s=list(h["first_s"]),
                                gap_s=list(h["gap_s"])), share_fixed=R2E_SHARE, level_start_default=True)
  cfg.events["curriculum_start"] = EventTermCfg(func=curriculum_start, mode="reset", params=dict(
    frames_file=str(R2.pinned(R2.START_FRAMES)), feed_dir=str(feed),
    anchor_human_fraction=0.0 if r2e else float(starts["human_fraction"]),
    clearance_m=float(starts["clearance_m"]), manifest_sha256=seen, code_sha256=R2.CODE_SHA256,
    distribution_sha256=R2.R2_DIST["sha256"], frames_sha256=R2.START_FRAMES["sha256"], **extra))
  cfg.events["curriculum_hits"] = EventTermCfg(func=curriculum_hits, mode="step", params=dict(code_sha256=R2.CODE_SHA256))
  return cfg
