"""R8 scene task of the positive agent (positive session 13; coordinator 2026-11-18 23:0x KST): the A-GMM task (careful_walk_r7a_gmm,
Hold objective r = L + D, r6d long parents' environment) with indoor scene patches around the stand point.

User rule (11-18, verbatim): "실내 3D 스캔은 네거티브 환경용으로 충분히 많이 활용해줘. 바닥을 제외하고 그 어떠한 충돌도 허용되지 않아.
아주 확실한 경우( 소파, 책상, 벽 등 ) 에서만 자연스럽게 손을 이용해서 가볍게 중심을 잡는 정도만 허용되."

Changed from A-GMM (everything else is A-GMM's: objective, hits, push, randomisation, commands, PPO, mirror augmentation):
  scenes        the negative's library v2.2 (rounds/R04/r7_scene_lane.md 8-9: ScanNet++ vobb, ProcTHOR boxes, cut to 1.5 m), TRAIN
                split only (SO.PatchSampler refuses the holdout), sources merged (or one source), padded to one K. Per reset: with probability flat_frac the env has no
                patch (flat floor, boxes parked), else a seeded draw (patch, side, rotation) placed on the robot's reset pose.
                Every draw reads only (seed, env id, reset count, k): the scenario stream is reproducible whatever the policy does.
  termination   'scene_collision': the online judge (SO.OnlineJudge, RULES_V2, window from the reset) ends the episode at the
                first collision failure, exactly like a fall (time_out False, no extra penalty: the objective has no
                termination term). Allowed light hand touches are not failures and never end an episode.
  observations  (actor and critic, appended after the deploy frames; options) rays (144): 2D free distance at body height
                (0.05-1.25 m, cap 2.5 m) as nearness 1 - d / 2.5; boxes (32): the nearest boxes in the robot frame with the
                allowed-support flag (SO.nearest_boxes, distance as nearness); hands: per hand [touching an allowed surface
                within the limits, 20 ms force / 10 %BW, 0.5 s mean force / 5 %BW, touch duration / 2 s]. All exactly 0 on a
                flat floor, so a flat-floor episode feeds the network the parent's inputs plus zeros.
  learning      ScenePPO = ParentLrPPO (A-GMM) + (a) a parent with 810 actor / 905 critic inputs is widened at load: new
                first-layer columns 0, normaliser mean 0 / std 1 on the new inputs, Adam first moment 0 and second moment =
                the row mean of the parent's columns (no un-bias-corrected step burst); so the first action equals the
                parent's on any input; (b) mirror augmentation = r1lab Mirror on the first 810 / 905 inputs + the scene mirror
                (rays j -> -j, boxes y and yaw sin negated, hands swapped).
Slot for the negative (rounds/R04/08B_r8_synthesis.md 2 (d), 08A adversarial_environment_designer): spec 'proposer' =
'module:Class' with propose(env_ids, resets, rt) -> dict | None and observe(env_ids, episode, outcome); see ScenarioProposer.
"""
from __future__ import annotations

import dataclasses
import hashlib
import importlib
import math
from pathlib import Path

import torch

from r1adv.negative import scene_online as SO
from r1adv.positive.s2f_r7a import ParentLrPPO, ParentLrPpoCfg

ACTOR_BASE = 810                         # r1lab deploy frames (C.OBS_DIM)
CRITIC_BASE = 905                        # sum of common.CRITIC_LAYOUT
CAP = 2.5                                # ray / box nearness range (m), = SceneOnline.ray_profile cap
ZLO, ZHI = 0.05, 1.25                    # body-height band of the rays (SceneOnline.ray_profile)
LIBS_V2 = (dict(file="lib_procthor_v2.json", sha256="d9a7cecc220c4cb99037255be6e2395d0aae40fbcf71db3987253f08788dacdd"),
           dict(file="lib_scannetpp_v2.json", sha256="21f6038fbc421aba45251037b5ba480d65c4a0e0c1dca9fe2fe5bdccd5ca305e"))
# v2.1 (coordinator 23:4x, r7_scene_lane.md 8): ScanNet++ vobb colliders (V-HACD 32 parts, one yaw box each), ProcTHOR v2 boxes;
# both cut to the boxes within 1.5 m of the stand point (nearest 1,536 at most). The default.
LIBS_V21 = {"procthor": dict(file="lib_procthor_v21.json", sha256="944f1f570797fe0ad80c9fdde72f77ad220366304a2de20abc9bf3fac4afc143"),
            "scannetpp": dict(file="lib_scannetpp_v21.json", sha256="ec01e59edb54ea639fafba19c09c785bcec1c8eec08b62bfda66fc35f9c0639a")}
# v2.2 (coordinator 00:2x, r7_scene_lane.md 9): v2.1 with the unlabelled floor faces dropped again (the 11 start-in-contact
# ScanNet++ patches moved or removed). The default.
LIBS_V22 = {"procthor": dict(file="lib_procthor_v22.json", sha256="ebd2483ecfc1dab4bd2bd3790741817d679428cd9b19abaf8058ac5d61548038"),
            "scannetpp": dict(file="lib_scannetpp_v22.json", sha256="e89816164612a8d8c67a88ae302ebd7f0ca606cb92c8a0c93ee51fc07833cf2e")}
SOURCES = {"both": (LIBS_V22["procthor"], LIBS_V22["scannetpp"]), "procthor": (LIBS_V22["procthor"],),
           "scannetpp": (LIBS_V22["scannetpp"],)}
INPUT_DIRS = (Path("/rl-agent/.agent/r1adv_inputs"),
              Path(__file__).resolve().parents[4] / ".agent/adversarial/negative/r7sl/out")
HAND_FEATS = 4
BOX_FEATS = 10


@dataclasses.dataclass(frozen=True)
class SceneSpec:
  libs: tuple = SOURCES["both"]
  flat_frac: float = 0.25                # share of resets without a patch (flat floor)
  side_weights: tuple = (0.35, 0.35, 0.10, 0.20)   # furniture / wall / empty / uniform side at phi (SO.PatchSampler)
  rays: int = 144                        # 0 = off; 144 (2.5 deg) and boxes 32: the smallest pair meeting the README 2.1 rule on v2.1
  boxes: int = 32                        # 0 = off
  hands: bool = True
  terminate: bool = True                 # False: judge and log only (the r6_scene_env.md 5 'other way' would add a cost)
  proposer: str = ""                     # '' = seeded uniform draws; 'module:Class' = the negative's scenario proposer
  per_geom: int = 32                     # intrusion candidates per (env, link); 32 for the small vobb boxes (r7_scene_lane.md 8.3)
  hand: str = "stock"                    # robot body: 'stock' (r1lab, A-GMM's) or 'o6' (r1lab.robot.use_o6_hands = the r1wb body)


# ---------------------------------------------------------------------------------------------------- library
def pinned(entry: dict) -> Path:
  for d in INPUT_DIRS:
    p = d / entry["file"]
    if p.is_file():
      got = hashlib.sha256(p.read_bytes()).hexdigest()
      if got != entry["sha256"]:
        raise ValueError(f"{p} sha256 {got} != pinned {entry['sha256']}")
      return p
  raise FileNotFoundError(f"{entry['file']} not found in {INPUT_DIRS}")


class MergedLibrary:
  """SO.PatchLibrary objects concatenated and padded to the largest K, with PatchLibrary's attributes (local, cls, split,
  beta, train_idx, holdout_idx, classes, K, N) so that SO.PatchSampler and SceneOnline.place_lib take it unchanged."""

  def __init__(self, entries, device="cpu", _parts=None):
    parts = _parts if _parts is not None else [SO.PatchLibrary(pinned(e), "cpu") for e in entries]
    for p in parts:
      if tuple(p.classes) != tuple(SO.RULES_V2.classes):
        raise ValueError(f"library classes {p.classes} != RULES_V2 {SO.RULES_V2.classes}")
      if p.schema != "r7sl-patchlib-v2" and not p.schema.startswith("r7sl-patchlib-v2"):
        raise ValueError(f"library schema {p.schema}: training uses v2 libraries only (v1 = the holdout baseline)")
    self.parts, self.entries = parts, tuple(dict(e) for e in entries)
    self.classes = tuple(parts[0].classes)
    self.K = max(p.K for p in parts)
    self.schema = "merged:" + "+".join(p.schema for p in parts)
    local, cls = [], []
    for p in parts:
      lo = torch.zeros(p.N, self.K, 8)
      lo[..., 3] = 1.0
      lo[:, :p.K] = p.local.cpu()
      c = torch.full((p.N, self.K), -1, dtype=torch.long)
      c[:, :p.K] = p.cls.cpu()
      local.append(lo)
      cls.append(c)
    self.local, self.cls = torch.cat(local).to(device), torch.cat(cls).to(device)
    self.split = torch.cat([p.split.cpu() for p in parts]).to(device)
    self.beta = torch.cat([p.beta.cpu() for p in parts]).to(device)
    self.source = torch.cat([torch.full((p.N,), i, dtype=torch.long) for i, p in enumerate(parts)]).to(device)
    self.ids = [i for p in parts for i in p.ids]
    self.N = len(self.ids)
    self.train_idx = (self.split == 0).nonzero().squeeze(1)
    self.holdout_idx = (self.split == 1).nonzero().squeeze(1)
    self.sha256 = [p.sha256 for p in parts]

  def to(self, device) -> "MergedLibrary":
    return MergedLibrary(self.entries, device, _parts=self.parts)

  def summary(self) -> dict:
    return dict(files=[e["file"] for e in self.entries], sha256=list(self.sha256), K=self.K, patches=self.N,
                train=int(len(self.train_idx)), holdout=int(len(self.holdout_idx)),
                train_by_source=[int(((self.source == i) & (self.split == 0)).sum()) for i in range(len(self.parts))])


_LIB_CACHE: dict = {}


def merged_library(libs) -> MergedLibrary:
  key = tuple((e["file"], e["sha256"]) for e in libs)
  if key not in _LIB_CACHE:
    _LIB_CACHE[key] = MergedLibrary(libs, "cpu")
  return _LIB_CACHE[key]


def derive_seed(seed: int, what: str) -> int:
  return int(hashlib.sha256(f"r8sc|{what}|{int(seed)}".encode()).hexdigest()[:15], 16)


# ---------------------------------------------------------------------------------------------------- proposer slot
class ScenarioProposer:
  """Slot for the negative's witness-gated adversarial sampler (interface only; the default None draws seeded uniform).

  propose(env_ids, resets, rt) -> None or dict of (m,) tensors on rt.env.device, every key optional:
    idx  long, library patch index in rt.lib (-1 = flat floor); must be a TRAIN patch (checked, raises otherwise)
    rot  float, patch-frame rotation relative to the robot heading (rad)
    phi  float, robot-frame angle (rad) where the sampler puts the drawn side (used when idx / rot are not given)
    hit  reserved for (azimuth rad, impulse N s, duration s, first time s); NOT consumed by this task yet (the hit event
         stays A-GMM's own seeded stream): a later task wires it into the hit event
  observe(env_ids, episode, outcome) is called before every draw for envs whose previous episode just ended; outcome is a
  dict of (m,) tensors: has (patch present), patch (idx, -1 flat), fall (judge fall criterion reached), collision
  (judge failure), kinds (SO.KINDS bit mask), steps (control steps), allowed (closed allowed touches).
  Every choice must come from a recorded seed (reproducibility rule, user 11-16); the witness itself is the negative's."""

  def __init__(self, rt) -> None:
    self.rt = rt

  def propose(self, env_ids: torch.Tensor, resets: torch.Tensor, rt) -> dict | None:
    return None

  def observe(self, env_ids: torch.Tensor, episode: torch.Tensor, outcome: dict) -> None:
    return None


# ---------------------------------------------------------------------------------------------------- runtime
class SceneRuntime:
  """One per env (env._r8_scene): SceneOnline (boxes, recorder, judge), merged library, sampler, reset counters, cached
  observations. Created by the first term that needs it (the observation manager's shape probe, after the event manager
  expanded its randomisation fields)."""

  def __init__(self, env, spec: dict):
    self.env, self.spec = env, spec
    dev = env.device
    self.n = env.num_envs
    self.lib = merged_library(spec["libs"]).to(dev)
    if self.lib.K != spec["K"]:
      raise ValueError(f"library K {self.lib.K} != configured {spec['K']}")
    self.sc = SO.SceneOnline(env, rules=SO.RULES_V2, per_geom=int(spec["per_geom"]))
    if self.sc.K < self.lib.K:
      raise ValueError(f"scene entity has {self.sc.K} boxes, the library needs {self.lib.K}")
    self.sc.hook()
    self.robot = env.scene["robot"]
    self.seed = derive_seed(env.cfg.seed, "patch")
    self.seed_flat = derive_seed(env.cfg.seed, "flat")
    self.smp = SO.PatchSampler(self.lib, self.seed, side_weights=tuple(spec["side_weights"]))
    self.resets = torch.zeros(self.n, dtype=torch.long, device=dev)
    self.patch = torch.full((self.n,), -1, dtype=torch.long, device=dev)
    self.has = torch.zeros(self.n, dtype=torch.bool, device=dev)
    self.rot = torch.zeros(self.n, device=dev)
    self.out = None
    self.version = 0
    self._key, self._cache = None, None
    self.body_ids = torch.as_tensor(self.robot.indexing.body_ids, device=dev).long()
    self.hand_allowed = self.sc.judge.allowed
    self.proposer = None
    if spec.get("proposer"):
      mod, cls = spec["proposer"].split(":")
      self.proposer = getattr(importlib.import_module(mod), cls)(self)
    self._ray_k = None
    self.flat_override = None              # test hook (smoke timing): a share in [0, 1] replacing spec flat_frac
    self.meta = dict(seed_patch=self.seed, seed_flat=self.seed_flat, library=self.lib.summary(), boxes=self.sc.K,
                     rules=self.sc.rules.name, backend=self.sc.backend)
    print(f"[INFO] r8 scene runtime: {self.meta}", flush=True)

  # -------------------------------------------------------------------------------------------- resets
  def body_weight(self, ids: torch.Tensor) -> torch.Tensor:
    m = self.env.sim.model.body_mass
    m = m[ids][:, self.body_ids] if m.dim() == 2 else m[self.body_ids][None, :].expand(len(ids), -1)
    return m.sum(1) * 9.81

  def outcome(self, ids: torch.Tensor) -> dict:
    J = self.sc.judge
    return dict(has=self.has[ids], patch=self.patch[ids], fall=J.fallen[ids], collision=J.fail[ids], kinds=J.kinds[ids],
                steps=self.sc.step_count[ids], allowed=J.ok[ids])

  def place(self, ids: torch.Tensor) -> None:
    ids = ids.to(self.env.device).long()
    r = self.resets[ids]
    prop = None
    if self.proposer is not None:
      ended = r > 0
      if bool(ended.any()):
        e = ids[ended]
        self.proposer.observe(e, r[ended] - 1, self.outcome(e))
      prop = self.proposer.propose(ids, r, self)
    ff = float(self.spec["flat_frac"]) if self.flat_override is None else float(self.flat_override)
    flat = SO.counter_uniform(self.seed_flat, ids, r, 0) < ff
    idx, rot, _side = self.smp.draw(ids, r, phi=None if prop is None else prop.get("phi"))
    idx = torch.where(flat, torch.full_like(idx, -1), idx)
    if prop is not None:
      if prop.get("idx") is not None:
        idx = prop["idx"].to(idx)
        bad = (idx >= 0) & (self.lib.split[idx.clamp_min(0)] != 0)
        if bool(bad.any()):
          raise ValueError("the proposer chose a holdout patch; training draws the train split only")
      if prop.get("rot") is not None:
        rot = prop["rot"].to(rot)
    self.sc.place_lib(ids, self.lib, idx, rot)
    self.sc.reset_judge(ids, win0=0, bw=self.body_weight(ids))
    self.patch[ids] = idx
    self.has[ids] = idx >= 0
    self.rot[ids] = rot.float()
    self.resets[ids] = r + 1
    self.version += 1

  # -------------------------------------------------------------------------------------------- control step
  def down(self) -> torch.Tensor:
    d = self.robot.data
    tilt = torch.acos((-d.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    h = d.root_link_pos_w[:, 2] - self.env.scene.env_origins[:, 2]
    return (tilt > self.sc.rules.fall_tilt) | (h < self.sc.rules.fall_height)

  def end_step(self) -> dict:
    out = self.sc.end_step(self.down())
    # the judge's fail / kinds / ok tensors are cleared in place by the reset that follows; keep this step's values
    self.out = dict(out, fail=out["fail"].clone(), kinds=out["kinds"].clone(), ok=out["ok"].clone())
    self.version += 1
    return self.out

  # -------------------------------------------------------------------------------------------- observations
  def rays(self, R: int) -> torch.Tensor:
    """(n, R) SceneOnline.ray_profile(R) semantics; on CUDA one fused warp kernel (thread per env x ray)."""
    if not str(self.env.device).startswith("cuda"):
      return self.sc.ray_profile(R)
    import warp as wp
    if self._ray_k is None:
      self._ray_k = _ray_kernel()
      self._ray_out = torch.zeros(self.n, R, device=self.env.device)
      self._dang = float(torch.tensor(2.0 * math.pi / R, dtype=torch.float32))
    data = self.env.sim.data
    fq = self.robot.indexing.free_joint_q_adr.long()
    hd = SO.yaw_of_quat(data.qpos[:, fq[3:7]]).float().contiguous()
    pel = data.qpos[:, fq[:2]].float().contiguous()
    b = self.sc.box
    zc = b[..., 2] - self.sc.origins[:, None, 2]
    body = (self.sc.live & (zc - b[..., 7] < ZHI) & (zc + b[..., 7] > ZLO)).int().contiguous()
    with wp.ScopedDevice(self.env.sim.wp_device):
      wp.launch(self._ray_k, dim=(self.n, R),
                inputs=[wp.from_torch(b.contiguous(), dtype=wp.float32), wp.from_torch(body, dtype=wp.int32),
                        wp.from_torch(pel, dtype=wp.float32), wp.from_torch(hd, dtype=wp.float32), self.sc.K, self._dang,
                        CAP, wp.from_torch(self._ray_out, dtype=wp.float32)])
    return self._ray_out

  def hand_features(self) -> torch.Tensor:
    n, dev = self.n, self.env.device
    if self.out is None or self.sc.cells is None:
      return torch.zeros(n, 2 * HAND_FEATS, device=dev)
    J, r = self.sc.judge, self.sc.rules
    act = J.active
    bw = J.bw[:, None]
    f20 = self.sc.cells["fmean"][:, [1, 2]][:, :, self.hand_allowed].sum(-1)
    f05 = J.ring.sum(-1) / J.W
    dur = (J.last - J.t0 + 1).to(f20.dtype) * J.dt
    feat = torch.stack([(J.flags == 0).to(f20.dtype), f20 / bw / r.hand_fpeak_bw, f05 / bw / r.hand_f_bw,
                        dur / r.hand_t_s], -1)
    return torch.where(act[..., None], feat, torch.zeros_like(feat)).reshape(n, 2 * HAND_FEATS)

  def obs(self) -> dict:
    key = (self.version, int(self.env._sim_step_counter))
    if key == self._key:
      return self._cache
    s, out = self.spec, {}
    if s["rays"]:
      out["rays"] = (1.0 - self.rays(s["rays"]) / CAP).clamp(0.0, 1.0)
    if s["boxes"]:
      b = self.sc.nearest_boxes(s["boxes"])
      near = (1.0 - b[..., 9:10] / CAP).clamp(0.0, 1.0)
      out["boxes"] = torch.cat([b[..., :9], near], -1).reshape(self.n, -1)
    if s["hands"]:
      out["hands"] = self.hand_features()
    self._key, self._cache = key, out
    return out


def runtime(env, spec: dict) -> SceneRuntime:
  rt = getattr(env, "_r8_scene", None)
  if rt is None:
    rt = SceneRuntime(env, spec)
    env._r8_scene = rt
  elif rt.spec != spec:
    raise ValueError("scene terms carry different specs")
  return rt


_RK: dict = {}


def _ray_kernel():
  if "k" in _RK:
    return _RK["k"]
  import warp as wp

  @wp.kernel
  def ray_first_hit(box: wp.array3d(dtype=wp.float32), body: wp.array2d(dtype=wp.int32), pel: wp.array2d(dtype=wp.float32),
                    hd: wp.array(dtype=wp.float32), K: int, dang: float, cap: float, out: wp.array2d(dtype=wp.float32)):
    e, r = wp.tid()
    ang = hd[e] + float(r) * dang
    ux = wp.cos(ang)
    uy = wp.sin(ang)
    px = pel[e, 0]
    py = pel[e, 1]
    best = cap
    big = float(1.0e9)
    for k in range(K):
      if body[e, k] == 0:
        continue
      c = box[e, k, 3]
      s = box[e, k, 4]
      ox = px - box[e, k, 0]
      oy = py - box[e, k, 1]
      lox = c * ox + s * oy
      loy = -s * ox + c * oy
      ldx = c * ux + s * uy
      ldy = -s * ux + c * uy
      hx = box[e, k, 5]
      hy = box[e, k, 6]
      lo_x = big
      hi_x = -big
      if wp.abs(ldx) > 1.0e-9:
        t1 = (-hx - lox) * (1.0 / ldx)
        t2 = (hx - lox) * (1.0 / ldx)
        lo_x = wp.min(t1, t2)
        hi_x = wp.max(t1, t2)
      elif wp.abs(lox) <= hx:
        lo_x = -big
        hi_x = big
      lo_y = big
      hi_y = -big
      if wp.abs(ldy) > 1.0e-9:
        t1 = (-hy - loy) * (1.0 / ldy)
        t2 = (hy - loy) * (1.0 / ldy)
        lo_y = wp.min(t1, t2)
        hi_y = wp.max(t1, t2)
      elif wp.abs(loy) <= hy:
        lo_y = -big
        hi_y = big
      tmin = wp.max(wp.max(lo_x, lo_y), 0.0)
      if wp.min(hi_x, hi_y) >= tmin:
        if tmin < best:
          best = tmin
    out[e, r] = best

  _RK["k"] = ray_first_hit
  return ray_first_hit


# ---------------------------------------------------------------------------------------------------- manager terms
class ScenePlace:
  """Reset event (last in the reset order: after the root / joint resets and the mass randomisation)."""

  def __init__(self, cfg, env):
    self.spec = cfg.params["spec"]

  def reset(self, env_ids=None):
    return None

  def __call__(self, env, env_ids, spec):
    ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids
    runtime(env, spec).place(ids)


class SceneCollision:
  """Termination: the online judge's collision failure (once per control step, before the rewards). Logs Scene/* means."""

  def __init__(self, cfg, env):
    self.spec = cfg.params["spec"]

  def reset(self, env_ids=None):
    return None

  def __call__(self, env, spec):
    rt = runtime(env, spec)
    out = rt.end_step()
    log = env.extras.get("log")
    if log is not None:
      fn = out["fail_new"]
      hasf = rt.has.float()
      log["Scene/has_scene"] = hasf.mean()
      log["Scene/fail_new"] = fn.float().mean()
      log["Scene/fail_new_per_scene_env"] = fn.float().sum() / hasf.sum().clamp_min(1.0)
      for b, name in enumerate(SO.KINDS):
        log[f"Scene/kind_{name}"] = (fn & ((out["kinds"] >> b) & 1).bool()).float().mean()
      log["Scene/hand_touch_ok"] = (out["touch"] == 1).any(1).float().mean()
      log["Scene/hand_touch_ok_per_scene_env"] = (out["touch"] == 1).any(1).float().sum() / hasf.sum().clamp_min(1.0)
      log["Scene/allowed_touches_closed"] = out["ok"].float().mean()
      log["Scene/judge_fallen"] = rt.sc.judge.fallen.float().mean()
      log["Scene/overflow_steps"] = rt.sc.stat["overflow_steps"].float()
      log["Scene/geom_overflow"] = rt.sc.stat["geom_overflow"].float()
    if spec["terminate"]:
      return out["fail"].clone()
    return torch.zeros_like(out["fail"])


def scene_rays(env, spec):
  return runtime(env, spec).obs()["rays"]


def scene_boxes(env, spec):
  return runtime(env, spec).obs()["boxes"]


def scene_hands(env, spec):
  return runtime(env, spec).obs()["hands"]


def spec_dict(spec: SceneSpec) -> dict:
  lib = merged_library(spec.libs)
  d = dataclasses.asdict(spec)
  d["libs"] = [dict(e) for e in spec.libs]
  d["side_weights"] = [float(x) for x in spec.side_weights]
  d.update(K=lib.K, library=lib.summary(), rules="v2", judge_window="reset", hand_x_min=SO.HAND_X_MIN,
           obs_layout=layout(spec))
  return d


def layout(spec: SceneSpec) -> list:
  out = []
  if spec.rays:
    out.append(["rays", int(spec.rays)])
  if spec.boxes:
    out.append(["boxes", int(spec.boxes)])
  if spec.hands:
    out.append(["hands", 2])
  return out


def scene_dim(lay) -> int:
  return sum(n if kind == "rays" else n * (BOX_FEATS if kind == "boxes" else HAND_FEATS) for kind, n in lay)


def scene_env_cfg(play: bool = False, t0: int = 1499, spec: SceneSpec | None = None, source: str = "both", hand: str = "stock"):
  """A-GMM's environment (s2f_r7a.r7a_env_cfg) + the scene patches, judge termination and scene observations.
  source: 'both' (ProcTHOR + ScanNet++ v2.1, K 1,536 for every env), 'procthor' (K 176) or 'scannetpp' (K 1,536)."""
  from mjlab.managers.event_manager import EventTermCfg
  from mjlab.managers.observation_manager import ObservationTermCfg
  from mjlab.managers.termination_manager import TerminationTermCfg
  from r1adv.positive.s2f_r7a import r7a_env_cfg
  spec = spec or SceneSpec(libs=SOURCES[source], hand=hand)
  cfg = r7a_env_cfg(play=play, t0=t0)
  if spec.hand == "o6":
    from r1lab.robot import use_o6_hands
    use_o6_hands(cfg)
  elif spec.hand != "stock":
    raise ValueError(f"unknown hand {spec.hand!r}")
  sp = spec_dict(spec)
  SO.add_entity(cfg, sp["K"])
  cfg.events["scene_place"] = EventTermCfg(func=ScenePlace, mode="reset", params=dict(spec=sp))
  cfg.terminations["scene_collision"] = TerminationTermCfg(func=SceneCollision, params=dict(spec=sp))
  for group in ("actor", "critic"):
    terms = cfg.observations[group].terms
    for kind, _n in sp["obs_layout"]:
      terms[f"scene_{kind}"] = ObservationTermCfg(func={"rays": scene_rays, "boxes": scene_boxes, "hands": scene_hands}[kind],
                                                  params=dict(spec=sp))
  return cfg


# ---------------------------------------------------------------------------------------------------- learning
def widen_module_state(src: dict, dst: dict, base: int) -> tuple[dict, list]:
  """Parent state dict with `base` inputs -> the shapes of dst (more inputs): new first-layer columns 0, normaliser
  mean 0 / var 1 / std 1 on the new inputs. Equal shapes pass through; any other difference raises."""
  out, changed = {}, []
  for k, want in dst.items():
    if k not in src:
      raise KeyError(f"parent state lacks {k}")
    v = src[k]
    if tuple(v.shape) == tuple(want.shape):
      out[k] = v
      continue
    if k == "mlp.0.weight" and v.shape[0] == want.shape[0] and v.shape[1] == base and want.shape[1] > base:
      w = torch.zeros(want.shape, dtype=v.dtype, device=v.device)
      w[:, :base] = v
    elif k in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std") and v.shape[-1] == base:
      w = (torch.zeros if k.endswith("_mean") else torch.ones)(want.shape, dtype=v.dtype, device=v.device)
      w[..., :base] = v
    else:
      raise ValueError(f"cannot widen {k}: {tuple(v.shape)} -> {tuple(want.shape)}")
    out[k] = w
    changed.append(f"{k} {tuple(v.shape)}->{tuple(w.shape)}")
  extra = set(src) - set(dst)
  if extra:
    raise KeyError(f"parent state has keys the model lacks: {sorted(extra)[:4]}")
  return out, changed


def widen_optimizer_state(state: dict, params: list, bases: list) -> tuple[dict, list]:
  """Adam state for params (list of (name, tensor) in optimizer order) whose first-layer weights got new input columns:
  exp_avg 0, exp_avg_sq = the row mean of the parent's columns on the new columns."""
  st = {"param_groups": state["param_groups"], "state": {}}
  changed = []
  for i, s in state["state"].items():
    name, p = params[int(i)]
    s = dict(s)
    for key in ("exp_avg", "exp_avg_sq"):
      v = s.get(key)
      if v is None or tuple(v.shape) == tuple(p.shape):
        continue
      base = bases[int(i)]
      if name != "mlp.0.weight" or v.shape[1] != base or v.shape[0] != p.shape[0]:
        raise ValueError(f"optimizer state {i} ({name}) {tuple(v.shape)} cannot become {tuple(p.shape)}")
      w = torch.zeros(p.shape, dtype=v.dtype, device=v.device)
      w[:, :base] = v
      if key == "exp_avg_sq":
        w[:, base:] = v.mean(dim=1, keepdim=True)
      s[key] = w
      changed.append(f"{i}:{name}.{key}")
    st["state"][i] = s
  return st, changed


class SceneMirror:
  """Mirror augmentation for [r1lab frames | scene] inputs: the r1lab Mirror on the first actor_base / critic_base
  inputs, the scene layout mirrored here (rays j -> -j mod R; boxes y and yaw sin negated; hands left <-> right)."""

  def __init__(self, base, actor_base: int, critic_base: int, scene_layout):
    self.base = base
    self.actor_base, self.critic_base = int(actor_base), int(critic_base)
    self.layout = [(str(k), int(n)) for k, n in scene_layout]
    self.dim = scene_dim(self.layout)
    self._idx = {}

  def scene(self, x: torch.Tensor) -> torch.Tensor:
    if x.shape[1] != self.dim:
      raise ValueError(f"scene part has {x.shape[1]} inputs, layout {self.layout} needs {self.dim}")
    if not self.layout:                                                      # every scene observation switched off
      return x
    out, o = [], 0
    for kind, n in self.layout:
      if kind == "rays":
        key = (n, x.device)
        if key not in self._idx:
          self._idx[key] = (-torch.arange(n, device=x.device)) % n
        out.append(x[:, o:o + n][:, self._idx[key]])
        o += n
      elif kind == "boxes":
        sign = torch.ones(BOX_FEATS, device=x.device)
        sign[1] = -1.0
        sign[7] = -1.0
        out.append((x[:, o:o + n * BOX_FEATS].view(-1, n, BOX_FEATS) * sign).reshape(-1, n * BOX_FEATS))
        o += n * BOX_FEATS
      elif kind == "hands":
        out.append(x[:, o:o + 2 * HAND_FEATS].view(-1, 2, HAND_FEATS)[:, [1, 0]].reshape(-1, 2 * HAND_FEATS))
        o += 2 * HAND_FEATS
      else:
        raise ValueError(kind)
    return torch.cat(out, 1)

  def mirror_group(self, key: str, v: torch.Tensor) -> torch.Tensor:
    nb = self.actor_base if key == "actor" else self.critic_base
    head = self.base.actor(v[:, :nb]) if key == "actor" else self.base.critic(v[:, :nb])
    return torch.cat([head, self.scene(v[:, nb:])], 1)

  def __call__(self, env=None, obs=None, actions=None):
    from tensordict import TensorDict
    obs_aug = None
    if obs is not None:
      b = obs.batch_size[0]
      data = {k: torch.cat([v, self.mirror_group(k, v)], dim=0) for k, v in obs.items()}
      obs_aug = TensorDict(data, batch_size=[2 * b], device=obs.device)
    act_aug = None
    if actions is not None:
      act_aug = torch.cat([actions, self.base.act(actions)], dim=0)
    return obs_aug, act_aug


@dataclasses.dataclass
class ScenePpoCfg(ParentLrPpoCfg):
  actor_base_dim: int = ACTOR_BASE
  critic_base_dim: int = CRITIC_BASE
  scene_layout: tuple = ()


class ScenePPO(ParentLrPPO):
  """ParentLrPPO + parent widening at load + the scene mirror (see the module docstring)."""

  def __init__(self, *args, actor_base_dim: int = ACTOR_BASE, critic_base_dim: int = CRITIC_BASE, scene_layout=(),
               **kwargs):
    super().__init__(*args, **kwargs)
    self.actor_base_dim, self.critic_base_dim = int(actor_base_dim), int(critic_base_dim)
    self.scene_layout = [(str(k), int(n)) for k, n in scene_layout]
    dims = (int(self._raw_actor.obs_dim), int(self._raw_critic.obs_dim))
    want = (self.actor_base_dim + scene_dim(self.scene_layout), self.critic_base_dim + scene_dim(self.scene_layout))
    if dims != want:
      raise ValueError(f"actor / critic inputs {dims} != {want} (base + scene layout {self.scene_layout})")
    if self.symmetry is not None:
      self.symmetry.data_augmentation_func = SceneMirror(self.symmetry.data_augmentation_func, self.actor_base_dim,
                                                         self.critic_base_dim, self.scene_layout)
    self.widened: list = []

  def load(self, loaded_dict, load_cfg, strict):
    d = dict(loaded_dict)
    changed = []
    if "actor_state_dict" in d:
      d["actor_state_dict"], c = widen_module_state(d["actor_state_dict"], self._raw_actor.state_dict(), self.actor_base_dim)
      changed += ["actor." + x for x in c]
    if "critic_state_dict" in d:
      d["critic_state_dict"], c = widen_module_state(d["critic_state_dict"], self._raw_critic.state_dict(), self.critic_base_dim)
      changed += ["critic." + x for x in c]
    if "optimizer_state_dict" in d and (load_cfg is None or load_cfg.get("optimizer", True)):
      params = list(self._raw_actor.named_parameters()) + list(self._raw_critic.named_parameters())
      bases = [self.actor_base_dim] * len(list(self._raw_actor.parameters())) + [self.critic_base_dim] * len(list(self._raw_critic.parameters()))
      d["optimizer_state_dict"], c = widen_optimizer_state(d["optimizer_state_dict"], params, bases)
      changed += ["optimizer." + x for x in c]
    self.widened = changed
    print(f"[INFO] r8 scene load: {'widened ' + '; '.join(changed) if changed else 'same widths, no widening'}", flush=True)
    return super().load(d, load_cfg, strict)


def scene_rl_cfg(spec: SceneSpec | None = None):
  """A-GMM's PPO recipe (s2f_r7a.r7a_rl_cfg: careful_walk PPO, adaptive lr restart 1.139e-4) with ScenePPO."""
  from r1adv.positive.s2f_r7a import r7a_rl_cfg
  spec = spec or SceneSpec()
  cfg = r7a_rl_cfg()
  base = cfg.algorithm
  cfg.algorithm = ScenePpoCfg(**{f.name: getattr(base, f.name) for f in dataclasses.fields(base)},
                              actor_base_dim=ACTOR_BASE, critic_base_dim=CRITIC_BASE,
                              scene_layout=tuple(tuple(x) for x in layout(spec)))
  cfg.algorithm.class_name = "r1adv.positive_scene.scene_r8:ScenePPO"
  return cfg
