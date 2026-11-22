"""R7 scene lane (rounds/R04/r7_scene_lane.md): indoor scene patches for TRAINING environments, with the contact judge
of rounds/R04/r6_scene_env.md section 3 running online on the GPU without host synchronisation.

User rule (11-18, verbatim): "바닥을 제외하고 그 어떠한 충돌도 허용되지 않아. 아주 확실한 경우( 소파, 책상, 벽 등 ) 에서만
자연스럽게 손을 이용해서 가볍게 중심을 잡는 정도만 허용되."

Pieces (fixed-size tensor code: nothing inside a physics or control step reads a GPU value on the host):
  Rules         rule sets. 'v1' = r6sc_judge.py f1473302 (r6_scene_env.md 3); 'v2' = r7_scene_lane.md 2 (scenes_v1 grades,
                UNDECIDED forbidden, support heights: furniture box top 0.60-1.30 m, wall contact height 0.60-1.80 m);
                'v3' = r9_scene_v3.md 2 (opt-in, RULES / rules_of('v3')): v2 with both bands 0.60-0.978 m (the robot's
                standing shoulder height); the furniture top band is applied when a patch is placed (relabel_top_band).
                Defaults (RULES_V1 / RULES_V2, SceneOnline(rules=RULES_V1)) are unchanged.
  OnlineJudge   per-environment state machine over control steps: failure flag, failure kinds, hand-touch state.
  SceneOnline   scene boxes placed per env_ids on the robot's reset pose, per-physics-step contact recorder (fixed-size
                scatter over a prefix of the contact buffer), visual-mesh intrusion with per-link culling, observation
                helpers (nearest boxes, 2D ray profile). Boxes are yaw-only (both libraries are).
  PatchLibrary  library JSON (r6sc-patchlib-v1 or r7sl-patchlib-v2) -> padded tensors and the train / holdout split.
  PatchSampler  counter-based seeded draws u(seed, env id, reset count, k): reproducible whatever the policy does.

Training wiring (the positive's choice; see r7_scene_lane.md 4):
  cfg time      add_entity(cfg, K)                      K = library.K (boxes per patch, multiple of 16)
  after build   sc = SceneOnline(env, rules=RULES_V2); sc.hook(); lib = PatchLibrary(path, device); smp = PatchSampler(lib, seed)
  reset event   idx, rot, side = smp.draw(ids, reset_count[ids]); sc.place_lib(ids, lib, idx, rot); sc.reset_judge(ids)
                (after the robot's root/joint reset is written; the step's sim.forward() refreshes the rest)
  termination   out = sc.end_step(down)                 once per control step, before rewards; out['fail'] ends the episode
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

PARK_Z = -50.0
PARTS = ("foot", "hand_l", "hand_r", "arm", "leg", "trunk", "head")
BODY = (0, 3, 4, 5, 6)                 # parts that fail on any non-floor contact
HANDS = (1, 2)
CLASSES_V1 = ("floor", "wall", "sofa", "table", "counter", "railing", "other")
CLASSES_V2 = CLASSES_V1 + ("undecided", "oob")
SUPPORT = ("wall", "sofa", "table", "counter", "railing")
GROUP = 1
FRICTION = (0.7, 0.005, 0.0001)
HAND_X_MIN = 0.08                      # wrist_roll_link frame: the O6 hand box spans x 0.08-0.22
VIS_VOXEL = 0.03
KINDS = ("body", "intrusion", "hand_other", "hand_hard", "hand_fast", "hand_long")
K_BODY, K_INTR, K_OTHER, K_HARD, K_FAST, K_LONG = (1 << k for k in range(6))


@dataclass(frozen=True)
class Rules:
  name: str
  classes: tuple
  allowed: tuple = SUPPORT
  hand_f_bw: float = 0.05             # 0.5 s trailing mean of the hand's normal force on allowed classes, x body weight
  hand_win_s: float = 0.5
  hand_fpeak_bw: float = 0.10         # 20 ms (one control step) mean, x body weight
  hand_v_mps: float = 0.25            # approach speed at a contact onset
  hand_t_s: float = 2.0               # duration of one touch
  gap_s: float = 0.10                 # shorter gaps join one touch
  intr_m: float = 0.01                # visual-mesh intrusion depth
  fall_tilt: float = 1.0
  fall_height: float = 0.35
  quantize: bool = True               # compare as r6sc_judge does: force ratios rounded to 1e-4, speed to 1e-3
  wall_band: tuple | None = None      # v2: allowed wall touch only at contact height lo..hi above the floor
  top_band: tuple | None = None       # v2: allowed furniture box only with its top lo..hi above the floor (library)
  undecided_as_floor: tuple = field(default_factory=tuple)
  relabel_top: bool = False           # v3: apply top_band to the furniture support classes when a patch is placed


RULES_V1 = Rules("v1", CLASSES_V1)
RULES_V2 = Rules("v2", CLASSES_V2, wall_band=(0.60, 1.80), top_band=(0.60, 1.30))
# v3 (rounds/R04/r9_scene_v3.md 2, frozen 62afd403): both bands 0.60 m .. the standing shoulder height S of the O6 body
# (shoulder pitch link origin, 0.978 m, body signature 15446e6f); thresholds, classes and windows as v2. Opt-in only.
SHOULDER_S = 0.978
RULES_V3 = Rules("v3", CLASSES_V2, wall_band=(0.60, SHOULDER_S), top_band=(0.60, SHOULDER_S), relabel_top=True)
RULES = {"v1": RULES_V1, "v2": RULES_V2, "v3": RULES_V3}
SUPPORT_TOP = ("sofa", "table", "counter", "railing")   # furniture classes under the top band (walls: wall_band, at contact)
V2_TOP_BAND = (0.60, 1.30)                             # the band the r7sl-patchlib-v2 libraries (v2, v2.1, v2.2) were labelled with


def rules_of(name: str) -> Rules:
  return RULES[name]


def relabel_top_band(local: torch.Tensor, cls: torch.Tensor, rules: Rules) -> torch.Tensor:
  """Class index (..., K) with rules.top_band applied: a box of a furniture support class (SUPPORT_TOP) whose top
  (local z + half z, above the floor; local (..., K, 8) = cx, cy, cz, cos, sin, hx, hy, hz) lies outside the band becomes
  'oob'. Exact on a library labelled with V2_TOP_BAND when rules.top_band lies inside it (checked)."""
  lo, hi = rules.top_band
  if not (V2_TOP_BAND[0] <= lo <= hi <= V2_TOP_BAND[1]):
    raise ValueError(f"top band {rules.top_band} is not inside the library band {V2_TOP_BAND}")
  sup = torch.tensor([rules.classes.index(c) for c in SUPPORT_TOP], device=cls.device)
  top = local[..., 2] + local[..., 7]
  out = torch.isin(cls, sup) & ((top < lo) | (top > hi))
  return torch.where(out, torch.full_like(cls, rules.classes.index("oob")), cls)


def _q(x: torch.Tensor, decimals: int, on: bool) -> torch.Tensor:
  if not on:
    return x
  s = 10.0 ** decimals
  return torch.round(x * s) / s


class OnlineJudge:
  """Per-environment judge over control steps (rules: r6_scene_env.md 3 / r7_scene_lane.md 2).

  step(s, cnt, fmean, von, intr, intr_cls, down) once per control step with that step's recorder cells:
    cnt (n, P, C) substeps in contact, fmean (n, P, C) mean summed normal force (N), von (n, P, C) approach speed at an
    onset (m/s), intr (n, P) visual intrusion depth (m), intr_cls (n, P) class of that box (-1 none), down (n,) the
    fall criterion on this step's post-step state (tilt > 1 rad or pelvis < 0.35 m). s: this step's index (int or (n,)).
  Window: steps >= win0 (per env) until the fall step (exclusive). Steps before win0 decide 'valid' only.
  Returns dict(fail (n,) cumulative, fail_new (n,), kinds (n,) bit mask of KINDS, touch (n, 2) 0 none / 1 touching
  within the limits / 2 touching over a limit, ok (n,) closed allowed touches so far).
  """

  def __init__(self, n: int, rules: Rules, bw, device, dt: float = 0.02, dtype=torch.float32, log: bool = False,
               max_touches: int = 16):
    self.n, self.r, self.dev, self.dt, self.dtype, self.log = n, rules, device, dt, dtype, log
    C = len(rules.classes)
    self.C = C
    self.allowed = torch.tensor([rules.classes.index(c) for c in rules.allowed], device=device)
    self.forbid = torch.tensor([k for k in range(1, C) if rules.classes[k] not in rules.allowed], device=device)
    self.cls_bit = (1 << torch.arange(len(rules.allowed), device=device)).to(torch.int64)
    self.W = max(1, round(rules.hand_win_s / dt))
    self.gap_n = round(rules.gap_s / dt)
    self.bw = torch.as_tensor(bw, dtype=dtype, device=device).expand(n).clone()
    self.max_touches = max_touches
    self.k = 0                                                   # global control-step counter (ring index)
    z = lambda *s, d=torch.long, v=0: torch.full(s, v, dtype=d, device=device)
    self.win0 = z(n)
    self.fallen, self.fall_step = z(n, d=torch.bool), z(n, v=-1)
    self.pre_contact, self.pre_intr = z(n), z(n, d=dtype)
    self.fail, self.kinds, self.fail_step = z(n, d=torch.bool), z(n), z(n, v=-1)
    self.active, self.t0, self.last = z(n, 2, d=torch.bool), z(n, 2), z(n, 2, v=-1)
    self.fm_max, self.fpeak, self.vmax = z(n, 2, d=dtype), z(n, 2, d=dtype), z(n, 2, d=dtype)
    self.flags, self.clsm = z(n, 2), z(n, 2)
    self.ring = z(n, 2, self.W, d=dtype)
    self.ok, self.hand_event = z(n), z(n, d=torch.bool)
    self.any_scene, self.post_fall, self.hand_floor = z(n, d=torch.bool), z(n, d=torch.bool), z(n)
    if log:
      nb = len(BODY)
      self.first_body = z(n, nb, C - 1, v=-1)
      self.first_intr, self.intr_cls_first = z(n, nb, v=-1), z(n, nb, v=-1)
      self.first_other = z(n, 2, v=-1)
      self.tlog = z(n, 2, max_touches, 8, d=torch.float64, v=float("nan"))   # t0, dur, fm, fpeak, v, clsm, open, flags
      self.tcount = z(n, 2)

  # ------------------------------------------------------------------------------------------------ state
  def reset(self, env_ids: torch.Tensor | None = None, win0: torch.Tensor | int = 0, bw=None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.win0[ids] = torch.as_tensor(win0, device=self.dev)
    if bw is not None:
      self.bw[ids] = torch.as_tensor(bw, dtype=self.dtype, device=self.dev)
    for t, v in ((self.fallen, False), (self.fall_step, -1), (self.pre_contact, 0), (self.pre_intr, 0.0),
                 (self.fail, False), (self.kinds, 0), (self.fail_step, -1), (self.active, False), (self.t0, 0),
                 (self.last, -1), (self.fm_max, 0.0), (self.fpeak, 0.0), (self.vmax, 0.0), (self.flags, 0),
                 (self.clsm, 0), (self.ring, 0.0), (self.ok, 0), (self.hand_event, False), (self.any_scene, False),
                 (self.post_fall, False), (self.hand_floor, 0)):
      t[ids] = v
    if self.log:
      for t, v in ((self.first_body, -1), (self.first_intr, -1), (self.intr_cls_first, -1), (self.first_other, -1),
                   (self.tlog, float("nan")), (self.tcount, 0)):
        t[ids] = v

  def _flags_of(self) -> torch.Tensor:
    r, q = self.r, self.r.quantize
    hard = (_q(self.fm_max / self.bw[:, None], 4, q) > r.hand_f_bw) | (_q(self.fpeak / self.bw[:, None], 4, q) > r.hand_fpeak_bw)
    fast = _q(self.vmax, 3, q) > r.hand_v_mps
    dur = (self.last - self.t0 + 1).to(torch.float64) * self.dt
    long = torch.round(dur * 1000.0) / 1000.0 > r.hand_t_s
    return hard.long() * K_HARD + fast.long() * K_FAST + long.long() * K_LONG

  def _log_touch(self, mask: torch.Tensor, open_: bool) -> None:
    """Append the touches selected by mask (n, 2) to the per-env touch log (verbose mode only)."""
    if not self.log or not bool(mask.any()):
      return
    e, h = mask.nonzero(as_tuple=True)
    slot = self.tcount[e, h].clamp_max(self.max_touches - 1)
    row = torch.stack([self.t0[e, h].double(), (self.last[e, h] - self.t0[e, h] + 1).double(),
                       (self.fm_max[e, h] / self.bw[e]).double(), (self.fpeak[e, h] / self.bw[e]).double(),
                       self.vmax[e, h].double(), self.clsm[e, h].double(),
                       torch.full_like(self.vmax[e, h], float(open_)).double(), self.flags[e, h].double()], dim=1)
    self.tlog[e, h, slot] = row
    self.tcount[e, h] += 1

  # ------------------------------------------------------------------------------------------------ step
  def step(self, s, cnt, fmean, von, intr, intr_cls, down) -> dict:
    r, dev = self.r, self.dev
    s = torch.as_tensor(s, device=dev).expand(self.n)
    cnt = cnt > 0
    inwin = s >= self.win0
    pre = ~inwin
    nf = cnt[:, :, 1:]                                                       # non-floor contact cells
    nf_any = nf.flatten(1).any(1)
    bint = intr[:, list(BODY)].to(self.dtype)
    self.pre_contact += pre.long() * nf.flatten(1).sum(1)
    self.pre_intr = torch.where(pre, torch.maximum(self.pre_intr, bint.amax(1)), self.pre_intr)
    newly = inwin & ~self.fallen & down
    self.fall_step = torch.where(newly, s, self.fall_step)
    self.fallen |= newly
    act = inwin & ~self.fallen                                               # pre-fall window
    # report-only window accumulators (window start .. end, the fall included)
    self.any_scene |= inwin & (nf_any | (bint > r.intr_m).any(1))
    self.hand_floor += inwin.long() * cnt[:, list(HANDS), 0].sum(1)
    self.post_fall |= inwin & self.fallen & nf_any
    # body parts on any non-floor class, visual intrusion, hand on a forbidden class
    bc = nf[:, list(BODY)]                                                   # (n, 5, C-1)
    new_body = act[:, None, None] & bc
    deep = act[:, None] & (bint > r.intr_m)
    hc = cnt[:, list(HANDS)]                                                 # (n, 2, C)
    other = act[:, None] & hc[:, :, self.forbid].any(-1)
    if self.log:
      fb = new_body & (self.first_body < 0)
      self.first_body = torch.where(fb, s[:, None, None], self.first_body)
      fi = deep & (self.first_intr < 0)
      self.first_intr = torch.where(fi, s[:, None], self.first_intr)
      self.intr_cls_first = torch.where(fi, intr_cls[:, list(BODY)].long(), self.intr_cls_first)
      fo = other & (self.first_other < 0)
      self.first_other = torch.where(fo, s[:, None], self.first_other)
    # hand touches on allowed classes
    al = hc[:, :, self.allowed]                                              # (n, 2, A)
    on = act[:, None] & al.any(-1)
    fh = fmean[:, list(HANDS)][:, :, self.allowed].to(self.dtype).sum(-1)
    f = torch.where(act[:, None], fh, torch.zeros_like(fh))
    self.ring[:, :, self.k % self.W] = f
    self.k += 1
    fm = self.ring.sum(-1) / self.W
    v = von[:, list(HANDS)][:, :, self.allowed].to(self.dtype).amax(-1)
    # close touches whose gap has run out (an off step more than gap_n steps after the last contact)
    expire = self.active & ~on & act[:, None] & ((s[:, None] - self.last) > self.gap_n)
    self.ok += (expire & (self.flags == 0)).sum(1)
    self._log_touch(expire, False)
    self.active &= ~expire
    start = on & ~self.active
    self.t0 = torch.where(start, s[:, None], self.t0)
    for t in (self.fm_max, self.fpeak, self.vmax):
      t.masked_fill_(start, 0.0)
    self.flags = torch.where(start, torch.zeros_like(self.flags), self.flags)
    self.clsm = torch.where(start, torch.zeros_like(self.clsm), self.clsm)
    self.active |= on
    self.last = torch.where(on, s[:, None], self.last)
    self.fm_max = torch.where(on, torch.maximum(self.fm_max, fm), self.fm_max)
    self.fpeak = torch.where(on, torch.maximum(self.fpeak, f), self.fpeak)
    self.vmax = torch.where(on, torch.maximum(self.vmax, v), self.vmax)
    self.clsm = self.clsm | torch.where(on, (al.long() * self.cls_bit).sum(-1), torch.zeros_like(self.clsm))
    flags = torch.where(self.active, self._flags_of(), self.flags) | self.flags
    self.flags = flags
    hand_bits = torch.where(self.active, flags, torch.zeros_like(flags))
    kinds = (new_body.flatten(1).any(1).long() * K_BODY | deep.any(1).long() * K_INTR | other.any(1).long() * K_OTHER
             | hand_bits[:, 0] | hand_bits[:, 1])
    self.hand_event |= other.any(1) | (hand_bits != 0).any(1)
    now = kinds != 0
    fail_new = now & ~self.fail
    self.fail |= now
    self.kinds |= kinds
    self.fail_step = torch.where(fail_new, s, self.fail_step)
    touch = torch.where(self.active, torch.where(self.flags != 0, 2, 1), 0)
    return dict(fail=self.fail, fail_new=fail_new, kinds=self.kinds, touch=touch, ok=self.ok)

  def finalize(self, end: torch.Tensor | int) -> dict:
    """Close the window: end = first step after the episode (horizon) unless the env fell (then the fall step).
    A touch still in contact on the last window step is 'open' (never allowed; fails only if it broke a limit)."""
    end = torch.as_tensor(end, device=self.dev).expand(self.n)
    wend = torch.where(self.fallen, self.fall_step, end)
    open_ = self.active & (self.last >= (wend - 1)[:, None])
    closed = self.active & ~open_
    self.ok += (closed & (self.flags == 0)).sum(1)
    self._log_touch(closed, False)
    self._log_touch(open_, True)
    self.active[:] = False
    valid = (self.pre_contact == 0) & (self.pre_intr <= self.r.intr_m)
    return dict(valid=valid, fall=self.fallen.clone(), fall_step=self.fall_step.clone(), collision=self.fail.clone(),
                kinds=self.kinds.clone(), fail_step=self.fail_step.clone(),
                allowed_touch=(self.ok > 0) & ~self.hand_event, ok=self.ok.clone(), hand_event=self.hand_event.clone(),
                any_scene_contact=self.any_scene.clone(), post_fall=self.post_fall.clone(),
                hand_floor_steps=self.hand_floor.clone(), pre_contact=self.pre_contact.clone(),
                pre_intr=self.pre_intr.clone())


# ---------------------------------------------------------------------------------------------------- scene entity
def add_entity(cfg, count: int) -> None:
  """K box geoms on one fixed body (group 1, contype 0 / conaffinity 1: they touch the robot only, condim 3)."""
  import mujoco
  from mjlab.entity import EntityCfg

  def spec():
    made = mujoco.MjSpec()
    body = made.worldbody.add_body(name="scenepatch")
    for k in range(count):
      body.add_geom(name=f"box{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.01, 0.01], pos=[0.0, 0.0, PARK_Z],
                    rgba=[0.62, 0.56, 0.50, 1.0], group=GROUP, contype=0, conaffinity=1, condim=3, priority=0,
                    friction=list(FRICTION))
    return made
  cfg.scene.entities["scenepatch"] = EntityCfg(spec_fn=spec)
  cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 250)
  cfg.sim.njmax = max(cfg.sim.njmax or 0, 1600)


def part_of_name(name: str) -> int:
  """Robot geom / body name -> PARTS index (-1: untracked). Same map as r6sc scene_patch.py fb74ba3a."""
  n = name.split("/")[-1]
  if "foot" in n or "ankle_roll" in n:
    return 0
  if "o6" in n or "hand" in n or "wrist" in n:
    return 1 if n.startswith("left") else 2
  if "shoulder" in n or "elbow" in n:
    return 3
  if any(k in n for k in ("hip", "thigh", "knee", "shin", "linkage", "ankle")):
    return 4
  if "head" in n:
    return 6
  if any(k in n for k in ("pelvis", "torso", "waist", "imu")):
    return 5
  return -1


def yaw_of_quat(q: torch.Tensor) -> torch.Tensor:
  return torch.atan2(2 * (q[..., 0] * q[..., 3] + q[..., 1] * q[..., 2]), 1 - 2 * (q[..., 2] ** 2 + q[..., 3] ** 2))


class SceneOnline:
  """Scene boxes + fixed-size contact recorder + intrusion + online judge for one mjlab env.

  Box state per env (world frame, yaw-only): box (n, K, 8) = cx, cy, cz, cos, sin, hx, hy, hz; cls (n, K) class index
  (-1 unused); top (n, K) box top above the env floor. Classes follow rules.classes; v2 wall contacts outside
  rules.wall_band are recorded as class 'oob'.
  """

  def __init__(self, env, origins: torch.Tensor | None = None, rules: Rules = RULES_V1, *, prefix_per_env: int = 80,
               per_geom: int = 16, bw=None, judge_log: bool = False, judge_dtype=torch.float32,
               hand_points: bool = False, backend: str | None = None):
    import os
    import mujoco
    self.env, dev = env, env.device
    self.backend = backend or ("warp" if str(dev).startswith("cuda") else "torch")
    self.prof = {k: 0.0 for k in ("substep", "cells", "intrusion", "judge")} if os.environ.get("R7SL_PROF") else None
    self.dev = dev
    self.n = env.num_envs
    self.rules = rules
    if rules.relabel_top:
      relabel_top_band(torch.zeros(1, 8), torch.zeros(1, dtype=torch.long), rules)    # band containment check
    self.C = len(rules.classes)
    self.P = len(PARTS)
    self.origins = env.scene.env_origins if origins is None else origins
    mj = env.sim.mj_model
    names = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(mj.ngeom)]
    self.ids = sorted([g for g, nm in enumerate(names) if nm.startswith("scenepatch/box")],
                      key=lambda g: int(names[g].split("box")[-1]))
    self.ids_t = torch.tensor(self.ids, device=dev)
    self.K = len(self.ids)
    self.body = int(mj.geom_bodyid[self.ids[0]])
    env.sim.expand_model_fields(("geom_pos", "geom_quat", "geom_size", "geom_aabb", "geom_rbound"))
    m = env.sim.model
    self.f_pos, self.f_quat, self.f_size, self.f_aabb, self.f_rbound = m.geom_pos, m.geom_quat, m.geom_size, m.geom_aabb, m.geom_rbound
    bq = env.sim.data.xquat[:, self.body]
    if not torch.allclose(bq, torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).expand_as(bq), atol=1e-6):
      raise RuntimeError("scenepatch body is rotated")
    self.body_xpos = env.sim.data.xpos[:, self.body].clone()
    n, K = self.n, self.K
    self.box = torch.zeros(n, K, 8, device=dev)
    self.box[..., 3] = 1.0
    self.cls = torch.full((n, K), -1, dtype=torch.long, device=dev)
    self.top = torch.zeros(n, K, device=dev)
    self.live = torch.zeros(n, K, dtype=torch.bool, device=dev)          # placed, non-floor
    self.cls32 = torch.full((n, K), -1, dtype=torch.int32, device=dev)
    self.live32 = torch.zeros(n, K, dtype=torch.int32, device=dev)
    self.theta = torch.zeros(n, device=dev)
    self.has = torch.zeros(n, dtype=torch.bool, device=dev)
    self._park(torch.arange(n, device=dev))
    # robot geoms -> parts; scene geoms -> slots
    robot = env.scene["robot"]
    robot_bodies = set(int(b) for b in robot.indexing.body_ids.tolist())
    part = torch.full((mj.ngeom,), -1, dtype=torch.long)
    coll = []
    for g in range(mj.ngeom):
      if int(mj.geom_bodyid[g]) in robot_bodies and (mj.geom_contype[g] or mj.geom_conaffinity[g]):
        part[g] = part_of_name(names[g])
        if part[g] < 0:
          raise ValueError(f"robot collision geom without a part: {names[g]}")
        coll.append(g)
    self.robot_coll = torch.tensor(coll, device=dev)
    self.part_of_geom = part.to(dev)
    self.is_robot = self.part_of_geom >= 0
    slot = torch.full((mj.ngeom,), -1, dtype=torch.long)
    slot[self.ids] = torch.arange(K)
    self.slot_of_geom = slot.to(dev)
    self.coll_index = torch.full((mj.ngeom,), 0, dtype=torch.long, device=dev)
    self.coll_index[self.robot_coll] = torch.arange(len(coll), device=dev)
    self.hand_caps = torch.full((mj.ngeom,), -1, dtype=torch.long, device=dev)
    for g in coll:
      if names[g].split("/")[-1] in ("left_hand_collision", "right_hand_collision"):
        self.hand_caps[g] = int(mj.geom_bodyid[g])
    self.dt = float(env.physics_dt)
    self.prefix_per_env = prefix_per_env
    self.per_geom = per_geom
    self.prev_gpos = env.sim.data.geom_xpos[:, self.robot_coll].clone()
    self.fresh = torch.ones(n, dtype=torch.bool, device=dev)
    P, C = self.P, self.C
    self.NPC = n * P * C
    self._acc_reset()
    self.prev_present = torch.zeros(n, P, C, dtype=torch.bool, device=dev)
    self.stat = dict(substeps=torch.zeros((), dtype=torch.long, device=dev),
                     max_nacon=torch.zeros((), dtype=torch.long, device=dev),
                     overflow_steps=torch.zeros((), dtype=torch.long, device=dev),
                     geom_overflow=torch.zeros((), dtype=torch.long, device=dev))
    self.hooked = False
    self.active = True                    # timing A/B: False skips the recorder and the judge (boxes stay)
    bnames = [mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(mj.nbody)]
    self.pelvis_body = next(b for b, nm in enumerate(bnames) if nm.split("/")[-1] == "pelvis" and b in robot_bodies)
    self._visual_points(mj, names, robot_bodies, hand_points)
    if bw is None:
      bw = float(mj.body_mass[sorted(robot_bodies)].sum()) * 9.81
    self.judge = OnlineJudge(n, rules, bw, dev, dt=float(env.step_dt), dtype=judge_dtype, log=judge_log)
    self.step_count = torch.zeros(n, dtype=torch.long, device=dev)
    self.cells = None

  # ---------------------------------------------------------------------------------------------- placement
  def _park(self, ids: torch.Tensor) -> None:
    g = self.ids_t
    park = (self.origins[ids] - self.body_xpos[ids]) + torch.tensor([0.0, 0.0, PARK_Z], device=self.dev)
    self.f_pos[ids[:, None], g[None, :]] = park[:, None, :].expand(-1, self.K, -1)
    self.f_quat[ids[:, None], g[None, :]] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.dev)
    self.f_size[ids[:, None], g[None, :]] = 0.01
    self.f_aabb[ids[:, None], g[None, :], 0] = 0.0
    self.f_aabb[ids[:, None], g[None, :], 1] = 0.01
    self.f_rbound[ids[:, None], g[None, :]] = 0.01 * math.sqrt(3.0)
    self.live[ids] = False
    self.cls[ids] = -1
    self.cls32[ids] = -1
    self.live32[ids] = 0
    self.has[ids] = False

  def place(self, env_ids: torch.Tensor, local: torch.Tensor, cls: torch.Tensor, rot_rel: torch.Tensor,
            anchor: torch.Tensor | None = None, has: torch.Tensor | None = None, forward: bool = False) -> None:
    """Place patches on the robots' current pose (call after the reset wrote the root state).

    env_ids (m,) long; local (m, K, 8) boxes in patch coordinates (cx, cy, cz, cos, sin, hx, hy, hz; z above the
    floor; yaw-only); cls (m, K) class index or -1; rot_rel (m,) patch-frame rotation relative to the robot heading;
    anchor (m, 2) stand point in patch coordinates (default 0); has (m,) False parks that env's boxes (no scene)."""
    dev = self.dev
    ids = env_ids.to(dev).long()
    m = len(ids)
    if self.rules.relabel_top:                                               # v3: furniture top band at placement
      cls = relabel_top_band(local.to(dev), cls.to(dev).long(), self.rules)
    if anchor is None:
      anchor = torch.zeros(m, 2, device=dev)
    if has is None:
      has = torch.ones(m, dtype=torch.bool, device=dev)
    robot = self.env.scene["robot"]
    fq = robot.indexing.free_joint_q_adr.long()
    q = self.env.sim.data.qpos[ids][:, fq[3:7]]
    heading = yaw_of_quat(q)
    pelvis = self.env.sim.data.qpos[ids][:, fq[:2]]
    theta = heading + rot_rel.to(dev)
    c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    lx = local[..., 0] - anchor[:, None, 0]
    ly = local[..., 1] - anchor[:, None, 1]
    wx = pelvis[:, None, 0] + c * lx - s * ly
    wy = pelvis[:, None, 1] + s * lx + c * ly
    wz = self.origins[ids][:, None, 2] + local[..., 2]
    bc, bs = local[..., 3], local[..., 4]
    yc, ys = c * bc - s * bs, s * bc + c * bs                               # box yaw in the world
    half = local[..., 5:8]
    live = (cls >= 0) & has[:, None]
    world = torch.stack([wx, wy, wz], dim=-1)
    yaw = torch.atan2(ys, yc)
    quat = torch.stack([torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)], dim=-1)
    g = self.ids_t
    rel = world - self.body_xpos[ids][:, None, :]
    park = (self.origins[ids] - self.body_xpos[ids])[:, None, :] + torch.tensor([0.0, 0.0, PARK_Z], device=dev)
    self.f_pos[ids[:, None], g[None, :]] = torch.where(live[..., None], rel, park.expand(-1, self.K, -1))
    self.f_quat[ids[:, None], g[None, :]] = torch.where(live[..., None], quat, torch.tensor([1.0, 0, 0, 0], device=dev))
    size = torch.where(live[..., None], half, torch.full_like(half, 0.01))
    self.f_size[ids[:, None], g[None, :]] = size
    self.f_aabb[ids[:, None], g[None, :], 0] = 0.0
    self.f_aabb[ids[:, None], g[None, :], 1] = size
    self.f_rbound[ids[:, None], g[None, :]] = size.norm(dim=-1)
    self.box[ids] = torch.stack([wx, wy, wz, yc, ys, half[..., 0], half[..., 1], half[..., 2]], dim=-1)
    self.cls[ids] = torch.where(live, cls.to(dev).long(), torch.full_like(cls.to(dev).long(), -1))
    self.top[ids] = local[..., 2] + half[..., 2]
    self.live[ids] = live & (cls.to(dev) != 0)
    self.cls32[ids] = self.cls[ids].int()
    self.live32[ids] = self.live[ids].int()
    self.theta[ids] = theta
    self.has[ids] = has
    self.fresh[ids] = True
    if forward:
      self.env.sim.forward()

  def place_lib(self, env_ids: torch.Tensor, lib: "PatchLibrary", idx: torch.Tensor, rot_rel: torch.Tensor) -> None:
    """idx (m,) library patch per env (-1: no scene). The library's classes must be the rules' classes."""
    if tuple(lib.classes) != tuple(self.rules.classes):
      raise ValueError(f"library classes {lib.classes} != rules {self.rules.name} classes {self.rules.classes}")
    if lib.K > self.K:
      raise ValueError(f"library needs {lib.K} boxes, the entity has {self.K}")
    j = idx.clamp_min(0)
    local = torch.zeros(len(idx), self.K, 8, device=self.dev)
    local[..., 3] = 1.0
    cls = torch.full((len(idx), self.K), -1, dtype=torch.long, device=self.dev)
    local[:, :lib.K] = lib.local[j]
    cls[:, :lib.K] = lib.cls[j]
    self.place(env_ids, local, cls, rot_rel, has=idx >= 0)

  def check_placed(self) -> torch.Tensor:
    """Largest distance between intended and simulated box centres (needs a forward after place). Host read."""
    gx = self.env.sim.data.geom_xpos[:, self.ids]
    live = self.cls >= 0
    if not bool(live.any()):
      return torch.zeros(())
    return (gx - self.box[..., :3]).norm(dim=-1)[live].max()

  # ---------------------------------------------------------------------------------------------- recorder
  def _acc_reset(self) -> None:
    dev, N = self.dev, self.NPC + 1                                         # + 1: dump slot for skipped contacts
    if getattr(self, "a_fsum", None) is not None and self.backend == "warp":
      for t in (self.a_fsum, self.a_fmax, self.a_cnt, self.a_onset, self.a_von, self.a_pen):
        t.zero_()
      self.a_n = 0
      return
    self.a_fsum = torch.zeros(N, device=dev)
    self.a_fmax = torch.zeros(N, device=dev)
    self.a_cnt = torch.zeros(N, dtype=torch.int32, device=dev)
    self.a_onset = torch.zeros(N, dtype=torch.int32, device=dev)
    self.a_von = torch.zeros(N, device=dev)
    self.a_pen = torch.zeros(self.n + 1, device=dev)
    self.a_n = 0

  def hook(self) -> None:
    import mujoco_warp as mjw  # noqa: F401
    import warp as wp
    env = self.env
    robot = env.scene["robot"]
    original = robot.update
    wd = env.sim.wp_data
    cap = int(wd.contact.pos.shape[0])
    L = min(cap, max(1, self.prefix_per_env) * self.n)
    self.cap, self.L = cap, L
    self._ids = wp.array(torch.arange(L, dtype=torch.int32).numpy(), dtype=int, device=env.sim.wp_device)
    self._out = wp.zeros(L, dtype=wp.spatial_vector, device=env.sim.wp_device)
    self._t = dict(pos=wp.to_torch(wd.contact.pos), geom=wp.to_torch(wd.contact.geom), world=wp.to_torch(wd.contact.worldid),
                   frame=wp.to_torch(wd.contact.frame), dist=wp.to_torch(wd.contact.dist), efc=wp.to_torch(wd.contact.efc_address),
                   nacon=wp.to_torch(wd.nacon), force=wp.to_torch(self._out))
    self._index = torch.arange(L, device=self.dev)
    self._dump = torch.full((L,), self.NPC, dtype=torch.long, device=self.dev)
    if self.backend == "warp":
      self._warp_setup()

    def hooked(dt):
      original(dt)
      if self.active:
        self.substep()
    robot.update = hooked
    self.hooked = True

  def _warp_setup(self) -> None:
    import warp as wp
    dev, wd = self.dev, self.env.sim.wp_data
    i32 = lambda t: wp.from_torch(t.int().contiguous(), dtype=wp.int32)
    self._wk = _kernels()
    self._w_is_robot, self._w_part = i32(self.is_robot), i32(self.part_of_geom)
    self._w_slot, self._w_cap, self._w_ci = i32(self.slot_of_geom), i32(self.hand_caps), i32(self.coll_index)
    self._keep_alive = [self.is_robot.int(), self.part_of_geom.int(), self.slot_of_geom.int(), self.hand_caps.int(), self.coll_index.int()]
    self._w_is_robot, self._w_part, self._w_slot, self._w_cap, self._w_ci = (wp.from_torch(t, dtype=wp.int32) for t in self._keep_alive)
    self._w_cls = wp.from_torch(self.cls32, dtype=wp.int32)
    self._w_live = wp.from_torch(self.live32, dtype=wp.int32)
    self._oz = self.origins[:, 2].contiguous().float()
    self._w_oz = wp.from_torch(self._oz, dtype=wp.float32)
    self._vel = torch.zeros(self.n, len(self.robot_coll), 3, device=dev)
    self._w_vel = wp.from_torch(self._vel, dtype=wp.vec3)
    N = self.NPC + 1
    self._fsub = torch.zeros(N, device=dev)
    self._present = torch.zeros(N, dtype=torch.int32, device=dev)
    self._vsub = torch.zeros(N, device=dev)
    self._prev = torch.zeros(N, dtype=torch.int32, device=dev)
    self._oobw = torch.zeros(self.n * self.P, dtype=torch.int32, device=dev)     # wall contacts relabelled 'oob' (v2)
    self._w = {k: wp.from_torch(getattr(self, k), dtype=(wp.int32 if getattr(self, k).dtype == torch.int32 else wp.float32))
               for k in ("_fsub", "_present", "_vsub", "_prev", "a_fsum", "a_fmax", "a_cnt", "a_onset", "a_von", "a_pen", "_oobw")}
    self._w_box = wp.from_torch(self.box, dtype=wp.float32)
    G = len(self.vg)
    self._lg = torch.tensor([v["geom"] for v in self.vg], dtype=torch.int32, device=dev)
    self._lc = torch.stack([v["centre"] for v in self.vg]).float().contiguous()
    self._lr = torch.tensor([v["radius"] for v in self.vg], dtype=torch.float32, device=dev)
    self._pts = torch.cat([v["pts"] for v in self.vg]).float().contiguous()
    self._plink = torch.cat([torch.full((len(v["pts"]),), j, dtype=torch.int32, device=dev) for j, v in enumerate(self.vg)])
    self._ppart = torch.cat([v["part"] for v in self.vg]).int().contiguous()
    M = min(self.per_geom, self.K)
    self._cand = torch.zeros(self.n, G, M, dtype=torch.int32, device=dev)
    self._ncand = torch.zeros(self.n, G, dtype=torch.int32, device=dev)
    self._over = torch.zeros(1, dtype=torch.int32, device=dev)
    self._best = torch.zeros(self.n, self.P, dtype=torch.int32, device=dev)
    self._wi = dict(lg=wp.from_torch(self._lg, dtype=wp.int32), lc=wp.from_torch(self._lc, dtype=wp.vec3),
                    lr=wp.from_torch(self._lr, dtype=wp.float32), pts=wp.from_torch(self._pts, dtype=wp.vec3),
                    plink=wp.from_torch(self._plink, dtype=wp.int32), ppart=wp.from_torch(self._ppart, dtype=wp.int32),
                    cand=wp.from_torch(self._cand, dtype=wp.int32), ncand=wp.from_torch(self._ncand, dtype=wp.int32),
                    over=wp.from_torch(self._over, dtype=wp.int32), best=wp.from_torch(self._best, dtype=wp.int32))
    self._M = M

  def _sync_time(self, key: str, t0: float | None = None):
    if self.prof is None:
      return None
    import time
    torch.cuda.synchronize()
    now = time.perf_counter()
    if t0 is not None:
      self.prof[key] += now - t0
    return now

  def substep(self) -> None:
    t0 = self._sync_time("substep")
    if self.backend == "warp":
      self._substep_warp()
    else:
      self._substep_torch()
    self._sync_time("substep", t0)

  def _substep_warp(self) -> None:
    """Fused path: contact_force + one kernel over the first L contact slots + one fold kernel over the cells."""
    import mujoco_warp as mjw
    import warp as wp
    env, wd = self.env, self.env.sim.wp_data
    data = env.sim.data
    gpos = data.geom_xpos[:, self.robot_coll]
    torch.sub(gpos, self.prev_gpos, out=self._vel)
    self._vel.div_(self.dt)
    self._vel.masked_fill_(self.fresh[:, None, None], 0.0)
    self.prev_gpos.copy_(gpos)
    self.fresh.zero_()
    nacon = self._t["nacon"][0]
    st = self.stat
    st["substeps"] += 1
    st["max_nacon"] = torch.maximum(st["max_nacon"], nacon.long())
    st["overflow_steps"] += (nacon > self.L).long()
    k, w = self._wk, self._w
    band = self.rules.wall_band
    with wp.ScopedDevice(env.sim.wp_device):
      mjw.contact_force(env.sim.wp_model, wd, self._ids, False, self._out)
      wp.launch(k["contact_cells"], dim=self.L,
                inputs=[wd.nacon, wd.contact.geom, wd.contact.worldid, wd.contact.efc_address, wd.contact.pos, wd.contact.frame,
                        wd.contact.dist, self._out, self._w_is_robot, self._w_part, self._w_slot, self._w_cap, self._w_ci,
                        wd.xpos, wd.xmat, self._w_cls, self._w_oz, self._w_vel, self.n, self.P, self.C,
                        1 if band is not None else 0, float(band[0]) if band else 0.0, float(band[1]) if band else 0.0,
                        self.C - 1, HAND_X_MIN, w["_fsub"], w["_present"], w["_vsub"], w["a_pen"], w["_oobw"]])
      wp.launch(k["fold"], dim=self.NPC,
                inputs=[w["_fsub"], w["_present"], w["_vsub"], w["_prev"], w["a_fsum"], w["a_fmax"], w["a_cnt"], w["a_onset"], w["a_von"]])
    self.a_n += 1

  def _substep_torch(self) -> None:
    """One physics step: fold its robot-scene contacts into the (env, part, class) accumulators. Fixed-size: the
    first L contact slots are processed with masks; contacts past L (nacon > L) are counted in stat overflow_steps."""
    import mujoco_warp as mjw
    import warp as wp
    env, t, dev = self.env, self._t, self.dev
    P, C, L = self.P, self.C, self.L
    data = env.sim.data
    gpos = data.geom_xpos[:, self.robot_coll]
    vel = torch.where(self.fresh[:, None, None], torch.zeros_like(gpos), (gpos - self.prev_gpos) / self.dt)
    self.prev_gpos.copy_(gpos)
    self.fresh.zero_()
    nacon = t["nacon"][0]
    st = self.stat
    st["substeps"] += 1
    st["max_nacon"] = torch.maximum(st["max_nacon"], nacon.long())
    st["overflow_steps"] += (nacon > L).long()
    with wp.ScopedDevice(env.sim.wp_device):
      mjw.contact_force(env.sim.wp_model, env.sim.wp_data, self._ids, False, self._out)
    geom = t["geom"][:L].long()
    live = (self._index < nacon) & (geom.amin(dim=1) >= 0) & (t["efc"][:L, 0] >= 0)
    g0, g1 = geom[:, 0].clamp_min(0), geom[:, 1].clamp_min(0)
    r0, r1 = self.is_robot[g0], self.is_robot[g1]
    rg = torch.where(r1, g1, g0)
    og = torch.where(r1, g0, g1)
    slot = self.slot_of_geom[og]
    keep = live & (r0 ^ r1)
    w = t["world"][:L].long().clamp(0, self.n - 1)
    part = self.part_of_geom[rg].clamp_min(0)
    caps = self.hand_caps[rg]
    body = caps.clamp_min(0)
    xpos = data.xpos[w, body]
    xmat = data.xmat[w, body].reshape(-1, 3, 3)
    pos = t["pos"][:L]
    xloc = ((pos - xpos)[:, None, :] @ xmat).squeeze(1)[:, 0]
    part = torch.where((caps >= 0) & (xloc < HAND_X_MIN), torch.full_like(part, 3), part)
    cls = torch.where(slot >= 0, self.cls[w, slot.clamp_min(0)], torch.zeros_like(slot)).clamp_min(0)
    if self.rules.wall_band is not None:                                     # v2: a wall touched outside the hand band
      lo, hi = self.rules.wall_band
      z = pos[:, 2] - self.origins[w, 2]
      cls = torch.where((cls == 1) & ((z < lo) | (z > hi)), torch.full_like(cls, self.C - 1), cls)
    fn = t["force"][:, 0].clamp_min(0.0)
    normal = t["frame"][:L, 0, :]
    n_out = torch.where(r1[:, None], normal, -normal)                       # from the scene into the robot
    v = vel[w, self.coll_index[rg]]
    approach = (-(v * n_out).sum(dim=1)).clamp_min(0.0)
    flat = torch.where(keep, (w * P + part) * C + cls, self._dump)
    fsub = torch.zeros(self.NPC + 1, device=dev).index_add_(0, flat, torch.where(keep, fn, torch.zeros_like(fn)))
    present = torch.zeros(self.NPC + 1, dtype=torch.bool, device=dev).index_fill_(0, flat, True)
    vsub = torch.zeros(self.NPC + 1, device=dev).scatter_reduce_(0, flat, torch.where(keep, approach, torch.zeros_like(approach)),
                                                                  reduce="amax", include_self=True)
    pen = torch.where(keep & (cls != 0), (-t["dist"][:L]).clamp_min(0.0), torch.zeros_like(fn))
    wk = torch.where(keep & (cls != 0), w, torch.full_like(w, self.n))
    self.a_pen.scatter_reduce_(0, wk, pen, reduce="amax", include_self=True)
    present[-1] = False
    prev = torch.cat([self.prev_present.view(-1), torch.zeros(1, dtype=torch.bool, device=dev)])
    onset = present & ~prev
    self.prev_present = present[:-1].view(self.n, P, C)
    self.a_fsum += fsub
    self.a_fmax = torch.maximum(self.a_fmax, fsub)
    self.a_cnt += present.int()
    self.a_onset += onset.int()
    self.a_von = torch.maximum(self.a_von, torch.where(onset, vsub, torch.zeros_like(vsub)))
    self.a_n += 1

  def cells_now(self) -> dict:
    """This control step's recorder cells (n, P, C) and penetration; resets the accumulators."""
    n, P, C = self.n, self.P, self.C
    k = max(self.a_n, 1)
    t0 = self._sync_time("cells")
    cp = (lambda x: x.clone()) if self.backend == "warp" else (lambda x: x)
    out = dict(fmean=(self.a_fsum[:-1] / k).view(n, P, C), fmax=cp(self.a_fmax[:-1]).view(n, P, C),
               cnt=cp(self.a_cnt[:-1]).view(n, P, C), onset=cp(self.a_onset[:-1]).view(n, P, C),
               von=cp(self.a_von[:-1]).view(n, P, C), pen=self.a_pen[:-1].clone())
    if getattr(self, "_oobw", None) is not None:
      out["oobwall"] = self._oobw.view(n, P).clone()
      self._oobw.zero_()
    self._acc_reset()
    self._sync_time("cells", t0)
    return out

  def end_step(self, down: torch.Tensor, step: torch.Tensor | None = None) -> dict:
    """Once per control step after the physics substeps: recorder cells -> intrusion -> judge.
    down (n,) bool: the fall criterion on the post-step state. step: per-env step index in the episode (default:
    an internal counter advanced here and cleared by reset_judge). Returns the judge output (OnlineJudge.step)."""
    s = self.step_count if step is None else step
    cells = self.cells_now()
    t0 = self._sync_time("intrusion")
    depth, which = self.intrusion()
    self._sync_time("intrusion", t0)
    self.cells = dict(cells, intr=depth, intr_cls=which)
    t0 = self._sync_time("judge")
    out = self.judge.step(s, cells["cnt"], cells["fmean"], cells["von"], depth, which, down)
    self._sync_time("judge", t0)
    if step is None:
      self.step_count += 1
    return out

  def reset_judge(self, env_ids: torch.Tensor, win0=0, bw=None) -> None:
    self.judge.reset(env_ids, win0, bw)
    self.step_count[env_ids] = 0
    self.prev_present[env_ids] = False
    if getattr(self, "_prev", None) is not None:
      self._prev[:-1].view(self.n, self.P, self.C)[env_ids] = 0
    self.fresh[env_ids] = True

  # ---------------------------------------------------------------------------------------------- intrusion
  def _visual_points(self, mj, names, robot_bodies, hand_points: bool) -> None:
    import numpy as np
    import mujoco
    by_geom = {}
    for g in range(mj.ngeom):
      if int(mj.geom_bodyid[g]) not in robot_bodies or mj.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
        continue
      if mj.geom_contype[g] or mj.geom_conaffinity[g]:
        continue
      mid = int(mj.geom_dataid[g])
      v = np.asarray(mj.mesh_vert[mj.mesh_vertadr[mid]: mj.mesh_vertadr[mid] + mj.mesh_vertnum[mid]], dtype=np.float64)
      key = np.floor(v / VIS_VOXEL).astype(np.int64)
      _, first = np.unique(key, axis=0, return_index=True)
      v = v[np.sort(first)]
      mesh_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_MESH, mid) or names[g]
      body_name = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(mj.geom_bodyid[g])) or ""
      p = part_of_name(mesh_name)
      if p < 0:
        p = part_of_name(body_name)
      if p < 0:
        continue
      parts = np.full(len(v), p)
      if p in (1, 2) and "wrist" in body_name:
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, mj.geom_quat[g])
        inbody = v @ rot.reshape(3, 3).T + mj.geom_pos[g]
        parts = np.where(inbody[:, 0] >= HAND_X_MIN, p, 3)
      if not hand_points:
        keep = ~np.isin(parts, (1, 2))                                     # the judge never reads hand intrusion
        v, parts = v[keep], parts[keep]
      if len(v):
        by_geom[g] = (v, parts)
    dev = self.dev
    self.vg = []
    for g, (v, parts) in by_geom.items():
      lo, hi = v.min(0), v.max(0)
      c = (lo + hi) / 2
      r = float(np.linalg.norm(v - c, axis=1).max())
      self.vg.append(dict(geom=g, pts=torch.tensor(v, dtype=torch.float, device=dev), part=torch.tensor(parts, device=dev).long(),
                          centre=torch.tensor(c, dtype=torch.float, device=dev), radius=r,
                          parts=sorted(set(int(x) for x in parts))))
    self.vis_points = int(sum(len(x["pts"]) for x in self.vg))

  def intrusion(self) -> tuple[torch.Tensor, torch.Tensor]:
    if self.backend == "warp" and getattr(self, "_wi", None) is not None:
      return self._intrusion_warp()
    return self._intrusion_torch()

  def _intrusion_warp(self) -> tuple[torch.Tensor, torch.Tensor]:
    import warp as wp
    wd, wi, k = self.env.sim.wp_data, self._wi, self._wk
    self._best.zero_()
    with wp.ScopedDevice(self.env.sim.wp_device):
      wp.launch(k["link_cands"], dim=(self.n, len(self.vg)),
                inputs=[self._w_box, self._w_live, wd.geom_xpos, wd.geom_xmat, wi["lg"], wi["lc"], wi["lr"], self.K, self._M,
                        wi["cand"], wi["ncand"], wi["over"]])
      wp.launch(k["point_depth"], dim=(self.n, int(self._pts.shape[0])),
                inputs=[self._w_box, self._w_cls, wd.geom_xpos, wd.geom_xmat, wi["lg"], wi["plink"], wi["pts"], wi["ppart"],
                        wi["cand"], wi["ncand"], wi["best"]])
    q = self._best.long()
    depth = (q // 16).float() * 1.0e-6
    which = torch.where(depth > 0, q % 16, torch.full_like(q, -1))
    self.stat["geom_overflow"] = self._over[0].long()
    return depth, which

  def _intrusion_torch(self) -> tuple[torch.Tensor, torch.Tensor]:
    """(n, P) deepest visual-sample penetration into a non-floor box per part, and that box's class (-1: none).
    Per visual link: boxes whose exact distance to the link's bounding sphere centre is <= its radius are the
    candidates (the nearest per_geom of them; more than per_geom counts in stat geom_overflow)."""
    data, dev = self.env.sim.data, self.dev
    n, P, m = self.n, self.P, min(self.per_geom, self.K)
    out = torch.zeros(n, P, device=dev)
    which = torch.full((n, P), -1, dtype=torch.long, device=dev)
    b = self.box
    for vg in self.vg:
      gx = data.geom_xpos[:, vg["geom"]]                                     # (n, 3)
      gm = data.geom_xmat[:, vg["geom"]].reshape(n, 3, 3)
      sc = gx + gm @ vg["centre"]
      dx, dy, dz = sc[:, None, 0] - b[..., 0], sc[:, None, 1] - b[..., 1], sc[:, None, 2] - b[..., 2]
      qx, qy = b[..., 3] * dx + b[..., 4] * dy, -b[..., 4] * dx + b[..., 3] * dy
      ex = (qx.abs() - b[..., 5]).clamp_min(0.0)
      ey = (qy.abs() - b[..., 6]).clamp_min(0.0)
      ez = (dz.abs() - b[..., 7]).clamp_min(0.0)
      dist = torch.sqrt(ex * ex + ey * ey + ez * ez)
      ok = self.live & (dist <= vg["radius"])
      self.stat["geom_overflow"] += (ok.sum(dim=1) > m).sum()
      cd, ck = torch.topk(torch.where(ok, dist, torch.full_like(dist, float("inf"))), m, dim=1, largest=False)
      cand = torch.isfinite(cd)                                               # (n, m)
      cb = torch.gather(b, 1, ck[..., None].expand(-1, -1, 8))                # (n, m, 8)
      ccls = torch.gather(self.cls, 1, ck)
      pw = gx[:, None, :] + vg["pts"][None, :, :] @ gm.transpose(1, 2)        # (n, Ng, 3)
      dx = pw[:, :, None, 0] - cb[:, None, :, 0]
      dy = pw[:, :, None, 1] - cb[:, None, :, 1]
      dz = pw[:, :, None, 2] - cb[:, None, :, 2]
      qx = cb[:, None, :, 3] * dx + cb[:, None, :, 4] * dy
      qy = -cb[:, None, :, 4] * dx + cb[:, None, :, 3] * dy
      inside = torch.minimum(torch.minimum(cb[:, None, :, 5] - qx.abs(), cb[:, None, :, 6] - qy.abs()), cb[:, None, :, 7] - dz.abs())
      inside = torch.where(cand[:, None, :], inside, torch.full_like(inside, -1.0))
      best, j = inside.max(dim=2)                                             # (n, Ng)
      best = best.clamp_min(0.0)
      bcls = torch.gather(ccls, 1, j)
      for p in vg["parts"]:
        sel = vg["part"] == p
        bp = torch.where(sel[None, :], best, torch.full_like(best, -1.0))
        v, jj = bp.max(dim=1)
        better = v > out[:, p]
        out[:, p] = torch.where(better, v, out[:, p])
        which[:, p] = torch.where(better & (v > 0), torch.gather(bcls, 1, jj[:, None]).squeeze(1), which[:, p])
    return out, which

  # ---------------------------------------------------------------------------------------------- observations
  def nearest_boxes(self, k: int = 16) -> torch.Tensor:
    """(n, k, 10) the k live non-floor boxes nearest the pelvis (2D footprint distance), in the robot's yaw frame:
    x, y (centre), z_bottom, z_top (above the env floor), half x, half y, cos, sin (box yaw relative to the heading),
    allowed (1: an allowed hand-support class), distance. Missing boxes are zeros with distance 9."""
    data, dev = self.env.sim.data, self.dev
    robot = self.env.scene["robot"]
    fq = robot.indexing.free_joint_q_adr.long()
    q = data.qpos[:, fq[3:7]]
    hd = yaw_of_quat(q)
    pel = data.qpos[:, fq[:2]]
    b = self.box
    dx, dy = pel[:, None, 0] - b[..., 0], pel[:, None, 1] - b[..., 1]
    qx, qy = b[..., 3] * dx + b[..., 4] * dy, -b[..., 4] * dx + b[..., 3] * dy
    d2 = torch.sqrt((qx.abs() - b[..., 5]).clamp_min(0) ** 2 + (qy.abs() - b[..., 6]).clamp_min(0) ** 2)
    d2 = torch.where(self.live, d2, torch.full_like(d2, 9.0))
    k = min(k, self.K)
    dk, ik = torch.topk(d2, k, dim=1, largest=False)
    g = torch.gather(b, 1, ik[..., None].expand(-1, -1, 8))
    ch, sh = torch.cos(hd)[:, None], torch.sin(hd)[:, None]
    rx, ry = g[..., 0] - pel[:, None, 0], g[..., 1] - pel[:, None, 1]
    x, y = ch * rx + sh * ry, -sh * rx + ch * ry
    zc = g[..., 2] - self.origins[:, None, 2]
    yc, ys = g[..., 3] * ch + g[..., 4] * sh, g[..., 4] * ch - g[..., 3] * sh
    allowed = torch.zeros(self.C, device=dev)
    allowed[self.judge.allowed] = 1.0
    al = allowed[torch.gather(self.cls, 1, ik).clamp_min(0)]
    feat = torch.stack([x, y, zc - g[..., 7], zc + g[..., 7], g[..., 5], g[..., 6], yc, ys, al, dk], dim=-1)
    missing = dk >= 9.0
    return torch.where(missing[..., None], torch.cat([torch.zeros_like(feat[..., :9]), torch.full_like(feat[..., 9:], 9.0)], -1), feat)

  def ray_profile(self, rays: int = 36, zlo: float = 0.05, zhi: float = 1.25, cap: float = 2.5) -> torch.Tensor:
    """(n, rays) 2D free distance from the pelvis along rays at body heights (robot yaw frame, ray 0 = forward,
    counter-clockwise) to the first live non-floor box footprint overlapping zlo..zhi, capped at cap."""
    data = self.env.sim.data
    robot = self.env.scene["robot"]
    fq = robot.indexing.free_joint_q_adr.long()
    hd = yaw_of_quat(data.qpos[:, fq[3:7]])
    pel = data.qpos[:, fq[:2]]
    b = self.box
    zc = b[..., 2] - self.origins[:, None, 2]
    body = self.live & (zc - b[..., 7] < zhi) & (zc + b[..., 7] > zlo)
    ox, oy = pel[:, None, 0] - b[..., 0], pel[:, None, 1] - b[..., 1]                     # (n, K)
    lox, loy = b[..., 3] * ox + b[..., 4] * oy, -b[..., 4] * ox + b[..., 3] * oy
    out = torch.full((self.n, rays), cap, device=self.dev)
    big = 1e9

    def slab(o, d, h):
      ok = d.abs() > 1e-9
      inv = 1.0 / torch.where(ok, d, torch.ones_like(d))
      t1, t2 = (-h - o) * inv, (h - o) * inv
      inside = o.abs() <= h
      lo = torch.where(ok, torch.minimum(t1, t2), torch.where(inside, torch.full_like(d, -big), torch.full_like(d, big)))
      hi = torch.where(ok, torch.maximum(t1, t2), torch.where(inside, torch.full_like(d, big), torch.full_like(d, -big)))
      return lo, hi
    for r0 in range(0, rays, 6):                                                          # 6 rays at a time
      ang = hd[:, None] + torch.arange(r0, min(rays, r0 + 6), device=self.dev) * (2 * math.pi / rays)
      ux, uy = torch.cos(ang)[..., None], torch.sin(ang)[..., None]                      # (n, r, 1)
      ldx = b[:, None, :, 3] * ux + b[:, None, :, 4] * uy                                # (n, r, K)
      ldy = -b[:, None, :, 4] * ux + b[:, None, :, 3] * uy
      lx, hx = slab(lox[:, None, :], ldx, b[:, None, :, 5])
      ly, hy = slab(loy[:, None, :], ldy, b[:, None, :, 6])
      tmin = torch.maximum(lx, ly).clamp_min(0.0)
      hit = body[:, None, :] & (torch.minimum(hx, hy) >= tmin)
      out[:, r0:r0 + ang.shape[1]] = torch.where(hit, tmin, torch.full_like(tmin, cap)).amin(dim=2).clamp_max(cap)
    return out

  def meta(self) -> dict:
    st = {k: int(v) for k, v in self.stat.items()}
    return dict(module="scene_online", rules=self.rules.name, classes=list(self.rules.classes), boxes=self.K,
                prefix=int(getattr(self, "L", 0)), contact_capacity=int(getattr(self, "cap", 0)), per_geom=self.per_geom,
                vis_points=self.vis_points, vis_links=len(self.vg), saturated=bool(st["max_nacon"] >= getattr(self, "cap", 1 << 62)),
                backend=self.backend, profile_s=self.prof, **st)


# ---------------------------------------------------------------------------------------------------- warp kernels
_K: dict = {}


def _kernels() -> dict:
  """Fused GPU kernels (built once): contact_cells + fold per physics step, link_cands + point_depth per control step."""
  if _K:
    return _K
  import warp as wp

  @wp.kernel
  def contact_cells(nacon: wp.array(dtype=wp.int32), geom: wp.array(dtype=wp.vec2i), worldid: wp.array(dtype=wp.int32),
                    efc: wp.array2d(dtype=wp.int32), pos: wp.array(dtype=wp.vec3), frame: wp.array(dtype=wp.mat33),
                    dist: wp.array(dtype=wp.float32), force: wp.array(dtype=wp.spatial_vector),
                    is_robot: wp.array(dtype=wp.int32), part_of: wp.array(dtype=wp.int32), slot_of: wp.array(dtype=wp.int32),
                    hand_cap: wp.array(dtype=wp.int32), coll_index: wp.array(dtype=wp.int32),
                    xpos: wp.array2d(dtype=wp.vec3), xmat: wp.array2d(dtype=wp.mat33), cls: wp.array2d(dtype=wp.int32),
                    origin_z: wp.array(dtype=wp.float32), vel: wp.array2d(dtype=wp.vec3), nworld: int, P: int, C: int,
                    band: int, lo: float, hi: float, oob: int, hand_x_min: float,
                    fsub: wp.array(dtype=wp.float32), present: wp.array(dtype=wp.int32), vsub: wp.array(dtype=wp.float32),
                    pen: wp.array(dtype=wp.float32), oobw: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    if tid >= nacon[0]:
      return
    g = geom[tid]
    if g[0] < 0 or g[1] < 0:
      return
    if efc[tid, 0] < 0:
      return
    r0 = is_robot[g[0]]
    r1 = is_robot[g[1]]
    if r0 == r1:
      return
    rg = g[0]
    og = g[1]
    if r1 == 1:
      rg = g[1]
      og = g[0]
    w = worldid[tid]
    if w < 0 or w >= nworld:
      return
    part = part_of[rg]
    cap = hand_cap[rg]
    p = pos[tid]
    if cap >= 0:
      m = xmat[w, cap]
      d = p - xpos[w, cap]
      xl = d[0] * m[0, 0] + d[1] * m[1, 0] + d[2] * m[2, 0]
      if xl < hand_x_min:
        part = 3
    c = int(0)
    s = slot_of[og]
    if s >= 0:
      c = wp.max(cls[w, s], 0)
    if band == 1 and c == 1:
      z = p[2] - origin_z[w]
      if z < lo or z > hi:
        c = oob
        wp.atomic_add(oobw, w * P + part, 1)
    fn = wp.max(force[tid][0], 0.0)
    fr = frame[tid]
    nrm = wp.vec3(fr[0, 0], fr[0, 1], fr[0, 2])
    if r1 == 0:
      nrm = -nrm
    v = vel[w, coll_index[rg]]
    ap = wp.max(-wp.dot(v, nrm), 0.0)
    flat = (w * P + part) * C + c
    wp.atomic_add(fsub, flat, fn)
    present[flat] = 1
    wp.atomic_max(vsub, flat, ap)
    if c != 0:
      wp.atomic_max(pen, w, wp.max(-dist[tid], 0.0))

  @wp.kernel
  def fold(fsub: wp.array(dtype=wp.float32), present: wp.array(dtype=wp.int32), vsub: wp.array(dtype=wp.float32),
           prev: wp.array(dtype=wp.int32), a_fsum: wp.array(dtype=wp.float32), a_fmax: wp.array(dtype=wp.float32),
           a_cnt: wp.array(dtype=wp.int32), a_onset: wp.array(dtype=wp.int32), a_von: wp.array(dtype=wp.float32)):
    i = wp.tid()
    p = present[i]
    f = fsub[i]
    a_fsum[i] = a_fsum[i] + f
    a_fmax[i] = wp.max(a_fmax[i], f)
    a_cnt[i] = a_cnt[i] + p
    if p == 1 and prev[i] == 0:
      a_onset[i] = a_onset[i] + 1
      a_von[i] = wp.max(a_von[i], vsub[i])
    prev[i] = p
    fsub[i] = 0.0
    present[i] = 0
    vsub[i] = 0.0

  @wp.kernel
  def link_cands(box: wp.array3d(dtype=wp.float32), live: wp.array2d(dtype=wp.int32), gx: wp.array2d(dtype=wp.vec3),
                 gm: wp.array2d(dtype=wp.mat33), link_geom: wp.array(dtype=wp.int32), link_c: wp.array(dtype=wp.vec3),
                 link_r: wp.array(dtype=wp.float32), K: int, M: int, cand: wp.array3d(dtype=wp.int32),
                 ncand: wp.array2d(dtype=wp.int32), over: wp.array(dtype=wp.int32)):
    e, j = wp.tid()
    gid = link_geom[j]
    sc = gx[e, gid] + gm[e, gid] * link_c[j]
    r = link_r[j]
    n = int(0)
    for k in range(K):
      if live[e, k] == 0:
        continue
      dx = sc[0] - box[e, k, 0]
      dy = sc[1] - box[e, k, 1]
      dz = sc[2] - box[e, k, 2]
      qx = box[e, k, 3] * dx + box[e, k, 4] * dy
      qy = -box[e, k, 4] * dx + box[e, k, 3] * dy
      ex = wp.max(wp.abs(qx) - box[e, k, 5], 0.0)
      ey = wp.max(wp.abs(qy) - box[e, k, 6], 0.0)
      ez = wp.max(wp.abs(dz) - box[e, k, 7], 0.0)
      if ex * ex + ey * ey + ez * ez <= r * r:
        if n < M:
          cand[e, j, n] = k
        n += 1
    ncand[e, j] = wp.min(n, M)
    if n > M:
      wp.atomic_add(over, 0, 1)

  @wp.kernel
  def point_depth(box: wp.array3d(dtype=wp.float32), cls: wp.array2d(dtype=wp.int32), gx: wp.array2d(dtype=wp.vec3),
                  gm: wp.array2d(dtype=wp.mat33), link_geom: wp.array(dtype=wp.int32), plink: wp.array(dtype=wp.int32),
                  pts: wp.array(dtype=wp.vec3), ppart: wp.array(dtype=wp.int32), cand: wp.array3d(dtype=wp.int32),
                  ncand: wp.array2d(dtype=wp.int32), best: wp.array2d(dtype=wp.int32)):
    e, i = wp.tid()
    j = plink[i]
    nb = ncand[e, j]
    if nb == 0:
      return
    gid = link_geom[j]
    pw = gx[e, gid] + gm[e, gid] * pts[i]
    bd = float(0.0)
    bc = int(-1)
    for t in range(nb):
      k = cand[e, j, t]
      dx = pw[0] - box[e, k, 0]
      dy = pw[1] - box[e, k, 1]
      dz = pw[2] - box[e, k, 2]
      qx = box[e, k, 3] * dx + box[e, k, 4] * dy
      qy = -box[e, k, 4] * dx + box[e, k, 3] * dy
      dep = wp.min(wp.min(box[e, k, 5] - wp.abs(qx), box[e, k, 6] - wp.abs(qy)), box[e, k, 7] - wp.abs(dz))
      if dep > bd:
        bd = dep
        bc = cls[e, k]
    if bc >= 0:
      q = int(wp.min(bd, 1.0) * 1.0e6) * 16 + bc
      wp.atomic_max(best, e, ppart[i], q)

  _K.update(contact_cells=contact_cells, fold=fold, link_cands=link_cands, point_depth=point_depth)
  return _K


# ---------------------------------------------------------------------------------------------------- library / draws
class PatchLibrary:
  """Library JSON -> padded tensors on a device. local (N, K, 8) = cx, cy, cz, cos, sin, hx, hy, hz (patch frame,
  stand point at the origin), cls (N, K) class index in `classes` (-1 pad), split (N,) 0 train / 1 holdout,
  beta (N, 3) directions (deg) of furniture / wall / empty sides (nan if none)."""

  def __init__(self, path, device, classes: tuple | None = None, k_multiple: int = 16):
    import hashlib
    import json
    from pathlib import Path
    raw = Path(path).read_bytes()
    self.sha256 = hashlib.sha256(raw).hexdigest()
    lib = json.loads(raw)
    self.schema = lib["schema"]
    self.classes = tuple(lib["classes"]) if classes is None else classes
    pats = lib["patches"]
    nmax = max(len(p["boxes"]) for p in pats)
    K = int(k_multiple * math.ceil(nmax / k_multiple))
    N = len(pats)
    local = torch.zeros(N, K, 8)
    local[..., 3] = 1.0
    cls = torch.full((N, K), -1, dtype=torch.long)
    for i, p in enumerate(pats):
      b = torch.tensor(p["boxes"], dtype=torch.float64)
      if len(b) == 0:
        continue
      qw, qx, qy, qz = b[:, 6], b[:, 7], b[:, 8], b[:, 9]
      if float(qx.abs().max()) > 1e-6 or float(qy.abs().max()) > 1e-6:
        raise ValueError(f"patch {p['id']}: a box is not yaw-only")
      yaw = 2.0 * torch.atan2(qz, qw)
      local[i, :len(b)] = torch.stack([b[:, 0], b[:, 1], b[:, 2], torch.cos(yaw), torch.sin(yaw), b[:, 3], b[:, 4], b[:, 5]], 1).float()
      cls[i, :len(b)] = b[:, 10].long()
    self.K, self.N = K, N
    self.local, self.cls = local.to(device), cls.to(device)
    self.ids = [p["id"] for p in pats]
    if self.schema == "r6sc-patchlib-v1":                 # the R6 baseline scenes: all holdout (r7_scene_lane.md 3)
      split = ["holdout"] * N
    else:
      missing = [p["id"] for p in pats if p.get("split") not in ("train", "holdout")]
      if missing:
        raise ValueError(f"{len(missing)} patches without a train/holdout split, e.g. {missing[:3]}")
      split = [p["split"] for p in pats]
    self.split = torch.tensor([1 if s == "holdout" else 0 for s in split], device=device)
    nan = float("nan")
    self.beta = torch.tensor([[(p.get("beta_deg") or {}).get(k) if (p.get("beta_deg") or {}).get(k) is not None else nan
                               for k in ("furniture", "wall", "empty")] for p in pats], device=device)
    self.train_idx = (self.split == 0).nonzero().squeeze(1)
    self.holdout_idx = (self.split == 1).nonzero().squeeze(1)


_C1, _C2, _C3 = -7046029254386353131, -4658895280553007687, -7723592293110705685   # 0x9e3779b97f4a7c15, splitmix64


def _mix64(x: torch.Tensor) -> torch.Tensor:
  """splitmix64 finaliser on int64 (wrapping arithmetic; logical shifts emulated with masks)."""
  x = x ^ ((x >> 30) & 0x3FFFFFFFF)
  x = x * _C2
  x = x ^ ((x >> 27) & 0x1FFFFFFFFF)
  x = x * _C3
  x = x ^ ((x >> 31) & 0x1FFFFFFFF)
  return x


def counter_uniform(seed: int, env_ids: torch.Tensor, resets: torch.Tensor, k: int) -> torch.Tensor:
  """u in [0, 1) from (seed, env id, reset count, draw index k): the same numbers on any device or batch."""
  x = torch.full_like(env_ids.long(), int(seed) & 0x7FFFFFFFFFFFFFFF)
  x = _mix64(x + _C1 * (env_ids.long() + 1))
  x = _mix64(x + _C1 * (resets.long() + 1))
  x = _mix64(x + _C1 * torch.full_like(x, int(k) + 1))
  return ((x >> 11) & ((1 << 53) - 1)).double() * (2.0 ** -53)


class PatchSampler:
  """Seeded patch draws for training: reads only (env id, reset count). Holdout patches are never drawn
  (split == 'train' only). side weights: furniture / wall / empty / uniform direction of the patch side placed at
  the robot-frame angle phi (e.g. the planned push azimuth); default phi from the same counter stream."""

  def __init__(self, lib: PatchLibrary, seed: int, side_weights=(0.35, 0.35, 0.10, 0.20), split: str = "train"):
    if split != "train":
      raise ValueError("the training sampler draws the train split only")
    self.lib, self.seed = lib, int(seed)
    self.pool = lib.train_idx
    w = torch.tensor(side_weights, dtype=torch.float64)
    self.cum = (w / w.sum()).cumsum(0).to(lib.local.device)

  def draw(self, env_ids: torch.Tensor, resets: torch.Tensor, phi: torch.Tensor | None = None):
    dev = self.lib.local.device
    env_ids, resets = env_ids.to(dev), resets.to(dev)
    u = [counter_uniform(self.seed, env_ids, resets, k) for k in range(4)]
    idx = self.pool[(u[0] * len(self.pool)).long().clamp_max(len(self.pool) - 1)]
    side = torch.searchsorted(self.cum, u[1].to(self.cum.dtype).contiguous(), right=True).clamp_max(3)
    if phi is None:
      phi = (u[2] * 2.0 - 1.0) * math.pi
    beta = torch.cat([self.lib.beta[idx], torch.full((len(idx), 1), float("nan"), device=dev)], 1)
    b = torch.gather(beta, 1, side[:, None]).squeeze(1)
    b = torch.where(torch.isnan(b), (u[3] * 360.0).float(), b)                # side missing or 'uniform'
    rot = phi.float() - torch.deg2rad(b.float())
    return idx, rot, side
