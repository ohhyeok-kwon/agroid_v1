"""Negative 30 motion-range gate (coordinator 18:56, user "당연히 동작의 폭을 늘리고 싶어"): human R1 clip tracking per motion type.

Pool (frozen list, sha256 of its JSON):
  unseen   passed clips (bones_seed_t2 strict / loose) ABSENT from human_v2_armneutral/sonic_pkl, i.e. never in the R5-R10 LoRA
           motion library. Least self-penetrating arm-neutral form first; values are recorded.
  seen     (historical key name; read 'held-out') list_heldout clips present in the arm-neutral folder. They are NOT trained on:
           R5-R10 configs remove them with motion_lib_cfg.remove_motion_keys (the 587 held-out prefixes, mirrors included;
           motion_lib_base.py 441-448). Corrected 20:4x: an earlier note here said R5-R10 had seen them (wrong; the removal
           step after filter_motion_keys was missed). Feasible arm-neutral references (they passed the self-contact filter).
  failtier fail-tier clips (never converted into any sonic_pkl), only for types still short (run, balance); flagged.
  One clip per (family, actor), no mirror twins. Types are given by name hints AND checked on the reference kinematics;
  the window (<= 10 s) is placed on the type's own signal. Clips whose reference needs non-foot support (sitting low,
  kneeling: no chair in the sim) are kept apart (type 'support').
Episode: the clip's window is played once from its first frame (root pose, joints, velocities by finite difference), the
  T1 harness (sonic_g1.DeployR1Native, R1-IL reference), PD at 200 Hz with an action delay (5 ms steps; the delay buffer is
  filled with the first-frame targets). Reference version 'an' = arm-neutral constants of human_v2_armneutral (DATA_CARD)
  added to the 10 arm joints, clipped to the MJCF range (what the LoRA trackers learnt on); 'orig' = the clip as is.
  Failure = SONIC im_eval rule (reference-relative: anchor z error > 0.25 m, anchor orientation error^2 > 1 rad^2, any
  end-effector height error > 0.25 m); the episode stops there. Per frame: MPJPE g / l (14 bodies, mm), joint error
  (24, rad), arm joints, pelvis z of robot and reference.
  python -I neg30_clips.py pool OUT.json
  python -I neg30_clips.py run --list L.json --policy TAG=CKPT --version an --delay 0 [--jit 0] --out DIR --procs 24
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/rl-agent/.agent/tmp/t1_sonic/work")
sys.path.insert(0, "/rl-agent/src/r1adv/negative")
import neg30_body as NB  # noqa: E402  (NEG30_BODY r1wb | b1: R1_BODY / R1_XML before sonic_g1; --out/body.json)
B = "/rl-agent/dataset/r1_retarget"
CLIPS = f"{B}/bones_seed_t2/clips/bones_seed"
DT = 0.02
WIN_MAX, WIN_MIN = 10.0, 3.0
PER_TYPE, PER_TYPE_SEEN = 30, 15
ORDER = ["support", "run", "crouch_sit", "balance", "bend", "arm", "side", "turn", "transition", "walk", "other"]
HINT = {"run": r"\b(run|running|jog|jogging|sprint)", "crouch_sit": r"(crouch|squat|kneel|sit_|sitting|_sit\b|sit down|crawl|ground)",
        "bend": r"(bend|pick|stoop|bow|lean|tie_shoe|ground|reach_down|look_under)",
        "arm": r"(reach|lift|wave|waving|point|gesture|clap|throw|carry|push|pull|eat|drink|sip|phone|cigarette|mosquito|feeding|read|book|"
               r"crossed_arms|zippo|box|swing|punch|stretch|scratch|catch|grab|hold)",
        "balance": r"(one_leg|one_foot|single_leg|balance|stand_on_one|kick|tiptoe|stork)",
        "side": r"(sideway|side_step|sidestep|strafe|shuffle)", "turn": r"(turn|rotate|pivot|_arc_)",
        "transition": r"(start|stop|_to_|change_)", "walk": r"walk"}
ARM_MJ = ["left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
          "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"]


def family_actor(name):
  base, _, actor = name.partition("__")
  return re.sub(r"_\d+$", "", base), actor


def sm(x, n=25):
  return np.convolve(x, np.ones(n) / n, mode="same") if len(x) > n else x


def yaw_of(Q):
  return np.unwrap(np.arctan2(2 * (Q[:, 0] * Q[:, 3] + Q[:, 1] * Q[:, 2]), 1 - 2 * (Q[:, 2] ** 2 + Q[:, 3] ** 2)))


def pitch_of(Q):
  return np.arcsin(np.clip(2 * (Q[:, 0] * Q[:, 2] - Q[:, 3] * Q[:, 1]), -1, 1))


# ------------------------------------------------------------------ pool
def features(name, model, data, mj):
  z = np.load(f"{CLIPS}/{name}.npz", allow_pickle=False)
  if abs(float(z["fps"]) - 50.0) > 1e-6:
    return None
  q = z["qpos_m1"].astype(np.float64)
  T = len(q)
  if T < int(WIN_MIN / DT):
    return None
  P, Q = q[:, :3], q[:, 3:7]
  yaw = yaw_of(Q)
  v = np.gradient(P[:, :2], DT, axis=0)
  vf = np.stack([sm(v[:, 0] * np.cos(yaw) + v[:, 1] * np.sin(yaw)), sm(-v[:, 0] * np.sin(yaw) + v[:, 1] * np.cos(yaw))], 1)
  speed = np.linalg.norm(vf, axis=1)
  wz = sm(np.gradient(yaw, DT))
  jn = [str(x) for x in z["joint_names"]]
  ai = [jn.index(n) for n in ARM_MJ]
  arm = q[:, 7:][:, ai]
  bodies = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link"]
  bid = [model.body(b).id for b in bodies]
  sub = range(0, T, 2)
  X = np.zeros((len(sub), len(bodies), 3))
  for i, k in enumerate(sub):
    data.qpos[:] = q[k]
    mj.mj_kinematics(model, data)
    X[i] = data.xpos[bid]
  t2 = np.array(list(sub)) * DT
  pel_z = X[:, 0, 2]
  feet_z = X[:, 2:4, 2] - X[:, 2:4, 2].min()
  knee_z = X[:, 4:6, 2].min(1)
  torso_pitch = pitch_of(q[:, 3:7])[::2]
  stand_z = float(np.percentile(pel_z, 90))
  return dict(T=T, t2=t2, speed=speed, vf=vf, wz=wz, yaw=yaw, arm=arm, pel_z=pel_z, feet_z=feet_z, knee_z=knee_z,
              torso_pitch=torso_pitch, stand_z=stand_z)


def longest(mask):
  best, run, end = 0, 0, -1
  for i, m in enumerate(mask):
    run = run + 1 if m else 0
    if run > best:
      best, end = run, i
  return best, end


STAND_Z = 0.72        # R1 standing pelvis height (bones_seed_t2 idle clips: pelvis_z_med 0.72 m)


def classify(name, caption, F):
  """(type, event time s) from the reference kinematics (absolute pelvis height, single support, pelvis pitch, arm range,
  root speed / heading), name hints only as a tie-break for arm actions and transitions."""
  txt = (name + " " + (caption or "")).lower()
  hint = {k for k, rx in HINT.items() if re.search(rx, txt)}
  t2 = F["t2"]
  sp, sp2 = F["speed"], F["speed"][::2]
  n_kn, e_kn = longest(F["knee_z"] < 0.10)
  n_lo, e_lo = longest(F["pel_z"] < 0.30)
  if n_kn * 2 * DT >= 0.5 or n_lo * 2 * DT >= 0.5:
    return "support", float(t2[e_kn if n_kn >= n_lo else e_lo])
  if sp.max() > 1.6 or ("run" in hint and sp.max() > 1.2):
    return "run", float(np.argmax(sp) * DT)
  if F["pel_z"].min() < STAND_Z - 0.15:
    return "crouch_sit", float(t2[int(np.argmin(F["pel_z"]))])
  up = F["feet_z"] > 0.06
  single = up.any(1) & ~up.all(1)
  n_l, e_l = longest(single & (sp2[: len(single)] < 0.15))
  if n_l * 2 * DT >= 1.5:
    return "balance", float(t2[e_l] - n_l * DT)
  if F["torso_pitch"].max() > math.radians(25):
    return "bend", float(t2[int(np.argmax(F["torso_pitch"]))])
  rng = (F["arm"].max(0) - F["arm"].min(0)).max()
  k_ev = int(np.argmax(np.abs(F["arm"] - F["arm"][:1]).max(1)))
  still = sp[max(0, k_ev - 50):k_ev + 50].max() < 0.15
  if still and (rng > 1.0 or ("arm" in hint and rng > 0.5)):
    return "arm", float(k_ev * DT)
  lat = np.abs(F["vf"][:, 1])
  n_s, e_s = longest((lat > 0.2) & (lat > np.abs(F["vf"][:, 0])))
  if n_s * DT >= 1.0:
    return "side", float((e_s - n_s) * DT)
  if np.abs(F["yaw"][-1] - F["yaw"][0]) > 1.57 or np.abs(np.diff(F["yaw"][:: 50])).sum() > 1.57:
    return "turn", float(np.argmax(np.abs(F["wz"])) * DT)
  moving = sp > 0.3
  if "transition" in hint and moving.any() and not moving.all():
    k = int(np.argmax(np.abs(np.diff(moving.astype(int)))))
    return "transition", float(k * DT)
  n_w, e_w = longest(moving)
  if n_w * DT >= 2.0:
    return "walk", float((e_w - n_w) * DT)
  return "other", 0.0


def window(typ, t_ev, T):
  dur = T * DT
  L = min(WIN_MAX, dur)
  if typ in ("walk", "side", "run"):
    a = max(0.0, t_ev - 1.0)
  elif typ == "other":
    a = 0.0
  else:
    a = max(0.0, t_ev - L / 2)
  a = min(a, max(0.0, dur - L))
  return int(round(a / DT)), int(round(min(dur, a + L) / DT))


def build_pool(out):
  import mujoco as mj
  fi = [json.loads(l) for l in open(f"{B}/bones_seed_t2/filter_index.jsonl")]
  an_idx = {json.loads(l)["name"]: json.loads(l) for l in open(f"{B}/human_v2_armneutral/index.jsonl")}
  an_files = set()
  for dp, _dn, fn in os.walk(f"{B}/human_v2_armneutral/sonic_pkl"):
    an_files |= {f[:-4] for f in fn if f.endswith(".pkl") and f != "metadata.pkl"}
  npz = {f[:-4] for f in os.listdir(CLIPS)}
  model = mj.MjModel.from_xml_path(os.environ.get("R1_XML", "/rl-agent/src/r1lab/assets/r1/r1.xml"))
  assert model.nq == 31, model.nq            # free joint + the 24 joints in the clips' MJCF order
  data = mj.MjData(model)
  cand = []
  for r in fi:
    n = r["clip_id"]
    if n not in npz:
      continue
    if r.get("tier") in ("strict", "loose"):
      pool = "seen" if n in an_files else "unseen"
      if pool == "seen" and not r.get("heldout"):
        continue                                   # trained clip that is not even on the held-out list: not used
    elif r.get("tier") == "fail":
      pool = "failtier"
    else:
      continue
    cand.append((r, pool))
  rows = []
  t0 = time.time()
  for r, pool in cand:
    n = r["clip_id"]
    if pool == "failtier" and not re.search(HINT["run"] + "|" + HINT["balance"], (n + " " + (r.get("caption") or "")).lower()):
      continue
    F = features(n, model, data, mj)
    if F is None:
      continue
    typ, t_ev = classify(n, r.get("caption"), F)
    f0, f1 = window(typ, t_ev, F["T"])
    ai = an_idx.get(n, {})
    own_ok = ai.get("selfpen_share_1cm", 1.0) <= 0.05 and ai.get("selfpen_max_m", 1.0) <= 0.03
    rows.append(dict(name=n, pool=pool, type=typ, t_event=round(t_ev, 2), window=[f0, f1], tier=r.get("tier"), heldout=bool(r.get("heldout")),
                     own_selfpen_ok=bool(own_ok), selfpen_share_1cm=ai.get("selfpen_share_1cm"), selfpen_max_m=ai.get("selfpen_max_m"),
                     family_actor="|".join(family_actor(n)), caption=r.get("caption")))
  # selection: one clip per (family, actor); per type and pool, hash order; unseen prefers own self-penetration pass
  key = lambda x: hashlib.sha256(x["name"].encode()).hexdigest()  # noqa: E731
  pick, seen_fa = [], set()
  quota = {"unseen": PER_TYPE, "seen": PER_TYPE_SEEN, "failtier": PER_TYPE}
  for pool in ("unseen", "seen", "failtier"):
    for typ in ORDER:
      have = sum(1 for x in pick if x["type"] == typ and x["pool"] in ("unseen", "failtier")) if pool == "failtier" else 0
      if pool == "failtier" and (typ not in ("run", "balance") or have >= 20):
        continue
      C = sorted([x for x in rows if x["pool"] == pool and x["type"] == typ],    # least self-penetrating arm-neutral form first
                 key=lambda x: (not x["own_selfpen_ok"], round(x["selfpen_max_m"] if x["selfpen_max_m"] is not None else 9.0, 2), key(x)))
      k = 0
      for x in C:
        if x["family_actor"] in seen_fa or k >= quota[pool] - have:
          continue
        seen_fa.add(x["family_actor"])
        pick.append(x)
        k += 1
  body = dict(made=time.strftime("%F %T"), rule=__doc__.split("Episode:")[0].strip(), n=len(pick), clips=pick,
              counts={f"{p}:{t}": sum(1 for x in pick if x["pool"] == p and x["type"] == t) for p in ("unseen", "seen", "failtier") for t in ORDER},
              candidates={f"{p}:{t}": sum(1 for x in rows if x["pool"] == p and x["type"] == t) for p in ("unseen", "seen", "failtier") for t in ORDER},
              wall_s=round(time.time() - t0, 1))
  s = json.dumps(body, sort_keys=True)
  body["list_sha256"] = hashlib.sha256(json.dumps(pick, sort_keys=True).encode()).hexdigest()
  Path(out).write_text(json.dumps(body, indent=1, sort_keys=True))
  print(json.dumps({k: v for k, v in body["counts"].items() if v}, indent=0), "\nlist_sha256", body["list_sha256"][:16], "wall", body["wall_s"], len(s))


# ------------------------------------------------------------------ episode
def arm_constants():
  d = json.load(open(f"{B}/human_v2_armneutral/DATA_CARD.json"))
  return d["constants"]


_RANGE = {}


def arm_ranges():
  if not _RANGE:
    import mujoco as mj
    m = mj.MjModel.from_xml_path(os.environ.get("R1_XML", "/rl-agent/src/r1lab/assets/r1/r1.xml"))
    for n in ARM_MJ:
      j = m.joint(n).id
      _RANGE[n] = (float(m.jnt_range[j][0]), float(m.jnt_range[j][1])) if m.jnt_limited[j] else (-np.inf, np.inf)
  return _RANGE


def run_one(job):
  import mujoco
  import clip_eval as CE
  import sonic_g1 as S
  NB.patch(S)
  tag, ck, clip, version, delay_ms, jit, outdir = job
  name = clip["name"]
  f0, f1 = clip["window"]
  f0 = min(f0 + 2 * jit, f1 - int(WIN_MIN / DT))
  P, Q, J, jn = CE.load_r1_npz(f"{CLIPS}/{name}.npz")
  P, Q, J = P[f0:f1].copy(), Q[f0:f1].copy(), J[f0:f1].copy()
  T = len(P)
  t_start = time.time()
  if version == "an":                       # arm-neutral constants, then the MJCF hard range (as human_v2_armneutral)
    c, rg = arm_constants(), arm_ranges()
    for n in ARM_MJ:
      J[:, jn.index(n)] = np.clip(J[:, jn.index(n)] + c[n], *rg[n])
  JV = CE.fd(J)
  order = [jn.index(n) for n in S.R1_IL]
  mo = S.Motion(J[:, order], JV[:, order], P, Q, name)
  mo.space = "r1"
  dep = S.DeployR1Native("clip", mo, ck)
  sim = dep.sim
  m = sim.m
  dep.play, dep.play_ok, dep.t_play = True, True, 0.0
  qn = {n: J[0, i] for i, n in enumerate(jn)}
  dqn = {n: JV[0, i] for i, n in enumerate(jn)}
  for n in sim.rec_names:
    qn.setdefault(n, 0.0)
    dqn.setdefault(n, 0.0)
  sim.reset_to(P[0], Q[0], qn, CE.fd(P)[0], CE.ang_vel_body(Q)[0], dqn)
  rec = list(sim.rec_names)
  jref = np.array([[J[f, jn.index(n)] if n in jn else 0.0 for n in rec] for f in range(T)])
  dly = int(round(float(delay_ms) / 5.0))
  hist = [(-10 ** 9, jref[0].copy())]       # delay buffer filled with the first-frame targets
  dref = mujoco.MjData(m)
  bid = [m.body(b).id for b in CE.R1_BODIES]
  ee = [m.body(b).id for b in ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_roll_link", "right_wrist_roll_link")]
  pel = m.body("pelvis").id
  d = sim.d
  eg, el, err, pz, pzr = [], [], [], [], []
  fail_at, why, s_idx = None, "", 0
  for k in range(T - 1):
    dep.control_tick(k, k * S.CTRL_DT)
    qt = sim.target24()
    hist.append((s_idx, np.asarray(qt, float).copy()))
    hist = hist[-8:]
    for _ in range(S.DECIM):
      tgt = hist[0][1]
      for si, qq in hist:
        if si + dly <= s_idx:
          tgt = qq
      q, dq = d.qpos[sim.jq], d.qvel[sim.jv]
      tau = np.clip(sim.kp * (tgt - q) - sim.kd * dq, -sim.eff, sim.eff)
      d.qfrc_applied[:] = 0.0
      d.qfrc_applied[sim.jv] = tau
      d.xfrc_applied[:] = 0.0
      mujoco.mj_step(m, d)
      s_idx += 1
    f = k + 1
    dref.qpos[:] = 0
    dref.qpos[0:3], dref.qpos[3:7] = P[f], Q[f]
    for i, n in enumerate(rec):
      dref.qpos[sim.jq[i]] = jref[f, i]
    mujoco.mj_kinematics(m, dref)
    gt, pr = dref.xpos[bid].copy(), d.xpos[bid].copy()
    eg.append(np.linalg.norm(gt - pr, axis=1).mean() * 1000)
    el.append(np.linalg.norm((gt - gt[:1]) - (pr - pr[:1]), axis=1).mean() * 1000)
    err.append(d.qpos[sim.jq] - jref[f])
    pz.append(d.xpos[pel][2])
    pzr.append(dref.xpos[pel][2])
    qa, qb = dref.xquat[pel], d.xquat[pel]
    dq_ = S.qmul(S.qconj(qa), qb)
    ang = 2 * math.acos(min(1.0, abs(dq_[0])))
    if abs(dref.xpos[pel][2] - d.xpos[pel][2]) > 0.25:
      fail_at, why = k, "anchor_height"
    elif ang * ang > 1.0:
      fail_at, why = k, "anchor_ori"
    elif np.any(np.abs(dref.xpos[ee][:, 2] - d.xpos[ee][:, 2]) > 0.25):
      fail_at, why = k, "ee_height"
    if fail_at is not None:
      break
  arm_i = [rec.index(n) for n in ARM_MJ]
  E = np.array(err)
  sha = hashlib.sha256(f"{name}|{version}|{delay_ms}|{jit}".encode()).hexdigest()[:16]
  pdir = Path(outdir) / "npz" / tag
  pdir.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(pdir / f"{sha}.npz", eg=np.array(eg, np.float32), el=np.array(el, np.float32), err=E.astype(np.float16),
                      qref_arm=jref[1:len(E) + 1][:, arm_i].astype(np.float32), pz=np.array(pz, np.float32), pzr=np.array(pzr, np.float32))
  row = dict(key=sha, policy=tag, clip=name, type=clip["type"], pool=clip["pool"], version=version, delay_ms=delay_ms, jit=jit,
             frames=T, survived=len(eg), success=fail_at is None, fail_at=fail_at, why=why,
             mpjpe_g=float(np.mean(eg)) if eg else None, wall_s=round(time.time() - t_start, 2))
  if NB.BODY != "r1wb":
    row.update(NB.row_tag(S))
  return row


def main():
  if len(sys.argv) > 1 and sys.argv[1] == "pool":
    build_pool(sys.argv[2])
    return
  ap = argparse.ArgumentParser()
  ap.add_argument("cmd")
  ap.add_argument("--list", required=True)
  ap.add_argument("--policy", required=True)
  ap.add_argument("--version", default="an")
  ap.add_argument("--delay", type=float, default=0.0)
  ap.add_argument("--jit", type=int, default=0)
  ap.add_argument("--types", default="")
  ap.add_argument("--pools", default="")
  ap.add_argument("--limit", type=int, default=0)
  ap.add_argument("--out", required=True)
  ap.add_argument("--procs", type=int, default=24)
  a = ap.parse_args()
  tag, ck = a.policy.split("=", 1)
  L = json.load(open(a.list))
  clips = [c for c in L["clips"] if (not a.types or c["type"] in a.types.split(",")) and (not a.pools or c["pool"] in a.pools.split(","))]
  if a.limit:
    clips = clips[: a.limit]
  out = Path(a.out)
  out.mkdir(parents=True, exist_ok=True)
  NB.claim(out)
  rp = out / "rows.jsonl"
  done = set()
  if rp.exists():
    for l in rp.read_text().splitlines():
      if l.strip():
        r = json.loads(l)
        done.add((r["policy"], r["key"]))
  jobs = []
  for c in clips:
    sha = hashlib.sha256(f"{c['name']}|{a.version}|{a.delay}|{a.jit}".encode()).hexdigest()[:16]
    if (tag, sha) not in done:
      jobs.append((tag, ck, c, a.version, a.delay, a.jit, str(out)))
  import multiprocessing as mp
  t0 = time.time()
  with mp.get_context("spawn").Pool(min(a.procs, max(1, len(jobs)))) as pool, open(rp, "a") as fo:
    for i, r in enumerate(pool.imap_unordered(_safe, jobs), 1):
      fo.write(json.dumps(r) + "\n")
      fo.flush()
      if i % 50 == 0:
        print(f"{i}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
  print(f"done {len(jobs)} episodes {time.time() - t0:.0f}s list {L.get('list_sha256', '')[:16]}", flush=True)


def _safe(job):
  try:
    return run_one(job)
  except Exception as e:  # noqa: BLE001
    import traceback
    tag, _ck, clip, version, delay_ms, jit, _o = job
    return dict(key=hashlib.sha256(f"{clip['name']}|{version}|{delay_ms}|{jit}".encode()).hexdigest()[:16], policy=tag, clip=clip["name"],
                version=version, delay_ms=delay_ms, jit=jit, error=repr(e)[:300], trace=traceback.format_exc()[-1500:])


if __name__ == "__main__":
  main()
