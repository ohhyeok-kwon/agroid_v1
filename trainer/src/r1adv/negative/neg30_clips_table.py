"""Negative 30 motion-range table (coordinator 18:56): type x tracker from neg30_clips.py rows.
Paired on identical references: every tracker vs B@0 on the arm-neutral reference ('an'), per clip; B@0 on the original
reference ('orig') is the shift-cost column. Displayed continuous values use, per clip, the frames ALL listed trackers survive;
verdicts use the frames the pair (tracker, B@0) survives (>= 1 s; clip_eval's own mean stops at each tracker's failure and
would favour early failers; pairwise frames keep a cached tracker's verdict independent of a newly added one).
  completion   whole window without the SONIC im_eval failure; verdict on discordant clips (sign test, two-sided p < 0.05)
  mpjpe_l      body-relative MPJPE, mm (14 bodies)        mpjpe_g  world MPJPE, mm
  arm_rmse     arm joints, rad                            dyn_gain within-clip mean-removed slope of robot arm on reference
                                                                   arm (clips with arm range >= 0.5 rad)
  pel_mae      pelvis height error, mm                    hf       4-25 Hz joint-error RMS, deg (tremor)
  static gain  (mean arm with 'orig' - mean arm with 'an') projected on the reference difference: the P31 shift experiment
Verdict per metric: mean paired difference d vs thr = max(t(0.975, n-1) SD(d)/sqrt(n), floor, 2 x jitter SD), jitter SD =
SD over B@0 re-runs of the same clips started 0 / 2 / 4 frames later (type mean). lost = worse beyond thr, gained = better.
  python -I neg30_clips_table.py --run DIR --list L.json --ref B0 --tags R6s600,R8s400,R9s300 --out OUT.md [--json OUT.json]"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

DT = 0.02
R1_MJ = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint", "left_ankle_pitch_joint",
         "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
         "right_ankle_pitch_joint", "right_ankle_roll_joint", "waist_roll_joint", "waist_yaw_joint", "left_shoulder_pitch_joint",
         "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint", "right_shoulder_pitch_joint",
         "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint"]
ARM_I = list(range(14, 24))
TYPES = ["walk", "side", "turn", "transition", "run", "crouch_sit", "bend", "arm", "balance", "other", "support"]
KO = {"walk": "걷기", "side": "옆걸음", "turn": "돌기", "transition": "전환", "run": "달리기", "crouch_sit": "웅크리기·앉기", "bend": "굽히기",
      "arm": "팔 동작", "balance": "균형(한 발)", "other": "기타", "support": "지지 필요"}
FLOOR = dict(mpjpe_l=3.0, mpjpe_g=10.0, arm_rmse=0.01, pel_mae=3.0, hf=0.05, dyn_gain=0.05)
LOWER_BETTER = dict(mpjpe_l=True, mpjpe_g=True, arm_rmse=True, pel_mae=True, hf=True, dyn_gain=False)
TCRIT = {1: 12.71, 2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57, 6: 2.45, 7: 2.36, 8: 2.31, 9: 2.26, 10: 2.23, 15: 2.13, 20: 2.09, 30: 2.04}


def tcrit(df):
  ks = sorted(TCRIT)
  for k in ks:
    if df <= k:
      return TCRIT[k]
  return 1.96


def binom_two_sided(k, n):
  if n == 0:
    return 1.0
  from math import comb
  p = sum(comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
  return min(1.0, 2 * p)


class Runs:
  def __init__(self, run):
    self.run = Path(run)
    self.rows = {}
    for l in (self.run / "rows.jsonl").read_text().splitlines():
      if l.strip():
        r = json.loads(l)
        if "error" not in r:
          self.rows[(r["policy"], r["version"], float(r["delay_ms"]), int(r["jit"]), r["clip"])] = r
    self._a = {}

  def get(self, pol, ver, d, jit, clip):
    return self.rows.get((pol, ver, float(d), int(jit), clip))

  def arr(self, r):
    k = (r["policy"], r["key"])
    if k not in self._a:
      z = np.load(self.run / "npz" / r["policy"] / f"{r['key']}.npz")
      self._a[k] = {x: z[x].astype(np.float64) for x in z.files}
    return self._a[k]


def metrics(A, n):
  err = A["err"][:n]
  out = dict(mpjpe_l=float(A["el"][:n].mean()), mpjpe_g=float(A["eg"][:n].mean()),
             arm_rmse=float(np.sqrt((err[:, ARM_I] ** 2).mean())), pel_mae=float(np.abs(A["pz"][:n] - A["pzr"][:n]).mean() * 1000))
  qr = A["qref_arm"][:n]
  if (qr.max(0) - qr.min(0)).max() >= 0.5:
    x = qr - qr.mean(0)
    y = (qr + err[:, ARM_I]) - (qr + err[:, ARM_I]).mean(0)
    out["dyn_gain"] = float((x * y).sum() / max((x * x).sum(), 1e-9))
  if n >= 100:
    e = err - err.mean(0)
    X = np.fft.rfft(e, axis=0)
    X[np.fft.rfftfreq(n, DT) < 4.0] = 0
    out["hf"] = float(np.degrees(np.sqrt((np.fft.irfft(X, n=n, axis=0) ** 2).mean())))
  out["arm_mean"] = (qr + err[:, ARM_I]).mean(0)
  out["ref_arm_mean"] = qr.mean(0)
  return out


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--run", required=True)
  ap.add_argument("--list", required=True)
  ap.add_argument("--ref", default="B0")
  ap.add_argument("--tags", required=True)
  ap.add_argument("--delay", type=float, default=0.0)
  ap.add_argument("--out", required=True)
  ap.add_argument("--json", default="")
  a = ap.parse_args()
  import sys
  sys.path.insert(0, str(Path(__file__).resolve().parent))
  import neg30_body as NB
  NB.require(a.run)                       # NEG30_BODY: the run dir must be of this body
  R = Runs(a.run)
  L = json.load(open(a.list))
  tags = [t for t in a.tags.split(",") if t]
  cols = [(a.ref, "an")] + [(t, "an") for t in tags]
  res = {}
  for delay in sorted({a.delay, 15.0}):
    groups = defaultdict(list)
    for c in L["clips"]:
      groups[(c["pool"], c["type"])].append(c)
    out = {}
    for g, C in groups.items():
      per = defaultdict(dict)          # col -> clip -> metrics
      comp = defaultdict(dict)
      for c in C:
        rs = {col: R.get(col[0], col[1], delay, 0, c["name"]) for col in cols}
        ro = R.get(a.ref, "orig", delay, 0, c["name"])
        for col, r in rs.items():
          if r is not None:
            comp[col][c["name"]] = bool(r["success"])
        if ro is not None:
          comp[(a.ref, "orig")][c["name"]] = bool(ro["success"])
        have = {col: r for col, r in rs.items() if r is not None}
        if len(have) < 2 or (a.ref, "an") not in have:
          continue
        n = min(r["survived"] for r in have.values())
        if n >= 50:
          for col, r in have.items():
            per[col][c["name"]] = metrics(R.arr(r), n)
        if ro is not None and rs.get((a.ref, "an")) is not None:          # shift cost and B@0 static gain: common frames of the two
          n2 = min(ro["survived"], rs[(a.ref, "an")]["survived"])
          if n2 >= 50:
            mo, ma = metrics(R.arr(ro), n2), metrics(R.arr(rs[(a.ref, "an")]), n2)
            per[(a.ref, "orig")][c["name"]] = mo
            per[("__static__", a.ref)][c["name"]] = (mo, ma)
        for t in tags:                                                    # tracker static gain (orig run only where it exists)
          rto, rta = R.get(t, "orig", delay, 0, c["name"]), R.get(t, "an", delay, 0, c["name"])
          if rto is not None and rta is not None:
            n3 = min(rto["survived"], rta["survived"])
            if n3 >= 50:
              per[("__static__", t)][c["name"]] = (metrics(R.arr(rto), n3), metrics(R.arr(rta), n3))
      # jitter SD of the type mean (B@0, an, re-runs started 0 / 2 / 4 frames later), per metric
      jit = {}
      for k in (0, 1, 2):
        vals = defaultdict(list)
        for c in C:
          r = R.get(a.ref, "an", delay, k, c["name"])
          if r is None:
            continue
          n = r["survived"]
          if n >= 50:
            m = metrics(R.arr(r), n)
            for key in FLOOR:
              if key in m:
                vals[key].append(m[key])
          vals["completion"].append(float(r["success"]))
        jit[k] = {key: float(np.mean(v)) for key, v in vals.items() if v}
      jsd = {}
      if len([k for k in jit if jit[k]]) >= 2:
        for key in set().union(*[set(v) for v in jit.values()]):
          xs = [jit[k][key] for k in jit if key in jit[k]]
          if len(xs) >= 2:
            jsd[key] = float(np.std(xs, ddof=1))
      G = dict(n=len(C), cols={}, verdict={}, jit_sd=jsd)
      for col in cols + [(a.ref, "orig")]:
        cc = comp.get(col, {})
        mm = per.get(col, {})
        G["cols"]["/".join(col)] = dict(n=len(cc), completion=(float(np.mean(list(cc.values()))) if cc else None),
                                         **{key: (float(np.mean([m[key] for m in mm.values() if key in m])) if any(key in m for m in mm.values()) else None)
                                            for key in FLOOR}, n_metric=len(mm))
      base = per.get((a.ref, "an"), {})
      for t in tags:
        V = {}
        cb, ct = comp.get((a.ref, "an"), {}), comp.get((t, "an"), {})
        both = [k for k in cb if k in ct]
        lost = sum(1 for k in both if cb[k] and not ct[k])
        gained = sum(1 for k in both if ct[k] and not cb[k])
        p = binom_two_sided(min(lost, gained), lost + gained)
        V["completion"] = dict(lost=lost, gained=gained, n=len(both), p=p, call=("lost" if lost > gained and p < 0.05 else "gained" if gained > lost and p < 0.05 else "same"))
        # verdict frames: pairwise (this tracker vs B@0) per clip, so a new checkpoint never changes a cached tracker's verdict
        pair = {}
        for c in C:
          rb, rt = R.get(a.ref, "an", delay, 0, c["name"]), R.get(t, "an", delay, 0, c["name"])
          if rb is None or rt is None:
            continue
          n = min(rb["survived"], rt["survived"])
          if n >= 50:
            pair[c["name"]] = (metrics(R.arr(rt), n), metrics(R.arr(rb), n))
        for key in FLOOR:
          d = [mt_[key] - mb_[key] for mt_, mb_ in pair.values() if key in mt_ and key in mb_]
          if len(d) < 3:
            continue
          d = np.array(d)
          thr = max(tcrit(len(d) - 1) * d.std(ddof=1) / math.sqrt(len(d)), FLOOR[key], 2 * jsd.get(key, 0.0))
          worse = d.mean() > thr if LOWER_BETTER[key] else d.mean() < -thr
          better = d.mean() < -thr if LOWER_BETTER[key] else d.mean() > thr
          V[key] = dict(d=float(d.mean()), thr=float(thr), n=len(d), call="lost" if worse else "gained" if better else "same")
        G["verdict"][t] = V
      # static gains
      sg = {}
      for who in [a.ref] + tags:
        pairs = per.get(("__static__", who), {})
        num = den = 0.0
        for mo, ma in pairs.values():
          u = mo["ref_arm_mean"] - ma["ref_arm_mean"]
          off = mo["arm_mean"] - ma["arm_mean"]
          num += float(off @ u)
          den += float(u @ u)
        if den > 0:
          sg[who] = dict(gain=num / den, n=len(pairs))
      G["static_gain"] = sg
      out["|".join(g)] = G
    res[str(delay)] = out
  if a.json:
    Path(a.json).write_text(json.dumps(res, indent=1, default=float))
  # ---- markdown
  sym = {"lost": "↓", "gained": "↑", "same": "="}
  def keys_for(ty):
    return ["completion", "mpjpe_l"] + (["arm_rmse"] if ty == "arm" else []) + (["pel_mae"] if ty in ("crouch_sit", "bend") else [])

  def cell(G, col, t=None, ty=""):
    c = G["cols"].get(col)
    if not c or c["completion"] is None:
      return "—"
    s = f"{100 * c['completion']:.0f}% · {c['mpjpe_l']:.0f}" if c["mpjpe_l"] is not None else f"{100 * c['completion']:.0f}% · —"
    if t:
      V = G["verdict"].get(t, {})
      calls = [V[k]["call"] for k in keys_for(ty) if k in V]
      v = "lost" if "lost" in calls else ("gained" if "gained" in calls else "same")
      s += f" {sym[v]}"
    return s
  Lm = [f"# 동작 폭 합격표 — 클립 목록 {L.get('list_sha256', '')[:16]} (사람 R1 클립, T1 고리, 칸 = 완주 % · mpjpe_l mm · 판정 대 B@0(팔 중립 참조))", "",
        "- 묶음: '안 본 것' = 통과 등급이지만 human_v2_armneutral 폴더에 없는 클립(R5–R10 미세조정이 안 본 것); '안 본 것(거른 등급)' = 어느 sonic_pkl 에도 없음; "
        "'보류' = list_heldout 의 클립(R5–R10 은 remove_motion_keys 로 학습에서 뺐다; 20:4x 고침 — 앞서 '본 것'이라 적은 것은 틀림). B@0 의 사전학습 노출은 알 수 없다.",
        "- 판정 ↓ 잃음 / ↑ 얻음 / = 같음: 완주(어긋난 클립 부호 검정 p < 0.05)·mpjpe_l(모든 꼴)·팔 RMSE(팔 동작)·골반 높이 오차(웅크리기·굽히기) 가운데 하나라도 문턱(짝 t, 바닥값, 2 × 흔들기 SD 중 큰 값)을 넘으면.", ""]
  for delay in sorted(res, key=float):
    out = res[delay]
    Lm += [f"## 행동 지연 {float(delay):g} ms" + (" (주 판정)" if float(delay) == a.delay else " (곁 값)"), "",
           "| 묶음 | 꼴 | n | B@0 원본 참조 | B@0 팔 중립 | " + " | ".join(tags) + " |", "|---|---|---|---|---|" + "---|" * len(tags)]
    for pool in ("unseen", "failtier", "seen"):
      for ty in TYPES:
        G = out.get(f"{pool}|{ty}")
        if not G:
          continue
        Lm.append(f"| {dict(unseen='안 본 것', failtier='안 본 것(거른 등급)', seen='보류(학습 제외)')[pool]} | {KO[ty]} | {G['n']} | "
                  f"{cell(G, a.ref + '/orig')} | {cell(G, a.ref + '/an')} | " + " | ".join(cell(G, f'{t}/an', t, ty) for t in tags) + " |")
    Lm.append("")
  # details (main delay): arm / pelvis / tremor per type and tracker, static arm gains
  out0 = res[str(a.delay)]
  Lm += [f"## 자세한 값 (지연 {a.delay:g} ms, 같은 클립·모든 추종기가 버틴 프레임; 칸 = 팔 RMSE rad · 팔 동적 이득 · 골반 높이 오차 mm · 떨림 4–25 Hz °)", "",
         "| 묶음 | 꼴 | B@0 원본 | B@0 팔 중립 | " + " | ".join(tags) + " |", "|---|---|---|---|" + "---|" * len(tags)]
  f3 = lambda x, nd: "—" if x is None else f"{x:.{nd}f}"  # noqa: E731
  for pool in ("unseen", "failtier", "seen"):
    for ty in TYPES:
      G = out0.get(f"{pool}|{ty}")
      if not G:
        continue
      cells = []
      for col in [a.ref + "/orig", a.ref + "/an"] + [f"{t}/an" for t in tags]:
        c = G["cols"].get(col) or {}
        cells.append(f"{f3(c.get('arm_rmse'), 3)} · {f3(c.get('dyn_gain'), 2)} · {f3(c.get('pel_mae'), 0)} · {f3(c.get('hf'), 2)}")
      Lm.append(f"| {dict(unseen='안 본 것', failtier='거른 등급', seen='보류')[pool]} | {KO[ty]} | " + " | ".join(cells) + " |")
  sg = defaultdict(lambda: [0.0, 0.0, 0])
  for g, G in out0.items():
    for who, v in G.get("static_gain", {}).items():
      pass
  Lm += ["", "- 팔 정적 이득(원본 참조 − 팔 중립 참조의 팔 평균 차를 참조 차에 사영, P31 밀기 실험과 같은 꼴; 꼴별): " + "; ".join(
    f"{KO[g.split('|')[1]]}({dict(unseen='안 본', failtier='거른', seen='보류')[g.split('|')[0]]}) " + ", ".join(f"{w} {v['gain']:.2f} (n {v['n']})" for w, v in G.get("static_gain", {}).items())
    for g, G in sorted(out0.items()) if G.get("static_gain")), ""]
  Lm += posture_line(R, L, a)
  Path(a.out).write_text("\n".join(Lm) + "\n")
  print("\n".join(Lm))


POSTURE_TYPES = {"crouch_sit": ("unseen",), "bend": ("unseen",), "arm": ("unseen",), "balance": ("unseen", "failtier")}
POSTURE_KEYS = {"crouch_sit": ("completion", "mpjpe_l", "pel_mae"), "bend": ("completion", "mpjpe_l", "pel_mae"),
                "arm": ("completion", "mpjpe_l", "arm_rmse"), "balance": ("completion", "mpjpe_l")}


def posture_line(R, L, a, old="R8s400", new="R9s300"):
  """Pre-registered (coordinator 18:52, written 18:51-18:52 before the values): R9 vs R8@400 on the non-walking types of the
  unseen pool, arm-neutral reference, delay 0, paired per clip. Noise = SD over the three runs started 0 / 2 / 4 frames later
  of the mean paired difference; threshold = max(2 SD, floor; completion floor = 1 clip). Compared value = the main run's
  (0 frames) mean paired difference; the three-run mean is shown beside it. Any type worse beyond the threshold ->
  'revise the posture terms', none -> 'keep'."""
  out = ["## 바른자세 항 줄 (등록: 조율자 18:52, 값 보기 전 18:51–18:52 에 적음) — " + f"{new} − {old}, 안 본 것, 지연 0, 팔 중립 참조", "",
         "| 꼴 | n | 잣대 | " + old + " | " + new + " | 차이 (주 판) | 세 판 평균 | 잡음 SD | 문턱 | 판정 |", "|---|---|---|---|---|---|---|---|---|---|"]
  bad = []
  for ty, pools in POSTURE_TYPES.items():
    C = [c for c in L["clips"] if c["type"] == ty and c["pool"] in pools]
    per_k = {}
    vals0 = {}
    for k in (0, 1, 2):
      D = defaultdict(list)
      for c in C:
        r8, r9 = R.get(old, "an", 0.0, k, c["name"]), R.get(new, "an", 0.0, k, c["name"])
        if r8 is None or r9 is None:
          continue
        D["completion"].append(float(r9["success"]) - float(r8["success"]))
        if k == 0:
          vals0.setdefault("completion", ([], []))
          vals0["completion"][0].append(float(r8["success"]))
          vals0["completion"][1].append(float(r9["success"]))
        n = min(r8["survived"], r9["survived"])
        if n >= 50:
          m8, m9 = metrics(R.arr(r8), n), metrics(R.arr(r9), n)
          for key in POSTURE_KEYS[ty]:
            if key != "completion" and key in m8 and key in m9:
              D[key].append(m9[key] - m8[key])
              if k == 0:
                vals0.setdefault(key, ([], []))
                vals0[key][0].append(m8[key])
                vals0[key][1].append(m9[key])
      per_k[k] = {key: (float(np.mean(v)), len(v)) for key, v in D.items() if v}
    for key in POSTURE_KEYS[ty]:
      if key not in per_k.get(0, {}):
        continue
      d0, n0 = per_k[0][key]
      ds = [per_k[k][key][0] for k in per_k if key in per_k[k]]
      sd = float(np.std(ds, ddof=1)) if len(ds) >= 2 else float("nan")
      floor = (1.0 / n0) if key == "completion" else FLOOR[key]
      thr = max(2 * sd if math.isfinite(sd) else 0.0, floor)
      worse = (d0 < -thr) if key == "completion" else (d0 > thr)
      if worse:
        bad.append(f"{KO[ty]}:{key}")
      a8, a9 = vals0[key]
      fmt = (lambda x: f"{100 * x:.0f}%") if key == "completion" else (lambda x: f"{x:.3f}" if key == "arm_rmse" else f"{x:.1f}")
      dfmt = (lambda x: f"{100 * x:+.1f}%p") if key == "completion" else (lambda x: f"{x:+.3f}" if key == "arm_rmse" else f"{x:+.1f}")
      out.append(f"| {KO[ty]} | {n0} | {key} | {fmt(float(np.mean(a8)))} | {fmt(float(np.mean(a9)))} | {dfmt(d0)} | {dfmt(float(np.mean(ds)))} | "
                 f"{(dfmt(sd).lstrip('+') if math.isfinite(sd) else '—')} (판 {len(ds)}) | {dfmt(thr).lstrip('+')} | {'나빠짐' if worse else ('나아짐' if ((d0 > thr) if key == 'completion' else (d0 < -thr)) else '같음')} |")
  out += ["", "- 판정: " + ("**바른자세 항 다시 손봄** — " + ", ".join(bad) if bad else "**그대로** — 문턱을 넘어 나빠진 꼴 없음"), ""]
  return out


if __name__ == "__main__":
  main()
