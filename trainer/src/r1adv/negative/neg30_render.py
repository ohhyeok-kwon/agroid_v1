"""Negative 30: posture stills ('바른자세' card, eye check) rendered on the server CPU (MUJOCO_GL=osmesa) from neg30_pass.py
npz records - nothing is simulated here, recorded states are replayed. Tiles (front / side): standing on the planner idle
reference (stand_push, t = 15 s), standing on the symmetric static reference (stand_static, t = 8 s), forward 0.3 (moves,
t = 6 s), forward 0.5 (walk_stop, t = 10 s). One image per checkpoint with the manufacturer row above it, plus one
overview with every policy.

  MUJOCO_GL=osmesa python -I neg30_render.py OUT_DIR SPECS_GLOB RUN_DIR CELL POLICY [POLICY ...]
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("R1_BODY", "r1wb")
for _p in ("/rl-agent/src", "/rl-agent/.agent/tmp/t1_sonic/work"):
  if _p not in sys.path:
    sys.path.insert(0, _p)

TILES = (("stand_push", 15.0, "stand idle"), ("stand_static", 8.0, "stand sym-ref"), ("moves", 6.0, "walk 0.3"),
         ("walk_stop", 10.0, "walk 0.5"))
W_, H_ = 300, 380


def yaw_of(q):
  return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def main():
  import mujoco
  import sonic_g1 as S
  from PIL import Image, ImageDraw
  out, spec_glob, run, cell, pols = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5:]
  out.mkdir(parents=True, exist_ok=True)
  specs = {}
  for fn in glob.glob(spec_glob):
    for l in open(fn):
      if l.strip():
        s = json.loads(l)
        specs[s["spec_sha256"]] = s
  rows = [json.loads(l) for l in (run / "rows.jsonl").read_text().splitlines() if l.strip()]
  sim = S.SimR1("r1")
  m, d = sim.m, mujoco.MjData(sim.m)
  ren = mujoco.Renderer(m, H_, W_)

  def tile(pol, fam, t_s, view):
    cands = [r for r in rows if r.get("policy") == pol and r.get("family") == fam and "error" not in r
             and (r.get("cell") == cell or (fam == "stand_static" and r.get("cell", "").startswith("ol_L01_d00")))]
    cands.sort(key=lambda r: r.get("i", 0))
    if not cands:
      return None
    r = cands[0]
    z = np.load(run / "npz" / pol / f"{r['spec_sha256'][:16]}.npz")
    k = min(int(round(t_s / 0.02)) - 1, len(z["pos"]) - 1)
    d.qpos[:] = 0
    d.qpos[0:3], d.qpos[3:7] = z["pos"][k], z["quat"][k]
    d.qpos[sim.jq] = z["q"][k]
    mujoco.mj_forward(m, d)
    psi = math.degrees(yaw_of(z["quat"][k]))
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [float(z["pos"][k][0]), float(z["pos"][k][1]), 0.55]
    cam.distance, cam.elevation = 2.3, -8.0
    cam.azimuth = psi + 180.0 if view == "front" else psi - 90.0
    ren.update_scene(d, cam)
    return Image.fromarray(ren.render())

  def strip(pol):
    ims = []
    for fam, t_s, lab in TILES:
      for view in ("front", "side"):
        if fam == "stand_static" and view == "side":
          continue
        im = tile(pol, fam, t_s, view) if not (pol == "mf" and fam == "stand_static") else None
        if im is None:
          im = Image.new("RGB", (W_, H_), (40, 40, 40))
        dr = ImageDraw.Draw(im)
        dr.text((6, 6), f"{pol} | {lab} | {view}", fill=(255, 255, 0))
        ims.append(im)
    row = Image.new("RGB", (W_ * len(ims), H_))
    for i, im in enumerate(ims):
      row.paste(im, (W_ * i, 0))
    return row

  strips = {p: strip(p) for p in pols}
  for p in pols:
    if p == "mf":
      continue
    a, b = strips.get("mf"), strips[p]
    img = Image.new("RGB", (b.width, b.height * (2 if a is not None else 1)))
    if a is not None:
      img.paste(a, (0, 0))
    img.paste(b, (0, b.height if a is not None else 0))
    img.save(out / f"posture_{cell}_{p}.png")
  allimg = Image.new("RGB", (strips[pols[0]].width, H_ * len(pols)))
  for i, p in enumerate(pols):
    allimg.paste(strips[p], (0, H_ * i))
  allimg.save(out / f"posture_{cell}_all.png")
  print("rendered", len(pols), "policies to", out)


if __name__ == "__main__":
  main()
