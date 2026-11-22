"""Negative 30 (coordinator 12:06): side-by-side posture stills from recorded runs (no simulation; recorded states are replayed
and rendered on the server CPU, MUJOCO_GL=osmesa). One PNG per scene and view; columns = sources (manufacturer first).
Scene spec (JSON list): [{"scene": name, "t": [times s], "cols": [{"label": L, "npz": path, "kind": "neg"|"t1", "robot": i}]}]
  kind 'neg': neg28 / neg30 runner npz (pos (T,3), quat (T,4), q (T,24) entity order)
  kind 't1':  T1 sonic_g1 cmd npz (pos (T,5,3), quat (T,5,4), q (T,5,24)); 'robot' selects the column of the 5 robots
  python -I neg30_render2.py SCENES.json OUT_DIR"""
from __future__ import annotations

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
W_, H_ = 360, 460


def yaw_of(q):
  return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def main():
  import mujoco
  import sonic_g1 as S
  from PIL import Image, ImageDraw
  scenes = json.load(open(sys.argv[1]))
  out = Path(sys.argv[2])
  out.mkdir(parents=True, exist_ok=True)
  sim = S.SimR1("r1")
  m, d = sim.m, mujoco.MjData(sim.m)
  ren = mujoco.Renderer(m, H_, W_)
  for sc in scenes:
    for view in ("front", "side"):
      rows = []
      for t_s in sc["t"]:
        tiles = []
        for col in sc["cols"]:
          z = np.load(col["npz"], allow_pickle=True)
          if col["kind"] == "t1":
            r = int(col.get("robot", 0))
            pos, quat, q = z["pos"][:, r], z["quat"][:, r], z["q"][:, r]
          else:
            pos, quat, q = z["pos"], z["quat"], z["q"]
          k = min(int(round(t_s / 0.02)) - 1, len(pos) - 1)
          d.qpos[:] = 0
          d.qpos[0:3], d.qpos[3:7] = pos[k], quat[k]
          d.qpos[sim.jq] = q[k]
          mujoco.mj_forward(m, d)
          psi = math.degrees(yaw_of(quat[k]))
          cam = mujoco.MjvCamera()
          cam.type = mujoco.mjtCamera.mjCAMERA_FREE
          cam.lookat[:] = [float(pos[k][0]), float(pos[k][1]), 0.55]
          cam.distance, cam.elevation = 2.2, -6.0
          cam.azimuth = psi + 180.0 if view == "front" else psi - 90.0
          ren.update_scene(d, cam)
          im = Image.fromarray(ren.render())
          ImageDraw.Draw(im).text((6, 6), f"{col['label']} | {sc['scene']} | {view} | t={t_s:.1f}s", fill=(255, 255, 0))
          tiles.append(im)
        row = Image.new("RGB", (W_ * len(tiles), H_))
        for i, im in enumerate(tiles):
          row.paste(im, (W_ * i, 0))
        rows.append(row)
      img = Image.new("RGB", (rows[0].width, H_ * len(rows)))
      for i, r in enumerate(rows):
        img.paste(r, (0, H_ * i))
      img.save(out / f"{sc['scene']}_{view}.png")
      print("saved", out / f"{sc['scene']}_{view}.png")


if __name__ == "__main__":
  main()
