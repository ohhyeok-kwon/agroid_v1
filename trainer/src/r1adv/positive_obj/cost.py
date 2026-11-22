"""Simulation cost of objects in an RL scene (rules.md 4 '비용', positive 18). Free-base R1 (training body, self-collision)
standing on a plane under the deploy PD (default pose), with: base (no table, no object), one (table + box on it), three
(table + box, cylinder, capsule on it), park (table + the three bodies, box on the table, two parked on the floor).
Control-step time (4 physics steps, CUDA graph). rules_amend A3: the GPU could not be held exclusively (another agent's
evaluation started on it), so each scene is timed in an A-B-A-B-A block against the base scene of the same cone and size
and r_phys = median(scene) / median(base) of the block. rules_amend A4: the PD-only free robot starts to fall about
1.5 s after a reset (C check: pelvis travel 0.14 m and no robot-table contact up to 1.3 s, on the table / floor by 1.9 s),
so each run warms up 10 and measures 50 control steps (0.2-1.2 s after the reset, robot upright). Table near edge 0.6 m.

  python -m r1adv.positive_obj.cost <out.json> [N ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import warp as wp

from r1adv.positive_obj import ladder as L
from r1adv.positive_obj import scene as S

SCENES = {"base": ((), False), "one": (("box",), True), "three": (S.SHAPES, True), "park": (S.SHAPES, True)}
ON_TABLE = {"one": {"box": (0.85, 0.0)}, "three": {"box": (0.85, -0.3), "cylinder": (0.85, 0.0), "capsule": (0.85, 0.3)},
            "park": {"box": (0.85, 0.0)}}
TABLE_POSE = (0.60, 0.0, 0.70)
WARM, MEAS = 10, 50                                   # 0.2 s + 1.0 s after each reset: the robot is still upright


class Bench:
  """One scene's simulation with its initial state, timed on demand."""

  def __init__(self, scene: str, cone: str, condim: int, n: int, device: str = "cuda:0"):
    from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
    shapes, table = SCENES[scene]
    m = L.build_model(cone, condim, free_robot=True, shapes=shapes, table=table)
    idx = S.SceneIndex(m)
    q0 = m.key_qpos[0].copy()
    for shape, qa in idx.obj_qadr.items():
      if shape in ON_TABLE.get(scene, {}):
        x, y = ON_TABLE[scene][shape]
        q0[qa:qa + 3] = (x, y, TABLE_POSE[2] + S.NOMINAL_DIMS[2] / 2 + 0.002)
      else:
        px, py = S.PARK_POS[shape]
        q0[qa:qa + 3] = (px, py, S.NOMINAL_DIMS[2] / 2 + 0.002)
      q0[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
    m.qpos0[:] = q0                                   # put_data checks contacts at qpos0
    if table and m.nmocap:
      m.body_pos[m.body("table/mocap_base").id] = TABLE_POSE
    cfg = SimulationCfg(nconmax=64, njmax=512, mujoco=MujocoCfg(cone=cone, **L.SIM_OPT))
    t0 = time.time()
    self.sim = Simulation(num_envs=n, cfg=cfg, model=m, device=device)
    self.build_s = time.time() - t0
    D = self.sim.wp_data
    self.qpos, self.qvel, self.ctrl = wp.to_torch(D.qpos), wp.to_torch(D.qvel), wp.to_torch(D.ctrl)
    self.q0 = torch.as_tensor(q0, device=self.qpos.device, dtype=self.qpos.dtype)
    self.c0 = torch.as_tensor(m.key_ctrl[0], device=self.ctrl.device, dtype=self.ctrl.dtype)
    self.mocap = wp.to_torch(D.mocap_pos) if (table and m.nmocap) else None
    self.table_mocap = idx.table_mocap
    self.n, self.scene = n, scene
    self.nacon_max, self.nefc_max = 0, 0

  def time_once(self) -> float:
    self.qpos.copy_(self.q0.expand_as(self.qpos))
    self.qvel.zero_()
    self.ctrl.copy_(self.c0.expand_as(self.ctrl))
    if self.mocap is not None:
      self.mocap[:, self.table_mocap] = torch.as_tensor(TABLE_POSE, device=self.mocap.device, dtype=self.mocap.dtype)
    self.sim.forward()
    for _ in range(WARM * L.DECIM):
      self.sim.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(MEAS * L.DECIM):
      self.sim.step()
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / MEAS
    D = self.sim.wp_data
    self.nacon_max = max(self.nacon_max, int(wp.to_torch(D.nacon)[0].item()))
    self.nefc_max = max(self.nefc_max, int(wp.to_torch(D.nefc).max().item()))
    return 1000.0 * dt


def others_on_gpu() -> str:
  import os
  import subprocess
  try:
    me = os.getpid()
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=10).stdout
    vis = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    uu = subprocess.run(["nvidia-smi", "-i", vis, "--query-gpu=uuid", "--format=csv,noheader"], capture_output=True,
                        text=True, timeout=10).stdout.strip() if vis else ""
    pids = [ln.split(",")[0].strip() for ln in out.strip().splitlines() if uu and uu in ln]
    return ",".join(p for p in pids if p and int(p) != me)
  except Exception as e:      # noqa: BLE001
    return f"unknown ({e})"


def main(argv: list[str]) -> None:
  out_path = argv[0]
  ns = [int(a) for a in argv[1:]] or [4096, 8192]
  rows = []
  for n in ns:
    for cone in ("pyramidal", "elliptic"):
      base = Bench("base", cone, 3, n)
      for condim in (3, 4):
        for scene in ("one", "three", "park"):
          sc = Bench(scene, cone, condim, n)
          tb, ts = [base.time_once()], []
          for _ in range(2):
            ts.append(sc.time_once())
            tb.append(base.time_once())
          row = dict(scene=scene, cone=cone, condim=condim, n=n, t_base_ms=tb, t_scene_ms=ts,
                     r_phys=float(np.median(ts) / np.median(tb)), build_s=sc.build_s, nacon_max=sc.nacon_max,
                     nefc_max=sc.nefc_max, base_nacon_max=base.nacon_max, overflow=bool(sc.nacon_max >= 64 * n or sc.nefc_max >= 512),
                     others_on_gpu=others_on_gpu())
          rows.append(row)
          print(row, flush=True)
          del sc
          torch.cuda.empty_cache()
      del base
      torch.cuda.empty_cache()
  res = {"rows": rows, "r_phys": {}, "R4096": {}, "R8192": {}}
  for r in rows:
    key = f"{r['cone']}_{r['condim']}"
    R = (41.6 + r["r_phys"] * 8.4e-3 * r["n"]) / (41.6 + 8.4e-3 * r["n"])
    res["r_phys"].setdefault(str(r["n"]), {}).setdefault(key, {})[r["scene"]] = r["r_phys"]
    res.setdefault(f"R{r['n']}", {}).setdefault(key, {})[r["scene"]] = R
  with open(out_path, "w") as f:
    json.dump(res, f, indent=1)
  print(json.dumps({k: res[k] for k in ("r_phys", "R4096", "R8192")}, indent=1))


if __name__ == "__main__":
  main(sys.argv[1:])
