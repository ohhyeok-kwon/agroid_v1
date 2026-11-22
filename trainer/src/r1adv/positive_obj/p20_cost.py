"""Simulation cost of the P20 arm bodies with objects (rules.md 5.3 + rules_amend A1-3/A1-4), positive 18 cost.py method:
free-base robot standing on a plane under the deploy PD (default pose), control-step time (4 physics steps, CUDA graph),
A-B-A-B-A blocks against the base scene (training O6 body, no table / object, nconmax 64, njmax 512), each run warms up 10
and measures 50 control steps (0.2-1.2 s after the reset, robot upright). Scene 'park': table + the three object bodies of
p20_scene (box, mesh cylinder, mesh capsule), the mesh cylinder on the table and the other two parked on the floor, the arm's
ladder buffers (p20_ladder_mjw.buffers).

  python -m r1adv.positive_obj.p20_cost <out.json> [arm ...]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch
import warp as wp

from r1adv.positive_obj import cost as C18
from r1adv.positive_obj import p20_bodies as B
from r1adv.positive_obj import p20_ladder as L
from r1adv.positive_obj import p20_ladder_mjw as W
from r1adv.positive_obj import p20_scene as PS
from r1adv.positive_obj import scene as S

TABLE_POSE = C18.TABLE_POSE
WARM, MEAS = C18.WARM, C18.MEAS


def build(arm: str | None, cone: str, condim: int):
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sim import MujocoCfg
  from mjlab.terrains import TerrainEntityCfg
  from r1lab import robot as RB
  if arm is None:
    ents = {"robot": RB.get_r1_cfg(command_delay_max=0, self_collision=True, hand="o6")}
  else:
    a = L.ARMS[arm]
    ents = {"robot": B.robot_cfg(a["body"], fixed=False, force_scale=a["scale"], pads=bool(a.get("pads"))),
            "table": S.table_entity_cfg()}
    for shape in PS.SHAPES:
      ents[PS.ENTITY_OF[shape]] = PS.object_entity_cfg(shape, condim)
  m = Scene(SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type="plane"), entities=ents), device="cpu").compile()
  MujocoCfg(cone=cone, **L.SIM_OPT).apply(m)
  return m


class Bench:
  def __init__(self, arm: str | None, cone: str, condim: int, n: int, device: str = "cuda:0"):
    from mjlab.sim import MujocoCfg, Simulation, SimulationCfg
    m = build(arm, cone, condim)
    q0 = m.key_qpos[0].copy()
    if arm is not None:
      idx = PS.Index(m)
      for shape, qa in idx.obj_qadr.items():
        if shape == "cylinder":
          q0[qa:qa + 3] = (0.85, 0.0, TABLE_POSE[2] + PS.NOMINAL[shape][2] / 2 + PS.MESH_OFFSET + 0.002)
        else:
          px, py = PS.PARK_POS[shape]
          q0[qa:qa + 3] = (px, py, PS.NOMINAL[shape][2] / 2 + 0.002)
        q0[qa + 3:qa + 7] = (1.0, 0.0, 0.0, 0.0)
      m.body_pos[m.body("table/mocap_base").id] = TABLE_POSE
      self.table_mocap = idx.table_mocap
      nconmax, njmax = W.buffers(arm)
    else:
      self.table_mocap = -1
      nconmax, njmax = 64, 512
    m.qpos0[:] = q0
    cfg = SimulationCfg(nconmax=nconmax, njmax=njmax, mujoco=MujocoCfg(cone=cone, **L.SIM_OPT))
    t0 = time.time()
    self.sim = Simulation(num_envs=n, cfg=cfg, model=m, device=device)
    self.build_s = time.time() - t0
    D = self.sim.wp_data
    self.qpos, self.qvel, self.ctrl = wp.to_torch(D.qpos), wp.to_torch(D.qvel), wp.to_torch(D.ctrl)
    self.q0 = torch.as_tensor(q0, device=self.qpos.device, dtype=self.qpos.dtype)
    self.c0 = torch.as_tensor(m.key_ctrl[0], device=self.ctrl.device, dtype=self.ctrl.dtype)
    self.mocap = wp.to_torch(D.mocap_pos) if self.table_mocap >= 0 else None
    self.n, self.nconmax, self.njmax = n, nconmax, njmax
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


def main(argv):
  out = argv[0]
  arms = argv[1:] or ["box", "o6", "boxpads", "o6pads"]
  plan = [(4096, cone, condim) for cone in ("pyramidal", "elliptic") for condim in (3, 4)] + [(8192, "pyramidal", 3)]
  rows = []
  for n, cone, condim in plan:
    base = Bench(None, cone, 3, n)
    for arm in arms:
      sc = Bench(arm, cone, condim, n)
      tb, ts = [base.time_once()], []
      for _ in range(2):
        ts.append(sc.time_once())
        tb.append(base.time_once())
      row = dict(arm=arm, cone=cone, condim=condim, n=n, t_base_ms=tb, t_scene_ms=ts, r_phys=float(np.median(ts) / np.median(tb)),
                 nconmax=sc.nconmax, njmax=sc.njmax, nacon_max=sc.nacon_max, nefc_max=sc.nefc_max,
                 overflow=bool(sc.nacon_max >= sc.nconmax * n or sc.nefc_max >= sc.njmax), others_on_gpu=C18.others_on_gpu())
      row["R"] = (41.6 + row["r_phys"] * 8.4e-3 * n) / (41.6 + 8.4e-3 * n)
      rows.append(row)
      print(row, flush=True)
      del sc
      torch.cuda.empty_cache()
    del base
    torch.cuda.empty_cache()
  json.dump(dict(rows=rows), open(out, "w"), indent=1)


if __name__ == "__main__":
  main(sys.argv[1:])
