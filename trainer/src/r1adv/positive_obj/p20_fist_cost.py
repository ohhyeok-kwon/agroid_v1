"""Simulation cost R of the fist bodies (rounds/R04/positive/r11_hold/rules.md 8.4): p20_cost (positive 18 cost.py method,
A-B-A-B-A blocks, 'park' scene = table + the three object bodies) with the fist arms' bodies and buffers; the base scene is
the training stock body (r1lab hand='stock', no table / object, nconmax 64, njmax 512). GPU exclusive.

  python -m r1adv.positive_obj.p20_fist_cost <out.json> [arm ...]
"""
from __future__ import annotations

import sys
import types

from r1adv.positive_obj import p20_cost as PC
from r1adv.positive_obj import p20_fist as F
from r1adv.positive_obj import p20_hold_mjw as HW


def build(arm, cone, condim):
  from mjlab.scene import Scene, SceneCfg
  from mjlab.sim import MujocoCfg
  from mjlab.terrains import TerrainEntityCfg
  from r1lab import robot as RB
  if arm is None:
    ents = {"robot": RB.get_r1_cfg(command_delay_max=0, self_collision=True, hand="stock")}
  else:
    from r1adv.positive_obj import p20_hold as H
    ents = {"robot": F.robot_cfg(arm, fixed=False), "table": PC.S.table_entity_cfg()}
    for shape in PC.PS.SHAPES:                                 # hold-test objects (bits 3/3: pads 2/2 pair with them)
      ents[PC.PS.ENTITY_OF[shape]] = H.object_entity_cfg(shape, condim)
  m = Scene(SceneCfg(num_envs=1, terrain=TerrainEntityCfg(terrain_type="plane"), entities=ents), device="cpu").compile()
  MujocoCfg(cone=cone, **PC.L.SIM_OPT).apply(m)
  return m


PC.build = build
PC.W = types.SimpleNamespace(buffers=lambda arm: HW.BUFFERS[arm])

if __name__ == "__main__":
  PC.main([sys.argv[1]] + (sys.argv[2:] or list(F.ARMS)))
