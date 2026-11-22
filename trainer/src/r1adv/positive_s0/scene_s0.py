"""R9 cell S0 'scene sentinel' (positive session 14; coordinator 2026-11-19 03:xx KST, .agent/adversarial/rounds/R04/
09B_r9_synthesis.md 2 / 4): the B0 prescription (A-GMM procedure and Hold objective on the unified O6 body, task
careful_walk_r7a_gmm_o6) plus indoor scenes = positive 13's scene task careful_walk_r8_scene_o6 (r1adv.positive_scene.scene_r8,
unchanged: both sources, library v2.2, train split only, teacher observations 810 + 472 with zero-initialised new inputs,
collision = a fall-like termination, allowed light hand touches continue) with
  flat_frac   0.30 (instead of 0.25)
  sampling    S-U: seeded uniform draws (proposer '', side weights unchanged)
  judge       rules v3 (negative 13, rounds/R04/r9_scene_v3.md, frozen 62afd403; scene_online.RULES_V3: allowed support band
              0.60-0.978 m = the robot's standing shoulder height, furniture top band applied at placement) instead of v2.
How v3 is switched on without touching scene_r8.py (e612efd5): every scene term of this task first installs
SceneRuntimeV3 as the env's runtime (env._r8_scene); scene_r8.runtime() then returns it to all of them. SceneRuntimeV3 is
scene_r8.SceneRuntime built while scene_online.RULES_V2 is temporarily bound to the requested rules (scene_r8 reads
SO.RULES_V2 only in its constructor path: the library class check and SceneOnline(rules=...)); the classes of v2 and v3 are
the same (checked). The spec dict carries rules='v3' (env.yaml) and the runtime's meta / judge report the rules it uses."""
from __future__ import annotations

from r1adv.negative import scene_online as SO
from r1adv.positive_scene import scene_r8 as R8

RULES = "v3"
FLAT_FRAC = 0.30


class SceneRuntimeV3(R8.SceneRuntime):
  """scene_r8.SceneRuntime whose SceneOnline / judge use SO.rules_of(spec['rules'])."""

  def __init__(self, env, spec: dict):
    rules = SO.rules_of(spec["rules"])
    if tuple(rules.classes) != tuple(SO.RULES_V2.classes):
      raise ValueError(f"rules {rules.name} classes differ from v2's (library labels)")
    saved = SO.RULES_V2
    SO.RULES_V2 = rules
    try:
      super().__init__(env, spec)
    finally:
      SO.RULES_V2 = saved
    if self.sc.rules.name != spec["rules"]:
      raise RuntimeError(f"scene judge built with {self.sc.rules.name}, spec asks {spec['rules']}")


def install(env, spec: dict):
  rt = getattr(env, "_r8_scene", None)
  if rt is None:
    rt = SceneRuntimeV3(env, spec)
    env._r8_scene = rt
  if not isinstance(rt, SceneRuntimeV3) or rt.sc.rules.name != spec["rules"]:
    raise RuntimeError("the env's scene runtime is not the v3 runtime of this task")
  return rt


class ScenePlaceV3(R8.ScenePlace):
  def __call__(self, env, env_ids, spec):
    install(env, spec)
    return super().__call__(env, env_ids, spec)


class SceneCollisionV3(R8.SceneCollision):
  def __call__(self, env, spec):
    install(env, spec)
    return super().__call__(env, spec)


def scene_rays_v3(env, spec):
  install(env, spec)
  return R8.scene_rays(env, spec)


def scene_boxes_v3(env, spec):
  install(env, spec)
  return R8.scene_boxes(env, spec)


def scene_hands_v3(env, spec):
  install(env, spec)
  return R8.scene_hands(env, spec)


OBS_V3 = {R8.scene_rays: scene_rays_v3, R8.scene_boxes: scene_boxes_v3, R8.scene_hands: scene_hands_v3}


def s0_spec() -> R8.SceneSpec:
  return R8.SceneSpec(libs=R8.SOURCES["both"], hand="o6", flat_frac=FLAT_FRAC, proposer="")


def s0_env_cfg(play: bool = False, t0: int = 1499):
  cfg = R8.scene_env_cfg(play=play, t0=t0, spec=s0_spec())
  sp = cfg.events["scene_place"].params["spec"]
  sp["rules"] = RULES                      # one dict shared by every scene term (scene_r8.scene_env_cfg)
  cfg.events["scene_place"].func = ScenePlaceV3
  cfg.terminations["scene_collision"].func = SceneCollisionV3
  for group in ("actor", "critic"):
    for name, term in cfg.observations[group].terms.items():
      if name.startswith("scene_"):
        if term.params["spec"] is not sp:
          raise ValueError("scene terms carry different spec dicts")
        term.func = OBS_V3[term.func]
  return cfg


def s0_rl_cfg():
  return R8.scene_rl_cfg(s0_spec())
