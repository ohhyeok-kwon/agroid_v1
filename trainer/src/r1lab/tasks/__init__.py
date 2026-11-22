"""Task registry: name -> env/rl cfg factories, mirror command sign, critic layout,
action (what is learned) and tags (key settings, part of the run ID)."""
from functools import partial

from r1lab import contract as C
from r1lab.registry import TASK_SPECS
from r1lab.tasks import careful_walk, safe_crouch
from r1lab.tasks.common import CRITIC_LAYOUT, CRITIC_LAYOUT_SCAN

TASKS = {
  "careful_walk": dict(env=careful_walk.careful_walk_env_cfg, rl=careful_walk.careful_walk_rl_cfg,
                       cmd_sign=C.VEL_CMD_SIGN, critic_layout=CRITIC_LAYOUT),
  "careful_walk_flat": dict(env=lambda play=False: careful_walk.careful_walk_env_cfg(play, flat=True),
                            rl=careful_walk.careful_walk_rl_cfg,
                            cmd_sign=C.VEL_CMD_SIGN, critic_layout=CRITIC_LAYOUT),
  "careful_walk_s1": dict(env=lambda play=False, t0=0: careful_walk.careful_walk_env_cfg(play, stage=1, t0=t0),
                          rl=careful_walk.careful_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN,
                          critic_layout=CRITIC_LAYOUT),
  "careful_walk_s2": dict(env=lambda play=False, t0=0: careful_walk.careful_walk_env_cfg(play, stage=2, t0=t0),
                          rl=careful_walk.careful_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN,
                          critic_layout=CRITIC_LAYOUT),
  "careful_walk_s3": dict(env=lambda play=False, t0=0: careful_walk.careful_walk_env_cfg(play, stage=2, t0=t0, slow_turn_focus=True),
                          rl=careful_walk.careful_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN,
                          critic_layout=CRITIC_LAYOUT),
  "careful_walk_s4": dict(env=lambda play=False, t0=0: careful_walk.careful_walk_env_cfg(play, stage=2, t0=t0, slow_turn_focus=True, clearance=0.09),
                          rl=careful_walk.careful_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN,
                          critic_layout=CRITIC_LAYOUT),
  # s5: obstacle robustness (foot-stumble penalty, harder debris/steps), from s4.
  "careful_walk_s5": dict(env=lambda play=False, t0=0: careful_walk.careful_walk_env_cfg(play, stage=2, t0=t0, slow_turn_focus=True, clearance=0.09, stumble=-1.0, hard_terrain=True),
                          rl=careful_walk.careful_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN,
                          critic_layout=CRITIC_LAYOUT),
  # s5p: s5 + privileged terrain scan in the critic (start with --actor-only, see r1lab.train).
  "careful_walk_s5p": dict(env=lambda play=False, t0=0: careful_walk.careful_walk_env_cfg(play, stage=2, t0=t0, slow_turn_focus=True, clearance=0.09, stumble=-1.0, hard_terrain=True, scan=not play),
                           rl=careful_walk.careful_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN,
                           critic_layout=CRITIC_LAYOUT_SCAN),
  "safe_crouch": dict(env=safe_crouch.safe_crouch_env_cfg, rl=safe_crouch.safe_crouch_rl_cfg,
                      cmd_sign=(1.0, 1.0, 1.0), critic_layout=CRITIC_LAYOUT),
}

for suffix, variant in zip("abcd", ("control", "burst", "gated", "combined")):
  TASKS[f"careful_walk_s6{suffix}"] = dict(
    env=partial(careful_walk.recovery_walk_env_cfg, variant=variant),
    rl=careful_walk.careful_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN,
    critic_layout=CRITIC_LAYOUT)

for suffix, min_scale in (("a", .25), ("b", .05)):
  TASKS[f"careful_walk_s7{suffix}"] = dict(
    env=partial(careful_walk.projectile_walk_env_cfg, min_scale=min_scale),
    rl=careful_walk.projectile_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN,
    critic_layout=CRITIC_LAYOUT)

from r1lab.tasks.adversarial_walk import adversarial_walk_env_cfg, adversarial_walk_rl_cfg

TASKS["careful_walk_s8"] = dict(env=adversarial_walk_env_cfg, rl=adversarial_walk_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.reference_walk import reference_walk_env_cfg
TASKS["careful_walk_s8r"] = dict(env=reference_walk_env_cfg, rl=adversarial_walk_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.motion_teacher import motion_teacher_env_cfg, motion_teacher_rl_cfg
TASKS["careful_walk_teacher"] = dict(env=motion_teacher_env_cfg, rl=motion_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.precise_teacher import precise_teacher_env_cfg, precise_teacher_rl_cfg
TASKS["careful_walk_teacher_precise"] = dict(env=precise_teacher_env_cfg, rl=precise_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.coordinated_teacher import coordinated_teacher_env_cfg, coordinated_teacher_rl_cfg
TASKS["careful_walk_teacher_coordinated"] = dict(env=coordinated_teacher_env_cfg, rl=coordinated_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.fullcapture_teacher import fullcapture_teacher_env_cfg, fullcapture_teacher_rl_cfg
TASKS["careful_walk_teacher_fullcapture"] = dict(env=fullcapture_teacher_env_cfg, rl=fullcapture_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.phasefree_teacher import phasefree_teacher_env_cfg, phasefree_teacher_rl_cfg
TASKS["careful_walk_teacher_phasefree"] = dict(env=phasefree_teacher_env_cfg, rl=phasefree_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.library_teacher import library_teacher_env_cfg, library_teacher_rl_cfg
TASKS["careful_walk_teacher_library"] = dict(env=library_teacher_env_cfg, rl=library_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.library_teacher import library_xy_teacher_env_cfg
TASKS["careful_walk_teacher_library_xy"] = dict(env=library_xy_teacher_env_cfg, rl=library_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_teacher_env_cfg, matching_teacher_rl_cfg
TASKS["careful_walk_teacher_matching"] = dict(env=matching_teacher_env_cfg, rl=matching_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_push_teacher_env_cfg
TASKS["careful_walk_teacher_matching_push"] = dict(env=matching_push_teacher_env_cfg, rl=matching_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_quiet_teacher_env_cfg
TASKS["careful_walk_teacher_matching_quiet"] = dict(env=matching_quiet_teacher_env_cfg, rl=matching_teacher_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_projectile_teacher_env_cfg
TASKS["careful_walk_teacher_matching_projectile"] = dict(env=matching_projectile_teacher_env_cfg,
    rl=matching_teacher_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_soft_teacher_env_cfg
TASKS["careful_walk_teacher_matching_soft"] = dict(env=matching_soft_teacher_env_cfg,
    rl=matching_teacher_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_v4_teacher_env_cfg
TASKS["careful_walk_teacher_matching_v4"] = dict(env=matching_v4_teacher_env_cfg,
    rl=matching_teacher_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_recover_teacher_env_cfg
TASKS["careful_walk_teacher_matching_recover"] = dict(env=matching_recover_teacher_env_cfg,
    rl=matching_teacher_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_v4_ball_teacher_env_cfg
TASKS["careful_walk_teacher_matching_v4_ball"] = dict(env=matching_v4_ball_teacher_env_cfg,
    rl=matching_teacher_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_still_teacher_env_cfg
TASKS["careful_walk_teacher_matching_still"] = dict(env=matching_still_teacher_env_cfg,
    rl=matching_teacher_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_arms_teacher_env_cfg
TASKS["careful_walk_teacher_matching_arms"] = dict(env=matching_arms_teacher_env_cfg,
    rl=matching_teacher_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_teacher import matching_lateral_teacher_env_cfg
TASKS["careful_walk_teacher_matching_lateral"] = dict(env=matching_lateral_teacher_env_cfg,
    rl=matching_teacher_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_student import matching_student_env_cfg, matching_student_still_env_cfg, matching_student_rl_cfg
TASKS["careful_walk_student_matching"] = dict(env=matching_student_still_env_cfg, rl=matching_student_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_student import matching_student_crisis_env_cfg
TASKS["careful_walk_student_matching_crisis"] = dict(env=matching_student_crisis_env_cfg, rl=matching_student_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.matching_student import (STUDENT_CONTRACT_LATERAL, matching_student_crisis_lateral_env_cfg,
                                          matching_student_lateral_rl_cfg)
TASKS["careful_walk_student_matching_crisis_lateral"] = dict(env=matching_student_crisis_lateral_env_cfg,
    rl=matching_student_lateral_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False,
    student_contract=STUDENT_CONTRACT_LATERAL, random_episode_start=True)
from r1lab.tasks.matching_student import matching_student_crisis_arms_env_cfg
TASKS["careful_walk_student_matching_crisis_arms"] = dict(env=matching_student_crisis_arms_env_cfg,
    rl=matching_student_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False)
from r1lab.tasks.student_rl import (STUDENT_CONTRACT_LATERAL_RL, student_rl_lateral_env_cfg,
                                    student_rl_lateral_rl_cfg)
TASKS["careful_walk_student_rl_lateral"] = dict(env=student_rl_lateral_env_cfg, rl=student_rl_lateral_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False,
    student_contract=STUDENT_CONTRACT_LATERAL_RL, random_episode_start=True)
from r1lab.tasks.student_rl import (STUDENT_CONTRACT_TERRAIN, student_terrain_env_cfg,
                                    student_terrain_rl_cfg)
TASKS["careful_walk_student_terrain"] = dict(env=student_terrain_env_cfg, rl=student_terrain_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False,
    student_contract=STUDENT_CONTRACT_TERRAIN, random_episode_start=True)
from r1lab.tasks.student_rl import (STUDENT_CONTRACT_TERRAIN_DENSE, student_terrain_dense_env_cfg,
                                    student_terrain_dense_rl_cfg)
TASKS["careful_walk_student_terrain_dense"] = dict(env=student_terrain_dense_env_cfg, rl=student_terrain_dense_rl_cfg,
    cmd_sign=C.VEL_CMD_SIGN, critic_layout=None, symmetry=False,
    student_contract=STUDENT_CONTRACT_TERRAIN_DENSE, random_episode_start=True)

# Whole-body family: environments and training loop live in r1wb; listed here so both tables agree.
from r1wb.tasks import TASKS as WHOLE_BODY_TASKS
for _name, _task in WHOLE_BODY_TASKS.items():
  TASKS[_name] = dict(env=_task["env"], rl=_task["rl"], cmd_sign=None, critic_layout=None, symmetry=False)

# Adversarial co-evolution program, positive agent: R1 Hold bracket (code in r1adv.positive.hold_task, imported lazily
# so that an error there cannot break the other tasks' registration).
def _adv_hold(prev_action, beta):
  def env(**kwargs):
    from r1adv.positive.hold_task import hold_env_cfg
    return hold_env_cfg(prev_action=prev_action, beta=beta, **kwargs)
  def rl():
    from r1adv.positive.hold_task import hold_rl_cfg
    return hold_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
for _name, _prev, _beta in (("adv_hold_b0_prev", True, 0.0), ("adv_hold_b0_noprev", False, 0.0),
                            ("adv_hold_b1_prev", True, 1.0), ("adv_hold_b1_noprev", False, 1.0)):
  TASKS[_name] = _adv_hold(_prev, _beta)

# Adversarial co-evolution program, positive agent: R2 Hold teachers C0 / T (coordinator lock 2026-11-17 10:12 KST;
# code in r1adv.positive.hold_r2, imported lazily so that an error there cannot break the other tasks' registration).
def _adv_hold_r2(actor_clean, velocity_input=True, arm="c0"):
  def env(**kwargs):
    from r1adv.positive.hold_r2 import hold_r2_env_cfg
    return hold_r2_env_cfg(actor_clean=actor_clean, velocity_input=velocity_input, **kwargs)
  def rl():
    from r1adv.positive.hold_r2 import hold_r2_rl_cfg
    return hold_r2_rl_cfg(arm=arm)
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
for _name, _clean in (("adv_hold_r2_c0", False), ("adv_hold_r2_t", True)):
  TASKS[_name] = _adv_hold_r2(_clean)
TASKS["adv_hold_r2_v"] = _adv_hold_r2(False, velocity_input=False)    # R2 wave 2 arm V (no joint-velocity input)
TASKS["adv_hold_r2_a"] = _adv_hold_r2(False, arm="a")                 # R2 wave 2 arm A (residual AR(1), exact likelihood)

# R2 one-factor runs (coordinator 2026-11-17 12:18 KST, coordinator_decisions_wave1_collapse.md): all T form (actor reads
# the noise-free frames), one change at a time from N = the R2 code path with default starts and no hits.
def _adv_hold_r2_of(human_starts, hit_impulse, entropy_coef=None):
  def env(**kwargs):
    from r1adv.positive.hold_r2 import hold_r2_env_cfg
    return hold_r2_env_cfg(actor_clean=True, human_starts=human_starts, hit_impulse=hit_impulse, **kwargs)
  def rl():
    from r1adv.positive.hold_r2 import hold_r2_rl_cfg
    return hold_r2_rl_cfg(entropy_coef=entropy_coef)
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
TASKS["adv_hold_r2_of_n"] = _adv_hold_r2_of(False, None)                 # N: default starts, no hits
TASKS["adv_hold_r2_of_s"] = _adv_hold_r2_of(True, None)                  # S: N + human standing starts
TASKS["adv_hold_r2_of_p"] = _adv_hold_r2_of(False, "dist")               # P: N + hits 4-25 N s
TASKS["adv_hold_r2_of_plow"] = _adv_hold_r2_of(False, (2.0, 6.0))        # P-low: N + hits 2-6 N s
TASKS["adv_hold_r2_of_pe0"] = _adv_hold_r2_of(False, "dist", 0.0)        # P-e0: P + entropy bonus 0

# R2c (coordinator 13:0x): T-form Hold teacher with the negative's adversarial hit curriculum (level feed directory from
# R1ADV_FEED_DIR at configuration time; anchor = no hit + the canonical start spread).
def _adv_hold_r2c():
  def env(**kwargs):
    from r1adv.positive.hold_r2c import hold_r2c_env_cfg
    return hold_r2c_env_cfg(**kwargs)
  def rl():
    from r1adv.positive.hold_r2 import hold_r2_rl_cfg
    return hold_r2_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
TASKS["adv_hold_r2c"] = _adv_hold_r2c()


# R2d (coordinator 15:3x, conditional lock): the R2c curriculum teacher + per-episode actuator PD randomisation (main
# arm) / without the joint-velocity input (pair arm, 330 frames); plus a T-form 330 evaluation env.
def _adv_hold_r2d(pd_random, velocity_input):
  def env(**kwargs):
    from r1adv.positive.hold_r2c import hold_r2c_env_cfg
    return hold_r2c_env_cfg(pd_random=pd_random, velocity_input=velocity_input, **kwargs)
  def rl():
    from r1adv.positive.hold_r2 import hold_r2_rl_cfg
    return hold_r2_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
TASKS["adv_hold_r2d_pd"] = _adv_hold_r2d(True, True)
TASKS["adv_hold_r2d_v"] = _adv_hold_r2d(False, False)
def _adv_hold_r2_of_nv():
  def env(**kwargs):
    from r1adv.positive.hold_r2 import hold_r2_env_cfg
    return hold_r2_env_cfg(actor_clean=True, human_starts=False, hit_impulse=None, velocity_input=False, **kwargs)
  def rl():
    from r1adv.positive.hold_r2 import hold_r2_rl_cfg
    return hold_r2_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
TASKS["adv_hold_r2_of_nv"] = _adv_hold_r2_of_nv()                      # evaluation env of the 330 arm (no starts/hits)

# R2e (coordinator 17:17, coordinator_decisions_r2e.md 4): fair adversarial-structure comparison. Curriculum arm =
# base episodes with the R2 distribution's uniform 4-25 N s hits, level episodes (fixed share 0.3) from the negative's
# bundle, default-pose starts only (as P-e0), entropy 0; control arm = adv_hold_r2_of_pe0 (seed 2).
def _adv_hold_r2e():
  def env(**kwargs):
    from r1adv.positive.hold_r2c import hold_r2c_env_cfg
    return hold_r2c_env_cfg(r2e=True, **kwargs)
  def rl():
    from r1adv.positive.hold_r2 import hold_r2_rl_cfg
    return hold_r2_rl_cfg(entropy_coef=0.0)
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
TASKS["adv_hold_r2e_cur"] = _adv_hold_r2e()

# R3 wave 1 (coordinator 2026-11-17 21:41 KST, .agent/adversarial/rounds/R03/coordinator_decisions_r3_wave1.md): 2x2 from
# scratch = foot-lead input b (absent / present) x posture channel of L (point q* / human distribution), the P-e0 recipe
# otherwise (code in r1adv.positive.hold_w1, imported lazily so that an error there cannot break the other tasks).
def _adv_hold_w1(lead, posture):
  def env(**kwargs):
    from r1adv.positive.hold_w1 import hold_w1_env_cfg
    return hold_w1_env_cfg(lead=lead, posture=posture, **kwargs)
  def rl():
    from r1adv.positive.hold_w1 import hold_w1_rl_cfg
    return hold_w1_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
for _name, _lead, _posture in (("adv_hold_w1_c0", False, "point"), ("adv_hold_w1_b1", True, "point"),
                               ("adv_hold_w1_l1", False, "dist"), ("adv_hold_w1_bl", True, "dist")):
  TASKS[_name] = _adv_hold_w1(_lead, _posture)

# R4 wave 2 (coordinator 2026-11-18 01:4x KST, .agent/adversarial/rounds/R04/04_r4_state.md 3): training-side arms on the
# W1 recipe, one axis each (code in r1adv.positive.hold_w2, imported lazily so that an error there cannot break the
# other tasks): XA / XB / XC and the follow-up variants XAs / XAf / XAn.
def _adv_hold_w2(cell):
  def env(**kwargs):
    from r1adv.positive.hold_w2 import cell_env_cfg
    return cell_env_cfg(cell, **kwargs)
  def rl():
    from r1adv.positive.hold_w2 import cell_rl_cfg
    return cell_rl_cfg(cell)
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
for _name, _cell in (("adv_hold_w2_xa", "XA"), ("adv_hold_w2_xb", "XB"), ("adv_hold_w2_xc", "XC"),
                     ("adv_hold_w2_xas", "XAs"), ("adv_hold_w2_xaf", "XAf"), ("adv_hold_w2_xan", "XAn")):
  TASKS[_name] = _adv_hold_w2(_cell)

# R5 wave 3 (coordinator 2026-11-18 07:1x KST, .agent/adversarial/rounds/R04/06_wave3_plan.md): the friction-world far end
# on C0 / XB (code in r1adv.positive.hold_w3, imported lazily): XBF / XBN (XB-null) / C0F.
def _adv_hold_w3(cell):
  def env(**kwargs):
    from r1adv.positive.hold_w3 import cell_env_cfg
    return cell_env_cfg(cell, **kwargs)
  def rl():
    from r1adv.positive.hold_w3 import cell_rl_cfg
    return cell_rl_cfg(cell)
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
for _name, _cell in (("adv_hold_w3_xbf", "XBF"), ("adv_hold_w3_xbn", "XBN"), ("adv_hold_w3_c0f", "C0F")):
  TASKS[_name] = _adv_hold_w3(_cell)

# R6 wave-4 option S (coordinator 2026-11-18 11:0x KST, .agent/adversarial/rounds/R04/06B_r6_synthesis.md lever S): the
# human-push disturbance (code in r1adv.positive.hold_w4s, imported lazily); XBS = XB + S.
def _adv_hold_w4s(cell):
  def env(**kwargs):
    from r1adv.positive.hold_w4s import cell_env_cfg
    return cell_env_cfg(cell, **kwargs)
  def rl():
    from r1adv.positive.hold_w4s import cell_rl_cfg
    return cell_rl_cfg(cell)
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
TASKS["adv_hold_w4_xbs"] = _adv_hold_w4s("XBS")

# R7 wave-4 branch B (positive 10, 2026-11-18): XBS with only the D slot of r = L + D replaced by the online phi
# discriminator D_online, panel design (code in r1adv.positive.hold_w4amp, imported lazily).
def _adv_hold_w4amp(variant):
  def env(**kwargs):
    from r1adv.positive.hold_w4amp import cell_env_cfg
    return cell_env_cfg(variant, **kwargs)
  def rl():
    from r1adv.positive.hold_w4amp import cell_rl_cfg
    return cell_rl_cfg(variant)
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
TASKS["adv_hold_w4_xbs_amp"] = _adv_hold_w4amp("DON")

# Adversarial co-evolution program, branch W: W1-J tracker E1 bracket (code in r1adv.wam.track_task, imported lazily
# so that an error there cannot break the other tasks' registration).
def _adv_wam(arm):
  def env(**kwargs):
    from r1adv.wam.track_task import env_cfg
    return env_cfg(arm, **kwargs)
  def rl():
    from r1adv.wam.track_task import rl_cfg
    return rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
for _name, _arm in (("adv_wam_e1a", "A"), ("adv_wam_e1b", "B")):
  TASKS[_name] = _adv_wam(_arm)

# Adversarial co-evolution program, branch W: W1-J tracker E1 relaunch, init std 1.0 (r1adv.wam.track_task, lazy
# so that an error there cannot break the other tasks' registration).
def _adv_wam_s1(arm):
  def env(**kwargs):
    from r1adv.wam.track_task import env_cfg
    return env_cfg(arm, **kwargs)
  def rl():
    from r1adv.wam.track_task import rl_cfg
    return rl_cfg(init_std=1.0)
  return dict(env=env, rl=rl, cmd_sign=None, critic_layout=None, symmetry=False)
for _name, _arm in (("adv_wam_e1a_s1", "A"), ("adv_wam_e1b_s1", "B")):
  TASKS[_name] = _adv_wam_s1(_arm)

# R6 diagnosis arm r6d (coordinator 2026-11-18 10:5x KST, .agent/adversarial/rounds/R04/04_r4_state.md 10:58): s2f/3300
# continued with its own careful_walk stage-2 objective (t0 1499), only the hits change (code in r1adv.positive.s2f_r6d,
# imported lazily): ctl / imp (Hold hit stream) / long (imp + option S). Same mirror augmentation as careful_walk_s2.
def _s2f_r6d(cell):
  def env(play=False, t0=1499):
    from r1adv.positive.s2f_r6d import s2f_env_cfg
    return s2f_env_cfg(cell, play=play, t0=t0)
  def rl():
    from r1adv.positive.s2f_r6d import s2f_rl_cfg
    return s2f_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=C.VEL_CMD_SIGN, critic_layout=CRITIC_LAYOUT)
for _name, _cell in (("careful_walk_r6d_ctl", "ctl"), ("careful_walk_r6d_imp", "imp"), ("careful_walk_r6d_long", "long")):
  TASKS[_name] = _s2f_r6d(_cell)

# R7 branch A control cell A-GMM (coordinator 2026-11-18 20:2x KST, .agent/adversarial/rounds/R04/07B_r7_synthesis.md
# 5.1): the r6d long parents' environment with the Hold objective (r = L + D, arm target r7a), all-standing commands, flat
# floor, no curricula (code in r1adv.positive.s2f_r7a, imported lazily). Same mirror augmentation as careful_walk_s2.
def _s2f_r7a():
  def env(play=False, t0=1499):
    from r1adv.positive.s2f_r7a import r7a_env_cfg
    return r7a_env_cfg(play=play, t0=t0)
  def rl():
    from r1adv.positive.s2f_r7a import r7a_rl_cfg
    return r7a_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=C.VEL_CMD_SIGN, critic_layout=CRITIC_LAYOUT)
TASKS["careful_walk_r7a_gmm"] = _s2f_r7a()

# R9 S0 scene sentinel (positive 14, coordinator 2026-11-19 03:xx KST, rounds/R04/09B_r9_synthesis.md 2 / 4): the B0
# prescription (A-GMM procedure and Hold objective on the unified O6 body) + indoor scenes: positive 13's
# careful_walk_r8_scene_o6 with flat share 0.30, seeded uniform draws (S-U) and the scene judge on rules v3
# (r1adv.positive_s0.scene_s0, imported lazily; scene_r8.py and r1adv.positive untouched).
def _r9_s0():
  def env(play=False, t0=1499):
    from r1adv.positive_s0.scene_s0 import s0_env_cfg
    return s0_env_cfg(play=play, t0=t0)
  def rl():
    from r1adv.positive_s0.scene_s0 import s0_rl_cfg
    return s0_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=C.VEL_CMD_SIGN, critic_layout=CRITIC_LAYOUT)
TASKS["careful_walk_r9_s0"] = _r9_s0()

# R10 / W7 candidate cells (positive 16, coordinator 2026-11-19 09:xx KST, rounds/R04/10_r10_facts.md): the S0 task
# (careful_walk_r9_s0: B0 + scenes, judge v3, flat 0.30, O6 body, ScenePPO) with one change each, defined here so that
# r1adv and its code hashes stay unchanged. (a) S0-e0: entropy 0. (b) S0-sfix: the actor std is not learned (rsl_rl
# GaussianDistribution learn_std=False; the loaded std vector stays). (c) STAND-S: the reward block of A-STAND (the
# parents' s2f objective, 31 terms + AMP style; terminated -100 also covers scene_collision). (d) S0-cont: S0 itself
# (parent B0 /4800 is a launch choice); (d') the same with the resumed optimizer's own learning rate.
def _r10_s0e0():
  base = _r9_s0()
  def rl():
    cfg = base["rl"]()
    cfg.algorithm.entropy_coef = 0.0
    return cfg
  return dict(base, rl=rl)
def _r10_s0sfix():
  base = _r9_s0()
  def rl():
    cfg = base["rl"]()
    cfg.actor.distribution_cfg = dict(cfg.actor.distribution_cfg, learn_std=False)
    return cfg
  return dict(base, rl=rl)
def _r10_stands():
  base = _r9_s0()
  def env(play=False, t0=1499):
    from r1adv.positive.s2f_r6d import s2f_env_cfg
    cfg = base["env"](play=play, t0=t0)
    cfg.rewards = s2f_env_cfg("long", play=play, t0=t0).rewards
    return cfg
  return dict(base, env=env)
def _r10_s0contlr():
  base = _r9_s0()
  def rl():
    cfg = base["rl"]()
    cfg.algorithm.start_learning_rate = None
    return cfg
  return dict(base, rl=rl)
TASKS["careful_walk_r10_s0e0"] = _r10_s0e0()
TASKS["careful_walk_r10_s0sfix"] = _r10_s0sfix()
TASKS["careful_walk_r10_stands"] = _r10_stands()
TASKS["careful_walk_r10_s0cont"] = _r9_s0()
TASKS["careful_walk_r10_s0contlr"] = _r10_s0contlr()

# R10 cells b' S0-ssched and f SHA-s (positive 16, coordinator 2026-11-19 09:2x KST; r1adv.positive_r10.r10, imported
# lazily): S0 with the actor std set every update to B0's recorded trajectory (table r10_sigma_b0_v1.json pinned by sha,
# cell from R1ADV_SIGMA_CELL), entropy 0; SHA-s also replaces the Hold objective's D by the walking AMP style (weight 1).
def _r10_s0ssch():
  base = _r9_s0()
  def rl():
    from r1adv.positive_r10.r10 import ssched_rl_cfg
    return ssched_rl_cfg()
  return dict(base, rl=rl)
def _r10_shas():
  base = _r10_s0ssch()
  def env(play=False, t0=1499):
    from r1adv.positive_r10.r10 import shas_env_cfg
    return shas_env_cfg(play=play, t0=t0)
  return dict(base, env=env)
TASKS["careful_walk_r10_s0ssch"] = _r10_s0ssch()
TASKS["careful_walk_r10_shas"] = _r10_shas()

# A-GMM-O6 (positive 14, coordinator 2026-11-19 02:0x KST: common control of the next wave): careful_walk_r7a_gmm with
# only the robot body changed to the unified O6 body: r1lab.robot.use_o6_hands = robot entity get_r1_cfg(hand="o6")
# (body signature 15446e6f) and the 'hand_load' event moved from the wrist links onto the O6 bodies (-0.15 .. +0.37 kg).
# Objective, commands, floor, hits, push, terminations, episode, PPO as A-GMM. Defined here so that r1adv.positive and
# its code hash stay unchanged.
def _s2f_r7a_o6():
  base = _s2f_r7a()
  def env(play=False, t0=1499):
    from r1lab.robot import use_o6_hands
    cfg = base["env"](play=play, t0=t0)
    use_o6_hands(cfg)
    return cfg
  return dict(base, env=env)
TASKS["careful_walk_r7a_gmm_o6"] = _s2f_r7a_o6()

# R10 contrast cell C2-flat (positive 16, coordinator 2026-11-19 09:5x KST; r1adv.positive_r10.r10, imported lazily):
# B0 (careful_walk_r7a_gmm_o6) with r = L + D_AMP (walking AMP style, weight 1, objective beta 0) and the std set every
# update to B0's own trajectory (SigmaSchedPPO, entropy 0, learn_std False); flat floor, no scene inputs.
def _r10_shaflat():
  base = _s2f_r7a_o6()
  def env(play=False, t0=1499):
    from r1adv.positive_r10.r10 import use_d_amp
    return use_d_amp(base["env"](play=play, t0=t0), play=play, t0=t0)
  def rl():
    from r1adv.positive_r10.r10 import ssched_rl
    return ssched_rl(base["rl"](), scene=False)
  return dict(base, env=env, rl=rl)
TASKS["careful_walk_r10_shaflat"] = _r10_shaflat()

# D_shape (positive 15, coordinator 2026-11-19 02:4x KST, rounds/R04/09B_r9_synthesis.md 4.5): careful_walk_r7a_gmm_o6
# (B0) with only the D slot of its objective replaced: positive 10's online discriminator v1 (LSGAN, AMP map, no context)
# on the 27 position-free phi channels + ten arm / trunk shape channels (r1adv.positive_shape.dshape). L, commands,
# floor, hits, push, terminations, episode, PPO as B0; the play / evaluation env is B0's.
def _s2f_r7a_o6_dshape():
  base = _s2f_r7a_o6()
  def env(play=False, t0=1499):
    from r1adv.positive_shape.dshape import dshape_env_cfg
    return dshape_env_cfg(play=play, t0=t0)
  return dict(base, env=env)
TASKS["careful_walk_r9_dshape_o6"] = _s2f_r7a_o6_dshape()

# ARMLOCK (positive 15, coordinator 2026-11-19 02:4x KST, rounds/R04/09B_r9_synthesis.md 4.3; diagnosis only, never a
# deployment body): careful_walk_r7a_gmm_o6 (B0) with only the arm lock of r1lab.armlock.use_o6_armlock: the ten arm
# joints welded at the arm entries of q*_r7a by joint equality constraints, the arm actuators inert (kp = kd = 0), and a
# reset event writing the arms to the lock pose. Objective, commands, floor, hits, push, terminations, episode, PPO
# as B0. Defined here so that r1adv.positive and its code hash stay unchanged.
def _s2f_r7a_o6_armlock():
  base = _s2f_r7a_o6()
  def env(play=False, t0=1499):
    from r1lab.armlock import use_o6_armlock
    cfg = base["env"](play=play, t0=t0)
    use_o6_armlock(cfg)
    return cfg
  return dict(base, env=env)
TASKS["careful_walk_r9_armlock_o6"] = _s2f_r7a_o6_armlock()

# R8 diagnosis cell A-STAND (positive 14, coordinator 2026-11-19 00:1x KST, rounds/R04/08B_r8_synthesis.md 6):
# careful_walk_r7a_gmm with only the reward block changed back to the parents' own s2f objective = the careful_walk
# stage-2 rewards of careful_walk_r6d_long (31 terms + AMP style 1.5, pose 0.4; the configured weights are the parents'
# weights since every robustness_ramp ends at 1.0 x the configured value). Commands (all standing), flat floor, no
# curricula, hits, push, terminations, episode, PPO (ParentLrPPO, start lr 1.139e-4, entropy 0.005) as A-GMM. The AMP
# discriminator is the parent run's amp_disc.pt (--amp-disc). Defined here so that r1adv.positive and its code hash
# stay unchanged.
def _s2f_r8_astand():
  base = _s2f_r7a()
  def env(play=False, t0=1499):
    from r1adv.positive.s2f_r6d import s2f_env_cfg
    cfg = base["env"](play=play, t0=t0)
    cfg.rewards = s2f_env_cfg("long", play=play, t0=t0).rewards
    return cfg
  return dict(base, env=env)
TASKS["careful_walk_r8_astand"] = _s2f_r8_astand()

# R8 scratch probe (coordinator 2026-11-18 22:5x): careful_walk_r7a_gmm trained from scratch (no parent weights, no
# critic warm-up); only the checkpoint interval differs (50 instead of 100). Defined here so that r1adv.positive and
# its code hash stay unchanged.
def _s2f_r7a_scratch():
  base = _s2f_r7a()
  def rl():
    cfg = base["rl"]()
    cfg.save_interval = 50
    return cfg
  return dict(base, rl=rl)
TASKS["careful_walk_r7a_gmm_scratch"] = _s2f_r7a_scratch()

# R7 branch A cell A-GMMe0 (coordinator 2026-11-18 21:2x): A-GMM with entropy_coef 0.0 (the Hold runs' value) instead of
# careful_walk's 0.005, nothing else (defined here so that r1adv.positive and its code hash stay unchanged).
def _s2f_r7a_e0():
  base = _s2f_r7a()
  def rl():
    cfg = base["rl"]()
    cfg.algorithm.entropy_coef = 0.0
    return cfg
  return dict(base, rl=rl)
TASKS["careful_walk_r7a_gmm_e0"] = _s2f_r7a_e0()

# R8 scene task (positive 13, coordinator 2026-11-18 23:0x KST, .agent/adversarial/rounds/R04/positive/r8_scene/README.md):
# A-GMM + indoor scene patches (negative library v2.1, train split, seeded draws), the online contact judge ending the episode
# like a fall, scene observations appended to actor and critic (parents widened with zero weights). Code in
# r1adv.positive_scene.scene_r8 (imported lazily; outside r1adv.positive so that package's code hash stays unchanged).
# Library v2.1 (coordinator 23:4x); source 'both' (K 1,536 for every env) or one source (_pt K 176, _sn K 1,536).
def _r8_scene(source, hand="stock"):
  def env(play=False, t0=1499):
    from r1adv.positive_scene.scene_r8 import scene_env_cfg
    return scene_env_cfg(play=play, t0=t0, source=source, hand=hand)
  def rl():
    from r1adv.positive_scene.scene_r8 import scene_rl_cfg
    return scene_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=C.VEL_CMD_SIGN, critic_layout=CRITIC_LAYOUT)
for _name, _source in (("careful_walk_r8_scene", "both"), ("careful_walk_r8_scene_pt", "procthor"),
                       ("careful_walk_r8_scene_sn", "scannetpp")):
  TASKS[_name] = _r8_scene(_source)
  TASKS[_name + "_o6"] = _r8_scene(_source, "o6")   # the same on the O6-hand body (r1lab.robot.use_o6_hands = r1wb body)

# R13 anchor careful_walk_s1_repro (positive 23, coordinator 2026-11-20 14:4x): careful_walk stage 1 from random weights
# in today's code with the walking run's recorded tracking terms (r1lab.tasks.careful_walk_repro, imported lazily).
def _cw_repro():
  def env(play=False, t0=0):
    from r1lab.tasks.careful_walk_repro import repro_env_cfg
    return repro_env_cfg(play=play, t0=t0)
  return dict(env=env, rl=careful_walk.careful_walk_rl_cfg, cmd_sign=C.VEL_CMD_SIGN, critic_layout=CRITIC_LAYOUT)
TASKS["careful_walk_s1_repro"] = _cw_repro()

# E1 c13 careful_walk_c13 (positive 23, coordinator 2026-11-20 16:0x): anchor + S1 environment leaves (r1) / + S1
# representation-optimisation leaves (r2); rewards careful_walk's (r1lab.tasks.careful_walk_c13, imported lazily).
def _cw_c13(arm):
  def env(play=False, t0=0):
    from r1lab.tasks import careful_walk_c13 as M
    return (M.r1_env_cfg if arm == "r1" else M.r2_env_cfg)(play=play, t0=t0)
  def rl():
    from r1lab.tasks import careful_walk_c13 as M
    return careful_walk.careful_walk_rl_cfg() if arm == "r1" else M.r2_rl_cfg()
  extra = {} if arm == "r1" else {"symmetry": False}
  return dict(env=env, rl=rl, cmd_sign=C.VEL_CMD_SIGN, critic_layout=CRITIC_LAYOUT, **extra)
TASKS["careful_walk_c13_r1"] = _cw_c13("r1")
TASKS["careful_walk_c13_r2"] = _cw_c13("r2")

# B0 careful_walk_b0 (positive 23, coordinator 2026-11-20 18:2x): S1 objective v2 env + the anchor's representation and
# trainer (r1lab.tasks.careful_walk_b0, imported lazily).
def _cw_b0():
  def env(play=False, t0=0):
    from r1lab.tasks import careful_walk_b0 as M
    return M.b0_env_cfg(play=play, t0=t0)
  def rl():
    from r1lab.tasks import careful_walk_b0 as M
    return M.b0_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=(1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
              critic_layout=(("actor", 870), ("frame", 87), ("vec3", 3), ("scalar", 1), ("lr2", 2), ("lr2", 2), ("lr_vec3", 6)))
TASKS["careful_walk_b0"] = _cw_b0()

# B0AMP careful_walk_b0amp (positive 23, coordinator 2026-11-20 18:4x): B0 + the anchor's AMP style term
# (r1lab.tasks.careful_walk_b0amp, imported lazily).
def _cw_b0amp():
  def env(play=False, t0=0):
    from r1lab.tasks import careful_walk_b0amp as M
    return M.b0amp_env_cfg(play=play, t0=t0)
  def rl():
    from r1lab.tasks import careful_walk_b0amp as M
    return M.b0amp_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=(1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
              critic_layout=(("actor", 870), ("frame", 87), ("vec3", 3), ("scalar", 1), ("lr2", 2), ("lr2", 2), ("lr_vec3", 6)))
TASKS["careful_walk_b0amp"] = _cw_b0amp()

# E3 careful_walk_e3 (positive 23, coordinator 2026-11-20 22:0x): L = B0 + sigma 0.34 fixed, a04 = B0 + push
# boundary d (r1lab.tasks.careful_walk_e3, imported lazily).
def _cw_e3(arm):
  def env(play=False, t0=0):
    if arm == "l":
      from r1lab.tasks import careful_walk_b0 as M
      return M.b0_env_cfg(play=play, t0=t0)
    from r1lab.tasks import careful_walk_e3 as M
    return M.a04_env_cfg(play=play, t0=t0)
  def rl():
    from r1lab.tasks import careful_walk_e3 as M
    from r1lab.tasks import careful_walk_b0 as B
    return M.l_rl_cfg() if arm == "l" else B.b0_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=(1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
              critic_layout=(("actor", 870), ("frame", 87), ("vec3", 3), ("scalar", 1), ("lr2", 2), ("lr2", 2), ("lr_vec3", 6)))
TASKS["careful_walk_e3l"] = _cw_e3("l")
TASKS["careful_walk_e3a04"] = _cw_e3("a04")

# E3X careful_walk_e3x (positive 23, coordinator 2026-11-20 23:0x): one seed per arm - cap = B0 + std capped at 0.8,
# capa04 = cap + push boundary d, col = B0 + time-correlated exploration noise (r1lab.tasks.careful_walk_e3x, lazy).
def _cw_e3x(arm):
  def env(play=False, t0=0):
    if arm == "capa04":
      from r1lab.tasks import careful_walk_e3 as M
      return M.a04_env_cfg(play=play, t0=t0)
    from r1lab.tasks import careful_walk_b0 as M
    return M.b0_env_cfg(play=play, t0=t0)
  def rl():
    from r1lab.tasks import careful_walk_e3x as M
    return M.col_rl_cfg() if arm == "col" else M.cap_rl_cfg()
  return dict(env=env, rl=rl, cmd_sign=(1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
              critic_layout=(("actor", 870), ("frame", 87), ("vec3", 3), ("scalar", 1), ("lr2", 2), ("lr2", 2), ("lr_vec3", 6)))
for _arm in ("cap", "capa04", "col"):
  TASKS[f"careful_walk_e3{_arm}"] = _cw_e3x(_arm)

# E4M careful_walk_e4m (positive 25, coordinator 2026-11-20 23:0x-23:2x): board-2 method arms on B0 - m1 objective
# structure, m2p gait/contact human prior, m4 actuator signal-dependent noise, mx = m1 + m2p + m4 + J-lim, jl = B0 + J-lim;
# suffix _cap = B0 + std cap 0.8 base, _jl = J-lim on (r1lab.tasks.careful_walk_e4m, imported lazily).
def _cw_e4m(arm, cap, jl):
  def env(play=False, t0=0):
    from r1lab.tasks import careful_walk_e4m as M
    return M.e4m_env_cfg(arm, play=play, t0=t0, jl=jl)
  def rl():
    from r1lab.tasks import careful_walk_e4m as M
    return M.e4m_rl_cfg(cap=cap)
  return dict(env=env, rl=rl, cmd_sign=(1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
              critic_layout=(("actor", 870), ("frame", 87), ("vec3", 3), ("scalar", 1), ("lr2", 2), ("lr2", 2), ("lr_vec3", 6)))
for _name, _arm, _cap, _jl in [('careful_walk_e4m1', 'm1', False, False), ('careful_walk_e4m1_jl', 'm1', False, True), ('careful_walk_e4m1_cap', 'm1', True, False), ('careful_walk_e4m1_cap_jl', 'm1', True, True), ('careful_walk_e4m2p', 'm2p', False, False), ('careful_walk_e4m2p_jl', 'm2p', False, True), ('careful_walk_e4m2p_cap', 'm2p', True, False), ('careful_walk_e4m2p_cap_jl', 'm2p', True, True), ('careful_walk_e4m4', 'm4', False, False), ('careful_walk_e4m4_jl', 'm4', False, True), ('careful_walk_e4m4_cap', 'm4', True, False), ('careful_walk_e4m4_cap_jl', 'm4', True, True), ('careful_walk_e4mx', 'mx', False, True), ('careful_walk_e4mx_cap', 'mx', True, True), ('careful_walk_e4jl', 'jl', False, True), ('careful_walk_e4jl_cap', 'jl', True, True)]:
  TASKS[_name] = _cw_e4m(_arm, _cap, _jl)

# E4N careful_walk_e4n (positive 25, coordinator 2026-11-20 23:5x): board-3 arms on B0 - m1m3 = M1 + no fall
# termination, m5 = input bottleneck (VIB), m6 = continuous rest success, m7 = hold share 0.4 / hold 4-10 s
# (r1lab.tasks.careful_walk_e4n, imported lazily).
def _cw_e4n(arm):
  def env(play=False, t0=0):
    from r1lab.tasks import careful_walk_e4n as M
    return M.e4n_env_cfg(arm, play=play, t0=t0)
  def rl():
    from r1lab.tasks import careful_walk_e4n as M
    return M.e4n_rl_cfg(arm)
  return dict(env=env, rl=rl, cmd_sign=(1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
              critic_layout=(("actor", 870), ("frame", 87), ("vec3", 3), ("scalar", 1), ("lr2", 2), ("lr2", 2), ("lr_vec3", 6)))
for _arm in ("m1m3", "m5", "m6", "m7"):
  TASKS[f"careful_walk_e4{_arm}"] = _cw_e4n(_arm)

if TASKS.keys() != TASK_SPECS.keys():
  raise RuntimeError("모든 학습 태스크를 r1lab.registry.TASK_SPECS에 등록해야 합니다.")
for name, spec in TASK_SPECS.items():
  TASKS[name].update(spec)
