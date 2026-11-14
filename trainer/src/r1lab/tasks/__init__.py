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

if TASKS.keys() != TASK_SPECS.keys():
  raise RuntimeError("모든 학습 태스크를 r1lab.registry.TASK_SPECS에 등록해야 합니다.")
for name, spec in TASK_SPECS.items():
  TASKS[name].update(spec)
