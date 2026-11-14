"""D6: PPO fine-tune of the D5 student in the D5 scene (r1lab.student_rl).

Single change against D5: the learning rule. The scene is D5's (I13 teacher scene in play mode, student command
with heading hold, teacher pushes and 1 kg balls during training, library v5 command mix); its rewards are the
I13 teacher's own (body/arm tracking, stance slip, landing speed, sway excess, anchor height, falls). Only the
teacher run's privileged critic group is added back so the teacher's critic weights fit. The deployment contract
(810 inputs, command processing, heading hold) is STUDENT_CONTRACT_LATERAL unchanged.
"""
from __future__ import annotations

import copy

from r1lab.tasks.matching_student import STUDENT_CONTRACT_LATERAL, matching_student_crisis_lateral_env_cfg
from r1lab.tasks.matching_teacher import matching_lateral_teacher_env_cfg

STUDENT_CONTRACT_LATERAL_RL = copy.deepcopy(STUDENT_CONTRACT_LATERAL)
STUDENT_CONTRACT_LATERAL_RL["fine_tune"] = (
    "PPO on the teacher task's rewards from the completed D5 student, anchored by bc_coef * MSE to the frozen "
    "teacher's mean action on the student's own rollout states; actor normalizer frozen as distilled, fixed "
    "acting std 0.1, teacher-run critic warmed up first (r1lab.student_rl)")


def student_rl_lateral_env_cfg(play=False, t0=0, teacher_env=matching_lateral_teacher_env_cfg):
    """D5 scene + the teacher's privileged critic group (training only; play scenes are D5's exactly).

    teacher_env stays a parameter: train.py records its default name for the results service."""
    cfg = matching_student_crisis_lateral_env_cfg(play=play, t0=t0, teacher_env=teacher_env)
    if not play:
        critic = teacher_env(play=False).observations.get("critic")
        if critic is None:
            raise ValueError("The teacher scene has no critic observation group")
        cfg.observations["critic"] = critic
    return cfg


def student_rl_lateral_rl_cfg():
    from r1lab.student_rl import StudentRLRunnerCfg
    from r1lab.tasks.matching_teacher import COMMAND_MIX_V5
    cfg = StudentRLRunnerCfg()
    cfg.algorithm.command_kinds = tuple(m[1] for m in COMMAND_MIX_V5)
    return cfg


# ----------------------------------------------------------------------------- D7T: terrain
STUDENT_CONTRACT_TERRAIN = copy.deepcopy(STUDENT_CONTRACT_LATERAL)
STUDENT_CONTRACT_TERRAIN["fine_tune"] = (
    "PPO on the careful_walk terrain task (rough, debris boxes up to 8.5 cm, slopes up to 0.25, stairs up to 9 cm, "
    "pushes and hits, physics randomization, sensor noise) from the completed D5 student, anchored by "
    "bc_coef * MSE to the frozen D5 student itself (no teacher on terrain); actor normalizer frozen, fixed std 0.1")
# careful_walk curricula (command stages, penalty and disturbance ramps) are complete from this step on.
TERRAIN_T0 = 3000
TERRAIN_COMMAND = {"lin_vel_x": (-0.25, 0.30), "lin_vel_y": (-0.28, 0.28), "ang_vel_z": (-1.0, 1.0)}


def student_terrain_env_cfg(play=False, t0=0):
    """D7T scene: careful_walk_env_cfg(hard_terrain, no AMP) with the student contract's command ranges.

    Single change against D5 for the learner: the world (terrain, its rewards and disturbances). The actor input is
    the same 810 frame stack (common.actor_obs on the "twist" command); play scenes keep the terrain."""
    del t0
    from r1lab.tasks.careful_walk import careful_walk_env_cfg
    cfg = careful_walk_env_cfg(play=play, flat=False, use_amp=False, stage=2, t0=TERRAIN_T0, hard_terrain=True,
                               scan=False)
    ranges = cfg.commands["twist"].ranges
    for key, value in TERRAIN_COMMAND.items():
        setattr(ranges, key, value)
    cfg.curriculum.pop("command_vel", None)
    return cfg


def student_terrain_rl_cfg():
    from r1lab.student_rl import StudentRLRunnerCfg, student_actor_cfg, terrain_critic_cfg
    cfg = StudentRLRunnerCfg()
    cfg.teacher = student_actor_cfg()  # the frozen initial student is the anchor
    cfg.critic = terrain_critic_cfg()
    cfg.obs_groups = {"actor": ("actor",), "critic": ("critic",), "teacher": ("actor",)}
    cfg.algorithm.bc_coef = 0.2
    cfg.algorithm.learning_rate = 5e-5
    cfg.algorithm.actor_warmup_updates = 40
    cfg.algorithm.command_kinds = ()
    cfg.max_iterations = 400
    return cfg


# ----------------------------------------------------------------------------- D7T-b: obstacle-dense terrain
def dense_terrain():
    """Obstacle-dense tiles at fixed random difficulty (no curriculum): the survival curriculum promoted robots
    that stood on the 1-2.5 m start platforms (D7T levels 4->9 without crossing obstacles). Small platforms,
    steps .02-.07 m on .3 m treads (evaluation stairs: 4.6 cm on 28 cm), boxes up to 8.5 cm (rocks 4-9 cm)."""
    from mjlab.terrains import config as tcfg
    from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
    return TerrainGeneratorCfg(
        size=(8.0, 8.0), border_width=10.0, num_rows=10, num_cols=12, curriculum=False, difficulty_range=(0.3, 1.0),
        sub_terrains={
            "rough": tcfg.random_rough(proportion=0.15, noise_range=(0.005, 0.04), noise_step=0.005),
            "debris": tcfg.random_spread_boxes(proportion=0.35, num_boxes=90, box_width_range=(0.06, 0.35),
                                               box_length_range=(0.06, 0.5), box_height_range=(0.015, 0.085),
                                               platform_width=0.6),
            "slope": tcfg.hf_pyramid_slope(proportion=0.075, slope_range=(0.0, 0.25)),
            "slope_inv": tcfg.hf_pyramid_slope_inv(proportion=0.075, slope_range=(0.0, 0.25)),
            "steps": tcfg.pyramid_stairs(proportion=0.2, step_height_range=(0.02, 0.07), step_width=0.3,
                                         platform_width=1.0),
            "steps_inv": tcfg.pyramid_stairs_inv(proportion=0.15, step_height_range=(0.02, 0.07), step_width=0.3,
                                                 platform_width=1.0),
        },
        add_lights=True,
    )


STUDENT_CONTRACT_TERRAIN_DENSE = copy.deepcopy(STUDENT_CONTRACT_LATERAL)
STUDENT_CONTRACT_TERRAIN_DENSE["fine_tune"] = (
    "PPO on obstacle-dense careful_walk terrain (fixed random difficulty .3-1, steps .02-.07 m, boxes to 8.5 cm, "
    "slopes to .25; moving commands only; foot stumble penalty) from the completed D5 student, anchored to the "
    "frozen D5 itself (bc .05); fixed acting std .15")


def student_terrain_dense_env_cfg(play=False, t0=0):
    """D7T-b: D7T scene with obstacle-dense terrain at fixed difficulty, moving commands and a stumble penalty.

    Changes against D7T (one concept: meet obstacles every episode): terrain layout/curriculum, command mix
    (no standing envs, few pure turns), foot-against-obstacle penalty -1."""
    del t0
    from r1lab.tasks.careful_walk import careful_walk_env_cfg
    cfg = careful_walk_env_cfg(play=play, flat=False, use_amp=False, stage=2, t0=TERRAIN_T0, hard_terrain=True,
                               scan=False, stumble=-1.0)
    cfg.scene.terrain.terrain_generator = dense_terrain()
    cfg.scene.terrain.max_init_terrain_level = None
    command = cfg.commands["twist"]
    for key, value in TERRAIN_COMMAND.items():
        setattr(command.ranges, key, value)
    command.rel_standing_envs = 0.0
    command.mode_probs = (0.2, 0.6, 0.05, 0.15)
    cfg.curriculum.pop("command_vel", None)
    cfg.curriculum.pop("terrain_levels", None)
    return cfg


def student_terrain_dense_rl_cfg():
    cfg = student_terrain_rl_cfg()
    cfg.algorithm.bc_coef = 0.05
    cfg.algorithm.learning_rate = 1.5e-4
    cfg.algorithm.clip_param = 0.2
    cfg.algorithm.acting_std = 0.15
    cfg.max_iterations = 800
    return cfg
