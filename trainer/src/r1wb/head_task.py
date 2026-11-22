"""Stage H: one 26-joint teacher, with an unchanged 24-joint human reference.

v12h0 and v12h share the articulated head, sensors, reset and dynamics. Only
the three bounded perception reward weights differ. No gaze action override.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import copy
from pathlib import Path
import torch
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ObjRef, RayCastSensorCfg
from r1wb import robot, recovery_task, teacher_task, head_contract as C
from r1wb.student.stream import MixedGoalCommand, MixedGoalCommandCfg
from r1wb import head_rewards as gaze

HEAD_XML = robot.R1_XML.with_name('r1_head26.xml')
CONTRACT_VERSION = 'r1wb-head26-reference24-scan-v1'


def get_spec(hand_mass=robot.HAND_MASS):
    spec=mujoco.MjSpec.from_file(str(HEAD_XML))
    for side in ('left','right'):
        hand=spec.body(side+'_wrist_roll_link').add_body(name=side+'_o6_hand',pos=list(robot.HAND_POS))
        hand.add_geom(name=side+'_o6_collision',type=mujoco.mjtGeom.mjGEOM_BOX,size=list(robot.HAND_HALF),
                      mass=hand_mass,rgba=[.2,.2,.6,.5],group=3,condim=1)
    return spec


def head_actuators(delay_steps):
    return tuple(BuiltinPositionActuatorCfg(target_names_expr=(name,),stiffness=g['kp'],damping=g['kd'],
                    effort_limit=g['effort'],armature=g['armature'],frictionloss=g['frictionloss'],
                    viscous_damping=g['viscous_damping'],delay_min_lag=0,delay_max_lag=delay_steps,
                    delay_hold_prob=.9 if delay_steps else 0.) for name,g in C.pd_candidates().items())


class HeadMixedGoalCommand(MixedGoalCommand):
    """Reference comparisons use old 24; reset writes all 26 exactly once."""
    def __init__(self,cfg,env):
        super().__init__(cfg,env)
        if tuple(self.robot.joint_names)!=C.STATE_JOINTS:
            raise ValueError('Head26 entity order differs from the frozen action/reference contract')
        if self.motion.joint_pos.shape[1]!=24 or self.motion.joint_vel.shape[1]!=24:
            raise ValueError('Stage H requires the unchanged human reference24 library')

    @property
    def robot_joint_pos(self):return self.robot.data.joint_pos[:,:24]

    @property
    def robot_joint_vel(self):return self.robot.data.joint_vel[:,:24]

    def _write_reference_state_to_sim(self,env_ids,root_pos,root_ori,root_lin_vel,root_ang_vel,joint_pos,joint_vel):
        if joint_pos.shape[-1]!=24 or joint_vel.shape[-1]!=24:
            raise ValueError('Reset reference width must remain 24')
        # Reset-only physical initial condition; never freezes head actions.
        q=self.robot.data.default_joint_pos[env_ids].clone()
        dq=self.robot.data.default_joint_vel[env_ids].clone()
        q[:,:24],dq[:,:24]=joint_pos,joint_vel
        super()._write_reference_state_to_sim(env_ids,root_pos,root_ori,root_lin_vel,root_ang_vel,q,dq)


@dataclass(kw_only=True)
class HeadMixedGoalCommandCfg(MixedGoalCommandCfg):
    def build(self,env):return HeadMixedGoalCommand(self,env)


def body_action_rate_l2(env):
    delta=env.action_manager.action[:,:24]-env.action_manager.prev_action[:,:24]
    return delta.square().sum(dim=1)


def body_joint_motion_l1(env):
    return env.scene['robot'].data.joint_vel[:,:24].abs().mean(dim=1)


def contract_record(perception,base_teacher='v10s'):
    if base_teacher not in ('v10s','v11a','v11c'):
        raise ValueError('Head26 supports only v10s, v11a and explicit body24-preserving v11c')
    return dict(version=CONTRACT_VERSION,reference_joints=24,actions=26,actor=417,critic=553,
                body_teacher_base=base_teacher,
                task='v12h' if perception else 'v12h0',perception_weights=gaze.WEIGHTS if perception else dict.fromkeys(gaze.WEIGHTS,0.),
                pd=C.pd_candidates(),pd_calibrated=False,head_reset='zero physical q/dq only at episode reset',
                body_reward_normalization='24 original joints',
                body_consistency=('original24 Nm^2 mean; original411 features and63 draws; head auxiliary noise0'
                                  if base_teacher=='v11c' else None),
                torso_dr='same numerical ranges on residual torso; not exact old aggregate spatial-inertia DR',
                optics=gaze.OPTICS,ray_grid=gaze.RAY_GRID,paired_seeds='TBD before the paired H experiment',
                prior_prediction=dict(camera_angular_rms_reduction=.30,coverage='increase standing and walking',
                    fall_task_W='compare under identical settings; seed spread not yet measured'),
                ray_timing='static world hits sampled at preceding sense; current camera; centre shift logged; control dt .02 s',
                deployment_verified=False,physical_velocity_limit_is_metadata_only=True)


def env_cfg(perception=True,play=False,*,base_teacher='v10s',**kwargs):
    factories={'v10s':recovery_task.height_scan_env_cfg,'v11a':recovery_task.upper_body_scan_env_cfg,
               'v11c':recovery_task.noise_consistency_scan_env_cfg}
    if base_teacher not in factories:
        raise ValueError('Unknown Stage-H body teacher')
    cfg=factories[base_teacher]('control',play,**kwargs)
    entity=cfg.scene.entities['robot']
    entity.spec_fn=get_spec
    entity.init_state.joint_pos={**entity.init_state.joint_pos,**dict.fromkeys(C.HEAD_JOINTS,0.)}
    entity.init_state.joint_vel={'.*':0.}
    delay=max(a.delay_max_lag for a in entity.articulation.actuators)
    entity.articulation.actuators=tuple(entity.articulation.actuators)+head_actuators(delay)
    # Existing action targets resolve in natural entity order, now 24+pitch+yaw.
    cfg.actions['joint_pos'].scale={**robot.ACTION_SCALE,**{n:g['action_scale'] for n,g in C.pd_candidates().items()}}
    if base_teacher=='v11c':
        from r1wb import noise_consistency as N, head_noise_consistency as HN
        cfg.observations[N.GAIN].terms['gain'].func=HN.BodyPdGainScale
    old=cfg.commands['motion']
    cfg.commands['motion']=HeadMixedGoalCommandCfg(**{f.name:getattr(old,f.name) for f in fields(old)})
    # The stock ghost accepts a reference for every actual joint. No fabricated
    # head reference is supplied: reference24 ghost visualization is disabled.
    cfg.commands['motion'].debug_vis=False
    for sensor in cfg.scene.sensors:
        if sensor.name=='trunk_ground':
            sensor.primary.pattern=r'^(pelvis|torso_link|waist_roll_link|head_pitch_link|head_yaw_link)$'
    cfg.scene.sensors=tuple(cfg.scene.sensors)+(RayCastSensorCfg(name='head_gaze_ground',
        frame=ObjRef(type='body',name='pelvis',entity='robot'),pattern=gaze.GazeGridPatternCfg(),
        ray_alignment='yaw',max_distance=4.,exclude_parent_body=True,include_geom_groups=(0,),debug_vis=False),)
    # Keep all existing body costs and their normalisation. Head effort is separate.
    cfg.rewards['action_rate_l2'].func=body_action_rate_l2
    if 'joint_motion' in cfg.rewards:cfg.rewards['joint_motion'].func=body_joint_motion_l1
    cfg.rewards['joint_limit'].params['asset_cfg']=SceneEntityCfg('robot',joint_names=C.REFERENCE_JOINTS)
    # Same-head baseline also pays ordinary physical joint-limit cost; no target rule.
    cfg.rewards['head_joint_limit']=copy.deepcopy(cfg.rewards['joint_limit'])
    cfg.rewards['head_joint_limit'].params['asset_cfg']=SceneEntityCfg('robot',joint_names=C.HEAD_JOINTS)
    for name,func in (('head_view',gaze.view_fraction),('head_camera_motion',gaze.camera_motion_cost),('head_effort',gaze.head_effort_cost)):
        cfg.rewards[name]=RewardTermCfg(func=func,weight=gaze.WEIGHTS[name] if perception else 0.)
    from r1wb.reward_contract import require_no_hit_exemptions
    require_no_hit_exemptions(cfg)
    return cfg


def rl_cfg(base_teacher='v10s'):
    if base_teacher=='v11c':
        from r1wb.head_noise_consistency import rl_cfg as consistency_rl_cfg
        return consistency_rl_cfg()
    if base_teacher not in ('v10s','v11a'):raise ValueError('Unknown Stage-H body teacher')
    return teacher_task.teacher_task_rl_cfg(symmetric=True)


def validate_env(env):
    entity=env.scene['robot'];action=env.action_manager.get_term('joint_pos')
    if tuple(entity.joint_names)!=C.STATE_JOINTS or tuple(action.target_names)!=C.STATE_JOINTS:
        raise ValueError('Head26 physical/action ordering mismatch')
    term=env.command_manager.get_term('motion')
    if not isinstance(term,HeadMixedGoalCommand) or term.joint_pos.shape[-1]!=24:
        raise ValueError('Head26 reference adapter is missing')
    if env.observation_manager.group_obs_dim['actor']!=(417,) or env.observation_manager.group_obs_dim['critic']!=(553,):
        raise ValueError('Head26 expected actor417/critic553')
    from r1wb.noise_consistency import CLEAN, GAIN
    dims=env.observation_manager.group_obs_dim
    if CLEAN in dims or GAIN in dims:
        if dims.get(CLEAN)!=(417,) or dims.get(GAIN)!=(24,):
            raise ValueError('Head v11c auxiliary observations must be clean417/gain24')
    return dict(passed=True,version=CONTRACT_VERSION,actions=26,reference=24,actor=417,critic=553)
