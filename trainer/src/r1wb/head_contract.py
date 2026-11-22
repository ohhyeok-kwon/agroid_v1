"""Head26 semantic contract, keeping the legacy manufacturer contract unchanged.

Reference trajectories retain their measured/retargeted 24 joints. Head state and
actions expand to 26; no invented neutral-head tracking target is introduced.
The named PPO migration is implemented separately in head_resume.
"""
from __future__ import annotations
import numpy as np

from r1lab import contract as LEGACY

REFERENCE_JOINTS = LEGACY.ENTITY_JOINTS
HEAD_JOINTS = ('head_pitch_joint', 'head_yaw_joint')
STATE_JOINTS = REFERENCE_JOINTS + HEAD_JOINTS
LEG_JOINTS = STATE_JOINTS[:12]
WAIST_JOINTS = STATE_JOINTS[12:14]
ARM_JOINTS = STATE_JOINTS[14:24]
DEFAULT_POSE = {**LEGACY.DEFAULT_POS_BY_NAME, **dict.fromkeys(HEAD_JOINTS, 0.)}
HEAD_LIMITS = {'head_pitch_joint':dict(lower=-.62832, upper=.62832, effort=33., velocity=33.4),
               'head_yaw_joint':dict(lower=-2.0071, upper=2.0071, effort=.86, velocity=11.)}
MIRROR_PERM = tuple(STATE_JOINTS.index(n.replace('left','@').replace('right','left').replace('@','right')) for n in STATE_JOINTS)
MIRROR_SIGN = tuple(-1. if ('_roll_' in n or '_yaw_' in n) else 1. for n in STATE_JOINTS)


def actor_features(*, head=False, senses='', task=True):
    """Semantic feature order; all current reference fields stay 24-dimensional."""
    if senses not in ('', 'history', 'scan', 'both'):
        raise ValueError(senses)
    joints = STATE_JOINTS if head else REFERENCE_JOINTS
    keys=[]
    def add(prefix,names):keys.extend(prefix+'.'+str(n) for n in names)
    for prefix in ('reference.q','reference.dq'):add(prefix,REFERENCE_JOINTS)
    for i in range(4):
        add(f'preview{i}.q',REFERENCE_JOINTS)
        add(f'preview{i}.position',range(3));add(f'preview{i}.orientation6',range(6))
    add('anchor.position',range(3));add('anchor.orientation6',range(6))
    add('base.linear_velocity',range(3));add('base.angular_velocity',range(3))
    for prefix in ('body.q','body.dq','body.previous_action'):add(prefix,joints)
    add('reference.linear_velocity',range(3));add('reference.angular_velocity',range(3))
    if task:add('goal',range(18));add('lag',range(3))
    if senses in ('history','both'):
        for i in range(10):
            for prefix in ('q','dq','action'):add(f'history{i}.{prefix}',joints)
            add(f'history{i}.omega',range(3));add(f'history{i}.gravity',range(3))
    if senses in ('scan','both'):add('scan',range(117))
    return tuple(keys)


def feature_embedding(old, new):
    if len(set(old))!=len(old) or len(set(new))!=len(new):
        raise ValueError('Feature names must be unique')
    if not set(old).issubset(new):
        raise ValueError('Old feature lost or renamed')
    return np.asarray([new.index(n) for n in old],dtype=np.int64)


TRACKED_BODIES = ('pelvis','left_hip_roll_link','left_knee_link','left_ankle_roll_link',
                  'right_hip_roll_link','right_knee_link','right_ankle_roll_link','torso_link',
                  'left_shoulder_roll_link','left_elbow_link','left_wrist_roll_link',
                  'right_shoulder_roll_link','right_elbow_link','right_wrist_roll_link')


def critic_features(*, head=False, scan=False):
    """Current mixed teacher critic: reference24, tracked bodies14, arm intent10.

    Head state extends only q/dq/previous action, by six. Existing privileged
    tracked body pose terms stay 14 x (3+6); no fabricated head reference pose.
    """
    actor=list(actor_features(head=head,task=False))
    insert=actor.index('base.linear_velocity.0')
    body=[f'body_pose.position.{n}.{i}' for n in TRACKED_BODIES for i in range(3)]
    body += [f'body_pose.orientation6.{n}.{i}' for n in TRACKED_BODIES for i in range(6)]
    actor[insert:insert]=body
    actor += [f'goal.{i}' for i in range(18)]
    actor += [f'arm_intent.{n}' for n in ARM_JOINTS]
    actor += [f'lag.{i}' for i in range(3)]
    if scan:actor += [f'scan.{i}' for i in range(117)]
    return tuple(actor)


def expand_input_weights(weight, old, new):
    if weight.ndim!=2 or weight.shape[1]!=len(old):raise ValueError('Old input width differs')
    out=np.zeros((weight.shape[0],len(new)),dtype=weight.dtype)
    out[:,feature_embedding(old,new)]=weight
    return out


def expand_normalizer(value, old, new, *, fill):
    if value.shape[-1]!=len(old):raise ValueError('Old normalizer width differs')
    out=np.full(value.shape[:-1]+(len(new),),fill,dtype=value.dtype)
    out[...,feature_embedding(old,new)]=value
    return out


def expand_mean_head(weight,bias):
    """Preserves old 24 deterministic outputs; the new two means equal exactly 0."""
    if weight.ndim!=2 or weight.shape[0]!=24 or bias.shape!=(24,):
        raise ValueError('Expected old 24-output mean layer')
    return np.concatenate((weight,np.zeros((2,weight.shape[1]),dtype=weight.dtype))),np.concatenate((bias,np.zeros(2,dtype=bias.dtype)))


def pd_candidates():
    """Uncalibrated choices for review, not measured manufacturer head gains.

    Pitch reuses the existing 33 Nm group. Yaw scales that group's gain and
    armature by its URDF torque ratio, retaining the existing action-scale and
    nominal gain/armature convention. Physical load differs, so this does not
    preserve closed-loop bandwidth and must be validated before training.
    """
    group=LEGACY.GROUPS['small'];out={}
    for name in HEAD_JOINTS:
        ratio=HEAD_LIMITS[name]['effort']/group['effort']
        kp,kd,armature=(group[k]*ratio for k in ('kp','kd','armature'))
        out[name]=dict(kp=kp,kd=kd,armature=armature,effort=HEAD_LIMITS[name]['effort'],
                       action_scale=.25*HEAD_LIMITS[name]['effort']/kp,
                       frictionloss=.1,viscous_damping=.05,delay_physics_steps=[0,2],
                       delay_hold_prob=.9,calibrated=False)
    return out
