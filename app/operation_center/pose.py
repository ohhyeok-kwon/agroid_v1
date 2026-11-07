"""Operation Center pose validation and URDF joint mapping; no robot transport."""
import math

# https://github.com/unitreerobotics/unitree_sdk2_python/blob/814556d15970dd2ecf1c9984e845ca02ab07e206/example/r1/low_level/r1_low_level_example.py
JOINTS = [
    'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
    'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
    'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
    'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
    'waist_roll_joint', 'waist_yaw_joint',
    'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
    'left_elbow_joint', 'left_wrist_roll_joint',
    'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
    'right_elbow_joint', 'right_wrist_roll_joint', 'head_pitch_joint', 'head_yaw_joint',
]
MOTOR_SLOTS = list(range(14)) + list(range(15, 20)) + list(range(22, 27)) + [29, 30]


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('관절·IMU 값에 유효하지 않은 숫자가 있습니다.')
    return float(value)


def normalize_pose(pose):
    tick = pose['tick']
    if isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick <= 2**53 - 1:
        raise ValueError('상태 메시지의 시각 값이 잘못되었습니다.')
    names = set(pose['joints'])
    if not names or not names <= set(JOINTS):
        raise ValueError('R1 관절 이름이 아니거나 수신한 관절이 없습니다.')
    joints = {name: number(pose['joints'][name]) for name in JOINTS if name in names}
    # R1 omits joints without a valid measurement; the view keeps their last shown angle.
    missing = [name for name in JOINTS if name not in names]
    if pose['quaternion'] is None and pose['rpy'] is None:
        transforms = {}
        for child, value in pose.get('transforms', {}).items():
            xyz = [number(v) for v in value['translation']]
            rotation = [number(v) for v in value['rotation']]
            if len(xyz) != 3 or len(rotation) != 4 or abs(sum(v*v for v in rotation)-1) > 1e-4:
                raise ValueError('TF 좌표 형식이 올바르지 않습니다.')
            transforms[child] = {'translation': xyz, 'rotation': rotation}
        return {'tick': tick, 'joints': joints, 'missing': missing, 'quaternion': None, 'rpy': None,
                'transforms': transforms, 'frame': 'pelvis_link'}
    if missing:
        raise ValueError('R1의 26개 관절이 모두 필요합니다.')
    quaternion = [number(v) for v in pose['quaternion']]
    rpy = [number(v) for v in pose['rpy']]
    if len(quaternion) != 4 or len(rpy) != 3:
        raise ValueError('IMU 벡터 길이가 잘못되었습니다.')
    norm = math.sqrt(sum(v * v for v in quaternion))
    if norm < 1e-6:
        raise ValueError('IMU 자세가 아직 유효하지 않습니다.')
    return {'tick': tick, 'joints': joints, 'missing': [], 'quaternion': [v / norm for v in quaternion], 'rpy': rpy}


def normalize(sample):
    if sample.get('mode_pr') != 0:
        raise ValueError('병렬 구동 모드의 발목 각도는 표시할 수 없습니다. PR 자세 수신을 기다립니다.')
    tick = sample['tick']
    if isinstance(tick, bool) or not isinstance(tick, int) or not 0 <= tick <= 0xffffffff:
        raise ValueError('상태 메시지의 시각 값이 잘못되었습니다.')
    return normalize_pose({
        'tick': tick,
        'joints': {name: sample['motor_state'][index]['q'] for name, index in zip(JOINTS, MOTOR_SLOTS)},
        'quaternion': sample['imu_state']['quaternion'], 'rpy': sample['imu_state']['rpy'],
    })


def quaternion_rpy(quaternion):
    w, x, y, z = quaternion
    return [math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
            math.asin(max(-1, min(1, 2 * (w*y - z*x)))),
            math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))]
