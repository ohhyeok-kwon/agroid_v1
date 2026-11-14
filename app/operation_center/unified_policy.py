"""통합 정책 S1(실험): 전신 학습 계열(r1wb)의 System 1 정책 ONNX를 Simulation 물리에서 실행한다.

정책 하나가 관측 이력과 목표만 보고 24관절 목표를 낸다(참조 동작·감지기·전환 규칙 없음). 학습 환경
(r1wb.student)과 같은 계약을 이 MuJoCo 로봇에 그대로 연결한다:
  관측 1198 = 15프레임 이력 x [골반 IMU 각속도 3, 골반 기준 중력 방향 3, 관절각 - 기본자세 24, 관절속도 24,
              직전 행동 24] (항목별로 오래된 것부터 이어 붙임) + 목표 18 + 팔 의도 10
  목표 18   = [전후·좌우 속도, 회전 속도, 골반 높이 x 사용여부, 숙임 x 사용여부, 손 목표 10(미사용 0),
              사용여부 3(속도, 높이, 숙임)]
  행동 24   = 다리·허리: 기본자세 + 행동 x 스케일, 팔: 팔 의도 + 0.5 rad x clip(행동, -1, 1)
명령은 학습의 명령 스트림처럼 가속 한계를 두고 따라간다. R1 배포 정책과는 별개이고 Simulation 전용이다.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from r1lab import contract as C
from r1wb import robot
from r1wb.student.intent import ArmIntent

MANIFEST = '.agent/operation_center/unified_policy.json'
MANIFEST_KEYS = {'version', 'label', 'run_id', 'iteration', 'path', 'sha256', 'selected_at', 'note', 'lateral'}
HISTORY = 15
CONTROL_DT = 0.02
TERMS = (('gyro', 3), ('gravity', 3), ('joint_pos', 24), ('joint_vel', 24), ('action', 24))
INPUT_DIMENSION = HISTORY * sum(width for _, width in TERMS) + 18 + 10
ARM_LIMIT = 0.5                                   # rad, r1wb.student.actions.IntentResidualActionCfg.residual_limit
SLEW = np.array([1.0, 1.0, 2.0]) * CONTROL_DT     # 명령 스트림의 가속 한계(m/s^2, m/s^2, rad/s^2) x 제어 주기
FORWARD_SPEED, REVERSE_SPEED, LATERAL_SPEED, YAW_RATE = 0.4, 0.25, 0.2, 0.8
DEFAULT_HEIGHT = 0.731


def lateral_ignored(axes, lateral=False):
    """A/D만 누른 순수 옆걸음을 배우지 않은 정책(선택 파일에 lateral 표시가 없음)에서는 그 입력을 무시한다
    (모션 자료 공백: 그런 정책은 왼쪽 0.06 m/s, 오른쪽은 기울기만 함)."""
    return not lateral and abs(float(axes[1])) > 1e-6 and abs(float(axes[0])) < 1e-6


def command_from_axes(axes, lateral=False):
    """W/S -> 전후 속도, A/D -> 좌우 속도(m/s). lateral=False면 좌우는 전후 입력과 함께일 때만 쓴다."""
    forward, sideways = float(axes[0]), float(axes[1])
    if lateral_ignored(axes, lateral):
        sideways = 0.0
    return np.array([forward * (FORWARD_SPEED if forward >= 0 else REVERSE_SPEED), sideways * LATERAL_SPEED, 0.0])


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


class Selection:
    def __init__(self, session, info):
        self.session, self.info = session, info

    def controller(self, model):
        return UnifiedPolicyController(model, self)


def load_selection(project, manifest=None):
    """선택 파일(.agent/operation_center/unified_policy.json)의 체크포인트를 해시로 확인하고 연다."""
    project = Path(project).resolve()
    path = project / (manifest or MANIFEST)
    if not path.exists():
        raise ValueError('통합 정책 선택 파일이 없습니다(.agent/operation_center/unified_policy.json).')
    encoded = path.read_bytes()
    entry = json.loads(encoded)
    if not isinstance(entry, dict) or set(entry) - MANIFEST_KEYS or entry.get('version') != 1:
        raise ValueError('통합 정책 선택 파일의 형식이 올바르지 않습니다.')
    relative = Path(entry['path'])
    policy = (project / relative).resolve()
    runs = (project / '.agent/trainer/runs').resolve()
    if (relative.is_absolute() or not policy.is_relative_to(runs) or policy.parent.name != entry['run_id']
            or policy.name != f"policy_{entry['iteration']}.onnx"):
        raise ValueError('통합 정책: 등록된 학습의 체크포인트 경로가 아닙니다.')
    model = policy.read_bytes()
    if _sha256(model) != entry['sha256']:
        raise ValueError('통합 정책: 정책 파일 해시가 선택한 체크포인트와 다릅니다.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(model, options, providers=['CPUExecutionProvider'])
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if (len(inputs) != 1 or inputs[0].name != 'obs' or inputs[0].shape != [1, INPUT_DIMENSION]
            or len(outputs) != 1 or outputs[0].name != 'actions' or outputs[0].shape != [1, 24]):
        raise ValueError('통합 정책: 관측·관절 출력 계약(1198 -> 24)이 일치하지 않습니다.')
    info = {'label': entry.get('label', f"통합 정책 · {entry['iteration']}"), 'run_id': entry['run_id'],
            'iteration': entry['iteration'], 'sha256': entry['sha256'], 'path': str(relative),
            'input_dimension': INPUT_DIMENSION, 'selection_sha256': _sha256(encoded),
            'lateral': bool(entry.get('lateral', False)),
            'source': 'unified_policy_manifest', 'note': entry.get('note', '')}
    return Selection(session, info)


class UnifiedPolicyController:
    """50Hz마다 step()으로 1198관측 -> ONNX -> ENTITY 순서 관절 목표와 kp·kd를 돌려준다."""

    def command_from_axes(self, axes):
        return command_from_axes(axes, self.info['lateral'])

    def lateral_ignored(self, axes):
        return lateral_ignored(axes, self.info['lateral'])

    def __init__(self, model, selection):
        self.selection, self.session, self.info = selection, selection.session, selection.info
        joints = list(robot.JOINTS)                                   # ENTITY 순서: 다리 12, 허리 2, 팔 10
        self.q_index = np.array([model.jnt_qposadr[model.joint(name).id] for name in joints])
        self.v_index = np.array([model.jnt_dofadr[model.joint(name).id] for name in joints])
        self.ang_sensor = slice(model.sensor('imu_ang_vel').adr[0], model.sensor('imu_ang_vel').adr[0] + 3)
        self.default = np.array([robot.DEFAULT_POSE[name] for name in joints])
        self.scale = np.array([robot.ACTION_SCALE[name] for name in joints])
        order = np.array(C.ACTION_IDX)
        self.kp, self.kd = np.array(C.KP)[order], np.array(C.KD)[order]
        self.arm = np.array([joints.index(name) for name in robot.ARM_JOINTS])
        self.intent = ArmIntent(1, 'cpu', CONTROL_DT)
        self.history = None
        self.last_action = np.zeros(24, dtype=np.float32)
        self.twist = np.zeros(3)
        self.height, self.height_on, self.pitch, self.pitch_on = DEFAULT_HEIGHT, 0.0, 0.0, 0.0
        self.arm_intent = self.intent.rest.numpy().copy()
        self.steps = 0

    # ------------------------------------------------------------ pieces of the contract (tested against mjlab)
    @staticmethod
    def projected_gravity(quaternion):
        """세계 -z를 골반 좌표로 옮긴 것(w, x, y, z 사원수)."""
        w, x, y, z = quaternion
        return np.array([-2 * (x * z - w * y), -2 * (y * z + w * x), -(1 - 2 * (x * x + y * y))])

    def goal(self):
        hands = np.zeros(10)
        return np.concatenate([self.twist, [self.height * self.height_on, self.pitch * self.pitch_on], hands,
                               [1.0, self.height_on, self.pitch_on]])

    def observation(self, gyro, gravity, joint_pos_rel, joint_vel):
        """새 프레임을 이력에 넣고 1198 관측을 만든다. 첫 호출에서는 이력을 첫 프레임으로 채운다."""
        frame = {'gyro': gyro, 'gravity': gravity, 'joint_pos': joint_pos_rel, 'joint_vel': joint_vel,
                 'action': self.last_action}
        if self.history is None:
            self.history = {name: np.repeat(np.asarray(frame[name], dtype=np.float32)[None], HISTORY, axis=0)
                            for name, _ in TERMS}
        else:
            for name, _ in TERMS:
                self.history[name] = np.concatenate([self.history[name][1:], np.asarray(frame[name], dtype=np.float32)[None]])
        parts = [self.history[name].reshape(-1) for name, _ in TERMS]
        return np.concatenate(parts + [self.goal(), self.arm_intent]).astype(np.float32)

    def targets(self, action):
        """행동 -> ENTITY 순서 관절 목표(rad)."""
        target = self.default + self.scale * action
        target[self.arm] = self.arm_intent + ARM_LIMIT * np.clip(action[self.arm], -1.0, 1.0)
        return target

    def update_intent(self, joint_pos):
        speed = torch.tensor([float(np.hypot(self.twist[0], self.twist[1]))])
        self.arm_intent = self.intent.step(torch.as_tensor(joint_pos, dtype=torch.float32)[None], speed)[0].numpy()

    # ------------------------------------------------------------ control
    def reset(self, model, data):
        joint_pos = data.qpos[self.q_index]
        self.intent.reset(torch.arange(1), torch.as_tensor(joint_pos, dtype=torch.float32)[None])
        self.arm_intent = self.intent.rest.numpy().copy()
        self.history = None
        self.last_action = np.zeros(24, dtype=np.float32)
        self.twist = np.zeros(3)
        self.steps = 0

    def step(self, model, data, command, heading_target=None):
        """command: 목표 (전후 m/s, 좌우 m/s, 회전 rad/s). 가속 한계를 두고 따라간다."""
        self.twist += np.clip(np.asarray(command, dtype=float) - self.twist, -SLEW, SLEW)
        joint_pos = data.qpos[self.q_index]
        self.update_intent(joint_pos)
        observation = self.observation(data.sensordata[self.ang_sensor], self.projected_gravity(data.qpos[3:7]),
                                       joint_pos - self.default, data.qvel[self.v_index])
        action = self.session.run(['actions'], {'obs': observation[None]})[0][0]
        self.last_action = action.astype(np.float32)
        self.steps += 1
        return self.targets(action.astype(float)), self.kp, self.kd

    def diagnostics(self):
        return {'twist': [round(float(v), 3) for v in self.twist], 'steps': self.steps}
