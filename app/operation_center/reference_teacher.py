"""사람 모션 교사(실험): 모션 매칭 참조를 따라가는 233입력 교사 ONNX를 Simulation 물리에서 실행한다.

학습(careful_walk_teacher_matching)과 같은 r1lab 구현을 그대로 쓴다. 참조는
MotionMatcher(배치 1, MatchingMotionCommandCfg 기본값)가 사용자 명령과 로봇 상태로
만들고, 명령 변환·로봇 특징·관측 항목은 MatchingMotionCommand와 학습 관측 함수를 이
MuJoCo 로봇에 연결해 호출한다. 재고정은 평가 기준(MotionMatcher 기본 0.3m/0.3rad)이며
학습 전용 완화 값은 쓰지 않는다. 위상 입력은 0이고 R1 배포 정책과는 별개다.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime as ort
import torch

from mjlab.tasks.tracking.mdp import observations as tracking
from r1lab import contract as C
from r1lab.mdp import matching_command as matching
from r1lab.mdp import motion_matching as mm
from r1lab.mdp import motion_teacher as teacher
from r1lab.tasks.motion_teacher import OBSERVATION_CONTRACT

MANIFEST = '.agent/operation_center/reference_teacher.json'
MANIFEST_KEYS = {'version', 'label', 'run_id', 'iteration', 'path', 'sha256', 'library', 'selected_at', 'note'}
ONNX_CONTRACT = {'r1lab.observation': 'reference-teacher-v1', 'r1lab.input_dimension': '233',
                 'r1lab.phase': 'zero', 'r1lab.reference_required': 'true',
                 'r1lab.deployment': 'simulation-only'}
LIBRARY_KEYS = ('clip_names', 'segment_start', 'segment_end', 'segment_clip', 'segment_heldout', 'frame_segment')
# W/S 속도(m/s). 학습 명령 분포: 전진 0.15~0.30, 후진 -0.25~-0.17(라이브러리 v4: 사람 후진 자료가 0.2m/s대이고
# 0.16m/s 미만은 출발 프레임이 없어 반쯤 출발한 참조가 된다; v3 교사는 후진 출발 자료 없음), 옆걸음 없음.
FORWARD_SPEED, REVERSE_SPEED = 0.25, 0.2
CONTROL_DT = 1.0 / teacher.REFERENCE_FPS
TERMS = tuple((name, int(width)) for name, width in
              (term.split(':') for term in OBSERVATION_CONTRACT['actor_terms']))


def _defaults():
    """MatchingMotionCommandCfg 기본값(매처·명령 설정). tasks/matching_teacher.py의 학습 전용 덮어쓰기는 제외한다."""
    fields = {field.name: field for field in dataclasses.fields(matching.MatchingMotionCommandCfg)}
    def value(name):
        field = fields[name]
        return field.default_factory() if field.default_factory is not dataclasses.MISSING else field.default
    return SimpleNamespace(matcher=value('matcher'), yaw_rate_filter_s=value('yaw_rate_filter_s'),
                           heading_gain=value('heading_gain'), heading_rate_limit=value('heading_rate_limit'))


def command_from_axes(axes):
    """W/S → 전후 속도(m/s). A/D 요청은 그대로 담고 제어기가 0으로 바꿔 무시 여부를 표시한다."""
    forward, lateral = float(axes[0]), float(axes[1])
    return np.array([forward * (FORWARD_SPEED if forward >= 0 else REVERSE_SPEED), lateral * FORWARD_SPEED, 0.0])


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _inside(project, relative, folder, label):
    relative = Path(relative)
    path = (project / relative).resolve()
    if relative.is_absolute() or not path.is_relative_to((project / folder).resolve()):
        raise ValueError(f'{label} 경로는 {folder} 아래의 상대 경로여야 합니다.')
    return path


class TeacherSelection:
    """검증된 교사 ONNX·매칭 라이브러리. controller(model)로 실행기를 만든다."""

    def __init__(self, session, info, library):
        self.session, self.info, self.library = session, info, library

    def controller(self, model):
        return ReferenceTeacherController(model, self)


def load_selection(project, manifest=None):
    """선택 파일과 ONNX·라이브러리 해시·계약을 검증한다. 실패 시 대체 모델 없이 중단한다."""
    project = Path(project).resolve()
    path = Path(manifest) if manifest is not None else project / MANIFEST
    try:
        encoded = path.read_bytes()
        document = json.loads(encoded)
        if not isinstance(document, dict) or document.get('version') != 1 or set(document) - MANIFEST_KEYS:
            raise ValueError('선택 파일 형식이 올바르지 않습니다.')
        run_id, iteration = document['run_id'], document['iteration']
        if not isinstance(run_id, str) or type(iteration) is not int or iteration < 0:
            raise ValueError('학습 ID·반복 번호가 올바르지 않습니다.')
        policy = _inside(project, document['path'], '.agent/trainer/runs', '교사 정책')
        if policy.parent.name != run_id or policy.name != f'policy_{iteration}.onnx':
            raise ValueError('선택한 학습 ID·반복의 체크포인트가 아닙니다.')
        model_bytes = policy.read_bytes()
        digest = _sha256(model_bytes)
        if digest != document['sha256']:
            raise ValueError('교사 ONNX 해시가 선택 파일과 다릅니다.')
        library = document['library']
        if not isinstance(library, dict) or set(library) != {'path', 'sha256'}:
            raise ValueError('매칭 라이브러리 항목이 올바르지 않습니다.')
        library_path = _inside(project, library['path'], '.agent/trainer/data/tracking', '매칭 라이브러리')
        if library_path.suffix != '.npz' or _sha256(library_path.read_bytes()) != library['sha256']:
            raise ValueError('매칭 라이브러리 해시가 선택 파일과 다릅니다.')
        teacher.validate_reference_file(library_path)
        with np.load(library_path, allow_pickle=False) as data:
            if missing := [key for key in LIBRARY_KEYS if key not in data.files]:
                raise ValueError(f'매칭 라이브러리에 {missing} 항목이 없습니다.')
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        session = ort.InferenceSession(model_bytes, options, providers=['CPUExecutionProvider'])
        metadata = session.get_modelmeta().custom_metadata_map
        if any(metadata.get(key) != value for key, value in ONNX_CONTRACT.items()):
            raise ValueError('위상 없는 참조 교사(233입력) ONNX 메타데이터가 아닙니다.')
        inputs, outputs = session.get_inputs(), session.get_outputs()
        if (len(inputs) != 1 or len(outputs) != 1 or inputs[0].name != 'obs' or inputs[0].shape != [1, 233]
                or inputs[0].type != 'tensor(float)' or outputs[0].name != 'actions'
                or outputs[0].shape != [1, 24] or outputs[0].type != 'tensor(float)'):
            raise ValueError('교사의 관측·관절 출력 계약이 일치하지 않습니다.')
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise ValueError(f'사람 모션 교사를 불러올 수 없습니다: {error}') from error
    training_reference = metadata.get('r1lab.reference_sha256', '')
    info = {'label': document.get('label') or f'{run_id} · {iteration}', 'run_id': run_id,
            'iteration': iteration, 'sha256': digest, 'path': str(policy.relative_to(project)),
            'library': str(library_path.relative_to(project)), 'library_sha256': library['sha256'],
            'training_reference_sha256': training_reference,
            # 다르면 학습 때 본 참조와 다른 라이브러리로 매칭한다(실험 표시용, 거부하지 않음).
            'library_matches_training': training_reference == library['sha256'],
            'selection_sha256': _sha256(encoded), 'source': 'reference_teacher_manifest',
            'matcher_overrides': training_matcher_overrides(policy.parent)}
    return TeacherSelection(session, info, library_path)


def training_matcher_overrides(run_dir):
    """선택한 교사가 학습한 생성기 옵션(run_info provenance). 학습과 같은 참조 생성 조건으로 실행한다."""
    try:
        record = json.loads((Path(run_dir) / 'run_info.json').read_text())
    except (OSError, ValueError):
        return {}
    provenance = record.get('reference_teacher') or {}
    return {'hold_settled_heading': True} if provenance.get('hold_settled_heading') else {}


class _MatchingView:
    """MatchingMotionCommand 속성 중 명령·로봇 특징·관측 함수가 읽는 부분을 MuJoCo 로봇 1대로 제공한다."""

    motion_anchor_body_index = 0  # BODY_NAMES[0] == pelvis
    joint_pos = matching.MatchingMotionCommand.joint_pos
    joint_vel = matching.MatchingMotionCommand.joint_vel
    body_pos_w = matching.MatchingMotionCommand.body_pos_w
    body_quat_w = matching.MatchingMotionCommand.body_quat_w
    anchor_pos_w = matching.MatchingMotionCommand.anchor_pos_w
    anchor_quat_w = matching.MatchingMotionCommand.anchor_quat_w
    current_command = matching.MatchingMotionCommand.current_command
    robot_features = matching.MatchingMotionCommand.robot_features

    def __init__(self, matcher, cfg):
        self.matcher, self.cfg = matcher, cfg
        self._env = SimpleNamespace(scene=SimpleNamespace(env_origins=torch.zeros(1, 3)), num_envs=1,
                                    command_manager=SimpleNamespace(get_term=lambda name: self))
        self.time_steps = torch.zeros(1, dtype=torch.long)
        self.user_command = torch.zeros(1, 3)
        self.heading_target = torch.zeros(1)
        self.heading_mode = torch.zeros(1, dtype=torch.bool)
        self.yaw_rate_filtered = torch.zeros(1)
        self.ref = matcher.reference()

    @property
    def robot_anchor_pos_w(self):
        return self.robot_body_pos_w[:, self.motion_anchor_body_index]

    @property
    def robot_anchor_quat_w(self):
        return self.robot_body_quat_w[:, self.motion_anchor_body_index]


class ReferenceTeacherController:
    """50Hz마다 step()으로 233관측→ONNX→ENTITY 순서 관절 목표와 kp·kd를 돌려준다."""

    command_from_axes = staticmethod(command_from_axes)

    def __init__(self, model, selection):
        self.selection, self.session, self.info = selection, selection.session, selection.info
        self.cfg = _defaults()
        self.cfg.matcher = {**self.cfg.matcher, **selection.info.get('matcher_overrides', {})}
        self.matcher = mm.MotionMatcher(selection.library, 1, 'cpu', **self.cfg.matcher)
        self.db = self.matcher.db
        self.matcher_config = {**self.cfg.matcher, 'reanchor_m': self.matcher.reanchor_m,
                               'reanchor_rad': self.matcher.reanchor_rad,
                               'pose_source': getattr(self.matcher, 'pose_source', 'robot')}
        with np.load(selection.library, allow_pickle=False) as data:
            self.clip_names = [str(name) for name in data['clip_names']]
            self.segment_clip = np.asarray(data['segment_clip'])
        # 학습 시작 자세와 같은 대기(StandIdle·IdleWaiting) 후보 프레임.
        from r1lab.mdp.matching_command import idle_candidates
        self.idle_candidates = idle_candidates(self.db, self.clip_names, self.segment_clip)
        if len(self.idle_candidates) == 0:
            raise ValueError('매칭 라이브러리에 대기 자세 프레임이 없습니다.')
        self.idle_frame = torch.zeros(self.db.frames, dtype=torch.bool)
        self.idle_frame[self.idle_candidates] = True
        self.body_ids = np.array([model.body(name).id for name in teacher.BODY_NAMES])
        self.root_body = model.body(teacher.BODY_NAMES[0]).id
        self.q_index = np.array([model.jnt_qposadr[model.joint(name).id] for name in C.ENTITY_JOINTS])
        self.v_index = np.array([model.jnt_dofadr[model.joint(name).id] for name in C.ENTITY_JOINTS])
        self.lin_sensor = slice(model.sensor('imu_lin_vel').adr[0], model.sensor('imu_lin_vel').adr[0] + 3)
        self.ang_sensor = slice(model.sensor('imu_ang_vel').adr[0], model.sensor('imu_ang_vel').adr[0] + 3)
        self.default = np.array([C.DEFAULT_POS_BY_NAME[name] for name in C.ENTITY_JOINTS])
        order = np.array(C.ACTION_IDX)
        self.action_index = order
        self.action_scale = np.array(C.ACTION_SCALE)[order]
        self.action_offset = np.array(C.DEFAULT_POS)[order]
        self.kp, self.kd = np.array(C.KP)[order], np.array(C.KD)[order]
        self.view = _MatchingView(self.matcher, self.cfg)
        self.started = False
        self.last_action = np.zeros(24, dtype=np.float32)
        self.lateral_requested = False
        self.start_frame = None
        self.command = self.view.user_command.clone()

    # ------------------------------------------------------------ robot state
    def _read_robot(self, data):
        """mjlab EntityData와 같은 정의: body link 위치·자세, cvel 기반 세계 속도."""
        ids = self.body_ids
        pos, cvel = data.xpos[ids], data.cvel[ids]
        offset = data.subtree_com[self.root_body] - pos
        lin = cvel[:, 3:6] - np.cross(cvel[:, 0:3], offset)
        view = self.view
        view.robot_body_pos_w = torch.as_tensor(pos[None], dtype=torch.float32)
        view.robot_body_quat_w = torch.as_tensor(data.xquat[ids][None], dtype=torch.float32)
        view.robot_body_lin_vel_w = torch.as_tensor(lin[None], dtype=torch.float32)
        view.robot_body_ang_vel_w = torch.as_tensor(cvel[None, :, 0:3], dtype=torch.float32)

    # ------------------------------------------------------------ control
    def reset(self, model, data):
        """로봇은 그대로 두고 매처를 현재 서 있는 자세에 맞춘다.

        학습의 기립 시작처럼 대기 프레임에서 시작하되(관절이 가장 가까운 프레임),
        참조 골반 xy·방향을 로봇에 둔다(MotionMatcher 재고정과 같은 식).
        """
        mujoco.mj_forward(model, data)
        self._read_robot(data)
        joints = torch.as_tensor(data.qpos[self.q_index], dtype=torch.float32)
        distance = (self.db.joint_pos[self.idle_candidates] - joints).square().sum(-1)
        frame = self.idle_candidates[int(distance.argmin())].reshape(1)
        pelvis_yaw = mm.yaw_of(self.view.robot_body_quat_w[:, 0])
        facing = mm.wrap(pelvis_yaw - self.db.sway[frame])
        self.matcher.reset(torch.arange(1), frame, self.view.robot_body_pos_w[:, 0, :2].clone(), facing)
        self.matcher.timer[:] = 0  # 무작위 첫 탐색 시점 대신 결정적으로 첫 틱에 탐색한다.
        view = self.view
        view.yaw_rate_filtered = self.db.facing_rate[frame].clone()
        view.user_command.zero_()
        view.heading_mode[:] = False
        view.heading_target[:] = facing
        view.ref = self.matcher.reference()
        self.started = False
        self.last_action = np.zeros(24, dtype=np.float32)
        self.lateral_requested = False
        self.start_frame = int(frame)

    def set_command(self, command, heading_target=None):
        """사용자 명령(vx, vy, wz)과 세계 방향 목표. 방향 목표가 있으면 학습처럼 회전 속도로 바꾼다."""
        view = self.view
        self.lateral_requested = abs(float(command[1])) > 1e-6
        view.user_command[0, 0] = float(command[0])
        view.user_command[0, 1] = 0.0  # 옆걸음 자료 없음: 학습 명령도 vy=0
        view.user_command[0, 2] = 0.0 if heading_target is not None else float(command[2])
        view.heading_mode[0] = heading_target is not None
        if heading_target is not None:
            view.heading_target[0] = float(heading_target)

    def advance_reference(self):
        """MatchingMotionCommand._update_command와 같은 순서로 참조를 한 틱 진행한다."""
        view = self.view
        alpha = 1.0 - math.exp(-CONTROL_DT / max(self.cfg.yaw_rate_filter_s, 1e-3))
        view.yaw_rate_filtered = view.yaw_rate_filtered + alpha * (
            view.robot_body_ang_vel_w[:, 0, 2] - view.yaw_rate_filtered)
        command = view.current_command()
        view.ref = self.matcher.step(command, view.robot_features(), dt=CONTROL_DT,
                                     heading_target=view.heading_target, heading_mode=view.heading_mode)
        return command

    def observation(self, data):
        """학습 actor 관측 233(항목 순서·배율은 OBSERVATION_CONTRACT·학습 함수와 같다)."""
        env = self.view._env
        terms = {
            'command': teacher.current_reference(env),
            'motion_anchor_pos_b': tracking.motion_anchor_pos_b(env, 'motion'),
            'motion_anchor_ori_b': tracking.motion_anchor_ori_b(env, 'motion'),
            'base_lin_vel': torch.as_tensor(data.sensordata[self.lin_sensor][None], dtype=torch.float32)
            * OBSERVATION_CONTRACT['base_linear_velocity_scale'],
            'base_ang_vel': torch.as_tensor(data.sensordata[self.ang_sensor][None], dtype=torch.float32)
            * OBSERVATION_CONTRACT['base_angular_velocity_scale'],
            'joint_pos': torch.as_tensor((data.qpos[self.q_index] - self.default)[None], dtype=torch.float32),
            'joint_vel': torch.as_tensor(data.qvel[self.v_index][None], dtype=torch.float32)
            * OBSERVATION_CONTRACT['joint_velocity_scale'],
            'actions': torch.as_tensor(self.last_action[None]),
            'phase': teacher.zero_phase_observation(env),
            'future_reference': matching.matching_future_reference(env),
        }
        rows = [terms[name].reshape(1, -1) for name, _ in TERMS]
        if [row.shape[1] for row in rows] != [width for _, width in TERMS]:
            raise ValueError('교사 관측 항목 크기가 계약과 다릅니다.')
        return torch.cat(rows, dim=-1).numpy().astype(np.float32)

    def step(self, model, data, command, heading_target=None):
        """한 50Hz 틱: 상태 갱신 → (첫 틱 제외) 참조 진행 → 관측 → ONNX → 목표각·kp·kd(ENTITY 순서)."""
        mujoco.mj_forward(model, data)
        self._read_robot(data)
        self.set_command(command, heading_target)
        self.command = self.advance_reference() if self.started else self.view.current_command()
        self.started = True
        self.obs = self.observation(data)
        raw = self.session.run(['actions'], {'obs': self.obs})[0][0]
        if not np.isfinite(raw).all():
            raise ValueError('교사 정책 출력에 유효하지 않은 값이 있습니다.')
        self.last_action = raw.astype(np.float32)
        target = raw[self.action_index] * self.action_scale + self.action_offset
        return target, self.kp.copy(), self.kd.copy()

    # ------------------------------------------------------------ diagnostics
    def diagnostics(self):
        matcher, view = self.matcher, self.view
        play = int(matcher.play[0])
        clip = self.clip_names[int(self.segment_clip[int(self.db.frame_segment[play])])]
        error = view.robot_body_pos_w[0, 0, :2] - view.ref['body_pos_w'][0, 0, :2]
        command = self.command
        return {'frame': play, 'clip': clip.removesuffix('.npz'), 'idle_frame': bool(self.idle_frame[play]),
                'start_frame': self.start_frame, 'blend': int(matcher.blend[0]),
                'switches': int(matcher.switches[0]), 'reanchors': int(matcher.reanchors[0]),
                'heading_error_rad': float(mm.wrap(matcher.command_heading[0] - matcher.facing[0]).abs()),
                'tracking_error_m': float(error.norm()),
                'command': [float(command[0, 0]), float(command[0, 2])],
                'lateral_ignored': self.lateral_requested}
