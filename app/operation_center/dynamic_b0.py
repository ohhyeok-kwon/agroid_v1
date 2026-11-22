"""DynamicMode 의 B0 연결(실험): S1 목적 v2 + 닻 표현 정책 하나(careful_walk_b0 시드 1, 갱신 2,400)를 학습 env 에서 돌린다.
DynamicMode 의 기본 모델은 E114 이고, B0 는 '연결 모델과 평가'에서 고를 때만 이 파일로 연다(simulation.DYNAMIC_MODELS).

B0 는 참조 동작 없이 골반의 결과 목표(S1 목표 9값)를 보고 24관절 목표를 낸다. 그래서 E114 전용 부품(조향 고리 n18b,
운동 토큰 머리 h12, 토큰기 v2d, 모션 라이브러리 corebal)은 B0 에서 쓰지 않는다. 문장 → 지시를 읽는 목표 모델(goal2b_v1)과
입력 사상(dynamic_mode: 키·방향·자세·문장 → 27개 어휘 → 5값 명령)은 그대로 쓰고, 그 5값을 아래 어댑터가 S1 목표로 바꾼다.

  장면     B0 판독기(r12_b0drive)와 같은 play env: anchor_representation(s1_env_cfg(effort="power", play=True), play=True)
           (잡음·지연·무작위화·밂 없음, 물리 200 Hz · 제어 50 Hz · 위치 구동기). 앱이 더한 것: 공 8개, 지면 광선 감지기,
           머리 카메라(문장 읽기), 넘어져도 다시 세우지 않음(time_out 만), 초기화 자세 고정(방향 0, 관절 오프셋 0).
  관측     env 자신의 actor 관측 870 = 10프레임 x [각속도 3, 중력 3, S1 목표 9, q-q0 24, dq 24, 직전 행동 24] (정책 순서)
  정책     고정 파일의 ONNX(정규화기 포함, 평균 행동)를 CPU onnxruntime 으로. 행동 = 정책 순서 관절 목표 오프셋(닻의 행동 항)
  목표     판독기의 방법 그대로: 명령 항의 갱신 뒤에 목표(마스크·세계 목표·남은 시간)를 덮어쓰고 관측을 다시 만든다.
           hold 를 1000 s 로 두어 표집기가 새 목표를 뽑지 않게 한다.

어댑터(5값 명령 -> S1 목표; 학습 표집 범위 r1s1.numbers v2 안에서, 지시가 바뀔 때 그 순간의 자세로 고정):
  걷기 (vx, vy)      이동 목표(마스크 평면·방향): 명령 속도로 걸어가는 점(지시 때 골반에서 출발, 골반과 속도 x 2 s 이상
                     벌어지지 않음)에서 (vx, vy) x 2 s 앞(0.4~1.0 m; 학습 0~2 m), 방향 목표 = 지시 시작 때의 방향,
                     마감 = 0.5 + 거리/속도 (학습: SETTLE 0.5 + 거리/보행 속도 0.2~0.6 m/s). ROLL_S 마다 다시 굴린다.
                     (골반에서 바로 굴리면 늘 '새 목표의 시작'이라 0.35 m/s 명령에 0.54~0.63 m/s 로 걸었다.)
  회전 (wz)          제자리 이동 목표: 평면 목표 = 회전 시작 위치, 방향 목표 = 지금 방향 + wz x 2 s(최대 pi/2; 학습 -pi~pi),
                     마감 = 0.5 + |회전|/회전율(학습 회전율 0.4~1.0 rad/s). ROLL_S 마다 다시 굴린다.
  서 있기·멈춤        버티기 목표(마스크 높이): 그 순간의 골반 높이를 유지, 마감 = 지금(학습과 같음). 앉은 채 '멈춰'는 앉은 높이를 유지한다.
                     초기화 직후의 첫 버티기 목표는 학습처럼 초기화의 목표 갱신에서 잡는다(골반 0.741 m).
  앉기·일어서기        높이 목표(마스크 셋 다): 그 자리·방향에서 골반 높이 0.62 / 0.54 / 0.731 m(학습 0.40~0.7313 m),
                     마감 = 0.5 + |높이 차|/0.2 m/s (학습 하강 속도 0.1~0.3 m/s 의 가운데).
평지에서만 학습했고 높이 목표가 절대 골반 높이이므로 평지에서만 연다. Simulation 전용이며 R1 배포 정책과는 별개다.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from dynamic_core import BALLS, DynamicCore

MANIFEST = '.agent/operation_center/dynamic_mode_b0.json'
MANIFEST_KEYS = {'version', 'label', 'run_id', 'iteration', 'path', 'sha256', 'checkpoint', 'goal_model', 'selected_at', 'note'}
WIDTH = 870
STAND = (0.0, 0.0, 0.0, 0.731, 0.0)
# ---- adapter numbers (r1s1.numbers v2 ranges in brackets)
HORIZON_S = 2.0          # move distance = speed x 2 s: 0.4 / 0.7 / 1.0 m for 0.2 / 0.35 / 0.5 m/s  [LOCO_DIST_M 0-2 m]
TURN_MAX = math.pi / 2   # turn per goal <= pi/2 (the reader's addendum 11 step, clear of the +-pi wrap)  [LOCO_TURN_RAD -pi..pi]
SETTLE_S = 0.5           # [SETTLE_S]
PACE = (0.2, 0.6)        # m/s the deadline is paced with  [PACE_MPS]
TURN_RATE = (0.4, 1.0)   # rad/s  [TURN_RATE_RPS]
CROUCH_RATE = 0.2        # m/s, middle of [CROUCH_RATE_MPS 0.1-0.3]
ROLL_S = 1.0             # s between re-issues of a held move / turn goal (a new instruction issues at once); between
                         # them the deadline counts down as in training. Headless check (.agent/operation_center/s1_b0):
                         # every step 0.41 m/s ahead, 0.13 m/s sideways, 0.47 rad/s; every 1 s 0.38 / 0.18 m/s, 0.67 rad/s
                         # for the 0.35 m/s / 0.6 rad/s commands (no fall in either)
HOLD_BIG = 1000.0        # s, the goal term's hold: no resampling (as the reader)
KINDS = ('hold', 'move', 'crouch')


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def load_manifest(project, manifest=None):
    """B0 선택 파일(.agent/operation_center/dynamic_mode_b0.json)을 읽고 ONNX·체크포인트 해시를 확인한다."""
    project = Path(project).resolve()
    path = project / (manifest or MANIFEST)
    if not path.exists():
        raise ValueError('B0 선택 파일이 없습니다(.agent/operation_center/dynamic_mode_b0.json).')
    encoded = path.read_bytes()
    entry = json.loads(encoded)
    if not isinstance(entry, dict) or set(entry) - MANIFEST_KEYS or entry.get('version') != 1:
        raise ValueError('B0 선택 파일의 형식이 올바르지 않습니다.')
    runs, work = (project / '.agent/trainer/runs').resolve(), (project / '.agent/trainer').resolve()
    relative = Path(entry['path'])
    policy = (project / relative).resolve()
    if (relative.is_absolute() or not policy.is_relative_to(runs) or policy.parent.name != entry['run_id']
            or policy.name != f"policy_{entry['iteration']}.onnx"):
        raise ValueError('B0: 등록된 학습의 정책 경로가 아닙니다.')
    model = policy.read_bytes()
    if _sha256(model) != entry['sha256']:
        raise ValueError('B0: 정책 파일 해시가 선택한 것과 다릅니다.')
    out = {'label': entry.get('label', f"B0 · {entry['iteration']}"), 'note': entry.get('note', ''),
           'run_id': entry['run_id'], 'iteration': entry['iteration'], 'path': str(policy), 'sha256': entry['sha256'],
           'model': model, 'selection_sha256': _sha256(encoded), 'goal_model': None}
    checkpoint = entry.get('checkpoint')
    if checkpoint is not None:
        file = (project / checkpoint['path']).resolve()
        if file.parent != policy.parent or _sha256(file.read_bytes()) != checkpoint['sha256']:
            raise ValueError('B0: 체크포인트 해시가 선택한 것과 다릅니다.')
    item = entry.get('goal_model')
    if item is not None:
        file, backbone = (project / item['path']).resolve(), (project / item.get('backbone', '')).resolve()
        if not file.is_relative_to(work) or not file.is_file() or not backbone.is_relative_to(work) or not backbone.is_dir():
            raise ValueError('B0: 목표 모델 파일이 트레이너 작업 폴더에 없습니다.')
        out['goal_model'] = {**item, 'path': str(file), 'backbone': str(backbone)}   # hash: dynamic_mode.verify_goal_model
    return out


def _wrap(angle):
    return torch.remainder(angle + math.pi, 2 * math.pi) - math.pi


class Adapter:
    """5값 명령(vx, vy, 회전율, 높이, 높이 사용) -> S1 목표. 명령이 바뀌면 그 순간의 자세로 고정하고, 이동·회전은 ROLL_S 마다 굴린다."""

    def __init__(self, n, device, roll_s=ROLL_S, pace=True):
        self.roll_s = roll_s
        self.request = torch.tensor(STAND, device=device).repeat(n, 1)
        self.active = torch.full((n, 5), float('nan'), device=device)
        self.psi_ref = torch.zeros(n, device=device)          # heading a walk keeps
        self.p_turn = torch.zeros(n, 2, device=device)        # where a turn in place stays
        self.carrot = torch.zeros(n, 2, device=device)        # the point that walks at the commanded speed
        self.t_issue = torch.zeros(n, device=device)
        self.t_plan = torch.zeros(n, device=device)
        self.pace = pace

    def tell(self, goal):
        goal = torch.as_tensor(goal, dtype=torch.float, device=self.request.device)
        self.request = goal.repeat(len(self.request), 1) if goal.dim() == 1 else goal.clone()

    def restart(self):
        self.request = torch.tensor(STAND, device=self.request.device).repeat(len(self.request), 1)
        self.active[:] = float('nan')

    def plan(self, goal, t, pos, yaw):
        """Called inside the goal term's update (fresh state after physics). Writes the goals it (re-)issues."""
        want = self.request
        changed = ~(want == self.active).all(dim=1)
        v = want[:, :2]
        speed = v.norm(dim=1)
        moving, turning = speed > 1e-6, want[:, 2].abs() > 1e-6
        loco = moving | turning
        roll = loco & (t - self.t_issue >= self.roll_s - 1e-6)
        issue = changed | roll
        if not issue.any():
            return
        ids = issue.nonzero().flatten()
        fresh = changed[ids]
        self.psi_ref[ids] = torch.where(fresh, yaw[ids], self.psi_ref[ids])
        self.p_turn[ids] = torch.where(fresh.unsqueeze(1), pos[ids, :2], self.p_turn[ids])
        w, z = want[ids], pos[ids, 2]
        kind = torch.where(loco[ids], 1, torch.where(w[:, 4] > 0.5, 2, 0))
        # move / turn
        c, s = torch.cos(self.psi_ref[ids]), torch.sin(self.psi_ref[ids])
        rot = lambda xy: torch.stack([c * xy[:, 0] - s * xy[:, 1], s * xy[:, 0] + c * xy[:, 1]], dim=1)  # noqa: E731
        here = pos[ids, :2]
        if self.pace:
            # the goal is rolled from a point that walks at the commanded speed (it starts at the pelvis and never
            # leads or trails it by more than speed x horizon), so the robot is asked for the commanded pace, not
            # for a fresh goal every step
            elapsed = torch.where(fresh, torch.zeros_like(z), t - self.t_plan[ids]).unsqueeze(1)
            carrot = torch.where(fresh.unsqueeze(1), here, self.carrot[ids] + rot(w[:, :2]) * elapsed)
            off = carrot - here
            lead = (speed[ids] * HORIZON_S).unsqueeze(1)
            carrot = here + off * (lead / off.norm(dim=1, keepdim=True).clamp_min(1e-6)).clamp(max=1.0)
            self.carrot[ids] = carrot
        else:
            carrot = here
        ahead = carrot + rot(w[:, :2] * HORIZON_S)
        dist = (ahead - here).norm(dim=1)
        p_loco = torch.where(moving[ids].unsqueeze(1), ahead, self.p_turn[ids])
        dpsi = (w[:, 2] * HORIZON_S).clamp(-TURN_MAX, TURN_MAX)
        psi_loco = torch.where(turning[ids], yaw[ids] + dpsi, self.psi_ref[ids])
        tau_loco = (SETTLE_S + dist / speed[ids].clamp(*PACE)
                    + dpsi.abs() / w[:, 2].abs().clamp(*TURN_RATE))
        # crouch / rise, hold
        tau_crouch = SETTLE_S + (w[:, 3] - z).abs() / CROUCH_RATE
        mask = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], device=pos.device)[kind]
        p = torch.where((kind == 1).unsqueeze(1), p_loco, pos[ids, :2])
        psi = torch.where(kind == 1, psi_loco, yaw[ids])
        h = torch.where(kind == 2, w[:, 3], z)
        tau = torch.where(kind == 1, tau_loco, torch.where(kind == 2, tau_crouch, torch.zeros_like(z)))
        goal.write(ids, kind, mask, p, psi, h, t + tau)
        self.t_issue[ids] = torch.where(changed[ids] | roll[ids], torch.full_like(z, t), self.t_issue[ids])
        self.t_plan[ids] = t
        self.active[ids] = w


class Schedule:
    """Positive control / tests: the reader's goal list (r12_b0drive.goals_for semantics) instead of the adapter."""

    def __init__(self, rows):
        self.rows, self.index, self.p0 = rows, -1, None      # rows: [(t_issue, kind, dxy_start_frame, dyaw, h, t_dead)]

    def plan(self, goal, t, pos, yaw):
        if self.p0 is None:
            self.p0, self.psi0 = pos[:, :2].clone(), yaw.clone()
        index = sum(1 for row in self.rows if row[0] <= t + 1e-6) - 1
        if index == self.index:
            return
        self.index = index
        _, kind, dxy, dyaw, h, t_dead = self.rows[index]
        n = len(pos)
        ids = torch.arange(n, device=pos.device)
        c, s = torch.cos(self.psi0), torch.sin(self.psi0)
        p_move = self.p0 + torch.stack([c * dxy[0] - s * dxy[1], s * dxy[0] + c * dxy[1]], dim=1)
        k = KINDS.index(kind)
        mask = torch.tensor([[0.0, 0.0, 1.0], [1.0, 1.0, 0.0], [1.0, 1.0, 1.0]], device=pos.device)[k].repeat(n, 1)
        p = p_move if kind == 'move' else (pos[:, :2] if kind == 'crouch' else self.p0)
        psi = self.psi0 + dyaw if kind == 'move' else (yaw if kind == 'crouch' else self.psi0)
        hh = torch.full((n,), float(h), device=pos.device) if h is not None else pos[:, 2].clone()
        goal.write(ids, torch.full((n,), k, device=pos.device), mask, p, psi, hh, torch.full((n,), float(t_dead), device=pos.device))


class Goal:
    """The S1 goal the app sets (what the actor's goal observation is built from), written into r1s1's goal term."""

    def __init__(self, n, device):
        self.kind = torch.zeros(n, dtype=torch.long, device=device)
        self.mask = torch.tensor([0.0, 0.0, 1.0], device=device).repeat(n, 1)
        self.p = torch.zeros(n, 2, device=device)
        self.psi = torch.zeros(n, device=device)
        self.h = torch.full((n,), STAND[3], device=device)
        self.t_dead = torch.zeros(n, device=device)
        self.issued = 0

    def write(self, ids, kind, mask, p, psi, h, t_dead):
        self.kind[ids], self.mask[ids], self.p[ids] = kind, mask, p
        self.psi[ids], self.h[ids], self.t_dead[ids] = _wrap(psi), h, t_dead
        self.issued += 1


class B0Core(DynamicCore):
    """DynamicCore 와 같은 겉모양(tell / step / reset / 공 / 읽기)으로 B0 를 돌린다. 로봇 수는 인자(앱은 1)."""

    def __init__(self, selection: dict, environment: str = 'flat', envs: int = 1, device: str = 'cuda:0',
                 camera: bool = False, planner=None):
        if environment != 'flat':
            raise ValueError('B0 는 평지에서만 학습했고 높이 목표가 절대 골반 높이라 평지에서만 엽니다. '
                             '지형을 평지로 바꾸거나 연결 모델을 E114 로 고르세요.')
        import onnxruntime as ort
        from mjlab.envs import ManagerBasedRlEnv

        from r1lab.tasks.careful_walk_b0 import anchor_representation
        from r1s1.env_cfg import s1_env_cfg
        from r1wb import world

        self.selection, self.environment, self.count, self.device = selection, environment, envs, device
        options = ort.SessionOptions()
        options.intra_op_num_threads, options.inter_op_num_threads = 1, 1
        self.session = ort.InferenceSession(selection['model'], options, providers=['CPUExecutionProvider'])
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != 'obs' or inputs[0].shape != [1, WIDTH]
                or len(outputs) != 1 or outputs[0].name != 'actions' or outputs[0].shape != [1, 24]):
            raise ValueError('B0: 관측·관절 출력 계약(870 -> 24)이 일치하지 않습니다.')
        cfg = self.env_cfg(envs, camera)
        self.ball_names = world.add_throwables(cfg, 'ball', BALLS)
        cfg.sim.nconmax = max(cfg.sim.nconmax or 0, 400)
        cfg.sim.njmax = max(cfg.sim.njmax or 0, 1600)
        self.cfg = cfg
        self.env = ManagerBasedRlEnv(cfg=cfg, device=device)
        env = self.env
        self.robot = env.scene['robot']
        self.dt = float(env.step_dt)
        self.term = env.command_manager.get_term('goal')
        self.target = Goal(envs, device)
        self.adapter = Adapter(envs, device)
        self.planner = planner or self.adapter
        self.t_next = 0.0
        original = self.term._update_command

        def update(env_ids=None):
            original(env_ids)
            self._write_goal()
        self.term._update_command = update
        self._world = world
        self.fallen = torch.zeros(envs, dtype=torch.bool, device=device)
        self.steps = 0
        self.balls = [env.scene[name] for name in self.ball_names]
        self.ball_used = torch.zeros(envs, dtype=torch.long, device=device)
        self.ball_age = torch.full((BALLS, envs), 1.0e3, device=device)
        self.ball_flight = torch.zeros(BALLS, envs, 3, device=device)
        self.ball_measure = torch.zeros(BALLS, envs, device=device)
        self.ball_given = torch.full((BALLS, envs), float('nan'), device=device)
        self.ball_mass = float(world.BALL['mass'])
        self.last_action = np.zeros((envs, 24), dtype=np.float32)
        self.obs = None
        self.reset()

    @staticmethod
    def env_cfg(envs: int = 1, camera: bool = False):
        """The reader's play env plus the app leaves (listed in the module docstring)."""
        from r1lab.tasks.careful_walk_b0 import anchor_representation
        from r1s1.env_cfg import s1_env_cfg
        from r1wb import world
        cfg = anchor_representation(s1_env_cfg(effort='power', play=True), play=True)
        cfg.scene.num_envs = envs
        cfg.scene.sensors = tuple(cfg.scene.sensors) + world.ray_sensors()
        if camera:
            from r1wb.s2.collect import head_camera
            cfg.scene.sensors = tuple(cfg.scene.sensors) + (head_camera(224),)
        cfg.terminations = {'time_out': cfg.terminations['time_out']}            # a fall is shown, not reset
        base, joints = cfg.events['reset_base'], cfg.events['reset_joints']
        base.params = {**base.params, 'pose_range': {'yaw': (0.0, 0.0)}, 'velocity_range': {}}
        joints.params = {**joints.params, 'position_range': (0.0, 0.0)}
        return cfg

    # ------------------------------------------------------------------ goal
    def _write_goal(self) -> None:
        term, goal = self.term, self.target
        pos, yaw, _, _ = term.state()
        t = self.t_next
        self.planner.plan(goal, t, pos, yaw)
        term.mask[:] = goal.mask
        term.p_target[:] = goal.p
        term.psi_target[:] = goal.psi
        term.h_target[:] = goal.h
        term.hold[:] = HOLD_BIG
        term.time_left[:] = HOLD_BIG + (goal.t_dead - t).clamp_min(-100.0)       # tau = t_dead - t; never resamples
        term.pending[:] = False
        term._goal_obs = term._observe()

    # ------------------------------------------------------------------ control
    def tell(self, goal) -> None:
        self.adapter.tell(goal)

    def act(self, seen: np.ndarray) -> np.ndarray:
        return np.stack([self.session.run(['actions'], {'obs': seen[i:i + 1]})[0][0] for i in range(len(seen))])

    def step(self) -> None:
        with torch.no_grad():
            seen = self.obs.detach().to('cpu').numpy().astype(np.float32)
            actions = self.act(seen)
            self.seen, self.last_action = seen, actions
            self.t_next = (self.steps + 1) * self.dt
            out = self.env.step(torch.as_tensor(actions, device=self.device))
            self.obs = out[0]['actor']
            self._balls_after_step()
            self.fallen |= self._world.fallen(self.env)
        self.steps += 1

    def reset(self) -> None:
        everyone = torch.arange(self.count, device=self.device)
        if hasattr(self.planner, 'restart'):
            self.planner.restart()
        self.steps, self.t_next = 0, 0.0
        out = self.env.reset()               # the first goal is latched in the reset's goal update, as in training
        self.obs = out[0]['actor']
        self.fallen[:] = False
        self.last_action[:] = 0.0
        for index in range(BALLS):
            self._park(index, everyone)

    # ------------------------------------------------------------------ what the caller reads
    def plan_minus_robot_height(self) -> torch.Tensor:
        return torch.zeros(self.count, device=self.device)

    def told_and_command(self):
        """(the 5-value command told, the requested (vx, vy, yaw rate)) like DynamicCore (no slow loop in B0)."""
        return self.adapter.request, self.adapter.request[:, :3]

    def goal_state(self, env: int = 0) -> dict:
        """The S1 goal now (heading-frame errors as the actor's goal observation sees them)."""
        pos, yaw, _, _ = self.term.state()
        tau = float(self.term.tau[env])
        d = self.target.p[env] - pos[env, :2]
        c, s = math.cos(float(yaw[env])), math.sin(float(yaw[env]))
        return {'kind': KINDS[int(self.target.kind[env])], 'mask': [int(v) for v in self.target.mask[env].tolist()],
                'ahead_m': round(c * float(d[0]) + s * float(d[1]), 3), 'left_m': round(-s * float(d[0]) + c * float(d[1]), 3),
                'heading_error_rad': round(float(_wrap(self.target.psi[env] - yaw[env])), 3),
                'height_target_m': round(float(self.target.h[env]), 3), 'height_m': round(float(pos[env, 2]), 3),
                'deadline_s': round(max(tau, 0.0), 2)}
