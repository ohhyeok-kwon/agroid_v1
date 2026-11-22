"""DynamicMode 의 SONIC-R1 연결(실험): 지금 학습 중인 SONIC → R1 추종기를 사람이 키·방향·문장으로 몰아 본다.

  추종기   SONIC universal token 정책(인코더 → FSQ → 디코더, 50 Hz)을 R1 24 관절에 옮긴 것. 선택 파일
           .agent/operation_center/dynamic_mode_sonic.json(현재 학습의 최신 저장본) · dynamic_mode_sonic_b0.json(이식 원판
           B@0, 비교)의 경로·SHA256 으로 고정한다(.agent/operation_center/sonic/pin_sonic.sh). CPU 로 돈다.
  물리     T1 SONIC MuJoCo 하네스(sonic_g1.SimR1: r1wb 몸 = r1.xml + O6 손 무게, 계약 armature·kp·kd·effort, 바닥 평면)와
           같다: 200 Hz 명시적 PD(tau = clip(kp (q* - q) - kd dq, effort), qfrc_applied), 50 Hz 정책, 잡음 없음. 행동 지연은
           선택 파일의 action_delay_steps(물리 걸음 수 = 5 ms 단위, 기본 0 = 없음)만큼: 새 관절 목표를 그 걸음 수 뒤부터 적용한다
           (3 = 15 ms). 서버 판(P34 calm2_ub.py 의 지연 0 / 15 / 30 ms 판)의 규칙을 그대로 옮긴 것이다.
           관측·행동은 sonic_g1.DeployR1Native.control_tick 그대로(참조는 이미 R1 관절 공간: mo.space = "r1"). 하네스 사본은
           .agent/operation_center/sonic/harness/(서버 원본과 경로만 다름, SOURCES.json 의 해시로 확인).
  참조     sonic_reference.CalmCausalReference: calm 인과 합성기(판은 sonic_reference.GENERATOR; 학습·평가용, 배포 계획기 아님)를 실시간으로. 참조는 서버
           판(npz, float32)처럼 float32 로 반올림해 읽고 관절 속도는 clip_eval.fd 와 같은 중심 차분으로 낸다.
  명령     dynamic_mode 의 입력 사상(키·방향·자세·문장 → 27개 어휘 → 5값)에서 (vx, vy, 회전율)만 쓴다. 앉기·일어서기는 이
           모델에 없어 서 있기로 둔다(작업 프로세스가 화면에 알린다). 합성기 범위(LIMITS)는 줄여 쓰고 표시한다.
  앱 몫    공 8개(1 kg, 반지름 7 cm, 무게 없음, 멀리 띄워 둠 = r1wb.world 의 던질 물체), 넘어져도 다시 세우지 않음(높이 0.35 m 또는
           기울기 0.9 rad 를 넘으면 넘어짐으로 표시), 평지만(학습·검증 모두 평지).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

from sonic_reference import CalmCausalReference, ReferenceFailed, ReferenceSource, wait_started

PROJECT = Path(__file__).resolve().parents[2]
MANIFESTS = {'sonic': '.agent/operation_center/dynamic_mode_sonic.json',
             'sonic_b0': '.agent/operation_center/dynamic_mode_sonic_b0.json'}
MANIFEST_KEYS = {'version', 'slot', 'label', 'path', 'sha256', 'step', 'source', 'reference', 'selected_at', 'note',
                 'action_delay_steps'}
MAX_DELAY_STEPS = 6                # action_delay_steps: 0 .. 6 physics steps (0 .. 30 ms, the range the server panel runs)
SONIC_DIR = '.agent/operation_center/sonic'
PARTS = f'SONIC 추종기(인코더 → FSQ → 디코더) · {CalmCausalReference.name} 참조 · T1 하네스 물리(200 Hz PD)'
INPUTS = '540 + 780'               # encoder: 10 frames x (q 24, dq 24, orientation 6); decoder: token 64 + history 10 x 78
FLAT_ONLY = ('SONIC-R1 연결은 평지에서만 학습·검증했고 하네스도 평면 바닥이라 평지에서만 엽니다. '
             '지형을 평지로 바꾸거나 연결 모델을 E114 로 고르세요.')
BALLS, BALL_LIFETIME, BALL = 8, 5.0, dict(radius=0.07, mass=1.0)
PARK = 60.0                        # m, where the balls wait (r1wb.world.PARK)
FALL_HEIGHT, FALL_TILT = 0.35, 0.9  # r1wb.world.fallen on flat ground
SHIFT_AT, KEEP = 20000, 200        # frames: the reference buffer is moved down once the tracker is this far in
NOTE_TICKS = 150                   # ticks (3 s) a note about a reduced command / a restarted reference stays on screen
REDUCED = '명령을 줄였습니다: 합성기가 이 명령을 그대로는 디딜 수 없어 줄여서 걷습니다.'
RESTARTED = '참조 합성기가 이 명령을 풀지 못해 그 자리에서 서기로 돌아갔습니다. 입력을 바꾸면 다시 움직입니다.'


def _sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def delay_steps(value) -> int:
    """선택 파일의 action_delay_steps: 물리 걸음 수(5 ms 단위)의 정수 0 ~ MAX_DELAY_STEPS. 틀린 값은 0 으로 바꾸지 않고 거절한다."""
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_DELAY_STEPS:
        raise ValueError(f'SONIC-R1: action_delay_steps 는 0~{MAX_DELAY_STEPS} 의 정수(물리 걸음 수, 5 ms 단위)여야 합니다({value!r}).')
    return value


def load_manifest(project, key):
    """SONIC-R1 선택 파일을 읽고 모델 파일·하네스 사본의 해시를 확인한다. 문장 목표 모델은 E114 선택 파일의 것을 쓴다."""
    project = Path(project).resolve()
    path = project / MANIFESTS[key]
    if not path.exists():
        raise ValueError(f'SONIC-R1 선택 파일이 없습니다({MANIFESTS[key]}). {SONIC_DIR}/pin_sonic.sh 로 고정하세요.')
    encoded = path.read_bytes()
    entry = json.loads(encoded)
    if (not isinstance(entry, dict) or set(entry) - MANIFEST_KEYS or entry.get('version') != 1 or entry.get('slot') != key
            or not {'label', 'path', 'sha256'} <= set(entry)):
        raise ValueError('SONIC-R1 선택 파일의 형식이 올바르지 않습니다.')
    delay = delay_steps(entry.get('action_delay_steps', 0))
    models = (project / SONIC_DIR / 'models').resolve()
    relative = Path(entry['path'])
    file = (project / relative).resolve()
    if relative.is_absolute() or not file.is_relative_to(models) or not file.is_file():
        raise ValueError(f'SONIC-R1: 모델 파일이 {SONIC_DIR}/models 에 없습니다({entry["path"]}).')
    if _sha256(file) != entry['sha256']:
        raise ValueError('SONIC-R1: 모델 파일 해시가 선택한 것과 다릅니다.')
    harness = project / SONIC_DIR / 'harness'
    sources = json.loads((project / SONIC_DIR / 'SOURCES.json').read_text())
    for name, item in sources['files'].items():
        if _sha256(harness / name) != item['pc_sha256']:
            raise ValueError(f'SONIC-R1: 하네스 사본 {name} 이 가져온 것과 다릅니다(fetch_harness.py).')
    goal_model = None
    try:
        import dynamic_mode
        goal_model = dynamic_mode.load_manifest(project, verify=False)['goal_model']
    except (ValueError, OSError, KeyError):
        pass
    source = entry.get('source') or {}
    return {'label': entry['label'], 'note': entry.get('note', ''), 'step': entry.get('step'), 'path': str(file),
            'sha256': entry['sha256'], 'source': source, 'run_id': Path(source.get('path', file.name)).name,
            'selected_at': entry.get('selected_at'), 'selection_sha256': hashlib.sha256(encoded).hexdigest(),
            'goal_model': goal_model, 'harness': str(harness), 'action_delay_steps': delay}


def selection_note(selection) -> str:
    """One line for the screen: the selection file's evaluation note (as E114 / B0 show theirs; not pin_sonic.sh's default
    note), then the reference's standing, which server save is pinned and when."""
    source = selection.get('source') or {}
    when = (selection.get('selected_at') or '')[:16].replace('T', ' ')
    delay = selection.get('action_delay_steps', 0)
    late = f' · 행동 지연 {5 * delay} ms' if delay else ''
    note = str(selection.get('note') or '').strip()
    rated = f'{note}\n' if note and not note.startswith('pin_sonic.sh:') else ''
    return (f"{rated}{CalmCausalReference.note} · 저장본 {Path(source.get('path', '')).name or '?'}"
            f" (step {selection.get('step')}, 고정 {when}){late}")


def _import_harness(harness):
    os.environ.setdefault('R1_BODY', 'r1wb')
    source = os.environ.get('AGROID_TRAINER_SRC') or str(PROJECT / 'trainer/src')    # r1wb, r1lab (the harness's body)
    for path in (source, harness):
        if path not in sys.path:
            sys.path.insert(0, path)
    import clip_eval
    import sonic_g1
    return sonic_g1, clip_eval


def add_balls(spec, count=BALLS):
    """r1wb.world's throwable balls in one MuJoCo model (free bodies with gravity compensation, parked high up)."""
    import mujoco
    for index in range(count):
        body = spec.worldbody.add_body(name=f'ball_{index}', pos=[0.0, 0.0, PARK + index], gravcomp=1.0)
        body.add_freejoint(name=f'ball_{index}_free')
        body.add_geom(name=f'ball_{index}_shape', type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[BALL['radius'], 0, 0],
                      mass=BALL['mass'], rgba=[0.9, 0.35, 0.1, 1.0], group=2)
    return spec


def build_sim(S, balls=BALLS):
    """sonic_g1.SimR1('r1') unchanged; the balls are added to the body spec it compiles (after the robot)."""
    import r1wb.robot as robot
    original = robot.get_spec
    robot.get_spec = lambda *a, **k: add_balls(original(*a, **k), balls)
    try:
        sim = S.SimR1('r1')
    finally:
        robot.get_spec = original
    return sim


def make_tracker(S, sim, checkpoint):
    """sonic_g1.DeployR1Native for a reference in R1 joint space, without the parts it does not use then (the G1 ONNX
    policy of Deploy, the G1 hip map); control_tick / log_state / update_heading / advance are the harness's own."""
    dep = S.DeployR1Native.__new__(S.DeployR1Native)
    dep.sim, dep.pol, dep.ref, dep.cmd, dep.planner, dep.motion, dep.t_play = sim, None, 'clip', None, None, None, 0.0
    dep.npol = S.NativeR1Policy(checkpoint)
    dep.ent_to_il = np.array([S.R1_IL.index(n) for n in sim.rec_names])
    dep.il_to_ent = np.array([sim.rec_names.index(n) for n in S.R1_IL])
    dep.static_r1 = False
    restart_tracker(dep)
    return dep


def restart_tracker(dep):
    """The state Deploy.__init__ / DeployR1Native.__init__ start with (history empty, reference frame 0)."""
    dep.hist, dep.last_action, dep.last_a24 = [], np.zeros(29), np.zeros(24)
    dep.cur, dep.play, dep.play_ok, dep.reinit = 0, True, True, True
    dep.init_base, dep.init_ref = np.array([1.0, 0, 0, 0]), np.array([1.0, 0, 0, 0])
    dep.last_ms, dep.replan_ctr, dep.pending = None, 0.0, None
    dep.readapt_stored, dep.readapt_state, dep.readapt_orig = False, 'IDLE', np.zeros(29)


class _RobotView:
    """What the worker reads of the robot, in the shapes mjlab's entity data has (one world)."""

    def __init__(self, names):
        self.joint_names = list(names)
        self.data = type('Data', (), {})()

    def update(self, d, jq):
        self.data.root_link_pos_w = d.qpos[None, 0:3].copy()
        self.data.root_link_quat_w = d.qpos[None, 3:7].copy()
        self.data.root_link_lin_vel_w = d.qvel[None, 0:3].copy()
        self.data.joint_pos = d.qpos[None, jq].copy()


def _np(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


class SonicCore:
    """DynamicCore 와 같은 겉모양(tell / step / reset / 공 / 읽기)으로 SONIC-R1 추종기를 돌린다(로봇 1대, CPU)."""

    def __init__(self, selection: dict, environment: str = 'flat', envs: int = 1, reference: ReferenceSource | None = None,
                 balls: int = BALLS, record: bool = False):
        if environment != 'flat':
            raise ValueError(FLAT_ONLY)
        if envs != 1:
            raise ValueError('SONIC-R1 연결은 로봇 1대만 돌립니다.')
        import mujoco
        import torch
        self.mj = mujoco
        harness = selection.get('harness') or str(PROJECT / SONIC_DIR / 'harness')
        # the synthesizer process starts first: it imports its harness while the tracker loads here
        self.reference = reference if reference is not None else CalmCausalReference(harness)
        self.S, self.CE = _import_harness(harness)
        S = self.S
        self.selection, self.environment, self.count = selection, environment, 1
        self.delay = delay_steps(selection.get('action_delay_steps', 0))       # physics steps a new joint target acts late
        self.device = 'cpu'
        self.reader_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'   # the sentence reader (2B) only
        self.sim = build_sim(S, balls)
        if list(self.sim.rec_names) != list(self.CE.R1_MJ):
            raise ValueError('SONIC-R1: 하네스 관절 순서가 참조(R1 MJCF 순서)와 다릅니다.')
        self.tracker = make_tracker(S, self.sim, selection['path'])
        self.order = [list(self.CE.R1_MJ).index(n) for n in S.R1_IL]
        frames = self.reference.length or len(self.reference.qpos)
        self.motion = S.Motion(np.zeros((frames, 24)), np.zeros((frames, 24)), np.zeros((frames, 3)),
                               np.tile([1.0, 0.0, 0.0, 0.0], (frames, 1)), 'live')
        self.motion.space = 'r1'
        self.tracker.motion = self.motion
        self.dt = float(S.CTRL_DT)
        m = self.sim.m
        self.ball_q = [int(m.jnt_qposadr[m.joint(f'ball_{i}_free').id]) for i in range(balls)]
        self.ball_v = [int(m.jnt_dofadr[m.joint(f'ball_{i}_free').id]) for i in range(balls)]
        self.balls = list(range(balls))
        self.ball_mass = BALL['mass']
        self.floor = m.geom('floor').id
        self.foot_of = {g: self.sim.feet.index(int(m.geom_bodyid[g])) for g in range(m.ngeom)
                        if int(m.geom_bodyid[g]) in self.sim.feet}
        self.robot = _RobotView(self.sim.rec_names)
        self.record = {} if record else None
        self.goal = np.array([0.0, 0.0, 0.0, 0.731, 0.0])
        self.twist, self.note = (0.0, 0.0, 0.0), ''
        wait_started(self.reference)
        self.reset()

    # ------------------------------------------------------------------ reference -> tracker arrays
    def _derive(self, changed):
        """Motion arrays (R1 IsaacLab order) of the reference frames from local frame `changed` on (float32 like the
        server's npz; joint velocity = clip_eval.fd: central difference, one-sided at the ends of a fixed clip)."""
        ref, mo = self.reference, self.motion
        end = ref.length if ref.length else ref.valid
        if changed is None and self.derived >= end:
            return
        start = self.derived if changed is None else min(self.derived, changed)
        start = max(start - 1, 0)
        if start >= end:
            return
        q = ref.qpos[start:end].astype(np.float32).astype(np.float64)
        mo.rpos[start:end], mo.rquat[start:end] = q[:, 0:3], q[:, 3:7]
        mo.jp[start:end] = q[:, 7:31][:, self.order]
        dt = self.dt
        lo, hi = max(start, 1), end - 1                  # central difference where both neighbours are known
        if hi > lo:
            mo.jv[lo:hi] = (mo.jp[lo + 1:hi + 1] - mo.jp[lo - 1:hi - 1]) / (2 * dt)
        if start == 0 and end > 1:
            mo.jv[0] = (mo.jp[1] - mo.jp[0]) / dt
        if ref.length and end == ref.length:
            mo.jv[end - 1] = (mo.jp[end - 1] - mo.jp[end - 2]) / dt
            self.derived = end
        else:
            self.derived = end - 1                       # the last frame's velocity waits for its next neighbour

    # ------------------------------------------------------------------ control
    def tell(self, goal) -> None:
        """goal: the 5 values of the input mapping (vx, vy, yaw rate, height, height on). Height goals are not this model's.
        After a restart of the reference the command that could not be made is held at stand until the input changes."""
        goal = _np(goal).reshape(-1)[:5].astype(float)
        self.goal = goal
        self.note = ''
        if goal[4] > 0.5:
            self.note = '이 모델(SONIC-R1)은 앉기·일어서기를 지원하지 않아 서 있습니다.'
            twist = (0.0, 0.0, 0.0)
        else:
            twist = (float(goal[0]), float(goal[1]), float(goal[2]))
        if self.hold_off is not None:
            if twist == self.hold_off:
                twist = (0.0, 0.0, 0.0)
            else:
                self.hold_off = None
        self.twist = twist

    def step(self) -> None:
        S, sim, dep = self.S, self.sim, self.tracker
        n = self.steps
        try:
            self.used = self.reference.tell(n, self.twist)
            changed = self.reference.ready(n)
        except ReferenceFailed as error:
            changed = self._recover(n, error)
        self._derive(changed)
        reduced = getattr(self.reference, 'reductions', 0)
        if reduced != self.reductions_seen:
            self.reductions_seen = reduced
            self.screen = (REDUCED, n + NOTE_TICKS)
        dep.control_tick(n, n * self.dt)
        m, d = sim.m, sim.d
        # calm2_ub's action delay, line for line (neg28 Actuator rule): a target issued at physics step s acts from step
        # s + action_delay_steps; before the first one acts the target is the default pose. With 0 this is the newest target.
        self.issued.append((self.physics_steps, sim.native_qt24.copy()))
        self.issued = self.issued[-6:]
        for _ in range(S.DECIM):                         # calm2_ub / SimR1.step_ctrl: 200 Hz explicit PD
            qt = self.issued[0][1]
            for issued_at, target in self.issued:
                if issued_at + self.delay <= self.physics_steps:
                    qt = target
            qj, dqj = d.qpos[sim.jq], d.qvel[sim.jv]
            tau = np.clip(sim.kp * (qt - qj) - sim.kd * dqj, -sim.eff, sim.eff)
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[sim.jv] = tau
            d.xfrc_applied[:] = 0.0
            self.mj.mj_step(m, d)
            self.physics_steps += 1
        self.steps += 1
        if self.record is not None:
            self._record()
        self._balls_after_step()
        up = 1.0 - 2.0 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)
        self.fallen[0] |= bool(d.qpos[2] < FALL_HEIGHT or math.acos(max(-1.0, min(1.0, up))) > FALL_TILT)
        self.robot.update(d, sim.jq)
        if not self.reference.length and dep.cur > SHIFT_AT:
            self._shift(dep.cur - KEEP)

    def _recover(self, n, error):
        """The reference could not go on: it starts again standing still at this tick, the tracker re-aligns its heading to
        the robot's (Deploy.update_heading at reinit), and the command that failed is held at stand until the input changes."""
        self.recoveries.append(dict(tick=n, twist=list(self.twist), error=str(error)[:200]))
        self.hold_off, self.twist = (self.twist if any(self.twist) else None), (0.0, 0.0, 0.0)
        self.reference.restart(n)
        dep = self.tracker
        dep.cur, dep.reinit, dep.play, dep.play_ok = 0, True, True, True
        self.derived = 0
        self.used = self.reference.tell(n, self.twist)
        self.screen = (RESTARTED, n + NOTE_TICKS)
        return self.reference.ready(n)

    def screen_note(self) -> str:
        """A note for the screen while it is fresh (a reduced command, a restarted reference)."""
        text, until = self.screen
        return text if self.steps < until else ''

    def _shift(self, frames):
        self.reference.shift(frames)
        mo = self.motion
        for array in (mo.jp, mo.jv, mo.rpos, mo.rquat):
            array[:-frames] = array[frames:]
        self.tracker.cur -= frames
        self.derived -= frames

    def reset(self) -> None:
        ref = self.reference
        ref.reset()
        self.twist, self.goal, self.note = (0.0, 0.0, 0.0), np.array([0.0, 0.0, 0.0, 0.731, 0.0]), ''
        self.hold_off, self.screen, self.recoveries, self.reductions_seen = None, ('', 0), [], 0
        ref.tell(0, self.twist)
        self.derived = 0
        self._derive(ref.ready(0))
        restart_tracker(self.tracker)
        names = self.sim.rec_names
        J = ref.qpos[0:2, 7:31].astype(np.float32).astype(np.float64)
        P = ref.qpos[0:2, 0:3].astype(np.float32).astype(np.float64)
        Q = ref.qpos[0:2, 3:7].astype(np.float32).astype(np.float64)
        dq = (J[1] - J[0]) / self.dt
        self.sim.reset_to(P[0], Q[0], {n: J[0, i] for i, n in enumerate(names)}, (P[1] - P[0]) / self.dt,
                          self.CE.ang_vel_body(Q)[0], {n: dq[i] for i, n in enumerate(names)})
        self.steps = 0
        self.issued, self.physics_steps = [(-10 ** 9, self.sim.rec_default.copy())], 0   # (calm2_ub's start of an episode)
        self.used = self.twist
        self.fallen = np.zeros(1, dtype=bool)
        self.ball_used = np.zeros(1, dtype=int)
        self.ball_age = np.full((BALLS, 1), 1.0e3)
        self.ball_flight = np.zeros((BALLS, 1, 3))
        self.ball_measure = np.zeros((BALLS, 1))
        self.ball_given = np.full((BALLS, 1), np.nan)
        for index in self.balls:
            self._park(index)
        self.robot.update(self.sim.d, self.sim.jq)
        if self.record is not None:
            self.record.clear()

    def close(self) -> None:
        self.reference.close()

    # ------------------------------------------------------------------ test record (calm2_ub's npz layout)
    def _record(self):
        sim, d, m = self.sim, self.sim.d, self.sim.m
        F, other = np.zeros((2, 3)), False
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if self.floor not in (g1, g2):
                continue
            g = g2 if g1 == self.floor else g1
            if g in self.foot_of:
                self.mj.mj_contactForce(m, d, i, f6)
                fw = c.frame.reshape(3, 3).T @ f6[:3]
                F[self.foot_of[g]] += fw if g1 == self.floor else -fw
            else:
                other = True
        sole = np.array(self.S.R1_SOLE)
        rec = self.record
        for key, value in (('pos', d.qpos[0:3]), ('quat', d.qpos[3:7]), ('v', d.qvel[0:3]), ('torso_quat', d.xquat[sim.torso]),
                           ('q', d.qpos[sim.jq]), ('dq', d.qvel[sim.jv]),
                           ('sole', np.stack([d.xpos[b][None] + sole @ d.xmat[b].reshape(3, 3).T for b in sim.feet])),
                           ('force', F), ('other', other)):
            rec.setdefault(key, []).append(np.array(value, copy=True))

    # ------------------------------------------------------------------ balls (as dynamic_core.DynamicCore, one world)
    def throw(self, envs, origin, target, speed) -> None:
        origin, target, speed = _np(origin).reshape(-1, 3)[0], _np(target).reshape(-1, 3)[0], float(_np(speed).reshape(-1)[0])
        along = target - origin
        distance = max(float(np.linalg.norm(along)), 1.0e-6)
        along = along / distance
        index = int(self.ball_used[0]) % len(self.balls)
        d = self.sim.d
        a, v = self.ball_q[index], self.ball_v[index]
        d.qpos[a:a + 7] = [*origin, 1.0, 0.0, 0.0, 0.0]
        d.qvel[v:v + 6] = [*(along * speed), 0.0, 0.0, 0.0]
        self.ball_age[index, 0] = 0.0
        self.ball_flight[index, 0] = along * speed
        self.ball_measure[index, 0] = distance / max(speed, 0.1) + 0.15
        self.ball_given[index, 0] = np.nan
        self.ball_used[0] += 1

    def _park(self, index) -> None:
        d = self.sim.d
        a, v = self.ball_q[index], self.ball_v[index]
        d.qpos[a:a + 7] = [0.0, 0.0, PARK + index, 1.0, 0.0, 0.0, 0.0]
        d.qvel[v:v + 6] = 0.0
        self.ball_age[index, 0] = 1.0e3

    def _balls_after_step(self) -> None:
        d = self.sim.d
        self.ball_age += self.dt
        for index in self.balls:
            if self.ball_age[index, 0] >= 1.0e3:
                continue
            if self.ball_age[index, 0] >= self.ball_measure[index, 0] and np.isnan(self.ball_given[index, 0]):
                flight = self.ball_flight[index, 0]
                speed = max(float(np.linalg.norm(flight)), 1.0e-6)
                kept = float(d.qvel[self.ball_v[index]:self.ball_v[index] + 3] @ flight) / speed
                self.ball_given[index, 0] = self.ball_mass * (speed - kept)
            if self.ball_age[index, 0] > BALL_LIFETIME:
                self._park(index)

    def ball_positions(self, env: int = 0) -> list:
        d = self.sim.d
        return [(index, d.qpos[self.ball_q[index]:self.ball_q[index] + 3].tolist()) for index in self.balls
                if self.ball_age[index, 0] < 1.0e3]

    # ------------------------------------------------------------------ what the caller reads
    def yaw(self) -> np.ndarray:
        w, x, y, z = self.sim.d.qpos[3:7]
        return np.array([math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])

    def plan_minus_robot_height(self) -> np.ndarray:
        return np.zeros(1)

    def told_and_command(self):
        """(the 5-value command told, the (vx, vy, yaw rate) the reference uses after its limits)."""
        return self.goal[None], np.array(self.used, float)[None]

    def reference_state(self) -> dict:
        """The reference for the screen: its gait state, the command it uses, the heading the tracker is asked for."""
        ref = self.reference
        out = dict(ref.state())
        local = self.tracker.cur
        if 0 <= local < len(self.motion.rquat):
            w, x, y, z = self.motion.rquat[local]
            heading = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            out['heading_error_rad'] = round(math.remainder(heading - float(self.yaw()[0]), 2 * math.pi), 3)
        out.update(asked=[round(float(v), 3) for v in self.twist], note=self.note, reference_note=ref.note,
                   recoveries=len(self.recoveries), action_delay_ms=5 * self.delay)
        return out


class SonicRenderer:
    """SonicCore 의 상태를 그리는 화면(960 x 600, 로봇을 따라가는 자유 시점)과 문장 읽기용 머리 카메라(224, r1wb.s2.collect
    head_camera 와 같은 자리·방향·화각). 물리 모델과 같은 몸·공에 바닥 무늬·하늘·조명만 더한 별도 모델이다."""

    def __init__(self, core: SonicCore, width: int = 960, height: int = 600):
        import mujoco
        import r1wb.robot as robot
        self.mj, self.core, self.width, self.height = mujoco, core, width, height
        spec = add_balls(robot.get_spec(), len(core.balls))
        spec.add_texture(name='sky', type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.62, 0.72, 0.82], rgb2=[0.16, 0.2, 0.26], width=256, height=256)
        spec.add_texture(name='grid', type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                         rgb1=[0.5, 0.52, 0.54], rgb2=[0.42, 0.44, 0.46], mark=mujoco.mjtMark.mjMARK_EDGE,
                         markrgb=[0.6, 0.62, 0.64], width=512, height=512)
        material = spec.add_material(name='grid', texrepeat=[2.0, 2.0], texuniform=True, reflectance=0.05)
        material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = 'grid'
        spec.worldbody.add_geom(name='floor', type=mujoco.mjtGeom.mjGEOM_PLANE, size=[30.0, 30.0, 0.05], material='grid')
        spec.worldbody.add_light(name='sun', pos=[0.0, 0.0, 6.0], dir=[0.2, 0.3, -1.0], diffuse=[0.75, 0.75, 0.75],
                                 castshadow=True, type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
        tilt = math.radians(25.0)                     # r1wb.s2.collect.head_camera
        w1, x1, y1, z1 = 0.5, 0.5, -0.5, -0.5
        w2, x2 = math.cos(-0.5 * tilt), math.sin(-0.5 * tilt)
        quat = [w1 * w2 - x1 * x2, w1 * x2 + x1 * w2, y1 * w2 + z1 * x2, z1 * w2 - y1 * x2]
        spec.body('torso_link').add_camera(name='head_camera', pos=[0.12, 0.0, 0.42], quat=quat, fovy=70.0)
        self.model = spec.compile()
        if self.model.nq != core.sim.m.nq or self.model.nv != core.sim.m.nv:
            raise ValueError('SONIC-R1: 화면 모델과 물리 모델의 상태 크기가 다릅니다.')
        self.model.stat.extent = 10.0                   # (dynamic_core.SceneRenderer: parked balls must not stretch the shadow map)
        self.model.vis.map.znear = 0.02 / self.model.stat.extent
        self.model.vis.map.zfar = 200.0 / self.model.stat.extent
        self.model.vis.global_.offwidth = max(int(self.model.vis.global_.offwidth), width)
        self.model.vis.global_.offheight = max(int(self.model.vis.global_.offheight), height)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.head = None
        self.camera = mujoco.MjvCamera()
        self.camera.azimuth, self.camera.elevation, self.camera.distance = 0.0, -15.0, 2.8
        self.root_body = int(self.model.body('pelvis').id)
        self.robot_bodies = {i for i in range(1, self.model.nbody) if not self.model.body(i).name.startswith('ball_')}

    def sync(self, env: int = 0) -> None:
        d = self.core.sim.d
        self.data.qpos[:] = d.qpos
        self.data.qvel[:] = d.qvel
        self.mj.mj_kinematics(self.model, self.data)
        self.mj.mj_camlight(self.model, self.data)

    def scene(self):
        self.camera.lookat[:] = self.data.xpos[self.root_body]
        self.model.stat.center[:] = self.camera.lookat
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.scene

    def render(self) -> np.ndarray:
        self.scene()
        return self.renderer.render()

    def head_image(self) -> np.ndarray:
        """What the head camera sees now (224 x 224 RGB), for the sentence reader."""
        if self.head is None:
            self.head = self.mj.Renderer(self.model, height=224, width=224)
        self.sync()
        self.head.update_scene(self.data, camera='head_camera')
        return self.head.render().copy()

    def pick(self, x: float, y: float):
        """화면 위치(0..1) -> (발사 원점, 조준점, 조준점이 로봇 위인가). dynamic_core.SceneRenderer.pick 과 같은 계산."""
        mujoco = self.mj
        scene = self.scene()
        camera = mujoco.mjv_averageCamera(scene.camera[0], scene.camera[1])
        aspect = self.width / self.height
        up, forward = np.array(camera.up, dtype=float), np.array(camera.forward, dtype=float)
        half = (camera.frustum_top - camera.frustum_bottom) / 2
        direction = forward * camera.frustum_near + np.cross(forward, up) * (2 * x - 1) * half * aspect + up * (1 - 2 * y) * half
        direction /= np.linalg.norm(direction)
        origin = np.array(camera.pos, dtype=float) + direction * 0.25
        target = np.array(camera.pos, dtype=float) + direction * 12.0
        point = np.zeros(3)
        geom = np.zeros(1, dtype=np.int32)
        flex, skin = geom.copy(), geom.copy()
        selected = mujoco.mjv_select(self.model, self.data, mujoco.MjvOption(), aspect, x, 1 - y, scene, point, geom, flex, skin)
        on_robot = False
        if selected >= 0 and 0.4 < np.linalg.norm(point - camera.pos) < 12:
            target = point.copy()
            on_robot = int(geom[0]) >= 0 and int(self.model.geom_bodyid[int(geom[0])]) in self.robot_bodies
        return origin, target, on_robot

    def close(self) -> None:
        self.renderer.close()
        if self.head is not None:
            self.head.close()

