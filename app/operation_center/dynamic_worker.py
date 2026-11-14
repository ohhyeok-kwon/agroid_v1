"""DynamicMode 작업 프로세스: 평가와 같은 전신 스택(dynamic_core)을 Simulation 화면에 연결한다. 로봇 전송은 없다.

simulation_worker 와 같은 통로를 쓴다: 표준 입력으로 JSON 한 줄씩의 명령, 표준 출력으로 상태 JSON 한 줄 + JPEG 바이트.
트레이너의 평가 가상환경(.agent/trainer/rlinf/.venv: mjlab + 목표 모델)에서 실행하고 PC GPU 를 쓴다. 다른 액션을
고르거나 30초 동안 입력이 없으면 끝난다(GPU 를 놓는다). 지형을 바꾸면 서버가 이 프로세스를 다시 띄운다.

입력 -> 지시(27개 어휘 중 하나, dynamic_mode):
  W/A/S/D + 속도 선택   걷기 8방향          우클릭 방향   제자리 회전(먼저 돌고 나서 걷는다)
  자세 버튼            살짝·깊게 앉기, 일어서기   문장        목표 모델(머리 카메라 + 문장 -> 어휘에서 가장 가까운 목표)
마지막 입력이 이긴다: 문장 지시는 다음 키 누름·방향·자세 입력이나 새 문장까지 유지된다.
"""
import argparse
import io
import json
import math
import sys
import threading
import time
from collections import deque
from pathlib import Path

if __name__ == '__main__':
    import ctypes
    import os
    import signal
    owner = os.getppid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)
    if os.getppid() != owner or owner == 1:
        raise SystemExit(0)

PROJECT = Path(__file__).resolve().parents[2]
IDLE_EXIT = 30.0               # s without any input: the process ends and frees the GPU
DECISION = 16                  # control steps between two System 2 decisions (0.32 s)


def emit(state, frame=b''):
    state['frame_bytes'] = len(frame)
    sys.stdout.buffer.write(json.dumps(state, allow_nan=False, separators=(',', ':')).encode() + b'\n' + frame)
    sys.stdout.buffer.flush()


class GoalReader:
    """문장 -> 목표. 목표 모델(2B)을 처음 문장이 올 때 별도 스레드에서 불러오고, 그 뒤로는 그 스레드가 읽는다."""

    def __init__(self, selection, device):
        self.selection, self.device = selection, device
        self.state, self.message = 'off', ''          # off | loading | reading | ready | error
        self.lock = threading.Lock()
        self.work = None                              # (text, image) waiting to be read
        self.result = None                            # (text, instruction, seconds) not yet taken
        self.wake = threading.Event()
        self.thread = None
        self.seconds, self.load_seconds, self.memory = None, None, None

    def available(self):
        return self.selection.get('goal_model') is not None

    def submit(self, text, image):
        with self.lock:
            self.work = (text, image)
            if self.state in ('off', 'error'):
                self.state, self.message = 'loading', ''
            elif self.state == 'ready':
                self.state = 'reading'
        if self.thread is None or not self.thread.is_alive():
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()
        self.wake.set()

    def idle(self):
        with self.lock:
            return self.state == 'ready' and self.work is None

    def take(self):
        with self.lock:
            result, self.result = self.result, None
            return result

    def run(self):
        import torch
        try:
            if getattr(self, 'model', None) is None:
                began = time.perf_counter()
                import dynamic_mode
                from r1wb.s2 import instructions
                from r1wb.s2.small import goal_of
                from r1wb.s2.vla import GoalVLA
                item = self.selection['goal_model']
                dynamic_mode.verify_goal_model(self.selection)
                before = torch.cuda.memory_allocated()
                model = GoalVLA(Path(item['backbone'])).to(self.device).eval()
                weights = torch.load(item['path'], map_location='cpu', weights_only=False)
                model.load_state_dict(weights['state'], strict=False)
                del weights
                self.entries = instructions.every_instruction()
                self.vocabulary = torch.tensor([goal_of(entry) for entry in self.entries], device=self.device)
                self.model = model
                self.load_seconds = time.perf_counter() - began
                self.memory = (torch.cuda.memory_allocated() - before) / 2 ** 30
            while True:
                with self.lock:
                    work, self.work = self.work, None
                    if work is None:
                        self.state = 'ready'
                        self.wake.clear()             # (inside the lock: a submit after this sets it again)
                if work is None:
                    self.wake.wait(timeout=60.0)
                    continue
                with self.lock:
                    self.state = 'reading'
                text, image = work
                began = time.perf_counter()
                with torch.no_grad():
                    packed = self.model.pack([image], [text]).to(self.device)
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        goal = self.model.decide(packed, self.vocabulary).float()
                    index = int(torch.cdist(goal, self.vocabulary).argmin(dim=1)[0])
                    torch.cuda.synchronize()
                self.seconds = time.perf_counter() - began
                with self.lock:
                    self.result = (text, dict(self.entries[index]), self.seconds)
        except Exception as error:                    # the mode keeps running on keys
            with self.lock:
                self.state, self.work = 'error', None
                self.message = str(error) if isinstance(error, ValueError) else f'목표 모델을 쓸 수 없습니다: {error}'


class Inputs:
    """키·방향·자세·문장을 지금의 지시 하나로(dynamic_mode 의 사상). 마지막 입력이 이긴다."""

    def __init__(self):
        import dynamic_mode
        self.rules = dynamic_mode
        self.heading = dynamic_mode.HeadingInput()
        self.reset()

    def reset(self):
        self.source, self.posture, self.speed = 'keys', 'stand', 'normal'
        self.text, self.text_words, self.awaiting, self.pressed, self.note = None, '', False, False, ''
        self.heading.clear()

    def by_hand(self):
        """A key press, a direction or a posture button: it replaces a sentence (also one still being read)."""
        self.source, self.awaiting, self.text = 'keys', False, None

    def key_edge(self, axes):
        pressed = self.rules.walk_instruction(axes, self.speed) is not None
        edge, self.pressed = pressed and not self.pressed, pressed
        if edge:
            self.by_hand()
        return edge

    def sentence(self, words):
        self.text_words, self.awaiting = words, True

    def reading(self, words, instruction):
        """The goal model read `words`. The first reading of a new sentence takes over; later readings of the same
        sentence (with what the camera sees then) replace it while the sentence is still in force."""
        if words != self.text_words or not (self.awaiting or self.source == 'text'):
            return
        self.awaiting, self.source, self.text = False, 'text', instruction
        self.heading.clear()
        kind = instruction['kind']
        self.posture = instruction['depth'] if kind == 'crouch' else ('rise' if kind == 'rise' else 'stand')

    def instruction(self, axes, yaw):
        self.note = ''
        moving = self.rules.walk_instruction(axes, self.speed)
        if self.source == 'text' and self.text is not None:
            return self.text
        if self.posture in ('slight', 'deep'):
            if moving is not None or self.heading.target is not None:
                self.note = '앉은 자세에서는 이동·회전 지시를 보내지 않습니다. ‘일어서기’ 후 움직이세요.'
                self.heading.clear()
            return self.rules.posture_instruction(self.posture)
        turn = self.heading.instruction(yaw)
        if turn is not None:
            return turn
        if moving is not None:
            return moving
        return self.rules.posture_instruction(self.posture)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', default='flat')
    args = parser.parse_args()
    lock = threading.Lock()
    command = {'axes': [0, 0], 'active': True, 'reset': False, 'received': time.monotonic(),
               'camera': {'azimuth': 0, 'elevation': -15, 'distance': 2.8}}
    launches, instructions, postures = deque(maxlen=32), deque(maxlen=4), deque(maxlen=4)
    pending = {'heading': None}
    closed = threading.Event()

    def inputs():
        for line in sys.stdin:
            try:
                message = json.loads(line)
                with lock:
                    if 'launch' in message:
                        launches.append(message.pop('launch'))
                    if 'heading' in message:
                        pending['heading'] = message.pop('heading')
                    if 'instruction' in message:
                        instructions.append(str(message.pop('instruction')))
                    if 'posture' in message:
                        postures.append(message.pop('posture'))
                    message.pop('action', None)
                    message.pop('environment', None)
                    command.update(message)
                    command['received'] = time.monotonic()
            except (ValueError, TypeError):
                pass
        closed.set()
    threading.Thread(target=inputs, daemon=True).start()

    emit({'state': 'loading', 'action': 'dynamic_mode', 'environment': args.environment,
          'message': 'DynamicMode 스택(목표 → 조향 고리 → 운동 토큰 → 추종 정책)을 불러오는 중입니다.'})
    import numpy as np
    import torch
    from PIL import Image
    from r1lab import contract as C
    from r1wb.s2 import instructions as vocabulary
    from r1wb.s2.small import goal_of
    from simulation import PROJECTILE_SETTINGS, validate_launch
    from simulation_worker import terrain

    import dynamic_mode
    from dynamic_core import DynamicCore, SceneRenderer

    selection = dynamic_mode.load_manifest(PROJECT)
    core = DynamicCore(selection, environment=args.environment, envs=1, camera=selection.get('goal_model') is not None)
    renderer = SceneRenderer(core)
    reader = GoalReader(selection, core.device)
    control = Inputs()
    objects = terrain(args.environment)
    joints = list(core.robot.joint_names)
    system1 = selection['system1']
    info = {'label': selection['label'], 'run_id': system1['run_id'], 'iteration': system1['iteration'],
            'parts': ' / '.join(selection[key]['name'] for key in ('head', 'tokens', 'steer', 'library')),
            'goal_model': selection['goal_model']['name'] if selection.get('goal_model') else None,
            'inputs': int(core.tracker.obs_dim), 'selection_sha256': selection['selection_sha256']}
    shots, hits, last_shot, records = 0, 0, None, {}
    instruction = {'kind': 'stand'}
    ratio, message = 1.0, ''
    sim_time, sim_mark, wall_mark = 0.0, 0.0, time.monotonic()
    deadline = time.monotonic()
    first = True
    while not closed.is_set():
        started = time.monotonic()
        with lock:
            current = command.copy()
            command['reset'] = False
            thrown = list(launches); launches.clear()
            typed = list(instructions); instructions.clear()
            posed = list(postures); postures.clear()
            heading, pending['heading'] = pending['heading'], None
        if started - current['received'] > IDLE_EXIT:
            break
        if current['reset']:
            core.reset()
            control.reset()
            shots, hits, last_shot, records = 0, 0, None, {}
            sim_time, sim_mark, wall_mark = 0.0, 0.0, started
            message = ''
        camera_changed = any(getattr(renderer.camera, key) != value for key, value in current['camera'].items())
        for key, value in current['camera'].items():
            setattr(renderer.camera, key, value)
        active = current['active'] and started - current['received'] < 2
        fresh = started - current['received'] < 0.4
        if active:
            axes = current['axes'] if fresh else [0, 0]
            if current.get('speed') in dynamic_mode.SPEEDS:
                control.speed = current['speed']
            if fresh:
                if heading is not None:
                    control.by_hand()
                    control.heading.set(heading)
                for posture in posed:
                    if posture in dynamic_mode.POSTURES:
                        control.by_hand()
                        control.posture = posture
                for words in typed:
                    words = words.strip()[:200]
                    if not words:
                        continue
                    if not reader.available():
                        message = '이 구성에는 목표 모델이 없어 문장 지시를 읽을 수 없습니다.'
                        continue
                    control.sentence(words)
                    message = ''
                    reader.submit(words, core.env.scene['head_camera'].data.rgb[0].cpu().numpy())
                for launch in thrown:
                    try:
                        validate_launch(launch)
                    except ValueError:
                        continue
                    renderer.sync()
                    origin, target, on_robot = renderer.pick(*launch['point'])
                    slot = int(core.ball_used[0]) % len(core.balls)
                    world = lambda point: torch.tensor(point, dtype=torch.float, device=core.device).unsqueeze(0)  # noqa: E731
                    core.throw(torch.zeros(1, dtype=torch.long, device=core.device), world(origin), world(target),
                               torch.tensor([float(launch['speed'])], device=core.device))
                    shots += 1
                    last_shot = {'id': shots, 'speed': launch['speed'], 'momentum': launch['speed'] * PROJECTILE_SETTINGS['mass'],
                                 'time': sim_time, 'hit': False}
                    records[slot] = (last_shot, on_robot)
            control.key_edge(axes)
            read = reader.take()
            if read is not None:
                control.reading(read[0], read[1])
            yaw = float(core.yaw()[0])
            instruction = control.instruction(axes, yaw)
            core.tell(goal_of(instruction))
            core.step()
            sim_time += core.dt
            # a sentence is read again at the decisions, with what the camera sees now (as the evaluation does), when the reader is free
            if control.source == 'text' and control.text is not None and core.steps % DECISION == 0 and reader.idle():
                reader.submit(control.text_words, core.env.scene['head_camera'].data.rgb[0].cpu().numpy())
            for slot, (record, on_robot) in list(records.items()):
                given = float(core.ball_given[slot, 0])
                if not math.isnan(given):
                    if on_robot and given > 0.5 and not record['hit']:
                        record['hit'] = True
                        hits += 1
                    del records[slot]
            if started - wall_mark >= 1.0:
                ratio = (sim_time - sim_mark) / (started - wall_mark)
                sim_mark, wall_mark = sim_time, started
        else:
            sim_mark, wall_mark = sim_time, started
        data = core.robot.data
        position = data.root_link_pos_w[0].tolist()
        quaternion = data.root_link_quat_w[0].tolist()
        told, steered = core.told_and_command()
        alert = message or control.note or (reader.message if reader.state == 'error' else '')
        state = {
            'state': 'running' if active else 'paused', 'action': 'dynamic_mode', 'environment': args.environment,
            'time': round(sim_time, 3), 'tick': core.steps, 'fallen': bool(core.fallen[0]), 'message': alert,
            'policy': None, 'teacher': None, 'unified': None,
            'dynamic': {**info, 'goal': dynamic_mode.describe(instruction), 'goal_id': vocabulary.identifier(instruction),
                        'source': control.source, 'posture': control.posture, 'speed': control.speed,
                        'told': [round(float(v), 3) for v in told[0]], 'command': [round(float(v), 3) for v in steered[0]],
                        'text': {'state': reader.state if reader.available() else 'none', 'words': control.text_words,
                                 'read': dynamic_mode.describe(control.text) if control.text else '',
                                 'awaiting': control.awaiting,
                                 'seconds': None if reader.seconds is None else round(reader.seconds, 3),
                                 'load_seconds': None if reader.load_seconds is None else round(reader.load_seconds, 1)},
                        'realtime': round(ratio, 2)},
            'velocity': data.root_link_lin_vel_w[0].tolist(), 'obstacles': objects,
            'launcher': {'shots': shots, 'hits': hits, 'last': last_shot},
            'render': {'engine': 'MuJoCo', 'width': 960, 'height': 600, 'fps': 50},
            'heading': float(core.yaw()[0]), 'heading_target': control.heading.target,
            'camera': {key: float(getattr(renderer.camera, key)) for key in ('azimuth', 'elevation', 'distance')},
            'projectiles': [{'id': slot, 'position': point, 'hit': False} for slot, point in core.ball_positions()],
            'pose': {'tick': core.steps, 'joints': dict(zip(joints, map(float, data.joint_pos[0]))),
                     'quaternion': quaternion, 'position': position}}
        frame = b''
        if active or camera_changed or first:
            renderer.sync()
            output = io.BytesIO()
            Image.fromarray(renderer.render()).save(output, format='JPEG', quality=85)
            frame, first = output.getvalue(), False
        emit(state, frame)
        deadline += core.dt
        remaining = deadline - time.monotonic()
        if remaining < -core.dt:
            deadline = time.monotonic()            # behind: the wall clock slips, the physics is not touched
        time.sleep(max(0.0, remaining))
    renderer.close()
    core.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'state': 'error', 'action': 'dynamic_mode', 'message': f'DynamicMode: {error}'}), flush=True)
        raise
