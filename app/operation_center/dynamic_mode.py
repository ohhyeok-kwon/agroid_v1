"""DynamicMode: 선택 파일과 입력(키·방향·자세·문장)을 지시 어휘로 옮기는 부분. Simulation 전용이며 R1 배포와는 별개다.

전신 학습 계열(r1wb)의 스택 — 지시 → 목표 → 조향 고리 → 운동 토큰 머리 → 추종 정책(System 1) — 을 평가와 같은 실행기로
돌린다(dynamic_core). 이 파일은 무거운 모듈 없이 불러올 수 있는 부분만 둔다.

  선택 파일   .agent/operation_center/dynamic_mode.json: 구성 요소마다 이름·경로·SHA256 (학습이 같은 이름으로 덮어쓰면 거부)
  입력 사상   W/A/S/D, 우클릭 방향, 자세 버튼 -> r1wb.s2.instructions 의 지시 하나(27개 어휘). 문장은 목표 모델이 읽는다.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

MANIFEST = '.agent/operation_center/dynamic_mode.json'
MANIFEST_KEYS = {'version', 'label', 'system1', 'head', 'tokens', 'steer', 'library', 'goal_model', 'ground_reference',
                 'selected_at', 'note'}
PARTS = ('system1', 'head', 'tokens', 'steer', 'library')          # 시작할 때 해시를 확인하는 파일(목표 모델은 불러올 때)
SPEEDS = ('slow', 'normal', 'fast')
POSTURES = ('stand', 'rise', 'slight', 'deep')
# 방향 지시(우클릭): 이 오차를 넘으면 돌기 시작하고(재시작 폭), 이 오차 안으로 들어오면 멈추라고 지시한다. 지시는 결정
# 주기(0.32초) 한두 번 뒤에 몸에 나타나므로 멈춤 지시 뒤에도 0.6 rad/s 로 약 0.3 rad 더 돈다.
TURN_START, TURN_STOP, TURN_FAST = 0.7, 0.35, 1.3
LABELS = {
    'direction': {'forward': '앞으로', 'backward': '뒤로', 'left': '왼쪽으로', 'right': '오른쪽으로',
                  'forward_left': '왼쪽 앞 대각선으로', 'forward_right': '오른쪽 앞 대각선으로',
                  'backward_left': '왼쪽 뒤 대각선으로', 'backward_right': '오른쪽 뒤 대각선으로'},
    'side': {'left': '왼쪽으로', 'right': '오른쪽으로'},
    'depth': {'slight': '살짝 앉기', 'deep': '깊게 앉기'},
}


def _sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(project, manifest=None, verify=True):
    """선택 파일을 읽고 구성 요소의 경로·해시를 확인한다. 돌려주는 값: 항목별 {'name', 'path'(절대), 'sha256', ...}."""
    project = Path(project).resolve()
    path = project / (manifest or MANIFEST)
    if not path.exists():
        raise ValueError('DynamicMode 선택 파일이 없습니다(.agent/operation_center/dynamic_mode.json).')
    encoded = path.read_bytes()
    entry = json.loads(encoded)
    if not isinstance(entry, dict) or set(entry) - MANIFEST_KEYS or entry.get('version') != 1:
        raise ValueError('DynamicMode 선택 파일의 형식이 올바르지 않습니다.')
    work = (project / '.agent/trainer').resolve()
    out = {'label': entry.get('label', 'DynamicMode'), 'note': entry.get('note', ''),
           'ground_reference': entry.get('ground_reference', 'feet'),
           'selection_sha256': hashlib.sha256(encoded).hexdigest()}
    if out['ground_reference'] not in ('', 'pelvis', 'feet'):
        raise ValueError('DynamicMode: ground_reference 는 "", "pelvis", "feet" 중 하나입니다.')
    for part in PARTS + ('goal_model',):
        item = entry.get(part)
        if part == 'goal_model' and item is None:
            out[part] = None
            continue
        if not isinstance(item, dict) or not {'name', 'path', 'sha256'} <= set(item):
            raise ValueError(f'DynamicMode: {part} 항목에 name·path·sha256 이 필요합니다.')
        relative = Path(item['path'])
        file = (project / relative).resolve()
        if relative.is_absolute() or not file.is_relative_to(work) or not file.is_file():
            raise ValueError(f'DynamicMode: {part} 파일이 트레이너 작업 폴더에 없습니다({item["path"]}).')
        if verify and part in PARTS and _sha256(file) != item['sha256']:
            raise ValueError(f'DynamicMode: {part} 파일 해시가 선택한 것과 다릅니다({item["path"]}).')
        out[part] = {**item, 'path': str(file)}
    system1 = out['system1']
    if 'run_id' not in system1 or 'iteration' not in system1:
        raise ValueError('DynamicMode: system1 항목에 run_id·iteration 이 필요합니다.')
    file = Path(system1['path'])
    if file.parent.name != system1['run_id'] or file.name != f"model_{system1['iteration']}.pt":
        raise ValueError('DynamicMode: system1 이 등록된 학습의 체크포인트 경로가 아닙니다.')
    if out['goal_model'] is not None:
        backbone = out['goal_model'].get('backbone')
        if not backbone or not (project / backbone).resolve().is_relative_to(work) or not (project / backbone).is_dir():
            raise ValueError('DynamicMode: goal_model 의 backbone 폴더가 트레이너 작업 폴더에 없습니다.')
        out['goal_model']['backbone'] = str((project / backbone).resolve())
    return out


def verify_goal_model(selection):
    """목표 모델 가중치(수 GB)의 해시는 불러오기 직전에 확인한다."""
    item = selection['goal_model']
    if _sha256(item['path']) != item['sha256']:
        raise ValueError('DynamicMode: 목표 모델 가중치 해시가 선택한 것과 다릅니다.')


# ---------------------------------------------------------------- 입력 -> 지시(어휘)

def walk_instruction(axes, speed='normal'):
    """W/S, A/D 축(-1..1) -> 걷기 지시 또는 None(입력 없음). 'fast' 는 자료가 있는 전진 계열에서만 쓴다."""
    forward, side = float(axes[0]), float(axes[1])
    x = (forward > 0.3) - (forward < -0.3)
    y = (side > 0.3) - (side < -0.3)
    if not x and not y:
        return None
    name = {(1, 0): 'forward', (-1, 0): 'backward', (0, 1): 'left', (0, -1): 'right', (1, 1): 'forward_left',
            (1, -1): 'forward_right', (-1, 1): 'backward_left', (-1, -1): 'backward_right'}[(x, y)]
    if speed == 'fast' and name not in ('forward', 'forward_left', 'forward_right'):
        speed = 'normal'
    return {'kind': 'walk', 'direction': name, 'speed': speed if speed in SPEEDS else 'normal'}


class HeadingInput:
    """우클릭으로 정한 방향을 향해 돌라는 지시를 만든다(어휘의 제자리 회전 넷 중 하나). 입력 장치의 사상이다."""

    def __init__(self):
        self.target, self.turning = None, 0

    def set(self, heading):
        self.target = None if heading is None else math.remainder(float(heading), 2 * math.pi)

    def clear(self):
        self.target, self.turning = None, 0

    def instruction(self, yaw):
        if self.target is None:
            self.turning = 0
            return None
        error = math.remainder(self.target - float(yaw), 2 * math.pi)
        if self.turning and (abs(error) < TURN_STOP or (error > 0) != (self.turning > 0)):
            self.turning = 0
        if not self.turning and abs(error) > TURN_START:
            self.turning = 1 if error > 0 else -1
        if not self.turning:
            return None
        return {'kind': 'turn', 'side': 'left' if self.turning > 0 else 'right', 'rate': 'fast' if abs(error) > TURN_FAST else 'slow'}


def posture_instruction(posture):
    if posture in ('slight', 'deep'):
        return {'kind': 'crouch', 'depth': posture}
    return {'kind': 'rise'} if posture == 'rise' else {'kind': 'stand'}


def describe(instruction):
    """지시 하나를 화면에 보일 한국어로."""
    kind = instruction['kind']
    if kind == 'walk':
        speed = {'slow': 0.2, 'normal': 0.35, 'fast': 0.5}[instruction['speed']]
        return f"{LABELS['direction'][instruction['direction']]} {speed:g} m/s"
    if kind == 'turn':
        rate = {'slow': 0.6, 'fast': 1.0}[instruction['rate']]
        return f"제자리에서 {LABELS['side'][instruction['side']]} {rate:g} rad/s"
    if kind == 'crouch':
        return LABELS['depth'][instruction['depth']]
    return '일어서기' if kind == 'rise' else '서 있기'
