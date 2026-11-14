"""Pinned local policies for the interactive Simulation, separate from R1 deployment."""
import hashlib
import json
import math
from pathlib import Path


class SimulationPolicies:
    def __init__(self, project):
        self.project = Path(project).resolve()
        path = self.project / '.agent/operation_center/simulation_policies.json'
        encoded = path.read_bytes() if path.exists() else b'{"version":1,"actions":{}}'
        document = json.loads(encoded)
        self.manifest_sha256 = hashlib.sha256(encoded).hexdigest()
        self.source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if document.get('version') != 1 or not isinstance(document.get('actions'), dict):
            raise ValueError('Simulation 정책 선택 파일의 형식이 올바르지 않습니다.')
        self.actions = document['actions']
        if set(self.actions) - {'careful_walk', 'safe_crouch'}:
            raise ValueError('지원하지 않는 액션의 Simulation 정책입니다.')
        for action, entry in self.actions.items():
            if not isinstance(entry, dict):
                raise ValueError(f'{action}: Simulation 정책 설정이 올바르지 않습니다.')
            limits = entry.get('command_limits', {})
            if set(limits) - {'forward', 'reverse', 'lateral', 'yaw'} or any(
                type(value) not in (int, float) or not math.isfinite(value) or value <= 0
                for value in limits.values()
            ):
                raise ValueError(f'{action}: Simulation 이동 범위가 올바르지 않습니다.')
            if type(entry.get('heading_hold', False)) is not bool:
                raise ValueError(f'{action}: heading_hold는 true/false여야 합니다.')

    def limits(self, action):
        return self.actions.get(action, {}).get('command_limits', {})

    def heading_hold(self, action):
        """The pinned policy was trained with the student heading hold (contract v2): hold the start heading."""
        return bool(self.actions.get(action, {}).get('heading_hold', False))

    def open(self, action, options):
        """Hash the exact bytes passed to ORT; never silently fall back on a bad pin."""
        entry = self.actions.get(action)
        if entry is None:
            return None
        try:
            run_id, iteration = entry['run_id'], entry['iteration']
            relative = Path(entry['path'])
            path = (self.project / relative).resolve()
            runs = (self.project / '.agent/trainer/runs').resolve()
            if (relative.is_absolute() or not path.is_relative_to(runs)
                    or path.parent.name != run_id or type(iteration) is not int or iteration < 0
                    or path.name != f'policy_{iteration}.onnx'):
                raise ValueError('등록된 학습의 체크포인트 경로가 아닙니다.')
            model = path.read_bytes()
            digest = hashlib.sha256(model).hexdigest()
            if digest != entry['sha256']:
                raise ValueError('정책 파일 해시가 선택한 체크포인트와 다릅니다.')
            evidence = entry['evaluation']
            report_path = (self.project / evidence['path']).resolve()
            if not report_path.is_relative_to(self.project / '.agent/trainer/result/render' / run_id):
                raise ValueError('선택한 학습의 로컬 평가 결과가 아닙니다.')
            encoded_report = report_path.read_bytes()
            report = json.loads(encoded_report)
            execution = report.get('execution', {})
            if (hashlib.sha256(encoded_report).hexdigest() != evidence['sha256']
                    or report.get('policy_sha256', {}).get('candidate') != digest
                    or execution.get('state') != 'completed' or execution.get('worker_errors') != 0
                    or not execution.get('jobs_expected')
                    or execution.get('jobs_completed') != execution['jobs_expected']):
                raise ValueError('선택 정책의 완료된 비교 평가를 확인할 수 없습니다.')
            control = self.project / f'src/motion_manager/{action}_action_manager/config/{action}.json'
            if hashlib.sha256(control.read_bytes()).hexdigest() != entry['control_sha256']:
                raise ValueError('정책을 평가한 제어 설정과 현재 설정이 다릅니다.')
            import onnxruntime as ort
            session = ort.InferenceSession(model, options, providers=['CPUExecutionProvider'])
            inputs, outputs = session.get_inputs(), session.get_outputs()
            expected = entry['input_dimension']
            if (expected not in (810, 927) or len(inputs) != 1 or len(outputs) != 1
                    or inputs[0].name != 'obs' or inputs[0].type != 'tensor(float)'
                    or inputs[0].shape != [1, expected] or outputs[0].name != 'actions'
                    or outputs[0].type != 'tensor(float)' or outputs[0].shape != [1, 24]):
                raise ValueError('정책의 관측·관절 출력 계약이 일치하지 않습니다.')
            return session, {
                'label': entry.get('label', f'{action} · {iteration}'),
                'run_id': run_id, 'iteration': iteration, 'sha256': digest,
                'path': str(path.relative_to(self.project)), 'input_dimension': expected,
                'source': 'evaluated_checkpoint', 'command_limits': self.limits(action),
                'control_sha256': entry['control_sha256'],
                'selection_sha256': self.manifest_sha256, 'selector_sha256': self.source_sha256,
            }
        except (KeyError, OSError, TypeError, ValueError) as error:
            raise ValueError(f'{action} 선택 정책을 불러올 수 없습니다: {error}') from error
