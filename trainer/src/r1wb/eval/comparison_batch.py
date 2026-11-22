"""Finite server-side evaluation capture; transport states for workstation-only video rendering.

The JSON specification and every child command are retained. A captured result is not a completed video or a
confirmed performance improvement. This runner never trains a policy or renders on the GPU server.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from r1wb.eval.paths import AGENT, ON_SERVER, ROOT


def atomic(path, value):
  temporary = path.with_suffix('.tmp')
  temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
  temporary.replace(path)


def diagnostic_args(job, profile):
  """Only these two job fields may alter impulse diagnostics; command-suite argv stays unchanged."""
  requested = {name: job[name] for name in ('goal_lag_grace', 'arm_response_scale') if name in job}
  if requested and job.get('policy', 's2') == 'manufacturer':
    raise ValueError('Manufacturer jobs cannot receive goal-lag or arm interventions')
  for name, value in requested.items():
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
      raise ValueError(f'{name} must be a finite number')
    if name == 'goal_lag_grace' and value not in (0, 2):
      raise ValueError('goal_lag_grace must be 0 or 2 seconds')
    if name == 'arm_response_scale' and not 0 <= value <= 1:
      raise ValueError('arm_response_scale must be in [0, 1]')
  if profile not in ('impulse_smoke', 'impulse_screen'):
    return []
  return [value for name, number in requested.items() for value in ('--' + name.replace('_', '-'), f'{number:g}')]


def validate(spec):
  if not re.fullmatch(r'[A-Za-z0-9_.-]+', spec.get('batch_id', '')):
    raise ValueError('A unique simple batch_id is required')
  profiles = spec.get('profiles')
  if not profiles or not set(profiles) <= {'impulse_smoke', 'impulse_screen', 'commands'}:
    raise ValueError('Select impulse_smoke, impulse_screen or commands')
  if not isinstance(spec.get('seed'), int) or not spec.get('jobs'):
    raise ValueError('Explicit seed and jobs are required')
  names = []
  for job in spec['jobs']:
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', job.get('name', '')):
      raise ValueError('Invalid job name')
    names.append(job['name'])
    diagnostic_args(job, 'commands')  # Validate even when a job is skipped by a profile filter.
    if any(name in job for name in ('goal_lag_grace', 'arm_response_scale')):
      selected = set(job.get('profiles', profiles)) & set(profiles)
      if not selected & {'impulse_smoke', 'impulse_screen'}:
        raise ValueError('Diagnostic job has no selected impulse profile')
    run = (AGENT / 'runs/r1_wholebody' / job['run']).resolve()
    if run.parent != (AGENT / 'runs/r1_wholebody').resolve():
      raise ValueError('Run must be inside the registered whole-body directory')
    if job.get('policy', 's2') == 'manufacturer':
      from r1wb.eval.manufacturer import inspect_bundle
      inspect_bundle(job['manufacturer_bundle'])
    else:
      checkpoint = run / f"model_{int(job['iteration'])}.pt"
      if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if job.get('steer', 'n18b') not in ('n18b', 'n18', ''):
      raise ValueError('Unsupported steering comparison')
  if len(set(names)) != len(names):
    raise ValueError('Duplicate job name')


def run_batch(spec):
  if not ON_SERVER:
    raise ValueError('comparison_batch captures on the GPU server; no local GPU fallback')
  validate(spec)
  result_root = AGENT / 'result'
  batch_id = spec['batch_id']
  directory = result_root / 'evaluation_batches' / batch_id
  directory.mkdir(parents=True, exist_ok=False)
  protocol = directory / 'protocol.json'
  protocol.write_text(json.dumps(spec, ensure_ascii=False, indent=2))
  progress = dict(batch_id=batch_id, status='capturing', owner_pid=os.getpid(), started_at=time.time(),
                  protocol=str(protocol), protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
                  capture_only=True, video_rendering='workstation pending', jobs=[])
  status = directory / 'status.json'
  atomic(status, progress)
  package_paths = {directory.relative_to(result_root)}
  env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'), PYTHONDONTWRITEBYTECODE='1',
             OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1')
  try:
    for job in spec['jobs']:
      run = AGENT / 'runs/r1_wholebody' / job['run']
      for profile in spec['profiles']:
        if profile not in job.get('profiles', spec['profiles']):
          continue
        tag = f"{batch_id}-{job['name']}-{profile}"
        logs = result_root / 'logs' / run.name
        logs.mkdir(parents=True, exist_ok=True)
        log = logs / (tag + '.log')
        trace = logs / (tag + '.npz')
        manufacturer = job.get('policy', 's2') == 'manufacturer'
        argv = [sys.executable, '-B', '-m', 'r1wb.eval.' + ('eval_stream' if profile == 'commands' else 'impulse_probe'),
                '--run', str(run), '--app-pid', str(os.getpid()), '--seed', str(spec['seed']), '--tag', tag,
                '--protocol', str(protocol), '--archive-compression']
        if not manufacturer:
          argv += ['--checkpoint', str(run / f"model_{int(job['iteration'])}.pt")]
        else:
          argv += ['--manufacturer-bundle', job['manufacturer_bundle']]
        if profile == 'commands':
          argv += ['--policy', 'manufacturer' if manufacturer else 's2', '--head', 'h12', '--tokens', 'v2d', '--suite', 'compass',
                   '--comparison-contract', 'native-r1', '--repeats', str(spec.get('command_repeats', 1)),
                   '--seconds', str(spec.get('command_seconds', 21.0))]
        else:
          argv += ['--task', 'manufacturer' if manufacturer else 'stream_label', '--tokens', 'v2d', '--library', 'corebal',
                   '--sync-twins', '--match-pose', 'reference', '--commit', '--ground-reference', 'feet',
                   '--trace', str(trace), '--comparison-contract', 'native-r1']
          if not manufacturer:
            argv += ['--s2-head', 'h12']
          if profile == 'impulse_smoke':
            argv += ['--twists', '0', '0', '0', '--impulses', '0', '--directions', 'front',
                     '--repeats', '1', '--first-hit', '.8', '--record', '.8']
          else:
            argv += ['--twists', '0', '0', '0', '.35', '0', '0', '--impulses', '12', '25',
                     '--repeats', str(spec.get('impulse_repeats', 3)),
                     '--first-hit', str(spec.get('first_hit', 4.28))]
        if not manufacturer and job.get('steer', 'n18b'):
          argv += ['--steer', job.get('steer', 'n18b')]
        argv += diagnostic_args(job, profile)
        entry = dict(name=job['name'], profile=profile, run=run.name, iteration=int(job['iteration']),
                     command=argv, status='capturing', started_at=time.time(), log=str(log))
        progress['jobs'].append(entry)
        atomic(status, progress)
        with log.open('x') as output:
          child = subprocess.run(argv, cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
        entry.update(returncode=child.returncode, ended_at=time.time())
        package_paths.add(log.relative_to(result_root))
        if child.returncode:
          raise RuntimeError(f"{tag} failed; preserved log: {log}")
        results = list((result_root / 'render' / run.name).glob('*' + tag + '*.json'))
        if len(results) != 1:
          raise RuntimeError(f'Missing or ambiguous result: {tag}')
        result = results[0]
        data = json.loads(result.read_text())
        archive = Path(data['video']['directory'])
        manifest = json.loads((archive / 'manifest.json').read_text())
        if not manifest.get('capture_complete') or manifest['status'] == 'unavailable':
          raise RuntimeError(f'Incomplete state archive: {archive}')
        entry.update(status='captured', result=str(result), archive=str(archive),
                     result_sha256=hashlib.sha256(result.read_bytes()).hexdigest())
        package_paths.update((result.relative_to(result_root), archive.relative_to(result_root)))
        if trace.exists():
          package_paths.add(trace.relative_to(result_root))
        commands = data.get('comparison_contract', {}).get('commands', {}).get('path')
        if commands:
          package_paths.add(Path(commands).relative_to(result_root))
        atomic(status, progress)
    progress.update(status='captured', captured_at=time.time())
    atomic(status, progress)
    transit = result_root / 'transit'
    transit.mkdir(exist_ok=True)
    bundle = transit / (batch_id + '.tar.zst')
    if bundle.exists():
      raise FileExistsError(bundle)
    subprocess.run(['tar', '--zstd', '-cf', str(bundle), '-C', str(result_root),
                    *[str(path) for path in sorted(package_paths)]], check=True)
    with bundle.open('rb') as stream:
      digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    progress.update(status='packed', bundle=str(bundle), bundle_sha256=digest,
                    bundle_bytes=bundle.stat().st_size, finished_at=time.time())
    atomic(status, progress)
    print(json.dumps(progress, ensure_ascii=False), flush=True)
  except BaseException as error:
    progress.update(status='failed', error=f'{type(error).__name__}: {error}', failed_at=time.time())
    atomic(status, progress)
    # Preserve failed/partial captures too; never report their package as completed evaluation.
    try:
      for entry in progress['jobs']:
        tag = entry['command'][entry['command'].index('--tag') + 1]
        video_root = result_root / 'video' / entry['run']
        package_paths.update(path.relative_to(result_root) for path in video_root.glob('*' + tag + '*') if path.is_dir())
        render_root = result_root / 'render' / entry['run']
        package_paths.update(path.relative_to(result_root) for path in render_root.glob('*' + tag + '*') if path.is_file())
        log_root = result_root / 'logs' / entry['run']
        package_paths.update(path.relative_to(result_root) for path in log_root.glob(tag + '*') if path.is_file())
      transit = result_root / 'transit'
      transit.mkdir(exist_ok=True)
      partial = transit / (batch_id + '.partial.tar.zst')
      if not partial.exists():
        subprocess.run(['tar', '--zstd', '-cf', str(partial), '-C', str(result_root),
                        *[str(path) for path in sorted(package_paths) if (result_root / path).exists()]], check=True)
        with partial.open('rb') as stream:
          digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        progress.update(partial_bundle=str(partial), partial_bundle_sha256=digest,
                        partial_bundle_bytes=partial.stat().st_size)
    except BaseException as packaging_error:
      progress['partial_packaging_error'] = f'{type(packaging_error).__name__}: {packaging_error}'
    atomic(status, progress)
    raise


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--spec-json', required=True, help='Explicit finite comparison specification, preserved verbatim')
  args = parser.parse_args()
  run_batch(json.loads(args.spec_json))


if __name__ == '__main__':
  main()
