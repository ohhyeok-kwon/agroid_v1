"""Stream atomic vector-evaluation state chunks, then render without resimulation.

No in-memory frame collection and no physics/RNG changes. Pending is not video
completion. Only local result/video/<run>/ paths are permitted.
"""
import argparse
import hashlib
import fcntl
import importlib.metadata
from datetime import datetime
from zoneinfo import ZoneInfo
import io
import json
import os
import platform
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import zipfile
import numpy as np
from r1wb.eval.paths import ROOT,AGENT

RESERVE=512*1024**2

def file_sha256(path):
  with Path(path).open('rb') as stream:
    return hashlib.file_digest(stream,'sha256').hexdigest()

def copy_verified(source,target,expected):
  """Link/copy immutable content without replacing any existing destination."""
  source,target=Path(source),Path(target)
  if source.is_symlink() or target.is_symlink():raise ValueError('Snapshot files must not be symlinks')
  if not isinstance(expected,str) or len(expected)!=64 or file_sha256(source)!=expected:
    raise ValueError(f'SHA256 differs or is missing: {source}')
  target.parent.mkdir(parents=True,exist_ok=True)
  if not target.exists():
    try:os.link(source,target)
    except FileExistsError:pass
    except OSError:
      try:
        with source.open('rb') as incoming,target.open('xb') as outgoing:shutil.copyfileobj(incoming,outgoing)
      except FileExistsError:pass
  if file_sha256(target)!=expected:raise ValueError(f'Existing/copied SHA256 differs: {target}')
  return target

def _json_atomic(path,value):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix='.import-',delete=False) as stream:
    json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n');temporary=Path(stream.name)
  temporary.replace(path)

def _preserve_bytes(path,payload):
  path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
  try:
    with path.open('xb') as stream:stream.write(payload)
  except FileExistsError:
    if path.read_bytes()!=payload:raise ValueError(f'Preserved source differs: {path}')
  return path

def _safe_name(value):
  if not value or value in ('.','..') or Path(value).name!=value:raise ValueError(f'Invalid artifact name: {value}')
  return value

def _locate_import_file(raw,source,remote,tree,*,extra_name=None):
  raw=PurePosixPath(raw);candidates=[]
  for old,new in ((remote,source),(remote.parents[2],tree),(remote.parent.parent,tree)):
    if raw.is_relative_to(old):candidates.append(new/str(raw.relative_to(old)))
  if extra_name:candidates.append(source/extra_name)
  candidates.append(source/raw.name)
  for candidate in candidates:
    if candidate.is_file() and candidate.resolve().is_relative_to(tree.resolve()):return candidate
  raise FileNotFoundError(f'Artifact absent from import tree: {raw}')

def _result_destination(root,run,name,payload):
  digest=hashlib.sha256(payload).hexdigest();target=root/'render'/run/_safe_name(name)
  if target.exists():
    old=target.read_bytes()
    try:identity=json.loads(old).get('capture_import',{}).get('original_result_sha256')
    except (ValueError,AttributeError):identity=None
    if hashlib.sha256(old).hexdigest()!=digest and identity!=digest:
      target=target.with_name(f'{target.stem}_import-{digest[:16]}{target.suffix}')
      if target.exists() and json.loads(target.read_text()).get('capture_import',{}).get('original_result_sha256')!=digest:
        raise FileExistsError(f'Existing result preserved: {target}')
  return target,digest

def _comparison_commands(*records):
  return [record['comparison_contract']['commands'] for record in records
          if record and record.get('comparison_contract',{}).get('commands')]

def _import_commands(records,result_root,run,locate):
  """Rebase the recorded command stream only; contract source/asset paths are provenance."""
  references=_comparison_commands(*records)
  if not references:return None
  expected=references[0]['sha256']
  if any(entry['sha256']!=expected for entry in references):
    raise ValueError('Commands differ between result and manifest')
  name=_safe_name(PurePosixPath(references[0]['path']).name)
  destination=result_root/'render'/run/name
  alternate=destination.with_name(f'{destination.stem}_import-{expected[:16]}{destination.suffix}')
  try:source=locate(references[0]['path'])
  except FileNotFoundError:
    source=next((path for path in (destination,alternate) if path.is_file() and file_sha256(path)==expected),destination)
  if destination.exists() and file_sha256(destination)!=expected:destination=alternate
  copy_verified(source,destination,expected)
  for entry in references:entry['path']=str(destination)
  return dict(references[0])

def _rebase_imported_commands(commands,*records):
  """Update old imports without replacing their later render/progress metadata."""
  changed=False
  for entry in _comparison_commands(*records):
    if entry['sha256']!=commands['sha256']:raise ValueError('Existing imported commands checksum differs')
    if entry['path']!=commands['path']:entry['path']=commands['path'];changed=True
  return changed

def import_results_tree(tree,result_root):
  """Preserve small remote results first; video readiness waits for the verified archive."""
  tree,result_root=Path(tree).resolve(),Path(result_root).resolve();outputs=[]
  for source in sorted((tree/'render').glob('*/*.json')):
    run=_safe_name(source.parent.name);payload=source.read_bytes();data=json.loads(payload)
    target,digest=_result_destination(result_root,run,source.name,payload)
    original=_preserve_bytes(result_root/'transit'/'original-results'/run/(digest+'.json'),payload)
    def locate_commands(raw):
      path=source.parent/PurePosixPath(raw).name
      if not path.is_file():raise FileNotFoundError(path)
      return path
    commands=_import_commands((data,data.get('video')),result_root,run,locate_commands)
    if target.exists() and json.loads(target.read_text()).get('capture_import',{}).get('original_result_sha256')==digest:
      previous=target.read_bytes();existing=json.loads(previous)
      if commands and _rebase_imported_commands(commands,existing,existing.get('video')):
        _preserve_bytes(original.parent/(hashlib.sha256(previous).hexdigest()+'.before-commands.json'),previous)
        _json_atomic(target,existing)
      outputs.append(str(target));continue
    data['capture_import']=dict(status='awaiting_verified_archive',original_result=str(original),original_result_sha256=digest)
    if target.exists():_preserve_bytes(original.parent/(file_sha256(target)+'.previous.json'),target.read_bytes())
    _json_atomic(target,data);outputs.append(str(target))
  for source in sorted((tree/'logs').glob('*/*')):
    if not source.is_file():continue
    target=result_root/'logs'/_safe_name(source.parent.name)/source.name;digest=file_sha256(source)
    if target.exists() and file_sha256(target)!=digest:target=target.with_name(f'{target.stem}_import-{digest[:16]}{target.suffix}')
    copy_verified(source,target,digest)
  return outputs

def import_archive(source,result_root,*,tree=None):
  """Verify a server capture, preserve its original JSON, and rebase only operational paths."""
  source,result_root=Path(source).resolve(),Path(result_root).resolve();tree=Path(tree or source).resolve()
  if any(path.is_symlink() for path in source.rglob('*')):raise ValueError('Import source contains an archive symlink')
  payload=(source/'manifest.json').read_bytes();manifest=json.loads(payload);remote=PurePosixPath(manifest['directory'])
  if remote.parent.parent.name!='video':raise ValueError('Expected remote video/<run>/<archive> layout')
  run,name=_safe_name(remote.parent.name),_safe_name(remote.name)
  target=result_root/'video'/run/name
  if not target.resolve().is_relative_to(result_root):raise ValueError('Archive destination escapes result root')
  target.mkdir(parents=True,exist_ok=True)
  with (target/'import.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    digest=hashlib.sha256(payload).hexdigest();existing=None
    if (target/'manifest.json').exists():
      existing=json.loads((target/'manifest.json').read_text())
      if existing.get('capture_import',{}).get('original_manifest_sha256')!=digest:
        raise FileExistsError(f'Existing archive preserved: {target}')
    original=_preserve_bytes(target/f'manifest.original-{digest}.json',payload)
    hashes={}
    def install(path,relative,expected):
      destination=copy_verified(path,target/relative,expected);hashes[str(relative)]=expected;return str(destination)
    if PurePosixPath(manifest['model_path'])!=remote/'model.mjb' or PurePosixPath(manifest['state_path'])!=remote/'states':
      raise ValueError('Remote model/state paths are not archive-local')
    manifest['model_path']=install(source/'model.mjb',Path('model.mjb'),manifest['model_sha256'])
    chunks=manifest['chunk_sha256'];names=sorted(path.name for path in (source/'states').glob('*.npz'))
    if names!=[f'{i:08d}.npz' for i in range(manifest['frames'])] or set(names)!=set(chunks):
      raise ValueError('State chunks/count/checksums differ')
    for name in names:install(source/'states'/name,Path('states')/name,chunks[name])
    for entry in manifest.get('source_snapshot',{}).values():
      path=_locate_import_file(entry['snapshot'],source,remote,tree)
      entry['snapshot']=install(path,Path('sources')/path.name,entry['sha256'])
    for entry in manifest.get('model_assets',[]):
      path=_locate_import_file(entry['snapshot'],source,remote,tree)
      entry['snapshot']=install(path,Path('assets')/(entry['sha256']+path.suffix),entry['sha256'])
    policy=manifest.get('policy')
    if policy:
      path=_locate_import_file(policy['shared_file'],source,remote,tree)
      policy['shared_file']=install(path,Path('assets')/(policy['sha256']+path.suffix),policy['sha256'])
    localization=manifest.get('asset_localization')
    if localization:
      path=_locate_import_file(localization['original_manifest'],source,remote,tree)
      localization['original_manifest']=install(path,Path(path.name),localization['sha256'])
    result=None;result_target=None;result_digest=None
    if manifest.get('result_json'):
      try:result_path=_locate_import_file(manifest['result_json'],source,remote,tree,extra_name='result.json')
      except FileNotFoundError:
        prior=result_root/'render'/run/PurePosixPath(manifest['result_json']).name
        record=json.loads(prior.read_text()).get('capture_import',{})
        result_path=Path(record['original_result'])
        if file_sha256(result_path)!=record['original_result_sha256']:raise ValueError('Preserved result checksum differs')
      result_payload=result_path.read_bytes();result=json.loads(result_payload)
      for key in ('model_sha256','chunk_sha256','frames','directory'):
        if result.get('video',{}).get(key)!=json.loads(payload)[key]:raise ValueError(f'Result belongs to another capture: {key}')
      result_target,result_digest=_result_destination(result_root,run,PurePosixPath(manifest['result_json']).name,result_payload)
      _preserve_bytes(target/f'result.original-{result_digest}.json',result_payload)
      manifest['result_json']=str(result_target)
    trace=manifest.get('trace') or (result.get('trace') if result else None)
    if trace:
      try:path=_locate_import_file(trace['path'],source,remote,tree)
      except FileNotFoundError:
        path=result_root/'logs'/run/PurePosixPath(trace['path']).name
      trace_path=result_root/'logs'/run/path.name
      if trace_path.exists() and file_sha256(trace_path)!=trace['sha256']:
        trace_path=trace_path.with_name(f'{trace_path.stem}_import-{trace["sha256"][:16]}{trace_path.suffix}')
      copy_verified(path,trace_path,trace['sha256'])
      trace=dict(trace,path=str(trace_path));manifest['trace']=trace
      if result is not None and result.get('trace'):
        if result['trace']['sha256']!=trace['sha256']:raise ValueError('Trace differs between result and manifest')
        result['trace']=dict(result['trace'],path=str(trace_path))
    commands=_import_commands((manifest,result,result.get('video') if result else None),result_root,run,
                              lambda raw:_locate_import_file(raw,source,remote,tree))
    # A proper actor trace has an independent manifest: verify its declared
    # checksums, not merely the bytes found by the generic preservation pass.
    def actor_references(*records):
      return [record['proper_evaluation']['actor_trace'] for record in records
              if record and record.get('proper_evaluation',{}).get('actor_trace')]
    actor_refs=actor_references(manifest,result,result.get('video') if result else None)
    actor_trace=None
    if actor_refs:
      actor_digest=actor_refs[0]['sha256']
      if any(entry['sha256']!=actor_digest for entry in actor_refs):
        raise ValueError('Actor input traces differ between result and manifest')
      actor_remote=PurePosixPath(actor_refs[0]['path'])
      if actor_remote!=remote/'actor_inputs'/'manifest.json':
        raise ValueError('Actor input manifest must be archive-local')
      actor_source=source/'actor_inputs';actor_manifest=actor_source/'manifest.json'
      if file_sha256(actor_manifest)!=actor_digest:raise ValueError('Actor input manifest checksum differs')
      actor=json.loads(actor_manifest.read_text())
      if actor.get('schema')!='r1wb-actor-input-trace-v1' or actor.get('state')!='complete':
        raise ValueError('Actor input trace is not complete')
      count=len(actor.get('metadata',{}).get('noise_seeds',[]))
      if count!=len(manifest['environments']):raise ValueError('Actor trace environment coverage differs')
      expected_step=0;declared=set()
      for entry in actor['chunks']:
        name=_safe_name(entry['path']);declared.add(name)
        installed=install(actor_source/name,Path('actor_inputs')/name,entry['sha256'])
        with np.load(installed,allow_pickle=False) as values:
          steps=values['step'];frames=int(entry['frames'])
          if (frames<=0 or not np.array_equal(steps,np.arange(expected_step,expected_step+frames))
              or entry['first_step']!=expected_step or entry['last_step']!=expected_step+frames-1):
            raise ValueError('Actor input frame gap/count mismatch')
          if values['actor_input'].shape[:2]!=(frames,count) or values['actor_clean'].shape!=values['actor_input'].shape:
            raise ValueError('Actor input tensor coverage differs')
          expected_step+=frames
      if (expected_step!=actor['frames'] or expected_step!=actor['expected_steps']
          or expected_step!=actor_refs[0]['frames']):raise ValueError('Actor trace expected horizon differs')
      outcome=actor['outcomes'];name=_safe_name(outcome['path']);declared.add(name)
      installed=install(actor_source/name,Path('actor_inputs')/name,outcome['sha256'])
      with np.load(installed,allow_pickle=False) as values:
        if outcome['frames']!=expected_step or not np.array_equal(values['step'],np.arange(expected_step)):
          raise ValueError('Actor outcome frame coverage differs')
        for key in ('done','physical_terminated','first_physical_fall','alive_after'):
          if values[key].shape!=(expected_step,count):raise ValueError('Actor outcome environment coverage differs')
      if {p.name for p in actor_source.glob('*.npz')}!=declared:
        raise ValueError('Actor trace contains undeclared or missing NPZ chunks')
      actor_local=install(actor_manifest,Path('actor_inputs/manifest.json'),actor_digest)
      actor_trace=dict(actor_refs[0],path=actor_local)
      for entry in actor_refs:entry['path']=actor_local
    def rebase_actor_trace(*records):
      changed=False
      if actor_trace is not None:
        for entry in actor_references(*records):
          if entry['sha256']!=actor_trace['sha256']:raise ValueError('Existing actor trace checksum differs')
          if entry['path']!=actor_trace['path']:entry['path']=actor_trace['path'];changed=True
      return changed
    for path in source.rglob('*'):
      if path.is_symlink():raise ValueError(f'Archive symlink is not self-contained: {path}')
      if path.is_file() and path!=source/'manifest.json':
        relative=path.relative_to(source)
        if str(relative) not in hashes:install(path,relative,file_sha256(path))
    manifest.update(directory=str(target),state_path=str(target/'states'))
    manifest['capture_import']=dict(status='verified',original_manifest=str(original),original_manifest_sha256=digest,
      original_result_sha256=result_digest,artifact_sha256=hashes)
    if existing is not None:
      if commands or actor_trace:
        updated=bool(commands and _rebase_imported_commands(commands,existing))
        updated=rebase_actor_trace(existing) or updated
        if result_target is not None and result_target.exists():
          previous=result_target.read_bytes();local_result=json.loads(previous)
          if local_result.get('capture_import',{}).get('original_result_sha256')!=result_digest:
            raise ValueError('Existing imported result identity differs')
          result_updated=bool(commands and _rebase_imported_commands(commands,local_result,local_result.get('video')))
          result_updated=rebase_actor_trace(local_result,local_result.get('video')) or result_updated
          if result_updated:
            _preserve_bytes(target/f'result.before-commands-{hashlib.sha256(previous).hexdigest()}.json',previous)
            _json_atomic(result_target,local_result)
        if updated:
          previous=(target/'manifest.json').read_bytes()
          _preserve_bytes(target/f'manifest.before-commands-{hashlib.sha256(previous).hexdigest()}.json',previous)
          _json_atomic(target/'manifest.json',existing)
      return existing  # Preserve all later local render/progress metadata on repeated imports.
    if result is not None:
      if result_target.exists():_preserve_bytes(target/f'result.previous-{file_sha256(result_target)}.json',result_target.read_bytes())
      result['video']=manifest
      result['capture_import']=dict(status='verified',original_result=str(target/f'result.original-{result_digest}.json'),
                                    original_result_sha256=result_digest,archive=str(target))
      _json_atomic(result_target,result)
    _json_atomic(target/'manifest.json',manifest)
    return manifest

def import_archive_tree(tree,result_root):
  tree=Path(tree).resolve();results=[]
  for path in sorted(tree.rglob('manifest.json')):
    if (path.parent/'states').is_dir():results.append(import_archive(path.parent,result_root,tree=tree))
  if not results:raise ValueError('No captured vector archives in import tree')
  return results

def import_bundle(bundle,result_root,expected_sha256=None):
  """Extract into a fresh retained staging area; no server/local originals are deleted."""
  bundle,result_root=Path(bundle).resolve(),Path(result_root).resolve();digest=file_sha256(bundle)
  if expected_sha256 is not None and digest!=expected_sha256:raise ValueError('Transferred bundle checksum differs')
  transit=result_root/'transit';transit.mkdir(parents=True,exist_ok=True)
  staging=Path(tempfile.mkdtemp(prefix='import-',dir=transit));tree=staging/'extracted';tree.mkdir()
  receipt=dict(bundle=str(bundle),sha256=digest,server_sha_verified=expected_sha256 is not None,status='extracting',tree=str(tree))
  _json_atomic(staging/'receipt.json',receipt)
  try:
    if bundle.name.endswith('.zst'):
      with subprocess.Popen(['zstd','-d','--long=27','-q','-c',str(bundle)],stdout=subprocess.PIPE) as process:
        try:
          with tarfile.open(fileobj=process.stdout,mode='r|') as archive:archive.extractall(tree,filter='data')
        finally:process.stdout.close()
        if process.wait()!=0:raise ValueError('Bundle decompression failed')
    else:
      with tarfile.open(bundle) as archive:archive.extractall(tree,filter='data')
    imported=import_archive_tree(tree,result_root)
    receipt.update(status='verified',archives=[m['directory'] for m in imported]);return receipt
  except BaseException as error:
    receipt.update(status='partial',reason=f'{type(error).__name__}: {error}');raise
  finally:_json_atomic(staging/'receipt.json',receipt)

def _localize_asset_snapshots(directory,manifest):
  localized=[]
  for asset in manifest.get('model_assets',[]):
    source=Path(asset['snapshot']);expected=asset['sha256']
    target=copy_verified(source,directory/'assets'/(expected+source.suffix),expected)
    entry=dict(asset);entry.setdefault('original_snapshot',str(source));entry['snapshot']=str(target)
    localized.append(entry)
  policy=manifest.get('policy')
  if policy:
    source=Path(policy['shared_file']);expected=policy['sha256']
    target=copy_verified(source,directory/'assets'/(expected+source.suffix),expected)
    policy.setdefault('original_shared_file',str(source));policy['shared_file']=str(target)
  manifest['model_assets']=localized
  return localized

def localize_archive_tree(tree):
  """Prepare old captures for packing while retaining their exact pre-localization manifests."""
  outputs=[]
  for path in sorted(Path(tree).rglob('manifest.json')):
    if not (path.parent/'states').is_dir():continue
    payload=path.read_bytes();manifest=json.loads(payload)
    if not manifest.get('capture_complete'):
      pid=int(manifest.get('capture_pid',-1))
      if pid>1 and manifest.get('capture_host',platform.node())==platform.node():
        try:os.kill(pid,0)
        except OSError:pass
        else:raise RuntimeError(f'Incomplete capture is still live: {path}')
    digest=hashlib.sha256(payload).hexdigest()
    _localize_asset_snapshots(path.parent,manifest)
    if manifest==json.loads(payload):
      outputs.append(str(path.parent));continue
    backup=_preserve_bytes(path.parent/f'manifest.before-assets-{digest}.json',payload)
    manifest['asset_localization']=dict(original_manifest=str(backup),sha256=digest)
    _json_atomic(path,manifest);outputs.append(str(path.parent))
  return outputs

def array(value):
  return value.detach().cpu().numpy() if hasattr(value,'detach') else np.asarray(value)

def destination(onnx,out=None):
  # State capture may run where the evaluation runs (since 2026-11-12 the GPU server); the archive is then pulled to
  # the workstation (scripts/06_evaluate.sh pull) and only rendered there (_render stays workstation-only).
  path=Path(onnx).resolve();run=path.parent.name
  if out:
    parts=Path(out).resolve().parts
    if 'render' in parts and len(parts)>parts.index('render')+1:run=parts[parts.index('render')+1]
  run=''.join(c if c.isalnum() or c in '-_@.' else '_' for c in run)
  stamp=datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%d-%H%M%S-%f')
  return AGENT/'result/video'/run/f'{stamp}_{Path(out).stem if out else "vector_eval"}'

class VectorVideoArchive:
  def __init__(self,env,directory,labels,min_free_bytes=RESERVE,*,compressed=False):
    self.env=env;self.directory=Path(directory);self.labels=labels;self.reserve=min_free_bytes
    self.compressed=bool(compressed)
    self.episodes=np.zeros(env.num_envs,dtype=np.int32);self.step=0;self.started=False;self.archive=None
    self.manifest=dict(protocol_version=2,capture_pid=os.getpid(),status='pending',reason='State capture awaiting local rendering',
        directory=str(self.directory),state_path=str(self.directory/'states'),model_path=str(self.directory/'model.mjb'),
        step_dt=env.step_dt,frames=0,capture_complete=False,chunk_sha256={},capture_host=platform.node(),
        state_compression='npz_deflate' if self.compressed else 'npz_store',
        sampling_scope='qpos/qvel/ctrl/action/command/wrench at every policy tick plus pre-reset terminal; no substep forces, contact impulses or RNG stream. DR model fields sampled at the same ticks.',
        replay_scope='Exact stored visual states at policy ticks and pre-reset terminals; physics replay is not exact: substep contacts/noise RNG are not recorded.',environments=[dict(env_id=i,command=label,videos=[],status='pending') for i,label in enumerate(labels)])
    self.original_reset=env._reset_idx
    def reset(env_ids=None):
      if self.started:
        ids=np.arange(env.num_envs) if env_ids is None else np.arange(env.num_envs)[env_ids] if isinstance(env_ids,slice) else array(env_ids).astype(int)
        self.capture('terminal_pre_reset',ids)
        self.episodes[ids]+=1
      return self.original_reset(env_ids)
    env._reset_idx=reset

  def begin(self):
    try:
      root=(AGENT/'result/video').resolve()
      if not self.directory.resolve().is_relative_to(root):
        raise ValueError('Archive must be under the work area result/video')
      self.directory.mkdir(parents=True,exist_ok=False)
      if shutil.disk_usage(self.directory).free<self.reserve:raise OSError('Insufficient disk space (512MiB reserve)')
      import mujoco
      self.manifest['libraries']={'mujoco':mujoco.__version__,'numpy':np.__version__,
          'python':platform.python_version(),'platform':platform.platform()}
      for package in ('torch','mujoco-warp','mjlab','onnxruntime'):
        try:self.manifest['libraries'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:pass
      sources={}
      source_files=[Path(__file__),Path(__file__).with_name('eval_policy.py'),
          ROOT/'src/r1lab/mdp/observations.py',ROOT/'src/r1lab/tasks/careful_walk.py',
          ROOT/'src/r1lab/terrain_observation.py',ROOT/'src/r1lab/contract.py',
          ROOT/'src/r1lab/robot.py',ROOT/'src/r1lab/tasks/common.py',
          ROOT/'src/r1lab/mdp/events.py',ROOT/'src/r1lab/deploy_runtime.py']
      evaluator=self.manifest.get('evaluator')
      if isinstance(evaluator,str):source_files.append(Path(evaluator))
      for source in dict.fromkeys(source_files):
        if source.is_file():
          content=source.read_bytes();digest=hashlib.sha256(content).hexdigest()
          saved=self.directory/'sources'/(digest+'_'+source.name);saved.parent.mkdir(exist_ok=True);saved.write_bytes(content)
          sources[str(source)]={'sha256':digest,'snapshot':str(saved)}
      self.manifest['source_snapshot']=sources
      model=self.env.sim.mj_model
      self.manifest['physics']={k:float(getattr(model.opt,k)) for k in ('timestep','integrator','iterations','ls_iterations')}
      self.manifest['joint_mapping']=[dict(name=model.joint(i).name,qpos_address=int(model.jnt_qposadr[i]),dof_address=int(model.jnt_dofadr[i])) for i in range(model.njnt)]
      policy=self.manifest.get('evaluation_args',{}).get('onnx')
      if policy:
        source=Path(policy);digest=hashlib.sha256(source.read_bytes()).hexdigest()
        assets=AGENT/'result/video/_assets';assets.mkdir(parents=True,exist_ok=True)
        target=assets/(digest+'.onnx')
        if not target.exists():shutil.copyfile(source,target)
        local=self.directory.parent/'assets'/target.name;local.parent.mkdir(exist_ok=True)
        if not local.exists():
          try:os.link(target,local)
          except OSError:shutil.copyfile(target,local)
        self.manifest['policy']={'sha256':digest,'shared_file':str(local),'deduplicated_source':str(target),'original_file':str(source.resolve())}
      mujoco.mj_saveModel(self.env.sim.mj_model,str(self.directory/'model.mjb'),None)
      self.manifest['model_sha256']=hashlib.sha256((self.directory/'model.mjb').read_bytes()).hexdigest()
      (self.directory/'states').mkdir()
      self.started=True;self.capture('initial')
      (self.directory/'manifest.json').write_text(json.dumps(self.manifest,indent=2))
    except Exception as error:self.fail(error)

  def localize_assets(self):
    """After begin(), make model/policy snapshots self-contained and verify every byte hash."""
    if not self.started:raise RuntimeError('localize_assets requires a successfully started archive')
    try:
      localized=_localize_asset_snapshots(self.directory,self.manifest)
      temporary=self.directory/'manifest.tmp'
      temporary.write_text(json.dumps(self.manifest,indent=2));temporary.replace(self.directory/'manifest.json')
      return localized
    except Exception as error:
      self.fail(error)
      temporary=self.directory/'manifest.tmp'
      temporary.write_text(json.dumps(self.manifest,indent=2));temporary.replace(self.directory/'manifest.json')
      raise

  def fail(self,error):
    self.manifest.update(status='unavailable',reason=f'{type(error).__name__}: {error}')
    for row in self.manifest['environments']:row.update(status='unavailable',reason=self.manifest['reason'])
    self.started=False

  def capture(self,stage='post_step',terminal_ids=None):
    if not self.started:return
    try:
      if shutil.disk_usage(self.directory).free<self.reserve:raise OSError('Insufficient disk space; state capture stopped, existing files retained')
      data=self.env.sim.data
      values=dict(qpos=array(data.qpos),qvel=array(data.qvel),episodes=self.episodes.copy(),
          terminal_ids=np.asarray([] if terminal_ids is None else terminal_ids,dtype=np.int32),stage=np.asarray(stage),step=np.asarray(self.step+(stage=="terminal_pre_reset")))
      for field in ('ctrl','xfrc_applied','qfrc_applied','qfrc_actuator','actuator_force'):
        value=getattr(data,field,None)
        if value is not None:values[field]=array(value)
      manager=getattr(self.env,'command_manager',None)
      if manager is not None:
        for command in ('twist','crouch'):
          try:values['command_'+command]=array(manager.get_command(command))
          except (KeyError,ValueError):pass
      action=getattr(getattr(self.env,'action_manager',None),'action',None)
      if action is not None:values['policy_action']=array(action)
      # Preserve per-world DR, including state changes at episode reset.
      for field in ('body_mass','body_inertia','body_ipos','body_iquat','dof_armature','dof_damping','jnt_stiffness','geom_friction','actuator_gainprm','actuator_biasprm'):
        value=getattr(self.env.sim.model,field,None)
        if value is not None:values['model_'+field]=array(value)
      # Preserve visually mutable per-world geometry, when present/expanded.
      for name in ('geom_pos','geom_quat','geom_size','geom_rgba'):
        field=getattr(self.env.sim.model,name,None)
        if field is not None:values[name]=array(field)
      buffer=io.BytesIO()
      (np.savez_compressed if self.compressed else np.savez)(buffer,**values)
      path=self.directory/'states'/f'{self.manifest["frames"]:08d}.npz'
      payload=buffer.getvalue()
      temporary=path.with_suffix('.tmp');temporary.write_bytes(payload);temporary.replace(path)
      self.manifest['chunk_sha256'][path.name]=hashlib.sha256(payload).hexdigest()
      self.manifest['frames']+=1
      self.manifest['recorded_end_s']=float(values['step'])*self.env.step_dt
      manifest_tmp=self.directory/'manifest.tmp'
      manifest_tmp.write_text(json.dumps(self.manifest,indent=2));manifest_tmp.replace(self.directory/'manifest.json')
    except Exception as error:self.fail(error)

  def after_step(self):
    self.step+=1;self.capture()

  def close(self):
    self.env._reset_idx=self.original_reset
    if self.archive is not None:
      try:self.archive.close()
      except OSError as error:self.fail(error)
    self.manifest['capture_complete']=bool(self.started)
    if self.directory.is_dir():
      try:(self.directory/'manifest.json').write_text(json.dumps(self.manifest,indent=2))
      except OSError as error:self.fail(error)
    return self.manifest


def _render(directory,owner):
  """Render saved qpos only; never call mj_step or a policy."""
  from r1wb.eval.evaluation_video import configure_local_video,EvaluationVideoRecorder
  if str(ROOT).startswith('/rl-agent'):raise ValueError('Rendering is workstation-only; the server only captures states')
  configure_local_video()
  os.environ["RL_EVALUATION_APP_PID"]=str(owner.pid)
  import mujoco
  directory=Path(directory);manifest=json.loads((directory/'manifest.json').read_text())
  if not directory.resolve().is_relative_to((AGENT/'result/video').resolve()):raise ValueError('Not a local evaluation archive')
  if manifest.get('model_sha256') and hashlib.sha256(Path(manifest['model_path']).read_bytes()).hexdigest()!=manifest['model_sha256']:raise ValueError('Compiled model checksum mismatch')
  model=mujoco.MjModel.from_binary_path(manifest['model_path']);data=mujoco.MjData(model)
  names=sorted(Path(manifest['state_path']).glob('*.npz'))
  if not names:raise ValueError('No recoverable atomic state chunks')
  for index,path in enumerate(names):
    checksum=manifest.get('chunk_sha256',{}).get(path.name)
    if checksum and hashlib.sha256(path.read_bytes()).hexdigest()!=checksum:raise ValueError('State chunk checksum mismatch: '+path.name)
    if path.stem!=f'{index:08d}':raise ValueError('Gap in saved state chunks')
  manifest['recovered_chunks']=len(names)
  with np.load(names[-1],allow_pickle=False) as last:
    manifest['recoverable_end_s']=float(last['step'])*manifest['step_dt']
  expected=manifest.get('evaluation_args',{}).get('seconds')
  manifest['evaluation_complete']=bool(manifest.get('capture_complete') and not manifest.get('evaluation_interrupted') and (expected is None or manifest['recoverable_end_s']>=expected-manifest['step_dt']))
  for row in manifest['environments']:
    if row['status']=='written':continue
    recorder=None;episode=None
    try:
      for name in names:
        if not owner.alive():raise RuntimeError('Operation Center closed; rendering remains pending')
        with np.load(name,allow_pickle=False) as state:
          i=row['env_id'];current=int(state['episodes'][i])
          if str(state['stage'])=='terminal_pre_reset' and i not in state['terminal_ids']:continue
          if current!=episode:
            if recorder is not None:row['videos'].append(recorder.close())
            episode=current
            path=directory.parent/f'{directory.name}_env{i:04d}_episode{episode:04d}.mp4'
            if path.exists():path=path.with_stem(path.stem+'_retry'+datetime.now().strftime('%H%M%S%f'))
            recorder=EvaluationVideoRecorder(model,data,path,
                fps=1/manifest['step_dt'],label=f'env{i} {row["command"]} episode{episode}',
                root_position=lambda:data.qpos[:3],record_replay=False,
                metadata={'visual_replay_only':True,'source_state_manifest':str(directory/'manifest.json')})
          for field in ('geom_pos','geom_quat','geom_size','geom_rgba'):
            if field not in state:continue
            value=state[field];target=getattr(model,field)
            target[:]=value[i if len(value)>1 else 0] if value.ndim==target.ndim+1 else value
          data.qpos[:]=state['qpos'][i];data.qvel[:]=state['qvel'][i]
          data.time=float(state['step'])*manifest['step_dt']
          mujoco.mj_forward(model,data)
          recorder.capture(force=True)
      if recorder is not None:row['videos'].append(recorder.close());recorder=None
      row['status']='written' if row['videos'] and all(v['status']=='written' for v in row['videos']) else 'partial'
    finally:
      if recorder is not None:row['videos'].append(recorder.close())
      (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
  manifest['status']='written' if all(r['status']=='written' for r in manifest['environments']) else 'partial'
  manifest['render_complete']=manifest['status']=='written'
  if manifest['render_complete'] and not manifest['evaluation_complete']:manifest['status']='partial'
  manifest['reason']=None if manifest['status']=='written' else 'Partial evaluation or video; see recoverable_end_s and per-environment metadata'
  (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
  source=manifest.get('result_json')
  if source and Path(source).is_file():
    result=json.loads(Path(source).read_text());result['video']=manifest
    temporary=Path(source).with_suffix('.video-tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False));temporary.replace(source)
  return manifest

def render(directory,owner):
  with (Path(directory)/'render.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return _render(directory,owner)


def attach_rendered_video(directory,metadata):
  """Attach an already-written env0 demo; never re-encode the same trajectory."""
  path=Path(directory)/'manifest.json';manifest=json.loads(path.read_text())
  if len(manifest['environments'])!=1:raise ValueError('One primary demo can only cover one environment')
  manifest['environments'][0].update(videos=[metadata],status=metadata['status'])
  manifest['render_complete']=metadata['status']=='written'
  manifest['evaluation_complete']=bool(manifest.get('capture_complete') and not manifest.get('evaluation_interrupted'))
  manifest['status']='written' if manifest['evaluation_complete'] and manifest['render_complete'] else 'partial'
  manifest['reason']=metadata.get('reason') or (None if manifest['status']=='written' else 'Evaluation capture interrupted/incomplete or video incomplete')
  path.write_text(json.dumps(manifest,indent=2));return manifest


def resume_pending_archives(result_root,run_id=None):
  """App-owned cycle API. Returns per-archive status; leaves existing data intact."""
  root=Path(result_root);root=root if root.name=='video' else root/'video'
  if root.resolve()!=(AGENT/'result/video').resolve():raise ValueError('Local canonical video root required')
  if run_id:
    if Path(run_id).name!=run_id:raise ValueError('Invalid run ID')
    root=root/run_id
  results=[]
  for path in sorted(root.glob('**/manifest.json')):
    try:
      manifest=json.loads(path.read_text())
      if 'environments' not in manifest or manifest.get('render_complete'):continue
      if manifest.get('status') not in ('pending','partial'):continue
      if not manifest.get('capture_complete'):
        try:os.kill(int(manifest.get('capture_pid',-1)),0)
        except (ProcessLookupError,ValueError):pass
        else:continue  # Do not race a live evaluation capture.
      result=render_default(path.parent)
      results.append({'manifest':str(path),'status':result['status']})
    except Exception as error:results.append({'manifest':str(path),'status':'pending','reason':str(error)})
  return results


def render_default(directory):
  """App-owned default path. On interruption archive stays available for resume."""
  from r1wb.eval.evaluation_video import require_app_owner
  from eval_learned_adversary import AppOwner
  pid=require_app_owner(bind=True)
  owner=AppOwner(pid);owner.start()
  try:return render(directory,owner)
  finally:owner.close()

if __name__=='__main__':
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path,nargs='?');parser.add_argument('--owner-pid',type=int)
  operation=parser.add_mutually_exclusive_group()
  operation.add_argument('--import-bundle',type=Path)
  operation.add_argument('--import-tree',type=Path)
  operation.add_argument('--import-results',type=Path)
  operation.add_argument('--localize-tree',type=Path)
  parser.add_argument('--result-root',type=Path,default=AGENT/'result')
  parser.add_argument('--sha256')
  args=parser.parse_args()
  if args.sha256 and not args.import_bundle:parser.error('--sha256 requires --import-bundle')
  if args.import_bundle:
    print(json.dumps(import_bundle(args.import_bundle,args.result_root,args.sha256)))
  elif args.import_tree:
    print(json.dumps({'status':'verified','archives':[m['directory'] for m in import_archive_tree(args.import_tree,args.result_root)]}))
  elif args.import_results:
    print(json.dumps({'status':'preserved','results':import_results_tree(args.import_results,args.result_root)}))
  elif args.localize_tree:
    print(json.dumps({'status':'localized','archives':localize_archive_tree(args.localize_tree)}))
  else:
    if args.directory is None or args.owner_pid is None:parser.error('rendering requires directory and --owner-pid')
    from eval_learned_adversary import AppOwner
    owner=AppOwner(args.owner_pid);owner.start()
    try:render(args.directory,owner)
    finally:owner.close()
