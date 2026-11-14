"""Stream atomic vector-evaluation state chunks, then render without resimulation.

No in-memory frame collection and no physics/RNG changes. Pending is not video
completion. Only local result/video/<run>/ paths are permitted.
"""
import argparse
import hashlib
import fcntl
import importlib.metadata
from datetime import datetime
import io
import json
import os
import platform
from pathlib import Path
import shutil
import zipfile
import numpy as np
from r1wb.eval.paths import ROOT,AGENT

RESERVE=512*1024**2

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
  stamp=datetime.now().strftime('%Y%m%d-%H%M%S-%f')
  return AGENT/'result/video'/run/f'{stamp}_{Path(out).stem if out else "vector_eval"}'

class VectorVideoArchive:
  def __init__(self,env,directory,labels,min_free_bytes=RESERVE):
    self.env=env;self.directory=Path(directory);self.labels=labels;self.reserve=min_free_bytes
    self.episodes=np.zeros(env.num_envs,dtype=np.int32);self.step=0;self.started=False;self.archive=None
    self.manifest=dict(protocol_version=2,capture_pid=os.getpid(),status='pending',reason='State capture awaiting local rendering',
        directory=str(self.directory),state_path=str(self.directory/'states'),model_path=str(self.directory/'model.mjb'),
        step_dt=env.step_dt,frames=0,capture_complete=False,chunk_sha256={},
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
      buffer=io.BytesIO();np.savez(buffer,**values)
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
  parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--owner-pid',type=int,required=True)
  args=parser.parse_args()
  from eval_learned_adversary import AppOwner
  owner=AppOwner(args.owner_pid);owner.start()
  try:render(args.directory,owner)
  finally:owner.close()
