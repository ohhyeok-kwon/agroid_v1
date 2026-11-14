"""Simulation-only actor contract: original 810 history followed by 117 heights.

Scan v1: pelvis body origin (not COM), yaw-aligned x forward/y left, x=-.6:.1:.6
and y=-.4:.1:.4. Flatten [9 y,13 x], x fastest. Parallel world-down rays, 3 m
maximum, geometry group 0 only (terrain/objects; no robot/projectiles).
Value = pelvis_z - hit_z - .7313 metres, clipped [-.5,.5]. Miss/nonfinite=+1.
One CURRENT scan, not ten stacked scans. This contract is not approved for R1.
NumPy-only import; MuJoCo, torch and mjlab imports are lazy.
"""
from functools import partial
import numpy as np

HISTORY_DIM=810
SCAN_DIM=117
ACTOR_INPUT_DIM=927
SCAN_PROTOCOL='r1_height_scan_v1'
HEIGHT_REFERENCE=.7313
MAX_DISTANCE=3.
MISS_VALUE=1.
X_COORDS=np.linspace(-.6,.6,13,dtype=np.float32)
Y_COORDS=np.linspace(-.4,.4,9,dtype=np.float32)


def actor_input_dimension(session):
  inputs=session.get_inputs()
  if len(inputs)!=1 or len(inputs[0].shape)!=2:
    raise ValueError('Expected one batched actor input')
  width=inputs[0].shape[-1]
  if width not in (HISTORY_DIM,ACTOR_INPUT_DIM):
    raise ValueError(f'Unsupported actor input {width}; expected 810 or 927')
  return int(width)


def local_offsets_numpy():
  x,y=np.meshgrid(X_COORDS,Y_COORDS,indexing='xy')
  return np.stack((x,y,np.zeros_like(x)),-1).reshape(SCAN_DIM,3)


def ray_origins_numpy(position,quaternion):
  position=np.asarray(position,dtype=np.float64)
  w,x,y,z=np.asarray(quaternion,dtype=np.float64)
  yaw=np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
  offsets=local_offsets_numpy().astype(np.float64)
  c,s=np.cos(yaw),np.sin(yaw)
  offsets[:,:2]=offsets[:,:2]@np.array([[c,s],[-s,c]])
  return offsets+position


def encode_height_numpy(frame_z,hit_z,distances):
  hit_z=np.asarray(hit_z);distances=np.asarray(distances)
  if hit_z.shape != distances.shape or hit_z.shape[-1]!=SCAN_DIM:
    raise ValueError('Expected matching [...,117] hit heights and distances')
  height=np.asarray(frame_z)[...,None]-hit_z-HEIGHT_REFERENCE
  valid=np.isfinite(height)&np.isfinite(distances)&(distances>=0)&(distances<=MAX_DISTANCE)
  return np.where(valid,np.clip(height,-.5,.5),MISS_VALUE).astype(np.float32)


def encode_height_torch(frame_z,hit_z,distances):
  import torch
  if hit_z.shape!=distances.shape or hit_z.shape[-1]!=SCAN_DIM:
    raise ValueError('Expected matching [...,117] hit heights and distances')
  height=frame_z[...,None]-hit_z-HEIGHT_REFERENCE
  valid=torch.isfinite(height)&torch.isfinite(distances)&(distances>=0)&(distances<=MAX_DISTANCE)
  return torch.where(valid,height.clamp(-.5,.5),torch.full_like(height,MISS_VALUE))


def _pelvis_id(model):
  for name in ('pelvis','robot/pelvis'):
    try:return model.body(name).id
    except KeyError:pass
  raise ValueError('No pelvis body found; pass body_id explicitly')


def height_scan_numpy(model,data,*,body_id=None):
  """CPU counterpart of actor_height_scan; does not advance physics.

  mj_multiRay has one shared origin and cannot represent this translated grid.
  Do not call for legacy 810 actors. Temporary ray-group filtering is restored
  before return; callers must serialize access to a shared MuJoCo model.
  """
  import mujoco
  mujoco.mj_kinematics(model,data)
  body_id=_pelvis_id(model) if body_id is None else int(body_id)
  origins=ray_origins_numpy(data.xpos[body_id],data.xquat[body_id])
  original_groups=model.geom_group.copy()
  for i in range(model.ngeom):
    body=int(model.geom_bodyid[i]);name=model.body(body).name
    excluded=name.startswith(('ball','projectile'))
    ancestor=body
    while ancestor:
      if ancestor==body_id:excluded=True;break
      ancestor=int(model.body_parentid[ancestor])
    if excluded:model.geom_group[i]=1
  group=np.array([1,0,0,0,0,0],dtype=np.uint8)
  direction=np.array([0.,0.,-1.],dtype=np.float64)
  geom_id=np.empty(1,dtype=np.int32)
  distances=np.empty(SCAN_DIM,dtype=np.float64)
  try:
    for i,origin in enumerate(origins):
      distances[i]=mujoco.mj_ray(model,data,origin,direction,group,True,body_id,geom_id)
  finally:model.geom_group[:]=original_groups
  hit_z=origins[:,2]-np.maximum(distances,0.)
  return encode_height_numpy(data.xpos[body_id,2],hit_z,distances)


def actor_height_scan(env,sensor_name='terrain_scan'):
  data=env.scene[sensor_name].data
  return encode_height_torch(data.frame_pos_w[:,0,2],data.hit_pos_w[...,2],data.distances)


def filter_terrain_scan_spec(spec,previous=None):
  """Ray groups do not change collision masks. Group 1 is rendered by default."""
  if previous is not None:previous(spec)
  for geom in spec.geoms:
    if geom.parent.name.startswith(('ball','projectile')):
      geom.group=1


def add_actor_height_scan(cfg):
  """Append only actor observation; preserve terrain and the critic's terms."""
  from mjlab.managers.observation_manager import ObservationTermCfg
  from r1lab.tasks.common import terrain_scan
  if not any(sensor.name=='terrain_scan' for sensor in cfg.scene.sensors):
    cfg.scene.sensors += (terrain_scan(),)
  cfg.observations['actor'].terms['terrain_scan']=ObservationTermCfg(func=actor_height_scan)
  previous=cfg.scene.spec_fn
  if previous is not filter_terrain_scan_spec:
    cfg.scene.spec_fn=(filter_terrain_scan_spec if previous is None else
                       partial(filter_terrain_scan_spec,previous=previous))
  return cfg
