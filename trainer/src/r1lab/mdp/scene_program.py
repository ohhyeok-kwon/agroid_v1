"""Learned continuous typed scene programs over fixed GPU-friendly object pools.

Topology is compiled once. A program changes geometry, materials and a sequence
of independently parameterized physical events. Finite validation bounds are a
physical operating envelope, not a catalogue of complete scenarios.
"""
from __future__ import annotations
import copy
import math
import mujoco
import torch
from torch import nn
from torch.distributions import Normal, Bernoulli, Categorical
from mjlab.entity import EntityCfg
from r1lab.mdp.adversary import AdversaryController

OBJECTS, EVENTS = 12, 8
CONTINUOUS = OBJECTS*8 + EVENTS*7
MODEL_FIELDS = ("geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_rbound", "geom_aabb")


def _scene_spec():
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="scene")
  for i in range(OBJECTS):
    body.add_geom(name=f"object_{i}", type=(mujoco.mjtGeom.mjGEOM_CYLINDER if i >= 8 else mujoco.mjtGeom.mjGEOM_BOX),
                  # Nonidentity pose prevents compile-time sameframe shortcuts
                  # from ignoring later per-environment geom_pos/quat updates.
                  size=(.15, .05, .05), pos=(25+2*i, 0, -5),
                  quat=(math.cos(.005), 0., math.sin(.005), 0.), contype=1, conaffinity=1,
                  friction=(.8, .02, .01), condim=6, priority=2)
  return spec


def scene_entities():
  return {"adversary_scene": EntityCfg(spec_fn=_scene_spec)}


def decode_program(raw, object_mask, event_types, ceiling=1.):
  """Bounded continuous decoder. Slots 0:4 boxes, 4:8 ramps, 8:12 cylinders.

  Event types: 0 inactive, 1 ball, 2 finite force, 3 velocity jump, 4 material change.
  At most one ball per event; eight slots fit the existing projectile pool.
  """
  if raw.ndim != 2 or raw.shape[1] != CONTINUOUS or not torch.isfinite(raw).all():
    raise ValueError("Malformed or nonfinite scene program")
  if object_mask.shape != (len(raw),OBJECTS) or event_types.shape != (len(raw),EVENTS):
    raise ValueError("Wrong pool shape")
  if ((event_types<0)|(event_types>4)).any() or not 0 <= ceiling <= 1:
    raise ValueError("Invalid typed event or ceiling")
  x = raw.tanh()
  o = x[:, :OBJECTS*8].reshape(-1,OBJECTS,8)
  e = x[:, OBJECTS*8:].reshape(-1,EVENTS,7)
  # Radial exclusion leaves a clear reset footprint even after object extents.
  angle = math.pi*o[...,0]
  distance = 1.0+1.5*(o[...,1]+1)/2
  half = torch.stack((.06+.29*(o[...,2]+1)/2, .06+.29*(o[...,3]+1)/2,
                      .008+(.072*ceiling)*(o[...,4]+1)/2), -1)
  pitch = o[...,5]*(.25*ceiling)
  pitch[:, :4] = 0; pitch[:, 8:] = 0
  yaw = math.pi*o[...,6]
  cy,sy,cp,sp = (yaw/2).cos(),(yaw/2).sin(),(pitch/2).cos(),(pitch/2).sin()
  quat = torch.stack((cy*cp,-sy*sp,cy*sp,sy*cp),-1)
  # Cylinder size conventions are (radius, half-height, unused).
  half[:,8:,1] = half[:,8:,2].clone();half[:,8:,2] = 0
  height = torch.where(torch.arange(OBJECTS,device=raw.device)[None,:]>=8,
                       half[...,1],half[...,0]*pitch.sin().abs()+half[...,2]*pitch.cos().abs())
  pos = torch.stack((distance*angle.cos(),distance*angle.sin(),height),-1)
  friction = .35+1.05*(o[...,7]+1)/2
  times = 2.+10*(e[...,0]+1)/2
  types = event_types.long().clone()
  # Sort the program chronologically, retaining each event's parameters.
  order = torch.where(types>0,times,torch.full_like(times,100.)).argsort(-1)
  e = e.gather(1,order[...,None].expand(-1,-1,7));times=times.gather(1,order);types=types.gather(1,order)
  # Non-overlapping starts bound the density of arbitrary combinations.
  for i in range(1,EVENTS): times[:,i]=torch.where(types[:,i]>0,torch.maximum(times[:,i],times[:,i-1]+.3),times[:,i])
  strength = (e[...,1]+1)/2*ceiling
  return dict(version=1, object_mask=object_mask.bool().clone(), object_pos=pos,
      object_size=half,object_quat=quat,object_friction=friction,
      event_type=types,event_time=times,event_angle=math.pi*e[...,2],
      ball_speed=6.+12*strength,force_n=20.+40*strength,push_dv=.1+.5*strength,
      duration=.05+.15*(e[...,3]+1)/2, height_offset=.1*e[...,4],
      body_choice=((e[...,5]+1)*1.999).long(), point_offset=.08*e[...,6],
      material_friction=.35+1.05*(e[...,1]+1)/2,ceiling=ceiling)


def program_json(program, index):
  """Exact decoded program for replay; no RNG or policy required on replay."""
  return {k:(v[index].detach().cpu().tolist() if isinstance(v,torch.Tensor) else v)
          for k,v in program.items()}


class GenerativeSceneController(AdversaryController):
  """Joint continuous/discrete contextual PPO over a whole typed program."""
  def __init__(self,context_dim=8,device="cpu",learning_rate=1e-4):
    nn.Module.__init__(self)
    self.context_dim=context_dim;self.epochs=2;self.updates=0;self.pending=[]
    self.net=nn.Sequential(nn.Linear(context_dim,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh())
    self.mean=nn.Linear(128,CONTINUOUS);self.log_std=nn.Parameter(torch.full((CONTINUOUS,),math.log(2.3/.7)))
    self.objects=nn.Linear(128,OBJECTS);self.types=nn.Linear(128,EVENTS*5);self.value=nn.Linear(128,1)
    nn.init.zeros_(self.mean.weight);nn.init.zeros_(self.mean.bias)
    nn.init.zeros_(self.objects.weight);nn.init.constant_(self.objects.bias,-1.5)
    nn.init.zeros_(self.types.weight);nn.init.zeros_(self.types.bias)
    with torch.no_grad(): self.types.bias.reshape(EVENTS,5)[:,0]=1.5
    self.to(device);self.optimizer=torch.optim.Adam(self.parameters(),lr=learning_rate)

  def _dists(self,context):
    h=self.net(context)
    return (Normal(self.mean(h),(-3+3*torch.sigmoid(self.log_std)).exp()),
            Bernoulli(logits=self.objects(h)),Categorical(logits=self.types(h).reshape(-1,EVENTS,5)),
            self.value(h).squeeze(-1))

  def _log_prob(self,normal,objects,types,raw,mask,event,entropy_sample=False):
    # Marginalize irrelevant continuous coordinates; discrete existence/type
    # decisions remain part of the policy even when a slot is switched off.
    active_o=mask.bool()[...,None].expand(-1,-1,8).clone()
    active_o[:,:4,5]=False;active_o[:,8:,3]=False
    active_o[:,8:,5:7]=False  # cylinder rotation and unused half-width
    active_e=torch.zeros((*event.shape,7),device=raw.device,dtype=torch.bool)
    active_e[...,0]=event>0
    active_e[...,1]=event>0
    active_e[...,2]=(event>0)&(event<4)
    active_e[...,3]=event==2
    active_e[...,4]=event==1
    active_e[...,5:7]=(event==2)[...,None]
    active=torch.cat((active_o.flatten(1),active_e.flatten(1)),-1)
    lp=(normal.log_prob(raw)*active).sum(-1)+objects.log_prob(mask).sum(-1)+types.log_prob(event).sum(-1)
    # Entropy of the transformed action, not unbounded raw Normal entropy.
    z=normal.rsample() if entropy_sample else raw
    jac=2*(math.log(2)-z-torch.nn.functional.softplus(-2*z))
    entropy=((normal.entropy()+jac)*active).sum(-1)+objects.entropy().sum(-1)+types.entropy().sum(-1)
    return lp,entropy

  @torch.no_grad()
  def sample(self,context,clean_mask=None,max_difficulty=4,max_terrain_level=4,deterministic=False):
    context=context.detach().to(self.device,dtype=torch.float32)
    if context.ndim!=2 or context.shape[1]!=self.context_dim or not torch.isfinite(context).all():
      raise ValueError("Invalid context")
    normal,objects,types,value=self._dists(context)
    raw=normal.mean if deterministic else normal.sample()
    mask=(objects.probs>=.5).float() if deterministic else objects.sample()
    event=types.probs.argmax(-1) if deterministic else types.sample()
    clean=torch.zeros(len(context),dtype=torch.bool,device=self.device) if clean_mask is None else clean_mask.bool()
    # Reserving clean modifies the outcome, so these samples are never trained.
    mask[clean]=0;event[clean]=0
    log_prob,_=self._log_prob(normal,objects,types,raw,mask,event)
    # Metadata only, not a categorical scenario catalogue.
    active_types=torch.stack([(event==i).any(-1) for i in range(1,5)],-1)
    group=torch.where(active_types.sum(-1)>1,4,active_types.long().argmax(-1)+1)
    group=torch.where(active_types.any(-1),group,torch.zeros_like(group))
    group=torch.where(mask.bool().any(-1),torch.full_like(group,4),group)  # geometry is its own challenge too
    action=torch.zeros(len(context),5,dtype=torch.long,device=self.device)
    action[:,0]=group;action[:,1]=max_difficulty
    return dict(context=context.clone(),raw=raw,object_mask=mask,event_types=event,
                log_prob=log_prob,value=value,trainable=~clean,action=action,
                version=torch.full((len(context),),self.updates,device=self.device,dtype=torch.long))

  def observe(self,sample,score,valid=None,success=None):
    score=torch.as_tensor(score,device=self.device,dtype=torch.float32)
    keep=sample['trainable'] & torch.isfinite(score)
    if valid is not None: keep &= valid.bool()
    keep &= (self.updates-sample['version'])<=256
    record={k:sample[k][keep].detach().clone() for k in ('context','raw','object_mask','event_types','log_prob','value','version')}
    record['score']=score[keep].detach().clone()
    record['success']=(torch.full_like(score,float('nan')) if success is None else success)[keep].detach().clone()
    self.pending.append(record)

  def update(self,reduce_sum=None):
    n=sum(len(b['score']) for b in self.pending)
    count=self._sum(torch.tensor(float(n),device=self.device),reduce_sum)
    if count.item()==0:
      self.pending.clear();return {'samples':0,'updates':self.updates}
    data={k:torch.cat([b[k] for b in self.pending]) for k in self.pending[0]} if n else None
    stats=torch.zeros(5,device=self.device)
    if n:
      adv=data['score']-data['value']; success=data['success']
      stats[:]=torch.stack((adv.sum(),adv.square().sum(),data['score'].sum(),success.nan_to_num().sum(),torch.isfinite(success).float().sum()))
    stats=self._sum(stats,reduce_sum);mean=stats[0]/count
    std=(stats[1]/count-mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    if n:
      with torch.no_grad():
        normal,objects,types,_=self._dists(data['context'])
        prox,_=self._log_prob(normal,objects,types,data['raw'],data['object_mask'],data['event_types'])
        importance=(prox-data['log_prob']).clamp(max=0,min=-20).exp()
    params=list(self.parameters())
    for _ in range(self.epochs):
      self.optimizer.zero_grad(set_to_none=True)
      if n:
        normal,objects,types,value=self._dists(data['context'])
        lp,entropy=self._log_prob(normal,objects,types,data['raw'],data['object_mask'],data['event_types'],entropy_sample=True)
        ratio=(lp-prox).clamp(-10,10).exp();adv=(data['score']-data['value']-mean)/std
        loss=(-importance*torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv)+.5*(value-data['score']).square()-.005*entropy).sum()
      else: loss=sum(p.sum()*0 for p in params)
      loss.backward();grad=self._sum(torch.cat([p.grad.reshape(-1) for p in params]),reduce_sum)/count
      start=0
      for p in params:
        p.grad.copy_(grad[start:start+p.numel()].view_as(p));start+=p.numel()
      nn.utils.clip_grad_norm_(params,1.);self.optimizer.step()
    self.pending.clear();self.updates+=1
    return dict(samples=int(count.item()),updates=self.updates,score=float((stats[2]/count).item()),
                actor_success=float((stats[3]/stats[4]).item()) if stats[4]>0 else None)

  def checkpoint(self):
    return copy.deepcopy(dict(version=3,context_dim=self.context_dim,model=self.state_dict(),
                              optimizer=self.optimizer.state_dict(),updates=self.updates))

  def restore(self,state):
    if state['version']!=3 or state['context_dim']!=self.context_dim: raise ValueError('Wrong scene controller checkpoint')
    self.load_state_dict(state['model']);self.optimizer.load_state_dict(state['optimizer'])
    self.updates=state['updates'];self.pending.clear()


def validate_program(program):
  """Vectorized physical-envelope checks; not a solvability certificate."""
  p=program
  n=len(p['object_pos']);device=p['object_pos'].device
  valid=torch.ones(n,dtype=torch.bool,device=device)
  for value in p.values():
    if isinstance(value,torch.Tensor):valid &= torch.isfinite(value).reshape(n,-1).all(-1)
  size=p['object_size']
  valid &= (size[:,:8]>0).reshape(n,-1).all(-1)
  valid &= (size[:,8:,:2]>0).reshape(n,-1).all(-1)
  valid &= (size<=.36).reshape(n,-1).all(-1)
  valid &= ((p['object_quat'].norm(dim=-1)-1).abs()<1e-4).all(-1)
  valid &= ((p['object_friction']>=.35)&(p['object_friction']<=1.4)).all(-1)
  horizontal_bound=size[:,:,:2].norm(dim=-1)
  horizontal_bound[:,8:]=size[:,8:,0]
  clearance=p['object_pos'][...,:2].norm(dim=-1)-horizontal_bound
  valid &= ((clearance>=.45)|~p['object_mask']).all(-1)
  limits={'ball_speed':(6,18),'force_n':(20,60),'push_dv':(.1,.6),
          'duration':(.05,.2),'height_offset':(-.1,.1),'point_offset':(-.08,.08),
          'material_friction':(.35,1.4),'event_time':(2,15)}
  for key,(low,high) in limits.items():
    valid &= ((p[key]>=low-1e-6)&(p[key]<=high+1e-6)).all(-1)
  valid &= ((p['event_type']>=0)&(p['event_type']<=4)).all(-1)
  valid &= ((p['event_time'][:,1:]-p['event_time'][:,:-1]>=.3-1e-5)|(p['event_type'][:,1:]==0)).all(-1)
  return valid
