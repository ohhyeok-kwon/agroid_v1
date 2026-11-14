"""Training-only fixed anchors and asynchronous four-replica program accounting.

Independent DR/noise is intended; actor drift means frontier estimates are not
an unconditional IID guarantee. No optimizer or distributed calls occur here.
"""
from collections import deque
import torch
from r1lab.mdp.scene_program import CONTINUOUS,OBJECTS,EVENTS


def frontier_score(successes,k=4,severity=0.):
  if not 0<=successes<=k or k<2:raise ValueError('Invalid replica outcomes')
  return 4*successes*(k-successes)/(k*(k-1))+.05*severity*(successes>0)


class ReplicaQueue:
  def __init__(self,k=4):
    self.k=k;self.next_id=0;self.waiting=deque();self.records={}

  def assign(self,fresh,version=None):
    """One reset row in/one stored row out. At most K active replicas per case."""
    if self.waiting:
      key=self.waiting.popleft();record=self.records[key]
    else:
      key=self.next_id;self.next_id+=1
      record={'sample':{k:v.detach() for k,v in fresh.items()},'outcomes':[],
              'severity':[],'invalid':False,'completed':0,
              'version':int(fresh['version'][0]) if version is None else version}
      self.records[key]=record
      self.waiting.extend([key]*(self.k-1))
    return key,record['sample']

  def complete(self,key,success,valid,censored,severity,current_version):
    record=self.records.get(key)
    if record is None:return None
    record['completed']+=1
    old=record['version']
    stale=current_version-old>256
    record['invalid'] |= censored or stale or not valid
    record['outcomes'].append(bool(success));record['severity'].append(float(severity))
    if record['completed']<self.k:return None
    del self.records[key]
    if record['invalid']:return None
    score=frontier_score(sum(record['outcomes']),self.k,sum(record['severity'])/self.k)
    return record['sample'],score,sum(record['outcomes'])/self.k


def anchor_bank(device,count=64):
  """Fixed training seed, never imported from an evaluation bank.

  Groups rotate ball / finite-force+velocity-push / geometry / material.
  No generator cap dependence. Explicit strengths fit the existing training ODD.
  """
  gen=torch.Generator(device='cpu').manual_seed(814731)
  raw=torch.randn(count,CONTINUOUS,generator=gen)*.6
  mask=torch.zeros(count,OBJECTS);types=torch.zeros(count,EVENTS,dtype=torch.long)
  group=torch.arange(count)%4+1
  e=raw[:,OBJECTS*8:].reshape(count,EVENTS,7)
  def inv(x):return torch.atanh(torch.as_tensor(x).clamp(-.999,.999))
  for i,g in enumerate(group.tolist()):
    if g==1:
      types[i,:3]=1;e[i,:3,0]=inv(torch.tensor([-.6,-.54,-.48])) # 4,4.3,4.6s
      e[i,:3,1]=inv(torch.tensor(0.)) # 12m/s
      e[i,:3,2]=inv(torch.tensor((i%4)*.5-.75))
    elif g==2:
      types[i,:2]=torch.tensor([2,3]);e[i,:2,0]=inv(torch.tensor([-.6,.0]))
      e[i,0,1]=inv(torch.tensor(.5));e[i,0,3]=inv(torch.tensor(.999)) #50N, .2s
      e[i,1,1]=inv(torch.tensor(.2)) #dv .4m/s
    elif g==3:
      mask[i,i%12]=1
      o=raw[i,:OBJECTS*8].reshape(OBJECTS,8)
      o[i%12,0]=0.;o[i%12,1]=-2.;o[i%12,4]=-.5
    else:
      types[i,0]=4;e[i,0,0]=inv(torch.tensor(-.6));e[i,0,1]=inv(torch.tensor(-.7))
  return {k:v.to(device) for k,v in dict(raw=raw,object_mask=mask,event_types=types,group=group).items()}
