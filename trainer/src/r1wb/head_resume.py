"""Named semantic checkpoint expansion for the Stage-H 24->26 transition.

Copies the old tensor coordinates and optimizer state exactly. New actor means
are zero; positive scalar head noise is the old scalar noise median. No training
or optimizer update is performed here. New coordinates share their tensor's old
Adam step, so their optimization history is explicitly a transfer initialization.
"""
from __future__ import annotations
import copy
import os
from pathlib import Path
import torch
from r1wb import head_contract as C

VERSION='r1wb-head26-reference24-scan-v1'


def mirror_tensors(device):
    from r1wb.symmetry import ActorMirror
    old=ActorMirror(411,device)
    index=torch.arange(417,device=device);sign=torch.ones(417,device=device)
    embedding=torch.tensor(C.feature_embedding(C.actor_features(senses='scan'),C.actor_features(head=True,senses='scan')),device=device)
    index[embedding]=embedding[old.obs_index];sign[embedding]=old.obs_sign
    names=C.actor_features(head=True,senses='scan')
    for prefix in ('body.q','body.dq','body.previous_action'):
        sign[names.index(prefix+'.head_yaw_joint')]=-1.
    return index,sign,torch.tensor(C.MIRROR_PERM,device=device),torch.tensor(C.MIRROR_SIGN,device=device)


def _layers(state):
    keys=sorted((k for k in state if k.startswith('mlp.') and k.endswith('.weight')),key=lambda k:int(k.split('.')[1]))
    if len(keys)<2:raise ValueError('Expected ordinary feed-forward actor/critic MLP')
    return keys


def _finite(value,label):
    if not torch.isfinite(value).all():raise ValueError('Nonfinite inherited tensor: '+label)


def _resize(value,key,state,own,embedding,*,moment=None):
    wanted=own[key].shape
    if value.shape==wanted:return value.clone()
    fill=0. if moment!='second' else float(value.mean())
    out=torch.full(wanted,fill,dtype=value.dtype,device=value.device)
    first,last=_layers(state)[0],_layers(state)[-1]
    if key==first:
        if value.ndim!=2 or value.shape[0]!=wanted[0]:raise ValueError('Input expansion shape differs')
        out[:,embedding]=value
    elif key in ('obs_normalizer._mean','obs_normalizer._var','obs_normalizer._std'):
        out.fill_(0. if key.endswith('_mean') else 1.)
        out[...,embedding]=value
    elif key in (last,last.replace('weight','bias'),'distribution.std_param') and value.shape[0]==24 and wanted[0]==26:
        out[:24]=value
        if key=='distribution.std_param' and moment is None:
            if (value<=0).any():raise ValueError('Scalar exploration std must be positive')
            out[24:]=value.median()
    else:raise ValueError(f'Unaccounted checkpoint shape: {key}: {value.shape} -> {wanted}')
    return out


def expand_loaded(loaded,actor,critic,optimizer):
    if actor.obs_dim!=417 or critic.obs_dim!=553 or actor.distribution.std_type!='scalar':
        raise ValueError('Stage H requires actor417/critic553 and scalar Gaussian noise')
    own_parts={'actor_state_dict':actor.state_dict(),'critic_state_dict':critic.state_dict()}
    shape=tuple(loaded[p][_layers(loaded[p])[0]].shape[1] for p in own_parts)
    if shape==(417,553):
        if (loaded.get('infos') or {}).get('head_contract')!=VERSION:
            raise ValueError('Existing head26 checkpoint lacks its explicit contract')
        for part,own in own_parts.items():
            if set(loaded[part])!=set(own) or any(loaded[part][k].shape!=own[k].shape for k in own):
                raise ValueError('Resumed head26 state shape differs')
            for key,value in loaded[part].items():_finite(value,part+'.'+key)
        if (loaded['actor_state_dict']['distribution.std_param']<=0).any():
            raise ValueError('Resumed head26 scalar exploration std must be positive')
        return loaded,False
    if shape!=(411,547):raise ValueError('Stage H starts from a completed scan teacher (411/547), not an unscanned model')
    result=copy.deepcopy(loaded)
    mappings={}
    for part,own in own_parts.items():
        state=loaded[part]
        if set(state)!=set(own):raise ValueError('Unsupported checkpoint keys: '+part)
        old,new=(C.actor_features(senses='scan'),C.actor_features(head=True,senses='scan')) if part=='actor_state_dict' else (C.critic_features(scan=True),C.critic_features(head=True,scan=True))
        embedding=torch.tensor(C.feature_embedding(old,new),device=state[_layers(state)[0]].device)
        mappings[part]=embedding
        for key,value in state.items():
            _finite(value,part+'.'+key)
            result[part][key]=_resize(value,key,state,own,embedding)
    parameters=[('actor_state_dict',name,param) for name,param in actor.named_parameters()]
    parameters += [('critic_state_dict',name,param) for name,param in critic.named_parameters()]
    actual=[p for g in optimizer.param_groups for p in g['params']]
    if len(actual)!=len(parameters) or any(a is not row[2] for a,row in zip(actual,parameters)):
        raise ValueError('Optimizer ordering is not the verified actor-then-critic parameter contract')
    old_opt=loaded['optimizer_state_dict'];new_opt=result['optimizer_state_dict']
    old_ids=[i for g in old_opt['param_groups'] for i in g['params']]
    if len(old_ids)!=len(parameters) or len(set(old_ids))!=len(old_ids):raise ValueError('Optimizer parameter identity/count differs')
    for identity,(part,name,param) in zip(old_ids,parameters):
        if name not in loaded[part]:raise ValueError('Optimizer parameter is not in the named model state')
        state=old_opt['state'].get(identity)
        if state is None:continue # A never-updated parameter legitimately has no Adam state yet.
        if not {'step','exp_avg','exp_avg_sq'}.issubset(state) or set(state)-{'step','exp_avg','exp_avg_sq','max_exp_avg_sq'}:
            raise ValueError('Unsupported Adam state fields')
        for key,value in state.items():
            _finite(value,'optimizer.'+name+'.'+key)
            if key=='step':continue
            if value.shape!=loaded[part][name].shape:raise ValueError('Optimizer/name shape pairing differs: '+name)
            if key!='exp_avg' and (value<0).any():raise ValueError('Negative Adam second moment')
            new_opt['state'][identity][key]=_resize(value,name,loaded[part],own_parts[part],mappings[part],
                                                   moment='first' if key=='exp_avg' else 'second')
    result.setdefault('infos',{})['head_contract']=VERSION
    result['infos']['head_expansion']=dict(actor_input=[411,417],critic_input=[547,553],action=[24,26],
        new_mean='zero',new_std='old positive scalar median',new_adam_first='zero',
        new_adam_second='same tensor old mean',new_adam_step='inherits tensor step',optimizer_update_performed=False)
    return result,True


def prepare_resume(checkpoint,runner,directory,device):
    actor=getattr(runner.alg,'_raw_actor',runner.alg.actor)
    critic=getattr(runner.alg,'_raw_critic',runner.alg.critic)
    loaded=torch.load(checkpoint,map_location=device,weights_only=False)
    from r1wb.head_noise_consistency import HeadNoiseConsistencyPPO
    if 'noise_consistency' in loaded and not isinstance(runner.alg,HeadNoiseConsistencyPPO):
        raise ValueError('A v11c checkpoint cannot silently lose its body consistency objective')
    expanded,changed=expand_loaded(loaded,actor,critic,runner.alg.optimizer)
    if isinstance(runner.alg,HeadNoiseConsistencyPPO):
        expanded=runner.alg.migrate_checkpoint(loaded,expanded,changed)
    if not changed:return checkpoint
    path=Path(directory)/f'.resume_head26_{os.getpid()}.pt'
    torch.save(expanded,path)
    return str(path)
