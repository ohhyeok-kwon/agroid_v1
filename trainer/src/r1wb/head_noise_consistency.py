"""Stage-H extension of v11c's original 24-body-action consistency objective.

The 411 inherited features are embedded in 417 inputs. Exactly the old 63 sensor
draws and 24-action Nm^2 mean remain; the six new head state coordinates are
identical in both auxiliary observations. Native rollout noise is unchanged.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, fields
import hashlib
import json
import math

import torch
from tensordict import TensorDict
from mjlab.actuator import BuiltinPositionActuator
from r1wb import head_contract as C, noise_consistency as N, recovery_task
from r1wb.learning import ImitationPPO

VERSION = 'r1wb-head417-v11c-body24-consistency-v1'
HEAD_VERSION = 'r1wb-head26-reference24-scan-v1'


def embedding():
    return C.feature_embedding(C.actor_features(senses='scan'),
                               C.actor_features(head=True, senses='scan')).tolist()


def preservation_contract(config_sha):
    return dict(version=VERSION, source_config_sha256=config_sha,
                source_actor=411, target_actor=417, source_actions=24, target_actions=26,
                loss_actions=list(C.REFERENCE_JOINTS), denominator=24,
                input_embedding=embedding(), new_head_auxiliary_noise=False,
                noise_draws_per_pair_per_state=2*63, native_head_rollout_noise_unchanged=True)


class BodyPdGainScale:
    """Read the original 24 gains from the real 26-actuator/action mapping."""
    def __init__(self, cfg, env):
        entity=env.scene['robot']; action=env.action_manager.get_term('joint_pos')
        if tuple(entity.joint_names)!=C.STATE_JOINTS or tuple(action._target_names)!=C.STATE_JOINTS:
            raise ValueError('Head consistency needs the canonical body24+head2 order')
        if action.cfg.clip is not None or type(action).__name__!='JointPositionAction':
            raise ValueError('Head consistency requires plain joint position actions')
        ids,names=[],[]
        for actuator in entity.actuators:
            if not isinstance(actuator,BuiltinPositionActuator) or actuator.transmission_type.value!='joint':
                raise ValueError('Only native joint position actuators are supported')
            if len(actuator.target_names)!=len(actuator.global_ctrl_ids):
                raise ValueError('Ambiguous physical actuator mapping')
            ids.append(actuator.global_ctrl_ids); names.extend(actuator.target_names)
        if len(names)!=26 or set(names)!=set(C.STATE_JOINTS):
            raise ValueError('Missing or duplicate head/body physical actuator')
        order=torch.tensor([names.index(name) for name in C.REFERENCE_JOINTS],device=env.device)
        self.global_ids=torch.cat(ids)[order]; self.action=action

    def __call__(self,env):
        if self.action._scale.shape[-1]!=26:
            raise ValueError('Head action scale must have all 26 targets')
        return N.ordered_gain_scale(env.sim.model.actuator_gainprm,self.global_ids,
                                    self.action._scale[:,:24],env.num_envs)


class HeadNoiseConsistencyPPO(ImitationPPO):
    """Native PPO update plus exactly the inherited body-only auxiliary loss."""
    def __init__(self,*args,consistency_coef,consistency_low,consistency_high,
                 consistency_seed=20261115,**kwargs):
        if not math.isfinite(consistency_coef) or consistency_coef<=0:
            raise ValueError('Consistency coefficient must be finite and positive')
        if type(consistency_seed)is not int or not 0<=consistency_seed<2**63:
            raise ValueError('Invalid dedicated noise seed')
        original=recovery_task.noise_consistency_scan_rl_cfg().algorithm
        if (tuple(consistency_low)!=original.consistency_low or tuple(consistency_high)!=original.consistency_high
                or consistency_coef!=original.consistency_coef or consistency_seed!=original.consistency_seed):
            raise ValueError('Head extension must retain the exact registered v11c configuration')
        super().__init__(*args,**kwargs)
        if (tuple(self.actor.obs_groups)!=('actor',) or self.actor.obs_dim!=417
                or self.critic.obs_dim!=553 or tuple(self.actor.distribution.std_param.shape)!=(26,)):
            raise ValueError('Head consistency requires actor417, critic553 and 26 scalar-Gaussian actions')
        self.consistency_coef=float(consistency_coef)
        old_low=torch.tensor(consistency_low,device=self.device)
        old_span=torch.tensor(consistency_high,device=self.device)-old_low
        mapping=torch.tensor(embedding(),device=self.device)
        self.consistency_low=torch.zeros(417,device=self.device)
        self.consistency_span=torch.zeros(417,device=self.device)
        self.consistency_low[mapping]=old_low; self.consistency_span[mapping]=old_span
        self.consistency_indices=mapping[old_span.nonzero().flatten()]
        if self.consistency_indices.numel()!=63:
            raise ValueError('Inherited v11c noise stream must have exactly 63 coordinates')
        self.consistency_seed,self.consistency_calls=consistency_seed,0
        self.consistency_generator=torch.Generator(device=self.device)
        config=dict(coefficient=consistency_coef,low=consistency_low,high=consistency_high,seed=consistency_seed)
        self.consistency_config_sha=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
        self.head_body_contract=preservation_contract(self.consistency_config_sha)

    def _extra_loss(self,observations):
        clean,gain=observations[N.CLEAN],observations[N.GAIN]
        if clean.ndim!=2 or clean.shape[1]!=417 or gain.shape!=(len(clean),24):
            raise ValueError('Expected clean417 and the original 24 PD gain scales')
        rank=self.gpu_global_rank if self.is_multi_gpu else 0
        seed=(self.consistency_seed+1000003*rank+1000000007*self.consistency_calls)%(2**63)
        self.consistency_generator.manual_seed(seed)
        samples=torch.rand((2,len(clean),len(self.consistency_indices)),device=clean.device,dtype=clean.dtype,
                           generator=self.consistency_generator)
        noisy=clean.unsqueeze(0).expand(2,-1,-1).clone()
        noisy.add_(self.consistency_low)
        noisy[:,:,self.consistency_indices]+=samples*self.consistency_span[self.consistency_indices]
        means=self.actor(TensorDict({'actor':noisy.flatten(0,1)},batch_size=[2*len(clean)]),
                         stochastic_output=False).reshape(2,len(clean),26)
        raw=((means[0,:,:24]-means[1,:,:24])*gain).square().mean()
        weighted=self.consistency_coef*raw; self.consistency_calls+=1
        return weighted,{'noise_consistency_nm2':raw,'noise_consistency_weighted':weighted}

    def _require_state(self,loaded,*,head):
        state=loaded.get('noise_consistency')
        expected={'config_sha256','calls'}|({'head_body_contract'}if head else set())
        if (not isinstance(state,dict) or set(state)!=expected
                or state.get('config_sha256')!=self.consistency_config_sha
                or type(state.get('calls'))is not int or state['calls']<0):
            raise ValueError('Missing or incompatible inherited v11c config/call state')
        if head and (state['head_body_contract']!=self.head_body_contract
                     or (loaded.get('infos')or{}).get('head_contract')!=HEAD_VERSION):
            raise ValueError('Head v11c checkpoint requires its explicit preservation contract')
        return state

    def migrate_checkpoint(self,original,expanded,changed):
        state=self._require_state(original,head=not changed)
        if not changed:return expanded
        if ((original.get('infos')or{}).get('head_contract') is not None
                or (expanded.get('infos')or{}).get('head_contract')!=HEAD_VERSION
                or expanded['infos'].get('head_expansion',{}).get('actor_input')!=[411,417]
                or tuple(original['actor_state_dict']['mlp.0.weight'].shape)[1]!=411
                or tuple(original['critic_state_dict']['mlp.0.weight'].shape)[1]!=547):
            raise ValueError('Only the explicit scan411/547 to head417/553 migration is permitted')
        result=deepcopy(expanded)
        result['noise_consistency']={**deepcopy(state),'head_body_contract':deepcopy(self.head_body_contract)}
        result['infos']['head_body_consistency_migration']=dict(version=VERSION,
            source_config_sha256=state['config_sha256'],inherited_calls=state['calls'],objective_actions=24)
        return result

    def save(self):
        state=super().save()
        state['noise_consistency']=dict(config_sha256=self.consistency_config_sha,calls=self.consistency_calls,
                                       head_body_contract=deepcopy(self.head_body_contract))
        return state

    def load(self,loaded_dict,load_cfg,strict):
        if load_cfg is not None or strict is not True:
            raise ValueError('Head v11c permits only strict full-state resume')
        state=self._require_state(loaded_dict,head=True)
        result=super().load(loaded_dict,load_cfg,strict)
        self.consistency_calls=state['calls']
        return result


@dataclass
class HeadNoiseConsistencyPpoCfg(N.NoiseConsistencyPpoCfg):
    class_name:str='r1wb.head_noise_consistency:HeadNoiseConsistencyPPO'


def rl_cfg():
    cfg=recovery_task.noise_consistency_scan_rl_cfg(); previous=cfg.algorithm
    cfg.algorithm=HeadNoiseConsistencyPpoCfg(**{f.name:getattr(previous,f.name)
                                               for f in fields(previous)if f.name!='class_name'})
    return cfg
