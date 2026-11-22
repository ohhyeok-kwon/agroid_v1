"""Candidate only: independent same-state sensor noise loss for the scan teacher.

The auxiliary groups do not enter actor/critic forward inputs or normalization.
The loss is a squared *unsaturated PD command difference* in Nm, not measured
mechanical work/force. No temporal smoothing or disturbance-clock exemptions.
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
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.utils.noise import UniformNoiseCfg
from r1wb import robot
from r1wb.learning import ImitationPPO
from r1wb.student.env_cfg import ImitationPpoCfg
from r1wb.teacher_task import teacher_task_rl_cfg

CLEAN = 'consistency_clean'
GAIN = 'consistency_pd_gain'
WIDTHS = dict(command=180, motion_anchor_pos_b=3, motion_anchor_ori_b=6,
              base_lin_vel=3, base_ang_vel=3, joint_pos=24, joint_vel=24,
              actions=24, reference_velocity=6, goal=18, goal_lag=3, senses=117)


def training_noise_profile(actor_cfg):
  """Reuse the native training term amplitudes/order; fail on unsupported transforms."""
  if tuple(actor_cfg.terms) != tuple(WIDTHS) or not actor_cfg.concatenate_terms:
    raise ValueError('Consistency requires the exact 411-wide scan teacher layout')
  if actor_cfg.history_length not in (None, 0):
    raise ValueError('No history resampling is defined for this auxiliary loss')
  low, high, profile = [], [], []
  for name, term in actor_cfg.terms.items():
    if term.scale is not None or term.clip is not None or term.history_length or term.delay_max_lag:
      raise ValueError('Unsupported transformed observation: '+name)
    if name == 'senses' and term.params != {'kind':'scan'}:
      raise ValueError('Only the existing 117-value terrain scan is supported')
    lo = hi = 0.0
    if term.noise is not None:
      if not isinstance(term.noise, UniformNoiseCfg) or term.noise.operation != 'add':
        raise ValueError('Only native additive uniform sensor noise is supported')
      lo, hi = float(term.noise.n_min), float(term.noise.n_max)
      if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError('Invalid native sensor noise bounds')
    first = len(low)
    low.extend([lo]*WIDTHS[name]); high.extend([hi]*WIDTHS[name])
    profile.append(dict(name=name,first=first,stop=len(low),low=lo,high=hi))
  return tuple(low), tuple(high), profile


def ordered_gain_scale(model_gain, global_ids, scale, num_envs):
  """Read current randomized MuJoCo stiffness, including a TorchArray view."""
  tensor = model_gain if isinstance(model_gain, torch.Tensor) else getattr(model_gain, '_tensor', None)
  if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3 or tensor.shape[0] not in (1,num_envs):
    raise ValueError('PD gain field needs [1|world, actuator, parameter] tensor view')
  if scale.ndim != 2 or scale.shape[1] != 24 or scale.shape[0] not in (1,num_envs):
    raise ValueError('Unexpected plain action scale layout')
  if global_ids.shape != (24,):
    raise ValueError('Every one of 24 action targets needs exactly one position actuator')
  return (tensor[:,global_ids,0]*scale).expand(num_envs,-1)


class PdGainScale:
  def __init__(self, cfg, env):
    entity = env.scene['robot']; action = env.action_manager.get_term('joint_pos')
    if tuple(entity.joint_names) != tuple(robot.JOINTS) or tuple(action._target_names) != tuple(robot.JOINTS):
      raise ValueError('PD gain and action need the canonical 24-joint order')
    if action.cfg.clip is not None or type(action).__name__ != 'JointPositionAction':
      raise ValueError('Consistency requires plain unfiltered joint position actions')
    ids, names = [], []
    for actuator in entity.actuators:
      if not isinstance(actuator, BuiltinPositionActuator) or actuator.transmission_type.value != 'joint':
        raise ValueError('Only native joint position actuators have this PD difference contract')
      if len(actuator.target_names) != len(actuator.global_ctrl_ids):
        raise ValueError('Ambiguous actuator-to-joint mapping')
      ids.append(actuator.global_ctrl_ids); names.extend(actuator.target_names)
    if len(names) != 24 or set(names) != set(robot.JOINTS):
      raise ValueError('Missing/duplicate physical actuator target')
    order = torch.tensor([names.index(n) for n in robot.JOINTS],device=env.device)
    self.global_ids = torch.cat(ids)[order]
    self.action = action

  def __call__(self, env):
    return ordered_gain_scale(env.sim.model.actuator_gainprm, self.global_ids, self.action._scale, env.num_envs)


def add_consistency_observations(cfg):
  """Mutate only two new auxiliary groups on the newly created v10s config."""
  training_noise_profile(cfg.observations['actor'])
  if CLEAN in cfg.observations or GAIN in cfg.observations:
    raise ValueError('Consistency groups already registered')
  clean = deepcopy(cfg.observations['actor'])
  clean.enable_corruption = False
  cfg.observations[CLEAN] = clean
  cfg.observations[GAIN] = ObservationGroupCfg(
    terms={'gain':ObservationTermCfg(func=PdGainScale)}, concatenate_terms=True, enable_corruption=False)
  return cfg


class NoiseConsistencyPPO(ImitationPPO):
  """The inherited PPO update/optimizer are unchanged; exactly one extra loss hook."""
  def __init__(self, *args, consistency_coef: float, consistency_low, consistency_high,
               consistency_seed: int = 20261115, **kwargs):
    if not math.isfinite(consistency_coef) or consistency_coef <= 0:
      raise ValueError('Noise consistency coefficient must be finite and positive')
    if type(consistency_seed) is not int or not 0 <= consistency_seed < 2**63:
      raise ValueError('Invalid dedicated noise seed')
    if (len(consistency_low) != 411 or len(consistency_high) != 411
        or any(not math.isfinite(a) or not math.isfinite(b) or a>b for a,b in zip(consistency_low,consistency_high))):
      raise ValueError('Malformed 411-coordinate noise bounds')
    super().__init__(*args,**kwargs)
    if tuple(self.actor.obs_groups) != ('actor',) or self.actor.obs_dim != 411:
      raise ValueError('Auxiliary observations must not enter the policy input')
    self.consistency_coef = float(consistency_coef)
    self.consistency_low = torch.tensor(consistency_low,device=self.device)
    self.consistency_span = torch.tensor(consistency_high,device=self.device)-self.consistency_low
    self.consistency_indices = self.consistency_span.nonzero().flatten()
    self.consistency_seed, self.consistency_calls = consistency_seed, 0
    self.consistency_generator = torch.Generator(device=self.device)
    config = dict(coefficient=consistency_coef,low=consistency_low,high=consistency_high,seed=consistency_seed)
    self.consistency_config_sha = hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()

  def _extra_loss(self, observations):
    clean, gain = observations[CLEAN], observations[GAIN]
    if clean.ndim != 2 or clean.shape[1] != 411 or gain.shape != (len(clean),24):
      raise ValueError('Missing same-state clean actor/PD gain observations')
    # A checkpoint saves the call counter. The rank salt reconstructs distinct
    # streams after loading the common rank-0 checkpoint, without global RNG use.
    rank = self.gpu_global_rank if self.is_multi_gpu else 0
    seed = (self.consistency_seed + 1000003*rank + 1000000007*self.consistency_calls) % (2**63)
    self.consistency_generator.manual_seed(seed)
    samples = torch.rand((2,len(clean),len(self.consistency_indices)),device=clean.device,dtype=clean.dtype,
                         generator=self.consistency_generator)
    noisy = clean.unsqueeze(0).expand(2,-1,-1).clone()
    noisy.add_(self.consistency_low)
    noisy[:,:,self.consistency_indices] += samples*self.consistency_span[self.consistency_indices]
    means = self.actor(TensorDict({'actor':noisy.flatten(0,1)},batch_size=[2*len(clean)]),
                       stochastic_output=False).reshape(2,len(clean),24)
    raw = ((means[0]-means[1])*gain).square().mean()
    weighted = self.consistency_coef*raw
    self.consistency_calls += 1
    return weighted, {'noise_consistency_nm2':raw,'noise_consistency_weighted':weighted}

  def save(self):
    state = super().save()
    state['noise_consistency'] = dict(config_sha256=self.consistency_config_sha,calls=self.consistency_calls)
    return state

  def load(self, loaded_dict, load_cfg, strict):
    state = loaded_dict.get('noise_consistency')
    if state is not None and (state.get('config_sha256') != self.consistency_config_sha
                             or type(state.get('calls')) is not int or state['calls'] < 0):
      raise ValueError('Noise consistency resume configuration differs')
    result = super().load(loaded_dict,load_cfg,strict)
    self.consistency_calls = state['calls'] if state is not None else 0
    return result


@dataclass
class NoiseConsistencyPpoCfg(ImitationPpoCfg):
  class_name: str = 'r1wb.noise_consistency:NoiseConsistencyPPO'
  consistency_coef: float = 0.0
  consistency_low: tuple[float,...] = ()
  consistency_high: tuple[float,...] = ()
  consistency_seed: int = 20261115


def consistency_rl_cfg(actor_cfg, coefficient):
  if not math.isfinite(coefficient) or coefficient <= 0:
    raise ValueError('Set a preregistered finite positive consistency coefficient')
  low,high,_ = training_noise_profile(actor_cfg)
  cfg = teacher_task_rl_cfg(symmetric=True)
  previous = cfg.algorithm
  cfg.algorithm = NoiseConsistencyPpoCfg(
    **{f.name:getattr(previous,f.name) for f in fields(previous) if f.name != 'class_name'},
    consistency_coef=coefficient,consistency_low=low,consistency_high=high)
  return cfg
