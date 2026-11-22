"""Bounded physical body24 costs, sampled once per 20 ms control interval.

Only reward and diagnostic state are changed. No action, reset pose, force,
sensor or hit-clock input is modified. Derivatives use adjacent control samples
within an episode; the first sample after a real reset has no derivative pair.
"""
from __future__ import annotations

import hashlib
import json
import math
import torch
from r1wb.head_contract import STATE_JOINTS, REFERENCE_JOINTS

VERSION = 'r1wb-wholebody-calm-body24-v2'
DT = .02
SCALES = (3987.46586737681, 36.62048347128767, 134.21899591126459)
BUDGETS = (2., 2., 1.)
SCALES_SHA256 = hashlib.sha256(json.dumps(dict(scales=SCALES, budgets=BUDGETS),
                                           sort_keys=True).encode()).hexdigest()
DIAGNOSTIC_COLUMNS = ('acceleration_mean_square', 'target_rate_mean_square',
                      'abs_mechanical_power_sum', 'acceleration_cost',
                      'target_rate_cost', 'power_cost', 'total_cost', 'history_valid')
REWARD_NAMES = ('body_acceleration_cost', 'body_target_rate_cost', 'body_power_cost')


def costs(features):
    """Nonnegative per-second costs; NaN/Inf remain visible to the raw ledger."""
    if features.ndim != 2 or features.shape[1] != 3:
        raise ValueError('Expected N x 3 physical body24 features')
    scale = features.new_tensor(SCALES)
    budget = features.new_tensor(BUDGETS)
    return budget * (features / (scale + features))


def physical_features(dq, target, torque, previous_dq, previous_target, history_valid, dt=DT):
    if not math.isclose(dt, DT, rel_tol=0., abs_tol=1e-12):
        raise ValueError('Calibrated features require a 20 ms control interval')
    shape = dq.shape
    if dq.ndim != 2 or shape[1] != 24 or any(x.shape != shape for x in
                                           (target, torque, previous_dq, previous_target)):
        raise ValueError('All physical feature arrays must be N x original body24')
    if history_valid.shape != (len(dq),) or history_valid.dtype != torch.bool:
        raise ValueError('History validity must be a per-environment boolean')
    acc = ((dq - previous_dq) / dt).square().mean(dim=1)
    rate = ((target - previous_target) / dt).square().mean(dim=1)
    # Missing pre-reset history is not an exemption from a defined derivative.
    # Nonfinite *current* inputs remain nonfinite even on the first sample.
    acc = torch.where(history_valid, acc,
                      torch.where(torch.isfinite(dq).all(dim=1), 0., float('nan')))
    rate = torch.where(history_valid, rate,
                       torch.where(torch.isfinite(target).all(dim=1), 0., float('nan')))
    power = (torque * dq).abs().sum(dim=1)
    return torch.stack((acc, rate, power), dim=1)


class ControlSamples:
    """Read-only physical observer; writes only its own history and feature cache."""
    def __init__(self, env):
        self.env = env
        entity = env.scene['robot']
        if tuple(entity.joint_names) != STATE_JOINTS:
            raise ValueError('Calm H requires the canonical original body24 + head2')
        if not math.isclose(float(env.step_dt), DT, rel_tol=0., abs_tol=1e-12):
            raise ValueError('Calm H requires control dt .02')
        if not env.reward_manager._scale_by_dt:
            raise ValueError('Native reward scaling by dt must remain enabled')
        self.ids = torch.tensor([entity.joint_names.index(name) for name in REFERENCE_JOINTS],
                                device=env.device, dtype=torch.long)
        self.previous_dq = torch.zeros(env.num_envs, 24, device=env.device)
        self.previous_target = torch.zeros_like(self.previous_dq)
        self.history_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.values = torch.empty(env.num_envs, 8, device=env.device)
        self.last_step = None
        self.captures = 0

    @torch.no_grad()
    def capture(self, dt):
        step = int(self.env.common_step_counter)
        if not math.isclose(float(dt), DT, rel_tol=0., abs_tol=1e-12):
            raise ValueError('Reward compute dt differs from the calibrated control interval')
        if step == self.last_step:
            return
        if self.last_step is not None and step != self.last_step + 1:
            raise RuntimeError('Missing or reordered control sample; do not join discontinuous history')
        data = self.env.scene['robot'].data
        # qvel is post-integration. qfrc_actuator is the existing final substep
        # force buffer, which can lag qvel by 5 ms; no new forward/step is run.
        dq = data.joint_vel[:, self.ids].detach()
        target = data.joint_pos_target[:, self.ids].detach()
        torque = data.qfrc_actuator[:, self.ids].detach()
        features = physical_features(dq, target, torque, self.previous_dq,
                                     self.previous_target, self.history_valid, float(dt))
        bounded = costs(features)
        self.values[:, :3] = features
        self.values[:, 3:6] = bounded
        self.values[:, 6] = bounded.sum(dim=1)
        self.values[:, 7] = self.history_valid.to(self.values.dtype)
        self.previous_dq.copy_(dq)
        self.previous_target.copy_(target)
        self.history_valid.fill_(True)
        self.last_step = step
        self.captures += 1

    def reset(self, env_ids=None):
        # RewardManager.reset runs before command reset. Invalidate only; do not
        # sample the old terminal pose or the not-yet-resampled reference here.
        self.history_valid[slice(None) if env_ids is None else env_ids] = False

    def read(self):
        if self.last_step is None or self.last_step != int(self.env.common_step_counter):
            raise RuntimeError('Calm monitor must capture before native reward evaluation')
        return self.values


def install(env):
    """Install after native managers/optimizations, before the outer reward ledger."""
    if '_wholebody_calm' in env.__dict__:
        return env._wholebody_calm
    monitor = ControlSamples(env)
    native_compute, native_reset = env.reward_manager.compute, env.reward_manager.reset

    def compute(dt):
        monitor.capture(dt)
        return native_compute(dt)

    def reset(env_ids=None):
        result = native_reset(env_ids)
        monitor.reset(env_ids)
        return result

    env.reward_manager.compute, env.reward_manager.reset = compute, reset
    env._wholebody_calm = monitor
    return monitor


def diagnostic_values(env):
    # Pure callback: history advances only in the explicit compute-entry monitor.
    return env._wholebody_calm.read()


def acceleration_cost(env):
    return diagnostic_values(env)[:, 3]


def target_rate_cost(env):
    return diagnostic_values(env)[:, 4]


def power_cost(env):
    return diagnostic_values(env)[:, 5]


def total_cost(env):
    """Zero-weight registered anchor for the ledger's eight-column diagnostic."""
    return diagnostic_values(env)[:, 6]


def contract_record(cell):
    if cell not in ('10', '01', '11'):
        raise ValueError('Only the three newly approved factorial cells are registered')
    return dict(version=VERSION, cell=cell, kappa=int(cell[1]),
                cost_cap_per_second=5., scope=list(REFERENCE_JOINTS),
                scales=SCALES, budgets=BUDGETS, scales_sha256=SCALES_SHA256,
                control_dt=DT, reward_dt='native manager once', torque_squared=False,
                fresh_consistency=cell[0] == '1', head_perception=True,
                diagnostic_anchor='wholebody_calm; always weight zero, never another charged cost',
                history='adjacent post-control samples; first sample after reset has derivative undefined/zero',
                torque_timing='existing final 5 ms substep qfrc_actuator times current qvel',
                target='scaled position target before actuator delay; radians, not raw action',
                diagnostics_columns=DIAGNOSTIC_COLUMNS, no_hit_time_inputs=True)
