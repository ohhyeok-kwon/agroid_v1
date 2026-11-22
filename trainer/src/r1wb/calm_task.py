"""Approved H 2 x 2 factor cells; 00 is the completed original v12h seed1."""
from mjlab.managers.reward_manager import RewardTermCfg
from r1wb import head_task, wholebody_calm as costs


def env_cfg(cell, play=False, **kwargs):
    costs.contract_record(cell)
    cfg = head_task.env_cfg(True, play, base_teacher='v11c' if cell[0] == '1' else 'v10s', **kwargs)
    weight = -float(int(cell[1]))
    for name, func in zip(costs.REWARD_NAMES,
                          (costs.acceleration_cost, costs.target_rate_cost, costs.power_cost), strict=True):
        cfg.rewards[name] = RewardTermCfg(func=func, weight=weight)
    cfg.rewards['wholebody_calm'] = RewardTermCfg(func=costs.total_cost, weight=0.)
    return cfg


def rl_cfg(cell):
    costs.contract_record(cell)
    if cell[0] == '1':
        from r1wb.calm_consistency import rl_cfg as consistency_rl_cfg
        return consistency_rl_cfg()
    return head_task.rl_cfg('v10s')
