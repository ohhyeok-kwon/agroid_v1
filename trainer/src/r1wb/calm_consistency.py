"""Explicit introduction of v11c's unchanged loss into a v10s-derived head H.

This new subclass does not weaken the existing inherited-v11c resume path.
The parent file's identity is pinned by the study launcher. Here we require the
exact original dimensions/update counts and absence of any pre-existing loss
state, then initialize its dedicated noise call counter to zero.
"""
from copy import deepcopy
from dataclasses import dataclass, fields
from r1wb import head_noise_consistency as HN

VERSION = 'r1wb-calm-factorial-fresh-v11c-objective-v1'
PARENT_SHA256 = '6995c2a7ba9b4e69c24076b6cca60f33518eaef06c93377952c6971b69afc2df'


def origin_contract(config_sha):
    return dict(version=VERSION, parent_checkpoint_sha256=PARENT_SHA256,
                parent_iteration=2748, parent_update_count=2750,
                config_sha256=config_sha, initial_calls=0, original_body_actions=24,
                head_actions_excluded=True, native_rng_unchanged=True)


class FreshHeadNoiseConsistencyPPO(HN.HeadNoiseConsistencyPPO):
    def migrate_checkpoint(self, original, expanded, changed):
        if not changed:
            self._require_state(original, head=True)
            if original.get('fresh_body_consistency') != origin_contract(self.consistency_config_sha):
                raise ValueError('Resume lacks the explicit fresh-objective origin')
            return expanded
        if ('noise_consistency' in original or 'fresh_body_consistency' in original
                or (original.get('infos') or {}).get('head_contract') is not None
                or original.get('iter') != 2748 or original.get('imitation_update_count') != 2750
                or original['actor_state_dict']['mlp.0.weight'].shape[1] != 411
                or original['critic_state_dict']['mlp.0.weight'].shape[1] != 547
                or tuple(original['actor_state_dict']['distribution.std_param'].shape) != (24,)
                or expanded.get('infos', {}).get('head_contract') != HN.HEAD_VERSION
                or expanded['infos'].get('head_expansion', {}).get('actor_input') != [411, 417]):
            raise ValueError('Fresh objective only permits the pinned v10s2748 scan-to-head full transfer')
        result = deepcopy(expanded)
        result['noise_consistency'] = dict(config_sha256=self.consistency_config_sha, calls=0,
                                          head_body_contract=deepcopy(self.head_body_contract))
        result['fresh_body_consistency'] = origin_contract(self.consistency_config_sha)
        return result

    def load(self, loaded_dict, load_cfg, strict):
        if loaded_dict.get('fresh_body_consistency') != origin_contract(self.consistency_config_sha):
            raise ValueError('Missing fresh-objective origin; inherited v11c must use its original class')
        return super().load(loaded_dict, load_cfg, strict)

    def save(self):
        result = super().save()
        result['fresh_body_consistency'] = origin_contract(self.consistency_config_sha)
        return result


@dataclass
class FreshHeadNoiseConsistencyPpoCfg(HN.HeadNoiseConsistencyPpoCfg):
    class_name: str = 'r1wb.calm_consistency:FreshHeadNoiseConsistencyPPO'


def rl_cfg():
    cfg = HN.rl_cfg()
    cfg.algorithm = FreshHeadNoiseConsistencyPpoCfg(**{
        f.name: getattr(cfg.algorithm, f.name) for f in fields(cfg.algorithm) if f.name != 'class_name'})
    return cfg
