"""AMP reference-state exploration; deployment still starts from real standing."""
from r1lab.run_record import work_root
from r1lab.tasks.adversarial_walk import adversarial_walk_env_cfg


def reference_walk_env_cfg(play=False, t0=0):
  cfg = adversarial_walk_env_cfg(play=play, t0=t0)
  if not play:
    cfg.events['scenarios'].params.update(
        rsi_data_file=str(work_root()/'data/reference_initialization/cautious_rsi_v1.npz'),
        rsi_clean_fraction=.4, rsi_seed=1)
  return cfg
