"""S1 learner: plain rsl_rl PPO from random weights (mjlab's from-scratch humanoid recipe), symmetric actor, learned
per-joint std, fixed physical observation scales (no empirical normaliser, no inherited statistics)."""
from __future__ import annotations

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from r1s1 import numbers as N
from r1s1.env_cfg import EXPERIMENT


def s1_rl_cfg(iterations: int | None = None, entropy: float | None = None,
              algorithm: str | None = None, sigma: str = "scalar") -> RslRlOnPolicyRunnerCfg:
  cfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(class_name="r1s1.model:S1SymmetricMLPModel", hidden_dims=N.HIDDEN, activation="elu",
                        obs_normalization=False,
                        distribution_cfg={"class_name": "GaussianDistribution", "init_std": N.INIT_STD,
                                          "std_type": "scalar"}),
    critic=RslRlModelCfg(hidden_dims=N.HIDDEN, activation="elu", obs_normalization=False),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0, use_clipped_value_loss=True, clip_param=N.CLIP, entropy_coef=N.ENTROPY,
      num_learning_epochs=N.EPOCHS, num_mini_batches=N.MINI_BATCHES, learning_rate=N.LEARNING_RATE,
      schedule="adaptive", gamma=N.GAMMA, lam=N.LAM, desired_kl=N.DESIRED_KL, max_grad_norm=1.0),
    # entropy: None = the registered N.ENTROPY (set below); the R12 sigma measurement's cell (b) passes 0.0
    obs_groups={"actor": ("actor",), "critic": ("critic",)},
    experiment_name=EXPERIMENT, save_interval=N.SAVE_INTERVAL, num_steps_per_env=N.STEPS_PER_ENV,
    max_iterations=int(iterations or N.ITERATIONS), logger="tensorboard", upload_model=False)
  if entropy is not None:
    cfg.algorithm.entropy_coef = float(entropy)
  if algorithm is not None:   # R12 S0b: "r1s1.learning:S1PPO" (S0's PPO + records + in-run gates)
    cfg.algorithm.class_name = algorithm
  if sigma not in N.SIGMA_PARAMS:
    raise ValueError(f"sigma {sigma!r} not in {N.SIGMA_PARAMS}")
  if sigma == "fixed":        # S0b-fixed (coordinator 07:5x): constant per-joint std, not in the optimizer
    cfg.actor.distribution_cfg = {"class_name": "r1s1.distribution:FixedGaussian", "std_values": list(N.FIXED_STD),
                                  "scale": N.SIGMA_FIXED_SCALE}
  elif sigma != "scalar":     # R12 S0b arms (ㄱ) / (ㄷ), prepared (numbers.SIGMA_PARAMS); "scalar" leaves S0's dict as is
    cfg.actor.distribution_cfg = {
      "class_name": "r1s1.distribution:MirrorTiedGaussian" if sigma.startswith("tied") else "GaussianDistribution",
      "init_std": N.INIT_STD, "std_type": "log" if sigma.endswith("log") else "scalar"}
  return cfg
