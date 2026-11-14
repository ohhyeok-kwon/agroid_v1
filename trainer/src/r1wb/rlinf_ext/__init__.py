"""Trial: the whole-body teacher task as an RLinf environment (RLinf = a Ray-based RL infrastructure built on
PyTorch; a separate package, tried on the user's request of 2026-11-12). Nothing here is used by the normal trainer.

  env.py           the mjlab task behind RLinf's simulator-environment interface (as its Genesis environment does)
  config/          Hydra configs for RLinf's embodied PPO with its MLP policy
  setup_server.sh  registers the environment type in an RLinf checkout and links the configs (server work area)
"""
