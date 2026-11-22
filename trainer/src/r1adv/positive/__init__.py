"""Positive agent of the adversarial co-evolution program (R0 foundation, plan v0).

Fixed objective (coordinator decision 2026-11-17 01:35, objective.py):  r_t = L_t + beta * D_t
  L_t  mission success likelihood exp(-1/2 sum_k (e_k / sigma_k)^2), sigma_k fixed per mission by the coordinator
  D_t  fixed human-motion density score in (0, 1] (human_surprise.py; R0 G0 gate passed for pairs and 3 Hz)
Falls end the episode. Knobs of this agent: D's features/window, human data mix, beta, curriculum, distillation.
multipliers.py is not part of the objective (optional difficulty signal only). Nothing here is deployed: the deployed
network maps its inputs to joint targets, with no filter or rule in the control path.

Records: .agent/adversarial/rounds/R00/positive/{diagnostics,g0,training_plan_v0}.md
"""
