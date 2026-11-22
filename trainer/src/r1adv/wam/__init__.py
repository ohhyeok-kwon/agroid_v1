"""Branch W of the adversarial co-evolution program (GR00T 2 / world action models / SONIC-family tracker).

W1-J: a reference executor for the judge (physical replay of human reference clips that leaves no tracker trace).
E1 bracket (track_task.py): two far ends of how the objective prices smoothness, same data and budget.
  A = SONIC as published (absolute smoothness prices, default-offset actions).
  B = reference-relative prices + reference-residual actions + spectral augmentation of the references.
Nothing here is deployed on the robot. Records: .agent/adversarial/rounds/R01/wam/ (w0_proposal.md, e0b.md, e1_plan.md).
Owner: the W agent (.agent/adversarial/agents/wam). Other agents do not edit this package.
"""
