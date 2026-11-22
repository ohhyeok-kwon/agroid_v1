"""S1: System 1 trained from random weights on one outcome objective (R12, positive 22, 2026-11-19).

The policy is told WHAT to reach (a masked goal on the pelvis: planar displacement, heading, height, with the time
left until its deadline; every goal ends at rest) and never HOW: no reference motion, no joint-path price, no rule
arms, no reference-height terminations, no contact rules, no lag integrator, no post-hit windows. The episode ends on
a physical failure only (a body other than the feet on the floor, a self-collision). The reward is one success
indicator (occupancy of the time-varying admissible set), a potential difference (policy-invariant shaping) and one
effort price (Joule heating or mechanical power, one of the two switched on). The physics the deployed robot meets is
randomised (actuation and observation latency from the deployment chain, foot friction, trunk mass and centre of
mass, hand loads 0-3 kg, PD gains, pushes); the actor reads its own proprioception and actions of the last 0.5 s
plus the goal, the critic additionally the physical state (asymmetric).

The objective is written out in `.agent/adversarial/rounds/R04/positive/r12_s1/objective.md`; the registered
numbers with their reasons are in r1s1.numbers (one place). The human-motion prior (S+) is only a disconnected hook
(r1s1.prior, off): the coordinator's G0 check (19:0x) found the discriminator saturated.
"""
