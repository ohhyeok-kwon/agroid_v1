"""Negative agent of the r1adv co-evolution program: reproducible hard-but-solvable scenario generation.

Spec side (pure Python, runs anywhere): seedtree, bank, generator. Runtime side (server GPU): world_build, runner.
Scoring (CPU): score, diagnostic. No module here trains a policy or changes r1wb.
"""
NAME = "r1adv.negative"
VERSION = "v0.1"
