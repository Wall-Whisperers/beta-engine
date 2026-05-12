"""Reinforcement-learning environments for the Beta Engine.

`rl.env.ClimbingEnv` is a Gymnasium-compatible wrapper around the
pymunk physics simulation in `physics/`. The action space is a flat
discrete index over `(limb, hold)` pairs; the observation is a fixed-
length feature vector capturing climber state + which holds are
currently occupied.

The environment follows the Gymnasium 1.0 API
(`reset() -> (obs, info)`, `step(a) -> (obs, reward, terminated,
truncated, info)`) so Stable-Baselines3, RLlib, or CleanRL agents can
be dropped in without further glue. Today we ship a tiny random-policy
runner in `rl.random_policy` to validate the env end-to-end without
pulling in a heavyweight RL framework.
"""
from rl.env import ClimbingEnv

__all__ = ["ClimbingEnv"]
