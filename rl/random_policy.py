"""Random + masked-random policies for end-to-end smoke-testing the env.

These exist so we can run

    python -m rl --wall example-v2-boulder --policy random

and verify the Gymnasium env actually steps, terminates, and produces
reasonable rewards before bringing in heavyweight agents (PPO, SAC).
"""
from __future__ import annotations

import random
from typing import Optional

import numpy as np

from rl.env import ClimbingEnv


def random_policy(env: ClimbingEnv, rng: random.Random) -> int:
    """Pick any action — including illegal ones (so the agent learns)."""
    return rng.randrange(env.action_space.n)


def masked_random_policy(env: ClimbingEnv, rng: random.Random) -> int:
    """Pick uniformly among currently-legal actions. Falls back to a
    purely random action if no moves are legal (very unlikely)."""
    legal = env.legal_actions()
    if not legal:
        return random_policy(env, rng)
    return rng.choice(legal)


def rollout(
    env: ClimbingEnv,
    *,
    masked: bool = True,
    seed: int = 0,
    max_steps: Optional[int] = None,
    verbose: bool = False,
) -> tuple[float, list[dict]]:
    """Run one episode, return (total_reward, per-step info list)."""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    obs, info = env.reset(seed=seed)
    total = 0.0
    transcript: list[dict] = []
    step_cap = max_steps if max_steps is not None else env.env_config.max_steps
    for _ in range(step_cap):
        action = (
            masked_random_policy(env, rng) if masked
            else random_policy(env, rng)
        )
        obs, reward, terminated, truncated, info = env.step(int(action))
        total += reward
        transcript.append(dict(action=int(action), reward=reward, **info))
        if verbose:
            print(f"  step {len(transcript):>2}: "
                  f"{info.get('limb','??')}→{info.get('hold','??')}  "
                  f"r={reward:+.2f}  total={total:+.2f}  "
                  f"{'illegal' if info.get('illegal') else ''}")
        if terminated or truncated:
            break
    return total, transcript
