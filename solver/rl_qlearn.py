"""Tabular Q-learning solver.

A deliberately tiny RL implementation, dependency-free beyond NumPy. The
state is the pose tuple `(LH, RH, LF, RF)`; the action is the
`(limb, target_hold)` pair returned by `reachable_moves`.

A "really basic 2D RL" reference implementation. It's not meant to scale
beyond a handful of holds — once the wall gets bigger we should swap in
PPO + Stable Baselines3 by keeping the same env semantics
(step/reset/reward).

Reward shaping:
    + progress_reward   — COM closer to the finish (per step)
    + completion_bonus  — large reward for landing a hand on a finish hold
    - efficiency_penalty — small fixed cost per move
    - stability_penalty  — large cost for unstable poses (filtered out by
                           reachable_moves but kept here for safety)
"""
from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import numpy as np

from solver.astar import SolveResult, starting_poses
from solver.body import BodyModel, Limb
from solver.reachability import Pose, estimate_com, is_stable, reachable_moves
from solver.wall import Wall

PROGRESS_REWARD_PER_CM = 0.05
EFFICIENCY_PENALTY = 0.5
STABILITY_PENALTY = 50.0
COMPLETION_BONUS = 100.0
DEAD_END_PENALTY = 5.0
MAX_STEPS_PER_EPISODE = 30


class ClimbingEnv:
    """Minimal Gym-style env. We deliberately avoid pulling in `gymnasium`
    so this stays a single-file primitive. The interface is shaped like
    `reset() -> obs` / `step(action) -> (obs, reward, done, info)` so a
    drop-in PPO upgrade is a small lift later.
    """

    def __init__(self, wall: Wall, body: BodyModel | None = None) -> None:
        self.wall = wall
        self.body = body or BodyModel()
        self._starts = starting_poses(wall, self.body)
        if not self._starts:
            raise RuntimeError("No valid starting pose for this wall.")
        self._finish_ids = {h.hold_id for h in wall.finishes()}
        self.pose: Pose = self._starts[0]
        self._steps = 0

    # ------------------------------------------------------------------

    def reset(self, rng: random.Random | None = None) -> Pose:
        rng = rng or random
        self.pose = rng.choice(self._starts)
        self._steps = 0
        return self.pose

    def actions(self) -> list[tuple[Limb, str]]:
        return reachable_moves(self.body, self.wall, self.pose)

    def step(self, action: tuple[Limb, str]) -> tuple[Pose, float, bool, dict]:
        limb, target_id = action
        prev_pose = self.pose
        new_pose = prev_pose.with_limb(limb, target_id)

        prev_com = estimate_com(self.wall, prev_pose)
        new_com = estimate_com(self.wall, new_pose)

        reward = -EFFICIENCY_PENALTY
        if not is_stable(self.wall, new_pose):
            reward -= STABILITY_PENALTY

        # Distance-to-finish progress reward (uses nearest finish hold).
        finishes = self.wall.finishes()
        if finishes:
            target = min(
                finishes,
                key=lambda h: float(np.linalg.norm(new_com - np.array([h.x_cm, h.y_cm]))),
            )
            target_pt = np.array([target.x_cm, target.y_cm])
            prev_d = float(np.linalg.norm(prev_com - target_pt))
            new_d = float(np.linalg.norm(new_com - target_pt))
            reward += PROGRESS_REWARD_PER_CM * (prev_d - new_d)

        done = False
        if (new_pose.LH in self._finish_ids) or (new_pose.RH in self._finish_ids):
            reward += COMPLETION_BONUS
            done = True

        self.pose = new_pose
        self._steps += 1
        if self._steps >= MAX_STEPS_PER_EPISODE and not done:
            done = True
            reward -= DEAD_END_PENALTY

        return new_pose, reward, done, {"steps": self._steps}


def solve_qlearn(
    wall: Wall,
    body: BodyModel | None = None,
    *,
    episodes: int = 2000,
    alpha: float = 0.5,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 0,
) -> Optional[SolveResult]:
    """Train a tabular Q-table, then roll out a greedy policy from a
    starting pose to produce the beta. Returns None if training never
    found a goal-reaching trajectory.
    """
    body = body or BodyModel()
    env = ClimbingEnv(wall, body)
    rng = random.Random(seed)

    Q: dict[tuple, dict[tuple[Limb, str], float]] = defaultdict(dict)

    def q(state: Pose, action: tuple[Limb, str]) -> float:
        return Q[state.as_tuple()].get(action, 0.0)

    def best_action(state: Pose, actions: list[tuple[Limb, str]]) -> tuple[Limb, str]:
        return max(actions, key=lambda a: q(state, a))

    completions = 0
    for ep in range(episodes):
        state = env.reset(rng)
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * (ep / max(1, episodes - 1))
        done = False
        while not done:
            actions = env.actions()
            if not actions:
                # Dead end — penalise the visit and abandon the episode.
                if Q[state.as_tuple()]:
                    for a in list(Q[state.as_tuple()]):
                        Q[state.as_tuple()][a] -= DEAD_END_PENALTY
                break

            action = (
                rng.choice(actions) if rng.random() < epsilon
                else best_action(state, actions)
            )
            next_state, reward, done, _ = env.step(action)

            future = 0.0
            if not done:
                next_actions = reachable_moves(body, wall, next_state)
                if next_actions:
                    future = max(q(next_state, a) for a in next_actions)
            target = reward + gamma * future
            old = q(state, action)
            Q[state.as_tuple()][action] = old + alpha * (target - old)
            state = next_state

            if done and reward >= COMPLETION_BONUS - EFFICIENCY_PENALTY:
                completions += 1

    # Greedy rollout from the best starting pose.
    best: Optional[SolveResult] = None
    for start in env._starts:
        rollout = _greedy_rollout(wall, body, start, Q)
        if rollout is None:
            continue
        if best is None or len(rollout.moves) < len(best.moves):
            best = rollout

    if best is not None:
        # Patch in training stats.
        return SolveResult(
            poses=best.poses,
            moves=best.moves,
            expanded=completions,
            method=f"qlearn(eps={episodes}, completions={completions})",
        )
    return None


def _greedy_rollout(
    wall: Wall,
    body: BodyModel,
    start: Pose,
    Q: dict[tuple, dict[tuple[Limb, str], float]],
    max_steps: int = MAX_STEPS_PER_EPISODE,
) -> Optional[SolveResult]:
    state = start
    poses = [state]
    moves: list[tuple[Limb, str]] = []
    finish_ids = {h.hold_id for h in wall.finishes()}
    visited: set[tuple] = {state.as_tuple()}

    for _ in range(max_steps):
        if (state.LH in finish_ids) or (state.RH in finish_ids):
            return SolveResult(poses=poses, moves=moves, expanded=0, method="qlearn")
        actions = reachable_moves(body, wall, state)
        if not actions:
            return None
        scored = sorted(
            actions,
            key=lambda a: Q.get(state.as_tuple(), {}).get(a, 0.0),
            reverse=True,
        )
        # Take the highest-Q action that doesn't immediately revisit.
        chosen = None
        for a in scored:
            cand = state.with_limb(a[0], a[1])
            if cand.as_tuple() not in visited:
                chosen = a
                break
        if chosen is None:
            chosen = scored[0]
        next_state = state.with_limb(chosen[0], chosen[1])
        moves.append(chosen)
        poses.append(next_state)
        visited.add(next_state.as_tuple())
        state = next_state
    if (state.LH in finish_ids) or (state.RH in finish_ids):
        return SolveResult(poses=poses, moves=moves, expanded=0, method="qlearn")
    return None
