"""Day 4 test: validate MoonBoardEnv with gymnasium.utils.env_checker.

Usage:
    python3 scripts/test_env.py
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import warnings

import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
import mujoco

from src.parsers import format1
from src.envs.moonboard_env import MoonBoardEnv
from src.xml_gen import scene as scene_mod

_MOONBOARD1 = os.path.join(_PROJECT_ROOT, "moonboard_data", "moonboard1.json")
_HUMANOID = os.path.join(_PROJECT_ROOT, "assets", "humanoid.xml")


def _select_route(routes):
    candidates = [r for r in routes if r.grade_v in (4, 5)]
    return max(candidates, key=lambda r: r.repeats)


def _termination_reason(info: dict, fell_default: bool) -> str:
    if info.get("fall_penalty", 0.0) < 0.0:
        return "fell"
    if info.get("finish_bonus", 0.0) > 0.0:
        return "finished"
    return "truncated"


def main() -> None:
    routes = format1.load_routes(_MOONBOARD1)
    route = _select_route(routes)

    start_holds = [h for h in route.holds if h.role == "start"]
    mid_holds = [h for h in route.holds if h.role == "mid"]
    end_holds = [h for h in route.holds if h.role == "end"]

    print("=" * 60)
    print(f"Route: '{route.name}'  V{route.grade_v}  repeats={route.repeats}")
    print(
        f"Holds: {len(route.holds)} total  "
        f"({len(start_holds)} start, {len(mid_holds)} mid, {len(end_holds)} end)"
    )
    print("=" * 60)

    env = MoonBoardEnv(route=route, humanoid_xml_path=_HUMANOID)

    print(f"\nAction space:       {env.action_space}")
    print(f"Observation space:  {env.observation_space}")
    print(f"obs dim: {env.observation_space.shape[0]}")
    print(
        f"  stream1 (proprio):  {env._stream1_dim}\n"
        f"  stream2 (exterocep): {env._stream2_dim}\n"
        f"  stream3 (goal):      {env._stream3_dim}"
    )

    # ── check_env ─────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Running gymnasium.utils.env_checker.check_env ...")
    print(f"{'='*60}")

    checker_errors: list[str] = []
    checker_warnings: list[str] = []

    # Phrases that identify gymnasium-internal housekeeping warnings (not env bugs).
    _INTERNAL_WARN_PHRASES = (
        "check_env(warn",     # deprecated warn= parameter
        "not having a spec",  # render-mode spec warning when env is not gym.make'd
        "alternative render", # same render-mode check
    )

    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            # gymnasium 1.x removed the warn= parameter; call without it.
            check_env(env)
            print("check_env: PASSED")
        except Exception as exc:
            checker_errors.append(str(exc))
            print(f"check_env: FAILED — {exc}")

    # Only surface warnings that are about our env, not gymnasium's own API.
    for w in caught_warnings:
        msg = str(w.message)
        if not any(phrase in msg for phrase in _INTERNAL_WARN_PHRASES):
            checker_warnings.append(msg)
            print(f"check_env WARNING: {msg}")

    if not checker_errors and not checker_warnings:
        print("  (zero errors, zero warnings)")

    # ── 3 random rollouts ─────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Running 3 random episodes ...")
    print(f"{'='*60}")

    episode_stats: list[dict] = []
    for ep in range(3):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        comp_sums: dict[str, float] = {
            "height_progress": 0.0,
            "hold_match_bonus": 0.0,
            "alive_bonus": 0.0,
            "fall_penalty": 0.0,
            "finish_bonus": 0.0,
        }
        ep_len = 0
        term_reason = "truncated"
        last_info: dict = {}

        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            for k in comp_sums:
                comp_sums[k] += info.get(k, 0.0)
            ep_len += 1
            last_info = info
            done = terminated or truncated
            if terminated:
                term_reason = _termination_reason(info, fell_default=True)

        episode_stats.append({
            "ep": ep + 1,
            "length": ep_len,
            "total_reward": total_reward,
            "components": dict(comp_sums),
            "reason": term_reason,
        })

        print(f"\nEpisode {ep + 1}:")
        print(f"  Length      : {ep_len} steps")
        print(f"  Total reward: {total_reward:.3f}")
        print("  Components  :")
        for k, v in comp_sums.items():
            print(f"    {k}: {v:.3f}")
        print(f"  Termination : {term_reason}")

    env.close()

    # ── DAY 4 COMPLETE summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DAY 4 COMPLETE")
    print(f"{'='*60}")
    print(f"  nq={env._nj_pos + 7}  nv={env._nj_vel + 6}  nu={env._nu}")

    _tmp_routes = format1.load_routes(_MOONBOARD1)
    _tmp_route = _select_route(_tmp_routes)
    _xml = scene_mod.build_scene_xml(_tmp_route, _HUMANOID)
    _model = mujoco.MjModel.from_xml_string(_xml)
    pp = float(_model.opt.timestep) * 10
    print(f"  Policy period: {pp:.6f} s  (timestep={_model.opt.timestep} × substeps=10)")
    print(
        f"  Observation dim: {env._obs_dim}  "
        f"(stream1={env._stream1_dim}, stream2={env._stream2_dim}, stream3={env._stream3_dim})"
    )
    print(f"  Action space shape: {env.action_space.shape}")
    print(
        f"  check_env: {'PASS' if not checker_errors else 'FAIL'}  "
        f"({len(checker_warnings)} warnings, {len(checker_errors)} errors)"
    )
    print("  Episode rollouts:")
    for s in episode_stats:
        print(
            f"    ep{s['ep']}: len={s['length']}  "
            f"reward={s['total_reward']:.2f}  reason={s['reason']}"
        )


if __name__ == "__main__":
    main()
