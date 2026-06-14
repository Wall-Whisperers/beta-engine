"""Deterministic evaluation harness for continuous sim3d policies.

Runs saved PPO checkpoints on fixed seeds and reports non-gameable progress
metrics (completion, max gripped-hold height, unique higher holds, fall rate).
This is intentionally continuous-only: evaluation always uses EnvConfig's
continuous-joint action mode and refuses the legacy ground-reach start helper.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Optional

import numpy as np

from solver.wall import load_wall
from sim3d.body import ClimberProfile
from sim3d.env import Climbing3DEnv, EnvConfig
from sim3d.train import _require_sb3


def _make_env(args) -> Climbing3DEnv:
    profile = ClimberProfile(
        height_cm=args.height,
        wingspan_cm=args.wingspan,
        mass_kg=args.mass,
    )
    env_cfg = EnvConfig(
        task_mode=args.task_mode,
        max_steps=args.episode_steps,
        start_mode="seed",
        enable_slip=not args.no_slip,
        grip_intent_deadband=args.grip_deadband,
        finish_approach_coeff=args.finish_approach_coeff,
        reach_approach_coeff=args.reach_approach_coeff,
        survival_bonus_coeff=args.survival_bonus_coeff,
        hwm_height_scale=args.hwm_height_scale,
        fall_penalty=args.fall_penalty,
        reach_max_target_dist=args.reach_max_target_dist,
        reach_require_higher_target=not args.allow_lateral_reach_targets,
    )
    wall = load_wall(args.wall)
    return Climbing3DEnv(wall, profile=profile, config=env_cfg)


def _load_obs_normalizer(model_path: Path, env: Climbing3DEnv):
    vn_path = model_path.parent / "vec_normalize.pkl"
    if not vn_path.exists():
        return None
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        dummy = DummyVecEnv([lambda: env])
        normalizer = VecNormalize.load(str(vn_path), dummy)
        normalizer.training = False
        normalizer.norm_reward = False
        return normalizer
    except Exception as e:  # noqa: BLE001
        print(f"warning: could not load VecNormalize stats from {vn_path}: {e}")
        return None


def _norm_obs(obs: np.ndarray, normalizer) -> np.ndarray:
    if normalizer is None:
        return obs
    return normalizer.normalize_obs(np.array([obs]))[0]


def evaluate(args) -> list[dict]:
    sb3, _BaseCallback, _DummyVecEnv, _SubprocVecEnv = _require_sb3()
    model_path = Path(args.model)
    model = sb3.PPO.load(str(model_path), device=args.device)
    env = _make_env(args)
    normalizer = _load_obs_normalizer(model_path, env)

    rows: list[dict] = []
    for ep in range(args.episodes):
        seed = args.seed + ep
        obs, info = env.reset(seed=seed)
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(_norm_obs(obs, normalizer), deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)

        row = {
            "episode": ep + 1,
            "seed": seed,
            "reward": round(total_reward, 3),
            "outcome": info.get("outcome", "?"),
            "task_mode": info.get("task_mode", args.task_mode),
            "length": int(info.get("step", 0)),
            "final_com_z": round(float(info.get("com", (0, 0, 0))[2]), 3),
            "max_com_z": round(float(info.get("max_com_z", 0.0)), 3),
            "max_pelvis_z": round(float(info.get("max_pelvis_z", 0.0)), 3),
            "max_grip_z": round(float(info.get("max_grip_z", 0.0)), 3),
            "unique_holds_gripped": int(info.get("unique_holds_gripped", 0)),
            "unique_higher_holds_gripped": int(info.get("unique_higher_holds_gripped", 0)),
            "finish_streak": int(info.get("finish_streak", 0)),
            "slips": int(info.get("slips", 0)),
            "body_intersections": int(info.get("body_intersections", 0)),
        }
        rows.append(row)
    env.close()
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sim3d.evaluate")
    p.add_argument("--model", required=True, help="Path to model.zip")
    p.add_argument("--wall", default="baby-v1")
    p.add_argument("--task-mode", choices=("hang", "reach-one", "climb"), default="climb")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=1000)
    p.add_argument("--episode-steps", type=int, default=1000)
    p.add_argument("--height", type=float, default=175.0)
    p.add_argument("--wingspan", type=float, default=175.0)
    p.add_argument("--mass", type=float, default=70.0)
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--no-slip", action="store_true")
    p.add_argument("--grip-deadband", type=float, default=0.5)
    p.add_argument("--finish-approach-coeff", type=float, default=100.0)
    p.add_argument("--reach-approach-coeff", type=float, default=20.0)
    p.add_argument("--survival-bonus-coeff", type=float, default=0.01)
    p.add_argument("--hwm-height-scale", type=float, default=50.0)
    p.add_argument("--fall-penalty", type=float, default=50.0)
    p.add_argument("--reach-max-target-dist", type=float, default=1.25)
    p.add_argument("--allow-lateral-reach-targets", action="store_true")
    p.add_argument("--csv", default="", help="Optional per-episode CSV output path")
    args = p.parse_args(argv)

    rows = evaluate(args)
    completed = sum(1 for r in rows if r["outcome"] == "completed")
    fell = sum(1 for r in rows if r["outcome"] == "fell")
    mean_reward = sum(float(r["reward"]) for r in rows) / len(rows)
    mean_max_grip = sum(float(r["max_grip_z"]) for r in rows) / len(rows)
    mean_unique_higher = sum(int(r["unique_higher_holds_gripped"]) for r in rows) / len(rows)

    print(f"episodes:        {len(rows)}")
    print(f"completion_rate: {completed / len(rows):.3f}")
    print(f"fall_rate:       {fell / len(rows):.3f}")
    print(f"mean_reward:     {mean_reward:+.3f}")
    print(f"mean_max_grip_z: {mean_max_grip:.3f}")
    print(f"mean_new_holds:  {mean_unique_higher:.3f}")
    if args.csv:
        _write_csv(Path(args.csv), rows)
        print(f"csv:             {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
