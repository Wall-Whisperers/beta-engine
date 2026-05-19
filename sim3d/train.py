"""RL training entry point for the 3D climbing simulator.

Wraps Stable-Baselines3 PPO around `Climbing3DEnv`, logs episode
stats to CSV + TensorBoard, and saves the trained policy to a run
directory you can later replay in the viewer.

Stable-Baselines3 is pinned in the project requirements because this module
is the supported PPO training entry point. If it is missing, this module raises
a clear ImportError with the install command.

Run directory layout:

    data/runs/sim3d/<run_id>/
    ├── config.json          # training config (wall, profile, hyperparams)
    ├── episode_stats.csv    # one row per finished episode (reward, length, outcome, ...)
    ├── tb/                  # TensorBoard event files
    └── model.zip            # the trained PPO policy
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from solver.wall import load_wall
from sim3d.body import ClimberProfile
from sim3d.env import Climbing3DEnv, EnvConfig


def _require_sb3():
    try:
        import stable_baselines3 as sb3
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
        return sb3, BaseCallback, DummyVecEnv, SubprocVecEnv
    except ImportError as e:
        raise ImportError(
            "sim3d.train requires stable-baselines3. "
            "Install project dependencies with:  pip install -r requirements.txt"
        ) from e


@dataclass
class TrainConfig:
    wall: str = "example-v2-boulder"
    moonboard_file: Optional[str] = None
    moonboard_problem_id: Optional[int] = None
    moonboard_split: str = "train"            # train | validation | test when sampling a corpus
    split_seed: int = 42
    train_fraction: float = 0.80
    validation_fraction: float = 0.10
    moonboard_vertical_projection: bool = False
    height_cm: float = 175.0
    wingspan_cm: float = 175.0
    mass_kg: float = 70.0
    total_timesteps: int = 100_000
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 64
    gamma: float = 0.99
    seed: int = 42
    move_mode: str = "reach"            # snap is faster but less realistic
    start_mode: str = "seed"            # seed | ground-reach
    move_frames: int = 24               # shorter for snap, longer for reach
    max_episode_steps: int = 30
    enable_slip: bool = True
    body_intersection_penalty: float = 20.0
    out_dir: str = "data/runs/sim3d"
    run_id: Optional[str] = None        # auto-generated from timestamp if None
    n_envs: int = 1                     # parallel CPU env workers
    device: str = "auto"                # SB3/PyTorch device: auto | cpu | cuda


class _EpisodeStatsCallback:
    """SB3-compatible callback that streams episode stats to a CSV.

    SB3 already logs to TensorBoard out of the box. This adds a flat
    CSV that's easy to grep / plot externally — one row per completed
    episode, columns: episode, total_steps, reward, length, outcome.
    """

    def __init__(self, csv_path: Path, BaseCallback) -> None:
        self._csv_path = csv_path
        self._fh = None
        self._writer = None
        self._episode = 0
        self._BaseCallback = BaseCallback
        self._impl = None  # The real BaseCallback subclass

    def build(self):
        outer = self

        class _Cb(self._BaseCallback):
            def _on_training_start(self) -> None:
                outer._fh = open(outer._csv_path, "w", newline="", encoding="utf-8")
                outer._writer = csv.writer(outer._fh)
                outer._writer.writerow([
                    "episode", "total_steps", "reward", "length",
                    "outcome", "final_com_z", "n_slips", "body_intersections",
                ])

            def _on_step(self) -> bool:
                # SB3 stuffs episode info into self.locals when an
                # episode finishes. Each parallel env emits its own.
                infos = self.locals.get("infos")
                dones = self.locals.get("dones")
                if infos is None:
                    infos = []
                if dones is None:
                    dones = []
                if hasattr(dones, "__len__") and not isinstance(dones, list):
                    dones = list(dones)
                for done, info in zip(dones, infos):
                    if not done:
                        continue
                    outer._episode += 1
                    # Episode return / length are stuffed under "episode" by SB3's Monitor.
                    ep = info.get("episode", {}) or {}
                    outcome = info.get("outcome", "?")
                    com = info.get("com", (0, 0, 0))
                    outer._writer.writerow([
                        outer._episode,
                        self.num_timesteps,
                        round(float(ep.get("r", 0.0)), 3),
                        int(ep.get("l", 0)),
                        outcome,
                        round(float(com[2]), 3),
                        int(info.get("slips", 0)),
                        int(info.get("body_intersections", 0)),
                    ])
                outer._fh.flush()
                return True

            def _on_training_end(self) -> None:
                if outer._fh:
                    outer._fh.close()

        outer._impl = _Cb()
        return outer._impl


def _moonboard_splits_for_config(cfg: TrainConfig):
    if not cfg.moonboard_file or cfg.moonboard_problem_id is not None:
        return None
    from sim3d.moonboard import load_moonboard_corpus, split_moonboard_problems

    corpus = load_moonboard_corpus(cfg.moonboard_file)
    splits = split_moonboard_problems(
        corpus,
        seed=cfg.split_seed,
        train_fraction=cfg.train_fraction,
        validation_fraction=cfg.validation_fraction,
    )
    if cfg.moonboard_split not in splits:
        raise ValueError(
            f"unknown MoonBoard split {cfg.moonboard_split!r}; "
            f"expected one of {sorted(splits)}"
        )
    if not splits[cfg.moonboard_split]:
        raise ValueError(f"MoonBoard split {cfg.moonboard_split!r} is empty")
    return splits


def _make_env_factory(cfg: TrainConfig, moonboard_splits=None):
    """Returns a Gymnasium env factory for SB3's vec-env wrappers."""

    profile = ClimberProfile(
        height_cm=cfg.height_cm,
        wingspan_cm=cfg.wingspan_cm,
        mass_kg=cfg.mass_kg,
    )
    env_cfg = EnvConfig(
        move_mode=cfg.move_mode,
        start_mode=cfg.start_mode,
        move_frames=cfg.move_frames,
        max_steps=cfg.max_episode_steps,
        enable_slip=cfg.enable_slip,
        body_intersection_penalty=cfg.body_intersection_penalty,
    )

    def _factory():
        if cfg.moonboard_file:
            from sim3d.moonboard import (
                load_moonboard_problems, moonboard_problem_to_wall, find_problem,
            )
            if cfg.moonboard_problem_id is not None:
                problems = load_moonboard_problems(cfg.moonboard_file)
                problem = find_problem(problems, id=cfg.moonboard_problem_id)
                if problem is None:
                    raise ValueError(
                        f"problem id {cfg.moonboard_problem_id} not found in "
                        f"{cfg.moonboard_file}"
                    )
                wall = moonboard_problem_to_wall(
                    problem,
                    vertical_projection=cfg.moonboard_vertical_projection,
                )
                return Climbing3DEnv(wall, profile=profile, config=env_cfg)

            from sim3d.moonboard_env import MoonboardClimbing3DEnv

            assert moonboard_splits is not None
            return MoonboardClimbing3DEnv(
                moonboard_splits[cfg.moonboard_split],
                profile=profile,
                config=env_cfg,
                vertical_projection=cfg.moonboard_vertical_projection,
            )

        wall = load_wall(cfg.wall)
        return Climbing3DEnv(wall, profile=profile, config=env_cfg)

    return _factory


def train(cfg: TrainConfig) -> Path:
    sb3, BaseCallback, DummyVecEnv, SubprocVecEnv = _require_sb3()
    from stable_baselines3.common.monitor import Monitor

    if cfg.run_id is None:
        cfg.run_id = time.strftime("run_%Y%m%d_%H%M%S")
    out_dir = Path(cfg.out_dir) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save the config for reproducibility.
    with open(out_dir / "config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    moonboard_splits = _moonboard_splits_for_config(cfg)
    if moonboard_splits is not None:
        from sim3d.moonboard import moonboard_split_manifest

        with open(out_dir / "moonboard_splits.json", "w", encoding="utf-8") as f:
            json.dump(moonboard_split_manifest(moonboard_splits), f, indent=2)

    factory = _make_env_factory(cfg, moonboard_splits)

    def _make_monitored_env(rank: int):
        def _init():
            env = factory()
            env.reset(seed=cfg.seed + rank)
            return Monitor(env)
        return _init

    n_envs = max(1, int(cfg.n_envs))
    env_fns = [_make_monitored_env(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns) if n_envs == 1 else SubprocVecEnv(env_fns)

    # TensorBoard is an optional sub-dep. Enable logging only if it's
    # importable; otherwise SB3 errors out at .learn() time.
    tb_log = None
    try:
        import tensorboard  # noqa: F401
        (out_dir / "tb").mkdir(exist_ok=True)
        tb_log = str(out_dir / "tb")
    except ImportError:
        print("(tensorboard not installed — skipping TB logging. "
              "Install with `pip install tensorboard` to enable.)")

    model = sb3.PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        seed=cfg.seed,
        tensorboard_log=tb_log,
        device=cfg.device,
        verbose=1,
    )

    csv_cb = _EpisodeStatsCallback(out_dir / "episode_stats.csv", BaseCallback).build()
    model.learn(total_timesteps=cfg.total_timesteps, callback=csv_cb,
                progress_bar=False)
    model.save(out_dir / "model.zip")

    print(f"\nTraining complete. Run dir: {out_dir}")
    print(f"  model:        {out_dir / 'model.zip'}")
    print(f"  episode CSV:  {out_dir / 'episode_stats.csv'}")
    print(f"  env workers:   {n_envs}")
    print(f"  torch device:  {model.device}")
    if tb_log:
        print(f"  tensorboard:  tensorboard --logdir {out_dir / 'tb'}")
    replay_cmd = f"python -m sim3d --play {out_dir / 'model.zip'}"
    if cfg.moonboard_file:
        replay_cmd += f" --moonboard {cfg.moonboard_file}"
        if cfg.moonboard_problem_id is not None:
            replay_cmd += f" --problem {cfg.moonboard_problem_id}"
        elif moonboard_splits is not None:
            first = moonboard_splits[cfg.moonboard_split][0]
            replay_cmd += f" --problem {first.id} --moonboard-full-board"
        if cfg.moonboard_vertical_projection:
            replay_cmd += " --vertical-projection"
        if cfg.start_mode != "seed":
            replay_cmd += f" --start-mode {cfg.start_mode}"
    else:
        replay_cmd += f" --wall {cfg.wall}"
    replay_cmd += (
        f" --height {cfg.height_cm}"
        f" --wingspan {cfg.wingspan_cm}"
        f" --mass {cfg.mass_kg}"
        f" --move-mode {cfg.move_mode}"
        f" --play-frames {cfg.move_frames}"
    )
    print(f"  replay:       {replay_cmd}")
    return out_dir / "model.zip"


def play(model_path: str | Path, env: Climbing3DEnv, *, deterministic: bool = True) -> dict:
    """Run a single deterministic episode in the given env using a saved
    SB3 model. Returns the final info dict."""
    sb3, _BaseCallback, _DummyVecEnv, _SubprocVecEnv = _require_sb3()
    model = sb3.PPO.load(str(model_path), device="auto")
    obs, info = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, term, trunc, info = env.step(action)
        if term or trunc:
            return info


# ─── CLI ──────────────────────────────────────────────────────────────────
def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sim3d.train")
    p.add_argument("--wall", default="example-v2-boulder")
    p.add_argument("--moonboard", dest="moonboard_file")
    p.add_argument("--problem", dest="moonboard_problem_id", type=int,
                   help="Train on one MoonBoard problem id. Omit to sample a deterministic split.")
    p.add_argument("--moonboard-split", default="train", choices=("train", "validation", "test"),
                   help="MoonBoard corpus split to sample when --problem is omitted.")
    p.add_argument("--split-seed", type=int, default=42,
                   help="Deterministic MoonBoard train/validation/test split seed.")
    p.add_argument("--train-fraction", type=float, default=0.80)
    p.add_argument("--validation-fraction", type=float, default=0.10)
    p.add_argument("--vertical-projection", action="store_true",
                   help="Use MoonBoard vertical-projection geometry.")
    p.add_argument("--height", type=float, default=175.0)
    p.add_argument("--wingspan", type=float, default=175.0)
    p.add_argument("--mass", type=float, default=70.0)
    p.add_argument("--steps", type=int, default=100_000,
                   help="Total environment steps (PPO default budget).")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--n-steps", type=int, default=1024,
                   help="PPO rollout length per update.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--move-mode", default="reach", choices=("snap", "reach", "dyno"))
    p.add_argument("--start-mode", default="seed", choices=("seed", "ground-reach"),
                   help="seed starts welded on route holds; ground-reach starts on the floor and reaches to start hands.")
    p.add_argument("--move-frames", type=int, default=24)
    p.add_argument("--episode-steps", type=int, default=30)
    p.add_argument("--no-slip", "--no_slip", dest="no_slip", action="store_true")
    p.add_argument("--body-intersection-penalty", type=float, default=2.0,
                   help="Reward penalty per limb-vs-torso/pelvis intersection contact.")
    p.add_argument("--out-dir", default="data/runs/sim3d")
    p.add_argument("--run-id", default=None,
                   help="Custom run id; default = run_<timestamp>.")
    p.add_argument("--n-envs", type=int, default=1,
                   help="Parallel environment workers. Use >1 to speed up CPU-bound MuJoCo rollouts.")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"),
                   help="PyTorch device for PPO policy updates. Env simulation still runs on CPU.")
    args = p.parse_args(argv)

    cfg = TrainConfig(
        wall=args.wall,
        moonboard_file=args.moonboard_file,
        moonboard_problem_id=args.moonboard_problem_id,
        moonboard_split=args.moonboard_split,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        moonboard_vertical_projection=args.vertical_projection,
        height_cm=args.height,
        wingspan_cm=args.wingspan,
        mass_kg=args.mass,
        total_timesteps=args.steps,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        move_mode=args.move_mode,
        start_mode=args.start_mode,
        move_frames=args.move_frames,
        max_episode_steps=args.episode_steps,
        enable_slip=not args.no_slip,
        body_intersection_penalty=args.body_intersection_penalty,
        out_dir=args.out_dir,
        run_id=args.run_id,
        n_envs=args.n_envs,
        device=args.device,
    )
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
