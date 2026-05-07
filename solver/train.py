"""PPO training for the climbing solver.

Uses MaskablePPO from sb3-contrib so invalid actions are zeroed out of
the policy distribution at every step — the agent never needs to learn
"don't pick a hold you can't reach", it only learns which legal move is
best.

Curriculum
----------
Training starts on easy walls (difficulty 0.2) and ramps up as the agent
improves. A rolling success-rate window tracks how often the agent reaches
a finish hold. When success rate exceeds `CURRICULUM_THRESHOLD` for the
current tier, difficulty is bumped by `CURRICULUM_STEP`. This runs inside
a custom SB3 callback so it happens automatically during `model.learn()`.

Usage
-----
    python -m solver train                          # default 500k steps
    python -m solver train --timesteps 2000000      # longer run
    python -m solver train --n-envs 8 --no-physics  # fast, no Pymunk
    python -m solver train --eval-wall big-wall     # pin an eval wall

Output is saved to data/runs/ppo-<timestamp>/  with:
    best_model.zip     best checkpoint by eval success rate
    final_model.zip    model at end of training
    training.log       episode rewards + curriculum progress
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

import numpy as np

from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

from solver.body import BodyModel
from solver.env import ClimbingEnv
from solver.generate import GeneratorConfig, generate_wall

# ── Hyperparameters ──────────────────────────────────────────────────────────

N_ENVS        = 4
TIMESTEPS     = 500_000
N_STEPS       = 1024      # rollout steps per env before each update
BATCH_SIZE    = 256
N_EPOCHS      = 4
GAMMA         = 0.99
GAE_LAMBDA    = 0.95
ENT_COEF      = 0.02      # exploration entropy — higher = more exploration
LR            = 3e-4
CLIP_RANGE    = 0.2
NET_ARCH      = [256, 256]

# ── Curriculum ───────────────────────────────────────────────────────────────

CURRICULUM_START     = 0.2   # starting difficulty
CURRICULUM_END       = 0.9   # max difficulty
CURRICULUM_STEP      = 0.1   # difficulty bump per tier
CURRICULUM_THRESHOLD = 0.40  # success rate needed to advance tier
CURRICULUM_WINDOW    = 100   # episodes to average for success rate

# ── Eval ─────────────────────────────────────────────────────────────────────

EVAL_FREQ      = 20_000    # steps between evals
EVAL_EPISODES  = 20        # episodes per eval run
EVAL_N_WALLS   = 10        # pre-generate this many fixed eval walls


# ── Callbacks ────────────────────────────────────────────────────────────────

class CurriculumCallback(BaseCallback):
    """Watches episode success rate and bumps wall difficulty when ready."""

    def __init__(self, envs: list[ClimbingEnv], verbose: int = 0) -> None:
        super().__init__(verbose)
        self._envs = envs
        self._difficulty = CURRICULUM_START
        self._successes: list[bool] = []
        self._log_path: Optional[Path] = None

    def set_log_path(self, path: Path) -> None:
        self._log_path = path

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for done, info in zip(dones, infos):
            if done:
                self._successes.append(bool(info.get("success", False)))

        if len(self._successes) >= CURRICULUM_WINDOW:
            rate = sum(self._successes[-CURRICULUM_WINDOW:]) / CURRICULUM_WINDOW
            if (rate >= CURRICULUM_THRESHOLD
                    and self._difficulty < CURRICULUM_END):
                self._difficulty = min(
                    CURRICULUM_END,
                    round(self._difficulty + CURRICULUM_STEP, 2),
                )
                self._update_envs()
                if self.verbose:
                    print(f"\n  [curriculum] difficulty → {self._difficulty:.2f}  "
                          f"(success rate {rate:.2%})")
                if self._log_path:
                    with self._log_path.open("a") as f:
                        f.write(f"step={self.num_timesteps}  "
                                f"difficulty={self._difficulty:.2f}  "
                                f"success_rate={rate:.4f}\n")

        return True

    def _update_envs(self) -> None:
        for env in self._envs:
            env.gen_config = GeneratorConfig(
                **{**env.gen_config.__dict__, "difficulty": self._difficulty}
            )


class EpisodeLogger(BaseCallback):
    """Logs mean episode reward to file every N episodes."""

    def __init__(self, log_path: Path, log_every: int = 200, verbose: int = 0):
        super().__init__(verbose)
        self._log_path = log_path
        self._log_every = log_every
        self._ep_rewards: list[float] = []
        self._ep_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("episode")
            if ep:
                self._ep_rewards.append(ep["r"])
                self._ep_count += 1
                if self._ep_count % self._log_every == 0:
                    mean_r = np.mean(self._ep_rewards[-self._log_every:])
                    with self._log_path.open("a") as f:
                        f.write(f"episodes={self._ep_count}  "
                                f"steps={self.num_timesteps}  "
                                f"mean_reward={mean_r:.3f}\n")
        return True


# ── Env factory ──────────────────────────────────────────────────────────────

def _make_masked_env(
    body: BodyModel,
    gen_config: GeneratorConfig,
    wall_dict: Optional[dict] = None,
    use_physics: bool = True,
) -> ActionMasker:
    env = ClimbingEnv(body=body, gen_config=gen_config,
                     wall=wall_dict, physics=use_physics)
    return ActionMasker(env, lambda e: e.action_masks())


def _make_vec_env(
    body: BodyModel,
    gen_config: GeneratorConfig,
    n_envs: int,
    use_physics: bool,
    use_subproc: bool,
) -> tuple[SubprocVecEnv | DummyVecEnv, list[ClimbingEnv]]:
    """Build a vectorised env and return it alongside the raw ClimbingEnv
    instances so the curriculum callback can update gen_config in place."""
    raw_envs: list[ClimbingEnv] = []

    def _factory(i: int):
        def _init():
            cfg = GeneratorConfig(**{**gen_config.__dict__, "seed": gen_config.seed or i * 997})
            env = ClimbingEnv(body=body, gen_config=cfg, physics=use_physics)
            raw_envs.append(env)
            return ActionMasker(env, lambda e: e.action_masks())
        return _init

    fns = [_factory(i) for i in range(n_envs)]
    # SubprocVecEnv gives real parallelism but raw_envs won't be in the
    # main process — use DummyVecEnv when we need curriculum access.
    vec_cls = DummyVecEnv  # SubprocVecEnv breaks in-process env references
    return vec_cls(fns), raw_envs


# ── Eval env ─────────────────────────────────────────────────────────────────

def _build_eval_env(
    body: BodyModel,
    gen_config: GeneratorConfig,
    n_walls: int,
    use_physics: bool,
) -> ActionMasker:
    """Single eval env cycling through N pre-generated walls."""
    walls = []
    for i in range(n_walls):
        cfg = GeneratorConfig(**{**gen_config.__dict__, "seed": 900_000 + i})
        w = generate_wall(cfg, body)
        if w:
            walls.append(w)

    class CyclingWallEnv(ClimbingEnv):
        def __init__(self):
            super().__init__(body=body, gen_config=gen_config,
                             wall=walls[0] if walls else None,
                             physics=use_physics)
            self._wall_cycle = walls
            self._idx = 0

        def reset(self, *, seed=None, options=None):
            if self._wall_cycle:
                self._idx = (self._idx + 1) % len(self._wall_cycle)
                options = {"wall": self._wall_cycle[self._idx]}
            return super().reset(seed=seed, options=options)

    env = CyclingWallEnv()
    return ActionMasker(env, lambda e: e.action_masks())


# ── Runs directory ───────────────────────────────────────────────────────────

def _runs_dir(tag: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for base in (Path("/data/runs"), Path(__file__).resolve().parent.parent / "data" / "runs"):
        try:
            d = base / f"ppo-{stamp}-{tag}"
            d.mkdir(parents=True, exist_ok=True)
            return d
        except (PermissionError, OSError):
            continue
    raise RuntimeError("Cannot create runs directory")


# ── Main ─────────────────────────────────────────────────────────────────────

def train(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="solver train", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--timesteps",   type=int,   default=TIMESTEPS)
    p.add_argument("--n-envs",      type=int,   default=N_ENVS)
    p.add_argument("--height-cm",   type=float, default=175.0)
    p.add_argument("--wingspan-cm", type=float, default=175.0)
    p.add_argument("--cols",        type=int,   default=12)
    p.add_argument("--rows",        type=int,   default=18)
    p.add_argument("--difficulty",  type=float, default=CURRICULUM_START,
                   help="Starting difficulty (curriculum ramps from here).")
    p.add_argument("--no-physics",  action="store_true",
                   help="Skip Pymunk settle — faster, no fall detection.")
    p.add_argument("--no-curriculum", action="store_true",
                   help="Fixed difficulty, no curriculum ramp.")
    p.add_argument("--eval-wall",   type=str, default=None,
                   help="Pin eval to a specific wall_id or JSON path.")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--tag",         type=str, default="run",
                   help="Label appended to the output directory name.")
    p.add_argument("--verbose",     type=int, default=1)
    args = p.parse_args(argv)

    warnings.filterwarnings("ignore")

    body = BodyModel(height_cm=args.height_cm, wingspan_cm=args.wingspan_cm)
    gen_config = GeneratorConfig(
        cols=args.cols,
        rows=args.rows,
        difficulty=args.difficulty,
        seed=args.seed,
    )
    use_physics = not args.no_physics

    runs = _runs_dir(args.tag)
    log_path = runs / "training.log"
    print(f"Output → {runs}")

    # ── Build envs ───────────────────────────────────────────────────────
    print(f"Building {args.n_envs} training envs …")
    vec_env, raw_envs = _make_vec_env(body, gen_config, args.n_envs, use_physics,
                                      use_subproc=False)

    # Eval env.
    if args.eval_wall:
        from solver.wall import load_wall
        eval_wall_dict = load_wall(args.eval_wall).__dict__  # crude but works for fixed
        eval_env = _make_masked_env(body, gen_config, use_physics=use_physics)
    else:
        eval_env = _build_eval_env(body, gen_config, EVAL_N_WALLS, use_physics)

    # ── Model ────────────────────────────────────────────────────────────
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        ent_coef=ENT_COEF,
        learning_rate=LR,
        clip_range=CLIP_RANGE,
        policy_kwargs={"net_arch": NET_ARCH},
        verbose=args.verbose,
        seed=args.seed,
        tensorboard_log=None,  # install tensorboard separately to enable
    )

    # ── Callbacks ────────────────────────────────────────────────────────
    callbacks = []

    if not args.no_curriculum:
        curr_cb = CurriculumCallback(raw_envs, verbose=args.verbose)
        curr_cb.set_log_path(log_path)
        callbacks.append(curr_cb)

    callbacks.append(EpisodeLogger(log_path, verbose=0))

    eval_cb = MaskableEvalCallback(
        eval_env,
        best_model_save_path=str(runs),
        log_path=str(runs),
        eval_freq=max(EVAL_FREQ // args.n_envs, 1),
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        verbose=args.verbose,
    )
    callbacks.append(eval_cb)

    # ── Train ────────────────────────────────────────────────────────────
    body_str = f"{body.height_cm:.0f}cm / {body.wingspan_cm:.0f}cm"
    print(f"Body: {body_str}   Physics: {use_physics}   "
          f"Curriculum: {not args.no_curriculum}")
    print(f"Training for {args.timesteps:,} timesteps …\n")

    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=CallbackList(callbacks),
        progress_bar=True,
    )
    elapsed = time.time() - t0

    # ── Save ─────────────────────────────────────────────────────────────
    final_path = runs / "final_model"
    model.save(str(final_path))
    print(f"\nTraining done in {elapsed:.0f}s")
    print(f"  final model → {final_path}.zip")
    print(f"  best model  → {runs / 'best_model.zip'}")

    return 0


if __name__ == "__main__":
    sys.exit(train())
