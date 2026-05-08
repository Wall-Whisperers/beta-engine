"""Day 5 Part B — PPO smoke test for MoonBoardEnv.

Runs 1M PPO steps on the highest-repeat V4 route, logging per-reward-component
scalars to TensorBoard and saving an mp4 rollout every 100k steps.

Usage:
    python3 scripts/train_ppo_smoke.py
    python3 scripts/train_ppo_smoke.py --seed 42
    python3 scripts/train_ppo_smoke.py --timesteps 200000   # quick smoke

Monitor TensorBoard while training:
    tensorboard --logdir moonboard-rl/output/tb_logs

Watch a saved video:
    open moonboard-rl/output/videos/smoke_0.mp4   # macOS
    xdg-open moonboard-rl/output/videos/smoke_0.mp4  # Linux

Load and evaluate the final model:
    python3 scripts/watch_random_agent.py --policy output/checkpoints/smoke_final.zip
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from src.parsers import format1
from src.envs.moonboard_env import MoonBoardEnv

_MOONBOARD1  = os.path.join(_PROJECT_ROOT, "moonboard_data", "moonboard1.json")
_HUMANOID    = os.path.join(_PROJECT_ROOT, "assets", "humanoid.xml")
_OUT_ROOT    = os.path.join(_PROJECT_ROOT, "output")
_TB_LOGDIR   = os.path.join(_OUT_ROOT, "tb_logs")
_CKPT_DIR    = os.path.join(_OUT_ROOT, "checkpoints")
_VIDEO_DIR   = os.path.join(_OUT_ROOT, "videos")

# ── PPO hyperparameters ───────────────────────────────────────────────────────
PPO_CONFIG = {
    "policy":        "MlpPolicy",
    "n_steps":        2048,
    "batch_size":     64,
    "n_epochs":       10,
    "learning_rate":  3e-4,
    "ent_coef":       0.01,
    "clip_range":     0.2,
    "gamma":          0.99,
    "gae_lambda":     0.95,
    "verbose":        1,
    "tensorboard_log": _TB_LOGDIR,
}

# Save an evaluation video every VIDEO_INTERVAL training timesteps.
VIDEO_INTERVAL = 100_000


def _select_route(routes):
    candidates = [r for r in routes if r.grade_v in (4, 5)]
    return max(candidates, key=lambda r: r.repeats)


def _make_env(route, seed: int = 0):
    """Return a factory function that creates a monitored MoonBoardEnv."""
    def _init():
        env = MoonBoardEnv(route=route, humanoid_xml_path=_HUMANOID)
        env = Monitor(env, filename=None)
        return env
    return _init


# ── Callback: log reward components to TensorBoard ───────────────────────────

class RewardComponentCallback(BaseCallback):
    """Reads per-step info dicts and logs mean reward components to TensorBoard.

    SB3 stores the info list for the most recent rollout batch in
    ``self.locals["infos"]``.  Each element is the info dict returned by
    ``env.step()`` for one environment.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self._component_keys = (
            "height_reward", "hold_match_bonus",
            "energy_penalty", "fall_penalty", "finish_bonus",
        )
        self._buf: dict[str, list[float]] = {k: [] for k in self._component_keys}

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            for k in self._component_keys:
                if k in info:
                    self._buf[k].append(float(info[k]))
        # Flush to TensorBoard every 512 steps.
        if self.n_calls % 512 == 0:
            for k, vals in self._buf.items():
                if vals:
                    self.logger.record(f"reward/{k}", float(np.mean(vals)))
                    self._buf[k].clear()
        return True


# ── Callback: render an evaluation episode and save as mp4 ───────────────────

class VideoRolloutCallback(BaseCallback):
    """Saves an mp4 of one deterministic evaluation episode every VIDEO_INTERVAL steps.

    Uses a separate eval env with render_mode="rgb_array".  Frames are written
    to output/videos/smoke_{step}.mp4 at 30 fps using imageio.
    """

    def __init__(self, route, eval_freq: int = VIDEO_INTERVAL, verbose: int = 0):
        super().__init__(verbose)
        self._route     = route
        self._eval_freq = eval_freq
        self._last_save = 0
        os.makedirs(_VIDEO_DIR, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save < self._eval_freq:
            return True

        self._last_save = self.num_timesteps
        mp4_path = os.path.join(_VIDEO_DIR, f"smoke_{self.num_timesteps}.mp4")

        try:
            import imageio
        except ImportError:
            print("[VideoRolloutCallback] imageio not installed — skipping video.")
            return True

        # Build a fresh eval env with rgb_array rendering.
        eval_env = MoonBoardEnv(
            route=self._route,
            humanoid_xml_path=_HUMANOID,
            render_mode="rgb_array",
        )

        frames: list[np.ndarray] = []
        obs, _ = eval_env.reset()
        done   = False
        steps  = 0

        try:
            import mujoco as _mj
            cam = _mj.MjvCamera()
            cam.type      = _mj.mjtCamera.mjCAMERA_FREE
            cam.lookat    = [0.0, -0.3, 1.5]
            cam.distance  = 6.0
            cam.azimuth   = 235
            cam.elevation = -20
            renderer = _mj.Renderer(eval_env._model, height=480, width=640)
        except Exception:
            renderer = None
            cam = None

        # Use the current model's policy (deterministic).
        while not done and steps < 300:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = eval_env.step(action)
            if renderer is not None:
                renderer.update_scene(eval_env._data, camera=cam)
                frames.append(renderer.render().copy())
            done = terminated or truncated
            steps += 1

        if renderer is not None:
            renderer.close()
        eval_env.close()

        if frames:
            imageio.mimwrite(mp4_path, frames, fps=30)
            if self.verbose:
                print(f"[VideoRolloutCallback] saved {len(frames)} frames → {mp4_path}")
        return True



# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO smoke test on MoonBoard-v0.")
    parser.add_argument("--seed",       type=int, default=0,         help="RNG seed.")
    parser.add_argument("--timesteps",  type=int, default=1_000_000, help="Total training timesteps.")
    args = parser.parse_args()

    os.makedirs(_TB_LOGDIR, exist_ok=True)
    os.makedirs(_CKPT_DIR,  exist_ok=True)
    os.makedirs(_VIDEO_DIR, exist_ok=True)

    # ── Load route ────────────────────────────────────────────────────────────
    routes = format1.load_routes(_MOONBOARD1)
    route  = _select_route(routes)
    print(f"Route: '{route.name}'  V{route.grade_v}  repeats={route.repeats}")

    # ── Build vectorised env ──────────────────────────────────────────────────
    vec_env = DummyVecEnv([_make_env(route, seed=args.seed)])

    # ── Instantiate PPO ───────────────────────────────────────────────────────
    model = PPO(
        env=vec_env,
        seed=args.seed,
        **PPO_CONFIG,
    )

    print(f"\nPPO config: {PPO_CONFIG}")
    print(f"Total timesteps: {args.timesteps:,}")
    print(f"TensorBoard log: {_TB_LOGDIR}")
    print(f"Videos:          {_VIDEO_DIR}")
    print(f"Checkpoint:      {_CKPT_DIR}/smoke_final.zip")
    print("\nTo monitor training:")
    print(f"  tensorboard --logdir {_TB_LOGDIR}\n")

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        RewardComponentCallback(verbose=1),
        VideoRolloutCallback(route=route, eval_freq=VIDEO_INTERVAL, verbose=1),
    ]

    # ── Train ─────────────────────────────────────────────────────────────────
    model.learn(
        total_timesteps=args.timesteps,
        callback=callbacks,
        progress_bar=False,
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    save_path = os.path.join(_CKPT_DIR, "smoke_final")
    model.save(save_path)
    print(f"\nModel saved → {save_path}.zip")
    print(f"\nTo watch the trained agent:")
    print(f"  mjpython scripts/watch_random_agent.py --policy {save_path}.zip")


if __name__ == "__main__":
    main()
