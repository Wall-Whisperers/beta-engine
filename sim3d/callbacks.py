"""Stable-Baselines3 callbacks for sim3d training.

Currently provides ``VideoRolloutCallback``: every ``eval_freq`` env steps it
runs one deterministic episode in a fresh eval env (separate from the train
vec env) and writes an mp4 next to the run's CSV.

Requires ``imageio`` for the mp4 writer. If it is not installed the callback
prints a one-time warning and becomes a no-op.
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

try:
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sim3d.callbacks requires stable-baselines3. "
        "Install with: pip install -r requirements.txt"
    ) from e


class FirstMidLastCheckpointCallback(BaseCallback):
    """Save the policy at three named points: first, mid, last.

    Args:
        out_dir: directory to write the .zip files into.
        total_timesteps: matches the ``learn(total_timesteps=...)`` call.
        first_at: env-step count at which to save ``model_first.zip``.
            Defaults to a small value so you get an "untrained-ish" baseline
            that still has the first PPO update applied (so it isn't pure
            random init).
    """

    def __init__(
        self,
        out_dir: str,
        total_timesteps: int,
        first_at: int = 1024,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._out_dir = out_dir
        self._first_at = int(first_at)
        self._mid_at = int(total_timesteps // 2)
        self._saved_first = False
        self._saved_mid = False
        os.makedirs(self._out_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if not self._saved_first and self.num_timesteps >= self._first_at:
            self._save("model_first.zip")
            self._saved_first = True
        if not self._saved_mid and self.num_timesteps >= self._mid_at:
            self._save("model_mid.zip")
            self._saved_mid = True
        return True

    def _on_training_end(self) -> None:
        # `model.zip` (final) is saved by train.py; we save `model_last.zip`
        # alongside it so the three named files have parallel names.
        self._save("model_last.zip")

    def _save(self, name: str) -> None:
        path = os.path.join(self._out_dir, name)
        self.model.save(path)
        if self.verbose:
            print(f"[FirstMidLastCheckpointCallback] saved {path} at step {self.num_timesteps}")


class VideoRolloutCallback(BaseCallback):
    """Save an mp4 rollout of the current policy every N training timesteps.

    Args:
        eval_env: a fresh env instance with ``render_mode="rgb_array"``-like
            access to its underlying MuJoCo model (we render via
            ``mujoco.Renderer`` against ``eval_env.world.data``).
        video_dir: directory to write videos into.
        eval_freq: env-steps between video captures (default 100_000).
        max_frames: cap per-episode frame count.
        verbose: 0/1 SB3 verbosity.
    """

    def __init__(
        self,
        eval_env,
        video_dir: str,
        eval_freq: int = 100_000,
        max_frames: int = 600,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._eval_env = eval_env
        self._video_dir = video_dir
        self._eval_freq = int(eval_freq)
        self._max_frames = int(max_frames)
        self._last_save = 0
        self._imageio_missing_warned = False
        os.makedirs(self._video_dir, exist_ok=True)

    def _on_training_start(self) -> None:
        # Capture an "untrained policy" baseline.
        self._save_video()

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self._eval_freq:
            self._save_video()
        return True

    def _save_video(self) -> None:
        self._last_save = self.num_timesteps
        try:
            import imageio.v2 as imageio
        except ImportError:
            try:
                import imageio  # type: ignore[no-redef]
            except ImportError:
                if not self._imageio_missing_warned:
                    print("[VideoRolloutCallback] imageio not installed — skipping video.")
                    self._imageio_missing_warned = True
                return

        try:
            import mujoco
        except ImportError:
            return

        mp4_path = os.path.join(
            self._video_dir, f"rollout_{self.num_timesteps:010d}.mp4"
        )

        env = self._eval_env
        # Resolve the underlying Climb3DWorld for rendering.
        underlying = env
        if hasattr(env, "_env"):  # MoonboardClimbing3DEnv wraps Climbing3DEnv
            underlying = env._env
        world = getattr(underlying, "world", None)
        if world is None:
            return

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat = [0.0, -0.3, 1.5]
        cam.distance = 6.0
        cam.azimuth = 235
        cam.elevation = -20
        try:
            renderer = mujoco.Renderer(world.model, height=480, width=640)
        except Exception:
            return

        frames: list[np.ndarray] = []
        obs, _ = env.reset()
        done = False
        steps = 0
        # eval_env's underlying world may have been swapped on reset (e.g.
        # MoonboardClimbing3DEnv rebuilds). Re-resolve for the renderer.
        if hasattr(env, "_env"):
            underlying = env._env
            world = getattr(underlying, "world", None)
            if world is not None:
                try:
                    renderer = mujoco.Renderer(world.model, height=480, width=640)
                except Exception:
                    renderer.close()
                    return

        while not done and steps < self._max_frames:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            renderer.update_scene(world.data, camera=cam)
            frames.append(renderer.render().copy())
            done = bool(terminated or truncated)
            steps += 1

        renderer.close()
        if frames:
            imageio.mimwrite(mp4_path, frames, fps=30)
            if self.verbose:
                print(f"[VideoRolloutCallback] saved {len(frames)} frames → {mp4_path}")
