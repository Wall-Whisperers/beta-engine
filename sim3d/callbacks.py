"""Stable-Baselines3 callbacks for sim3d training.

Currently provides ``VideoRolloutCallback``: every ``eval_freq`` env steps it
runs one deterministic episode in a fresh eval env (separate from the train
vec env) and writes an mp4 next to the run's CSV.

Requires ``imageio`` for the mp4 writer. If it is not installed the callback
prints a one-time warning and becomes a no-op.
"""
from __future__ import annotations

import collections
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


class RollingBestCheckpointCallback(BaseCallback):
    """Crash-safe checkpoints + best-policy tracking for overnight runs.

    Writes three .zip files in ``out_dir``, each saved atomically (via
    write-to-temp + os.replace) so a kill mid-write cannot corrupt them:

        model_latest.zip         overwritten every ``save_freq`` env steps
        model_best_reward.zip    overwritten when rolling avg episode reward
                                 over the last ``window`` episodes hits a
                                 new high (with a small improvement floor
                                 so noise can't trigger constant rewrites)
        model_best_height.zip    overwritten when a new episode max progress
                                 height is observed across the run

    The two metrics tracked are intentional: rolling reward catches "the
    overall policy is getting better", and progress height catches "the
    climber actually moved upward" — neither alone is sufficient (a
    perfect-hang policy maxes reward but plateaus on higher holds; a one-time
    lucky height spike doesn't mean the policy learned anything).
    """

    REWARD_IMPROVE_FLOOR = 1.0   # require ≥ +1.0 reward gain to overwrite
    HEIGHT_IMPROVE_FLOOR = 0.005  # require ≥ 5 mm to overwrite

    def __init__(
        self,
        out_dir: str,
        save_freq: int = 25_000,
        window: int = 100,
        min_episodes_for_best: int = 25,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self._out_dir = out_dir
        self._save_freq = int(save_freq)
        self._window = int(window)
        self._min_for_best = int(min_episodes_for_best)
        self._last_latest_save = 0
        self._rewards: collections.deque[float] = collections.deque(maxlen=window)
        self._best_rolling_rew = -float("inf")
        self._best_com_z = -float("inf")
        os.makedirs(out_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # 1. Periodic crash-recovery snapshot.
        if self.num_timesteps - self._last_latest_save >= self._save_freq:
            self._save_atomic("model_latest.zip")
            self._last_latest_save = self.num_timesteps

        # 2. On episode terminations, update rolling stats + best-model files.
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
        if dones is None:
            dones = [False] * len(infos)
        for info, done in zip(infos, dones):
            if not done:
                continue
            # Monitor wraps each env and injects 'episode': {'r': total_r, 'l': len}
            ep = info.get("episode") if isinstance(info, dict) else None
            if ep is not None:
                self._rewards.append(float(ep["r"]))
                if len(self._rewards) >= self._min_for_best:
                    avg = sum(self._rewards) / len(self._rewards)
                    if avg > self._best_rolling_rew + self.REWARD_IMPROVE_FLOOR:
                        self._best_rolling_rew = avg
                        self._save_atomic("model_best_reward.zip")
                        if self.verbose:
                            print(f"[RollingBest] new best avg_rew={avg:+.2f} "
                                  f"at step {self.num_timesteps}")
            # Prefer non-gameable route progress: highest gripped hold.  Fall
            # back to max COM/final COM for older envs that don't report it.
            z = None
            if isinstance(info, dict):
                try:
                    z = float(info.get("max_grip_z", info.get("max_com_z")))
                except (TypeError, ValueError):
                    z = None
                if z is None:
                    com = info.get("com")
                    if com is not None:
                        try:
                            z = float(com[2])
                        except (TypeError, IndexError):
                            z = None
            if z is not None and z > self._best_com_z + self.HEIGHT_IMPROVE_FLOOR:
                self._best_com_z = z
                self._save_atomic("model_best_height.zip")
                if self.verbose:
                    print(f"[RollingBest] new best progress_z={z:.3f} m "
                          f"at step {self.num_timesteps}")
        return True

    def _on_training_end(self) -> None:
        # One final latest snapshot on graceful exit.
        self._save_atomic("model_latest.zip")

    def _save_atomic(self, name: str) -> None:
        """Atomic save: write to a sibling .tmp_<name> file, then os.replace.

        os.replace is atomic on both Windows and POSIX, so a crash during
        the underlying zip write leaves the previous (intact) file in
        place. A torn temp can be ignored on restart.

        SB3's PPO.save uses ``pathlib.Path.with_suffix(".zip")`` which
        REPLACES any existing suffix (so "foo.tmp" becomes "foo.zip", not
        "foo.tmp.zip"). To prevent SB3 from mangling our temp path we
        give it a name that already ends in ``.zip`` — Path.with_suffix
        leaves a matching suffix untouched.
        """
        final_path = os.path.join(self._out_dir, name)
        base_no_ext = name[:-4] if name.endswith(".zip") else name
        # The leading underscore-prefix lives ONLY on the temp file so
        # tools that glob *.zip don't get confused mid-write.
        tmp_path = os.path.join(self._out_dir, f"_tmp_{base_no_ext}.zip")
        try:
            self.model.save(tmp_path)
            if not os.path.exists(tmp_path):
                # Defensive: if SB3 ever changes its path handling and
                # writes elsewhere, fall back to a direct save and skip
                # atomicity rather than crash.
                self.model.save(final_path)
                return
            os.replace(tmp_path, final_path)
            # Also keep VecNormalize stats up-to-date so --resume can reload them.
            try:
                from stable_baselines3.common.vec_env import VecNormalize as _VN
                env = self.training_env
                if isinstance(env, _VN):
                    env.save(os.path.join(self._out_dir, "vec_normalize.pkl"))
            except Exception:
                pass
        except Exception as e:  # noqa: BLE001
            # Don't kill training over a checkpoint hiccup.
            if self.verbose:
                print(f"[RollingBest] save failed for {name}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass

class ContinuousTaskCurriculumCallback(BaseCallback):
    """Advance continuous task_mode based on rolling completion rate.

    This is intentionally orthogonal to wall-difficulty curriculum.  It keeps
    the action space continuous-joint for every stage and only changes the
    reward/termination target exposed through EnvConfig.task_mode.
    """

    def __init__(
        self,
        stages: list[str],
        *,
        window: int = 20,
        threshold: float = 0.80,
        min_episodes: int = 10,
        eval_env=None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        clean = [s.strip() for s in stages if s.strip()]
        if not clean:
            raise ValueError("ContinuousTaskCurriculumCallback requires at least one stage")
        for stage in clean:
            if stage not in ("hang", "reach-one", "climb"):
                raise ValueError(f"unknown task curriculum stage: {stage}")
        self._stages = clean
        self._window = max(1, int(window))
        self._threshold = float(threshold)
        self._min_episodes = max(1, int(min_episodes))
        self._stage_idx = 0
        self._history: collections.deque[bool] = collections.deque(maxlen=self._window)
        self._eval_env = eval_env

    @property
    def current_stage(self) -> str:
        return self._stages[self._stage_idx]

    def _on_training_start(self) -> None:
        self._apply_stage(self.current_stage)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", None)
        if dones is None:
            dones = [False] * len(infos)
        for info, done in zip(infos, dones):
            if not done or not isinstance(info, dict):
                continue
            if info.get("task_mode") != self.current_stage:
                continue
            self._history.append(info.get("outcome") == "completed")
            self._maybe_advance()
        return True

    def _maybe_advance(self) -> None:
        if self._stage_idx >= len(self._stages) - 1:
            return
        if len(self._history) < min(self._window, self._min_episodes):
            return
        rate = sum(self._history) / len(self._history)
        if rate < self._threshold:
            return
        old = self.current_stage
        self._stage_idx += 1
        self._history.clear()
        new = self.current_stage
        self._apply_stage(new)
        if self.verbose:
            print(
                f"[TaskCurriculum] advanced {old} -> {new} "
                f"at step {self.num_timesteps} (success_rate={rate:.2f})"
            )

    def _apply_stage(self, stage: str) -> None:
        try:
            self.training_env.env_method("set_task_mode", stage)
            if self._eval_env is not None and hasattr(self._eval_env, "set_task_mode"):
                self._eval_env.set_task_mode(stage)
        except Exception as e:  # noqa: BLE001
            if self.verbose:
                print(f"[TaskCurriculum] failed to set task_mode={stage}: {e}")


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
        # 3/4 side view: you can see the wall face AND the climber's depth
        # relative to it. Elevation -25 tilts down enough to see the floor
        # when the climber falls.
        cam.lookat = [0.0, 0.3, 1.4]
        cam.distance = 5.5
        cam.azimuth = 200
        cam.elevation = -25
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
            policy_obs = obs
            # If training used VecNormalize, the fresh eval env emits raw obs.
            # Feed the policy normalized obs so videos reflect the trained model.
            try:
                from stable_baselines3.common.vec_env import VecNormalize as _VN
                if isinstance(self.training_env, _VN):
                    policy_obs = self.training_env.normalize_obs(np.array([obs]))[0]
            except Exception:
                policy_obs = obs
            action, _ = self.model.predict(policy_obs, deterministic=True)
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
