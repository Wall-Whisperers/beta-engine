"""Procedural curriculum environment for the 3D climbing simulator.

Wraps ``Climbing3DEnv`` to generate a *new synthetic wall on every reset*,
with automatic difficulty scheduling driven by recent episode outcomes.

How the scheduler works
-----------------------
Every time an episode ends (``terminated`` or ``truncated``), the outcome is
recorded in a rolling window of the last ``window`` episodes.  The window
success-rate drives the difficulty:

    success_rate >= up_threshold   → difficulty += difficulty_step
    success_rate < down_threshold  → difficulty -= difficulty_step / 2

Difficulty is clamped to ``[min_difficulty, max_difficulty]`` after each
adjustment.  The scheduler only fires once the window has at least
``window // 2`` episodes in it (avoids jumping on the very first episode).

Why per-reset generation?
-------------------------
Generating a new wall each episode gives the policy maximum layout variety,
which is exactly what prevents it from memorising a single route.  The
generation + A* verification cost is ~50–200 ms; the MuJoCo world build
adds another ~100–300 ms.  With ``move_frames=24`` that is a ~15–20 %
overhead vs the physics — acceptable for the diversity benefit.

Because we fixed the obs/action space sizes in the previous change, the
spaces are stable across all generated walls and SB3 never sees a mismatch.

Usage
-----
    from sim3d.curriculum import CurriculumEnv, CurriculumConfig
    from sim3d.body import ClimberProfile
    from sim3d.env import EnvConfig

    env = CurriculumEnv(CurriculumConfig(), ClimberProfile(), EnvConfig())
    obs, info = env.reset()
    for _ in range(30):
        obs, rew, term, trunc, info = env.step(env.action_space.sample())
        print(info["curriculum_difficulty"], info["curriculum_success_rate"])
        if term or trunc:
            obs, info = env.reset()
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sim3d.curriculum requires gymnasium. "
        "Install project dependencies with:  pip install -r requirements.txt"
    ) from e

from sim3d.body import ClimberProfile
from sim3d.env import Climbing3DEnv, EnvConfig
from solver.generate import GeneratorConfig, generate_wall
from solver.wall import load_wall, DEFAULT_CELL_SIZE_CM


@dataclass
class CurriculumConfig:
    """Knobs for the curriculum difficulty scheduler and synthetic wall generator.

    Difficulty scheduler
    --------------------
    start_difficulty  Starting difficulty level (0.0 = easy jugs, 1.0 = hard).
    min_difficulty    Floor — difficulty never drops below this.
    max_difficulty    Ceiling — difficulty never rises above this.
    window            Rolling window size (number of recent episodes).
    up_threshold      Success rate (0–1) above which difficulty increases.
    down_threshold    Success rate (0–1) below which difficulty decreases.
    difficulty_step   Increment on an up-tick; halved for a down-tick.

    Wall generator
    --------------
    gen_cols          Grid columns for each generated wall.
    gen_rows          Grid rows.
    gen_cell_size_cm  Physical cell size in centimetres (sets real-world scale).
    gen_extra_holds   Scatter holds beyond the spine hold sequence.

    Fallback
    --------
    fallback_wall     Wall id to use if generation fails all retries.  None →
                      retry at a slightly easier difficulty before raising.
    """
    # Scheduler
    start_difficulty:  float = 0.0
    min_difficulty:    float = 0.0
    max_difficulty:    float = 1.0
    window:            int   = 20
    up_threshold:      float = 0.60
    down_threshold:    float = 0.20
    difficulty_step:   float = 0.05

    # Generator
    gen_cols:          int   = 12
    gen_rows:          int   = 20  # see GeneratorConfig.rows: start sits 2 rows up
    gen_cell_size_cm:  float = DEFAULT_CELL_SIZE_CM
    gen_extra_holds:   int   = 8

    # Fallback
    fallback_wall:     Optional[str] = "example-v2-boulder"


class CurriculumEnv(gym.Env):
    """Gymnasium env that generates a new synthetic wall on every episode.

    Observation and action spaces are identical to ``Climbing3DEnv`` and
    never change between episodes, so this is drop-in compatible with
    SB3 ``PPO("MlpPolicy", env)``.

    Additional info-dict keys (present on every ``step`` and ``reset``):
        curriculum_difficulty     float  current difficulty level (0–1)
        curriculum_success_rate   float  fraction of window episodes completed
        curriculum_window_size    int    how many episodes are in the window
        curriculum_wall_id        str    wall_id of the current episode's wall
    """

    metadata = Climbing3DEnv.metadata

    def __init__(
        self,
        curriculum_config: Optional[CurriculumConfig] = None,
        profile: Optional[ClimberProfile] = None,
        env_config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._ccfg = curriculum_config or CurriculumConfig()
        self._profile = profile or ClimberProfile()
        self._env_cfg = env_config or EnvConfig()
        self._render_mode = render_mode

        self.difficulty: float = float(self._ccfg.start_difficulty)
        self._history: deque[bool] = deque(maxlen=self._ccfg.window)
        self._current_wall_id: str = ""

        # Build one initial inner env to lock in stable obs/action spaces.
        # SB3 reads these once at construction time and never re-checks them.
        self._env = self._build_env(self.difficulty)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    # ─── Gym API ──────────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self._env = self._build_env(self.difficulty)
        obs, info = self._env.reset(seed=seed, options=options)
        info.update(self._curriculum_info())
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self._env.step(action)

        if terminated or truncated:
            success = info.get("outcome") == "completed"
            self._history.append(success)
            self._maybe_adjust_difficulty()

        info.update(self._curriculum_info())
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[dict]:
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    # ─── Helpers ──────────────────────────────────────────────────────────────

    @property
    def success_rate(self) -> float:
        """Fraction of recent episodes that ended with outcome='completed'."""
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def _curriculum_info(self) -> dict[str, Any]:
        return {
            "curriculum_difficulty":   round(self.difficulty, 4),
            "curriculum_success_rate": round(self.success_rate, 4),
            "curriculum_window_size":  len(self._history),
            "curriculum_wall_id":      self._current_wall_id,
        }

    def _maybe_adjust_difficulty(self) -> None:
        """Adjust difficulty based on the rolling success rate.

        Only fires once the window is at least half-full, to avoid
        over-reacting to the very first episodes of training.
        """
        min_data = max(1, self._ccfg.window // 2)
        if len(self._history) < min_data:
            return

        rate = self.success_rate
        if rate >= self._ccfg.up_threshold:
            self.difficulty = min(
                self._ccfg.max_difficulty,
                self.difficulty + self._ccfg.difficulty_step,
            )
        elif rate < self._ccfg.down_threshold:
            self.difficulty = max(
                self._ccfg.min_difficulty,
                self.difficulty - self._ccfg.difficulty_step / 2,
            )
        # Round to avoid floating-point drift accumulating over thousands of steps.
        self.difficulty = round(self.difficulty, 6)

    def _build_env(self, difficulty: float) -> Climbing3DEnv:
        """Generate a synthetic wall at `difficulty` and return a fresh inner env.

        Falls back to `fallback_wall` (or a slightly easier generation) if
        every retry attempt fails.
        """
        gen_cfg = GeneratorConfig(
            cols=self._ccfg.gen_cols,
            rows=self._ccfg.gen_rows,
            cell_size_cm=self._ccfg.gen_cell_size_cm,
            difficulty=difficulty,
            extra_holds=self._ccfg.gen_extra_holds,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress cell_size_cm default warning
            wall_dict = generate_wall(gen_cfg)

        if wall_dict is None:
            wall = self._fallback_wall(difficulty, gen_cfg)
        else:
            self._current_wall_id = wall_dict.get("wall_id", "synth-?")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wall = load_wall(wall_dict, cell_size_cm=self._ccfg.gen_cell_size_cm)

        return Climbing3DEnv(
            wall,
            profile=self._profile,
            config=self._env_cfg,
            render_mode=self._render_mode,
        )

    def _fallback_wall(self, difficulty: float, original_gen_cfg: GeneratorConfig):
        """Return a wall to use when generation fails all retries."""
        # Try generating at a slightly easier difficulty first.
        easier = max(self._ccfg.min_difficulty, difficulty - 0.15)
        if easier < difficulty:
            gen_cfg_easy = GeneratorConfig(
                cols=original_gen_cfg.cols,
                rows=original_gen_cfg.rows,
                cell_size_cm=original_gen_cfg.cell_size_cm,
                difficulty=easier,
                extra_holds=original_gen_cfg.extra_holds,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wall_dict = generate_wall(gen_cfg_easy)
            if wall_dict is not None:
                self._current_wall_id = wall_dict.get("wall_id", "synth-fallback")
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return load_wall(wall_dict, cell_size_cm=self._ccfg.gen_cell_size_cm)

        # Hard fallback: a known-good static wall.
        if self._ccfg.fallback_wall:
            warnings.warn(
                f"CurriculumEnv: wall generation failed at difficulty={difficulty:.2f}; "
                f"falling back to static wall {self._ccfg.fallback_wall!r}.",
                RuntimeWarning,
                stacklevel=3,
            )
            self._current_wall_id = self._ccfg.fallback_wall
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return load_wall(self._ccfg.fallback_wall)

        raise RuntimeError(
            f"CurriculumEnv: wall generation failed at difficulty={difficulty:.2f} "
            "and no fallback_wall is configured."
        )
