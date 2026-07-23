"""MoonBoard-generalized Gymnasium environment helpers.

This wrapper keeps the Gym action/observation spaces stable by representing
MoonBoard problems on the full fixed 11×18 board, then sampling a new official
route at reset time. Off-route full-board holds are present for a fixed index
space but are masked by ``EnvConfig.official_route_only`` so hands and feet may
only attach to holds in the current problem.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sim3d.moonboard_env requires gymnasium. "
        "It's listed in requirements.txt — run `pip install -r requirements.txt`."
    ) from e

from sim3d.body import ClimberProfile, Limb
from sim3d.env import Climbing3DEnv, EnvConfig
from sim3d.moonboard import MoonboardProblem, moonboard_problem_to_wall


@dataclass(frozen=True)
class SampledProblemInfo:
    """Metadata for the MoonBoard problem active in the current episode."""

    id: int
    name: str
    grade: int | str
    setter: str

    @classmethod
    def from_problem(cls, problem: MoonboardProblem) -> "SampledProblemInfo":
        return cls(
            id=problem.id,
            name=problem.name,
            grade=problem.grade,
            setter=problem.setter,
        )


class MoonboardClimbing3DEnv(gym.Env):
    """Sample MoonBoard problems while preserving fixed Gym spaces.

    The wrapped ``Climbing3DEnv`` is rebuilt on reset with a full 198-hold
    MoonBoard wall for the sampled problem. Because every sampled wall contains
    the same 11×18 hold IDs, PPO sees stable spaces while the active route mask
    changes between episodes.
    """

    metadata = Climbing3DEnv.metadata

    def __init__(
        self,
        problems: list[MoonboardProblem],
        *,
        profile: Optional[ClimberProfile] = None,
        config: Optional[EnvConfig] = None,
        vertical_projection: bool = False,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not problems:
            raise ValueError("MoonboardClimbing3DEnv requires at least one problem")
        self.problems = list(problems)
        self.profile = profile or ClimberProfile()
        # MoonBoard climbs always include the kickboard for canonical
        # foot starts; route-only contacts are enforced.
        base_cfg = config or EnvConfig()
        self.cfg_env = replace(
            base_cfg, official_route_only=True, include_kickboard=True,
        )
        self.vertical_projection = vertical_projection
        self.render_mode = render_mode
        self._rng = None
        self._active_problem = self.problems[0]
        self._env = self._build_env(self._active_problem)
        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space

    @property
    def active_problem(self) -> MoonboardProblem:
        return self._active_problem

    def _build_env(self, problem: MoonboardProblem) -> Climbing3DEnv:
        wall = moonboard_problem_to_wall(
            problem,
            include_full_board=True,
            vertical_projection=self.vertical_projection,
        )
        return Climbing3DEnv(
            wall,
            profile=self.profile,
            config=self.cfg_env,
            render_mode=self.render_mode,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None or self._rng is None:
            self._rng = self.np_random
        problem_index = int(self._rng.integers(0, len(self.problems)))
        self._active_problem = self.problems[problem_index]
        self._env = self._build_env(self._active_problem)
        obs, info = self._env.reset(seed=seed, options=options)
        info = dict(info)
        info["moonboard_problem"] = SampledProblemInfo.from_problem(
            self._active_problem
        ).__dict__
        info["moonboard_problem_index"] = problem_index
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        info = dict(info)
        info["moonboard_problem"] = SampledProblemInfo.from_problem(
            self._active_problem
        ).__dict__
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self) -> None:
        self._env.close()

    def encode_move(self, limb: Limb, hold_id: str) -> int:
        return self._env.encode_move(limb, hold_id)

    def decode_move(self, action: int) -> tuple[Limb, str]:
        return self._env.decode_move(action)
