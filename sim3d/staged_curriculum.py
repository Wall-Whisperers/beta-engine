"""Task-stage curriculum for the 3D climber (NEXT_STEPS A1).

The plain height reward teaches the agent to *hang* but not to *climb*: from a
4-grip hang, releasing a hand to reach the next hold risks the −50 fall, so the
locally-optimal "hold on tight" blocks the globally-optimal "let go and reach
up". This is an exploration trap, not a reward-shape problem (confirmed
empirically: a 300k run learned to hang 8× longer but never moved upward).

This module breaks the trap with a staged curriculum that hands the agent the
reach primitive instead of making it stumble into a full climb:

    hang  →  reach-one  →  climb

* **hang** — survive N steps on the seed grips (clears fast; the height reward
  already solves it).
* **reach-one** — *combine both* exploration aids:
    1. **Reverse-curriculum start**: seed the body already high on the route and
       practise just the *last* hand move first; as it succeeds, walk the start
       back toward the bottom (learn the route end-first).
    2. **Dense reach reward**: a signed-potential pull of the designated mover
       hand toward its target hold, gated on the other 3 limbs staying anchored
       (so the old fall-and-swing farm can't return).
* **climb** — the full clean reward, now from a policy that knows each move.

``StagedCurriculumEnv`` auto-advances the stage (and, within reach-one, the
reverse-curriculum position) on rolling success rate, mirroring the difficulty
scheduler in ``curriculum.py``.

The route helpers (``extract_route`` / ``reach_one_seed``) are shared with the
feasibility probe so the probe validates the exact seeds the wrapper will use.
"""
from __future__ import annotations

import warnings
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sim3d.staged_curriculum requires gymnasium. "
        "Install project dependencies with:  pip install -r requirements.txt"
    ) from e

from sim3d.body import ClimberProfile
from sim3d.env import Climbing3DEnv, EnvConfig
from solver.generate import GeneratorConfig, generate_wall
from solver.wall import DEFAULT_CELL_SIZE_CM, Wall, load_wall


# ─── Route extraction (shared with the feasibility probe) ────────────────────

def extract_route(wall: Wall):
    """Return ``(hand_seq, footholds, finish_ids)`` for a wall.

    * ``hand_seq``  — hand-usable holds sorted bottom→top: the climb sequence.
    * ``footholds`` — foothold-type holds sorted bottom→top.
    * ``finish_ids`` — set of finish hold ids.
    """
    hand_seq = sorted(
        (h for h in wall.holds if h.usable_for_hand()),
        key=lambda h: (h.y_cm, h.x_cm),
    )
    footholds = sorted(
        (h for h in wall.holds if h.hold_type == "foothold"),
        key=lambda h: (h.y_cm, h.x_cm),
    )
    finish_ids = {h.hold_id for h in wall.holds if h.is_finish}
    return hand_seq, footholds, finish_ids


def n_reach_moves(hand_seq) -> int:
    """Number of practisable reach-one moves on a route (moves 1 .. n-2)."""
    return max(0, len(hand_seq) - 2)


def reach_one_seed(hand_seq, footholds, move_k: int, wall: Wall):
    """Build the seed + target for practising hand-move ``move_k``.

    The body is seeded in a stable **4-grip stance** — the *mover* hand on
    ``hand_seq[k-1]``, the *anchor* hand on ``hand_seq[k]``, and two feet — and
    must move the mover up to ``hand_seq[k+1]`` and grip it (an alternating-hands
    move, the way a climber actually moves). The mover starts *gripped* (a
    1-hand stance is too unstable to hang), so the env gives the **mover hand a
    deadband of 0** in reach-one mode (it releases readily under exploration)
    while the anchors keep the protective 0.5 deadband. Returns
    ``(seed_kwargs, mover_limb, target_hold_id)`` or ``None`` if ``k`` is out of
    range.
    """
    n = len(hand_seq)
    if not (1 <= move_k <= n - 2):
        return None
    mover_hold = hand_seq[move_k - 1]
    anchor_hold = hand_seq[move_k]
    target_hold = hand_seq[move_k + 1]

    # Assign hands by horizontal position: the leftward hold is the left hand.
    if mover_hold.grid_x <= anchor_hold.grid_x:
        mover_limb, anchor_limb = "LH", "RH"
    else:
        mover_limb, anchor_limb = "RH", "LH"

    seed: dict[str, str] = {}
    seed["lh" if mover_limb == "LH" else "rh"] = mover_hold.hold_id
    seed["lh" if anchor_limb == "LH" else "rh"] = anchor_hold.hold_id

    # Feet: prefer one foothold on each side, both below the anchor stance.
    below = [f for f in footholds if f.grid_y < anchor_hold.grid_y]
    cx = wall.cols / 2.0
    if len(below) >= 2:
        left = next((f for f in reversed(below) if f.grid_x <= cx), None)
        right = next((f for f in reversed(below) if f.grid_x > cx), None)
        if left is None:
            left = below[-1]
        if right is None or right is left:
            right = next((f for f in reversed(below) if f is not left), below[-1])
        seed["lf"] = left.hold_id
        seed["rf"] = right.hold_id
    elif below:
        seed["lf"] = seed["rf"] = below[-1].hold_id
    # else: no footholds below — seed_pose will hang from the hands only.

    return seed, mover_limb, target_hold.hold_id


# ─── Feasibility pre-vetting ─────────────────────────────────────────────────
# Empirically only ~half of reverse-curriculum stances actually hang, and many
# targets sit beyond arm's reach. A single infeasible move would stall the
# whole reverse curriculum (success-rate stuck at 0 ⇒ never advances), so we
# pre-vet every move at construction and keep only the usable ones.

def feasible_reach_moves(
    wall: Wall,
    profile: ClimberProfile,
    *,
    hang_steps: int = 40,
    max_reach_m: float = 0.65,
) -> list[dict]:
    """Return the reach-one moves whose reverse-curriculum seed pose both hangs
    (no fall under zero action) and whose target is within ``max_reach_m`` of
    the seeded mover tip. Each entry: ``{move_k, seed_kwargs, mover, target,
    reach_d0}``, ordered bottom→top. Reuses ONE probe env (mutating its config +
    re-seeding per move) so vetting a 30-move wall costs ~one env build.
    """
    hand_seq, footholds, _finish = extract_route(wall)
    nmoves = n_reach_moves(hand_seq)
    if nmoves <= 0:
        return []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = Climbing3DEnv(wall, profile=profile,
                            config=EnvConfig(task_mode="reach-one"))
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    out: list[dict] = []
    try:
        for k in range(1, nmoves + 1):
            spec = reach_one_seed(hand_seq, footholds, k, wall)
            if spec is None:
                continue
            seed_kwargs, mover, target = spec
            env.cfg_env.seed_kwargs = seed_kwargs
            env.cfg_env.reach_target_hold_id = target
            env.cfg_env.reach_mover_limb = mover
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    env.reset(seed=0)
            except Exception:  # noqa: BLE001
                continue
            n_grips = sum(1 for l in ("LH", "RH", "LF", "RF")
                          if env.world.on_hold(l) is not None)
            d0 = env._reach_one_dist()
            if n_grips < 3 or d0 > max_reach_m:
                continue
            fell = False
            for _ in range(hang_steps):
                _o, _r, term, _tr, info = env.step(zero)
                if term and info.get("outcome") == "fell":
                    fell = True
                    break
            if not fell:
                out.append({
                    "move_k": k, "seed_kwargs": seed_kwargs, "mover": mover,
                    "target": target, "reach_d0": round(float(d0), 3),
                })
    finally:
        env.close()
    return out


# ─── Staged curriculum env ───────────────────────────────────────────────────

@dataclass
class StagedCurriculumConfig:
    """Knobs for the hang → reach-one → climb task-stage curriculum."""
    # Wall: use a static wall id, or generate one (seeded, fixed for the run).
    wall: Optional[str] = None
    gen_seed: int = 7
    gen_difficulty: float = 0.0
    gen_cols: int = 12
    gen_rows: int = 20
    gen_cell_size_cm: float = DEFAULT_CELL_SIZE_CM
    gen_max_wall_attempts: int = 6        # retry seeds until enough feasible moves
    min_feasible_moves: int = 3
    # Scheduler.
    window: int = 30                      # rolling success window per (stage, rc_pos)
    up_threshold: float = 0.60            # advance when success_rate ≥ this
    # Give up on a reach-one move that won't train after this many episodes and
    # skip to the next one. Some moves pass the "does it hang" feasibility filter
    # but are not learnable (awkward target geometry); without this, one bad move
    # wedges the whole reverse curriculum at 0% forever (observed at rc 14).
    skip_after_episodes: int = 300
    # Stage rewards / caps.
    hang_target_steps: int = 60           # hang success = survive this many steps
    reach_one_coeff: float = 30.0
    reach_regrip_bonus: float = 50.0
    reach_episode_steps: int = 200        # reach-one truncation cap
    climb_episode_steps: int = 1000       # climb truncation cap
    # Feasibility vetting. 0.45 m keeps the first reaches short/completable —
    # at 0.65 m the agent learned the reach direction but couldn't close the
    # final gap to regrip within an episode.
    max_reach_m: float = 0.45


class StagedCurriculumEnv(gym.Env):
    """Auto-advancing hang → reach-one → climb curriculum on a fixed wall.

    One inner ``Climbing3DEnv`` is built once (the wall never changes), and the
    per-episode task is selected by mutating ``inner.cfg_env`` before
    ``reset`` — no MjModel rebuild per episode. The stage (and, within
    reach-one, the reverse-curriculum position ``rc_pos``) advances when the
    rolling success rate clears ``up_threshold``; the window is cleared on each
    advance so the next sub-task starts fresh.

    Reverse curriculum: ``rc_pos`` starts at the TOP feasible move (nearest the
    finish) and walks down toward the bottom as each move is mastered.

    Extra info keys: ``stage``, ``rc_pos``, ``rc_total``, ``stage_success_rate``.
    """

    metadata = Climbing3DEnv.metadata
    STAGES = ("hang", "reach-one", "climb")

    def __init__(
        self,
        config: Optional[StagedCurriculumConfig] = None,
        profile: Optional[ClimberProfile] = None,
        env_config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._scfg = config or StagedCurriculumConfig()
        self._profile = profile or ClimberProfile()
        self._base_cfg = env_config or EnvConfig()
        self._render_mode = render_mode

        self._wall, self._feasible = self._build_wall_and_vet()
        if len(self._feasible) < self._scfg.min_feasible_moves:
            raise RuntimeError(
                "StagedCurriculumEnv: only "
                f"{len(self._feasible)} feasible reach moves found "
                f"(need ≥ {self._scfg.min_feasible_moves}); try another gen_seed."
            )

        self._stage_idx = 0                      # index into STAGES
        self._rc_pos = len(self._feasible) - 1   # reverse curriculum: top move first
        self._history: deque[bool] = deque(maxlen=self._scfg.window)
        self._eps_since_advance = 0              # for the skip-stuck-move guard

        # One inner env, reused for the whole run (wall is fixed).
        self._env = Climbing3DEnv(
            self._wall, profile=self._profile,
            config=self._episode_cfg(), render_mode=self._render_mode,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    # ─── Wall build + feasibility vetting ────────────────────────────────────
    def _build_wall_and_vet(self):
        if self._scfg.wall:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wall = load_wall(self._scfg.wall)
            feas = feasible_reach_moves(
                wall, self._profile, max_reach_m=self._scfg.max_reach_m,
            )
            return wall, feas

        # Generate a wall; retry seeds until enough moves are feasible.
        for attempt in range(self._scfg.gen_max_wall_attempts):
            gen = GeneratorConfig(
                cols=self._scfg.gen_cols, rows=self._scfg.gen_rows,
                cell_size_cm=self._scfg.gen_cell_size_cm,
                difficulty=self._scfg.gen_difficulty,
                seed=self._scfg.gen_seed + attempt,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wd = generate_wall(gen)
            if wd is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wall = load_wall(wd, cell_size_cm=self._scfg.gen_cell_size_cm)
            feas = feasible_reach_moves(
                wall, self._profile, max_reach_m=self._scfg.max_reach_m,
            )
            if len(feas) >= self._scfg.min_feasible_moves:
                return wall, feas
        # Last wall (may be below the bar — __init__ will raise).
        return wall, feas

    # ─── Per-episode config ──────────────────────────────────────────────────
    @property
    def stage(self) -> str:
        return self.STAGES[self._stage_idx]

    def _episode_cfg(self) -> EnvConfig:
        """Build the inner EnvConfig for the current (stage, rc_pos)."""
        stage = self.stage
        common = dict(
            task_mode=stage,
            hang_target_steps=self._scfg.hang_target_steps,
            reach_one_coeff=self._scfg.reach_one_coeff,
            reach_regrip_bonus=self._scfg.reach_regrip_bonus,
        )
        if stage == "reach-one":
            mv = self._feasible[self._rc_pos]
            return replace(
                self._base_cfg, **common,
                seed_kwargs=dict(mv["seed_kwargs"]),
                reach_target_hold_id=mv["target"],
                reach_mover_limb=mv["mover"],
                max_steps=self._scfg.reach_episode_steps,
            )
        # hang / climb start from the default 4-grip start stance.
        return replace(
            self._base_cfg, **common,
            seed_kwargs={},
            reach_target_hold_id=None,
            reach_mover_limb=None,
            max_steps=(self._scfg.hang_target_steps + 10 if stage == "hang"
                       else self._scfg.climb_episode_steps),
        )

    # ─── Gym API ─────────────────────────────────────────────────────────────
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Re-point the (already-built) inner env at the current sub-task.
        self._env.cfg_env = self._episode_cfg()
        obs, info = self._env.reset(seed=seed, options=options)
        info.update(self._stage_info())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if terminated or truncated:
            self._history.append(info.get("outcome") == "completed")
            self._maybe_advance()
        info.update(self._stage_info())
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # ─── Scheduler ───────────────────────────────────────────────────────────
    @property
    def stage_success_rate(self) -> float:
        if not self._history:
            return 0.0
        return sum(self._history) / len(self._history)

    def _maybe_advance(self) -> None:
        """Promote on rolling success — or, within reach-one, give up on a move
        that won't train after `skip_after_episodes` and skip to the next one, so
        one unlearnable-but-hangable move can't wedge the whole curriculum.
        Reach-one walks rc_pos down (top → bottom); when the bottom is reached
        (mastered or skipped) advance to the full climb."""
        self._eps_since_advance += 1
        if len(self._history) < max(1, self._scfg.window // 2):
            return
        mastered = self.stage_success_rate >= self._scfg.up_threshold

        if self.stage == "hang":
            if mastered:
                self._stage_idx = 1                   # → reach-one
                self._rc_pos = len(self._feasible) - 1
                self._reset_subtask()
        elif self.stage == "reach-one":
            stuck = self._eps_since_advance >= self._scfg.skip_after_episodes
            if mastered or stuck:
                if self._rc_pos > 0:
                    self._rc_pos -= 1                  # walk start toward the bottom
                else:
                    self._stage_idx = 2               # → climb
                self._reset_subtask()
        # climb is terminal — no further advance.

    def _reset_subtask(self) -> None:
        self._history.clear()
        self._eps_since_advance = 0

    def _stage_info(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "rc_pos": int(self._rc_pos),
            "rc_total": len(self._feasible),
            "stage_success_rate": round(self.stage_success_rate, 4),
            "curriculum_difficulty": round(self._stage_idx / 2.0, 4),  # 0/.5/1
        }


# ─── Climb reverse curriculum ────────────────────────────────────────────────
# Per-move practice (reach-one) proved the agent can learn a reach but does NOT
# chain into a full climb: most isolated moves are unlearnable and they don't
# transfer. The reverse curriculum on the FULL climb instead seeds the body
# progressively lower down the route — always with the real FINISH as the goal —
# so the agent learns the climb end-first and each lower start reuses the
# upper-climb skill it already has. This is the textbook fix for hard sequential
# exploration (Florensa et al., reverse curriculum generation).

def _pick_feet(footholds, below_row: int, wall: Wall) -> dict[str, str]:
    """Two footholds below `below_row`, one per side where possible."""
    below = [f for f in footholds if f.grid_y < below_row]
    cx = wall.cols / 2.0
    out: dict[str, str] = {}
    if len(below) >= 2:
        left = next((f for f in reversed(below) if f.grid_x <= cx), None)
        right = next((f for f in reversed(below) if f.grid_x > cx), None)
        if left is None:
            left = below[-1]
        if right is None or right is left:
            right = next((f for f in reversed(below) if f is not left), below[-1])
        out["lf"], out["rf"] = left.hold_id, right.hold_id
    elif below:
        out["lf"] = out["rf"] = below[-1].hold_id
    return out


def _stance_seed(hold_a, hold_b, footholds, wall: Wall) -> dict[str, str]:
    """seed_kwargs for a 2-hand stance on hold_a/hold_b (assigned L/R by x) +
    two feet below the lower hand."""
    pair = sorted([hold_a, hold_b], key=lambda h: h.grid_x)
    seed = {"lh": pair[0].hold_id, "rh": pair[1].hold_id}
    seed.update(_pick_feet(footholds, min(hold_a.grid_y, hold_b.grid_y), wall))
    return seed


def feasible_climb_stances(
    wall: Wall, profile: ClimberProfile, *, hang_steps: int = 20,
) -> list[dict]:
    """Hangable 4-grip stances at each height for the climb reverse curriculum.
    Stance i = hands on consecutive route holds (hand_seq[i], hand_seq[i+1]) +
    feet below, excluding the finish (the agent must climb TO it). Ordered
    bottom→top; each vetted to hang under zero action. Each entry:
    ``{seed_kwargs, top_hand_row}``."""
    hand_seq, footholds, _finish = extract_route(wall)
    n = len(hand_seq)
    if n < 3:
        return []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = Climbing3DEnv(wall, profile=profile, config=EnvConfig(task_mode="climb"))
    zero = np.zeros(env.action_space.shape, dtype=np.float32)
    out: list[dict] = []
    try:
        for i in range(n - 1):
            a, b = hand_seq[i], hand_seq[i + 1]
            if a.is_finish or b.is_finish:
                continue  # don't seed on the finish — climb TO it
            seed = _stance_seed(a, b, footholds, wall)
            env.cfg_env.seed_kwargs = seed
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    env.reset(seed=0)
            except Exception:  # noqa: BLE001
                continue
            if sum(1 for l in ("LH", "RH", "LF", "RF")
                   if env.world.on_hold(l) is not None) < 3:
                continue
            fell = False
            for _ in range(hang_steps):
                _o, _r, term, _tr, info = env.step(zero)
                if term and info.get("outcome") == "fell":
                    fell = True
                    break
            if not fell:
                out.append({"seed_kwargs": seed,
                            "top_hand_row": max(a.grid_y, b.grid_y)})
    finally:
        env.close()
    return out


@dataclass
class ClimbCurriculumConfig:
    """Knobs for the full-climb reverse curriculum."""
    wall: Optional[str] = None
    gen_seed: int = 7
    gen_difficulty: float = 0.0
    gen_cols: int = 12
    gen_rows: int = 20
    gen_cell_size_cm: float = DEFAULT_CELL_SIZE_CM
    gen_max_wall_attempts: int = 6
    min_levels: int = 3
    window: int = 30
    up_threshold: float = 0.40        # top-out rate to drop the start lower
    skip_after_episodes: int = 400    # give up on a stuck level and drop anyway
    finish_approach_coeff: float = 50.0
    climb_episode_steps: int = 1000


class ClimbCurriculumEnv(gym.Env):
    """Reverse curriculum on the full climb: seed the body in a vetted stance at
    a height, run the full climb reward toward the finish, and lower the start
    one stance at a time as the top-out rate clears ``up_threshold`` (skip-stuck
    guard). rc_pos starts at the highest stance (nearest the finish) and walks
    down to the start. One inner env, reused (wall is fixed); the per-episode
    task is selected by mutating cfg_env before reset.

    info keys: stage='climb-reverse', rc_pos, rc_total, stage_success_rate.
    """

    metadata = Climbing3DEnv.metadata

    def __init__(self, config=None, profile=None, env_config=None, render_mode=None):
        super().__init__()
        self._ccfg = config or ClimbCurriculumConfig()
        self._profile = profile or ClimberProfile()
        self._base_cfg = env_config or EnvConfig()
        self._render_mode = render_mode

        self._wall, self._levels = self._build_wall_and_vet()
        if len(self._levels) < self._ccfg.min_levels:
            raise RuntimeError(
                f"ClimbCurriculumEnv: only {len(self._levels)} hangable stances "
                f"(need ≥ {self._ccfg.min_levels}); try another gen_seed."
            )
        self._rc = len(self._levels) - 1        # highest stance (near finish) first
        self._history: deque[bool] = deque(maxlen=self._ccfg.window)
        self._eps_since_advance = 0

        self._env = Climbing3DEnv(
            self._wall, profile=self._profile,
            config=self._episode_cfg(), render_mode=self._render_mode,
        )
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space

    def _build_wall_and_vet(self):
        if self._ccfg.wall:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wall = load_wall(self._ccfg.wall)
            return wall, feasible_climb_stances(wall, self._profile)
        for attempt in range(self._ccfg.gen_max_wall_attempts):
            gen = GeneratorConfig(
                cols=self._ccfg.gen_cols, rows=self._ccfg.gen_rows,
                cell_size_cm=self._ccfg.gen_cell_size_cm,
                difficulty=self._ccfg.gen_difficulty,
                seed=self._ccfg.gen_seed + attempt,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wd = generate_wall(gen)
            if wd is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wall = load_wall(wd, cell_size_cm=self._ccfg.gen_cell_size_cm)
            levels = feasible_climb_stances(wall, self._profile)
            if len(levels) >= self._ccfg.min_levels:
                return wall, levels
        return wall, levels

    def _episode_cfg(self) -> EnvConfig:
        lvl = self._levels[self._rc]
        return replace(
            self._base_cfg,
            task_mode="climb",
            seed_kwargs=dict(lvl["seed_kwargs"]),
            finish_approach_coeff=self._ccfg.finish_approach_coeff,
            reach_target_hold_id=None,
            reach_mover_limb=None,
            max_steps=self._ccfg.climb_episode_steps,
        )

    # ─── Gym API ─────────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._env.cfg_env = self._episode_cfg()
        obs, info = self._env.reset(seed=seed, options=options)
        info.update(self._info())
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        if terminated or truncated:
            self._history.append(info.get("outcome") == "completed")
            self._maybe_advance()
        info.update(self._info())
        return obs, reward, terminated, truncated, info

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    @property
    def success_rate(self) -> float:
        return sum(self._history) / len(self._history) if self._history else 0.0

    def _maybe_advance(self) -> None:
        """Drop the start one stance lower on a cleared top-out rate, or skip a
        stance that won't train after skip_after_episodes."""
        self._eps_since_advance += 1
        if len(self._history) < max(1, self._ccfg.window // 2):
            return
        mastered = self.success_rate >= self._ccfg.up_threshold
        stuck = self._eps_since_advance >= self._ccfg.skip_after_episodes
        if (mastered or stuck) and self._rc > 0:
            self._rc -= 1
            self._history.clear()
            self._eps_since_advance = 0

    def _info(self) -> dict[str, Any]:
        return {
            "stage": "climb-reverse",
            "rc_pos": int(self._rc),
            "rc_total": len(self._levels),
            "stage_success_rate": round(self.success_rate, 4),
            # rc_pos high (near finish) = early/easy → low difficulty; walking
            # down to 0 (full climb) = difficulty 1.
            "curriculum_difficulty": round(
                1.0 - self._rc / max(1, len(self._levels) - 1), 4),
        }
