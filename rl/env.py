"""Gymnasium environment wrapping the pymunk physics climber.

Action space:
    Discrete(num_limbs × num_holds) — encoded as
        action_index = limb_idx × num_holds + hold_idx
    The agent picks "move limb L to hold H" each step. Illegal moves
    (limb already on that hold; foothold target for a hand limb; etc.)
    are penalised but not blocked, so the agent learns to avoid them.

Observation:
    A flat float32 vector concatenating
      • COM position (x, y) in metres
      • COM velocity (vx, vy) in m/s
      • Per-hold occupancy mask (0/1 for each hold, length num_holds × 4
        — one block per limb)
      • Per-limb force fraction (4 floats; 0 if limb is in flight)
    See `_observation_dim()` for the exact length.

Reward (`config.py`-tunable, defaults reproduce solver/rl_qlearn.py):
    - per-step efficiency penalty
    - shaped progress reward (closer to nearest finish hold = positive)
    - completion bonus when a hand reaches a finish hold
    - illegal-move penalty
    - slip penalty if any attached limb is past its force budget

Episode end:
    `terminated = True` when a hand lands on a finish hold OR the
    physics goes unstable for too long. `truncated = True` after
    `max_steps` actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from physics.body import LIMBS, ClimberProfile, FOOT_LIMBS, HAND_LIMBS
from physics.world import ClimbWorld
from physics import config as cfg
from solver.wall import Wall


# ─── Reward shaping ───────────────────────────────────────────────────────
EFFICIENCY_PENALTY = 0.5            # per step
PROGRESS_REWARD_PER_CM = 0.05       # nearest finish, COM proximity
ILLEGAL_MOVE_PENALTY = 5.0          # tried to move to an unusable hold
SLIP_PENALTY = 10.0                 # any limb past its force budget
COMPLETION_BONUS = 100.0
MAX_STEPS_DEFAULT = 30
SETTLE_FRAMES_PER_STEP = 8          # physics frames simulated per action


@dataclass
class EnvConfig:
    """Tunable environment knobs separate from per-episode state."""

    max_steps: int = MAX_STEPS_DEFAULT
    settle_frames: int = SETTLE_FRAMES_PER_STEP
    seed_pose: Optional[dict[str, str]] = None
    """Override starting pose. Keys: 'lh','rh','lf','rf'."""


class ClimbingEnv(gym.Env):
    """Gymnasium env. One climber, one wall, one episode = one ascent attempt."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        wall: Wall,
        profile: Optional[ClimberProfile] = None,
        env_config: Optional[EnvConfig] = None,
    ) -> None:
        super().__init__()
        self.wall = wall
        self.profile = profile or ClimberProfile()
        self.env_config = env_config or EnvConfig()

        # Stable hold ordering. The action / observation indexing
        # depends on this — never reshuffle.
        self._hold_ids: list[str] = [h.hold_id for h in wall.holds]
        self._hold_idx: dict[str, int] = {hid: i for i, hid in enumerate(self._hold_ids)}
        self._finish_ids = {h.hold_id for h in wall.finishes()}

        n_holds = len(self._hold_ids)
        self.action_space = spaces.Discrete(len(LIMBS) * n_holds)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self._observation_dim(),),
            dtype=np.float32,
        )

        self.world: Optional[ClimbWorld] = None
        self._steps = 0
        self._last_progress: float = 0.0

    # ─── Gymnasium API ─────────────────────────────────────────────────────

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.world = ClimbWorld(self.wall, self.profile)
        starts = self._default_starts()
        self.world.seed_pose(**starts)
        # Settle briefly so the body is at rest before the first action.
        self.world.step(self.env_config.settle_frames)
        self._steps = 0
        self._last_progress = self._distance_to_finish()
        return self._obs(), {"pose": self.world.pose()}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self.world is not None, "Call reset() before step()."
        self._steps += 1
        limb_idx, hold_idx = divmod(int(action), len(self._hold_ids))
        limb = LIMBS[limb_idx]
        hold_id = self._hold_ids[hold_idx]
        hold = self.wall.by_id(hold_id)

        reward = -EFFICIENCY_PENALTY
        terminated = False
        truncated = False
        info: dict[str, Any] = {"limb": limb, "hold": hold_id}

        # ── Legality check ────────────────────────────────────────────
        # We *don't* reject illegal actions (the agent should learn) —
        # we charge a penalty and skip executing the move.
        legal = self._is_legal(limb, hold)
        if not legal:
            reward -= ILLEGAL_MOVE_PENALTY
            info["illegal"] = True
        else:
            self.world.move_limb(limb, hold_id, mode="snap")
            self.world.step(self.env_config.settle_frames)

        # ── Reward shaping ────────────────────────────────────────────
        new_progress = self._distance_to_finish()
        # `progress` shrinks as we get closer to the finish, so
        # (last - new) > 0 means the climber moved upward.
        reward += PROGRESS_REWARD_PER_CM * (self._last_progress - new_progress)
        self._last_progress = new_progress

        # Slip penalty: any attached limb maxed out on its force budget.
        for l in LIMBS:
            frac = self.world.body.per_limb_force_fraction(l)
            if frac is not None and frac >= 1.0:
                reward -= SLIP_PENALTY
                info["slipped"] = info.get("slipped", []) + [l]

        # ── Termination ──────────────────────────────────────────────
        pose = self.world.pose()
        if (pose["LH"] in self._finish_ids) or (pose["RH"] in self._finish_ids):
            reward += COMPLETION_BONUS
            terminated = True
            info["reason"] = "finish"
        elif self._steps >= self.env_config.max_steps:
            truncated = True
            info["reason"] = "timeout"

        info["pose"] = pose
        return self._obs(), float(reward), terminated, truncated, info

    def render(self):
        """Returns an RGB array of the current pose (Gymnasium 1.0 style)."""
        from physics.render import render_frame
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 8), dpi=90)
        fig.patch.set_facecolor("#0f172a")
        render_frame(self.world, ax, t=self._steps * cfg.PHYS_DT * cfg.SUBSTEPS_PER_FRAME)
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        return img

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _is_legal(self, limb, hold) -> bool:
        # No-op check: limb already on that hold.
        if self.world.on_hold(limb) == hold.hold_id:
            return False
        # Type-of-hold check.
        if limb in HAND_LIMBS and not hold.usable_for_hand():
            return False
        if limb in FOOT_LIMBS and not hold.usable_for_foot():
            return False
        # Reachability: anchor → hold within max reach for that limb.
        anchor = self.world.shoulder_or_hip_cm(limb)
        target = np.array([hold.x_cm, hold.y_cm])
        # Use 0.95 × physical limb length (the SlideJoint max) as the
        # legality bound; matches what physics will actually accept.
        max_reach_cm = self.world.body._max_reach[limb] * cfg.CM_PER_M * 0.95
        if float(np.linalg.norm(anchor - target)) > max_reach_cm:
            return False
        return True

    def _distance_to_finish(self) -> float:
        finishes = self.wall.finishes()
        if not finishes or self.world is None:
            return 0.0
        com = self.world.com_cm()
        ds = [
            float(np.linalg.norm(com - np.array([f.x_cm, f.y_cm])))
            for f in finishes
        ]
        return min(ds)

    def _default_starts(self) -> dict:
        if self.env_config.seed_pose is not None:
            return self.env_config.seed_pose
        starts = self.wall.starts()
        foot_holds = sorted(
            [h for h in self.wall.holds if h.usable_for_foot()],
            key=lambda h: h.y_cm,
        )
        out = {"lh": None, "rh": None, "lf": None, "rf": None}
        if len(starts) >= 2:
            sorted_starts = sorted(starts, key=lambda h: h.x_cm)
            out["lh"] = sorted_starts[0].hold_id
            out["rh"] = sorted_starts[-1].hold_id
        elif len(starts) == 1:
            out["lh"] = out["rh"] = starts[0].hold_id
        if len(foot_holds) >= 2:
            l, r = sorted(foot_holds[:2], key=lambda h: h.x_cm)
            out["lf"] = l.hold_id
            out["rf"] = r.hold_id
        return out

    # ─── Observation construction ─────────────────────────────────────────

    def _observation_dim(self) -> int:
        n_holds = len(self._hold_ids)
        # 4 (COM x,y,vx,vy) + 4 limbs × n_holds occupancy + 4 force fracs
        return 4 + 4 * n_holds + 4

    def _obs(self) -> np.ndarray:
        n_holds = len(self._hold_ids)
        out = np.zeros(self._observation_dim(), dtype=np.float32)
        if self.world is None:
            return out
        # COM position + velocity (in metres / m·s⁻¹).
        com = np.array(self.world.body.torso.position)
        vel = np.array(self.world.body.torso.velocity)
        out[0:2] = com
        out[2:4] = vel

        # Per-limb occupancy: a one-hot block per limb showing which
        # hold it's on (all zeros means in flight).
        for li, limb in enumerate(LIMBS):
            hid = self.world.on_hold(limb)
            if hid is not None:
                out[4 + li * n_holds + self._hold_idx[hid]] = 1.0

        # Per-limb force fraction.
        force_offset = 4 + 4 * n_holds
        for li, limb in enumerate(LIMBS):
            frac = self.world.body.per_limb_force_fraction(limb)
            out[force_offset + li] = float(frac) if frac is not None else 0.0

        return out

    def action_for(self, limb: str, hold_id: str) -> int:
        """Inverse of `divmod` in step(): pack (limb, hold) → discrete index."""
        return LIMBS.index(limb) * len(self._hold_ids) + self._hold_idx[hold_id]

    def legal_actions(self) -> list[int]:
        """All currently legal discrete-action indices. Useful for masked
        random policies / debugging — most RL agents won't use this."""
        out = []
        for li, limb in enumerate(LIMBS):
            for hi, hid in enumerate(self._hold_ids):
                if self._is_legal(limb, self.wall.by_id(hid)):
                    out.append(li * len(self._hold_ids) + hi)
        return out
