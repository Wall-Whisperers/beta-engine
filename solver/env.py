"""Gymnasium environment for the climbing solver.

ClimbingEnv wraps the wall, body model, IK reachability, and Pymunk
physics into a standard gymnasium.Env that Stable Baselines 3 (PPO via
MaskablePPO) can train against.

Observation
-----------
A flat float32 Box of shape (OBS_DIM,):

  [0 : 8]              Limb positions relative to COM (4 limbs × x, y),
                       normalised by arm_length.
  [8 : 12]             Grip encoding: hold index / MAX_HOLDS per limb,
                       or −1 if ungripped, then shifted to [−1, 1].
  [12 : 12 + H*4]      Hold features (padded to MAX_HOLDS holds), each
                       (x_rel, y_rel, type_norm, is_finish):
                         x_rel / arm_length, y_rel / arm_length  (relative to COM)
                         type_norm ∈ [0, 1]  (positivity score)
                         is_finish ∈ {0, 1}
  [12 + H*4 : end]     Body state: (com_y_norm, com_vx_norm, com_vy_norm).

Total dim with MAX_HOLDS=64: 8 + 4 + 256 + 3 = 271.

Action
------
Discrete(4 × MAX_HOLDS).
  limb_idx  = action // MAX_HOLDS   (0=LH, 1=RH, 2=LF, 3=RF)
  hold_idx  = action %  MAX_HOLDS   (index into self._holds list)

Invalid actions (hold doesn't exist, out of reach, occupied, etc.) are
masked via `action_masks()` — compatible with sb3-contrib MaskablePPO.

Curriculum
----------
Each `reset()` call generates a fresh wall via `generate_wall()`.
Pass `options={"wall": <dict>}` to reset() to pin a specific wall
(useful for evaluation and A* baselines).

Physics
-------
After each discrete move the Pymunk `PhysicsClimber` settles for
`SETTLE_STEPS` sub-steps. If the COM falls (is_fallen) before settling,
the episode terminates with a fall penalty even if the IK/reachability
said the move was legal. This catches edge-case instabilities that the
geometric checks miss.
"""
from __future__ import annotations

import random
from typing import Any, Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from solver.body import BodyModel, LIMBS, HAND_LIMBS, FOOT_LIMBS, Limb
from solver.generate import GeneratorConfig, generate_wall
from solver.physics import PhysicsClimber
from solver.reachability import (
    Pose,
    estimate_com,
    is_stable,
    reachable_moves,
    pose_anatomy_ok,
)
from solver.wall import Wall, Hold, load_wall

# ── Constants ────────────────────────────────────────────────────────────────

MAX_HOLDS = 64          # observation / action space is always this size
LIMB_ORDER: list[Limb] = ["LH", "RH", "LF", "RF"]
LIMB_INDEX: dict[Limb, int] = {l: i for i, l in enumerate(LIMB_ORDER)}

# Observation slices
OBS_LIMB_POS   = slice(0, 8)          # 4 limbs × (x, y) rel to COM
OBS_GRIP_ENC   = slice(8, 12)         # 4 limbs, grip encoding
OBS_HOLDS      = slice(12, 12 + MAX_HOLDS * 4)   # MAX_HOLDS × 4 features
OBS_BODY       = slice(12 + MAX_HOLDS * 4, 12 + MAX_HOLDS * 4 + 3)
OBS_DIM        = 12 + MAX_HOLDS * 4 + 3

# Physics settle params
SETTLE_STEPS  = 40
SETTLE_DT     = 1 / 60.0

# Reward shaping (same semantics as rl_qlearn.py, now physics-augmented)
COMPLETION_BONUS    =  100.0
PROGRESS_PER_CM     =    0.05
EFFICIENCY_PENALTY  =    0.5
FALL_PENALTY        =   50.0
DEAD_END_PENALTY    =    5.0
MAX_STEPS           =   60


class ClimbingEnv(gym.Env):
    """Gymnasium climbing environment.

    Parameters
    ----------
    body : BodyModel | None
        Climber anthropometrics. None → default 175 cm / 175 cm.
    gen_config : GeneratorConfig | None
        Wall generator settings. None → defaults (12×18, difficulty 0.5).
    wall : dict | None
        Fixed wall JSON dict. If given, `reset()` always uses this wall
        instead of generating a new one (useful for eval / debugging).
    physics : bool
        If True (default), run Pymunk settle after each move. Set False
        to skip physics for faster A*-baseline evaluation.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        body: Optional[BodyModel] = None,
        gen_config: Optional[GeneratorConfig] = None,
        wall: Optional[dict] = None,
        physics: bool = True,
    ) -> None:
        super().__init__()
        self.body = body or BodyModel()
        self.gen_config = gen_config or GeneratorConfig()
        self._fixed_wall_dict = wall
        self._use_physics = physics

        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(4 * MAX_HOLDS)

        # Runtime state — populated in reset().
        self._wall: Optional[Wall] = None
        self._holds: list[Hold] = []
        self._pose: Optional[Pose] = None
        self._physics: Optional[PhysicsClimber] = None
        self._steps: int = 0
        self._finish_ids: set[str] = set()

    # ── gymnasium API ────────────────────────────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        rng = random.Random(seed)

        # Load / generate wall.
        if options and "wall" in options:
            wall_dict = options["wall"]
        elif self._fixed_wall_dict is not None:
            wall_dict = self._fixed_wall_dict
        else:
            cfg_seed = rng.randrange(10 ** 9)
            wall_dict = generate_wall(
                GeneratorConfig(**{**self.gen_config.__dict__, "seed": cfg_seed}),
                self.body,
            )
            if wall_dict is None:
                # Fallback: retry without a seed constraint.
                wall_dict = generate_wall(self.gen_config, self.body)
            if wall_dict is None:
                raise RuntimeError("Wall generator failed all retries.")

        self._wall = load_wall(wall_dict, cell_size_cm=self.gen_config.cell_size_cm)
        self._holds = self._wall.holds[:]
        self._finish_ids = {h.hold_id for h in self._wall.finishes()}

        # Find a valid starting pose.
        from solver.astar import starting_poses
        starts = starting_poses(self._wall, self.body)
        if not starts:
            # Extremely rare — generator verified A* solvability, so this
            # should never happen. Fall back to a fresh generate.
            return self.reset(seed=seed, options=None)

        self._pose = rng.choice(starts)
        self._steps = 0

        # Initialise physics.
        if self._use_physics:
            com = estimate_com(self._wall, self._pose, self.body)
            grips = self._grip_positions(self._pose)
            self._physics = PhysicsClimber(self.body, (float(com[0]), float(com[1])))
            for limb, pos in grips.items():
                self._physics.grip(limb, pos)

        return self._obs(), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        assert self._pose is not None, "Call reset() before step()."

        limb_idx = action // MAX_HOLDS
        hold_idx = action % MAX_HOLDS

        limb = LIMB_ORDER[limb_idx]

        # ── Validate action ──────────────────────────────────────────────
        if hold_idx >= len(self._holds):
            # Masked-out padding slot — shouldn't happen with MaskablePPO.
            return self._obs(), -EFFICIENCY_PENALTY, False, False, {"invalid": True}

        target = self._holds[hold_idx]
        new_pose = self._pose.with_limb(limb, target.hold_id)

        legal_moves = reachable_moves(self.body, self._wall, self._pose)
        if (limb, target.hold_id) not in legal_moves:
            # Illegal action slipped past the mask — count as a wasted step.
            self._steps += 1
            truncated = self._steps >= MAX_STEPS
            return self._obs(), -EFFICIENCY_PENALTY, False, truncated, {"illegal": True}

        # ── Apply move ───────────────────────────────────────────────────
        prev_com = estimate_com(self._wall, self._pose, self.body)
        self._pose = new_pose
        new_com = estimate_com(self._wall, self._pose, self.body)
        self._steps += 1

        # ── Physics settle ───────────────────────────────────────────────
        fallen = False
        if self._use_physics and self._physics is not None:
            # Reposition physics COM and update grips.
            grips = self._grip_positions(self._pose)
            self._physics.reset(
                (float(new_com[0]), float(new_com[1])), grips
            )
            self._physics.step(SETTLE_STEPS, SETTLE_DT)
            fallen = self._physics.is_fallen(floor_y=0.0)

        # ── Reward ───────────────────────────────────────────────────────
        reward = -EFFICIENCY_PENALTY

        if fallen:
            reward -= FALL_PENALTY
            return self._obs(), reward, True, False, {"fallen": True}

        # Progress: upward COM movement.
        dy = float(new_com[1] - prev_com[1])
        reward += PROGRESS_PER_CM * max(dy, 0.0)

        # Completion.
        done = (self._pose.LH in self._finish_ids) or (self._pose.RH in self._finish_ids)
        if done:
            reward += COMPLETION_BONUS

        # Step limit.
        truncated = (not done) and (self._steps >= MAX_STEPS)
        if truncated:
            reward -= DEAD_END_PENALTY

        return self._obs(), reward, done, truncated, {"success": done}

    def action_masks(self) -> np.ndarray:
        """Boolean mask of shape (4 × MAX_HOLDS,) for MaskablePPO.

        True = action is legal from the current pose.
        """
        mask = np.zeros(4 * MAX_HOLDS, dtype=bool)
        if self._pose is None:
            return mask

        legal = set(reachable_moves(self.body, self._wall, self._pose))
        for limb_idx, limb in enumerate(LIMB_ORDER):
            for hold_idx, hold in enumerate(self._holds):
                if (limb, hold.hold_id) in legal:
                    mask[limb_idx * MAX_HOLDS + hold_idx] = True
        return mask

    # ── Observation builder ──────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        if self._pose is None or self._wall is None:
            return obs

        com = estimate_com(self._wall, self._pose, self.body)
        arm = self.body.arm_length or 1.0

        # Build hold-id → index map for grip encoding.
        hold_id_to_idx = {h.hold_id: i for i, h in enumerate(self._holds)}

        # Limb positions relative to COM, normalised by arm length.
        for i, limb in enumerate(LIMB_ORDER):
            hid = self._pose.get(limb)
            if hid is not None:
                h = self._wall.by_id(hid)
                obs[OBS_LIMB_POS.start + i * 2]     = (h.x_cm - com[0]) / arm
                obs[OBS_LIMB_POS.start + i * 2 + 1] = (h.y_cm - com[1]) / arm
            # else stays 0.0

        # Grip encoding: normalised hold index, −1 if ungripped.
        for i, limb in enumerate(LIMB_ORDER):
            hid = self._pose.get(limb)
            if hid is not None and hid in hold_id_to_idx:
                obs[OBS_GRIP_ENC.start + i] = hold_id_to_idx[hid] / MAX_HOLDS
            else:
                obs[OBS_GRIP_ENC.start + i] = -1.0

        # Hold features (padded to MAX_HOLDS).
        for idx, hold in enumerate(self._holds[:MAX_HOLDS]):
            base = OBS_HOLDS.start + idx * 4
            obs[base]     = (hold.x_cm - com[0]) / arm
            obs[base + 1] = (hold.y_cm - com[1]) / arm
            obs[base + 2] = hold.positivity          # [0, 1]
            obs[base + 3] = 1.0 if hold.is_finish else 0.0

        # Body state.
        wall_h = self._wall.height_cm or 1.0
        obs[OBS_BODY.start] = float(com[1]) / wall_h  # progress [0, 1]
        if self._use_physics and self._physics is not None:
            vx, vy = self._physics.com_vel()
            max_v = 200.0  # cm/s — normalise velocity
            obs[OBS_BODY.start + 1] = np.clip(vx / max_v, -1.0, 1.0)
            obs[OBS_BODY.start + 2] = np.clip(vy / max_v, -1.0, 1.0)

        return obs

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _grip_positions(self, pose: Pose) -> dict[Limb, tuple[float, float]]:
        """World positions of all gripped holds."""
        out: dict[Limb, tuple[float, float]] = {}
        for limb in LIMBS:
            hid = pose.get(limb)
            if hid is not None:
                h = self._wall.by_id(hid)
                out[limb] = (h.x_cm, h.y_cm)
        return out

    # ── Convenience ──────────────────────────────────────────────────────────

    def current_pose(self) -> Optional[Pose]:
        return self._pose

    def current_wall(self) -> Optional[Wall]:
        return self._wall

    def decode_action(self, action: int) -> tuple[Limb, Optional[Hold]]:
        """Human-readable decode of an action index."""
        limb = LIMB_ORDER[action // MAX_HOLDS]
        hold_idx = action % MAX_HOLDS
        hold = self._holds[hold_idx] if hold_idx < len(self._holds) else None
        return limb, hold
