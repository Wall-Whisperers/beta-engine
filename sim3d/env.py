"""Gymnasium environment for the 3D climbing simulator.

Two flavours of action space, picked at construction time:

    "discrete-move"  (default) — high-level "move limb X to hold Y".
        action = limb_id * n_holds + hold_id
        Best for the kind of beta-finding RL the project wants.
        Episode is a sequence of moves; each move runs `move_frames`
        of physics afterwards to let the body settle.

    "continuous-joint" — direct joint-target control.
        action ∈ Box(-1, 1, (n_actuators,)) scaled to joint range.
        For learning low-level motor control. Much harder to train,
        but gives the agent full flexibility.

Observation (both modes) packs:
    - pelvis world pos (3)
    - pelvis quat (4)
    - centre-of-mass world pos (3)
    - joint qpos[7:] (n_actuators)
    - joint qvel[6:] (n_actuators)
    - per-limb tip world pos (4 × 3)
    - per-limb on-hold one-hot (4 × n_holds)  — masked to which holds
      the limb is *eligible* for (e.g. hands can't use foothold-only).
    - distance from highest hand to finish hold (1)

Reward (per step):
    + UPWARD_REWARD × (com_z - prev_com_z)         # progress
    + ON_FINISH_BONUS if a hand is on a finish hold
    - PER_STEP_PENALTY                             # be efficient
    - FALL_PENALTY (terminal) if pelvis_z < FALL_Z
    - SLIP_PENALTY × n_slips                       # don't blow up grips

Termination:
    - hand on finish hold for ≥ FINISH_HOLD_FRAMES
    - pelvis_z < FALL_Z (fell off)
    - max_steps reached (truncation)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "sim3d.env requires gymnasium. "
        "It's listed in requirements.txt — run `pip install -r requirements.txt`."
    ) from e

from sim3d import config as cfg
from sim3d.body import HAND_LIMBS, FOOT_LIMBS, LIMBS, ClimberProfile, Limb
from sim3d.world import Climb3DWorld
from solver.wall import Wall


@dataclass
class EnvConfig:
    action_mode: str = "discrete-move"           # or "continuous-joint"
    move_mode: str = "reach"                     # "snap" | "reach" | "dyno"
    move_frames: int = 60                        # physics frames between actions (1s @ 60Hz)
    max_steps: int = 60                          # episode cap
    fall_z: float = 0.20                         # below this pelvis-z = fall
    finish_hold_frames: int = 6                  # hand must stay on finish for this long
    upward_reward: float = 8.0
    on_finish_bonus: float = 100.0
    per_step_penalty: float = 0.02
    fall_penalty: float = 50.0
    slip_penalty: float = 5.0
    enable_slip: bool = True                     # hold-overload model on by default for training
    seed_pose: bool = True
    seed_kwargs: dict = field(default_factory=dict)


class Climbing3DEnv(gym.Env):
    """Single-wall single-climber Gymnasium environment.

    Used like any other gym env:

        from solver.wall import load_wall
        from sim3d import ClimberProfile
        from sim3d.env import Climbing3DEnv, EnvConfig

        env = Climbing3DEnv(load_wall("example-v2-boulder"))
        obs, info = env.reset()
        for _ in range(60):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            if term or trunc: break

    Drop-in compatible with Stable-Baselines3 `PPO("MlpPolicy", env)`
    and RLlib `AlgorithmConfig().environment(env_creator=...)`.
    """

    metadata = {"render_modes": ["pose-snapshot"], "render_fps": 30}

    def __init__(
        self,
        wall: Wall,
        profile: Optional[ClimberProfile] = None,
        config: Optional[EnvConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.wall = wall
        self.profile = profile or ClimberProfile()
        self.cfg_env = config or EnvConfig()
        self.render_mode = render_mode

        if self.cfg_env.action_mode not in ("discrete-move", "continuous-joint"):
            raise ValueError(f"unknown action_mode: {self.cfg_env.action_mode}")

        self.world = Climb3DWorld(wall, self.profile)
        self._hold_ids: list[str] = list(self.world._hold_meta_by_id.keys())
        self.n_holds = len(self._hold_ids)
        self._hold_index: dict[str, int] = {h: i for i, h in enumerate(self._hold_ids)}

        # Eligibility mask — hands can't use "foothold-only" holds.
        self._hand_eligible = np.array([
            not self.world._hold_meta_by_id[h]["is_foothold_only"]
            for h in self._hold_ids
        ], dtype=np.bool_)
        self._foot_eligible = np.ones(self.n_holds, dtype=np.bool_)  # feet can use any hold

        self._finish_hold_ids = [
            h for h in self._hold_ids
            if self.world._hold_meta_by_id[h]["is_finish"]
        ]
        if not self._finish_hold_ids:
            # Treat the highest hold as the finish if none marked.
            self._finish_hold_ids = [
                max(self._hold_ids,
                    key=lambda h: self.world._hold_meta_by_id[h]["world_pos"][2])
            ]
        self._finish_z = max(
            self.world._hold_meta_by_id[h]["world_pos"][2]
            for h in self._finish_hold_ids
        )

        # Action space
        n_act = self.world.model.nu
        if self.cfg_env.action_mode == "discrete-move":
            self.action_space = spaces.Discrete(4 * self.n_holds)
        else:
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(n_act,), dtype=np.float32,
            )
            # Cache joint ranges for action-scaling.
            self._act_lo = np.zeros(n_act, dtype=np.float64)
            self._act_hi = np.zeros(n_act, dtype=np.float64)
            for i in range(n_act):
                jid = int(self.world.model.actuator_trnid[i, 0])
                self._act_lo[i] = self.world.model.jnt_range[jid, 0]
                self._act_hi[i] = self.world.model.jnt_range[jid, 1]

        # Observation space — large flat vector.
        obs_dim = (
            3 + 4 + 3                              # pelvis pos, quat, com
            + n_act + n_act                        # qpos[7:], qvel[6:]
            + 4 * 3                                # 4 limb tips
            + 4 * self.n_holds                     # one-hot per limb
            + 1                                    # finish-distance
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        self._n_act = n_act
        self._step_count = 0
        self._finish_streak = 0
        self._prev_com_z = 0.0

    # ─── Gym API ──────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.world.reset()
        if self.cfg_env.seed_pose:
            kw = dict(self.cfg_env.seed_kwargs)
            if not kw:
                # Auto-seed from start holds + lowest two foothold-eligible holds.
                starts = self.wall.starts()
                if len(starts) >= 2:
                    kw["lh"] = starts[0].hold_id
                    kw["rh"] = starts[1].hold_id
                feet_candidates = sorted(
                    (h for h in self.wall.holds if h.hold_type == "foothold"),
                    key=lambda h: h.y_cm,
                )
                if len(feet_candidates) >= 2:
                    kw["lf"] = feet_candidates[0].hold_id
                    kw["rf"] = feet_candidates[1].hold_id
            self.world.seed_pose(**kw)
        else:
            # Reset and let the climber dangle from gravity.
            pass

        self._step_count = 0
        self._finish_streak = 0
        self._prev_com_z = float(self.world.com()[2])
        return self._obs(), self._info()

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        info: dict[str, Any] = {}
        slips = 0

        if self.cfg_env.action_mode == "discrete-move":
            limb_id = int(action) // self.n_holds
            hold_id_idx = int(action) % self.n_holds
            limb = LIMBS[limb_id % 4]
            hold_id = self._hold_ids[hold_id_idx]
            # Eligibility: hands can't use foothold-only.
            if limb in HAND_LIMBS and not self._hand_eligible[hold_id_idx]:
                info["invalid_action"] = "hand on foothold-only"
            else:
                self.world.move_limb(limb, hold_id, mode=self.cfg_env.move_mode)
            slips = self.world.step(
                self.cfg_env.move_frames,
                check_slip=self.cfg_env.enable_slip,
            )
        else:
            # Continuous: action ∈ [-1, 1] → joint angle in joint range.
            action = np.asarray(action, dtype=np.float64).clip(-1.0, 1.0)
            ctrl = 0.5 * (action + 1.0) * (self._act_hi - self._act_lo) + self._act_lo
            self.world.data.ctrl[:self._n_act] = ctrl
            slips = self.world.step(
                1, check_slip=self.cfg_env.enable_slip,
            )

        # ── Reward shaping ─────────────────────────────────────────
        com_z = float(self.world.com()[2])
        progress = com_z - self._prev_com_z
        self._prev_com_z = com_z

        reward = (
            self.cfg_env.upward_reward * progress
            - self.cfg_env.per_step_penalty
            - self.cfg_env.slip_penalty * slips
        )

        # Finish hold — either hand counts.
        on_finish = (
            self.world.on_hold("LH") in self._finish_hold_ids
            or self.world.on_hold("RH") in self._finish_hold_ids
        )
        if on_finish:
            self._finish_streak += 1
        else:
            self._finish_streak = 0

        terminated = False
        truncated = False
        pelvis_z = float(self.world.pelvis_pos()[2])
        if self._finish_streak >= self.cfg_env.finish_hold_frames:
            reward += self.cfg_env.on_finish_bonus
            terminated = True
            info["outcome"] = "completed"
        elif pelvis_z < self.cfg_env.fall_z:
            reward -= self.cfg_env.fall_penalty
            terminated = True
            info["outcome"] = "fell"
        elif self._step_count >= self.cfg_env.max_steps:
            truncated = True
            info["outcome"] = "timeout"

        info.update(self._info())
        info["slips"] = slips
        info["progress"] = progress
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> Optional[dict]:
        if self.render_mode == "pose-snapshot":
            return self.world.pose_snapshot()
        return None

    def close(self) -> None:
        # MuJoCo data is GC-managed, nothing to release.
        pass

    # ─── Action helpers ───────────────────────────────────────────────
    def encode_move(self, limb: Limb, hold_id: str) -> int:
        """For tests / scripted policies."""
        return LIMBS.index(limb) * self.n_holds + self._hold_index[hold_id]

    def decode_move(self, action: int) -> tuple[Limb, str]:
        return LIMBS[action // self.n_holds], self._hold_ids[action % self.n_holds]

    # ─── Observation builder ──────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        d = self.world.data
        m = self.world.model
        n_act = self._n_act

        pelvis = np.array(d.qpos[0:3])
        pelvis_quat = np.array(d.qpos[3:7])
        com = self.world.com()
        joint_pos = np.array(d.qpos[7: 7 + n_act])
        # qvel for non-free joints starts at index 6 (free is 6 DOF in qvel).
        joint_vel = np.array(d.qvel[6: 6 + n_act])

        tips = np.concatenate([self.world.limb_tip_pos(l) for l in LIMBS])

        onehot = np.zeros(4 * self.n_holds, dtype=np.float32)
        for li, l in enumerate(LIMBS):
            hid = self.world.on_hold(l)
            if hid is not None:
                onehot[li * self.n_holds + self._hold_index[hid]] = 1.0

        hand_z = max(
            float(self.world.limb_tip_pos("LH")[2]),
            float(self.world.limb_tip_pos("RH")[2]),
        )
        finish_dist = np.array([self._finish_z - hand_z], dtype=np.float32)

        obs = np.concatenate([
            pelvis, pelvis_quat, com,
            joint_pos, joint_vel,
            tips, onehot, finish_dist,
        ]).astype(np.float32)
        return obs

    def _info(self) -> dict[str, Any]:
        return {
            "t": float(self.world.data.time),
            "pelvis_z": float(self.world.pelvis_pos()[2]),
            "com": tuple(float(v) for v in self.world.com()),
            "limbs": {l: self.world.on_hold(l) for l in LIMBS},
            "step": self._step_count,
        }
