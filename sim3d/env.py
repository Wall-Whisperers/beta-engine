"""Gymnasium environment for the 3D climbing simulator.

Primary action mode: ``continuous-joint``. The agent emits a flat
``Box(nu + 4,)`` action — the first ``nu`` values are normalised joint
targets in [-1, 1] (rescaled per-joint), and the last 4 are per-limb grip
intents in [LH, RH, LF, RF]. A grip engages iff its intent is > 0 AND the
limb tip is within ``GRIP_PROXIMITY_M`` of an unoccupied, valid hold. Non-
positive intents release any active weld. There is no auto-grip.

``discrete-move`` mode is preserved behind ``EnvConfig.action_mode`` as a
debug / curriculum tool that picks (limb, hold) and uses the Cartesian-
impedance reach controller. It is NOT the default and should not be used
for the primary training experiments.

Observation: see ``sim3d.obs.build_observation``.

Reward (per step, continuous-joint) — designed to be un-hackable:
    + 50.0 × max(0, com_z − episode_max_com_z)      # HWM progress (state-based)
    + 10.0 rising-edge per (limb, hold) per episode # base match
    + 75.0 when a grip event raises episode_max_grip_z (NEW high mark)
    + 5.0  × (prev_finish_dist − cur_finish_dist)   # potential-based approach
    + 0.1  × weighted_grip_fraction (hand-gated)    # tiny stay-on-wall bonus
    − 1.0  × n_limbs_released_this_step             # discourages dangle, not climbing
    − 5.0  × n_slips
    − 20.0 × n_body_intersections                   # gating, not shaping
    − 0.25 if action was invalid (discrete-move only)
    − 0.0005 × Σ ctrl²                              # energy
    + 200.0 on_finish_bonus, − 50.0 fall_penalty (terminal)

`upward_velocity_coeff` is retained for back-compat but is now GATED on HWM
gain (so it becomes a multiplier on the HWM term, not a separate path-
dependent farm). Default 0.0 — leave it off unless you intentionally want
to scale HWM further.
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
from sim3d.obs import build_observation, observation_dim
from sim3d.world import Climb3DWorld
from solver.wall import Wall


@dataclass
class EnvConfig:
    action_mode: str = "continuous-joint"        # "continuous-joint" | "discrete-move"
    move_mode: str = "reach"                     # discrete-move only: "snap"|"reach"|"dyno"
    move_frames: int = 60                        # discrete-move only
    sim_substeps: int = 8                        # physics substeps per env.step() in continuous mode
    max_steps: int = 2000
    fall_z: float = 0.20
    finish_hold_frames: int = 6
    # Reward coefficients.
    hwm_height_scale: float = 50.0               # × max(0, com_z − max_com_z)
    hold_match_bonus: float = 10.0               # rising-edge first-touch
    # Bonus paid when a grip event raises the episode's max gripped-hold z.
    # Big discrete jackpot for *vertical* progress through grips; each height
    # level only pays once, so it can't be re-claimed by swapping limbs onto
    # holds at the same height.
    new_high_grip_bonus: float = 75.0
    on_finish_bonus: float = 200.0
    fall_penalty: float = 50.0
    slip_penalty: float = 5.0
    # Body intersection penalty raised to GATE (not shape) — coefficient
    # chosen so a single intersection wipes out roughly the largest possible
    # single-step HWM gain. This makes self-intersecting poses an outright
    # negative-EV action rather than a shaping nudge.
    body_intersection_penalty: float = 20.0
    energy_penalty_coeff: float = 0.001          # × Σ ctrl² (was 0.005 — too large vs HWM)
    invalid_action_penalty: float = 0.25
    # Dense finish-approach shaping (potential-based).
    # Per-step reward = finish_approach_coeff × (prev_dist − cur_dist).
    # Positive when the highest gripped hand moves closer to the finish hold.
    # Total reward for a full climb = coeff × route_length_m. At coeff=100
    # a 2 m route gives +200 approach reward — comparable to the finish bonus.
    # Must be large enough to beat the fall-penalty on every path to the top.
    finish_approach_coeff: float = 0.0
    # Per-step survival bonus — weighted fraction of limbs currently gripped.
    # Hands are weighted 2× feet (max value = coeff when all 4 limbs gripped).
    # Teaches "stay on the wall" before the height signal kicks in.
    # WARNING: any positive value creates a floor-hanging attractor. Only use
    # a tiny coeff (≤0.02) so the per-episode ceiling is small vs the fall
    # penalty. Default 0; use --finish-approach-coeff as the dense signal
    # instead.
    survival_bonus_coeff: float = 0.0
    # Penalty per limb that was gripping last step but isn't now. Without
    # this, releasing is free and dangling-off-the-wall becomes an attractor.
    grip_release_penalty: float = 0.0
    # DEPRECATED-as-of-2026-05-27: the original "per-step max(0, dz)"
    # semantics were exploitable — bouncing the pelvis in place earned
    # ~5 reward per up-frame with no downward penalty, dominating the
    # signal. Now GATED on HWM gain (only paid when com_z exceeds the
    # episode's previous max), so it behaves as a second coefficient on
    # the HWM term. Leave at 0 unless you intentionally want to amplify
    # HWM; the default HWM scale (50) is already the dominant signal.
    upward_velocity_coeff: float = 0.0
    enable_slip: bool = True
    # Grip intent deadband.  Any intent in (-grip_intent_deadband,
    # +grip_intent_deadband) leaves the current grip state unchanged.
    # A zero-centred Gaussian policy (SB3 default) releases grips on
    # ~50 % of steps when deadband=0, causing an immediate fall.  Set
    # to ~0.2 so near-zero actions default to "hold what you have".
    grip_intent_deadband: float = 0.0
    seed_pose: bool = True
    seed_kwargs: dict = field(default_factory=dict)
    start_mode: str = "seed"                     # "seed" | "ground-reach"
    official_route_only: bool = False
    reset_max_retries: int = 5
    include_kickboard: bool = False


class Climbing3DEnv(gym.Env):
    """Single-wall single-climber Gymnasium environment.

    Continuous-joint is the canonical training mode. Drop-in compatible
    with SB3 ``PPO("MlpPolicy", env)``.
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
        if self.cfg_env.start_mode not in ("seed", "ground-reach"):
            raise ValueError(f"unknown start_mode: {self.cfg_env.start_mode}")

        self.world = Climb3DWorld(
            wall, self.profile, include_kickboard=self.cfg_env.include_kickboard,
        )
        self._hold_ids: list[str] = list(self.world._hold_meta_by_id.keys())
        self.n_holds = len(self._hold_ids)
        self._hold_index: dict[str, int] = {h: i for i, h in enumerate(self._hold_ids)}

        self._route_eligible = np.array([
            self._is_official_route_hold(h) for h in self._hold_ids
        ], dtype=np.bool_)
        self._hand_eligible = np.array([
            (not self.world._hold_meta_by_id[h]["is_foothold_only"])
            and (not self.cfg_env.official_route_only or self._route_eligible[i])
            for i, h in enumerate(self._hold_ids)
        ], dtype=np.bool_)
        self._foot_eligible = np.array([
            (not self.cfg_env.official_route_only or self._route_eligible[i])
            for i, _h in enumerate(self._hold_ids)
        ], dtype=np.bool_)

        self._finish_hold_ids = [
            h for h in self._hold_ids
            if self.world._hold_meta_by_id[h]["is_finish"]
        ]
        if not self._finish_hold_ids:
            self._finish_hold_ids = [
                max(self._hold_ids,
                    key=lambda h: self.world._hold_meta_by_id[h]["world_pos"][2])
            ]

        # ── Action space ─────────────────────────────────────────────
        n_act = self.world.model.nu
        self._n_act = n_act
        # Cache per-joint ctrl ranges (for normalised → world rescaling).
        self._act_lo = np.zeros(n_act, dtype=np.float64)
        self._act_hi = np.zeros(n_act, dtype=np.float64)
        for i in range(n_act):
            jid = int(self.world.model.actuator_trnid[i, 0])
            self._act_lo[i] = self.world.model.jnt_range[jid, 0]
            self._act_hi[i] = self.world.model.jnt_range[jid, 1]

        if self.cfg_env.action_mode == "discrete-move":
            self.action_space = spaces.Discrete(4 * self.n_holds)
        else:
            # nu joint targets in [-1, 1] + 4 grip intents in [-1, 1].
            low = np.full(n_act + 4, -1.0, dtype=np.float32)
            high = np.full(n_act + 4, 1.0, dtype=np.float32)
            self.action_space = spaces.Box(
                low=low, high=high, shape=(n_act + 4,), dtype=np.float32,
            )

        # ── Observation space ────────────────────────────────────────
        obs_dim = observation_dim(self.world)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32,
        )

        # Episode state.
        self._step_count = 0
        self._finish_streak = 0
        self._max_com_z = 0.0
        self._holds_matched_this_episode: set[tuple[str, str]] = set()
        self._prev_grip: dict[Limb, Optional[str]] = {l: None for l in LIMBS}
        self._prev_finish_dist: float = 0.0    # for dense finish-approach shaping
        # Per-episode seed-pose joint targets; residual base for continuous
        # actions (set in reset()). Midpoint until the first reset runs.
        self._seed_ctrl = 0.5 * (self._act_lo + self._act_hi)

    # ─── Gym API ──────────────────────────────────────────────────────
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        last_err: Optional[str] = None
        for attempt in range(max(1, self.cfg_env.reset_max_retries)):
            self.world.reset()
            try:
                if self.cfg_env.seed_pose:
                    if self.cfg_env.start_mode == "ground-reach":
                        self._begin_ground_reach_start()
                    else:
                        kw = dict(self.cfg_env.seed_kwargs)
                        if not kw:
                            kw = self._default_seed_kwargs()
                        self.world.seed_pose(**kw)
            except Exception as e:  # noqa: BLE001
                last_err = str(e)
                continue

            n_grips = sum(
                1 for l in LIMBS if self.world.on_hold(l) is not None
            )
            if n_grips >= 3 or not self.cfg_env.seed_pose:
                break

            if attempt < self.cfg_env.reset_max_retries - 1 and self.np_random is not None:
                # Perturb pelvis xy with a small random offset so the next
                # settle starts from a slightly different basin.
                jitter = self.np_random.normal(scale=0.02, size=2)
                self.world.data.qpos[0] += float(jitter[0])
                self.world.data.qpos[1] += float(jitter[1])
        else:
            # Loop exhausted without break. Warn and proceed.
            print(
                f"[sim3d.env] WARNING: reset settled with <3 grips after "
                f"{self.cfg_env.reset_max_retries} attempts ({last_err}). "
                "Continuing — episode may be unstable."
            )

        self._step_count = 0
        self._finish_streak = 0
        # Init HWM at the seed com_z, not at 0. Initing at 0 gives the agent
        # a one-time "free" +hwm_scale×seed_height every reset (e.g. +60
        # when seed lands at 1.2 m) — an un-earned constant reward that can
        # be combined with survival bonus to make floor-hanging profitable
        # without ever climbing. Initing at the seed height means the agent
        # must climb ABOVE the seed position to earn any HWM reward.
        self._max_com_z = float(self.world.com()[2])
        # Track the highest hold-z any grip has reached this episode. Used
        # to gate the new_high_grip_bonus so each height level only pays
        # once per episode regardless of which limb arrives there.
        # Init at the highest *currently* gripped hold so the seed pose
        # doesn't get an unearned bonus on step 1.
        self._max_grip_z = self._current_max_grip_z()
        self._holds_matched_this_episode = set()
        self._prev_grip = {l: self.world.on_hold(l) for l in LIMBS}
        self._prev_finish_dist = self._finish_dist()
        self._prev_com_z = float(self.world.com()[2])
        # Snapshot the settled seed-pose joint targets. Continuous-joint
        # actions are interpreted as residuals around THIS pose (see
        # _step_continuous), so action≈0 means "hold the hang" rather than
        # "yank every joint to its ctrlrange midpoint".
        self._seed_ctrl = self.world.data.ctrl[: self._n_act].copy()
        return self._obs(), self._info()

    def _current_max_grip_z(self) -> float:
        """Highest z of any currently gripped hold; 0.0 if no grips."""
        zs = []
        for limb in LIMBS:
            hid = self.world.on_hold(limb)
            if hid is not None:
                zs.append(float(self.world._hold_meta_by_id[hid]["world_pos"][2]))
        return max(zs) if zs else 0.0

    # ─── Pose seeding helpers (unchanged from prior implementation) ──
    def _default_seed_kwargs(self) -> dict[str, str]:
        kw: dict[str, str] = {}
        starts = self.wall.starts()
        if len(starts) >= 2:
            kw["lh"] = starts[0].hold_id
            kw["rh"] = starts[1].hold_id
        elif len(starts) == 1:
            kw["lh"] = kw["rh"] = starts[0].hold_id

        feet_candidates = sorted(
            (h for h in self.wall.holds
             if h.hold_type == "foothold"
             and self._is_official_route_hold(h.hold_id)),
            key=lambda h: h.y_cm,
        )
        if len(feet_candidates) < 2:
            used_for_hands = {kw.get("lh"), kw.get("rh")}
            extras = sorted(
                (h for h in self.wall.holds
                 if h.hold_id not in used_for_hands
                 and not h.is_finish
                 and self._is_official_route_hold(h.hold_id)),
                key=lambda h: h.y_cm,
            )
            feet_candidates = (feet_candidates or []) + extras
        if len(feet_candidates) >= 2:
            left = next((h for h in feet_candidates if h.grid_x <= self.wall.cols / 2), None)
            right = next((h for h in feet_candidates if h.grid_x > self.wall.cols / 2), None)
            if left is None:
                left = feet_candidates[0]
            if right is None or right is left:
                right = feet_candidates[1]
            kw["lf"] = left.hold_id
            kw["rf"] = right.hold_id
        return kw

    def _start_hand_targets(self) -> tuple[str | None, str | None]:
        starts = sorted(self.wall.starts(), key=lambda h: h.x_cm)
        if len(starts) >= 2:
            return starts[0].hold_id, starts[-1].hold_id
        if len(starts) == 1:
            return starts[0].hold_id, starts[0].hold_id
        hand_low = sorted(
            [h for h in self.wall.holds
             if h.usable_for_hand() and self._is_official_route_hold(h.hold_id)],
            key=lambda h: (h.y_cm, h.x_cm),
        )[:2]
        if len(hand_low) >= 2:
            return hand_low[0].hold_id, hand_low[-1].hold_id
        if len(hand_low) == 1:
            return hand_low[0].hold_id, hand_low[0].hold_id
        return None, None

    def _begin_ground_reach_start(self) -> None:
        self.world._sync_actuator_targets_to_pose()
        lh, rh = self._start_hand_targets()
        if lh is not None:
            self.world.move_limb("LH", lh, mode=self.cfg_env.move_mode)
        if rh is not None:
            self.world.move_limb("RH", rh, mode=self.cfg_env.move_mode)

    # ─── Step ────────────────────────────────────────────────────────
    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        self._step_count += 1
        info: dict[str, Any] = {}
        invalid = False
        ctrl_l2_sq = 0.0

        if self.cfg_env.action_mode == "discrete-move":
            slips, invalid_reason = self._step_discrete(action)
            if invalid_reason is not None:
                invalid = True
                info["invalid_action"] = invalid_reason
        else:
            slips, ctrl_l2_sq = self._step_continuous(action)

        # ── Reward ───────────────────────────────────────────────────
        body_intersections = self.world.body_intersection_count()
        com_z = float(self.world.com()[2])

        # High-water-mark progress (state-based, un-hackable).
        height_reward = 0.0
        hwm_gain = 0.0
        if com_z > self._max_com_z:
            hwm_gain = com_z - self._max_com_z
            height_reward = self.cfg_env.hwm_height_scale * hwm_gain
            self._max_com_z = com_z

        # Upward-velocity term — GATED on HWM gain. With this gate the term
        # is mathematically equivalent to an extra scalar on hwm_height_scale
        # (un-farmable). The original "per-step max(0, dz)" semantics was
        # exploited by pelvis-bouncing in place. Default coeff = 0; leave
        # at 0 unless intentionally amplifying HWM.
        upward_reward = 0.0
        if self.cfg_env.upward_velocity_coeff > 0.0 and hwm_gain > 0.0:
            upward_reward = self.cfg_env.upward_velocity_coeff * hwm_gain
        self._prev_com_z = com_z

        # Rising-edge per (limb, hold) hold-match bonus + new-high-grip
        # bonus when this grip event raises the episode's max gripped-z.
        # Also count releases.
        match_bonus = 0.0
        high_grip_bonus = 0.0
        n_released = 0
        for limb in LIMBS:
            cur = self.world.on_hold(limb)
            prev = self._prev_grip.get(limb)
            if cur is not None and cur != prev:
                key = (limb, cur)
                if key not in self._holds_matched_this_episode:
                    self._holds_matched_this_episode.add(key)
                    match_bonus += self.cfg_env.hold_match_bonus
                # Check whether this grip advances the episode's max
                # gripped-hold z. Each height level only pays once per
                # episode — re-grabbing the same hold with another limb
                # doesn't trigger it (since it didn't raise the max).
                hold_z = float(self.world._hold_meta_by_id[cur]["world_pos"][2])
                if hold_z > self._max_grip_z:
                    high_grip_bonus += self.cfg_env.new_high_grip_bonus
                    self._max_grip_z = hold_z
            if prev is not None and cur is None:
                n_released += 1
            self._prev_grip[limb] = cur

        # Dense finish-approach shaping — potential-based so it cannot be
        # exploited by oscillating near the finish without touching it.
        # Φ(s) = −dist(highest_hand, finish).  Shaping = Φ(s') − Φ(s)
        #       = prev_dist − cur_dist  (positive when hand moved closer).
        approach_reward = 0.0
        if self.cfg_env.finish_approach_coeff > 0.0:
            cur_dist = self._finish_dist()
            approach_reward = self.cfg_env.finish_approach_coeff * (
                self._prev_finish_dist - cur_dist
            )
            self._prev_finish_dist = cur_dist

        # Per-step survival bonus: weighted fraction of limbs gripped, GATED
        # on at least one hand being engaged. Without the hand-gate the agent
        # learns to sit on the footholds with both hands free — technically
        # "on the wall" but the opposite of climbing. With the gate, no
        # bonus accrues until a hand is on a hold.
        # Hands count 2×, feet count 1×; max weighted sum = 6 → bonus is
        # capped at coeff/step when all 4 limbs are gripped.
        survival_reward = 0.0
        if self.cfg_env.survival_bonus_coeff > 0.0:
            n_hand = sum(1 for l in HAND_LIMBS if self.world.on_hold(l) is not None)
            if n_hand >= 1:
                n_foot = sum(1 for l in FOOT_LIMBS if self.world.on_hold(l) is not None)
                weighted = (2 * n_hand + n_foot) / 6.0
                survival_reward = self.cfg_env.survival_bonus_coeff * weighted

        # Penalty for each limb that released a grip this step. Combined with
        # the grip_intent_deadband this discourages the "let go and dangle"
        # local optimum.
        release_penalty = self.cfg_env.grip_release_penalty * n_released

        reward = (
            height_reward
            + upward_reward
            + approach_reward
            + survival_reward
            + match_bonus
            + high_grip_bonus
            - release_penalty
            - self.cfg_env.slip_penalty * slips
            - self.cfg_env.body_intersection_penalty * body_intersections
            - self.cfg_env.energy_penalty_coeff * ctrl_l2_sq
        )
        if invalid:
            reward -= self.cfg_env.invalid_action_penalty

        # ── Termination ──────────────────────────────────────────────
        terminated = False
        truncated = False
        pelvis_z = float(self.world.pelvis_pos()[2])
        on_finish = (
            self.world.on_hold("LH") in self._finish_hold_ids
            or self.world.on_hold("RH") in self._finish_hold_ids
        )
        if on_finish:
            self._finish_streak += 1
        else:
            self._finish_streak = 0

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
        info["body_intersections"] = body_intersections
        info["height_reward"] = float(height_reward)
        info["upward_reward"] = float(upward_reward)
        info["approach_reward"] = float(approach_reward)
        info["survival_reward"] = float(survival_reward)
        info["match_bonus"] = float(match_bonus)
        info["high_grip_bonus"] = float(high_grip_bonus)
        info["release_penalty"] = float(release_penalty)
        info["energy_penalty"] = float(self.cfg_env.energy_penalty_coeff * ctrl_l2_sq)
        return self._obs(), float(reward), terminated, truncated, info

    # ─── Action handling ─────────────────────────────────────────────
    def _step_continuous(self, action) -> tuple[int, float]:
        """Apply joint targets + grip intents; return (slips, Σctrl²)."""
        action = np.asarray(action, dtype=np.float64)
        n = self._n_act
        joint_norm = np.clip(action[:n], -1.0, 1.0)
        # Residual-around-seed mapping. action=0 holds the settled seed pose;
        # action=+1 drives a joint to its upper limit, -1 to its lower limit.
        # This keeps full reach authority while making "do nothing" == "hold
        # the hang", instead of the old midpoint mapping that yanked every
        # joint ~42° off the seed and slipped both hands on step 1.
        seed = self._seed_ctrl
        reach = np.where(joint_norm >= 0.0, self._act_hi - seed, seed - self._act_lo)
        ctrl = seed + joint_norm * reach
        self.world.data.ctrl[:n] = ctrl

        # Grip intents come *before* stepping physics so the welds can hold
        # the body through the upcoming substeps.
        #
        # Deadband semantics (grip_intent_deadband = db):
        #   intent >  db  →  try to engage (if tip near an eligible hold)
        #   intent < -db  →  release (if currently gripping)
        #   else          →  hold current grip state unchanged
        #
        # With SB3's default Gaussian init (mean=0, std=1), without a
        # deadband every limb releases on ~50% of steps → immediate fall.
        # A deadband of 0.2 combined with log_std_init=-1.5 (std≈0.22)
        # means ~85% of steps hold the current grip, giving PPO time to
        # discover the height reward before the body hits the floor.
        db = self.cfg_env.grip_intent_deadband
        intents = action[n: n + 4]
        for i, limb in enumerate(LIMBS):
            intent = float(intents[i]) if i < len(intents) else 0.0
            if intent > db:
                self._maybe_engage_grip(limb)
            elif intent < -db:
                if self.world.on_hold(limb) is not None:
                    self.world.release_limb(limb)
            # else: inside deadband — leave grip state unchanged

        slips = self.world.step(
            self.cfg_env.sim_substeps,
            check_slip=self.cfg_env.enable_slip,
        )
        return slips, float(np.sum(ctrl * ctrl))

    def _maybe_engage_grip(self, limb: Limb) -> None:
        """Engage the weld if the tip is within proximity of a valid,
        unoccupied hold. Picks the closest eligible hold within range."""
        if self.world.on_hold(limb) is not None:
            return
        tip = self.world.limb_tip_pos(limb)
        # Holds currently occupied by another limb are off limits.
        occupied = {
            self.world.on_hold(l) for l in LIMBS if l != limb
        }
        occupied.discard(None)

        best_id: Optional[str] = None
        best_d = cfg.GRIP_PROXIMITY_M
        for i, hid in enumerate(self._hold_ids):
            if hid in occupied:
                continue
            if limb in HAND_LIMBS and not self._hand_eligible[i]:
                continue
            if limb in FOOT_LIMBS and not self._foot_eligible[i]:
                continue
            meta = self.world._hold_meta_by_id[hid]
            d = float(np.linalg.norm(np.array(meta["world_pos"]) - tip))
            if d <= best_d:
                best_d = d
                best_id = hid
        if best_id is not None:
            self.world.attach_limb(limb, best_id)

    def _step_discrete(self, action) -> tuple[int, Optional[str]]:
        limb_id = int(action) // self.n_holds
        hold_id_idx = int(action) % self.n_holds
        limb = LIMBS[limb_id % 4]
        hold_id = self._hold_ids[hold_id_idx]
        invalid_reason = self._invalid_move_reason(limb, hold_id_idx)
        if invalid_reason is None:
            self.world.move_limb(limb, hold_id, mode=self.cfg_env.move_mode)
        slips = self.world.step(
            self.cfg_env.move_frames,
            check_slip=self.cfg_env.enable_slip,
        )
        return slips, invalid_reason

    # ─── Misc helpers ────────────────────────────────────────────────
    def render(self) -> Optional[dict]:
        if self.render_mode == "pose-snapshot":
            return self.world.pose_snapshot()
        return None

    def close(self) -> None:
        pass

    def _is_official_route_hold(self, hold_id: str) -> bool:
        meta = self.world._hold_meta_by_id[hold_id]
        return bool(
            meta["is_start"]
            or meta["is_finish"]
            or str(meta.get("color", "")).lower() != "#888888"
        )

    def _invalid_move_reason(self, limb: Limb, hold_id_idx: int) -> Optional[str]:
        if self.cfg_env.official_route_only and not self._route_eligible[hold_id_idx]:
            return "off-route hold"
        if limb in HAND_LIMBS and not self._hand_eligible[hold_id_idx]:
            return "hand on foothold-only"
        if limb in FOOT_LIMBS and not self._foot_eligible[hold_id_idx]:
            return "foot on ineligible hold"
        return None

    def encode_move(self, limb: Limb, hold_id: str) -> int:
        return LIMBS.index(limb) * self.n_holds + self._hold_index[hold_id]

    def decode_move(self, action: int) -> tuple[Limb, str]:
        return LIMBS[action // self.n_holds], self._hold_ids[action % self.n_holds]

    # ─── Helpers ─────────────────────────────────────────────────────
    def _finish_dist(self) -> float:
        """Euclidean distance from the highest gripped hand (or highest hand
        tip if no hand is gripped) to the nearest finish hold."""
        finish_positions = [
            np.array(self.world._hold_meta_by_id[h]["world_pos"])
            for h in self._finish_hold_ids
        ]
        hand_tips = [(self.world.limb_tip_pos(l), l) for l in ("LH", "RH")]
        gripped = [(pos, l) for pos, l in hand_tips
                   if self.world.on_hold(l) is not None]
        ref_pos = max(gripped or hand_tips, key=lambda pl: pl[0][2])[0]
        return float(min(
            np.linalg.norm(fp - ref_pos) for fp in finish_positions
        ))

    # ─── Observation / info ──────────────────────────────────────────
    def _obs(self) -> np.ndarray:
        return build_observation(self.world, self.cfg_env)

    def _info(self) -> dict[str, Any]:
        return {
            "t": float(self.world.data.time),
            "pelvis_z": float(self.world.pelvis_pos()[2]),
            "com": tuple(float(v) for v in self.world.com()),
            "limbs": {l: self.world.on_hold(l) for l in LIMBS},
            "step": self._step_count,
            "start_mode": self.cfg_env.start_mode,
        }
