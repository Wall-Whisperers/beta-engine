"""Gymnasium environment for MoonBoard climbing RL (Day 4).

Wraps a MuJoCo scene (wall + humanoid + grip constraints) as a standard
Gymnasium Env compatible with all standard RL libraries including SB3.

Observation  : flat Box of shape (133,), dtype float32
Action       : flat Box of shape (21,), dtype float32
                 first 17 — joint position targets (ctrlrange bounded)
                 last  4  — continuous grip intent signals in [-1, 1]

Day 4 target : env.reset() + env.step() pass check_env with zero errors/warnings.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.join(_DIR, "..", "..")
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.parsers.canonical import Route
from src.xml_gen.scene import build_scene_xml, LIMB_SITE_NAMES
from src.xml_gen.wall import hold_body_name, hold_position_world
from src.xml_gen.holds import RADIUS, _NY, _NZ
import src.grip.grip_manager as _gm_mod
from src.grip.grip_manager import GripManager

# ── Observation stream dimensions (derived from model at runtime) ──────────────
# Stream 1 — Proprioception:  17 + 17 + 3 + 6 + 3 + 3 + 12 + 4  = 65
# Stream 2 — Exteroception:   8 holds × 7 values                  = 56
# Stream 3 — Goal:            4 limbs × 3 values                  = 12
# Grand total                                                      = 133
_NUM_NEAR_HOLDS: int = 8
_HOLD_OBS_DIM: int = 7  # 3 rel_pos + 3 role_onehot + 1 gripping_flag

# Reset pose constants — place humanoid with both hands close to the start hold
# (hold_5_5, sphere centre ≈ (0.0, -0.955, 0.887)).
# Grid-search confirmed lhand dist=0.167 m, rhand dist=0.178 m from start hold.
# Proximity threshold is temporarily relaxed to 0.20 during reset (see reset()).
_RESET_TORSO_X: float = 0.35
_RESET_TORSO_Y: float = -0.95
_RESET_TORSO_Z: float = 0.80
# Quaternion [w, x, y, z] for 180° rotation about Z (torso faces −Y, toward wall).
_RESET_QUAT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

# Proximity threshold to use temporarily during reset so the default A-pose
# (arms not raised) can engage both start-hold grips.
_RESET_PROXIMITY: float = 0.20
_RESET_ALIGNMENT: float = -1.0


class MoonBoardEnv(gym.Env):
    """Gymnasium environment for MoonBoard climbing simulation.

    A MuJoCo humanoid is controlled by joint position targets and must climb a
    MoonBoard wall by gripping holds in sequence.  The episode terminates when
    the humanoid falls (pelvis z < 0.2 m) or successfully grips the finish hold
    with both hands for 10 consecutive steps.

    Args:
        route: Route object from any MoonBoard parser.
        humanoid_xml_path: Absolute path to humanoid.xml on disk.
        sim_substeps: Physics steps per policy step.  Policy period =
            model.opt.timestep × sim_substeps; should be in [0.02, 0.04] s.
        max_episode_steps: Steps before truncation.
        render_mode: Unused; present for Gymnasium API compatibility.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        route: Route,
        humanoid_xml_path: str,
        sim_substeps: int = 10,
        max_episode_steps: int = 2000,
        render_mode: str | None = None,
    ) -> None:
        """Construct the environment and validate the MuJoCo model.

        Performs the Step 1 model inspection during construction: prints nq,
        nv, actuator names, and the computed policy period.
        """
        super().__init__()
        import mujoco as _mj
        self._mj = _mj

        self._route = route
        self._sim_substeps = sim_substeps
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # ── Build and load the scene XML ──────────────────────────────────────
        xml_str = build_scene_xml(route, humanoid_xml_path)
        self._model = _mj.MjModel.from_xml_string(xml_str)
        self._data = _mj.MjData(self._model)

        # ── Policy period validation (Step 1) ─────────────────────────────────
        policy_period = float(self._model.opt.timestep) * sim_substeps
        if not (0.02 <= policy_period <= 0.04):
            print(
                f"[MoonBoardEnv] WARNING: policy_period={policy_period:.4f} s is outside "
                f"[0.02, 0.04] s — expect unstable humanoid dynamics."
            )
        print(
            f"[MoonBoardEnv] Policy period = {policy_period:.6f} s "
            f"(timestep={self._model.opt.timestep} × substeps={sim_substeps})"
        )

        # ── Torso / pelvis body ───────────────────────────────────────────────
        self._torso_id = _mj.mj_name2id(self._model, _mj.mjtObj.mjOBJ_BODY, "torso")
        if self._torso_id < 0:
            raise ValueError("Body 'torso' not found in humanoid XML.")

        # ── Limb sites (hand / foot tips) ─────────────────────────────────────
        self._site_ids: list[int] = []
        for sname in LIMB_SITE_NAMES:
            sid = _mj.mj_name2id(self._model, _mj.mjtObj.mjOBJ_SITE, sname)
            if sid < 0:
                raise ValueError(
                    f"Site '{sname}' not found in model.  "
                    "Ensure build_scene_xml called _inject_limb_sites."
                )
            self._site_ids.append(sid)

        # ── Hold lookup dicts (for GripManager) ──────────────────────────────
        self._hold_positions: dict[str, np.ndarray] = {}
        self._hold_body_ids: dict[str, int] = {}
        for h in route.holds:
            bname = hold_body_name(h.col, h.row)
            bid = _mj.mj_name2id(self._model, _mj.mjtObj.mjOBJ_BODY, bname)
            if bid < 0:
                print(f"[MoonBoardEnv] WARNING: hold body '{bname}' not found — skipping.")
                continue
            wx, wy, wz = hold_position_world(h.col, h.row)
            self._hold_positions[bname] = np.array(
                [wx, wy + RADIUS * _NY, wz + RADIUS * _NZ], dtype=np.float64
            )
            self._hold_body_ids[bname] = bid

        # ── Role-sorted hold lists ────────────────────────────────────────────
        self._start_holds = [h for h in route.holds if h.role == "start"]
        self._mid_holds = [h for h in route.holds if h.role == "mid"]
        self._end_holds = [h for h in route.holds if h.role == "end"]

        # Indices into route.holds for each role group.
        self._start_holds_ri: list[int] = [route.holds.index(h) for h in self._start_holds]
        self._mid_holds_ri: list[int] = [route.holds.index(h) for h in self._mid_holds]
        self._end_holds_ri: list[int] = [route.holds.index(h) for h in self._end_holds]

        # Name → Hold lookup (for exteroception stream).
        self._name_to_hold = {
            hold_body_name(h.col, h.row): h for h in route.holds
        }

        # ── GripManager ───────────────────────────────────────────────────────
        self._grip_manager = GripManager(
            self._model, self._data,
            self._hold_positions, self._hold_body_ids,
        )

        # ── Action space ──────────────────────────────────────────────────────
        nu = int(self._model.nu)
        self._nu = nu
        _FALLBACK_BOUND = 3.14

        act_lo = np.empty(nu, dtype=np.float32)
        act_hi = np.empty(nu, dtype=np.float32)
        for i in range(nu):
            lo = float(self._model.actuator_ctrlrange[i, 0])
            hi = float(self._model.actuator_ctrlrange[i, 1])
            if not np.isfinite(lo):
                aname = _mj.mj_id2name(self._model, _mj.mjtObj.mjOBJ_ACTUATOR, i)
                print(
                    f"[MoonBoardEnv] WARNING: actuator '{aname}' has infinite lower "
                    f"bound; clamping to -{_FALLBACK_BOUND}"
                )
                lo = -_FALLBACK_BOUND
            if not np.isfinite(hi):
                aname = _mj.mj_id2name(self._model, _mj.mjtObj.mjOBJ_ACTUATOR, i)
                print(
                    f"[MoonBoardEnv] WARNING: actuator '{aname}' has infinite upper "
                    f"bound; clamping to {_FALLBACK_BOUND}"
                )
                hi = _FALLBACK_BOUND
            act_lo[i] = lo
            act_hi[i] = hi

        self._act_lo = act_lo
        self._act_hi = act_hi

        # Grip intent: 4 continuous signals in [-1, 1]; > 0 means attempt grip.
        grip_lo = np.full(4, -1.0, dtype=np.float32)
        grip_hi = np.full(4, 1.0, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.concatenate([act_lo, grip_lo]),
            high=np.concatenate([act_hi, grip_hi]),
            dtype=np.float32,
        )

        # ── Observation space ─────────────────────────────────────────────────
        nj_pos = int(self._model.nq) - 7   # skip freejoint 7 DOF
        nj_vel = int(self._model.nv) - 6   # skip freejoint 6 DOF
        self._nj_pos = nj_pos
        self._nj_vel = nj_vel

        # [0 : nj_pos]                   joint positions        (17)
        # [nj_pos : nj_pos+nj_vel]       joint velocities       (17)
        # [34 : 37]                       pelvis world pos       ( 3)
        # [37 : 43]                       pelvis rot6d           ( 6)
        # [43 : 46]                       pelvis linear vel      ( 3)
        # [46 : 49]                       pelvis angular vel     ( 3)
        # [49 : 61]                       4 limb sites (pelvis)  (12)
        # [61 : 65]                       grip state             ( 4)
        self._stream1_dim = nj_pos + nj_vel + 3 + 6 + 3 + 3 + 12 + 4  # 65

        # [65 : 121]  8 nearest holds × 7 values                        (56)
        self._stream2_dim = _NUM_NEAR_HOLDS * _HOLD_OBS_DIM             # 56

        # [121 : 133]  4 limb goal vectors × 3 values                   (12)
        self._stream3_dim = 4 * 3                                        # 12

        self._obs_dim = self._stream1_dim + self._stream2_dim + self._stream3_dim  # 133

        # Wide bounds avoid check_env complaints from large velocity/force values.
        self.observation_space = spaces.Box(
            low=-1e6, high=1e6,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # ── Episode state (values set in reset) ───────────────────────────────
        self._prev_pelvis_z: float = 0.0
        self._step_count: int = 0
        self._consec_finish_count: int = 0
        # Route-holds index of the current target hold for each limb slot.
        self._target_hold_indices: list[int] = [0, 0, 0, 0]
        # True if slot i was gripping its target hold on the previous step.
        # Used for rising-edge detection in the hold-match reward.
        self._prev_grip_target_state: list[bool] = [False, False, False, False]
        # Per-hand pointer into self._mid_holds_ri (which mid hold each hand targets).
        self._hand_mid_ptr: list[int] = [0, 0]
        # Ordered history of mid-hold pointers for foot lag computation.
        self._hand_mid_history: list[list[int]] = [[], []]

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to an initial climbing state.

        Places the humanoid near the start hold, engages both hand grips on
        the start hold, and resets all episode counters and target pointers.

        Alignment threshold is temporarily set to -1.0 and proximity to 0.20
        so the default A-pose can engage grips without requiring RSI.
        TODO (Week 2): replace hardcoded pose with Reference State Initialization.

        Args:
            seed: RNG seed forwarded to super().reset() for reproducibility.
            options: Ignored; present for Gymnasium API compliance.

        Returns:
            Tuple of (observation, info_dict).

        Raises:
            RuntimeError: If both hand slots fail to engage on the start hold,
                with diagnostic printout of site and hold world positions.
        """
        super().reset(seed=seed)

        # Release any grips from the previous episode so GripManager._active_holds
        # is clean (mj_resetData handles data.eq_active, but not the Python dict).
        for slot in range(4):
            self._grip_manager.release_grip(slot)

        # ── Physics reset ─────────────────────────────────────────────────────
        self._mj.mj_resetData(self._model, self._data)

        # Hardcoded starting pose: torso near wall with arms in default A-pose.
        # Grid search found this places lhand at 0.167 m and rhand at 0.178 m
        # from hold_5_5 (the start hold sphere centre).
        # TODO (Week 2): replace with Reference State Initialization (RSI).
        q = np.zeros(self._model.nq, dtype=np.float64)
        q[0] = _RESET_TORSO_X
        q[1] = _RESET_TORSO_Y
        q[2] = _RESET_TORSO_Z
        q[3], q[4], q[5], q[6] = _RESET_QUAT  # 180° around Z
        self._data.qpos[:] = q
        self._mj.mj_forward(self._model, self._data)

        # ── Engage start grips ────────────────────────────────────────────────
        # Both hand slots grip the first (and for this route, only) start hold.
        # Thresholds are relaxed because the A-pose arm Z-axis does not face the
        # wall and hand-to-hold distance (~0.17 m) exceeds the default 0.12 m
        # proximity threshold.  Restored immediately after.
        orig_prox = _gm_mod.PROXIMITY_THRESHOLD
        orig_align = _gm_mod.ALIGNMENT_THRESHOLD
        _gm_mod.PROXIMITY_THRESHOLD = _RESET_PROXIMITY
        _gm_mod.ALIGNMENT_THRESHOLD = _RESET_ALIGNMENT
        try:
            start_names = [hold_body_name(h.col, h.row) for h in self._start_holds]
            for hand_slot in (0, 1):
                target_name = start_names[min(hand_slot, len(start_names) - 1)]
                self._grip_manager.try_grip(hand_slot, target_name)
        finally:
            _gm_mod.PROXIMITY_THRESHOLD = orig_prox
            _gm_mod.ALIGNMENT_THRESHOLD = orig_align

        # ── Verify both hands engaged ─────────────────────────────────────────
        active = self._grip_manager.get_active_hold_ids()
        if 0 not in active or 1 not in active:
            lhand_pos = np.array(self._data.site_xpos[self._site_ids[0]])
            rhand_pos = np.array(self._data.site_xpos[self._site_ids[1]])
            start_worlds = [
                self._hold_positions[n]
                for n in (hold_body_name(h.col, h.row) for h in self._start_holds)
                if n in self._hold_positions
            ]
            raise RuntimeError(
                f"[MoonBoardEnv] reset(): failed to grip both start holds.\n"
                f"  Active grips after attempt: {active}\n"
                f"  lhand site world pos: {lhand_pos}\n"
                f"  rhand site world pos: {rhand_pos}\n"
                f"  start hold world positions: {start_worlds}\n"
            )

        self._mj.mj_forward(self._model, self._data)

        # ── Reset episode counters ────────────────────────────────────────────
        self._step_count = 0
        self._consec_finish_count = 0

        # ── Target hold sequencing init ───────────────────────────────────────
        # TODO (Week 2): replace with beta planner.
        self._hand_mid_ptr = [0, 0]
        self._hand_mid_history = [[], []]
        first_mid_ri = self._mid_holds_ri[0] if self._mid_holds_ri else 0
        first_start_ri = self._start_holds_ri[0] if self._start_holds_ri else 0
        self._target_hold_indices = [
            first_mid_ri,    # slot 0 (lhand) → first mid hold
            first_mid_ri,    # slot 1 (rhand) → first mid hold (same)
            first_start_ri,  # slot 2 (lfoot) → first start hold
            first_start_ri,  # slot 3 (rfoot) → first start hold (same)
        ]
        self._prev_grip_target_state = [False, False, False, False]

        # Initialise pelvis z for height-progress reward.
        pelvis_pos = np.array(self._data.xpos[self._torso_id])
        self._prev_pelvis_z = float(pelvis_pos[2])

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by one policy step.

        Applies joint targets and grip intents for sim_substeps physics steps,
        then checks slips, advances target holds on rising-edge grips, computes
        reward, and checks termination / truncation.

        Args:
            action: Flat float32 array of shape (nu + 4,).
                First nu values are joint position targets.
                Last 4 are grip intent signals (> 0.0 → attempt grip).

        Returns:
            (obs, reward, terminated, truncated, info) where info contains
            every reward component as an individual float.
        """
        joint_targets, grip_intents = self._parse_action(action)

        # Apply joint targets and advance physics.
        self._data.ctrl[:] = joint_targets
        for _ in range(self._sim_substeps):
            self._mj.mj_step(self._model, self._data)

        # ── Process grip intents ──────────────────────────────────────────────
        for slot in range(4):
            if grip_intents[slot] > 0.0:
                th = self._route.holds[self._target_hold_indices[slot]]
                self._grip_manager.try_grip(slot, hold_body_name(th.col, th.row))
            else:
                self._grip_manager.release_grip(slot)

        # ── Slip detection ────────────────────────────────────────────────────
        self._grip_manager.check_slip()

        # ── Compute per-slot "gripping its target" state ──────────────────────
        active = self._grip_manager.get_active_hold_ids()
        curr_grip_target: list[bool] = [False, False, False, False]
        for slot in range(4):
            th = self._route.holds[self._target_hold_indices[slot]]
            curr_grip_target[slot] = (
                active.get(slot) == hold_body_name(th.col, th.row)
            )

        # ── Advance target holds on rising edge ───────────────────────────────
        for slot in range(4):
            rising = curr_grip_target[slot] and not self._prev_grip_target_state[slot]
            if rising:
                if slot in (0, 1):
                    self._advance_hand_target(slot)
                else:
                    self._advance_foot_target(slot)

        # ── Reward ────────────────────────────────────────────────────────────
        reward, info = self._compute_reward(curr_grip_target)

        # Update grip-target history after computing reward (rising-edge uses prev).
        self._prev_grip_target_state = list(curr_grip_target)

        # ── Termination ───────────────────────────────────────────────────────
        pelvis_pos = np.array(self._data.xpos[self._torso_id])
        fell = bool(pelvis_pos[2] < 0.2)

        both_on_finish = False
        if self._end_holds_ri:
            finish_name = hold_body_name(
                self._route.holds[self._end_holds_ri[0]].col,
                self._route.holds[self._end_holds_ri[0]].row,
            )
            lh_finish = (active.get(0) == finish_name)
            rh_finish = (active.get(1) == finish_name)
            if lh_finish and rh_finish:
                self._consec_finish_count += 1
            else:
                self._consec_finish_count = 0
            both_on_finish = (self._consec_finish_count >= 10)

        terminated = fell or both_on_finish

        if fell:
            reward += -10.0
            info["fall_penalty"] = -10.0
        else:
            info["fall_penalty"] = 0.0

        if both_on_finish:
            reward += 50.0
            info["finish_bonus"] = 50.0
        else:
            info["finish_bonus"] = 0.0

        # ── Truncation ────────────────────────────────────────────────────────
        self._step_count += 1
        truncated = (self._step_count >= self._max_episode_steps)

        obs = self._get_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ─────────────────────────────────────────────────────────────────────────
    # Observation assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build and return the flat observation vector.

        Index map (nj_pos=17, nj_vel=17):
        Stream 1 — Proprioception [0 : 65]:
          [0  : 17]   joint positions (qpos[7:])            (17)
          [17 : 34]   joint velocities (qvel[6:])           (17)
          [34 : 37]   pelvis world position                  ( 3)
          [37 : 43]   pelvis orientation 6D (rot mat cols)  ( 6)
          [43 : 46]   pelvis linear velocity (world frame)  ( 3)
          [46 : 49]   pelvis angular velocity (world frame) ( 3)
          [49 : 61]   4 limb sites in pelvis frame (4×3)    (12)
          [61 : 65]   grip state (4 binary floats)          ( 4)
        Stream 2 — Exteroception [65 : 121]:
          8 nearest holds × (3 rel_pos + 3 role_onehot + 1 gripping) (56)
        Stream 3 — Goal [121 : 133]:
          4 × (target_world − site_world)                           (12)

        Returns:
            Float32 array of shape (133,).

        Raises:
            RuntimeError: If any NaN or Inf is detected, identifying the stream.
        """
        data = self._data

        # ── Stream 1: Proprioception ──────────────────────────────────────────
        joint_pos = np.array(data.qpos[7:], dtype=np.float32)          # 17

        joint_vel = np.array(data.qvel[6:], dtype=np.float32)          # 17

        pelvis_world = np.array(data.xpos[self._torso_id], dtype=np.float32)  # 3

        pelvis_mat = np.array(data.xmat[self._torso_id]).reshape(3, 3)
        rot6d = pelvis_mat[:, :2].flatten().astype(np.float32)          # 6

        linvel = np.array(data.qvel[0:3], dtype=np.float32)            # 3
        angvel = np.array(data.qvel[3:6], dtype=np.float32)            # 3

        site_in_pelvis = np.empty(12, dtype=np.float32)                 # 12
        for k, sid in enumerate(self._site_ids):
            site_world = np.array(data.site_xpos[sid])
            rel = site_world - pelvis_world.astype(np.float64)
            site_in_pelvis[k * 3: k * 3 + 3] = (pelvis_mat.T @ rel).astype(np.float32)

        grip_state = self._grip_manager.get_grip_state()                # 4

        stream1 = np.concatenate([
            joint_pos, joint_vel, pelvis_world, rot6d,
            linvel, angvel, site_in_pelvis, grip_state,
        ])  # 65

        # ── Stream 2: Exteroception ───────────────────────────────────────────
        active = self._grip_manager.get_active_hold_ids()
        active_hold_names: set[str] = set(active.values())

        # Sort all holds by distance from pelvis; take the 8 nearest.
        pelvis_64 = pelvis_world.astype(np.float64)
        sorted_holds = sorted(
            self._hold_positions.items(),
            key=lambda kv: float(np.linalg.norm(kv[1] - pelvis_64)),
        )
        nearest = sorted_holds[:_NUM_NEAR_HOLDS]

        _role_map = {"start": 0, "mid": 1, "end": 2}
        stream2 = np.zeros(_NUM_NEAR_HOLDS * _HOLD_OBS_DIM, dtype=np.float32)
        for i, (hname, hworld) in enumerate(nearest):
            rel = (hworld - pelvis_64).astype(np.float32)
            hold = self._name_to_hold.get(hname)
            if hold is not None:
                one_hot = np.zeros(3, dtype=np.float32)
                one_hot[_role_map.get(hold.role, 1)] = 1.0
            else:
                one_hot = np.zeros(3, dtype=np.float32)
            gripping = np.float32(hname in active_hold_names)
            base = i * _HOLD_OBS_DIM
            stream2[base: base + 3] = rel
            stream2[base + 3: base + 6] = one_hot
            stream2[base + 6] = gripping  # 56 total

        # ── Stream 3: Goal vectors ────────────────────────────────────────────
        stream3 = np.zeros(12, dtype=np.float32)
        for slot in range(4):
            ri = self._target_hold_indices[slot]
            if 0 <= ri < len(self._route.holds):
                th = self._route.holds[ri]
                tname = hold_body_name(th.col, th.row)
                if tname in self._hold_positions:
                    target_world = self._hold_positions[tname]
                    site_world = np.array(data.site_xpos[self._site_ids[slot]])
                    stream3[slot * 3: slot * 3 + 3] = (
                        (target_world - site_world).astype(np.float32)
                    )
            # If no valid target, leave as zero vector.
            # TODO (Week 3): replace zero-vector with a sentinel (e.g. large constant)
            # to avoid ambiguity with "limb is exactly at target."

        obs = np.concatenate([stream1, stream2, stream3]).astype(np.float32)

        # ── NaN / Inf guard ───────────────────────────────────────────────────
        if not np.all(np.isfinite(obs)):
            s1_end = self._stream1_dim
            s2_end = s1_end + self._stream2_dim
            for start, end, name in [
                (0, s1_end, "stream1/proprioception"),
                (s1_end, s2_end, "stream2/exteroception"),
                (s2_end, self._obs_dim, "stream3/goal"),
            ]:
                chunk = obs[start:end]
                if not np.all(np.isfinite(chunk)):
                    bad = (np.where(~np.isfinite(chunk))[0] + start).tolist()
                    raise RuntimeError(
                        f"[MoonBoardEnv] NaN/Inf in {name} [{start}:{end}]. "
                        f"Bad global indices: {bad}"
                    )

        return obs

    # ─────────────────────────────────────────────────────────────────────────
    # Reward function v0
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self, curr_grip_target: list[bool]
    ) -> tuple[float, dict[str, float]]:
        """Compute base reward components (fall/finish bonuses added by step()).

        Args:
            curr_grip_target: Per-slot bool — True if slot is gripping its
                current target hold this step.

        Returns:
            Tuple of (total_base_reward, info_dict).  info_dict keys:
                height_progress, hold_match_bonus, alive_bonus.
        """
        info: dict[str, float] = {}

        # ── Height progress ───────────────────────────────────────────────────
        pelvis_z = float(np.array(self._data.xpos[self._torso_id])[2])
        delta_z = pelvis_z - self._prev_pelvis_z
        height_progress = float(np.clip(delta_z * 2.0, -0.1, 0.1))
        self._prev_pelvis_z = pelvis_z
        info["height_progress"] = height_progress

        # ── Hold match bonus (rising edge only) ───────────────────────────────
        hold_match_bonus = 0.0
        for slot in range(4):
            if curr_grip_target[slot] and not self._prev_grip_target_state[slot]:
                hold_match_bonus += 5.0
        info["hold_match_bonus"] = hold_match_bonus

        # ── Alive bonus ───────────────────────────────────────────────────────
        info["alive_bonus"] = 0.1

        total = height_progress + hold_match_bonus + 0.1
        return total, info

    # ─────────────────────────────────────────────────────────────────────────
    # Target hold sequencing
    # ─────────────────────────────────────────────────────────────────────────

    def _advance_hand_target(self, hand_slot: int) -> None:
        """Advance a hand slot's target to the next available mid hold.

        When hand i grips its target:
          - Record the current mid-hold pointer in history (for foot lag).
          - Find the next mid hold not currently targeted by the other hand.
          - If no mid holds remain: target the finish hold.

        TODO (Week 2): replace with beta planner.

        Args:
            hand_slot: 0 (left hand) or 1 (right hand).
        """
        ptr = self._hand_mid_ptr[hand_slot]
        self._hand_mid_history[hand_slot].append(ptr)
        other = 1 - hand_slot

        next_ptr = ptr + 1
        while (
            next_ptr < len(self._mid_holds_ri)
            and self._hand_mid_ptr[other] == next_ptr
        ):
            next_ptr += 1

        if next_ptr < len(self._mid_holds_ri):
            self._hand_mid_ptr[hand_slot] = next_ptr
            self._target_hold_indices[hand_slot] = self._mid_holds_ri[next_ptr]
        else:
            # All mid holds exhausted; target the finish hold.
            if self._end_holds_ri:
                self._target_hold_indices[hand_slot] = self._end_holds_ri[0]
            self._hand_mid_ptr[hand_slot] = len(self._mid_holds_ri)  # sentinel

    def _advance_foot_target(self, foot_slot: int) -> None:
        """Advance a foot slot's target using a two-step lag behind its hand.

        Foot j targets the mid hold that hand (j-2) was targeting exactly two
        advances ago.  Fallback: lowest-index available mid hold not already
        targeted by the other foot.

        TODO (Week 2): replace with beta planner.

        Args:
            foot_slot: 2 (left foot) or 3 (right foot).
        """
        hand_slot = foot_slot - 2    # 2→0, 3→1
        other_foot = 5 - foot_slot   # 2→3, 3→2
        history = self._hand_mid_history[hand_slot]

        new_ri: int | None = None
        two_ago_idx = len(history) - 2
        if two_ago_idx >= 0:
            mid_ptr = history[two_ago_idx]
            if mid_ptr < len(self._mid_holds_ri):
                new_ri = self._mid_holds_ri[mid_ptr]

        if new_ri is None:
            other_target = self._target_hold_indices[other_foot]
            for ri in self._mid_holds_ri:
                if ri != other_target:
                    new_ri = ri
                    break
            if new_ri is None and self._mid_holds_ri:
                new_ri = self._mid_holds_ri[0]

        if new_ri is not None:
            self._target_hold_indices[foot_slot] = new_ri

    # ─────────────────────────────────────────────────────────────────────────
    # Action parsing
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_action(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Slice a flat action into joint targets and grip intents.

        Args:
            action: Float32 array of shape (nu + 4,).

        Returns:
            Tuple of:
                joint_targets: Float32 array of shape (nu,), clamped to ctrlrange.
                grip_intents: Float32 array of shape (4,), clamped to [-1, 1].
        """
        action = np.asarray(action, dtype=np.float32)
        joint_targets = np.clip(action[: self._nu], self._act_lo, self._act_hi)
        grip_intents = np.clip(action[self._nu: self._nu + 4], -1.0, 1.0)
        return joint_targets, grip_intents

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium rendering / lifecycle stubs
    # ─────────────────────────────────────────────────────────────────────────

    def render(self) -> None:
        """Render a frame (no-op in headless mode; use interactive_grip.py)."""
        pass

    def close(self) -> None:
        """Release any held resources."""
        pass
