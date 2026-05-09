"""Gymnasium environment for MoonBoard climbing RL.

Wraps a MuJoCo scene (wall + humanoid + grip constraints) as a standard
Gymnasium Env compatible with all standard RL libraries including SB3.

Observation  : flat Box of shape (139,), dtype float32
Action       : flat Box of shape (21,), dtype float32
                 first 17 — joint position targets (ctrlrange bounded)
                 last  4  — continuous grip intent signals in [-1, 1]

Observation layout (139 total):
  Stream 1 — Proprioception [0 : 65]:
    [0  : 17]  joint positions          (17)
    [17 : 34]  joint velocities         (17)
    [34 : 37]  pelvis world pos         ( 3)
    [37 : 43]  pelvis rot6d             ( 6)
    [43 : 46]  pelvis linear vel        ( 3)
    [46 : 49]  pelvis angular vel       ( 3)
    [49 : 61]  4 limb sites in pelvis   (12)
    [61 : 65]  grip state               ( 4)
  Stream 2 — Exteroception [65 : 121]:
    8 nearest holds × (3 rel_pos + 3 role_onehot + 1 gripping) (56)
  Stream 3 — Goal [121 : 133]:
    4 × (target_world − site_world)                            (12)
  Stream 4 — Foot proximity [133 : 139]:
    2 feet × (kickboard_hold_pos − foot_site_pos) in pelvis    ( 6)
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
from src.xml_gen.wall import (
    hold_body_name, hold_position_world,
    kickboard_hold_positions_world, KICKBOARD_HOLD_NAMES,
)
from src.xml_gen.holds import RADIUS, _NY, _NZ
import src.grip.grip_manager as _gm_mod
from src.grip.grip_manager import GripManager

# ── Observation stream dimensions ─────────────────────────────────────────────
_NUM_NEAR_HOLDS: int = 8
_HOLD_OBS_DIM: int = 7   # 3 rel_pos + 3 role_onehot + 1 gripping_flag

# Reset pose constants — place humanoid with both hands close to the start hold.
_RESET_TORSO_X: float = 0.35
_RESET_TORSO_Y: float = -0.95
_RESET_TORSO_Z: float = 0.80
_RESET_QUAT: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

# Relaxed thresholds used only during reset to engage grips from A-pose.
_RESET_PROXIMITY: float = 0.20
_RESET_ALIGNMENT: float = -1.0
# Foot snap distance: set to 1.0 m so kickboard holds at x=±0.61 m engage even
# when the IK only partially converges.  The connect constraint closes the gap
# during warmup; check_slip() is not called during warmup steps.
_RESET_PROXIMITY_FEET: float = 1.0

# qpos indices for hip/knee joints (freejoint occupies qpos[0:7]).
#   [7]=abdomen_z  [8]=abdomen_y  [9]=abdomen_x
#   [10]=right_hip_x  [11]=right_hip_z  [12]=right_hip_y  [13]=right_knee
#   [14]=left_hip_x   [15]=left_hip_z   [16]=left_hip_y   [17]=left_knee
_QI_RIGHT_HIP_Z: int = 11   # internal/external rotation, range −60° to +35°
_QI_RIGHT_HIP_Y: int = 12   # flexion/extension, range −110° to +20°
_QI_RIGHT_KNEE:  int = 13   # flexion, range −160° to −2°
_QI_LEFT_HIP_Z:  int = 15
_QI_LEFT_HIP_Y:  int = 16
_QI_LEFT_KNEE:   int = 17

# Initial hip/knee angles (starting guess before IK refinement).
# −60° hip flex (forward) brings the thigh toward the wall face.
# hip_z=0 is neutral; IK will rotate hip_z to achieve the lateral spread.
_INIT_HIP_Z: float = np.radians(0.0)
_INIT_HIP_Y: float = np.radians(-60.0)
_INIT_KNEE:  float = np.radians(-60.0)

# Warmup steps after engaging grips (50 × 2 ms = 100 ms ≈ 5 × time constant).
_RESET_WARMUP_STEPS: int = 50

# Fall detection thresholds.
_FALL_Z_THRESHOLD: float = 0.30
_FALL_Y_THRESHOLD: float = 1.0

# Reward coefficients.
_HWM_HEIGHT_SCALE: float = 5.0
_ENERGY_PENALTY_COEFF: float = 0.01


class MoonBoardEnv(gym.Env):
    """Gymnasium environment for MoonBoard climbing simulation.

    A MuJoCo humanoid is controlled by joint position targets and must climb a
    MoonBoard wall by gripping holds in sequence.  The episode terminates when
    the humanoid falls (pelvis z < 0.30 m) or grips the finish hold with both
    hands for 10 consecutive steps.

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
        sim_substeps: int = 7,
        max_episode_steps: int = 2000,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        import mujoco as _mj
        self._mj = _mj

        self._route = route
        self._sim_substeps = sim_substeps
        self._max_episode_steps = max_episode_steps
        self.render_mode = render_mode

        # ── Build and load scene XML ───────────────────────────────────────────
        xml_str = build_scene_xml(route, humanoid_xml_path)
        self._model = _mj.MjModel.from_xml_string(xml_str)
        self._data = _mj.MjData(self._model)

        # ── Policy period validation ───────────────────────────────────────────
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

        # ── Hold lookup dicts (for GripManager + goal vector) ────────────────
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

        # Add kickboard holds so they are accessible for goal-vector computation.
        _kb_pos = kickboard_hold_positions_world()
        for kb_name in KICKBOARD_HOLD_NAMES:
            bid = _mj.mj_name2id(self._model, _mj.mjtObj.mjOBJ_BODY, kb_name)
            if bid >= 0:
                self._hold_positions[kb_name] = _kb_pos[kb_name]
                self._hold_body_ids[kb_name] = bid

        # Cache kickboard world positions for fast lookup during obs/IK.
        self._kb_positions: dict[str, np.ndarray] = kickboard_hold_positions_world()

        # ── Role-sorted hold lists ────────────────────────────────────────────
        self._start_holds = [h for h in route.holds if h.role == "start"]
        self._mid_holds = [h for h in route.holds if h.role == "mid"]
        self._end_holds = [h for h in route.holds if h.role == "end"]

        self._start_holds_ri: list[int] = [route.holds.index(h) for h in self._start_holds]
        self._mid_holds_ri: list[int] = [route.holds.index(h) for h in self._mid_holds]
        self._end_holds_ri: list[int] = [route.holds.index(h) for h in self._end_holds]

        self._name_to_hold = {
            hold_body_name(h.col, h.row): h for h in route.holds
        }

        # ── GripManager ───────────────────────────────────────────────────────
        # GripManager augments hold_positions / hold_body_ids with kickboard holds
        # in its __init__, so no extra wiring needed here.
        self._grip_manager = GripManager(
            self._model, self._data,
            self._hold_positions, self._hold_body_ids,
            verbose=False,
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

        grip_lo = np.full(4, -1.0, dtype=np.float32)
        grip_hi = np.full(4,  1.0, dtype=np.float32)

        self.action_space = spaces.Box(
            low=np.concatenate([act_lo, grip_lo]),
            high=np.concatenate([act_hi, grip_hi]),
            dtype=np.float32,
        )

        # ── Observation space ─────────────────────────────────────────────────
        nj_pos = int(self._model.nq) - 7
        nj_vel = int(self._model.nv) - 6
        self._nj_pos = nj_pos
        self._nj_vel = nj_vel

        self._stream1_dim = nj_pos + nj_vel + 3 + 6 + 3 + 3 + 12 + 4   # 65
        self._stream2_dim = _NUM_NEAR_HOLDS * _HOLD_OBS_DIM             # 56
        self._stream3_dim = 4 * 3                                        # 12
        self._stream4_dim = 2 * 3                                        # 6 — foot→kickboard

        self._obs_dim = (
            self._stream1_dim + self._stream2_dim +
            self._stream3_dim + self._stream4_dim
        )  # 139

        self.observation_space = spaces.Box(
            low=-1e6, high=1e6,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

        # ── Foot hold selection (recomputed in reset) ─────────────────────────
        # _select_foot_holds() always returns the two upper kickboard holds.
        self._foot_hold_names: tuple[str, str] = self._select_foot_holds()

        # Per-foot current target hold names (may advance during episode).
        # Index 0 = left foot (slot 2), index 1 = right foot (slot 3).
        self._foot_target_hold_names: list[str] = list(self._foot_hold_names)

        # ── Episode state ─────────────────────────────────────────────────────
        self._reset_pelvis_z: float = 0.0
        self._max_pelvis_z: float = 0.0
        self._step_count: int = 0
        self._consec_finish_count: int = 0
        self._target_hold_indices: list[int] = [0, 0, 0, 0]
        self._prev_grip_target_state: list[bool] = [False, False, False, False]
        self._hand_mid_ptr: list[int] = [0, 0]
        self._hand_mid_history: list[list[int]] = [[], []]
        self._holds_matched_this_episode: set[str] = set()

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium interface
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Reset the environment to a 4-point kickboard start.

        Flow:
          1. Release any previous grips.
          2. Set torso position and initial hip/knee angles.
          3. Phase 1: engage hand grips on start hold (relaxed thresholds).
          4. Warmup 50 steps (hands only).
          5. DLS IK for feet: 30-iteration damped-least-squares on hip_z+hip_y+knee,
             run in the settled post-Phase-1 body configuration.
          6. Phase 2: engage foot grips on kickboard upper holds.
          7. Warmup 50 steps (4-point contact).
          8. Zero qvel, settle 20 steps, mj_forward.
          9. Assert ≥ 3 grips active (RuntimeError if not).

        Args:
            seed: RNG seed forwarded to super().reset() for reproducibility.
            options: Ignored; present for Gymnasium API compliance.

        Returns:
            Tuple of (observation, info_dict).

        Raises:
            RuntimeError: If fewer than 3 grips are active after reset.
        """
        super().reset(seed=seed)

        # Release any grips from the previous episode.
        for slot in range(4):
            self._grip_manager.release_grip(slot)

        # ── Physics reset ─────────────────────────────────────────────────────
        self._mj.mj_resetData(self._model, self._data)

        # Initial pose: torso in front of wall, hip/knee pre-flexed as IK start.
        q = np.zeros(self._model.nq, dtype=np.float64)
        q[0] = _RESET_TORSO_X
        q[1] = _RESET_TORSO_Y
        q[2] = _RESET_TORSO_Z
        q[3], q[4], q[5], q[6] = _RESET_QUAT
        q[_QI_LEFT_HIP_Z]  = _INIT_HIP_Z
        q[_QI_LEFT_HIP_Y]  = _INIT_HIP_Y
        q[_QI_LEFT_KNEE]   = _INIT_KNEE
        q[_QI_RIGHT_HIP_Z] = _INIT_HIP_Z
        q[_QI_RIGHT_HIP_Y] = _INIT_HIP_Y
        q[_QI_RIGHT_KNEE]  = _INIT_KNEE
        self._data.qpos[:] = q

        self._mj.mj_forward(self._model, self._data)

        # ── Phase 1: engage hand grips, warmup ───────────────────────────────
        orig_prox  = _gm_mod.PROXIMITY_THRESHOLD
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

        active = self._grip_manager.get_active_hold_ids()
        if 0 not in active or 1 not in active:
            lhand_pos = np.array(self._data.site_xpos[self._site_ids[0]])
            rhand_pos = np.array(self._data.site_xpos[self._site_ids[1]])
            raise RuntimeError(
                f"[MoonBoardEnv] reset(): failed to grip both start holds.\n"
                f"  Active grips after attempt: {active}\n"
                f"  lhand site world pos: {lhand_pos}\n"
                f"  rhand site world pos: {rhand_pos}\n"
                f"  start hold world positions: "
                f"{[self._hold_positions[n] for n in start_names if n in self._hold_positions]}\n"
            )

        self._data.ctrl[:] = 0.0
        for _ in range(_RESET_WARMUP_STEPS):
            self._mj.mj_step(self._model, self._data)

        # ── IK after Phase 1: bring feet near kickboard holds ─────────────────
        # Now that the body has settled into its hanging configuration (hands
        # gripped, gravity balanced), the hip–foot kinematics are accurate.
        # Run IK in this settled state to minimise the snap distance at Phase 2.
        self._ik_feet_to_kickboard()
        self._mj.mj_forward(self._model, self._data)

        # ── Phase 2: engage kickboard foot grips, warmup ──────────────────────
        # Always use kickboard upper holds — they are at z ≈ 0.270 m.
        # PROXIMITY_THRESHOLD is relaxed to 1.0 m; check_slip() is not called
        # during warmup so large transient snap forces are tolerated.
        lfoot_hold, rfoot_hold = self._foot_hold_names   # "kb_hold_KB_LU", "kb_hold_KB_RU"
        _gm_mod.PROXIMITY_THRESHOLD = _RESET_PROXIMITY_FEET
        _gm_mod.ALIGNMENT_THRESHOLD = _RESET_ALIGNMENT
        try:
            # snap_to_center=True: the constraint spring pulls the foot TO the
            # hold centre during the 50-step warmup, regardless of how far the
            # foot started.  This is reset-only behaviour; episode grips use
            # snap_to_center=False to preserve the foot's contact position.
            self._grip_manager.try_grip(2, lfoot_hold, snap_to_center=True)
            self._grip_manager.try_grip(3, rfoot_hold, snap_to_center=True)
        finally:
            _gm_mod.PROXIMITY_THRESHOLD = orig_prox
            _gm_mod.ALIGNMENT_THRESHOLD = orig_align

        for _ in range(_RESET_WARMUP_STEPS):
            self._mj.mj_step(self._model, self._data)

        # Zero residual velocities; re-settle to true static equilibrium.
        self._data.qvel[:] = 0.0
        for _ in range(20):
            self._mj.mj_step(self._model, self._data)
        self._mj.mj_forward(self._model, self._data)

        # ── Verify grip count ─────────────────────────────────────────────────
        active = self._grip_manager.get_active_hold_ids()
        n_active = len(active)
        if n_active < 3:
            raise RuntimeError(
                f"[MoonBoardEnv] reset(): only {n_active}/4 grips active after warmup. "
                f"Active slots: {sorted(active.keys())}. "
                "Check kickboard IK and foot proximity threshold."
            )
        if n_active < 4:
            print(
                f"[MoonBoardEnv] WARNING: reset() achieved {n_active}/4 grips "
                f"(active slots: {sorted(active.keys())}).  "
                f"Foot holds '{lfoot_hold}'/'{rfoot_hold}' may be out of reach."
            )

        # ── Reset episode counters ────────────────────────────────────────────
        self._step_count = 0
        self._consec_finish_count = 0
        self._holds_matched_this_episode = set()

        self._hand_mid_ptr = [0, 0]
        self._hand_mid_history = [[], []]
        first_mid_ri = self._mid_holds_ri[0] if self._mid_holds_ri else 0

        self._target_hold_indices = [
            first_mid_ri,   # slot 0 (lhand) → first mid hold
            first_mid_ri,   # slot 1 (rhand) → first mid hold
            first_mid_ri,   # slot 2 (lfoot) — unused; feet use _foot_target_hold_names
            first_mid_ri,   # slot 3 (rfoot) — unused; feet use _foot_target_hold_names
        ]
        # Foot targets always start on kickboard upper holds.
        self._foot_target_hold_names = list(self._foot_hold_names)
        self._prev_grip_target_state = [False, False, False, False]

        pelvis_pos = np.array(self._data.xpos[self._torso_id])
        self._reset_pelvis_z = float(pelvis_pos[2])
        self._max_pelvis_z   = self._reset_pelvis_z

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the simulation by one policy step.

        For hand slots: tries the scripted route-hold target.
        For foot slots: tries the current foot target hold first; if that fails,
        searches all holds registered for that slot (including kickboard holds)
        and grips the first within proximity.

        Args:
            action: Flat float32 array of shape (nu + 4,).

        Returns:
            (obs, reward, terminated, truncated, info) with per-component rewards.
        """
        joint_targets, grip_intents = self._parse_action(action)

        self._data.ctrl[:] = joint_targets
        for _ in range(self._sim_substeps):
            self._mj.mj_step(self._model, self._data)

        # ── Process grip intents ──────────────────────────────────────────────
        for slot in range(4):
            if grip_intents[slot] > 0.0:
                if slot in (0, 1):
                    # Hand: scripted route-hold target only.
                    th = self._route.holds[self._target_hold_indices[slot]]
                    scripted_target = hold_body_name(th.col, th.row)
                    self._grip_manager.try_grip(slot, scripted_target)
                else:
                    # Foot: try scripted kickboard/route target first, then fallback.
                    scripted_target = self._foot_target_hold_names[slot - 2]
                    success = self._grip_manager.try_grip(slot, scripted_target)
                    if not success:
                        # Fallback: search all valid holds for this slot.
                        for hold_name in self._grip_manager.get_available_holds_for_slot(slot):
                            if self._grip_manager.try_grip(slot, hold_name):
                                break
            else:
                self._grip_manager.release_grip(slot)

        # ── Slip detection ────────────────────────────────────────────────────
        self._grip_manager.check_slip()

        # ── Compute per-slot "gripping its target" state ──────────────────────
        active = self._grip_manager.get_active_hold_ids()
        curr_grip_target: list[bool] = [False, False, False, False]
        for slot in range(4):
            if slot in (0, 1):
                th = self._route.holds[self._target_hold_indices[slot]]
                target_name = hold_body_name(th.col, th.row)
            else:
                target_name = self._foot_target_hold_names[slot - 2]
            curr_grip_target[slot] = (active.get(slot) == target_name)

        # ── Advance target holds on rising edge ───────────────────────────────
        for slot in range(4):
            rising = curr_grip_target[slot] and not self._prev_grip_target_state[slot]
            if rising:
                if slot in (0, 1):
                    self._advance_hand_target(slot)
                else:
                    self._advance_foot_target(slot)

        # ── Reward ────────────────────────────────────────────────────────────
        reward, info = self._compute_reward(curr_grip_target, joint_targets)
        self._prev_grip_target_state = list(curr_grip_target)

        # ── Termination ───────────────────────────────────────────────────────
        pelvis_pos = np.array(self._data.xpos[self._torso_id])
        fell = bool(
            pelvis_pos[2] < _FALL_Z_THRESHOLD or pelvis_pos[1] > _FALL_Y_THRESHOLD
        )

        both_on_finish = False
        if self._end_holds_ri:
            finish_name = hold_body_name(
                self._route.holds[self._end_holds_ri[0]].col,
                self._route.holds[self._end_holds_ri[0]].row,
            )
            if active.get(0) == finish_name and active.get(1) == finish_name:
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

        # ── Foot hold info ────────────────────────────────────────────────────
        info["foot_hold_names"] = [
            self._grip_manager.get_gripped_hold(2),
            self._grip_manager.get_gripped_hold(3),
        ]

        # ── Truncation ────────────────────────────────────────────────────────
        self._step_count += 1
        truncated = (self._step_count >= self._max_episode_steps)

        obs = self._get_obs()
        return obs, float(reward), bool(terminated), bool(truncated), info

    # ─────────────────────────────────────────────────────────────────────────
    # Observation assembly
    # ─────────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Build and return the flat observation vector (139 dimensions).

        Stream 4 appends 6 new dims: for each foot (slots 2 and 3), the relative
        XYZ from the foot site to the nearest kickboard hold, expressed in the
        pelvis (torso) frame.  These give the policy a direct signal for foot
        placement on the kickboard without searching the exteroception stream.

        Returns:
            Float32 array of shape (139,).

        Raises:
            RuntimeError: If any NaN or Inf is detected, identifying the stream.
        """
        data = self._data

        # ── Stream 1: Proprioception ──────────────────────────────────────────
        joint_pos = np.array(data.qpos[7:], dtype=np.float32)

        joint_vel = np.array(data.qvel[6:], dtype=np.float32)

        pelvis_world = np.array(data.xpos[self._torso_id], dtype=np.float32)

        pelvis_mat = np.array(data.xmat[self._torso_id]).reshape(3, 3)
        rot6d = pelvis_mat[:, :2].flatten().astype(np.float32)

        linvel = np.array(data.qvel[0:3], dtype=np.float32)
        angvel = np.array(data.qvel[3:6], dtype=np.float32)

        site_in_pelvis = np.empty(12, dtype=np.float32)
        for k, sid in enumerate(self._site_ids):
            site_world = np.array(data.site_xpos[sid])
            rel = site_world - pelvis_world.astype(np.float64)
            site_in_pelvis[k * 3: k * 3 + 3] = (pelvis_mat.T @ rel).astype(np.float32)

        grip_state = self._grip_manager.get_grip_state()

        stream1 = np.concatenate([
            joint_pos, joint_vel, pelvis_world, rot6d,
            linvel, angvel, site_in_pelvis, grip_state,
        ])  # 65

        # ── Stream 2: Exteroception ───────────────────────────────────────────
        active = self._grip_manager.get_active_hold_ids()
        active_hold_names: set[str] = set(active.values())

        pelvis_64 = pelvis_world.astype(np.float64)
        # Only sort main-wall holds for nearest-hold display; kickboard holds
        # appear in stream4 instead.
        main_hold_items = [
            (n, p) for n, p in self._hold_positions.items()
            if not n.startswith("kb_hold_")
        ]
        sorted_holds = sorted(
            main_hold_items,
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
            if slot in (0, 1):
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
            else:
                # Foot: goal vector toward current foot target hold.
                fname = self._foot_target_hold_names[slot - 2]
                if fname in self._hold_positions:
                    target_world = self._hold_positions[fname]
                    site_world = np.array(data.site_xpos[self._site_ids[slot]])
                    stream3[slot * 3: slot * 3 + 3] = (
                        (target_world - site_world).astype(np.float32)
                    )

        # ── Stream 4: Foot → nearest kickboard hold (in pelvis frame) ────────
        # 3 dims per foot × 2 feet = 6 dims total.
        # Gives the policy a direct, stable proximity signal for foot placement
        # on the kickboard without needing to search stream2.
        stream4 = np.zeros(6, dtype=np.float32)
        # Nearest kickboard hold for each foot (by Euclidean distance from site).
        for k, foot_slot in enumerate((2, 3)):
            site_world = np.array(data.site_xpos[self._site_ids[foot_slot]])
            nearest_kb = min(
                self._kb_positions.values(),
                key=lambda p: float(np.linalg.norm(site_world - p)),
            )
            rel_world = (nearest_kb - site_world).astype(np.float64)
            rel_pelvis = (pelvis_mat.T @ rel_world).astype(np.float32)
            stream4[k * 3: k * 3 + 3] = rel_pelvis

        obs = np.concatenate([stream1, stream2, stream3, stream4]).astype(np.float32)

        # ── NaN / Inf guard ───────────────────────────────────────────────────
        if not np.all(np.isfinite(obs)):
            s1_end = self._stream1_dim
            s2_end = s1_end + self._stream2_dim
            s3_end = s2_end + self._stream3_dim
            for start, end, name in [
                (0, s1_end, "stream1/proprioception"),
                (s1_end, s2_end, "stream2/exteroception"),
                (s2_end, s3_end, "stream3/goal"),
                (s3_end, self._obs_dim, "stream4/foot-kickboard"),
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
    # Reward function
    # ─────────────────────────────────────────────────────────────────────────

    def _compute_reward(
        self,
        curr_grip_target: list[bool],
        joint_targets: np.ndarray,
    ) -> tuple[float, dict[str, float]]:
        """Compute base reward components (fall/finish bonuses added by step()).

        Args:
            curr_grip_target: Per-slot bool — True if slot grips its target hold.
            joint_targets: Float32 array (nu,) of joint position targets.

        Returns:
            Tuple of (total_base_reward, info_dict).
        """
        info: dict[str, float] = {}

        # High-Water Mark height reward: only new upward progress earns reward.
        pelvis_z = float(np.array(self._data.xpos[self._torso_id])[2])
        if pelvis_z > self._max_pelvis_z:
            height_reward = float((pelvis_z - self._max_pelvis_z) * _HWM_HEIGHT_SCALE)
            self._max_pelvis_z = pelvis_z
        else:
            height_reward = 0.0
        info["height_reward"] = height_reward

        # Hold match bonus (rising edge + yo-yo guard).
        hold_match_bonus = 0.0
        active = self._grip_manager.get_active_hold_ids()
        for slot in range(4):
            rising = curr_grip_target[slot] and not self._prev_grip_target_state[slot]
            if not rising:
                continue
            if slot in (0, 1):
                th = self._route.holds[self._target_hold_indices[slot]]
                hold_name = hold_body_name(th.col, th.row)
            else:
                hold_name = self._foot_target_hold_names[slot - 2]
            if hold_name not in self._holds_matched_this_episode:
                hold_match_bonus += 5.0
                self._holds_matched_this_episode.add(hold_name)
        info["hold_match_bonus"] = hold_match_bonus

        # Energy penalty: discourage maximum-torque jitter.
        energy_penalty = float(
            -_ENERGY_PENALTY_COEFF * np.sum(np.square(joint_targets))
        )
        info["energy_penalty"] = energy_penalty

        total = height_reward + hold_match_bonus + energy_penalty
        return total, info

    # ─────────────────────────────────────────────────────────────────────────
    # Target hold sequencing
    # ─────────────────────────────────────────────────────────────────────────

    def _select_foot_holds(self) -> tuple[str, str]:
        """Return the kickboard upper holds as the canonical foot start positions.

        Left foot always starts on kb_hold_KB_LU, right foot on kb_hold_KB_RU.
        These holds are at z ≈ 0.270 m, within reach of the IK-positioned legs.
        This is route-independent: every episode starts from the kickboard.

        Returns:
            Tuple (lfoot_hold_name, rfoot_hold_name).
        """
        return ("kb_hold_KB_LU", "kb_hold_KB_RU")

    def _advance_hand_target(self, hand_slot: int) -> None:
        """Advance a hand slot's target to the next available mid hold.

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
            if self._end_holds_ri:
                self._target_hold_indices[hand_slot] = self._end_holds_ri[0]
            self._hand_mid_ptr[hand_slot] = len(self._mid_holds_ri)

    def _advance_foot_target(self, foot_slot: int) -> None:
        """Advance a foot slot's target using a two-step lag behind its hand.

        When the foot grips its current target, it advances to the route hold
        that the corresponding hand was targeting two moves ago.  This keeps
        feet tracking hands with a natural lag.

        Args:
            foot_slot: 2 (left foot) or 3 (right foot).
        """
        hand_slot = foot_slot - 2
        history = self._hand_mid_history[hand_slot]

        new_ri: int | None = None
        two_ago_idx = len(history) - 2
        if two_ago_idx >= 0:
            mid_ptr = history[two_ago_idx]
            if mid_ptr < len(self._mid_holds_ri):
                new_ri = self._mid_holds_ri[mid_ptr]

        if new_ri is None and self._mid_holds_ri:
            other_foot = 5 - foot_slot
            other_target = self._foot_target_hold_names[other_foot - 2]
            for ri in self._mid_holds_ri:
                candidate = hold_body_name(
                    self._route.holds[ri].col, self._route.holds[ri].row
                )
                if candidate != other_target:
                    new_ri = ri
                    break
            if new_ri is None:
                new_ri = self._mid_holds_ri[0]

        if new_ri is not None:
            # Transition from kickboard to route hold.
            h = self._route.holds[new_ri]
            self._foot_target_hold_names[foot_slot - 2] = hold_body_name(h.col, h.row)

    # ─────────────────────────────────────────────────────────────────────────
    # IK helper
    # ─────────────────────────────────────────────────────────────────────────

    def _ik_feet_to_kickboard(self) -> None:
        """Damped-Least-Squares IK to place foot sites near the kickboard upper holds.

        Uses 4 DOFs per leg (hip_x, hip_z, hip_y, knee) and Damped Least Squares
        (DLS) for robust convergence, which avoids the divergence that simple
        gradient descent suffers near joint limits.

        DLS update: dq = J.T @ (J @ J.T + λI)^-1 @ err   (λ=0.01)

        Runs 30 iterations, step scale 0.5.  Terminates early if foot site is
        within 150 mm of target.

        Should be called while the body is in a settled state (e.g., after
        Phase 1 hand-warmup) so the Jacobian reflects realistic leg kinematics.
        Caller must call mj_forward after this method returns.

        Target holds (units: metres):
          Left  foot → kb_hold_KB_LU
          Right foot → kb_hold_KB_RU
        """
        # Joint qpos indices for each leg (4 DOFs each).
        slot_params = [
            # (foot_slot, target_name, [qi0, qi1, qi2, qi3], [(lo,hi)...])
            (
                2, "kb_hold_KB_LU",
                [_QI_LEFT_HIP_Z,  _QI_LEFT_HIP_Y,  _QI_LEFT_KNEE],
                [(-60, 35), (-110, 20), (-160, -2)],
            ),
            (
                3, "kb_hold_KB_RU",
                [_QI_RIGHT_HIP_Z, _QI_RIGHT_HIP_Y, _QI_RIGHT_KNEE],
                [(-60, 35), (-110, 20), (-160, -2)],
            ),
        ]

        _lam = 0.01   # DLS damping — keeps updates stable near singularities

        for foot_slot, target_name, qi_list, ranges_deg in slot_params:
            target = self._kb_positions[target_name]
            site_id = self._site_ids[foot_slot]
            site_body_id = int(self._model.site_bodyid[site_id])

            # qpos index → qvel index: vi = qi - 1 (freejoint offset).
            vi_list = [qi - 1 for qi in qi_list]
            lo_arr = np.array([np.radians(r[0]) for r in ranges_deg])
            hi_arr = np.array([np.radians(r[1]) for r in ranges_deg])
            n_dofs = len(qi_list)

            for _ in range(30):
                self._mj.mj_forward(self._model, self._data)
                site_pos = np.array(self._data.site_xpos[site_id])
                err = target - site_pos
                if np.linalg.norm(err) < 0.15:
                    break

                jacp = np.zeros((3, self._model.nv))
                jacr = np.zeros((3, self._model.nv))
                self._mj.mj_jac(
                    self._model, self._data, jacp, jacr,
                    site_pos, site_body_id,
                )

                # Sub-Jacobian: columns for the selected DOFs (shape 3 × n_dofs).
                J_sub = jacp[:, vi_list]   # 3 × n_dofs

                # DLS solution: dq = J.T @ (J @ J.T + λI)^-1 @ err
                A = J_sub @ J_sub.T + _lam * np.eye(3)   # 3×3
                dq = 0.5 * (J_sub.T @ np.linalg.solve(A, err))  # n_dofs

                # Apply update, clip to joint limits.
                for k, qi in enumerate(qi_list):
                    new_val = float(self._data.qpos[qi]) + float(dq[k])
                    self._data.qpos[qi] = float(np.clip(new_val, lo_arr[k], hi_arr[k]))

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
            Tuple of (joint_targets, grip_intents).
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
