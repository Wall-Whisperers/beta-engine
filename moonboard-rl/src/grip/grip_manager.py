"""Virtual grip mechanic for the MoonBoard RL environment.

Design
------
Each limb (2 hands, 2 feet) maps to one MuJoCo *connect* equality constraint.
A connect constraint pins a point in body1's local frame to a point in body2's
local frame.  We use body1 = limb body (hand/foot), body2 = target hold body.

Why connect, not weld?
  A connect constraint is a ball-and-socket: it locks the 3-D position of the
  anchor point but leaves all rotational DOF free.  This lets the arm/leg pivot
  naturally on the hold — exactly what a real hand or foot does.  A weld would
  lock all 6 DOF, producing unrealistically stiff arms and numerically stiff
  Jacobians.

Why sites instead of body origins?
  data.xpos[body_id] returns the *body origin*, which for left_lower_arm /
  right_lower_arm is the elbow joint — not the hand tip.  Sites placed at the
  hand/foot sphere geom centres (see scene.py:_inject_limb_sites) let us use
  data.site_xpos[site_id] for accurate proximity checks and anchor placement.

Runtime retargeting protocol (must happen in this exact order)
--------------------------------------------------------------
1. Set model.eq_obj2id[eq_id] to the integer body index of the target hold.
2. Compute anchor1 = site position expressed in body1's local frame.
   (This is the site's pos attribute — constant in body frame.)
3. Compute anchor2 = site world position expressed in body2 (hold) local frame:
       rel    = site_world - hold_world
       anchor2 = R_hold.T @ rel
   Without this step, MuJoCo enforces the relative pose from model-load time
   and the arm snaps violently to an arbitrary position.
4. Set data.eq_active[eq_id] = 1.
   Note: model.eq_active0 is the load-time default only — it has no effect
   after simulation starts.  data.eq_active is the runtime-mutable field.

Limb slot convention (matches scene.py GRIP_CONSTRAINT_NAMES order)
---------------------------------------------------------------------
  0 = left hand  → body "left_lower_arm",  site "site_lhand", constraint "grip_lhand"
  1 = right hand → body "right_lower_arm", site "site_rhand", constraint "grip_rhand"
  2 = left foot  → body "left_foot",       site "site_lfoot", constraint "grip_lfoot"
  3 = right foot → body "right_foot",      site "site_rfoot", constraint "grip_rfoot"
"""

from __future__ import annotations

import numpy as np

# Import WALL_NORMAL so callers and tests can use the canonical value.
from ..xml_gen.wall import WALL_NORMAL  # noqa: F401 (re-exported intentionally)
from ..xml_gen.wall import KICKBOARD_HOLD_NAMES, kickboard_hold_positions_world
from ..xml_gen.scene import GRIP_CONSTRAINT_NAMES, LIMB_BODY_NAMES, LIMB_SITE_NAMES

# ── Grip physics thresholds ───────────────────────────────────────────────────

PROXIMITY_THRESHOLD: float = 0.12
"""Maximum distance (metres) between limb site (hand/foot tip) and hold centre
for grip to engage.  Relax to 0.20 m temporarily when debugging placement."""

ALIGNMENT_THRESHOLD: float = 0.70
"""Minimum dot product of the limb body's local Z-axis with the wall outward
normal for grip to engage.  Cosine 0.70 ≈ 45.5°.  Set to -1.0 in tests/
interactive mode where the arm is in a neutral default orientation."""

MAX_CONSTRAINT_FORCE: float = 6000.0
"""Force magnitude (Newtons) above which a grip auto-releases (slip detection).

Rationale: the MuJoCo humanoid weighs 43.7 kg (429 N).
  Static two-handed hang:  ~215 N per constraint.
  Dynamic deadpoint moves:  3-5× per-limb BW → 650-1070 N.
  Initialization transient: the connect constraint engages ~0.17 m from the
    hold surface at reset; solref="0.02 1" then produces a stiff spring force
    of ~4000-4400 N for the first 1-2 substeps until the constraint gap closes.
  True free-fall slip: body falling at ≥ 3 m/s stopped in ~20 ms →
    F = m·Δv/Δt ≈ 43.7 × 3 / 0.02 ≈ 6500 N.
6000 N sits above the initialization transient (4400 N) and just below the
free-fall threshold (≈6500 N), so the constraint releases on genuine falls
while tolerating the reset settling spike."""

SLOT_MAX_FORCE: dict[int, float] = {
    0: MAX_CONSTRAINT_FORCE,          # lhand — standard threshold
    1: MAX_CONSTRAINT_FORCE,          # rhand — standard threshold
    2: MAX_CONSTRAINT_FORCE * 1.5,    # lfoot — push load spikes during reset
    3: MAX_CONSTRAINT_FORCE * 1.5,    # rfoot — push load spikes during reset
}
"""Per-slot maximum constraint force (Newtons) before auto-release.

Feet carry compressive (push) loads on an overhanging wall and can spike
higher than hands during the reset constraint-force transient.  1.5× gives
headroom for the warmup phase without masking genuine fall-induced slip."""

SLOT_ALIGNMENT_THRESHOLD: dict[int, float] = {
    0: ALIGNMENT_THRESHOLD,   # lhand — check alignment with wall normal
    1: ALIGNMENT_THRESHOLD,   # rhand — check alignment with wall normal
    2: -1.0,                  # lfoot — orientation varies widely on overhangs
    3: -1.0,                  # rfoot — kickboard normal differs from wall normal
}
"""Per-slot minimum alignment dot-product for grip to engage.

Foot bodies rotate in many orientations during climbing (toe-on, heel-on,
smearing). The kickboard hold normal (0,1,0) also differs from WALL_NORMAL,
so feet use -1.0 (accept any orientation) while hands remain strict."""


class GripManager:
    """Manages four MuJoCo connect equality constraints for limb-to-hold gripping.

    Each instance is bound to a specific (MjModel, MjData) pair.  The manager
    reads and writes model and data fields directly; callers must call
    mujoco.mj_forward or mujoco.mj_step to propagate changes.

    Args:
        model: A loaded ``mujoco.MjModel`` instance for the scene.
        data:  The corresponding ``mujoco.MjData`` instance.
        hold_positions: Dict mapping hold body name (e.g. ``"hold_5_8"``) to a
            length-3 numpy array giving the hold body's world position.
            Used for proximity checks.  Kickboard holds are added automatically.
        hold_body_ids: Dict mapping hold body name to integer MuJoCo body index.
            Used to retarget constraints at runtime.  Kickboard holds are added
            automatically.
    """

    def __init__(
        self,
        model,
        data,
        hold_positions: dict[str, np.ndarray],
        hold_body_ids: dict[str, int],
        verbose: bool = False,
    ) -> None:
        import mujoco  # local import so the module loads without mujoco installed

        self._model = model
        self._data = data
        self._hold_positions = hold_positions
        self._hold_body_ids = hold_body_ids
        self._mj = mujoco
        self._verbose = verbose

        # Resolve constraint indices once; crash early if names are missing.
        self._eq_ids: list[int] = []
        for name in GRIP_CONSTRAINT_NAMES:
            eid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
            if eid < 0:
                raise ValueError(
                    f"Equality constraint '{name}' not found in model. "
                    "Ensure scene.py added it to the XML."
                )
            self._eq_ids.append(eid)

        # Resolve limb body indices.
        self._limb_ids: list[int] = []
        for name in LIMB_BODY_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid < 0:
                raise ValueError(f"Body '{name}' not found in model.")
            self._limb_ids.append(bid)

        # Resolve limb site indices (hand/foot tips — not body origins).
        self._site_ids: list[int] = []
        for name in LIMB_SITE_NAMES:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(
                    f"Site '{name}' not found in model. "
                    "Ensure scene.py:_inject_limb_sites ran during build."
                )
            self._site_ids.append(sid)

        # ── Register kickboard holds alongside main-wall holds ─────────────────
        # Kickboard hold bodies are injected into the XML by scene.py via
        # kickboard_holds_xml().  We add them here so try_grip() / nearest_hold()
        # work transparently for kickboard hold names.
        kb_world_positions = kickboard_hold_positions_world()
        for kb_name in KICKBOARD_HOLD_NAMES:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, kb_name)
            if bid >= 0:
                self._hold_positions[kb_name] = kb_world_positions[kb_name]
                self._hold_body_ids[kb_name] = bid
            else:
                print(f"[GripManager] WARNING: kickboard body '{kb_name}' not in model — skipped.")

        # Track kickboard hold names for get_available_holds_for_slot().
        self._kickboard_hold_names: frozenset[str] = frozenset(KICKBOARD_HOLD_NAMES)

        # Slot → active hold body name (None if not gripping).
        self._active_holds: dict[int, str | None] = {i: None for i in range(4)}

        # Track peak constraint force per slot for diagnostics.
        self.peak_forces: dict[int, float] = {i: 0.0 for i in range(4)}

    # ── Public API ─────────────────────────────────────────────────────────────

    def try_grip(self, limb_slot: int, hold_id: str, snap_to_center: bool = False) -> bool:
        """Attempt to engage a grip constraint between a limb site and a hold.

        Uses data.site_xpos[site_id] (the hand/foot tip) for distance and
        anchor calculations instead of the body origin, which is the elbow.

        Performs two checks before activating:
        1. Proximity: distance from limb site world pos to hold centre
           ≤ PROXIMITY_THRESHOLD.
        2. Alignment: dot product of limb body's local Z-axis with WALL_NORMAL
           ≥ SLOT_ALIGNMENT_THRESHOLD[limb_slot].  Foot slots use -1.0 (any
           orientation accepted) so kickboard holds engage regardless of foot
           body rotation.

        If both pass, the constraint is retargeted and activated (see module
        docstring for the exact 4-step protocol).

        Args:
            limb_slot: Integer 0–3 identifying the limb (see module docstring).
            hold_id: String key into hold_positions / hold_body_ids.
            snap_to_center: If True, set anchor2=(0,0,0) so the constraint
                spring pulls the limb site TO the hold centre during warmup.
                Use for reset foot snapping.  If False (default), anchor2 is set
                to the current limb–hold offset so position is preserved in
                place — correct for episode gripping.

        Returns:
            True if the grip was successfully engaged, False otherwise.
        """
        if limb_slot not in range(4):
            print(f"[GripManager] Invalid limb_slot {limb_slot} (must be 0–3).")
            return False

        if hold_id not in self._hold_positions:
            print(f"[GripManager] Unknown hold_id '{hold_id}'.")
            return False

        limb_body_id = self._limb_ids[limb_slot]
        site_id      = self._site_ids[limb_slot]
        hold_world_pos = self._hold_positions[hold_id]
        hold_body_id   = self._hold_body_ids[hold_id]
        eq_id          = self._eq_ids[limb_slot]

        # ── Check 1: Proximity (site = hand/foot tip, not elbow) ─────────────
        site_world_pos = np.array(self._data.site_xpos[site_id])
        dist = float(np.linalg.norm(site_world_pos - hold_world_pos))
        if dist > PROXIMITY_THRESHOLD:
            if self._verbose:
                print(
                    f"[GripManager] try_grip FAIL slot={limb_slot} hold={hold_id}: "
                    f"site dist {dist:.3f} m > threshold {PROXIMITY_THRESHOLD:.3f} m"
                )
            return False

        # ── Check 2: Alignment (per-slot threshold) ───────────────────────────
        # Foot slots (2, 3) always use -1.0 (any orientation); hand slots use
        # the current module-level ALIGNMENT_THRESHOLD so that the env can
        # temporarily override it to -1.0 during reset via _gm_mod.ALIGNMENT_THRESHOLD.
        slot_align_thresh = -1.0 if limb_slot in (2, 3) else ALIGNMENT_THRESHOLD
        limb_mat = np.array(self._data.xmat[limb_body_id]).reshape(3, 3)
        limb_z_world = limb_mat[:, 2]
        alignment = float(np.dot(limb_z_world, WALL_NORMAL))
        if alignment < slot_align_thresh:
            if self._verbose:
                print(
                    f"[GripManager] try_grip FAIL slot={limb_slot} hold={hold_id}: "
                    f"alignment {alignment:.3f} < threshold {slot_align_thresh:.3f}"
                )
            return False

        # ── Retarget and activate constraint ──────────────────────────────────
        # Step 1: Point body2 at the hold.
        self._model.eq_obj2id[eq_id] = hold_body_id

        # Step 2: anchor1 = site position in body1's local frame.
        #   site_world = xpos_body1 + R_body1 @ anchor1
        #   → anchor1  = R_body1.T @ (site_world - xpos_body1)
        limb_world_pos = np.array(self._data.xpos[limb_body_id])
        anchor1 = limb_mat.T @ (site_world_pos - limb_world_pos)

        # Step 3: anchor2 in body2 (hold) local frame.
        #   Normal mode (snap_to_center=False): preserves the current limb–hold
        #   offset so the constraint holds the limb in place (correct for episodes).
        #   Snap mode (snap_to_center=True): sets anchor2=(0,0,0) so the spring
        #   pulls the limb site TO the hold centre during warmup (correct for reset).
        if snap_to_center:
            anchor2 = np.zeros(3)
        else:
            hold_world_mat = np.array(self._data.xmat[hold_body_id]).reshape(3, 3)
            hold_world_pos_now = np.array(self._data.xpos[hold_body_id])
            rel = site_world_pos - hold_world_pos_now
            anchor2 = hold_world_mat.T @ rel

        self._model.eq_data[eq_id, 0:3] = anchor1
        self._model.eq_data[eq_id, 3:6] = anchor2

        # Step 4: Activate via data.eq_active (runtime mutable field).
        self._data.eq_active[eq_id] = 1
        self._active_holds[limb_slot] = hold_id

        if self._verbose:
            print(
                f"[GripManager] GRIP ENGAGED slot={limb_slot} "
                f"({LIMB_BODY_NAMES[limb_slot]}) → {hold_id} "
                f"site_dist={dist:.3f} m  align={alignment:.3f}  "
                f"anchor1={anchor1}  anchor2={anchor2}"
            )
        return True

    def release_grip(self, limb_slot: int) -> None:
        """Deactivate the connect constraint for the given limb slot.

        Args:
            limb_slot: Integer 0–3 identifying the limb.
        """
        eq_id = self._eq_ids[limb_slot]
        self._data.eq_active[eq_id] = 0
        released_hold = self._active_holds.get(limb_slot)
        self._active_holds[limb_slot] = None
        if self._verbose:
            print(
                f"[GripManager] GRIP RELEASED slot={limb_slot} "
                f"({LIMB_BODY_NAMES[limb_slot]}) was on {released_hold}"
            )

    def check_slip(self) -> list[int]:
        """Check constraint forces and auto-release any grip exceeding its slot's threshold.

        Uses SLOT_MAX_FORCE[slot] rather than a single global threshold, so foot
        slots (2, 3) tolerate the higher transient forces that arise from push
        loads and the reset constraint-force spike without spurious auto-release.

        Force estimation: iterates over data.efc_force rows that belong to
        active equality constraints.  For each active connect constraint (3 efc
        rows), computes the Euclidean norm of the force vector.  Falls back to
        data.qfrc_constraint at the freejoint DoFs as a proxy if the efc
        mapping yields no rows.

        Returns:
            List of limb slot indices whose grip was auto-released due to slip.
        """
        auto_released: list[int] = []
        forces = self._measure_constraint_forces()

        for slot, force_mag in forces.items():
            if force_mag > self.peak_forces[slot]:
                self.peak_forces[slot] = force_mag
            eq_id = self._eq_ids[slot]
            max_force = SLOT_MAX_FORCE.get(slot, MAX_CONSTRAINT_FORCE)
            if self._data.eq_active[eq_id] and force_mag > max_force:
                if self._verbose:
                    print(
                        f"[GripManager] SLIP slot={slot}: force {force_mag:.1f} N "
                        f"> {max_force:.1f} N  → auto-releasing"
                    )
                self.release_grip(slot)
                auto_released.append(slot)

        return auto_released

    def get_grip_state(self) -> np.ndarray:
        """Return a binary array indicating which limb slots are currently gripping.

        Returns:
            Numpy array of shape (4,) with dtype float32.  1.0 = gripping.
            Index order: [lhand, rhand, lfoot, rfoot].
        """
        return np.array(
            [1.0 if self._active_holds[i] is not None else 0.0 for i in range(4)],
            dtype=np.float32,
        )

    def get_active_hold_ids(self) -> dict[int, str]:
        """Return a dict mapping active limb slot indices to their hold_id strings.

        Returns:
            Dict where each key is a limb slot int (0–3) and each value is the
            hold_id string currently gripped by that slot.  Inactive slots omitted.
        """
        return {k: v for k, v in self._active_holds.items() if v is not None}

    def get_gripped_hold(self, slot: int) -> str | None:
        """Return the hold_id currently gripped by the given slot, or None.

        Args:
            slot: Integer 0–3 identifying the limb.

        Returns:
            Hold body name string if gripping, else None.
        """
        return self._active_holds.get(slot)

    def get_available_holds_for_slot(self, slot: int) -> list[str]:
        """Return the list of valid grip targets for the given limb slot.

        Hand slots (0, 1) can only grip main-wall holds.
        Foot slots (2, 3) can grip main-wall holds AND kickboard holds.

        Args:
            slot: Integer 0–3 identifying the limb.

        Returns:
            List of hold body name strings.  Order is not guaranteed.
        """
        main_holds = [
            name for name in self._hold_positions
            if name not in self._kickboard_hold_names
        ]
        if slot in (2, 3):
            return main_holds + list(KICKBOARD_HOLD_NAMES)
        return main_holds

    def nearest_hold(self, limb_slot: int) -> tuple[str, float] | None:
        """Find the hold whose centre is closest to the given limb's site.

        Args:
            limb_slot: Integer 0–3 identifying the limb.

        Returns:
            Tuple (hold_id, distance_metres) for the nearest hold, or None if
            hold_positions is empty.
        """
        site_id = self._site_ids[limb_slot]
        site_pos = np.array(self._data.site_xpos[site_id])
        best_id, best_dist = None, float("inf")
        for hid, hpos in self._hold_positions.items():
            d = float(np.linalg.norm(site_pos - hpos))
            if d < best_dist:
                best_dist, best_id = d, hid
        return (best_id, best_dist) if best_id is not None else None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _measure_constraint_forces(self) -> dict[int, float]:
        """Measure per-slot constraint force magnitudes from MuJoCo data arrays.

        Attempts efc_force with efc_type/efc_id matching (type 0 = equality).
        Falls back to qfrc_constraint at freejoint DoFs as a proxy if no rows
        are found.

        TODO (Week 2): verify efc_id→eq_id mapping holds for all MuJoCo 3.x
        model configurations and remove the fallback.

        Returns:
            Dict mapping active limb slot → force magnitude in Newtons.
        """
        forces = {i: 0.0 for i in range(4)}

        try:
            nefc = int(self._data.nefc)
            if nefc == 0:
                return forces

            efc_type  = np.array(self._data.efc_type[:nefc])
            efc_id    = np.array(self._data.efc_id[:nefc])
            efc_force = np.array(self._data.efc_force[:nefc])
            eq_type_val = int(self._mj.mjtConstraint.mjCNSTR_EQUALITY)

            for slot in range(4):
                if self._active_holds[slot] is None:
                    continue
                eid = self._eq_ids[slot]
                mask = (efc_type == eq_type_val) & (efc_id == eid)
                if np.any(mask):
                    forces[slot] = float(np.linalg.norm(efc_force[mask]))

        except Exception:
            # ── Fallback: qfrc_constraint at freejoint DoFs ───────────────────
            try:
                proxy = float(np.linalg.norm(self._data.qfrc_constraint[:6]))
                active_slots = [i for i in range(4) if self._active_holds[i] is not None]
                per_slot = proxy / max(len(active_slots), 1)
                for slot in active_slots:
                    forces[slot] = per_slot
            except Exception:
                pass

        return forces
