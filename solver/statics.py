"""Static equilibrium analysis for climbing holds.

This module computes the hold forces on a climber in quasi-static equilibrium
and determines whether any grip will slip under the applied loads.  It is the
physics layer used by ``validate_move()`` to check move legality without
running a Pymunk simulation.

Background
----------
For quasi-static climbing (each move completes before the next begins),
static equilibrium gives the correct hold loads directly — no integration
loop, no numerical instability, no joint-motor noise.  The friction-cone
condition is:

    slip if  f_tangential > μ × f_normal

where:
  f_tangential  wall-parallel (downward, gravity direction) force at the grip
  f_normal      wall-perpendicular force — active grip strength for hands,
                passive foot press for feet

Hold loads are distributed by solving the 2-equation moment-balance system
(vertical force balance + torque about the y-axis) via minimum-norm
least-squares.  This reduces to an exact solution for two active grips and
distributes load proportionally to horizontal proximity for three or four
grips.

Units
-----
All forces are in Newtons (N).  Body weight uses g = 9.81 m/s² (SI).  The
grip-force constants are calibrated so that:

  • A single jug safely supports a 65 kg climber (slip margin > 0).
  • A single sloper under full bodyweight slips   (slip margin < 0).
  • A standard 4-point jug/foothold stance has comfortable positive margins.

Note on mu values
-----------------
``_MU_BY_TYPE`` and ``_MU_DEFAULT`` are imported from ``solver.physics`` so
the friction coefficients are defined in exactly one place.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from solver.body import BodyModel, HAND_LIMBS, LIMBS
from solver.physics import _MU_BY_TYPE, _MU_DEFAULT
from solver.reachability import Pose, can_reach, estimate_com
from solver.wall import Wall

# ─── Physical constants ────────────────────────────────────────────────────
_GRAVITY = 9.81            # m/s²  (SI)
_DEFAULT_MASS_KG = 65.0    # kg    (average climber)

# ─── Grip normal-force baselines (N) ──────────────────────────────────────
# These represent the maximum wall-perpendicular force each hold type can
# sustain given realistic grip technique.  They are calibrated so that the
# friction-cone model reproduces the intuitive climbing result:
#   jug   : holds a single-handed hang (μ × f_normal > bodyweight)
#   sloper: slips under a single-handed hang (μ × f_normal < bodyweight)
#
# 65 kg × 9.81 m/s² ≈ 637 N.
# Jug  threshold: 637 / 1.2  ≈ 531 N  → use 600 N  (1.2 × 600 = 720 > 637 ✓)
# Sloper ceiling: 637 / 0.4  = 1593 N → use  80 N  (0.4 × 80  =  32 < 637 ✓)
# Foot threshold: need μ × f_foot > W even if one foot bears full load →
#   0.9 × 800 = 720 > 637 ✓  (foot-press set high: on a vertical wall the
#   climber loads a foothold with close to their full bodyweight via leg drive)

_HAND_GRIP_FORCE: dict[str, float] = {
    "jug":    600.0,   # N — deep positive lip; finger curl + arm tension
    "pinch":  250.0,   # N — thumb-opposition pinch; moderate capacity
    "crimp":  200.0,   # N — crimped lip; limited by finger flexor strength
    "sloper":  80.0,   # N — friction-only contact; very low normal force
}
_HAND_GRIP_DEFAULT = 150.0   # N — fallback for unknown hand-hold types
_FOOT_PRESS_FORCE  = 800.0   # N — foot smear/press on any foot-usable hold


# ─── Dataclasses ──────────────────────────────────────────────────────────

@dataclass
class HoldForces:
    """Computed grip force components and slip margin for one active hold."""

    limb:         str    # "LH" | "RH" | "LF" | "RF"
    hold_id:      str
    hold_type:    str
    f_tangential: float  # downward (gravity-direction) force at the grip (N)
    f_normal:     float  # wall-perpendicular force: grip strength / foot press (N)
    mu:           float  # friction coefficient for this hold type
    slip_margin:  float  # μ × f_normal − f_tangential  (positive = safe)


@dataclass
class MoveResult:
    """Result of ``validate_move()``."""

    success:        bool
    failure_reason: Optional[str]        # "out_of_reach" | "slip_at_source" | "slip_at_target"
    slip_margins:   dict[str, float]     # per-limb margin at the evaluated pose (N)
    final_pose:     Optional[Pose]       # new Pose on success, None on failure


# ─── Private helpers ──────────────────────────────────────────────────────

def _total_mass_kg(body_model: Optional[BodyModel]) -> float:
    """Estimate total climber mass from a BodyModel, or return the default.

    ``BodyModel`` stores height and wingspan but not mass.  We use a standard
    anthropometric estimate: 65 kg scales linearly with height from 175 cm.
    """
    if body_model is None:
        return _DEFAULT_MASS_KG
    return _DEFAULT_MASS_KG * (body_model.height_cm / 175.0)


# ─── Public API ───────────────────────────────────────────────────────────

def compute_hold_forces(
    wall: Wall,
    pose: Pose,
    body_model: Optional[BodyModel] = None,
) -> dict[str, HoldForces]:
    """Compute the tangential and normal force at every active grip.

    Uses two-equation moment equilibrium (vertical force balance + torque
    about the world y-axis) solved with minimum-norm least-squares.  This
    gives an exact solution for two-grip stances and the most evenly
    distributed load for three or four grips.  Negative tangential forces
    (which would imply a hold pushing the climber upward) are clipped to zero.

    The wall-normal (grip-strength / foot-press) forces are hold-type
    constants, independent of body position.  They represent the force the
    climber can exert perpendicular to the wall surface, not the reaction to
    gravity.

    Args:
        wall:        Wall geometry and hold positions.
        pose:        Current 4-limb hold assignment (``None`` = limb in flight).
        body_model:  Climber anthropometrics.  Defaults to 175 cm / 65 kg.

    Returns:
        ``dict[limb_name → HoldForces]``, one entry per active (non-None) grip.
        Returns an empty dict if no limbs are on holds.
    """
    # ── Step 1: active grips ───────────────────────────────────────────────
    active: dict[str, str] = {
        limb: pose.get(limb)  # type: ignore[assignment]
        for limb in LIMBS
        if pose.get(limb) is not None
    }
    if not active:
        return {}

    # ── Step 2: total body weight ──────────────────────────────────────────
    W = _total_mass_kg(body_model) * _GRAVITY   # N

    # ── Step 3: hold x-positions and COM x-position ───────────────────────
    n = len(active)
    limb_order = list(active.keys())
    x_holds = np.array(
        [wall.by_id(active[l]).x_cm for l in limb_order], dtype=float
    )

    com = estimate_com(wall, pose, body_model)
    x_com = float(com[0])

    # ── Step 4: moment-equilibrium solve ──────────────────────────────────
    # System:
    #   Σ f_i         = W          (vertical force balance)
    #   Σ f_i * x_i   = W * x_com  (moment balance — torques about y-axis)
    #
    # Matrix form:  A @ f = b,  A is (2, n), f is (n,), b is (2,).
    # np.linalg.lstsq returns the minimum-norm solution for n > 2 (under-
    # determined) and the least-squares best fit for n < 2 (over-determined).
    A = np.vstack([np.ones(n), x_holds])        # (2, n)
    b = np.array([W, W * x_com])                # (2,)
    f_raw, _, _, _ = np.linalg.lstsq(A, b, rcond=None)   # (n,)

    # Grips cannot push the climber upward; clip to zero.
    f_tangential = np.clip(f_raw, 0.0, None)

    # ── Steps 5–6: normal forces, mu, slip margins ────────────────────────
    result: dict[str, HoldForces] = {}
    for i, limb in enumerate(limb_order):
        hold      = wall.by_id(active[limb])
        hold_type = hold.hold_type

        if limb in HAND_LIMBS:
            f_norm = _HAND_GRIP_FORCE.get(hold_type, _HAND_GRIP_DEFAULT)
        else:
            f_norm = _FOOT_PRESS_FORCE

        mu      = _MU_BY_TYPE.get(hold_type, _MU_DEFAULT)
        f_tang  = float(f_tangential[i])
        margin  = mu * f_norm - f_tang

        result[limb] = HoldForces(
            limb=limb,
            hold_id=active[limb],
            hold_type=hold_type,
            f_tangential=f_tang,
            f_normal=f_norm,
            mu=mu,
            slip_margin=margin,
        )

    return result


def is_slip_free(
    wall: Wall,
    pose: Pose,
    body_model: Optional[BodyModel] = None,
) -> bool:
    """Return ``True`` if every active grip has a non-negative slip margin.

    A slip margin ≥ 0 means the hold can support the applied tangential
    (downward) load without slipping.  A negative margin means the grip is
    overloaded and will fail.

    An empty pose (no active grips) is considered slip-free — there is nothing
    to slip from.
    """
    forces = compute_hold_forces(wall, pose, body_model)
    return all(hf.slip_margin >= 0.0 for hf in forces.values())


def validate_move(
    wall: Wall,
    pose: Pose,
    move: tuple[str, str],
    body_model: Optional[BodyModel] = None,
) -> MoveResult:
    """Check whether a one-limb move is valid under static equilibrium.

    Three checks are applied in order:

    1. **Geometric reachability** — the target hold must be within the limb's
       anatomical envelope and IK-solvable (delegates to
       ``reachability.can_reach``).

    2. **Source-pose stability** — the current stance must not already be
       slipping.  A climber whose grip is failing cannot safely initiate a
       new move.

    3. **Target-pose stability** — the new stance (current pose with the one
       moved limb reassigned) must also be slip-free.

    This is a quasi-static model — it validates end-states only.  Dynamic
    forces during the limb transition are not evaluated; that is the concern
    of the Pymunk layer (``solver.physics``).

    Args:
        wall:        Wall geometry and holds.
        pose:        Current 4-limb hold assignment.
        move:        ``(limb_name, target_hold_id)``, e.g. ``("LH", "h_007")``.
        body_model:  Climber anthropometrics.  Defaults to ``BodyModel()``.

    Returns:
        :class:`MoveResult` with ``success``, ``failure_reason``,
        per-grip ``slip_margins`` (N, positive = safe), and ``final_pose``
        on success.
    """
    limb, target_hold_id = move
    bm = body_model if body_model is not None else BodyModel()

    # ── Check 1: geometric reachability ───────────────────────────────────
    com         = estimate_com(wall, pose, bm)
    target_hold = wall.by_id(target_hold_id)
    if not can_reach(bm, com, limb, target_hold):  # type: ignore[arg-type]
        return MoveResult(
            success=False,
            failure_reason="out_of_reach",
            slip_margins={},
            final_pose=None,
        )

    # ── Check 2: source pose is slip-free ─────────────────────────────────
    if not is_slip_free(wall, pose, bm):
        forces  = compute_hold_forces(wall, pose, bm)
        margins = {l: hf.slip_margin for l, hf in forces.items()}
        return MoveResult(
            success=False,
            failure_reason="slip_at_source",
            slip_margins=margins,
            final_pose=None,
        )

    # ── Check 3: target pose is slip-free ─────────────────────────────────
    new_pose = pose.with_limb(limb, target_hold_id)  # type: ignore[arg-type]
    if not is_slip_free(wall, new_pose, bm):
        forces  = compute_hold_forces(wall, new_pose, bm)
        margins = {l: hf.slip_margin for l, hf in forces.items()}
        return MoveResult(
            success=False,
            failure_reason="slip_at_target",
            slip_margins=margins,
            final_pose=None,
        )

    forces  = compute_hold_forces(wall, new_pose, bm)
    margins = {l: hf.slip_margin for l, hf in forces.items()}
    return MoveResult(
        success=True,
        failure_reason=None,
        slip_margins=margins,
        final_pose=new_pose,
    )
