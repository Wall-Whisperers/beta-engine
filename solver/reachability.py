"""Pose, reachability + stability checks.

A `Pose` is which hold each limb is currently on (or `None` if a limb is
mid-flight — only used during a one-limb-at-a-time move).

Reachability is now a *three-stage* test from each limb's body anchor to
the candidate hold's centre:

    1. Distance test  — target within REACH_SAFETY × max_reach
    2. Envelope test  — target inside the limb's anatomical box
                        (see `BodyModel.envelope_box`)
    3. IK test        — closed-form 2-link IK actually solves
                        (catches "almost in reach but folds the limb")

Stability (vertical-wall MVP): the COM x-coord must lie within the
horizontal span between the two feet. With only one foot, the COM x-coord
must lie within `STABILITY_TOLERANCE_CM` of that foot.

Pose-level anatomy: hands/feet can cross body but only by limited
amounts; no two end-effectors share the same patch of wall; feet stay
below the shoulders. See `pose_anatomy_ok`.

═══════════════════════════════════════════════════════════════════════════
  Everything here is 2D. There is no body twist, no out-of-plane drop-knee,
  no friction model. The constants below are knobs — bias them looser for
  more aggressive (acrobatic) betas, tighter for safer/more conservative
  betas.
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional

import numpy as np

from solver.body import (
    BodyModel,
    END_EFFECTOR_MIN_SEPARATION_CM,
    FOOT_CROSSOVER_LIMIT_CM,
    FOOT_LIMBS,
    HAND_CROSSOVER_LIMIT_CM,
    HAND_LIMBS,
    LIMBS,
    Limb,
    solve_2link_ik,
)
from solver.wall import Hold, Wall

REACH_SAFETY = 0.75       # use ≤75 % of max reach — comfortable, non-extended moves only
STABILITY_TOLERANCE_CM = 5.0
COM_HIP_HEIGHT_FRAC = 0.45  # pelvis sits at 45 % of leg length above the feet (≈ hip height)


@dataclass(frozen=True)
class Pose:
    """Which hold each limb sits on. `None` = limb is in flight."""

    LH: Optional[str]
    RH: Optional[str]
    LF: Optional[str]
    RF: Optional[str]

    def get(self, limb: Limb) -> Optional[str]:
        return getattr(self, limb)

    def with_limb(self, limb: Limb, hold_id: Optional[str]) -> "Pose":
        return replace(self, **{limb: hold_id})

    def hand_holds(self) -> tuple[Optional[str], Optional[str]]:
        return self.LH, self.RH

    def foot_holds(self) -> tuple[Optional[str], Optional[str]]:
        return self.LF, self.RF

    def as_tuple(self) -> tuple[Optional[str], ...]:
        return (self.LH, self.RH, self.LF, self.RF)


def estimate_com(wall: Wall, pose: Pose, body: "BodyModel | None" = None) -> np.ndarray:
    """Estimate pelvis/COM position for a pose.

    Anchored to the feet: COM sits at hip height (half leg length) above the
    foot midpoint. Hand position only nudges the x-coordinate slightly so the
    body leans toward the hands — it does not move the COM vertically.
    This keeps the shoulder/hip anchors stable between steps so the IK pole
    vectors don't flip as hands move to different holds.
    """
    from solver.body import BodyModel  # local import avoids circular dep
    if body is None:
        body = BodyModel()

    hand_pts = [_pt(wall, pose.get(l)) for l in HAND_LIMBS]
    foot_pts = [_pt(wall, pose.get(l)) for l in FOOT_LIMBS]
    hand_pts = [p for p in hand_pts if p is not None]
    foot_pts = [p for p in foot_pts if p is not None]

    if foot_pts:
        foot_mid = np.mean(foot_pts, axis=0)
        com_y = foot_mid[1] + body.leg_length * COM_HIP_HEIGHT_FRAC
        if hand_pts:
            hand_mid = np.mean(hand_pts, axis=0)
            # Small horizontal lean toward hands (20 % weight) — keeps COM over feet.
            com_x = 0.8 * foot_mid[0] + 0.2 * hand_mid[0]
        else:
            com_x = foot_mid[0]
        return np.array([com_x, com_y])

    # No feet on wall — fall back to hand midpoint (e.g. starting position lookup).
    if hand_pts:
        return np.mean(hand_pts, axis=0)
    return np.array([0.0, 0.0])


def _pt(wall: Wall, hold_id: Optional[str]) -> Optional[np.ndarray]:
    if hold_id is None:
        return None
    h = wall.by_id(hold_id)
    return np.array([h.x_cm, h.y_cm])


def is_stable(wall: Wall, pose: Pose) -> bool:
    """Vertical-wall stability: COM x-coord must sit between the two
    foot x-coords (or within tolerance of the single foot)."""
    com = estimate_com(wall, pose)
    feet = [_pt(wall, pose.get(l)) for l in FOOT_LIMBS]
    feet = [p for p in feet if p is not None]
    if not feet:
        return False
    if len(feet) == 1:
        return abs(com[0] - feet[0][0]) <= STABILITY_TOLERANCE_CM
    xs = [p[0] for p in feet]
    return min(xs) - STABILITY_TOLERANCE_CM <= com[0] <= max(xs) + STABILITY_TOLERANCE_CM


def can_reach(
    body: BodyModel,
    com: np.ndarray,
    limb: Limb,
    target: Hold,
) -> bool:
    """Three-stage reachability test:

    1. Distance: target within `REACH_SAFETY × max_reach` of the anchor.
       Fully extended limbs can technically reach further, but it's
       physiologically miserable (no margin for adjustment, joint locked
       out).
    2. Envelope: target sits inside the limb's anatomical box — captures
       'foot can't go above shoulder', cross-body limits, etc. See
       `BodyModel.envelope_box`.
    3. IK: closed-form 2-link IK actually solves. Catches near-anchor
       targets where the limb would have to fold past itself.
    """
    anchor = com + body.anchor_offset(limb)
    target_pt = np.array([target.x_cm, target.y_cm])

    dist = float(np.linalg.norm(anchor - target_pt))
    if dist > REACH_SAFETY * body.reach_radius(limb):
        return False

    if not body.in_envelope(limb, anchor, target_pt):
        return False

    # Hard rule: foot can never be above shoulder height in world space.
    if limb in FOOT_LIMBS and target_pt[1] > body.foot_world_ceiling(com):
        return False

    is_leg = limb in FOOT_LIMBS
    upper, lower = (body.upper_leg(), body.lower_leg()) if is_leg else (body.upper_arm(), body.lower_arm())
    # Either IK branch being solvable is enough for reachability — pole selection
    # only affects which branch visualize.py draws, not whether the move is legal.
    if solve_2link_ik(anchor, target_pt, upper, lower, elbow_up=True) is None and \
       solve_2link_ik(anchor, target_pt, upper, lower, elbow_up=False) is None:
        return False

    return True


def pose_anatomy_ok(wall: Wall, pose: Pose) -> bool:
    """Pose-level anatomy: limb crossover limits + minimum separation
    between any two end-effectors. Single-limb envelope checks live in
    `can_reach`; this catches the multi-limb cases."""
    pts = {
        l: _pt(wall, pose.get(l)) for l in LIMBS
    }
    # Hand crossover: LH should not be far to the right of RH.
    if pts["LH"] is not None and pts["RH"] is not None:
        if pts["LH"][0] - pts["RH"][0] > HAND_CROSSOVER_LIMIT_CM:
            return False
    # Foot crossover: LF should not be far to the right of RF.
    if pts["LF"] is not None and pts["RF"] is not None:
        if pts["LF"][0] - pts["RF"][0] > FOOT_CROSSOVER_LIMIT_CM:
            return False
    # No two end-effectors at the exact same patch of wall —
    # except both hands matching the same finish hold is explicitly allowed.
    finish_ids = {h.hold_id for h in wall.finishes()}
    placed = [(l, p) for l, p in pts.items() if p is not None]
    for i, (la, a) in enumerate(placed):
        for lb, b in placed[i + 1:]:
            if float(np.linalg.norm(a - b)) < END_EFFECTOR_MIN_SEPARATION_CM:
                matching_hands = (
                    la in HAND_LIMBS and lb in HAND_LIMBS
                    and pose.get(la) in finish_ids
                    and pose.get(la) == pose.get(lb)
                )
                if not matching_hands:
                    return False
    return True


def reachable_holds(
    body: BodyModel,
    wall: Wall,
    pose: Pose,
    limb: Limb,
) -> list[Hold]:
    """All holds the given limb could move to from the current pose."""
    # Compute the COM the body would hold while the moving limb is in flight
    # (briefly remove that limb's anchor contribution).
    com_pose = pose.with_limb(limb, None)
    com = estimate_com(wall, com_pose)

    candidates: Iterable[Hold]
    if limb in HAND_LIMBS:
        candidates = (h for h in wall.holds if h.usable_for_hand())
    else:
        candidates = (h for h in wall.holds if h.usable_for_foot())

    occupied = {pose.LH, pose.RH, pose.LF, pose.RF} - {None, pose.get(limb)}
    finish_ids = {h.hold_id for h in wall.finishes()}

    def _allowed(h: Hold) -> bool:
        if h.hold_id in occupied:
            # Allow both hands to match the same finish hold.
            if limb in HAND_LIMBS and h.hold_id in finish_ids:
                other_hand = "RH" if limb == "LH" else "LH"
                return pose.get(other_hand) == h.hold_id
            return False
        return True

    return [h for h in candidates if _allowed(h) and can_reach(body, com, limb, h)]


def _all_limbs_reachable(body: BodyModel, wall: Wall, pose: Pose) -> bool:
    """After a move, verify every stationary limb is still within reach.

    When feet advance they pull the COM up, which can push a shoulder far
    above a low hand hold. The one-limb reachability check doesn't catch
    this because it computes COM *without* the moving limb. This check uses
    the full 4-limb COM and is applied to the resulting pose.
    """
    com = estimate_com(wall, pose)
    for limb in LIMBS:
        hold_id = pose.get(limb)
        if hold_id is None:
            continue
        h = wall.by_id(hold_id)
        if not can_reach(body, com, limb, h):
            return False
    return True


def reachable_moves(
    body: BodyModel,
    wall: Wall,
    pose: Pose,
) -> list[tuple[Limb, str]]:
    """Every legal one-limb move from the current pose. A move is legal
    when:
      - the *moving* state (limb in flight, 3 points of contact) is stable,
      - the *target* hold is reachable + inside the limb's envelope + IK-solvable,
      - the *resulting* 4-limb pose is stable, anatomically OK, AND all
        stationary limbs remain within reach from the new COM.
    """
    moves: list[tuple[Limb, str]] = []
    for limb in LIMBS:
        if not _three_points_stable(wall, pose, limb):
            continue
        for target in reachable_holds(body, wall, pose, limb):
            new_pose = pose.with_limb(limb, target.hold_id)
            if (is_stable(wall, new_pose)
                    and pose_anatomy_ok(wall, new_pose)
                    and _all_limbs_reachable(body, wall, new_pose)):
                moves.append((limb, target.hold_id))
    return moves


def _three_points_stable(wall: Wall, pose: Pose, moving: Limb) -> bool:
    """While `moving` is in flight, the remaining 3 limbs must keep us
    stable (or, if we're moving a hand, at least both feet remain)."""
    intermediate = pose.with_limb(moving, None)
    if moving in FOOT_LIMBS:
        # Need the other foot at minimum.
        other_foot = "RF" if moving == "LF" else "LF"
        if intermediate.get(other_foot) is None:
            return False
    return is_stable(wall, intermediate)
