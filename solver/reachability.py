"""Pose, reachability + stability checks.

A `Pose` is which hold each limb is currently on (or `None` if a limb is
mid-flight — only used during a one-limb-at-a-time move).

Reachability is a circle test from each limb's body anchor to the
candidate hold's centre. We back off slightly from `max_reach` to avoid
fully-extended-limb poses, which are physiologically unstable.

Stability (vertical-wall MVP): the COM x-coord must lie within the
horizontal span between the two feet. With only one foot, the COM x-coord
must lie within `STABILITY_TOLERANCE_CM` of that foot.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Optional

import numpy as np

from solver.body import BodyModel, FOOT_LIMBS, HAND_LIMBS, LIMBS, Limb
from solver.wall import Hold, Wall

REACH_SAFETY = 0.92  # use ≤92 % of max reach for "comfortable" reachability
STABILITY_TOLERANCE_CM = 5.0
COM_HEIGHT_BIAS = 0.55  # COM sits 55 % of the way from feet to hands


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


def estimate_com(wall: Wall, pose: Pose) -> np.ndarray:
    """Approximate COM for a pose: midpoint of hands and feet, biased
    upward toward the hands (climbers' COM sits at the pelvis).
    """
    hand_pts = [_pt(wall, pose.get(l)) for l in HAND_LIMBS]
    foot_pts = [_pt(wall, pose.get(l)) for l in FOOT_LIMBS]
    hand_pts = [p for p in hand_pts if p is not None]
    foot_pts = [p for p in foot_pts if p is not None]

    if hand_pts and foot_pts:
        hand_mid = np.mean(hand_pts, axis=0)
        foot_mid = np.mean(foot_pts, axis=0)
        return (1 - COM_HEIGHT_BIAS) * hand_mid + COM_HEIGHT_BIAS * foot_mid
    if hand_pts:
        return np.mean(hand_pts, axis=0)
    if foot_pts:
        return np.mean(foot_pts, axis=0)
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
    """True if the target is within `REACH_SAFETY × max_reach` of the
    limb's body anchor for the given COM position."""
    anchor = com + body.anchor_offset(limb)
    dist = float(np.linalg.norm(anchor - np.array([target.x_cm, target.y_cm])))
    return dist <= REACH_SAFETY * body.reach_radius(limb)


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
    return [
        h for h in candidates
        if h.hold_id not in occupied and can_reach(body, com, limb, h)
    ]


def reachable_moves(
    body: BodyModel,
    wall: Wall,
    pose: Pose,
) -> list[tuple[Limb, str]]:
    """Every legal one-limb move from the current pose. A move is legal
    when both the *moving* state (limb in flight, three points of contact)
    and the *resulting* state are stable."""
    moves: list[tuple[Limb, str]] = []
    for limb in LIMBS:
        if not _three_points_stable(wall, pose, limb):
            continue
        for target in reachable_holds(body, wall, pose, limb):
            new_pose = pose.with_limb(limb, target.hold_id)
            if is_stable(wall, new_pose):
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
