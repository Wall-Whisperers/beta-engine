"""2D body model + closed-form IK for a stick-figure climber.

The model is the 5-point figure described in CLAUDE.md / planning-gabe.md:

    - 1 Center Of Mass (COM, treated as the climber's pelvis/torso anchor)
    - 4 end-effectors (left/right hand, left/right foot)
    - 2 limbs per side, each a 3-joint chain (shoulder→elbow→wrist or
      hip→knee→ankle), short enough for closed-form IK using the law of
      cosines.

Distances are in cm. Defaults assume the "average climber" from CLAUDE.md
(175 cm height / 175 cm wingspan).

This is deliberately 2D and kinematic — no slab angle, no friction. Phase 3
of the roadmap layers a Pymunk physics step on top of this.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Limb = Literal["LH", "RH", "LF", "RF"]
LIMBS: tuple[Limb, ...] = ("LH", "RH", "LF", "RF")
HAND_LIMBS: tuple[Limb, ...] = ("LH", "RH")
FOOT_LIMBS: tuple[Limb, ...] = ("LF", "RF")


@dataclass(frozen=True)
class BodyModel:
    """Anthropometric parameters for a 2D climber."""

    height_cm: float = 175.0
    wingspan_cm: float = 175.0

    @property
    def arm_length(self) -> float:
        # Wingspan = full reach across both arms + shoulder width.
        # ≈ 0.42 * wingspan per arm is a decent first-pass.
        return 0.42 * self.wingspan_cm

    @property
    def leg_length(self) -> float:
        # Legs ≈ 0.48 * height (hip-to-ankle), per standard anthropometry.
        return 0.48 * self.height_cm

    @property
    def shoulder_offset(self) -> float:
        # Half shoulder width — distance from COM to each shoulder anchor.
        return 0.10 * self.height_cm

    @property
    def hip_offset(self) -> float:
        # Half hip width.
        return 0.06 * self.height_cm

    @property
    def shoulder_height(self) -> float:
        # Shoulder above pelvis (COM).
        return 0.30 * self.height_cm

    def upper_arm(self) -> float:
        return 0.5 * self.arm_length

    def lower_arm(self) -> float:
        return 0.5 * self.arm_length

    def upper_leg(self) -> float:
        return 0.5 * self.leg_length

    def lower_leg(self) -> float:
        return 0.5 * self.leg_length

    def reach_radius(self, limb: Limb) -> float:
        """Maximum straight-line reach from the corresponding shoulder/hip."""
        if limb in HAND_LIMBS:
            return self.upper_arm() + self.lower_arm()
        return self.upper_leg() + self.lower_leg()

    def anchor_offset(self, limb: Limb) -> np.ndarray:
        """Offset from COM to the limb's body anchor (shoulder or hip)."""
        sx = -self.shoulder_offset if limb == "LH" else self.shoulder_offset
        hx = -self.hip_offset if limb == "LF" else self.hip_offset
        if limb in HAND_LIMBS:
            return np.array([sx, self.shoulder_height])
        return np.array([hx, 0.0])


def solve_2link_ik(
    anchor: np.ndarray,
    target: np.ndarray,
    upper_len: float,
    lower_len: float,
    elbow_up: bool = True,
) -> Optional[np.ndarray]:
    """Closed-form 2D IK for a 2-link chain (upper, lower).

    Returns the joint position (elbow / knee) or None if the target is out
    of reach. Uses the law of cosines, exactly as sketched in CLAUDE.md.

    `elbow_up=True` puts the joint above the anchor→target line — sensible
    default for arms; for legs we typically want elbow_up=False (knee
    bends forward).
    """
    delta = target - anchor
    dist = float(np.linalg.norm(delta))
    if dist > upper_len + lower_len:
        return None  # too far
    if dist < abs(upper_len - lower_len):
        return None  # too close (folds the limb past itself)
    if dist == 0:
        return None

    cos_angle = (upper_len ** 2 + dist ** 2 - lower_len ** 2) / (2 * upper_len * dist)
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    elbow_angle = np.arccos(cos_angle)
    angle_to_target = np.arctan2(delta[1], delta[0])
    sign = 1.0 if elbow_up else -1.0
    theta = angle_to_target + sign * elbow_angle
    return anchor + upper_len * np.array([np.cos(theta), np.sin(theta)])


@dataclass
class Skeleton:
    """A fully-resolved body pose with joint positions for visualization."""

    com: np.ndarray
    shoulders: dict[Limb, np.ndarray]
    hips: dict[Limb, np.ndarray]
    elbows: dict[Limb, np.ndarray]
    knees: dict[Limb, np.ndarray]
    end_effectors: dict[Limb, np.ndarray]


def resolve_skeleton(
    body: BodyModel,
    com: np.ndarray,
    targets: dict[Limb, np.ndarray],
) -> Skeleton:
    """Run IK for each limb. Joints unreachable by IK are placed at the
    midpoint of anchor→target so the visualizer still has something to draw.
    """
    shoulders: dict[Limb, np.ndarray] = {}
    hips: dict[Limb, np.ndarray] = {}
    elbows: dict[Limb, np.ndarray] = {}
    knees: dict[Limb, np.ndarray] = {}

    for limb in HAND_LIMBS:
        anchor = com + body.anchor_offset(limb)
        shoulders[limb] = anchor
        joint = solve_2link_ik(anchor, targets[limb], body.upper_arm(), body.lower_arm(), elbow_up=True)
        elbows[limb] = joint if joint is not None else 0.5 * (anchor + targets[limb])

    for limb in FOOT_LIMBS:
        anchor = com + body.anchor_offset(limb)
        hips[limb] = anchor
        joint = solve_2link_ik(anchor, targets[limb], body.upper_leg(), body.lower_leg(), elbow_up=False)
        knees[limb] = joint if joint is not None else 0.5 * (anchor + targets[limb])

    return Skeleton(
        com=com,
        shoulders=shoulders,
        hips=hips,
        elbows=elbows,
        knees=knees,
        end_effectors=dict(targets),
    )
