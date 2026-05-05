"""2D body model + closed-form IK for a stick-figure climber.

The model is the 5-point figure described in CLAUDE.md / planning-gabe.md:

    - 1 Center Of Mass (COM, treated as the climber's pelvis/torso anchor)
    - 4 end-effectors (left/right hand, left/right foot)
    - 2 limbs per side, each a 2-link chain (shoulder→elbow→wrist or
      hip→knee→ankle), short enough for closed-form IK using the law of
      cosines.

Distances are in cm. Defaults assume the "average climber" from CLAUDE.md
(175 cm height / 175 cm wingspan).

═══════════════════════════════════════════════════════════════════════════
  THIS IS A 2D MODEL — pose, IK, reachability, and stability are all
  evaluated in the wall plane (x = horizontal, y = vertical). There is
  no body twist, no hip rotation, no out-of-plane drop-knee. 3D moves
  (gastons that need shoulder roll, drop-knees that need hip twist) get
  approximated by 2D angle envelopes — the constants below.
═══════════════════════════════════════════════════════════════════════════

Phase 3 of the roadmap will layer a Pymunk physics step on top of this.

────────────────────────────────────────────────────────────────────────
  Joint angle envelopes — anatomically inspired, freely tweakable.
────────────────────────────────────────────────────────────────────────

Each limb has a *valid envelope* in its body-local frame: an axis-aligned
box (centred on the shoulder/hip anchor) inside which the end-effector
must lie. Outside the box, the move is rejected even if the IK formally
solves. The envelope is the cheapest 2D approximation of joint range-of-
motion that captures the rules a coach would actually call out:

  • Hands can reach overhead, sideways, and down to roughly hip level.
    Cross-body grabs (gastons, far crosses) are possible but limited.
  • Feet stay below the shoulders. Most climbers can't put their foot
    above hip level without serious flexibility — so the high-step ceiling
    is small. Drop-knee crossovers are allowed but tightly capped.

If you find the solver is rejecting moves a real climber would pull,
loosen the relevant *_FRAC constant. If it's accepting impossible-looking
moves, tighten them. They are deliberately the only knobs you need.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

Limb = Literal["LH", "RH", "LF", "RF"]
LIMBS: tuple[Limb, ...] = ("LH", "RH", "LF", "RF")
HAND_LIMBS: tuple[Limb, ...] = ("LH", "RH")
FOOT_LIMBS: tuple[Limb, ...] = ("LF", "RF")
LEFT_LIMBS: tuple[Limb, ...] = ("LH", "LF")

# ─── Joint envelope fractions (× arm_length or × leg_length) ──────────────
# Hand envelope: deliberately permissive at the single-limb level.
# Cross-body grabs (cross-overs / cross-unders) come up constantly in
# real climbing, so we let one hand reach well past the body midline.
# The *pose-level* HAND_CROSSOVER_LIMIT_CM below is the firmer guard
# against unrealistic "fully swapped" hand positions.
HAND_MAX_ABOVE_SHOULDER_FRAC = 1.00   # fully overhead
HAND_MAX_BELOW_SHOULDER_FRAC = 0.85   # past hip — mantling, low gastons
HAND_IPSILATERAL_REACH_FRAC = 1.00    # full arm sideways on own side
HAND_CROSS_BODY_FRAC = 0.75           # cross-over moves (LH past midline)

# Foot envelope: tighter than hands. The big anatomical limit is "foot
# above hip" — most climbers cap out around the knee-to-chest range, and
# almost nobody can put a foot above the shoulders. The latter is also
# enforced by `foot_world_ceiling()` below.
FOOT_MAX_ABOVE_HIP_FRAC = 0.30        # high-step ceiling (knee ≈ chest)
FOOT_MAX_BELOW_HIP_FRAC = 1.00        # full leg extension downward
FOOT_IPSILATERAL_REACH_FRAC = 1.00    # full leg sideways on own side
FOOT_CROSS_BODY_FRAC = 0.40           # drop-knee crossover limit

# ─── Pose-level (multi-limb) constraints ──────────────────────────────────
# Hands/feet shouldn't fully swap sides. A *cross* move is fine; *cross
# and keep going* (LH ends up wildly right of RH) isn't.
HAND_CROSSOVER_LIMIT_CM = 40.0        # max LH.x − RH.x when LH is right of RH
FOOT_CROSSOVER_LIMIT_CM = 35.0
# Two end-effectors can't share the same patch of wall (catches accidental
# overlap on small holds; per-hold occupancy is enforced separately).
END_EFFECTOR_MIN_SEPARATION_CM = 8.0
# Hard rule: a foot above the shoulders is anatomically impossible for
# almost everyone, regardless of how loose the envelope frac gets.
FOOT_ABS_CEILING_BELOW_SHOULDER_CM = 5.0


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

    def envelope_box(self, limb: Limb) -> tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) of the valid end-effector
        region in the limb's local frame (origin = body anchor).

        The frame is body-relative: +y points up, +x points to the
        climber's right. For a left-side limb the *cross-body* direction
        is +x; for a right-side limb it's −x.
        """
        is_left = limb in LEFT_LIMBS
        if limb in HAND_LIMBS:
            up = self.arm_length * HAND_MAX_ABOVE_SHOULDER_FRAC
            down = self.arm_length * HAND_MAX_BELOW_SHOULDER_FRAC
            ipsi = self.arm_length * HAND_IPSILATERAL_REACH_FRAC
            cross = self.arm_length * HAND_CROSS_BODY_FRAC
        else:
            up = self.leg_length * FOOT_MAX_ABOVE_HIP_FRAC
            down = self.leg_length * FOOT_MAX_BELOW_HIP_FRAC
            ipsi = self.leg_length * FOOT_IPSILATERAL_REACH_FRAC
            cross = self.leg_length * FOOT_CROSS_BODY_FRAC

        if is_left:
            x_min, x_max = -ipsi, +cross
        else:
            x_min, x_max = -cross, +ipsi
        return x_min, x_max, -down, up

    def in_envelope(self, limb: Limb, anchor: np.ndarray, target: np.ndarray) -> bool:
        """True if `target` is within the limb's anatomical envelope."""
        delta = target - anchor
        x_min, x_max, y_min, y_max = self.envelope_box(limb)
        if not (x_min <= delta[0] <= x_max):
            return False
        if not (y_min <= delta[1] <= y_max):
            return False
        # Hard ceiling: a foot can never be above the climber's shoulders.
        # Implemented in *world* y by the caller (it knows the COM).
        return True

    def foot_world_ceiling(self, com: np.ndarray) -> float:
        """Absolute world-y ceiling for a foot — a foot above the
        shoulders is anatomically impossible regardless of envelope
        tweaks (knee-to-shoulder pose simply doesn't exist for most
        humans). Caller checks `target.y <= foot_world_ceiling(com)`.
        """
        return float(com[1]) + self.shoulder_height - FOOT_ABS_CEILING_BELOW_SHOULDER_CM


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

    `elbow_up=True` puts the joint above the anchor→target line.
    `elbow_up=False` puts it below.
    Callers should use `natural_elbow_up` to pick the right branch.
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


# ─── Pole vectors (world space, un-normalised) ────────────────────────────
# The pole vector defines the preferred direction from the anchor to the
# joint (elbow / knee). The IK solution whose joint-anchor vector aligns
# best with the pole is chosen — the same technique used in Maya, Blender,
# Unity, and Unreal for disambiguating 2-solution IK chains.
#
# Arms: elbow prefers to go outward (away from body midline) and downward.
#   LH → left+down  RH → right+down
# Legs: knee prefers to go outward (away from body midline), roughly level.
#   LF → left       RF → right
LIMB_POLES: dict[str, np.ndarray] = {
    "LH": np.array([-1.0, -1.0]),
    "RH": np.array([ 1.0, -1.0]),
    "LF": np.array([-1.0,  0.0]),
    "RF": np.array([ 1.0,  0.0]),
}


def solve_2link_ik_pole(
    anchor: np.ndarray,
    target: np.ndarray,
    upper_len: float,
    lower_len: float,
    pole: np.ndarray,
) -> Optional[np.ndarray]:
    """2-link IK with pole-vector disambiguation.

    Computes both IK solutions and returns the joint position whose
    direction from the anchor best matches `pole`. This is the standard
    approach used in animation software to control elbow/knee direction.
    """
    j_up = solve_2link_ik(anchor, target, upper_len, lower_len, elbow_up=True)
    j_dn = solve_2link_ik(anchor, target, upper_len, lower_len, elbow_up=False)

    if j_up is None and j_dn is None:
        return None
    if j_up is None:
        return j_dn
    if j_dn is None:
        return j_up

    pole_n = pole / (float(np.linalg.norm(pole)) + 1e-9)
    score_up = float(np.dot(j_up - anchor, pole_n))
    score_dn = float(np.dot(j_dn - anchor, pole_n))
    return j_up if score_up >= score_dn else j_dn


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
        joint = solve_2link_ik_pole(
            anchor, targets[limb], body.upper_arm(), body.lower_arm(), LIMB_POLES[limb]
        )
        elbows[limb] = joint if joint is not None else 0.5 * (anchor + targets[limb])

    for limb in FOOT_LIMBS:
        anchor = com + body.anchor_offset(limb)
        hips[limb] = anchor
        joint = solve_2link_ik_pole(
            anchor, targets[limb], body.upper_leg(), body.lower_leg(), LIMB_POLES[limb]
        )
        knees[limb] = joint if joint is not None else 0.5 * (anchor + targets[limb])

    return Skeleton(
        com=com,
        shoulders=shoulders,
        hips=hips,
        elbows=elbows,
        knees=knees,
        end_effectors=dict(targets),
    )
