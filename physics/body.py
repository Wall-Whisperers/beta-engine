"""2D climber body for pymunk — rigid torso + leashed limbs.

After several iterations on a fully articulated model, we settled on
the simplest configuration that gives stable, meaningful physics:

    • The torso is a single rigid pymunk body. It carries the climber's
      mass, has a moment of inertia, responds to gravity, and can rotate.

    • Each "limb on a hold" is modelled as a `SlideJoint` between the
      torso's body-local shoulder/hip anchor and a static hold body.
      The joint is a leash: distance can range from 0 to `limb_length`
      (so the climber can reach in or fully extend) but can't exceed
      `limb_length` (the arm/leg can't stretch).

    • Grip strength is encoded as `joint.max_force`. If the body needs
      more force than that to stay leashed, the joint slides — the
      climber visibly slips off the hold.

    • The articulated stick figure (upper/lower arm, elbow, etc.) is
      computed *kinematically* via the closed-form IK in
      `solver.body` whenever someone reads it. It is visualisation
      only; it never feeds back into the physics. Limb segments don't
      have their own mass in this model — a deliberate simplification
      that gives the body stable, predictable behaviour without the
      tuning hell of a passive-muscle ragdoll.

Why not full articulated? A multi-segment chain of pivot joints folds
under gravity unless every joint has a reasonable rest-angle stiffness.
Tuning that gets you either oscillation (too stiff) or sag (too soft);
neither is useful for an RL training signal. The leash model side-steps
the whole problem and still answers the only physical question that
matters for early RL: "given which holds are attached, can the body
balance, and how hard is each limb working?"

Units: SI (metres, kg, seconds, newtons). Public API takes/returns
positions in metres; the world layer converts to/from cm at the
boundary.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pymunk

from physics import config as cfg
from solver.body import BodyModel, LIMB_POLES, solve_2link_ik_pole

Limb = Literal["LH", "RH", "LF", "RF"]
LIMBS: tuple[Limb, ...] = ("LH", "RH", "LF", "RF")
HAND_LIMBS: tuple[Limb, ...] = ("LH", "RH")
FOOT_LIMBS: tuple[Limb, ...] = ("LF", "RF")


@dataclass
class ClimberProfile:
    """Anthropometric + strength profile for a single climber.

    `body` drives where the body anchors and limb visualization sit;
    the strength fields (`grip_force_n`, `foot_push_force_n`) drive how
    much load each leash can take before slipping.
    """

    body: BodyModel = field(default_factory=BodyModel)
    mass_kg: float = cfg.DEFAULT_MASS_KG
    grip_force_n: float = cfg.DEFAULT_GRIP_FORCE_N
    foot_push_force_n: float = cfg.DEFAULT_FOOT_PUSH_FORCE_N
    friction_hand: float = cfg.DEFAULT_FRICTION_HAND
    friction_foot: float = cfg.DEFAULT_FRICTION_FOOT


class PhysicsBody:
    """Rigid torso + four optional limb leashes inside a pymunk Space."""

    def __init__(
        self,
        space: pymunk.Space,
        profile: ClimberProfile,
        com_position_m: tuple[float, float],
        collision_group: int = 1,
    ) -> None:
        self.space = space
        self.profile = profile
        self._collision_group = collision_group
        self._attachments: dict[Limb, pymunk.SlideJoint] = {}

        bm = profile.body

        # ─── The torso ────────────────────────────────────────────────────
        # Modelled as a rectangle covering hips → shoulders. Body-local
        # frame: y goes up, x is body-right. Origin sits at the COM
        # (≈ pelvis), as in solver/body.py.
        hip_y_m = -0.5 * bm.shoulder_height / cfg.CM_PER_M
        shoulder_y_m = +0.5 * bm.shoulder_height / cfg.CM_PER_M
        torso_height_m = bm.shoulder_height / cfg.CM_PER_M
        torso_width_m = max(bm.shoulder_offset, bm.hip_offset) * 2.0 / cfg.CM_PER_M

        moment = pymunk.moment_for_box(profile.mass_kg, (torso_width_m, torso_height_m))
        self.torso = pymunk.Body(profile.mass_kg, moment)
        self.torso.position = (float(com_position_m[0]), float(com_position_m[1]))

        # Visual + collision shape (sensor=True means it doesn't actually
        # collide with anything — climbing physics in 2D doesn't need
        # body-on-wall collision yet).
        torso_shape = pymunk.Poly.create_box(self.torso, (torso_width_m, torso_height_m))
        torso_shape.sensor = True
        torso_shape.filter = pymunk.ShapeFilter(group=collision_group)
        space.add(self.torso, torso_shape)
        self._torso_shape = torso_shape

        # "Core tension": a passive rotary spring against the world's
        # static body keeps the torso upright. Without it, gravity + a
        # single attached limb would flip the climber. Stiffness is
        # tuned so the body stays roughly vertical but can lean — real
        # climbers angle their torso at the wall.
        upright_spring = pymunk.DampedRotarySpring(
            space.static_body, self.torso,
            rest_angle=0.0,
            stiffness=cfg.TORSO_UPRIGHT_STIFFNESS,
            damping=cfg.TORSO_UPRIGHT_DAMPING,
        )
        upright_spring.collide_bodies = False
        space.add(upright_spring)
        self._upright_spring = upright_spring

        # Body-local anchors for each limb. The torso pivots around the
        # COM, so as the body rotates these anchors swing too.
        self._anchor_local: dict[Limb, tuple[float, float]] = {
            "LH": (-bm.shoulder_offset / cfg.CM_PER_M, shoulder_y_m),
            "RH": (+bm.shoulder_offset / cfg.CM_PER_M, shoulder_y_m),
            "LF": (-bm.hip_offset / cfg.CM_PER_M, hip_y_m),
            "RF": (+bm.hip_offset / cfg.CM_PER_M, hip_y_m),
        }

        # Per-limb maximum reach (m). Used as the SlideJoint's max length.
        self._max_reach: dict[Limb, float] = {
            "LH": bm.arm_length / cfg.CM_PER_M,
            "RH": bm.arm_length / cfg.CM_PER_M,
            "LF": bm.leg_length / cfg.CM_PER_M,
            "RF": bm.leg_length / cfg.CM_PER_M,
        }

    # ─── Public API ───────────────────────────────────────────────────────

    def shoulder_or_hip(self, limb: Limb) -> np.ndarray:
        """World-space position (in metres) of the limb's body anchor."""
        local = self._anchor_local[limb]
        return np.array(self.torso.local_to_world(local))

    def end_effector(self, limb: Limb) -> np.ndarray:
        """World-space position (in metres) of the limb tip.

        For attached limbs this is the hold's position (joint.b is the
        static hold body; anchor_b is the body-local anchor on it,
        which world_to_local'd to (0, 0) when we built the joint, so
        local_to_world((0,0)) gives the hold position).

        For limbs in flight we default to "limb hangs straight down
        from the body anchor at max reach" — visualisation only. Real
        in-flight motion needs an animated target tracked by the
        controller.
        """
        attached = self._attachments.get(limb)
        if attached is not None:
            return np.array(attached.b.local_to_world(attached.anchor_b))
        anchor = self.shoulder_or_hip(limb)
        return np.array([anchor[0], anchor[1] - self._max_reach[limb]])

    def elbow_or_knee(self, limb: Limb) -> np.ndarray:
        """Kinematic elbow/knee position (m), via 2-link IK from the
        body anchor to the current end-effector. Visual only — never
        feeds into physics."""
        bm = self.profile.body
        anchor = self.shoulder_or_hip(limb)
        target = self.end_effector(limb)
        is_arm = limb in HAND_LIMBS
        upper = (bm.upper_arm() if is_arm else bm.upper_leg()) / cfg.CM_PER_M
        lower = (bm.lower_arm() if is_arm else bm.lower_leg()) / cfg.CM_PER_M
        joint = solve_2link_ik_pole(anchor, target, upper, lower, LIMB_POLES[limb])
        if joint is None:
            return 0.5 * (anchor + target)
        return np.array(joint)

    def attach_to_hold(
        self,
        limb: Limb,
        hold_body: pymunk.Body,
        hold_world_m: tuple[float, float],
        max_force_n: float,
    ) -> None:
        """Pin the limb to a hold. Hands and feet attach differently:

        Hands → SlideJoint (a leash). Hands *pull*: they stop the body
            from falling away from the hold, but allow the body to drift
            in toward the hold. Force = how hard the climber is gripping.

        Feet → PinJoint (rigid distance). Feet *push*: the body stands
            on the foot, so the hip-to-foot distance stays whatever it
            was at the moment of attachment (the kinematic seed pose
            picks something sensible). This is the v1 way to capture
            "the leg holds the body up" without modelling muscles.

        For both, `max_force` caps the constraint impulse — exceed it
        and the joint slides, which is the simulator's way of saying
        "the climber slipped off this hold".
        """
        if limb in self._attachments:
            self.release(limb)

        anchor_local = self._anchor_local[limb]

        max_len = self._max_reach[limb] * 0.999
        joint = pymunk.SlideJoint(
            self.torso, hold_body,
            anchor_local,
            _local(hold_body, np.array(hold_world_m)),
            0.0,            # min distance — climber can pull right up to the hold
            max_len,        # max distance — limb can't stretch past its bone length
        )
        joint.max_force = float(max_force_n)
        self.space.add(joint)
        self._attachments[limb] = joint

    def release(self, limb: Limb) -> None:
        joint = self._attachments.pop(limb, None)
        if joint is not None and joint in self.space.constraints:
            self.space.remove(joint)

    def is_attached(self, limb: Limb) -> bool:
        return limb in self._attachments

    def attached_holds(self) -> dict[Limb, pymunk.SlideJoint]:
        return dict(self._attachments)

    def per_limb_force(self, limb: Limb) -> Optional[float]:
        """Force currently in the leash for that limb, in newtons.
        Returns None if the limb is in flight."""
        joint = self._attachments.get(limb)
        if joint is None:
            return None
        return float(np.linalg.norm(joint.impulse)) / cfg.PHYS_DT

    def per_limb_force_fraction(self, limb: Limb) -> Optional[float]:
        """Force on the limb as a fraction of its max-force budget.
        ≥1.0 means the limb is at or past its slip threshold."""
        joint = self._attachments.get(limb)
        if joint is None or joint.max_force <= 0:
            return None
        force = float(np.linalg.norm(joint.impulse)) / cfg.PHYS_DT
        return force / joint.max_force


def _local(body: pymunk.Body, world_pt: np.ndarray) -> tuple[float, float]:
    """Convert a world-space point into a pymunk Body's local frame."""
    p = body.world_to_local((float(world_pt[0]), float(world_pt[1])))
    return (float(p[0]), float(p[1]))
