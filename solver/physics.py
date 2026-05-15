"""2D physics for the climbing body — Pymunk "COM on strings" model.

Model
-----
The climber's entire mass is concentrated at the pelvis/COM — a single
Pymunk dynamic body. Each gripped hold becomes a PinJoint anchor from
that body to a static point in the world. Gravity pulls the COM down;
the grip joints resist.

Why this model?
  • Captures the most important climbing physics: will this configuration
    support the climber's weight, and how much does the body swing?
  • The IK solver in body.py already gives exact joint positions for
    visualisation and reachability checks — we don't need Pymunk to
    recompute those.
  • A full articulated-body simulation (9 rigid bodies, 8 joints) is
    numerically fragile and slow to train on. The COM model trains in
    the same loop as the RL env without blowing up.

Coordinates
-----------
All units are centimetres. y increases upward (same as world-space in
wall.py). Gravity = (0, −980) cm s⁻².

Usage
-----
    phys = PhysicsClimber(body_model, start_com=(x, y))
    phys.grip("LH", (hold_x, hold_y))
    phys.grip("RF", (hold_x2, hold_y2))
    phys.step(30)
    stable = not phys.is_fallen(floor_y=0)
    cx, cy = phys.com_pos()
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pymunk

from solver.body import BodyModel, LIMBS, Limb

# Gravity in cm s⁻².
GRAVITY_Y = -980.0
# Space damping: 1.0 = no damping, 0.0 = instant stop. 0.85 kills oscillation.
SPACE_DAMPING = 0.85
# Climber body mass in grams (70 kg).
CLIMBER_MASS_G = 70_000.0
# Radius of the COM collision shape (cm). Small — just a point mass.
COM_RADIUS_CM = 5.0
# Distance threshold (cm) beyond which the climber is considered "fallen".
FALL_THRESHOLD_CM = 30.0
# Default physics sub-steps per env step.
DEFAULT_SUBSTEPS = 40
DEFAULT_DT = 1 / 60.0


class PhysicsClimber:
    """Pymunk COM-on-strings physics model of a climbing body.

    Parameters
    ----------
    body_model : BodyModel
        Anthropometric params — used to compute anchor offsets for
        grip-strength estimation (not yet used, reserved for Phase 4).
    start_com : (x_cm, y_cm)
        Initial pelvis/COM world position.
    gravity_y : float
        Gravity acceleration (cm s⁻², negative = downward).
    """

    def __init__(
        self,
        body_model: BodyModel,
        start_com: tuple[float, float],
        gravity_y: float = GRAVITY_Y,
    ) -> None:
        self.body_model = body_model

        self.space = pymunk.Space()
        self.space.gravity = (0.0, gravity_y)
        self.space.damping = SPACE_DAMPING

        # ── COM body ──────────────────────────────────────────────────────
        moment = pymunk.moment_for_circle(CLIMBER_MASS_G, 0, COM_RADIUS_CM)
        self._com = pymunk.Body(CLIMBER_MASS_G, moment)
        self._com.position = pymunk.Vec2d(*start_com)
        com_shape = pymunk.Circle(self._com, COM_RADIUS_CM)
        com_shape.filter = pymunk.ShapeFilter(mask=0)  # no collisions with anything
        self.space.add(self._com, com_shape)

        # ── Per-limb grip state ───────────────────────────────────────────
        # Each entry is (static_anchor_body, PinJoint) or None.
        self._grips: dict[Limb, Optional[tuple[pymunk.Body, pymunk.Constraint]]] = {
            l: None for l in LIMBS
        }

    # ── Grip management ───────────────────────────────────────────────────

    def grip(self, limb: Limb, hold_pos: tuple[float, float]) -> None:
        """Attach `limb` to the hold at `hold_pos` (creates a PinJoint)."""
        self.release(limb)

        anchor = pymunk.Body(body_type=pymunk.Body.STATIC)
        anchor.position = pymunk.Vec2d(*hold_pos)
        # No shape needed — static body is purely a constraint anchor.
        self.space.add(anchor)

        joint = pymunk.PinJoint(self._com, anchor)
        self.space.add(joint)

        self._grips[limb] = (anchor, joint)

    def release(self, limb: Limb) -> None:
        """Detach `limb` from its current hold (removes the PinJoint)."""
        if self._grips[limb] is None:
            return
        anchor, joint = self._grips[limb]
        self.space.remove(joint, anchor)
        self._grips[limb] = None

    def release_all(self) -> None:
        for limb in LIMBS:
            self.release(limb)

    def gripped_limbs(self) -> list[Limb]:
        return [l for l in LIMBS if self._grips[l] is not None]

    # ── Physics step ──────────────────────────────────────────────────────

    def step(self, n: int = DEFAULT_SUBSTEPS, dt: float = DEFAULT_DT) -> None:
        """Advance the simulation by `n` sub-steps of `dt` seconds each."""
        for _ in range(n):
            self.space.step(dt)

    # ── State queries ─────────────────────────────────────────────────────

    def com_pos(self) -> tuple[float, float]:
        p = self._com.position
        return float(p.x), float(p.y)

    def com_vel(self) -> tuple[float, float]:
        v = self._com.velocity
        return float(v.x), float(v.y)

    def is_fallen(self, floor_y: float = 0.0) -> bool:
        """True if the COM has dropped near or below the floor — i.e. the
        climber has come off the wall."""
        return self._com.position.y < floor_y + FALL_THRESHOLD_CM

    def displacement_from(self, ref_pos: tuple[float, float]) -> float:
        """Euclidean distance (cm) COM has moved from a reference position."""
        dx = self._com.position.x - ref_pos[0]
        dy = self._com.position.y - ref_pos[1]
        return float(np.hypot(dx, dy))

    # ── State management ─────────────────────────────────────────────────

    def reset(
        self,
        com_pos: tuple[float, float],
        grips: Optional[dict[Limb, tuple[float, float]]] = None,
    ) -> None:
        """Reset the simulation to a new pose.

        Parameters
        ----------
        com_pos : (x, y)
            New COM position.
        grips : dict mapping limb → hold_pos, or None
            Which holds to grip immediately after reset.
        """
        self.release_all()
        self._com.position = pymunk.Vec2d(*com_pos)
        self._com.velocity = pymunk.Vec2d(0, 0)
        self._com.angular_velocity = 0.0
        self._com.angle = 0.0

        if grips:
            for limb, pos in grips.items():
                self.grip(limb, pos)

    def settle(
        self,
        n: int = DEFAULT_SUBSTEPS,
        dt: float = DEFAULT_DT,
        stability_threshold_cm: float = 5.0,
    ) -> bool:
        """Step the simulation and return True if the COM is stable (small
        displacement) at the end — i.e. the current grip configuration can
        support the climber's weight."""
        before = self.com_pos()
        self.step(n, dt)
        return self.displacement_from(before) <= stability_threshold_cm
