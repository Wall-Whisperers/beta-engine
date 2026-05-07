"""ClimbWorld — pymunk Space + wall + holds + climber, wired together.

This is the top-level handle the rest of the codebase (RL env, demo
runner) talks to. All time progression goes through `step()`. All
high-level actions (attach a limb, release a limb, advance the sim by
one move) live here too — the body and joint-level details stay in
`physics.body`.

Coordinate system:
    Inside this class everything is in metres (pymunk's natural unit).
    The Wall passed in is in cm, so positions are converted at the
    boundary. Render code reads positions back in metres and converts
    to cm itself.

Wall angle:
    `wall.wall_angle_deg` rotates gravity in the wall plane. 0 =
    vertical (gravity straight down), positive = overhang (top tilts
    toward climber → gravity pulls the body slightly to one side as
    well as down). For 2D in-plane physics we only project the in-plane
    component; the perpendicular pull-off-the-wall component is a
    Phase-4 / 3D extension.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pymunk

from physics import config as cfg
from physics.body import (
    HAND_LIMBS,
    LIMBS,
    ClimberProfile,
    Limb,
    PhysicsBody,
)
from solver.wall import Hold, Wall


@dataclass
class HoldAnchor:
    """A static pymunk body planted at a hold's centre.

    We need a real body (not a Vec2d) because PivotJoint connects two
    bodies. `space.static_body` would technically work but a per-hold
    static body lets us track impulse on each contact independently.
    """

    hold: Hold
    body: pymunk.Body
    position_m: tuple[float, float]


class ClimbWorld:
    """Wraps a pymunk Space + the wall geometry + a single climber.

    Typical use:

        world = ClimbWorld(wall, ClimberProfile())
        world.seed_pose(lh="h_003", rh="h_004", lf="h_001", rf="h_002")
        for _ in range(60):
            world.step()
        world.move_limb("RH", "h_008")     # try the next move
        for _ in range(60):
            world.step()
    """

    def __init__(
        self,
        wall: Wall,
        profile: Optional[ClimberProfile] = None,
        *,
        gravity_scale: float = 1.0,
    ) -> None:
        self.wall = wall
        self.profile = profile or ClimberProfile()

        # ─── Build pymunk Space ───────────────────────────────────────
        self.space = pymunk.Space()
        self.space.damping = cfg.DEFAULT_DAMPING
        # The body has 9 segments + ~20 joints by the time all four
        # limbs are anchored. Pymunk's default 10 solver iterations
        # leaves visible drift — bump to 30 for tight constraint
        # convergence (cheap, simulation is small).
        self.space.iterations = 30

        # Gravity is in m/s². Wall angle rotates it in-plane: a positive
        # `wall_angle_deg` represents an overhang where the wall top
        # tilts toward the climber, which (in our 2D projection) shows
        # up as a small horizontal pull plus a slight reduction in the
        # vertical component. We rotate gravity *clockwise* by the wall
        # angle (positive overhang → gravity vector tilts, climber feels
        # pulled "out and down" in wall-plane coords).
        theta = math.radians(wall.wall_angle_deg)
        gx = -gravity_scale * cfg.GRAVITY_M_S2 * math.sin(theta)
        gy = -gravity_scale * cfg.GRAVITY_M_S2 * math.cos(theta)
        self.space.gravity = (gx, gy)

        # ─── Hold anchors (one static body per hold) ──────────────────
        self.holds: dict[str, HoldAnchor] = {}
        for h in wall.holds:
            pos_m = (h.x_cm / cfg.CM_PER_M, h.y_cm / cfg.CM_PER_M)
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
            body.position = pos_m
            # Give it a tiny shape so debug renderers can see it (not
            # required for the joint to work). We disable collision via
            # a unique filter so the climber's limbs don't bounce off
            # the holds — contact is mediated by joints, not collision.
            shape = pymunk.Circle(body, 0.04)
            shape.filter = pymunk.ShapeFilter(
                categories=0,  # collide with nothing
                mask=0,
            )
            shape.sensor = True
            self.space.add(body, shape)
            self.holds[h.hold_id] = HoldAnchor(hold=h, body=body, position_m=pos_m)

        # ─── The climber ──────────────────────────────────────────────
        # Initial COM: roughly the middle of the wall, mid-height. The
        # caller is expected to call `seed_pose()` immediately to snap
        # the limbs to the starting holds.
        com_init = (
            wall.width_cm / 2.0 / cfg.CM_PER_M,
            wall.height_cm / 3.0 / cfg.CM_PER_M,
        )
        self.body = PhysicsBody(self.space, self.profile, com_init)

        # Track which hold each limb is on (or None).
        self._on_hold: dict[Limb, Optional[str]] = {l: None for l in LIMBS}

    # ─── Pose seeding ──────────────────────────────────────────────────────

    def seed_pose(
        self,
        *,
        lh: Optional[str] = None,
        rh: Optional[str] = None,
        lf: Optional[str] = None,
        rf: Optional[str] = None,
    ) -> None:
        """Snap the climber so each named limb is on its hold and create
        the corresponding pivot joints. Pass None to leave a limb in
        flight.

        Also re-positions the torso so the geometry is roughly
        consistent: COM goes between the foot midpoint and hand
        midpoint, at hip height above the feet (matches the kinematic
        guess the solver uses)."""
        bm = self.profile.body
        targets: dict[Limb, Optional[str]] = {
            "LH": lh, "RH": rh, "LF": lf, "RF": rf,
        }

        # Estimate COM from the seeded targets so the body starts in a
        # reasonable layout.
        foot_pts = []
        hand_pts = []
        for limb, hid in targets.items():
            if hid is None:
                continue
            anchor = self.holds[hid]
            (foot_pts if limb in ("LF", "RF") else hand_pts).append(
                np.array(anchor.position_m)
            )

        if foot_pts:
            foot_mid = np.mean(foot_pts, axis=0)
            com_y = foot_mid[1] + (bm.leg_length * 0.45) / cfg.CM_PER_M
            com_x = foot_mid[0]
            if hand_pts:
                hand_mid = np.mean(hand_pts, axis=0)
                com_x = 0.8 * foot_mid[0] + 0.2 * hand_mid[0]
            self.body.torso.position = (float(com_x), float(com_y))
        elif hand_pts:
            self.body.torso.position = tuple(np.mean(hand_pts, axis=0))

        self.body.torso.velocity = (0, 0)
        self.body.torso.angular_velocity = 0.0
        self.body.torso.angle = 0.0

        # Attach each limb (snap-then-attach so the joint isn't born
        # under huge stress).
        for limb in LIMBS:
            self.body.release(limb)
            self._on_hold[limb] = None

        for limb, hid in targets.items():
            if hid is None:
                continue
            self._attach(limb, hid)


    def _attach(self, limb: Limb, hold_id: str) -> None:
        """Pin a limb to a hold via a force-limited leash."""
        anchor = self.holds[hold_id]
        max_force = self._max_force_for(limb, anchor.hold)
        self.body.attach_to_hold(
            limb, anchor.body, anchor.position_m, max_force
        )
        self._on_hold[limb] = hold_id

    def _max_force_for(self, limb: Limb, hold: Hold) -> float:
        """Combine climber strength + hold positivity + (optional) per-hold
        cap into a single newton number used as the PivotJoint's
        max_force."""
        base = (
            self.profile.grip_force_n if limb in HAND_LIMBS
            else self.profile.foot_push_force_n
        )
        cap = base * hold.positivity * cfg.ATTACH_MAX_FORCE_SCALE
        if hold.max_force_n is not None:
            cap = min(cap, hold.max_force_n)
        return cap

    # ─── High-level actions ────────────────────────────────────────────────

    def move_limb(
        self,
        limb: Limb,
        target_hold_id: str,
        *,
        mode: str = "snap",
    ) -> None:
        """Move a single limb to a new hold.

        mode="snap": teleport the limb tip to the new hold (instant).
            Useful for fast simulation / RL training.
        mode="reach": release the limb, run physics with no attachment
            for one render frame so the body shifts weight, then
            snap-attach. A first-pass approximation of reaching.
        """
        if target_hold_id not in self.holds:
            raise KeyError(f"Unknown hold: {target_hold_id}")

        if mode == "snap":
            self._attach(limb, target_hold_id)
            return

        if mode == "reach":
            self.body.release(limb)
            self._on_hold[limb] = None
            for _ in range(cfg.SUBSTEPS_PER_FRAME * 2):
                self.space.step(cfg.PHYS_DT)
            self._attach(limb, target_hold_id)
            return

        raise ValueError(f"unknown mode: {mode}")

    def release_limb(self, limb: Limb) -> None:
        self.body.release(limb)
        self._on_hold[limb] = None

    def step(self, frames: int = 1) -> None:
        """Advance the simulation by `frames` render frames (each frame
        is `SUBSTEPS_PER_FRAME` physics sub-steps).

        Before each physics sub-step we apply an "active posture" force
        nudging the torso toward its kinematic ideal COM (computed from
        the currently-attached holds). Without it the body hangs
        passively from its leashes; with it the body holds the upright
        pose a climber actually adopts.
        """
        for _ in range(frames * cfg.SUBSTEPS_PER_FRAME):
            self._apply_posture_force()
            self.space.step(cfg.PHYS_DT)

    def _apply_posture_force(self) -> None:
        """Active "stand-up" force on the torso.

        Equation: F = m·g_compensate + Kp·(target - actual) - Kd·velocity

        The first term cancels gravity exactly so the controller doesn't
        have to fight a steady DC offset (the body lands on the target
        instead of below it). The PD term tracks the kinematic ideal
        COM. Together they emulate the active leg/core work a real
        climber does to hold a static pose.
        """
        target = self._kinematic_target_com_m()
        if target is None:
            return
        torso = self.body.torso
        actual = np.array(torso.position)
        delta = target - actual
        velocity = np.array(torso.velocity)
        # Gravity feed-forward: cancel the world's gravity on the torso
        # so the PD controller only has to track position, not also
        # support body weight.
        gravity = np.array(self.space.gravity)
        gravity_comp = -torso.mass * gravity
        force = (
            gravity_comp
            + cfg.POSTURE_GAIN_N_PER_M * delta
            - cfg.POSTURE_DAMPING * velocity
        )
        torso.apply_force_at_world_point(
            (float(force[0]), float(force[1])),
            torso.position,
        )

    def _kinematic_target_com_m(self) -> Optional[np.ndarray]:
        """Where the torso WANTS to be: the kinematic COM the solver
        would compute from the attached holds. Pelvis sits at hip
        height above the foot midpoint, slightly biased toward the
        hand midpoint."""
        bm = self.profile.body
        foot_pts = []
        hand_pts = []
        for limb in LIMBS:
            hid = self._on_hold[limb]
            if hid is None:
                continue
            anchor = self.holds[hid]
            (foot_pts if limb in ("LF", "RF") else hand_pts).append(
                np.array(anchor.position_m)
            )
        if not foot_pts and not hand_pts:
            return None
        if foot_pts:
            foot_mid = np.mean(foot_pts, axis=0)
            target_y = foot_mid[1] + (bm.leg_length * 0.45) / cfg.CM_PER_M
            target_x = float(foot_mid[0])
            if hand_pts:
                hand_mid = np.mean(hand_pts, axis=0)
                target_x = 0.8 * float(foot_mid[0]) + 0.2 * float(hand_mid[0])
            return np.array([target_x, target_y])
        # No feet → hang from hands.
        hand_mid = np.mean(hand_pts, axis=0)
        return np.array([float(hand_mid[0]),
                         float(hand_mid[1]) - bm.shoulder_height / cfg.CM_PER_M])

    # ─── State queries (used by the renderer + the RL env) ───────────────

    def on_hold(self, limb: Limb) -> Optional[str]:
        return self._on_hold[limb]

    def pose(self) -> dict[Limb, Optional[str]]:
        return dict(self._on_hold)

    def com_m(self) -> np.ndarray:
        return np.array(self.body.torso.position)

    def com_cm(self) -> np.ndarray:
        return self.com_m() * cfg.CM_PER_M

    def end_effector_cm(self, limb: Limb) -> np.ndarray:
        return self.body.end_effector(limb) * cfg.CM_PER_M

    def joint_cm(self, limb: Limb) -> np.ndarray:
        """Elbow / knee position in cm."""
        return self.body.elbow_or_knee(limb) * cfg.CM_PER_M

    def shoulder_or_hip_cm(self, limb: Limb) -> np.ndarray:
        return self.body.shoulder_or_hip(limb) * cfg.CM_PER_M

    def is_stable(self, slip_threshold: float = 0.95) -> bool:
        """Crude: a pose is stable if no attached limb is at >95% of
        its allowed force budget. If any limb is slipping, the climber
        is about to fall."""
        for limb in LIMBS:
            joint = self.body.attached_holds().get(limb)
            if joint is None:
                continue
            current = float(np.linalg.norm(joint.impulse)) / cfg.PHYS_DT
            cap = joint.max_force or 1.0
            if current >= slip_threshold * cap:
                return False
        # Also: COM shouldn't have fallen below the lowest foot by a lot
        # (sanity check that we're still on the wall).
        foot_pts = [self.end_effector_cm(l) for l in ("LF", "RF")
                    if self._on_hold[l] is not None]
        if foot_pts:
            lowest = min(p[1] for p in foot_pts)
            if self.com_cm()[1] < lowest - 30.0:  # 30 cm of slack
                return False
        return True
