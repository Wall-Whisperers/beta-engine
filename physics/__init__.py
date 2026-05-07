"""Pymunk-backed 2D physics for the Beta Engine climber.

This package replaces the static, kinematic-only solver with an
articulated rigid-body simulation:

    physics.world.ClimbWorld   — wraps a pymunk.Space, owns the wall
                                 + holds + gravity (rotated by wall angle)
    physics.body.PhysicsBody   — torso + 4 two-link limbs, joints,
                                 grip-limited hold attachments
    physics.render             — matplotlib renderer for the live state
                                 (headless-safe, no pygame needed)

The wall + hold loaders live in `solver/wall.py` and are re-used here —
this package is *additive*; the existing solver keeps working.

Units inside the physics layer are SI (metres, kg, seconds, newtons).
The wall JSON is in centimetres, so positions are converted at the
boundary. Render still works in cm to match `solver/visualize.py`.
"""

from physics.body import PhysicsBody, ClimberProfile
from physics.world import ClimbWorld

__all__ = ["ClimbWorld", "PhysicsBody", "ClimberProfile"]
