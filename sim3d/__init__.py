"""3D climbing simulator built on MuJoCo.

This package is the Phase-4 successor to `physics/` (which is 2D
pymunk). The two coexist: the 2D solver still runs the demo and is
referenced by the README. `sim3d` is where the project moves once the
3D body / hold model proves out.

Coordinate system (world frame, metres):

    +X  →  along the wall, climber's right
    +Y  →  perpendicular to the wall, AWAY from it (toward viewer)
    +Z  →  up

The wall plane sits roughly at Y=0 with its outward normal +Y.
`wall.wall_angle_deg` rotates the wall around the X axis:

    angle = 0   → vertical wall
    angle < 0   → slab (top tilts away, friend to feet)
    angle > 0   → overhang (top tilts toward climber)

Gravity is always world -Z; we tilt the wall, not gravity. This is
the opposite convention from `physics/world.py` (which tilts gravity)
but matches how MuJoCo prefers to see things — geometry stays mobile
in MJCF, gravity is a global.
"""

from sim3d.body import ClimberProfile, Limb, LIMBS, HAND_LIMBS, FOOT_LIMBS
from sim3d.world import Climb3DWorld

__all__ = [
    "Climb3DWorld",
    "ClimberProfile",
    "Limb",
    "LIMBS",
    "HAND_LIMBS",
    "FOOT_LIMBS",
]
