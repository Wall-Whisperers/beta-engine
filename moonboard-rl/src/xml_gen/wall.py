"""MuJoCo XML generation for the MoonBoard wall geometry.

The MoonBoard is an 11-column × 18-row wall with holds spaced 20 cm apart.
It overhangs at 40° from vertical — the top of the wall is closer to the
climber than the bottom.

Coordinate system (world frame):
  X: horizontal, perpendicular to the climbing direction (A-col = left, K-col = right).
  Y: horizontal, pointing from the wall toward the climber (positive = toward climber).
  Z: vertical, up.

As you travel up the wall surface by arc-length s (metres):
  Δy = +s · sin(40°)   ← top is closer to the climber
  Δz = +s · cos(40°)   ← top is higher

Wall surface outward normal (pointing toward climber):
  n = (0, cos(40°), −sin(40°)) ≈ (0, 0.766, −0.643)
"""

import math

# ── Physical constants ────────────────────────────────────────────────────────
SPACING: float = 0.20       # metres between adjacent holds
NUM_COLS: int = 11          # columns A–K
NUM_ROWS: int = 18          # rows 1–18
OVERHANG_DEG: float = 40.0  # degrees from vertical

# Bottom-left hold (A1) reference position on the wall face.
Y_BASE: float = -1.5   # wall-face y-coordinate at row 1 (negative = behind climber)
Z_BASE: float = 0.30   # floor-to-row-1 height in metres

# Precomputed trig values.
_ANGLE_RAD = math.radians(OVERHANG_DEG)
SIN_A: float = math.sin(_ANGLE_RAD)   # ≈ 0.6428
COS_A: float = math.cos(_ANGLE_RAD)   # ≈ 0.7660

# Wall outward normal (toward climber).
WALL_NORMAL = (0.0, COS_A, -SIN_A)

# Wall box half-extents in the box's LOCAL frame (before rotation):
#   x-half: (10 columns × 0.20 m / 2) + 0.2 m margin
#   y-half: 0.1 m wall thickness
#   z-half: (17 rows × 0.20 m / 2) + 0.3 m margin  (along the wall surface)
_WALL_HALF_X = 1.2
_WALL_HALF_Y = 0.10
_WALL_HALF_Z = 2.0


def hold_position_world(col: int, row: int) -> tuple[float, float, float]:
    """Return the 3-D world position of a hold's surface point.

    The returned coordinate lies exactly on the wall face (no outward radius
    offset).  Callers that need the centre of a sphere sitting on the wall
    must add RADIUS × WALL_NORMAL.

    Args:
        col: 0-indexed column number (0 = column A, 10 = column K).
        row: 1-indexed row number (1 = bottom, 18 = top).

    Returns:
        Tuple (x, y, z) in world-frame metres.
    """
    x = (col - 5) * SPACING           # centred at column F (index 5)
    s = (row - 1) * SPACING            # arc-length up the wall from row 1
    y = Y_BASE + s * SIN_A
    z = Z_BASE + s * COS_A
    return x, y, z


def _wall_center_world() -> tuple[float, float, float]:
    """Compute the geometric centre of the wall surface in world coordinates."""
    # Mid-point arc-length: 8.5 row-spacings from row 1.
    s_mid = 8.5 * SPACING
    x_c = 0.0
    y_c = Y_BASE + s_mid * SIN_A
    z_c = Z_BASE + s_mid * COS_A
    # Shift inward (away from climber) by the half-thickness so the box centre
    # sits behind the surface rather than on it.
    y_c -= _WALL_HALF_Y * COS_A
    z_c += _WALL_HALF_Y * SIN_A
    return x_c, y_c, z_c


def wall_xml() -> str:
    """Return a MuJoCo XML fragment containing the tilted wall box and floor plane.

    The fragment contains two <geom> elements intended to be placed directly
    inside <worldbody>.  They are NOT wrapped in any parent tag so that
    scene.py can embed them into a larger XML structure.

    Returns:
        Multi-line XML string with wall and floor geoms.
    """
    cx, cy, cz = _wall_center_world()

    lines = [
        # Floor plane — matches humanoid.xml material names so textures render.
        (
            f'    <geom name="floor" type="plane" condim="3" friction="1 .1 .1"'
            f' material="MatPlane" pos="0 0 0" size="20 20 0.125" rgba="0.8 0.9 0.8 1"/>'
        ),
        # Wall box — rotated −40° about X so the surface normal faces the climber.
        (
            f'    <geom name="moonboard_wall" type="box"'
            f' pos="{cx:.4f} {cy:.4f} {cz:.4f}"'
            f' euler="{-OVERHANG_DEG:.1f} 0 0"'
            f' size="{_WALL_HALF_X} {_WALL_HALF_Y} {_WALL_HALF_Z}"'
            f' rgba="0.55 0.45 0.35 1" contype="1" conaffinity="1"/>'
        ),
    ]
    return "\n".join(lines)
