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

import numpy as np

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

# Wall outward normal (toward climber) as a numpy unit vector.
WALL_NORMAL: np.ndarray = np.array([0.0, COS_A, -SIN_A])

# Wall box half-extents in the box's LOCAL frame (before rotation):
#   x-half: (10 columns × 0.20 m / 2) + 0.2 m margin
#   y-half: 0.1 m wall thickness
#   z-half: (17 rows × 0.20 m / 2) + 0.3 m margin  (along the wall surface)
_WALL_HALF_X = 1.2
_WALL_HALF_Y = 0.10
_WALL_HALF_Z = 2.0

# ── Kickboard physical constants ──────────────────────────────────────────────
# Source: standard MoonBoard 2016 spec — kickboard is a vertical panel below
# the 40-degree main board.

KB_HEIGHT: float = 0.370   # vertical panel height, metres (physical: 370 mm)
KB_WIDTH:  float = 2.440   # same as main board width, metres (physical: 2440 mm)
KB_DEPTH:  float = 0.018   # plywood thickness, metres (physical: 18 mm)

# Y position of the kickboard face (vertical panel, facing +Y toward climber).
# Derived by finding where the main board face is at z = KB_HEIGHT (the junction
# between the kickboard top and main board bottom):
#   s_jct = (KB_HEIGHT - Z_BASE) / COS_A = (0.370 - 0.30) / 0.766 ≈ 0.0913 m
#   y_jct = Y_BASE + s_jct * SIN_A = -1.5 + 0.0913 * 0.643 ≈ -1.4413 m
_KB_FACE_Y: float = Y_BASE + ((KB_HEIGHT - Z_BASE) / COS_A) * SIN_A   # ≈ -1.4413 m

# Kickboard box centre in world frame (vertical panel centred between floor and KB_HEIGHT).
_KB_CENTER_X: float = 0.0
_KB_CENTER_Y: float = _KB_FACE_Y - KB_DEPTH / 2   # ≈ -1.4503 m (behind face)
_KB_CENTER_Z: float = KB_HEIGHT / 2               # ≈ 0.1850 m

# Kickboard outward normal (facing +Y toward climber; panel is vertical).
KB_NORMAL: np.ndarray = np.array([0.0, 1.0, 0.0])

# ── Kickboard hold constants ──────────────────────────────────────────────────
# Four permanent footholds on the kickboard, independent of route.
# Positions derived from 25% / 75% width fractions and two height rows.

_KB_HOLD_RADIUS: float = 0.040   # metres (same as main wall holds)

# X positions: 40% and 60% of board width, centred at x=0.
# Physical spec is 25%/75% (±0.61 m), but the MuJoCo humanoid's hip joint
# limits (hip_x max abduction 5°, hip_z max rotation 60°) make ±0.61 m
# unreachable.  40%/60% = ±0.244 m is within the kinematic reach of the
# humanoid's hip_z-driven lateral positioning during IK.
_KB_LEFT_X:  float = -KB_WIDTH / 2 + KB_WIDTH * 0.40   # ≈ -0.244 m
_KB_RIGHT_X: float = -KB_WIDTH / 2 + KB_WIDTH * 0.60   # ≈ +0.244 m

# Z positions: lower row at 100 mm, upper row at 270 mm above floor.
_KB_LOWER_Z: float = 0.100   # metres
_KB_UPPER_Z: float = 0.270   # metres

# Y position of kickboard hold centres (face + radius, offset toward climber).
_KB_HOLD_Y: float = _KB_FACE_Y + _KB_HOLD_RADIUS   # ≈ -1.4013 m

# Canonical body names for kickboard footholds (4 total).
KICKBOARD_HOLD_NAMES: list[str] = [
    "kb_hold_KB_LL",   # left  lower
    "kb_hold_KB_LU",   # left  upper
    "kb_hold_KB_RL",   # right lower
    "kb_hold_KB_RU",   # right upper
]


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


def kickboard_hold_positions_world() -> dict[str, np.ndarray]:
    """Return world-frame XYZ centres of all four kickboard hold grip sites.

    Positions include the sphere radius offset from the kickboard face so
    the returned coordinates match the hold sphere centre (not the wall surface).

    Returns:
        Dict mapping each name in KICKBOARD_HOLD_NAMES to a length-3 float64
        array in metres: [x, y, z] in world frame.
    """
    return {
        "kb_hold_KB_LL": np.array([_KB_LEFT_X,  _KB_HOLD_Y, _KB_LOWER_Z], dtype=np.float64),
        "kb_hold_KB_LU": np.array([_KB_LEFT_X,  _KB_HOLD_Y, _KB_UPPER_Z], dtype=np.float64),
        "kb_hold_KB_RL": np.array([_KB_RIGHT_X, _KB_HOLD_Y, _KB_LOWER_Z], dtype=np.float64),
        "kb_hold_KB_RU": np.array([_KB_RIGHT_X, _KB_HOLD_Y, _KB_UPPER_Z], dtype=np.float64),
    }


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


def kickboard_geom_xml() -> str:
    """Return a MuJoCo XML fragment for the vertical kickboard panel below the main wall.

    The kickboard is a vertical box geom (no X rotation), with its face coplanar
    with the main board face at z = KB_HEIGHT (the physical junction point).
    Its bottom edge sits on the floor (z = 0).

    Returns:
        Multi-line XML string containing a <body> with the kickboard box geom
        and a debugging <site> at the face centre. Units: metres.
    """
    lines = [
        f'    <body name="kickboard" pos="0 0 0">',
        f'      <!-- Vertical kickboard panel: {KB_WIDTH*1000:.0f}mm wide × {KB_HEIGHT*1000:.0f}mm tall × {KB_DEPTH*1000:.0f}mm deep -->',
        f'      <geom name="kickboard_panel" type="box"'
        f' pos="{_KB_CENTER_X:.4f} {_KB_CENTER_Y:.4f} {_KB_CENTER_Z:.4f}"'
        f' size="{KB_WIDTH/2:.4f} {KB_DEPTH/2:.4f} {KB_HEIGHT/2:.4f}"'
        f' rgba="0.25 0.25 0.28 1"'
        f' friction="1.5 0.05 0.001"'
        f' contype="1" conaffinity="1"/>',
        f'      <site name="kickboard_face_center"'
        f' pos="{_KB_CENTER_X:.4f} {_KB_FACE_Y:.4f} {_KB_CENTER_Z:.4f}"'
        f' size="0.01" rgba="1 0 0 1"/>',
        f'    </body>',
    ]
    return "\n".join(lines)


def kickboard_holds_xml() -> str:
    """Return MuJoCo XML fragments for the four permanent kickboard footholds.

    Each hold has:
      - A visual sphere geom (radius 40 mm, orange).
      - A grip-volume sphere geom (radius 60 mm, semi-transparent blue,
        contype/conaffinity=2 for separate contact group).
      - A canonical grip site for GripManager proximity checks.

    The kickboard surface normal is (0, 1, 0) — horizontal, toward the climber.
    Hold positions are at _KB_HOLD_Y (kickboard face + radius offset).

    Returns:
        Multi-line XML string — one <body> element per kickboard hold. Units: metres.
    """
    holds = [
        ("KB_LL", _KB_LEFT_X,  _KB_LOWER_Z),
        ("KB_LU", _KB_LEFT_X,  _KB_UPPER_Z),
        ("KB_RL", _KB_RIGHT_X, _KB_LOWER_Z),
        ("KB_RU", _KB_RIGHT_X, _KB_UPPER_Z),
    ]
    lines: list[str] = []
    for short, hx, hz in holds:
        body_name = f"kb_hold_{short}"
        hy = _KB_HOLD_Y
        lines.append(
            f'    <body name="{body_name}" pos="{hx:.4f} {hy:.4f} {hz:.4f}">\n'
            f'      <!-- Kickboard foothold {short}: visual + grip volume + site -->\n'
            f'      <geom name="{body_name}_vis" type="sphere" size="{_KB_HOLD_RADIUS:.3f}"'
            f' rgba="0.85 0.35 0.05 1" friction="1.8 0.05 0.001"'
            f' contype="0" conaffinity="0"/>\n'
            f'      <geom name="{body_name}_grip" type="sphere" size="0.060"'
            f' rgba="0 0 1 0.15" contype="2" conaffinity="2"'
            f' friction="1.8 0.05 0.001"/>\n'
            f'      <site name="{body_name}_site" size="0.010" rgba="1 1 0 1"/>\n'
            f'    </body>'
        )
    return "\n".join(lines)


def wall_xml() -> str:
    """Return a MuJoCo XML fragment containing the tilted wall box, floor plane, and kickboard.

    The fragment contains <geom> elements and one <body> intended to be placed
    directly inside <worldbody>.  They are NOT wrapped in any parent tag so that
    scene.py can embed them into a larger XML structure.

    Returns:
        Multi-line XML string with wall, floor, and kickboard geoms.
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
        # Kickboard — vertical panel below the main board.
        kickboard_geom_xml(),
    ]
    return "\n".join(lines)


def hold_body_name(col: int, row: int) -> str:
    """Return the deterministic MuJoCo body name for a hold at (col, row).

    The name format is ``hold_{col}_{row}`` where col is 0-indexed (0=A, 10=K)
    and row is 1-indexed (1–18).  This matches the names emitted by holds_xml()
    and is used by GripManager to look up body indices via mj_name2id at runtime.

    Args:
        col: 0-indexed column number (0 = column A, 10 = column K).
        row: 1-indexed row number (1 = bottom, 18 = top).

    Returns:
        Body name string, e.g. ``"hold_5_8"`` for column F, row 8.
    """
    return f"hold_{col}_{row}"
