"""MuJoCo XML generation for MoonBoard hold geoms.

Each hold is represented as a sphere of radius RADIUS metres, colour-coded
by role:
  - start holds: green
  - mid holds:   blue
  - end holds:   red

Hold bodies are placed directly in worldbody (no joints) so they are welded
rigidly to the world.  The sphere centre is offset outward from the wall face
by exactly RADIUS metres along the wall surface normal so the sphere sits on
(rather than inside) the wall.
"""

from ..parsers.canonical import Hold
from .wall import COS_A, SIN_A, hold_body_name, hold_position_world

RADIUS: float = 0.04  # metres

# Wall outward normal components (Y and Z; X is always 0).
_NY: float = COS_A   # ≈ 0.766
_NZ: float = -SIN_A  # ≈ −0.643

# Role → RGBA colour string.
_ROLE_RGBA: dict[str, str] = {
    "start": "0 0.8 0 1",
    "mid":   "0.2 0.4 1 1",
    "end":   "0.9 0.1 0.1 1",
}
_DEFAULT_RGBA = "0.7 0.7 0.7 1"


def holds_xml(holds: list[Hold]) -> str:
    """Return MuJoCo XML fragments for a list of holds.

    Each hold becomes a <body>/<geom> pair.  The fragment is meant to be
    inserted inside a <worldbody> element by scene.py.

    Args:
        holds: List of Hold objects from any parser.

    Returns:
        Multi-line XML string — one <body> element per hold.
    """
    lines: list[str] = []
    for h in holds:
        sx, sy, sz = hold_position_world(h.col, h.row)
        # Offset sphere centre outward from the wall face.
        hx = sx
        hy = sy + RADIUS * _NY
        hz = sz + RADIUS * _NZ
        rgba = _ROLE_RGBA.get(h.role, _DEFAULT_RGBA)
        col_letter = chr(ord("A") + h.col)
        body_name = hold_body_name(h.col, h.row)
        lines.append(
            f'    <body name="{body_name}" pos="{hx:.4f} {hy:.4f} {hz:.4f}">\n'
            f'      <geom type="sphere" size="{RADIUS}" rgba="{rgba}"'
            f' contype="0" conaffinity="0"'
            f' name="geom_{body_name}"/>\n'
            f'      <!-- {col_letter}{h.row} ({h.role}) -->\n'
            f'    </body>'
        )
    return "\n".join(lines)
