"""Canonical data structures and utilities shared across all MoonBoard data format parsers."""

from dataclasses import dataclass, field


@dataclass
class Hold:
    """A single hold on the MoonBoard wall.

    Args:
        hold_id: Unique identifier string for this hold within its route.
        col: 0-indexed column number (A=0, B=1, ..., K=10).
        row: 1-indexed row number (1 at bottom, 18 at top).
        role: Role of this hold — "start", "mid", or "end".
        hold_type: Physical hold type (e.g. "jug", "crimp"). Defaults to "unknown".
    """

    hold_id: str
    col: int
    row: int
    role: str
    hold_type: str = "unknown"


@dataclass
class Route:
    """A climbing problem (route) on the MoonBoard.

    Args:
        route_id: Unique identifier string for this route.
        name: Human-readable route name.
        grade_v: V-scale difficulty grade (e.g. 4 for V4).
        holds: Ordered list of Hold objects making up this route.
        repeats: Number of times this route has been logged as completed.
    """

    route_id: str
    name: str
    grade_v: int
    holds: list = field(default_factory=list)
    repeats: int = 0


def col_letter_to_int(letter: str) -> int:
    """Convert a MoonBoard column letter to a 0-indexed integer.

    Args:
        letter: Single uppercase letter A through K.

    Returns:
        0-indexed column index (A=0, B=1, ..., K=10).

    Raises:
        ValueError: If the letter is not in the range A–K.
    """
    letter = letter.strip().upper()
    idx = ord(letter) - ord("A")
    if idx < 0 or idx > 10:
        raise ValueError(f"Column letter '{letter}' is out of MoonBoard range A–K")
    return idx


# Mapping from Font grade strings to approximate V-grades.
_FONT_TO_V: dict[str, int] = {
    "6A": 3,
    "6A+": 3,
    "6B": 4,
    "6B+": 4,
    "6C": 5,
    "6C+": 5,
    "7A": 6,
    "7A+": 7,
    "7B": 8,
    "7B+": 9,
    "7C": 10,
    "7C+": 11,
    "8A": 11,
    "8A+": 12,
    "8B": 13,
    "8B+": 14,
    "8C": 15,
    "8C+": 16,
}


def font_to_v_grade(font: str) -> int:
    """Convert a Font (French) grade string to an approximate V-grade integer.

    Mapping used:
        6A/6A+ → V3, 6B/6B+ → V4, 6C/6C+ → V5,
        7A → V6, 7A+ → V7, 7B → V8, 7B+ → V9,
        7C/7C+ → V10/V11, 8A → V11, 8A+ → V12,
        8B → V13, 8B+ → V14, 8C → V15, 8C+ → V16.

    Args:
        font: Font grade string such as "6C", "7A+", "8B+".
              Leading/trailing whitespace is stripped.

    Returns:
        Approximate V-grade integer. Returns 0 if the grade is unrecognised.
    """
    cleaned = font.strip().upper()
    if cleaned in _FONT_TO_V:
        return _FONT_TO_V[cleaned]
    # Try without trailing '+' as a fallback
    base = cleaned.rstrip("+")
    if base in _FONT_TO_V:
        return _FONT_TO_V[base]
    return 0
