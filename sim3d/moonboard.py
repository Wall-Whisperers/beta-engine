"""MoonBoard problem → Wall adapter.

Loads JSON in the format used by the public MoonBoard datasets
(e.g. moonGen-style) and produces a `solver.wall.Wall` ready for the
simulator. Each problem JSON looks like:

    {
      "id": 19215,
      "name": "Far from the Madding Crowd",
      "setter": "Ben Moon",
      "grade": 4,
      "holdsets": [2],
      "start_holds": ["E6", "C5"],
      "mid_holds":   ["E8", "F11", "C13", "D15"],
      "end_holds":   ["D18"]
    }

Hold positions are letter+number where:
    letter A–K = column index 0–10 (11 columns)
    number 1–18 = row index from bottom (1 = bottom row)

Standard MoonBoard 2016 board geometry:
    - 11 columns × 18 rows of T-nut positions
    - 198 mm horizontal × 198 mm vertical bolt spacing
    - 40° overhang (positive in our convention)
    - Wall surface ≈ 2.40 m wide × 3.16 m tall

We model spacing at 200 mm = 20 cm for round numbers — within 1% of
the real board. Hold types default to "jug" (the MoonBoard hold-set
is mostly positive holds); per-position type / size hints can be
overridden via the `hold_type_overrides` argument.

Note on `holdsets`: the MoonBoard has three hold-sets (A=2016 only,
B=2017, C=2019), and a problem's `holdsets` list says which sets must
be installed. We don't model individual hold geometry — every position
becomes a generic "jug" cylinder at the right grid coordinate. This is
a coarse approximation; the actual board has a mix of slopers, crimps,
pinches and pockets. See the docstring for `moonboard_problem_to_wall`
for the override hooks.
"""
from __future__ import annotations

import json
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from solver.wall import Hold, Wall


# ─── MoonBoard 2016 default geometry ──────────────────────────────────────
MB_COLS = 11           # A..K
MB_ROWS = 18           # 1..18
MB_CELL_SIZE_CM = 20.0
MB_WALL_ANGLE_DEG = 40.0   # standard 40° overhang
MB_SURFACE_FRICTION = 0.7

LETTER_TO_COL: dict[str, int] = {ch: i for i, ch in enumerate(string.ascii_uppercase[:MB_COLS])}


# ─── Position parsing ─────────────────────────────────────────────────────
def position_to_grid(pos: str) -> tuple[int, int]:
    """`'E6'` → `(grid_x=4, grid_y=5)`. Climbing convention: row 1 is
    the bottom of the wall, so we map row N → grid_y = N - 1."""
    if not pos or len(pos) < 2:
        raise ValueError(f"bad MoonBoard position: {pos!r}")
    letter = pos[0].upper()
    if letter not in LETTER_TO_COL:
        raise ValueError(f"bad MoonBoard column: {pos!r} (must be A-K)")
    try:
        row = int(pos[1:])
    except ValueError as e:
        raise ValueError(f"bad MoonBoard row: {pos!r}") from e
    if not 1 <= row <= MB_ROWS:
        raise ValueError(f"MoonBoard row out of range: {pos!r}")
    return (LETTER_TO_COL[letter], row - 1)


def grid_to_position(grid_x: int, grid_y: int) -> str:
    """Inverse of `position_to_grid` — useful for dumping a beta back
    into MoonBoard notation."""
    return f"{string.ascii_uppercase[grid_x]}{grid_y + 1}"


# ─── Problem → Wall ───────────────────────────────────────────────────────
@dataclass
class MoonboardProblem:
    """Lightly-typed view over a single MoonBoard problem JSON entry."""

    id: int
    name: str
    setter: str
    grade: int
    holdsets: list[int]
    start_holds: list[str]
    mid_holds: list[str]
    end_holds: list[str]

    @classmethod
    def from_dict(cls, d: dict) -> "MoonboardProblem":
        return cls(
            id=int(d.get("id", 0)),
            name=str(d.get("name", "unnamed")),
            setter=str(d.get("setter", "")),
            grade=int(d.get("grade", 0)),
            holdsets=list(d.get("holdsets", [])),
            start_holds=list(d.get("start_holds", [])),
            mid_holds=list(d.get("mid_holds", [])),
            end_holds=list(d.get("end_holds", [])),
        )


def _safe_wall_id(problem: MoonboardProblem) -> str:
    """Slug a moonboard problem id+name into a wall_id valid against
    the locked schema (`[A-Za-z0-9_\\-]{1,64}`)."""
    raw = f"mb-{problem.id}-" + "".join(
        c if c.isalnum() else "-" for c in problem.name.lower()
    )
    raw = "".join(ch for ch in raw if ch.isalnum() or ch in "-_")[:64]
    return raw or f"mb-{problem.id}"


def moonboard_problem_to_wall(
    problem: MoonboardProblem | dict,
    *,
    hold_type_overrides: Optional[dict[str, str]] = None,
    hold_size_overrides: Optional[dict[str, str]] = None,
    cell_size_cm: float = MB_CELL_SIZE_CM,
    wall_angle_deg: float = MB_WALL_ANGLE_DEG,
    include_full_board: bool = False,
    vertical_projection: bool = False,
) -> Wall:
    """Build a `Wall` from a single MoonBoard problem.

    By default the wall contains only the problem's holds (start +
    middle + finish). Pass `include_full_board=True` to include all
    198 T-nut positions as available holds — useful for letting the
    solver discover off-route footholds.

    `hold_type_overrides`: position → hold_type, e.g. `{"E6": "crimp"}`
    `hold_size_overrides`: position → "small" | "medium" | "large"

    The MoonBoard JSON doesn't carry per-position hold type info — we
    default everything to a medium jug. The overrides let you mark up
    a specific board (e.g. our friend's gym dumped its own board here).

    `vertical_projection=True` flattens the row spacing in the wall's
    own plane so that each row's *vertical* (world-Z) position matches
    a vertical board. Useful when you want "row 6 is at the same height
    as a vertical reference board" rather than "row 6 is 6 cell-units
    along the tilted surface." Visually this is what looking at a
    MoonBoard photo head-on shows. We adjust cell_size by 1/cos(angle)
    so the world-Z spacing equals `cell_size_cm`. Horizontal spacing is
    unchanged because columns are along world-X (perpendicular to the
    tilt axis).
    """
    import math as _math
    effective_cell = cell_size_cm
    if vertical_projection and abs(wall_angle_deg) > 1e-3:
        effective_cell = cell_size_cm / _math.cos(_math.radians(wall_angle_deg))
    if isinstance(problem, dict):
        problem = MoonboardProblem.from_dict(problem)

    type_o = hold_type_overrides or {}
    size_o = hold_size_overrides or {}

    starts = set(problem.start_holds)
    finishes = set(problem.end_holds)
    mids = set(problem.mid_holds)

    if include_full_board:
        positions = [
            f"{string.ascii_uppercase[c]}{r}"
            for r in range(1, MB_ROWS + 1)
            for c in range(MB_COLS)
        ]
    else:
        positions = sorted(starts | mids | finishes)

    # Determinism: assign hold_ids in a stable order for downstream
    # diff-friendliness (same problem ⇒ same hold_id mapping).
    holds: list[Hold] = []
    for pos in positions:
        gx, gy = position_to_grid(pos)
        x_cm = (gx + 0.5) * cell_size_cm
        y_cm = (gy + 0.5) * effective_cell

        is_start = pos in starts
        is_finish = pos in finishes
        hold_type = type_o.get(pos, "jug")
        size = size_o.get(pos, "medium")

        if is_start:
            color = "#22c55e"
        elif is_finish:
            color = "#ef4444"
        elif pos in mids:
            color = "#3b82f6"
        else:
            color = "#888888"   # off-route bonus hold

        holds.append(Hold(
            hold_id=f"mb_{pos}",
            grid_x=gx,
            grid_y=gy,
            x_cm=x_cm,
            y_cm=y_cm,
            hold_type=hold_type,
            orientation_deg=0.0,
            size=size,
            color=color,
            is_start=is_start,
            is_finish=is_finish,
        ))

    return Wall(
        wall_id=_safe_wall_id(problem),
        name=f"MoonBoard · {problem.name} (V{problem.grade})",
        cols=MB_COLS,
        rows=MB_ROWS,
        cell_size_cm=effective_cell,
        holds=holds,
        wall_angle_deg=wall_angle_deg,
        surface_friction=MB_SURFACE_FRICTION,
    )


# ─── Loading from disk ────────────────────────────────────────────────────
def load_moonboard_problems(
    source: str | Path | Iterable[dict],
) -> list[MoonboardProblem]:
    """Read a MoonBoard problem-list JSON file (the public format is a
    top-level array of problem dicts). Also accepts an already-parsed
    iterable of dicts."""
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8") as f:
            raw = json.load(f)
    else:
        raw = list(source)

    if isinstance(raw, dict):
        # Some dumps wrap the array under "problems" or similar.
        for key in ("problems", "data", "items"):
            if key in raw and isinstance(raw[key], list):
                raw = raw[key]
                break
        else:
            raise ValueError("MoonBoard JSON: expected a list of problems")

    return [MoonboardProblem.from_dict(p) for p in raw]


def find_problem(
    problems: Iterable[MoonboardProblem],
    *,
    id: Optional[int] = None,
    name: Optional[str] = None,
    grade: Optional[int] = None,
) -> Optional[MoonboardProblem]:
    """Pick the first problem matching the given filters. None means
    "don't filter on this field". Useful for `--problem 19215` or
    `--problem-name "Far from..."` CLI flags."""
    for p in problems:
        if id is not None and p.id != id:
            continue
        if name is not None and name.lower() not in p.name.lower():
            continue
        if grade is not None and p.grade != grade:
            continue
        return p
    return None
