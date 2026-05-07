"""Wall + Hold loaders.

Reads the JSON produced by the grid editor and converts grid-space holds
into world-space (cm) coordinates the body model can reason about.

The locked schema (`schemas/wall.schema.json`) does not yet carry a real
cell size, so we hard-default `cell_size_cm = DEFAULT_CELL_SIZE_CM` and
emit a warning. See `planning-gabe.md` — this is a Phase-2 schema
discussion item.
"""
from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_CELL_SIZE_CM = 20.0

# Per-hold-type "positivity": a 0-1 grippability score used as a tie-break
# in the solver. Until the schema carries a per-hold value we use defaults.
POSITIVITY_BY_TYPE: dict[str, float] = {
    "jug": 0.95,
    "foothold": 0.80,
    "pinch": 0.60,
    "crimp": 0.50,
    "sloper": 0.40,
}

# Foothold-only holds can't be used by the hands in our simple model.
# Feet CAN use hand holds in a pinch but strongly prefer dedicated footholds.
HAND_USABLE_TYPES = {"jug", "crimp", "sloper", "pinch"}  # hands never use footholds
FOOT_USABLE_TYPES = {"jug", "crimp", "sloper", "pinch", "foothold"}  # feet can use anything


# Per-hold-type default friction coefficient. Used by the physics layer
# when a hold's JSON doesn't carry an explicit `friction` value. Numbers
# are intentionally rough — tuned by what feels right rather than measured.
FRICTION_BY_TYPE: dict[str, float] = {
    "jug": 0.9,
    "crimp": 0.85,
    "pinch": 0.8,
    "foothold": 0.85,
    "sloper": 0.6,
}


@dataclass(frozen=True)
class Hold:
    """A single hold in world coordinates (cm).

    The first ten fields are the locked schema. The trailing three
    (`friction_override`, `positivity_override`, `max_force_n`) are
    optional physics-layer extensions added in Phase 3 — `None` means
    "use the per-type default".
    """

    hold_id: str
    grid_x: int
    grid_y: int
    x_cm: float
    y_cm: float
    hold_type: str
    orientation_deg: float
    size: str
    color: str
    is_start: bool
    is_finish: bool
    # Optional physics extensions (additive — older walls still load).
    friction_override: float | None = None
    positivity_override: float | None = None
    max_force_n: float | None = None

    @property
    def positivity(self) -> float:
        if self.positivity_override is not None:
            return self.positivity_override
        return POSITIVITY_BY_TYPE.get(self.hold_type, 0.5)

    @property
    def friction(self) -> float:
        """Friction coefficient — explicit override wins, else per-type default."""
        if self.friction_override is not None:
            return self.friction_override
        return FRICTION_BY_TYPE.get(self.hold_type, 0.7)

    def usable_for_hand(self) -> bool:
        return self.hold_type in HAND_USABLE_TYPES

    def usable_for_foot(self) -> bool:
        return self.hold_type in FOOT_USABLE_TYPES


@dataclass
class Wall:
    """A wall plus its holds in world (cm) coordinates.

    `wall_angle_deg` and `surface_friction` are physics extensions; they
    default to a vertical wall with mid-grippy plastic if the JSON
    doesn't specify them.
    """

    wall_id: str
    name: str
    cols: int
    rows: int
    cell_size_cm: float
    holds: list[Hold] = field(default_factory=list)
    # Physics extensions (defaults match the original "vertical wall" assumption).
    wall_angle_deg: float = 0.0
    surface_friction: float = 0.7

    @property
    def width_cm(self) -> float:
        return self.cols * self.cell_size_cm

    @property
    def height_cm(self) -> float:
        return self.rows * self.cell_size_cm

    def by_id(self, hold_id: str) -> Hold:
        for h in self.holds:
            if h.hold_id == hold_id:
                return h
        raise KeyError(hold_id)

    def starts(self) -> list[Hold]:
        return [h for h in self.holds if h.is_start]

    def finishes(self) -> list[Hold]:
        return [h for h in self.holds if h.is_finish]


def _grid_to_world(
    grid_x: int, grid_y: int, cell_size_cm: float
) -> tuple[float, float]:
    """Centre of a grid cell, with y growing upward (climbing convention)."""
    return (
        (grid_x + 0.5) * cell_size_cm,
        (grid_y + 0.5) * cell_size_cm,
    )


def load_wall(
    source: str | Path | dict,
    cell_size_cm: float | None = None,
) -> Wall:
    """Load a wall from a path, a wall_id (resolved against /data/walls/),
    or a parsed dict.

    `cell_size_cm` overrides the schema default. If neither the schema nor
    the caller provides one, we fall back to DEFAULT_CELL_SIZE_CM with a
    warning.
    """
    payload = _resolve_payload(source)

    grid = payload.get("grid", {}) or {}
    cols = int(grid.get("cols", 10))
    rows = int(grid.get("rows", 14))

    schema_cell = grid.get("cell_size_cm")
    if cell_size_cm is None and schema_cell is None:
        warnings.warn(
            f"wall '{payload.get('wall_id')}' has no cell_size_cm; defaulting "
            f"to {DEFAULT_CELL_SIZE_CM} cm. See planning-gabe.md.",
            stacklevel=2,
        )
    resolved_cell = float(
        cell_size_cm if cell_size_cm is not None
        else schema_cell if schema_cell is not None
        else DEFAULT_CELL_SIZE_CM
    )

    holds = [_hold_from_json(h, resolved_cell) for h in payload.get("holds", [])]

    return Wall(
        wall_id=str(payload.get("wall_id", "unnamed")),
        name=str(payload.get("name", payload.get("wall_id", "unnamed"))),
        cols=cols,
        rows=rows,
        cell_size_cm=resolved_cell,
        holds=holds,
        wall_angle_deg=float(payload.get("wall_angle_deg", 0.0)),
        surface_friction=float(payload.get("surface_friction", 0.7)),
    )


def _resolve_payload(source: str | Path | dict) -> dict:
    if isinstance(source, dict):
        return source

    path = Path(source)
    if not path.suffix:  # treat as wall_id
        for candidate in (
            Path("/data/walls") / f"{path.name}.json",
            Path(__file__).resolve().parent.parent / "data" / "examples" / f"{path.name}.json",
        ):
            if candidate.exists():
                path = candidate
                break

    if not path.exists():
        raise FileNotFoundError(f"wall not found: {source}")
    return json.loads(path.read_text(encoding="utf-8"))


def _hold_from_json(h: dict, cell_size_cm: float) -> Hold:
    x_cm, y_cm = _grid_to_world(h["grid_x"], h["grid_y"], cell_size_cm)
    return Hold(
        hold_id=h["hold_id"],
        grid_x=int(h["grid_x"]),
        grid_y=int(h["grid_y"]),
        x_cm=x_cm,
        y_cm=y_cm,
        hold_type=h["hold_type"],
        orientation_deg=float(h["orientation_deg"]),
        size=h["size"],
        color=h["color"],
        is_start=bool(h["is_start"]),
        is_finish=bool(h["is_finish"]),
        friction_override=(float(h["friction"]) if "friction" in h else None),
        positivity_override=(float(h["positivity"]) if "positivity" in h else None),
        max_force_n=(float(h["max_force_n"]) if "max_force_n" in h else None),
    )


def hand_holds(holds: Iterable[Hold]) -> list[Hold]:
    return [h for h in holds if h.usable_for_hand()]


def foot_holds(holds: Iterable[Hold]) -> list[Hold]:
    return [h for h in holds if h.usable_for_foot()]
