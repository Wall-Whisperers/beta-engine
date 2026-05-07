"""Parser for MoonBoard data Format 2 (moonboard2.json).

Format 2 is a dict keyed by string index. Key fields:
  - Grade (str): Font grade with optional leading space, e.g. " 6C"
  - Name (str): route name
  - Moves (list): each move has:
      - Description (str): hold coordinate — either "16F" (row-first: digits then letter)
                           or "F4" (col-first: letter then digits)
      - IsStart (bool): True if this is a starting hold
      - IsEnd (bool): True if this is a finishing hold

Hold coordinates use two possible orderings:
  - Row-first: "16F" → row=16, col=F (leading digits, trailing letters)
  - Col-first: "F4"  → col=F, row=4  (leading letters, trailing digits)
The parser detects which style is present using regex.
"""

import json
import os
import re
import sys

# Support running as a standalone script (python3 src/parsers/format2.py).
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.parsers.canonical import Hold, Route, col_letter_to_int, font_to_v_grade
else:
    from .canonical import Hold, Route, col_letter_to_int, font_to_v_grade

_ROW_FIRST = re.compile(r"^(\d+)([A-Ka-k]+)$")
_COL_FIRST = re.compile(r"^([A-Ka-k]+)(\d+)$")


def _parse_coord(desc: str) -> tuple[int, int]:
    """Parse a Format-2 hold coordinate description into (col, row).

    Handles both "16F" (row-first) and "F4" (col-first) styles.

    Args:
        desc: Raw coordinate string from the Description field.

    Returns:
        Tuple of (col_int, row_int) where col is 0-indexed and row is 1-indexed.

    Raises:
        ValueError: If the string matches neither known pattern.
    """
    s = desc.strip()
    m = _ROW_FIRST.match(s)
    if m:
        row = int(m.group(1))
        col = col_letter_to_int(m.group(2))
        return col, row
    m = _COL_FIRST.match(s)
    if m:
        col = col_letter_to_int(m.group(1))
        row = int(m.group(2))
        return col, row
    raise ValueError(f"Cannot parse Format-2 coordinate: '{desc}'")


def load_routes(path: str) -> list[Route]:
    """Load all routes from a Format-2 JSON file.

    Args:
        path: Filesystem path to moonboard2.json.

    Returns:
        List of Route objects. Routes with no parseable holds are skipped with a warning.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    routes: list[Route] = []
    for idx, entry in raw.items():
        route_id = str(idx)
        name = entry.get("Name", f"route_{route_id}")
        grade_v = font_to_v_grade(entry.get("Grade", ""))

        holds: list[Hold] = []
        for move_i, move in enumerate(entry.get("Moves") or []):
            desc = move.get("Description", "").strip()
            if not desc:
                continue
            is_start = bool(move.get("IsStart", False))
            is_end = bool(move.get("IsEnd", False))
            role = "start" if is_start else ("end" if is_end else "mid")
            try:
                col, row = _parse_coord(desc)
            except ValueError as exc:
                print(f"[format2] Warning: {exc} in route {route_id} move {move_i}")
                continue
            hid = f"{route_id}_h{move_i + 1:03d}"
            holds.append(Hold(hold_id=hid, col=col, row=row, role=role))

        routes.append(Route(route_id=route_id, name=name, grade_v=grade_v, holds=holds))

    return routes


if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), "../../moonboard_data/moonboard2.json")
    if not os.path.exists(data_path):
        print(f"ERROR: File not found: {data_path}")
        sys.exit(1)

    all_routes = load_routes(data_path)
    print(f"Loaded {len(all_routes)} routes from Format 2\n")
    for r in all_routes[:3]:
        print(f"  Route {r.route_id}: '{r.name}' V{r.grade_v} — {len(r.holds)} holds")
        for h in r.holds:
            col_letter = chr(ord("A") + h.col)
            print(f"    [{h.role:5s}] col={col_letter} row={h.row}")
        print()
