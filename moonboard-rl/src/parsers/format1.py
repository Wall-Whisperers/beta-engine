"""Parser for MoonBoard data Format 1 (moonboard1.json).

Format 1 is an array of route objects. Key fields:
  - id (int): unique route identifier
  - name (str): route name
  - grade (int): V-scale grade directly (e.g. 4 → V4)
  - repeats (int): number of logged ascents
  - start_holds, mid_holds, end_holds (list[str]): hold coordinates like "F4"
    where the letter is the column (A–K) and the number is the row (1–18).
"""

import json
import re

from .canonical import Hold, Route, col_letter_to_int


def _parse_coord(coord: str, hold_id: str, role: str) -> Hold:
    """Parse a Format-1 hold coordinate string into a Hold object.

    Args:
        coord: Coordinate string such as "F4" (column letter + row number).
        hold_id: Identifier to assign to the returned Hold.
        role: Role string — "start", "mid", or "end".

    Returns:
        Hold with col and row populated.

    Raises:
        ValueError: If coord does not match the expected pattern.
    """
    m = re.fullmatch(r"([A-Ka-k]+)(\d+)", coord.strip())
    if not m:
        raise ValueError(f"Cannot parse Format-1 coordinate: '{coord}'")
    col = col_letter_to_int(m.group(1))
    row = int(m.group(2))
    return Hold(hold_id=hold_id, col=col, row=row, role=role)


def load_routes(path: str) -> list[Route]:
    """Load all routes from a Format-1 JSON file.

    Args:
        path: Filesystem path to moonboard1.json.

    Returns:
        List of Route objects in the order they appear in the file.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    routes: list[Route] = []
    for entry in raw:
        route_id = str(entry["id"])
        name = entry.get("name", f"route_{route_id}")
        grade_v = int(entry.get("grade", 0))
        repeats = int(entry.get("repeats", 0))

        holds: list[Hold] = []
        hold_counter = 0

        for role, key in [("start", "start_holds"), ("mid", "mid_holds"), ("end", "end_holds")]:
            for coord in entry.get(key) or []:
                hold_counter += 1
                hid = f"{route_id}_h{hold_counter:03d}"
                try:
                    holds.append(_parse_coord(coord, hid, role))
                except ValueError as exc:
                    print(f"[format1] Warning: {exc} in route {route_id}")

        routes.append(Route(route_id=route_id, name=name, grade_v=grade_v, holds=holds, repeats=repeats))

    return routes
