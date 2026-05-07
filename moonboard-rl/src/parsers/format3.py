"""Parser for MoonBoard data Format 3 (moonboard3.json).

Format 3 is a dict keyed by route ID. It is the richest schema:
  - Grade (str): Font grade string, e.g. "6A", "7A+"
  - Name (str): route name
  - Repeats (int): logged ascent count
  - Holdsetup (dict): contains Id (int) and Description (str).
    IMPORTANT: Different Holdsetup.Id values correspond to different physical
    hold configurations. Routes with uncommon IDs may not map correctly to the
    standard hold positions and will emit a warning.
  - Moves (list): each move has:
      - Id (int): unique move identifier
      - Description (str): hold coordinate like "J4" (col letter + row number)
      - IsStart (bool), IsEnd (bool): role markers

Hold coordinates use the same col-first style as Format 1: "J4" → col=J, row=4.
"""

import json
import logging
import os
import re
import sys
from collections import Counter

# Support running as a standalone script (python3 src/parsers/format3.py).
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
    from src.parsers.canonical import Hold, Route, col_letter_to_int, font_to_v_grade
else:
    from .canonical import Hold, Route, col_letter_to_int, font_to_v_grade

logging.basicConfig(level=logging.INFO, format="[format3] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

_COORD_RE = re.compile(r"^([A-Ka-k]+)(\d+)$")


def _parse_coord(desc: str) -> tuple[int, int]:
    """Parse a Format-3 hold coordinate string into (col, row).

    Args:
        desc: Coordinate string such as "J4" (col letter + row number).

    Returns:
        Tuple of (col_int, row_int) where col is 0-indexed and row is 1-indexed.

    Raises:
        ValueError: If the string does not match the expected col-first pattern.
    """
    m = _COORD_RE.match(desc.strip())
    if not m:
        raise ValueError(f"Cannot parse Format-3 coordinate: '{desc}'")
    col = col_letter_to_int(m.group(1))
    row = int(m.group(2))
    return col, row


def load_routes(path: str) -> list[Route]:
    """Load all routes from a Format-3 JSON file.

    Emits a WARNING for any route whose Holdsetup.Id is not one of the two
    most common IDs in the dataset, since those routes may reference holds
    that do not exist on the standard MoonBoard configuration.

    Args:
        path: Filesystem path to moonboard3.json.

    Returns:
        List of Route objects in iteration order of the source dict.
    """
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)

    # Find the two most common Holdsetup.Id values.
    id_counts: Counter = Counter()
    for entry in raw.values():
        hs = entry.get("Holdsetup") or {}
        hsid = hs.get("Id")
        if hsid is not None:
            id_counts[hsid] += 1
    common_ids = {hsid for hsid, _ in id_counts.most_common(2)}
    logger.info("Most common Holdsetup.Id values: %s", common_ids)

    routes: list[Route] = []
    for route_id, entry in raw.items():
        name = entry.get("Name", f"route_{route_id}")
        grade_v = font_to_v_grade(entry.get("Grade", ""))
        repeats = int(entry.get("Repeats", 0))

        hs = entry.get("Holdsetup") or {}
        hsid = hs.get("Id")
        if hsid not in common_ids:
            logger.warning(
                "Route %s ('%s') has uncommon Holdsetup.Id=%s — hold positions may not be standard",
                route_id,
                name,
                hsid,
            )

        holds: list[Hold] = []
        for move in entry.get("Moves") or []:
            desc = move.get("Description", "").strip()
            if not desc:
                continue
            is_start = bool(move.get("IsStart", False))
            is_end = bool(move.get("IsEnd", False))
            role = "start" if is_start else ("end" if is_end else "mid")
            mid = move.get("Id", len(holds))
            hid = f"{route_id}_h{mid}"
            try:
                col, row = _parse_coord(desc)
            except ValueError as exc:
                logger.warning("%s in route %s", exc, route_id)
                continue
            holds.append(Hold(hold_id=hid, col=col, row=row, role=role))

        routes.append(Route(route_id=route_id, name=name, grade_v=grade_v, holds=holds, repeats=repeats))

    return routes


if __name__ == "__main__":
    import os

    data_path = os.path.join(os.path.dirname(__file__), "../../moonboard_data/moonboard3.json")
    if not os.path.exists(data_path):
        print(f"ERROR: File not found: {data_path}")
        sys.exit(1)

    all_routes = load_routes(data_path)
    print(f"\nLoaded {len(all_routes)} routes from Format 3\n")
    for r in all_routes[:3]:
        print(f"  Route {r.route_id}: '{r.name}' V{r.grade_v} repeats={r.repeats} — {len(r.holds)} holds")
        for h in r.holds:
            col_letter = chr(ord("A") + h.col)
            print(f"    [{h.role:5s}] col={col_letter} row={h.row}")
        print()
