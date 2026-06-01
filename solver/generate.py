"""Procedural wall generator for RL training data.

Algorithm
---------
1. Place start holds: two hands low-center, two feet below them.
2. Build a "spine" by alternating LH/RH, advancing 1-3 rows per step.
   Each new hold is placed within REACH_SAFETY * arm_length of the current
   hold for the same hand (hold-to-hold approximation — conservative and
   fast; the A* check catches anything that slips through).
3. Add footholds every ~4 rows of spine advancement to keep feet viable.
4. Scatter extra holds for route variety.
5. Mark the top 1-2 hand holds as finish.
6. Verify with A* (max 8 000 expansions). Retry with a perturbed seed
   up to MAX_RETRIES times.

Output is the locked wall JSON schema — same format the grid editor saves.
"""
from __future__ import annotations

import json
import math
import random
import uuid
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import Optional

import numpy as np

from solver.body import BodyModel
from solver.wall import DEFAULT_CELL_SIZE_CM

# ── Visual constants (match the grid editor palette) ───────────────────────
COLOR_BY_TYPE: dict[str, str] = {
    "jug":      "#22c55e",
    "crimp":    "#ef4444",
    "sloper":   "#f59e0b",
    "pinch":    "#3b82f6",
    "foothold": "#a855f7",
}
SIZE_BY_TYPE: dict[str, str] = {
    "jug":      "large",
    "crimp":    "small",
    "sloper":   "large",
    "pinch":    "medium",
    "foothold": "small",
}

MAX_RETRIES = 10


def _difficulty_params(difficulty: float) -> dict:
    """Derive all difficulty-sensitive generation parameters from a 0–1 scalar.

    Returns a dict with:
      n_scatter    – scatter holds beyond the spine (more = easier, more options)
      min_step_dy  – minimum row advance per hand move (higher = forced big moves)
      max_step_dy  – maximum row advance per hand move (higher = longer moves)
      foot_lag     – rows hands can outrun feet before forcing a new foothold
                     (higher = fewer footholds = harder)
      reach_frac   – fraction of arm_length for hold-to-hold spacing
                     (higher = near max reach = harder)
      scatter_bias – probability scatter lands near an existing spine hold
                     (high = easy route has many reachable alternatives;
                      low = hard route scatters holds randomly as noise)
      size_penalty – whether to downgrade hold sizes one step (hard = smaller holds)
    """
    d = max(0.0, min(1.0, difficulty))
    return {
        "n_scatter":    max(3, round(14 - d * 11)),      # 14 (easy) → 3 (hard)
        "min_step_dy":  1 + round(d * 1),                # 1 (easy) → 2 (hard)
        "max_step_dy":  2 + round(d * 2),                # 2 (easy) → 4 (hard)
        "foot_lag":     max(3, round(3 + d * 5)),        # 3 (easy) → 8 (hard)
        "reach_frac":   0.60 + d * 0.30,                 # 0.60 (easy) → 0.90 (hard)
        "scatter_bias": max(0.0, 0.8 - d * 1.0),        # 0.8 (easy, near spine) → 0.0 (hard, random)
        "size_penalty": d >= 0.6,                        # shrink hold sizes on hard routes
    }


@dataclass
class GeneratorConfig:
    """Knobs for the procedural generator.

    Attributes
    ----------
    cols, rows      Grid dimensions.
    cell_size_cm    Physical size of each grid cell (sets real-world scale).
    difficulty      0.0 = easy (jugs), 1.0 = hard (crimps/slopers).
    extra_holds     Scatter holds added beyond the spine.
    seed            Fixed seed for reproducibility; None = random each call.
    """

    cols: int = 12
    # 20, not 18: the start hands sit ~2 rows up (see _build_holds) so the
    # body can hang instead of crouch. That offset eats 2 rows of climb span,
    # and at difficulty 1.0 the route no longer fits/verifies in 18 rows.
    # 20 restores the original ~14-row climbable span above the start.
    rows: int = 20
    cell_size_cm: float = DEFAULT_CELL_SIZE_CM
    difficulty: float = 0.5
    extra_holds: int = 8
    seed: Optional[int] = None


# ── Public API ──────────────────────────────────────────────────────────────

def generate_wall(
    config: Optional[GeneratorConfig] = None,
    body: Optional[BodyModel] = None,
    wall_id: Optional[str] = None,
) -> Optional[dict]:
    """Generate a solvable wall and return it as a JSON-schema-compliant dict.

    Returns None if every retry attempt fails A* verification (rare — usually
    means the config is geometrically impossible, e.g. cols < 4).
    """
    cfg = config or GeneratorConfig()
    body = body or BodyModel()
    wall_id = wall_id or f"synth-{uuid.uuid4().hex[:8]}"

    if cfg.cols < 4:
        raise ValueError("cols must be ≥ 4")

    base_seed = cfg.seed if cfg.seed is not None else random.randrange(10 ** 9)

    for attempt in range(MAX_RETRIES):
        rng = random.Random(base_seed + attempt)
        holds = _build_holds(cfg, body, rng)
        if holds is None:
            continue
        wall_dict = _pack_json(holds, cfg, wall_id)
        if _verify(wall_dict, body, cfg.cell_size_cm):
            return wall_dict

    return None


def generate_batch(
    n: int,
    config: Optional[GeneratorConfig] = None,
    body: Optional[BodyModel] = None,
) -> list[dict]:
    """Generate `n` solvable walls. Walls that fail all retries are skipped."""
    cfg = config or GeneratorConfig()
    body = body or BodyModel()
    results: list[dict] = []
    for i in range(n):
        seed = (cfg.seed or 0) + i * 997 if cfg.seed is not None else None
        w = generate_wall(GeneratorConfig(**{**cfg.__dict__, "seed": seed}), body)
        if w is not None:
            results.append(w)
    return results


# ── Internal hold construction ──────────────────────────────────────────────

def _build_holds(
    cfg: GeneratorConfig,
    body: BodyModel,
    rng: random.Random,
) -> Optional[list[dict]]:
    """Build the holds list. Returns None if spine construction gets stuck."""
    occupied: set[tuple[int, int]] = set()
    holds: list[dict] = []
    _counter = [0]

    def add_hold(
        gx: int, gy: int, hold_type: str,
        is_start: bool = False, is_finish: bool = False,
    ) -> str:
        _counter[0] += 1
        hid = f"h_{_counter[0]:03d}"
        holds.append({
            "hold_id": hid,
            "grid_x": gx,
            "grid_y": gy,
            "hold_type": hold_type,
            "orientation_deg": float(rng.randrange(0, 72) * 5),
            "size": hold_size(hold_type) if not is_start else SIZE_BY_TYPE[hold_type],
            "color": COLOR_BY_TYPE[hold_type],
            "is_start": is_start,
            "is_finish": is_finish,
        })
        occupied.add((gx, gy))
        return hid

    cx = cfg.cols // 2

    # Derive all difficulty-sensitive params up front.
    dp = _difficulty_params(cfg.difficulty)
    n_scatter    = dp["n_scatter"]
    min_step_dy  = dp["min_step_dy"]
    max_step_dy  = dp["max_step_dy"]
    foot_lag     = dp["foot_lag"]
    scatter_bias = dp["scatter_bias"]
    size_penalty = dp["size_penalty"]
    arm_cells    = body.arm_length * dp["reach_frac"] / cfg.cell_size_cm

    def hold_size(hold_type: str) -> str:
        base = SIZE_BY_TYPE[hold_type]
        if size_penalty and hold_type != "foothold":
            return {"large": "medium", "medium": "small", "small": "small"}[base]
        return base

    # ── Phase 1: Start holds ────────────────────────────────────────────────
    # Two hands shoulder-width apart, two footholds well below them.
    # Start holds are always jugs regardless of difficulty.
    #
    # Start height sets whether the seed pose is a clean hang. The real
    # training climber (wingspan 175 cm → 0.61 m arm reach) has its shoulders
    # settle at ~1.2 m in the extended seed pose. There is a narrow clean
    # window for the start hands, measured by sweeping hand height vs the
    # settled per-limb weld force:
    #   hand z ≤ 0.9 m  → hands reach DOWN past full extension → 1.3x over cap
    #   hand z 1.1–1.3 m → shoulders ≈ hand height: true hang, ~0.3 kN/limb
    #   hand z ≥ 1.5 m  → body hangs, legs over-extend to the low feet → feet
    #                      blow 2.5x over cap
    # So target hand z ≈ 1.0 m (mid-window, margin from both cliffs) and keep
    # feet low (~0.2 m) for a ~0.8 m gap. Derived from cell size for any grid.
    start_hand_row = max(2, round(100.0 / cfg.cell_size_cm))
    start_foot_row = max(0, round(20.0 / cfg.cell_size_cm))
    start_foot_row = min(start_foot_row, start_hand_row - 3)  # keep ≥0.5 m gap
    lh_pos = (cx - 1, start_hand_row)
    rh_pos = (cx + 1, start_hand_row)
    add_hold(*lh_pos, "jug", is_start=True)
    add_hold(*rh_pos, "jug", is_start=True)
    add_hold(cx - 1, start_foot_row, "foothold")
    add_hold(cx + 1, start_foot_row, "foothold")

    finish_row = cfg.rows - 2
    foot_frontier = start_foot_row  # highest foothold row placed so far

    # ── Phase 2: Build hand spine ───────────────────────────────────────────
    alt = cycle(["LH", "RH"])
    max_steps = cfg.rows * 4  # safety cap

    for _ in range(max_steps):
        if lh_pos[1] >= finish_row and rh_pos[1] >= finish_row:
            break

        limb = next(alt)
        cur = lh_pos if limb == "LH" else rh_pos

        if cur[1] >= finish_row:
            # This hand is done; let the loop drain.
            continue

        cell = _sample_hand_cell(cur, arm_cells, cfg, occupied, rng,
                                  min_dy=min_step_dy, max_dy=max_step_dy)
        if cell is None:
            # Widen search once before giving up (relax min_dy too).
            cell = _sample_hand_cell(cur, arm_cells * 1.15, cfg, occupied, rng,
                                      min_dy=max(1, min_step_dy - 1), max_dy=max_step_dy + 1)
        if cell is None:
            return None  # stuck — caller will retry with perturbed seed

        gx, gy = cell
        hold_type = _pick_hand_type(gy, cfg.rows, cfg.difficulty, rng)
        add_hold(gx, gy, hold_type)

        if limb == "LH":
            lh_pos = (gx, gy)
        else:
            rh_pos = (gx, gy)

        # Footholds: advance when hands are pulling too far ahead of feet.
        hand_avg_row = (lh_pos[1] + rh_pos[1]) // 2
        if hand_avg_row - foot_frontier > foot_lag:
            foot_row = max(0, hand_avg_row - 2)
            # Try to place 1-2 footholds in a horizontal band around center.
            placed = 0
            for attempt_col in rng.sample(range(max(0, cx - 3), min(cfg.cols, cx + 4)), k=min(7, cfg.cols)):
                if (attempt_col, foot_row) not in occupied:
                    add_hold(attempt_col, foot_row, "foothold")
                    foot_frontier = foot_row
                    placed += 1
                    if placed >= 2:
                        break

    # ── Phase 3: Mark finish holds ──────────────────────────────────────────
    hand_holds_placed = [
        h for h in holds
        if not h["is_start"] and h["hold_type"] != "foothold"
    ]
    if not hand_holds_placed:
        return None

    top_row = max(h["grid_y"] for h in hand_holds_placed)
    marked = 0
    for h in hand_holds_placed:
        if h["grid_y"] >= top_row - 1:
            h["is_finish"] = True
            marked += 1
            if marked >= 2:
                break

    if marked == 0:
        return None

    # ── Phase 4: Scatter extra holds ────────────────────────────────────────
    # Easy routes: scatter near spine holds (reachable alternatives).
    # Hard routes: fully random placement (mostly useless noise / traps).
    spine_cells = [(h["grid_x"], h["grid_y"]) for h in holds if h["hold_type"] != "foothold"]
    scatter_radius = max(2, round(arm_cells * 0.5))
    added = 0
    for _ in range(n_scatter * 8):
        if added >= n_scatter:
            break
        if scatter_bias > 0 and spine_cells and rng.random() < scatter_bias:
            scx, scy = rng.choice(spine_cells)
            gx = scx + rng.randint(-scatter_radius, scatter_radius)
            gy = scy + rng.randint(-scatter_radius, scatter_radius)
            gx = max(0, min(cfg.cols - 1, gx))
            gy = max(1, min(cfg.rows - 2, gy))
        else:
            gx = rng.randint(0, cfg.cols - 1)
            gy = rng.randint(1, cfg.rows - 2)
        if (gx, gy) in occupied:
            continue
        hold_type = _pick_scatter_type(gy, cfg.rows, cfg.difficulty, rng)
        add_hold(gx, gy, hold_type)
        added += 1

    return holds


# ── Cell sampling ────────────────────────────────────────────────────────────

def _sample_hand_cell(
    cur: tuple[int, int],
    arm_cells: float,
    cfg: GeneratorConfig,
    occupied: set[tuple[int, int]],
    rng: random.Random,
    min_dy: int = 1,
    max_dy: int = 3,
) -> Optional[tuple[int, int]]:
    """Return a random grid cell reachable by a hand from `cur`."""
    cx, cy = cur
    r = math.ceil(arm_cells)
    candidates: list[tuple[int, int]] = []

    for dy in range(min_dy, max_dy + 1):
        gy = cy + dy
        if gy >= cfg.rows:
            continue
        for dx in range(-r, r + 1):
            gx = cx + dx
            if not (0 <= gx < cfg.cols):
                continue
            if (gx, gy) in occupied:
                continue
            dist = math.hypot(dx, dy)
            if dist <= arm_cells:
                candidates.append((gx, gy))

    if not candidates:
        return None
    return rng.choice(candidates)


# ── Hold type selection ──────────────────────────────────────────────────────

def _pick_hand_type(row: int, total_rows: int, difficulty: float, rng: random.Random) -> str:
    """Pick a hold type scaled by position (row) and route difficulty.

    Difficulty is weighted 70% so hard routes feel hard throughout, not just
    at the top. Row fraction adds 30% so the top of any route is always hardest.
    """
    row_frac = row / max(1, total_rows - 1)
    d = difficulty * 0.70 + row_frac * 0.30  # difficulty-dominant blend

    if d < 0.25:
        pool = ["jug", "jug", "jug", "crimp"]
    elif d < 0.45:
        pool = ["jug", "jug", "crimp", "crimp"]
    elif d < 0.65:
        pool = ["jug", "crimp", "crimp", "pinch"]
    else:
        pool = ["crimp", "sloper", "pinch", "pinch"]

    return rng.choice(pool)


def _pick_scatter_type(row: int, total_rows: int, difficulty: float, rng: random.Random) -> str:
    row_frac = row / max(1, total_rows - 1)
    d = difficulty * 0.70 + row_frac * 0.30
    # Scatter holds include footholds for variety.
    if d < 0.35:
        pool = ["jug", "foothold", "foothold", "crimp"]
    elif d < 0.60:
        pool = ["crimp", "foothold", "pinch", "jug"]
    else:
        pool = ["crimp", "sloper", "pinch", "foothold"]
    return rng.choice(pool)


# ── JSON packaging ───────────────────────────────────────────────────────────

def _pack_json(holds: list[dict], cfg: GeneratorConfig, wall_id: str) -> dict:
    return {
        "wall_id": wall_id,
        "grid": {
            "cols": cfg.cols,
            "rows": cfg.rows,
            "cell_size_cm": cfg.cell_size_cm,
        },
        "holds": holds,
    }


# ── A* verification ──────────────────────────────────────────────────────────

def _verify(wall_dict: dict, body: BodyModel, cell_size_cm: float) -> bool:
    """Return True if the wall is solvable by A*."""
    import warnings
    from solver.wall import load_wall
    from solver.astar import solve_astar

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wall = load_wall(wall_dict, cell_size_cm=cell_size_cm)

    return solve_astar(wall, body, max_expansions=8_000) is not None


# ── CLI ──────────────────────────────────────────────────────────────────────

def _cli(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(prog="solver.generate",
                                description="Generate synthetic solvable walls.")
    p.add_argument("--count", type=int, default=1, help="Number of walls to generate.")
    p.add_argument("--cols",  type=int, default=12)
    p.add_argument("--rows",  type=int, default=18)
    p.add_argument("--cell-size-cm", type=float, default=DEFAULT_CELL_SIZE_CM)
    p.add_argument("--difficulty", type=float, default=0.5,
                   help="0.0 = easy (jugs) … 1.0 = hard (crimps/slopers)")
    p.add_argument("--extra-holds", type=int, default=8)
    p.add_argument("--height-cm",  type=float, default=175.0)
    p.add_argument("--wingspan-cm", type=float, default=175.0)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Directory to write JSON files. Defaults to data/walls/.")
    p.add_argument("--stdout", action="store_true",
                   help="Print the first generated wall to stdout (ignores --output-dir).")
    args = p.parse_args(argv)

    cfg = GeneratorConfig(
        cols=args.cols,
        rows=args.rows,
        cell_size_cm=args.cell_size_cm,
        difficulty=args.difficulty,
        extra_holds=args.extra_holds,
        seed=args.seed,
    )
    body = BodyModel(height_cm=args.height_cm, wingspan_cm=args.wingspan_cm)

    out_dir: Optional[Path] = None
    if not args.stdout:
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
        else:
            # Try the Docker-mounted path first, fall back to local.
            candidate = Path("/data/walls")
            try:
                candidate.mkdir(parents=True, exist_ok=True)
                out_dir = candidate
            except (PermissionError, OSError):
                out_dir = Path(__file__).resolve().parent.parent / "data" / "walls"
                out_dir.mkdir(parents=True, exist_ok=True)

    generated = 0
    for i in range(args.count):
        seed_i = (args.seed + i * 997) if args.seed is not None else None
        wall_id = f"synth-{uuid.uuid4().hex[:8]}"
        w = generate_wall(
            GeneratorConfig(**{**cfg.__dict__, "seed": seed_i}),
            body,
            wall_id=wall_id,
        )
        if w is None:
            print(f"  [!] Wall {i+1}/{args.count} failed all retries — skipped.",
                  file=sys.stderr)
            continue

        generated += 1
        if args.stdout:
            print(json.dumps(w, indent=2))
            break  # only first wall to stdout

        path = out_dir / f"{wall_id}.json"  # type: ignore[operator]
        path.write_text(json.dumps(w, indent=2), encoding="utf-8")
        n_holds = len(w["holds"])
        n_finish = sum(1 for h in w["holds"] if h["is_finish"])
        print(f"  [{i+1}/{args.count}] {wall_id}  {n_holds} holds, "
              f"{n_finish} finish → {path}")

    if not args.stdout:
        print(f"\nGenerated {generated}/{args.count} walls.")
    return 0 if generated > 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_cli())
