"""CLI entry point: `python -m solver --wall <wall_id|path>`.

Loads a wall JSON, solves it (A* by default, optionally Q-learning), and
writes a PNG/GIF visualization plus a text move list to `data/runs/`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from solver.astar import solve_astar
from solver.body import BodyModel
from solver.rl_qlearn import solve_qlearn
from solver.visualize import render_animation, render_panels
from solver.wall import load_wall


def _runs_dir() -> Path:
    candidate = Path("/data/runs")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        return candidate
    except (PermissionError, OSError):
        fallback = Path(__file__).resolve().parent.parent / "data" / "runs"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="solver", description=__doc__)
    p.add_argument("--wall", required=True,
                   help="wall_id (resolved to /data/walls/) or a path to a JSON file")
    p.add_argument("--method", choices=("astar", "qlearn", "both"), default="astar")
    p.add_argument("--height-cm", type=float, default=175.0)
    p.add_argument("--wingspan-cm", type=float, default=175.0)
    p.add_argument("--cell-size-cm", type=float, default=None,
                   help="override the wall's cell_size_cm (default 20 cm if missing)")
    p.add_argument("--episodes", type=int, default=2000,
                   help="Q-learning episodes (only used by qlearn / both)")
    p.add_argument("--no-viz", action="store_true", help="skip rendering")
    p.add_argument("--gif", action="store_true", help="also render an animated GIF")
    args = p.parse_args(argv)

    wall = load_wall(args.wall, cell_size_cm=args.cell_size_cm)
    body = BodyModel(height_cm=args.height_cm, wingspan_cm=args.wingspan_cm)

    print(f"Wall: {wall.name} ({wall.cols}×{wall.rows} grid, "
          f"{wall.cell_size_cm:.1f} cm cells, {len(wall.holds)} holds)")
    print(f"Body: {body.height_cm:.0f} cm tall, {body.wingspan_cm:.0f} cm wingspan")

    methods = ["astar", "qlearn"] if args.method == "both" else [args.method]
    runs = _runs_dir()
    any_success = False

    for method in methods:
        print(f"\n— Solving with {method} —")
        if method == "astar":
            result = solve_astar(wall, body)
        else:
            result = solve_qlearn(wall, body, episodes=args.episodes)

        if result is None:
            print(f"  No solution found via {method}.")
            continue
        any_success = True
        print(f"  Found {len(result.moves)}-move beta "
              f"(expanded={result.expanded}, method={result.method})")
        for line in result.text_steps():
            print(f"  {line}")

        if args.no_viz:
            continue

        png_path = runs / f"{wall.wall_id}-{method}.png"
        render_panels(wall, result, png_path, body=body)
        print(f"  Wrote {png_path}")
        if args.gif:
            gif_path = render_animation(wall, result, png_path.with_suffix(".gif"), body=body)
            if gif_path:
                print(f"  Wrote {gif_path}")

    return 0 if any_success else 1


if __name__ == "__main__":
    sys.exit(main())
