"""CLI entry point: `python -m solver [generate|solve] ...`.

Sub-commands
------------
  solve     (default) Load a wall JSON and solve it (A* or Q-learning).
  generate  Procedurally generate one or more solvable walls.

Legacy form `python -m solver --wall <id>` still works (treated as `solve`).
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
    # Detect legacy invocation (`--wall` as first meaningful arg) so old
    # scripts keep working without a sub-command.
    args_raw = argv if argv is not None else sys.argv[1:]
    if args_raw and args_raw[0] not in ("solve", "generate") and "--wall" in args_raw:
        return _solve(args_raw)

    p = argparse.ArgumentParser(prog="solver", description=__doc__)
    sub = p.add_subparsers(dest="cmd")

    # ── solve ──
    sp = sub.add_parser("solve", help="Solve a wall (A* or Q-learning).")
    sp.add_argument("--wall", required=True,
                    help="wall_id (resolved to /data/walls/) or a JSON path")
    sp.add_argument("--method", choices=("astar", "qlearn", "both"), default="astar")
    sp.add_argument("--height-cm",   type=float, default=175.0)
    sp.add_argument("--wingspan-cm", type=float, default=175.0)
    sp.add_argument("--cell-size-cm", type=float, default=None)
    sp.add_argument("--episodes", type=int, default=2000)
    sp.add_argument("--no-viz", action="store_true")
    sp.add_argument("--gif",    action="store_true")

    # ── generate ──
    gp = sub.add_parser("generate", help="Generate synthetic solvable walls.")
    gp.add_argument("--count",       type=int,   default=1)
    gp.add_argument("--cols",        type=int,   default=12)
    gp.add_argument("--rows",        type=int,   default=18)
    gp.add_argument("--cell-size-cm", type=float, default=20.0)
    gp.add_argument("--difficulty",  type=float, default=0.5,
                    help="0.0 = easy (jugs) … 1.0 = hard (crimps/slopers)")
    gp.add_argument("--extra-holds", type=int,   default=8)
    gp.add_argument("--height-cm",   type=float, default=175.0)
    gp.add_argument("--wingspan-cm", type=float, default=175.0)
    gp.add_argument("--seed",        type=int,   default=None)
    gp.add_argument("--output-dir",  type=str,   default=None)
    gp.add_argument("--stdout",      action="store_true",
                    help="Print first wall to stdout instead of saving files.")

    # ── train ──
    tp = sub.add_parser("train", help="Train a MaskablePPO agent.")
    tp.add_argument("--timesteps",    type=int,   default=500_000)
    tp.add_argument("--n-envs",       type=int,   default=4)
    tp.add_argument("--height-cm",    type=float, default=175.0)
    tp.add_argument("--wingspan-cm",  type=float, default=175.0)
    tp.add_argument("--no-physics",   action="store_true")
    tp.add_argument("--no-curriculum", action="store_true")
    tp.add_argument("--seed",         type=int,   default=0)
    tp.add_argument("--tag",          type=str,   default="run")

    args = p.parse_args(args_raw)

    if args.cmd == "generate":
        from solver.generate import _cli
        return _cli(args_raw[1:])

    if args.cmd == "solve":
        return _solve(args_raw[1:])

    if args.cmd == "train":
        from solver.train import train
        return train(args_raw[1:])

    p.print_help()
    return 1


def _solve(argv: list[str]) -> int:
    p = argparse.ArgumentParser(prog="solver solve")
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
