"""CLI demo: drop a climber on a wall, run the physics simulation,
optionally execute a sequence of moves, render to PNG / GIF.

Usage examples (run inside the docker container or in a local venv):

    # Just settle on the starting pose, render a still image
    python -m physics --wall example-v2-boulder

    # Settle + animate 3 seconds of physics (no moves)
    python -m physics --wall example-v2-boulder --frames 90 --gif

    # Plan a beta with the existing solver, then play the moves through
    # the physics engine and animate the whole thing.
    python -m physics --wall example-v2-boulder --solve --gif

The output goes to /data/runs/<wall>-physics.{png,gif} inside the
container, which maps to ./data/runs/ on the host.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from physics.body import ClimberProfile
from physics.render import render_animation, render_still
from physics.world import ClimbWorld
from solver.body import BodyModel
from solver.wall import load_wall


def _runs_dir() -> Path:
    # Inside Docker the /data volume is bind-mounted to ./data/ on the host.
    # Outside Docker we write directly into the repo's data/runs/ so the
    # user can find the files next to their code.
    in_docker = Path("/.dockerenv").exists()
    if in_docker:
        d = Path("/data/runs")
    else:
        d = Path(__file__).resolve().parent.parent / "data" / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _starting_holds(wall) -> tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Pick a sensible 4-limb starting pose.

    - Hands on `is_start` holds (left-most → LH, right-most → RH).
    - Feet on the two lowest foot-usable holds.
    """
    starts = wall.starts()
    foots = sorted(
        [h for h in wall.holds if h.usable_for_foot()],
        key=lambda h: h.y_cm,
    )
    if len(starts) >= 2:
        sorted_starts = sorted(starts, key=lambda h: h.x_cm)
        lh, rh = sorted_starts[0].hold_id, sorted_starts[-1].hold_id
    elif len(starts) == 1:
        lh = rh = starts[0].hold_id
    else:
        # No start markers — pick the two lowest hand-usable holds.
        hand_low = sorted(
            [h for h in wall.holds if h.usable_for_hand()],
            key=lambda h: h.y_cm,
        )[:2]
        if len(hand_low) >= 2:
            l, r = sorted(hand_low, key=lambda h: h.x_cm)
            lh, rh = l.hold_id, r.hold_id
        else:
            lh = rh = None
    if len(foots) >= 2:
        l, r = sorted(foots[:2], key=lambda h: h.x_cm)
        lf, rf = l.hold_id, r.hold_id
    else:
        lf = rf = None
    return lh, rh, lf, rf


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="physics", description=__doc__)
    p.add_argument("--wall", required=True,
                   help="wall_id or path to a JSON file")
    p.add_argument("--height-cm", type=float, default=175.0)
    p.add_argument("--wingspan-cm", type=float, default=175.0)
    p.add_argument("--mass-kg", type=float, default=70.0)
    p.add_argument("--cell-size-cm", type=float, default=None)
    p.add_argument("--frames", type=int, default=60,
                   help="number of rendered frames (gif mode only)")
    p.add_argument("--gif", action="store_true",
                   help="render an animated GIF instead of a still PNG")
    p.add_argument("--solve", action="store_true",
                   help="plan a beta with solver A* and play it through "
                        "the physics engine (falls back to --moves if "
                        "the solver returns nothing)")
    p.add_argument("--moves",
                   help="comma-separated 'LIMB:hold_id' pairs, e.g. "
                        "'RF:h_005,LF:h_007,RH:h_008,LH:h_006,RH:h_013'. "
                        "Lets you script a beta directly; useful when "
                        "the solver can't find one yet.")
    p.add_argument("--frames-per-move", type=int, default=20,
                   help="how many rendered frames each move occupies")
    args = p.parse_args(argv)

    wall = load_wall(args.wall, cell_size_cm=args.cell_size_cm)
    body_model = BodyModel(height_cm=args.height_cm, wingspan_cm=args.wingspan_cm)
    profile = ClimberProfile(body=body_model, mass_kg=args.mass_kg)

    print(f"Wall: {wall.name} ({wall.cols}×{wall.rows} grid, "
          f"cell={wall.cell_size_cm:.1f} cm, angle={wall.wall_angle_deg:.1f}°, "
          f"{len(wall.holds)} holds)")
    print(f"Climber: height={body_model.height_cm:.0f} cm, "
          f"wingspan={body_model.wingspan_cm:.0f} cm, mass={profile.mass_kg:.0f} kg")

    world = ClimbWorld(wall, profile)
    lh, rh, lf, rf = _starting_holds(wall)
    print(f"Seeding pose: LH={lh} RH={rh} LF={lf} RF={rf}")
    world.seed_pose(lh=lh, rh=rh, lf=lf, rf=rf)

    runs = _runs_dir()

    # Build the move plan: --moves wins, then --solve, else just settle.
    move_plan: list[tuple[str, str]] = []
    if args.moves:
        for tok in args.moves.split(","):
            tok = tok.strip()
            if not tok:
                continue
            limb, hid = tok.split(":")
            move_plan.append((limb.strip(), hid.strip()))
        print(f"using --moves: {len(move_plan)} hand-rolled steps.")
    elif args.solve:
        from solver.astar import solve_astar
        result = solve_astar(wall, body_model)
        if result is None:
            print("solver: no beta found — pass --moves to script one manually.")
        else:
            move_plan = list(result.moves)
            print(f"solver: planned {len(move_plan)} moves.")
            for i, (limb, target) in enumerate(move_plan, 1):
                print(f"  step {i}: {limb} → {target}")

    # ── Output ─────────────────────────────────────────────────────────
    if not args.gif:
        # Settle the body for a moment, then snapshot.
        world.step(60)
        png_path = runs / f"{wall.wall_id}-physics.png"
        print(f"Saving PNG → {png_path.resolve()}")
        render_still(world, png_path)
        print(f"Wrote {png_path.resolve()}")
        return 0

    # GIF mode: play the move plan over the chosen number of frames.
    if move_plan:
        # Allocate `frames_per_move` to each move + a settle window at end.
        n_frames = args.frames_per_move * len(move_plan) + 30
    else:
        n_frames = args.frames

    move_state = {"idx": 0, "applied": -1}

    def on_frame(world, frame_idx):
        if not move_plan:
            return
        # Schedule one move at the start of each move window.
        target_idx = frame_idx // args.frames_per_move
        if target_idx < len(move_plan) and target_idx != move_state["applied"]:
            limb, target = move_plan[target_idx]
            print(f"  frame {frame_idx}: {limb} → {target}")
            world.move_limb(limb, target, mode="snap")
            move_state["applied"] = target_idx

    gif_path = runs / f"{wall.wall_id}-physics.gif"
    print(f"Saving GIF → {gif_path.resolve()}")
    render_animation(
        world, gif_path,
        n_frames=n_frames,
        on_frame=on_frame,
    )
    print(f"Wrote {gif_path.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
