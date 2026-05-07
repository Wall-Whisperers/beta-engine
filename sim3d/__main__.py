"""`python -m sim3d` — open the 3D climbing simulator on a wall.

Usage:

    # Default: load the bundled example wall, open the native MuJoCo
    # viewer, hang the climber from the start holds, and step forever.
    python -m sim3d

    # A specific wall (resolved against /data/walls/<id>.json or
    # ./data/examples/<id>.json):
    python -m sim3d --wall my-wall

    # Headless mode — no viewer, just step and dump pose snapshots
    # (useful for CI / Docker / sanity checks):
    python -m sim3d --headless --frames 60

    # Override climber dimensions (cm):
    python -m sim3d --height 190 --wingspan 195

    # Run a scripted move sequence as a smoke test:
    python -m sim3d --beta h_003 h_006 RH:h_008 LF:h_005

The `--beta` flag is a quick way to drive the simulator without the
solver attached: each token is either a hold ID (next limb in
LH→RH→LF→RF order) or `LIMB:hold_id` to specify the limb explicitly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Iterable

from solver.wall import load_wall
from sim3d import Climb3DWorld, ClimberProfile
from sim3d.body import LIMBS


def parse_beta(tokens: Iterable[str]) -> list[tuple[str, str]]:
    """Parse the --beta argument into a list of (limb, hold_id) moves.

    Token forms:
        "h_006"       → use the next limb in round-robin order
        "RH:h_008"    → specify both limb and hold
    """
    moves: list[tuple[str, str]] = []
    auto_order = ["LH", "RH", "LF", "RF"]
    auto_idx = 0
    for tok in tokens:
        if ":" in tok:
            limb, hold_id = tok.split(":", 1)
            limb = limb.upper()
            if limb not in LIMBS:
                raise SystemExit(f"unknown limb {limb!r}; expected one of {LIMBS}")
            moves.append((limb, hold_id))
        else:
            limb = auto_order[auto_idx % len(auto_order)]
            auto_idx += 1
            moves.append((limb, tok))
    return moves


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sim3d")
    p.add_argument(
        "--wall", default="example-v2-boulder",
        help="Wall ID or path to a wall JSON.",
    )
    p.add_argument("--height", type=float, default=175.0,
                   help="Climber total height in cm.")
    p.add_argument("--wingspan", type=float, default=175.0,
                   help="Climber wingspan in cm.")
    p.add_argument("--mass", type=float, default=70.0,
                   help="Climber total mass in kg.")
    p.add_argument(
        "--headless", action="store_true",
        help="Don't open the native viewer; step and exit.",
    )
    p.add_argument(
        "--frames", type=int, default=120,
        help="Number of render frames to step in headless mode.",
    )
    p.add_argument(
        "--duration", type=float, default=60.0,
        help="Real-time seconds to run the viewer (default: 60).",
    )
    p.add_argument(
        "--beta", nargs="*", default=[],
        help="Scripted move sequence (e.g. 'h_006 RH:h_008'). "
             "Each move runs after the previous one settles.",
    )
    p.add_argument(
        "--snapshot", action="store_true",
        help="Print one pose-snapshot JSON object to stdout after "
             "seed_pose. Useful for piping into the web viewer for tests.",
    )
    args = p.parse_args(argv)

    wall = load_wall(args.wall)
    profile = ClimberProfile(
        height_cm=args.height,
        wingspan_cm=args.wingspan,
        mass_kg=args.mass,
    )
    world = Climb3DWorld(wall, profile)

    starts = wall.starts()
    foots = [h for h in wall.holds if h.hold_type == "foothold"][:2]
    if len(starts) >= 2 and len(foots) >= 2:
        world.seed_pose(
            lh=starts[0].hold_id, rh=starts[1].hold_id,
            lf=foots[0].hold_id, rf=foots[1].hold_id,
        )
    else:
        # No starts marked — just hang at the bottom-most two holds.
        bottom = sorted(wall.holds, key=lambda h: h.y_cm)[:4]
        if len(bottom) >= 4:
            world.seed_pose(
                lh=bottom[2].hold_id, rh=bottom[3].hold_id,
                lf=bottom[0].hold_id, rf=bottom[1].hold_id,
            )

    if args.snapshot:
        snap = world.pose_snapshot()
        json.dump(snap, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0

    moves = parse_beta(args.beta)

    if args.headless:
        for _ in range(args.frames):
            world.step()
        for limb, hold_id in moves:
            print(f"  → {limb} → {hold_id}")
            world.move_limb(limb, hold_id, mode="snap")
            for _ in range(int(0.5 * 60)):  # 0.5 s settle
                world.step()
        print("final pelvis:", world.pelvis_pos())
        print("final COM:   ", world.com())
        return 0

    # ── Native viewer with optional scripted beta ────────────────────
    move_iter = iter(moves)
    next_move_at = 2.0  # seconds before the first scripted move
    pending_move: tuple[str, str] | None = next(move_iter, None)

    def on_frame(w: Climb3DWorld, t: float) -> None:
        nonlocal pending_move, next_move_at
        if pending_move is not None and t >= next_move_at:
            limb, hid = pending_move
            print(f"[t={t:5.1f}s] {limb} → {hid}")
            w.move_limb(limb, hid, mode="reach")
            pending_move = next(move_iter, None)
            next_move_at = t + 1.5

    from sim3d.viewer import run_demo
    print(f"Loaded wall '{wall.name}' — {len(wall.holds)} holds, "
          f"angle {wall.wall_angle_deg}°")
    print(f"Climber: {profile.height_cm:.0f} cm / {profile.wingspan_cm:.0f} cm wingspan / {profile.mass_kg:.0f} kg")
    print("Native MuJoCo viewer running. Close the window or Ctrl+C to quit.")
    try:
        run_demo(world, duration_s=args.duration, on_frame=on_frame)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
