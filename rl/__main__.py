"""CLI: run a random-policy episode through the Gymnasium climbing env.

Usage:

    python -m rl --wall example-v2-boulder
    python -m rl --wall example-v2-boulder --gif
    python -m rl --wall example-v2-boulder --policy random --episodes 5 --seed 42

Useful as a smoke-test that the env, physics, and renderer still hold
hands. For actual training, drop the env into Stable-Baselines3:

    from stable_baselines3 import PPO
    from rl.env import ClimbingEnv
    from solver.wall import load_wall
    env = ClimbingEnv(load_wall('example-v2-boulder'))
    PPO('MlpPolicy', env).learn(100_000)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from physics.body import ClimberProfile
from rl.env import ClimbingEnv, EnvConfig
from rl.random_policy import rollout
from solver.body import BodyModel
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
    p = argparse.ArgumentParser(prog="rl", description=__doc__)
    p.add_argument("--wall", required=True)
    p.add_argument("--policy", choices=("random", "masked"), default="masked",
                   help="random = uniform over all actions (incl. illegal); "
                        "masked = uniform over currently-legal actions only.")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--height-cm", type=float, default=175.0)
    p.add_argument("--wingspan-cm", type=float, default=175.0)
    p.add_argument("--mass-kg", type=float, default=70.0)
    p.add_argument("--cell-size-cm", type=float, default=None)
    p.add_argument("--gif", action="store_true",
                   help="render the LAST episode to a GIF in /data/runs/.")
    args = p.parse_args(argv)

    wall = load_wall(args.wall, cell_size_cm=args.cell_size_cm)
    body_model = BodyModel(height_cm=args.height_cm, wingspan_cm=args.wingspan_cm)
    profile = ClimberProfile(body=body_model, mass_kg=args.mass_kg)
    env = ClimbingEnv(
        wall, profile,
        env_config=EnvConfig(max_steps=args.max_steps),
    )

    print(f"Wall {wall.wall_id}: action space = {env.action_space.n} "
          f"(4 limbs × {len(wall.holds)} holds), "
          f"obs dim = {env.observation_space.shape[0]}")

    last_transcript = None
    rewards: list[float] = []
    for ep in range(args.episodes):
        total, transcript = rollout(
            env,
            masked=(args.policy == "masked"),
            seed=args.seed + ep,
            max_steps=args.max_steps,
            verbose=(args.episodes == 1),
        )
        rewards.append(total)
        last_transcript = transcript
        outcome = (
            "FINISH" if transcript and transcript[-1].get("reason") == "finish"
            else "timeout"
        )
        print(f"episode {ep+1}/{args.episodes}: total reward = {total:+.2f} "
              f"({len(transcript)} steps, {outcome})")

    if args.episodes > 1:
        import statistics as st
        print(f"mean reward = {st.mean(rewards):+.2f} ± "
              f"{st.stdev(rewards) if len(rewards) > 1 else 0:.2f}")

    # ── Optional GIF of the last episode ─────────────────────────────
    if args.gif and last_transcript:
        from physics.render import render_animation
        from physics.world import ClimbWorld

        # Replay the same actions on a fresh world so we can render the
        # trajectory frame-by-frame.
        world = ClimbWorld(wall, profile)
        starts = env._default_starts()
        world.seed_pose(**starts)

        moves = [
            (LIMB_NAMES[t["action"] // len(wall.holds)],
             wall.holds[t["action"] % len(wall.holds)].hold_id)
            for t in last_transcript if not t.get("illegal")
        ]
        frames_per_move = 12
        n_frames = max(30, frames_per_move * len(moves) + 30)

        def on_frame(world, idx):
            slot = idx // frames_per_move
            if slot < len(moves) and (idx % frames_per_move == 0):
                limb, hid = moves[slot]
                world.move_limb(limb, hid, mode="snap")

        gif_path = _runs_dir() / f"{wall.wall_id}-rl.gif"
        render_animation(world, gif_path, n_frames=n_frames, on_frame=on_frame)
        print(f"Wrote {gif_path}")

    return 0


# Constants used by the GIF replay above.
from physics.body import LIMBS as LIMB_NAMES  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())
