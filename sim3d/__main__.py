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
from pathlib import Path
from typing import Iterable

from solver.wall import load_wall
from sim3d import Climb3DWorld, ClimberProfile
from sim3d.body import LIMBS


def replay_command_from_run_config(model_path: str) -> str | None:
    """Return a replay command reconstructed from a training run config.

    Older training output could suggest replaying a MoonBoard-trained model
    with ``--wall``. When the model lives beside ``config.json``, use that
    config to show the wall/source and profile that match the saved policy.
    """
    cfg_path = Path(model_path).with_name("config.json")
    if not cfg_path.exists():
        return None
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    cmd = f"python -m sim3d --play {model_path}"
    if cfg.get("moonboard_file"):
        cmd += f" --moonboard {cfg['moonboard_file']}"
        if cfg.get("moonboard_problem_id") is not None:
            cmd += f" --problem {cfg['moonboard_problem_id']}"
        else:
            split_path = cfg_path.with_name("moonboard_splits.json")
            if split_path.exists():
                try:
                    splits = json.loads(split_path.read_text(encoding="utf-8"))
                    split_name = str(cfg.get("moonboard_split", "train"))
                    selected = splits.get(split_name) or splits.get("train") or []
                    if selected:
                        cmd += f" --problem {selected[0]['id']}"
                except (OSError, KeyError, TypeError, json.JSONDecodeError):
                    pass
            cmd += " --moonboard-full-board"
    else:
        cmd += f" --wall {cfg.get('wall', 'example-v2-boulder')}"

    for arg, key in (
        ("height", "height_cm"),
        ("wingspan", "wingspan_cm"),
        ("mass", "mass_kg"),
    ):
        if key in cfg:
            cmd += f" --{arg} {cfg[key]}"
    if cfg.get("move_mode"):
        cmd += f" --move-mode {cfg['move_mode']}"
    if cfg.get("start_mode") and cfg.get("start_mode") != "seed":
        cmd += f" --start-mode {cfg['start_mode']}"
    if cfg.get("move_frames"):
        cmd += f" --play-frames {cfg['move_frames']}"
    return cmd

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
    p.add_argument(
        "--moonboard", metavar="PATH",
        help="Path to a MoonBoard problem-list JSON. Overrides --wall.",
    )
    p.add_argument(
        "--problem", type=int, metavar="ID",
        help="MoonBoard problem id to load (with --moonboard).",
    )
    p.add_argument(
        "--problem-name", metavar="NAME",
        help="Substring of MoonBoard problem name (with --moonboard). "
             "First match wins. Use this if you don't have the id.",
    )
    p.add_argument(
        "--moonboard-full-board", action="store_true",
        help="MoonBoard: include all 198 T-nut positions. Required to replay generalized split-trained policies.",
    )
    p.add_argument(
        "--vertical-projection", action="store_true",
        help="MoonBoard: lay holds out so each row's WORLD-Z spacing "
             "equals the cell size — looks like the photo of a vertical "
             "MoonBoard, even when the wall is tilted at 40°. Default "
             "(off) keeps holds on the actual angled surface (physically "
             "real but visually denser when tilted).",
    )
    p.add_argument(
        "--gym", action="store_true",
        help="Run a Gymnasium random-policy episode instead of the viewer. "
             "Useful sanity check for the env wrapper.",
    )
    p.add_argument(
        "--gym-episodes", type=int, default=1,
        help="Number of episodes when --gym is set.",
    )
    p.add_argument(
        "--slip", action="store_true",
        help="Enable slip detection (welds release when force exceeds capacity).",
    )
    p.add_argument(
        "--play", metavar="MODEL.zip",
        help="Run a saved SB3 PPO model in the viewer instead of "
             "manual controls. Pair with --wall or --moonboard. "
             "Installed by `pip install -r requirements.txt`.",
    )
    p.add_argument(
        "--play-frames", type=int, default=120,
        help="Render frames per policy action when replaying (default 120 = 2 s).",
    )
    p.add_argument(
        "--slowmo", type=float, default=1.0,
        help="Replay slow-motion factor. 1.0=real time, 4.0=4x slower, "
             "10.0=ultra slow. Each policy action's physics is broken into "
             "single-frame chunks with viewer.sync + sleep between them.",
    )
    p.add_argument(
        "--move-mode", default="reach", choices=("snap", "reach", "dyno"),
        help="Limb-move mode for scripted betas and policy replay.",
    )
    p.add_argument(
        "--start-mode", default="seed", choices=("seed", "ground-reach"),
        help="seed = start welded on route holds; ground-reach = start on floor and reach to start hand holds.",
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

    if args.moonboard:
        from sim3d.moonboard import (
            load_moonboard_problems, moonboard_problem_to_wall, find_problem,
        )
        problems = load_moonboard_problems(args.moonboard)
        if args.problem is not None:
            problem = find_problem(problems, id=args.problem)
            if problem is None:
                raise SystemExit(f"problem id {args.problem} not found in {args.moonboard}")
        elif args.problem_name is not None:
            problem = find_problem(problems, name=args.problem_name)
            if problem is None:
                raise SystemExit(f"no problem matches name {args.problem_name!r}")
        else:
            problem = problems[0]
            print(f"(picked first problem: #{problem.id} {problem.name!r}; "
                  f"use --problem ID or --problem-name to choose)")
        wall = moonboard_problem_to_wall(
            problem,
            include_full_board=args.moonboard_full_board,
            vertical_projection=args.vertical_projection,
        )
        proj_note = " (vertical-projection)" if args.vertical_projection else ""
        print(f"MoonBoard: \"{problem.name}\" by {problem.setter} — V{problem.grade}{proj_note}")
    else:
        wall = load_wall(args.wall)
    profile = ClimberProfile(
        height_cm=args.height,
        wingspan_cm=args.wingspan,
        mass_kg=args.mass,
    )
    world = Climb3DWorld(wall, profile)

    def _start_hand_targets() -> tuple[str | None, str | None]:
        starts = sorted(wall.starts(), key=lambda h: h.x_cm)
        if len(starts) >= 2:
            return starts[0].hold_id, starts[-1].hold_id
        if len(starts) == 1:
            return starts[0].hold_id, starts[0].hold_id
        hand_low = sorted(
            [h for h in wall.holds if h.usable_for_hand()],
            key=lambda h: (h.y_cm, h.x_cm),
        )[:2]
        if len(hand_low) >= 2:
            return hand_low[0].hold_id, hand_low[-1].hold_id
        if len(hand_low) == 1:
            return hand_low[0].hold_id, hand_low[0].hold_id
        return None, None

    if args.start_mode == "ground-reach":
        lh, rh = _start_hand_targets()
        world._sync_actuator_targets_to_pose()
        if lh is not None:
            world.move_limb("LH", lh, mode=args.move_mode)
        if rh is not None:
            world.move_limb("RH", rh, mode=args.move_mode)
    else:
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

    if args.gym:
        # Gym random-policy episode(s) — useful smoke test for the env.
        from sim3d.env import Climbing3DEnv, EnvConfig
        env = Climbing3DEnv(
            wall, profile,
            config=EnvConfig(
                max_steps=30,
                move_mode=args.move_mode,
                start_mode=args.start_mode,
                enable_slip=args.slip,
                official_route_only=args.moonboard_full_board,
            ),
        )
        for ep in range(args.gym_episodes):
            obs, info = env.reset()
            ep_reward = 0.0
            for _ in range(env.cfg_env.max_steps):
                a = env.action_space.sample()
                obs, r, term, trunc, info = env.step(a)
                ep_reward += r
                if term or trunc:
                    break
            print(f"episode {ep+1}: reward={ep_reward:+.2f}, "
                  f"outcome={info.get('outcome')}, com_z={info['com'][2]:.2f}")
        return 0

    if args.play:
        # Replay a trained policy in the native viewer.
        from sim3d.env import Climbing3DEnv, EnvConfig
        from sim3d.train import _require_sb3
        sb3, _, _, _ = _require_sb3()
        env = Climbing3DEnv(
            wall, profile,
            config=EnvConfig(
                max_steps=30,
                move_mode=args.move_mode,
                start_mode=args.start_mode,
                move_frames=args.play_frames,
                enable_slip=args.slip,
                official_route_only=args.moonboard_full_board,
            ),
        )
        model = sb3.PPO.load(args.play)
        model_obs_shape = getattr(model.observation_space, "shape", None)
        env_obs_shape = getattr(env.observation_space, "shape", None)
        if model_obs_shape != env_obs_shape:
            expected = model_obs_shape[0] if model_obs_shape else model_obs_shape
            actual = env_obs_shape[0] if env_obs_shape else env_obs_shape
            hint = replay_command_from_run_config(args.play)
            hint_msg = f"\nTry the saved run config replay command:\n  {hint}" if hint else ""
            raise SystemExit(
                "Saved policy is incompatible with the replay environment: "
                f"model expects observation shape {model_obs_shape}, but "
                f"the selected wall/config produces {env_obs_shape}.\n"
                "For this env, observation size changes with the number of holds "
                "(four limb-on-hold one-hot vectors), climber model/action mode, "
                "and other training config. Replay with the exact wall/config used "
                "for training, or retrain the model. "
                f"Observed dimensions: model={expected}, replay_env={actual}."
                f"{hint_msg}"
            )
        if getattr(model.action_space, "n", None) != getattr(env.action_space, "n", None):
            hint = replay_command_from_run_config(args.play)
            hint_msg = f" Try: {hint}" if hint else ""
            raise SystemExit(
                "Saved policy action space is incompatible with the replay "
                f"environment: model={model.action_space}, env={env.action_space}. "
                "Replay with the exact training env or retrain."
                f"{hint_msg}"
            )
        print(f"Loaded policy from {args.play}")
        print(f"Wall: {wall.name} ({len(wall.holds)} holds)")
        print("Native viewer running. Each policy action takes "
              f"{args.play_frames} frames (~{args.play_frames/60:.1f}s).")

        # Drive the env's world (which the viewer attaches to).
        from sim3d.viewer import native_viewer
        from sim3d import config as cfg
        obs, info = env.reset()
        # One env.step() advances sim_substeps (default 8) render frames
        # of physics. For smooth slow-motion we want the viewer to refresh
        # *during* that physics chunk, not just at the end. So in slow-mo
        # mode we manually apply the action's ctrl + grip intents (mirror
        # of Climbing3DEnv._step_continuous) and then call world.step(1)
        # in a tight loop with viewer.sync() + sleep between frames.
        slowmo = max(1.0, float(args.slowmo))
        substeps = env.cfg_env.sim_substeps
        frame_dt = 1.0 / cfg.RENDER_HZ
        sleep_per_frame = (slowmo - 1.0) * frame_dt
        with native_viewer(env.world) as viewer:
            done = False
            while viewer.is_running() and not done:
                action, _ = model.predict(obs, deterministic=True)
                if env.cfg_env.action_mode == "discrete-move":
                    limb, hold_id = env.decode_move(int(action))
                    print(f"  policy: {limb} → {hold_id}")
                    obs, r, term, trunc, info = env.step(action)
                    viewer.sync()
                    if sleep_per_frame > 0:
                        time.sleep(sleep_per_frame * substeps)
                elif slowmo > 1.0:
                    # Slow-mo: apply ctrl + grip intents, then step one
                    # render frame at a time so the viewer can refresh
                    # mid-physics-chunk.
                    import numpy as np
                    from sim3d.body import LIMBS
                    a = np.asarray(action, dtype=np.float64)
                    n = env._n_act
                    joint_norm = np.clip(a[:n], -1.0, 1.0)
                    ctrl = (0.5 * (joint_norm + 1.0)
                            * (env._act_hi - env._act_lo) + env._act_lo)
                    env.world.data.ctrl[:n] = ctrl
                    db = env.cfg_env.grip_intent_deadband
                    for i, limb in enumerate(LIMBS):
                        intent = float(a[n + i]) if n + i < len(a) else 0.0
                        if intent > db:
                            env._maybe_engage_grip(limb)
                        elif intent < -db:
                            if env.world.on_hold(limb) is not None:
                                env.world.release_limb(limb)
                    for _ in range(substeps):
                        env.world.step(1, check_slip=env.cfg_env.enable_slip)
                        viewer.sync()
                        if sleep_per_frame > 0:
                            time.sleep(sleep_per_frame)
                    # Bookkeeping step (no extra physics) — env.step would
                    # advance physics again, so build obs/info manually:
                    obs = env._obs()
                    # Update HWM + done flags by calling env.step with a
                    # neutral action would re-physics; instead just check
                    # the basic done conditions.
                    pelvis_z = float(env.world.pelvis_pos()[2])
                    done = pelvis_z < 0.20
                    info = {"com": env.world.com(), "outcome": "running"}
                    if done:
                        info["outcome"] = "fell"
                else:
                    obs, r, term, trunc, info = env.step(action)
                    viewer.sync()
                    done = term or trunc
            print(f"Outcome: {info.get('outcome')} | "
                  f"final COM_z = {info['com'][2]:.2f} m")
        return 0

    if args.headless:
        slip_total = world.step(args.frames, check_slip=args.slip)
        for limb, hold_id in moves:
            print(f"  → {limb} → {hold_id}")
            world.move_limb(limb, hold_id, mode="snap")
            slip_total += world.step(30, check_slip=args.slip)  # 0.5 s settle
        print(f"final pelvis: {world.pelvis_pos()}")
        print(f"final COM:    {world.com()}")
        if args.slip:
            print(f"slip events:  {slip_total}")
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
