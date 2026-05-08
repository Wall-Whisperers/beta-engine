"""Watch a random (or trained) agent interact with MoonBoardEnv in real time.

Usage — random agent (Day 4+):
    mjpython scripts/watch_random_agent.py
    mjpython scripts/watch_random_agent.py --episodes 5

Usage — trained SB3 policy (Week 2+):
    mjpython scripts/watch_random_agent.py --policy path/to/model.zip

Controls: close the viewer window or press Ctrl-C to stop.
"""

import argparse
import os
import sys
import time
import mujoco
try:
    import mujoco.viewer
except ImportError:
    from mujoco import viewer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from src.parsers import format1
from src.envs.moonboard_env import MoonBoardEnv

_MOONBOARD1 = os.path.join(_PROJECT_ROOT, "moonboard_data", "moonboard1.json")
_HUMANOID = os.path.join(_PROJECT_ROOT, "assets", "humanoid.xml")


def _select_route(routes):
    candidates = [r for r in routes if r.grade_v in (4, 5)]
    return max(candidates, key=lambda r: r.repeats)


def _load_policy(model_path: str):
    """Load a Stable-Baselines3 model from disk.

    Returns a callable policy(obs) → action, or raises ImportError/FileNotFoundError.
    """
    try:
        from stable_baselines3 import PPO
    except ImportError:
        raise ImportError("stable-baselines3 is required to load a trained policy.")
    model = PPO.load(model_path)
    print(f"[watch] Loaded SB3 policy from '{model_path}'")

    def policy(obs):
        action, _ = model.predict(obs, deterministic=True)
        return action

    return policy


def main():
    parser = argparse.ArgumentParser(description="Watch an agent on MoonBoardEnv.")
    parser.add_argument(
        "--policy",
        type=str,
        default=None,
        help="Path to a Stable-Baselines3 .zip model file.  "
             "Omit to use a random agent.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = loop forever, default=0).",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=True,
        help="Sleep between steps to run at real-time speed (default: on).",
    )
    parser.add_argument(
        "--no-realtime",
        dest="realtime",
        action="store_false",
        help="Run as fast as the viewer allows (no sleep).",
    )
    args = parser.parse_args()

    # ── Build environment ─────────────────────────────────────────────────────
    routes = format1.load_routes(_MOONBOARD1)
    route = _select_route(routes)
    print(f"[watch] Route: '{route.name}'  V{route.grade_v}  repeats={route.repeats}")

    env = MoonBoardEnv(route=route, humanoid_xml_path=_HUMANOID)
    step_dt = env._model.opt.timestep * env._sim_substeps  # seconds per policy step

    # ── Load policy or use random ─────────────────────────────────────────────
    if args.policy:
        policy = _load_policy(args.policy)
        agent_label = f"SB3 policy ({os.path.basename(args.policy)})"
    else:
        policy = lambda obs: env.action_space.sample()  # noqa: E731
        agent_label = "random agent"

    print(f"[watch] Agent:  {agent_label}")
    print(f"[watch] Policy period: {step_dt*1000:.1f} ms  |  real-time: {args.realtime}")
    if args.episodes:
        print(f"[watch] Episodes: {args.episodes}")
    else:
        print("[watch] Episodes: ∞  (Ctrl-C or close window to stop)")
    print()

    # ── Launch passive viewer ─────────────────────────────────────────────────

    obs, _ = env.reset()
    ep = 0
    ep_step = 0
    ep_reward = 0.0

    with mujoco.viewer.launch_passive(env._model, env._data) as viewer:
        # az=235 places camera diagonally front-right of the wall face.
        # Verified: sees wall face, holds, humanoid, and overhang angle.
        viewer.cam.lookat[:] = [0.0, -0.3, 1.5]
        viewer.cam.distance  = 6.0
        viewer.cam.azimuth   = 235
        viewer.cam.elevation = -20

        print(f"  Episode {ep + 1} started")

        try:
            while viewer.is_running():
                action = policy(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                ep_step += 1

                viewer.sync()

                if args.realtime:
                    time.sleep(step_dt)

                if terminated or truncated:
                    reason = (
                        "fell"      if info.get("fall_penalty", 0) < 0 else
                        "finished"  if info.get("finish_bonus", 0) > 0 else
                        "truncated"
                    )
                    print(
                        f"  Episode {ep + 1} done — "
                        f"steps={ep_step}  reward={ep_reward:.2f}  reason={reason}"
                    )
                    ep += 1
                    if args.episodes and ep >= args.episodes:
                        print(f"[watch] Reached {args.episodes} episode(s). Stopping.")
                        break

                    obs, _ = env.reset()
                    ep_step = 0
                    ep_reward = 0.0
                    print(f"  Episode {ep + 1} started")

        except KeyboardInterrupt:
            print("\n[watch] Interrupted.")
        finally:
            env.close()

    print("[watch] Viewer closed.")


if __name__ == "__main__":
    main()
