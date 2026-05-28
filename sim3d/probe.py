"""Reward-shaping probe — run scripted policies, dump per-component reward.

Run this BEFORE launching a 500k-step training run to verify the reward
function is sensibly shaped. A workable reward should pass these checks:

    1. "hold-tight" (max-positive grip intents, zero joint targets)
       earns SIGNIFICANTLY more total reward than "do-nothing".
    2. "release-all" (max-negative grip intents) earns LESS reward than
       "do-nothing" and ideally goes terminal-on-fall quickly.
    3. Reward components individually make sense — survival_reward
       accumulates for hold-tight, upward_reward fires when the body
       moves up, energy_penalty bites only for high-ctrl policies.

If hold-tight doesn't beat do-nothing, no amount of PPO will discover
"hang on the wall" — the reward gradient doesn't point there.

Usage:
    python -m sim3d.probe                       # default baby-v1 wall
    python -m sim3d.probe --wall my-wall
    python -m sim3d.probe --steps 500           # longer rollouts
"""
from __future__ import annotations

import argparse
import time
from typing import Callable

import numpy as np

from solver.wall import load_wall
from sim3d import Climb3DWorld, ClimberProfile
from sim3d.env import Climbing3DEnv, EnvConfig


# All scripted policies share this action layout (continuous-joint):
#   action[:21]  = normalised joint targets in [-1, 1]
#   action[21:]  = grip intents [LH, RH, LF, RF]
N_ACT = 21
N_GRIP = 4


def policy_do_nothing(t: int) -> np.ndarray:
    """All zeros. Joint targets at midrange, grip intents in deadband."""
    return np.zeros(N_ACT + N_GRIP, dtype=np.float32)


def policy_hold_tight(t: int) -> np.ndarray:
    """Zero joint targets + maximally positive grip intents.

    Grip intents > deadband re-engage any limb within proximity of a hold
    and prevent the random-release problem at episode start. This is the
    minimum-viable 'hang from the seed pose' policy."""
    a = np.zeros(N_ACT + N_GRIP, dtype=np.float32)
    a[N_ACT:] = 1.0
    return a


def policy_release_all(t: int) -> np.ndarray:
    """Zero joint targets + maximally negative grip intents.

    Should release every grip on step 1 and immediately drop into a fall."""
    a = np.zeros(N_ACT + N_GRIP, dtype=np.float32)
    a[N_ACT:] = -1.0
    return a


def policy_random(t: int) -> np.ndarray:
    """Uniform random — what PPO sees in early rollouts before learning."""
    rng = np.random.default_rng(t)
    return rng.uniform(-0.5, 0.5, size=N_ACT + N_GRIP).astype(np.float32)


def policy_leg_push(t: int) -> np.ndarray:
    """Hold grips tight, push legs to extend hips/knees → stand up.

    This is the *intended* solution to baby-v1: stand up on the footholds
    while gripping the start jugs, then the body is high enough that one
    hand can reach the finish."""
    a = np.zeros(N_ACT + N_GRIP, dtype=np.float32)
    a[N_ACT:] = 1.0   # grip everything tight
    # Joint target indices depend on actuator order; bias the first 21
    # toward "extend" without knowing exact indices by setting hip/knee/
    # ankle joints to a positive value. This is approximate — the probe
    # isn't trying to actually climb, just to test the reward signal under
    # a "tries to push up" pose.
    a[:N_ACT] = 0.3
    return a


POLICIES: dict[str, Callable[[int], np.ndarray]] = {
    "do-nothing":   policy_do_nothing,
    "hold-tight":   policy_hold_tight,
    "release-all":  policy_release_all,
    "random":       policy_random,
    "leg-push":     policy_leg_push,
}


# Reward component names exposed by Climbing3DEnv.step() info dict.
REWARD_KEYS = (
    "height_reward",
    "upward_reward",
    "approach_reward",
    "survival_reward",
    "match_bonus",
    "release_penalty",
    "energy_penalty",
)


def run_probe(env: Climbing3DEnv, policy: Callable[[int], np.ndarray],
              steps: int) -> dict:
    """Run one scripted-policy rollout and return aggregate stats."""
    obs, info = env.reset(seed=0)
    totals = {k: 0.0 for k in REWARD_KEYS}
    total_reward = 0.0
    com_z_max = float(env.world.com()[2])
    com_z_start = com_z_max
    n_steps = 0
    outcome = "ran-to-end"
    for t in range(steps):
        a = policy(t)
        obs, r, term, trunc, info = env.step(a)
        total_reward += r
        for k in REWARD_KEYS:
            totals[k] += info.get(k, 0.0)
        com_z = float(env.world.com()[2])
        if com_z > com_z_max:
            com_z_max = com_z
        n_steps = t + 1
        if term or trunc:
            outcome = info.get("outcome", "?")
            break

    n_grips_final = sum(
        1 for l in ("LH", "RH", "LF", "RF")
        if env.world.on_hold(l) is not None
    )
    return {
        "total_reward": total_reward,
        "components":   totals,
        "steps":        n_steps,
        "outcome":      outcome,
        "com_z_start":  com_z_start,
        "com_z_final":  float(env.world.com()[2]),
        "com_z_max":    com_z_max,
        "n_grips_final": n_grips_final,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m sim3d.probe")
    p.add_argument("--wall", default="baby-v1")
    p.add_argument("--steps", type=int, default=200,
                   help="Max steps per rollout (probe terminates early on fall).")
    p.add_argument("--episode-steps", type=int, default=1000,
                   help="Env max_steps (for finish detection).")
    p.add_argument("--policies", nargs="+", default=list(POLICIES.keys()),
                   choices=list(POLICIES.keys()))
    # Reward coefficients — match training defaults so we probe the SAME
    # reward function PPO will see. Override per experiment.
    p.add_argument("--finish-approach-coeff", type=float, default=2.0)
    p.add_argument("--survival-bonus-coeff", type=float, default=0.5)
    p.add_argument("--grip-release-penalty", type=float, default=2.0)
    p.add_argument("--upward-velocity-coeff", type=float, default=50.0)
    p.add_argument("--energy-penalty-coeff", type=float, default=0.001)
    p.add_argument("--grip-deadband", type=float, default=0.15)
    args = p.parse_args(argv)

    wall = load_wall(args.wall)
    profile = ClimberProfile()
    cfg = EnvConfig(
        max_steps=args.episode_steps,
        enable_slip=False,
        finish_approach_coeff=args.finish_approach_coeff,
        survival_bonus_coeff=args.survival_bonus_coeff,
        grip_release_penalty=args.grip_release_penalty,
        upward_velocity_coeff=args.upward_velocity_coeff,
        energy_penalty_coeff=args.energy_penalty_coeff,
        grip_intent_deadband=args.grip_deadband,
    )

    print(f"Wall: {wall.name} ({len(wall.holds)} holds)")
    print(f"Reward config: approach={args.finish_approach_coeff} "
          f"survival={args.survival_bonus_coeff} "
          f"release_pen={args.grip_release_penalty} "
          f"upward={args.upward_velocity_coeff} "
          f"energy={args.energy_penalty_coeff} "
          f"deadband={args.grip_deadband}")
    print(f"Probe length: {args.steps} steps per policy\n")

    results: dict[str, dict] = {}
    t0 = time.time()
    for name in args.policies:
        env = Climbing3DEnv(wall, profile, config=cfg)
        results[name] = run_probe(env, POLICIES[name], args.steps)

    elapsed = time.time() - t0

    # ── Table output ─────────────────────────────────────────────────────
    col_w = max(len(n) for n in args.policies) + 2
    print(f"{'policy':<{col_w}} | "
          f"{'reward':>8} | "
          f"{'steps':>5} | "
          f"{'com_z_max':>9} | "
          f"{'com_z_end':>9} | "
          f"{'grips':>5} | "
          f"{'outcome':<10}")
    print("-" * (col_w + 70))
    for name in args.policies:
        r = results[name]
        print(f"{name:<{col_w}} | "
              f"{r['total_reward']:>+8.2f} | "
              f"{r['steps']:>5} | "
              f"{r['com_z_max']:>9.3f} | "
              f"{r['com_z_final']:>9.3f} | "
              f"{r['n_grips_final']:>5} | "
              f"{r['outcome']:<10}")

    # ── Per-component breakdown ──────────────────────────────────────────
    print(f"\nReward component sums (per policy):")
    print(f"{'policy':<{col_w}} | " +
          " | ".join(f"{k:>16}" for k in REWARD_KEYS))
    print("-" * (col_w + 19 * len(REWARD_KEYS)))
    for name in args.policies:
        comps = results[name]["components"]
        print(f"{name:<{col_w}} | " +
              " | ".join(f"{comps[k]:>+16.2f}" for k in REWARD_KEYS))

    # ── Sanity-check verdict ─────────────────────────────────────────────
    print(f"\nElapsed: {elapsed:.1f}s")
    if "hold-tight" in results and "do-nothing" in results:
        ht = results["hold-tight"]["total_reward"]
        dn = results["do-nothing"]["total_reward"]
        if ht > dn + 5.0:
            print(f"[OK]   hold-tight (+{ht:.1f}) beats do-nothing (+{dn:.1f}) "
                  "- reward shape points toward staying gripped.")
        else:
            print(f"[BAD]  hold-tight ({ht:+.1f}) does NOT beat do-nothing "
                  f"({dn:+.1f}). Reward gradient does not favour gripping; "
                  "fix it before training.")
    if "hold-tight" in results and "release-all" in results:
        ht = results["hold-tight"]["total_reward"]
        ra = results["release-all"]["total_reward"]
        if ht > ra + 5.0:
            print(f"[OK]   hold-tight (+{ht:.1f}) beats release-all "
                  f"({ra:+.1f}) - releasing is worse than holding.")
        else:
            print(f"[BAD]  hold-tight ({ht:+.1f}) does NOT beat release-all "
                  f"({ra:+.1f}) - release is not penalised enough.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
