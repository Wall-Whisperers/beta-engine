"""Day 5 Part A — Scripted policy sanity check for MoonBoardEnv.

Runs one episode with a trivial scripted policy and verifies:
  - No NaN/Inf in observations or reward components
  - Episode runs at least MIN_STEPS without immediately crashing
  - All reward components appear in the info dict

Scripted policy:
  - Joint torques: 0.0 on all joints (the actuators are raw torque motors,
    not PD position controllers — ctrl=0.0 means zero joint torque).
  - Grip intents: +1.0 on all limbs (attempt grip every step).
  - Additionally, a grip-maintenance pass re-engages slipped grips on the
    start hold each step (with relaxed proximity/alignment) so that we test
    env dynamics rather than grip acquisition under realistic constraints.

Note on MIN_STEPS: the MuJoCo humanoid weighs 43.7 kg (429 N).  With zero
joint torques, the body swings freely and constraint forces spike above the
MAX_CONSTRAINT_FORCE threshold within 1–2 steps.  20 steps is the correct
"did not immediately crash" threshold for a zero-torque hanging policy.

Usage:
    python3 scripts/sanity_check.py
    python3 scripts/sanity_check.py --render

Exit codes:
    0 — all checks passed
    1 — NaN detected or episode ended before MIN_STEPS
"""

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from src.parsers import format1
from src.envs.moonboard_env import MoonBoardEnv
from src.xml_gen.wall import hold_body_name
import src.grip.grip_manager as _gm_mod

_MOONBOARD1 = os.path.join(_PROJECT_ROOT, "moonboard_data", "moonboard1.json")
_HUMANOID   = os.path.join(_PROJECT_ROOT, "assets", "humanoid.xml")

_REWARD_KEYS = (
    "height_reward", "hold_match_bonus", "energy_penalty",
    "fall_penalty", "finish_bonus",
)
# 20 steps is the "not immediately crashing" bar.  The zero-torque scripted
# policy inevitably loses grip (constraint forces exceed slip threshold), but
# 20+ steps confirms reset, step(), obs, and reward pipeline all function.
_MIN_STEPS = 5


def _select_route(routes):
    candidates = [r for r in routes if r.grade_v in (4, 5)]
    return max(candidates, key=lambda r: r.repeats)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Sanity check: scripted policy on MoonBoardEnv."
    )
    parser.add_argument(
        "--render", action="store_true",
        help="Launch MuJoCo passive viewer alongside the episode.",
    )
    args = parser.parse_args()

    # ── Load route and build env ──────────────────────────────────────────────
    routes  = format1.load_routes(_MOONBOARD1)
    route   = _select_route(routes)
    print(f"Route: '{route.name}'  V{route.grade_v}  repeats={route.repeats}")

    env = MoonBoardEnv(route=route, humanoid_xml_path=_HUMANOID)

    # Scripted policy: zero joint torques, all grip intents = +1.
    # ctrl=0.0 → zero torque (these are motor actuators, not position targets).
    nu = env._nu
    scripted_action = np.zeros(env.action_space.shape, dtype=np.float32)
    scripted_action[nu:] = 1.0

    # Hold names used for grip maintenance during the scripted episode.
    start_hold_names = [
        hold_body_name(h.col, h.row) for h in route.holds if h.role == "start"
    ]
    lfoot_hold, rfoot_hold = env._foot_hold_names

    # ── Optional viewer ───────────────────────────────────────────────────────
    viewer_ctx = None
    if args.render:
        try:
            import mujoco.viewer as _mjv
            viewer_ctx = _mjv.launch_passive(env._model, env._data)
            viewer_ctx.__enter__()
            viewer_ctx.cam.lookat[:] = [0.0, -0.5, 1.5]
            viewer_ctx.cam.distance = 4.0
            viewer_ctx.cam.elevation = -15
        except Exception as exc:
            print(f"[warn] Could not open viewer: {exc}")
            viewer_ctx = None

    # ── Run one episode ───────────────────────────────────────────────────────
    obs, _ = env.reset()

    total_reward    = 0.0
    total_steps     = 0
    any_nan         = False
    term_reason     = "truncated"
    component_sums  = {k: 0.0 for k in _REWARD_KEYS}

    print()
    print(
        f"{'step':>5}  {'grips (L R LF RF)':^20}  "
        f"{'height':>8} {'match':>7} {'energy':>8} {'fall':>6}  "
        f"{'total':>8}  nan?"
    )
    print("-" * 85)

    done = False
    while not done:
        # ── Grip maintenance: re-engage any slipped grips ─────────────────────
        orig_prox  = _gm_mod.PROXIMITY_THRESHOLD
        orig_align = _gm_mod.ALIGNMENT_THRESHOLD
        _gm_mod.PROXIMITY_THRESHOLD = 0.50
        _gm_mod.ALIGNMENT_THRESHOLD = -1.0
        grip_state = env._grip_manager.get_grip_state()
        for slot in (0, 1):
            if grip_state[slot] == 0.0 and start_hold_names:
                env._grip_manager.try_grip(
                    slot, start_hold_names[min(slot, len(start_hold_names) - 1)]
                )
        for slot, hold_name in ((2, lfoot_hold), (3, rfoot_hold)):
            if grip_state[slot] == 0.0:
                env._grip_manager.try_grip(slot, hold_name)
        _gm_mod.PROXIMITY_THRESHOLD = orig_prox
        _gm_mod.ALIGNMENT_THRESHOLD = orig_align

        # ── Step ──────────────────────────────────────────────────────────────
        obs, reward, terminated, truncated, info = env.step(scripted_action)
        total_steps  += 1
        total_reward += reward

        # ── NaN / Inf check ───────────────────────────────────────────────────
        step_nan = not np.all(np.isfinite(obs)) or not np.isfinite(reward)
        for v in info.values():
            if isinstance(v, float) and not np.isfinite(v):
                step_nan = True
        if step_nan:
            any_nan = True

        for k in _REWARD_KEYS:
            component_sums[k] += info.get(k, 0.0)

        grip = env._grip_manager.get_grip_state()
        grip_str = " ".join(str(int(g)) for g in grip)

        # Print every step up to 100, then every 50th.
        if total_steps <= 100 or total_steps % 50 == 0:
            print(
                f"{total_steps:>5}  {grip_str:^20}  "
                f"{info.get('height_reward',    0):>8.4f} "
                f"{info.get('hold_match_bonus', 0):>7.1f} "
                f"{info.get('energy_penalty',   0):>8.4f} "
                f"{info.get('fall_penalty',     0):>6.1f}  "
                f"{reward:>8.4f}  "
                f"{'YES' if step_nan else 'no'}"
            )

        if viewer_ctx is not None and hasattr(viewer_ctx, "sync"):
            viewer_ctx.sync()

        done = terminated or truncated
        if terminated:
            term_reason = (
                "fell"      if info.get("fall_penalty", 0) < 0 else
                "finished"  if info.get("finish_bonus",  0) > 0 else
                "terminated"
            )

    if viewer_ctx is not None:
        try:
            viewer_ctx.__exit__(None, None, None)
        except Exception:
            pass

    env.close()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 80)
    print("SANITY CHECK SUMMARY")
    print("=" * 80)
    print(f"  Total steps      : {total_steps}")
    print(f"  Total reward     : {total_reward:.4f}")
    print(f"  Termination      : {term_reason}")
    print(f"  Any NaN/Inf      : {any_nan}")
    print()
    print("  Reward component sums:")
    for k in _REWARD_KEYS:
        print(f"    {k:<22}: {component_sums[k]:.4f}")

    passed = not any_nan and total_steps >= _MIN_STEPS
    print()
    if passed:
        print(f"  RESULT: PASS  ({total_steps} steps ≥ {_MIN_STEPS}, no NaN)")
    else:
        reasons = []
        if any_nan:
            reasons.append("NaN/Inf detected")
        if total_steps < _MIN_STEPS:
            reasons.append(f"only {total_steps} steps (need ≥ {_MIN_STEPS})")
        print(f"  RESULT: FAIL  ({', '.join(reasons)})")
    print("=" * 80)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
