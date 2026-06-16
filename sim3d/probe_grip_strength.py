"""Grip-strength probe — measured justification for the force multipliers.

HAND_FORCE_MULTIPLIER / FOOT_FORCE_MULTIPLIER were raised to 2.5 / 3.0 when
"the body could not hold a one-hand stance". The 2026-06-11 zero-strain attach
fix removed the actual culprit (kN-scale constraint strain misread as grip
load), so the multipliers can come back down — but to a *measured* floor, not
a guess.

For each vetted seed stance on the probe wall, this:
  1. seeds the stance and measures steady-state anchor forces over a hold;
  2. releases each hand in turn (the move every climb must survive) and
     records the peak force each remaining anchor sees over the transient,
     with slip DISABLED so the requirement is observed, not censored;
  3. converts peaks to the minimum multiplier that keeps that anchor under
     its slip threshold:  required_m = peak / (base × positivity × SLACK).

The headline is the worst-case required multiplier per limb class across all
stances × releases. Config should sit ~20% above it.

Run:
    python -m sim3d.probe_grip_strength
    python -m sim3d.probe_grip_strength --wall-seed 14 --stances 6
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np

from sim3d import config as cfg
from sim3d.body import FOOT_LIMBS, HAND_LIMBS, LIMBS
from sim3d.discover import build_tight_wall
from sim3d.world import Climb3DWorld


def _base_force(profile, limb: str) -> float:
    return (profile.grip_force_n if limb in HAND_LIMBS
            else profile.foot_push_force_n)


def _required_mult(w: Climb3DWorld, profile, peak: dict[str, float]) -> dict[str, float]:
    """Per-limb minimum multiplier so peak ≤ base × m × positivity × SLACK."""
    req = {}
    for l in LIMBS:
        a = w._on_hold[l]
        if a is None or peak[l] <= 0:
            continue
        # max_force_n = base × current_mult × positivity ⇒ recover positivity
        cur_mult = (cfg.HAND_FORCE_MULTIPLIER if l in HAND_LIMBS
                    else cfg.FOOT_FORCE_MULTIPLIER)
        positivity = a.max_force_n / (_base_force(profile, l) * cur_mult)
        req[l] = peak[l] / (_base_force(profile, l) * positivity
                            * cfg.SLIP_FORCE_SLACK)
    return req


def probe(*, wall_seed: int = 9, reach_frac: float = 0.42, n_stances: int = 4,
          hold_frames: int = 100, release_frames: int = 60) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _wd, wall, profile, feas = build_tight_wall(wall_seed, reach_frac)
        w = Climb3DWorld(wall, profile)

    # Unique stances among the feasible moves.
    seen, stances = set(), []
    for m in feas:
        key = tuple(sorted(m["seed_kwargs"].items()))
        if key not in seen:
            seen.add(key)
            stances.append(m["seed_kwargs"])
        if len(stances) >= n_stances:
            break

    worst_hand, worst_foot = 0.0, 0.0
    details = []
    for sk in stances:
        for mover in (None, "LH", "RH"):     # None = plain hang
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.seed_pose(**sk)
            if mover is not None:
                if w.on_hold(mover) is None:
                    continue
                w.release_limb(mover)
            peak = {l: 0.0 for l in LIMBS}
            frames = hold_frames if mover is None else release_frames
            for _ in range(frames):
                w.step(cfg.SUBSTEPS_PER_FRAME, check_slip=False)
                for l in LIMBS:
                    peak[l] = max(peak[l], w.limb_grip_force(l))
            fell = float(w.pelvis_pos()[2]) < 0.20
            req = _required_mult(w, profile, peak)
            for l, m in req.items():
                if l in HAND_LIMBS:
                    worst_hand = max(worst_hand, m)
                else:
                    worst_foot = max(worst_foot, m)
            details.append({"stance": sk, "released": mover, "fell": fell,
                            "peaks": {l: round(p) for l, p in peak.items()},
                            "required_mult": {l: round(m, 2) for l, m in req.items()}})

    print(f"probed {len(stances)} stances × (hang, LH-release, RH-release) "
          f"on wall seed {wall_seed}")
    for d in details:
        rel = d["released"] or "hang"
        print(f"  {rel:11s} {'FELL ' if d['fell'] else ''}peaks {d['peaks']}  "
              f"req_mult {d['required_mult']}")
    print(f"\nworst-case required multiplier:  hand {worst_hand:.2f}   foot {worst_foot:.2f}")
    print(f"current config:                  hand {cfg.HAND_FORCE_MULTIPLIER}   "
          f"foot {cfg.FOOT_FORCE_MULTIPLIER}")
    return {"worst_hand": worst_hand, "worst_foot": worst_foot, "details": details}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wall-seed", type=int, default=9)
    ap.add_argument("--stances", type=int, default=4)
    args = ap.parse_args()
    probe(wall_seed=args.wall_seed, n_stances=args.stances)


if __name__ == "__main__":
    main()
