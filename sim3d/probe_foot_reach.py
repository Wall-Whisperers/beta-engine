"""Foot-reach envelope probe — measured justification for hip changes.

NEXT_STEPS N1 claims foot moves are blocked: "the hip_flex axis rotates the
leg forward (into the wall), not upward, capping foot z". Before touching the
body (CLAUDE.md: body changes need a measured justification), this probe
measures what the CURRENT hips can actually do:

  1. Build the same tight wall + vetted 4-limb stance discovery uses.
  2. Release one foot (the mover).
  3. Grid-sweep the mover leg's joint targets (hip_flex × hip_abduct ×
     hip_rot × knee × ankle), settle each combo, and record where the foot
     tip can go *while staying placeable*: SETTLED (measured on the last
     frames, not a swing-through), tip within --wall-band of the hold plane,
     within 0.5 m laterally of the pelvis (a usable foothold, not a side
     kick), and without collapsing the stance (≥3 anchors kept).

The headline number is ``max placeable tip z relative to the pelvis``: a foot
move to the next hold row needs roughly pelvis−0.5 m … pelvis−0.3 m. If the
envelope tops out far below that, the hips are the blocker and the follow-up
body change has its measurement.

Run:
    python -m sim3d.probe_foot_reach                  # LF on the seed-9 wall
    python -m sim3d.probe_foot_reach --mover RF --steps 20
"""
from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np

from sim3d import config as cfg
from sim3d.body import LIMBS
from sim3d.discover import _snapshot, build_tight_wall
from sim3d.reference import ENV_SUBSTEPS

_LEG_JOINTS = ["hip_flex", "hip_abduct", "hip_rot", "knee", "ankle"]


def _leg_ctrl_indices(w, mover: str) -> list[int]:
    side = "l" if mover == "LF" else "r"
    return [w.actuator_id_by_joint[f"{side}_{j}"] for j in _LEG_JOINTS]

DEG = np.pi / 180.0


def _grid(joint: str, n: int) -> np.ndarray:
    lo, hi = cfg.JOINT_LIMITS_RAD[joint]
    return np.linspace(lo, hi, n)


def probe_envelope(mover: str = "LF", *, wall_seed: int = 9,
                   reach_frac: float = 0.42, steps: int = 16,
                   wall_band_m: float = 0.12, grid_n: tuple = (6, 4, 3, 5, 3),
                   verbose: bool = True) -> dict:
    """Sweep the mover leg's joint targets from the vetted stance and return
    the reachable foot-tip envelope. ``grid_n`` = points per joint in
    ``_LEG_JOINTS`` order."""
    from sim3d.world import Climb3DWorld

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _wd, wall, profile, feas = build_tight_wall(wall_seed, reach_frac)
        if not feas:
            raise RuntimeError(f"no feasible stance on seed {wall_seed}")
        w = Climb3DWorld(wall, profile)
        w.seed_pose(**feas[0]["seed_kwargs"])
    stance = _snapshot(w)
    grips0 = {LIMBS[i]: (g or None) for i, g in enumerate(stance["grips"])}
    pelvis_z0 = float(w.pelvis_pos()[2])
    tip0 = w.limb_tip_pos(mover).copy()

    # Hold plane: the y where a foot must be to stand on a hold (vertical wall
    # ⇒ constant). Use the median hold-tip y.
    wall_y = float(np.median([m["world_pos"][1] for m in w._hold_meta_by_id.values()]))

    axes = [_grid(j, n) for j, n in zip(_LEG_JOINTS, grid_n)]
    ctrl_idx = _leg_ctrl_indices(w, mover)

    best = {"placeable_z": -np.inf}
    n_eval = 0
    max_tip_z_any = -np.inf
    for combo in itertools.product(*axes):
        n_eval += 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.rsi(stance["qpos"], np.zeros(w.model.nv), grips0)
        w.release_limb(mover)
        for i, v in zip(ctrl_idx, combo):
            w.data.ctrl[i] = float(v)
        placeable_z = -np.inf
        tip_at_best = None
        for k in range(steps):
            w.step(ENV_SUBSTEPS, check_slip=True)
            tip = w.limb_tip_pos(mover)
            max_tip_z_any = max(max_tip_z_any, float(tip[2]))
            if k < steps - 3:        # settled poses only — no swing-throughs
                continue
            n_anchor = sum(1 for l in LIMBS if l != mover and w.on_hold(l))
            pelvis_x = float(w.pelvis_pos()[0])
            if (n_anchor >= 3
                    and abs(float(tip[1]) - wall_y) <= wall_band_m
                    and abs(float(tip[0]) - pelvis_x) <= 0.50):
                if float(tip[2]) > placeable_z:
                    placeable_z = float(tip[2])
                    tip_at_best = tip.copy()
        if placeable_z > best["placeable_z"]:
            best = {
                "placeable_z": placeable_z,
                "tip": None if tip_at_best is None else [round(float(v), 3) for v in tip_at_best],
                "targets_deg": {j: round(float(v) / DEG, 1)
                                for j, v in zip(_LEG_JOINTS, combo)},
            }

    result = {
        "mover": mover,
        "wall_seed": wall_seed,
        "n_eval": n_eval,
        "pelvis_z0": round(pelvis_z0, 3),
        "tip_z_start": round(float(tip0[2]), 3),
        "wall_y": round(wall_y, 3),
        "max_tip_z_any": round(max_tip_z_any, 3),
        "max_placeable_tip_z": (None if not np.isfinite(best["placeable_z"])
                                else round(best["placeable_z"], 3)),
        "placeable_minus_pelvis": (None if not np.isfinite(best["placeable_z"])
                                   else round(best["placeable_z"] - pelvis_z0, 3)),
        "best": best,
    }
    if verbose:
        print(f"mover {mover}  stance pelvis_z {pelvis_z0:.3f}  "
              f"start tip_z {float(tip0[2]):.3f}  wall_y {wall_y:.3f}")
        print(f"swept {n_eval} combos × {steps} frames")
        print(f"  max tip z anywhere:               {max_tip_z_any:.3f}")
        if np.isfinite(best["placeable_z"]):
            print(f"  max PLACEABLE tip z (±{wall_band_m*100:.0f}cm of wall, "
                  f"anchors kept): {best['placeable_z']:.3f}  "
                  f"(pelvis {best['placeable_z']-pelvis_z0:+.3f})")
            print(f"  at targets: {best['targets_deg']}  tip {best['tip']}")
        else:
            print("  no placeable pose found (every combo left the wall band "
                  "or collapsed the stance)")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mover", choices=("LF", "RF"), default="LF")
    ap.add_argument("--wall-seed", type=int, default=9)
    ap.add_argument("--steps", type=int, default=16,
                    help="env frames to settle each combo")
    ap.add_argument("--wall-band", type=float, default=0.12,
                    help="max |tip_y − hold_plane_y| (m) to count as placeable")
    args = ap.parse_args()
    probe_envelope(args.mover, wall_seed=args.wall_seed, steps=args.steps,
                   wall_band_m=args.wall_band)


if __name__ == "__main__":
    main()
