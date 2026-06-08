"""CMA-ES move discovery — the non-RL discovery stage (Naderi 2017).

The reach *controller* is too marginal to author feasible multi-move references
(it won't close to the grip radius, and over-cap stances shed — see NEXT_STEPS
A1d). So instead of executing a hand-coded reach, **search** for joint targets
that land the move: CMA-ES optimizes the mover's arm + spine + hips toward a pose
that places the mover's tip on the target hold *while* keeping the anchors gripped
and the body balanced. That's IK + balance + grip-retention solved jointly — the
trajectory-optimizer discovery the imitation loop assumes.

Each move is discovered from the previous move's end state (RSI-chained), so a
full climb is a sequence of discovered, individually-feasible moves. The recorded
rollout to the optimized pose is the reference (`sim3d.reference.Reference`),
which `sim3d.imitation` then trains a policy to track.

    python -m sim3d.discover --seed 9 --moves 28,29,30,31 --out ref_climb.npz
"""
from __future__ import annotations

import warnings
from typing import Optional

import numpy as np

try:
    import cma
except ImportError as e:  # pragma: no cover
    raise ImportError("sim3d.discover requires pycma:  pip install cma") from e

from sim3d import config as cfg
from sim3d.body import HAND_LIMBS, LIMBS, ClimberProfile
from sim3d.reference import ENV_SUBSTEPS, FALL_Z, Reference
from sim3d.world import Climb3DWorld
from solver.wall import Wall

# Actuator indices (see `actuator joints` order in builder): spine=0, LH arm
# 1–5, RH arm 6–10, hips flex/abduct l=11/12, r=16/17.
_ARM = {"LH": [1, 2, 3, 4, 5], "RH": [6, 7, 8, 9, 10]}
_SPINE_HIPS = [0, 11, 12, 16, 17]


def _search_dofs(mover: str) -> list[int]:
    """The DOFs CMA-ES searches for a move: the mover's arm (the reach) + spine
    and both hips (the weight-shift). Anchor arm + knees/ankles stay at the seed
    pose, holding the stance."""
    return _ARM[mover] + _SPINE_HIPS


def _snapshot(w: Climb3DWorld) -> dict:
    return {
        "qpos": w.data.qpos.copy(), "qvel": w.data.qvel.copy(),
        "eef": np.array([w.limb_tip_pos(l) for l in LIMBS]),
        "com": w.com().copy(),
        "grips": tuple((w.on_hold(l) or "") for l in LIMBS),
    }


def _grips_dict(grips_tuple) -> dict:
    return {LIMBS[i]: (g or None) for i, g in enumerate(grips_tuple)}


def build_tight_wall(seed: int, reach_frac: float, *, cols: int = 12, rows: int = 20,
                     max_attempts: int = 8, max_reach_m: float = 0.45, min_moves: int = 3):
    """Generate a wall with a given ``reach_frac`` (hold spacing as a fraction of
    arm length) and return ``(wall_dict, wall, profile, feasible_moves)``. The
    default route spacing (reach_frac 0.60 ≈ 0.37 m) is beyond the body's static
    reach; ~0.42 (≈ 0.25 m) is reachable, so CMA-ES can land the moves. Returns
    the JSON dict too so the exact wall can be persisted and reloaded for training
    (rebuilding from seed alone wouldn't reproduce a non-default reach_frac)."""
    import solver.generate as gen
    from solver.generate import GeneratorConfig, generate_wall
    from solver.wall import DEFAULT_CELL_SIZE_CM, load_wall
    from sim3d.staged_curriculum import feasible_reach_moves
    profile = ClimberProfile()
    orig = gen._difficulty_params
    wd = wall = feas = None
    try:
        gen._difficulty_params = lambda d, _o=orig: {**_o(0.0), "reach_frac": reach_frac}
        for a in range(max_attempts):
            gc = GeneratorConfig(cols=cols, rows=rows, cell_size_cm=DEFAULT_CELL_SIZE_CM,
                                 difficulty=0.0, seed=seed + a)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wd = generate_wall(gc)
            if wd is None:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wall = load_wall(wd, cell_size_cm=DEFAULT_CELL_SIZE_CM)
                feas = feasible_reach_moves(wall, profile, max_reach_m=max_reach_m)
            if feas and len(feas) >= min_moves:
                return wd, wall, profile, feas
    finally:
        gen._difficulty_params = orig
    return wd, wall, profile, (feas or [])


def discover_move(
    w: Climb3DWorld, start_frame: dict, mover: str, target: str, *,
    horizon: int = 20, max_evals: int = 250, sigma0: float = 0.4,
) -> tuple[list[dict], dict]:
    """CMA-ES-discover a single move from ``start_frame``. Returns
    ``(recorded_frames, info)``. ``w`` is a scratch world reused across the
    search (RSI-reset every rollout)."""
    anchor = next(l for l in HAND_LIMBS if l != mover)
    lo = np.array([w.model.jnt_range[int(w.model.actuator_trnid[i, 0]), 0] for i in range(w.model.nu)])
    hi = np.array([w.model.jnt_range[int(w.model.actuator_trnid[i, 0]), 1] for i in range(w.model.nu)])
    dofs = _search_dofs(mover)
    target_pos = np.array(w._hold_meta_by_id[target]["world_pos"])
    grips0 = _grips_dict(start_frame["grips"])
    n_anchor0 = sum(1 for l in LIMBS if l != mover and grips0.get(l))

    def rollout(residual: np.ndarray, record: bool = False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.rsi(start_frame["qpos"], np.zeros(w.model.nv), grips0)
        w.release_limb(mover)
        ctrl = w.data.ctrl.copy()                    # = stance pose after rsi sync
        for k, i in enumerate(dofs):
            ctrl[i] = float(np.clip(ctrl[i] + residual[k], lo[i], hi[i]))
        w.data.ctrl[:] = ctrl
        pelvis_y0 = float(w.pelvis_pos()[1])
        com_z0 = float(start_frame["com"][2])
        com_z_min = com_z0          # track worst trough during the reach
        frames: list[dict] = []
        for _ in range(horizon):
            if record:
                frames.append(_snapshot(w))
            w.step(ENV_SUBSTEPS, check_slip=True)
            com_z_min = min(com_z_min, float(w.com()[2]))
            if w.on_hold(mover) is None:
                gap = float(np.linalg.norm(target_pos - w.limb_tip_pos(mover)))
                if gap < cfg.GRIP_PROXIMITY_M:
                    w.attach_limb(mover, target)     # grip when in proximity
        if record:
            frames.append(_snapshot(w))

        gap = float(np.linalg.norm(target_pos - w.limb_tip_pos(mover)))
        n_anchor = sum(1 for l in LIMBS if l != mover and w.on_hold(l) is not None)
        fell = float(w.pelvis_pos()[2]) < FALL_Z
        # Directional lean: only penalize moving *away* from the wall (positive Y).
        lean_back = max(0.0, float(w.pelvis_pos()[1]) - pelvis_y0)
        # CoM trough: penalize the worst mid-reach dip below the start height.
        # com_drop (end-minus-start) missed poses that dip and recover — the policy
        # can't track a sharp trough even if the final height is fine.
        com_trough = max(0.0, com_z0 - com_z_min)
        # End-height: also penalize ending lower (catches moves that don't recover).
        com_drop = max(0.0, com_z0 - float(w.com()[2]))
        landed = w.on_hold(mover) == target
        # Land (gap→0) dominates; hard penalties for dropped anchors or falling;
        # posture terms keep the body upright with a smooth CoM arc, no sharp troughs.
        cost = (10.0 * gap + 8.0 * max(0, n_anchor0 - n_anchor)
                + (25.0 if fell else 0.0)
                + 3.0 * lean_back + 4.0 * com_trough + 3.0 * com_drop
                + 0.08 * float(np.linalg.norm(residual)))
        if record:
            return cost, frames, {"gap": round(gap, 3), "landed": bool(landed),
                                  "n_anchor": n_anchor, "fell": bool(fell),
                                  "lean_back": round(lean_back, 3),
                                  "com_trough": round(com_trough, 3),
                                  "com_drop": round(com_drop, 3)}
        return cost

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xbest, _es = cma.fmin2(
            rollout, np.zeros(len(dofs)), sigma0,
            {"maxfevals": max_evals, "bounds": [-1.6, 1.6], "verbose": -9},
        )
    _cost, frames, info = rollout(np.asarray(xbest), record=True)
    return frames, info


def discover_climb(
    wall: Wall, profile: ClimberProfile, moves: list[dict], *,
    horizon: int = 20, max_evals: int = 250, sigma0: float = 0.4,
    settle_pre: int = 2, wall_gen_seed: int = 7,
) -> tuple[Reference, list[dict]]:
    """Discover a multi-move climb: seed the bottom stance, then CMA-ES-discover
    each move from the previous move's end frame (RSI-chained). Stops at the first
    move that can't be discovered (a genuine infeasibility)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
        w.seed_pose(**moves[0]["seed_kwargs"])

    frames: list[dict] = [_snapshot(w) for _ in range(settle_pre)]
    start = _snapshot(w)
    diag: list[dict] = []
    move_starts: list[int] = []
    for m in moves:
        mover, target = m["mover"], m["target"]
        if target not in w._hold_meta_by_id:
            diag.append({"move_k": m["move_k"], "status": "aborted: unknown target"})
            break
        move_starts.append(len(frames))
        move_frames, info = discover_move(
            w, start, mover, target, horizon=horizon, max_evals=max_evals, sigma0=sigma0)
        diag.append({"move_k": m["move_k"], "mover": mover, "target": target, **info})
        frames.extend(move_frames)
        if not info["landed"]:
            diag.append({"status": f"aborted: move {m['move_k']} not discovered "
                                   f"(gap {info['gap']} m)"})
            break
        start = move_frames[-1]      # chain from this move's end

    ref = Reference(
        qpos=np.array([f["qpos"] for f in frames]),
        qvel=np.array([f["qvel"] for f in frames]),
        eef=np.array([f["eef"] for f in frames]),
        com=np.array([f["com"] for f in frames]),
        grips=np.array([f["grips"] for f in frames], dtype="<U24"),
        wall_gen_seed=wall_gen_seed,
        meta={"discovered": [d.get("move_k") for d in diag if "mover" in d],
              "move_starts": move_starts, "method": "cma-es"},
    )
    return ref, diag


def main() -> None:
    import argparse
    import contextlib
    import io
    import json
    from pathlib import Path
    from sim3d.reference import holdable_fraction

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--reach-frac", type=float, default=0.42,
                    help="hold spacing as fraction of arm length (lower = reachable)")
    ap.add_argument("--moves", type=str, default="9,10,11",
                    help="comma-separated move_k values to discover (consecutive)")
    ap.add_argument("--max-evals", type=int, default=160)
    ap.add_argument("--out", type=str, default="data/runs/sim3d/imitation/ref_cma.npz")
    args = ap.parse_args()

    with contextlib.redirect_stderr(io.StringIO()):
        wd, wall, profile, feas = build_tight_wall(args.seed, args.reach_frac)
    want = [int(x) for x in args.moves.split(",")]
    moves = [next((m for m in feas if m["move_k"] == k), None) for k in want]
    if any(m is None for m in moves):
        print(f"moves not feasible on seed {args.seed}: "
              f"{[k for k, m in zip(want, moves) if m is None]}  (feasible: {[m['move_k'] for m in feas]})")
        return
    print(f"Discovering {len(moves)} moves {want} on '{getattr(wall, 'name', '?')}' "
          f"(reach_frac {args.reach_frac}, CMA-ES {args.max_evals} evals/move)…")
    ref, diag = discover_climb(wall, profile, moves, max_evals=args.max_evals,
                               wall_gen_seed=args.seed)
    for d in diag:
        print("  ", d)
    landed = sum(1 for d in diag if d.get("landed"))
    with contextlib.redirect_stderr(io.StringIO()):
        frac = holdable_fraction(ref, wall, profile)
    print(f"discovered {landed}/{len(moves)} moves | {len(ref)} frames | holdable {frac*100:.0f}%")
    if landed >= 1:
        ref.save(args.out)
        # Persist the exact wall so training reproduces it (a non-default
        # reach_frac wouldn't rebuild from seed alone).
        wall_path = Path(args.out).with_suffix(".wall.json")
        wall_path.write_text(json.dumps(wd))
        print(f"saved {args.out}  +  {wall_path}")


if __name__ == "__main__":
    main()
