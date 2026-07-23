"""Trackability probe — is each move of a reference action-space FEASIBLE?

The decisive question behind the foot-move grind (NEXT_STEPS / the
infeasible-reference saga): a reference authored by the *reach controller*
(Kp≈2000 + weld-snap) can contain stances/moves the PD-servo action space
cannot reproduce, so PPO can never track them. A reference authored by
``sim3d.discover`` is feasible by construction (CMA-ES vets every move against
the live env physics).

We answer it by reusing the discovery oracle itself. ``discover.discover_move``
searches for a single CONSTANT action that lands a move under the real env
physics — no balance assist, anchors must stay gripped, the body must not fall.
That is exactly the imitation env's contract. So for each move in a reference:

  1. RSI a scratch world to the move's recorded START stance,
  2. run ``discover_move`` toward the move's recorded TARGET hold,
  3. report whether a constant action lands it (gap < grip radius) and the
     residual tip gap.

A move that ``discover_move`` lands is re-authorable in the action space (just
re-author the reference with ``sim3d.discover`` — Increment 2/4). A move it
CANNOT land from that stance is servo-infeasible — the controller mismatch /
scrunched-stance root cause, which trajectory search (Increment 3) or a
different stance must address.

NOTE: a separate open-loop pose-replay was tried first and rejected — feeding
the reference pose as the servo target barn-doors the body on hand releases
(the pelvis is a free joint with no corrective control open-loop), so it has
false negatives on every hand move regardless of how the ref was authored.
``discover_move`` sidesteps this: its CMA search actively finds a pose that
keeps the anchors and the body upright, the same way a trained policy would.

    python -m sim3d.probe_trackability \
        --ref data/runs/sim3d/imitation/ref_cma9_6moves.npz \
        --ref data/runs/sim3d/imitation/ref_ladder_natural_v2.npz
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import warnings
from pathlib import Path

import numpy as np

from sim3d import config as cfg
from sim3d.body import LIMBS, ClimberProfile
from sim3d.discover import _snapshot, discover_move
from sim3d.reference import Reference, migrate_reference_spine
from sim3d.world import Climb3DWorld
from solver.wall import DEFAULT_CELL_SIZE_CM, load_wall


def _load_wall(ref_path: Path, wall_arg: str | None):
    """Resolve the wall for a reference. Priority: explicit --wall, then the
    ``<ref>.wall.json`` sidecar (discover writes it), then ladder fallback."""
    candidates = []
    if wall_arg:
        candidates.append(Path(wall_arg))
    candidates.append(ref_path.with_suffix(".wall.json"))
    candidates.append(Path("data/examples/ladder-v1.json"))
    for c in candidates:
        if c.exists():
            wd = json.loads(c.read_text())
            with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wall = load_wall(wd, cell_size_cm=DEFAULT_CELL_SIZE_CM)
            return wall, c
    raise FileNotFoundError(
        f"no wall for {ref_path.name}: pass --wall, or place a "
        f"{ref_path.with_suffix('.wall.json').name} sidecar")


def _move_segments(ref: Reference) -> list[tuple[int, int]]:
    """Return [(start, end)] frame ranges per move. Prefer meta['move_starts'];
    fall back to grip rising-edges (a limb acquiring a hold it wasn't on)."""
    ms = list(ref.meta.get("move_starts") or [])
    if ms:
        bounds = sorted(set(int(x) for x in ms if 0 <= int(x) < len(ref)))
    else:
        bounds = [0]
        for t in range(1, len(ref)):
            prev, cur = ref.grips[t - 1], ref.grips[t]
            if any(cur[i] and cur[i] != prev[i] for i in range(4)):
                bounds.append(t)
        bounds = sorted(set(bounds))
    ends = bounds[1:] + [len(ref)]
    return [(s, e) for s, e in zip(bounds, ends) if e - s >= 1]


def _segment_mover(ref: Reference, s: int, e: int) -> tuple[str | None, str | None]:
    """The limb that acquires a NEW hold across the segment, and that hold."""
    g0, g1 = ref.frame_grips(s), ref.frame_grips(min(e - 1, len(ref) - 1))
    for limb in LIMBS:
        if g1[limb] and g1[limb] != g0[limb]:
            return limb, g1[limb]
    return None, None


def probe_reference(ref_path: Path, wall_arg: str | None, *, max_evals: int) -> dict:
    ref = migrate_reference_spine(Reference.load(ref_path))
    wall, wall_src = _load_wall(ref_path, wall_arg)
    profile = ClimberProfile()
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)

    rows: list[dict] = []
    for k, (s, e) in enumerate(_move_segments(ref)):
        mover, target = _segment_mover(ref, s, e)
        if mover is None or target is None or target not in w._hold_meta_by_id:
            continue
        # Build a discover-style start frame from the move's recorded START
        # stance (anchors welded exactly as the reference had them).
        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.rsi(ref.qpos[s], np.zeros(w.model.nv), ref.frame_grips(s))
            start = _snapshot(w)
            _frames, info = discover_move(
                w, start, mover, target, max_evals=max_evals, restarts=1)
        # Reference's own recorded gap at the move end, for comparison.
        li = LIMBS.index(mover)
        ref_tip_gap = float(np.linalg.norm(
            np.array(w._hold_meta_by_id[target]["world_pos"]) - ref.eef[e - 1, li]))
        rows.append({
            "move": k, "mover": mover, "target": target,
            "feasible": bool(info["landed"]), "gap": float(info["gap"]),
            "ref_gap": ref_tip_gap, "fell": bool(info.get("fell", False)),
            "com_drop": float(info.get("com_drop", 0.0)),
        })
    return {"ref": ref_path.name, "wall": wall_src.name,
            "method": ref.meta.get("method", "?"), "n_frames": len(ref), "rows": rows}


def _print_report(rep: dict) -> None:
    print(f"\n=== {rep['ref']}  (wall: {rep['wall']}, method: {rep['method']}, "
          f"{rep['n_frames']} frames) ===")
    print(f"  {'move':>4} {'mover':>5} {'target':>10} {'feasible':>9} "
          f"{'gap(m)':>8} {'refGap(m)':>10} {'comDrop':>8} {'fell':>5}")
    feas = 0
    for r in rep["rows"]:
        print(f"  {r['move']:>4} {r['mover']:>5} {r['target']:>10} "
              f"{('YES' if r['feasible'] else 'no'):>9} {r['gap']:>8.3f} "
              f"{r['ref_gap']:>10.3f} {r['com_drop']:>8.3f} "
              f"{('Y' if r['fell'] else '-'):>5}")
        feas += int(r["feasible"])
    n = len(rep["rows"])
    pct = (100.0 * feas / n) if n else 0.0
    verdict = ("re-authorable in action space" if feas == n
               else f"{n - feas} move(s) servo-INFEASIBLE from their stance")
    print(f"  → {feas}/{n} moves action-space feasible ({pct:.0f}%)  ⇒  {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", action="append", required=True,
                    help="reference .npz (repeatable)")
    ap.add_argument("--wall", default=None,
                    help="wall JSON override (default: <ref>.wall.json sidecar, "
                         "then data/examples/ladder-v1.json)")
    ap.add_argument("--max-evals", type=int, default=300,
                    help="CMA-ES evals per move (higher = stronger feasibility "
                         "claim; default 300)")
    args = ap.parse_args()
    for rp in args.ref:
        try:
            rep = probe_reference(Path(rp), args.wall, max_evals=args.max_evals)
        except Exception as ex:  # noqa: BLE001
            import traceback
            print(f"\n=== {rp} ===\n  ERROR: {ex}")
            traceback.print_exc()
            continue
        _print_report(rep)


if __name__ == "__main__":
    main()
