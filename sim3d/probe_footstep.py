"""Open-loop feasibility probe for a single foot-move stage of a reference.

Decisive test for the question: "can the PD position servos reproduce the
reference's own foot move, when fed the reference's own joint targets?"

Method (mirrors the stage-3 RF probe in the infeasible-reference memory):
  1. RSI the world to the frame just before the mover releases (start of move).
  2. Release the mover limb's weld.
  3. For each subsequent reference frame, write ctrl = ref.qpos[t, 7:]
     (the reference joint angles ARE the position-servo targets) and step.
  4. Track the mover tip's gap to its target hold and its peak height.

If the tip never closes to GRIP_PROXIMITY_M of the target hold, the reference
move is servo-infeasible — no reward shaping can make PPO track it, because
the body cannot reproduce its own recorded trajectory under the controller the
policy actually drives.

Usage:
  python -m sim3d.probe_footstep --ref data/runs/sim3d/imitation/ref_ladder_v6_smooth.npz --move 1
"""
from __future__ import annotations

import argparse

import numpy as np

import sim3d.config as cfg
from sim3d.body import LIMBS
from sim3d.imitation import _load_wall_for_ref
from sim3d.reference import Reference
from sim3d.world import Climb3DWorld


def _move_bounds(ref: Reference) -> list[int]:
    """move_starts from meta if present, else grip-change frames."""
    meta = getattr(ref, "meta", None)
    if isinstance(meta, dict) and "move_starts" in meta:
        return list(meta["move_starts"])
    starts = []
    for i in range(1, len(ref)):
        if not np.array_equal(ref.grips[i], ref.grips[i - 1]):
            starts.append(i)
    return starts


def probe_move(ref: Reference, wall, profile, move_idx: int, mode: str = "perframe") -> dict:
    starts = _move_bounds(ref)
    # The move's "release" frame is move_starts[move_idx]; the move completes
    # by the next move_start (or end of ref).
    rel = starts[move_idx]
    end = starts[move_idx + 1] if move_idx + 1 < len(starts) else len(ref) - 1

    # Which limb moves, and which hold it targets (the hold it grips by `end`).
    g_before = ref.frame_grips(rel - 1)
    g_after = ref.frame_grips(end)
    mover = None
    target = None
    for limb in LIMBS:
        if g_before.get(limb) != g_after.get(limb) and g_after.get(limb) is not None:
            mover, target = limb, g_after[limb]
            break
    if mover is None:
        raise SystemExit(f"move {move_idx}: no limb regrips between f{rel} and f{end}")

    w = Climb3DWorld(wall, profile)
    # Hold world position of the target (true hold centre, not the ref tip).
    hold_pos = np.array(w._hold_meta_by_id[target]["world_pos"], dtype=float)

    # RSI to the frame just before release, with the reference's grips.
    start_frame = rel - 1
    grips0 = {l: ref.frame_grips(start_frame).get(l) for l in LIMBS}
    w.rsi(ref.qpos[start_frame], ref.qvel[start_frame], grips0, settle_frames=4)

    # Release the mover.
    w.release_limb(mover)

    # Open-loop replay. Two modes:
    #   "perframe" — feed ref.qpos[t] as the target each frame (DeepMimic pose
    #     tracking: what the imitation reward rewards when the mover ISN'T freed).
    #   "constant" — hold ref.qpos[end] as a single target for the whole swing
    #     (what discover_move actually does: one constant residual ctrl, then a
    #     weld grabs the last few cm — the relevant test under free_mover).
    best_gap = float("inf")
    peak_z = -float("inf")
    final_tip = None
    n_act = w.data.ctrl.shape[0]
    horizon = end - rel + 1
    if mode == "constant":
        w.data.ctrl[:n_act] = ref.qpos[end, 7:7 + n_act]
    for t in range(rel, end + 1):
        if mode == "perframe":
            w.data.ctrl[:n_act] = ref.qpos[t, 7:7 + n_act]
        w.step(1, check_slip=False)
        tip = w.limb_tip_pos(mover)
        final_tip = tip
        gap = float(np.linalg.norm(tip - hold_pos))
        best_gap = min(best_gap, gap)
        peak_z = max(peak_z, float(tip[2]))

    ref_target_z = float(ref.eef[end, LIMBS.index(mover)][2])
    return {
        "move_idx": move_idx,
        "mover": mover,
        "target_hold": target,
        "frames": (rel, end),
        "best_gap_m": best_gap,
        "grip_radius_m": cfg.GRIP_PROXIMITY_M,
        "feasible": best_gap <= cfg.GRIP_PROXIMITY_M,
        "tip_peak_z": peak_z,
        "tip_final_z": float(final_tip[2]) if final_tip is not None else None,
        "ref_target_z": ref_target_z,
        "z_shortfall_m": ref_target_z - peak_z,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--move", type=int, default=None,
                    help="move index (into move_starts); default: probe all foot moves")
    ap.add_argument("--wall-json", default=None)
    ap.add_argument("--mode", choices=["perframe", "constant", "both"], default="both",
                    help="perframe = DeepMimic pose tracking; constant = hold the "
                         "final target (what discover_move does + free_mover)")
    args = ap.parse_args()

    ref = Reference.load(args.ref)
    wall, profile = _load_wall_for_ref(ref, wall_json=args.wall_json, ref_path=args.ref)
    starts = _move_bounds(ref)

    if args.move is not None:
        moves = [args.move]
    else:
        # Probe every move whose mover is a foot.
        moves = []
        for mi in range(len(starts)):
            rel = starts[mi]
            end = starts[mi + 1] if mi + 1 < len(starts) else len(ref) - 1
            gb, ga = ref.frame_grips(rel - 1), ref.frame_grips(end)
            for limb in LIMBS:
                if gb.get(limb) != ga.get(limb) and ga.get(limb) is not None:
                    if limb in ("LF", "RF"):
                        moves.append(mi)
                    break

    modes = ["perframe", "constant"] if args.mode == "both" else [args.mode]
    print(f"reference: {args.ref}  ({len(ref)} frames, {len(starts)} moves)")
    print(f"grip radius: {cfg.GRIP_PROXIMITY_M:.3f} m\n")
    for mi in moves:
        rs = {m: probe_move(ref, wall, profile, mi, mode=m) for m in modes}
        r0 = next(iter(rs.values()))
        print(f"move {mi}  {r0['mover']}→{r0['target_hold']}  f{r0['frames'][0]}–{r0['frames'][1]}")
        for m, r in rs.items():
            verdict = "FEASIBLE ✅" if r["feasible"] else "INFEASIBLE ❌"
            print(f"    [{m:8}] best gap {r['best_gap_m']*100:6.1f} cm  "
                  f"(need ≤ {r['grip_radius_m']*100:.1f})  {verdict}   "
                  f"tip peak z {r['tip_peak_z']:.3f} / target {r['ref_target_z']:.3f}")
        print()


if __name__ == "__main__":
    main()
