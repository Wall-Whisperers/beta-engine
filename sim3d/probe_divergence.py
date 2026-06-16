"""Divergence probe — WHERE and HOW a tracking policy departs from its reference.

The headline metric (`imitation.eval_frame0`) tells you *whether* a policy climbs
the whole reference from the bottom. It does not tell you where it falls off or
what the body is doing wrong when it does. Mean episode length is worse than
useless here (a policy that nails move 1 then dies looks identical to one that
flails immediately if both happen to end at the same frame).

This probe answers the mechanical question. It rolls out the model deterministically
from reference frame 0 (exactly as `eval_frame0` does), records the full per-frame
state, and for each episode finds the **divergence frame**: the first frame at which
a limb the reference has gripped is NOT on that hold in the sim — i.e. the first move
that failed to land. It then:

  * histograms divergence frames by reference MOVE, so you see which transition is
    the wall (e.g. "37/40 die in move 2, the LF foot step");
  * at the modal failing move, dumps the body state at the boundary, policy vs
    reference: CoM height + **distance-to-wall** (the body-tension signal), the
    mover limb's distance to its target hold (did the reach even close?), and the
    anchor limbs' grip forces (did weight transfer, or did an anchor slip?).

This is the "trace actions, not rewards; find the first divergence frame" diagnostic
— grounded in this body's actual contact/force state rather than a tracking-camera
video (which lies; see NEXT_STEPS.md). Run:

    python -m sim3d.probe_divergence --model <run>/model.zip --ref <ref>.npz \
        --vecnorm <run>/vecnormalize.pkl [--episodes 40]
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from sim3d.body import LIMBS
from sim3d.imitation import ImitationConfig, ImitationEnv, _load_wall_for_ref
from sim3d.reference import Reference


def _move_of_frame(t: int, move_starts: list[int]) -> int:
    """Reference move index whose window contains frame ``t`` (move k spans
    [move_starts[k], move_starts[k+1]))."""
    m = 0
    for k, fr in enumerate(move_starts):
        if t >= fr:
            m = k
    return m


def probe(model_path: str, ref_path: str, *, vecnorm: Optional[str] = None,
          wall_json: Optional[str] = None, n_episodes: int = 40,
          r_min: float = 0.5) -> dict:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ref = Reference.load(ref_path)
    wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
    move_starts = list(ref.meta.get("move_starts") or [0])
    # Boundary frame that defines "move k landed": the START of move k+1 is the
    # established end-stance of move k. Final move lands at the last frame.
    move_boundaries = move_starts[1:] + [len(ref) - 1]

    # Frame-0, full-reference, no chain windows — identical setup to eval_frame0.
    icfg = ImitationConfig(rsi_phase_max=0, rsi_anneal_steps=0, chain_stages=False,
                           r_min_start=r_min, r_min_end=r_min)
    env = ImitationEnv(ref, wall, profile, imitation_config=icfg)
    model = PPO.load(model_path)
    vn = None
    if vecnorm and Path(vecnorm).exists():
        vn = VecNormalize.load(vecnorm, DummyVecEnv([lambda: ImitationEnv(ref, wall, profile, icfg)]))
        vn.training = False

    world = env.env.world
    term_moves: list[int] = []         # reference move the episode TERMINATED in
    term_frames: list[int] = []        # frame it terminated at
    reached: list[int] = []            # highest move boundary actually landed
    n_succ = 0
    # Per-move-boundary mechanical snapshots, accumulated over the episodes that
    # reached that boundary's window, keyed by move index.
    boundary_dumps: dict[int, list[dict]] = {}

    for ep in range(n_episodes):
        obs, info = env.reset(seed=10_000 + ep)
        done = False
        last_phase = env._phase
        max_landed = -1                # highest move whose stance grips matched
        while not done:
            o = vn.normalize_obs(obs) if vn is not None else obs
            action, _ = model.predict(o, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            t = min(env._phase, len(ref) - 1)
            last_phase = t

            # Snapshot at each boundary crossing; record it as "landed" when the
            # stance's reference grips are all satisfied (the move actually
            # completed, not merely that we passed the frame).
            for k, bf in enumerate(move_boundaries):
                if t == bf:
                    boundary_dumps.setdefault(k, []).append(
                        _snapshot(world, ref, bf, k, move_boundaries))
                    g = ref.frame_grips(bf)
                    if all(world.on_hold(l) == h for l, h in g.items() if h is not None):
                        max_landed = max(max_landed, k)
            done = term or trunc

        reached.append(max_landed)
        if info.get("is_success"):
            n_succ += 1
        else:
            term_frames.append(last_phase)
            term_moves.append(_move_of_frame(last_phase, move_starts))

    env.close()
    return {
        "n_episodes": n_episodes, "n_succ": n_succ,
        "term_moves": term_moves, "term_frames": term_frames, "reached": reached,
        "boundary_dumps": boundary_dumps, "move_starts": move_starts,
        "move_boundaries": move_boundaries, "ref": ref,
    }


def _snapshot(world, ref: Reference, bf: int, move_k: int,
              move_boundaries: list[int]) -> dict:
    """Body state at boundary frame ``bf``, policy vs reference. The 'mover' is
    the limb the reference NEWLY grips at this boundary (its reach is the move)."""
    com = world.com()
    ref_com = ref.com[bf]
    # Mover = limb gripped at this boundary but not at the previous one.
    prev_bf = move_boundaries[move_k - 1] if move_k > 0 else 0
    g_now = ref.frame_grips(bf)
    g_prev = ref.frame_grips(prev_bf)
    movers = [l for l in LIMBS if g_now[l] is not None and g_prev.get(l) != g_now[l]]
    mover = movers[0] if movers else None
    d = {
        "com_z": float(com[2]), "ref_com_z": float(ref_com[2]),
        "com_y": float(com[1]), "ref_com_y": float(ref_com[1]),  # +Y = away from wall
        "mover": mover,
        "grip_forces": {l: round(world.limb_grip_force(l), 1) for l in LIMBS},
        "on_hold": {l: world.on_hold(l) for l in LIMBS},
        "ref_grips": {l: g_now[l] for l in LIMBS},
    }
    if mover is not None:
        tip = world.limb_tip_pos(mover)
        target = ref.eef[bf, LIMBS.index(mover)]   # ref tip when gripped ≈ hold pos
        d["mover_reach_gap"] = float(np.linalg.norm(tip - target))
    return d


def report(res: dict) -> None:
    n, ns = res["n_episodes"], res["n_succ"]
    print(f"\nFRAME-0 rollouts: {ns}/{n} full-reference successes "
          f"({100*ns/max(1,n):.0f}%)\n")

    # How far the body actually got — highest move whose stance grips it landed.
    reached = res["reached"]
    rcnt = Counter(reached)
    print("Highest move LANDED (grips matched the reference's stance):")
    for mv in sorted(rcnt):
        label = "(none — never completed move 0)" if mv < 0 else f"move {mv}"
        print(f"  {label:>32}: {rcnt[mv]:>3}/{n}  {'█'*rcnt[mv]}")
    # Landing boundary index k = completing move k = k+1 new holds vs the
    # start hang (boundary 0 is the first reach off the start stance).
    best = max(reached)
    holds = best + 1
    print(f"  → best case: completed move {best}  "
          f"({holds} new hold(s) climbed)" if best >= 0
          else "  → best case: never held even the first stance\n")
    print()

    if not res["term_moves"]:
        print("All episodes succeeded — nothing terminated early.")
        return

    print("Where episodes TERMINATED (by reference move):")
    cnt = Counter(res["term_moves"])
    total_fail = len(res["term_moves"])
    for mv in sorted(cnt):
        bar = "█" * cnt[mv]
        print(f"  move {mv:>2}: {cnt[mv]:>3}/{total_fail}  {bar}")
    modal_move = cnt.most_common(1)[0][0]
    print(f"\nModal termination: move {modal_move}  "
          f"(median frame {int(np.median(res['term_frames']))})")

    dumps = res["boundary_dumps"].get(modal_move)
    if not dumps:
        # Most episodes died BEFORE reaching this boundary — inspect the prior
        # one (the stance they launch the failing move from).
        prior = modal_move - 1
        dumps = res["boundary_dumps"].get(prior)
        if dumps:
            print(f"\n(No episode reached move {modal_move}'s boundary; showing the "
                  f"launch stance at move {prior}'s boundary instead.)")
            modal_move = prior
    if not dumps:
        print("\nNo boundary snapshots captured for the failing move.")
        return

    mover = dumps[0]["mover"]
    com_z = np.mean([d["com_z"] for d in dumps])
    com_y = np.mean([d["com_y"] for d in dumps])
    ref_z = dumps[0]["ref_com_z"]
    ref_y = dumps[0]["ref_com_y"]
    print(f"\nBody state at move {modal_move}'s boundary  "
          f"(mover limb: {mover}, n={len(dumps)} episodes reached it):")
    print(f"  CoM height     policy {com_z:.3f} m   ref {ref_z:.3f} m   "
          f"(Δ {com_z-ref_z:+.3f})")
    print(f"  CoM dist-to-wall (+Y)  policy {com_y:.3f} m   ref {ref_y:.3f} m   "
          f"(Δ {com_y-ref_y:+.3f})   ← body tension: larger = further off the wall")
    if "mover_reach_gap" in dumps[0]:
        gap = np.mean([d["mover_reach_gap"] for d in dumps])
        print(f"  {mover} reach gap to target hold  {gap:.3f} m   "
              f"(GRIP_PROXIMITY is 0.08 m — above it the grip can't engage)")
    # Anchor grip forces, averaged.
    print("  grip forces (N), policy mean per limb:")
    for l in LIMBS:
        f = np.mean([d["grip_forces"][l] for d in dumps])
        held = sum(d["on_hold"][l] is not None for d in dumps)
        print(f"    {l}: {f:7.1f} N   (on a hold in {held}/{len(dumps)} eps)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--vecnorm", default=None)
    ap.add_argument("--wall-json", default=None)
    ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--r-min", type=float, default=0.15,
                    help="R_min termination threshold — must match training (default 0.15)")
    args = ap.parse_args()

    wall_json = args.wall_json
    if wall_json is None:
        sib = Path(args.ref).with_suffix(".wall.json")
        if sib.exists():
            wall_json = str(sib)

    res = probe(args.model, args.ref, vecnorm=args.vecnorm, wall_json=wall_json,
                n_episodes=args.episodes, r_min=args.r_min)
    report(res)


if __name__ == "__main__":
    main()
