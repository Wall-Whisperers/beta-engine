"""Transition-holdability probe — de-risk for the imitation + RSI plan.

The imitation/DeepMimic plan (record a reference, RSI into its frames, train a
policy to track it with a termination curriculum) rests on one physical
assumption: **the body can hold the *transitional* stances a climb passes
through**, not just the tuned seed/hang stances. `NEXT_STEPS.md` A1c found the
opposite for a hand-coded expert — it chains 1–4 moves then a transition tips it
off. This probe adjudicates that finding against the imitation thesis *before*
we build the full training loop.

It separates the question A1c conflated into three artifact-free sub-tests, each
starting from the gentle, vetted ``seed_pose`` stance (no instant state-slamming,
so no reconstruction transients):

  1. **base hang** — seed the 4-grip stance, hold passively. Sanity: it was
     pre-vetted to hang, so it must. If it doesn't, the harness/physics is off.
  2. **one-hand release (the physical question)** — seed the same stance, release
     the *mover* hand, hold the same pose passively on the remaining 3 grips.
     Does releasing a hand — with no weight-shift — drop the body?
  3. **expert reach (A1c reproduction)** — seed, fire the KP-1500 Cartesian
     reach, and watch whether it sheds *anchor* grips (factor 1) and barn-doors
     (factor 2) during the motion.

The decision:

* release-hold OK **but** reach sheds anchors → the instability is the *fixed
  reach controller* overloading grips, which a closed-loop imitation policy that
  commands grips directly can fix. **Green** — build the loop.
* release-hold FAILS → releasing a hand drops the body even from a good static
  pose, so it needs an active weight-shift first. Imitation can still do it *iff*
  the reference encodes that CoM shift — but a naive same-pose reference won't.
  **Yellow** — author references with weight-shift, or address grip/balance.

A fourth, separately-reported check — **RSI faithfulness** — slams the captured
seed state back in (instant qpos + re-weld) and asks whether it reproduces the
hang. This is an *engineering* property the real RSI loop needs, kept distinct
from the physics so a reconstruction artifact can't masquerade as instability.

Run:
    python -m sim3d.probe_transitions                 # default gen wall (seed 7)
    python -m sim3d.probe_transitions --hold-steps 20 --frames 100
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from sim3d import config as cfg
from sim3d.body import LIMBS, ClimberProfile
from sim3d.staged_curriculum import feasible_reach_moves
from sim3d.world import Climb3DWorld
from solver.generate import GeneratorConfig, generate_wall
from solver.wall import DEFAULT_CELL_SIZE_CM, load_wall

FALL_Z = 0.20  # EnvConfig.fall_z — pelvis below this counts as fallen.


# ─── Wall + feasible moves (mirrors StagedCurriculumEnv._build_wall_and_vet) ──

def build_wall_and_moves(
    *, seed: int = 7, cols: int = 12, rows: int = 20, difficulty: float = 0.0,
    max_attempts: int = 6, max_reach_m: float = 0.45, min_moves: int = 3,
):
    """Generate the same curriculum wall the staged runs used and return
    ``(wall, profile, feasible_moves)``."""
    profile = ClimberProfile()
    wall = feas = None
    for attempt in range(max_attempts):
        gen = GeneratorConfig(
            cols=cols, rows=rows, cell_size_cm=DEFAULT_CELL_SIZE_CM,
            difficulty=difficulty, seed=seed + attempt,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wd = generate_wall(gen)
        if wd is None:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wall = load_wall(wd, cell_size_cm=DEFAULT_CELL_SIZE_CM)
            feas = feasible_reach_moves(wall, profile, max_reach_m=max_reach_m)
        if feas and len(feas) >= min_moves:
            return wall, profile, feas
    return wall, profile, (feas or [])


def _seed_world(wall, profile, seed_kwargs) -> Climb3DWorld:
    """A fresh world gently seeded onto ``seed_kwargs`` (vetted hang stance)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
        w.seed_pose(**seed_kwargs)
    return w


def _hold(w: Climb3DWorld, k_steps: int) -> dict:
    """Hold the current pose passively (ctrl already = settled pose, grips
    unchanged) for ``k_steps`` env-steps with slip on. This is the exact
    condition a zero action produces in the env.

    Returns a trichotomy ``outcome`` — the binary "held?" hid the real signal:
      * ``fall``     — pelvis dropped below FALL_Z (catastrophic; needs an active
                       weight-shift the passive hold can't supply).
      * ``shed``     — kept the body up but lost ≥1 grip (marginal; a near-cap
                       grip the passive servo couldn't maintain, but a closed-loop
                       policy actively managing load plausibly can).
      * ``hold``     — stayed up and kept every grip.
    Matches the curriculum's vetting on the ``fall`` line (it too only checks
    not-fallen), so ``hold``/``shed`` both clear that bar."""
    grips0 = sum(1 for l in LIMBS if w.on_hold(l) is not None)
    min_pz = float(w.pelvis_pos()[2])
    for _ in range(k_steps):
        w.step(cfg.SUBSTEPS_PER_FRAME, check_slip=True)
        min_pz = min(min_pz, float(w.pelvis_pos()[2]))
        if float(w.pelvis_pos()[2]) < FALL_Z:
            break
    grips_end = sum(1 for l in LIMBS if w.on_hold(l) is not None)
    fell = min_pz < FALL_Z
    outcome = "fall" if fell else ("shed" if grips_end < grips0 else "hold")
    return {"outcome": outcome, "fell": fell, "grips0": grips0,
            "grips_end": grips_end, "min_pelvis_z": round(min_pz, 3)}


def _run_reach(w: Climb3DWorld, mover: str, target: str, n_frames: int) -> dict:
    """Fire the Cartesian-impedance reach from the seeded stance and watch the
    anchors. Reproduces A1c factors 1 (anchor shed) and 2 (barn-door)."""
    anchors = [l for l in LIMBS if l != mover and w.on_hold(l) is not None]
    target_pos = np.array(w._hold_meta_by_id[target]["world_pos"])
    pelvis_y0 = float(w.pelvis_pos()[1])
    w.move_limb(mover, target, mode="reach")
    min_anchors = len(anchors)
    slips = 0
    min_pz = float(w.pelvis_pos()[2])
    min_reach_d = 1e9
    for _ in range(n_frames):
        min_reach_d = min(min_reach_d, float(np.linalg.norm(w.limb_tip_pos(mover) - target_pos)))
        slips += w.step(1, check_slip=True)
        min_anchors = min(min_anchors, sum(1 for l in anchors if w.on_hold(l) is not None))
        min_pz = min(min_pz, float(w.pelvis_pos()[2]))
    return {"n_anchors0": len(anchors), "min_anchors": min_anchors, "slips": slips,
            "barn_door_dy": round(float(w.pelvis_pos()[1]) - pelvis_y0, 3),
            "min_pelvis_z": round(min_pz, 3), "min_reach_d": round(min_reach_d, 3),
            "landed": min_reach_d < cfg.REACH_ATTACH_RADIUS}


def _rsi_faithful(wall, profile, ref_qpos, ref_grips, k_steps: int,
                  settle_frames: int = 4) -> bool:
    """Engineering check (NOT physics): slam ``ref_qpos`` + re-weld ``ref_grips``
    into a fresh world, absorb the weld transient with a slip-off settle, then
    hold with slip on. True iff the instant reconstruction reproduces the hang.
    Tells the real RSI loop whether instant state-set needs a gentle re-attach."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
    w.reset()
    w.data.qpos[:] = ref_qpos
    w.data.qvel[:] = 0.0
    mujoco.mj_forward(w.model, w.data)
    for limb, hid in ref_grips.items():
        if hid is not None:
            w.attach_limb(limb, hid)
    mujoco.mj_forward(w.model, w.data)
    w._sync_actuator_targets_to_pose()
    if settle_frames > 0:
        w.step(settle_frames, check_slip=False)
        w.data.qvel[:] = 0.0
    n_ref = sum(1 for v in ref_grips.values() if v is not None)
    n_after = sum(1 for l in LIMBS if w.on_hold(l) is not None)
    # Faithful = the slammed-in seed still hangs (not fallen) without losing the
    # grips the reference held.
    return (not _hold(w, k_steps)["fell"]) and n_after >= n_ref


# ─── Driver ──────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7, help="wall gen seed (default 7)")
    ap.add_argument("--frames", type=int, default=100, help="frames per expert reach (~1.6 s)")
    ap.add_argument("--hold-steps", type=int, default=16, help="env-steps to hold (16 ≈ 2.0 s)")
    ap.add_argument("--out", type=str, default="data/runs/sim3d/transition_probe/result.json")
    args = ap.parse_args()

    t0 = time.time()
    print(f"Building curriculum wall (gen_seed={args.seed}) + vetting reach moves…")
    wall, profile, feasible = build_wall_and_moves(seed=args.seed)
    print(f"  wall '{getattr(wall, 'name', '?')}' — {len(feasible)} feasible reach moves\n")
    if not feasible:
        print("No feasible moves — try another --seed.")
        return

    rows = []
    for mv in feasible:
        mover, target, seed_kw = mv["mover"], mv["target"], mv["seed_kwargs"]
        tag = f"move {mv['move_k']:>2}  {mover}→{target:<10} d0={mv['reach_d0']:.2f}m"

        # (1) base hang — sanity that the vetted seed hangs in this harness.
        base = _hold(_seed_world(wall, profile, seed_kw), args.hold_steps)
        # (2) one-hand release — the physical question (artifact-free).
        wr = _seed_world(wall, profile, seed_kw)
        ref_qpos = wr.data.qpos.copy()
        ref_grips = {l: wr.on_hold(l) for l in LIMBS}
        wr.release_limb(mover)
        rel = _hold(wr, args.hold_steps)
        # (3) expert reach — A1c factors 1 & 2.
        reach = _run_reach(_seed_world(wall, profile, seed_kw), mover, target, args.frames)
        # (4) RSI faithfulness — engineering, reported separately.
        rsi_ok = _rsi_faithful(wall, profile, ref_qpos, ref_grips, args.hold_steps)

        anchor_note = ("anchors OK" if reach["min_anchors"] == reach["n_anchors0"]
                       else f"ANCHOR SHED {reach['n_anchors0']}→{reach['min_anchors']}")
        oc = {"hold": "HOLD", "shed": "shed", "fall": "FALL"}
        print(f"{tag}")
        print(f"    base-hang(4grip)={oc[base['outcome']]}   "
              f"release(3grip)={oc[rel['outcome']]}"
              f"  (grips {rel['grips0']}→{rel['grips_end']}, min_pelvis_z={rel['min_pelvis_z']})")
        _snap = "" if reach['landed'] else ("snap@%.2fm" % reach['min_reach_d'])
        print(f"    expert-reach: {'land' if reach['landed'] else _snap}"
              f"  {anchor_note}  slips={reach['slips']}  barn-door Δy={reach['barn_door_dy']:+.3f}m")
        print(f"    rsi-faithful={'OK' if rsi_ok else 'no'}")
        print()

        rows.append({
            "move_k": mv["move_k"], "mover": mover, "target": target,
            "reach_d0": mv["reach_d0"],
            "base_outcome": base["outcome"], "release_outcome": rel["outcome"],
            "release_grips": [rel["grips0"], rel["grips_end"]],
            "release_min_pelvis_z": rel["min_pelvis_z"],
            "reach_landed": reach["landed"], "n_anchors0": reach["n_anchors0"],
            "min_anchors": reach["min_anchors"], "reach_slips": reach["slips"],
            "barn_door_dy": reach["barn_door_dy"], "rsi_faithful": rsi_ok,
        })

    # ── Verdict ──────────────────────────────────────────────────────────────
    n = len(rows)
    base_survive = sum(r["base_outcome"] != "fall" for r in rows)   # not-fallen (matches vetting)
    rel_survive = sum(r["release_outcome"] != "fall" for r in rows)
    rel_fall = sum(r["release_outcome"] == "fall" for r in rows)
    rel_clean = sum(r["release_outcome"] == "hold" for r in rows)
    reach_sheds = sum(r["min_anchors"] < r["n_anchors0"] for r in rows)
    # The discriminator: release survives (so losing a hand isn't itself
    # catastrophic), yet the *reach* sheds anchors → the reach controller's force
    # is the culprit, not the pose. That's the imitation-fixable case.
    controller_fault = sum(r["release_outcome"] != "fall" and r["min_anchors"] < r["n_anchors0"]
                           for r in rows)
    rsi_ok = sum(r["rsi_faithful"] for r in rows)
    barn = float(np.mean([abs(r["barn_door_dy"]) for r in rows]))

    print("═" * 74)
    print(f"VERDICT  ({n} moves on '{getattr(wall, 'name', '?')}')")
    print(f"  base 4-grip seed survives passive hold (sanity): {base_survive}/{n}"
          + ("" if base_survive >= 0.85 * n else "   ⚠ see INVALID note"))
    print(f"  one-hand release — body SURVIVES (no fall)     : {rel_survive}/{n}"
          f"   ({rel_clean} clean-hold, {rel_survive - rel_clean} shed-but-upright, {rel_fall} fell)")
    print(f"  expert reach sheds ≥1 anchor (A1c factor 1)    : {reach_sheds}/{n}")
    print(f"  └ release survived AND reach shed anchors       : {controller_fault}/{n}"
          f"   → reach-controller fault, imitation-fixable")
    print(f"  mean |barn-door Δy| during reach (A1c factor 2): {barn:.3f} m")
    print(f"  RSI faithfulness (instant state-set reproduces): {rsi_ok}/{n}"
          f"   (engineering; real loop may need gentle re-attach)")
    print()

    if base_survive < 0.85 * n:
        print("  → INVALID. The pre-vetted 4-grip seed doesn't even survive passive "
              "holding\n    here — the probe's stepping disagrees with the curriculum's "
              "own vetting.\n    Resolve before trusting the rest.")
    elif rel_fall >= 0.5 * n:
        print("  → YELLOW/RED. Releasing a hand makes the body FALL on most moves, "
              "even from a\n    good static stance — the transition genuinely needs an "
              "active weight-shift\n    first. Imitation can still do it, but ONLY if "
              "the reference encodes that CoM\n    shift (a hands-only stitched "
              "reach-one reference is NOT sufficient). Author\n    references with "
              "weight-shift, or address grip/balance before the loop.")
    elif controller_fault >= 0.5 * n:
        print("  → GREEN-ish. Losing a hand rarely drops the body (it survives, often "
              "just\n    shedding a near-cap grip the *passive* servo can't hold), yet "
              "the KP-1500\n    reach reliably sheds anchors (A1c factor 1). So the "
              "catastrophe is the fixed\n    open-loop reach force, not the stance. A "
              "closed-loop imitation policy that\n    commands grip intents + joint "
              "targets directly — and is rewarded for matching\n    the reference's "
              "contacts/CoM — can hold what the reach can't. Build the loop.\n    Watch "
              "the 'shed-but-upright' count: that's the margin the policy must actively\n"
              "    manage, so keep the CoM-tracking term and consider nudging grip caps "
              "if it\n    struggles.")
    else:
        print("  → MIXED. Read per-move: release survives + reach sheds ⇒ controller "
              "fault\n    (fixable); release falls ⇒ that stance needs an active "
              "weight-shift.")
    print("═" * 74)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": vars(args), "wall": getattr(wall, "name", None),
        "n_moves": n, "base_survive": base_survive, "release_survive": rel_survive,
        "release_clean_hold": rel_clean, "release_fall": rel_fall,
        "reach_sheds_anchor": reach_sheds, "controller_fault": controller_fault,
        "rsi_faithful": rsi_ok, "mean_abs_barn_door_dy": round(barn, 4), "moves": rows,
    }, indent=2))
    print(f"\nWrote {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
