"""Reference trajectories + the bounded DeepMimic imitation reward.

This is the "discovery stage" half of the imitation plan (Babadi/Naderi/
Hämäläinen): produce a per-route reference motion *without RL*, then let PPO
learn a robust controller that tracks it (``sim3d.imitation``). The probe
(``sim3d.probe_transitions``) established the prerequisite — the body can hold
the transitional stances; the chaining failure was the open-loop reach
controller — and flagged the key requirement: **the reference must carry a CoM
weight-shift**, or the "shed-but-upright" margin bites.

Three things live here:

* ``Reference`` — a per-env-step trajectory: full ``qpos``/``qvel`` (for RSI),
  precomputed end-effector + CoM positions and per-limb grip hold-ids (for the
  reward). Sampled at the env control rate so phase advances 1 frame per step.
* ``imitation_reward`` — the bounded [0,1] DeepMimic reward, the single source
  shared by calibration and the training env. Capped at 1/step and maximised by
  *matching* the reference, so it cannot be farmed by oscillating (the failure
  mode of every dense shaping term tried before).
* ``author_weight_shift_move`` — author one reference move with a CoM shift, by
  recording a **balance-assisted** reach: the pelvis is pulled toward the anchor
  hand while the mover reaches, so the CoM shifts onto the support (instead of
  barn-dooring as the raw KP-1500 reach does). Physically recorded ⇒ the frames
  are self-consistent and RSI-faithful by construction.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from sim3d import config as _cfg
from sim3d.body import HAND_LIMBS, LIMBS, ClimberProfile
from sim3d.env import EnvConfig
from sim3d.world import Climb3DWorld
from solver.wall import Wall

ENV_SUBSTEPS = EnvConfig().sim_substeps   # world.step(frames) per env control step
FALL_Z = 0.20


# ─── Bounded DeepMimic imitation reward ──────────────────────────────────────

@dataclass
class ImitationCoeffs:
    """Weights (sum to 1) + sensitivity exponents for the bounded reward.

    ``r_imit = w_pose·r_pose + w_vel·r_vel + w_endeff·r_endeff + w_com·r_com``,
    each term ``exp(-k · mean_sq_error) ∈ (0,1]``, so ``r_imit ∈ (0,1]``. Weights
    are DeepMimic's (0.65/0.10/0.15/0.10); the ``k`` exponents are calibrated for
    our units (rad, m, m/s) so a faithful track scores ~1 and a typical deviation
    ~0.5 — tune with ``python -m sim3d.reference --calibrate``."""
    w_pose: float = 0.65
    w_vel: float = 0.10
    w_endeff: float = 0.15
    w_com: float = 0.10
    k_pose: float = 12.0      # over mean sq joint-angle error (rad²)
    k_vel: float = 0.30       # over mean sq joint-velocity error ((rad/s)²)
    k_endeff: float = 40.0    # over mean per-limb sq tip error (m²)
    k_com: float = 30.0       # over sq CoM error (m²)


def imitation_reward(world: Climb3DWorld, ref: "Reference", t: int,
                     c: ImitationCoeffs) -> tuple[float, dict]:
    """Bounded [0,1] DeepMimic reward of the world's current state against
    reference frame ``t``. Returns ``(r_imit, components)``."""
    q, qd = world.data.qpos, world.data.qvel
    pose_err = q[7:] - ref.qpos[t, 7:]                       # 21 joint angles
    r_pose = float(np.exp(-c.k_pose * np.mean(pose_err ** 2)))
    vel_err = qd[6:] - ref.qvel[t, 6:]                       # 21 joint velocities
    r_vel = float(np.exp(-c.k_vel * np.mean(vel_err ** 2)))
    eef = np.array([world.limb_tip_pos(l) for l in LIMBS])   # (4,3)
    eef_err = eef - ref.eef[t]
    r_endeff = float(np.exp(-c.k_endeff * np.mean(np.sum(eef_err ** 2, axis=1))))
    com_err = world.com() - ref.com[t]
    r_com = float(np.exp(-c.k_com * np.sum(com_err ** 2)))
    r = c.w_pose * r_pose + c.w_vel * r_vel + c.w_endeff * r_endeff + c.w_com * r_com
    return r, {"pose": r_pose, "vel": r_vel, "endeff": r_endeff, "com": r_com}


# ─── Reference trajectory container ──────────────────────────────────────────

@dataclass
class Reference:
    """A per-env-step reference motion. Arrays are length-T (T frames)."""
    qpos: np.ndarray            # (T, nq)
    qvel: np.ndarray            # (T, nv)
    eef: np.ndarray             # (T, 4, 3) tip world positions, LIMBS order
    com: np.ndarray             # (T, 3)
    grips: np.ndarray           # (T, 4) hold-id strings, "" = released
    wall_gen_seed: int = 7      # the gen seed that rebuilds the matching wall
    meta: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.qpos.shape[0])

    def frame_grips(self, t: int) -> dict[str, Optional[str]]:
        return {LIMBS[i]: (g if g else None) for i, g in enumerate(self.grips[t])}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path, qpos=self.qpos, qvel=self.qvel, eef=self.eef, com=self.com,
            grips=self.grips, wall_gen_seed=self.wall_gen_seed,
            meta=np.array(repr(self.meta), dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "Reference":
        d = np.load(Path(path), allow_pickle=True)
        meta = {}
        try:
            meta = eval(str(d["meta"]))  # noqa: S307 — our own repr, trusted
        except Exception:  # noqa: BLE001
            pass
        return cls(
            qpos=d["qpos"], qvel=d["qvel"], eef=d["eef"], com=d["com"],
            grips=d["grips"], wall_gen_seed=int(d["wall_gen_seed"]), meta=meta,
        )


# ─── Authoring: a balance-assisted reach with a CoM weight-shift ─────────────

def author_weight_shift_move(
    wall: Wall, profile: ClimberProfile, move: dict, *,
    settle_pre: int = 3, reach_frames: int = 16, settle_post: int = 6,
    shift_frac: float = 0.6, toward_wall_m: float = 0.04, balance_kp: float = 900.0,
    wall_gen_seed: int = 7,
) -> tuple[Reference, dict]:
    """Author one reference move (from a ``feasible_reach_moves`` entry) and
    return ``(Reference, diagnostics)``.

    The motion: settle on the vetted stance → release the mover and reach to the
    target with the **balance assist** pulling the pelvis toward the anchor hand
    (CoM shifts onto the support, no barn-door) → settle on the new stance.
    Recorded at the env control rate (one frame per ``world.step(ENV_SUBSTEPS)``)
    so the imitation env advances phase 1 frame per step.

    The balance assist is the existing (off-by-default) pelvis PD; we enable it
    only for the duration of authoring by setting ``config.BALANCE_KP``."""
    mover, target = move["mover"], move["target"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
        w.seed_pose(**move["seed_kwargs"])
    anchor_hand = next((l for l in HAND_LIMBS
                        if l != mover and w.on_hold(l) is not None), None)

    frames: list[dict] = []

    def rec() -> None:
        frames.append({
            "qpos": w.data.qpos.copy(), "qvel": w.data.qvel.copy(),
            "eef": np.array([w.limb_tip_pos(l) for l in LIMBS]),
            "com": w.com().copy(),
            "grips": tuple((w.on_hold(l) or "") for l in LIMBS),
        })

    mover_force_seed = w.limb_grip_force(mover)
    com_x0 = float(w.com()[0])

    # Pre-settle on the vetted 4-grip stance.
    for _ in range(settle_pre):
        rec()
        w.step(ENV_SUBSTEPS, check_slip=True)

    # Balance-assisted reach: release mover, pull pelvis toward the anchor.
    w.move_limb(mover, target, mode="reach")
    pelvis_tgt = w.pelvis_pos().copy()
    if anchor_hand is not None:
        ax = float(w.limb_tip_pos(anchor_hand)[0])
        pelvis_tgt[0] += shift_frac * (ax - pelvis_tgt[0])
    pelvis_tgt[1] -= toward_wall_m
    kp0 = _cfg.BALANCE_KP
    _cfg.BALANCE_KP = balance_kp
    w._balance_target = pelvis_tgt
    min_anchors = 3
    try:
        for _ in range(reach_frames):
            rec()
            w.step(ENV_SUBSTEPS, check_slip=True)
            min_anchors = min(min_anchors, sum(
                1 for l in LIMBS if l != mover and w.on_hold(l) is not None))
    finally:
        _cfg.BALANCE_KP = kp0
        w._balance_target = None

    # Post-settle on the new stance.
    for _ in range(settle_post):
        rec()
        w.step(ENV_SUBSTEPS, check_slip=True)

    ref = Reference(
        qpos=np.array([f["qpos"] for f in frames]),
        qvel=np.array([f["qvel"] for f in frames]),
        eef=np.array([f["eef"] for f in frames]),
        com=np.array([f["com"] for f in frames]),
        grips=np.array([f["grips"] for f in frames], dtype="<U24"),
        wall_gen_seed=wall_gen_seed,
        meta={"move_k": move["move_k"], "mover": mover, "target": target,
              "reach_d0": move["reach_d0"]},
    )
    diag = {
        "n_frames": len(ref),
        "landed": bool(ref.grips[-1, LIMBS.index(mover)] == target),
        "com_shift_x": round(float(np.max(ref.com[:, 0]) - com_x0), 3),
        "com_total_travel": round(float(np.sum(np.linalg.norm(np.diff(ref.com, axis=0), axis=1))), 3),
        "min_anchors_during_reach": min_anchors,
        "mover_force_seed_n": round(float(mover_force_seed), 1),
    }
    return ref, diag


def author_climb_reference(
    wall: Wall, profile: ClimberProfile, moves: list[dict], *,
    settle_pre: int = 2, reach_frames: int = 20, settle_post: int = 4,
    shift_frac: float = 0.6, toward_wall_m: float = 0.04, balance_kp: float = 250.0,
    snap_dist: float = 0.22, wall_gen_seed: int = 7,
) -> tuple["Reference", list[dict]]:
    """Author a multi-move climb by **RSI-chaining** — the fix for both failure
    modes that broke the earlier versions.

    Independent authoring + concatenation gave smooth *within* moves but teleports
    at the boundaries (the seed pose ≠ the prior move's end). A single continuous
    rollout fixed the boundaries but reproduced A1c — momentum/instability
    accumulated across moves and the body collapsed (grips 3→1→0).

    RSI-chaining takes the best of both: author move 1 from its vetted seed, then
    start each next move by RSI-ing the world into the *previous move's clean end
    frame* (zeroed velocity + re-welded grips) before authoring its reach. Move
    k+1 begins exactly at move k's end pose (smooth boundary) from a fresh,
    non-collapsing state (no accumulated instability). The static reach often
    stalls ~0.13 m short, so it snaps to its **closest-approach** pose (drift
    frames discarded) — a genuine miss ends the climb there. Returns
    ``(Reference, per_move_diagnostics)``."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
        w.seed_pose(**moves[0]["seed_kwargs"])

    frames: list[dict] = []

    def rec() -> None:
        frames.append({
            "qpos": w.data.qpos.copy(), "qvel": w.data.qvel.copy(),
            "eef": np.array([w.limb_tip_pos(l) for l in LIMBS]),
            "com": w.com().copy(),
            "grips": tuple((w.on_hold(l) or "") for l in LIMBS),
        })

    for _ in range(settle_pre):
        rec()
        w.step(ENV_SUBSTEPS, check_slip=True)

    diag: list[dict] = []
    move_starts: list[int] = []
    for mi, m in enumerate(moves):
        # RSI-chain: every move after the first starts from the PREVIOUS move's
        # clean end frame (zeroed velocity + re-welded grips), so no momentum or
        # instability carries over (the A1c collapse) while the boundary stays
        # smooth (move k+1 begins at move k's end pose). The spec movers alternate
        # (LH,RH,…) and chain; only the anchor must hold (the mover releases).
        if mi > 0:
            last = frames[-1]
            grips = {LIMBS[i]: (g or None) for i, g in enumerate(last["grips"])}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.rsi(last["qpos"], np.zeros(w.model.nv), grips)

        mover, target = m["mover"], m["target"]
        anchor = next(l for l in HAND_LIMBS if l != mover)
        if w.on_hold(anchor) is None or target not in w._hold_meta_by_id:
            diag.append({"move_k": m["move_k"], "status": "aborted: anchor hand lost"})
            break

        move_starts.append(len(frames))
        tgt = np.array(w._hold_meta_by_id[target]["world_pos"])
        w.move_limb(mover, target, mode="reach")
        pelvis_tgt = w.pelvis_pos().copy()
        pelvis_tgt[0] += shift_frac * (float(w.limb_tip_pos(anchor)[0]) - pelvis_tgt[0])
        pelvis_tgt[1] -= toward_wall_m
        kp0 = _cfg.BALANCE_KP
        _cfg.BALANCE_KP = balance_kp
        w._balance_target = pelvis_tgt
        # Track the closest approach so a marginal reach snaps to its CLOSEST
        # pose, not the drifted-away final one.
        best = (1e9, len(frames), None, None)   # (gap, frame_idx, qpos, grips)
        try:
            for _ in range(reach_frames):
                rec()
                gap = float(np.linalg.norm(tgt - w.limb_tip_pos(mover)))
                if gap < best[0]:
                    best = (gap, len(frames) - 1, w.data.qpos.copy(),
                            tuple(w.on_hold(l) for l in LIMBS))
                w.step(ENV_SUBSTEPS, check_slip=True)
                if w.on_hold(mover) == target:
                    break
        finally:
            _cfg.BALANCE_KP = kp0
            w._balance_target = None

        # Land it: if the reach didn't auto-grip, rewind to the closest-approach
        # pose (discard the drift frames) and snap onto the hold — but only if it
        # got close enough; a genuine miss ends the climb here.
        if w.on_hold(mover) != target:
            if best[0] <= snap_dist and best[2] is not None:
                del frames[best[1] + 1:]
                grips_best = {l: h for l, h in zip(LIMBS, best[3]) if h}
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w.rsi(best[2], np.zeros(w.model.nv), grips_best, settle_frames=0)
                w.move_limb(mover, target, mode="snap")
            else:
                diag.append({"move_k": m["move_k"],
                             "status": f"aborted: reach fell {best[0]:.2f} m short"})
                break

        for _ in range(settle_post):
            rec()
            w.step(ENV_SUBSTEPS, check_slip=True)

        n_grip = sum(1 for l in LIMBS if w.on_hold(l) is not None)
        diag.append({"move_k": m["move_k"], "mover": mover, "target": target,
                     "landed": bool(w.on_hold(mover) == target),
                     "min_gap": round(best[0], 3), "n_grip_after": n_grip})
        if float(w.pelvis_pos()[2]) < FALL_Z:
            diag.append({"status": "aborted: body fell"})
            break

    ref = Reference(
        qpos=np.array([f["qpos"] for f in frames]),
        qvel=np.array([f["qvel"] for f in frames]),
        eef=np.array([f["eef"] for f in frames]),
        com=np.array([f["com"] for f in frames]),
        grips=np.array([f["grips"] for f in frames], dtype="<U24"),
        wall_gen_seed=wall_gen_seed,
        meta={"moves": [d.get("move_k") for d in diag if "mover" in d],
              "move_starts": move_starts, "continuous": True},
    )
    return ref, diag


def stitch_references(refs: list["Reference"]) -> "Reference":
    """Concatenate per-move references into one full-climb trajectory.

    Consecutive feasible reach moves chain by construction — move k ends with
    the hands on holds (k, k+1), which is exactly move k+1's start stance — so
    independently-authored moves (each from its own vetted stance, hence
    RSI-faithful) concatenate into a continuous climb. Small pose jumps at the
    boundaries are fine: RSI re-seeds each frame and the policy tracks locally,
    so a boundary is just another transition to learn."""
    if not refs:
        raise ValueError("stitch_references: need ≥1 reference")
    return Reference(
        qpos=np.concatenate([r.qpos for r in refs]),
        qvel=np.concatenate([r.qvel for r in refs]),
        eef=np.concatenate([r.eef for r in refs]),
        com=np.concatenate([r.com for r in refs]),
        grips=np.concatenate([r.grips for r in refs]),
        wall_gen_seed=refs[0].wall_gen_seed,
        meta={"stitched": [r.meta for r in refs], "boundaries":
              list(np.cumsum([len(r) for r in refs])[:-1])},
    )


def holdable_fraction(ref: Reference, wall: Wall, profile: ClimberProfile, *,
                      k_steps: int = 12) -> float:
    """Fraction of reference frames that, RSI'd and held passively, don't fall.
    Reuses the probe's verdict mechanism to confirm the authored reference is
    trackable before we train on it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
    held = 0
    for t in range(len(ref)):
        w.rsi(ref.qpos[t], np.zeros_like(ref.qvel[t]), ref.frame_grips(t))
        ok = True
        for _ in range(k_steps):
            w.step(ENV_SUBSTEPS, check_slip=True)
            if float(w.pelvis_pos()[2]) < FALL_Z:
                ok = False
                break
        held += int(ok)
    return held / max(1, len(ref))


# ─── CLI: author + self-check + calibrate ────────────────────────────────────

def _calibrate(ref: Reference, wall: Wall, profile: ClimberProfile,
               c: ImitationCoeffs) -> None:
    """RSI to each frame, take ONE zero action, and report the imitation reward
    vs the *next* frame — the score a do-nothing policy earns. It should start
    high (RSI holds the pose) and fall as the reference moves away, i.e. the
    gradient the policy must climb. If it's ~1 everywhere, the k's are too soft;
    if ~0, too sharp."""
    import numpy as _np
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
    zero_ctrl_rows = []
    for t in range(len(ref) - 1):
        w.rsi(ref.qpos[t], ref.qvel[t], ref.frame_grips(t))
        w.step(ENV_SUBSTEPS, check_slip=True)        # zero action ≈ hold ref pose
        r, comp = imitation_reward(w, ref, t + 1, c)
        zero_ctrl_rows.append((r, comp))
    rs = _np.array([r for r, _ in zero_ctrl_rows])
    print(f"  null-policy r_imit vs next frame: mean {rs.mean():.2f}  "
          f"min {rs.min():.2f}  max {rs.max():.2f}")
    for name in ("pose", "vel", "endeff", "com"):
        vals = _np.array([comp[name] for _, comp in zero_ctrl_rows])
        print(f"    r_{name:7s} mean {vals.mean():.2f}  min {vals.min():.2f}")


def main() -> None:
    import argparse
    from sim3d.probe_transitions import build_wall_and_moves

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--move-index", type=int, default=-1,
                    help="index into feasible moves to author (default: a mid move)")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--out", type=str, default="data/runs/sim3d/imitation/ref_move.npz")
    args = ap.parse_args()

    wall, profile, feasible = build_wall_and_moves(seed=args.seed)
    if not feasible:
        print("No feasible moves — try another --seed.")
        return
    idx = args.move_index if args.move_index >= 0 else len(feasible) // 2
    move = feasible[idx]
    print(f"Authoring move {move['move_k']} ({move['mover']}→{move['target']}, "
          f"d0={move['reach_d0']:.2f}m) on '{getattr(wall, 'name', '?')}'…")

    ref, diag = author_weight_shift_move(wall, profile, move, wall_gen_seed=args.seed)
    print("  diagnostics:", diag)
    frac = holdable_fraction(ref, wall, profile)
    print(f"  holdable fraction (RSI + passive hold): {frac*100:.0f}%  "
          f"({len(ref)} frames)")
    if args.calibrate:
        _calibrate(ref, wall, profile, ImitationCoeffs())

    ref.save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
