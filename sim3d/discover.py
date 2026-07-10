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

from sim3d import artifact_meta as am
from sim3d import config as cfg
from sim3d.body import LIMBS, ClimberProfile
from sim3d.reference import ENV_SUBSTEPS, FALL_Z, Reference
from sim3d.world import Climb3DWorld
from solver.wall import Wall

# Joint names per mover chain. Ctrl indices are looked up on the live model
# (``w.actuator_id_by_joint``) — never hardcode actuator order, it shifts
# whenever the body gains joints (e.g. the 2026-06-11 spine_lat/spine_twist).
_ARM = {"LH": ["l_shoulder_az", "l_shoulder_el", "l_shoulder_roll", "l_elbow", "l_wrist"],
        "RH": ["r_shoulder_az", "r_shoulder_el", "r_shoulder_roll", "r_elbow", "r_wrist"]}
_LEG = {"LF": ["l_hip_flex", "l_hip_abduct", "l_hip_rot", "l_knee", "l_ankle"],
        "RF": ["r_hip_flex", "r_hip_abduct", "r_hip_rot", "r_knee", "r_ankle"]}
_SPINE = ["spine_lean", "spine_lat", "spine_twist"]
_HIPS = {"LF": ["l_hip_flex", "l_hip_abduct"], "RF": ["r_hip_flex", "r_hip_abduct"]}
_KNEES = ["l_knee", "r_knee"]

# 4-limb move cycle: LH → RH → LF → RF → LH → …
_CYCLE = ["LH", "RH", "LF", "RF"]

# Posture joints whose ROM exhaustion is the validated "scrunch" signal: a
# folded-in-half stance (feet too high relative to hands — a 2-rung instead of
# 3-rung span) pins the spine AND hip_abduct at their limits, while an extended
# natural stance leaves them relaxed (measured 2026-06-14, see the
# infeasible-reference-foot-move note). Penalizing closeness-to-limit steers
# discovery toward extended stances a 3-limb hang can hold WITHOUT a balance
# assist — exactly the stances the trackability probe showed reach-controller
# refs lacked (hand reaches slumped 0.27-0.33 m on release from scrunched setups).
_POSTURE_JOINTS = ["spine_lean", "spine_lat", "spine_twist",
                   "l_hip_abduct", "r_hip_abduct"]


def _posture_strain(w: Climb3DWorld, margin_rad: float = 0.15) -> float:
    """Sum over posture joints of how deep each sits in the last ``margin_rad``
    of its ROM toward either limit (0 = relaxed, 1 = pinned at a limit). A
    fully-folded stance maxes all five ⇒ strain ≈ 5."""
    strain = 0.0
    for name in _POSTURE_JOINTS:
        if name not in w.actuator_id_by_joint:
            continue
        jid = int(w.model.actuator_trnid[w.actuator_id_by_joint[name], 0])
        adr = int(w.model.jnt_qposadr[jid])
        lo, hi = w.model.jnt_range[jid]
        margin = min(float(w.data.qpos[adr]) - lo, hi - float(w.data.qpos[adr]))
        if margin < margin_rad:
            strain += (margin_rad - margin) / margin_rad
    return strain


def _search_dofs(w, mover: str, whole_body: bool = False) -> list[int]:
    """The ctrl indices CMA-ES searches for a move.
    For arm movers: the mover's arm + spine + both hips + BOTH KNEES.
    Knees added 2026-06-11: after a foot move the body is crouched on high
    feet, and converting that into hand reach is knee extension — standing
    up. Without the knees the optimizer literally could not stand, which is
    why every 4-limb chain stalled right after its foot steps (hands "out
    of reach" 0.7–1.0 m that leg extension closes).
    For foot movers: the mover's leg + spine + opposing hip (weight-shift).

    ``whole_body`` (foot movers) — ALSO search BOTH arms + the standing leg's
    knee/ankle + the standing hip's rot. A real climber places a high foot by
    pulling in on the hands and pressing through the standing leg — a whole-body
    action. The narrow default froze the arms and standing knee/ankle, so the
    authored foot move could only reach with the swinging leg and landed at the
    grip-radius edge (RF gap 0.080, no margin). Recruiting the whole body lets
    discovery pull the hips in/up and close the last cm with margin. Added
    2026-07-07 on the observation that each move should move the WHOLE body."""
    if mover in _ARM:
        names = _ARM[mover] + _SPINE + _HIPS["LF"] + _HIPS["RF"] + _KNEES
    elif mover in _LEG:
        other = "RF" if mover == "LF" else "LF"
        names = _LEG[mover] + _SPINE + _HIPS[other]
        if whole_body:
            stand = "l" if mover == "RF" else "r"     # the planted leg
            names = (names + _ARM["LH"] + _ARM["RH"]
                     + [f"{stand}_hip_rot", f"{stand}_knee", f"{stand}_ankle"])
    else:
        raise ValueError(f"unknown mover: {mover!r}")
    # De-dup while preserving order (opposing hip flex/abduct may recur).
    seen: set[str] = set()
    names = [n for n in names if not (n in seen or seen.add(n))]
    return [w.actuator_id_by_joint[n] for n in names]


def _apply_climbing_posture(w: Climb3DWorld, *, knee_bend_rad: float = 0.90,
                            hip_flex_rad: float = 0.55, settle: int = 24) -> None:
    """Re-settle the current welded stance into a climbing posture: drive the
    knees + hip_flex to a bent target and spine_lean→0 with actuators ON (welds
    keep the grips). Bent knees sink the hips DOWN and BACK over the feet (room
    for the shin to come forward / the foot to travel straight up on the next
    move); the stiff spine servo lifts the torso off its lean stop. seed_pose
    and discover_move both land near-straight-leg HANGS — this turns them into a
    STAND. A true actuator-feasible equilibrium (no balance assist ⇒ trackable)."""
    ctrl = w.data.ctrl.copy()
    for nm, val in (("l_knee", knee_bend_rad), ("r_knee", knee_bend_rad),
                    ("l_hip_flex", hip_flex_rad), ("r_hip_flex", hip_flex_rad),
                    ("spine_lean", 0.0)):
        if nm in w.actuator_id_by_joint:
            ctrl[w.actuator_id_by_joint[nm]] = val
    w.data.ctrl[:] = ctrl
    for _ in range(settle):
        w.step(ENV_SUBSTEPS, check_slip=False)


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
    horizon: int = 30, max_evals: int = 300, sigma0: float = 0.5,
    x0: np.ndarray | None = None, restarts: int = 0,
    balance_cap_n: float = 0.0, balance_kp: float = 500.0,
    com_drop_max: float = 0.10, max_gap_m: float | None = None,
    com_rise_reward: float = 0.0,
    stance_center_coeff: float = 0.0, stance_center_com_y: float = 0.16,
    whole_body: bool = False,
) -> tuple[list[dict], dict]:
    """CMA-ES-discover a single move from ``start_frame``. Returns
    ``(recorded_frames, info)``. ``w`` is a scratch world reused across the
    search (RSI-reset every rollout). Works for hand movers (LH/RH) and foot
    movers (LF/RF).

    ``restarts`` — if the move doesn't land, re-run CMA-ES up to this many
    extra times with escalating sigma (×1.6 per attempt) from a cold start.
    A single CMA run regularly stalls in a local minimum a wider restart
    escapes (the fragility behind every manually-stitched reference); the
    best attempt by (landed, cost) is returned.

    ``balance_cap_n`` — if > 0, a CAPPED pelvis-position balance assist holds
    the body during the swing (clamped to this many newtons ≈ what a trained
    policy could supply by weight-shifting). Open-loop discover_move otherwise
    sags when a limb releases on a steep wall; a small capped nudge closes that
    gap while keeping the recorded motion trackable (uncapped assist is what made
    reach-controller refs untrackable). The pelvis is held at its start position.

    ``stance_center_coeff`` — if > 0, penalize the LANDED stance for leaving the
    body off-balance: com_y above ``stance_center_com_y`` (leaning off the wall)
    plus the lateral (x) offset of the CoM from the centroid of the gripped-limb
    tips (the base of support). This steers a move to END centered over its
    stance, so the NEXT move launches from balance — the fix for the barn-door
    trap where a foot lands tight but leaves the body committed to one side
    (RF then unreachable from a post-LF stance). Applied to the move itself so no
    separate weight-shift keyframe is needed (those stall as imitation targets —
    a static posture with no reach goal to guide the policy in)."""
    lo = np.array([w.model.jnt_range[int(w.model.actuator_trnid[i, 0]), 0] for i in range(w.model.nu)])
    hi = np.array([w.model.jnt_range[int(w.model.actuator_trnid[i, 0]), 1] for i in range(w.model.nu)])
    dofs = _search_dofs(w, mover, whole_body=whole_body)
    target_pos = np.array(w._hold_meta_by_id[target]["world_pos"])
    grips0 = _grips_dict(start_frame["grips"])
    n_anchor0 = sum(1 for l in LIMBS if l != mover and grips0.get(l))

    # Per-DOF residual bounds spanning each joint's full ctrl range from the
    # stance pose. The old flat ±1.6 rad box silently amputated the search:
    # e.g. a high-step needs hip_flex ~2.4 rad from a hanging leg, so foot
    # moves could never even be EXPRESSED, let alone discovered.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.rsi(start_frame["qpos"], np.zeros(w.model.nv), grips0)
    ctrl0 = w.data.ctrl.copy()
    # Pad each side by 0.05 rad so x=0 (and warm starts at a joint limit) sit
    # strictly inside cma's box; the rollout clips ctrl to [lo, hi] regardless.
    blo = [float(lo[i] - ctrl0[i]) - 0.05 for i in dofs]
    bhi = [float(hi[i] - ctrl0[i]) + 0.05 for i in dofs]

    def rollout(residual: np.ndarray, record: bool = False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.rsi(start_frame["qpos"], np.zeros(w.model.nv), grips0)
        w.release_limb(mover)
        # Capped balance assist: hold the pelvis at its post-RSI (= start) pos.
        if balance_cap_n > 0.0:
            w._balance_target = w.pelvis_pos().copy()
            w._balance_force_active = True
            w._balance_cap_n = balance_cap_n
        ctrl = w.data.ctrl.copy()                    # = stance pose after rsi sync
        for k, i in enumerate(dofs):
            ctrl[i] = float(np.clip(ctrl[i] + residual[k], lo[i], hi[i]))
        w.data.ctrl[:] = ctrl
        pelvis_y0 = float(w.pelvis_pos()[1])
        com_z0 = float(start_frame["com"][2])
        com_z_min = com_z0          # track worst trough during the reach
        frames: list[dict] = []
        vel_sq = 0.0
        for _ in range(horizon):
            if record:
                frames.append(_snapshot(w))
            w.step(ENV_SUBSTEPS, check_slip=True)
            vel_sq += float(np.mean(w.data.qvel[6:] ** 2))
            com_z_min = min(com_z_min, float(w.com()[2]))
            if w.on_hold(mover) is None:
                gap = float(np.linalg.norm(target_pos - w.limb_tip_pos(mover)))
                if gap < cfg.GRIP_PROXIMITY_M:
                    # grip where the tip touched (zero strain)
                    w.attach_limb(mover, target, anchor="tip")
        if record:
            frames.append(_snapshot(w))
        vel_sq /= max(1, horizon)

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
        # A move that ends with the body >10 cm lower is not a climbing move,
        # no matter where the tip touched — touching the target from a slump
        # chains the NEXT move from a hole (observed: net −0.74 m "routes").
        landed = (w.on_hold(mover) == target) and com_drop <= com_drop_max
        # Posture strain: penalize ending in a scrunched fold (spine/hip_abduct
        # pinned at their limits). Such a stance can't support the NEXT hand
        # reach without a balance assist, so the chain stalls one move later —
        # the trackability probe's measured failure. Steering each move's end
        # pose toward extension fixes the stance the following reach starts from.
        posture = _posture_strain(w)
        # Land (gap→0) dominates; hard penalties for dropped anchors or falling;
        # posture terms keep the body upright with a smooth CoM arc, no sharp troughs.
        # Smoothness (0.4·mean qvel²): CMA otherwise authors jerky open-loop
        # reaches that no closed-loop policy can track within the R_min leash.
        # Net-rise reward: chains currently gain ~0 pelvis height (each move keeps
        # the com level — balance assist pins it + only com_drop is penalized).
        # Rewarding the com ENDING higher steers authored moves to pull the body
        # UP, so a chain accumulates real height. (Pair with a raised balance
        # target if the assist caps the rise.)
        com_rise = max(0.0, float(w.com()[2]) - com_z0)
        # Stance-center: penalize ending off-balance so the NEXT move launches
        # from a centered stance (com in toward the wall + laterally over the
        # base of support). Only when landed — an unlanded attempt's balance is
        # moot and shouldn't compete with closing the gap.
        stance_center = 0.0
        if stance_center_coeff > 0.0 and w.on_hold(mover) == target:
            com = w.com()
            com_y_pen = max(0.0, float(com[1]) - stance_center_com_y)
            tips = [w.limb_tip_pos(l) for l in LIMBS if w.on_hold(l) is not None]
            lateral = (abs(float(com[0]) - float(np.mean([t[0] for t in tips])))
                       if tips else 0.0)
            stance_center = com_y_pen + 0.5 * lateral
        cost = (10.0 * gap + 8.0 * max(0, n_anchor0 - n_anchor)
                + (25.0 if fell else 0.0)
                + 2.0 * lean_back + 1.5 * com_trough + 2.0 * com_drop
                + 1.5 * posture
                + 0.4 * vel_sq
                + 0.08 * float(np.linalg.norm(residual))
                + stance_center_coeff * stance_center
                - com_rise_reward * com_rise)
        if record:
            return cost, frames, {"gap": round(gap, 3), "landed": bool(landed),
                                  "n_anchor": n_anchor, "fell": bool(fell),
                                  "lean_back": round(lean_back, 3),
                                  "com_trough": round(com_trough, 3),
                                  "com_drop": round(com_drop, 3),
                                  "posture": round(posture, 2),
                                  "stance_center": round(stance_center, 3),
                                  "com_y": round(float(w.com()[1]), 3)}
        return cost

    # Enable the capped balance assist for the duration of this move's search.
    kp0 = cfg.BALANCE_KP
    if balance_cap_n > 0.0:
        cfg.BALANCE_KP = balance_kp
    try:
        best = None     # (landed_rank, cost, frames, info)
        for attempt in range(restarts + 1):
            # Warm start only on the first attempt — a restart exists to escape
            # the basin the warm start (or the zero pose) parked us in.
            x_init = x0 if (x0 is not None and attempt == 0) else np.zeros(len(dofs))
            # The weld settle can wrench a joint past its range, putting the
            # stance ctrl (residual 0) outside that dof's box — clip inside.
            x_init = np.clip(np.asarray(x_init, dtype=float),
                             np.asarray(blo) + 1e-3, np.asarray(bhi) - 1e-3)
            sigma = sigma0 * (1.6 ** attempt)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xbest, _es = cma.fmin2(
                    rollout, x_init, sigma,
                    {"maxfevals": max_evals, "bounds": [blo, bhi], "verbose": -9},
                )
            cost, frames, info = rollout(np.asarray(xbest), record=True)
            info["x_best"] = xbest   # expose best residual for warm-starting
            info["attempts"] = attempt + 1
            # Selection key. By default rank landed attempts by full cost
            # (posture/smoothness/lean matter for trackability). But when chasing
            # a tight landing, rank by GAP first — cost is dominated by the
            # posture term (~1.5) over 10·gap (~0.5 at small gaps), so a
            # cost-ranked best returned a loose 0.054 m attempt even though a
            # 0.017 m one was found, contradicting the chase. Tie-break by cost.
            if max_gap_m is not None:
                key = (0 if info["landed"] else 1, info["gap"], cost)
            else:
                key = (0 if info["landed"] else 1, cost)
            if best is None or key < best[0]:
                best = (key, frames, info)
            # Stop early only on a TIGHT landing. Without max_gap_m, any landing
            # inside the 0.08 m grip radius ends the search — so a loose 6 cm
            # attempt-0 result was returned and restarts never tightened it.
            # With max_gap_m set, keep restarting until the tip lands within it
            # (or restarts exhaust), returning the tightest attempt.
            tight_enough = info["landed"] and (max_gap_m is None or info["gap"] <= max_gap_m)
            if tight_enough:
                break
    finally:
        cfg.BALANCE_KP = kp0
        w._balance_force_active = False
        w._balance_target = None
        w._balance_cap_n = 0.0
    _, frames, info = best
    return frames, info


def discover_stand(
    w: Climb3DWorld, start_frame: dict, *,
    horizon: int = 32, max_evals: int = 200, sigma0: float = 0.4,
    restarts: int = 1,
) -> tuple[list[dict], dict]:
    """CMA-ES 'stand up': no mover, no target — raise the CoM over the
    current grips with all four limbs kept. Real climbing inserts this
    between a foot placement and the next reach (step, push the hips up,
    THEN reach); without it every chain ended its foot moves slumped in a
    crouch no hand reach could escape (measured: net pelvis −0.13 m on a
    chain whose foot had just gained +0.22 m). Searched DOFs: spine + both
    hips + knees + ankles."""
    names = (_SPINE + _HIPS["LF"] + _HIPS["RF"] + _KNEES + ["l_ankle", "r_ankle"])
    dofs = [w.actuator_id_by_joint[n] for n in names]
    lo = np.array([w.model.jnt_range[int(w.model.actuator_trnid[i, 0]), 0] for i in range(w.model.nu)])
    hi = np.array([w.model.jnt_range[int(w.model.actuator_trnid[i, 0]), 1] for i in range(w.model.nu)])
    grips0 = _grips_dict(start_frame["grips"])
    n_anchor0 = sum(1 for l in LIMBS if grips0.get(l))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.rsi(start_frame["qpos"], np.zeros(w.model.nv), grips0)
    ctrl0 = w.data.ctrl.copy()
    blo = [float(lo[i] - ctrl0[i]) - 0.05 for i in dofs]
    bhi = [float(hi[i] - ctrl0[i]) + 0.05 for i in dofs]
    com_z0 = float(start_frame["com"][2])

    def rollout(residual: np.ndarray, record: bool = False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.rsi(start_frame["qpos"], np.zeros(w.model.nv), grips0)
        ctrl = w.data.ctrl.copy()
        for k, i in enumerate(dofs):
            ctrl[i] = float(np.clip(ctrl[i] + residual[k], lo[i], hi[i]))
        w.data.ctrl[:] = ctrl
        pelvis_y0 = float(w.pelvis_pos()[1])
        frames: list[dict] = []
        vel_sq = 0.0
        for _ in range(horizon):
            if record:
                frames.append(_snapshot(w))
            w.step(ENV_SUBSTEPS, check_slip=True)
            vel_sq += float(np.mean(w.data.qvel[6:] ** 2))
        if record:
            frames.append(_snapshot(w))
        vel_sq /= max(1, horizon)
        com_gain = float(w.com()[2]) - com_z0
        n_anchor = sum(1 for l in LIMBS if w.on_hold(l) is not None)
        fell = float(w.pelvis_pos()[2]) < FALL_Z
        lean_back = max(0.0, float(w.pelvis_pos()[1]) - pelvis_y0)
        # Posture strain: a stand-up that raises the CoM but leaves the body
        # folded (spine/hip_abduct pinned) hasn't actually set up the next reach.
        # Extension is the whole point of standing — reward it explicitly.
        posture = _posture_strain(w)
        # Smoothness: a stand-up CMA finds without this is an open-loop jerk
        # no policy can track (every training run died at EXACTLY the frame a
        # recorded stand-up ends). Slow is smooth; smooth is learnable.
        cost = (-10.0 * com_gain + 8.0 * max(0, n_anchor0 - n_anchor)
                + (25.0 if fell else 0.0) + 2.0 * lean_back
                + 1.5 * posture
                + 0.4 * vel_sq
                + 0.05 * float(np.linalg.norm(residual)))
        if record:
            return cost, frames, {"com_gain": round(com_gain, 3),
                                  "n_anchor": n_anchor, "fell": bool(fell),
                                  "posture": round(posture, 2)}
        return cost

    best = None
    for attempt in range(restarts + 1):
        x_init = np.clip(np.zeros(len(dofs)), np.asarray(blo) + 1e-3,
                         np.asarray(bhi) - 1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbest, _es = cma.fmin2(
                rollout, x_init, sigma0 * (1.6 ** attempt),
                {"maxfevals": max_evals, "bounds": [blo, bhi], "verbose": -9},
            )
        cost, frames, info = rollout(np.asarray(xbest), record=True)
        if best is None or cost < best[0]:
            best = (cost, frames, info)
        if info["com_gain"] > 0.10 and info["n_anchor"] == n_anchor0:
            break
    _, frames, info = best
    return frames, info


def discover_shift(
    w: Climb3DWorld, start_frame: dict, *,
    com_y_target: float = 0.16, horizon: int = 32, max_evals: int = 200,
    sigma0: float = 0.4, restarts: int = 1,
) -> tuple[list[dict], dict]:
    """CMA-ES 'weight shift': no mover, no target — pull the CoM IN toward the
    wall (reduce com_y) over the current grips with all four limbs kept. This is
    the posture correction a launch needs when the body is left leaning off the
    wall after a foot move (e.g. com_y ≈ 0.5 m) and the next reach is physically
    out of range from there. Same search space and machinery as ``discover_stand``
    (which pulls com_z UP instead of com_y IN) — only the cost changes. Searched
    DOFs: spine + both hips + knees + ankles."""
    names = (_SPINE + _HIPS["LF"] + _HIPS["RF"] + _KNEES + ["l_ankle", "r_ankle"])
    dofs = [w.actuator_id_by_joint[n] for n in names]
    lo = np.array([w.model.jnt_range[int(w.model.actuator_trnid[i, 0]), 0] for i in range(w.model.nu)])
    hi = np.array([w.model.jnt_range[int(w.model.actuator_trnid[i, 0]), 1] for i in range(w.model.nu)])
    grips0 = _grips_dict(start_frame["grips"])
    n_anchor0 = sum(1 for l in LIMBS if grips0.get(l))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w.rsi(start_frame["qpos"], np.zeros(w.model.nv), grips0)
    ctrl0 = w.data.ctrl.copy()
    blo = [float(lo[i] - ctrl0[i]) - 0.05 for i in dofs]
    bhi = [float(hi[i] - ctrl0[i]) + 0.05 for i in dofs]
    com_z0 = float(start_frame["com"][2])

    def rollout(residual: np.ndarray, record: bool = False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.rsi(start_frame["qpos"], np.zeros(w.model.nv), grips0)
        ctrl = w.data.ctrl.copy()
        for k, i in enumerate(dofs):
            ctrl[i] = float(np.clip(ctrl[i] + residual[k], lo[i], hi[i]))
        w.data.ctrl[:] = ctrl
        frames: list[dict] = []
        vel_sq = 0.0
        for _ in range(horizon):
            if record:
                frames.append(_snapshot(w))
            w.step(ENV_SUBSTEPS, check_slip=True)
            vel_sq += float(np.mean(w.data.qvel[6:] ** 2))
        if record:
            frames.append(_snapshot(w))
        vel_sq /= max(1, horizon)
        com_y = float(w.com()[1])
        com_z_drop = max(0.0, com_z0 - float(w.com()[2]))
        n_anchor = sum(1 for l in LIMBS if w.on_hold(l) is not None)
        fell = float(w.pelvis_pos()[2]) < FALL_Z
        posture = _posture_strain(w)
        cost = (10.0 * max(0.0, com_y - com_y_target)
                + 8.0 * max(0, n_anchor0 - n_anchor)
                + (25.0 if fell else 0.0)
                + 2.0 * com_z_drop         # don't sag while shifting
                + 1.5 * posture
                + 0.4 * vel_sq
                + 0.05 * float(np.linalg.norm(residual)))
        if record:
            return cost, frames, {"com_y_end": round(com_y, 3),
                                  "com_z_drop": round(com_z_drop, 3),
                                  "n_anchor": n_anchor, "fell": bool(fell),
                                  "posture": round(posture, 2)}
        return cost

    best = None
    for attempt in range(restarts + 1):
        x_init = np.clip(np.zeros(len(dofs)), np.asarray(blo) + 1e-3,
                         np.asarray(bhi) - 1e-3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xbest, _es = cma.fmin2(
                rollout, x_init, sigma0 * (1.6 ** attempt),
                {"maxfevals": max_evals, "bounds": [blo, bhi], "verbose": -9},
            )
        cost, frames, info = rollout(np.asarray(xbest), record=True)
        if best is None or cost < best[0]:
            best = (cost, frames, info)
        if info["com_y_end"] < com_y_target + 0.02 and info["n_anchor"] == n_anchor0:
            break
    _, frames, info = best
    return frames, info


def stance_route_from_reference(ref: Reference) -> list[dict[str, str]]:
    """Extract the sequence of STABLE 4-grip stances (hold-sets) from a dense
    reference — the route skeleton, with the untrackable transition frames
    discarded. Returns ``[{"LH": id, "RH": id, "LF": id, "RF": id}, …]``, one
    entry per distinct stance where all four limbs are gripped."""
    route: list[dict[str, str]] = []
    for t in range(len(ref)):
        g = {LIMBS[i]: (ref.grips[t, i] or "") for i in range(4)}
        if not all(g.values()):
            continue                       # mid-move (a limb in flight)
        if not route or g != route[-1]:
            route.append(dict(g))
    return route


def author_stance_reference(
    wall: Wall, profile: ClimberProfile, route: list[dict[str, str]], *,
    settle_dwell: int = 4, wall_gen_seed: int = 7,
    knee_bend_rad: float = 0.90, hip_flex_rad: float = 0.55,
    posture_settle: int = 24,
) -> tuple[Reference, list[dict]]:
    """Author a STANCE-KEYFRAME reference: one settled, welded, *climbing-posture*
    stance per hold-set in ``route``. The transitions between stances are
    deliberately NOT authored — the policy learns to balance through them (the
    2026-06-15 reframe: discover owns route + stance geometry, RL owns the move).

    Each stance is produced by ``seed_pose`` (welds yank the body into a
    self-consistent rest pose), then **corrected into a climbing posture**:
    seed_pose alone leaves the legs near-straight (pelvis ~0.85·leg above the
    feet ⇒ knees ~0°) and the spine pinned at its lean limit — a HANG, not a
    STAND. A climber bends the knees on nearly every move and keeps the torso
    upright/close to the wall. So we re-settle with actuators ON, driving the
    knees to ``knee_bend_rad`` and spine_lean→0; the bent knees drop the hips
    DOWN and BACK over the feet (room for the shin to come forward / the foot to
    travel straight up on the next move), and the stiff spine actuator pulls the
    torso off its lean stop. ``settle_dwell`` identical frames per stance give a
    milestone imitation env a dwell target. ``move_starts`` marks each stance's
    first frame.

    Returns ``(Reference, per-stance diagnostics)`` reporting knee angle, spine
    lean, pelvis y/z, posture strain, intersections, and grips so a bad stance
    is caught before training."""
    diag: list[dict] = []
    frames: list[dict] = []
    move_starts: list[int] = []
    for k, hs in enumerate(route):
        kw = {l.lower(): hs[l] for l in ("LH", "RH", "LF", "RF") if hs.get(l)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = Climb3DWorld(wall, profile)
            w.seed_pose(**kw)
            _apply_climbing_posture(w, knee_bend_rad=knee_bend_rad,
                                    hip_flex_rad=hip_flex_rad, settle=posture_settle)
        move_starts.append(len(frames))
        for _ in range(settle_dwell):
            frames.append(_snapshot(w))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.step(ENV_SUBSTEPS, check_slip=False)
        n_grip = sum(1 for l in LIMBS if w.on_hold(l) is not None)
        kn = lambda nm: float(np.degrees(w.data.qpos[int(w.model.jnt_qposadr[
            int(w.model.actuator_trnid[w.actuator_id_by_joint[nm], 0])])]))
        diag.append({
            "stance": k, "holds": kw, "grips_held": n_grip,
            "l_knee": round(kn("l_knee"), 1), "r_knee": round(kn("r_knee"), 1),
            "spine_lean": round(kn("spine_lean"), 1),
            "posture": round(_posture_strain(w), 2),
            "intersections": int(w.body_intersection_count()),
            "pelvis_y": round(float(w.pelvis_pos()[1]), 3),
            "pelvis_z": round(float(w.pelvis_pos()[2]), 3),
        })

    ref = Reference(
        qpos=np.array([f["qpos"] for f in frames]),
        qvel=np.array([f["qvel"] for f in frames]),
        eef=np.array([f["eef"] for f in frames]),
        com=np.array([f["com"] for f in frames]),
        grips=np.array([f["grips"] for f in frames], dtype="<U24"),
        wall_gen_seed=wall_gen_seed,
        meta={"method": "stance-keyframe", "move_starts": move_starts,
              "n_stances": len(route)},
    )
    return ref, diag


def author_dense_from_stances(
    wall: Wall, profile: ClimberProfile, route: list[dict[str, str]], *,
    settle_dwell: int = 4, knee_bend_rad: float = 0.90, hip_flex_rad: float = 0.55,
    posture_settle: int = 20, horizon: int = 30, max_evals: int = 250,
    restarts: int = 1,
    balance_cap_n: float = 0.0, wall_gen_seed: int = 7,
    shift_com_y_target: float = 0.16, insert_foot_shift: bool = False,
    foot_stance_center_coeff: float = 0.0,
) -> tuple[Reference, list[dict]]:
    """Synthesis (2026-06-15): a DENSE reference built from a stance route by
    connecting each consecutive *posture-corrected* stance with a ``discover_move``
    transition. This reconciles the two failed lines:

      - stance-keyframe milestone training failed because pure stances have NO
        mid-transition frames to RSI into (the swing is undiscoverable);
      - dense reach-controller refs failed because their transitions are
        untrackable.

    Here the transition is authored by ``discover_move`` (action-space feasible —
    the probe lands moves from good stances at ~100%), so it BOTH supplies
    mid-transition frames the chain trainer can RSI into AND is reproducible by
    the PD servos. Each landing is posture-corrected back to a climbing stance
    (``_apply_climbing_posture``) so the good posture + bent knees carry through
    the whole climb. Train the result with the existing chain machinery
    (``--chain --free-mover-imitation --mover-grip-bonus``), which already learned
    single moves to 87%.

    Stops at the first transition ``discover_move`` can't land (a genuine
    infeasibility from that stance)."""
    def _next_mover_is_foot(k: int) -> bool:
        """Is the move LEAVING stance k a foot move? Hips back (hip_flex) helps
        a FOOT clear and step up, but moves the HANDS away from the wall — so we
        flex the hips only when a foot moves next, and keep hips IN (hip_flex 0)
        before a hand reach. The dynamic shift a real climber makes."""
        if k + 1 >= len(route):
            return False
        g0, g1 = route[k], route[k + 1]
        mv = next((l for l in ("LH", "RH", "LF", "RF")
                   if g1.get(l) and g1.get(l) != g0.get(l)), None)
        return mv in ("LF", "RF")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
        kw0 = {l.lower(): route[0][l] for l in ("LH", "RH", "LF", "RF") if route[0].get(l)}
        w.seed_pose(**kw0)
        _apply_climbing_posture(w, knee_bend_rad=knee_bend_rad,
                                hip_flex_rad=(hip_flex_rad if _next_mover_is_foot(0) else 0.0),
                                settle=posture_settle)

    frames: list[dict] = []
    move_starts: list[int] = []
    diag: list[dict] = []
    for _ in range(settle_dwell):              # stance-0 dwell (pre-roll)
        frames.append(_snapshot(w))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.step(ENV_SUBSTEPS, check_slip=False)
    start = _snapshot(w)

    for k in range(1, len(route)):
        g_prev, g_now = route[k - 1], route[k]
        mover = next((l for l in ("LH", "RH", "LF", "RF")
                      if g_now.get(l) and g_now.get(l) != g_prev.get(l)), None)
        if mover is None:
            continue
        target = g_now[mover]
        if target not in w._hold_meta_by_id:
            diag.append({"status": f"transition {k-1}->{k}: unknown target {target}"})
            break
        move_starts.append(len(frames))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            is_foot = mover in ("LF", "RF")
            move_frames, info = discover_move(
                w, start, mover, target,
                horizon=horizon, max_evals=max_evals,
                restarts=restarts, balance_cap_n=balance_cap_n,
                com_drop_max=(0.30 if is_foot else 0.10),
                stance_center_coeff=(foot_stance_center_coeff if is_foot else 0.0),
                stance_center_com_y=shift_com_y_target)
        frames.extend(move_frames)
        diag.append({"transition": f"{k-1}->{k}", "mover": mover, "target": target,
                     "landed": bool(info["landed"]), "gap": round(info["gap"], 3),
                     "com_y": info.get("com_y"),
                     "stance_center": info.get("stance_center")})
        print(f"  transition {k-1}->{k}: {mover}->{target}  gap={info['gap']:.3f}m  "
              f"landed={info['landed']}  com_y={info.get('com_y')}", flush=True)
        if not info["landed"]:
            diag.append({"status": f"aborted: transition {k-1}->{k} not feasible "
                                   f"(gap {info['gap']:.3f} m)"})
            break
        # Posture-correct the landing into stance k; record its dwell. Hips back
        # only if the NEXT move is a foot (else hips in for the coming hand reach).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _apply_climbing_posture(
                w, knee_bend_rad=knee_bend_rad,
                hip_flex_rad=(hip_flex_rad if _next_mover_is_foot(k) else 0.0),
                settle=posture_settle)
        for _ in range(settle_dwell):
            frames.append(_snapshot(w))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                w.step(ENV_SUBSTEPS, check_slip=False)
        start = _snapshot(w)

        # Stand-up: after a foot move followed by a hand move, raise the CoM
        # before the next reach so the hand can clear the next rung.  Without
        # this the body stays slumped from the foot placement and the hand reach
        # starts from a hole (measured: net -0.65 m after foot moves on this
        # wall, making the next hand hold 0.47 m out of range).
        next_is_foot = _next_mover_is_foot(k)
        if is_foot and not next_is_foot and k + 1 < len(route):
            print(f"  stand-up after foot move {k-1}->{k}…", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stand_frames, stand_info = discover_stand(
                    w, start, horizon=horizon, max_evals=max_evals,
                    restarts=restarts)
            frames.extend(stand_frames)
            print(f"    com_gain={stand_info['com_gain']:.3f}m  "
                  f"n_anchor={stand_info['n_anchor']}", flush=True)
            for _ in range(settle_dwell):
                frames.append(_snapshot(w))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w.step(ENV_SUBSTEPS, check_slip=False)
            start = _snapshot(w)

        # Weight-shift: after a foot move followed by ANOTHER foot move (the
        # free leg is still in the air, weight left on the other side after
        # the previous step), pull the CoM back toward the wall over the
        # planted limbs before authoring the next foot's reach. Mirrors the
        # stand-up block above (which raises com_z before a HAND move); this
        # one reduces com_y before a FOOT move. Without it the next foot
        # launches from a lopsided, leaning-out stance it can't reach from
        # (the barn-door trap — RF unreachable at 0.12-0.19 m from a
        # committed post-LF stance, never fixed by RL-side reward shaping;
        # baking the shift into the reference is the fix).
        if insert_foot_shift and is_foot and next_is_foot and k + 1 < len(route):
            print(f"  weight-shift after foot move {k-1}->{k}…", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shift_frames, shift_info = discover_shift(
                    w, start, com_y_target=shift_com_y_target,
                    horizon=horizon, max_evals=max_evals, restarts=restarts)
            frames.extend(shift_frames)
            print(f"    com_y_end={shift_info['com_y_end']:.3f}m  "
                  f"n_anchor={shift_info['n_anchor']}", flush=True)
            for _ in range(settle_dwell):
                frames.append(_snapshot(w))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w.step(ENV_SUBSTEPS, check_slip=False)
            start = _snapshot(w)

    ref = Reference(
        qpos=np.array([f["qpos"] for f in frames]),
        qvel=np.array([f["qvel"] for f in frames]),
        eef=np.array([f["eef"] for f in frames]),
        com=np.array([f["com"] for f in frames]),
        grips=np.array([f["grips"] for f in frames], dtype="<U24"),
        wall_gen_seed=wall_gen_seed,
        meta={"method": "dense-from-stances", "move_starts": move_starts,
              "n_transitions": len(move_starts)},
    )
    return ref, diag


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
            w, start, mover, target, horizon=horizon, max_evals=max_evals,
            sigma0=sigma0, restarts=1)
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


def _reachable_holds(
    w: Climb3DWorld, start_frame: dict, mover: str, *,
    max_dist_m: float = 0.50, max_probe_evals: int = 100,
    visited: set[str] | None = None, min_z_gain_m: float = 0.08,
) -> list[tuple[float, str, np.ndarray]]:
    """Return ``[(gap, hold_id)]`` for holds within ``max_dist_m`` of the mover
    tip at ``start_frame``, sorted by ascending probe gap.

    ``visited`` — hold IDs already gripped in this climb; excluded to prevent
    oscillation (the greedy nearest-hold strategy revisits released holds without
    this filter). Holds gripped by ANY limb at ``start_frame`` are always
    excluded — without this, a limb's own hold is the easiest "move" in every
    probe and greedy selection authors wiggle-in-place references (the
    2026-06-11 degenerate 3-move reference: feet "stepping" onto the holds
    they already stood on, net pelvis movement −3 cm).
    ``min_z_gain_m`` — candidate must be at least this high above the mover's
    *current* tip z-coordinate, enforcing upward progress. Default 0.08:
    zero-strain anchors leave tips a few cm BELOW their hold's centre, so a
    2 cm threshold let a limb's own hold count as "up".
    """
    tip = np.array(start_frame["eef"][LIMBS.index(mover)])
    tip_z = float(tip[2])
    visited = set(visited or set())
    visited.update(g for g in start_frame["grips"] if g)   # currently-held holds
    # Foot ceiling: feet may not climb above lowest_hand − 0.25 m. Reaches
    # need a 0.5–0.7 m hand−foot push window (ladder sweep 2026-06-12), but
    # the ceiling must leave room above the +0.08 gain floor or foot moves
    # deadlock; a mild interim crouch is fine — the stand-up move converts
    # it into reach height before the hands go again.
    foot_ceiling = None
    if mover in ("LF", "RF"):
        hand_zs = [float(start_frame["eef"][LIMBS.index(l)][2])
                   for l in ("LH", "RH")]
        foot_ceiling = min(hand_zs) - 0.25
    candidates = []
    for hid, meta in w._hold_meta_by_id.items():
        if hid in visited:
            continue
        # Eligibility: hands can never use foothold-type holds (the training
        # env refuses the grip, so a reference using one is untrackable).
        if mover in ("LH", "RH") and meta.get("is_foothold_only"):
            continue
        hold_pos = np.array(meta["world_pos"])
        if foot_ceiling is not None and hold_pos[2] > foot_ceiling:
            continue
        dist = float(np.linalg.norm(hold_pos - tip))
        if dist >= max_dist_m:
            continue
        if hold_pos[2] < tip_z + min_z_gain_m:
            continue   # must go up
        candidates.append((dist, hid))
    candidates.sort()
    results = []
    for _d, hid in candidates[:8]:   # probe at most 8 closest upward holds
        _, info = discover_move(w, start_frame, mover, hid,
                                max_evals=max_probe_evals, sigma0=0.4)
        # Return x_best so the main loop can warm-start from the probe solution,
        # avoiding probe/main-loop inconsistency (cold-start with more evals
        # sometimes diverges from a good early solution found with fewer evals).
        results.append((info["gap"], hid, info["x_best"]))
    # Aim HIGH, not near: among plausibly-landable probes (gap < 0.25 m —
    # full budget + restarts usually close that), prefer the highest hold.
    # Ranking by smallest gap made greedy discovery author timid micro-move
    # routes on dense walls (and, before held-hold exclusion, outright
    # wiggle-in-place re-grips). Hopeless probes keep the by-gap order as a
    # fallback tail.
    # FEET rank by closest gap, not highest hold. The "aim high" heuristic is
    # for hands (avoid timid micro-moves); applied to feet it picks a foothold
    # higher than the foot's ~5-6 cm open-loop reach ceiling, so the foot never
    # grips (measured 2026-06-19: the cycle kept trying 10-16 cm footholds and
    # skipping the 6 cm one that DOES grip). For feet, the closest reachable
    # foothold is the only one that closes, so try it first.
    if mover in ("LF", "RF"):
        results.sort(key=lambda r: r[0])
        return results
    plausible = [r for r in results if r[0] < 0.25]
    hopeless = [r for r in results if r[0] >= 0.25]
    plausible.sort(key=lambda r: -float(w._hold_meta_by_id[r[1]]["world_pos"][2]))
    hopeless.sort(key=lambda r: r[0])
    return plausible + hopeless


def select_first_move(
    wall: Wall, profile: ClimberProfile, feas: list[dict], *,
    max_candidates: int = 8, probe_evals: int = 250, sigma0: float = 0.5,
    restarts: int = 1, max_gap_m: float = 0.05,
) -> tuple[dict, dict]:
    """Probe the feasible first-moves and return the one that lands TIGHTEST.

    ``feasible_reach_moves`` only vets hang-stability + a 2D reach distance
    (≤ max_reach_m); it never checks the move actually closes to the grip
    radius under the 3D body. Greedy ``feas[0]`` therefore often starts the
    whole chain on an unclosable move (seed 14: LH→h_049 is "feasible" but CMA
    lands it at 0.37 m even full-strength). Rank candidates by their REAL CMA
    landing gap instead. Returns ``(best_move, best_info)``.

    Probes the closest-reach candidates first (smallest ``reach_d0``) and caps
    at ``max_candidates`` to bound cost; uses light ``probe_evals`` since this
    is a ranker — the chosen move is re-run at full budget by discovery.
    """
    cands = sorted(feas, key=lambda m: m.get("reach_d0", 1e9))[:max_candidates]
    scored: list[tuple[float, bool, dict, dict]] = []
    for m in cands:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w = Climb3DWorld(wall, profile)
            w.seed_pose(**m["seed_kwargs"])
        start = _snapshot(w)
        _frames, info = discover_move(
            w, start, m["mover"], m["target"], max_evals=probe_evals,
            sigma0=sigma0, restarts=restarts, max_gap_m=max_gap_m,
        )
        tight = bool(info["landed"]) and info["gap"] <= max_gap_m
        scored.append((info["gap"], tight, m, info))
        print(f"  first-move probe: {m['mover']}→{m['target']} "
              f"(k={m['move_k']}, reach_d0={m.get('reach_d0')})  "
              f"gap={info['gap']:.3f}m  landed={info['landed']}  tight={tight}",
              flush=True)
    # Prefer a tight landing, then any landing, then smallest gap.
    scored.sort(key=lambda s: (0 if s[1] else 1, 0 if s[3]["landed"] else 1, s[0]))
    best_gap, _tight, best_move, best_info = scored[0]
    print(f"  → selected first move {best_move['mover']}→{best_move['target']} "
          f"(k={best_move['move_k']}, probe gap {best_gap:.3f}m)", flush=True)
    return best_move, best_info


def discover_climb_adaptive(
    wall: Wall, profile: ClimberProfile, seed_move: dict, *,
    max_moves: int = 12, horizon: int = 40, max_evals: int = 350,
    sigma0: float = 0.5, settle_pre: int = 2, wall_gen_seed: int = 7,
    max_gap_m: float = 0.04, probe_evals: int = 80, restarts: int = 1,
    foot_balance_cap_n: float = 0.0, seed_x0: np.ndarray | None = None,
    foot_max_gap_m: float | None = None,
) -> tuple[Reference, list[dict]]:
    """Adaptive CMA-ES climb discovery: after each move, probe all reachable
    holds for the next free hand and greedily pick the closest-landing one.

    This fixes the static-feasibility bug in ``discover_climb``: the pre-computed
    ``feasible_reach_moves`` list measures reachability from the initial seed
    stance, but mid-climb stances are different — what looks reachable from the
    bottom is often unreachable after 2-3 moves, and vice versa. By probing from
    the *actual* end-of-move stance we discover whatever is genuinely reachable
    next, even if it wasn't in the original feasible list.

    ``seed_move`` is the first move dict (provides ``seed_kwargs``, ``mover``,
    ``target``). Subsequent moves alternate hands and greedily pick the best
    discoverable hold within ``max_gap_m`` after the full-budget CMA-ES search.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
        w.seed_pose(**seed_move["seed_kwargs"])

    frames: list[dict] = [_snapshot(w) for _ in range(settle_pre)]
    start = _snapshot(w)
    diag: list[dict] = []
    move_starts: list[int] = []
    # Holds excluded from future candidates: everything ever gripped,
    # INCLUDING the seed stance. Without seeding these, a vacated start hold
    # becomes a "new" target later (observed: a foot climbing onto the
    # hand's old start hold, corkscrewing the body 0.74 m downward).
    visited: set[str] = {g for g in start["grips"] if g}

    # Alternate movers: start with seed_move's mover, then flip each step.
    current_mover = seed_move["mover"]
    current_target = seed_move["target"]

    for step in range(max_moves):
        if current_target not in w._hold_meta_by_id:
            diag.append({"step": step, "status": f"unknown target {current_target}"})
            break

        move_starts.append(len(frames))
        # Warm-start the FIRST move from the selector's probe solution. CMA
        # landing gap has high run-to-run variance (same move: 0.013 m one run,
        # 0.062 m the next); without this the full-budget step-0 search cold-
        # starts and often lands in a looser basin than the probe already found,
        # so auto-selected moves were rejected by the tight bar they'd passed.
        move_x0 = seed_x0 if step == 0 else None
        move_frames, info = discover_move(
            w, start, current_mover, current_target,
            horizon=horizon, max_evals=max_evals, sigma0=sigma0, restarts=restarts,
            max_gap_m=max_gap_m, x0=move_x0,
        )
        entry = {"step": step, "mover": current_mover,
                 "target": current_target, **info}
        diag.append(entry)
        # Tight-landing bar: a move that grips but parks at gap ~0.07 m trains
        # poorly (the policy can't reliably get the tip inside 0.08 to grip).
        # Require gap ≤ max_gap_m, not just the loose `landed` (on_hold) flag.
        tight = bool(info["landed"]) and info["gap"] <= max_gap_m
        print(f"  step {step}: {current_mover}→{current_target}  "
              f"gap={info['gap']:.3f}m  landed={info['landed']}  "
              f"tight={tight} (≤{max_gap_m:.2f})", flush=True)
        frames.extend(move_frames)

        if not tight:
            why = "did not grip" if not info["landed"] else f"gap {info['gap']:.3f}m > {max_gap_m:.2f}m (marginal)"
            msg = f"aborted at step {step}: {why} on {current_mover}→{current_target}"
            diag.append({"status": msg})
            print(f"  {msg}", flush=True)
            break

        visited.add(current_target)   # mark this hold as used
        start = move_frames[-1]

        # After a foot placement, STAND UP on it (no-target CoM-raise move)
        # before probing — reaches are probed from the risen stance, which is
        # what makes a foot move actually buy the hands anything.
        if current_mover in _LEG:
            stand_frames, stand_info = discover_stand(w, start)
            print(f"  stand-up after {current_mover}: com {stand_info['com_gain']:+.3f} m  "
                  f"anchors {stand_info['n_anchor']}", flush=True)
            if stand_info["com_gain"] >= 0.05 and not stand_info["fell"]:
                frames.extend(stand_frames)
                start = stand_frames[-1]
                diag.append({"stand_after": current_target, **stand_info})

        # Advance through the 4-limb cycle (LH→RH→LF→RF→…), but don't abort
        # just because the next limb in line has nothing reachable — walk the
        # cycle until SOME limb has a discoverable move. Foot moves raise the
        # body so hands can reach the next row.
        cycle_idx = _CYCLE.index(current_mover)
        found = False
        for advance in range(1, 5):
            next_mover = _CYCLE[(cycle_idx + advance) % 4]
            # Foot movers search a larger radius (leg reach >> arm reach).
            max_dist = 0.55 if next_mover in _LEG else 0.80   # hands probe wide:
                # from a post-foot-step crouch the reachable holds sit 0.7-1.0 m
                # from the tip; knee extension (stand-up) closes them, but only
                # if the probe radius lets them be candidates at all
            # Feet: lower the upward-gain floor (default 0.08). The open-loop
            # foot reaches only ~5-6 cm up, so an 8 cm floor excludes every
            # foothold the foot can actually grip. 0.04 lets a small (5-7 cm)
            # foot step be an eligible target — the foot CAN grip those.
            min_gain = 0.04 if next_mover in _LEG else 0.08
            print(f"  probing {next_mover} candidates (visited={len(visited)})…", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidates = _reachable_holds(w, start, next_mover,
                                              max_dist_m=max_dist,
                                              max_probe_evals=probe_evals,
                                              visited=visited, min_z_gain_m=min_gain)
            if not candidates:
                diag.append({"status": f"no holds within range for {next_mover} "
                                       f"after step {step}"})
                continue
            # The probe ranks candidates cheaply. Now run full-budget on the
            # top ones (sorted by probe gap) until one lands. This always
            # tries the full budget — the probe is a ranker, not a filter.
            for probe_gap, hid, probe_x in candidates[:4]:
                print(f"    trying {next_mover}→{hid} (probe_gap={probe_gap:.3f}m)…", flush=True)
                # Foot moves: the open-loop swing tops out ~5-6 cm above its hang
                # start and won't close to a tight grip (NOT fixable by balance
                # assist or foothold spacing — measured 2026-06-19). So accept a
                # looser landing for feet (foot_max_gap_m, default = max_gap_m):
                # a foot that GRIPS inside the 0.08 m capture sphere is trainable
                # by free_mover_imitation even though it isn't tight. Hands keep
                # the tight bar. balance_cap_n is opt-in (default off — no effect).
                is_foot = next_mover in _LEG
                gap_bar = (foot_max_gap_m if foot_max_gap_m is not None else max_gap_m) if is_foot else max_gap_m
                bcap = foot_balance_cap_n if is_foot else 0.0
                _, full_info = discover_move(w, start, next_mover, hid,
                                             horizon=horizon, max_evals=max_evals,
                                             sigma0=sigma0 * 0.5,   # tighter sigma — warm-starting
                                             x0=probe_x, restarts=restarts,
                                             max_gap_m=gap_bar, balance_cap_n=bcap)
                diag.append({"probe": {"mover": next_mover, "target": hid,
                                       "probe_gap": round(probe_gap, 3),
                                       "full_gap": round(full_info["gap"], 3),
                                       "landed": full_info["landed"],
                                       "warm_start": probe_x is not None}})
                full_tight = bool(full_info["landed"]) and full_info["gap"] <= gap_bar
                print(f"    → gap={full_info['gap']:.3f}m  landed={full_info['landed']}  "
                      f"accept={full_tight} (≤{gap_bar:.2f})", flush=True)
                if full_tight:
                    current_mover, current_target = next_mover, hid
                    found = True
                    break
            if found:
                break
            diag.append({"status": f"no discoverable hold for {next_mover} "
                                   f"after step {step} (tried {len(candidates[:4])})"})
        if not found:
            diag.append({"status": f"aborted after step {step}: no limb has a "
                                   f"discoverable upward move"})
            break

    ref = Reference(
        qpos=np.array([f["qpos"] for f in frames]),
        qvel=np.array([f["qvel"] for f in frames]),
        eef=np.array([f["eef"] for f in frames]),
        com=np.array([f["com"] for f in frames]),
        grips=np.array([f["grips"] for f in frames], dtype="<U24"),
        wall_gen_seed=wall_gen_seed,
        meta={"discovered": [d.get("target") for d in diag if "mover" in d],
              "move_starts": move_starts, "method": "cma-es-adaptive"},
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
    ap.add_argument("--wall", type=str, default=None,
                    help="explicit wall JSON (e.g. data/examples/ladder-v1.json) — "
                         "bypasses the generated tight wall; route-shaped walls "
                         "are where chains actually extend")
    ap.add_argument("--reach-frac", type=float, default=0.42,
                    help="hold spacing as fraction of arm length (lower = reachable)")
    ap.add_argument("--moves", type=str, default="9,10,11",
                    help="comma-separated move_k values to discover (consecutive)")
    ap.add_argument("--max-evals", type=int, default=160)
    ap.add_argument("--out", type=str, default="data/runs/sim3d/imitation/ref_cma.npz")
    ap.add_argument("--stances", action="store_true",
                    help="author a STANCE-KEYFRAME reference (settled welded "
                         "stances only; transitions left to the policy) for the "
                         "route extracted from --from-ref. The 2026-06-15 reframe.")
    ap.add_argument("--from-ref", type=str, default=None,
                    help="--stances: dense reference whose 4-grip route is "
                         "re-authored as a stance skeleton")
    ap.add_argument("--dense", action="store_true",
                    help="--stances --dense: author a DENSE reference (discover_move "
                         "transitions between posture-corrected stances) instead of a "
                         "bare stance skeleton — trainable with --chain. The synthesis.")
    ap.add_argument("--balance-assist", type=float, default=0.0,
                    help="--dense: capped pelvis balance-assist force (N) during the "
                         "swing (≈ what a policy could weight-shift; 0 = off). Closes "
                         "the residual sag on steep walls while staying trackable.")
    ap.add_argument("--restarts", type=int, default=1,
                    help="--dense: CMA restarts per transition (default 1; raise to 2-3 "
                         "for marginal transitions near the grip radius).")
    ap.add_argument("--shift-com-y", type=float, default=0.16,
                    help="--dense: target com_y (m from wall) for both the "
                         "stance-center landing term and the (opt-in) auto-inserted "
                         "weight-shift. Lower = hug the wall harder.")
    ap.add_argument("--foot-stance-center", type=float, default=0.0,
                    help="--dense: penalty coeff steering FOOT moves to LAND "
                         "centered (com in toward wall + over the base of support) "
                         "so the next move launches from balance — no separate "
                         "weight-shift keyframe needed. 0 = off. Try 3.0.")
    ap.add_argument("--insert-foot-shift", action="store_true",
                    help="--dense: auto-insert a discover_shift weight-shift stance "
                         "between two consecutive foot moves (the separate-keyframe "
                         "approach). Prefer --foot-stance-center instead — shift "
                         "keyframes stall as static imitation targets.")
    ap.add_argument("--adaptive", action="store_true",
                    help="use adaptive discovery (probe reachable holds from each "
                         "end-of-move stance instead of a fixed move list)")
    ap.add_argument("--max-moves", type=int, default=12,
                    help="max moves to discover in adaptive mode")
    ap.add_argument("--max-gap", type=float, default=0.04,
                    help="adaptive mode: tight landing bar (m). A move is only "
                         "accepted if its tip lands within this gap of the hold, "
                         "not merely inside the 0.08 m grip radius. Marginal "
                         "landings (gap ~0.07) grip but don't train — see "
                         "NEXT_STEPS 'Landing tightness'. Default 0.04.")
    ap.add_argument("--first-move", type=str, default=None,
                    help="adaptive mode: 'move_k' of the first move (default: first feasible)")
    ap.add_argument("--auto-first-move", action="store_true",
                    help="adaptive mode: probe the feasible first-moves and pick "
                         "the tightest-landing one instead of feas[0]. feas[0] is "
                         "only 2D-vetted and is often unclosable by the 3D body.")
    ap.add_argument("--foot-max-gap", type=float, default=None,
                    help="adaptive mode: separate (looser) landing bar for FOOT "
                         "moves. Open-loop foot reach tops out ~5-6 cm short, so a "
                         "foot that merely GRIPS (gap < 0.08) is trainable by "
                         "free_mover_imitation. Hands keep --max-gap. Default: "
                         "same as --max-gap.")
    ap.add_argument("--foot-balance", type=float, default=0.0,
                    help="adaptive mode: capped pelvis balance-assist force (N) "
                         "during FOOT-move swings only. The open-loop foot reach "
                         "sags ~9 cm short of any target (leg-lift-under-hang); a "
                         "small cap (~150-300 N ≈ a policy's weight-shift) closes "
                         "it. 0 = off. Hands never use it.")
    ap.add_argument("--continue-from", type=str, default=None,
                    help="adaptive mode: continue from the last frame of an existing "
                         "reference .npz (loads the matching .wall.json alongside it); "
                         "bypasses seed_pose, starting from the real end state of a "
                         "previous discovery run. Use with --adaptive.")
    args = ap.parse_args()

    # --continue-from: load an existing reference and continue discovery from its
    # last frame. The wall JSON is expected at <ref>.wall.json next to the ref.
    if args.continue_from and args.adaptive:
        from sim3d.reference import Reference, migrate_reference_spine
        from solver.wall import DEFAULT_CELL_SIZE_CM, load_wall
        ref_path = Path(args.continue_from)
        wall_path = ref_path.with_suffix(".wall.json")
        if not ref_path.exists():
            print(f"reference not found: {ref_path}")
            return
        if not wall_path.exists():
            print(f"wall JSON not found: {wall_path}  (expected alongside {ref_path})")
            return
        prior_ref = migrate_reference_spine(Reference.load(ref_path))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wd = json.loads(wall_path.read_text())
            cell = wd.get("grid", {}).get("cell_size_cm")
            wall = load_wall(wd, cell_size_cm=cell or DEFAULT_CELL_SIZE_CM)
        am.validate_reference(prior_ref, wall, path=ref_path)
        profile = ClimberProfile()
        w = Climb3DWorld(wall, profile)
        # Build start_frame from the last STABLE frame (all 4 limbs gripped).
        # The absolute last frame may have the last mover released (mid-move),
        # so scan backward for the last frame where every gripped slot is set.
        stable_t = len(prior_ref) - 1
        for t in range(len(prior_ref) - 1, -1, -1):
            g = prior_ref.frame_grips(t)
            if all(v for v in g.values()):
                stable_t = t
                break
        print(f"  using stable frame {stable_t}/{len(prior_ref)-1} "
              f"(grips: {prior_ref.frame_grips(stable_t)})")
        last_qpos = prior_ref.qpos[stable_t]
        last_grips = prior_ref.frame_grips(stable_t)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w.rsi(last_qpos, np.zeros(w.model.nv), last_grips)
        start_frame = _snapshot(w)
        # Determine the next mover. For CMA-ES discovered refs, the last move's
        # mover is encoded in the meta "discovered" list as hold IDs; we infer
        # from the stable frame grips which limb moved last (the one just gripped).
        # For stitched refs the meta has "stitched[].mover".
        gripped = [l for l in LIMBS if start_frame["grips"][LIMBS.index(l)]]
        # Infer the last mover: scan BACKWARD for the most recent frame where
        # any limb's grip changed (the old "compare the last two frames only"
        # version missed movers whose regrip happened several settle frames
        # before the end, silently fell back to "RH", and pointed the
        # continuation at the wrong limb).
        last_mover_in_prior = prior_ref.meta.get("last_mover", "RH")
        for t in range(stable_t, 0, -1):
            prev_g, cur_g = prior_ref.frame_grips(t - 1), prior_ref.frame_grips(t)
            changed = [l for l in LIMBS if prev_g[l] != cur_g[l] and cur_g[l]]
            if changed:
                last_mover_in_prior = changed[0]
                break
        # Collect already-visited holds from the prior reference's grips.
        visited: set[str] = set()
        for t in range(len(prior_ref)):
            for l in LIMBS:
                g = prior_ref.grips[t, LIMBS.index(l)]
                if g:
                    visited.add(g)
        print(f"  visited holds from prior ref: {len(visited)}")
        # Run adaptive from this start_frame, not from a seed_pose, walking
        # the full 4-limb cycle. (Foot moves used to be excluded here because
        # the hip couldn't raise a foot past hip height — fixed 2026-06-11 by
        # the widened hip_rot/hip_abduct limits; see config.JOINT_LIMITS_RAD.)
        if last_mover_in_prior in _CYCLE:
            next_mover = _CYCLE[(_CYCLE.index(last_mover_in_prior) + 1) % 4]
        else:
            next_mover = "LH"
        print(f"Continuing from {ref_path.name}: {len(prior_ref)} prior frames, "
              f"grips={gripped}, last_mover={last_mover_in_prior}, next_mover={next_mover}")
        from_ref_diag: list[dict] = []
        frames: list[dict] = [start_frame]
        move_starts: list[int] = []
        current_mover = None
        current_target = None
        probe_x = None
        # Walk the full cycle from the inferred next mover — don't give up
        # because one limb has nothing reachable (mirror of the main adaptive
        # loop's cycle-walking).
        start_idx = _CYCLE.index(next_mover)
        for advance in range(4):
            mover = _CYCLE[(start_idx + advance) % 4]
            max_dist = 0.55 if mover in _LEG else 0.80
            print(f"  probing {mover} candidates (visited={len(visited)})…", flush=True)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                candidates = _reachable_holds(w, start_frame, mover,
                                              max_dist_m=max_dist,
                                              max_probe_evals=80,
                                              visited=visited)
            if not candidates:
                continue
            for probe_gap, hid, px in candidates[:4]:
                print(f"    trying {mover}→{hid} (probe_gap={probe_gap:.3f}m)…", flush=True)
                mf, full_info = discover_move(w, start_frame, mover, hid,
                                              horizon=30, max_evals=args.max_evals,
                                              sigma0=0.25, x0=px, restarts=1)
                from_ref_diag.append({"probe": {"mover": mover, "target": hid,
                                                "probe_gap": round(probe_gap, 3),
                                                "full_gap": round(full_info["gap"], 3),
                                                "landed": full_info["landed"]}})
                print(f"    → gap={full_info['gap']:.3f}m  landed={full_info['landed']}", flush=True)
                if full_info["landed"]:
                    current_mover, current_target, first_move_frames = mover, hid, mf
                    break
            if current_target is not None:
                break
        if current_target is None:
            print("No limb has a discoverable hold from the prior ref end state.")
            return
        # Now hand off to discover_climb_adaptive starting from the chained state.
        frames.extend(first_move_frames)
        move_starts.append(1)  # offset by the 1 initial frame
        visited.add(current_target)
        cont_start = first_move_frames[-1]
        # Continue with further adaptive moves, walking the full 4-limb cycle
        # at each step (don't stop because one limb is out of options).
        for step in range(1, args.max_moves):
            found = False
            base_idx = _CYCLE.index(current_mover)
            for advance in range(1, 5):
                next_mover = _CYCLE[(base_idx + advance) % 4]
                max_dist = 0.55 if next_mover in _LEG else 0.80   # hands probe wide:
                # from a post-foot-step crouch the reachable holds sit 0.7-1.0 m
                # from the tip; knee extension (stand-up) closes them, but only
                # if the probe radius lets them be candidates at all
                print(f"  probing {next_mover} candidates (visited={len(visited)})…", flush=True)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    candidates = _reachable_holds(w, cont_start, next_mover,
                                                  max_dist_m=max_dist,
                                                  max_probe_evals=80,
                                                  visited=visited)
                if not candidates:
                    continue
                for probe_gap, hid, px in candidates[:4]:
                    print(f"    trying {next_mover}→{hid} (probe_gap={probe_gap:.3f}m)…", flush=True)
                    move_frames2, full_info = discover_move(
                        w, cont_start, next_mover, hid,
                        horizon=30, max_evals=args.max_evals,
                        sigma0=0.25, x0=px, restarts=1)
                    print(f"    → gap={full_info['gap']:.3f}m  landed={full_info['landed']}", flush=True)
                    if full_info["landed"]:
                        frames.extend(move_frames2)
                        move_starts.append(len(frames) - len(move_frames2))
                        visited.add(hid)
                        cont_start = move_frames2[-1]
                        current_mover = next_mover
                        found = True
                        break
                if found:
                    break
            if not found:
                break
        # Build reference from the continuation frames.
        from sim3d.reference import Reference, holdable_fraction
        ref = Reference(
            qpos=np.array([f["qpos"] for f in frames]),
            qvel=np.array([f["qvel"] for f in frames]),
            eef=np.array([f["eef"] for f in frames]),
            com=np.array([f["com"] for f in frames]),
            grips=np.array([f["grips"] for f in frames], dtype="<U24"),
            wall_gen_seed=prior_ref.wall_gen_seed,
            meta={"method": "cma-es-continue", "continued_from": str(ref_path),
                  "move_starts": move_starts},
        )
        n_moves = len(move_starts)
        with contextlib.redirect_stderr(io.StringIO()):
            frac = holdable_fraction(ref, wall, profile)
        net_z = float(ref.qpos[-1, 2] - ref.qpos[0, 2])
        print(f"continued {n_moves} moves | {len(ref)} frames | holdable {frac*100:.0f}% | "
              f"net pelvis rise {net_z:+.2f} m{'  << NOT A CLIMB' if net_z < 0.10 else ''}")
        if n_moves >= 1:
            ref.save(args.out, wall=wall, env_mode="discover-continue",
                     parent=ref_path)
            out_wall = Path(args.out).with_suffix(".wall.json")
            out_wall.write_text(json.dumps(wd))
            print(f"saved {args.out}  +  {out_wall}")
        return

    # --stances: re-author a dense reference's 4-grip route as a stance skeleton.
    if args.stances:
        from solver.wall import DEFAULT_CELL_SIZE_CM, load_wall
        if not args.from_ref:
            print("--stances requires --from-ref <dense reference .npz>")
            return
        from sim3d.reference import Reference, migrate_reference_spine
        ref_path = Path(args.from_ref)
        prior = migrate_reference_spine(Reference.load(ref_path))
        # Wall: --wall, then <ref>.wall.json sidecar, then ladder default.
        wall_json = (Path(args.wall) if args.wall
                     else ref_path.with_suffix(".wall.json"))
        if not wall_json.exists():
            wall_json = Path("data/examples/ladder-v1.json")
        wd = json.loads(wall_json.read_text())
        profile = ClimberProfile()
        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Respect the wall's own cell_size_cm (fine-grid walls use 5 cm, not
            # the 20 cm default) — forcing DEFAULT rescaled every hold 4× and put
            # foot targets at z≈6 m, so no move could land.
            cell = wd.get("grid", {}).get("cell_size_cm")
            wall = load_wall(wd, cell_size_cm=cell or DEFAULT_CELL_SIZE_CM)
        am.validate_reference(prior, wall, path=ref_path)
        route = stance_route_from_reference(prior)
        mode = "DENSE (discover_move transitions)" if args.dense else "stance-keyframe"
        print(f"{mode} authoring: {len(route)} stances from "
              f"{ref_path.name} (wall {wall_json.name})")
        if args.dense:
            ref, diag = author_dense_from_stances(
                wall, profile, route, wall_gen_seed=prior.wall_gen_seed,
                max_evals=args.max_evals, restarts=args.restarts,
                balance_cap_n=args.balance_assist,
                shift_com_y_target=args.shift_com_y,
                insert_foot_shift=args.insert_foot_shift,
                foot_stance_center_coeff=args.foot_stance_center)
        else:
            with contextlib.redirect_stderr(io.StringIO()):
                ref, diag = author_stance_reference(wall, profile, route,
                                                    wall_gen_seed=prior.wall_gen_seed)
        for d in diag:
            if "stance" not in d:          # dense-mode diag (printed live above)
                if "status" in d:
                    print("  ", d["status"])
                continue
            flag = ""
            if d["grips_held"] < len(d["holds"]):
                flag += "  << LOST GRIP"
            if d["intersections"] > 0:
                flag += "  << SELF-INTERSECT"
            if d["posture"] > 2.0:
                flag += "  << SCRUNCHED"
            if max(d["l_knee"], d["r_knee"]) < 15.0:
                flag += "  << KNEES STRAIGHT"
            print(f"  stance {d['stance']:>2}: grips {d['grips_held']}/"
                  f"{len(d['holds'])}  knee L{d['l_knee']:>5}/R{d['r_knee']:>5}  "
                  f"lean {d['spine_lean']:>5}  pelvisY {d['pelvis_y']}  "
                  f"pelvisZ {d['pelvis_z']}  intersect {d['intersections']}{flag}")
        net_z = float(ref.qpos[-1, 2] - ref.qpos[0, 2])
        if args.dense:
            landed = sum(1 for d in diag if d.get("landed"))
            print(f"authored {landed} transitions | {len(ref)} frames | "
                  f"net pelvis rise {net_z:+.2f} m")
        else:
            worst_posture = max((d["posture"] for d in diag), default=0.0)
            print(f"authored {len(route)} stances | {len(ref)} frames | "
                  f"net pelvis rise {net_z:+.2f} m | worst posture {worst_posture}")
        env_mode = "discover-stances-dense" if args.dense else "discover-stances"
        ref.save(args.out, wall=wall, env_mode=env_mode, parent=ref_path)
        Path(args.out).with_suffix(".wall.json").write_text(json.dumps(wd))
        print(f"saved {args.out}  +  {Path(args.out).with_suffix('.wall.json')}")
        return

    if args.wall:
        from solver.wall import DEFAULT_CELL_SIZE_CM, load_wall
        from sim3d.staged_curriculum import feasible_reach_moves
        wd = json.loads(Path(args.wall).read_text())
        profile = ClimberProfile()
        with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Respect the wall's own cell_size_cm (fall back to the default only
            # if the JSON omits it). Forcing DEFAULT_CELL_SIZE_CM here silently
            # rescaled fine-grid walls (e.g. a 10 cm foot-step test wall) to
            # 20 cm, defeating finer foothold spacing.
            cell = wd.get("grid", {}).get("cell_size_cm")
            wall = load_wall(wd, cell_size_cm=cell or DEFAULT_CELL_SIZE_CM)
            feas = feasible_reach_moves(wall, profile)
    else:
        with contextlib.redirect_stderr(io.StringIO()):
            wd, wall, profile, feas = build_tight_wall(args.seed, args.reach_frac)

    if args.adaptive:
        if not feas:
            print(f"No feasible moves on {'wall ' + args.wall if args.wall else f'seed {args.seed}'}")
            return
        seed_x0 = None
        if args.first_move is not None:
            k0 = int(args.first_move)
            seed_move = next((m for m in feas if m["move_k"] == k0), None)
            if seed_move is None:
                print(f"move {k0} not feasible; feasible: {[m['move_k'] for m in feas]}")
                return
        elif args.auto_first_move:
            print(f"Auto-selecting first move from {len(feas)} feasible candidates…")
            seed_move, seed_info = select_first_move(
                wall, profile, feas, max_gap_m=args.max_gap, restarts=args.restarts)
            seed_x0 = seed_info.get("x_best")
        else:
            seed_move = feas[0]
        print(f"Adaptive CMA-ES discovery: seed {args.seed}, reach_frac {args.reach_frac}, "
              f"first move {seed_move['move_k']} ({seed_move['mover']}→{seed_move['target']}), "
              f"max {args.max_moves} moves, {args.max_evals} evals/move…")
        ref, diag = discover_climb_adaptive(
            wall, profile, seed_move,
            max_moves=args.max_moves, max_evals=args.max_evals,
            max_gap_m=args.max_gap, restarts=args.restarts,
            foot_balance_cap_n=args.foot_balance, seed_x0=seed_x0,
            foot_max_gap_m=args.foot_max_gap, wall_gen_seed=args.seed,
        )
    else:
        want = [int(x) for x in args.moves.split(",")]
        moves = [next((m for m in feas if m["move_k"] == k), None) for k in want]
        if any(m is None for m in moves):
            print(f"moves not feasible on seed {args.seed}: "
                  f"{[k for k, m in zip(want, moves) if m is None]}  "
                  f"(feasible: {[m['move_k'] for m in feas]})")
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
    net_z = float(ref.qpos[-1, 2] - ref.qpos[0, 2])
    print(f"discovered {landed} moves | {len(ref)} frames | holdable {frac*100:.0f}% | "
          f"net pelvis rise {net_z:+.2f} m{'  << NOT A CLIMB' if net_z < 0.10 else ''}")
    if landed >= 1:
        env_mode = "discover-adaptive" if args.adaptive else "discover"
        ref.save(args.out, wall=wall, env_mode=env_mode)
        wall_path = Path(args.out).with_suffix(".wall.json")
        wall_path.write_text(json.dumps(wd))
        print(f"saved {args.out}  +  {wall_path}")


if __name__ == "__main__":
    main()
