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
        # Stop reseeding once a real rise is achieved; below that, keep trying
        # (the caller commits the best partial rise regardless — a small stand-up
        # is strictly better than none). 0.05 is the retry TRIGGER, not a discard.
        if info["com_gain"] >= 0.05 and info["n_anchor"] == n_anchor0:
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


# ══════════════════════════════════════════════════════════════════════════════
# Batch authoring pipeline (Path A) — unattended multi-wall reference authoring.
#
# The one-off tools above (discover_climb_adaptive, the scratch re-author scripts)
# author ONE reference per babysat run. The batch pipeline runs the same proven
# recipe — whole-body foot search, stance-center foot landings, sequential RSI
# chaining, the anti-wiggle guards (held-hold exclusion, per-move gain floor,
# net-rise report) — but adds the machinery an unattended run needs: per-move
# FEASIBILITY GATES with bounded CMA reseeding, a zero-action holdability probe
# on every authored stance (standard practice per the dead-zone note), a per-wall
# timeout, skip-and-continue on failure, and a summary manifest.
#
# Posture/stand-up STANCES are out of scope as authored imitation targets (they
# park at an honest ceiling — see the standup memory notes). The stand-up *move*
# (discover_stand: raise the com over the feet before a hand reach) is kept — it
# is a transition the policy tracks, not a static keyframe target — but it is not
# gated as a "move" (no grip to gate on); it either raises the com or is dropped.
# ══════════════════════════════════════════════════════════════════════════════

# Gate defaults. Hands and feet differ because a foot step on a dense foothold
# ladder is legitimately SHORTER than a hand reach: the 0.08 m "gain floor" in
# the recipe is the HAND anti-wiggle threshold (a hand that gains <8 cm is a
# timid micro-move / re-grip); feet step 4-7 cm and that is real progress, not
# wiggle (the footstep-fine5 design). Net wiggle is caught at the chain level by
# the net-pelvis-rise report, not per foot move.
GATE_HAND_GAP_M = 0.05
GATE_FOOT_GAP_M = 0.06
GATE_HAND_GAIN_M = 0.07   # anti-wiggle floor, NOT physics; lowered 0.08→0.07
                          # 2026-07-12. It is an upward-progress heuristic, and the
                          # wiggle it guards is now independently blocked by three
                          # other gates (held-hold exclusion in _reachable_holds, the
                          # net-rise chain gate, and the four-limb chain gate). An
                          # otherwise-clean s4288 hand reach stalled the whole chain
                          # on a 2 mm miss against 0.08 (gain 0.078); 0.07 keeps the
                          # anti-wiggle intent with margin off that boundary.
GATE_FOOT_GAIN_M = 0.04

# Hand-foot separation (m) above which the move-ordering walk flips to FEET-FIRST
# so a trailing foot is brought up before the feet are stranded (see
# `_next_move_limb_order`). Below it, hands lead as in normal ladder climbing.
# Calibrated 2026-07-12 against the hand-foot lag profile — min(hand tip_z) −
# min(foot tip_z) at each committed move — of the three references known to
# alternate all four limbs: footstep-fine5 brought feet up at lag 0.62-0.68
# (hands led up to 0.65), s4288 at 0.69-0.77 (hands up to 0.74/0.78). 0.68 sits
# just above footstep-fine5's sustained hand-climb and inside s4288's foot-trigger
# band, so the flip fires right where the good walls alternated. This only sets
# PREFERENCE order — `_find_next_gated_move` still falls through to the other
# limb class if the preferred one has no gate-passing move, so the exact value
# degrades gracefully (it changes which limb is tried first, never feasibility).
FOOT_FIRST_LAG_M = 0.68
HOLDABILITY_STEPS = 12


def _stance_holdable(w_probe: Climb3DWorld, qpos: np.ndarray, grips: dict, *,
                     k_steps: int = HOLDABILITY_STEPS,
                     require_grips: int | None = None) -> tuple[bool, int]:
    """Zero-action holdability probe (dead-zone note, standard practice): RSI the
    stance, take zero action for ``k_steps``, and confirm it neither falls nor
    sheds a grip. Returns ``(ok, n_grips_at_end)``. ``w_probe`` is a scratch
    world that MUST be separate from the discovery world (RSI mutates it)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w_probe.rsi(qpos, np.zeros(w_probe.model.nv), grips)
    n0 = sum(1 for l in LIMBS if w_probe.on_hold(l) is not None)
    for _ in range(k_steps):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            w_probe.step(ENV_SUBSTEPS, check_slip=True)
        if float(w_probe.pelvis_pos()[2]) < FALL_Z:
            return False, 0
    n1 = sum(1 for l in LIMBS if w_probe.on_hold(l) is not None)
    need = require_grips if require_grips is not None else n0
    return (n1 >= need), n1


def _gate_move(mover: str, target: str, info: dict, landed_frame: dict,
               launch_tip_z: float, w_probe: Climb3DWorld, *,
               gap_m: float, gain_m: float) -> dict:
    """Evaluate all feasibility gates for one authored move. Returns a report
    dict with a boolean ``passed`` and every measured quantity (for the manifest
    and for ranking reseeds). Gates: (1) the mover GRIPPED the target; (2) grip
    gap ≤ ``gap_m``; (3) tip z-gain ≥ ``gain_m`` (anti-wiggle upward progress);
    (4) the landed 4-grip stance is zero-action holdable."""
    landed = bool(info["landed"])   # mover gripped the target (weld active)
    gap = float(info["gap"])
    tip_z = float(landed_frame["eef"][LIMBS.index(mover)][2])
    gain = tip_z - launch_tip_z
    holdable, n_hold = (False, 0)
    if landed:
        holdable, n_hold = _stance_holdable(
            w_probe, landed_frame["qpos"], _grips_dict(landed_frame["grips"]))
    reasons = []
    if not landed:
        reasons.append(f"no grip (gap {gap:.3f})")
    else:
        if gap > gap_m:
            reasons.append(f"gap {gap:.3f} > {gap_m:.2f}")
        if gain < gain_m:
            reasons.append(f"gain {gain:+.3f} < {gain_m:.2f}")
        if not holdable:
            reasons.append("stance not zero-action holdable")
    passed = landed and gap <= gap_m and gain >= gain_m and holdable
    return {"passed": passed, "landed": landed, "gap": round(gap, 3),
            "gain": round(gain, 3), "holdable": holdable, "n_hold": n_hold,
            "reason": "ok" if passed else "; ".join(reasons)}


def _author_move_gated(
    w: Climb3DWorld, w_probe: Climb3DWorld, start: dict, mover: str, target: str, *,
    launch_tip_z: float, retries: int, horizon: int, max_evals: int,
    sigma0: float, x0: np.ndarray | None = None, deadline: float | None = None,
) -> tuple[list[dict], dict, dict]:
    """Author one move and run it through the gates, RESEEDING CMA up to
    ``retries`` extra times to beat run-to-run variance (the proven pattern from
    the centered re-author: a single CMA run stalls in a loose basin; a reseed
    lands tight — reseeding beats variance). Returns ``(frames, info, report)``
    for the best attempt: a gate-passing attempt if any, else the closest one
    (ranked landed-then-gap, or com_y-centered for feet).

    Feet get the stance-center landing (land centered over the base of support)
    with a capped balance assist. The whole-body search (arms + standing leg
    recruited) is ESCALATED to only on later retries: measured, it lands close
    footholds LOOSER and ~1.5x slower than the narrow leg-swing search, and only
    pays off on the marginal high reaches (the solved wall's RF) — so try narrow
    first, recruit the whole body only if narrow can't close the gap."""
    import time
    is_foot = mover in _LEG
    gap_m = GATE_FOOT_GAP_M if is_foot else GATE_HAND_GAP_M
    gain_m = GATE_FOOT_GAIN_M if is_foot else GATE_HAND_GAIN_M
    best = None    # (rank_key, frames, info, report)
    n_attempts = 0
    for attempt in range(retries + 1):
        # Stop reseeding once the wall-clock budget is spent — the caller stops
        # gracefully with whatever it has (never blows past into the SIGALRM kill).
        if attempt > 0 and deadline is not None and time.time() >= deadline:
            break
        n_attempts += 1
        move_x0 = x0 if attempt == 0 else None
        base = dict(horizon=horizon, max_evals=max_evals, restarts=1, max_gap_m=gap_m)
        if is_foot:
            # Escalate to the whole-body search on the last two retries only.
            whole = attempt >= max(1, retries - 1)
            base.update(whole_body=whole, stance_center_coeff=2.0,
                        stance_center_com_y=0.16, balance_cap_n=250.0,
                        com_drop_max=0.30,
                        max_evals=(int(max_evals * 1.4) if whole else max_evals))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            frames, info = discover_move(w, start, mover, target, sigma0=sigma0,
                                         x0=move_x0, **base)
        report = _gate_move(mover, target, info, frames[-1], launch_tip_z,
                            w_probe, gap_m=gap_m, gain_m=gain_m)
        report["try"] = attempt + 1
        # Rank: a passing attempt beats a non-passing one; among non-passing,
        # prefer landed, then (feet) most-centered / (hands) tightest gap.
        secondary = (info.get("com_y", 9.9) if is_foot else info["gap"])
        key = (0 if report["passed"] else 1,
               0 if report["landed"] else 1, secondary)
        if best is None or key < best[0]:
            best = (key, frames, info, report)
        if report["passed"]:
            break
    _, frames, info, report = best
    report["tries"] = n_attempts     # actual reseeds run (best may be an earlier one)
    return frames, info, report


def _next_move_limb_order(start: dict, after_mover: str, *,
                          foot_first_lag_m: float = FOOT_FIRST_LAG_M) -> list[str]:
    """Preference order for which limb to move next, LOWER limb (smaller tip z)
    first within the leading class, just-moved limb deprioritized to last.

    Which class leads is SEPARATION-GATED (fix 2026-07-12). Normally hands lead
    and a foot comes up only when a hand reach stalls — real ladder climbing.
    But with closely-spaced hand holds every hand step keeps passing its gate, so
    the walk returned a hand every time and the feet were never brought up: v2's
    chains climbed on the arms and stranded the feet (RF never moved), the mirror
    of v1's stranded-hand failure. So once the hands climb more than
    ``foot_first_lag_m`` above the feet (min hand tip_z − min foot tip_z), flip to
    FEET-FIRST for that step to bring the trailing foot up; below it, hands lead.
    Symmetric in spirit: neither class can be starved indefinitely — hands
    outrunning the feet forces a foot, and a fresh foot drops the lag back so the
    hands resume. This only orders the candidates; ``_find_next_gated_move`` still
    tries the other class if the preferred one has no gate-passing move, so a foot
    is never forced onto a hand hold the way the old rigid LH→RH→LF→RF cycle did."""
    tip_z = {l: float(start["eef"][LIMBS.index(l)][2]) for l in LIMBS}
    lag = min(tip_z["LH"], tip_z["RH"]) - min(tip_z["LF"], tip_z["RF"])
    feet_first = lag > foot_first_lag_m
    # Primary sort key prefers the LAGGING class (feet when hands have run ahead,
    # else hands); secondary key brings the lower limb of that class up first.
    order = sorted(LIMBS, key=lambda l: (
        (l not in _LEG) if feet_first else (l in _LEG), tip_z[l]))
    # Move the just-moved limb to the end (prefer alternating).
    return [l for l in order if l != after_mover] + [after_mover]


def _find_next_gated_move(
    w: Climb3DWorld, w_probe: Climb3DWorld, start: dict, after_mover: str,
    visited: set, *, move_retries: int, horizon: int, max_evals: int,
    foot_max_evals: int, sigma0: float, probe_evals: int,
    deadline: float | None = None, max_candidates: int = 3,
) -> dict | None:
    """Try each limb (in ``_next_move_limb_order`` preference — hands before feet,
    lower first) and return the first gate-PASSING move as a ``pending`` dict
    (carrying its authored frames so the caller never re-authors it), or ``None``
    if no limb has one. Candidates come from ``_reachable_holds`` (held-hold
    exclusion + per-limb upward-gain floor — the anti-wiggle guards).

    Two-phase to stay cheap: EXPLORE every candidate with a SINGLE attempt
    (``retries=0``, warm-started from the probe) and return the first that passes;
    this is the common case on a ladder and costs one CMA run per candidate. Only
    if nothing passes on one attempt do we REseed the single most-promising
    candidate (tightest probe gap) up to ``move_retries`` — reserving the
    expensive reseeds for the one move most likely to land. Deadline-checked
    between every authoring so it can never blow past the graceful budget into the
    SIGALRM backstop (the bug that lost 4/5 walls' progress in the first run)."""
    import time
    best_cand = None    # (probe_gap, mover, hid, probe_x, is_foot, launch_z)
    for next_mover in _next_move_limb_order(start, after_mover):
        if deadline is not None and time.time() >= deadline:
            return None
        next_is_foot = next_mover in _LEG
        max_dist = 0.55 if next_is_foot else 0.80
        min_gain = GATE_FOOT_GAIN_M if next_is_foot else GATE_HAND_GAIN_M
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            candidates = _reachable_holds(w, start, next_mover, max_dist_m=max_dist,
                                          max_probe_evals=probe_evals,
                                          visited=visited, min_z_gain_m=min_gain)
        if not candidates:
            continue
        cand_launch_z = float(start["eef"][LIMBS.index(next_mover)][2])
        for probe_gap, hid, probe_x in candidates[:max_candidates]:
            if deadline is not None and time.time() >= deadline:
                return None
            cand_frames, cand_info, cand_report = _author_move_gated(
                w, w_probe, start, next_mover, hid, launch_tip_z=cand_launch_z,
                retries=0, horizon=horizon,
                max_evals=(foot_max_evals if next_is_foot else max_evals),
                sigma0=sigma0 * 0.6, x0=probe_x, deadline=deadline)
            print(f"    probe {next_mover}->{hid}: gap={cand_report['gap']:.3f} "
                  f"gain={cand_report['gain']:+.3f} hold={cand_report['holdable']} "
                  f"pass={cand_report['passed']}", flush=True)
            if cand_report["passed"]:
                return {"mover": next_mover, "target": hid, "x0": None,
                        "frames": cand_frames, "info": cand_info,
                        "report": cand_report, "launch_z": cand_launch_z}
            if best_cand is None or probe_gap < best_cand[0]:
                best_cand = (probe_gap, next_mover, hid, probe_x, next_is_foot,
                             cand_launch_z)
    # Nothing passed on a single attempt: reseed the most-promising candidate.
    if best_cand is not None and move_retries > 0 and (
            deadline is None or time.time() < deadline):
        _pg, mv, hid, px, isf, lz = best_cand
        cand_frames, cand_info, cand_report = _author_move_gated(
            w, w_probe, start, mv, hid, launch_tip_z=lz, retries=move_retries,
            horizon=horizon, max_evals=(foot_max_evals if isf else max_evals),
            sigma0=sigma0 * 0.6, x0=px, deadline=deadline)
        print(f"    reseed best {mv}->{hid}: gap={cand_report['gap']:.3f} "
              f"pass={cand_report['passed']} tries={cand_report['tries']}", flush=True)
        if cand_report["passed"]:
            return {"mover": mv, "target": hid, "x0": None, "frames": cand_frames,
                    "info": cand_info, "report": cand_report, "launch_z": lz}
    return None


def discover_climb_batch(
    wall: Wall, profile: ClimberProfile, seed_move: dict, *,
    max_moves: int = 10, horizon: int = 36, max_evals: int = 320,
    foot_max_evals: int = 460, sigma0: float = 0.45, settle_pre: int = 2,
    wall_gen_seed: int = 7, move_retries: int = 4, probe_evals: int = 80,
    seed_x0: np.ndarray | None = None, deadline: float | None = None,
) -> tuple[Reference, list[dict]]:
    """Batch-oriented adaptive discovery: like ``discover_climb_adaptive`` but
    every authored move passes the feasibility gates with bounded CMA reseeding
    (``move_retries``), the recipe (whole-body + stance-center) is on for feet,
    and each stance is holdability-probed. Returns ``(Reference, per_move_diag)``
    where ``per_move_diag`` is the manifest's per-move detail. Stops cleanly at
    the first move that can't be made to pass its gates within the retry budget
    (a genuine infeasibility) — the partial chain up to there is still returned."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w = Climb3DWorld(wall, profile)
        w.seed_pose(**seed_move["seed_kwargs"])
        w_probe = Climb3DWorld(wall, profile)   # separate world for holdability

    frames: list[dict] = [_snapshot(w) for _ in range(settle_pre)]
    start = _snapshot(w)
    move_starts: list[int] = []
    per_move: list[dict] = []
    visited: set[str] = {g for g in start["grips"] if g}

    # Gate the seed stance itself (a wall whose start won't hang is unusable).
    seed_ok, _ = _stance_holdable(w_probe, start["qpos"], _grips_dict(start["grips"]))
    if not seed_ok:
        per_move.append({"status": "seed stance not holdable — unclimbable start"})
        ref = _frames_to_reference(frames, wall_gen_seed,
                                   {"method": "cma-es-batch", "move_starts": []})
        return ref, per_move

    # Resolve the FIRST move: try the selected seed move; if it fails its gates,
    # fall back to a gain-respecting cycle-walk from the seed stance. The selector
    # (feasible_reach_moves) only vets reach distance, NOT the 0.08 m upward-gain
    # floor, so it can hand back a tight-but-timid micro-move (measured: RH→h_040
    # landed 0.02 m tight but gained only 0.05 m) that the gate rightly rejects —
    # without this fallback the whole chain aborted at step 0.
    seed_mover, seed_target = seed_move["mover"], seed_move["target"]
    seed_launch_z = float(start["eef"][LIMBS.index(seed_mover)][2])
    sframes, sinfo, sreport = _author_move_gated(
        w, w_probe, start, seed_mover, seed_target, launch_tip_z=seed_launch_z,
        retries=move_retries, horizon=horizon,
        max_evals=(foot_max_evals if seed_mover in _LEG else max_evals),
        sigma0=sigma0, x0=seed_x0, deadline=deadline)
    if sreport["passed"]:
        pending = {"mover": seed_mover, "target": seed_target, "x0": None,
                   "frames": sframes, "info": sinfo, "report": sreport,
                   "launch_z": seed_launch_z}
    else:
        print(f"  seed move {seed_mover}->{seed_target} rejected "
              f"({sreport['reason']}); cycle-walking from seed stance", flush=True)
        per_move.append({"seed_move_rejected": f"{seed_mover}->{seed_target}",
                         "reason": sreport["reason"]})
        pending = _find_next_gated_move(
            w, w_probe, start, seed_mover, visited, move_retries=move_retries,
            horizon=horizon, max_evals=max_evals, foot_max_evals=foot_max_evals,
            sigma0=sigma0, probe_evals=probe_evals, deadline=deadline)
    if pending is None:
        per_move.append({"status": "no gate-passing first move from seed stance"})
        return _frames_to_reference(
            frames, wall_gen_seed,
            {"method": "cma-es-batch", "move_starts": []}), per_move

    import time
    for step in range(max_moves):
        # Graceful wall-clock stop BETWEEN moves: return the partial chain (all
        # committed moves are already gated + saved-worthy) instead of being
        # hard-killed mid-move by the SIGALRM backstop, which would lose them.
        if deadline is not None and time.time() >= deadline:
            per_move.append({"status": f"stopped at step {step}: wall-clock budget "
                                       f"reached (partial chain kept)"})
            break
        mover, target = pending["mover"], pending["target"]
        if target not in w._hold_meta_by_id:
            per_move.append({"step": step, "status": f"unknown target {target}"})
            break
        is_foot = mover in _LEG
        # `pending` always carries frames pre-authored (seed resolved above; each
        # subsequent move authored during the prior step's cycle-walk) — never
        # re-authored here.
        mframes, info, report = pending["frames"], pending["info"], pending["report"]
        entry = {"step": step, "mover": mover, "target": target,
                 "gap": report["gap"], "gain": report["gain"],
                 "holdable": report["holdable"], "tries": report["tries"],
                 "passed": report["passed"], "reason": report["reason"]}
        per_move.append(entry)
        print(f"  step {step}: {mover}->{target}  gap={report['gap']:.3f}m  "
              f"gain={report['gain']:+.3f}m  hold={report['holdable']}  "
              f"tries={report['tries']}  passed={report['passed']}"
              f"{'' if report['passed'] else '  << ' + report['reason']}", flush=True)
        if not report["passed"]:
            # Don't include a non-gripping reach in the saved reference (it would
            # train the policy to reach-and-not-grip at the top). Record + stop.
            per_move.append({"status": f"aborted at step {step}: {report['reason']} "
                                       f"on {mover}->{target}"})
            break
        move_starts.append(len(frames))
        frames.extend(mframes)
        visited.add(target)
        start = mframes[-1]

        # Stand up on a freshly placed foot before probing the next reach (the
        # foot only buys the hands height once the body rises over it). Give the
        # stand-up the same bounded-reseed budget the moves get (was a single
        # shot) and COMMIT any positive rise: the old 0.05 discard turned a small
        # win into zero, leaving the body slumped so the feet re-stranded and the
        # next hand reach missed its gate — the s4288 3-move stall (2026-07-12).
        if is_foot:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stand_frames, stand_info = discover_stand(w, start, restarts=3)
            print(f"    stand after {target}: com_gain={stand_info['com_gain']:+.3f}m "
                  f"anchors={stand_info['n_anchor']} fell={stand_info['fell']}", flush=True)
            if stand_info["com_gain"] > 0.01 and not stand_info["fell"]:
                frames.extend(stand_frames)
                start = stand_frames[-1]
                per_move.append({"stand_after": target,
                                 "com_gain": stand_info["com_gain"]})

        # Walk the 4-limb cycle to the next limb with a gate-passing move.
        pending = _find_next_gated_move(
            w, w_probe, start, mover, visited, move_retries=move_retries,
            horizon=horizon, max_evals=max_evals, foot_max_evals=foot_max_evals,
            sigma0=sigma0, probe_evals=probe_evals, deadline=deadline)
        if pending is None:
            per_move.append({"status": f"no gate-passing move from step {step}; "
                                       f"chain complete or stuck"})
            break

    ref = _frames_to_reference(
        frames, wall_gen_seed,
        {"method": "cma-es-batch", "move_starts": move_starts,
         "discovered": [d.get("target") for d in per_move if d.get("passed")]})
    return ref, per_move


def _frames_to_reference(frames: list[dict], wall_gen_seed: int, meta: dict) -> Reference:
    return Reference(
        qpos=np.array([f["qpos"] for f in frames]),
        qvel=np.array([f["qvel"] for f in frames]),
        eef=np.array([f["eef"] for f in frames]),
        com=np.array([f["com"] for f in frames]),
        grips=np.array([f["grips"] for f in frames], dtype="<U24"),
        wall_gen_seed=wall_gen_seed, meta=meta,
    )


# ─── Wall-set generation (the batch's input) ────────────────────────────────

def _overlay_dense_footholds(wd: dict, *, band_offset: int = 2,
                             row_step: int = 1) -> dict:
    """Overlay two foothold columns (cx ± ``band_offset``, one foothold every
    ``row_step`` rows across the climb span) onto a generated wall — the
    footstep-fine5 design that made the SOLVED wall's foot moves tractable: a
    foot always has a target a short reach up, so it closes cheaply and lands
    tight. The generator's own sparse foothold bands leave feet stranded on
    steep walls. Existing (hand/start) cells are never overwritten.

    ``row_step`` controls foothold VERTICAL density. Keep it 1 (every row = 5 cm
    on the 5 cm grid): a 2026-07-12 smoke tried 10 cm spacing to force bigger
    stand-ups, but feet then could not close on the sparse holds (foot probes
    stalled at gap ~0.08 > the 0.06 gate), the body never rose, and the chain
    stalled at 2 hand moves. Dense footholds let a forced foot move actually grip
    tight. The 5 cm-shuffle / feet-lead worry is handled by the lag-gated move
    ordering (``FOOT_FIRST_LAG_M``), not by thinning the foothold ladder."""
    occupied = {(h["grid_x"], h["grid_y"]) for h in wd["holds"]}
    cx = wd["grid"]["cols"] // 2
    hand_rows = [h["grid_y"] for h in wd["holds"]
                 if h["hold_type"] != "foothold"]
    lo = max(1, min(hand_rows) - 6)
    hi = max(hand_rows) - 1
    n = 0
    for gy in range(lo, hi + 1, row_step):
        for gx in (cx - band_offset, cx + band_offset):
            if (gx, gy) in occupied or not (0 <= gx < wd["grid"]["cols"]):
                continue
            n += 1
            wd["holds"].append({
                "hold_id": f"fd_{n:03d}", "grid_x": gx, "grid_y": gy,
                "hold_type": "foothold", "orientation_deg": 0.0, "size": "medium",
                "color": "#a855f7", "is_start": False, "is_finish": False})
            occupied.add((gx, gy))
    return wd


def build_batch_walls(
    out_dir, n: int, *, base_seed: int = 300, cols: int = 21, rows: int = 52,
    cell_size_cm: float = 5.0, reach_frac: float = 0.5, min_step_dy: int = 3,
    max_step_dy: int = 3, foot_lag: int = 3, n_scatter: int = 6,
    foot_row_step: int = 1, max_seed_scan: int = 200,
) -> list:
    """Generate ``n`` climb walls comparable to the solved 4-move wall: a jug
    ladder on a 5 cm grid with central foothold columns, produced by
    ``solver.generate`` and A*-verified. Each wall's bottom stance is checked
    hangable before it is accepted (an over-braced start is unclimbable for this
    body). Returns the list of written Paths.

    The hand route comes from ``solver.generate`` (monkeypatched difficulty
    params for the ladder spacing, the same pattern ``build_tight_wall`` uses);
    the footholds are overlaid on top.

    Geometry tuned 2026-07-12 against the diagnosed feet-lead root cause. Hand
    steps are UNIFORM 15 cm (``max_step_dy`` 4→3): v1's 15-20 cm steps let the
    20 cm reaches land just outside the 0.05 gap / 0.08 gain hand gate from a low
    stance, so the search always took the cheaper foot move. 15 cm matches the
    upper hand-step of the two walls known to produce alternating climbs
    (footstep-fine5's clean 20 cm ladder solves, and s4288's scattered 5-15 cm
    hands alternate cleanly), and keeps a comfortable margin over the 8 cm gain
    gate (a fully-closed 15 cm reach gains 15 cm; even a 5 cm-short one gains
    10 cm).

    ``foot_row_step`` stays 1 (footholds every row = 5 cm). A one-wall smoke on
    2026-07-12 tried sparser 10 cm footholds to force bigger stand-ups, but the
    feet then could not CLOSE on the sparse holds (every foot probe stalled at
    gap ~0.08 > the 0.06 gate) → the body never rose → the chain stalled at 2
    hand moves. Dense footholds are what let a forced foot move actually grip
    (s4288 alternates all four WITH 5 cm footholds). Feet-lead is prevented by
    the lag-gated move ORDERING (``FOOT_FIRST_LAG_M`` / ``_next_move_limb_order``),
    not by starving the feet of holds — a single forced foot move drops the lag
    back below threshold, so hands resume and no double-shuffle occurs."""
    import json
    from pathlib import Path
    import solver.generate as gen
    from solver.generate import GeneratorConfig, generate_wall
    from solver.wall import load_wall
    from sim3d.staged_curriculum import feasible_reach_moves

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    profile = ClimberProfile()
    orig = gen._difficulty_params
    written: list = []
    try:
        gen._difficulty_params = lambda d, _o=orig: {
            **_o(0.0), "min_step_dy": min_step_dy, "max_step_dy": max_step_dy,
            "reach_frac": reach_frac, "foot_lag": foot_lag, "n_scatter": n_scatter}
        for off in range(max_seed_scan):
            if len(written) >= n:
                break
            # Stride the seeds by 997 (as generate_batch does): generate_wall
            # retries base_seed+attempt on A* failure, so consecutive base seeds
            # hit OVERLAPPING retry ranges and collapse to the SAME wall. A wide
            # stride keeps every wall structurally distinct.
            seed = base_seed + off * 997
            gc = GeneratorConfig(cols=cols, rows=rows, cell_size_cm=cell_size_cm,
                                 difficulty=0.0, seed=seed)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wd = generate_wall(gc, wall_id=f"batch-s{seed}")
            if wd is None:
                continue
            wd = _overlay_dense_footholds(wd, row_step=foot_row_step)
            with warnings.catch_warnings(), contextlib_redirect_stderr():
                warnings.simplefilter("ignore")
                wall = load_wall(wd, cell_size_cm=cell_size_cm)
                feas = feasible_reach_moves(wall, profile)
                if len(feas) < 3:
                    continue
                # Bottom-stance hangability: the first feasible move's stance.
                w_probe = Climb3DWorld(wall, profile)
                w_probe.seed_pose(**feas[0]["seed_kwargs"])
                start_qpos = w_probe.data.qpos.copy()
                start_grips = {l: (w_probe.on_hold(l) or None) for l in LIMBS}
            ok, _ = _stance_holdable(Climb3DWorld(wall, profile), start_qpos, start_grips)
            if not ok:
                continue
            path = out_dir / f"{wd['wall_id']}.json"
            path.write_text(json.dumps(wd))
            written.append(path)
            print(f"  wrote {path.name}: {len(wd['holds'])} holds, "
                  f"{len(feas)} feasible first-moves (seed {seed})", flush=True)
    finally:
        gen._difficulty_params = orig
    return written


def contextlib_redirect_stderr():
    import contextlib
    import io
    return contextlib.redirect_stderr(io.StringIO())


# ─── Batch driver: per-wall timeout, skip-on-failure, manifest ──────────────

class _WallTimeout(Exception):
    pass


class _timeout:
    """SIGALRM-based per-wall wall-clock cap. MuJoCo/CMA return to Python between
    physics steps, so the alarm fires promptly. ``seconds<=0`` disables it."""
    def __init__(self, seconds: float):
        self.seconds = int(seconds)

    def __enter__(self):
        import signal
        if self.seconds > 0:
            self._old = signal.signal(signal.SIGALRM, self._fire)
            signal.alarm(self.seconds)
        return self

    def __exit__(self, *exc):
        import signal
        if self.seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, self._old)
        return False

    def _fire(self, *a):
        raise _WallTimeout(f"per-wall timeout ({self.seconds}s) exceeded")


# Fields carried per authoring attempt (constant top-level wall fields are added
# separately). Kept as a named tuple of keys so incumbent-from-manifest and a
# fresh attempt build the same shape for _chain_quality_key ranking.
_ATTEMPT_FIELDS = ("n_moves", "moves", "net_pelvis_rise", "holdable_fraction",
                   "abort_reason", "limbs_moved", "all_gates_pass", "success")


def _chain_quality_key(fields: dict) -> tuple:
    """Rank chain-authoring attempts and harvest the best per wall. A gate-passing
    chain beats any non-passing one; then more committed moves; then higher net
    pelvis rise. Used by BOTH the chain-level reseed loop (keep the best of N
    CMA seeds) and best-per-wall harvesting (a worse re-authoring must never
    overwrite a better — esp. gate-passing — prior reference)."""
    return (1 if fields.get("all_gates_pass") else 0,
            int(fields.get("n_moves") or 0),
            float(fields.get("net_pelvis_rise") or 0.0))


def _author_one_wall(wall_path, out_dir, *, max_moves: int, move_retries: int,
                     max_evals: int, foot_max_evals: int,
                     wall_deadline: float | None = None,
                     attempt_timeout: float = 0.0, chain_reseeds: int = 1,
                     prior: dict | None = None) -> dict:
    """Author a reference for one wall. Returns a manifest row. Never raises for
    an authoring failure — records it in the row instead (except a timeout, which
    the caller catches so it can annotate the row).

    ``wall_deadline`` is a wall-clock time the wall stops at (total budget across
    all reseeds). ``attempt_timeout`` (>0) caps each individual chain attempt so
    that when a wall's budget is larger than one chain, the wall is re-authored
    from scratch with a fresh CMA seed (``chain_reseeds`` attempts max), keeping
    the best by ``_chain_quality_key``. With ``attempt_timeout=0`` and
    ``chain_reseeds=1`` (the defaults) this is exactly the old single-attempt
    behavior.

    Best-per-wall harvesting: a ``prior`` manifest row whose ref still exists on
    disk seeds the incumbent, so a worse re-authoring can never overwrite a
    better (or gate-passing) reference — the chain-level CMA variance that flipped
    a passing s4288 smoke to a 2-limb failing run costs nothing now."""
    import contextlib
    import io
    import json
    import time
    from pathlib import Path
    from sim3d.reference import holdable_fraction
    from solver.wall import DEFAULT_CELL_SIZE_CM, load_wall
    from sim3d.staged_curriculum import feasible_reach_moves

    wall_path = Path(wall_path)
    wd = json.loads(wall_path.read_text())
    cell = wd.get("grid", {}).get("cell_size_cm") or DEFAULT_CELL_SIZE_CM
    profile = ClimberProfile()
    with contextlib.redirect_stderr(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wall = load_wall(wd, cell_size_cm=cell)
        feas = feasible_reach_moves(wall, profile)
    base_row: dict = {"wall": str(wall_path), "wall_id": wd.get("wall_id"),
                      "cell_size_cm": cell, "n_holds": len(wd["holds"])}
    out_path = Path(out_dir) / f"ref_{wd['wall_id']}.npz"
    if not feas:
        base_row.update(success=False, reason="no feasible first move", moves=[],
                        all_gates_pass=False)
        return base_row
    with contextlib.redirect_stderr(io.StringIO()):
        # Cheap ranker (few candidates, light evals): only picks WHICH first move
        # to commit; discover_climb_batch re-authors it at full budget with retries.
        seed_move, seed_info = select_first_move(
            wall, profile, feas, max_gap_m=GATE_HAND_GAP_M, restarts=1,
            max_candidates=4, probe_evals=120)

    def _one_attempt(deadline: float | None) -> tuple[dict, "Reference | None"]:
        """Author one full chain with a fresh CMA seed (cma defaults to a
        time-based seed, so back-to-back attempts search independent basins) and
        distill it into the manifest fields + (ref if it committed ≥1 move)."""
        ref, per_move = discover_climb_batch(
            wall, profile, seed_move, max_moves=max_moves,
            move_retries=move_retries, max_evals=max_evals,
            foot_max_evals=foot_max_evals, seed_x0=seed_info.get("x_best"),
            wall_gen_seed=0, deadline=deadline)
        passed = [d for d in per_move if d.get("passed")]
        net_z = float(ref.qpos[-1, 2] - ref.qpos[0, 2]) if len(ref) else 0.0
        with contextlib.redirect_stderr(io.StringIO()):
            frac = holdable_fraction(ref, wall, profile) if len(ref) > 1 else 0.0
        abort = next((d["status"] for d in per_move
                      if isinstance(d, dict) and "status" in d), None)
        # Every move COMMITTED to the reference passed its gates by construction
        # (a failed move ends the chain and is never appended). CHAIN-LEVEL
        # train-worthiness gate (2026-07-12): every limb must move ≥1× AND both
        # hands must move — the separator between s4288's real climb and the
        # stranded-hand / stranded-foot rejects, on top of the per-move gates and
        # the net-rise (>= -0.01) anti-wiggle floor.
        movers = {d.get("mover") for d in passed}
        fields = dict(
            n_moves=len(passed),
            moves=[{k: d[k] for k in ("step", "mover", "target", "gap", "gain",
                                       "holdable", "tries", "passed", "reason")
                    if k in d} for d in per_move if "mover" in d],
            net_pelvis_rise=round(net_z, 3),
            holdable_fraction=round(frac, 3),
            abort_reason=abort,
            limbs_moved=sorted(m for m in movers if m),
            all_gates_pass=(len(passed) >= 2 and net_z >= -0.01
                            and {"LH", "RH", "LF", "RF"} <= movers
                            and {"LH", "RH"} <= movers),
            success=len(passed) >= 1,
        )
        return fields, (ref if fields["success"] else None)

    # Best-per-wall harvesting: seed the incumbent from a prior ref still on disk
    # (this wall's canonical out_path). best_ref=None means "already on disk, keep
    # the file"; a fresh attempt only overwrites it if it strictly wins.
    best_fields: dict | None = None
    best_ref = None
    best_from_disk = False
    if prior and prior.get("ref") and Path(prior["ref"]) == out_path \
            and out_path.exists():
        best_fields = {k: prior.get(k) for k in _ATTEMPT_FIELDS if k in prior}
        best_from_disk = True
        if best_fields.get("all_gates_pass"):
            # Nothing to beat a gate-passer here (resume skips these before we're
            # called; this guards --no-resume / a partial-manifest edge).
            return {**base_row, **best_fields, "ref": str(out_path),
                    "n_chain_attempts": 0,
                    "harvested": "kept prior all-gates-pass ref"}

    n_attempts = 0
    while True:
        n_attempts += 1
        if attempt_timeout and attempt_timeout > 0:
            att_deadline = time.time() + attempt_timeout
            if wall_deadline is not None:
                att_deadline = min(att_deadline, wall_deadline)
        else:
            att_deadline = wall_deadline
        fields, ref = _one_attempt(att_deadline)
        if best_fields is None or \
                _chain_quality_key(fields) > _chain_quality_key(best_fields):
            best_fields, best_ref, best_from_disk = fields, ref, False
        if best_fields.get("all_gates_pass"):
            break                              # four-limb gate met — done
        if n_attempts >= chain_reseeds:
            break                              # reseed budget spent
        if wall_deadline is not None and time.time() >= wall_deadline:
            break                              # per-wall wall-clock budget spent

    row = {**base_row, **best_fields, "n_chain_attempts": n_attempts}
    if best_fields.get("success"):
        # Only (re)write the ref when a fresh attempt actually won; if the on-disk
        # incumbent is still best, leave its file untouched (never overwrite a
        # better ref with a worse re-authoring).
        if best_ref is not None and not best_from_disk:
            best_ref.save(out_path, wall=wall, env_mode="discover-batch")
            out_path.with_suffix(".wall.json").write_text(json.dumps(wd))
        row["ref"] = str(out_path)
    return row


def run_batch(wall_dir, out_dir=None, *, per_wall_timeout: float = 1800.0,
              max_moves: int = 10, move_retries: int = 4, max_evals: int = 320,
              foot_max_evals: int = 460, manifest_name: str = "batch_manifest.json",
              resume: bool = True, chain_attempt_timeout: float = 0.0,
              chain_reseeds: int = 1) -> dict:
    """Author a reference for every wall JSON in ``wall_dir`` and write a manifest.
    Each wall runs under a wall-clock ``per_wall_timeout``; a timeout or an
    unexpected exception is caught and recorded so one bad wall never kills the
    batch.

    ``resume`` (default on) makes the batch RESTARTABLE: a wall whose prior
    manifest row already passed all gates AND whose saved reference still exists
    is skipped and its row carried forward. So re-running after a crash/interrupt
    only re-attempts the walls that haven't succeeded yet — an overnight run that
    dies at wall 3 costs nothing on restart.

    ``chain_attempt_timeout`` (>0) + ``chain_reseeds`` enable CHAIN-LEVEL reseeds:
    each wall re-authors its whole chain from a fresh CMA seed (up to
    ``chain_reseeds`` attempts, each capped at ``chain_attempt_timeout``) within
    the ``per_wall_timeout`` total budget, keeping the best. This beats the
    run-to-run CMA variance that flipped a passing s4288 smoke into a 2-limb
    failing run — one attempt going down a bad basin no longer wastes the wall.
    Defaults (0 / 1) preserve the single-attempt behavior. The prior manifest is
    always consulted for best-per-wall harvesting (a worse re-authoring never
    overwrites a better prior ref), independently of ``resume``'s skip."""
    import json
    import time
    import traceback
    from pathlib import Path

    wall_dir = Path(wall_dir)
    out_dir = Path(out_dir) if out_dir else wall_dir / "refs"
    out_dir.mkdir(parents=True, exist_ok=True)
    wall_paths = sorted(p for p in wall_dir.glob("*.json")
                        if not p.name.endswith(".wall.json"))
    # Load a prior manifest (if any). Used BOTH for --resume skips and — always —
    # for best-per-wall harvesting inside _author_one_wall.
    prior_by_wall: dict = {}
    prior_path = out_dir / manifest_name
    if prior_path.exists():
        try:
            for r in json.loads(prior_path.read_text()).get("walls", []):
                prior_by_wall[r.get("wall_id")] = r
        except Exception:  # noqa: BLE001 — a corrupt prior manifest just disables it
            prior_by_wall = {}
    manifest = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "wall_dir": str(wall_dir), "out_dir": str(out_dir),
                "config": {"per_wall_timeout": per_wall_timeout,
                           "max_moves": max_moves, "move_retries": move_retries,
                           "max_evals": max_evals, "foot_max_evals": foot_max_evals,
                           "resume": resume,
                           "chain_attempt_timeout": chain_attempt_timeout,
                           "chain_reseeds": chain_reseeds},
                "walls": []}
    for i, wp in enumerate(wall_paths):
        print(f"\n=== wall {i+1}/{len(wall_paths)}: {wp.name} ===", flush=True)
        # Resume: skip a wall already authored to all-gates-pass with its ref on disk.
        prior = prior_by_wall.get(wp.stem)
        if (resume and prior and prior.get("all_gates_pass")
                and prior.get("ref") and Path(prior["ref"]).exists()):
            prior = {**prior, "skipped": "reused prior all-gates-pass reference"}
            manifest["walls"].append(prior)
            print(f"  → SKIP (already passed all gates: {prior.get('n_moves')} moves, "
                  f"net_rise={prior.get('net_pelvis_rise')})", flush=True)
            (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2, default=str))
            continue
        t0 = time.time()
        # Two-tier timeout: the chain stops GRACEFULLY at ``per_wall_timeout``
        # (checked between moves, returning its partial gated reference), and
        # SIGALRM is a generous backstop (+600 s, enough to let one in-flight foot
        # move with its reseeds finish) that only fires if a single CMA move truly
        # hangs — the common "chain is just long" case keeps all its progress.
        deadline = t0 + per_wall_timeout
        try:
            with _timeout(per_wall_timeout + 600):
                row = _author_one_wall(wp, out_dir, max_moves=max_moves,
                                       move_retries=move_retries, max_evals=max_evals,
                                       foot_max_evals=foot_max_evals,
                                       wall_deadline=deadline,
                                       attempt_timeout=chain_attempt_timeout,
                                       chain_reseeds=chain_reseeds, prior=prior)
        except _WallTimeout as e:
            row = {"wall": str(wp), "wall_id": wp.stem, "success": False,
                   "all_gates_pass": False, "reason": str(e), "moves": []}
        except Exception as e:  # noqa: BLE001 — one bad wall must not kill the batch
            row = {"wall": str(wp), "wall_id": wp.stem, "success": False,
                   "all_gates_pass": False,
                   "reason": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc(), "moves": []}
        row["seconds"] = round(time.time() - t0, 1)
        manifest["walls"].append(row)
        print(f"  → {'OK' if row.get('success') else 'FAIL'} "
              f"({row.get('n_moves', 0)} moves, gates_pass={row.get('all_gates_pass')}, "
              f"net_rise={row.get('net_pelvis_rise')}, "
              f"attempts={row.get('n_chain_attempts', 1)}, {row['seconds']}s)", flush=True)
        # Persist the manifest after every wall so a crash keeps partial results.
        (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2, default=str))

    n_ok = sum(1 for r in manifest["walls"] if r.get("success"))
    n_gates = sum(1 for r in manifest["walls"] if r.get("all_gates_pass"))
    manifest["summary"] = {"n_walls": len(wall_paths), "n_success": n_ok,
                           "n_all_gates_pass": n_gates}
    (out_dir / manifest_name).write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\n=== BATCH DONE: {n_gates}/{len(wall_paths)} refs pass ALL gates "
          f"({n_ok}/{len(wall_paths)} authored ≥1 move) ===")
    print(f"manifest: {out_dir / manifest_name}")
    return manifest


def main() -> None:
    import argparse
    import contextlib
    import io
    import json
    from pathlib import Path
    from sim3d.reference import holdable_fraction

    ap = argparse.ArgumentParser(description=__doc__)
    # ── Batch pipeline (Path A) ──────────────────────────────────────────────
    ap.add_argument("--batch", type=str, default=None, metavar="WALL_DIR",
                    help="UNATTENDED batch mode: author one gated multi-move "
                         "reference per wall JSON in WALL_DIR and write a manifest. "
                         "Per-wall timeout + skip-on-failure. See --batch-out etc.")
    ap.add_argument("--batch-out", type=str, default=None,
                    help="--batch: output dir for refs + manifest (default: "
                         "<WALL_DIR>/refs)")
    ap.add_argument("--per-wall-timeout", type=float, default=1800.0,
                    help="--batch: wall-clock cap per wall in seconds (default 1800)")
    ap.add_argument("--move-retries", type=int, default=4,
                    help="--batch: CMA reseeds per move before giving up (default 4)")
    ap.add_argument("--foot-max-evals", type=int, default=460,
                    help="--batch: CMA evals per FOOT move (hands use --max-evals)")
    ap.add_argument("--no-resume", action="store_true",
                    help="--batch: re-author every wall even if a prior manifest "
                         "row already passed all gates (default: skip those)")
    ap.add_argument("--chain-attempt-timeout", type=float, default=0.0,
                    help="--batch: cap each single chain-authoring attempt (s). "
                         ">0 enables CHAIN-LEVEL reseeds — re-author the whole "
                         "chain with a fresh CMA seed within --per-wall-timeout, "
                         "keeping the best (beats run-to-run CMA variance). "
                         "0 (default) = one attempt fills the wall budget.")
    ap.add_argument("--chain-reseeds", type=int, default=1,
                    help="--batch: max chain-authoring attempts per wall (default "
                         "1). Reseeds stop early on a four-limb gate pass or when "
                         "--per-wall-timeout is spent.")
    ap.add_argument("--gen-walls", type=str, default=None, metavar="OUT_DIR",
                    help="Generate a batch wall set (jug ladder + dense footholds, "
                         "5cm grid, A*-verified, hangable start) into OUT_DIR and "
                         "exit. Use --n-walls and --gen-seed.")
    ap.add_argument("--n-walls", type=int, default=5, help="--gen-walls: how many")
    ap.add_argument("--gen-seed", type=int, default=300, help="--gen-walls: base seed")
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

    # --gen-walls: build the batch's input wall set and exit.
    if args.gen_walls:
        print(f"Generating {args.n_walls} batch walls into {args.gen_walls} "
              f"(base seed {args.gen_seed})…")
        written = build_batch_walls(args.gen_walls, args.n_walls,
                                    base_seed=args.gen_seed)
        print(f"wrote {len(written)}/{args.n_walls} walls to {args.gen_walls}")
        return

    # --batch: unattended multi-wall authoring with a manifest.
    if args.batch:
        # --max-evals defaults to 160 (fixed-move mode); batch hand moves need
        # ~320 to land tight, so treat the untouched default as "use the batch
        # default" while still honoring an explicit override.
        hand_evals = 320 if args.max_evals == 160 else args.max_evals
        run_batch(args.batch, args.batch_out,
                  per_wall_timeout=args.per_wall_timeout,
                  max_moves=args.max_moves, move_retries=args.move_retries,
                  max_evals=hand_evals, foot_max_evals=args.foot_max_evals,
                  resume=not args.no_resume,
                  chain_attempt_timeout=args.chain_attempt_timeout,
                  chain_reseeds=args.chain_reseeds)
        return

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
