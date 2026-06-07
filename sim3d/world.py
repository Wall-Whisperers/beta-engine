"""Climb3DWorld — top-level handle for the 3D simulator.

Wraps an `mujoco.MjModel` + `mujoco.MjData` and exposes the same kind
of API as `physics.world.ClimbWorld` (the 2D pymunk version):

    world = Climb3DWorld(wall, profile)
    world.seed_pose(lh="h_003", rh="h_004", lf="h_001", rf="h_002")
    for _ in range(60): world.step()
    world.move_limb("RH", "h_008")

The 3D and 2D worlds intentionally share method names so the upstream
solver / RL code can target either via duck-typing.

Hold attachment is implemented with a per-limb mocap body + weld
equality constraint:

    1. The mocap body is parented to the world (kinematic — it doesn't
       move under physics; we move it by writing data.mocap_pos[i]).
    2. The weld equality is created at compile time, initially inactive.
    3. To attach: position the mocap at the hold, set eq_active[i]=1
       and update eq_data so the weld's relative pose snaps the limb's
       fixed tip-anchor body to the mocap.
    4. To release: set eq_active[i]=0.

Why mocap+weld rather than connect with sites? `connect` needs two
bodies and only constrains position; sites can't be the second arg of a
weld. Mocap+weld lets us teleport the target without recompiling the
model — exactly what RL needs for fast resets.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Optional

import mujoco
import numpy as np

from sim3d import config as cfg
from sim3d.body import (
    HAND_LIMBS,
    LIMB_EQUALITY,
    LIMB_MOCAP_BODY,
    LIMB_TIP_SITE,
    LIMBS,
    ClimberProfile,
    Limb,
)
from sim3d.builder import build_mjcf_xml
from solver.wall import Wall


@dataclass
class HoldAttachment:
    """Live state for a single limb→hold attachment."""

    hold_id: str
    world_pos: tuple[float, float, float]
    max_force_n: float


@dataclass
class SlipEvent:
    """Recorded when a limb's grip is exceeded and the weld releases."""

    t: float
    limb: str
    hold_id: str
    force_n: float
    capacity_n: float


@dataclass
class ReachState:
    """Live state for a limb mid-flight between holds.

    Setting a ReachState on a limb causes `step()` to apply a
    Cartesian PD force on the limb's tip site each sub-step, pulling
    it toward `target_world_pos`. When the tip is within
    `REACH_ATTACH_RADIUS` of the target, or `t_remaining` runs out,
    the controller activates the weld and clears itself.
    """

    target_hold_id: str
    target_world_pos: np.ndarray
    t_remaining: float
    kp: float
    kd: float
    dyno: bool = False
    closest_dist: float = 1e9
    closest_t: float = 0.0


class Climb3DWorld:
    """Owns the MuJoCo model + data + per-limb attachment state.

    Threading note: MuJoCo's MjData is not thread-safe. The native
    viewer and the web pose streamer both read from `self.data`; the
    web streamer is read-only and does shallow numpy copies before
    handing data off, which is fine for visualisation. Don't call
    `step()` from two threads.
    """

    def __init__(
        self,
        wall: Wall,
        profile: Optional[ClimberProfile] = None,
        *,
        include_kickboard: bool = False,
    ) -> None:
        self.wall = wall
        self.profile = profile or ClimberProfile()
        self.include_kickboard = include_kickboard

        xml, hold_meta = build_mjcf_xml(
            wall, self.profile, include_kickboard=include_kickboard,
        )
        self._xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # ── Hold lookup tables ────────────────────────────────────
        self._hold_meta_by_id: dict[str, dict] = {
            h["hold_id"]: h for h in hold_meta
        }

        # ── ID lookups (mujoco prefers ids, not names, in hot loops)
        self._mocap_idx: dict[Limb, int] = {}
        self._eq_idx: dict[Limb, int] = {}
        self._tip_site_idx: dict[Limb, int] = {}
        self._limb_body_idx: dict[Limb, int] = {}
        for limb in LIMBS:
            mocap_body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY,
                LIMB_MOCAP_BODY[limb],
            )
            self._mocap_idx[limb] = self.model.body_mocapid[mocap_body_id]
            self._eq_idx[limb] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_EQUALITY,
                LIMB_EQUALITY[limb],
            )
            self._tip_site_idx[limb] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE,
                LIMB_TIP_SITE[limb],
            )
            limb_body_name = {
                "LH": "l_hand", "RH": "r_hand",
                "LF": "l_foot", "RF": "r_foot",
            }[limb]
            self._limb_body_idx[limb] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, limb_body_name,
            )

        self._on_hold: dict[Limb, Optional[HoldAttachment]] = {
            l: None for l in LIMBS
        }

        # Cache the qpos slice that each non-free joint occupies. Used
        # by the settle phase below to copy "current pose" → "actuator
        # targets" without writing a long lookup loop.
        self._jnt_qposadr: dict[str, int] = {}
        self._actuator_jnt_qposadr: list[int] = []
        for i in range(self.model.nu):
            jnt_id = self.model.actuator_trnid[i, 0]
            self._actuator_jnt_qposadr.append(int(self.model.jnt_qposadr[jnt_id]))

        # Map limb → list of actuator indices for that limb's chain.
        # Used by the reach controller to relax those actuators while
        # the limb is mid-flight (otherwise the actuator servos fight
        # the Cartesian impedance pulling the limb to the target).
        self._limb_actuator_ids: dict[Limb, list[int]] = {l: [] for l in LIMBS}
        limb_joint_prefix = {
            "LH": ("l_shoulder", "l_elbow", "l_wrist"),
            "RH": ("r_shoulder", "r_elbow", "r_wrist"),
            "LF": ("l_hip", "l_knee", "l_ankle"),
            "RF": ("r_hip", "r_knee", "r_ankle"),
        }
        for i in range(self.model.nu):
            jnt_id = int(self.model.actuator_trnid[i, 0])
            jname = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
            if jname is None:
                continue
            for limb, prefixes in limb_joint_prefix.items():
                if any(jname.startswith(p) for p in prefixes):
                    self._limb_actuator_ids[limb].append(i)
                    break

        # Geom groups used by RL reward shaping. MuJoCo's contact solver
        # tells us when a limb geom penetrates the torso/pelvis; we turn
        # those contacts into a soft penalty instead of pretending that
        # impossible self-intersecting poses are valid climbing beta.
        self._body_intersection_torso_geom_ids = self._geom_ids(
            "g_pelvis", "g_chest",
        )
        self._body_intersection_limb_geom_ids = self._geom_ids(
            "g_l_upperarm", "g_l_forearm", "g_l_hand",
            "g_r_upperarm", "g_r_forearm", "g_r_hand",
            "g_l_thigh", "g_l_shin", "g_l_foot",
            "g_r_thigh", "g_r_shin", "g_r_foot",
        )

        # Track slips for diagnostics / RL reward shaping.
        self.slip_events: list[SlipEvent] = []
        self._max_slip_log = 200

        # Per-limb settled force / cap from the last seed_pose() call.
        # Populated by the post-settle guard so callers can detect an
        # over-braced seed (a limb the settle leaves above its slip cap).
        self.seed_overload: dict[Limb, dict] = {}
        self._seed_overload_warned = False

        # Continuous-reach state per limb (None = not reaching).
        self._reaching: dict[Limb, Optional[ReachState]] = {l: None for l in LIMBS}
        # Pelvis position to hold during a reach (balance assist). Set when a
        # reach starts; the balance PD pins the pelvis here so the reach
        # controller's reaction can't tip the body off its support.
        self._balance_target: Optional[np.ndarray] = None

        # Tip-anchor bodies are fixed children at the contact sites, so hold
        # constraints can pin the actual hand/toe contact point rather than
        # approximating it from the wrist/ankle body origin.

        # First mj_forward so kinematics are populated for any caller
        # that asks for site positions before stepping.
        mujoco.mj_forward(self.model, self.data)
        self._sync_actuator_targets_to_pose()

    def _geom_ids(self, *names: str) -> set[int]:
        """Resolve geom names to ids, ignoring missing names so old MJCFs
        or experiments can still run."""
        ids: set[int] = set()
        for name in names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                ids.add(int(gid))
        return ids

    def body_intersection_contacts(self) -> list[dict]:
        """Return current limb-vs-torso/pelvis penetration contacts.

        This is intentionally diagnostic/reward-shaping only. We do not
        terminate episodes or modify physics here; MuJoCo has already
        advanced the state, and the RL environment can decide how much to
        penalize impossible self-intersections.
        """
        contacts: list[dict] = []
        torso = self._body_intersection_torso_geom_ids
        limbs = self._body_intersection_limb_geom_ids
        for i in range(int(self.data.ncon)):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if c.dist > 0:
                continue
            if g1 in torso and g2 in limbs:
                torso_gid, limb_gid = g1, g2
            elif g2 in torso and g1 in limbs:
                torso_gid, limb_gid = g2, g1
            else:
                continue
            torso_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, torso_gid,
            )
            limb_name = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, limb_gid,
            )
            contacts.append({
                "torso_geom": torso_name or str(torso_gid),
                "limb_geom": limb_name or str(limb_gid),
                "penetration_m": float(max(0.0, -c.dist)),
            })
        return contacts

    def body_intersection_count(self) -> int:
        """Number of current limb-vs-body intersection contacts."""
        return len(self.body_intersection_contacts())

    # ─── Pose seeding ─────────────────────────────────────────────────
    def seed_pose(
        self,
        *,
        lh: Optional[str] = None,
        rh: Optional[str] = None,
        lf: Optional[str] = None,
        rf: Optional[str] = None,
    ) -> None:
        """Snap the climber so each named limb is on its hold and turn
        on the corresponding equality constraints. Pass None for a limb
        in flight.

        How seeding works (this matters — the previous implementation
        let actuators servo toward zero while welds pulled the body
        onto the holds, which causes the body to fight itself):

            1. Reset qpos / qvel.
            2. Place pelvis at a guess that minimises constraint stress.
               If hands are above feet by more than 30 cm we assume an
               extended-body climb pose; otherwise a crouched pose.
            3. Activate welds.
            4. *Disable* the actuators (zero ctrl, also zero kp via
               temp scaling) and run physics for a half-second so the
               welds yank limbs into a self-consistent rest pose.
            5. Snapshot the joint angles that resulted, store them as
               actuator targets, re-enable the actuators.

        After this, the climber holds whatever pose the welds settled
        into, with no internal stress between the welds and the
        actuators.
        """
        targets: dict[Limb, Optional[str]] = {
            "LH": lh, "RH": rh, "LF": lf, "RF": rf,
        }
        positions = {
            l: np.array(self._hold_meta_by_id[hid]["world_pos"])
            for l, hid in targets.items() if hid is not None
        }
        foot_pts = [positions[l] for l in ("LF", "RF") if l in positions]
        hand_pts = [positions[l] for l in ("LH", "RH") if l in positions]

        s = self.profile.segments
        if foot_pts and hand_pts:
            foot_mid = np.mean(foot_pts, axis=0)
            hand_mid = np.mean(hand_pts, axis=0)
            hand_to_foot = float(hand_mid[2] - foot_mid[2])
            # Pelvis sits between the feet and shoulders, biased toward
            # feet on a high-step / crouch pose.
            if hand_to_foot >= 0.50:
                # Extended pose: pelvis at hip height above feet.
                pelvis_z = float(foot_mid[2] + s.standing_leg * 0.85)
            else:
                # Crouched pose: pelvis low between hands and feet.
                pelvis_z = float(0.5 * (foot_mid[2] + hand_mid[2]))
            pelvis_x = float(0.5 * foot_mid[0] + 0.5 * hand_mid[0])
            pelvis_y = float(max(foot_mid[1], hand_mid[1]) + 0.55)
        elif foot_pts:
            foot_mid = np.mean(foot_pts, axis=0)
            pelvis_x = float(foot_mid[0])
            pelvis_y = float(foot_mid[1] + 0.55)
            pelvis_z = float(foot_mid[2] + s.standing_leg * 0.85)
        elif hand_pts:
            hand_mid = np.mean(hand_pts, axis=0)
            pelvis_x = float(hand_mid[0])
            pelvis_y = float(hand_mid[1] + 0.55)
            pelvis_z = float(hand_mid[2] - s.spine - 0.10)
        else:
            pelvis_x, pelvis_y, pelvis_z = 0.0, 0.5, s.standing_leg + 0.20

        # Clamp pelvis_z above the floor by at least one foot length.
        pelvis_z = max(pelvis_z, cfg.FLOOR_Z + s.standing_leg * 0.5)

        # Wall-clearance clamp: even at maximum spine lean the chest must
        # stay in front of the wall surface. Without this, MuJoCo starts
        # the settle with the body already penetrating the wall, which the
        # contact solver cannot reliably recover from.
        #
        # Derivation: wall_surface_Y ≈ min(hold_Y) − HOLD_PROTRUDE_M.
        # Worst-case chest front = pelvis_Y − spine*sin(max_lean) − chest_half_depth.
        # We require that to be > wall_surface_Y + a small safety margin.
        if positions:
            wall_ref_y = float(min(p[1] for p in positions.values())) - cfg.HOLD_PROTRUDE_M
            _max_lean = cfg.JOINT_LIMITS_RAD["spine_lean"][1]
            # Worst-case body extent: the HEAD at the far end of the spine.
            # When the spine leans forward by max_lean, the head (at distance
            # spine + neck/2 + head_radius from chest) projects toward the wall.
            _spine_chain = s.spine + s.head / 2 + s.head_radius
            _min_clearance = _spine_chain * math.sin(_max_lean) + 0.06
            pelvis_y = max(pelvis_y, wall_ref_y + _min_clearance)

        # ── 1) Reset state ────────────────────────────────────────
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = (pelvis_x, pelvis_y, pelvis_z)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0
        for limb in LIMBS:
            self._on_hold[limb] = None
            self.data.eq_active[self._eq_idx[limb]] = 0
        mujoco.mj_forward(self.model, self.data)

        # ── 2) Activate welds for the requested pose ──────────────
        for limb, hid in targets.items():
            if hid is not None:
                self.attach_limb(limb, hid)

        # ── 3) Settle: disable actuator gains, ramp gravity from 0 to
        #         full so the welds aren't smashed by the body's full
        #         weight in a single instant. Slip detection is OFF
        #         during settle — the spike forces here are an
        #         artefact of the warm-start, not a real grip event.
        #
        #         Exception: the spine actuator stays active at full gain
        #         targeting a slight forward lean. Without it, the 66 Nm
        #         gravitational torque on the upper body overwhelms the
        #         6 Nm/rad passive spring and slams the spine to its limit,
        #         driving the head through the wall.
        kp_backup = self.model.actuator_gainprm[:, 0].copy()
        gravity_backup = np.array(self.model.opt.gravity)
        self.model.actuator_gainprm[:, 0] = 0.0
        for i in range(self.model.nu):
            jnt_id = int(self.model.actuator_trnid[i, 0])
            jname = mujoco.mj_id2name(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id,
            )
            if jname == "spine_lean":
                # During settle we need ≫ normal kp to resist the ~66 Nm
                # gravitational torque on the upper body. Normal kp=120 gives
                # equilibrium at ~44° (still clipping). kp=1000 → ~4° lean.
                self.model.actuator_gainprm[i, 0] = 1000.0
                self.data.ctrl[i] = 0.0  # target upright; gravity settles ~4°
                # Also zero the spine qpos so the ramp starts from upright.
                jnt_qpos_adr = self.model.jnt_qposadr[jnt_id]
                self.data.qpos[jnt_qpos_adr] = 0.0
                break
        try:
            ramp_steps = int(cfg.SEED_RAMP_S / cfg.PHYS_DT)
            hold_steps = int(cfg.SEED_HOLD_S / cfg.PHYS_DT)
            for i in range(ramp_steps):
                alpha = (i + 1) / ramp_steps      # 0 → 1
                self.model.opt.gravity[:] = gravity_backup * alpha
                mujoco.mj_step(self.model, self.data)
            self.model.opt.gravity[:] = gravity_backup
            for _ in range(hold_steps):
                mujoco.mj_step(self.model, self.data)
        finally:
            self.model.actuator_gainprm[:, 0] = kp_backup
            self.model.opt.gravity[:] = gravity_backup

        # ── 4) Capture settled joint angles → actuator targets ────
        self._sync_actuator_targets_to_pose()

        # Damp out residual velocities. The welds settled the pose;
        # the actuator targets now match it; momentum can be reset.
        self.data.qvel[:] = 0.0
        self.slip_events.clear()

        # ── 5) Over-brace guard ───────────────────────────────────
        # The settle finds a self-consistent pose, but on geometry where
        # the start holds force a contorted hang (e.g. hands near ankle
        # height) the welds end up fighting each other and a limb can
        # settle ABOVE its slip cap. With slip on, that limb releases on
        # step 1 and the climber falls; with slip off it freezes in a
        # tensioned pose and only jiggles. Neither is learnable. The
        # settle cannot fix this — it is fixed by hold geometry — so the
        # least we can do is surface it instead of failing silently.
        self._record_seed_overload()

    def _record_seed_overload(self) -> None:
        """Measure each attached limb's settled force against its slip cap
        and record/warn for any limb left over cap by the seed pose."""
        mujoco.mj_forward(self.model, self.data)
        self.seed_overload = {}
        for limb in LIMBS:
            attach = self._on_hold[limb]
            if attach is None:
                continue
            f_mag = self._weld_force_magnitude(self._eq_idx[limb])
            cap = attach.max_force_n * cfg.SLIP_FORCE_SLACK
            if f_mag > cap:
                self.seed_overload[limb] = {
                    "hold_id": attach.hold_id,
                    "force_n": float(f_mag),
                    "cap_n": float(cap),
                    "ratio": float(f_mag / cap) if cap > 0 else float("inf"),
                }
        if self.seed_overload and not self._seed_overload_warned:
            self._seed_overload_warned = True
            detail = ", ".join(
                f"{l} {d['force_n']:.0f}N/{d['cap_n']:.0f}N "
                f"({d['ratio']:.1f}x) on {d['hold_id']}"
                for l, d in self.seed_overload.items()
            )
            print(
                f"[seed_pose] WARNING: over-braced seed on wall "
                f"'{self.wall.name}': {detail}. These limbs settle above "
                f"their slip cap — with slip enabled the climber will shed "
                f"them and fall on the first steps. The start-hold geometry "
                f"is unhangable for this body; pick/generate holds with a "
                f"larger hand-foot vertical gap.",
                file=sys.stderr,
            )

    # ─── Attach / release ─────────────────────────────────────────────
    def attach_limb(self, limb: Limb, hold_id: str) -> None:
        """Activate the per-limb weld between the hold (mocap) and the
        limb's fixed tip-anchor body.

        On attach we:
            1. Set the mocap to the hold's world position.
            2. Update eq_data so the weld's "relative pose" target is
               identity (limb tip ↔ mocap → snap together).
            3. Set eq_active = 1.
        """
        if hold_id not in self._hold_meta_by_id:
            raise KeyError(f"unknown hold: {hold_id}")
        meta = self._hold_meta_by_id[hold_id]
        mocap_idx = self._mocap_idx[limb]
        eq_idx = self._eq_idx[limb]

        # Position the mocap exactly at the hold's outer surface; body2 of
        # the weld is the fixed tip-anchor body colocated with the site.
        target = np.array(meta["world_pos"], dtype=np.float64)
        self.data.mocap_pos[mocap_idx] = target
        self.data.mocap_quat[mocap_idx] = (1.0, 0.0, 0.0, 0.0)

        # Configure the weld. Body2 is the fixed tip-anchor body colocated
        # with the rendered tip site, so a zero relpose pins the actual
        # contact point to the mocap. torquescale=0 keeps this position-only:
        # hands and feet can still pivot naturally on holds.
        eq_data = self.model.eq_data[eq_idx]
        eq_data[0:3] = (0.0, 0.0, 0.0)
        eq_data[3:6] = (0.0, 0.0, 0.0)
        eq_data[6:10] = (1.0, 0.0, 0.0, 0.0)
        if eq_data.shape[0] >= 11:
            eq_data[10] = 0.0

        self.data.eq_active[eq_idx] = 1
        self._on_hold[limb] = HoldAttachment(
            hold_id=hold_id,
            world_pos=tuple(float(v) for v in target),
            max_force_n=self._max_force_for(limb, meta),
        )

    def release_limb(self, limb: Limb) -> None:
        eq_idx = self._eq_idx[limb]
        self.data.eq_active[eq_idx] = 0
        self._on_hold[limb] = None

    def move_limb(
        self,
        limb: Limb,
        target_hold_id: str,
        *,
        mode: str = "reach",
    ) -> None:
        """Initiate a limb move.

        Three modes:

        ``"snap"`` — instant teleport. Releases the weld, teleports the
            mocap, re-welds. Use for fast RL training where you don't
            care how the climber gets between holds, only that they
            do. The body stays put because all other welds are still
            engaged.

        ``"reach"`` (default) — continuous Cartesian-impedance reach.
            Releases the weld and registers a `ReachState`. Subsequent
            calls to `step()` apply a PD force pulling the limb tip
            toward the target hold. When the tip is within
            `REACH_ATTACH_RADIUS` (default 5 cm) of the target — or
            after `REACH_TIMEOUT_S` — the weld engages.
            **Non-blocking:** call `step()` afterwards to actually do
            the reach. The Gym env's `move_frames` setting is the
            number of frames it gives a reach to complete.

        ``"dyno"`` — explosive reach. Same as `reach` but with a
            higher PD gain on the moving limb plus a brief leg-extension
            push during the first 0.35 s. Useful for moves that are
            geometrically out of static reach.
        """
        if target_hold_id not in self._hold_meta_by_id:
            raise KeyError(f"unknown hold: {target_hold_id}")
        if mode == "snap":
            self.release_limb(limb)
            self.attach_limb(limb, target_hold_id)
            self._reaching[limb] = None
            return
        if mode in ("reach", "dyno"):
            self.release_limb(limb)
            # Snapshot the pelvis position to hold during this move (balance
            # assist) — pin it so the reach reaction can't barn-door the body.
            self._balance_target = self.pelvis_pos().copy()
            target = np.array(self._hold_meta_by_id[target_hold_id]["world_pos"])
            kp = cfg.REACH_KP_HAND if limb in HAND_LIMBS else cfg.REACH_KP_FOOT
            kd = cfg.REACH_KD_HAND if limb in HAND_LIMBS else cfg.REACH_KD_FOOT
            if mode == "dyno":
                kp *= cfg.DYNO_KP_BOOST
            self._reaching[limb] = ReachState(
                target_hold_id=target_hold_id,
                target_world_pos=target,
                t_remaining=cfg.REACH_TIMEOUT_S,
                kp=kp,
                kd=kd,
                dyno=(mode == "dyno"),
                closest_dist=1e9,
                closest_t=cfg.REACH_TIMEOUT_S,
            )
            return
        raise ValueError(f"unknown mode: {mode!r}")

    def _max_force_for(self, limb: Limb, meta: dict) -> float:
        if limb in HAND_LIMBS:
            base = self.profile.grip_force_n * cfg.HAND_FORCE_MULTIPLIER
        else:
            base = self.profile.foot_push_force_n * cfg.FOOT_FORCE_MULTIPLIER
        cap = base * meta["positivity"]
        if meta["max_force_n"] is not None:
            cap = min(cap, meta["max_force_n"])
        return cap

    # ─── Stepping ─────────────────────────────────────────────────────
    def step(self, frames: int = 1, *, check_slip: bool = False) -> int:
        """Advance physics by `frames` render frames.

        Per sub-step the loop runs:
            1. Relax actuators on any reaching-limb chains so the
               Cartesian impedance can actually move the limb.
            2. Apply Cartesian-impedance forces to reaching limb tips.
            3. Apply dyno leg-push if a dyno is in flight.
            4. mj_step(): integrate one physics tick.
            5. Check whether any reach completed (limb tip near target
               or timeout) — if so, weld and clear the reach state.
            6. Restore actuator gains.
            7. Optionally check for grip slip.
        """
        slip_count = 0
        # Snapshot original actuator gains AND bias; we temporarily zero BOTH
        # on the reaching limbs' joints during each substep. (Zeroing only the
        # gain leaves the position servo's -kp·qpos bias as a spring-to-zero
        # that fights the reach controller — the bug that stalled every move
        # ~0.1 m short of its hold.)
        kp_orig = self.model.actuator_gainprm[:, 0].copy()
        bias_orig = self.model.actuator_biasprm.copy()
        for _ in range(frames * cfg.SUBSTEPS_PER_FRAME):
            self._relax_reaching_actuators(kp_orig, bias_orig)
            self._apply_reach_forces()
            mujoco.mj_step(self.model, self.data)
            self._update_reach_state(cfg.PHYS_DT)
            if check_slip:
                slip_count += self._check_slip()
        # Restore gains/bias and clear applied forces.
        self.model.actuator_gainprm[:, 0] = kp_orig
        self.model.actuator_biasprm[:] = bias_orig
        self.data.qfrc_applied[:] = 0.0
        return slip_count

    def _relax_reaching_actuators(self, kp_orig: np.ndarray,
                                  bias_orig: np.ndarray) -> None:
        """Fully relax (zero gain AND the -kp/-kv bias) the actuators on any
        limb chain currently reaching, restore the rest. A position servo's
        force is gain·ctrl − kp·qpos − kv·qvel; zeroing only the gain leaves
        −kp·qpos − kv·qvel, i.e. a spring pulling the joint back to angle 0,
        which fought the Cartesian reach controller and pinned the limb short
        of its target. Zeroing the bias columns too makes the chain truly
        free so the reach can extend it. One-substep scoped."""
        self.model.actuator_gainprm[:, 0] = kp_orig
        self.model.actuator_biasprm[:] = bias_orig
        for limb in LIMBS:
            if self._reaching[limb] is None:
                continue
            for aid in self._limb_actuator_ids[limb]:
                self.model.actuator_gainprm[aid, 0] = 0.0
                self.model.actuator_biasprm[aid, 1] = 0.0   # −kp (spring)
                self.model.actuator_biasprm[aid, 2] = 0.0   # −kv (damping)

    # ─── Continuous-reach controller ──────────────────────────────────
    def _apply_reach_forces(self) -> None:
        """Cartesian PD on each reaching limb's tip site.

        Uses `mj_applyFT` to map a world-frame force at the tip site
        into the generalised-coordinate force vector qfrc_applied. We
        also temporarily zero the limb-chain actuator gains so the
        actuator's "hold the seeded angle" behaviour doesn't fight
        the reach controller.

        The dyno boost: while a dyno reach is in flight, push the
        knees/hips toward extension by adding torque to those joints'
        qfrc_applied. Crude but effective for the "throw yourself at
        the hold" dynamics.
        """
        # Reset just the qfrc slots we touch each substep so impulses
        # don't accumulate across substeps. (We can't blanket-zero
        # because other code may add to qfrc_applied.)
        self.data.qfrc_applied[:] = 0.0

        for limb in LIMBS:
            rs = self._reaching[limb]
            if rs is None:
                continue
            site_id = self._tip_site_idx[limb]
            tip_pos = np.array(self.data.site_xpos[site_id])
            # Tip linear velocity in world frame via 6-D site velocity.
            vel = np.zeros(6, dtype=np.float64)
            mujoco.mj_objectVelocity(
                self.model, self.data, mujoco.mjtObj.mjOBJ_SITE,
                site_id, vel, 0,   # 0 = world frame
            )
            tip_vel = vel[3:6]

            err = rs.target_world_pos - tip_pos
            force = rs.kp * err - rs.kd * tip_vel

            # Apply force at the tip world position to the limb body.
            body_id = self._limb_body_idx[limb]
            mujoco.mj_applyFT(
                self.model, self.data,
                np.asarray(force, dtype=np.float64),
                np.zeros(3, dtype=np.float64),     # no torque
                np.asarray(tip_pos, dtype=np.float64),
                body_id,
                self.data.qfrc_applied,
            )

            # Dyno leg push: drive knee + hip-flex toward extension.
            if rs.dyno and rs.t_remaining > (cfg.REACH_TIMEOUT_S - cfg.DYNO_DURATION_S):
                self._dyno_leg_push(cfg.DYNO_LEG_PUSH_NM)

        # ── Balance assist ──────────────────────────────────────────────
        # While any limb is reaching, hold the pelvis at its pre-move position
        # with a Cartesian PD on the free-joint root, so the reach reaction
        # can't tip the body off its support (the transitional-stance instability
        # that breaks chaining). qfrc_applied[0:3] are world-frame forces on the
        # free root translation; qvel[0:3] is its world-frame linear velocity.
        if (cfg.BALANCE_KP > 0.0 and self._balance_target is not None
                and any(self._reaching[l] is not None for l in LIMBS)):
            pelvis = self.data.qpos[0:3]
            pvel = self.data.qvel[0:3]
            self.data.qfrc_applied[0:3] += (
                cfg.BALANCE_KP * (self._balance_target - pelvis)
                - cfg.BALANCE_KD * pvel
            )

    def _dyno_leg_push(self, torque_nm: float) -> None:
        """Add torque to both knees and hip-flex joints to push the
        body upward / outward during a dyno. Direction: extend (negative
        knee angle isn't physical; the convention is knee=0 = extended,
        so we push toward 0 → negative torque if knee>0)."""
        for jname in ("l_knee", "r_knee", "l_hip_flex", "r_hip_flex"):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            vadr = int(self.model.jnt_dofadr[jid])
            qadr = int(self.model.jnt_qposadr[jid])
            cur = float(self.data.qpos[qadr])
            # Push toward 0 (extended) — same sign as -cur.
            self.data.qfrc_applied[vadr] += -np.sign(cur) * torque_nm

    def _update_reach_state(self, dt: float) -> None:
        """Decrement timers, track closest-approach, finalise reaches
        whose distance crossed the attach threshold or whose timer ran
        out."""
        for limb in LIMBS:
            rs = self._reaching[limb]
            if rs is None:
                continue
            tip = self.limb_tip_pos(limb)
            dist = float(np.linalg.norm(tip - rs.target_world_pos))
            if dist < rs.closest_dist:
                rs.closest_dist = dist
                rs.closest_t = rs.t_remaining
            rs.t_remaining -= dt
            if dist < cfg.REACH_ATTACH_RADIUS:
                # On target — engage weld and clear reach.
                self.attach_limb(limb, rs.target_hold_id)
                self._reaching[limb] = None
            elif rs.t_remaining <= 0.0:
                # Timeout — weld at the closest-approach if reasonable,
                # else snap (the move "failed" but we don't leave the
                # limb dangling forever).
                self.attach_limb(limb, rs.target_hold_id)
                self._reaching[limb] = None

    def limb_grip_force(self, limb: Limb) -> float:
        """Magnitude of the world-frame force the climber is currently
        applying through this limb (Newtons). Returns 0 if the limb
        is not on a hold."""
        if self._on_hold[limb] is None:
            return 0.0
        return self._weld_force_magnitude(self._eq_idx[limb])

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for limb in LIMBS:
            self._on_hold[limb] = None
        self.slip_events.clear()
        self.data.ctrl[:] = 0.0

    def _sync_actuator_targets_to_pose(self) -> None:
        """Snapshot the current joint angles and write them as actuator
        targets. After this, the actuators try to *hold* the current
        pose rather than servo toward zero."""
        for i, qadr in enumerate(self._actuator_jnt_qposadr):
            self.data.ctrl[i] = self.data.qpos[qadr]

    def _weld_force_magnitude(self, eq_idx: int) -> float:
        """Linear force (N) carried by a single weld equality.

        Reads the constraint-force rows belonging to *this* weld from
        `data.efc_force`, located via ``efc_id == eq_idx`` among the
        equality-type rows. Each weld emits 6 rows; because holds are
        welded with ``torquescale=0`` (position-only) the 3 rotational
        rows are zero, so the norm over this weld's rows equals the
        translational force magnitude in Newtons — isolated to this
        limb, unlike `cfrc_int` (which sums every constraint acting on a
        body, over-counting 6–16× when multiple limbs are loaded, and is
        only populated by `mj_rnePostConstraint`, which `mj_step` never
        calls — so it read identically zero and slip never fired).

        `data.efc_*` is filled by the constraint solver inside `mj_step`,
        so this is valid immediately after a step.
        """
        d = self.data
        nefc = int(d.nefc)
        if nefc == 0:
            return 0.0
        mask = (
            (d.efc_type[:nefc] == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            & (d.efc_id[:nefc] == eq_idx)
        )
        if not np.any(mask):
            return 0.0
        return float(np.linalg.norm(d.efc_force[:nefc][mask]))

    def _check_slip(self) -> int:
        """Detect any hold whose limb is exceeding its rated capacity
        and release it. Logs a SlipEvent for each release.

        Single-substep debounce: we release at most one limb per substep.
        The per-weld force is now exact (see `_weld_force_magnitude`), so
        this is no longer masking an over-count — it's a deliberate choice
        to shed load one limb at a time. When several limbs are over cap,
        popping the most-loaded one lets the next 2 ms substep redistribute
        force before deciding whether the rest also slip, which yields a
        graceful cascade instead of dropping all four from a single spike.
        """
        slip_count = 0
        slipped_this_substep = False
        for limb in LIMBS:
            if slipped_this_substep:
                break
            attach = self._on_hold[limb]
            if attach is None:
                continue
            f_mag = self._weld_force_magnitude(self._eq_idx[limb])
            cap = attach.max_force_n * cfg.SLIP_FORCE_SLACK
            if f_mag > cap:
                slipped_this_substep = True
                self.data.eq_active[self._eq_idx[limb]] = 0
                if len(self.slip_events) < self._max_slip_log:
                    self.slip_events.append(SlipEvent(
                        t=float(self.data.time),
                        limb=limb,
                        hold_id=attach.hold_id,
                        force_n=f_mag,
                        capacity_n=cap,
                    ))
                self._on_hold[limb] = None
                slip_count += 1
        return slip_count

    # ─── State accessors ──────────────────────────────────────────────
    def pelvis_pos(self) -> np.ndarray:
        return np.array(self.data.qpos[0:3])

    def com(self) -> np.ndarray:
        """Centre of mass (m) from the subtreecom sensor."""
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, "com",
        )
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return np.array(self.data.sensordata[adr:adr + dim])

    def limb_tip_pos(self, limb: Limb) -> np.ndarray:
        site_idx = self._tip_site_idx[limb]
        return np.array(self.data.site_xpos[site_idx])

    def on_hold(self, limb: Limb) -> Optional[str]:
        a = self._on_hold[limb]
        return a.hold_id if a else None

    # ─── Pose snapshot for visualisation ──────────────────────────────
    def pose_snapshot(self, *, include_static: bool = False) -> dict:
        """Body positions/orientations for the web viewer.

        Returned shape:
            {
              "t": float,                       # sim time (s)
              "bodies": {name: {"pos": [x,y,z], "quat": [w,x,y,z]}},
              "limbs":  {LH: hold_id|null, ...},
              "reaching": {LH: hold_id|null, ...}, # set if the limb is mid-reach
              "holds":  {hold_id: [x,y,z]}      # tip world position
            }

        If `include_static=True` we also include wall + per-hold geometry
        (radius, normal direction, plate size) under "static". The web
        viewer fetches this once on session create rather than every frame.
        """
        names = (
            "pelvis", "chest", "head",
            "l_upperarm", "l_forearm", "l_hand",
            "r_upperarm", "r_forearm", "r_hand",
            "l_thigh", "l_shin", "l_foot",
            "r_thigh", "r_shin", "r_foot",
        )
        bodies = {}
        for n in names:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, n)
            if bid < 0:
                continue
            pos = np.array(self.data.xpos[bid]).tolist()
            quat = np.array(self.data.xquat[bid]).tolist()  # [w,x,y,z]
            bodies[n] = {"pos": pos, "quat": quat}

        snap = {
            "t": float(self.data.time),
            "bodies": bodies,
            "limbs": {
                l: (self._on_hold[l].hold_id if self._on_hold[l] else None)
                for l in LIMBS
            },
            "reaching": {
                l: (self._reaching[l].target_hold_id if self._reaching[l] else None)
                for l in LIMBS
            },
            "holds": {
                meta["hold_id"]: list(meta["world_pos"])
                for meta in self._hold_meta_by_id.values()
            },
        }

        if include_static:
            # Static info needed by the viewer once. Re-derived from the
            # builder math so the viewer doesn't have to know MJCF
            # internals — the server is source of truth.
            from sim3d import config as _cfg
            theta = math.radians(self.wall.wall_angle_deg)
            width_m = self.wall.width_cm / 100.0
            height_m = self.wall.height_cm / 100.0
            pad = _cfg.WALL_PADDING_M
            plate_w = width_m + 2 * pad
            plate_h = height_m + pad     # pad above only
            wall_normal = (0.0, math.cos(theta), -math.sin(theta))

            holds_static = {}
            for meta in self._hold_meta_by_id.values():
                radius_key = self.wall.by_id(meta["hold_id"]).size
                radius = _cfg.HOLD_RADIUS_BY_SIZE_M.get(radius_key, 0.05)
                holds_static[meta["hold_id"]] = {
                    "world_pos": list(meta["world_pos"]),
                    "wall_normal": list(meta["wall_normal"]),
                    "radius": radius,
                    "protrude": _cfg.HOLD_PROTRUDE_M,
                    "is_start": meta["is_start"],
                    "is_finish": meta["is_finish"],
                    "color": self.wall.by_id(meta["hold_id"]).color,
                }

            # Plate centre must exactly match builder._build_wall_xml.
            # If this drifts, especially on 40° MoonBoard overhangs, the
            # browser shows the wall on one diagonal while MuJoCo/body/holds
            # occupy another. Keep this math in lockstep with the MJCF.
            cy_base = (plate_h / 2.0) * math.sin(theta)
            cz_base = (plate_h / 2.0) * math.cos(theta)
            if self.wall.holds:
                cos_t0 = math.cos(theta)
                sin_t0 = math.sin(theta)
                local_y_tip0 = _cfg.WALL_THICKNESS_M / 2.0 + _cfg.HOLD_PROTRUDE_M
                min_world_z = min(
                    cz_base - local_y_tip0 * sin_t0
                    + ((h.y_cm / 100.0) - plate_h / 2.0) * cos_t0
                    for h in self.wall.holds
                )
                deficit = _cfg.FLOOR_Z + _cfg.HOLD_FLOOR_CLEARANCE - min_world_z
                if deficit > 0:
                    cz_base += deficit

            snap["static"] = {
                "wall": {
                    "width_m": width_m,
                    "height_m": height_m,
                    "plate_w": plate_w,
                    "plate_h": plate_h,
                    "thickness": _cfg.WALL_THICKNESS_M,
                    "angle_rad": theta,
                    "angle_deg": self.wall.wall_angle_deg,
                    "centre": [0.0, cy_base, cz_base],
                    "normal": list(wall_normal),
                },
                "holds": holds_static,
            }
        return snap
