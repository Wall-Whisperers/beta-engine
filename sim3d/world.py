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
       hand/foot body to the mocap.
    4. To release: set eq_active[i]=0.

Why mocap+weld rather than connect with sites? `connect` needs two
bodies and only constrains position; sites can't be the second arg of a
weld. Mocap+weld lets us teleport the target without recompiling the
model — exactly what RL needs for fast resets.
"""
from __future__ import annotations

import math
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
    ) -> None:
        self.wall = wall
        self.profile = profile or ClimberProfile()

        xml, hold_meta = build_mjcf_xml(wall, self.profile)
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

        # Track slips for diagnostics / RL reward shaping.
        self.slip_events: list[SlipEvent] = []
        self._max_slip_log = 200

        # ── Tip-site offsets (in limb-body local frame) ────────────
        # The weld snaps the limb body origin to the mocap. To make
        # the actual *tip* site sit on the hold, we encode the body→site
        # offset into the weld's relpose. Without this, hands end up
        # with the wrist on the hold and fingers dangling 12 cm below.
        self._tip_site_offset: dict[Limb, np.ndarray] = {}
        for limb in LIMBS:
            site_id = self._tip_site_idx[limb]
            self._tip_site_offset[limb] = np.array(
                self.model.site_pos[site_id], dtype=np.float64,
            )

        # First mj_forward so kinematics are populated for any caller
        # that asks for site positions before stepping.
        mujoco.mj_forward(self.model, self.data)
        self._sync_actuator_targets_to_pose()

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
            pelvis_y = float(max(foot_mid[1], hand_mid[1]) + 0.35)
        elif foot_pts:
            foot_mid = np.mean(foot_pts, axis=0)
            pelvis_x = float(foot_mid[0])
            pelvis_y = float(foot_mid[1] + 0.35)
            pelvis_z = float(foot_mid[2] + s.standing_leg * 0.85)
        elif hand_pts:
            hand_mid = np.mean(hand_pts, axis=0)
            pelvis_x = float(hand_mid[0])
            pelvis_y = float(hand_mid[1] + 0.35)
            pelvis_z = float(hand_mid[2] - s.spine - 0.10)
        else:
            pelvis_x, pelvis_y, pelvis_z = 0.0, 0.5, s.standing_leg + 0.20

        # Clamp pelvis_z above the floor by at least one foot length.
        pelvis_z = max(pelvis_z, cfg.FLOOR_Z + s.standing_leg * 0.5)

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
        kp_backup = self.model.actuator_gainprm[:, 0].copy()
        gravity_backup = np.array(self.model.opt.gravity)
        self.model.actuator_gainprm[:, 0] = 0.0
        try:
            ramp_steps = int(0.3 / cfg.PHYS_DT)
            hold_steps = int(0.4 / cfg.PHYS_DT)
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

    # ─── Attach / release ─────────────────────────────────────────────
    def attach_limb(self, limb: Limb, hold_id: str) -> None:
        """Activate the per-limb weld between the hold (mocap) and the
        limb's tip body (l_hand / r_hand / l_foot / r_foot).

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

        # Position the mocap at the hold's outer surface, slightly
        # offset along the wall normal so the limb doesn't sink in.
        wp = np.array(meta["world_pos"])
        n = np.array(meta["wall_normal"])
        target = wp + n * 0.02
        self.data.mocap_pos[mocap_idx] = target
        self.data.mocap_quat[mocap_idx] = (1.0, 0.0, 0.0, 0.0)

        # Configure the weld.
        # eq_data layout for mjEQ_WELD is
        #     [anchor(3) | relpos(3) | relquat(4) | torquescale(1)]
        # and the constraint enforces
        #     body2_world = body1_world ⊕ relpose
        # i.e. body2 (limb) sits at body1 (mocap) plus the relative pose.
        #
        # We want the *tip site* of body2 to coincide with body1's
        # origin. The site is offset by `s` from body2's origin
        # (in body2-local coords). So:
        #     site_world = body2_world + R(body2_quat) · s
        #                = body1_world + R(body1_quat) · relpos
        #                  + R(body1_quat · relquat) · s
        # Setting relquat = identity (free rotation, see torquescale=0)
        # and assuming body1 (mocap) is roughly identity-rotated, the
        # condition site_world = body1_world reduces to
        #     relpos + s = 0   →   relpos = -s
        # which puts body2 origin "behind" the mocap by the site offset,
        # so the tip itself lands on the mocap.
        #
        # torquescale=0 turns off rotation locking — the limb can still
        # rotate freely on the hold (a real hand can pivot around a
        # crimp). Without this, every weld also welds the hand's
        # orientation rigidly, which produces unnatural torso twists.
        s = self._tip_site_offset[limb]
        eq_data = self.model.eq_data[eq_idx]
        eq_data[0:3] = (0.0, 0.0, 0.0)
        eq_data[3:6] = -s
        eq_data[6:10] = (1.0, 0.0, 0.0, 0.0)
        if eq_data.shape[0] >= 11:
            eq_data[10] = 0.0

        self.data.eq_active[eq_idx] = 1
        self._on_hold[limb] = HoldAttachment(
            hold_id=hold_id,
            world_pos=tuple(target),
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
        mode: str = "snap",
    ) -> None:
        """Move a limb to a new hold.

        mode="snap"  — release, teleport mocap, re-attach. Instant.
                       Best for RL training (no in-flight physics).
        mode="reach" — release, run physics ~0.3 s while the body
                       shifts weight, then re-attach. Slower but
                       produces dynamic-looking betas.
        """
        if mode == "snap":
            self.release_limb(limb)
            self.attach_limb(limb, target_hold_id)
            return
        if mode == "reach":
            self.release_limb(limb)
            for _ in range(int(0.3 / cfg.PHYS_DT)):
                mujoco.mj_step(self.model, self.data)
            self.attach_limb(limb, target_hold_id)
            return
        raise ValueError(f"unknown mode: {mode!r}")

    def _max_force_for(self, limb: Limb, meta: dict) -> float:
        base = (
            self.profile.grip_force_n if limb in HAND_LIMBS
            else self.profile.foot_push_force_n
        )
        cap = base * meta["positivity"]
        if meta["max_force_n"] is not None:
            cap = min(cap, meta["max_force_n"])
        return cap

    # ─── Stepping ─────────────────────────────────────────────────────
    def step(self, frames: int = 1, *, check_slip: bool = False) -> int:
        """Advance physics by `frames` render frames.

        If `check_slip` is True, after each physics sub-step we
        compute the world-frame force on each active weld and
        deactivate any whose force exceeds the hold's rated
        capacity × `SLIP_FORCE_SLACK`. Disabled by default — the slip
        model is opinionated and easy to tune wrong; turn it on in the
        Gym env or via `--slip` once you trust the numbers.

        Returns the number of slip events that occurred this call.
        """
        slip_count = 0
        for _ in range(frames * cfg.SUBSTEPS_PER_FRAME):
            mujoco.mj_step(self.model, self.data)
            if check_slip:
                slip_count += self._check_slip()
        return slip_count

    def limb_grip_force(self, limb: Limb) -> float:
        """Magnitude of the world-frame force the climber is currently
        applying through this limb (Newtons). Returns 0 if the limb
        is not on a hold."""
        if self._on_hold[limb] is None:
            return 0.0
        return self._weld_force_magnitude(self._eq_idx[limb], limb)

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

    def _weld_force_magnitude(self, eq_idx: int, limb: Limb) -> float:
        """Approximate the world-frame force on the limb body from this
        equality. Computed from `data.qfrc_constraint` projected onto
        the limb body's translational direction via the body Jacobian.

        Why not `efc_force`? The raw Lagrange multipliers for a 6D
        weld mix translation and rotation rows in different units
        (Newtons and Newton-metres) — summing them gives a number with
        no clean physical meaning. The Jacobian-projected approach
        below returns the linear force the body sees, in Newtons.

        This is still an approximation: the qfrc_constraint accumulates
        forces from ALL active constraints (not just this weld). When
        only one limb is welded, the result is exact; with four welds
        active it's an upper bound. Good enough for slip detection.
        """
        body_id = self._limb_body_idx[limb]
        # mj_objectVelocity / objectAcceleration / similar exists, but
        # we want force. Use the body's Jacobian: f = J^T λ ⇒
        # f_body = J · qfrc_constraint reverses out the world-frame
        # force at the body origin. Approximated as the mass × the
        # constraint-induced acceleration.
        # Easier route — cfrc_int (internal) and cfrc_ext (external)
        # constraint forces on each body are computed during step.
        # cfrc_int[body, 0:3] is the *torque* and cfrc_int[body, 3:6]
        # is the *linear force* applied to the body by joint/equality
        # constraints (MuJoCo convention).
        if self.data.cfrc_int is None or body_id >= len(self.data.cfrc_int):
            return 0.0
        f = self.data.cfrc_int[body_id, 3:6]
        return float(np.linalg.norm(f))

    def _check_slip(self) -> int:
        """Detect any hold whose limb is exceeding its rated capacity
        and release it. Logs a SlipEvent for each release."""
        slip_count = 0
        for limb in LIMBS:
            attach = self._on_hold[limb]
            if attach is None:
                continue
            f_mag = self._weld_force_magnitude(self._eq_idx[limb], limb)
            cap = attach.max_force_n * cfg.SLIP_FORCE_SLACK
            if f_mag > cap:
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
    def pose_snapshot(self) -> dict:
        """Body positions/orientations for the web viewer.

        Returned shape:
            {
              "t": float,                      # sim time (s)
              "bodies": {name: {"pos": [x,y,z], "quat": [w,x,y,z]}},
              "limbs":  {LH: hold_id|null, ...},
              "holds":  {hold_id: [x,y,z]}     # static — caller can cache
            }
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

        return {
            "t": float(self.data.time),
            "bodies": bodies,
            "limbs": {
                l: (self._on_hold[l].hold_id if self._on_hold[l] else None)
                for l in LIMBS
            },
            "holds": {
                meta["hold_id"]: list(meta["world_pos"])
                for meta in self._hold_meta_by_id.values()
            },
        }
