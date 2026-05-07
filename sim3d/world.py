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

        # ── Default actuator targets — current joint angles, so the
        # body holds whatever pose seed_pose puts it in.
        self._actuator_targets = np.zeros(self.model.nu)

        # First mj_forward so kinematics are populated for any caller
        # that asks for site positions before stepping.
        mujoco.mj_forward(self.model, self.data)

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

        Heuristic for the pelvis pose:
            X = average of foot X positions
            Y = ~0.35 m off the wall (hip clearance)
            Z = foot midpoint Z + leg_length × 0.85
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
        if foot_pts:
            foot_mid = np.mean(foot_pts, axis=0)
            pelvis_x = float(foot_mid[0])
            pelvis_y = float(foot_mid[1] + 0.30)
            pelvis_z = float(foot_mid[2] + s.standing_leg * 0.85)
        elif hand_pts:
            hand_mid = np.mean(hand_pts, axis=0)
            pelvis_x = float(hand_mid[0])
            pelvis_y = float(hand_mid[1] + 0.30)
            pelvis_z = float(hand_mid[2] - s.spine - 0.10)
        else:
            pelvis_x, pelvis_y, pelvis_z = 0.0, 0.5, s.standing_leg + 0.20

        # Reset qpos / qvel: free joint is the first 7 entries
        # (3 pos + 4 quat). Identity quaternion = (1,0,0,0).
        self.data.qpos[:] = 0.0
        self.data.qpos[0:3] = (pelvis_x, pelvis_y, pelvis_z)
        self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
        self.data.qvel[:] = 0.0
        self._actuator_targets = np.zeros(self.model.nu)
        self.data.ctrl[:] = 0.0

        # First propagate kinematics so site positions are valid.
        mujoco.mj_forward(self.model, self.data)

        # Detach everything, then attach per the requested mapping.
        for limb in LIMBS:
            self.release_limb(limb)
        for limb, hid in targets.items():
            if hid is not None:
                self.attach_limb(limb, hid)

        # Step a few times to let the constraints settle without
        # external posture forces. The weld will yank the limbs onto
        # the holds; we just need to settle the joints.
        for _ in range(int(0.2 / cfg.PHYS_DT)):
            mujoco.mj_step(self.model, self.data)

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

        # Configure the weld:
        # eq_data layout for weld is [anchor(3) | relpos(3) | relquat(4) | torquescale(1)]
        # in MuJoCo 3.x. We want anchor at the mocap origin (0,0,0),
        # relpos = limb_tip_in_mocap_frame = current offset, relquat = identity.
        # The simplest approach: set anchor=(0,0,0), set relpos to the
        # vector from the mocap to the limb tip in world frame at the
        # moment of attachment, set relquat=identity. This way the
        # weld's "rest pose" matches the current geometry — no yank.
        # Then we *teleport* the limb to the mocap by directly setting
        # relpos=0.
        # For a simpler first pass: set relpos=0, accept a small initial
        # yank as the constraint pulls the limb to the mocap. The
        # solver settles within ~50 ms.
        eq_data = self.model.eq_data[eq_idx]
        eq_data[0:3] = (0.0, 0.0, 0.0)   # anchor in body2 frame
        eq_data[3:6] = (0.0, 0.0, 0.0)   # relative position
        eq_data[6:10] = (1.0, 0.0, 0.0, 0.0)  # relative quat (identity)
        if eq_data.shape[0] >= 11:
            eq_data[10] = 1.0            # torquescale (full)

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
    def step(self, frames: int = 1) -> None:
        """Advance physics by `frames` render frames."""
        for _ in range(frames * cfg.SUBSTEPS_PER_FRAME):
            mujoco.mj_step(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        for limb in LIMBS:
            self._on_hold[limb] = None

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
