"""Build a complete MuJoCo MJCF scene from a Wall + ClimberProfile.

Why a Python builder rather than a static XML?
    1. The wall's hold layout, angle, and dimensions are data-driven
       (from the editor's JSON). Templating it at runtime is much
       cleaner than maintaining a parallel XML library.
    2. The climber's segment lengths come from the user's height /
       wingspan. Same reason — runtime templating is the right tool.
    3. Equality constraints reference sites by name, and we want
       names to be deterministic so the world layer can look them up.

The builder emits a single XML string. `Climb3DWorld` compiles that
string into an `MjModel` once at construction and re-uses it for the
lifetime of the simulator.

If you ever want to inspect the generated XML, call
`build_mjcf_xml(wall, profile)` and print the result.
"""
from __future__ import annotations

import math
from typing import Iterable
from xml.sax.saxutils import escape

from sim3d import config as cfg
from sim3d.body import (
    LIMB_EQUALITY,
    LIMB_MOCAP_BODY,
    LIMB_TIP_SITE,
    ClimberProfile,
    Segments,
)
from solver.wall import Hold, Wall


# ─── Coordinate transforms ────────────────────────────────────────────────
# The Wall stores hold positions in the wall's own 2D plane (cm). To
# place a hold in the 3D scene we map:
#     wall_x_cm  →  world_x  (along the wall, +X right)
#     wall_y_cm  →  along the tilted wall plane, projected onto (Y, Z)
#
# The wall plane rotates around the X axis by `wall_angle_deg`. Slab
# walls (negative angle) tilt the top *away* from the climber; overhang
# (positive) tilts it *toward* the climber.

def _wall_plane_to_world(
    wx_cm: float,
    wy_cm: float,
    wall_angle_deg: float,
    wall_width_cm: float,
) -> tuple[float, float, float]:
    """Map a point in wall-plane (cm) into world (m).

    The wall is centred horizontally on the world origin so the
    climber's COM starts at X≈0."""
    cx = (wx_cm - wall_width_cm / 2.0) / 100.0     # X centred
    plane_y = wy_cm / 100.0                         # distance up the wall
    theta = math.radians(wall_angle_deg)
    # Wall plane vector (unit, in world): up the wall is (0, sin θ, cos θ)
    # for our convention (positive θ = overhang, top toward climber → -Y).
    # Re-derived: a point at distance d up the wall sits at
    #     (cx, -d·sinθ, d·cosθ)
    return (cx, -plane_y * math.sin(theta), plane_y * math.cos(theta))


def _wall_normal_world(wall_angle_deg: float) -> tuple[float, float, float]:
    """Outward normal of the wall plane in world coords.

    Convention: climber stands at +Y looking toward -Y. The wall sits
    in the X-Z plane with bottom at z=0. Positive θ = overhang (top
    tilts toward climber, +Y); negative θ = slab.
    """
    theta = math.radians(wall_angle_deg)
    # Plate is rotated by +θ around +X. Plate-local +Y axis (the
    # outward normal in the plate frame) maps to world (0, cosθ, -sinθ).
    return (0.0, math.cos(theta), -math.sin(theta))


# ─── XML helpers ──────────────────────────────────────────────────────────
def _xyz(v: tuple[float, float, float]) -> str:
    return f"{v[0]:.5f} {v[1]:.5f} {v[2]:.5f}"


def _rgba_from_hex(hex_color: str, alpha: float = 1.0) -> str:
    """'#22c55e' → '0.13 0.77 0.37 1.0'. Falls back to grey on bad input."""
    s = hex_color.lstrip("#")
    if len(s) != 6:
        return f"0.5 0.5 0.5 {alpha:.2f}"
    try:
        r = int(s[0:2], 16) / 255.0
        g = int(s[2:4], 16) / 255.0
        b = int(s[4:6], 16) / 255.0
    except ValueError:
        return f"0.5 0.5 0.5 {alpha:.2f}"
    return f"{r:.3f} {g:.3f} {b:.3f} {alpha:.2f}"


# ─── Climber XML ──────────────────────────────────────────────────────────
def _build_climber_xml(profile: ClimberProfile, start_pos_m: tuple[float, float, float]) -> str:
    """Emit the <body> tree for the climber, with a free joint at the
    pelvis. Caller embeds this inside <worldbody>.

    Tree layout (joint count in parens):

        pelvis (free, 6)
        ├── chest (spine_lean, 1)
        │   ├── head
        │   ├── L shoulder (3) → upper_arm → elbow (1) → forearm → hand [site_lh_tip]
        │   └── R shoulder (3) → upper_arm → elbow (1) → forearm → hand [site_rh_tip]
        ├── L thigh (hip 3) → shin (knee 1) → foot (ankle 1) [site_lf_tip]
        └── R thigh (hip 3) → shin (knee 1) → foot (ankle 1) [site_rf_tip]

    Total non-free DOF: 1 (spine) + 4×2 (shoulders+elbow) + 5×2 (hip+knee+ankle)
                      = 19, plus the 6 free DOF = 25.
    """
    s = profile.segments
    m = profile.masses
    # Hand and foot half-extents (the rigid stubs at the end of each chain).
    hand_half = (0.04, 0.025, 0.06)
    foot_half = (0.05, 0.10, s.foot_h / 2.0)

    # Quick helpers — segment geoms are capsules along Z by default.
    def capsule(name: str, length: float, radius: float, mass: float, rgba: str) -> str:
        # Capsule oriented down (-Z) from the parent attach point so
        # the next child can sit at -Z * length.
        return (
            f'<geom name="{name}" type="capsule" '
            f'fromto="0 0 0  0 0 {-length:.4f}" '
            f'size="{radius:.4f}" mass="{mass:.3f}" rgba="{rgba}" '
            f'friction="1.0 0.005 0.001"/>'
        )

    pelvis_x, pelvis_y, pelvis_z = start_pos_m
    sw = s.shoulder_width / 2.0
    pw = s.pelvis_width / 2.0
    # Joint range strings (radians).
    R = lambda key: (
        f'{cfg.JOINT_LIMITS_RAD[key][0]:.4f} {cfg.JOINT_LIMITS_RAD[key][1]:.4f}'
    )

    skin = "0.86 0.72 0.55 1.0"
    cloth = "0.20 0.40 0.70 1.0"

    return f"""
    <body name="pelvis" pos="{pelvis_x:.4f} {pelvis_y:.4f} {pelvis_z:.4f}">
        <freejoint name="root"/>
        <geom name="g_pelvis" type="box"
              size="{pw:.4f} 0.08 {0.08:.4f}"
              mass="{m.pelvis:.3f}" rgba="{cloth}"
              friction="1.0 0.005 0.001"/>
        <site name="site_com" pos="0 0 0" size="0.02" rgba="1 1 0 0.4"/>

        <body name="chest" pos="0 0 {0.10:.4f}">
            <joint name="spine_lean" type="hinge" axis="1 0 0" range="{R('spine_lean')}"/>
            <geom name="g_chest" type="box"
                  size="{sw*0.9:.4f} 0.10 {s.spine/2:.4f}"
                  pos="0 0 {s.spine/2:.4f}"
                  mass="{m.chest:.3f}" rgba="{cloth}"
                  friction="1.0 0.005 0.001"/>

            <body name="head" pos="0 0 {s.spine + s.head/2:.4f}">
                <geom name="g_head" type="sphere" size="{s.head/2:.4f}"
                      mass="{m.head:.3f}" rgba="{skin}"/>
            </body>

            <!-- LEFT ARM -->
            <body name="l_upperarm" pos="{-sw:.4f} 0 {s.spine - 0.05:.4f}">
                <joint name="l_shoulder_az"   type="hinge" axis="0 1 0" range="{R('shoulder_az')}"/>
                <joint name="l_shoulder_el"   type="hinge" axis="1 0 0" range="{R('shoulder_el')}"/>
                <joint name="l_shoulder_roll" type="hinge" axis="0 0 1" range="{R('shoulder_roll')}"/>
                {capsule("g_l_upperarm", s.upper_arm, 0.045, m.upper_arm, skin)}
                <body name="l_forearm" pos="0 0 {-s.upper_arm:.4f}">
                    <joint name="l_elbow" type="hinge" axis="1 0 0" range="{R('elbow')}"/>
                    {capsule("g_l_forearm", s.forearm, 0.038, m.forearm, skin)}
                    <body name="l_hand" pos="0 0 {-s.forearm:.4f}">
                        <geom name="g_l_hand" type="box"
                              size="{hand_half[0]:.4f} {hand_half[1]:.4f} {hand_half[2]:.4f}"
                              pos="0 0 {-hand_half[2]:.4f}"
                              mass="{m.hand:.3f}" rgba="{skin}"
                              friction="1.5 0.005 0.001"/>
                        <site name="{LIMB_TIP_SITE['LH']}"
                              pos="0 0 {-hand_half[2]*2:.4f}" size="0.015"
                              rgba="0 1 0 0.6"/>
                    </body>
                </body>
            </body>

            <!-- RIGHT ARM -->
            <body name="r_upperarm" pos="{sw:.4f} 0 {s.spine - 0.05:.4f}">
                <joint name="r_shoulder_az"   type="hinge" axis="0 1 0" range="{R('shoulder_az')}"/>
                <joint name="r_shoulder_el"   type="hinge" axis="1 0 0" range="{R('shoulder_el')}"/>
                <joint name="r_shoulder_roll" type="hinge" axis="0 0 1" range="{R('shoulder_roll')}"/>
                {capsule("g_r_upperarm", s.upper_arm, 0.045, m.upper_arm, skin)}
                <body name="r_forearm" pos="0 0 {-s.upper_arm:.4f}">
                    <joint name="r_elbow" type="hinge" axis="1 0 0" range="{R('elbow')}"/>
                    {capsule("g_r_forearm", s.forearm, 0.038, m.forearm, skin)}
                    <body name="r_hand" pos="0 0 {-s.forearm:.4f}">
                        <geom name="g_r_hand" type="box"
                              size="{hand_half[0]:.4f} {hand_half[1]:.4f} {hand_half[2]:.4f}"
                              pos="0 0 {-hand_half[2]:.4f}"
                              mass="{m.hand:.3f}" rgba="{skin}"
                              friction="1.5 0.005 0.001"/>
                        <site name="{LIMB_TIP_SITE['RH']}"
                              pos="0 0 {-hand_half[2]*2:.4f}" size="0.015"
                              rgba="0 1 0 0.6"/>
                    </body>
                </body>
            </body>
        </body>

        <!-- LEFT LEG -->
        <body name="l_thigh" pos="{-pw:.4f} 0 -0.05">
            <joint name="l_hip_flex"   type="hinge" axis="1 0 0" range="{R('hip_flex')}"/>
            <joint name="l_hip_abduct" type="hinge" axis="0 1 0" range="{R('hip_abduct')}"/>
            <joint name="l_hip_rot"    type="hinge" axis="0 0 1" range="{R('hip_rot')}"/>
            {capsule("g_l_thigh", s.thigh, 0.07, m.thigh, cloth)}
            <body name="l_shin" pos="0 0 {-s.thigh:.4f}">
                <joint name="l_knee" type="hinge" axis="1 0 0" range="{R('knee')}"/>
                {capsule("g_l_shin", s.shin, 0.05, m.shin, skin)}
                <body name="l_foot" pos="0 0 {-s.shin - foot_half[2]:.4f}">
                    <joint name="l_ankle" type="hinge" axis="1 0 0" range="{R('ankle')}"/>
                    <geom name="g_l_foot" type="box"
                          size="{foot_half[0]:.4f} {foot_half[1]:.4f} {foot_half[2]:.4f}"
                          mass="{m.foot:.3f}" rgba="0.10 0.10 0.10 1"
                          friction="1.5 0.005 0.001"/>
                    <site name="{LIMB_TIP_SITE['LF']}"
                          pos="0 {foot_half[1]*0.6:.4f} {-foot_half[2]:.4f}"
                          size="0.015" rgba="0 1 0 0.6"/>
                </body>
            </body>
        </body>

        <!-- RIGHT LEG -->
        <body name="r_thigh" pos="{pw:.4f} 0 -0.05">
            <joint name="r_hip_flex"   type="hinge" axis="1 0 0" range="{R('hip_flex')}"/>
            <joint name="r_hip_abduct" type="hinge" axis="0 1 0" range="{R('hip_abduct')}"/>
            <joint name="r_hip_rot"    type="hinge" axis="0 0 1" range="{R('hip_rot')}"/>
            {capsule("g_r_thigh", s.thigh, 0.07, m.thigh, cloth)}
            <body name="r_shin" pos="0 0 {-s.thigh:.4f}">
                <joint name="r_knee" type="hinge" axis="1 0 0" range="{R('knee')}"/>
                {capsule("g_r_shin", s.shin, 0.05, m.shin, skin)}
                <body name="r_foot" pos="0 0 {-s.shin - foot_half[2]:.4f}">
                    <joint name="r_ankle" type="hinge" axis="1 0 0" range="{R('ankle')}"/>
                    <geom name="g_r_foot" type="box"
                          size="{foot_half[0]:.4f} {foot_half[1]:.4f} {foot_half[2]:.4f}"
                          mass="{m.foot:.3f}" rgba="0.10 0.10 0.10 1"
                          friction="1.5 0.005 0.001"/>
                    <site name="{LIMB_TIP_SITE['RF']}"
                          pos="0 {foot_half[1]*0.6:.4f} {-foot_half[2]:.4f}"
                          size="0.015" rgba="0 1 0 0.6"/>
                </body>
            </body>
        </body>
    </body>
    """


# ─── Wall + holds XML ─────────────────────────────────────────────────────
def _build_wall_xml(wall: Wall) -> tuple[str, list[dict]]:
    """Emit the wall plate + per-hold geoms as a worldbody fragment.

    Returns (xml, hold_meta) where hold_meta is a list of dicts the
    world layer uses to look up hold positions/normals at attach time.
    """
    width_m = wall.width_cm / 100.0
    height_m = wall.height_cm / 100.0
    pad = cfg.WALL_PADDING_M
    plate_w = width_m + 2 * pad
    plate_h = height_m + 2 * pad
    angle_deg = wall.wall_angle_deg
    theta = math.radians(angle_deg)

    # ── Coordinate convention for the wall plate ─────────────────────
    # Climber lives at +Y looking toward -Y. Wall bottom anchored at z=0.
    # We orient the plate body so:
    #     plate-local X  =  along the wall (horizontal)
    #     plate-local Y  =  outward normal (where holds protrude)
    #     plate-local Z  =  up the wall plane
    #
    # Rotating the plate by +θ around world +X tips the top of the wall
    # toward +Y for positive θ (overhang) and toward -Y for negative θ
    # (slab). Plate centre therefore sits at:
    #     (0,  +h/2 sinθ,  +h/2 cosθ)
    # so the bottom edge stays at world Z = 0.
    cx = 0.0
    cy = (height_m / 2.0) * math.sin(theta)
    cz = (height_m / 2.0) * math.cos(theta)
    plate_axisangle = f"1 0 0 {theta:.5f}"

    nx, ny, nz = _wall_normal_world(angle_deg)
    fric = cfg.DEFAULT_WALL_FRICTION

    # Plate half-sizes (X=along-wall, Y=thickness, Z=up-the-wall).
    plate_half = (plate_w / 2.0, cfg.WALL_THICKNESS_M / 2.0, plate_h / 2.0)
    plate_xml = f"""
    <body name="wall_plate" pos="{cx:.5f} {cy:.5f} {cz:.5f}" axisangle="{plate_axisangle}">
        <geom name="g_wall" type="box"
              size="{plate_half[0]:.4f} {plate_half[1]:.4f} {plate_half[2]:.4f}"
              rgba="0.85 0.82 0.75 1"
              friction="{fric[0]} {fric[1]} {fric[2]}"
              contype="1" conaffinity="1"/>
    </body>
    """

    # ── Holds — children of the wall plate. ──────────────────────────
    # In plate-local coords:
    #     local_x = wx_cm/100 - width/2          (centred on plate)
    #     local_y = thickness/2 + protrude/2     (sticking outward)
    #     local_z = wy_cm/100 - height/2         (up the wall)
    # Cylinder geom default-axis is +Z; rotate by -90° around X so the
    # cylinder's long axis points along plate-local +Y (out of the wall).
    cyl_axisangle = "1 0 0 -1.5708"
    hold_meta: list[dict] = []
    hold_geoms = []
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    for h in wall.holds:
        wx_cm = h.x_cm
        wy_cm = h.y_cm
        local_x = (wx_cm / 100.0) - width_m / 2.0
        local_z_plane = (wy_cm / 100.0) - height_m / 2.0
        local_y_out = plate_half[1] + cfg.HOLD_PROTRUDE_M / 2.0
        radius = cfg.HOLD_RADIUS_BY_SIZE_M.get(h.size, 0.05)

        # Start / finish ring — flat disc embedded just outside the
        # plate, large enough to be visible from the front.
        marker_rgba = None
        if h.is_finish:
            marker_rgba = "1.0 0.20 0.20 1.0"
        elif h.is_start:
            marker_rgba = "0.20 1.00 0.20 1.0"
        marker_xml = ""
        if marker_rgba is not None:
            marker_xml = (
                f'<geom name="ring_{h.hold_id}" type="cylinder" '
                f'pos="{local_x:.4f} {plate_half[1] + 0.001:.4f} {local_z_plane:.4f}" '
                f'size="{radius * 1.5:.4f} 0.005" '
                f'axisangle="{cyl_axisangle}" '
                f'rgba="{marker_rgba}" contype="0" conaffinity="0"/>'
            )

        # The hold itself. contype=2/conaffinity=2 means it only
        # collides with other contype=2 geoms (none) — the body doesn't
        # bounce off; grabbing is mediated by an equality constraint.
        hold_geoms.append(marker_xml)
        hold_geoms.append(
            f'<geom name="hold_{h.hold_id}" type="cylinder" '
            f'pos="{local_x:.4f} {local_y_out:.4f} {local_z_plane:.4f}" '
            f'size="{radius:.4f} {cfg.HOLD_PROTRUDE_M/2:.4f}" '
            f'axisangle="{cyl_axisangle}" '
            f'rgba="{_rgba_from_hex(h.color, 1.0)}" '
            f'friction="{h.friction:.3f} 0.005 0.001" '
            f'contype="2" conaffinity="2"/>'
        )

        # World position of the hold tip (centre of the cylinder face
        # facing the climber). Plate axes in world:
        #     plate_x_world = (1, 0, 0)
        #     plate_y_world = (0, cosθ, -sinθ)   ← outward normal
        #     plate_z_world = (0, sinθ,  cosθ)   ← up the wall
        # so for plate-local (lx, ly, lz):
        local_y_tip = local_y_out + cfg.HOLD_PROTRUDE_M / 2.0  # outer face
        world_x = cx + local_x
        world_y = cy + local_y_tip * cos_t + local_z_plane * sin_t
        world_z = cz - local_y_tip * sin_t + local_z_plane * cos_t

        hold_meta.append({
            "hold_id": h.hold_id,
            "world_pos": (world_x, world_y, world_z),
            "wall_normal": (nx, ny, nz),
            "geom_name": f"hold_{h.hold_id}",
            "friction": h.friction,
            "positivity": h.positivity,
            "max_force_n": h.max_force_n,
            "is_start": h.is_start,
            "is_finish": h.is_finish,
            "is_foothold_only": (h.hold_type == "foothold"),
        })

    return plate_xml + "\n".join(hold_geoms), hold_meta


# ─── Mocap targets + equality constraints ─────────────────────────────────
def _build_mocap_targets() -> str:
    """One mocap body per limb. Mocap bodies are kinematic — physics
    doesn't move them, but you can teleport them every step by writing
    `data.mocap_pos[i] = ...`. Each mocap is the "anchor" the limb tip
    welds to when on a hold."""
    return "".join(
        f'<body name="{LIMB_MOCAP_BODY[l]}" mocap="true" pos="0 0 -10">'
        f'<geom type="sphere" size="0.01" rgba="1 1 0 0.0" '
        f'contype="0" conaffinity="0"/>'
        f'</body>'
        for l in ("LH", "RH", "LF", "RF")
    )


def _build_equalities() -> str:
    """Connect equality per limb, initially inactive. The world layer
    flips `data.eq_active[i]` to attach/detach without recompiling the
    model."""
    parts = []
    for l in ("LH", "RH", "LF", "RF"):
        parts.append(
            f'<connect name="{LIMB_EQUALITY[l]}" '
            f'body1="{LIMB_MOCAP_BODY[l].replace("mocap_", "")}_anchor_dummy" '
            f'body2="{LIMB_MOCAP_BODY[l]}" anchor="0 0 0" active="false"/>'
        )
    return ""  # Replaced below — see _build_equalities_v2.


def _build_equalities_v2() -> str:
    """Use weld equalities between the mocap and the limb tip's parent
    body. weld locks both position and orientation; for climbing we
    actually want connect (position-only), but connect's `body2` must
    refer to a body, not a site.

    We use weld with relpose chosen at runtime so the limb tip lands on
    the mocap regardless of orientation. Weld's `relpose` param can be
    set in `data.eq_data` after compile, so we keep `relpose` empty here
    and update it on attach.

    Bodies welded:
        mocap_lh  ↔  l_hand
        mocap_rh  ↔  r_hand
        mocap_lf  ↔  l_foot
        mocap_rf  ↔  r_foot
    """
    pairs = [
        ("LH", "l_hand"),
        ("RH", "r_hand"),
        ("LF", "l_foot"),
        ("RF", "r_foot"),
    ]
    parts = []
    for limb, body_name in pairs:
        parts.append(
            f'<weld name="{LIMB_EQUALITY[limb]}" '
            f'body1="{LIMB_MOCAP_BODY[limb]}" body2="{body_name}" '
            f'active="false" '
            f'solref="{cfg.HOLD_CONSTRAINT_SOLREF[0]} {cfg.HOLD_CONSTRAINT_SOLREF[1]}"/>'
        )
    return "\n".join(parts)


# ─── Actuators ────────────────────────────────────────────────────────────
def _build_actuators() -> str:
    """Position actuators on every non-free joint. ctrl is a target
    angle (radians) the joint servos toward."""
    joints = [
        ("spine_lean", "spine"),
        ("l_shoulder_az", "shoulder"),
        ("l_shoulder_el", "shoulder"),
        ("l_shoulder_roll", "shoulder"),
        ("l_elbow", "elbow"),
        ("r_shoulder_az", "shoulder"),
        ("r_shoulder_el", "shoulder"),
        ("r_shoulder_roll", "shoulder"),
        ("r_elbow", "elbow"),
        ("l_hip_flex", "hip"),
        ("l_hip_abduct", "hip"),
        ("l_hip_rot", "hip"),
        ("l_knee", "knee"),
        ("l_ankle", "ankle"),
        ("r_hip_flex", "hip"),
        ("r_hip_abduct", "hip"),
        ("r_hip_rot", "hip"),
        ("r_knee", "knee"),
        ("r_ankle", "ankle"),
    ]
    parts = []
    for joint, group in joints:
        cap = cfg.TORQUE_CAP_NM[group]
        parts.append(
            f'<position name="act_{joint}" joint="{joint}" '
            f'kp="{cfg.ACTUATOR_KP}" '
            f'forcerange="{-cap} {cap}"/>'
        )
    return "\n".join(parts)


# ─── Top-level builder ────────────────────────────────────────────────────
def build_mjcf_xml(wall: Wall, profile: ClimberProfile | None = None) -> tuple[str, list[dict]]:
    """Return (xml_string, hold_meta) for the given wall + climber.

    The caller passes the result to `mujoco.MjModel.from_xml_string`.
    `hold_meta` is a list of dicts the world layer uses to attach
    limbs to holds.
    """
    profile = profile or ClimberProfile()
    s = profile.segments

    # Place the climber so feet hover ~0.5 m off the ground, body
    # centred on the wall, hands at start-hold height. The world layer
    # will overwrite this with `seed_pose()` immediately.
    start_pos = (0.0, 0.5, s.standing_leg + 0.10)

    climber_xml = _build_climber_xml(profile, start_pos)
    wall_xml, hold_meta = _build_wall_xml(wall)
    mocap_xml = _build_mocap_targets()
    eq_xml = _build_equalities_v2()
    act_xml = _build_actuators()

    # Floor geom — a large ground plane below the climber. Acts as a
    # safety net during early development; real climbers fall onto a
    # crash pad, not infinity.
    floor_xml = (
        f'<geom name="floor" type="plane" size="20 20 0.1" '
        f'pos="0 0 {-cfg.FLOOR_DROP_M}" '
        f'rgba="0.30 0.30 0.30 1" '
        f'friction="1.0 0.005 0.001" contype="1" conaffinity="1"/>'
    )

    xml = f"""<?xml version="1.0" ?>
<mujoco model="climbing-{escape(wall.wall_id)}">
  <compiler angle="radian" coordinate="local" autolimits="true"/>
  <option timestep="{cfg.PHYS_DT}" gravity="0 0 {-cfg.GRAVITY_M_S2}"
          integrator="implicitfast" cone="elliptic" iterations="50"/>

  <visual>
    <headlight diffuse="0.7 0.7 0.7" ambient="0.4 0.4 0.4" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <quality shadowsize="2048"/>
    <map fogstart="3" fogend="20" force="0.1" znear="0.05"/>
  </visual>

  <default>
    <joint armature="0.01" damping="0.5"/>
    <geom condim="3" margin="0.001"/>
  </default>

  <worldbody>
    <light name="top" pos="0 -3 6" dir="0 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <light name="side" pos="-3 -2 4" dir="0.5 0.3 -0.7" diffuse="0.4 0.4 0.4"/>
    {floor_xml}
    {wall_xml}
    {mocap_xml}
    {climber_xml}
  </worldbody>

  <equality>
    {eq_xml}
  </equality>

  <actuator>
    {act_xml}
  </actuator>

  <sensor>
    <subtreecom name="com" body="pelvis"/>
    <subtreecom name="com_chest" body="chest"/>
  </sensor>
</mujoco>
"""
    return xml, hold_meta
