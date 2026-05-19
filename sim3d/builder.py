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
    LIMB_TIP_BODY,
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

    Tree layout — DOF count in parens:

        pelvis (free, 6)
        ├── chest (spine_lean, 1)
        │   ├── head
        │   ├── L shoulder (3) → upper_arm → elbow (1) → forearm → wrist (1) → hand
        │   └── R shoulder (3) → upper_arm → elbow (1) → forearm → wrist (1) → hand
        ├── L thigh (hip 3) → shin (knee 1) → foot (ankle 1)
        └── R thigh (hip 3) → shin (knee 1) → foot (ankle 1)

    Total non-free DOF: 1 + 5×2 + 5×2 = 21, plus 6 free = 27.

    Collision groups:
        contype=1, conaffinity=1 — torso/limbs
        Wall plate is also contype=1, so the climber CAN press against
        the wall (essential for slab and overhang). Holds are contype=2
        and don't collide with anyone — grabbing is mediated by the
        weld equality, not contact.
    """
    s = profile.segments
    m = profile.masses
    hand_half = (0.045, 0.025, 0.0975)  # width=9cm, depth=5cm, length=19.5cm
    foot_half = (0.045, 0.13, max(0.025, s.foot_h / 2.0))  # width=9cm, length=26cm

    R = lambda key: (
        f'{cfg.JOINT_LIMITS_RAD[key][0]:.4f} {cfg.JOINT_LIMITS_RAD[key][1]:.4f}'
    )
    P = lambda key: (
        f'stiffness="{cfg.JOINT_PASSIVE[key][0]:.2f}" '
        f'damping="{cfg.JOINT_PASSIVE[key][1]:.2f}"'
    )

    def capsule(name: str, length: float, radius: float, mass: float, rgba: str) -> str:
        # Oriented down (-Z) from the parent so the next child sits at z = -length.
        return (
            f'<geom name="{name}" type="capsule" '
            f'fromto="0 0 0  0 0 {-length:.4f}" '
            f'size="{radius:.4f}" mass="{mass:.3f}" rgba="{rgba}" '
            f'friction="1.0 0.005 0.001"/>'
        )

    pelvis_x, pelvis_y, pelvis_z = start_pos_m
    sw = s.shoulder_width / 2.0
    pw = s.pelvis_width / 2.0
    skin = "0.86 0.72 0.55 1.0"
    cloth = "0.20 0.40 0.70 1.0"
    hand_fric = " ".join(f"{v:.3f}" for v in cfg.LIMB_TIP_FRICTION["hand"])
    foot_fric = " ".join(f"{v:.3f}" for v in cfg.LIMB_TIP_FRICTION["foot"])

    # The hand_z offset puts the hand body's site at the lower face of
    # the hand box (where a real climber's fingers would wrap). Mocap
    # attaches there, so the constraint puts that point at the hold.
    return f"""
    <body name="pelvis" pos="{pelvis_x:.4f} {pelvis_y:.4f} {pelvis_z:.4f}">
        <freejoint name="root"/>
        <geom name="g_pelvis" type="ellipsoid"
              size="{pw:.4f} 0.115 {0.09:.4f}"
              mass="{m.pelvis:.3f}" rgba="{cloth}"
              friction="1.0 0.005 0.001"/>
        <site name="site_com" pos="0 0 0" size="0.02" rgba="1 1 0 0.4"/>

        <body name="chest" pos="0 0 {0.12:.4f}">
            <joint name="spine_lean" type="hinge" axis="1 0 0" class="j_spine"
                   range="{R('spine_lean')}" {P('spine_lean')}/>
            <geom name="g_chest" type="ellipsoid"
                  size="{sw*0.9:.4f} 0.115 {s.spine/2:.4f}"
                  pos="0 0 {s.spine/2:.4f}"
                  mass="{m.chest:.3f}" rgba="{cloth}"
                  friction="1.0 0.005 0.001"/>
            <geom name="g_neck" type="cylinder"
                  fromto="0 0 {s.spine:.4f}  0 0 {s.spine + s.head/2 - s.head_radius:.4f}"
                  size="{cfg.NECK_RADIUS_M:.4f}" mass="0.3" rgba="{skin}"
                  friction="1.0 0.005 0.001"/>

            <body name="head" pos="0 0 {s.spine + s.head/2:.4f}">
                <geom name="g_head" type="sphere" size="{s.head_radius:.4f}"
                      mass="{m.head:.3f}" rgba="{skin}"/>
            </body>

            <!-- LEFT ARM -->
            <body name="l_upperarm" pos="{-sw:.4f} 0 {s.spine - 0.05:.4f}">
                <joint name="l_shoulder_az"   type="hinge" axis="0 1 0" class="j_shoulder"
                       range="{R('shoulder_az')}"   {P('shoulder_az')}/>
                <joint name="l_shoulder_el"   type="hinge" axis="1 0 0" class="j_shoulder"
                       range="{R('shoulder_el')}"   {P('shoulder_el')}/>
                <joint name="l_shoulder_roll" type="hinge" axis="0 0 1" class="j_shoulder"
                       range="{R('shoulder_roll')}" {P('shoulder_roll')}/>
                {capsule("g_l_upperarm", s.upper_arm, 0.051, m.upper_arm, skin)}
                <body name="l_forearm" pos="0 0 {-s.upper_arm:.4f}">
                    <joint name="l_elbow" type="hinge" axis="1 0 0" class="j_elbow"
                           range="{R('elbow')}" {P('elbow')}/>
                    {capsule("g_l_forearm", s.forearm, 0.046, m.forearm, skin)}
                    <body name="l_hand" pos="0 0 {-s.forearm:.4f}">
                        <joint name="l_wrist" type="hinge" axis="1 0 0" class="j_wrist"
                               range="{R('wrist')}" {P('wrist')}/>
                        <geom name="g_l_hand" type="ellipsoid"
                              size="{hand_half[0]:.4f} {hand_half[1]:.4f} {hand_half[2]:.4f}"
                              pos="0 0 {-hand_half[2]:.4f}"
                              mass="{m.hand:.3f}" rgba="{skin}"
                              friction="{hand_fric}"/>
                        <site name="{LIMB_TIP_SITE['LH']}"
                              pos="0 0 {-hand_half[2]*2:.4f}" size="0.015"
                              rgba="0 1 0 0.6"/>
                        <body name="{LIMB_TIP_BODY['LH']}"
                              pos="0 0 {-hand_half[2]*2:.4f}"/>
                    </body>
                </body>
            </body>

            <!-- RIGHT ARM -->
            <body name="r_upperarm" pos="{sw:.4f} 0 {s.spine - 0.05:.4f}">
                <joint name="r_shoulder_az"   type="hinge" axis="0 1 0" class="j_shoulder"
                       range="{R('shoulder_az')}"   {P('shoulder_az')}/>
                <joint name="r_shoulder_el"   type="hinge" axis="1 0 0" class="j_shoulder"
                       range="{R('shoulder_el')}"   {P('shoulder_el')}/>
                <joint name="r_shoulder_roll" type="hinge" axis="0 0 1" class="j_shoulder"
                       range="{R('shoulder_roll')}" {P('shoulder_roll')}/>
                {capsule("g_r_upperarm", s.upper_arm, 0.051, m.upper_arm, skin)}
                <body name="r_forearm" pos="0 0 {-s.upper_arm:.4f}">
                    <joint name="r_elbow" type="hinge" axis="1 0 0" class="j_elbow"
                           range="{R('elbow')}" {P('elbow')}/>
                    {capsule("g_r_forearm", s.forearm, 0.046, m.forearm, skin)}
                    <body name="r_hand" pos="0 0 {-s.forearm:.4f}">
                        <joint name="r_wrist" type="hinge" axis="1 0 0" class="j_wrist"
                               range="{R('wrist')}" {P('wrist')}/>
                        <geom name="g_r_hand" type="ellipsoid"
                              size="{hand_half[0]:.4f} {hand_half[1]:.4f} {hand_half[2]:.4f}"
                              pos="0 0 {-hand_half[2]:.4f}"
                              mass="{m.hand:.3f}" rgba="{skin}"
                              friction="{hand_fric}"/>
                        <site name="{LIMB_TIP_SITE['RH']}"
                              pos="0 0 {-hand_half[2]*2:.4f}" size="0.015"
                              rgba="0 1 0 0.6"/>
                        <body name="{LIMB_TIP_BODY['RH']}"
                              pos="0 0 {-hand_half[2]*2:.4f}"/>
                    </body>
                </body>
            </body>
        </body>

        <!-- LEFT LEG -->
        <body name="l_thigh" pos="{-pw:.4f} 0 {-0.06:.4f}">
            <joint name="l_hip_flex"   type="hinge" axis="1 0 0" class="j_hip"
                   range="{R('hip_flex')}"   {P('hip_flex')}/>
            <joint name="l_hip_abduct" type="hinge" axis="0 1 0" class="j_hip"
                   range="{R('hip_abduct')}" {P('hip_abduct')}/>
            <joint name="l_hip_rot"    type="hinge" axis="0 0 1" class="j_hip"
                   range="{R('hip_rot')}"    {P('hip_rot')}/>
            {capsule("g_l_thigh", s.thigh, 0.086, m.thigh, cloth)}
            <body name="l_shin" pos="0 0 {-s.thigh:.4f}">
                <joint name="l_knee" type="hinge" axis="1 0 0" class="j_knee"
                       range="{R('knee')}" {P('knee')}/>
                {capsule("g_l_shin", s.shin, 0.056, m.shin, skin)}
                <body name="l_foot" pos="0 0 {-s.shin - foot_half[2]:.4f}">
                    <joint name="l_ankle" type="hinge" axis="1 0 0" class="j_ankle"
                           range="{R('ankle')}" {P('ankle')}/>
                    <geom name="g_l_foot" type="ellipsoid"
                          size="{foot_half[0]:.4f} {foot_half[1]:.4f} {foot_half[2]:.4f}"
                          mass="{m.foot:.3f}" rgba="0.10 0.10 0.10 1"
                          friction="{foot_fric}"/>
                    <site name="{LIMB_TIP_SITE['LF']}"
                          pos="0 {foot_half[1]*0.6:.4f} {-foot_half[2]:.4f}"
                          size="0.015" rgba="0 1 0 0.6"/>
                    <body name="{LIMB_TIP_BODY['LF']}"
                          pos="0 {foot_half[1]*0.6:.4f} {-foot_half[2]:.4f}"/>
                </body>
            </body>
        </body>

        <!-- RIGHT LEG -->
        <body name="r_thigh" pos="{pw:.4f} 0 {-0.06:.4f}">
            <joint name="r_hip_flex"   type="hinge" axis="1 0 0" class="j_hip"
                   range="{R('hip_flex')}"   {P('hip_flex')}/>
            <joint name="r_hip_abduct" type="hinge" axis="0 1 0" class="j_hip"
                   range="{R('hip_abduct')}" {P('hip_abduct')}/>
            <joint name="r_hip_rot"    type="hinge" axis="0 0 1" class="j_hip"
                   range="{R('hip_rot')}"    {P('hip_rot')}/>
            {capsule("g_r_thigh", s.thigh, 0.086, m.thigh, cloth)}
            <body name="r_shin" pos="0 0 {-s.thigh:.4f}">
                <joint name="r_knee" type="hinge" axis="1 0 0" class="j_knee"
                       range="{R('knee')}" {P('knee')}/>
                {capsule("g_r_shin", s.shin, 0.056, m.shin, skin)}
                <body name="r_foot" pos="0 0 {-s.shin - foot_half[2]:.4f}">
                    <joint name="r_ankle" type="hinge" axis="1 0 0" class="j_ankle"
                           range="{R('ankle')}" {P('ankle')}/>
                    <geom name="g_r_foot" type="ellipsoid"
                          size="{foot_half[0]:.4f} {foot_half[1]:.4f} {foot_half[2]:.4f}"
                          mass="{m.foot:.3f}" rgba="0.10 0.10 0.10 1"
                          friction="{foot_fric}"/>
                    <site name="{LIMB_TIP_SITE['RF']}"
                          pos="0 {foot_half[1]*0.6:.4f} {-foot_half[2]:.4f}"
                          size="0.015" rgba="0 1 0 0.6"/>
                    <body name="{LIMB_TIP_BODY['RF']}"
                          pos="0 {foot_half[1]*0.6:.4f} {-foot_half[2]:.4f}"/>
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
    plate_w = width_m + 2 * pad      # pad on both sides
    plate_h = height_m + pad         # pad above only — bottom anchors to z=0
    angle_deg = wall.wall_angle_deg
    theta = math.radians(angle_deg)

    # ── Coordinate convention for the wall plate ─────────────────────
    # Climber lives at +Y looking toward -Y. Wall bottom anchored at z=0.
    # We orient the plate body so:
    #     plate-local X  =  along the wall (horizontal)
    #     plate-local Y  =  outward normal (where holds protrude)
    #     plate-local Z  =  up the wall plane
    #
    # Plate centre is at the midpoint of the playable wall region —
    # NOT at plate_h/2 — because plate_h has padding only above. The
    # plate bottom (plate-local z = -plate_h/2) maps to wall z = 0 by
    # placing the centre at z_offset = (plate_h - pad)/2 in plate
    # coords. Working it out: bottom edge in world = centre + (-plate_h/2)
    # × plate_z_world. Plate_z_world for rotation +θ around +X is
    # (0, sinθ, cosθ). For bottom = (0, 0, 0):
    #     centre = (0, +(plate_h/2) sinθ, +(plate_h/2) cosθ)
    # Then the playable region (height_m) sits between bottom and
    # bottom + height_m × plate_z_world.
    cx = 0.0
    cy = (plate_h / 2.0) * math.sin(theta)
    cz = (plate_h / 2.0) * math.cos(theta)

    # Floor-clearance lift: at high overhang angles, the lowest holds
    # can compute to z < 0 (below the floor). Find the minimum hold
    # world-z under the current wall placement, and if it's below the
    # required clearance, lift the whole plate (and therefore every
    # hold position) by the deficit. This keeps the convention
    # "wall bottom at z=0 for vertical walls" while gracefully handling
    # extreme angles.
    if wall.holds:
        cos_t0 = math.cos(theta)
        sin_t0 = math.sin(theta)
        local_y_tip0 = cfg.WALL_THICKNESS_M / 2.0 + cfg.HOLD_PROTRUDE_M
        min_world_z = min(
            cz - local_y_tip0 * sin_t0
              + ((h.y_cm / 100.0) - plate_h / 2.0) * cos_t0
            for h in wall.holds
        )
        deficit = cfg.FLOOR_Z + cfg.HOLD_FLOOR_CLEARANCE - min_world_z
        if deficit > 0:
            cz += deficit

    plate_axisangle = f"1 0 0 {theta:.5f}"

    nx, ny, nz = _wall_normal_world(angle_deg)
    fric = cfg.DEFAULT_WALL_FRICTION

    # Plate half-sizes (X=along-wall, Y=thickness, Z=up-the-wall).
    plate_half = (plate_w / 2.0, cfg.WALL_THICKNESS_M / 2.0, plate_h / 2.0)
    plate_open_xml = f"""
    <body name="wall_plate" pos="{cx:.5f} {cy:.5f} {cz:.5f}" axisangle="{plate_axisangle}">
        <geom name="g_wall" type="box"
              size="{plate_half[0]:.4f} {plate_half[1]:.4f} {plate_half[2]:.4f}"
              rgba="0.85 0.82 0.75 1"
              friction="{fric[0]} {fric[1]} {fric[2]}"
              contype="1" conaffinity="1"/>
    """

    # ── Holds — children of the wall plate. ──────────────────────────
    # In plate-local coords:
    #     local_x = wx_cm/100 - width/2          (centred horizontally)
    #     local_y = thickness/2 + protrude/2     (sticking outward)
    #     local_z = wy_cm/100 - plate_h/2        (up the wall)
    #
    # Note plate_h not height_m — the plate centre is offset by half
    # the *padded* height because we only pad above. A hold at
    # wy_cm = 0 sits at plate-local z = -plate_h/2 = wall bottom.
    # Cylinder geom default-axis is +Z; rotate by -90° around X so its
    # long axis points along plate-local +Y (out of the wall).
    cyl_axisangle = "1 0 0 -1.5708"
    hold_meta: list[dict] = []
    hold_geoms = []
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    for h in wall.holds:
        wx_cm = h.x_cm
        wy_cm = h.y_cm
        local_x = (wx_cm / 100.0) - width_m / 2.0
        local_z_plane = (wy_cm / 100.0) - plate_h / 2.0
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
            "hold_type": h.hold_type,
            "color": h.color,
            "is_foothold_only": (h.hold_type == "foothold"),
        })

    # Hold geoms must be children of the rotated wall body. If they are
    # emitted as worldbody siblings, MuJoCo interprets their positions in
    # world coordinates, so the renderer shows them floating in the wrong
    # place even though hold_meta still points limb constraints at the
    # intended wall-space locations.
    wall_xml = plate_open_xml + "\n".join(hold_geoms) + "\n    </body>"
    return wall_xml, hold_meta


# ─── Kickboard ────────────────────────────────────────────────────────────
def _build_kickboard_xml(
    wall: Wall,
) -> tuple[str, list[dict]]:
    """Emit a near-vertical kickboard plate below/in front of the main wall.

    Returns (xml, foothold_meta). Each foothold is synthesized as a real hold
    (contype=2 cylinder) so the env's grip machinery treats it identically to
    a wall hold. IDs are prefixed ``kb_`` and ``hold_type`` is set to
    ``foothold`` so hands cannot use them.
    """
    width = cfg.KICKBOARD_WIDTH_M
    height = cfg.KICKBOARD_HEIGHT_M
    thickness = cfg.KICKBOARD_THICKNESS_M
    angle = math.radians(cfg.KICKBOARD_ANGLE_DEG)

    # Position the kickboard centred at x=0, sitting on the floor in front of
    # the main wall surface. We push it forward in +Y so the plate's front
    # face is slightly in front of the main wall's lowest hold protrude depth.
    plate_cx = 0.0
    # Main wall bottom sits at z=0; kickboard rises from z=0 to height.
    plate_cz = height / 2.0
    # In front of the main wall: a small positive Y so the upper edge of the
    # kickboard sits clear of the main plate's near face.
    plate_cy = 0.35

    # Slight backward tilt (top toward main wall) so feet press into it.
    plate_axisangle = f"1 0 0 {-angle:.5f}"
    plate_half = (width / 2.0, thickness / 2.0, height / 2.0)

    xml_parts = [
        f'<body name="kickboard" pos="{plate_cx:.5f} {plate_cy:.5f} {plate_cz:.5f}" '
        f'axisangle="{plate_axisangle}">',
        f'<geom name="g_kickboard" type="box" '
        f'size="{plate_half[0]:.4f} {plate_half[1]:.4f} {plate_half[2]:.4f}" '
        f'rgba="0.40 0.30 0.22 1" friction="1.0 0.005 0.001" '
        f'contype="1" conaffinity="1"/>',
    ]

    foothold_meta: list[dict] = []
    cyl_axisangle = "1 0 0 -1.5708"
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    nx, ny, nz = 0.0, cos_a, -sin_a
    for hid_suffix, lx, lz in cfg.KICKBOARD_FOOTHOLDS:
        local_x = lx
        local_z_plane = lz - height / 2.0  # plate-local z (plate origin at centre)
        local_y_out = plate_half[1] + cfg.HOLD_PROTRUDE_M / 2.0
        radius = cfg.HOLD_RADIUS_BY_SIZE_M["large"]
        xml_parts.append(
            f'<geom name="hold_kb_{hid_suffix}" type="cylinder" '
            f'pos="{local_x:.4f} {local_y_out:.4f} {local_z_plane:.4f}" '
            f'size="{radius:.4f} {cfg.HOLD_PROTRUDE_M/2:.4f}" '
            f'axisangle="{cyl_axisangle}" '
            f'rgba="0.85 0.65 0.20 1.0" '
            f'friction="1.4 0.01 0.005" contype="2" conaffinity="2"/>'
        )
        # World position of the hold tip.
        local_y_tip = local_y_out + cfg.HOLD_PROTRUDE_M / 2.0
        world_x = plate_cx + local_x
        world_y = plate_cy + local_y_tip * cos_a + local_z_plane * sin_a
        world_z = plate_cz - local_y_tip * sin_a + local_z_plane * cos_a
        hold_id = f"kb_{hid_suffix}"
        foothold_meta.append({
            "hold_id": hold_id,
            "world_pos": (world_x, world_y, world_z),
            "wall_normal": (nx, ny, nz),
            "geom_name": f"hold_kb_{hid_suffix}",
            "friction": 1.4,
            "positivity": 1.0,
            "max_force_n": None,
            "is_start": False,
            "is_finish": False,
            "hold_type": "foothold",
            "color": "#d8a020",
            "is_foothold_only": True,
        })
    xml_parts.append("</body>")
    return "\n".join(xml_parts), foothold_meta


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
        mocap_lh  ↔  tip_body_lh
        mocap_rh  ↔  tip_body_rh
        mocap_lf  ↔  tip_body_lf
        mocap_rf  ↔  tip_body_rf
    """
    pairs = [(limb, LIMB_TIP_BODY[limb]) for limb in ("LH", "RH", "LF", "RF")]
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
    """Position actuators on every non-free joint. `ctrl[i]` is a
    target angle (radians); the actuator servos the joint to that
    angle with a stiff PD plus the joint's passive stiffness/damping."""
    joints = [
        ("spine_lean", "spine"),
        ("l_shoulder_az",   "shoulder"),
        ("l_shoulder_el",   "shoulder"),
        ("l_shoulder_roll", "shoulder"),
        ("l_elbow",         "elbow"),
        ("l_wrist",         "wrist"),
        ("r_shoulder_az",   "shoulder"),
        ("r_shoulder_el",   "shoulder"),
        ("r_shoulder_roll", "shoulder"),
        ("r_elbow",         "elbow"),
        ("r_wrist",         "wrist"),
        ("l_hip_flex",   "hip"),
        ("l_hip_abduct", "hip"),
        ("l_hip_rot",    "hip"),
        ("l_knee",       "knee"),
        ("l_ankle",      "ankle"),
        ("r_hip_flex",   "hip"),
        ("r_hip_abduct", "hip"),
        ("r_hip_rot",    "hip"),
        ("r_knee",       "knee"),
        ("r_ankle",      "ankle"),
    ]
    parts = []
    for joint, group in joints:
        cap = cfg.TORQUE_CAP_NM[group]
        kp, kv = cfg.ACTUATOR_GAINS_BY_GROUP.get(
            group, (cfg.ACTUATOR_KP, cfg.ACTUATOR_KV),
        )
        parts.append(
            f'<position name="act_{joint}" joint="{joint}" '
            f'kp="{kp}" kv="{kv}" '
            f'forcerange="{-cap} {cap}"/>'
        )
    return "\n".join(parts)


# ─── Per-joint armature lookup ────────────────────────────────────────────
_JOINT_GROUP = {
    "spine_lean":    "spine",
    "l_shoulder_az": "shoulder", "r_shoulder_az": "shoulder",
    "l_shoulder_el": "shoulder", "r_shoulder_el": "shoulder",
    "l_shoulder_roll": "shoulder", "r_shoulder_roll": "shoulder",
    "l_elbow": "elbow", "r_elbow": "elbow",
    "l_wrist": "wrist", "r_wrist": "wrist",
    "l_hip_flex": "hip", "r_hip_flex": "hip",
    "l_hip_abduct": "hip", "r_hip_abduct": "hip",
    "l_hip_rot": "hip", "r_hip_rot": "hip",
    "l_knee": "knee", "r_knee": "knee",
    "l_ankle": "ankle", "r_ankle": "ankle",
}


def _armature_for_joint(joint_name: str) -> float:
    group = _JOINT_GROUP.get(joint_name)
    if group is None:
        return 0.01
    return cfg.JOINT_ARMATURE_BY_GROUP.get(group, 0.01)


# ─── Top-level builder ────────────────────────────────────────────────────
def build_mjcf_xml(
    wall: Wall,
    profile: ClimberProfile | None = None,
    *,
    include_kickboard: bool = False,
) -> tuple[str, list[dict]]:
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
    kickboard_xml = ""
    if include_kickboard:
        kickboard_xml, kb_meta = _build_kickboard_xml(wall)
        hold_meta = hold_meta + kb_meta
    mocap_xml = _build_mocap_targets()
    eq_xml = _build_equalities_v2()
    act_xml = _build_actuators()

    # Floor — its top surface sits at z = FLOOR_Z (= 0). The MJCF
    # plane geom occupies the half-space z <= surface and is infinite
    # in extent; we visualise a 40×40 m patch via the size attribute,
    # and use contype=1 so the climber's body can land on it after a
    # fall. Hold cylinders are contype=2 so they don't collide here.
    floor_xml = (
        f'<geom name="floor" type="plane" size="20 20 0.1" '
        f'pos="0 0 {cfg.FLOOR_Z}" '
        f'rgba="0.18 0.18 0.20 1" '
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
    <!-- Per-group joint armature. See cfg.JOINT_ARMATURE_BY_GROUP for why
         the load-bearing joints (hip, spine) get an order of magnitude
         more reflected inertia than the wrists/ankles. -->
    <default class="j_shoulder"><joint armature="{cfg.JOINT_ARMATURE_BY_GROUP['shoulder']}"/></default>
    <default class="j_elbow"   ><joint armature="{cfg.JOINT_ARMATURE_BY_GROUP['elbow']}"/></default>
    <default class="j_wrist"   ><joint armature="{cfg.JOINT_ARMATURE_BY_GROUP['wrist']}"/></default>
    <default class="j_spine"   ><joint armature="{cfg.JOINT_ARMATURE_BY_GROUP['spine']}"/></default>
    <default class="j_hip"     ><joint armature="{cfg.JOINT_ARMATURE_BY_GROUP['hip']}"/></default>
    <default class="j_knee"    ><joint armature="{cfg.JOINT_ARMATURE_BY_GROUP['knee']}"/></default>
    <default class="j_ankle"   ><joint armature="{cfg.JOINT_ARMATURE_BY_GROUP['ankle']}"/></default>
    <!-- Margin bumped from 0.001 → 0.005. A 1 mm margin is too thin for
         fast-moving body parts to engage contact against the wall plate
         before they penetrate; 5 mm gives the constraint solver enough
         warning to push the head/torso back out. -->
    <geom condim="3" margin="0.005"/>
  </default>

  <worldbody>
    <light name="top" pos="0 -3 6" dir="0 0.5 -1" diffuse="0.9 0.9 0.9"/>
    <light name="side" pos="-3 -2 4" dir="0.5 0.3 -0.7" diffuse="0.4 0.4 0.4"/>
    {floor_xml}
    {wall_xml}
    {kickboard_xml}
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
