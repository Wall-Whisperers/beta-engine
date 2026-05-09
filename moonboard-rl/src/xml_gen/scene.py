"""Assembles a complete MuJoCo scene XML for the MoonBoard RL environment.

Strategy for merging the humanoid XML with scene elements:
  1. Parse humanoid.xml with xml.etree.ElementTree.
  2. Extract sub-trees: compiler, default, option, size, visual, asset,
     actuator, sensor, contact.
  3. Patch hand/foot geom friction to high values so gripping is physically
     meaningful.
  4. Build a fresh <worldbody> containing:
       - A directional light.
       - The floor plane and tilted wall box (from wall.py).
       - Sphere geoms for each hold in the route (from holds.py).
       - A zero-size "grip_anchor" body used as the default body2 for all
         grip constraints before they are retargeted at runtime.
       - The humanoid's root <body> with pos and euler overridden.
  5. Add an <equality> section with 4 connect constraints, one per limb slot,
     all starting inactive.  GripManager retargets these at runtime by
     overwriting model.eq_obj2id and model.eq_data before activating.
  6. Serialise the merged tree to an XML string.

No <include> tags are used so the output is a single self-contained file.

Limb slot convention:
  0 = left hand  (body: left_lower_arm)
  1 = right hand (body: right_lower_arm)
  2 = left foot  (body: left_foot)
  3 = right foot (body: right_foot)
"""

import xml.etree.ElementTree as ET

from ..parsers.canonical import Route
from .holds import holds_xml
from .wall import wall_xml, kickboard_holds_xml

# Humanoid placement: in front of the wall, facing -Y (toward wall face).
_HUMANOID_POS = "0 1.0 1.4"   # z=1.4 keeps feet above floor (matches original)
_HUMANOID_EULER = "0 0 180"    # rotate 180° about Z to face -Y

# MuJoCo body names confirmed from humanoid.xml inspection.
LIMB_BODY_NAMES = [
    "left_lower_arm",   # slot 0 — left hand
    "right_lower_arm",  # slot 1 — right hand
    "left_foot",        # slot 2 — left foot
    "right_foot",       # slot 3 — right foot
]

# Sites injected at the true hand/foot tip positions (same slot order).
LIMB_SITE_NAMES = [
    "site_lhand",   # slot 0
    "site_rhand",   # slot 1
    "site_lfoot",   # slot 2
    "site_rfoot",   # slot 3
]

# Geom whose pos drives each site position (read from humanoid.xml at build time).
_LIMB_GEOM_NAMES = [
    "left_hand",   # in left_lower_arm  — pos ".18 -.18 .18"
    "right_hand",  # in right_lower_arm — pos ".18 .18 .18"
    "left_foot",   # in left_foot       — pos "0 0 0.1"
    "right_foot",  # in right_foot      — pos "0 0 0.1"
]

# Grip constraint names (in the same slot order).
GRIP_CONSTRAINT_NAMES = [
    "grip_lhand",
    "grip_rhand",
    "grip_lfoot",
    "grip_rfoot",
]

# Geom names to receive high-friction patching.
_HAND_FOOT_GEOM_NAMES = {"left_hand", "right_hand", "left_foot", "right_foot"}
_HIGH_FRICTION = "2.0 0.05 0.001"
_HIGH_CONDIM = "3"


def _indent(elem: ET.Element, level: int = 0, indent: str = "  ") -> None:
    """Add pretty-print whitespace to an ElementTree in place."""
    pad = "\n" + level * indent
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = pad + indent
        if not elem.tail or not elem.tail.strip():
            elem.tail = pad
        for child in elem:
            _indent(child, level + 1, indent)
        if not child.tail or not child.tail.strip():  # noqa: F821
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def _inject_limb_sites(hum_root: ET.Element) -> None:
    """Inject <site> elements at the true hand/foot tip positions in the humanoid tree.

    data.xpos[body_id] returns the body *origin*, which for left_lower_arm /
    right_lower_arm is the elbow joint — not the hand tip.  Sites placed at the
    hand/foot sphere geom centres let GripManager use data.site_xpos[site_id]
    for accurate proximity checks and anchor calculations.

    The site pos is read from the corresponding geom's pos attribute so it is
    guaranteed to match the visual hand/foot sphere even if the humanoid XML
    is updated in the future.

    Args:
        hum_root: Root element of the parsed humanoid XML tree (modified in place).
    """
    for body_name, geom_name, site_name in zip(
        LIMB_BODY_NAMES, _LIMB_GEOM_NAMES, LIMB_SITE_NAMES
    ):
        # Find the geom anywhere in the tree to read its pos.
        geom_pos = "0 0 0"
        for geom in hum_root.iter("geom"):
            if geom.get("name") == geom_name:
                geom_pos = geom.get("pos", "0 0 0")
                break

        # Find the body and inject the site as its first child.
        for body in hum_root.iter("body"):
            if body.get("name") == body_name:
                site = ET.Element("site")
                site.set("name", site_name)
                site.set("pos", geom_pos)
                site.set("size", "0.01")   # tiny — for debugging only
                body.insert(0, site)
                break


def _patch_hand_foot_friction(hum_root: ET.Element) -> None:
    """Set high friction + condim=3 on hand and foot geoms in the humanoid tree.

    The humanoid's <default> sets condim=1 (frictionless in tangential
    directions).  For climbing, hands and feet must resist tangential sliding,
    so we override the named geoms individually.

    Args:
        hum_root: Root element of the parsed humanoid XML tree (modified in place).
    """
    for geom in hum_root.iter("geom"):
        if geom.get("name") in _HAND_FOOT_GEOM_NAMES:
            geom.set("friction", _HIGH_FRICTION)
            geom.set("condim", _HIGH_CONDIM)


def build_scene_xml(route: Route, humanoid_xml_path: str) -> str:
    """Build a complete MuJoCo XML scene for a MoonBoard route.

    Includes the wall, hold spheres, grip_anchor body, 4 inactive connect
    equality constraints, and the humanoid positioned in front of the wall.

    Args:
        route: The climbing route whose holds will be rendered as spheres.
        humanoid_xml_path: Absolute path to humanoid.xml on disk.

    Returns:
        Complete MuJoCo XML string, ready to be passed to
        mujoco.MjModel.from_xml_string or written to a file.
    """
    # ── Parse humanoid XML ────────────────────────────────────────────────────
    hum_tree = ET.parse(humanoid_xml_path)
    hum_root = hum_tree.getroot()   # <mujoco model="humanoid">

    # Inject sites at hand/foot tip positions (must run before friction patch).
    _inject_limb_sites(hum_root)
    # Patch hand/foot friction before the tree is copied into the scene.
    _patch_hand_foot_friction(hum_root)

    def _get(tag: str) -> ET.Element | None:
        return hum_root.find(tag)

    # ── Build root element ────────────────────────────────────────────────────
    scene = ET.Element("mujoco")
    scene.set("model", "moonboard_scene")

    for tag in ("compiler", "option", "size", "visual", "default"):
        elem = _get(tag)
        if elem is not None:
            scene.append(elem)

    asset = _get("asset")
    scene.append(asset if asset is not None else ET.SubElement(scene, "asset"))

    # ── Worldbody ─────────────────────────────────────────────────────────────
    worldbody = ET.SubElement(scene, "worldbody")

    # Main light: from above-behind, casts wall shadows.
    light = ET.SubElement(worldbody, "light")
    light.set("name", "main_light")
    light.set("diffuse", ".8 .8 .8")
    light.set("pos", "0 -1 5")
    light.set("dir", "0 0.2 -1")
    light.set("directional", "true")
    # Fill light: from in front of the wall so the climbing face is visible.
    fill = ET.SubElement(worldbody, "light")
    fill.set("name", "fill_light")
    fill.set("diffuse", ".6 .6 .6")
    fill.set("ambient", ".1 .1 .1")
    fill.set("pos", "0 3 4")
    fill.set("dir", "0 -0.4 -1")
    fill.set("directional", "true")

    _inject_xml_fragment(worldbody, wall_xml())
    _inject_xml_fragment(worldbody, holds_xml(route.holds))
    _inject_xml_fragment(worldbody, kickboard_holds_xml())

    # grip_anchor — a massless, zero-size body welded to worldbody.
    # Used as the default body2 for all connect constraints so MuJoCo accepts
    # the XML even before runtime retargeting assigns each constraint to a hold.
    grip_anchor = ET.SubElement(worldbody, "body")
    grip_anchor.set("name", "grip_anchor")
    grip_anchor.set("pos", "0 0 0")

    # Humanoid torso with overridden position and orientation.
    hum_wb = _get("worldbody")
    if hum_wb is None:
        raise ValueError("humanoid.xml has no <worldbody> element")
    torso = hum_wb.find("body[@name='torso']") or hum_wb.find("body")
    if torso is None:
        raise ValueError("humanoid.xml worldbody has no <body> element")
    torso.set("pos", _HUMANOID_POS)
    torso.set("euler", _HUMANOID_EULER)
    worldbody.append(torso)

    # ── Equality constraints ───────────────────────────────────────────────────
    # Four connect constraints, one per limb slot, all inactive at start.
    # At runtime GripManager updates eq_obj2id and eq_data before activating.
    #
    # connect semantics: the anchor (in body1's local frame) is pinned to the
    # corresponding point in body2's local frame.  Setting anchor="0 0 0" pins
    # body1's origin; GripManager will set the body2 anchor so no snap occurs.
    #
    # solref="0.02 1"      → 20 ms time constant, critically damped
    # solimp="0.9 0.95 0.001" → near-rigid but with a small compliant zone
    equality = ET.SubElement(scene, "equality")
    for limb_body, constraint_name in zip(LIMB_BODY_NAMES, GRIP_CONSTRAINT_NAMES):
        c = ET.SubElement(equality, "connect")
        c.set("name", constraint_name)
        c.set("active", "false")
        c.set("body1", limb_body)
        c.set("body2", "grip_anchor")
        c.set("anchor", "0 0 0")
        c.set("solref", "0.02 1")
        c.set("solimp", "0.9 0.95 0.001")

    # ── Actuator, sensor, contact ─────────────────────────────────────────────
    for tag in ("actuator", "sensor", "contact"):
        elem = _get(tag)
        if elem is not None:
            scene.append(elem)

    # ── Serialise ─────────────────────────────────────────────────────────────
    _indent(scene)
    return ET.tostring(scene, encoding="unicode", xml_declaration=False)


def _inject_xml_fragment(parent: ET.Element, xml_fragment: str) -> None:
    """Parse a multi-line XML snippet and append each top-level element to parent.

    Args:
        parent: The ElementTree element to append children into.
        xml_fragment: XML string with one or more sibling top-level elements.
    """
    if not xml_fragment.strip():
        return
    wrapped = f"<_fragment>{xml_fragment}</_fragment>"
    try:
        frag_root = ET.fromstring(wrapped)
    except ET.ParseError as exc:
        raise ValueError(f"Invalid XML fragment: {exc}\n\nFragment:\n{xml_fragment}") from exc
    for child in frag_root:
        parent.append(child)
