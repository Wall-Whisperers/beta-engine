"""Assembles a complete MuJoCo scene XML for Day-1 MoonBoard visualisation.

Strategy for merging the humanoid XML with scene elements:
  1. Parse humanoid.xml with xml.etree.ElementTree.
  2. Extract sub-trees: compiler, default, option, size, visual, asset,
     actuator, sensor, contact.
  3. Build a fresh <worldbody> containing:
       - A directional light.
       - The floor plane and tilted wall box (from wall.py).
       - Sphere geoms for each hold in the route (from holds.py).
       - The humanoid's root <body> with pos and euler overridden to place
         it in front of the wall, facing it.
  4. Serialise the merged tree to an XML string.

No <include> tags are used so the output is a single self-contained XML file
that can be loaded from any directory (e.g. a tempfile).
"""

import xml.etree.ElementTree as ET

from ..parsers.canonical import Route
from .holds import holds_xml
from .wall import wall_xml

# Humanoid placement: in front of the wall, facing -Y (toward wall face).
_HUMANOID_POS = "0 1.0 1.4"   # z=1.4 keeps feet above floor (matches original)
_HUMANOID_EULER = "0 0 180"    # rotate 180° about Z to face -Y


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
        # Last child gets the closing-tag indent.
        if not child.tail or not child.tail.strip():  # noqa: F821
            child.tail = pad
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = pad


def build_scene_xml(route: Route, humanoid_xml_path: str) -> str:
    """Build a complete MuJoCo XML scene for visualising a MoonBoard route.

    Parses humanoid.xml, merges its sections with the scene's wall/hold
    geometry, and returns the assembled XML as a string.

    Args:
        route: The climbing route whose holds will be rendered as spheres.
        humanoid_xml_path: Absolute path to humanoid.xml on disk.

    Returns:
        Complete MuJoCo XML string, ready to be passed to mujoco.MjModel.from_xml_string.
    """
    # ── Parse humanoid XML ────────────────────────────────────────────────────
    hum_tree = ET.parse(humanoid_xml_path)
    hum_root = hum_tree.getroot()   # <mujoco model="humanoid">

    def _get(tag: str) -> ET.Element | None:
        return hum_root.find(tag)

    # ── Build root element ────────────────────────────────────────────────────
    scene = ET.Element("mujoco")
    scene.set("model", "moonboard_scene")

    # compiler — keep humanoid's settings (angle="degree", etc.)
    compiler = _get("compiler")
    if compiler is not None:
        scene.append(compiler)

    # option — keep humanoid's physics settings.
    option = _get("option")
    if option is not None:
        scene.append(option)

    # size
    size = _get("size")
    if size is not None:
        scene.append(size)

    # visual
    visual = _get("visual")
    if visual is not None:
        scene.append(visual)

    # default
    default = _get("default")
    if default is not None:
        scene.append(default)

    # asset — humanoid materials + textures (we need MatPlane, geom textures).
    asset = _get("asset")
    if asset is None:
        asset = ET.SubElement(scene, "asset")
    else:
        scene.append(asset)

    # ── Worldbody ─────────────────────────────────────────────────────────────
    worldbody = ET.SubElement(scene, "worldbody")

    # Lighting.
    light = ET.SubElement(worldbody, "light")
    light.set("name", "main_light")
    light.set("diffuse", ".9 .9 .9")
    light.set("pos", "0 -1 5")
    light.set("dir", "0 0.2 -1")
    light.set("directional", "true")

    # Wall + floor geoms (raw XML — embed via string and re-parse as fragment).
    _inject_xml_fragment(worldbody, wall_xml())

    # Hold spheres.
    _inject_xml_fragment(worldbody, holds_xml(route.holds))

    # Humanoid root body: find the torso body in humanoid worldbody, override pos/euler.
    hum_wb = _get("worldbody")
    if hum_wb is None:
        raise ValueError("humanoid.xml has no <worldbody> element")
    torso = hum_wb.find("body[@name='torso']")
    if torso is None:
        # Fallback: take the first body element.
        torso = hum_wb.find("body")
    if torso is None:
        raise ValueError("humanoid.xml worldbody has no <body> element")

    torso.set("pos", _HUMANOID_POS)
    torso.set("euler", _HUMANOID_EULER)
    worldbody.append(torso)

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

    Wraps the fragment in a temporary root tag so ElementTree can parse it.

    Args:
        parent: The ElementTree element to append children into.
        xml_fragment: XML string containing one or more sibling elements.
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
