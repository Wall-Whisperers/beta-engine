"""Day 1 MoonBoard Viewer.

Loads the most-repeated V4/V5 route from moonboard1.json, builds a MuJoCo
scene containing the tilted wall and a humanoid, and opens the interactive
viewer.  Falls back gracefully if no display is available.

Usage:
    python scripts/day1_viewer.py

Output:
  - Prints the selected route and hold world coordinates.
  - Opens the MuJoCo viewer (if a display is present).
  - Falls back to writing the scene XML to output/scene_day1.xml.
"""

import os
import sys
import tempfile

# ── Path setup so src/ is importable from scripts/ ───────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import math

import numpy as np

from src.parsers import format1
from src.xml_gen import scene as scene_mod
from src.xml_gen.wall import (
    COS_A,
    SIN_A,
    WALL_NORMAL,
    hold_position_world,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(_PROJECT_ROOT, "moonboard_data")
_MOONBOARD1_PATH = os.path.abspath(os.path.join(_DATA_DIR, "moonboard1.json"))
_HUMANOID_PATH = os.path.join(_PROJECT_ROOT, "assets", "humanoid.xml")
_OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")
_OUTPUT_XML = os.path.join(_OUTPUT_DIR, "scene_day1.xml")


def _select_route(routes):
    """Filter to V4/V5 and return the route with the highest repeat count."""
    candidates = [r for r in routes if r.grade_v in (4, 5)]
    if not candidates:
        raise RuntimeError("No V4/V5 routes found in dataset")
    return max(candidates, key=lambda r: r.repeats)


def _print_sanity_checks(route, xml_str):
    """Run and print all Day-1 sanity checks."""
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # 1. Wall surface normal.
    print(f"\n[1] Wall surface normal (outward, toward climber):")
    print(f"    n = (0, cos(40°), -sin(40°)) = (0, {COS_A:.4f}, {-SIN_A:.4f})")
    print(f"    Magnitude = {math.sqrt(COS_A**2 + SIN_A**2):.4f}  (should be 1.0)")

    # 2. Corner hold positions.
    a1 = hold_position_world(0, 1)
    k18 = hold_position_world(10, 18)
    print(f"\n[2] Hold A1 (bottom-left):  x={a1[0]:.3f}  y={a1[1]:.3f}  z={a1[2]:.3f}")
    print(f"    Hold K18 (top-right):   x={k18[0]:.3f}  y={k18[1]:.3f}  z={k18[2]:.3f}")
    print(f"    A1.x < K18.x? {a1[0] < k18[0]}  (should be True — A is left of K)")
    print(f"    A1.z < K18.z? {a1[2] < k18[2]}  (should be True — row 1 below row 18)")
    print(f"    A1.y < K18.y? {a1[1] < k18[1]}  (should be True — overhang: top closer to climber)")

    # 3. Humanoid position.
    print(f"\n[3] Humanoid root body pos: {scene_mod._HUMANOID_POS}")
    print(f"    Humanoid euler:          {scene_mod._HUMANOID_EULER}")

    # 4. Route holds.
    print(f"\n[4] Hold positions for selected route ({len(route.holds)} holds):")
    xs, ys, zs = [], [], []
    for h in route.holds:
        col_letter = chr(ord("A") + h.col)
        wx, wy, wz = hold_position_world(h.col, h.row)
        xs.append(wx); ys.append(wy); zs.append(wz)
        print(f"    [{h.role:5s}] {col_letter}{h.row:2d}  x={wx:+.3f}  y={wy:+.3f}  z={wz:.3f}")

    # 5. Bounding box.
    print(f"\n[5] Hold bounding box:")
    print(f"    x: [{min(xs):+.3f}, {max(xs):+.3f}]  (expect ≈ [-1.0, +1.0])")
    print(f"    y: [{min(ys):+.3f}, {max(ys):+.3f}]")
    print(f"    z: [{min(zs):+.3f}, {max(zs):+.3f}]  (expect ≈ [0.30, 3.00])")
    print()
    return xs, ys, zs


def _launch_viewer(model, data):
    """Attempt to open the MuJoCo interactive viewer.

    On macOS, MuJoCo's passive viewer requires the script to be run under
    mjpython (MuJoCo's bundled Python runtime with display support).
    If launch_passive raises RuntimeError for this reason, the error is
    re-raised so the caller can fall back gracefully.
    """
    import mujoco.viewer
    print("\nOpening MuJoCo viewer — close the window to exit.")
    print("(On macOS: if this fails, run with:  mjpython scripts/day1_viewer.py)")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


def main():
    # ── Load and select route ─────────────────────────────────────────────────
    print(f"Loading routes from {_MOONBOARD1_PATH} ...")
    if not os.path.exists(_MOONBOARD1_PATH):
        print(f"ERROR: moonboard1.json not found at {_MOONBOARD1_PATH}")
        sys.exit(1)

    routes = format1.load_routes(_MOONBOARD1_PATH)
    print(f"  {len(routes)} routes loaded total.")

    route = _select_route(routes)
    print(f"\nSelected route:")
    print(f"  Name:    {route.name}")
    print(f"  Grade:   V{route.grade_v}")
    print(f"  Repeats: {route.repeats}")
    print(f"  Holds:   {len(route.holds)}")

    # ── Build scene XML ───────────────────────────────────────────────────────
    print(f"\nBuilding scene XML with humanoid from {_HUMANOID_PATH} ...")
    xml_str = scene_mod.build_scene_xml(route, _HUMANOID_PATH)

    # Always write the XML for inspection.
    os.makedirs(_OUTPUT_DIR, exist_ok=True)
    with open(_OUTPUT_XML, "w", encoding="utf-8") as fh:
        fh.write(xml_str)
    print(f"Scene XML written to {_OUTPUT_XML}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    xs, ys, zs = _print_sanity_checks(route, xml_str)

    # ── Load MuJoCo model ─────────────────────────────────────────────────────
    try:
        import mujoco
        print("Loading MuJoCo model from generated XML ...")
        model = mujoco.MjModel.from_xml_string(xml_str)
        data = mujoco.MjData(model)
        print(f"  Model loaded: {model.nbody} bodies, {model.ngeom} geoms.")
    except Exception as exc:
        print(f"\nERROR: MuJoCo failed to load the model: {exc}")
        print("\n--- Begin scene XML ---")
        print(xml_str)
        print("--- End scene XML ---")
        sys.exit(1)

    # ── Launch viewer ─────────────────────────────────────────────────────────
    viewer_ok = False
    try:
        _launch_viewer(model, data)
        viewer_ok = True
    except Exception as exc:
        print(f"\nVIEWER UNAVAILABLE ({type(exc).__name__}: {exc})")
        print(f"VIEWER UNAVAILABLE - scene XML written to output/scene_day1.xml")

if __name__ == "__main__":
    main()
