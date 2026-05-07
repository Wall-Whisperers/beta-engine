"""Interactive MoonBoard Grip Tester.

Loads the most-repeated V4/V5 route, opens the MuJoCo viewer, and lets you
manually test grip engagement by pressing keys while dragging the humanoid
with MuJoCo's built-in perturbation tool.

KEY BINDINGS
────────────
  1   → grip LEFT  hand on nearest hold  (try_grip slot 0)
  2   → grip RIGHT hand on nearest hold  (try_grip slot 1)
  3   → release LEFT  hand grip          (release_grip slot 0)
  4   → release RIGHT hand grip          (release_grip slot 1)
  g   → print current grip state and active hold IDs
  f   → print current constraint force magnitudes
  r   → reset simulation (release all grips, call mj_resetData + mj_forward)

HOW TO DRAG THE HUMANOID
─────────────────────────
  In the MuJoCo viewer window:
    • Double-click a body part to select it (highlight appears).
    • Hold Ctrl and drag to apply a perturbation force.
  Move the humanoid until a hand is near a hold sphere, then press 1 or 2.

USAGE (macOS requires mjpython)
─────────────────────────────────
    mjpython scripts/interactive_grip.py
    python   scripts/interactive_grip.py   (headless fallback, no drag)
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from src.parsers import format1
from src.xml_gen import scene as scene_mod
from src.xml_gen.scene import LIMB_SITE_NAMES
from src.xml_gen.wall import hold_body_name, hold_position_world
from src.grip.grip_manager import GripManager
from src.viewer import launch as viewer_launch
import src.grip.grip_manager as _gm_mod

# Disable alignment check for interactive testing — in any pose the user
# should be able to grip when a hand is visually close to a hold.
_gm_mod.ALIGNMENT_THRESHOLD = -1.0

_MOONBOARD1 = os.path.join(_PROJECT_ROOT, "moonboard_data", "moonboard1.json")
_HUMANOID   = os.path.join(_PROJECT_ROOT, "assets", "humanoid.xml")

_KEY_HELP = """
╔══════════════════════════════════════════════════╗
║        INTERACTIVE GRIP TESTER — KEY MAP         ║
╠══════════════════════════════════════════════════╣
║  1   grip LEFT  hand on nearest hold             ║
║  2   grip RIGHT hand on nearest hold             ║
║  3   release LEFT  hand                          ║
║  4   release RIGHT hand                          ║
║  g   print grip state + active holds             ║
║  f   print constraint force magnitudes           ║
║  r   reset simulation (all grips released)       ║
╠══════════════════════════════════════════════════╣
║  Drag humanoid: Ctrl + drag in viewer            ║
╚══════════════════════════════════════════════════╝
"""


def _select_route(routes):
    """Return the most-repeated V4/V5 route."""
    candidates = [r for r in routes if r.grade_v in (4, 5)]
    if not candidates:
        raise RuntimeError("No V4/V5 routes found")
    return max(candidates, key=lambda r: r.repeats)


def _build_hold_dicts(route, model, mujoco):
    """Build hold_positions and hold_body_ids for GripManager."""
    from src.xml_gen.holds import RADIUS, _NY, _NZ
    hold_positions = {}
    hold_body_ids  = {}
    for h in route.holds:
        name = hold_body_name(h.col, h.row)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            continue
        wx, wy, wz = hold_position_world(h.col, h.row)
        hold_positions[name] = np.array([wx, wy + RADIUS * _NY, wz + RADIUS * _NZ])
        hold_body_ids[name]  = bid
    return hold_positions, hold_body_ids


def main():
    import mujoco

    print(_KEY_HELP)

    # ── Load route and model ──────────────────────────────────────────────────
    routes = format1.load_routes(_MOONBOARD1)
    route  = _select_route(routes)
    print(f"Route: '{route.name}'  V{route.grade_v}  repeats={route.repeats}")
    print(f"Holds: {len(route.holds)}")

    xml_str = scene_mod.build_scene_xml(route, _HUMANOID)
    model   = mujoco.MjModel.from_xml_string(xml_str)
    data    = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print(f"Model: {model.nbody} bodies  {model.ngeom} geoms  "
          f"{model.nsite} sites  {model.neq} equality constraints")

    # Print confirmed site positions (world coords at start) for verification.
    print("\nSite world positions at spawn:")
    for i, sname in enumerate(LIMB_SITE_NAMES):
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, sname)
        if sid >= 0:
            pos = data.site_xpos[sid]
            print(f"  {sname}: ({pos[0]:+.3f}, {pos[1]:+.3f}, {pos[2]:+.3f})")

    # ── GripManager ──────────────────────────────────────────────────────────
    hold_positions, hold_body_ids = _build_hold_dicts(route, model, mujoco)
    gm = GripManager(model, data, hold_positions, hold_body_ids)

    # ── Key callback ──────────────────────────────────────────────────────────
    def on_key(key: str, m, d) -> None:
        """Handle key presses from the viewer."""
        if key == "1":
            result = gm.nearest_hold(0)
            if result is None:
                print("[key 1] No holds in scene.")
                return
            hold_id, dist = result
            sid = gm._site_ids[0]
            site_pos = d.site_xpos[sid]
            print(f"[key 1] Nearest hold to LEFT hand: {hold_id}  dist={dist:.3f} m")
            print(f"        Site pos: ({site_pos[0]:+.3f}, {site_pos[1]:+.3f}, {site_pos[2]:+.3f})")
            gm.try_grip(0, hold_id)

        elif key == "2":
            result = gm.nearest_hold(1)
            if result is None:
                print("[key 2] No holds in scene.")
                return
            hold_id, dist = result
            sid = gm._site_ids[1]
            site_pos = d.site_xpos[sid]
            print(f"[key 2] Nearest hold to RIGHT hand: {hold_id}  dist={dist:.3f} m")
            print(f"        Site pos: ({site_pos[0]:+.3f}, {site_pos[1]:+.3f}, {site_pos[2]:+.3f})")
            gm.try_grip(1, hold_id)

        elif key == "3":
            gm.release_grip(0)

        elif key == "4":
            gm.release_grip(1)

        elif key == "g":
            state = gm.get_grip_state()
            active = gm.get_active_hold_ids()
            print(f"[key g] Grip state: {state}  active holds: {active}")

        elif key == "f":
            forces = gm._measure_constraint_forces()
            print(f"[key f] Constraint forces: { {k: f'{v:.1f} N' for k, v in forces.items()} }")

        elif key == "r":
            for slot in range(4):
                if gm._active_holds[slot] is not None:
                    gm.release_grip(slot)
            mujoco.mj_resetData(m, d)
            mujoco.mj_forward(m, d)
            print("[key r] Simulation reset. All grips released.")

    # ── Run viewer ────────────────────────────────────────────────────────────
    viewer_launch(model, data, on_key=on_key, title="MoonBoard Interactive Grip")


if __name__ == "__main__":
    main()
