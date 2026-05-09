"""Wall geometry viewer — CHECK A for the pre-RL audit.

Loads the most popular V4 route from moonboard1.json, builds the scene XML
(including kickboard and kickboard footholds), and opens the MuJoCo passive
viewer so the geometry can be inspected visually.

Visual checklist (Section 7 CHECK A):
  [ ] Kickboard is present below the main wall
  [ ] Kickboard is vertical; main wall is at 40-degree overhang
  [ ] 4 kickboard footholds (orange spheres) are visible on the kickboard face
  [ ] Main wall holds are at correct positions (green=start, blue=mid, red=end)
  [ ] No obvious geometry interpenetration at the kickboard/wall junction

Usage:
    python scripts/view_wall.py
"""

import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

import mujoco
import mujoco.viewer

from src.parsers import format1
from src.xml_gen.scene import build_scene_xml

_MOONBOARD1 = os.path.join(_PROJECT_ROOT, "moonboard_data", "moonboard1.json")
_HUMANOID   = os.path.join(_PROJECT_ROOT, "assets", "humanoid.xml")


def _select_route(routes):
    candidates = [r for r in routes if r.grade_v in (4, 5)]
    return max(candidates, key=lambda r: r.repeats)


def main() -> None:
    routes = format1.load_routes(_MOONBOARD1)
    route  = _select_route(routes)
    print(f"Route: '{route.name}'  V{route.grade_v}  repeats={route.repeats}")
    print(f"Holds: {len(route.holds)} ({sum(1 for h in route.holds if h.role=='start')} start, "
          f"{sum(1 for h in route.holds if h.role=='mid')} mid, "
          f"{sum(1 for h in route.holds if h.role=='end')} end)")
    print()
    print("Building scene XML (wall + kickboard + holds + humanoid)...")

    xml_str = build_scene_xml(route, _HUMANOID)
    model   = mujoco.MjModel.from_xml_string(xml_str)
    data    = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("Opening passive viewer — close the window to exit.")
    print()
    print("Visual checklist (Section 7 CHECK A):")
    print("  [ ] Kickboard is present below the main wall")
    print("  [ ] Kickboard is vertical; main wall is at 40-degree overhang")
    print("  [ ] 4 kickboard footholds (orange spheres) on kickboard face")
    print("  [ ] Main wall holds at correct positions (green/blue/red)")
    print("  [ ] No geometry interpenetration at kickboard/wall junction")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.5, 1.5]
        viewer.cam.distance   = 5.0
        viewer.cam.elevation  = -10.0
        viewer.cam.azimuth    = 180.0
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
