"""Interactive 3D MoonBoard viewer.

Controls:
  Right-click near a limb  — drag it; release near a hold to snap on
  Ctrl+left-drag a limb    — alternate drag method
  Arrow keys               — move camera (up/down = zoom, left/right = pan)
  Left-click drag          — orbit camera
  1/2/3/4                  — grip or release LH / RH / LF / RF
  r                        — reset to start pose
  g                        — print grip state

Run with mjpython (required on macOS):

    mjpython -m sim3d.moonboard_interactive
    mjpython -m sim3d.moonboard_interactive --grade 5
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

import mujoco
import mujoco.viewer
import numpy as np

from sim3d import config as cfg
from sim3d.body import LIMBS
from sim3d.moonboard import load_moonboard_problems, moonboard_problem_to_wall
from sim3d.world import Climb3DWorld

_SNAP_RADIUS_M = 0.22
_GRAB_RADIUS_SCREEN_PT = 160.0  # screen-point radius for limb grab


# ── CoreGraphics mouse polling ─────────────────────────────────────────────
# These functions work from ANY thread on macOS; no GLFW context needed.

def _init_coregraphics():
    """Return (cursor_fn, right_down_fn) or raise if unavailable."""
    import ctypes, ctypes.util
    libCG = ctypes.CDLL(ctypes.util.find_library("CoreGraphics"))
    libCF = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    class _CGPoint(ctypes.Structure):
        _fields_ = [("x", ctypes.c_double), ("y", ctypes.c_double)]

    libCG.CGEventCreate.restype = ctypes.c_void_p
    libCG.CGEventCreate.argtypes = [ctypes.c_void_p]
    libCG.CGEventGetLocation.restype = _CGPoint
    libCG.CGEventGetLocation.argtypes = [ctypes.c_void_p]
    libCF.CFRelease.argtypes = [ctypes.c_void_p]
    libCG.CGEventSourceButtonState.argtypes = [ctypes.c_int32, ctypes.c_uint32]
    libCG.CGEventSourceButtonState.restype = ctypes.c_bool

    def _cursor():
        evt = libCG.CGEventCreate(None)
        pt = libCG.CGEventGetLocation(evt)
        libCF.CFRelease(evt)
        return float(pt.x), float(pt.y)

    def _right_down():
        return bool(libCG.CGEventSourceButtonState(0, 1))

    return _cursor, _right_down


def _reset_to_floor(world: Climb3DWorld) -> None:
    """Release all limbs and stand the dummy on the floor in front of the wall."""
    for lmb in LIMBS:
        world.release_limb(lmb)

    s = world.profile.segments
    theta = math.radians(world.wall.wall_angle_deg)

    # Board horizontal centre
    pelvis_x = (world.wall.cols * world.wall.cell_size_cm / 200.0)
    pelvis_z = s.standing_leg - 0.20

    # At standing height the overhanging wall face is at Y = pelvis_z * tan(θ).
    # Add clearance so the body lands in front of the wall, not inside it.
    wall_face_y = pelvis_z * math.tan(theta) + 0.10
    pelvis_y = wall_face_y + 0.50

    mujoco.mj_resetData(world.model, world.data)
    world.data.qpos[0:3] = (pelvis_x, pelvis_y, pelvis_z)
    world.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    world.data.qvel[:] = 0.0
    mujoco.mj_forward(world.model, world.data)
    print(f"  reset to floor  (pelvis y={pelvis_y:.2f} z={pelvis_z:.2f})")


def _build_world(problem, *, full_board: bool = True) -> Climb3DWorld:
    wall = moonboard_problem_to_wall(problem, include_full_board=full_board)
    return Climb3DWorld(wall)


def _seed(world: Climb3DWorld, problem) -> tuple:
    start_ids = [f"mb_{pos}" for pos in problem.start_holds
                 if f"mb_{pos}" in world._hold_meta_by_id]
    lh = rh = None
    if len(start_ids) >= 2:
        lh, rh = start_ids[0], start_ids[1]
    elif start_ids:
        lh = rh = start_ids[0]
    all_holds = sorted(world._hold_meta_by_id.items(),
                       key=lambda kv: kv[1]["world_pos"][2])
    lf = all_holds[0][0] if all_holds else None
    rf = all_holds[1][0] if len(all_holds) > 1 else lf
    world.seed_pose(lh=lh, rh=rh, lf=lf, rf=rf)
    return lh, rh, lf, rf


def _snap_nearest(world: Climb3DWorld, limb: str, radius: float = 0.40) -> None:
    tip = world.limb_tip_pos(limb)
    best_id: Optional[str] = None
    best_d = radius
    for hid, meta in world._hold_meta_by_id.items():
        d = float(np.linalg.norm(np.array(meta["world_pos"]) - tip))
        if d < best_d:
            best_d = d
            best_id = hid
    if best_id:
        world.move_limb(limb, best_id, mode="snap")
        print(f"  {limb} → {best_id}")
    else:
        print(f"  {limb}: no hold in range")


def run(world: Climb3DWorld, problem=None) -> None:
    # ── body-id → limb map (includes whole chain for Ctrl+drag) ──────────
    _chain = {
        "LH": ["l_upperarm", "l_forearm", "l_hand"],
        "RH": ["r_upperarm", "r_forearm", "r_hand"],
        "LF": ["l_thigh", "l_shin", "l_foot"],
        "RF": ["r_thigh", "r_shin", "r_foot"],
    }
    _body_to_limb: dict[int, str] = {}
    for lmb, names in _chain.items():
        for bname in names:
            bid = mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, bname)
            if bid >= 0:
                _body_to_limb[bid] = lmb

    # ── shared state between UI thread and main loop ──────────────────────
    _winfo: dict = {}        # window geometry — filled from key_callback
    _flags: dict = {"reset": False}

    def _key_cb(key: int) -> None:
        try:
            import glfw as _g
        except ImportError:
            return

        # One-time: grab window bounds and framebuffer size.
        # Requires the UI thread — key_callback is the only safe place.
        if not _winfo:
            win = _g.get_current_context()
            if win:
                wx, wy = _g.get_window_pos(win)
                ww, wh = _g.get_window_size(win)
                fw, fh = _g.get_framebuffer_size(win)
                _winfo.update(
                    x=float(wx), y=float(wy),
                    w=float(ww), h=float(wh),
                    fb_w=float(fw), fb_h=float(fh),
                )

        limb_keys = {_g.KEY_1: "LH", _g.KEY_2: "RH",
                     _g.KEY_3: "LF", _g.KEY_4: "RF"}
        if key in limb_keys:
            lmb = limb_keys[key]
            if world.on_hold(lmb):
                world.release_limb(lmb)
                print(f"  released {lmb}")
            else:
                _snap_nearest(world, lmb)
        elif key == _g.KEY_R and problem is not None:
            _flags["reset"] = True
        elif key == _g.KEY_G:
            print("  grip:", {l: world.on_hold(l) for l in LIMBS})
        # Camera via arrow keys (works in key_callback since we're on UI thread)
        elif key == _g.KEY_UP:
            viewer.cam.distance = max(1.0, viewer.cam.distance - 0.25)
        elif key == _g.KEY_DOWN:
            viewer.cam.distance += 0.25
        elif key == _g.KEY_LEFT:
            viewer.cam.lookat[0] -= 0.15
        elif key == _g.KEY_RIGHT:
            viewer.cam.lookat[0] += 0.15

    # ── CoreGraphics setup ────────────────────────────────────────────────
    try:
        _cg_cursor, _cg_right_down = _init_coregraphics()
        _HAS_CG = True
    except Exception as e:
        print(f"  [warn] CoreGraphics unavailable ({e}), right-click disabled")
        _HAS_CG = False

    # ── camera helpers ────────────────────────────────────────────────────
    def _cam_axes():
        el = math.radians(viewer.cam.elevation)
        az = math.radians(viewer.cam.azimuth)
        look = np.array(viewer.cam.lookat, dtype=float)
        dist = float(viewer.cam.distance)
        cam_pos = look + dist * np.array([
            math.cos(el) * math.sin(az),
            -math.cos(el) * math.cos(az),
            math.sin(el),
        ])
        fwd = look - cam_pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        rn = np.linalg.norm(right)
        right = np.array([1.0, 0.0, 0.0]) if rn < 1e-6 else right / rn
        up = np.cross(right, fwd)
        return cam_pos, fwd, right, up

    def _find_limb(cx: float, cy: float) -> Optional[str]:
        """Find the limb whose screen projection is closest to cursor (cx, cy)."""
        cam_pos, fwd, right, up = _cam_axes()
        half_tan = math.tan(math.radians(45.0) / 2.0)

        if _winfo:
            dpr = (_winfo["fb_h"] / _winfo["h"]) if _winfo["h"] > 0 else 2.0
            vp_x = (cx - _winfo["x"]) * dpr
            vp_y = (cy - _winfo["y"]) * dpr
            fb_w, fb_h = _winfo["fb_w"], _winfo["fb_h"]
            aspect = fb_w / fb_h if fb_h > 0 else 1.0
            max_d2 = (_GRAB_RADIUS_SCREEN_PT * dpr) ** 2

            best: Optional[str] = None
            best_d2 = max_d2
            for lmb in LIMBS:
                rel = world.limb_tip_pos(lmb) - cam_pos
                z = float(np.dot(rel, fwd))
                if z <= 0:
                    continue
                sx = float(np.dot(rel, right)) / (z * half_tan * aspect) * fb_w / 2 + fb_w / 2
                sy = -float(np.dot(rel, up)) / (z * half_tan) * fb_h / 2 + fb_h / 2
                d2 = (sx - vp_x) ** 2 + (sy - vp_y) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best = lmb
            if best:
                return best

        # Before window info is available: grab whichever limb is closest
        # to the camera look-at point in world space.
        look = np.array(viewer.cam.lookat)
        return min(LIMBS, key=lambda l: float(
            np.linalg.norm(world.limb_tip_pos(l) - look)
        ))

    def _delta_to_world(dcx: float, dcy: float) -> np.ndarray:
        """Convert a screen-point cursor delta to a world-space displacement."""
        _, _, right, up = _cam_axes()
        dist = float(viewer.cam.distance)
        fb_h = _winfo.get("fb_h", 1200.0)
        dpr = (_winfo["fb_h"] / _winfo["h"]) if _winfo and _winfo.get("h", 0) > 0 else 2.0
        m_per_pt = 2.0 * dist * math.tan(math.radians(45.0) / 2.0) / (fb_h / dpr)
        return dcx * m_per_pt * right - dcy * m_per_pt * up

    def _do_snap(lmb: str) -> None:
        mi = world._mocap_idx[lmb]
        drag_pos = np.array(world.data.mocap_pos[mi])
        best_id: Optional[str] = None
        best_d = _SNAP_RADIUS_M
        for hid, meta in world._hold_meta_by_id.items():
            d = float(np.linalg.norm(np.array(meta["world_pos"]) - drag_pos))
            if d < best_d:
                best_d = d
                best_id = hid
        if best_id:
            world.move_limb(lmb, best_id, mode="snap")
            print(f"  snapped {lmb} → {best_id}")
        else:
            world.data.eq_active[world._eq_idx[lmb]] = 0
            print(f"  {lmb}: no hold nearby (floating)")

    with mujoco.viewer.launch_passive(
        world.model, world.data, key_callback=_key_cb,
    ) as viewer:

        wall_h_m = world.wall.height_cm / 100.0
        ys = [m["world_pos"][1] for m in world._hold_meta_by_id.values()]
        mean_y = float(np.mean(ys)) if ys else 0.5

        viewer.cam.lookat[:] = [0.0, mean_y, wall_h_m * 0.42]
        viewer.cam.distance = max(4.5, wall_h_m * 1.4)
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -5.0

        print("Controls:")
        print("  Right-click near a limb  — drag it; release near hold to snap")
        print("  Ctrl+left-drag a limb    — alternate drag (no key required)")
        print("  Arrow keys               — up/down = zoom, left/right = pan")
        print("  1/2/3/4                 — grip / release LH/RH/LF/RF")
        print("  r / g                   — reset / grip state")

        # Right-click drag state
        _rdrag: dict = {"limb": None, "cx": 0.0, "cy": 0.0, "was_right": False,
                        "locked_lookat": None}

        # Ctrl+drag (viewer.perturb) state
        _pdrag: dict = {"limb": None, "was_active": False}

        frame_dt = 1.0 / cfg.RENDER_HZ
        next_frame = time.monotonic()

        while viewer.is_running():

            # ── Right-click drag via CoreGraphics ─────────────────────
            if _HAS_CG:
                right_now = _cg_right_down()
                cx, cy = _cg_cursor()

                if right_now and not _rdrag["was_right"]:
                    # Button just pressed: grab limb and lock camera lookat so
                    # MuJoCo's right-drag pan is suppressed this entire drag.
                    _rdrag["locked_lookat"] = np.copy(viewer.cam.lookat)
                    lmb = _find_limb(cx, cy)
                    if lmb:
                        world.release_limb(lmb)
                        mi = world._mocap_idx[lmb]
                        world.data.mocap_pos[mi] = world.limb_tip_pos(lmb).copy()
                        world.data.eq_active[world._eq_idx[lmb]] = 1
                        _rdrag.update(limb=lmb, cx=cx, cy=cy)
                        print(f"  grabbed {lmb}")

                elif right_now and _rdrag["limb"]:
                    # Dragging: apply screen delta as world offset.
                    # Restore lookat each frame to undo MuJoCo's camera pan.
                    if _rdrag["locked_lookat"] is not None:
                        viewer.cam.lookat[:] = _rdrag["locked_lookat"]
                    dcx = cx - _rdrag["cx"]
                    dcy = cy - _rdrag["cy"]
                    _rdrag["cx"] = cx
                    _rdrag["cy"] = cy
                    if abs(dcx) > 0.05 or abs(dcy) > 0.05:
                        dw = _delta_to_world(dcx, dcy)
                        lmb = _rdrag["limb"]
                        mi = world._mocap_idx[lmb]
                        world.data.mocap_pos[mi] = (
                            np.array(world.data.mocap_pos[mi]) + dw
                        )
                        world.data.eq_active[world._eq_idx[lmb]] = 1

                elif not right_now and _rdrag["was_right"] and _rdrag["limb"]:
                    # Button released: snap to nearest hold
                    _do_snap(_rdrag["limb"])
                    _rdrag["limb"] = None
                    _rdrag["locked_lookat"] = None

                _rdrag["was_right"] = right_now

            # ── Ctrl+drag (viewer.perturb) ────────────────────────────
            pert = viewer.perturb
            now_active = bool(pert.active)
            sel_body = int(pert.select)

            if now_active and not _pdrag["was_active"]:
                lmb = _body_to_limb.get(sel_body)
                if lmb:
                    _pdrag["limb"] = lmb
                    world.release_limb(lmb)
                    print(f"  ctrl-grabbed {lmb}")

            if now_active and _pdrag["limb"]:
                lmb = _pdrag["limb"]
                mi = world._mocap_idx[lmb]
                world.data.mocap_pos[mi] = np.array(pert.refpos, dtype=float)
                world.data.eq_active[world._eq_idx[lmb]] = 1

            if not now_active and _pdrag["was_active"] and _pdrag["limb"]:
                _do_snap(_pdrag["limb"])
                _pdrag["limb"] = None

            _pdrag["was_active"] = now_active

            # ── Reset (flag set by key_callback on UI thread) ────────
            if _flags["reset"]:
                _flags["reset"] = False
                _reset_to_floor(world)

            # ── Physics + render ──────────────────────────────────────
            world.step(1)
            viewer.sync()

            next_frame += frame_dt
            sleep_for = next_frame - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_frame = time.monotonic()


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent.parent
    p = argparse.ArgumentParser(
        prog="moonboard_interactive",
        description="Interactive 3D MoonBoard viewer",
    )
    p.add_argument("--data",
                   default=str(here / "moonboard_data" / "moonboard1.json"))
    p.add_argument("--grade", type=int, default=4)
    p.add_argument("--route", default=None)
    p.add_argument("--route-only", action="store_true")
    args = p.parse_args(argv)

    print(f"Loading {args.data} …")
    with open(args.data, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, list):
        raw.sort(key=lambda d: d.get("repeats", 0), reverse=True)

    problems = load_moonboard_problems(raw)
    candidates = [p for p in problems if p.grade == args.grade]
    if args.route:
        needle = args.route.lower()
        candidates = [p for p in candidates if needle in p.name.lower()]
    if not candidates:
        print(f"No V{args.grade} problems found.", file=sys.stderr)
        return 1

    problem = candidates[0]
    print(f"Route: {problem.name!r}  V{problem.grade}")
    print(f"  start={problem.start_holds}  finish={problem.end_holds}")

    world = _build_world(problem, full_board=not args.route_only)
    print(f"Wall: {len(world._hold_meta_by_id)} holds loaded")

    lh, rh, lf, rf = _seed(world, problem)
    print(f"Seeded: LH={lh} RH={rh} LF={lf} RF={rf}")

    for _ in range(120):
        world.step(1)

    run(world, problem=problem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
