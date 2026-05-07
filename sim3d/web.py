"""Flask blueprint exposing the 3D simulator over HTTP.

The grid editor (`grid_editor/server.py`) registers this blueprint
under `/sim3d/*`. From the browser:

    GET  /sim3d/                       → three.js viewer page
    GET  /sim3d/api/walls              → list of walls (proxies the editor)
    POST /sim3d/api/session            → start a new sim session for a wall
    GET  /sim3d/api/session/<sid>/pose → current pose snapshot (JSON)
    POST /sim3d/api/session/<sid>/step → step the sim by N frames
    POST /sim3d/api/session/<sid>/move → move a limb to a hold
    POST /sim3d/api/session/<sid>/seed → re-seed the pose
    DELETE /sim3d/api/session/<sid>    → drop the session

Sessions are kept in memory only — they're a debug aid, not a
multi-tenant production endpoint. One Climb3DWorld per session.

Threading note: Flask's dev server is multi-threaded, MuJoCo's MjData
is not thread-safe. Each session takes a per-session lock so
concurrent step/move requests serialise. Read-only `pose` requests
take the same lock; that's safe but means a long step blocks the
viewer. With 60 Hz steps the lock is held for <1 ms — fine.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from flask import Blueprint, abort, jsonify, request, send_from_directory

from sim3d import Climb3DWorld, ClimberProfile
from sim3d.body import LIMBS
from sim3d.moonboard import (
    load_moonboard_problems,
    moonboard_problem_to_wall,
    find_problem,
)
from solver.wall import load_wall

bp = Blueprint("sim3d", __name__, url_prefix="/sim3d")

# Standard search path for MoonBoard problem files. The browser viewer
# discovers problems by listing this dir; advanced users can drop their
# own JSON in here and it'll show up in the dropdown.
MOONBOARD_DIR = Path("/data/moonboard")
MOONBOARD_DIR_FALLBACK = Path(__file__).resolve().parent.parent / "data" / "moonboard"


@dataclass
class _Session:
    world: Climb3DWorld
    lock: threading.Lock = field(default_factory=threading.Lock)


_sessions: dict[str, _Session] = {}
_sessions_lock = threading.Lock()


# ─── Page route ───────────────────────────────────────────────────────────
@bp.route("/")
def viewer_page():
    static_dir = Path(__file__).resolve().parent.parent / "static"
    return send_from_directory(static_dir, "sim3d.html")


# ─── Session lifecycle ────────────────────────────────────────────────────
def _moonboard_dirs() -> list[Path]:
    dirs = [d for d in (MOONBOARD_DIR, MOONBOARD_DIR_FALLBACK) if d.exists()]
    # Dedupe: if both dirs exist and host the same canonical files,
    # keep only the primary. Prevents the listing endpoint from
    # returning each file twice.
    if len(dirs) == 2 and dirs[0].resolve() == dirs[1].resolve():
        return [dirs[0]]
    return dirs


def _moonboard_files_unique() -> list[Path]:
    """Yield each moonboard JSON exactly once, primary dir wins on
    name-collision."""
    seen: set[str] = set()
    out: list[Path] = []
    for d in _moonboard_dirs():
        for p in sorted(d.glob("*.json")):
            if p.name in seen:
                continue
            seen.add(p.name)
            out.append(p)
    return out


@bp.route("/api/moonboard", methods=["GET"])
def list_moonboard_files():
    """List MoonBoard problem-JSON files available to the server."""
    files = []
    for p in _moonboard_files_unique():
        try:
            problems = load_moonboard_problems(p)
        except Exception as e:
            files.append({"file": p.name, "dir": str(p.parent), "error": str(e)})
            continue
        files.append({
            "file": p.name,
            "dir": str(p.parent),
            "problem_count": len(problems),
            "sample": [
                {"id": pr.id, "name": pr.name, "grade": pr.grade}
                for pr in problems[:50]
            ],
        })
    return jsonify({"files": files})


@bp.route("/api/session", methods=["POST"])
def create_session():
    payload = request.get_json(silent=True) or {}
    wall_id = payload.get("wall_id", "example-v2-boulder")
    height_cm = float(payload.get("height_cm", 175))
    wingspan_cm = float(payload.get("wingspan_cm", 175))
    mass_kg = float(payload.get("mass_kg", 70))
    seed = bool(payload.get("seed", True))

    moonboard_file = payload.get("moonboard_file")
    moonboard_problem_id = payload.get("moonboard_problem_id")

    if moonboard_file is not None:
        # Load a MoonBoard problem rather than a stored wall.
        target = None
        for d in _moonboard_dirs():
            cand = d / moonboard_file
            if cand.exists():
                target = cand
                break
        if target is None:
            abort(404, description=f"moonboard file not found: {moonboard_file}")
        problems = load_moonboard_problems(target)
        problem = None
        if moonboard_problem_id is not None:
            problem = find_problem(problems, id=int(moonboard_problem_id))
        if problem is None:
            problem = problems[0] if problems else None
        if problem is None:
            abort(404, description="no problems in MoonBoard file")
        wall = moonboard_problem_to_wall(problem)
    else:
        try:
            wall = load_wall(wall_id)
        except (FileNotFoundError, KeyError) as e:
            abort(404, description=f"wall not found: {e}")

    profile = ClimberProfile(
        height_cm=height_cm,
        wingspan_cm=wingspan_cm,
        mass_kg=mass_kg,
    )
    world = Climb3DWorld(wall, profile)

    if seed:
        starts = wall.starts()
        foots = [h for h in wall.holds if h.hold_type == "foothold"][:2]
        if len(starts) >= 2 and len(foots) >= 2:
            world.seed_pose(
                lh=starts[0].hold_id, rh=starts[1].hold_id,
                lf=foots[0].hold_id, rf=foots[1].hold_id,
            )

    sid = uuid.uuid4().hex[:12]
    with _sessions_lock:
        _sessions[sid] = _Session(world=world)

    return jsonify({
        "session_id": sid,
        "wall": {
            "wall_id": wall.wall_id,
            "name": wall.name,
            "width_m": wall.width_cm / 100.0,
            "height_m": wall.height_cm / 100.0,
            "wall_angle_deg": wall.wall_angle_deg,
            "n_holds": len(wall.holds),
        },
        "profile": {
            "height_cm": profile.height_cm,
            "wingspan_cm": profile.wingspan_cm,
            "mass_kg": profile.mass_kg,
        },
        "pose": world.pose_snapshot(),
    })


@bp.route("/api/session/<sid>", methods=["DELETE"])
def drop_session(sid: str):
    with _sessions_lock:
        _sessions.pop(sid, None)
    return ("", 204)


def _get(sid: str) -> _Session:
    with _sessions_lock:
        s = _sessions.get(sid)
    if s is None:
        abort(404, description="session not found")
    return s


# ─── Per-session actions ──────────────────────────────────────────────────
@bp.route("/api/session/<sid>/pose", methods=["GET"])
def get_pose(sid: str):
    s = _get(sid)
    with s.lock:
        return jsonify(s.world.pose_snapshot())


@bp.route("/api/session/<sid>/step", methods=["POST"])
def step_session(sid: str):
    s = _get(sid)
    payload = request.get_json(silent=True) or {}
    frames = int(payload.get("frames", 1))
    frames = max(1, min(frames, 600))   # cap so a bad client can't lock us up
    with s.lock:
        s.world.step(frames=frames)
        return jsonify(s.world.pose_snapshot())


@bp.route("/api/session/<sid>/move", methods=["POST"])
def move_limb(sid: str):
    s = _get(sid)
    payload = request.get_json(silent=True) or {}
    limb = str(payload.get("limb", "")).upper()
    hold_id = str(payload.get("hold_id", ""))
    mode = str(payload.get("mode", "reach"))
    if limb not in LIMBS:
        abort(400, description=f"limb must be one of {LIMBS}")
    if not hold_id:
        abort(400, description="hold_id required")
    with s.lock:
        try:
            s.world.move_limb(limb, hold_id, mode=mode)
        except (KeyError, ValueError) as e:
            abort(400, description=str(e))
        return jsonify(s.world.pose_snapshot())


@bp.route("/api/session/<sid>/seed", methods=["POST"])
def seed_pose(sid: str):
    s = _get(sid)
    payload = request.get_json(silent=True) or {}
    kwargs = {k: payload.get(k) for k in ("lh", "rh", "lf", "rf")}
    with s.lock:
        try:
            s.world.seed_pose(**kwargs)
        except KeyError as e:
            abort(400, description=str(e))
        return jsonify(s.world.pose_snapshot())
