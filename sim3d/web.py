"""Flask blueprint exposing the 3D simulator over HTTP.

The grid editor (`grid_editor/server.py`) registers this blueprint
under `/sim3d/*`. From the browser:

    GET  /sim3d/                       → three.js viewer page
    GET  /sim3d/api/walls              → list of walls (proxies the editor)
    POST /sim3d/api/session            → start a new sim session for a wall
    GET  /sim3d/api/session/<sid>/pose → current pose snapshot (JSON)
    POST /sim3d/api/session/<sid>/step → step the sim by N frames
    POST /sim3d/api/session/<sid>/move → move a limb to a hold
    POST /sim3d/api/session/<sid>/seed        → re-seed the pose
    POST /sim3d/api/session/<sid>/policy      → load an SB3 PPO run/model
    POST /sim3d/api/session/<sid>/policy/step → apply one policy action
    DELETE /sim3d/api/session/<sid>/policy    → clear the loaded policy
    DELETE /sim3d/api/session/<sid>           → drop the session

Sessions are kept in memory only — they're a debug aid, not a
multi-tenant production endpoint. One Climb3DWorld per session.

Threading note: Flask's dev server is multi-threaded, MuJoCo's MjData
is not thread-safe. Each session takes a per-session lock so
concurrent step/move requests serialise. Read-only `pose` requests
take the same lock; that's safe but means a long step blocks the
viewer. With 60 Hz steps the lock is held for <1 ms — fine.
"""
from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, abort, jsonify, request, send_from_directory

from sim3d import Climb3DWorld, ClimberProfile
from sim3d.body import LIMBS
from sim3d.moonboard import (
    load_moonboard_problems,
    moonboard_problem_to_wall,
    find_problem,
)
from solver.wall import Wall, load_wall

bp = Blueprint("sim3d", __name__, url_prefix="/sim3d")

# Standard search path for MoonBoard problem files. The browser viewer
# discovers problems by listing this dir; advanced users can drop their
# own JSON in here and it'll show up in the dropdown.
MOONBOARD_DIR = Path("/data/moonboard")
MOONBOARD_DIR_FALLBACK = Path(__file__).resolve().parent.parent / "data" / "moonboard"
MOONBOARD_DATA_DIR_FALLBACK = Path(__file__).resolve().parent.parent / "moonboard_data"
REPO_ROOT = Path(__file__).resolve().parent.parent
SIM3D_RUNS_DIR = REPO_ROOT / "data" / "runs" / "sim3d"


@dataclass
class _PolicyState:
    model: Any
    env: Any
    obs: Any
    run_path: str
    model_path: str
    config: dict[str, Any]
    last_info: dict[str, Any] = field(default_factory=dict)
    done: bool = False


@dataclass
class _Session:
    world: Climb3DWorld
    wall: Wall
    profile: ClimberProfile
    source: dict[str, Any]
    policy: Optional[_PolicyState] = None
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
    dirs = [
        d for d in (MOONBOARD_DIR, MOONBOARD_DIR_FALLBACK, MOONBOARD_DATA_DIR_FALLBACK)
        if d.exists()
    ]
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


def _find_moonboard_file(name_or_path: str) -> Path:
    raw = Path(name_or_path)
    candidates = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.extend([raw, REPO_ROOT / raw])
        candidates.extend(d / raw.name for d in _moonboard_dirs())
    for cand in candidates:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"moonboard file not found: {name_or_path}")


def _wall_response(wall: Wall) -> dict[str, Any]:
    return {
        "wall_id": wall.wall_id,
        "name": wall.name,
        "width_m": wall.width_cm / 100.0,
        "height_m": wall.height_cm / 100.0,
        "wall_angle_deg": wall.wall_angle_deg,
        "n_holds": len(wall.holds),
    }


def _load_wall_from_request(payload: dict[str, Any]) -> tuple[Wall, dict[str, Any]]:
    wall_id = payload.get("wall_id", "example-v2-boulder")
    moonboard_file = payload.get("moonboard_file")
    moonboard_problem_id = payload.get("moonboard_problem_id")
    moonboard_vertical = bool(payload.get("moonboard_vertical_projection", False))
    moonboard_full_board = bool(payload.get("moonboard_full_board", False))

    if moonboard_file is not None:
        target = _find_moonboard_file(str(moonboard_file))
        problems = load_moonboard_problems(target)
        problem = None
        if moonboard_problem_id is not None:
            problem = find_problem(problems, id=int(moonboard_problem_id))
        if problem is None:
            problem = problems[0] if problems else None
        if problem is None:
            raise FileNotFoundError("no problems in MoonBoard file")
        wall = moonboard_problem_to_wall(
            problem,
            include_full_board=moonboard_full_board,
            vertical_projection=moonboard_vertical,
        )
        return wall, {
            "kind": "moonboard",
            "moonboard_file": target.name,
            "moonboard_problem_id": problem.id,
            "moonboard_vertical_projection": moonboard_vertical,
            "moonboard_full_board": moonboard_full_board,
            "selected_value": f"moonboard:{target.name}:{problem.id}",
        }

    wall = load_wall(wall_id)
    return wall, {
        "kind": "wall",
        "wall_id": wall.wall_id,
        "selected_value": wall.wall_id,
    }


def _load_wall_from_train_config(
    cfg: dict[str, Any], run_path: Path | None = None,
) -> tuple[Wall, dict[str, Any]]:
    if cfg.get("moonboard_file"):
        problem_id = cfg.get("moonboard_problem_id")
        full_board = problem_id is None
        if problem_id is None and run_path is not None:
            split_path = run_path / "moonboard_splits.json" if run_path.is_dir() else run_path.parent / "moonboard_splits.json"
            if split_path.exists():
                splits = json.loads(split_path.read_text(encoding="utf-8"))
                split_name = str(cfg.get("moonboard_split", "train"))
                selected = splits.get(split_name) or splits.get("train") or []
                if selected:
                    problem_id = selected[0].get("id")
        return _load_wall_from_request({
            "moonboard_file": cfg["moonboard_file"],
            "moonboard_problem_id": problem_id,
            "moonboard_vertical_projection": bool(
                cfg.get("moonboard_vertical_projection", False)
            ),
            "moonboard_full_board": full_board,
        })
    return _load_wall_from_request({"wall_id": cfg.get("wall", "example-v2-boulder")})


def _start_hand_targets(wall: Wall) -> tuple[str | None, str | None]:
    starts = sorted(wall.starts(), key=lambda h: h.x_cm)
    if len(starts) >= 2:
        return starts[0].hold_id, starts[-1].hold_id
    if len(starts) == 1:
        return starts[0].hold_id, starts[0].hold_id
    hand_low = sorted(
        [h for h in wall.holds if h.usable_for_hand()],
        key=lambda h: (h.y_cm, h.x_cm),
    )[:2]
    if len(hand_low) >= 2:
        return hand_low[0].hold_id, hand_low[-1].hold_id
    if len(hand_low) == 1:
        return hand_low[0].hold_id, hand_low[0].hold_id
    return None, None


def _seed_world(wall: Wall, world: Climb3DWorld, *, start_mode: str = "seed") -> None:
    if start_mode == "ground-reach":
        world._sync_actuator_targets_to_pose()
        lh, rh = _start_hand_targets(wall)
        if lh is not None:
            world.move_limb("LH", lh, mode="reach")
        if rh is not None:
            world.move_limb("RH", rh, mode="reach")
        return

    starts = sorted(wall.starts(), key=lambda h: h.x_cm)
    foots = sorted(
        [h for h in wall.holds if h.usable_for_foot()],
        key=lambda h: (h.y_cm, h.x_cm),
    )[:2]
    if len(starts) >= 2 and len(foots) >= 2:
        lh, rh = starts[0].hold_id, starts[-1].hold_id
    elif len(starts) == 1:
        lh = rh = starts[0].hold_id
    else:
        lh, rh = _start_hand_targets(wall)
    if lh is not None and rh is not None and len(foots) >= 2:
        l_foot, r_foot = sorted(foots, key=lambda h: h.x_cm)
        world.seed_pose(
            lh=lh, rh=rh,
            lf=l_foot.hold_id, rf=r_foot.hold_id,
        )


def _resolve_run_path(path_text: str) -> tuple[Path, Path, Path]:
    if not path_text.strip():
        raise FileNotFoundError("run path is required")
    raw = Path(path_text.strip()).expanduser()
    candidates = [raw]
    if not raw.is_absolute():
        candidates.extend([REPO_ROOT / raw, SIM3D_RUNS_DIR / raw])
    run_path = next((p for p in candidates if p.exists()), None)
    if run_path is None:
        raise FileNotFoundError(f"run path not found: {path_text}")

    model_path = run_path / "model.zip" if run_path.is_dir() else run_path
    if not model_path.exists():
        raise FileNotFoundError(f"model.zip not found under: {run_path}")

    config_path = model_path.parent / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found next to model: {model_path}"
        )
    return run_path, model_path, config_path


def _policy_response(s: _Session, *, include_static: bool = False) -> dict[str, Any]:
    policy = None
    if s.policy is not None:
        policy = {
            "loaded": True,
            "run_path": s.policy.run_path,
            "model_path": s.policy.model_path,
            "done": s.policy.done,
            "last_info": s.policy.last_info,
            "config": s.policy.config,
        }
    return {
        "wall": _wall_response(s.wall),
        "source": s.source,
        "profile": {
            "height_cm": s.profile.height_cm,
            "wingspan_cm": s.profile.wingspan_cm,
            "mass_kg": s.profile.mass_kg,
        },
        "pose": s.world.pose_snapshot(include_static=include_static),
        "policy": policy,
    }


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
    height_cm = float(payload.get("height_cm", 175))
    wingspan_cm = float(payload.get("wingspan_cm", 175))
    mass_kg = float(payload.get("mass_kg", 70))
    seed = bool(payload.get("seed", True))
    start_mode = str(payload.get("start_mode", "seed"))
    if start_mode not in ("seed", "ground-reach"):
        abort(400, description="start_mode must be 'seed' or 'ground-reach'")

    try:
        wall, source = _load_wall_from_request(payload)
    except (FileNotFoundError, KeyError, ValueError) as e:
        abort(404, description=str(e))

    profile = ClimberProfile(
        height_cm=height_cm,
        wingspan_cm=wingspan_cm,
        mass_kg=mass_kg,
    )
    world = Climb3DWorld(wall, profile)

    if seed:
        _seed_world(wall, world, start_mode=start_mode)

    sid = uuid.uuid4().hex[:12]
    session = _Session(world=world, wall=wall, profile=profile, source=source)
    with _sessions_lock:
        _sessions[sid] = session

    response = _policy_response(session, include_static=True)
    response["session_id"] = sid
    return jsonify(response)


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
    # Default to "reach" — the continuous Cartesian-impedance reach
    # so the body actually swings into position rather than teleporting.
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
        s.policy = None
        return jsonify(s.world.pose_snapshot())


@bp.route("/api/session/<sid>/policy", methods=["POST"])
def load_policy(sid: str):
    s = _get(sid)
    payload = request.get_json(silent=True) or {}
    run_text = str(payload.get("run_path", ""))
    try:
        run_path, model_path, config_path = _resolve_run_path(run_text)
        cfg = json.loads(config_path.read_text(encoding="utf-8"))
        wall, source = _load_wall_from_train_config(cfg, run_path)
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
        abort(400, description=str(e))

    profile = ClimberProfile(
        height_cm=float(cfg.get("height_cm", 175.0)),
        wingspan_cm=float(cfg.get("wingspan_cm", 175.0)),
        mass_kg=float(cfg.get("mass_kg", 70.0)),
    )
    try:
        from stable_baselines3 import PPO
        from sim3d.env import Climbing3DEnv, EnvConfig

        env_cfg = EnvConfig(
            move_mode=str(cfg.get("move_mode", "reach")),
            start_mode=str(cfg.get("start_mode", "seed")),
            move_frames=int(cfg.get("move_frames", 24)),
            max_steps=int(cfg.get("max_episode_steps", 30)),
            enable_slip=bool(cfg.get("enable_slip", True)),
            body_intersection_penalty=float(
                cfg.get("body_intersection_penalty", 2.0)
            ),
            official_route_only=bool(source.get("moonboard_full_board", False)),
        )
        model = PPO.load(str(model_path))
        env = Climbing3DEnv(wall, profile=profile, config=env_cfg)
        obs, info = env.reset()
    except Exception as e:  # SB3/MuJoCo shape errors should be surfaced clearly.
        abort(400, description=f"failed to load policy: {e}")

    with s.lock:
        s.wall = wall
        s.profile = profile
        s.source = source
        s.world = env.world
        s.policy = _PolicyState(
            model=model,
            env=env,
            obs=obs,
            run_path=str(run_path),
            model_path=str(model_path),
            config=cfg,
            last_info=info,
        )
        return jsonify(_policy_response(s, include_static=True))


@bp.route("/api/session/<sid>/policy/step", methods=["POST"])
def step_policy(sid: str):
    s = _get(sid)
    payload = request.get_json(silent=True) or {}
    deterministic = bool(payload.get("deterministic", True))
    with s.lock:
        if s.policy is None:
            abort(400, description="no policy loaded")
        if s.policy.done:
            obs, info = s.policy.env.reset()
            s.policy.obs = obs
            s.policy.last_info = info
            s.policy.done = False
            s.world = s.policy.env.world

        action, _ = s.policy.model.predict(
            s.policy.obs, deterministic=deterministic,
        )
        obs, reward, term, trunc, info = s.policy.env.step(action)
        s.policy.obs = obs
        s.policy.done = bool(term or trunc)
        s.policy.last_info = info
        s.world = s.policy.env.world

        response = _policy_response(s)
        action_serial = action.tolist() if hasattr(action, "tolist") else action
        response["policy_step"] = {
            "action": action_serial,
            "reward": float(reward),
            "terminated": bool(term),
            "truncated": bool(trunc),
            "info": info,
        }
        return jsonify(response)


@bp.route("/api/session/<sid>/policy", methods=["DELETE"])
def clear_policy(sid: str):
    s = _get(sid)
    with s.lock:
        s.policy = None
        return jsonify(_policy_response(s, include_static=True))


@bp.route("/api/session/<sid>/seed", methods=["POST"])
def seed_pose(sid: str):
    s = _get(sid)
    payload = request.get_json(silent=True) or {}
    start_mode = payload.get("start_mode")
    kwargs = {k: payload.get(k) for k in ("lh", "rh", "lf", "rf")}
    with s.lock:
        try:
            if start_mode is not None:
                start_mode = str(start_mode)
                if start_mode not in ("seed", "ground-reach"):
                    abort(400, description="start_mode must be 'seed' or 'ground-reach'")
                s.world.reset()
                _seed_world(s.wall, s.world, start_mode=start_mode)
            else:
                s.world.seed_pose(**kwargs)
        except KeyError as e:
            abort(400, description=str(e))
        s.policy = None
        return jsonify(s.world.pose_snapshot())
