"""Beta Engine — climbing wall grid editor + solver dashboard backend.

Serves the static editor UI and provides a small JSON API for listing,
loading, saving, and deleting wall files under /data/walls/.

Also exposes /api/solver/* for on-demand generation and solving so the
browser can trigger them without a CLI.

Run directly:    python -m grid_editor.server
Run via Docker:  docker compose up
"""
from __future__ import annotations

import base64
import json
import re
import shutil
import warnings
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

DATA_DIR = Path("/data/walls")
RUNS_DIR = Path("/data/runs")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = PROJECT_ROOT / "static"
SCHEMA_PATH = PROJECT_ROOT / "schemas" / "wall.schema.json"
SEED_DIR = PROJECT_ROOT / "data" / "examples"

HOLD_TYPES = {"jug", "crimp", "sloper", "pinch", "foothold"}
SIZES = {"small", "medium", "large"}
WALL_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")

app = Flask(__name__, static_folder=None)


def _wall_path(wall_id: str) -> Path:
    if not WALL_ID_RE.match(wall_id or ""):
        raise ValueError("wall_id must be 1–64 chars: letters, digits, '-' or '_'.")
    return DATA_DIR / f"{wall_id}.json"


def _validate_wall(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return "Wall must be a JSON object."
    if not WALL_ID_RE.match(str(payload.get("wall_id", ""))):
        return "wall_id must be 1–64 chars of letters, digits, '-' or '_'."
    holds = payload.get("holds")
    if not isinstance(holds, list):
        return "holds must be an array."

    seen_ids: set[str] = set()
    seen_cells: set[tuple[int, int]] = set()
    for i, h in enumerate(holds):
        err = _validate_hold(h, i, seen_ids, seen_cells)
        if err:
            return err
    return None


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_number(v: Any) -> bool:
    return (isinstance(v, int) or isinstance(v, float)) and not isinstance(v, bool)


def _validate_hold(
    h: Any, i: int, seen_ids: set[str], seen_cells: set[tuple[int, int]]
) -> str | None:
    if not isinstance(h, dict):
        return f"holds[{i}] must be an object."

    checks = {
        "hold_id": (isinstance(h.get("hold_id"), str), "string"),
        "grid_x": (_is_int(h.get("grid_x")), "integer"),
        "grid_y": (_is_int(h.get("grid_y")), "integer"),
        "hold_type": (isinstance(h.get("hold_type"), str), "string"),
        "orientation_deg": (_is_number(h.get("orientation_deg")), "number"),
        "size": (isinstance(h.get("size"), str), "string"),
        "color": (isinstance(h.get("color"), str), "string"),
        "is_start": (_is_bool(h.get("is_start")), "boolean"),
        "is_finish": (_is_bool(h.get("is_finish")), "boolean"),
    }
    for field, (ok, kind) in checks.items():
        if field not in h:
            return f"holds[{i}] missing '{field}'."
        if not ok:
            return f"holds[{i}].{field} must be {kind}."

    if h["hold_type"] not in HOLD_TYPES:
        return f"holds[{i}].hold_type must be one of {sorted(HOLD_TYPES)}."
    if h["size"] not in SIZES:
        return f"holds[{i}].size must be one of {sorted(SIZES)}."
    if not 0 <= float(h["orientation_deg"]) < 360:
        return f"holds[{i}].orientation_deg must be in [0, 360)."
    if h["grid_x"] < 0 or h["grid_y"] < 0:
        return f"holds[{i}] grid coords must be non-negative."

    if h["hold_id"] in seen_ids:
        return f"holds[{i}].hold_id '{h['hold_id']}' is duplicated."
    seen_ids.add(h["hold_id"])

    cell = (h["grid_x"], h["grid_y"])
    if cell in seen_cells:
        return f"holds[{i}] cell {cell} already occupied."
    seen_cells.add(cell)
    return None


@app.get("/")
def index() -> Any:
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/static/<path:filename>")
def static_files(filename: str) -> Any:
    return send_from_directory(STATIC_DIR, filename)


@app.get("/api/schema")
def get_schema() -> Any:
    if SCHEMA_PATH.exists():
        return send_from_directory(SCHEMA_PATH.parent, SCHEMA_PATH.name)
    return jsonify({"error": "schema not found"}), 404


@app.get("/api/walls")
def list_walls() -> Any:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ids = sorted(p.stem for p in DATA_DIR.glob("*.json"))
    return jsonify({"walls": ids})


@app.get("/api/walls/<wall_id>")
def load_wall(wall_id: str) -> Any:
    try:
        path = _wall_path(wall_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not path.exists():
        return jsonify({"error": "wall not found"}), 404
    return jsonify(json.loads(path.read_text(encoding="utf-8")))


@app.put("/api/walls/<wall_id>")
def save_wall(wall_id: str) -> Any:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "request body must be a JSON object"}), 400

    payload.setdefault("wall_id", wall_id)
    if payload.get("wall_id") != wall_id:
        return jsonify({"error": "wall_id in URL and body must match"}), 400

    err = _validate_wall(payload)
    if err:
        return jsonify({"error": err}), 400

    try:
        path = _wall_path(wall_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return jsonify({"wall_id": wall_id, "saved": True})


@app.delete("/api/walls/<wall_id>")
def delete_wall(wall_id: str) -> Any:
    try:
        path = _wall_path(wall_id)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not path.exists():
        return jsonify({"error": "wall not found"}), 404
    path.unlink()
    return jsonify({"wall_id": wall_id, "deleted": True})


@app.get("/healthz")
def healthz() -> Any:
    return jsonify({"ok": True})


# ── Solver API ────────────────────────────────────────────────────────────────

@app.get("/solver")
def solver_page() -> Any:
    return send_from_directory(STATIC_DIR, "solver.html")


@app.post("/api/solver/generate")
def solver_generate() -> Any:
    """Generate one synthetic wall, save it to /data/walls/, return its JSON."""
    body = request.get_json(silent=True) or {}
    try:
        cols       = int(body.get("cols", 12))
        rows       = int(body.get("rows", 18))
        difficulty = float(body.get("difficulty", 0.5))
        seed       = body.get("seed")
        seed       = int(seed) if seed is not None else None
        name       = body.get("name", "").strip() or None
    except (TypeError, ValueError) as e:
        return jsonify({"error": str(e)}), 400

    if name and not WALL_ID_RE.match(name):
        return jsonify({"error": "Name must be 1–64 chars: letters, digits, '-' or '_'."}), 400

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solver.generate import GeneratorConfig, generate_wall
        from solver.body import BodyModel

    cfg = GeneratorConfig(cols=cols, rows=rows, difficulty=difficulty, seed=seed)
    wall_dict = generate_wall(cfg, BodyModel(), wall_id=name)
    if wall_dict is None:
        return jsonify({"error": "Generator failed after all retries."}), 500

    # If a name was given but a file already exists, reject rather than silently overwrite.
    wall_id = wall_dict["wall_id"]
    if name and (DATA_DIR / f"{wall_id}.json").exists():
        return jsonify({"error": f"A wall named '{wall_id}' already exists."}), 409

    # Save to /data/walls/ so the editor can load it immediately.
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wall_id = wall_dict["wall_id"]
    path = DATA_DIR / f"{wall_id}.json"
    path.write_text(json.dumps(wall_dict, indent=2), encoding="utf-8")

    return jsonify({"wall_id": wall_id, "wall": wall_dict})


@app.post("/api/solver/solve")
def solver_solve() -> Any:
    """Solve a wall with A* and return the move list + a base64-encoded PNG."""
    body = request.get_json(silent=True) or {}
    wall_id = body.get("wall_id")
    if not wall_id:
        return jsonify({"error": "wall_id required"}), 400

    try:
        path = _wall_path(str(wall_id))
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    if not path.exists():
        return jsonify({"error": "wall not found"}), 404

    height_cm   = float(body.get("height_cm", 175.0))
    wingspan_cm = float(body.get("wingspan_cm", 175.0))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from solver.wall import load_wall
        from solver.body import BodyModel
        from solver.astar import solve_astar
        from solver.visualize import render_all_frames

    wall = load_wall(path)
    bmodel = BodyModel(height_cm=height_cm, wingspan_cm=wingspan_cm)
    result = solve_astar(wall, bmodel)

    if result is None:
        return jsonify({"wall_id": wall_id, "solved": False, "moves": []})

    # Render one PNG per pose frame and return as base64 array.
    frames_b64 = [
        base64.b64encode(png).decode()
        for png in render_all_frames(wall, result, body=bmodel)
    ]

    return jsonify({
        "wall_id": wall_id,
        "solved": True,
        "moves": result.text_steps(),
        "n_moves": len(result.moves),
        "expanded": result.expanded,
        "frames": frames_b64,
    })


@app.get("/api/runs")
def list_runs() -> Any:
    """List training run directories and their log snippets."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    runs = []
    for d in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        log_path = d / "training.log"
        log_tail = ""
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8").splitlines()
            log_tail = "\n".join(lines[-20:])
        images = [f.name for f in d.glob("*.png")] + [f.name for f in d.glob("*.gif")]
        runs.append({
            "run_id": d.name,
            "has_best_model": (d / "best_model.zip").exists(),
            "has_final_model": (d / "final_model.zip").exists(),
            "images": sorted(images),
            "log_tail": log_tail,
        })
    return jsonify({"runs": runs})


def _seed_examples() -> None:
    """Copy bundled example walls into /data/walls/ if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SEED_DIR.exists():
        return
    for src in SEED_DIR.glob("*.json"):
        dst = DATA_DIR / src.name
        if not dst.exists():
            shutil.copy(src, dst)


def main() -> None:
    _seed_examples()
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()
