from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
DATA_DIR = Path("/data/walls")

WALL_EDITOR_HTML = """<!doctype html>
"""Demo-only image conversion service.

This file is intentionally a little more involved than a hello-world script so
new Docker users can quickly see the value of packaging dependencies in one
portable container.

What this demo does:
- starts a small Flask web server,
- accepts an uploaded image,
- converts it to black-and-white with Pillow,
- returns the converted image for download,
- provides simple wall JSON save/load endpoints backed by /data/walls.

This is not intended as production-ready code. It is a teaching/demo app.
"""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path

from flask import Flask, Response, request, send_from_directory
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
WALLS_DIR = Path("/data/walls")

@app.get('/')
def home() -> Response:
    return send_from_directory('static', 'index.html')
HTML_FORM = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Wall Editor</title>
  </head>
  <body>
    <h1>Wall Editor</h1>
    <p>Wall editor UI placeholder for MVP backend integration.</p>
  </body>
</html>
"""

REQUIRED_HOLD_FIELDS = {
    "hold_id": str,
    "grid_x": int,
    "grid_y": int,
    "hold_type": str,
    "orientation_deg": (int, float),
    "size": (int, float),
    "color": str,
    "is_start": bool,
    "is_finish": bool,
}


def _wall_file(wall_id: str) -> Path:
    safe_id = wall_id.strip()
    if not safe_id or "/" in safe_id or "\\" in safe_id:
        raise ValueError("Invalid wall_id")

    WALLS_DIR.mkdir(parents=True, exist_ok=True)
    return WALLS_DIR / f"{safe_id}.json"


@app.get("/")
def home() -> str:
    return WALL_EDITOR_HTML


def _validate_hold(hold: object, index: int) -> str | None:
    if not isinstance(hold, dict):
        return f"Hold at index {index} must be an object."

    for field, expected_type in REQUIRED_HOLD_FIELDS.items():
        if field not in hold:
            return f"Hold at index {index} is missing required field '{field}'."
        if not isinstance(hold[field], expected_type):
            return f"Hold at index {index} field '{field}' has invalid type."

    return None


@app.post("/api/walls")
def save_wall() -> Response:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object."}), 400

    wall_id = payload.get("wall_id")
    holds = payload.get("holds")

    if not isinstance(wall_id, str) or not wall_id.strip():
        return jsonify({"error": "'wall_id' is required and must be a non-empty string."}), 400
    if not isinstance(holds, list):
        return jsonify({"error": "'holds' is required and must be an array."}), 400

    for index, hold in enumerate(holds):
        error = _validate_hold(hold, index)
        if error:
            return jsonify({"error": error}), 400

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    wall_path = DATA_DIR / f"{wall_id}.json"

    with wall_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    return jsonify({"wall_id": wall_id, "saved": True})


@app.get("/api/walls/<wall_id>")
def load_wall(wall_id: str) -> Response:
    wall_path = DATA_DIR / f"{wall_id}.json"
    if not wall_path.exists():
        return jsonify({"error": "Wall not found."}), 404

    with wall_path.open("r", encoding="utf-8") as fp:
        wall_data = json.load(fp)

    return jsonify(wall_data)


@app.put("/walls/<wall_id>")
def save_wall(wall_id: str) -> Response:
    payload = request.get_json(silent=True)
    if payload is None:
        return Response("Request body must be valid JSON.\n", status=400, mimetype="text/plain")

    try:
        wall_file = _wall_file(wall_id)
    except ValueError:
        return Response("Invalid wall_id.\n", status=400, mimetype="text/plain")

    wall_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return Response(status=204)


@app.get("/walls/<wall_id>")
def load_wall(wall_id: str) -> Response:
    try:
        wall_file = _wall_file(wall_id)
    except ValueError:
        return Response("Invalid wall_id.\n", status=400, mimetype="text/plain")

    if not wall_file.exists():
        return Response("Wall not found.\n", status=404, mimetype="text/plain")

    return Response(wall_file.read_text(encoding="utf-8"), mimetype="application/json")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
