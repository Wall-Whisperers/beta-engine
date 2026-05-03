from __future__ import annotations

import json
from pathlib import Path

from flask import Flask, Response, jsonify, request

app = Flask(__name__)
DATA_DIR = Path("/data/walls")

WALL_EDITOR_HTML = """<!doctype html>
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
