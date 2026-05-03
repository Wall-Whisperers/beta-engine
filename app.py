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

from flask import Flask, Response, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB
WALLS_DIR = Path("/data/walls")

HTML_FORM = """<!doctype html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <title>Beta Engine Docker Demo</title>
  </head>
  <body>
    <h1>Beta Engine – Docker Demo App</h1>
    <p>
      Upload an image and this demo service will convert it to black-and-white
      using Pillow.
    </p>
    <form action=\"/convert\" method=\"post\" enctype=\"multipart/form-data\">
      <input type=\"file\" name=\"image\" accept=\"image/*\" required />
      <button type=\"submit\">Convert</button>
    </form>
  </body>
</html>
"""


def _wall_file(wall_id: str) -> Path:
    safe_id = wall_id.strip()
    if not safe_id or "/" in safe_id or "\\" in safe_id:
        raise ValueError("Invalid wall_id")

    WALLS_DIR.mkdir(parents=True, exist_ok=True)
    return WALLS_DIR / f"{safe_id}.json"


@app.get("/")
def home() -> str:
    return HTML_FORM


@app.post("/convert")
def convert_image() -> Response:
    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        return Response("No file was uploaded.\n", status=400, mimetype="text/plain")

    raw_bytes = uploaded.read()
    if not raw_bytes:
        return Response("Uploaded file is empty.\n", status=400, mimetype="text/plain")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        return Response("File too large. Max size is 10MB.\n", status=413, mimetype="text/plain")

    try:
        image = Image.open(BytesIO(raw_bytes))
    except UnidentifiedImageError:
        return Response("Unsupported image format.\n", status=415, mimetype="text/plain")

    bw_image = image.convert("L")
    output = BytesIO()
    bw_image.save(output, format="PNG")
    output.seek(0)

    filename = uploaded.filename.rsplit(".", maxsplit=1)[0] or "converted"
    return Response(
        output.getvalue(),
        mimetype="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}-bw.png"'},
    )


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
