"""Demo-only image conversion service.

This file is intentionally a little more involved than a hello-world script so
new Docker users can quickly see the value of packaging dependencies in one
portable container.

What this demo does:
- starts a small Flask web server,
- accepts an uploaded image,
- converts it to black-and-white with Pillow,
- returns the converted image for download.

This is not intended as production-ready code. It is a teaching/demo app.
"""

from __future__ import annotations

from io import BytesIO

from flask import Flask, Response, request
from PIL import Image, UnidentifiedImageError

app = Flask(__name__)
MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB

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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
