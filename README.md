# Beta Engine – Docker Demo (Image to Black-and-White)

This repository is designed as a **clear Docker demo** for collaborators across macOS, Windows, and Linux.

The demo app:
- runs a small Flask web server,
- accepts image uploads,
- converts images to black-and-white using Pillow,
- returns a downloadable PNG,
- supports saving/loading wall JSON files in `/data/walls`.

---

## Data storage convention

Wall files are stored using this on-disk layout:

- `/data/walls/<wall_id>.json`

API behavior:
- `PUT /walls/<wall_id>` writes the JSON request body to `/data/walls/<wall_id>.json`.
- `GET /walls/<wall_id>` reads and returns `/data/walls/<wall_id>.json`.

---

## Run with Docker Compose

Use Compose to run the app with a named volume mounted at `/data`:

```bash
docker compose up --build
```

This uses `docker-compose.yml` and mounts the `shared_data` volume so wall JSON files persist across container restarts.

App URL:
- http://localhost:8000

Stop:

```bash
docker compose down
```

---

## Run with Docker (single container)

Build:

```bash
docker build -t beta-engine:demo .
```

Run:

```bash
docker run --rm -p 8000:8000 --name beta-engine-demo beta-engine:demo
```

Then open:
- http://localhost:8000

