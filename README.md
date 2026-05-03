# Beta Engine – MVP Collaboration Guide

This repo contains a lightweight Flask app and step-by-step team workflow documentation for the **Beta Engine MVP**, written to be beginner-friendly for first-time Docker users.

---

## 1) MVP scope and non-goals

### MVP scope
The MVP is intentionally narrow and focused on proving end-to-end team flow:

- Upload an image through the web UI.
- Convert the uploaded image into a black-and-white output image.
- Persist wall/hold metadata as JSON records on disk.
- Reload previously saved wall records from local storage.
- Support local development and Docker-based collaboration.

### Non-goals (for MVP)
The following are explicitly out of scope for this phase:

- Multi-tenant authentication/authorization.
- Cloud database infrastructure.
- Real-time collaborative editing.
- Production-grade observability/alerting.
- Mobile-native app clients.
- ML-based automatic hold detection.

---

## 2) JSON schema overview (holds)

The wall model is file-based and JSON-first. Each wall record should contain:

- `wall_id` (string, unique identifier)
- `name` (string)
- `created_at` (ISO-8601 UTC timestamp)
- `updated_at` (ISO-8601 UTC timestamp)
- `reference_image` (object)
- `holds` (array of hold objects)

### Hold object fields

- `id` (string)
- `label` (string)
- `color` (string, e.g. `"blue"`)
- `difficulty` (string enum suggestion: `"easy" | "moderate" | "hard"`)
- `x` (number, normalized coordinate `0.0..1.0`)
- `y` (number, normalized coordinate `0.0..1.0`)
- `notes` (string, optional)

### Sample wall record

```json
{
  "wall_id": "wall-demo-001",
  "name": "Training Wall A",
  "created_at": "2026-05-03T10:00:00Z",
  "updated_at": "2026-05-03T10:05:00Z",
  "reference_image": {
    "filename": "training-wall-a.jpg",
    "width": 1080,
    "height": 1440
  },
  "holds": [
    {
      "id": "H1",
      "label": "start-left",
      "color": "blue",
      "difficulty": "easy",
      "x": 0.22,
      "y": 0.81,
      "notes": "Large jug"
    },
    {
      "id": "H2",
      "label": "mid-crimp",
      "color": "black",
      "difficulty": "moderate",
      "x": 0.48,
      "y": 0.52,
      "notes": "Small crimp"
    }
  ]
}
```

---

## 3) Local run instructions and `docker-compose` flow (team of 3)

## Local run (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open: http://localhost:8000

## Docker run (single container)

```bash
docker build -t beta-engine:demo .
docker run --rm -p 8000:8000 --name beta-engine-demo beta-engine:demo
```

Open: http://localhost:8000

## `docker-compose` team flow (3 collaborators)

Use one shared branch and consistent runtime commands per developer.

1. Developer A/B/C each pull latest `dev`.
2. Each developer runs the same compose flow locally on their own machine:

```bash
docker compose up --build
```

3. Open the app at `http://localhost:8000`.
4. Stop and remove containers:

```bash
docker compose down
```

> Note: if `docker-compose.yml` is not yet present, add it before using the compose workflow above.

---

## 4) Save/load behavior and `/data/walls/` storage rules

Wall persistence is file-based for MVP.

### Save behavior

- On save, write one JSON file per wall to `/data/walls/`.
- File naming convention: `<wall_id>.json` (example: `wall-demo-001.json`).
- Save operations should be idempotent: same `wall_id` overwrites prior record.
- Always update `updated_at` on write.

### Load behavior

- Load by `wall_id` from `/data/walls/<wall_id>.json`.
- Return a not-found response when file is missing.
- Validate JSON shape before returning data to clients.

### Storage rules

- `/data/walls/` is the canonical directory for wall metadata.
- Keep files UTF-8 encoded JSON.
- Do not store binaries in `/data/walls/` (images belong in a separate image directory/object store).
- Treat `/data/walls/` as durable volume-mounted storage when running in Docker.

---

## 5) Branch strategy and PR expectations

We use a simple three-lane branching model:

- `main`: production-ready history only.
- `dev`: integration branch for approved features.
- `feature/*`: short-lived branches for task-level work (e.g., `feature/save-wall-json`).

### Standard flow

1. Branch from `dev`.
2. Implement in `feature/*`.
3. Open PR into `dev`.
4. After validation and review, merge `dev` into `main` on release.

### PR expectations

Every PR should include:

- Clear scope summary (what changed, what did not).
- Linked issue/task reference.
- Manual validation notes (commands + observed results).
- Screenshots for UI-visible changes.
- Confirmation that persistence rules (`/data/walls/`) were preserved where relevant.

---

## 6) Quick manual validation checklist

Use this checklist before opening a PR.

1. Start app locally or in Docker.
2. Upload a reference wall photo.
3. Confirm black-and-white conversion returns a downloadable PNG.
4. Create/recreate a sample wall record from the reference photo:
   - define 5–10 holds,
   - set normalized `x/y` coordinates,
   - save JSON under `/data/walls/<wall_id>.json`.
5. Reload that wall by `wall_id` and verify hold positions/metadata match saved values.
6. Restart app and verify saved wall still loads.
7. Run a second save with same `wall_id` and confirm update/overwrite behavior is correct.

---

## Helpful Docker commands

```bash
docker ps
docker ps -a
docker images
docker logs -f beta-engine-demo
docker system prune
```

Then open `http://localhost:8000`.

---

## Wall data schema (source of truth)

For climbing wall and hold payloads, the canonical schema is:

- `schemas/wall.schema.json`

Team examples that should validate against this schema:

- `data/examples/wall.example.minimal.json`
- `data/examples/wall.example.with-metadata.json`
