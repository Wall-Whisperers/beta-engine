# Grid Editor

A small single-page web tool for defining climbing walls on a grid and saving
them as JSON. The JSON files it produces are the source of truth for the whole
pipeline — the solver reads them directly.

Built with Flask (backend) and vanilla JS / HTML canvas (frontend). No build
step, no bundler.

---

## Using the editor

Open <http://localhost:8000> after `docker compose up -d`.

| Action | How |
|--------|-----|
| Place a hold | Click an empty cell — uses the current palette settings |
| Select a hold | Click an occupied cell |
| Edit a selected hold | Change the palette, then click **Apply to selected** |
| Remove a hold | Select it → **Remove selected** (or press <kbd>Del</kbd>) |
| Rotate selected +15° | Press <kbd>R</kbd> (or right-click any hold) |
| Move selection | Arrow keys |
| Clone a hold | Select it, then shift-click an empty cell |
| Resize the grid | Set cols/rows → **Resize** (out-of-bounds holds are dropped) |
| Save / Load / Delete | Top bar — **Save** writes `data/walls/<wall_id>.json` |
| Hand-edit JSON | Right panel — edit, then **Apply JSON** |
| Export / Import | **Download .json** / **Upload** |

### Climbing conventions baked in

- Grid origin (0, 0) is **bottom-left** so increasing `grid_y` goes up the wall.
- `orientation_deg` 0° points up, increases clockwise. For a sidepull crimp
  pulling down-right use ~315°; for a horizontal pinch use 90°.
- Start holds get a green ring; finish holds get a red ring.
- Default hold-type colors: jug (green) · crimp (red) · sloper (orange) · pinch (blue) · foothold (purple)

---

## Hold JSON schema

The full schema lives in `schemas/wall.schema.json` (JSON Schema 2020-12).
**Do not change the schema without team discussion** — it is the contract
everything downstream depends on.

### Hold object

```json
{
  "hold_id": "h_001",
  "grid_x": 3,
  "grid_y": 1,
  "hold_type": "jug",
  "orientation_deg": 0.0,
  "size": "small",
  "color": "#22c55e",
  "is_start": true,
  "is_finish": false
}
```

| Field | Type | Notes |
|-------|------|-------|
| `hold_id` | string | Unique within the wall. Editor auto-assigns `h_001`, `h_002`, … |
| `grid_x` | integer ≥ 0 | Column. 0 = left edge. |
| `grid_y` | integer ≥ 0 | Row. **0 = bottom** (climbing convention). |
| `hold_type` | `jug` \| `crimp` \| `sloper` \| `pinch` \| `foothold` | Color-coded in the UI. |
| `orientation_deg` | number in `[0, 360)` | 0° = up; clockwise. Direction of pull / grip axis. |
| `size` | `small` \| `medium` \| `large` | |
| `color` | string | Hex `#RRGGBB` or any color name. |
| `is_start` | boolean | Green ring in the UI. |
| `is_finish` | boolean | Red ring in the UI. |

### Wall file

```json
{
  "wall_id": "my-wall",
  "name": "optional display name",
  "grid": { "cols": 10, "rows": 14, "cell_size_cm": 20.0 },
  "wall_angle_deg": 0.0,
  "surface_friction": 0.7,
  "holds": [ ]
}
```

`wall_id` must match `^[A-Za-z0-9_\-]{1,64}$` — it doubles as the filename.
`grid.cell_size_cm` is optional for older files; loaders default to 20 cm and
warn if it is absent. `wall_angle_deg` and `surface_friction` are optional
physics/sim3d fields used by the downstream simulators.

---

## REST API

The backend exposes a small JSON API under `/api/walls`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Editor UI (serves `static/index.html`) |
| GET | `/api/walls` | List all saved wall IDs |
| GET | `/api/walls/<wall_id>` | Load one wall |
| PUT | `/api/walls/<wall_id>` | Save (creates or overwrites). Body must pass validation. |
| DELETE | `/api/walls/<wall_id>` | Delete |
| GET | `/api/schema` | Returns `schemas/wall.schema.json` |
| GET | `/healthz` | `{ "ok": true }` (Docker healthcheck) |

Quick smoke test (container running):

```bash
curl -s http://localhost:8000/api/walls
curl -s http://localhost:8000/api/walls/example-v2-boulder
```

---

## Implementation notes

- `grid_editor/server.py` — Flask app. Validates every wall on save
  (hold type, size, orientation, no duplicate IDs or cells). Reads/writes
  to `/data/walls/` so Docker bind-mount makes them appear on the host.
- `static/` — vanilla JS + HTML canvas. No build step; served directly by
  Flask.
- `data/examples/` — seeded walls tracked in git. Copied into
  `/data/walls/` on first start if missing.

---

## Manual validation checklist

Run this before opening a PR that touches the editor.

1. `docker compose up -d`, open <http://localhost:8000>.
2. The example V2 boulder loads automatically.
3. Place 2–3 new holds, set start/finish flags, rotate one hold.
4. Type a new `Wall ID` and click **Save**.
5. Confirm `data/walls/<wall_id>.json` exists on the host.
6. Click **New**, pick the saved wall from the dropdown, **Load** it.
7. `docker compose down && docker compose up -d` — wall still loads.
8. Re-save with the same `wall_id` — confirm it overwrites cleanly.
9. Run the solver against the saved wall and check the output makes sense:

```bash
docker compose exec beta-engine python -m solver --wall <wall_id> --method both --gif
```
