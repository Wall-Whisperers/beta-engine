# Beta Engine

A two-part climbing-wall MVP:

1. **Grid editor** (`grid_editor/`) — a small web tool for defining climbing
   walls on a grid. Click cells to place holds, set type / orientation /
   size / color / start-finish flags, then save the wall as JSON. Saved
   walls live as plain `*.json` files under `data/walls/`.
2. **Solver** (`solver/`) — a basic 2D inverse-kinematics + reinforcement-
   learning solver that takes a wall JSON and produces a move sequence
   (which limb goes to which hold), with a matplotlib visualization.

This is a Phase 1 + early Phase 2/3 MVP. The grid editor locks the JSON
schema everything downstream depends on; the solver is a deliberately
small first pass to prove the body-model → reachability → search →
visualizer pipeline works on hand-built walls before we layer on
photogrammetry, physics, or PPO.

---

## Layout

```
beta-engine/
├── grid_editor/       # Flask backend for the wall editor (Phase 1)
│   └── server.py      #   was: app.py
├── solver/            # 2D IK + reachability + A* + tabular Q-learning (Phase 2/3)
│   ├── wall.py        #   loads wall JSON, world-coord conversion
│   ├── body.py        #   BodyModel + closed-form 2-link IK
│   ├── reachability.py#   Pose, reach + stability checks
│   ├── astar.py       #   A* baseline solver
│   ├── rl_qlearn.py   #   tabular Q-learning RL solver
│   ├── visualize.py   #   matplotlib renderer (PNG + GIF)
│   └── __main__.py    #   CLI: python -m solver --wall <id>
├── static/            # Vanilla-JS editor frontend
├── schemas/           # wall.schema.json — source of truth
├── data/
│   ├── examples/      # Seeded example walls (tracked)
│   ├── walls/         # User-saved walls (gitignored)
│   └── runs/          # Solver outputs (gitignored)
├── planning-gabe.md   # Long-term architecture / Gabe's brainstorms
└── CLAUDE.md          # Team context + build phases
```

## What's in the box

- **Backend** — Flask (Python 3.12). Validates wall JSON, persists to `/data/walls/`.
- **Frontend** — single-page editor (vanilla JS, no build step).
- **Schema** — `schemas/wall.schema.json` (JSON Schema 2020-12, source of truth).
- **Solver** — pure-Python (NumPy + matplotlib) 2D IK + A* + Q-learning.
- **Example wall** — `data/examples/example-v2-boulder.json` auto-seeds on first
  start so you see a real boulder problem immediately.
- **Docker** — single container with bind-mounted `./data` so wall files and
  solver outputs appear on your host filesystem.

---

## Hold JSON schema

```json
{
  "hold_id": "h_001",
  "grid_x": 12,
  "grid_y": 24,
  "hold_type": "crimp",
  "orientation_deg": 47.5,
  "size": "small",
  "color": "#ef4444",
  "is_start": false,
  "is_finish": false
}
```

| Field             | Type                                                           | Notes |
|-------------------|----------------------------------------------------------------|-------|
| `hold_id`         | string                                                         | Unique within the wall. Editor auto-assigns `h_001`, `h_002`, … |
| `grid_x`          | integer ≥ 0                                                    | Column. 0 = left edge of wall. |
| `grid_y`          | integer ≥ 0                                                    | Row. **0 = bottom of wall** (climbing convention). |
| `hold_type`       | `jug` \| `crimp` \| `sloper` \| `pinch` \| `foothold`           | Color-coded in the UI. |
| `orientation_deg` | number in `[0, 360)`                                           | 0° = up; clockwise. Indicates pull / pinch-axis direction. |
| `size`            | `small` \| `medium` \| `large`                                  | |
| `color`           | string                                                         | Hex `#RRGGBB` or any color name (route color or physical hold color). |
| `is_start`        | boolean                                                        | Marked with a green ring. |
| `is_finish`       | boolean                                                        | Marked with a red ring. |

A full wall file looks like:

```json
{
  "wall_id": "my-wall",
  "name": "optional name",
  "grid": { "cols": 10, "rows": 14 },
  "holds": [ /* one or more hold objects */ ]
}
```

`wall_id` must match `^[A-Za-z0-9_\-]{1,64}$` (it's used as a filename).

---

## Quickstart with Docker

You need Docker Desktop (or Docker Engine + Compose v2). Verify with:

```bash
docker --version
docker compose version
```

### 1) Initial setup (clone + build)

```bash
git clone https://github.com/Wall-Whisperers/beta-engine.git beta-engine
cd beta-engine
docker compose build
```

### 2) Start running

```bash
docker compose up -d
```

Open <http://localhost:8000>. You should land in the editor with the example
V2 boulder loaded.

Tail logs while it runs:

```bash
docker compose logs -f
```

### 3) Close it down (keep the data)

```bash
docker compose down
```

Your saved walls persist in `./data/walls/` on your host — reopen the app any
time and they're still there.

### 4) Clean up (full reset)

Remove the container, its image, and any wall files you've saved:

```bash
docker compose down --rmi all
rm -rf data/walls
```

The seeded example will be re-copied into `data/walls/` the next time the app
starts.

### 5) Re-run

```bash
docker compose up -d
```

### 6) Stop again

```bash
docker compose down
```

---

## Running without Docker

```bash
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m grid_editor.server      # serves the editor on :8000
```

> Note: the editor hard-codes `/data/walls/` to match the Docker path. For
> a local run, either run inside Docker (recommended) or temporarily edit
> `DATA_DIR` at the top of `grid_editor/server.py`.

Run the solver locally:

```bash
python -m solver --wall example-v2-boulder --method both --gif
```

---

## Using the editor

| Action                         | How |
|--------------------------------|-----|
| Place a hold                   | Click an empty cell — uses the current palette settings |
| Select a hold                  | Click an occupied cell |
| Edit a selected hold           | Change the palette, then click **Apply to selected** |
| Remove a hold                  | Select it → **Remove selected** (or press <kbd>Del</kbd>) |
| Rotate selected by +15°        | Press <kbd>R</kbd> (or right-click any hold) |
| Move selection                 | Arrow keys |
| Clone a hold                   | Select it, then shift-click an empty cell |
| Resize the grid                | Set cols/rows → **Resize** (out-of-bounds holds are dropped) |
| Save / Load / Delete           | Top bar (`Save` writes `data/walls/<wall_id>.json`) |
| Hand-edit JSON                 | Right panel — edit, then **Apply JSON** |
| Export / Import a `.json` file | **Download .json** / **Upload** |

### Climbing notes baked into the editor

- Grid origin (0, 0) is **bottom-left** so increasing `grid_y` goes up the wall.
- Orientation 0° points up. For a sidepull crimp pulling down-right, use ~315°.
  For a horizontal pinch, use 90°.
- Start holds get a green ring; finish holds get a red ring; both = both rings.
- Hold-type colors used by default:
  - jug 🟢 · crimp 🔴 · sloper 🟠 · pinch 🔵 · foothold 🟣

---

## Solver

The solver consumes a wall JSON (the same files the editor produces) and
outputs a move sequence — which limb moves to which hold, in order — plus
a matplotlib visualization.

### Quickstart

With the container already running (`docker compose up -d`):

```bash
# A* baseline
docker compose exec beta-engine python -m solver --wall example-v2-boulder

# Tabular Q-learning RL
docker compose exec beta-engine python -m solver --wall example-v2-boulder --method qlearn --episodes 2000

# Both, with an animated GIF
docker compose exec beta-engine python -m solver --wall example-v2-boulder --method both --gif
```

Outputs land in `./data/runs/<wall_id>-<method>.{png,gif}` on the host.

### CLI flags

| Flag                  | Default | Notes                                                                 |
|-----------------------|---------|-----------------------------------------------------------------------|
| `--wall <id\|path>`   | —       | wall_id (resolves to `/data/walls/<id>.json` or the seeded examples) or a path to a JSON file. |
| `--method`            | `astar` | `astar`, `qlearn`, or `both`.                                         |
| `--height-cm`         | `175`   | Climber height (drives leg length).                                   |
| `--wingspan-cm`       | `175`   | Climber wingspan (drives arm length).                                 |
| `--cell-size-cm`      | `20`    | Override grid cell size in cm. The schema does not yet carry this — see `planning-gabe.md`. |
| `--episodes`          | `2000`  | Q-learning episodes (only used by `qlearn` / `both`).                 |
| `--no-viz`            | off     | Skip image rendering, just print the move list.                       |
| `--gif`               | off     | Also render an animated GIF.                                          |

### How it works (one paragraph each)

- **Body model (`solver/body.py`).** A 5-point stick figure: COM + four
  end-effectors (LH, RH, LF, RF). Each limb is a 2-link chain (upper +
  lower, equal length) solved with a closed-form law-of-cosines IK —
  exactly the recipe in `CLAUDE.md`. Defaults to a 175-cm climber with a
  175-cm wingspan.
- **Reachability (`solver/reachability.py`).** A pose is the four hold IDs
  each limb sits on. From any pose we enumerate one-limb moves: a candidate
  move is legal if (a) the moving limb's target is within ~92 % of its
  max reach from its body anchor, (b) the three-point intermediate pose is
  stable, and (c) the resulting pose is stable. "Stable" on a vertical wall
  means the COM x-coord lies between the two foot x-coords.
- **A\* baseline (`solver/astar.py`).** Heuristic: vertical distance from
  the higher hand to the nearest finish hold. Cost: 1 per move + a small
  positivity penalty (worse holds cost more). Multi-source: all plausible
  starting poses get pushed onto the open set.
- **Q-learning (`solver/rl_qlearn.py`).** A dependency-free tabular agent
  over the `(LH, RH, LF, RF)` state space. Reward: progress toward the
  finish, completion bonus, efficiency penalty, stability penalty —
  matches the shape in `planning-gabe.md` §3.2. The env class is
  Gym-shaped so we can swap to PPO + Stable Baselines3 later without
  re-plumbing.
- **Visualizer (`solver/visualize.py`).** Headless matplotlib (Agg
  backend). Renders one panel per pose plus an optional animated GIF.

### Known limits (intentional, MVP)

- **Vertical wall only.** No slab/overhang angle, no friction model.
- **Single-wall RL.** Tabular Q-learning over one wall; no procedural
  generation, no transfer. Fine for ~13 holds, will explode beyond ~20.
- **No real-world scale in the schema.** The solver hard-defaults
  `cell_size_cm = 20` cm with a warning. See `planning-gabe.md`.

---

## API

| Method | Path                       | Purpose |
|--------|----------------------------|---------|
| GET    | `/`                        | Editor UI |
| GET    | `/api/walls`               | List saved wall IDs |
| GET    | `/api/walls/<wall_id>`     | Load one wall |
| PUT    | `/api/walls/<wall_id>`     | Save (creates or overwrites). Body must validate. |
| DELETE | `/api/walls/<wall_id>`     | Delete |
| GET    | `/api/schema`              | Returns `wall.schema.json` |
| GET    | `/healthz`                 | `{ "ok": true }` (used by Docker healthcheck) |

Quick `curl` smoke test (with the app running):

```bash
curl -s http://localhost:8000/api/walls
curl -s http://localhost:8000/api/walls/example-v2-boulder | head
```

---

## Manual validation checklist

Use before opening a PR.

1. `docker compose up -d` and open <http://localhost:8000>.
2. The example V2 boulder loads automatically.
3. Place 2-3 new holds, set start/finish, change orientation.
4. Type a new `Wall ID` and click **Save**.
5. Confirm the new file appears at `data/walls/<wall_id>.json`.
6. Click **New**, then pick the saved wall from the dropdown and **Load**.
7. `docker compose down` and `docker compose up -d` again — the wall still loads.
8. Re-save with the same `wall_id`; confirm it overwrites cleanly.
9. Run the solver against the saved wall:
   `docker compose exec beta-engine python -m solver --wall <wall_id> --method both --gif`
10. Confirm `data/runs/<wall_id>-astar.png` and `…-qlearn.png` look sensible
    (start pose first, finish pose last, stick figure stays inside the wall).

---

## Branching & PRs

- `main` — production-ready history.
- `dev` — integration branch.
- `feature/<short-name>` — branch from `dev` for each task; PR back to `dev`.

PR description should include: scope summary, screenshots for any UI change,
and the manual validation checklist above with results.

---

## Helpful Docker commands

```bash
docker compose ps             # running containers for this project
docker compose logs -f        # follow logs
docker compose restart        # restart without rebuilding
docker compose build --no-cache  # force a clean rebuild
docker system prune           # reclaim disk (safe on a dev machine)
```
