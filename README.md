# Beta Engine

A two-part climbing-wall MVP:

| Part | Folder | What it does |
|------|--------|--------------|
| **Grid editor** | `grid_editor/` | Web tool for placing holds on a grid and saving walls as JSON |
| **Solver** | `solver/` | 2D IK + A\*/RL solver that reads those JSON files and outputs a move sequence |

The editor locks the JSON schema everything downstream depends on. The solver
is a first pass at the body-model → reachability → search → visualizer pipeline,
intended to work on hand-built walls before we layer on computer vision, physics,
or a full PPO agent.

For deeper detail on each part see:
- [`grid_editor/README.md`](grid_editor/README.md)
- [`solver/README.md`](solver/README.md)
- [`planning-gabe.md`](planning-gabe.md) — long-term architecture + decisions log
- [`CLAUDE.md`](CLAUDE.md) — team context + build phases

---

## Project layout

```
beta-engine/
├── grid_editor/       # Flask backend + static frontend for the wall editor
│   ├── server.py      #   REST API + static file serving
│   └── README.md      #   schema reference, API docs, editor controls
├── solver/            # 2D IK + A* + Q-learning solver
│   ├── wall.py        #   JSON loader + grid → cm conversion
│   ├── body.py        #   5-point stick figure + 2-link IK
│   ├── reachability.py#   Pose, reach + stability
│   ├── astar.py       #   A* baseline
│   ├── rl_qlearn.py   #   tabular Q-learning
│   ├── visualize.py   #   matplotlib PNG/GIF renderer
│   ├── __main__.py    #   CLI entry point
│   └── README.md      #   solver quickstart + flags + architecture
├── static/            # Vanilla-JS editor frontend (no build step)
├── schemas/           # wall.schema.json — JSON Schema 2020-12 source of truth
├── data/
│   ├── examples/      # Seeded example walls (tracked in git)
│   ├── walls/         # User-saved walls (gitignored)
│   └── runs/          # Solver PNG/GIF outputs (gitignored)
├── planning-gabe.md   # Long-term architecture notes
└── CLAUDE.md          # Team context + multi-phase build plan
```

---

## Quickstart with Docker

You need Docker Desktop (or Docker Engine + Compose v2):

```bash
docker --version
docker compose version
```

### 1) Clone + build

```bash
git clone https://github.com/Wall-Whisperers/beta-engine.git beta-engine
cd beta-engine
docker compose build
```

### 2) Start

```bash
docker compose up -d
```

Open <http://localhost:8000>. The example V2 boulder loads automatically.

### 3) Stop (keep your walls)

```bash
docker compose down
```

Walls persist in `./data/walls/` on your host.

### 4) Full reset

```bash
docker compose down --rmi all
rm -rf data/walls
```

### 5) Run the solver (container must be running)

```bash
# A* — fast, mathematically shortest path
docker compose exec beta-engine python -m solver --wall example-v2-boulder

# Q-learning RL
docker compose exec beta-engine python -m solver --wall example-v2-boulder --method qlearn

# Both + animated GIFs saved to ./data/runs/
docker compose exec beta-engine python -m solver --wall example-v2-boulder --method both --gif
```

See [`solver/README.md`](solver/README.md) for all CLI flags and details.

---

## Running without Docker

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Grid editor (serves on :8000)
python -m grid_editor.server

# Solver
python -m solver --wall example-v2-boulder --method both --gif
```

> The editor hard-codes `/data/walls/` to match the Docker path. For a
> local run, edit `DATA_DIR` at the top of `grid_editor/server.py`.

---

## Branching & PRs

- `main` — protected, always working
- `dev` — integration branch
- `feature/<short-name>` — branch from `dev`, PR back to `dev`

PR descriptions should include: scope summary, screenshots for any UI
change, and the manual validation checklist from
[`grid_editor/README.md`](grid_editor/README.md#manual-validation-checklist).

---

## Helpful Docker commands

```bash
docker compose ps                # running containers
docker compose logs -f           # follow logs
docker compose restart           # restart without rebuild
docker compose build --no-cache  # force clean rebuild
docker system prune              # reclaim disk
```
