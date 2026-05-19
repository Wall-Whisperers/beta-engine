# Beta Engine

An RL agent that learns to climb a MoonBoard by directly controlling a custom
27-DOF humanoid in MuJoCo. Each policy step emits joint targets for every
actuated joint plus four per-limb grip intents. The agent must discover
climbing from proprioception plus the 3D positions of the holds around it.

| Part            | Folder         | What it does |
|-----------------|----------------|--------------|
| **Grid editor** | `grid_editor/` | Flask + vanilla-JS web tool to place holds on a grid and save walls as JSON |
| **Solver (2D)** | `solver/`      | Legacy 2D IK + A\* solver, used now only for the editor's planning preview |
| **Sim 3D**      | `sim3d/`       | MuJoCo 3D simulator + Gymnasium env + Stable-Baselines3 PPO trainer |
| **Schemas**     | `schemas/`     | JSON Schema for the wall format |
| **Static**      | `static/`      | Editor + sim3d browser frontend (no build step) |
| **Data**        | `data/`        | Walls, MoonBoard problem corpus, and training run outputs |

The editor locks the JSON schema everything downstream depends on. The 2D
solver was the bootstrap path; **the trained RL stack is now `sim3d/`**, with
a fixed-shape observation (127,) and action (25,) regardless of wall size.

See [`CLAUDE.md`](CLAUDE.md) for the architectural contract, action and
observation byte layouts, reward function, and "what not to do" list.

For deeper detail on each subpackage:

- [`grid_editor/README.md`](grid_editor/README.md)
- [`solver/README.md`](solver/README.md)
- [`sim3d/README.md`](sim3d/README.md)

---

## Project layout

```
beta-engine/
├── grid_editor/         # Flask backend + REST API for the wall editor
│   └── server.py
├── solver/              # 2D IK + A* (used by the editor's "solve" preview)
│   ├── wall.py          # JSON loader + grid → cm conversion
│   ├── body.py          # 5-point stick figure + 2-link IK
│   ├── reachability.py
│   ├── astar.py
│   ├── rl_qlearn.py     # tabular Q-learning reference
│   └── visualize.py
├── sim3d/               # MuJoCo 3D world + Gymnasium env + PPO trainer
│   ├── config.py        # all physics + reward constants
│   ├── body.py          # ClimberProfile + segment math + limb names
│   ├── builder.py       # build_mjcf_xml(wall, profile, include_kickboard)
│   ├── world.py         # Climb3DWorld — MjModel/MjData + grip + slip
│   ├── obs.py           # build_observation — fixed-shape (127,) obs
│   ├── env.py           # Climbing3DEnv — Gymnasium wrapper
│   ├── moonboard_env.py # MoonboardClimbing3DEnv — samples problems per reset
│   ├── moonboard.py     # MoonBoard problem JSON → Wall adapter
│   ├── callbacks.py     # VideoRolloutCallback (mp4 every N steps)
│   ├── train.py         # SB3 PPO trainer + episode CSV logger
│   ├── viewer.py        # native MuJoCo viewer wrapper
│   ├── web.py           # Flask blueprint for the three.js front-end
│   └── __main__.py      # `python -m sim3d` CLI (incl. --play <model.zip>)
├── schemas/             # wall.schema.json
├── static/              # vanilla-JS editor + sim3d viewer
└── data/
    ├── examples/        # seeded example walls (in git)
    ├── walls/           # user-saved walls (gitignored)
    ├── moonboard/       # MoonBoard problem corpus
    └── runs/sim3d/      # PPO run outputs (gitignored)
```

The retired `physics/`, `rl/`, and standalone `moonboard-rl/` packages have
been merged into `sim3d/` and removed from the tree.

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

Open <http://localhost:8000> — the example V2 boulder loads automatically.
The 3D viewer is at <http://localhost:8000/sim3d/>.

### 3) Stop (keep your walls)

```bash
docker compose down
```

Walls persist in `./data/walls/` on the host.

### 4) Run the 2D editor solver preview

```bash
docker compose exec beta-engine \
  python -m solver --wall example-v2-boulder --method both --gif
```

### 5) Step a Climbing3DEnv from the CLI

```bash
# Native MuJoCo viewer with the bundled example wall.
docker compose exec beta-engine python -m sim3d

# MoonBoard problem in the native viewer.
docker compose exec beta-engine python -m sim3d \
  --moonboard data/moonboard/sample-problems.json --problem 19215

# Headless smoke test.
docker compose exec beta-engine python -m sim3d --headless --frames 120

# Random Gym episode (env wrapper smoke test).
docker compose exec beta-engine python -m sim3d --gym --gym-episodes 3
```

### 6) Train an RL agent

```bash
# Smoke (~1 min on CPU).
docker compose exec beta-engine \
  python -m sim3d.train --steps 1500 --run-id smoke

# Real run on a MoonBoard problem.
docker compose exec beta-engine \
  python -m sim3d.train \
    --moonboard data/moonboard/sample-problems.json \
    --problem 19215 --steps 200_000 --run-id mb_v1

# Multi-worker rollouts + GPU policy update.
docker compose exec beta-engine \
  python -m sim3d.train --steps 200_000 --n-envs 8 --device cuda --run-id fast
```

Outputs land in `data/runs/sim3d/<run_id>/`:

```
config.json          # training config (wall, profile, hyperparams)
episode_stats.csv    # one row per episode (reward, length, outcome, slips…)
tb/                  # TensorBoard event files
model.zip            # the trained PPO policy
```

### 7) Replay a trained policy

```bash
# Native viewer:
python -m sim3d --play data/runs/sim3d/mb_v1/model.zip \
                --moonboard data/moonboard/sample-problems.json --problem 19215

# Browser (Three.js): start the editor process and paste the run dir
# into "RL policy replay" at http://localhost:8000/sim3d/.
```

---

## Running without Docker

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Editor + 3D viewer share one Flask process (port 8000).
python -m grid_editor.server

# Headless sim3d smoke from a separate shell.
python -m sim3d --headless --frames 120

# Quick env / observation sanity check.
python -c "
from sim3d.moonboard_env import MoonboardClimbing3DEnv
from sim3d.moonboard  import load_moonboard_problems
from sim3d.env        import EnvConfig
problems = load_moonboard_problems('data/moonboard/sample-problems.json')
env = MoonboardClimbing3DEnv(problems[:1], config=EnvConfig())
obs, _ = env.reset(seed=0)
print('obs', obs.shape, 'action', env.action_space.shape)   # (127,) (25,)
"
```

The editor hard-codes `/data/walls/` to match the Docker bind-mount. For a
local run, edit `DATA_DIR` at the top of `grid_editor/server.py`.

---

## Concrete tensor shapes (so you don't have to run the code)

```
action_space      : Box(low=-1, high=1, shape=(25,), dtype=float32)
                    # 21 joint targets + 4 grip intents [LH, RH, LF, RF]
observation_space : Box(low=-inf, high=inf, shape=(127,), dtype=float32)
                    # see CLAUDE.md for the byte-level layout
```

A grip engages iff intent > 0 AND the limb tip is within
`sim3d.config.GRIP_PROXIMITY_M` (default 0.05 m) of an unoccupied valid hold.

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
