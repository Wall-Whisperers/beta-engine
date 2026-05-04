# Solver

A basic 2D inverse-kinematics + route-finding solver. Reads the wall JSON
files produced by the grid editor and outputs:

- A **move sequence** — which limb (LH/RH/LF/RF) moves to which hold, in order.
- A **PNG panel sheet** — one subplot per pose, with stick-figure overlay.
- An **animated GIF** (optional) — the same sequence as a flipbook.

This is the Phase 2/3 MVP from [`CLAUDE.md`](../CLAUDE.md): body model + IK +
reachability + graph search, all working together before computer vision or a
heavier physics engine gets layered on top.

---

## Quickstart

Container must be running (`docker compose up -d`):

```bash
# A* — fast, mathematically shortest path
docker compose exec beta-engine python -m solver --wall example-v2-boulder

# Tabular Q-learning RL
docker compose exec beta-engine python -m solver --wall example-v2-boulder --method qlearn --episodes 2000

# Both solvers + animated GIFs
docker compose exec beta-engine python -m solver --wall example-v2-boulder --method both --gif

# Your own wall (saved in the editor)
docker compose exec beta-engine python -m solver --wall <wall_id> --method both --gif

# Taller climber, no visualization, just print the moves
docker compose exec beta-engine python -m solver --wall example-v2-boulder \
  --height-cm 190 --wingspan-cm 185 --no-viz
```

Outputs land in `./data/runs/` on your host:

```
data/runs/
├── example-v2-boulder-astar.png
├── example-v2-boulder-astar.gif
├── example-v2-boulder-qlearn.png
└── example-v2-boulder-qlearn.gif
```

---

## CLI flags

```
python -m solver --wall <id|path> [options]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--wall <id\|path>` | — | Wall ID (resolves to `/data/walls/<id>.json` or the seeded examples) or a path to a JSON file |
| `--method` | `astar` | `astar`, `qlearn`, or `both` |
| `--height-cm` | `175` | Climber height in cm — scales leg length |
| `--wingspan-cm` | `175` | Climber wingspan in cm — scales arm length |
| `--cell-size-cm` | `20` | Override grid cell size in cm (the schema doesn't carry this yet — see below) |
| `--episodes` | `2000` | Q-learning training episodes (ignored by `astar`) |
| `--no-viz` | off | Skip rendering, just print the move list |
| `--gif` | off | Also render an animated GIF alongside the PNG |

---

## How it works

### Body model (`body.py`)

A 5-point stick figure:

- **1 Center of Mass (COM)** — approximate pelvis position.
- **4 end-effectors** — LH (left hand), RH, LF (left foot), RF.

Each limb is a **2-link chain** (upper + lower segment, equal length) solved
with a closed-form **law-of-cosines IK** — no iteration needed:

```
dist = |target - anchor|
cos_angle = (upper² + dist² - lower²) / (2 × upper × dist)
elbow = anchor + upper × [cos(atan2(dy,dx) ± arccos(cos_angle)), ...]
```

Anthropometric defaults (175 cm / 175 cm wingspan):

| Measurement | Value |
|-------------|-------|
| Arm length (per side) | 42% of wingspan ≈ 73.5 cm |
| Leg length (hip to ankle) | 48% of height ≈ 84 cm |
| Max hand reach from shoulder | ~73.5 cm |
| Max foot reach from hip | ~84 cm |

### Reachability (`reachability.py`)

A **pose** is a tuple of four hold IDs: `(LH, RH, LF, RF)`. From any pose,
a one-limb move is legal when:

1. The target hold is within **92% of max reach** from the limb's body anchor
   (safety margin to avoid full-extension poses).
2. The **three-point intermediate** (moving limb off the wall) is stable.
3. The **resulting four-point pose** is stable.

**Stability** (vertical wall assumption): COM x-coordinate must lie within
the horizontal span of the two active footholds (or within 5 cm of a single
foot).

### A\* solver (`astar.py`)

- **Nodes** = poses `(LH, RH, LF, RF)`
- **Edges** = legal one-limb moves
- **Heuristic** = vertical distance from the higher hand to the nearest finish hold
- **Cost** = 1 per move + a positivity penalty (worse holds cost slightly more)
- **Multi-source** = all plausible starting poses are pushed onto the open set simultaneously

This gives the "mathematically shortest" beta — a good baseline to compare the RL agent against.

### Q-learning solver (`rl_qlearn.py`)

A **dependency-free tabular agent** — no `gymnasium`, no `stable-baselines3`.
State = pose tuple; action = `(limb, target_hold)`.

Reward shaping:

| Signal | Amount |
|--------|--------|
| Progress toward finish (per cm closer) | +0.05 |
| Efficiency penalty (each move) | -0.5 |
| Stability penalty (unstable result) | -50 |
| Completion bonus (hand on finish) | +100 |
| Dead-end penalty (episode times out) | -5 |

The env class is shaped like a Gym `reset()/step()` so swapping in PPO
+ Stable Baselines3 later is a small change — just replace the training
loop in `rl_qlearn.py`, the env stays the same.

### Visualizer (`visualize.py`)

Headless matplotlib (Agg backend — no display required). Renders:

- **PNG panel sheet** — one subplot per pose in the solution. Each panel
  shows the wall (holds colour-coded by type, start/finish rings), the
  stick figure (coloured per limb: LH red, RH green, LF blue, RF yellow),
  and a title with the move description.
- **Animated GIF** (opt-in via `--gif`) — same frames at 2 fps.

Outputs go to `/data/runs/` inside the container, which maps to
`./data/runs/` on your host via the Docker bind mount.

---

## Known limits (intentional — MVP)

| Limitation | Why it's OK for now | How to fix later |
|------------|--------------------|--------------------|
| **Vertical wall only** | Simplifies stability check | Add `wall_angle_deg` + friction model in Phase 3 |
| **No `cell_size_cm` in schema** | Schema is locked — see [`planning-gabe.md`](../planning-gabe.md) | Add it to `schemas/wall.schema.json` before Phase 3 |
| **Tabular RL won't scale past ~20 holds** | State space is N⁴ — fine for ~13 holds | Swap in PPO + Stable Baselines3 when walls get bigger |
| **Single-wall RL** | No procedural generation yet | Phase 4: train on 1000 random walls |
| **No 3D / hip twist / drop-knee** | 2D projection is tractable | Phase 3+: Pymunk physics, then MuJoCo |

---

## Module reference

| File | Responsibility |
|------|---------------|
| `wall.py` | Load wall JSON, convert grid cells to world (cm) coords, hold type utilities |
| `body.py` | `BodyModel` dataclass, `solve_2link_ik`, `resolve_skeleton` for visualization |
| `reachability.py` | `Pose`, `can_reach`, `is_stable`, `reachable_moves` |
| `astar.py` | `solve_astar`, `starting_poses`, `SolveResult` |
| `rl_qlearn.py` | `ClimbingEnv`, `solve_qlearn` |
| `visualize.py` | `render_panels` (PNG), `render_animation` (GIF) |
| `__main__.py` | CLI — argument parsing, orchestrates load → solve → render |
