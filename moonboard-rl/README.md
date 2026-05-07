# MoonBoard RL Environment

A MuJoCo 3.x reinforcement learning environment for simulating a humanoid climbing a MoonBoard — a standardised 40-degree overhanging climbing wall.

## Project Goal

Train a humanoid agent to solve MoonBoard routes (climbing problems) using physics-based simulation and reinforcement learning. The pipeline goes:

```
JSON route data → MuJoCo scene XML → Physics simulation → RL policy
```

## Setup

```bash
pip install -r requirements.txt
```

Verify the installation:

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

## Running the Day-1 Viewer

```bash
cd moonboard-rl
mjpython scripts/day1_viewer.py    # opens the interactive viewer (macOS requires mjpython)
```

If `mjpython` isn't available, the script falls back automatically: it writes `output/scene_day1.xml` and prints all hold coordinates without crashing.

```bash
python scripts/day1_viewer.py      # fallback mode — no viewer, XML written to output/
```

This will:
1. Load `moonboard1.json` from `../moonboard_data/`
2. Filter to V4/V5 routes and pick the most-repeated one
3. Build a MuJoCo scene with the 40° MoonBoard wall and colored hold spheres
4. Open the interactive MuJoCo viewer (or fall back to writing `output/scene_day1.xml`)

## Directory Structure

```
moonboard-rl/
  src/
    parsers/
      canonical.py    — Hold/Route dataclasses, grade converters
      format1.py      — Parser for moonboard1.json (array, int grades)
      format2.py      — Parser for moonboard2.json (dict, Font grades, row-first coords)
      format3.py      — Parser for moonboard3.json (dict, Font grades, richest schema)
    xml_gen/
      wall.py         — Wall geometry and hold_position_world()
      holds.py        — Hold sphere geoms
      scene.py        — Assembles full scene XML from humanoid + wall + holds
    envs/             — Gymnasium environments (Week 1+, placeholder for now)
  assets/
    humanoid.xml      — MuJoCo humanoid model (from Gymnasium assets)
  scripts/
    day1_viewer.py    — Main visualisation script
  output/             — Auto-created; holds scene_day1.xml fallback output
  requirements.txt
```

## Coordinate System

```
X: horizontal, perpendicular to climbing direction
   col A (left) → col K (right) as X increases

Y: horizontal, pointing from wall toward climber (positive = toward climber)
   wall face is at Y ≈ -1.5 m (bottom) to Y ≈ -0.3 m (top)
   humanoid stands at Y ≈ +1.0 m

Z: vertical, up
   row 1 (bottom hold): Z ≈ 0.30 m above floor
   row 18 (top hold):   Z ≈ 2.90 m above floor
```

## 40-Degree Overhang Math

The MoonBoard overhangs at **40° from vertical** — the top of the wall is closer to the climber than the bottom.

For a hold at grid position `(col, row)`:

```
s = (row - 1) × 0.20 m          # arc-length up the wall surface
x = (col - 5) × 0.20 m          # centered at column F
y = Y_BASE + s × sin(40°)        # top rows are closer to climber
z = Z_BASE + s × cos(40°)        # top rows are higher
```

Wall surface outward normal (pointing toward climber):

```
n = (0,  cos(40°),  −sin(40°))
  ≈ (0,  0.766,    −0.643)
```

## Data Formats

| File | Format | Grades | Coord style | Today? |
|------|--------|--------|-------------|--------|
| `moonboard1.json` | Array of objects | V-grade int | `"F4"` (col then row) | ✅ Used |
| `moonboard2.json` | Dict by index | Font string `" 6C"` | `"16F"` (row then col) | Parsed, deferred |
| `moonboard3.json` | Dict by route ID | Font string `"6A"` | `"J4"` (col then row) | Parsed, deferred |

## Build Phases

- **Day 1** ✅ — Wall + holds visualisation, humanoid in scene
- **Day 2** — Weld constraints to pin limbs to holds; initial pose from start holds
- **Week 1** — Observation space, action space, basic reward function
- **Week 2+** — RL training with stable-baselines3 (PPO/SAC)
