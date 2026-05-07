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

## Scripts

| Script | Command | Purpose |
|--------|---------|---------|
| `day1_viewer.py` | `mjpython scripts/day1_viewer.py` | Pure visualisation — wall, holds, humanoid, no grip |
| `test_grip.py` | `mjpython scripts/test_grip.py` | Automated grip pipeline test (500 steps, logs forces) |
| `interactive_grip.py` | `mjpython scripts/interactive_grip.py` | Manual grip testing with key bindings |

All three fall back to headless/log-only mode automatically when `mjpython` is unavailable.

### Interactive Grip Key Bindings

```
1   grip LEFT  hand on nearest hold     3   release LEFT  hand
2   grip RIGHT hand on nearest hold     4   release RIGHT hand
g   print grip state + active holds     f   print constraint forces
r   reset simulation (all grips off)
```

Drag the humanoid with **Ctrl + drag** in the viewer window, then press `1` or `2` when a hand is near a hold sphere.

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
      wall.py         — Wall geometry, hold_position_world(), WALL_NORMAL
      holds.py        — Hold sphere geoms (deterministic hold_{col}_{row} names)
      scene.py        — Assembles full scene XML; injects limb sites + grip constraints
    grip/
      grip_manager.py — GripManager: connect constraint retargeting, slip detection
    viewer.py         — Unified viewer entry point (key callbacks, on_step hook)
    envs/             — Gymnasium environments (Day 4+, placeholder for now)
  assets/
    humanoid.xml      — MuJoCo humanoid model (from Gymnasium assets)
  scripts/
    day1_viewer.py      — Pure visualisation
    test_grip.py        — Automated grip pipeline test
    interactive_grip.py — Manual grip tester with key bindings
  output/             — Auto-created; scene XML written here on fallback
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

## Grip Mechanic

### Confirmed Limb Slots, Body Names, and Sites

| Slot | Limb | Body | Site | Site pos (body-local) |
|------|------|------|------|-----------------------|
| 0 | Left hand | `left_lower_arm` | `site_lhand` | `.18 -.18 .18` |
| 1 | Right hand | `right_lower_arm` | `site_rhand` | `.18 .18 .18` |
| 2 | Left foot | `left_foot` | `site_lfoot` | `0 0 0.1` |
| 3 | Right foot | `right_foot` | `site_rfoot` | `0 0 0.1` |

Sites are placed at the **hand/foot tip** (same position as the sphere geom centre), not the body origin. `data.xpos[body_id]` returns the elbow/ankle joint for arm/leg bodies — sites fix this by using `data.site_xpos[site_id]` for all proximity checks and anchor calculations.

### Thresholds

| Constant | Value | Rationale |
|----------|-------|-----------|
| `PROXIMITY_THRESHOLD` | 0.12 m | Max distance from limb **site** (tip) to hold centre for grip to engage. Relax to 0.20 m for pipeline testing. |
| `ALIGNMENT_THRESHOLD` | 0.70 | Min dot product of limb body's local Z-axis with wall outward normal. Cosine 0.70 ≈ 45.5°. Set −1.0 in tests/interactive mode. |
| `MAX_CONSTRAINT_FORCE` | 500 N | Force above which `check_slip()` auto-releases the grip. Roughly 3× single-limb bodyweight. |

### How Connect Constraint Retargeting Works

A MuJoCo `connect` equality constraint pins a point in body1's local frame to a point in body2's local frame.  When gripping a hold:

1. **`model.eq_obj2id[eq_id] = hold_body_id`** — retarget body2 from placeholder `grip_anchor` to the actual hold.
2. **`anchor1 = R_limb.T @ (site_world - limb_world)`** — the site's position in the limb body's local frame (constant, equals the site's `pos` attribute).
3. **`anchor2 = R_hold.T @ (site_world - hold_world)`** — where the site currently sits, in the hold body's local frame. **Critical:** without this step MuJoCo enforces the pose from model-load time and the arm snaps violently.
4. **`data.eq_active[eq_id] = 1`** — use `data.eq_active` (runtime), not `model.eq_active0` (load-time default; has no effect after simulation starts).

## Build Phases

- **Day 1** ✅ — Wall + holds visualisation, humanoid in scene, all three JSON parsers
- **Days 2–3** ✅ — Connect constraint grip mechanic, GripManager, slip detection, interactive viewer
- **Fixes** ✅ — Site-based anchoring (hand tip not elbow), unified viewer module, interactive key bindings
- **Day 4** — Gymnasium `Env` wrapper: observation space, action space, `reset()`, `step()`
- **Week 2** — Basic reward function, PPO training loop with stable-baselines3
- **Week 3+** — RL training on synthetic walls, transfer to real routes
