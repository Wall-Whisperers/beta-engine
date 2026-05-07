# sim3d — 3D climbing simulator

A MuJoCo-driven 3D successor to the 2D `physics/` package. Built so
the rest of the project (solver, RL env, eventual photo→pipeline) can
treat the climber as a real articulated body in 3D space.

> **Phase 4 status — environment only.** This package gives you a
> working 3D world: wall, holds, articulated climber, attach/release,
> step. Solver and RL training are not wired up yet — that's the next
> phase. The RL env in `rl/` still runs against the 2D world.

---

## Why MuJoCo

Three engines made the shortlist. The decision matrix:

| Engine | Why we considered it | Why we passed (or chose) |
|---|---|---|
| **MuJoCo 3.x** | Industry standard for humanoid RL since DeepMind open-sourced it; great contact solver; native Python; built-in viewer; MJX/JAX backend means the same MJCF runs at GPU speed for big-batch RL later. | **Chosen.** The contact solver behaves well under tendon-like leashing, which is what we need for force-limited holds. |
| PyBullet | Free, well-documented, easy to learn. | Less accurate contact integration; the climber's force-limited grip would be jittery. Slower for batch RL. |
| Genesis (GPU) | New (Dec 2024), claims 10–80× faster than MuJoCo. | Bleeding-edge APIs; we'd be on our own when something breaks. Worth re-evaluating once it has more publications behind it. |
| Brax / Isaac | Massive parallelism for RL. | Higher learning curve, hard to debug. We can move to MJX (MuJoCo-on-JAX) when training scale demands it without rewriting the model. |

Existing climbing simulators on the open-source side: there isn't one
that's seriously maintained. The closest analogue is **MuJoCo
Playground** (RSS 2025, DeepMind) — humanoid loco-manipulation tasks
that share our exact toolchain. We're explicitly building the
"climbing-task" sibling that doesn't yet exist.

---

## Architecture

```
sim3d/
├── config.py        # tunable constants (timestep, joint limits, masses, ...)
├── body.py          # ClimberProfile dataclass + segment math + limb names
├── builder.py       # build_mjcf_xml(wall, profile) → (xml, hold_meta)
├── world.py         # Climb3DWorld — owns MjModel + MjData
├── viewer.py        # native MuJoCo viewer wrapper
├── web.py           # Flask blueprint for the three.js front-end
└── __main__.py      # `python -m sim3d` CLI demo
```

**The data flow on each step:**

```
ClimberProfile + Wall  ──▶  builder.build_mjcf_xml  ──▶  MJCF string
                                                              │
                                                              ▼
                                                 mujoco.MjModel.from_xml_string
                                                              │
       ┌──────────────────────── Climb3DWorld ────────────────┘
       │   ├── attach_limb(limb, hold_id) — set mocap pos, eq_active=1
       │   ├── release_limb(limb)         — eq_active=0
       │   ├── step(frames)               — substep ×8 = 1 render frame
       │   └── pose_snapshot()            — body xpos/xquat for the viewer
       │
       ├──▶ native viewer (sim3d/viewer.py)
       └──▶ web bridge   (sim3d/web.py → static/sim3d.{html,js})
```

**Body model.** Climbing-specific, ~25 DOF (6 free + 19 hinge):

- pelvis (root, free joint = 6 DOF)
- spine (1 hinge: forward lean)
- 2 × shoulder (3 hinges: az / el / roll) + 2 × elbow (1 hinge)
- 2 × hip (3 hinges: flex / abduct / rot) + 2 × knee (1) + 2 × ankle (1)

Hands and feet are rigid stubs — that's where each limb's "tip site"
lives, and where the equality constraint to the hold attaches. We
deliberately did **not** model fingers / individual toes: in the MVP
the question "is this hold reachable?" is more important than
"how does the climber crimp?".

**Hold attachment.** One mocap body + weld equality constraint per
limb. To attach: position the mocap at the hold, set `data.eq_active[i]
= 1`, the limb's hand/foot body welds to it. To release: set
`eq_active[i] = 0`. No model recompile needed → fast for RL resets.

**Coordinate convention.** `+X` along the wall, `+Y` away from the wall
(toward the climber/camera), `+Z` up. Gravity is fixed at world `-Z`;
slab/overhang is implemented by tilting the wall, not gravity. The
opposite of `physics/world.py` (which tilts gravity) — but more
natural in MuJoCo where geometry is mobile and gravity is a global.

**Wall angle.**

| `wall_angle_deg` | Meaning | Effect on climber |
|---:|---|---|
| `0` | Vertical | Standard |
| `< 0` | Slab — top tilts away | Feet press into wall, easier |
| `> 0` | Overhang — top tilts toward climber | Body wants to swing out |

Slab is what we ship with first. Overhangs work geometrically (the
plate tilts correctly) but the body controller is tuned for vertical
walls; expect the climber to swing out on overhangs > ~15°.

---

## Running locally

The simulator is pure Python, but **the native MuJoCo viewer needs an
OpenGL display**. On Linux you'll need an X session; on macOS the
`mujoco.viewer` window opens via Cocoa (no extra setup); Windows works
out of the box.

```bash
# 1. Install deps (use a venv if you don't want to pollute your system).
pip install -r requirements.txt

# 2. Open the native viewer with the bundled example wall:
python -m sim3d

# 3. Same, but with a tall climber and a scripted move sequence:
python -m sim3d --height 190 --wingspan 195 --beta h_006 RH:h_008 LF:h_005

# 4. Headless smoke test (no display, exits after 120 frames):
python -m sim3d --headless --frames 120
```

`--beta` accepts either bare hold IDs (which round-robin through
`LH RH LF RF`) or `LIMB:hold_id` to pin both. The viewer plays
moves out at one move every 1.5 s.

---

## Running via Docker

Docker can't open a native GUI by default, so the **web viewer at
`/sim3d/`** is the in-container experience. The Flask grid editor and
the 3D simulator share one process:

```bash
docker compose up --build
# then open http://localhost:8000          → the 2D grid editor
# and  open http://localhost:8000/sim3d/    → the 3D viewer
```

The 3D viewer page lets you:
- pick a wall from the same data dir the grid editor saves to,
- set climber height / wingspan / mass,
- start a session, step 0.1 s / 1 s, or play live at 60 Hz,
- move any limb to any hold via dropdowns,
- watch the climber rendered in three.js (so the GPU stays in your
  browser, not in the container).

If you really want the *native* MuJoCo viewer inside Docker, you'll
need to forward an X display:

```bash
xhost +local:docker
docker compose run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  beta-engine python -m sim3d
```

The Linux Dockerfile installs `libgl1`, `libegl1`, and `libglib2.0-0`
specifically because mujoco's wheel `dlopen()`s them at import time
even when nothing renders.

---

## Programmatic use

```python
from solver.wall import load_wall
from sim3d import Climb3DWorld, ClimberProfile

wall = load_wall("example-v2-boulder")
profile = ClimberProfile(height_cm=175, wingspan_cm=175, mass_kg=70)

world = Climb3DWorld(wall, profile)
world.seed_pose(lh="h_003", rh="h_004", lf="h_001", rf="h_002")

# Step 1 second of physics
for _ in range(60):
    world.step()

# Make a move
world.move_limb("RH", "h_008", mode="snap")    # or "reach" for dynamic

# Read state
print(world.com())          # 3D centre of mass
print(world.pelvis_pos())
print(world.limb_tip_pos("RH"))
print(world.on_hold("RH"))  # → "h_008"

# Get a render-ready snapshot (for the web viewer / debugging)
snap = world.pose_snapshot()
```

---

## What this gives you (and what it doesn't, yet)

**You can:**
- Place a climber on any wall in the editor and watch them hang in 3D.
- Move limbs and have the body follow with realistic mass/inertia.
- Tune climber dimensions and see how the same wall looks for a 160 cm
  vs 190 cm climber.
- Stream poses to a browser without exposing MuJoCo to the network.

**You can't yet (next phases):**
- Train an RL agent — the actuators and `step()` API are RL-ready, but
  there's no Gymnasium env wrapper here. The `rl/` package still talks
  to the 2D world; porting it is Phase 5.
- Solve a route — the A\* solver in `solver/` runs against the 2D
  reachability checker. A 3D reachability checker on top of `Climb3DWorld`
  is the natural next step.
- Use a Hill-type muscle model. The brief asked us to "think about
  muscles" — we considered it, but a torque-limited PD position
  servo is the right level of detail for the MVP. Hill muscles add a
  ~5× model-complexity cost for a marginal RL benefit.

---

## Tuning notes

- **`PHYS_DT = 2 ms`** — necessary for the welded-limb constraints to
  not jitter. Larger steps cause visible spring-back when a limb
  attaches.
- **`ACTUATOR_KP = 200, KV = 20`** — tuned so the climber holds a
  static pose without overshoot. Too-low Kp = sagging body; too-high
  Kp = vibration when a limb moves.
- **`HOLD_CONSTRAINT_SOLREF = (0.01, 1.0)`** — the weld's spring time
  constant. Crank it up (`(0.005, 1.0)`) for a tighter "snap-on" feel
  but expect more impulse at the moment of attach.
- **Joint limits in `config.JOINT_LIMITS_RAD`** — start generous,
  tighten as the solver finds physically silly poses. The hip flex
  range, in particular, is the difference between a high-foot beta
  being possible vs unreachable.
