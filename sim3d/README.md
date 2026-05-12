# sim3d — 3D climbing simulator

A MuJoCo-driven 3D successor to the 2D `physics/` package. Built so
the rest of the project (solver, RL env, eventual photo→pipeline) can
treat the climber as a real articulated body in 3D space.

> **Current status — research prototype.** This package now includes the
> MuJoCo 3D world, a Gymnasium wrapper, and an SB3 PPO training entry
> point. The high-level RL action space can choose limb→hold moves while
> MuJoCo still integrates the body continuously between decisions. It is
> good enough for experiments and demos, but it is not yet a production
> climbing model: observations are wall-size dependent, reward shaping is
> early, and policies must be replayed with the same wall/config used for
> training.

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
├── world.py         # Climb3DWorld — owns MjModel + MjData + slip detection +
│                    #   continuous-reach Cartesian-impedance controller
├── env.py           # Climbing3DEnv — Gymnasium wrapper for RL training
├── moonboard.py     # MoonBoard problem JSON → Wall adapter
├── train.py         # PPO trainer (SB3) + episode CSV logger
├── viewer.py        # native MuJoCo viewer wrapper
├── web.py           # Flask blueprint for the three.js front-end
└── __main__.py      # `python -m sim3d` CLI demo (incl. --play <model>)
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

**Body model.** Climbing-specific, 27 DOF (6 free + 21 hinge):

- pelvis (root, free joint = 6 DOF)
- spine (1 hinge: forward lean)
- 2 × shoulder (3 hinges: az / el / roll) + 2 × elbow (1) + 2 × wrist (1)
- 2 × hip (3 hinges: flex / abduct / rot) + 2 × knee (1) + 2 × ankle (1)

Hands and feet are rigid stubs — that's where each limb's tip site
lives, and where the weld equality to the hold attaches. We
deliberately do **not** model fingers / individual toes: in the MVP
the question "is this hold reachable?" matters more than "how does
the climber crimp?". Wrist DOF gives the hand enough freedom to
rotate into the hold; ankles do the same for feet.

Every joint has anatomically motivated limits (see
`config.JOINT_LIMITS_RAD`):

| Joint | Range | Notes |
|---|---|---|
| spine_lean   | -15° → +60°  | climber tucks for high feet |
| shoulder_az  | -50° → +180° | no excessive backward swing |
| shoulder_el  |   0° → +180° | full overhead reach |
| shoulder_roll| -80° → +80°  | internal/external rotation |
| elbow        |   0° → +150° | **no hyperextension** |
| wrist        | -70° → +70°  | flex/extend, no radial deviation |
| hip_flex     | -20° → +140° | high-step capable |
| hip_abduct   | -20° → +70°  | drop-knee, frog flag |
| hip_rot      | -40° → +40°  | |
| knee         |   0° → +150° | **no hyperextension** |
| ankle        | -25° → +45°  | dorsi/plantar |

Joints also carry passive stiffness + damping so the body doesn't
flop into pretzels when the actuator is under-driving.

**Hold attachment.** One mocap body + weld equality per limb. To
attach: position the mocap at the hold, set `data.eq_active[i] = 1`,
the limb welds to it. To release: set `eq_active[i] = 0`. No model
recompile needed → fast for RL resets. The weld's `relpose` is set so
the limb's *tip site* (fingertips / toe) lands on the hold rather
than the wrist / ankle.

**Continuous limb motion.** `move_limb` supports three modes:

| Mode | Behaviour | Use when |
|---|---|---|
| `snap` | Instant teleport. | Fast RL training where you only care which holds the limb visits. |
| `reach` (default) | Cartesian-impedance PD pulls the limb tip toward the target through space. Body sways under gravity + the reach force. Welds when the tip is within 5 cm or after 1.5 s. | Manual play, dynamic visualisation, realistic RL. |
| `dyno` | Same as reach plus an explosive leg-extension push during the first 0.35 s — for moves that would be out of static reach. | Long throws between holds. |

While a limb is reaching, its actuator KP is temporarily zeroed so
the per-joint hold-pose servo doesn't fight the impedance controller.
The applied force at the tip is mapped into joint-space via
`mj_applyFT` (force at a world point through the body Jacobian). All
of this happens automatically inside `step()` — you call
`move_limb(..., mode="reach")` once and then `step()` until the limb
attaches.

**Slip model.** When `check_slip=True` (or `--slip` on the CLI, or
`enable_slip=True` in the Gym env config), every step we compute the
world-frame force on each active weld via `data.cfrc_int` and release
any whose force exceeds the hold's rated capacity ×
`SLIP_FORCE_SLACK`. A real climber blowing a crimp is exactly this —
finger force exceeds skin/tendon capacity, the grip releases. Slip
events are logged in `world.slip_events` for diagnostics.

Slip detection is **off by default** in the bare `Climb3DWorld` API
(it can fire spuriously during dynamic moves and needs careful
tuning). It's **on by default** in the Gym env, where it's a
fundamental part of the failure-cost reward signal.

**Friction / smearing.** The wall surface and each hold carry their
own friction tuples. Holds are `contype=2`/`conaffinity=2` so the
limb doesn't *collide* with them (grabbing is mediated by the weld);
the wall surface is `contype=1` and the limb tips can press against
it for slab smearing or to balance against an overhang.

**Self-intersection penalty.** The MuJoCo body can still discover poses
where an arm or leg passes through the torso/pelvis. That is physically
impossible beta, so the Gym env scans MuJoCo contacts after each step and
subtracts `body_intersection_penalty` for every limb-vs-body penetration.
Tune it with `--body-intersection-penalty` in `sim3d.train`; set it to
`0` if you want to disable this shaping term for debugging.

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

# 5. Load a MoonBoard problem and view it in the native viewer:
python -m sim3d --moonboard data/moonboard/sample-problems.json --problem 19215

# 6. Random Gym episode (smoke test for the env wrapper) with slip on:
python -m sim3d --gym --gym-episodes 3 --slip

# 7. Random Gym episode on a MoonBoard problem:
python -m sim3d --gym --moonboard data/moonboard/sample-problems.json --problem 19216
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
- load a trained `sim3d.train` run directory or `model.zip`, which
  reconfigures the session to the wall/profile stored in that run's
  `config.json`, then step/play the PPO policy in the browser,
- clear the loaded policy and return to manual browser control,
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

# Step 1 second of physics, with slip detection enabled.
slips = world.step(60, check_slip=True)
print(f"slip events: {slips}")

# Make a move
world.move_limb("RH", "h_008", mode="snap")    # or "reach" for dynamic

# Read state
print(world.com())              # 3D centre of mass
print(world.pelvis_pos())
print(world.limb_tip_pos("RH"))
print(world.on_hold("RH"))      # → "h_008"
print(world.limb_grip_force("RH"))   # newtons currently flowing through this limb

# Render-ready snapshot for the web viewer
snap = world.pose_snapshot()
```

### MoonBoard problems

```python
from sim3d.moonboard import load_moonboard_problems, moonboard_problem_to_wall
from sim3d import Climb3DWorld, ClimberProfile

problems = load_moonboard_problems("data/moonboard/sample-problems.json")
wall = moonboard_problem_to_wall(problems[0])   # 11×18 grid, 40° overhang
world = Climb3DWorld(wall, ClimberProfile())
```

The adapter accepts the standard public MoonBoard problem schema —
`{id, name, grade, holdsets, start_holds, mid_holds, end_holds}` with
`"E6"`-style position strings. By default only the problem's holds
appear on the wall. Pass `include_full_board=True` to also include all
198 T-nut positions as auxiliary holds (useful for letting an RL
agent discover off-route footholds).

**Two layout conventions** for the 40° overhang:

```python
# Default: holds sit on the actual angled surface (physically real).
wall = moonboard_problem_to_wall(problem)

# vertical_projection=True: holds are spaced so each row's WORLD-Z
# matches a vertical reference board. Looks like the photo of a
# MoonBoard rather than the foreshortened tilted surface. Internally
# we scale cell_size by 1/cos(40°) so the rows project to the same
# vertical pitch as a flat board.
wall = moonboard_problem_to_wall(problem, vertical_projection=True)
```

CLI: pass `--vertical-projection` to `python -m sim3d`. Web: send
`{"moonboard_vertical_projection": true}` in the session-create POST.

**Foothold fallback.** MoonBoard problems don't mark anything as
`"foothold"` — competitions allow climbers to use any hold for feet.
The Gym env handles this automatically: when no foothold-typed holds
exist, it picks the lowest non-start non-finish hold on each side as
the seeded foot anchor. So you get all four limbs on holds at reset
even on bare MoonBoard problems.

### Gymnasium environment for RL

```python
from solver.wall import load_wall
from sim3d.env import Climbing3DEnv, EnvConfig

env = Climbing3DEnv(load_wall("example-v2-boulder"),
                    config=EnvConfig(max_steps=30, enable_slip=True))
obs, info = env.reset()
for _ in range(30):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        print(info["outcome"])
        break
```

Two action modes are available:

- **`discrete-move`** (default): `Discrete(4 * n_holds)` — pick a limb
  and a hold. Best for the high-level beta-finding RL the project is
  built around.
- **`continuous-joint`**: `Box(-1, 1, (n_actuators,))` — direct joint
  targets, scaled into each joint's range. For low-level motor
  control. Trains slower but is more flexible.

Observation is a flat 117-d vector (for 13-hold walls; scales with
hold count): pelvis pose + COM + joint angles + joint velocities +
limb-tip positions + per-limb on-hold one-hot + distance-to-finish.

Drop-in compatible with Stable-Baselines3:

```python
from stable_baselines3 import PPO  # installed by requirements.txt
model = PPO("MlpPolicy", Climbing3DEnv(wall), verbose=1)
model.learn(total_timesteps=100_000)
```

### Training a policy + viewing results

The included `sim3d.train` module wraps PPO + episode-stat logging:

```bash
# 1. Train. Defaults to the example wall, ~100k timesteps, mode=reach.
python -m sim3d.train --steps 100_000

# Or train on a MoonBoard problem with vertical projection:
python -m sim3d.train --moonboard data/moonboard/sample-problems.json \
                     --problem 19215 --steps 200_000

# 2. View results.
ls data/runs/sim3d/run_<timestamp>/
#   ├── config.json         # hyperparams + wall + climber profile
#   ├── episode_stats.csv   # one row per episode (reward, length, outcome, com_z, slips, intersections)
#   ├── tb/                 # TensorBoard event files
#   └── model.zip           # trained PPO policy

# Plot reward curve:
tensorboard --logdir data/runs/sim3d/run_<timestamp>/tb
# or just grep the CSV:
column -ts, data/runs/sim3d/run_<timestamp>/episode_stats.csv | head -20

# 3a. Replay the policy in the native MuJoCo viewer:
python -m sim3d --play data/runs/sim3d/run_<timestamp>/model.zip

# 3b. Or replay in the browser:
python grid_editor/server.py
# open http://localhost:8000/sim3d/
# paste data/runs/sim3d/run_<timestamp>/ into "RL policy replay"
```


### Faster training: GPU, parallel CPU workers, and Colab

`sim3d.train` now exposes the Stable-Baselines3/PyTorch device and the
number of parallel MuJoCo rollout workers:

```bash
# Use CUDA for PPO neural-network updates when your PyTorch install sees a GPU.
# MuJoCo environment stepping is still CPU-bound, so pair this with workers.
python -m sim3d.train --steps 200_000 --device cuda --n-envs 8

# If CUDA is not available, keep device=auto/cpu and scale rollout workers.
python -m sim3d.train --steps 200_000 --device auto --n-envs 8
```

Important caveat: this project uses Stable-Baselines3 PPO with a MuJoCo
Python environment. The policy network can train on GPU, but physics rollout
still happens on CPU. For this codebase, `--n-envs` is usually the bigger
speedup than GPU unless you also increase `--steps`, `--n-steps`, or network
size enough for PPO updates to dominate runtime.

A simple Google Colab workflow is:

```bash
# In a Colab notebook after enabling a GPU runtime:
!git clone https://github.com/Wall-Whisperers/beta-engine.git
%cd beta-engine
!pip install -r requirements.txt
!python -m sim3d.train --steps 200000 --device cuda --n-envs 2 --run-id colab_gpu

# Download or archive data/runs/sim3d/colab_gpu/, then replay model.zip locally
# or paste the run directory into the browser viewer's RL policy replay field.
```

If Colab runs out of RAM, reduce `--n-envs`; if GPU utilization is low, that is
expected for CPU-bound MuJoCo rollouts.

**What you should see in the CSV during training:**
- Early episodes have rewards around 0–2 (mostly per-step penalties).
- Successful episodes show `outcome=completed` with reward > 100.
- `final_com_z` trends upward as the policy learns to climb (up from
  ~0.5 m baseline to ~1.5+ m for partial progress).
- The `slips` column tells you how often the policy is over-gripping
  beyond hold capacity — high values mean the slip-aware reward is
  saturating and you may want to lower `slip_penalty`.
- `body_intersections` in the Gym `info` dict counts limb-vs-torso/pelvis
  penetration contacts; each one is penalized during training so the
  policy avoids impossible arms/legs-through-body poses.

**Viewer integration**: when `--play` is set, the native MuJoCo
viewer opens and the policy drives the climber in real-time. Each
discrete-move action takes `--play-frames` (default 120 = 2 s) of
physics so the continuous-reach has time to actually swing the limb
to the next hold rather than teleporting.

---

## What this gives you (and what it doesn't, yet)

**You can:**
- Place a climber on any wall in the editor and watch them hang in 3D.
- Move limbs and have the body follow with realistic mass/inertia.
- Tune climber dimensions and see how the same wall looks for a 160 cm
  vs 190 cm climber.
- Stream poses to a browser without exposing MuJoCo to the network.

**You can't yet (next phases):**
- Solve a route — the A\* solver in `solver/` runs against the 2D
  reachability checker. A 3D reachability checker on top of
  `Climb3DWorld` is the natural next step.
- Use a Hill-type muscle model. The brief asked us to "think about
  muscles" — we considered it, but a torque-limited PD position
  servo is the right level of detail for the MVP. Hill muscles add a
  ~5× model-complexity cost for a marginal RL benefit.
- Train policies that complete unseen MoonBoard problems with high
  reliability — the env runs, but procedural curriculum (random
  walls of increasing difficulty) is needed to get there.
- Train policies that generalize to arbitrary unseen walls from a
  single saved `MlpPolicy` — browser replay now works, but the current
  observation shape still depends on hold count, so a saved policy is
  wall/config specific.

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
