# Beta Engine

An RL agent that learns to climb a MoonBoard by continuously controlling a
custom **27-DOF humanoid** in MuJoCo. Every control step the policy emits
joint position targets for all 21 actuated joints plus four grip intents
(one per limb). There is no discrete "snap to hold" teleportation — the body
must discover climbing movement from physics, proprioception, and the 3D
positions of the holds around it.

---

## The Vision: Continuous Muscle-Level Control

Most climbing-AI projects treat movement as a sequence of discrete decisions:
pick a hold, teleport the hand there, repeat. That is not how climbing works.

**The goal of this project is to learn climbing the way a climber does it:**
by coordinating dozens of muscles simultaneously to move the body through
space, maintain balance, and generate the precise forces needed to stay on
the wall.

The current architecture uses **PD position-servo actuators** (joint-target
→ torque via Kp/Kv gains) as a tractable first step. The roadmap leads
toward **torque actuators**, and eventually **Hill-type muscle models** where
the policy outputs activation signals that drive force–length–velocity
relationships — the same control signal that fires in a real climber's spinal
cord.

This distinction matters:
- A discrete-move agent memorises `(hold_A, limb_B)` mappings. It fails on
  any wall it hasn't seen.
- A continuous-control agent learns how the body works. It generalises to any
  wall, any body proportions, any hold geometry — because the solution is
  physics, not a lookup table.

---

## Project Components

| Component | Folder | What it does |
|---|---|---|
| **Wall editor** | `grid_editor/` | Flask + vanilla-JS tool to place holds on a grid and save walls as JSON |
| **2D solver** | `solver/` | Legacy 2D IK + A* used for the editor's planning preview and the procedural wall generator |
| **3D simulator** | `sim3d/` | MuJoCo 3D physics, custom 27-DOF humanoid, Gymnasium env, SB3 PPO trainer |
| **Schemas** | `schemas/` | JSON Schema for the wall format |
| **Static** | `static/` | Browser frontend for both the editor and the 3D viewer |
| **Data** | `data/` | Walls, MoonBoard problem corpus, training run outputs |

The wall editor locks the JSON schema that everything downstream depends on.
The 2D solver was the bootstrap path; **the active training stack is `sim3d/`**.

---

## Architecture

### Pipeline

```
Photo(s)  →  [future CV]  →  Wall JSON
                                  │
                    ┌─────────────▼──────────────┐
                    │      grid_editor/           │
                    │  Flask REST + JS editor     │
                    └─────────────┬───────────────┘
                                  │  wall.json
                    ┌─────────────▼───────────────┐
                    │         sim3d/              │
                    │  builder.build_mjcf_xml()   │
                    │    → MuJoCo MJCF string     │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Climb3DWorld (world.py)     │
                    │  27-DOF humanoid in MuJoCo  │
                    │  mocap+weld grip model       │
                    │  slip detection              │
                    └─────────────┬───────────────┘
                                  │  obs (127,)
                    ┌─────────────▼───────────────┐
                    │  Climbing3DEnv (env.py)      │
                    │  Gymnasium-compatible        │
                    │  SB3 PPO ("MlpPolicy", env) │
                    └─────────────┬───────────────┘
                                  │  action (25,)
                              trained policy
```

### The Body Model

The climber is a **custom 27-DOF humanoid** — not the stock MuJoCo humanoid.
It is anatomically grounded: segment lengths from measured joint-height
ratios for a 175 cm male, mass fractions from Winter biomechanics, joint
limits from the ACSM range-of-motion tables biased toward trained climbers.

```
pelvis (root, 6-DOF free joint)
└── spine (1 hinge: forward lean)
    ├── LEFT arm chain:  shoulder_az → shoulder_el → shoulder_roll → elbow → wrist
    │                    tip site: site_lh_tip  (where the weld attaches)
    ├── RIGHT arm chain: shoulder_az → shoulder_el → shoulder_roll → elbow → wrist
    │                    tip site: site_rh_tip
    ├── LEFT leg chain:  hip_flex → hip_abduct → hip_rot → knee → ankle
    │                    tip site: site_lf_tip
    └── RIGHT leg chain: hip_flex → hip_abduct → hip_rot → knee → ankle
                         tip site: site_rf_tip
```

**Joint limits (climbing-realistic):**

| Joint | Range | Key constraint |
|---|---|---|
| spine_lean | −15° → +15° | kept near 0° by raised passive stiffness |
| shoulder_az | −50° → +180° | no extreme backward swing |
| shoulder_el | 0° → +180° | full overhead reach |
| shoulder_roll | −80° → +80° | internal/external rotation |
| elbow | 0° → +150° | **no hyperextension** |
| wrist | −70° → +70° | flex/extend only |
| hip_flex | −20° → +140° | high-step capable |
| hip_abduct | −20° → +70° | drop-knee, frog-flag |
| hip_rot | −40° → +40° | |
| knee | 0° → +150° | **no hyperextension** |
| ankle | −25° → +45° | dorsi/plantar |

Each joint has per-group PD gains tuned near critical damping for the load
it carries (shoulder Kp=220, hip Kp=450, etc. — see `sim3d/config.py`).
Passive stiffness + damping give the body a "taut tendon" feel and stop it
from flopping when the actuator is under-driving.

### Hold Attachment

Holds are grabbed via **mocap body + weld equality constraint**, one per
limb. On attach: the mocap is teleported to the hold's world position and
`eq_active = 1` engages the weld. On release: `eq_active = 0`. No MJCF
recompile needed → fast RL resets. The weld's `relpose` is set so the
**tip site** (fingertip / toe stub) lands on the hold, not the wrist / ankle.

### Grip model

A grip engages when:
1. The per-limb **grip intent** in the action is `> 0.0`, **and**
2. The limb tip is within **5 cm** (`GRIP_PROXIMITY_M`) of an unoccupied,
   eligible hold.

There is no auto-grip. Proximity alone does not attach.

The **slip model** reads the constraint force through each active weld after
every physics step. If it exceeds the hold's rated capacity × 1.25, the weld
releases and a slip event is logged.

---

## Action & Observation Spaces

```
action_space:      Box(low=-1, high=1, shape=(25,), dtype=float32)
observation_space: Box(low=-inf, high=inf, shape=(127,), dtype=float32)
```

### Action (25,)

```
[:21]    normalised joint targets in [-1, 1]
         → rescaled per joint to its MuJoCo ctrlrange at step time
[21:25]  grip intents for [LH, RH, LF, RF]
         > 0  →  engage weld (if tip is within proximity + hold eligible)
         ≤ 0  →  release weld
```

### Observation (127,)

```
[  0:  3)  pelvis world position                    (3)
[  3:  9)  pelvis orientation as rot6d              (6)  (first 2 cols of R)
[  9: 12)  centre-of-mass world position            (3)
[ 12: 33)  joint positions qpos[7:]                (21)  one per actuated joint
[ 33: 54)  joint velocities qvel[6:]               (21)
[ 54: 58)  per-limb grip flags [LH, RH, LF, RF]    (4)
[ 58:114)  K=8 nearest holds × 7                  (56)
               per hold:
                 relative position in pelvis frame  (3)
                 role one-hot [start, mid, finish]  (3)
                 is_gripping flag                   (1)
[114:126)  per-limb goal vectors                   (12)
               zero if limb is gripped;
               (finish_world − tip_world) otherwise
[126:127)  distance: highest gripped hand → nearest finish hold (1)
```

The observation shape is **always (127,)** regardless of wall size or number
of holds. A trained policy transfers to any wall, as long as the climber body
configuration is unchanged.

---

## Reward Function

| Component | Value | Notes |
|---|---|---|
| HWM height gain | `+5.0 × max(0, com_z − episode_max_com_z)` | Anti-oscillation: only new highs count |
| First-touch hold match | `+5.0` rising-edge | Deduped per `(limb, hold_id)` per episode |
| Slip | `−5.0 × n_slips` | Grip force exceeded hold capacity |
| Body intersection | `−20.0 × n_contacts` | Hard gate — makes self-intersection outright negative EV |
| Energy | `−0.005 × Σ ctrl²` | Discourages max-torque jitter |
| Terminal: finish | `+100` | One hand on a finish hold for ≥ 6 consecutive steps |
| Terminal: fall | `−50` | pelvis_z < 0.20 m |

---

## Current Status

### What works

- The wall editor is complete and stable. Walls save, load, and export
  correctly. The JSON schema is locked.
- The 3D simulator runs without NaN. The body has been visually verified in
  `mjpython`: symmetric commands produce symmetric poses, zero
  self-intersections at seed pose (fixed 2026-05-26).
- The Gymnasium env constructs, resets to a 4-grip seed pose, and steps.
  SB3 PPO trains against it.
- The MoonBoard adapter loads problems, handles the kickboard, and splits
  the corpus into train / validation / test.
- The curriculum environment generates new synthetic walls per episode with
  automatic difficulty scheduling.
- Video rollout callback and first/mid/last checkpoints work out of the box.

### What doesn't work yet

**The training loop has known blockers that prevent any learning. Success
rate is 0%.** The blockers are documented in `NEXT_STEPS.md`. The short list:

1. **Episode length** — defaults to 30 steps = 3.84 s of simulated time.
   A real MoonBoard problem takes 10–60 s. Fix: `max_episode_steps` → 800–1500.
2. **Seed pose on MoonBoard** — feet spawn above hands because kickboard
   holds aren't exposed to the foot-seeding logic.
3. **Grip semantics** — SB3's Gaussian policy initialises at mean=0, so
   grip intents are positive ~50% of the time by chance. The body releases
   ~2 limbs on step 1 and falls almost immediately.
4. **No observation normalisation** — raw world positions (pelvis at ~2 m,
   joint angles in radians) are at very different scales. Without
   `VecNormalize` this is a known PPO failure mode on MuJoCo environments.
5. **No dense shaping reward** — goal vectors are in the observation but not
   the reward. There is no gradient toward the next hold; the reward is
   flat until the agent accidentally reaches something.

Fix Phase 0 before anything else. The simulator is solid — the training
configuration is what's broken.

---

## Future Direction: The Path to Muscle-Level Control

The long-term goal is an agent that controls its body the way a human does:
by coordinating muscle activations that produce forces, not by targeting
joint angles directly. The roadmap:

### Step 1 — Get PPO working with position servos (current)

Fix the training-loop blockers above. First milestones:
- **Hang** — hold the seed pose for 10+ seconds without falling.
- **Reach one** — from a hang, move one hand to a target hold and regrip.
- **Climb** — string multiple moves together to the finish hold.

### Step 2 — Torque actuators

Replace the position-servo actuators with **torque-limited motors**. The
action now commands `ctrl[i] ∈ [−τ_max, +τ_max]` — direct joint torque,
not a target angle. The policy must learn to balance stiffness and gravity
entirely through force, which is physically closer to muscle activation.

In MuJoCo this means changing the actuator `gear` and `forcelimited`
attributes in `builder.py`. The action space shape stays (25,); only the
interpretation of the first 21 values changes.

### Step 3 — Hill-type muscle model

Each actuator becomes a muscle with:
```
F = F_max · a · f_L(l) · f_V(v) + F_passive(l)
```
where `a` is the activation signal (what the policy outputs), `f_L` is the
force–length relationship, and `f_V` is the force–velocity relationship.

This lets the agent discover:
- **Stretch-shortening cycles** — pre-loading a muscle before release gives
  "free" elastic energy (the mechanism behind explosive dynos).
- **Co-contraction** — stiffening a joint by activating opposing muscles
  simultaneously (how climbers lock off on crimps under load).
- **Pre-activation** — firing muscles before the load arrives so they're
  already generating force at the right moment.

This step adds ~5× model complexity. Defer until Step 2 produces a climbing
policy.

### Step 4 — GPU-scale training with MJX

The same MJCF runs on MuJoCo's JAX backend (`mujoco.mjx`) with minimal code
changes. This unlocks thousands of parallel rollout workers on a single GPU
— the sample throughput that Hill-muscle exploration demands.

### Why this matters

A discrete-move agent solves a climbing wall by memorising a mapping from
hold positions to limb placements. It cannot generalise to a wall it has not
seen, and it cannot discover emergent techniques like flagging, drop-knee, or
dynamic throws — those require understanding how the body generates and
absorbs momentum.

A continuous-control agent trained on physics has no such limitation. The
same policy that learns to hang can discover that certain grip sequences create
momentum, that flagging a free leg shifts the centre of mass to make a distant
hold reachable, or that a small hip rotation unlocks a move that brute-force
reach cannot make. These are not programmed — they emerge from the reward and
the physics.

---

## Project Layout

```
beta-engine/
├── grid_editor/         # Flask backend + REST API for the wall editor
│   ├── server.py
│   └── README.md
├── solver/              # 2D IK + A* (editor planning preview + wall gen)
│   ├── wall.py          # JSON loader + grid → cm conversion
│   ├── body.py          # 5-point stick figure + 2-link IK
│   ├── reachability.py
│   ├── astar.py
│   ├── generate.py      # procedural wall generator for curriculum
│   └── README.md
├── sim3d/               # MuJoCo 3D world + Gymnasium env + PPO trainer
│   ├── config.py        # all physics + reward constants
│   ├── body.py          # ClimberProfile + segment math + limb names
│   ├── builder.py       # build_mjcf_xml(wall, profile, include_kickboard)
│   ├── world.py         # Climb3DWorld — MjModel/MjData + grip + slip
│   ├── obs.py           # build_observation — fixed-shape (127,) obs
│   ├── env.py           # Climbing3DEnv — Gymnasium wrapper
│   ├── moonboard_env.py # MoonboardClimbing3DEnv — samples problems per reset
│   ├── moonboard.py     # MoonBoard problem JSON → Wall adapter
│   ├── curriculum.py    # CurriculumEnv — new synthetic wall every episode
│   ├── callbacks.py     # VideoRolloutCallback + checkpoint callbacks
│   ├── train.py         # SB3 PPO trainer + episode CSV logger
│   ├── viewer.py        # native MuJoCo viewer wrapper
│   ├── web.py           # Flask blueprint for the three.js front-end
│   ├── __main__.py      # `python -m sim3d` CLI (incl. --play <model.zip>)
│   └── README.md
├── schemas/             # wall.schema.json
├── static/              # vanilla-JS editor + sim3d browser viewer
└── data/
    ├── examples/        # seeded example walls (in git)
    ├── walls/           # user-saved walls (gitignored)
    ├── moonboard/       # MoonBoard problem corpus
    └── runs/sim3d/      # PPO run outputs (gitignored)
```

---

## Quickstart with Docker

Docker Desktop (or Docker Engine + Compose v2) required.

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

- Wall editor: http://localhost:8000
- 3D browser viewer: http://localhost:8000/sim3d/

### 3) Stop (walls are preserved)

```bash
docker compose down
```

Walls persist in `./data/walls/` on the host (bind-mounted into the container).

### 4) Smoke-test the simulator (headless)

```bash
docker compose exec beta-engine python -m sim3d --headless --frames 120
```

### 5) Step a MoonBoard problem in the native viewer

```bash
# Requires X11 forwarding on Linux:
xhost +local:docker
docker compose run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  beta-engine python -m sim3d \
    --moonboard data/moonboard/sample-problems.json --problem 19215
```

### 6) Train an RL agent

```bash
# Smoke run (~1 min on CPU).
docker compose exec beta-engine \
  python -m sim3d.train --steps 1500 --run-id smoke

# Real run on a single MoonBoard problem.
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
model_first.zip      # checkpoint at first update
model_mid.zip        # checkpoint at midpoint
model_last.zip       # checkpoint at end
videos/              # mp4 rollouts (if --video-freq > 0)
```

### 7) Replay a trained policy

```bash
# Native MuJoCo viewer (requires display):
python -m sim3d --play data/runs/sim3d/mb_v1/model.zip \
                --moonboard data/moonboard/sample-problems.json --problem 19215

# Browser viewer (no display needed):
# Start the server, then paste the run directory into
# "RL policy replay" at http://localhost:8000/sim3d/
```

### 8) Train with curriculum (procedurally generated walls)

```bash
docker compose exec beta-engine \
  python -m sim3d.train \
    --curriculum \
    --curriculum-start-difficulty 0.0 \
    --curriculum-max-difficulty 1.0 \
    --steps 500_000 \
    --run-id curriculum_v1
```

The curriculum generates a new synthetic wall each episode and
automatically raises or lowers difficulty based on the rolling success rate.

### Helpful Docker commands

```bash
docker compose ps                # running containers
docker compose logs -f           # follow logs
docker compose restart           # restart without rebuild
docker compose build --no-cache  # force clean rebuild
docker system prune              # reclaim disk
```

---

## Running Without Docker

```bash
python -m venv .venv
source .venv/bin/activate          # Linux / macOS
.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt

# Wall editor + 3D browser viewer (port 8000).
python -m grid_editor.server

# Native MuJoCo viewer with the bundled example wall.
python -m sim3d

# MoonBoard problem in the native viewer.
python -m sim3d --moonboard data/moonboard/sample-problems.json --problem 19215

# Headless smoke test (no display needed).
python -m sim3d --headless --frames 120

# Random Gym episode (env wrapper smoke test).
python -m sim3d --gym --gym-episodes 3

# Train.
python -m sim3d.train --steps 100_000 --run-id local_v1

# Quick env sanity check.
python -c "
from sim3d.moonboard_env import MoonboardClimbing3DEnv
from sim3d.moonboard import load_moonboard_problems
from sim3d.env import EnvConfig
problems = load_moonboard_problems('data/moonboard/sample-problems.json')
env = MoonboardClimbing3DEnv(problems[:1], config=EnvConfig())
obs, _ = env.reset(seed=0)
print('obs', obs.shape, 'action', env.action_space.shape)   # (127,) (25,)
"
```

The editor hard-codes `/data/walls/` to match the Docker bind-mount. For a
local run, edit `DATA_DIR` at the top of `grid_editor/server.py`.

---

## Programmatic Use

```python
from solver.wall import load_wall
from sim3d import Climb3DWorld, ClimberProfile

wall = load_wall("example-v2-boulder")
profile = ClimberProfile(height_cm=175, wingspan_cm=181, mass_kg=70)

world = Climb3DWorld(wall, profile)
world.seed_pose(lh="h_003", rh="h_004", lf="h_001", rf="h_002")

# Step 1 second of physics with slip detection.
slips = world.step(60, check_slip=True)
print(f"slip events: {slips}")

# Move a limb (snap = instant, reach = Cartesian-impedance PD).
world.move_limb("RH", "h_008", mode="snap")

# Read state.
print(world.com())                    # 3D centre of mass
print(world.pelvis_pos())
print(world.limb_tip_pos("RH"))
print(world.on_hold("RH"))            # → "h_008"
print(world.limb_grip_force("RH"))    # Newtons through this weld
```

### MoonBoard problems

```python
from sim3d.moonboard import load_moonboard_problems, moonboard_problem_to_wall
from sim3d import Climb3DWorld, ClimberProfile

problems = load_moonboard_problems("data/moonboard/sample-problems.json")
wall = moonboard_problem_to_wall(problems[0])   # 11×18 grid, 40° overhang
world = Climb3DWorld(wall, ClimberProfile())
```

### Gymnasium environment for RL

```python
from solver.wall import load_wall
from sim3d.env import Climbing3DEnv, EnvConfig

env = Climbing3DEnv(
    load_wall("example-v2-boulder"),
    config=EnvConfig(max_steps=1000, enable_slip=True),
)
obs, info = env.reset()
for _ in range(1000):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if term or trunc:
        print(info["outcome"])  # "completed" | "fell" | "timeout"
        break

# Drop-in with SB3:
from stable_baselines3 import PPO
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```

---

## TensorBoard

```bash
tensorboard --logdir data/runs/sim3d/<run_id>/tb
```

**What to look for in `episode_stats.csv` during training:**

| Column | Healthy sign |
|---|---|
| `length` | Increasing as the policy learns not to fall immediately |
| `outcome` | `completed` > 0% (currently never happens — see blockers above) |
| `final_com_z` | Trending upward (partial progress) |
| `n_slips` | Low — high values mean the policy is over-gripping |
| `body_intersections` | Low — each contact is penalised |

---

## Faster Training

```bash
# GPU for policy updates + parallel CPU MuJoCo workers.
python -m sim3d.train --steps 200_000 --device cuda --n-envs 8

# Colab (after enabling GPU runtime):
#   git clone https://github.com/Wall-Whisperers/beta-engine.git
#   cd beta-engine && pip install -r requirements.txt
#   python -m sim3d.train --steps 200_000 --device cuda --n-envs 2
```

Physics rollouts are CPU-bound; `--n-envs` usually gives more speedup than
GPU for typical run sizes. GPU matters for larger policy networks or very
long training runs.

---

## Hold JSON Schema (locked)

```json
{
  "wall_id": "my-wall",
  "grid": { "cols": 15, "rows": 20, "cell_size_cm": 20.0 },
  "holds": [
    {
      "hold_id": "h_001",
      "grid_x": 3,
      "grid_y": 1,
      "hold_type": "jug",
      "orientation_deg": 0.0,
      "size": "medium",
      "color": "#22c55e",
      "is_start": true,
      "is_finish": false
    }
  ]
}
```

**Hold types:** `jug`, `crimp`, `sloper`, `pinch`, `foothold`
**Sizes:** `small`, `medium`, `large`
**Orientation:** float 0.0–359.9° stored precisely, rendered snapped to 5° in UI
**cell_size_cm:** sets real-world scale (20 cm for MoonBoard, required for IK)

The schema is the contract. Every component reads this format. Do not change
it without a migration plan and team discussion.

---

## Branching & PRs

- `main` — protected, always working, no direct pushes
- `dev` — integration branch
- `feature/<short-name>` — branch from `dev`, PR back to `dev`

PR descriptions should include: scope summary, screenshots for any UI change,
and the manual validation checklist from
[`grid_editor/README.md`](grid_editor/README.md).

---

## See Also

- [`CLAUDE.md`](CLAUDE.md) — Architectural contract: action/observation byte
  layout, reward function, body model decisions, what not to do.
- [`NEXT_STEPS.md`](NEXT_STEPS.md) — Living roadmap: current blockers,
  training-loop fixes, curriculum, reward tuning.
- [`sim3d/README.md`](sim3d/README.md) — Deep-dive on the 3D simulator:
  why MuJoCo, body DOF table, hold attachment, slip model, obs/reward detail.
- [`grid_editor/README.md`](grid_editor/README.md) — Wall editor usage and
  manual validation checklist.
- [`solver/README.md`](solver/README.md) — 2D IK + A* solver internals.
