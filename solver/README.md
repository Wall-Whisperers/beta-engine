# Solver

A **2D-only** inverse-kinematics + route-finding solver for climbing
walls. Reads the wall JSON files produced by the grid editor and outputs:

- A **move sequence** — which limb (LH/RH/LF/RF) moves to which hold, in order.
- A **PNG panel sheet** — one subplot per pose, with stick-figure overlay.
- An **animated GIF** (optional) — the same sequence as a flipbook.

This is the Phase 2/3 MVP from [`CLAUDE.md`](../CLAUDE.md): body model + IK
+ reachability + graph search, all working together before computer
vision or a heavier physics engine gets layered on top.

> ### ⚠️ This is a 2D model.
> Every check — pose, IK, reachability, stability, joint envelopes — is
> done in the **wall plane** (x = horizontal, y = vertical). There is
> **no body twist, no out-of-plane drop-knee, no shoulder roll, no
> friction**. Real climbing 3D moves (gastons, drop-knees, flagging) are
> approximated by 2D joint-angle envelopes — the constants in
> `body.py`. The intent is to ship a working pipeline first; the next
> phase swaps in Pymunk for proper 2D physics, and then 3D + MuJoCo
> later. See [`../CLAUDE.md`](../CLAUDE.md) for the current architecture.

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

# A different climber — height/wingspan literally change which betas exist
docker compose exec beta-engine python -m solver --wall example-v2-boulder \
  --height-cm 190 --wingspan-cm 185 --gif
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
| `--cell-size-cm` | `20` | Override grid cell size in cm (the schema doesn't carry this yet) |
| `--episodes` | `2000` | Q-learning training episodes (ignored by `astar`) |
| `--no-viz` | off | Skip rendering, just print the move list |
| `--gif` | off | Also render an animated GIF alongside the PNG |

---

## How it works (per module)

The pipeline is a strict left-to-right chain of files in `solver/`. Each
file is small enough to read in one sitting. The reasoning below is
intentionally explicit so the model is easy to tweak — every numeric
constant has a docstring saying what it represents and which direction
to push it.

### 1. `wall.py` — Wall + holds in cm

**Reads:** `data/walls/<id>.json` or `data/examples/<id>.json`.
**Produces:** `Wall` and `Hold` dataclasses with positions in **cm**, not grid cells.

Grid cells are converted with `cell_size_cm` (defaults to `20` cm with a
warning — the single biggest open schema question). Each hold's centre is at
`((grid_x + 0.5) * cell_size_cm, (grid_y + 0.5) * cell_size_cm)`. y grows
upward (climbing convention).

Holds are also bucketed by **per-type usability**:
- `HAND_USABLE_TYPES = {jug, crimp, sloper, pinch}` — feet can't anchor on jugs/crimps if you don't want them to (today they can).
- `FOOT_USABLE_TYPES = {jug, crimp, sloper, pinch, foothold}`.

Per-type **positivity** (0–1, default per type) acts as a tie-breaker in
A\*'s cost function: jugs are cheaper to use than slopers. When the
schema gains a per-hold `positivity_score` (likely after CV), drop the
defaults.

**Tweakable knobs:** `DEFAULT_CELL_SIZE_CM`, `POSITIVITY_BY_TYPE`,
`HAND_USABLE_TYPES`, `FOOT_USABLE_TYPES`.

### 2. `body.py` — Stick figure + IK + joint envelopes

A 5-point stick figure:

| Point | Role |
|-------|------|
| **COM** (Center Of Mass) | Approximated as the pelvis. Drives where shoulders/hips sit. |
| **LH** (Left Hand) | End-effector. Anchored at the left shoulder. |
| **RH** (Right Hand) | End-effector. Anchored at the right shoulder. |
| **LF** (Left Foot) | End-effector. Anchored at the left hip. |
| **RF** (Right Foot) | End-effector. Anchored at the right hip. |

Each limb is a **2-link chain** (upper + lower segment, equal length)
solved with closed-form **law-of-cosines IK** — no iteration, no library:

```
delta   = target - anchor
dist    = |delta|
cos_a   = (upper² + dist² - lower²) / (2 × upper × dist)
elbow   = anchor + upper × [cos(atan2(dy,dx) ± arccos(cos_a)),
                            sin(atan2(dy,dx) ± arccos(cos_a))]
```

`elbow_up=True` for arms (elbow above the shoulder→target line);
`elbow_up=False` for legs (knee bends forward and down).

**Anthropometric defaults** (175 cm / 175 cm wingspan):

| Measurement | Formula | Default |
|-------------|---------|---------|
| Arm length (per side) | `0.42 × wingspan_cm` | 73.5 cm |
| Leg length (hip → ankle) | `0.48 × height_cm` | 84.0 cm |
| Shoulder offset from COM | `0.10 × height_cm` | 17.5 cm |
| Hip offset from COM | `0.06 × height_cm` | 10.5 cm |
| Shoulder height above COM | `0.30 × height_cm` | 52.5 cm |

> All of these are properties on `BodyModel`; override them in a
> subclass or just bump the formulas. Real anthropometry is messier (arm
> length isn't exactly half wingspan), but the first-pass coefficients
> are good enough that the solver's results match what a real climber
> would do on the example wall.

**Joint angle envelopes — anatomically inspired**

A circle test alone lets the model put a foot above the climber's head.
Real climbers can't. So each limb also has a **valid envelope** — an
axis-aligned box in the limb's body-local frame (origin = shoulder/hip
anchor) that the end-effector must lie inside:

| Constant | Default | What it means |
|----------|---------|---------------|
| `HAND_MAX_ABOVE_SHOULDER_FRAC` | `1.00` | Hand can reach fully overhead — `1.0 × arm_length` above the shoulder. |
| `HAND_MAX_BELOW_SHOULDER_FRAC` | `0.85` | Hand can drop to roughly the climber's hip — `0.85 × arm_length` below the shoulder. Allows mantling, low gastons. |
| `HAND_IPSILATERAL_REACH_FRAC`  | `1.00` | Full arm length on the limb's own side (LH to the left, RH to the right). |
| `HAND_CROSS_BODY_FRAC`         | `0.75` | Cross-body grabs (cross-overs / cross-unders). Loose at the single-limb level — the pose-level limit below catches fully-swapped hands. |
| `FOOT_MAX_ABOVE_HIP_FRAC`      | `0.30` | High-step ceiling. **Most climbers can't put a foot above the hip without serious flexibility** — this is the constant to tighten if you want a more conservative model. |
| `FOOT_MAX_BELOW_HIP_FRAC`      | `1.00` | Full leg extension downward. |
| `FOOT_IPSILATERAL_REACH_FRAC`  | `1.00` | Full leg sideways on own side. |
| `FOOT_CROSS_BODY_FRAC`         | `0.40` | Drop-knee crossover limit. |

**Pose-level (multi-limb) constraints** in the same file:

| Constant | Default | What it means |
|----------|---------|---------------|
| `HAND_CROSSOVER_LIMIT_CM` | `40.0` | Max LH-x minus RH-x when LH is right of RH. A *cross* move is fine; *cross and keep going* (LH ends up wildly right of RH) is rejected. |
| `FOOT_CROSSOVER_LIMIT_CM` | `35.0` | Same idea for feet. |
| `END_EFFECTOR_MIN_SEPARATION_CM` | `8.0` | Two end-effectors can't be closer than this (basic body-collision approximation; per-hold occupancy is enforced separately). |
| `FOOT_ABS_CEILING_BELOW_SHOULDER_CM` | `5.0` | Hard rule: `target.y ≤ COM.y + shoulder_height − 5cm`. A foot above the shoulders is anatomically impossible regardless of how loose the per-limb fracs get. |

> **How to tune:** if the solver rejects moves a real climber would
> pull, loosen the relevant `*_FRAC` constant. If it accepts impossible
> moves, tighten them. The constants in `body.py` are deliberately the
> only knobs needed.

### 3. `reachability.py` — Pose, three-stage reach test, stability

A **`Pose`** is a tuple of four hold IDs: `(LH, RH, LF, RF)`. Whichever
limb is `None` is "in flight" — only used during a one-limb-at-a-time
move.

**`can_reach(body, com, limb, target)`** returns `True` only when *all
three* of these pass:

1. **Distance test** — `|anchor → target| ≤ REACH_SAFETY × max_reach`
   (`REACH_SAFETY = 0.92`). Fully extended limbs technically reach
   further but it's physiologically miserable: no margin for adjustment,
   joint locked out.
2. **Envelope test** — target sits inside `BodyModel.envelope_box(limb)`.
   This is the anatomical filter described above.
3. **IK test** — closed-form 2-link IK actually solves. Catches
   near-anchor targets where the limb would have to fold past itself
   (`dist < |upper - lower|`).

**`is_stable(wall, pose)`** — vertical-wall stability rule. The COM
x-coord must lie within the horizontal span of the two active footholds
(or within `STABILITY_TOLERANCE_CM = 5.0` cm of the single foot, in
intermediate three-point states). This is a deliberately simple rule
appropriate for a vertical wall and a static (no-momentum) model. For
overhanging or dynamic moves, swap in a real physics check.

**`pose_anatomy_ok(wall, pose)`** — pose-level (multi-limb) checks:
hand/foot crossover limits and end-effector minimum separation.

**`reachable_moves(body, wall, pose)`** is the move generator the
search algorithms call. A candidate one-limb move is legal when:

- The **moving** state (limb in flight, 3 points of contact) passes `_three_points_stable`.
- The **target** hold passes `can_reach`.
- The **resulting** 4-limb pose passes both `is_stable` AND `pose_anatomy_ok`.

**Tweakable knobs:** `REACH_SAFETY`, `STABILITY_TOLERANCE_CM`,
`COM_HEIGHT_BIAS` (how much the COM is biased toward the feet vs the
hands — currently `0.55`).

### 4. `astar.py` — A\* baseline (the "mathematically shortest" beta)

- **Nodes** = poses `(LH, RH, LF, RF)`
- **Edges** = legal one-limb moves from `reachable_moves`
- **Heuristic** = vertical distance from the higher hand to the nearest finish hold
- **Cost** per move = `1.0 + 0.5 × (1 - target.positivity)` — base cost of one move plus a penalty for using bad holds (slopers cost more than jugs)
- **Multi-source** = `starting_poses(wall, body)` enumerates plausible 4-limb starting matchups (start markers for hands, the closest-stable foot pair) and pushes them all onto the open set with cost 0

A\* is the natural baseline because the state space is tiny (a few
hundred to a few thousand poses for a typical wall). It also gives us a
ground-truth "shortest" beta that the RL agent can be compared
against.

**Why these heuristic + cost choices?**
- Vertical distance to finish is **admissible** (you have to gain that
  height eventually) and cheap — no IK or stability calls needed.
- Per-move cost = 1 makes A\* prefer fewer moves, the simplest "good
  beta" proxy.
- Positivity penalty is **small enough not to break admissibility** in
  practice (the heuristic dominates) but biases ties toward better
  holds.

**Tweakable knobs:** `_heuristic`, `_move_cost`, `max_expansions`
(default `20_000`).

### 5. `rl_qlearn.py` — Tabular Q-learning RL

A **dependency-free** RL implementation. State = pose tuple; action =
`(limb, target_hold)` from `reachable_moves`.

The env class (`ClimbingEnv`) is shaped like a Gym `reset()/step()`
interface, deliberately, so the project now also includes SB3-compatible environments in `rl/` and `sim3d/env.py`.

**Reward shaping:**

| Signal | Default | Why |
|--------|---------|-----|
| Progress toward finish (per cm closer) | `+0.05` | Dense reward — gives signal at every step instead of only at the goal. Tune up if learning is slow, down if the agent gets stuck oscillating. |
| Efficiency penalty (each move) | `−0.5` | Fixed cost to discourage wandering. Without it, the agent has no reason to take short paths. |
| Stability penalty (unstable result) | `−50` | Filtered out by `reachable_moves`, but kept here for safety in case future env edits introduce unstable transitions. |
| Completion bonus (hand on finish hold) | `+100` | Sparse goal reward. Big enough to dominate even after several efficiency penalties. |
| Dead-end penalty (timeout / no actions) | `−5` | Discourages the agent from painting itself into corners. |
| Max steps per episode | `30` | Hard cap on episode length. |

**Hyperparameters:**

| Knob | Default | Notes |
|------|---------|-------|
| `episodes` | `2000` | Episodes of training. The example wall converges in ~600 — bump up for bigger walls. |
| `alpha` | `0.5` | Learning rate. High because the env is deterministic; lower it for stochastic envs. |
| `gamma` | `0.95` | Discount factor. |
| `epsilon_start / epsilon_end` | `1.0 → 0.05` | Linear ε-decay. Pure exploration at the start, mostly greedy by the end. |

**Why tabular instead of PPO?**
- The state space for a 13-hold wall is at most `13⁴ ≈ 28k` tuples (in
  practice far less — most 4-tuples aren't valid poses). Tabular handles
  that easily, no neural network needed.
- Zero external deps (no `gymnasium`, no `stable-baselines3`, no
  PyTorch). The Docker image stays tiny.
- The point of this MVP is to prove the pipeline runs end-to-end. PPO
  is a one-day swap once the env contracts are right.

**When to upgrade to PPO:**
- Walls bigger than ~20 holds (state space starts to explode).
- Procedural-generation training — train on many random walls so the
  agent learns *principles* rather than memorizing one route.
- Continuous-action body models (the next step beyond pose-graph search,
  if you want the agent to learn smooth dynamic moves).

**Tweakable knobs:** all the reward constants and hyperparameters above
(top of `rl_qlearn.py`).

### 6. `visualize.py` — matplotlib renderer

Headless matplotlib (`MPLBACKEND=Agg`, no display required). Renders:

- **PNG panel sheet** — one subplot per pose in the solution. Each panel
  shows the wall (holds colour-coded by type, start/finish rings), the
  stick figure (coloured per limb: LH red, RH green, LF blue, RF
  yellow), and a title with the move description.
- **Animated GIF** (opt-in via `--gif`) — same frames at 2 fps.

The stick figure uses the same IK as the solver — the elbow/knee
positions in the visualization are exactly the joints the IK solver
would find for that pose. So if a pose looks weird, it's the model
saying "this is the IK solution but you should probably reject it" —
a useful debugging signal.

Outputs land in `/data/runs/` inside the container, which maps to
`./data/runs/` on your host via the Docker bind mount.

---

## Constants summary — where to tweak what

| Want to change… | Edit… | File |
|-----------------|-------|------|
| Real-world cell size | `DEFAULT_CELL_SIZE_CM` (or `--cell-size-cm`) | `wall.py` |
| Per-hold-type quality | `POSITIVITY_BY_TYPE` | `wall.py` |
| Climber proportions | `BodyModel.arm_length` etc. | `body.py` |
| Joint angle envelopes | `*_FRAC` constants | `body.py` |
| Pose-level crossover/collision | `*_LIMIT_CM` constants | `body.py` |
| Reach safety margin | `REACH_SAFETY` | `reachability.py` |
| Stability tolerance | `STABILITY_TOLERANCE_CM` | `reachability.py` |
| COM bias toward feet vs hands | `COM_HEIGHT_BIAS` | `reachability.py` |
| A\* heuristic / cost | `_heuristic`, `_move_cost` | `astar.py` |
| RL rewards | `PROGRESS_REWARD_PER_CM` etc. | `rl_qlearn.py` |
| RL hyperparameters | `solve_qlearn(episodes=…, alpha=…, gamma=…)` | `rl_qlearn.py` |

---

## Known limits (intentional, MVP)

| Limitation | Why it's OK for now | How to fix later |
|------------|---------------------|------------------|
| **2D wall-plane reasoning** | Solver is a fast graph/search baseline, not the physics source of truth | Use `physics/` for 2D dynamics or `sim3d/` for MuJoCo body motion |
| **Static (no momentum)** | A* needs deterministic reachability and cheap edge checks | Validate candidate betas in `physics/` or `sim3d/` after solving |
| **Tabular Q-learning won't scale past ~20 holds** | State space is N⁴ — fine for the bundled 13-hold example | Use SB3 PPO with `rl/` or `sim3d.train` for neural policies |
| **Single-wall tabular RL** | Keeps the toy baseline understandable | Train over MoonBoard/custom wall sets in `sim3d/` for generalisation |
| **No body-on-body collision** | Single rough end-effector separation check is enough for graph pruning | Let MuJoCo collision/joint limits handle this in `sim3d/` |

---

## Module reference

| File | Responsibility |
|------|----------------|
| `wall.py` | Load wall JSON, convert grid cells to world (cm) coords, hold-type utilities |
| `body.py` | `BodyModel`, `solve_2link_ik`, `envelope_box`, `resolve_skeleton` |
| `reachability.py` | `Pose`, `can_reach`, `is_stable`, `pose_anatomy_ok`, `reachable_moves` |
| `astar.py` | `solve_astar`, `starting_poses`, `SolveResult` |
| `rl_qlearn.py` | `ClimbingEnv`, `solve_qlearn` |
| `visualize.py` | `render_panels` (PNG), `render_animation` (GIF) |
| `__main__.py` | CLI — argument parsing, orchestrates load → solve → render |
