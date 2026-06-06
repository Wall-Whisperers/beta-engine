# Climbing Beta Engine — Architectural Contract

> This file is the authoritative reference for AI collaborators and human
> developers alike. It captures decisions, invariants, and things that must
> **not** change without team discussion. Keep it accurate; delete stale
> sections rather than appending to them. Cumulative history lives in git.

---

## What This Is

An RL agent that learns to climb a MoonBoard by **continuously controlling a
custom 27-DOF humanoid** in MuJoCo. The agent emits joint position targets
and grip intents every control step — there is no discrete "snap to hold"
teleportation in the primary training mode. The body must discover climbing
from proprioception plus the 3D positions of the holds around it.

```
Wall JSON  →  MuJoCo MJCF  →  27-DOF climber  →  Gymnasium env  →  PPO policy
                                                      ↑ Box(25,) action
                                                      ↓ (127,) observation
```

**The long-term goal is continuous muscle-level control** — an agent that
drives joint torques coordinated by something closer to muscle activation
patterns, the way a real climber uses their body. The current PD-servo
actuators are a tractable first step toward that goal.

---

## Repository Layout

```
beta-engine/
├── grid_editor/         # Flask backend + REST API for the wall editor
│   └── server.py
├── solver/              # 2D IK + A* (editor planning preview + wall gen)
│   ├── wall.py          # JSON loader + grid-to-cm conversion
│   ├── body.py          # 5-point stick figure + 2-link IK
│   ├── reachability.py
│   ├── astar.py
│   └── generate.py      # procedural wall generator for curriculum
├── sim3d/               # MuJoCo 3D simulator + Gymnasium env + PPO trainer
│   ├── config.py        # all physics + reward constants (single source of truth)
│   ├── body.py          # ClimberProfile dataclass + segment math + limb names
│   ├── builder.py       # build_mjcf_xml(wall, profile) → (xml, hold_meta)
│   ├── world.py         # Climb3DWorld: MjModel/MjData + grip + slip + reach
│   ├── obs.py           # build_observation → fixed-shape (127,) vector
│   ├── env.py           # Climbing3DEnv — Gymnasium wrapper
│   ├── moonboard_env.py # MoonboardClimbing3DEnv — samples problems per reset
│   ├── moonboard.py     # MoonBoard problem JSON → Wall adapter
│   ├── curriculum.py    # CurriculumEnv — new synthetic wall every episode (wall-difficulty)
│   ├── staged_curriculum.py # StagedCurriculumEnv (hang→reach-one→climb, A1) + ClimbCurriculumEnv (full-climb reverse curriculum)
│   ├── callbacks.py     # VideoRolloutCallback, FirstMidLastCheckpointCallback
│   ├── train.py         # SB3 PPO trainer + episode CSV logger
│   ├── plot_reward_terms.py # per-term reward decomposition viewer (diagnostics)
│   ├── viewer.py        # native MuJoCo viewer wrapper
│   ├── web.py           # Flask blueprint for the Three.js browser viewer
│   └── __main__.py      # `python -m sim3d` CLI (incl. --play <model.zip>)
├── schemas/             # wall.schema.json (locked — see §Hold JSON Schema)
├── static/              # vanilla-JS editor + sim3d browser viewer
└── data/
    ├── examples/        # seeded example walls (in git)
    ├── walls/           # user-saved walls (gitignored)
    ├── moonboard/       # MoonBoard problem corpus
    └── runs/sim3d/      # PPO run outputs (gitignored except config.json)
```

---

## Stack

| Layer | Choice | Notes |
|---|---|---|
| Backend | Python 3.11+, Flask | REST API for wall CRUD + sim3d web bridge |
| Frontend | Vanilla JS + HTML | No build step. Three.js for 3D viewer. |
| 3D Physics | MuJoCo 3.2.6 | Industry-standard humanoid RL; great contact solver |
| RL | Stable-Baselines3 2.x, PPO | MlpPolicy, continuous action |
| RL Interface | Gymnasium 1.x | Drop-in SB3 compatible |
| IK / Math | NumPy | Custom 2D geometric IK for the editor solver |
| Infra | Docker + docker-compose | Consistent dev env |
| Data | JSON files in `data/` | Wall schema is the contract |

---

## Current Status (as of 2026-06-05)

### ✅ Done

- **Wall grid editor** — Flask REST API + vanilla-JS frontend. Hold
  placement, types (jug/crimp/sloper/pinch/foothold), orientations,
  start/finish markers. Save/load/delete JSON walls. Docker + compose.
- **3D MuJoCo simulator** — custom 27-DOF anatomically-grounded humanoid,
  wall/kickboard MJCF builder, grip via mocap+weld constraints, slip model,
  Cartesian-impedance reach controller, body intersection penalty.
- **Body physics correct** — right-side hinge axes mirrored; hip_flex
  points the right way for a wall-facing climber; feet toward wall; shoulder
  mount at correct height; deltoid spheres added. Zero self-intersections
  at seed pose on the example wall (verified in `mjpython` 2026-05-26).
- **Gymnasium env** — `Climbing3DEnv` + `MoonboardClimbing3DEnv`. Fixed
  obs shape (127,) and action shape (25,) regardless of wall size.
  Drop-in compatible with SB3 `PPO("MlpPolicy", env)`.
- **PPO training pipeline** — `sim3d.train`, episode CSV + TensorBoard,
  video rollout callback, first/mid/last checkpoints, parallel envs, GPU.
- **MoonBoard adapter** — loads problem JSON, kickboard support, train/val/
  test split, vertical-projection geometry option.
- **Curriculum env** — generates new synthetic walls per episode, automatic
  difficulty scheduling by rolling success rate.
- **Browser viewer** — Flask blueprint + Three.js client; replay a trained
  `model.zip` without a native display.

### ⚠️ What Doesn't Work Yet

**As of 2026-06-05 the agent learns its first climbing move.** Via the A1
staged curriculum (`staged_curriculum.py`), a PPO policy learned **reach-one**
— release a hand, reach a target hold, regrip — stably (monotonic, no collapse)
on a generated wall. Getting here required, in order:

1. **A clean potential-based reward** (height progress + hold-match + terminals;
   the old patchwork was never shipped — `train.py` re-layered it). DONE.
2. **The default-wall foot-gun** — 3 of 5 demo walls are unhangable for this
   body; default is now `baby-v1` + a startup hang-check. DONE.
3. **The physical blocker** — the body could not hold a one-hand stance with
   the old grips, so "cling forever" was optimal and nothing ever climbed.
   Fixed by stronger hands/feet (see §Grip strength). DONE.
4. **Two training collapses** — a PPO trust-region blowout (→ `target_kl`,
   `clip_range 0.1`, fewer epochs) and a target-blind observation (→ the
   mover's goal vector now points at the reach target). DONE.

**Still open:** the stable config learns slowly, so reach-one needs a long run
(≥1 M steps) to reach high success and advance through the curriculum into the
full **climb** stage; no end-to-end MoonBoard top-out yet. Earlier Phase-0
items (episode length, VecNormalize, log_std_init) are resolved in `train.py`.

---

## Body Model — The 27-DOF Humanoid

**This is a custom climbing-specific body, NOT the stock MuJoCo humanoid.**
Do not replace it. It has anatomically motivated joint limits, mass
distributions (Winter biomechanics), and per-group actuator gains tuned for
climbing.

### Degrees of Freedom

| Joint | DOF | Limits | Notes |
|---|---|---|---|
| pelvis root | 6 | free | position + quaternion; not actuated |
| spine_lean | 1 | −15° → +15° | forward lean; raised stiffness keeps near 0° |
| shoulder_az (×2) | 2 | −50° → +180° | forward/back swing |
| shoulder_el (×2) | 2 | 0° → +180° | abduction (full overhead reach) |
| shoulder_roll (×2) | 2 | −80° → +80° | internal/external rotation |
| elbow (×2) | 2 | 0° → +150° | **no hyperextension** |
| wrist (×2) | 2 | −70° → +70° | flex/extend; no radial deviation |
| hip_flex (×2) | 2 | −20° → +140° | high-step capable |
| hip_abduct (×2) | 2 | −20° → +70° | drop-knee, frog flag |
| hip_rot (×2) | 2 | −40° → +40° | |
| knee (×2) | 2 | 0° → +150° | **no hyperextension** |
| ankle (×2) | 2 | −25° → +45° | dorsi/plantar |
| **Total actuated** | **21** | | |
| **Total DOF** | **27** | | (6 free + 21 actuated) |

### Actuator Model

**Position actuators with per-group PD gains** (`config.py:
ACTUATOR_GAINS_BY_GROUP`). Tuned near critical damping for the load each
joint carries:

| Group | Kp (N·m/rad) | Kv (N·m·s/rad) |
|---|---|---|
| shoulder | 220 | 20 |
| elbow | 80 | 6 |
| wrist | 15 | 1 |
| spine | 250 | 22 |
| hip | 450 | 45 |
| knee | 200 | 17 |
| ankle | 40 | 3 |

Per-joint torque caps and armature are also in `config.py`. Change them
only with a measured justification — the comments explain the rationale.

### Hold Attachment

One mocap body + weld equality per limb. To attach: position the mocap at
the hold, set `data.eq_active[i] = 1`. To release: `eq_active[i] = 0`.
**No MJCF recompile needed** → fast RL resets. The weld `relpose` is set
so the **tip site** (fingertip / toe) lands on the hold, not the wrist /
ankle.

### Grip strength (training-phase, 2026-06-05)

A weld slips when its force exceeds `cap × SLIP_FORCE_SLACK`, where
`cap = base_force × positivity` (clamped by the hold's rating) and
`base_force = grip_force_n × HAND_FORCE_MULTIPLIER` (hands) or
`foot_push_force_n × FOOT_FORCE_MULTIPLIER` (feet). The multipliers were
raised — **`HAND_FORCE_MULTIPLIER 1.0→2.5`, `FOOT_FORCE_MULTIPLIER 1.5→3.0`** —
after a decisive finding: with the old values **the body could not hold a
one-hand stance**. Releasing either hand for a move overloaded the remaining
grips (the seed stances sit ~1.3× over cap) and the climber dropped. That is
the physical reason every prior run learned to *cling* and never climb — "let
go and reach" was a losing move. With the stronger grips a hand release leaves
a stable stance (verified) and the agent can climb. `GRIP_PROXIMITY_M` was also
loosened `0.05→0.08` so learned near-reaches convert to grips. These are a
**training-phase choice** — the roadmap defers realistic grip force to the
torque/muscle phase; tighten back toward 1.0 once the agent reliably climbs.

---

## Action Space (Canonical: `continuous-joint`)

```
Box(low=-1, high=1, shape=(25,), dtype=float32)

  action[:21]   per-joint RESIDUALS around the settled seed pose, in [-1, 1]
                  0   → hold the per-episode seed-pose joint target
                 +1   → drive that joint to its upper ctrlrange limit
                 -1   → drive that joint to its lower ctrlrange limit
                (ctrl = seed + a·(hi−seed) for a≥0; seed + a·(seed−lo) for a<0)
  action[21:25] grip intents for [LH, RH, LF, RF]
                > 0  →  engage weld if tip is within GRIP_PROXIMITY_M (0.05 m)
                        of an unoccupied eligible hold
                ≤ 0  →  release weld (if active)
```

Residual-around-seed (not raw midpoint) is what makes the calm init policy
(`log_std_init≈−1.5`) **hold the hang** at action≈0. The old midpoint mapping
yanked every joint ~42° off the settled seed on step 1, spiking the hands past
their slip cap and dropping grips before the policy could learn anything. Full
ctrlrange authority is preserved at ±1; the per-episode seed is captured in
`Climbing3DEnv.reset()` after the pose settles.

There is **no auto-grip**. The agent must raise grip intent above 0 AND
be within 5 cm of a valid hold. Proximity alone does not engage.

`discrete-move` (pick limb + hold; Cartesian-impedance reach controller) is
preserved **only** as a debug / curriculum / behavior-cloning tool. It is
not the canonical training mode.

---

## Observation Space

```
Box(low=-inf, high=inf, shape=(127,), dtype=float32)

[  0:  3)  pelvis world position          (3)
[  3:  9)  pelvis rot6d (cols 0,1 of R)  (6)
[  9: 12)  centre-of-mass world pos       (3)
[ 12: 33)  joint qpos[7:]               (21)   n_act
[ 33: 54)  joint qvel[6:]               (21)   n_act
[ 54: 58)  per-limb grip flags           (4)   LH RH LF RF
[ 58:114)  K=8 nearest holds × 7        (56)
               per hold: rel_pos_in_pelvis_frame (3)
                         role_onehot [start, mid, finish] (3)
                         is_gripping (1)
[114:126)  per-limb anchor/goal vectors  (12)
               zero when gripped; (nearest_reachable_hold − tip) otherwise.
               reach-one task mode: the MOVER limb's slot points at the
               designated target hold (target_world − tip) even while gripped,
               so the policy can perceive WHICH hold to reach.
[126:127)  Euclidean dist: highest gripped hand → nearest finish (1)
```

**Invariants:**
- Shape is always (127,) regardless of wall size or hold count. A saved
  policy is portable to any wall, as long as the climber body is unchanged.
- Every stream is NaN/Inf-guarded with a one-time stderr warning (`obs.py`).
- Hold positions expressed in pelvis-local frame so the representation is
  pose-relative, not world-absolute.

---

## Reward Function (per step, `continuous-joint`)

**Clean restart (2026-06-05).** Every shaping term we ever added either got
farmed or blocked something else, so the reward was a patchwork of guards
against the previous week's exploit. It is now rebuilt around one honest idea:
**reward raising the body, symmetrically, and let the discount factor — not a
per-step penalty — create the "climb promptly" pressure.** Two climbing-shaping
terms, two terminals, three physics gates. Nothing else is on by default.

| Component | Value | Category |
|---|---|---|
| **Height progress** | `+60.0 × (com_z − prev_com_z)` | primary dense — potential-based, symmetric (up pays, down costs), telescopes ⇒ un-farmable |
| First-touch hold match | `+10.0` rising-edge, deduped per `(limb, hold_id)`/episode | sparse grip nudge — the only grip incentive |
| Terminal: finish | `+200` | one hand on finish hold ≥ 6 consecutive steps |
| Terminal: fall | `−50` | pelvis_z < 0.20 m |
| Body intersection | `−20.0 × n_contacts` | physics gate (validity, not shaping) |
| Slip | `−5.0 × n_slips` | physics gate |
| Energy | `−0.001 × Σ ctrl²` | physics gate (anti-jitter, tiny) |

**Design rationale:**
- **Why potential-based `com_z`, not HWM.** The old high-water-mark only paid
  for *new* max height; sliding back down was free, so it could not punish lost
  progress. `K·(com_z − prev_com_z)` is symmetric and, by the potential-shaping
  theorem (Ng et al. 1999), policy-invariant and un-farmable by oscillation —
  which was the whole reason HWM needed the one-way ratchet. `prev_com_z` inits
  at the settled seed com_z, so step 1 earns `(com_z_after − seed)`, never a
  free bonus.
- **No time / stagnation penalty.** A per-step living cost interacts lethally
  with the `−50` fall terminal: falling *ends* the episode, so any cost above
  ~0.05/step makes "release and fall on step 1" beat hanging for a full episode.
  Since the agent can't climb yet, PPO finds that death-spiral first. "Climb
  promptly" pressure comes from the PPO discount γ and the symmetric height
  potential (stalling earns 0 while climbing earns positive). Add an explicit
  efficiency penalty only *after* the agent reliably tops out.
- **Physics gates ≠ shaping.** Intersection / slip / energy keep the solution
  physical; they are not climbing-shaping and stay on.
- **Inert legacy levers (default 0):** `hwm_height_scale`, `finish_approach_coeff`
  (B2 reference-jump now fixed — `_finish_dist` measures from the highest hand
  *tip*, so it's an un-farmable signed potential; tried in the climb experiments
  but it does not crack the chaining frontier — see NEXT_STEPS A1b),
  `reach_approach_coeff` (fall-and-swing exploit — confirmed it still farms),
  `survival_bonus_coeff` (floor-hang attractor), `new_high_grip_bonus`
  (grab→fall→repeat magnet), `grip_release_penalty`, `upward_velocity_coeff`.
  Code paths are kept behind `if coeff > 0` so terms can be re-added **one at a
  time, diagnostics-driven** — never all at once.

**Diagnostics.** `episode_stats.csv` logs a per-term decomposition
(`r_height, r_match, r_reach, r_finish, r_fall, r_slip, r_intersect, r_energy,
r_other`) that sums to the episode return, plus `stage` / `rc_pos` for the
staged curriculum. `r_height` should dominate in climb mode; any other column
creeping up is the next exploit surfacing. View with
`python -m sim3d.plot_reward_terms <run>/episode_stats.csv`.

### Task-stage curriculum (A1) — `EnvConfig.task_mode`

To break the "hangs but won't climb" exploration trap, `task_mode` gates the
reward into an achievable progression (`StagedCurriculumEnv` auto-advances it on
rolling success; `python -m sim3d.train --staged-curriculum`):

- **`hang`** — reward staying on the wall; success = survive `hang_target_steps`.
- **`reach-one`** — one designated *mover* hand must release and grip a *target*
  hold. Reward = **dense signed-potential** pull toward the target
  (`reach_one_coeff × Δdist`, gated on the other ≥2 limbs anchored so the
  fall-and-swing farm can't return) + **`+50` regrip** bonus/success. Height
  reward is **off** in this mode (the high reverse-curriculum seed makes a fall's
  `−K·Δz` swamp the reach signal). The mover starts seeded on a stance vetted to
  hang; the obs points its goal vector at the target (see §Observation Space).
- **`climb`** — the full clean reward above.

Default `task_mode="climb"` — the curriculum is opt-in and leaves normal
training unchanged.

---

## Coordinate System

`+X` along the wall (left→right), `+Y` away from the wall (climber side),
`+Z` up. Gravity is world `−Z`. Wall base sits at `z = 0`. Slab/overhang
is implemented by tilting the wall plate, not rotating gravity.

---

## Wall JSON Schema (locked — discuss before changing)

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
      "size": "small | medium | large",
      "color": "#22c55e",
      "is_start": true,
      "is_finish": false
    }
  ]
}
```

**Hold types:** `jug`, `crimp`, `sloper`, `pinch`, `foothold`
**Sizes:** `small`, `medium`, `large`
**Orientation:** float 0.0–359.9°, stored precisely, rendered snapped to 5°
**cell_size_cm:** required for real-world scale — IK and 3D builder both
need it; set to 20 cm by the MoonBoard adapter.

The schema is the contract. Every subpackage reads this format. Changes
require migration of all saved walls and team approval.

---

## Architecture Decisions

### 1. Continuous joint control is the primary mode

The agent controls every actuated joint simultaneously every step, like a
neural motor cortex commanding all muscles at once. This is harder to learn
than discrete-move but is the correct framing for the long-term goal of
muscle-level control. Discrete-move remains available as a curriculum
bootstrap only.

### 2. Custom body — not the stock MuJoCo humanoid

The stock humanoid is tuned for locomotion. It has wrong joint limits for
climbing (no high-step, no overhead reach), wrong mass distribution, and no
hold-attachment machinery. The custom body is anatomically grounded. Do not
swap it out.

### 3. Hold attachment via mocap+weld

A connect constraint only constrains position. A weld without a mocap
requires recompiling MJCF to move the target. Mocap+weld lets us teleport
the anchor without model recompile — essential for fast RL resets.

### 4. Fixed-shape observation (K-nearest holds, not a board one-hot)

A board-sized one-hot would be wall-specific (MoonBoard 11×18 vs 15×20
editor wall) and a policy would not transfer. K=8 nearest holds in pelvis
frame is wall-size-agnostic. This is the invariant that makes a single
saved policy work across any wall with the same body.

### 5. Do not model fingers

Crimp vs jug is modeled through grip-force capacity (`max_force_n` per
hold), not finger flexion. This is the right complexity/fidelity tradeoff
for the current phase. Revisit only when predicting per-hold grade difficulty
for specific climber strength profiles.

### 6. MuJoCo over PyBullet / Genesis / Brax

MuJoCo's contact solver is stable under tendon-like weld constraints.
PyBullet jitters under force-limited grips. Genesis is too bleeding-edge.
We can move to MJX (MuJoCo on JAX) for GPU-scale training without
rewriting the model.

### 7. PD servos now, torque / muscle later

Hill-type muscles add ~5× model complexity. PD position servos are the
right level of detail for the current phase. Once PPO learns to climb with
servos, migrating to torque actuators (and eventually muscle activation) is
the next step — not a prerequisite.

---

## Future Direction: Toward Muscle-Level Control

The current PD-servo architecture is a stepping stone. The full roadmap:

**Step 1 — Get PPO working with position servos (Phase 0–A)**
Fix the training-loop blockers: episode length, VecNormalize, seed pose,
grip semantics, dense shaping reward. First targets: hang stably → reach
one hold → climb a route.

**Step 2 — Torque actuators**
Replace position actuators with torque-limited motors. The policy now
commands effort, not target angle. This forces the policy to implicitly
learn stiffness and damping — physically closer to muscle activation.
Action space shape stays the same; MuJoCo actuator gear type changes.

**Step 3 — Hill-type muscle model (long-term)**
Each actuator becomes a muscle: `force = f(activation, length, velocity)`
following the Hill model. The agent can then discover stretch-shortening
cycles, pre-activation, and elastic energy storage — real biomechanics.
Adds ~5× model complexity; defer until Step 2 trains.

**Step 4 — MJX / GPU-scale training**
The same MJCF runs on MuJoCo's JAX backend (MJX) with minimal code
changes. This unlocks massive parallelism for the rollouts that muscle-level
training demands.

**Why continuous over discrete-move:**
A discrete-move agent solves a wall by pattern-matching (hold_A → limb_B).
A continuous-control agent discovers movement from physics and generalises
to any wall, any body proportions — because the solution is understanding
how the body works, not memorising hold sequences.

---

## What NOT to Do

- **Switch back to `discrete-move` as the default.** Discrete-move hides
  the real problem. Keep it only as a curriculum bootstrap / expert for BC.
- **Replace the custom climber with the stock Gymnasium humanoid.** It
  trains faster on locomotion tasks; it is not a climbing body.
- **Add DOF before the agent can climb with the current DOF.** More joints =
  harder exploration. Add spine lateral + rotation (E1) only after the agent
  reliably climbs with the current 21 actuated joints.
- **Change obs/action shape without updating this file.** A shape mismatch
  silently produces garbage predictions on checkpoint load.
- **Add `KNOWN_ISSUES.md`, `ARCH_REVIEW.md`, or per-phase trackers.**
  `NEXT_STEPS.md` is the only roadmap. Keep it pruned.

---

## Running Locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows
source .venv/bin/activate           # Linux / macOS
pip install -r requirements.txt

# Wall editor + 3D browser viewer (port 8000)
python -m grid_editor.server

# Headless smoke test (no display needed)
python -m sim3d --headless --frames 120

# Env / obs sanity check
python -c "
from sim3d.moonboard_env import MoonboardClimbing3DEnv
from sim3d.moonboard import load_moonboard_problems
from sim3d.env import EnvConfig
problems = load_moonboard_problems('data/moonboard/sample-problems.json')
env = MoonboardClimbing3DEnv(problems[:1], config=EnvConfig())
obs, _ = env.reset(seed=0)
print('obs', obs.shape, 'action', env.action_space.shape)  # (127,) (25,)
"
```

## Running with Docker

```bash
docker compose up --build
# http://localhost:8000          → 2D wall editor
# http://localhost:8000/sim3d/   → 3D browser viewer

# Train
docker compose exec beta-engine \
  python -m sim3d.train --steps 200_000 --n-envs 8 --run-id my_run

# Replay
docker compose exec beta-engine \
  python -m sim3d --play data/runs/sim3d/my_run/model.zip
```

## Git Conventions

- `main` — protected, always working, no direct pushes
- `dev` — integration branch
- `feature/<name>` — branch from dev, PR back to dev
- One person owns merging to main
- Review PRs before merging to dev
