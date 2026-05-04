# Planning — Gabe's Notes

A working scratchpad for architecture thinking, decisions, and parking-lot
ideas. Anything Gabe drops in here is a brainstorm, not a contract — the
schema in `schemas/wall.schema.json` and the build phases in `CLAUDE.md`
remain the source of truth.

---

## End-to-end Blueprint (full project, multi-phase)

The eventual system turns 2D photos of a climbing wall into a structured
JSON world model, then has an RL agent solve the route. **The current
repo deliberately stops short of vision.** What follows is the long-term
target so we can keep the near-term work pointed in the right direction.

### Phase 1 — Vision Pipeline (image → JSON) — DEFERRED

Skipped for now. Captured here so the data contracts stay aligned.

#### 1.1 Data acquisition
- 5–10 ground-level photos from various angles.
- Problem: perspective distortion, foreshortening (top holds look small).
- Solution: **Structure-from-Motion (SfM)** via COLMAP / OpenCV to
  triangulate camera positions and produce a top-down orthomosaic.

#### 1.2 Grid calibration (the T-nut anchor)
- Use physical T-nut bolt holes as the coordinate system.
- **RANSAC grid fitting** can recover the full N×M grid even from ~50%
  visible bolts.
- Every detected hold snaps to the nearest T-nut coordinate.

#### 1.3 Semantic extraction (SAM + LLM)
- **SAM (Segment Anything)** for hold silhouettes.
- Multimodal LLM (e.g. GPT-4o) classifies each hold crop into:
  - `hold_type`: jug / crimp / sloper / pinch / foothold
  - `orientation_deg`: direction of the "good side"
  - `positivity_score`: 0.1–1.0, how grippable

### Phase 2 — Physical Model (JSON Schema)

The JSON is the source of truth. Gabe's brainstormed extension:

```json
{
  "wall_metadata": { "grid_unit_cm": 15, "angle_deg": 5 },
  "holds": [
    {
      "hold_id": "h_051",
      "grid_x": 4, "grid_y": 4,
      "type": "pinch",
      "orientation": 0,
      "positivity": 0.6,
      "is_start": false, "is_finish": false
    }
  ]
}
```

**Reconciliation with the locked schema in `schemas/wall.schema.json`:**

| Brainstorm field            | Current schema field          | Status                                                              |
|-----------------------------|--------------------------------|---------------------------------------------------------------------|
| `wall_metadata.grid_unit_cm`| (none — `grid.cell_size_cm` placeholder mentioned in CLAUDE.md) | **Schema discussion required** before Phase 3 (reachability). The solver currently hard-defaults `cell_size_cm = 20` cm with a warning. |
| `wall_metadata.angle_deg`   | (none)                         | Slab/overhang angle. Not in schema yet. Solver assumes a vertical wall (0°) for now. |
| `type`                      | `hold_type`                    | Naming difference only — keep `hold_type` (already locked).         |
| `orientation`               | `orientation_deg`              | Keep `orientation_deg` (already locked).                            |
| `positivity`                | (none)                         | Solver derives a default per `hold_type` (jug=0.95 … sloper=0.4). Could be added explicitly later. |

**Decision: do NOT change the schema yet.** Per CLAUDE.md ("Schema is the
contract"), schema changes need team discussion. We pull defaults inside
the solver and surface a `cell_size_cm` warning at load time.

### Phase 3 — RL Solver (the brain) — IN PROGRESS

Use a 2D physics environment (Pymunk for full physics later; for the MVP
we use a kinematic-only simulator).

#### 3.1 Agent — 2D MVP (5-point stick figure)
- 1 Center of Mass (COM)
- 4 end-effectors (LH, RH, LF, RF) with per-limb max-reach radii
- Stable iff the COM x-coord lies within the **stability triangle** —
  i.e. between the two active footholds (vertical-wall simplification)

#### 3.2 Reward function (sketch)
- `+ progress_reward` — COM closer to the finish hold
- `- efficiency_penalty` — every move costs a small amount
- `- stability_penalty` — large penalty if pose is unstable / unreachable
- `+ completion_bonus` — finish hold matched by a hand for 2 "ticks"

#### 3.3 Slab physics (later)
- Normal-force / friction model: leaning out drops foot friction.
- Out of scope for the MVP — assumes vertical wall, infinite friction.

### Phase 4 — Implementation Roadmap (MVP)

#### Step 1 — "Stick-figure gym"
- Build a 2D environment.
- Originally Pymunk + Gymnasium; **the MVP uses a pure-NumPy kinematic env**
  to avoid heavy deps. The interface is shaped like a Gym env so we can
  swap in Stable Baselines3 PPO later.
- Holds are points with grabbable zones derived from JSON orientation.

#### Step 2 — Training for generalization
- **Procedural generation**: train on 1,000 random walls so the agent learns
  principles (balance, weight shift) instead of memorizing a path.
- Out of scope for the MVP — single-wall demo first.

#### Step 3 — A* baseline
- Nodes = valid poses `(LH, RH, LF, RF)`.
- Edges = single-limb moves.
- A* gives the shortest mathematically-correct beta. We compare RL
  output against this baseline.

#### Step 4 — Final output
- "Beta map": visual overlay of move sequence on the wall.
- Step-by-step text instructions ("Step 4: shift weight right, LH → (4,7)").

### Tech stack summary

| Concern         | Long-term choice                | MVP choice (this repo right now)         |
|-----------------|---------------------------------|------------------------------------------|
| Vision          | COLMAP + OpenCV + SAM           | None (deferred)                          |
| Reasoning       | GPT-4o (hold classification)    | None (deferred)                          |
| 2D physics      | Pymunk                          | Pure NumPy kinematic stub                |
| 3D physics      | MuJoCo                          | Out of scope                             |
| RL framework    | Stable Baselines3 (PPO)         | Tabular Q-learning, no external RL dep   |
| Visualization   | matplotlib + photo overlay      | matplotlib (saves PNG/MP4 to `/data/runs/`) |
| Language        | Python                          | Python 3.12                              |

---

## Open questions for the team

1. **Schema: `cell_size_cm` and `angle_deg`.** When do we lock the
   real-world-scale fields into the schema? The solver is currently using
   defaults — that's fine for a demo but blocks Phase 3 generalization.
2. **Positivity score.** Per-`hold_type` defaults are workable, but a
   per-hold `positivity` would let the editor express "good crimp vs awful
   crimp" — worth adding when CV starts producing it anyway.
3. **Body model parameters.** Currently hard-coded to 175 cm height /
   175 cm wingspan (the "average climber" figure from CLAUDE.md). Where
   should user-supplied body params live? CLI flag for now, profile JSON
   later?
4. **RL upgrade path.** Tabular Q-learning is fine for ~13 holds. Beyond
   ~20 holds the state space (≈ N⁴ poses) explodes — that's the cue to
   move to PPO + Gymnasium + Stable Baselines3.

---

## Decisions log (most recent first)

- **2026-05-04** — Solver MVP uses pure-NumPy kinematic env + tabular
  Q-learning instead of Pymunk/SB3, to keep the dependency footprint
  small. Gym-style interface preserved for a clean upgrade path.
- **2026-05-04** — Default `cell_size_cm = 20` in the solver loader.
  Surfaces a warning. Schema unchanged pending team discussion.
- **2026-05-04** — Renamed `app.py` → `grid_editor/server.py` so the
  package boundary between *editor* and *solver* is explicit.
