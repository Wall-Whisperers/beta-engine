# Climbing Beta Engine — Team Context

## What This Is

An app that takes a photo of a climbing wall, extracts hold positions into a structured world model, and generates a personalized move sequence (beta) based on the user's height and wingspan.

The core insight: don't feed raw images into a solver. Extract a structured representation first, then solve on that. This makes the system tractable and debuggable.

**Pipeline:**
```
Photo(s) → Hold Detection → World Model (JSON) → Body Model + IK → Solver → Move Sequence
```

---

## Current Status

**Phase 1 is complete.** The wall grid editor (`beta-engine`) is built and running:
- Flask backend serving a REST API for wall CRUD (`/api/walls`)
- Vanilla JS frontend with a grid-based hold editor
- Hold placement, types, orientations, start/finish markers
- Save/load/delete walls as JSON, download/upload, live JSON panel
- Docker + docker-compose for consistent dev environments
- Example wall seeded on startup (`example-v2-boulder`)

**What's missing before Phase 2:**
- No body model or IK solver yet
- No reachability checker
- No route/move-sequence concept in the schema (holds aren't ordered)
- No real-world scale (cells have no cm/inch mapping — reachability can't be computed)

---

## Stack

| Layer | Choice |
|---|---|
| Backend | Python 3.11+, Flask (MVP) → FastAPI (later) |
| Frontend | Vanilla JS + HTML canvas (current) → React (later) |
| ML / CV | YOLOv8 (hold detection), PyTorch (RL solver) |
| IK / Math | NumPy, ikpy or custom 2D geometric solver |
| Infra | Docker + docker-compose |
| Data | JSON files in `/data/walls/` |

---

## Hold JSON Schema (locked — discuss before changing)

```json
{
  "wall_id": "my-wall",
  "grid": { "cols": 15, "rows": 20, "cell_size_cm": null },
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
**Orientation:** float 0.0–359.9°, stored precisely, rendered snapped to 5° in UI
**cell_size_cm:** `null` for now — must be set before reachability can be computed

**What the schema cannot express yet:**
- Ordered move sequence (no route object)
- Which limb goes to which hold
- Real-world scale

---

## MVP vs Ideal

### MVP
- Manual hold placement (user taps grid — no CV)
- Fixed body model using average height (175cm) and wingspan (175cm)
- 2D IK solver (law of cosines, geometric — no library needed)
- Graph search (A* or BFS) to find a valid move sequence
- Output: text sequence of moves (`LH → h_003, RF → h_007...`)
- No photo input yet

**Why manual first:** CV is the hardest part. Build the solver and body model first so you know the pipeline works. If CV output is bad you'll know it's CV, not the solver.

### Ideal
- Photo/video sweep → YOLOv8 hold detection → JSON world model
- User correction layer on detected holds (non-optional fallback)
- Personalized body model: user inputs height, wingspan, flexibility, strength proxy
- 3D IK with hip twist, drop-knee, flagging
- RL solver (PPO or SAC) trained on synthetic walls, generalizes to new routes
- AR overlay showing next move in real time (gym-mounted camera or phone)
- Route difficulty estimator as byproduct of solver
- B2B setter tool: design routes on a tablet before physically placing holds

---

## Build Phases

### Phase 1 — Data Foundation ✅ DONE
Wall editor, JSON schema, Docker, save/load.

### Phase 2 — Body Model + IK (next)
- `BodyModel` class: height, wingspan, arm/leg segment lengths
- 2D IK for arms and legs (geometric, law of cosines)
- Joint constraint checks (no hyperextension, angle limits)
- Center of mass validator (is this position stable?)
- Stick figure overlay rendered on the grid
- Manual drag test: move limbs to holds, system flags valid vs invalid

**Key decision:** start with 2D projection. 3D (hip twist, drop-knee) comes later.

### Phase 3 — Reachability + Solver
- Reachability checker: given body position, which holds can each limb reach?
- Hold graph: nodes = holds, edges = valid moves
- A* or BFS to find path from start to finish
- Text move sequence output
- Basic difficulty estimator (how many near-max-reach moves?)

**Missing prerequisite:** `cell_size_cm` must be added to the schema before this phase. Without real-world scale, reachability is meaningless.

### Phase 4 — Synthetic Wall Generator
- Procedural hold placer with solvability check
- Randomized rendering: lighting, chalk dust, wall texture
- 100+ labeled walls with known solutions → CV training data
- RL training environment

**Why this matters:** without synthetic data, CV training requires hand-labeling thousands of real gym photos. The generator sidesteps this.

### Phase 5 — Computer Vision
- Fine-tune YOLOv8 on Phase 4 synthetic renders + real gym photos
- Output: Hold JSON (same schema as manual editor)
- User correction UI: overlay detected holds on photo, tap to fix
- Hold orientation detection

### Phase 6 — Full Pipeline + App
- Photo → CV → JSON → user corrections → solver → move sequence overlaid on photo
- User onboarding: height + wingspan
- Internal beta test on a real gym wall
- Performance target: end-to-end in < 5 seconds

---

## IK Approach

Use a custom 2D geometric solver — not a library. A climber has exactly 4 contact points. Each limb is a 3-joint chain (short enough for closed-form solutions, no iteration needed).

```python
import numpy as np

def solve_arm_ik_2d(shoulder_pos, target_pos, upper_len, lower_len):
    dx = target_pos[0] - shoulder_pos[0]
    dy = target_pos[1] - shoulder_pos[1]
    dist = np.sqrt(dx**2 + dy**2)
    if dist > upper_len + lower_len:
        return None  # out of reach
    if dist < abs(upper_len - lower_len):
        return None  # too close
    cos_angle = (upper_len**2 + dist**2 - lower_len**2) / (2 * upper_len * dist)
    angle_to_target = np.arctan2(dy, dx)
    elbow_angle = np.arccos(np.clip(cos_angle, -1, 1))
    elbow_x = shoulder_pos[0] + upper_len * np.cos(angle_to_target + elbow_angle)
    elbow_y = shoulder_pos[1] + upper_len * np.sin(angle_to_target + elbow_angle)
    return np.array([elbow_x, elbow_y])
```

Use `ikpy` as a reference/sanity check. Only move to a full 3D library if 2D projection is fundamentally insufficient.

---

## RL Solver Design (Phase 4+)

- **State:** current hand/foot positions + remaining holds + body position
- **Action:** move a limb to a target hold
- **Reward:** +reaching the top; −invalid body states, excessive moves, near-falls
- **Architecture:** Graph Neural Network policy (holds as nodes, reachability as weighted edges) — naturally generalizes to any wall
- **Training:** PPO or SAC, start entirely on synthetic walls, then transfer to real
- **Why RL over hardcoded:** generalizes to any new wall automatically without reprogramming

---

## Key Architecture Decisions

1. **Manual input before CV.** Build the solver first so you can isolate bugs. CV is a separate failure mode.
2. **Store orientation as float, render snapped.** `orientation_deg` is a float in JSON. UI snaps to 5° increments. Never lose precision in the data layer.
3. **cell_size_cm must be added before Phase 3.** Reachability is physically meaningless without real-world scale.
4. **Schema is the contract.** Every phase depends on the same JSON format. Don't change it without team discussion.
5. **User correction is non-optional.** CV will be wrong sometimes. Always give the user a way to fix detections before the solver runs.
6. **Synthetic generator enables everything.** It is CV training data, RL environment, and a standalone product (setter tool, personalized home board generator) simultaneously.

---

## Market + Value Prop

**Who needs this most:** intermediate climbers (V3–V7 / 5.10–5.12) who have plateaued and want to understand why a route is hard for their specific body.

**B2B angle (stronger GTM):** Sell a setter tool to gyms. Given a target grade, suggest hold placements. Simulate how a route solves for different body types before any hold is physically placed. ~600 US climbing gyms, $500–1000/month each = real ARR.

**Consumer:** 25M climbers globally, ~5M regular gym climbers. 2% at $8/month = ~$96M theoretical ceiling. Consumer is harder to monetize but builds brand.

**Cultural note:** a large portion of climbers actively do not want to be told the beta — the puzzle-solving is the point. Make beta sharing optional and default off.

**Long-term:** personalized home board generation (Kilter/Moon Board competitor), outdoor rock face identification (harder — 5–10 years out), competition prep tools for elite climbers.

---

## Git Conventions

- `main` — always working, protected, no direct pushes
- `dev` — integration branch
- `feature/your-name/description` — personal branches
- One person owns merging to main
- Review each other's PRs before merging to dev

---

## Running Locally

```bash
git clone <repo>
cd beta-engine
docker compose up          # starts at http://localhost:8000
docker compose up --build  # after dependency changes
```

Walls are persisted to `./data/walls/` on the host (bind-mounted into the container).
