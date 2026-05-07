# Physics

A 2D rigid-body simulation of a climber on a wall, built on **pymunk**
(Chipmunk2D's Python wrapper). Replaces the static, kinematic-only
checks in `solver/` with an actual time-stepped sim — gravity, wall
angle, friction-aware holds, force-limited grip, articulated stick
figure.

This is the Phase-3 step the [solver README](../solver/README.md) and
[`planning-gabe.md`](../planning-gabe.md) call for.

> **Why pymunk and not Box2D?**
> Pymunk's pip wheels are reliably built across Linux/macOS/Windows
> (`pip install pymunk` just works); pybox2d has a long history of
> stale wheels and build failures. Pymunk also has a cleaner Pythonic
> API and is the engine used by most modern Python physics+RL
> tutorials, so it slots straight into Gymnasium.

---

## Quickstart

Container running (`docker compose up -d`):

```bash
# A still PNG of the climber settled on the start holds
docker compose exec beta-engine python -m physics --wall example-v2-boulder

# Animated GIF — body settles, then plays a 9-move beta
docker compose exec beta-engine python -m physics --wall example-v2-boulder \
  --gif --frames-per-move 12 \
  --moves 'RF:h_005,LF:h_007,RH:h_008,LH:h_006,RF:h_009,LF:h_011,RH:h_012,LH:h_010,RH:h_013'

# Different climber — height/wingspan/mass change which holds are reachable
docker compose exec beta-engine python -m physics --wall example-v2-boulder \
  --height-cm 190 --wingspan-cm 185 --mass-kg 80 --gif
```

Outputs land in `./data/runs/`:

```
data/runs/
├── example-v2-boulder-physics.png
└── example-v2-boulder-physics.gif
```

---

## CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--wall <id\|path>` | — | wall_id or path to a JSON file |
| `--height-cm` | 175 | climber height |
| `--wingspan-cm` | 175 | climber wingspan |
| `--mass-kg` | 70 | climber mass — affects gravity load on each limb |
| `--cell-size-cm` | (schema) | override the wall's cell size |
| `--frames` | 60 | number of rendered frames (no-moves mode) |
| `--frames-per-move` | 20 | how long each move takes on screen |
| `--gif` | off | render an animated GIF instead of a PNG |
| `--solve` | off | plan a beta with `solver.astar` and play it through physics |
| `--moves` | — | comma-separated `LIMB:hold_id` pairs; lets you script a beta directly |

---

## Module reference

| File | Responsibility |
|------|----------------|
| `world.py` | `ClimbWorld` — wraps a `pymunk.Space`, owns the wall + holds + climber, drives the active-posture controller |
| `body.py`  | `PhysicsBody` — rigid torso, anatomical anchor points, force-limited leashes to attached holds |
| `config.py` | All numeric tuning constants in one file |
| `render.py` | Headless matplotlib renderer (PNG + animated GIF) |
| `__main__.py` | CLI entry point |

---

## How the body model works

The body is **one rigid torso** plus four virtual "limbs". Each
attached limb is a force-limited [`SlideJoint`][slide] connecting one
of the torso's body-local anchor points (left shoulder, right shoulder,
left hip, right hip) to a static "hold body" at the hold's world
position. The joint is a leash:

- `min` distance = 0 — the climber can pull right up to the hold,
- `max` distance = limb length × 0.999 — the limb can't stretch past
  its bone length,
- `max_force` = climber's grip × hold positivity — exceed it and the
  joint slides, which is the simulator's way of saying *the climber
  slipped off the hold*.

The articulated stick figure (upper arm, elbow, forearm, etc.) is
**computed kinematically** via the closed-form 2-link IK in
`solver.body`. It's purely visual — limb segments don't have their own
mass or joints in pymunk. This is a deliberate v1 simplification: a
multi-segment ragdoll *can* be built but tunes badly under gravity (it
either folds up or oscillates), and for the question we actually want
to answer — *given which holds are attached, can the body balance, and
how loaded is each limb?* — the leash model is enough.

[slide]: https://www.pymunk.org/en/latest/pymunk.constraints.html#pymunk.constraints.SlideJoint

### Two assists keep the body upright

1. **`DampedRotarySpring(static_body, torso, rest_angle=0)`** — passive
   "core tension" that resists torso rotation. Without it the body
   spins on its leashes until it's upside-down.
2. **Active posture force** in `ClimbWorld._apply_posture_force` — at
   each substep we apply
   `F = m·g_compensate + Kp·(target − actual) − Kd·velocity` to the
   torso, where the target is the kinematic ideal COM (foot midpoint +
   hip-height, biased toward the hand midpoint). This emulates the
   active leg/core work a real climber does to *stand up* on their
   feet rather than hang from them.

Without the active posture term, the body settles passively into
"hanging from the lowest leash" — a physically valid pose but visually
wrong (think of a person on a vertical wall: feet *push*, they don't
just dangle). Climbing is fundamentally an active task; this term makes
that explicit. Tune `POSTURE_GAIN_N_PER_M` in `config.py` if you want
to see the body sag and the leashes engage.

### What the simulator answers

For each pose / time-step the simulator can tell you:

- The torso's world position and angle.
- The kinematic position of each elbow / knee (via IK).
- The force in each attached leash (newtons), and the *fraction* of
  that limb's max-force budget being used.
- Whether the body is stable (no limb past its slip threshold).

That force/fraction signal is what an RL agent should learn from —
moves that load slopers near their cap should be punished; moves that
keep load on jugs should be rewarded.

---

## Tweakable knobs (all in `config.py`)

| Constant | Default | What it controls |
|---|---|---|
| `GRAVITY_M_S2` | 9.81 | gravity magnitude |
| `PHYS_DT` | 1/240 s | physics step. Smaller = stabler joints |
| `SUBSTEPS_PER_FRAME` | 8 | physics steps per render frame |
| `DEFAULT_DAMPING` | 0.4 | global linear/angular damping |
| `DEFAULT_MASS_KG` | 70 | climber mass |
| `DEFAULT_GRIP_FORCE_N` | 600 | max hand pull force on a perfect hold |
| `DEFAULT_FOOT_PUSH_FORCE_N` | 1500 | max foot push force |
| `TORSO_UPRIGHT_STIFFNESS` | 200 | passive core-tension spring stiffness |
| `POSTURE_GAIN_N_PER_M` | 4000 | active stand-up controller gain |
| `POSTURE_DAMPING` | 800 | active stand-up controller damping |
| `ATTACH_MAX_FORCE_SCALE` | 1.0 | global multiplier on hold max-force budgets |

### Schema additions used by physics

The wall JSON schema gained three optional fields the physics layer
reads (additive — older walls still load):

```json
{
  "wall_angle_deg": 0,        // -45..45, +ve = overhang, -ve = slab
  "surface_friction": 0.7,    // for future smearing physics
  "grid": { "cell_size_cm": 20 },
  "holds": [
    {
      "...": "...standard fields...",
      "friction": 0.85,       // overrides per-type default
      "positivity": 0.65,     // overrides per-type default
      "max_force_n": 500      // hard cap on this hold's load
    }
  ]
}
```

If a field is absent the physics falls back to a per-hold-type default
(see `solver.wall.POSITIVITY_BY_TYPE` and `FRICTION_BY_TYPE`).

---

## Known limitations (intentional, v1)

| Limit | Why it's OK for now | Fix when |
|---|---|---|
| **Limbs are leashes, not full segmented bodies** | A multi-segment ragdoll tunes badly under gravity — folds or oscillates without active per-joint control. Leashes give meaningful forces with stable behaviour. | Phase 4 — once we have an RL policy outputting torques, we can drive a full articulated body. |
| **Posture force does most of the work in stable poses** | Means leash forces are ~0 when the climber is at the kinematic ideal. Forces only meaningfully engage near max reach. | Trade off `POSTURE_GAIN` against leash realism; or compute static contact forces analytically (added together in body-frame). |
| **No body-on-wall collision** | The "wall" is just a frame for hold positions; the body floats. Ok on vertical walls. Breaks down on overhangs where the body would press into the wall. | Add a static `pymunk.Segment` plane at x=… and collision filter the body against it. |
| **No friction cone / direction-dependent grip** | Hold orientation is in the JSON schema but the physics treats every hold as omnidirectional. | Replace `max_force` with a direction-dependent constraint (one Pin per hold direction, or a custom constraint). |
| **No core-tension model for overhangs** | Steep walls need extra hip/torso work to keep feet on. We don't model that yet. | Add a soft constraint relating hand-foot distance to required core force; penalise in reward. |
| **No dynamic moves (dynos, deadpoints)** | All moves are quasi-static snap+settle; momentum doesn't carry between frames. | Replace `move_limb(mode='snap')` with a target-velocity controller that releases the limb, applies impulse, catches at target. |

For a full taxonomy of what a "real" 2D climbing physics engine would
include, see the brainstorm at the bottom of `CLAUDE.md` / the user's
notes — those are the road-map items above and beyond v1.
