# ARCH_REVIEW — Three-branch climbing-RL merge audit

Audit date: 2026-05-12. Scope: the merged tree at `beta-engine/` after three
parallel branches were combined. The audit was done file-by-file (no code was
modified). All file paths below are relative to the repo root.

---

## Phase 1 — Map of the merged codebase

### How many implementations are there, and how do they declare themselves?

Three distinct climbing simulators + Gym environments coexist. They are
identifiable by directory, by which physics engine they import, and by which
body model they use:

| ID | Directory tree | Engine | Body | Gym env class | Notes |
|----|----------------|--------|------|---------------|-------|
| **A** (sim3d) | `sim3d/` + the upstream `solver/` + `grid_editor/` + `static/` | MuJoCo 3.x | Custom 27-DOF climber generated from `Wall` + `ClimberProfile` (`sim3d/builder.py`) | `sim3d.env.Climbing3DEnv` (`sim3d/env.py:84`) plus `sim3d.moonboard_env.MoonboardClimbing3DEnv` (`sim3d/moonboard_env.py:46`) | Owned by you. Treats MoonBoard as one wall flavour among many; pipes through `solver.wall.Wall`. |
| **B** (moonboard-rl) | `moonboard-rl/` (self-contained; runs from its own `cd moonboard-rl`) | MuJoCo 3.x | Stock Gymnasium humanoid asset, with sites injected and friction patched (`moonboard-rl/src/xml_gen/scene.py`) | `src.envs.moonboard_env.MoonBoardEnv` (`moonboard-rl/src/envs/moonboard_env.py:102`) | Collaborator-built. MoonBoard-only. Has its own `Hold`/`Route` types (`moonboard-rl/src/parsers/canonical.py`) and three JSON-format parsers. |
| **C** (rl + physics) | `physics/` + `rl/` + the same `solver/` | pymunk (2D rigid body) | Rigid torso with leashed limbs; visual stick figure via 2-link IK (`physics/body.py`) | `rl.env.ClimbingEnv` (`rl/env.py:66`) | Older 2D track. Plays the same JSON walls as A. |

The three trees do not import each other except that A and C share the
`solver/` package (wall loader, hold types, BodyModel) and use the same
on-disk wall JSON schema (`schemas/wall.schema.json`). B is genuinely
independent — its own `Hold` / `Route` types, its own coordinate convention,
its own scene assembler — and pulls a different MuJoCo asset (the stock
DeepMind humanoid in `moonboard-rl/assets/humanoid.xml`).

### Scope of each implementation

**A (sim3d)** — Trying to be the general-purpose 3D climbing simulator. Wall +
holds parameterised from the editor JSON. MoonBoard adapter
(`sim3d/moonboard.py`) projects a problem into a `Wall`, optionally with
all 198 T-nuts (full-board mode) or vertical-projection geometry.
`Climb3DWorld` exposes a clean attach/release/move/step API.
`Climbing3DEnv` has two action modes — `discrete-move` (`Discrete(4·n_holds)`,
high-level limb→hold) and `continuous-joint` (`Box(-1,1,(nu,))`, joint
targets). Observation is a flat 117-d vector for a 13-hold wall, scaling
linearly with hold count.
`MoonboardClimbing3DEnv` wraps the above so each `reset()` rebuilds the env
on a new sampled problem but keeps `action_space` / `observation_space`
constant by always using the full 11×18 board.
PPO trainer (`sim3d/train.py`) wraps SB3 with `DummyVecEnv` /
`SubprocVecEnv`, an episode-stats CSV callback, and a JSON config dump.

**B (moonboard-rl)** — MoonBoard-specific RL. Hardcodes the 40° overhang
geometry (`OVERHANG_DEG=40.0` in `moonboard-rl/src/xml_gen/wall.py:28`),
the 11×18 grid with 0.20 m spacing, and a vertical kickboard panel below
the main wall with four permanent kickboard footholds at ±0.244 m and
z ∈ {0.100, 0.270} m (`moonboard-rl/src/xml_gen/wall.py:84-89`). Action
space is `Box(21,) = 17 motor torques + 4 grip-intent signals`
(`moonboard-rl/src/envs/moonboard_env.py:252-256`). Observation is a flat
139-d vector explicitly partitioned into proprioception (65),
exteroception (8 nearest holds × 7), goal vectors (4×3), and a foot-to-kickboard
proximity stream (2×3). PPO smoke trainer
(`moonboard-rl/scripts/train_ppo_smoke.py`) is single-env, single-route
(picks the most-repeated V4/V5 problem). Reset is a multi-phase
sequence — torso warp, hand grips, settle, DLS-IK feet to the kickboard,
foot grips with `snap_to_center=True` to drag the feet onto the kickboard,
further settle — and asserts ≥3 grips active or raises.

**C (rl + physics)** — 2D RL on the same JSON walls as A. Single rigid
torso + four force-limited `SlideJoint`/`PinJoint` leashes (`physics/body.py`).
Active posture force on the torso (`physics/world.py:270-301`) and a
passive rotary spring keep the body upright. Action space is
`Discrete(4·n_holds)`, observation `4 + 4·n_holds + 4` (COM x/y/vx/vy +
limb occupancy one-hot blocks + per-limb force fraction). Reward is the
classic per-step penalty + per-cm progress + completion bonus + slip + illegal
penalty (`rl/env.py:46-52`). The "physics" runs `SUBSTEPS_PER_FRAME=8`
sub-steps per action with the active posture controller in the inner
loop. Move mode defaults to `snap`. There is no curriculum or generalisation
plumbing.

### Where they share code

* The wall JSON schema (`schemas/wall.schema.json`) and the loader
  (`solver/wall.py`) are the contract between A, C, and the grid editor.
  `Hold` / `Wall` dataclasses come from `solver/wall.py` and carry the
  optional physics extensions (`wall_angle_deg`, `surface_friction`,
  `friction_override`, `positivity_override`, `max_force_n`) added in
  Phase-3 (`solver/wall.py:67-86`).
* `solver/body.BodyModel` is used by both C (`physics/body.py:49`) and the
  pre-3D solver. A duplicates the body anthropometry in `sim3d/body.py`
  rather than re-using `solver.body`.
* `solver/__init__.py` re-exports the 2D Q-learning toy as another solver,
  not connected to A or B.
* B shares nothing with A or C: it loads its own Route via
  `moonboard-rl/src/parsers/format1.py` directly from `moonboard1.json`.

### Hard conflicts in the merged tree

* **Duplicate `ClimberProfile`.** `physics.body.ClimberProfile`
  (`physics/body.py:57-71`) carries `body: BodyModel`, `mass_kg`,
  `grip_force_n`, `foot_push_force_n`, `friction_*`. `sim3d.body.ClimberProfile`
  (`sim3d/body.py:137-165`) carries `height_cm`, `wingspan_cm`, `mass_kg`,
  `grip_force_n`, `foot_push_force_n`. Same name, different shape; either
  can be imported depending on which package you load first. Importing the
  wrong one silently passes Pylance/MyPy because both are dataclasses.
* **Duplicate `EnvConfig`.** `rl.env.EnvConfig` (`rl/env.py:57`) and
  `sim3d.env.EnvConfig` (`sim3d/env.py:62-81`). Disjoint fields, no
  inheritance.
* **Three competing `Limb` enumerations.** A and C agree on
  `('LH','RH','LF','RF')` strings. B uses integer slot indices 0–3 keyed
  by `LIMB_BODY_NAMES = ['left_lower_arm','right_lower_arm','left_foot','right_foot']`
  (`moonboard-rl/src/xml_gen/scene.py:41-46`). The hand→arm-body mapping
  in B is from elbow body, which is why B has to inject sites at the
  hand/foot tip pos via `_inject_limb_sites` to get sane proximity checks.
* **Two MoonBoard adapters.** `sim3d/moonboard.py` produces a `solver.wall.Wall`
  with letter+row → cell-cm coordinates (and optional vertical projection).
  `moonboard-rl/src/parsers/format1.py` returns a `Route` of `Hold(col,row,role)`.
  Same source data, two incompatible runtime representations.
* **Two wall coordinate conventions.** A: wall plate rotated by `+θ`
  around +X, bottom anchored at z=0, outward normal `(0, cosθ, −sinθ)`
  (`sim3d/builder.py:67-77`). B: wall box rotated `−40°` about X,
  centred at `(0, Y_BASE + 8.5·SPACING·sinα, Z_BASE + 8.5·SPACING·cosα)`,
  outward normal `(0, cos40, −sin40)` (`moonboard-rl/src/xml_gen/wall.py:142-153`).
  The numerical normals agree at 40°, but A's framework is angle-parameterised
  and B's is angle-hardcoded.
* **Two attach mechanisms.** A uses a per-limb mocap body + weld equality
  with `eq_data[10]=0` (`torquescale=0` ⇒ position-only, body can still
  pivot) — see `sim3d/world.py:444-455`. B uses a `connect` equality
  retargeted at runtime (`moonboard-rl/src/xml_gen/scene.py:236-245` and
  `grip_manager.py:265-290`). The README in B explicitly argues against
  weld ("Weld would lock all 6 DOF, producing unrealistically stiff arms",
  `grip_manager.py:12-14`). Both work; they encode opposite design choices.
* **Two slip models, both called slip.** A: per-limb capacity = climber
  strength × hold positivity × `SLIP_FORCE_SLACK`, applied to
  `data.cfrc_int[body, 3:6]` magnitude. B: hardcoded 6000 N for hands,
  9000 N for feet, applied to `efc_force` projected through
  `efc_id == eq_id`. Neither is wrong, but the units and semantics
  differ.

---

## Phase 2 — Dimension-by-dimension comparison

### 1. Physics and simulation setup

**A.** MuJoCo `implicitfast` integrator, `PHYS_DT=2 ms`, render at 60 Hz,
`SUBSTEPS_PER_FRAME=8` (`sim3d/config.py:12-14`). `cone="elliptic"`,
`iterations=50`. Position actuators with `kp=120`, `kv=8`
(`sim3d/config.py:155-156`) plus per-joint torque caps grouped by
{shoulder/elbow/wrist/spine/hip/knee/ankle} 30–220 Nm. Joints carry
passive stiffness/damping that has been re-tuned twice (the comments in
`sim3d/config.py:134-146` document the tuning history).

**B.** MuJoCo with whatever defaults the stock humanoid carries
(`integrator` from the inherited `<option>`; `timestep` defaults to 2 ms;
`sim_substeps=7` per policy step ⇒ ~14 ms policy period —
*not* the 30 ms documented in B's README at
`moonboard-rl/README.md:218-220`. The env emits a runtime warning if it
falls outside [20, 40] ms — see `moonboard_env.py:144-149` — and 7 × 2 ms
falls below the lower bound). Actuators are `motor` (raw torque) with
gear ratios 25/100/200/300 and `ctrlrange="-.4 .4"`. The result is that B's
control authority is gear-ratio dependent (hip_y gear=300, shoulder gear=25)
and bounded torque is `0.4 × gear` Nm.

**C.** pymunk 2D. `PHYS_DT = 1/240 s`, 8 substeps per action
(`physics/config.py:11-12`). `space.iterations=30`, `damping=0.4`.
Wall angle implemented by rotating the gravity vector
(`physics/world.py:97-100`).

**Assessment.** A is the most carefully tuned setup — explicit substep
loop, explicit reach controller that *temporarily zeros actuator KP* on
the limb being moved (`sim3d/world.py:567-577`) so the impedance and
the position servo don't fight. B inherits stock-humanoid defaults
and then bolts a foot-IK + multi-phase reset on top; the policy-period
warning at `moonboard_env.py:144-149` is itself evidence that the
configuration isn't fully nailed down. C is the simplest and the most
quasi-static — it explicitly chose not to model articulated limbs in
the dynamics (`physics/body.py:23-33`), which is the right call for 2D
but means we cannot RL into a 3D motion policy from it.

### 2. Wall and hold geometry

**A.** Wall is a plate body with thickness `WALL_THICKNESS_M=0.05`, with
hold cylinders as child geoms; `contype=2/conaffinity=2` makes them
non-colliding so grabbing is mediated by the weld only
(`sim3d/builder.py:420-430`). Plate placement keeps the bottom at z=0 for
vertical walls and lifts the plate when extreme overhangs would push the
lowest holds underground (`sim3d/builder.py:348-359`). MoonBoard adapter
optionally rescales `cell_size` by `1/cosθ` to project rows onto a
vertical world-Z pitch (`sim3d/moonboard.py:163-166`).

**B.** Wall is one large `box` geom rotated `−40°` about X at a fixed
centre. Holds are 0.04 m spheres with `contype=0/conaffinity=0` (visual
only) plus an inner grip-volume sphere with `contype=2/conaffinity=2`
for the kickboard holds (`moonboard-rl/src/xml_gen/wall.py:212-217`).
Main-wall holds only have the visual geom — no separate grip volume.
Hardcoded geometry: `_WALL_HALF_X=1.2`, `_WALL_HALF_Y=0.10`,
`_WALL_HALF_Z=2.0`. Kickboard half-widths sized to the MoonBoard spec.

**C.** Holds are static pymunk bodies with sensor circles
(`physics/world.py:103-118`) that collide with nothing — purely position
sources for joint anchors. Wall surface itself is not represented.

**Assessment.** A's geometry is genuinely parameterised over wall + climber.
B's is a hardcoded MoonBoard with extras (the kickboard) that the rest
of the codebase doesn't know about. C's is a hand-wave (the climber
floats in 2D). For physical correctness on MoonBoard problems
specifically, B is closest to the real board because it models the
kickboard's effect on foot placement; A handles it abstractly via
foothold-eligibility masking; C ignores it. *However* B has a known
limitation that bites it later (Issue 1 in `moonboard-rl/KNOWN_ISSUES.md`):
the feet stay on the kickboard for the whole episode because the foot
target sequencer never activates outside reset.

### 3. Contact and grip model

**A.** Weld equality between a kinematic mocap body and a child "tip body"
colocated with the hand/foot site. `torquescale=0` makes it
position-only (`sim3d/world.py:452-453`). Per-limb max-force =
`profile.grip/foot_force × hold.positivity` clipped by the hold's
`max_force_n`. Slip detection reads `data.cfrc_int[body,3:6]` magnitude
and releases the weld if it crosses `cap × SLIP_FORCE_SLACK` (default
1.25). The code itself flags that `cfrc_int` is an approximation
because it accumulates *all* active-constraint forces on that body
(`sim3d/world.py:702-707`) — fine when one limb is welded, an upper bound
when more.

**B.** `connect` (3-DOF ball joint) with anchor1/anchor2 retargeted at
runtime. Critical insight in the docstring: `eq_obj2id` is mutated to
point body2 at the actual hold, and `eq_data[0:3]/[3:6]` are the
anchor positions in body-local frames; without this step the constraint
enforces the model-load relative pose and the arm snaps violently
(`moonboard-rl/src/grip/grip_manager.py:264-300`). Slip is `efc_force`
norm vs `SLOT_MAX_FORCE[slot]` (hands 6000 N, feet 9000 N — see
`grip_manager.py:79-89`). The fallback `qfrc_constraint[:6]` proxy
(`grip_manager.py:460-468`) is brittle: it divides total root-DOF
constraint force evenly among active slots.

**C.** Hand attachment = `SlideJoint` (leash, 0..max-length) so the
hand pulls but the body can drift in. Foot attachment = `PinJoint`
(rigid distance fixed at attach time) so the foot pushes
(`physics/body.py:186-224`). Force = `joint.impulse / dt`
(`physics/body.py:237-243`). Capacity = `grip_force × hold.positivity`.

**Assessment.** B's grip model is *more physically grounded* in one
dimension (using ball joints to let the body pivot on the hold) and
*less* in another (hardcoded per-slot thresholds independent of the hold's
material/positivity). A's `torquescale=0` weld is functionally equivalent
to a connect-with-position-only behaviour but is more numerically stable.
The "per-hold capacity from positivity" pattern is the right physical
abstraction, and only A implements it. C's `SlideJoint`-vs-`PinJoint`
distinction is a clever 2D trick but doesn't generalise to 3D.

### 4. Observation design

**A.** 117-d flat vector for 13 holds (`sim3d/env.py:176-186`). Pelvis
position (3) + pelvis quaternion (4) + COM (3) + joint qpos[7:] (nu) +
joint qvel[6:] (nu) + 4 limb tip world positions (12) + 4×n_holds
occupancy one-hot + 1 (highest hand → finish z distance). Observations
scale with hold count — for full-board MoonBoard this becomes
4·198 = 792-d for the one-hot alone. The default is no normalisation.

**B.** 139-d flat vector with explicit streams (`moonboard_env.py:582-718`):
proprioception (65; includes 6-D rotation representation instead of quat,
which is well-known to be friendlier to MLPs), 8 nearest holds × (rel_pos
+ role one-hot + grip flag) = 56, goal vectors (4×3 = 12), foot→nearest
kickboard rel-vec in pelvis frame (2×3 = 6). The "8 nearest" trick keeps
exteroception dimension constant and removes hold-count dependence.
Foot→kickboard stream is a hand-crafted shaping channel. Includes
NaN/Inf guard with per-stream localisation (`moonboard_env.py:699-716`).

**C.** 4 + 4·n_holds + 4 (`rl/env.py:235-238`). Minimal but agreed-on
with the action space.

**Assessment.** B's observation is the most thoughtfully engineered:
- 6-D rotation (rot6d) is the right encoding for an MLP, much better
  than quaternion sign ambiguity (which A's pelvis quat carries).
- "K nearest holds" removes the wall-size dependence that A still has.
- Per-stream NaN guard is exactly the kind of cheap defensive code that
  saves a debugging weekend.

A's design has two real defects:
- Hold-count scaling means a policy trained on the 13-hold example
  cannot be transferred to a MoonBoard wall, and the MoonBoard env
  works around this by always using full-board (198 holds), which then
  produces a 4·198 = 792-d one-hot block — a huge sparse vector for an
  MLP.
- The pelvis quaternion is double-cover ambiguous; gradients near `q`
  and `−q` jitter.

Neither is fatal, but B's observation is the better blueprint.

### 5. Action space design

**A.** Two modes — discrete-move and continuous-joint. Discrete is
high-level beta selection: `Discrete(4·n_holds)`, decoded as
`limb_idx, hold_idx = divmod(action, n_holds)` (`sim3d/env.py:296-309`).
Continuous is a low-level joint-target controller scaled to each joint's
range (`sim3d/env.py:165-174`). On a `discrete-move` step, the chosen
limb does a continuous reach over `move_frames` (default 60 = 1 s @ 60 Hz)
while MuJoCo integrates the body.

**B.** `Box(21,) = 17 motor torques + 4 grip intent signals`. Torque
clipped to actuator `ctrlrange`. Grip intent > 0 ⇒ `try_grip(slot,
scripted_target)`; foot has a fallback that searches all available holds
(`moonboard_env.py:486-505`). Target sequencing is *scripted*, not
policy-decided: hand targets advance with `_advance_hand_target` after
a rising-edge grip-target match (`moonboard_env.py:791-815`), and foot
targets lag two moves behind hands (`moonboard_env.py:816-852`).

**C.** Same as A's discrete-move (`Discrete(4·n_holds)`), with `mode="snap"`.

**Assessment.** A and B reflect opposite philosophies. A treats the
*policy* as the move-sequence picker — the only thing it has to learn is
"which limb to which hold next." Physics, body control, and reach are
solved by the env. This makes A's discrete-move env *much* easier to
train, especially under 24 hours: the policy explores in a tiny discrete
action space and the env's continuous reach controller delivers the
reward signal. B asks the policy to learn *both* low-level motor control
and *when to* press the grip-engage button, but it can't pick *where*
because the target is scripted. That is the worst of both worlds: the
policy can only learn timing of grip activation, not what to do with
the joints (because the joints are buried inside a 17-d torque action
that the reward signal can barely localise to).

For a 24-hour run, A's discrete-move action space is far more sample-efficient.
B's continuous-torque + scripted-target action space is closer to a
research-grade humanoid-control problem and will not converge on a
single GPU in a day.

### 6. Reward structure and shaping

**A** (`sim3d/env.py:62-82`, `319-348`):
- `+upward_reward × Δcom_z` (default `upward_reward=8.0`).
- `−per_step_penalty=0.02`.
- `−slip_penalty=5.0 × n_slips`.
- `−body_intersection_penalty=2.0 × n_intersections` (uses MuJoCo
  contacts to penalise self-intersection between limb geoms and torso/pelvis,
  see `sim3d/world.py:223-261`).
- `−invalid_action_penalty=0.25` if the action picks a masked hold.
- `+on_finish_bonus=100.0` after `finish_hold_frames=6` consecutive
  steps with a hand on a finish hold.
- `−fall_penalty=50.0` if `pelvis_z < fall_z=0.20`.

**B** (`moonboard_env.py:724-773`, `551-563`):
- `+5 × (pelvis_z − max_pelvis_z)` (high-water mark — *only new upward
  progress earns reward*).
- `+5.0` per slot for a rising-edge grip on a new target hold,
  deduplicated by `_holds_matched_this_episode` (no farming).
- `−0.01 × Σctrl²` energy penalty.
- `−10.0` fall penalty when `pelvis_z < 0.30` OR `pelvis_y > 1.0`.
- `+50.0` finish bonus after 10 consecutive steps on the end hold with
  both hands.

**C** (`rl/env.py:46-52`):
- `−0.5` per step, `+0.05 × (last − new) cm` progress.
- `−5` illegal move penalty.
- `−10` slip penalty per slipped limb per step.
- `+100` on-finish.

**Reward hacking surface (most → least vulnerable):**

1. **A — most vulnerable.** Progress is *Δcom_z each step*. The agent can
   harvest this by oscillating up and down (`+ε` on up, `−ε` on down)
   if it can synthesise transient COM lift via the reach controller without
   the body being in a stable pose. The `body_intersection_penalty=2.0`
   is shaped not gated; in continuous-joint mode the agent can fold limbs
   through the torso to clip the kinematics into a high-pelvis-z pose
   without paying enough to offset progress. There is *no* check that
   feet are still on holds before crediting upward progress, so a "press
   off the wall on hands and dyno upward" exploit harvests
   `upward_reward` between fall detection and termination.
2. **B — moderately vulnerable.** HWM progress is the right shaping
   pattern (cannot oscillate), and `_holds_matched_this_episode` denies
   the "yo-yo to the same hold" exploit. The exposed surface is the
   *scripted* foot-target sequence: the foot fallback in
   `moonboard_env.py:496-505` grips *the first hold within proximity*,
   which on a high body pose can be a hand-only hold, then earns the
   match bonus on a hold that no real climber would foot. Energy penalty
   on `ctrl²` discourages explosive moves the agent would otherwise need
   for harder problems.
3. **C — least vulnerable** because the action space is "snap a limb to
   a hold," which constrains the search to legal transitions. The
   `_distance_to_finish` shaping is by COM, so the agent can park its
   feet far down the wall and reach the COM toward the finish — but
   reachability is enforced by `_is_legal`.

**Sample efficiency for a 24-hour CPU PPO run:**

- A in **discrete-move** mode trains fastest. Action space is small
  (e.g. 4 × 13 = 52 actions on the example wall, 4 × 198 = 792 on full
  MoonBoard), reward has dense `upward_reward × Δcom_z` shaping every
  step, and the per-step physics cost is only `move_frames` (default
  60) substeps. SB3 PPO with `n_envs=8` will see ~tens of thousands of
  episodes in 24 h. The reward hacks above probably manifest but a
  partially-successful policy still emerges.
- C is also fast (no MuJoCo, just pymunk) but is 2D so the learning
  doesn't transfer to anything we care about beyond.
- B is by far the slowest. 17-d continuous torque + scripted targets
  means the gradient signal on the joint actions has to come from the
  HWM height reward and the energy penalty — neither of which localises
  well to individual joints in a 17-d torque space. The "must achieve
  ≥3 grips" assertion at reset can also throw a `RuntimeError`
  mid-training run if some sampled state lands outside the warmup's
  tolerance.

**Physics correctness ranking:** B > A > C on contact/grip realism
(connect joint > weld-with-torquescale-0 > leash); A > B > C on
hold-capacity modelling (positivity × strength > hardcoded per-slot >
positivity in C but no real contact); A ≈ B on body-vs-wall collision;
A ≫ B on wall geometry generality.

### 7. Termination and reset logic

**A.** `seed_pose()` snaps limbs onto target holds, then runs a 0.5 s
gravity-ramp + 2 s settle with actuators zeroed *except spine_lean* which
gets a temporary `kp=1000` against the gravitational torque on the upper
body (`sim3d/world.py:381-410`). Captures the settled joint angles as
new actuator targets, zeroes velocities. There's also a wall-clearance
clamp on `pelvis_y` to keep the body from being seeded penetrating the
wall (`sim3d/world.py:344-352`). Optional `start_mode="ground-reach"`
puts the climber on the floor and reach-controllers the hands onto the
start holds.

**B.** Multi-phase: (1) reset qpos, set torso pos and rough hip/knee
flexion, (2) engage hands with relaxed thresholds (`_RESET_PROXIMITY=0.20`,
`_RESET_ALIGNMENT=−1.0`), (3) 50 warmup steps, (4) DLS-IK feet to
kickboard upper holds — 30 iters, 0.5 step scale, λ=0.01 (`moonboard_env.py:858-930`),
(5) engage feet with `snap_to_center=True` and `PROXIMITY_THRESHOLD=1.0` m
so the constraint spring drags them in (`moonboard_env.py:399-411`),
(6) 50 warmup + 20 settle steps, (7) assert ≥3 grips active or raise.

**C.** Builds the world fresh, seeds pose, settles 8 frames.

**Assessment.** B's reset is intricate but is doing real work — it
ensures the policy always starts from a self-consistent 4-point stance,
which is hugely valuable for stability. The dependence on relaxed
thresholds + `snap_to_center=True` + a 1.0 m proximity is a hack
(per the user's memory note about ±0.244 m kickboard vs the spec ±0.61 m)
and it papers over the fact that the stock humanoid's hip range is
insufficient. A's settle-with-gravity-ramp is cleaner because the body
model is custom-tuned to the joint limits; B is doing reset gymnastics
*because* the body model is stock.

**Reward-hack-relevant note**: A's reset also `slip_events.clear()`s
after seeding (`sim3d/world.py:419`). Good — otherwise the initial
constraint impulse spike would count against the agent.

### 8. Exploration strategy

None of the three implementations carries an explicit exploration
schedule beyond PPO's entropy bonus.

- A: PPO defaults from SB3, `ent_coef` is the SB3 default (0.0). No
  RND/ICM/curiosity. Discrete action space helps.
- B: `PPO_CONFIG.ent_coef=0.01` (`train_ppo_smoke.py:54`), the only
  explicit exploration knob.
- C: random policy only ever runs.

### 9. Curriculum

- **A** has the only meaningful curriculum scaffolding:
  `MoonboardClimbing3DEnv` samples a new problem each `reset()` from a
  deterministic train/val/test split (`sim3d/moonboard.py:354-387`;
  `sim3d/moonboard_env.py:97-110`). `EnvConfig.official_route_only`
  masks off-route holds so the action space stays at a stable 4·198 =
  792 even though only on-route holds are legal targets.
- **B** trains on one route (the max-repeat V4/V5) only. No corpus
  sweep.
- **C** trains on one wall only.

### 10. Training infrastructure and logging

- **A** has the most production-ish setup: `sim3d/train.py` writes
  `config.json`, `episode_stats.csv`, `tb/` event files, `model.zip`,
  and (for MoonBoard corpus) `moonboard_splits.json` into
  `data/runs/sim3d/<run_id>/`. Supports `--n-envs N` and `--device
  auto|cuda|cpu`. CSV columns include `final_com_z`, `n_slips`, and
  `body_intersections` per episode — useful diagnostics.
- **B** has TensorBoard + per-reward-component scalar logging and an
  evaluation-rollout mp4 callback every 100k steps
  (`train_ppo_smoke.py:82-191`). Single-env only (no SB3 vec env).
- **C** has only random-policy smoke testing.

### 11. Evaluation and video tooling

- **A**: trained policy replay in either the native MuJoCo viewer
  (`python -m sim3d --play model.zip`) or the three.js web viewer at
  `/sim3d/`. No evaluation script per se; you replay manually.
- **B**: scheduled mp4 evaluation videos during training (writes 30 fps
  rgb_array via `mujoco.Renderer`). Also `watch_random_agent.py` for
  live replay. **This is the best evaluation tooling of the three.**
- **C**: matplotlib PNG/GIF only, via `physics/render.py`.

### 12. Code quality, modularity, and extensibility

- **A** is the most modular: `body.py` (data), `builder.py` (XML),
  `world.py` (state + step), `env.py` (Gym), `moonboard.py` (adapter),
  `train.py` (PPO). The MJCF builder regenerates the model on each
  `Climb3DWorld()` construction, which is fine for resets between
  episodes but does mean wall changes cost a full compile.
- **B** mixes scene assembly, kickboard hacks, and reset logic across
  `xml_gen/`, `grip/`, and `envs/`. The `_RESET_PROXIMITY_FEET=1.0`
  pattern (override module-level constants with try/finally) is brittle
  but contained. Several constants are mutable at module scope by design
  (`grip_manager.PROXIMITY_THRESHOLD`) — that has cost us the
  `try/finally` pattern in multiple call sites and makes parallel
  envs unsafe.
- **C** is the cleanest and smallest, but its scope is narrower (2D
  only, one wall, one ascent attempt).

**For "two-week RSI + AMP + curriculum" extensibility:**
- A's `EnvConfig` already has `start_mode="ground-reach"` and the corpus
  splitting is in place. RSI would slot in as another `start_mode`
  variant; AMP needs a reference-motion stream that A's `pose_snapshot()`
  partially provides.
- B has a clearer reward-component decomposition that AMP wants
  (separate critic for the style discriminator), but the env state is
  scattered across the module-mutable thresholds. Plugging AMP into B
  requires refactoring grip thresholds to be instance attributes first.
- C is the wrong layer entirely.

---

## Phase 3 — Synthesis and recommendation

### KEEP from A (sim3d)
- **The whole package as the chassis.** `Climb3DWorld`, `Climbing3DEnv`,
  `MoonboardClimbing3DEnv`, the MJCF builder, and `sim3d/train.py`. This
  is the only implementation with a parameterised body, a parameterised
  wall, a clean attach/release API, a corpus sampler, multi-env support,
  and run-directory logging.
- **The discrete-move action space** (`sim3d/env.py:296-309`) as the
  Phase-1 training target. High sample efficiency, learns betas, easy
  to debug.
- **Per-hold capacity from `positivity × climber strength`**
  (`sim3d/world.py:526-534`). It's the right physical abstraction.
- **`SlipEvent` + body-intersection contact accounting**
  (`sim3d/world.py:223-261`, `724-746`). Both are well-designed
  diagnostic signals; keep them and surface them to TB.
- **The Cartesian-impedance reach controller**
  (`sim3d/world.py:579-628`). Compelling visually, RL-friendly, and the
  `_relax_reaching_actuators` zero-KP trick during reach
  (`sim3d/world.py:567-577`) avoids the controller-vs-actuator fight.
- **MoonBoard corpus split + manifest** (`sim3d/moonboard.py:354-407`).
  RSI and curriculum work both want a deterministic split.
- **`config.json` per run + `episode_stats.csv` callback**
  (`sim3d/train.py:80-145`). Continue this.

### KEEP from B (moonboard-rl)
- **The observation design** — and only the design:
  - 6-D pelvis rotation (`rot6d`) instead of quaternion.
  - "K nearest holds" with role one-hot + grip flag.
  - Goal-vector stream `(target_world − site_world)` per limb.
  - Per-stream NaN/Inf guard.
- **The evaluation mp4 callback** (`train_ppo_smoke.py:115-191`). Save
  rgb_array rollouts every N steps from a side-by-side env. Drop it
  into `sim3d/train.py` as another callback.
- **The HWM (high-water-mark) progress reward**
  (`moonboard_env.py:740-746`). This is the right shape: agents cannot
  yo-yo to farm reward. *Use this in place of A's `upward_reward × Δcom_z`.*
- **The "hold match bonus, deduplicated per episode" pattern**
  (`moonboard_env.py:749-764`). Dense waypoint signal that doesn't
  reward revisiting.
- **The energy penalty `−c·Σctrl²`** as a regulariser once continuous-joint
  mode is back in play.
- **The slot-aware grip threshold idea** (`SLOT_ALIGNMENT_THRESHOLD` is
  effectively "feet are torque-permissive, hands are not"). Keep the
  *idea*; drop the magic numbers.

### KEEP from C (rl + physics)
- **The 2D `physics/` package as a sanity testbed.** It is fast, easy
  to reason about, and surfaces solver bugs in seconds. Keep it
  available behind `python -m rl` as a smoke layer; do not invest in it
  beyond bug fixes.
- **`legal_actions()` for masked policies** (`rl/env.py:270-279`). The
  Phase-3D env should expose the same affordance — invalid moves are
  effectively a 4·n_holds binary mask the agent can be conditioned on.
- The `ClimberProfile.body: BodyModel` composition idea
  (`physics/body.py:57-71`): RL on a different climber size *should* just
  mean swapping `BodyModel`. A's `ClimberProfile` already does this
  with cm-scalars; keep that style.

### DISCARD
- **The custom Wall geometry in B (`moonboard-rl/src/xml_gen/wall.py`).**
  The hardcoded MoonBoard box is a special case of A's parameterised
  plate. The kickboard is a Phase-2 feature in A's model; add it as a
  generic "second plate" under `sim3d/builder.py` rather than carrying
  B's wall code.
- **The stock Gymnasium humanoid asset in B**
  (`moonboard-rl/assets/humanoid.xml`). It does not have the hip range
  to use the spec ±0.61 m kickboard holds — B works around this by
  narrowing the kickboard to ±0.244 m + snapping the feet. A's custom
  27-DOF body (`sim3d/builder.py:_build_climber_xml`) is purpose-built
  for climbing reach (`hip_flex: −20°…+140°`) and should be the
  go-forward body.
- **B's PPO trainer** (`moonboard-rl/scripts/train_ppo_smoke.py`).
  Single env, single route, no run-dir manifest. Use `sim3d.train`.
- **B's two parsers for `moonboard2.json` and `moonboard3.json`**
  (`moonboard-rl/src/parsers/format2.py`, `format3.py`). A's
  `_legacy_problem_to_public_format` (`sim3d/moonboard.py:266-313`)
  already covers the same data; don't keep both.
- **B's scripted target sequencer**
  (`moonboard_env.py:791-852`). For RL we want the agent to choose
  targets, not have them scripted. Replace with A's
  `Discrete(4·n_holds)` choice.
- **B's `qfrc_constraint[:6]/n_active` slip fallback**
  (`grip_manager.py:460-468`). Misleading — it divides root-DOF
  reaction forces evenly across active slots. Either fix the efc
  lookup or remove the fallback and assert on missing rows.
- **C's `solver/rl_qlearn.py`.** Tabular Q-learning on the static
  reachability graph is a different experiment from what we're doing
  here. Move it under `solver/` and leave it alone.

### BUILD FRESH (or fix in-place rather than copy)

1. **Observation builder.** A's observation has wall-size scaling and a
   quaternion; B's observation has 6-D rotation, k-nearest holds, and
   goal vectors but is MoonBoard-coupled. Write a *new* `sim3d.env._obs`
   that takes the best of both:
   - pelvis pos (3) + rot6d (6) + COM (3),
   - joint qpos[7:] + qvel[6:],
   - **K-nearest holds** with `(rel_pos, role_onehot, gripping_flag)`,
     K independent of `n_holds`,
   - per-limb goal vectors `(target_world − tip_world)`,
   - distance from highest hand to finish (already in A).
   This gives a wall-size-independent observation that A's policy can
   transfer to MoonBoard.
2. **Reward.** Re-anchor A's reward shaping around B's HWM idea:
   - Replace `upward_reward × Δcom_z` with `5.0 × max(0, com_z −
     max_com_z_so_far)`.
   - Add a `+5.0` rising-edge bonus on a new hold match, deduplicated
     per episode.
   - Keep A's `slip_penalty`, `body_intersection_penalty`,
     `invalid_action_penalty`, `fall_penalty`, `on_finish_bonus`.
   - Add `−0.01 × Σctrl²` only when continuous-joint mode is active.
3. **Per-hold slip threshold using positivity AND a per-slot scale.**
   Combine A's `grip_force × positivity × max_force_n` cap with B's
   per-slot multiplier (feet take ~1.5× hand force on overhangs because
   they push, not pull) — but expressed as a slot-multiplier on A's
   per-hold capacity, not as a hardcoded 6000/9000 N.
4. **A *single* `ClimberProfile`.** Delete `physics.body.ClimberProfile`
   (or rename it). The 2D physics layer should import from `sim3d.body`
   and synthesise its own `BodyModel` from height/wingspan. There must
   only be one source of truth for the climber's physical parameters
   once these branches are unified.
5. **A *single* `EnvConfig`.** Same argument. The 2D `rl.env.EnvConfig`
   should either be renamed or extended off `sim3d.env.EnvConfig`.

### Priority order for unification

1. **Lock the public API.** Delete duplicate `ClimberProfile` and
   `EnvConfig`. Make `physics/` import from `sim3d.body`. (~30 min.)
2. **Smoke-train A's discrete-move env on the example wall** with current
   reward and observation. This is the baseline against which every
   change is measured. Use SB3 PPO `--n-envs 4 --steps 200_000` so it
   completes in <2 h on CPU. Confirm `final_com_z` in CSV trends up.
3. **Rewrite the observation builder** (see "BUILD FRESH #1"). Rerun
   the baseline. The discrete-move policy should learn equally well; the
   wall-size independence buys MoonBoard transfer.
4. **Swap reward to HWM + new-hold-match-dedup** (see "BUILD FRESH #2").
   Rerun the baseline. Reward curves should look qualitatively similar
   but with less episode-to-episode noise.
5. **Port the mp4 evaluation callback from B into `sim3d.train`**
   (~1 h). Now every run has video evidence of policy improvement.
6. **Switch to `MoonboardClimbing3DEnv` with corpus split** and run a
   200k–500k step PPO training over the V3/V4 split. This is the first
   "real" generalisation experiment.
7. **Decide on the grip threshold model** (combined per-hold positivity +
   per-slot scale) and migrate. Re-baseline.
8. **Only then** start work on RSI / AMP. The first ~3 days of the
   two-week timeline should be steps 1–6.

---

## Landmines (silent-failure surface)

Each of the following will produce a result that *looks* fine in TensorBoard
but is wrong. Address before launching the long training run.

1. **A's progress reward farms COM oscillation.** `upward_reward ×
   Δcom_z` (`sim3d/env.py:325-326`) is positive whenever the COM rises
   and negative when it falls. Under continuous-joint control the agent
   can converge on a high-frequency bounce that nets +ε per step. Fix:
   switch to HWM (see BUILD FRESH #2). This is the single highest-value
   change to make before any 24 h run.

2. **A's body_intersection_penalty is shaping, not gating.** Limbs are
   physically allowed to pass through the torso; the penalty (default
   `2.0` per contact) can be cheaper than the reward for the resulting
   pose. The `body_intersection_count()` itself is correct; the
   coefficient is the issue. Set to a value at least as large as one
   step's max plausible `upward_reward × Δcom_z`, or convert to a hard
   termination above some threshold.

3. **A's `_finish_z` is the *world-Z of the highest finish hold*, but the
   observation feeds `finish_dist = finish_z − highest_hand_z`** — i.e.
   it ignores the X/Y of the finish. On an overhanging board with finish
   holds offset toward the climber, this can mislead the policy.
   Recompute as Euclidean distance to nearest finish.

4. **A's `Climbing3DEnv` recompiles the MJCF on every construction.**
   `Climb3DWorld()` calls `mujoco.MjModel.from_xml_string` at
   `sim3d/world.py:114`. `MoonboardClimbing3DEnv._build_env` is called
   inside *every* `reset()` (`sim3d/moonboard_env.py:97-104`). At 8
   parallel envs and a real corpus this is hundreds of millisecond per
   reset. Make sure this is what we want; if not, cache MjModel by wall.

5. **B's `sim_substeps=7 × timestep=0.002 = 14 ms` policy period violates
   the documented 20–40 ms range** and prints a warning at construction
   (`moonboard_env.py:144-149`). If the documented range is correct, the
   humanoid dynamics are unstable. If the range is wrong, the warning
   needs to be removed. Either way the code as-written is contradicting
   itself.

6. **B's reset can `raise RuntimeError`** if fewer than 3 grips engage
   (`moonboard_env.py:425-430`). Inside SB3 vec envs this kills the
   whole training process; with `SubprocVecEnv` the worker dies silently
   and PPO degrades to fewer rollouts. If we keep B's reset, wrap it in
   a retry loop with bounded attempts and fall back to a known-good
   pose.

7. **B's grip thresholds are *module globals* mutated with `try/finally`**
   (`moonboard_env.py:358-369`, `400-411`). With `SubprocVecEnv` each
   worker has its own copy, but with `DummyVecEnv` (one process, multiple
   env instances) a `try/finally` race across envs can leak relaxed
   thresholds between resets. Switch to instance attributes.

8. **B's `_advance_foot_target` never fires in a normal episode.**
   `moonboard-rl/KNOWN_ISSUES.md` (Issue 1) confirms feet stay on the
   kickboard for the whole episode. Higher-climbing episodes therefore
   have a non-physical posture and the reward signal is misleading. Do
   not train B's env at length until this is fixed.

9. **A's `_max_force_for` reads `meta["max_force_n"]` even when the
   schema field is absent.** `solver/wall.py` defaults
   `max_force_n=None`; `_max_force_for` guards on `is not None`
   (`sim3d/world.py:531-534`) — correct here, but other call sites should
   be audited because some MoonBoard problems set this to 0 by accident
   in the legacy exports.

10. **A's `cfrc_int` slip detection is documented as an upper bound when
    multiple welds are active** (`sim3d/world.py:702-707`). On a 4-point
    stance, every limb's reported "force" is the sum of all four — so the
    first weld to exceed its capacity will release, then the recomputed
    forces drop. Practically OK but means `info["slips"]` can over-count
    one event as multiple. Add a single-substep dedup, or use
    `efc_force` row matching (B's primary approach, minus its broken
    fallback).

11. **C's `_distance_to_finish` uses Euclidean COM-to-finish in cm**, but
    the "progress" shaping is multiplied by a small constant
    (`PROGRESS_REWARD_PER_CM=0.05`). On a 320 cm wall the maximum
    progress reward across an episode is bounded but the gradient
    relative to the slip penalty is tiny — the agent may prefer to
    stand still and avoid slipping. Not a landmine for the merged
    project (C is not the training target) but worth knowing if anyone
    re-baselines on it.

12. **All three implementations crash differently on degenerate walls** (no
    holds, no starts, no finishes). A falls back to "highest hold = finish"
    (`sim3d/env.py:149-154`) — quietly. B raises. C produces zero
    `_distance_to_finish` and never terminates. Pick a convention.

---

## TL;DR

Keep A (sim3d) as the chassis. Steal B's observation design, HWM
progress reward, and mp4 evaluation callback. Discard B's wall geometry,
its stock humanoid, and its scripted target sequencer. Keep C only as a
2D smoke layer. Before any long training run, switch A's progress reward
to HWM, fix the body-intersection penalty coefficient, and re-baseline
on the example wall before moving to MoonBoard.
