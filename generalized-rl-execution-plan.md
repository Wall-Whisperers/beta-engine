# Generalized MoonBoard RL Climber Execution Plan

## 1. Purpose

This document defines the execution plan for turning Beta Engine's current 3D
climbing prototype into a MoonBoard-generalized reinforcement-learning climber.

The central idea is simple:

> The problem is not merely knowing which holds are on the route. The problem is
> learning whether a simulated body can actually move through the route under
> gravity, contact constraints, grip limits, balance, and timing.

For MoonBoard problems, the official route holds are usually known. A route can
be represented as a set of start, middle, and finish holds. A simple planner or
human can often describe a plausible sequence of holds. What is much harder is
executing that sequence with a body that has limited reach, mass, inertia, joint
limits, grip force, and the need to release and catch holds without falling.

Therefore, this plan prioritizes **continuous embodied reaching and gripping**
over snap-based route planning.

---

## 2. User Decisions Incorporated

The following product and research decisions are locked into this execution plan.

| Topic | Decision |
|---|---|
| Foot usage rule | Feet may use only explicitly marked official route holds. Any hold in the MoonBoard problem may be used by hands or feet, but off-route holds are not allowed. |
| Finish condition | Controlled finish for 2 seconds. The climber must reach the finish and remain stable rather than merely touch it for one frame. |
| Data source | Use `moonboard_data/` as the primary MoonBoard corpus. |
| Generalization target | Start with MoonBoard only. Do not attempt arbitrary generated walls until MoonBoard generalization works. |
| Control approach | Choose Option B: continuous reach-vector / movement-primitive control. Document Option A as a future, more intriguing but harder-to-train path. |
| Viewer | The web viewer should be the main demo and debugging experience. The plan should improve smoothness through real continuous reach, not fake snap playback. |
| Hardware target | CPU first. Single NVIDIA GPU may be used for PPO neural-network updates if available, but simulation should be designed to run on CPU for now. |

---

## 3. Current Project Foundation

Beta Engine already contains most of the foundation required for this plan:

1. A wall JSON schema shared across the editor, solver, physics, and simulator.
2. A 2D solver and 2D physics environment.
3. A MuJoCo-based 3D climber.
4. Weld-style hand and foot attachments to holds.
5. Continuous `reach` and `dyno` movement modes.
6. A Gymnasium-compatible 3D environment.
7. Stable-Baselines3 PPO training support.
8. A browser-based three.js viewer.
9. MoonBoard problem parsers and adapters.

This plan does not replace those pieces. It organizes them around a clearer RL
objective: **learn continuous, physically plausible MoonBoard climbing behavior**.

---

## 4. Core Principle: Do Not Make Snapping the Main Model

The current simulator has a `snap` mode, and that mode is useful for debugging.
However, snapping is not climbing. Snapping says:

> Put this limb on this hold.

A real climber must solve a harder question:

> Can I move my body so that this limb reaches the hold, catches it, loads it,
> and stabilizes without losing the other points of contact?

If the model snaps, it is almost equivalent to someone telling it, "just climb
it." That may be useful for verifying hold order, but it does not test the real
body feasibility problem.

Therefore:

- `snap` remains as a debug baseline and optional curriculum shortcut.
- `reach` becomes the main execution mode.
- Grip attachment should become controlled by proximity and grip commands, not
  guaranteed teleportation.
- The web viewer should show continuous movement from the physics simulation,
  not a sequence of instant pose changes.

---

## 5. Chosen Control Strategy: Option B

### 5.1 Summary

The chosen implementation path is **Option B: continuous reach-vector /
movement-primitive control**.

Instead of asking the policy to output raw torques for every joint immediately,
the policy outputs higher-level continuous movement intentions, such as:

- which limb should reach,
- where the limb should reach relative to the body or nearby route holds,
- how strongly it should reach,
- whether each hand or foot should grip,
- whether each attached limb should release,
- how the torso/core should bias posture.

The environment then converts those continuous intentions into MuJoCo forces,
controller targets, and grip constraints.

### 5.2 Why Option B Is the Right Starting Point

Option B is the best starting point because it sits between two extremes:

1. **Too abstract:** snap-based hold selection.
2. **Too low-level:** raw torque control for a 27-DOF humanoid.

Option B still requires the climber to move continuously through space. It must
reach, swing, load grips, shift weight, and stabilize. But it avoids forcing PPO
to discover all low-level motor control from scratch at the same time it is
learning climbing strategy.

This matches the product goal:

> The system does not need to be a perfectly human neural motor controller. It
> needs to show how a body can realistically perform the climb.

Option B should produce useful, inspectable, physically meaningful climbing much
sooner than full torque control.

### 5.3 How Option B Differs From Snap

In snap mode, the policy or user chooses a target hold and the limb is instantly
attached.

In Option B, the policy must:

1. Move the limb tip through space.
2. Keep enough other contacts to avoid falling.
3. Bring the limb close enough to an official route hold.
4. Activate grip at the right time.
5. Load the new grip without exceeding hold or limb capacity.
6. Release old contacts only when stable enough.
7. Continue upward.

The route holds may be known, but the motion remains physically meaningful.

---

## 6. Documented Future Strategy: Option A

### 6.1 Summary

Option A is **continuous joint-target or torque-level control**.

In this mode, the policy would output low-level control for most or all body
joints, plus grip commands:

```text
action = [
  shoulder targets / torques,
  elbow targets / torques,
  wrist targets / torques,
  hip targets / torques,
  knee targets / torques,
  ankle targets / torques,
  spine target / torque,
  LH grip,
  RH grip,
  LF grip,
  RF grip
]
```

This is the most intriguing long-term direction because it could discover
movement patterns that a hand-designed reach controller might miss:

- unusual drop-knees,
- flags,
- bicycles,
- heel hooks,
- dynamic catches,
- compression positions,
- body tension strategies.

### 6.2 Why Option A Is Not First

Option A is not the first implementation path because it makes the training
problem dramatically harder.

The policy would need to learn all of the following at once:

1. Basic balance.
2. Joint coordination.
3. Reaching.
4. Gripping.
5. Releasing.
6. Route progress.
7. Contact timing.
8. Slip avoidance.
9. Finish stabilization.

For a humanoid-like body, that can require very large simulation budgets and
careful curriculum design. It may eventually be worth doing, but it is not the
fastest route to a useful MoonBoard climbing assistant.

### 6.3 How Option B Prepares for Option A

Option B should be designed so it does not block Option A later.

Specifically:

- Observations should include real proprioception, not only route state.
- Rewards should be based on physical outcomes, not hand-authored move success.
- Grip should already be represented as an actuator-like command.
- The viewer should display actual body motion and contact forces.
- Training logs should record enough information to compare movement quality.

Once Option B works, Option A can be introduced as a lower-level replacement for
some or all of the reach-controller behavior.

---

## 7. Why Not Option C First: Full Raw Torque Control

For clarity, this plan treats full raw torque control as a later research phase,
not the initial implementation.

Raw torque control is attractive because it is the most physically direct. But
it is also the slowest and hardest path to a useful product demonstration.

The goal is not to prove that a policy can rediscover human motor control from
nothing. The goal is to show a climber-like body performing a route in a way
that is continuous, plausible, and useful for understanding beta feasibility.

Option B offers the best tradeoff:

- much more realistic than snapping,
- much easier to train than raw torques,
- compatible with the existing MuJoCo reach machinery,
- appropriate for CPU-first development,
- understandable in the web viewer.

---

## 8. MoonBoard-Only Generalization Scope

### 8.1 Why Start With MoonBoard

MoonBoard is the right first generalization target because it is structured:

- fixed grid width,
- fixed grid height,
- known hold positions,
- known route holds,
- known start and finish holds,
- many problems in a consistent format.

This avoids the need to solve arbitrary wall perception, arbitrary hold geometry,
or procedural generation before the embodied climbing problem is solved.

### 8.2 Primary Corpus

Use `moonboard_data/` as the primary corpus.

The training system should load all usable problems from that directory and
split them into:

- training problems,
- validation problems,
- held-out test problems.

The exact split should be deterministic and saved with each training run.

### 8.3 Problem Sampling

Each episode should sample one MoonBoard problem from the training split.

The environment reset should:

1. Load the selected problem.
2. Convert it to the internal wall representation.
3. Mark official route holds.
4. Mark start holds.
5. Mark finish holds.
6. Seed the climber on the start holds.
7. Restrict all hand and foot contacts to official route holds only.

This creates a clear task distribution:

> Learn to climb official MoonBoard routes, not arbitrary surrounding holds.

---

## 9. Official-Route-Only Contact Rule

### 9.1 Rule

Hands and feet may only use holds that are explicitly part of the official
MoonBoard problem.

That includes:

- start holds,
- middle holds,
- finish holds.

It excludes:

- all off-route MoonBoard holds,
- all hypothetical extra footholds,
- all generated helper holds.

### 9.2 Why This Rule Was Chosen

This rule keeps the problem honest.

If the agent can use any MoonBoard hold as a foot, many problems become easier
than the official route. The model may learn to exploit the board rather than
solve the intended climb.

Official-route-only contact means the learned movement corresponds to the route
as set.

### 9.3 Implementation Implication

The environment should maintain a route mask. A grip can only attach if:

1. The limb tip is near a hold.
2. The hold is part of the official route mask.
3. The limb's grip command is active.
4. The hold is compatible with the limb under the current rules.

Invalid grip attempts should not attach and should receive a small penalty if
they are repeated excessively.

---

## 10. Controlled Finish Condition

### 10.1 Rule

A route is complete only when the climber achieves a controlled finish for 2
seconds.

The minimum finish condition should be:

1. A hand is on an official finish hold.
2. The climber remains attached and stable for 2 seconds.
3. Body velocity remains below a configured threshold.
4. The climber does not slip, fall, or lose the finish contact during the hold.

A stricter mode can later require both hands matched on the finish hold or both
hands on finish holds, but the initial locked requirement is a controlled
2-second finish.

### 10.2 Why Controlled Finish Matters

Without a controlled finish, the policy can exploit the reward by briefly
slapping the finish hold while falling away. That is not a completed climb.

The 2-second finish condition encourages:

- stable catches,
- realistic weight transfer,
- reduced flailing,
- controlled movement quality,
- viewer demonstrations that look like climbing rather than collision exploits.

---

## 11. Observation Design

The observation should combine body state and route context.

### 11.1 Body State

The policy should observe:

- pelvis position,
- pelvis orientation,
- pelvis velocity,
- center of mass position,
- center of mass velocity,
- joint positions,
- joint velocities,
- limb tip positions,
- limb tip velocities,
- current attachment state for LH, RH, LF, RF,
- grip force per attached limb,
- whether each limb is free, reaching, or attached,
- recent slip flags,
- body-intersection/contact warning flags.

### 11.2 MoonBoard Route State

MoonBoard should use a fixed board representation.

MoonBoard has a fixed grid, so represent it as a tensor over board positions:

```text
11 columns × 18 rows × feature channels
```

Recommended channels:

| Channel | Meaning |
|---|---|
| `is_route_hold` | Hold belongs to the official problem. |
| `is_start` | Hold is a start hold. |
| `is_finish` | Hold is a finish hold. |
| `is_current_LH` | Left hand is attached here. |
| `is_current_RH` | Right hand is attached here. |
| `is_current_LF` | Left foot is attached here. |
| `is_current_RF` | Right foot is attached here. |
| `x_normalized` | Board x-coordinate. |
| `z_normalized` | Board height coordinate. |
| `distance_to_LH` | Distance from left hand tip. |
| `distance_to_RH` | Distance from right hand tip. |
| `distance_to_LF` | Distance from left foot tip. |
| `distance_to_RF` | Distance from right foot tip. |
| `hold_quality` | Placeholder for future hold quality annotation. |
| `friction` | Hold or default friction. |
| `positivity` | Hold catch quality. |

### 11.3 Why Fixed MoonBoard Encoding

A fixed MoonBoard encoding allows one policy to train across many problems.

A variable list of holds makes the observation and action dimensions change from
problem to problem. That is bad for a single PPO policy. The fixed 11×18 board
solves that problem for MoonBoard without needing a graph neural network or
transformer in the first version.

---

## 12. Action Design for Option B

### 12.1 Recommended Action Components

The Option B action should include:

```text
action = [
  limb_select_or_limb_weights,
  reach_direction_LH,
  reach_direction_RH,
  reach_direction_LF,
  reach_direction_RF,
  reach_strength_LH,
  reach_strength_RH,
  reach_strength_LF,
  reach_strength_RF,
  grip_LH,
  grip_RH,
  grip_LF,
  grip_RF,
  release_LH,
  release_RH,
  release_LF,
  release_RF,
  torso_posture_bias
]
```

There are two possible versions.

### 12.2 Version B1: Single Active Reaching Limb

At each high-level control step, the policy selects one primary limb to move and
outputs a continuous reach command for that limb.

Advantages:

- easier to train,
- closer to real climbing rhythm,
- easier to debug,
- lower-dimensional action space.

Disadvantages:

- less natural for coordinated two-limb movements,
- dynos may require special handling.

### 12.3 Version B2: Multi-Limb Continuous Control

At each control step, the policy outputs reach and grip commands for all four
limbs.

Advantages:

- more expressive,
- better for coordinated movement,
- better for dynamic moves later.

Disadvantages:

- harder to train,
- easier to exploit,
- more likely to produce noisy flailing early.

### 12.4 Recommendation

Start with B1, then move to B2 after single-limb reaching is stable.

This gives a clean progression:

1. One limb moves while three maintain contact.
2. The policy learns stable releases and catches.
3. Later, coordinated movement is enabled.

---

## 13. Grip Attachment Semantics

Grip should behave like an actuator.

### 13.1 Attach Rule

A limb attaches when:

1. Its grip command is active.
2. The limb tip is within an attach radius of a hold.
3. The hold is an official route hold.
4. The hold is not disallowed by task rules.
5. The contact does not immediately violate force/slip constraints.

### 13.2 Release Rule

A limb releases when:

1. The release command is active, or
2. The grip command drops below a release threshold, or
3. The grip force exceeds the hold/limb capacity and a slip occurs.

### 13.3 Why This Matters

This prevents the environment from giving the policy free catches. The policy
must learn when to close and open grip, which is one of the key realities of
climbing movement.

---

## 14. Reward Design

The reward should make physically plausible climbing the easiest way to get a
high score.

### 14.1 Core Reward Components

| Reward | Purpose |
|---|---|
| Vertical progress | Encourage upward movement. |
| Route progress | Encourage touching new official route holds. |
| Controlled finish bonus | Reward stable route completion. |
| Fall penalty | Make falling catastrophic. |
| Slip penalty | Discourage overloading grips. |
| Invalid grip penalty | Discourage trying to grip off-route or empty space. |
| Step/time penalty | Encourage efficient progress. |
| Energy penalty | Reduce unrealistic thrashing. |
| Jitter penalty | Reduce noisy high-frequency control. |
| Body-intersection penalty | Reject physically impossible poses. |
| Wall-distance/style penalty | Encourage hips/COM closer to wall. |

### 14.2 Vertical Progress

Reward change in center-of-mass height and/or highest-hand height.

Use clipping to avoid rewarding unrealistic jumps or simulation artifacts.

### 14.3 Route Discovery Bonus

Give a bonus the first time any limb attaches to a new official route hold.

This encourages exploration across the route and prevents the policy from
staying near the start holds.

### 14.4 Controlled Finish Bonus

Give the largest positive reward only after the 2-second controlled finish is
achieved.

Touching the finish without control should be useful but incomplete. A brief
finish touch may receive a small shaping reward, but not episode success.

### 14.5 Energy and Jitter Penalties

Penalize:

- large reach forces,
- excessive joint target changes,
- rapid grip toggling,
- unnecessary limb movement,
- high-frequency oscillation.

This is important because without these penalties the policy may find movements
that technically work in simulation but look nothing like climbing.

### 14.6 Style Penalty

Penalize excessive center-of-mass or pelvis distance away from the wall.

This encourages the climber to keep hips closer to the wall, which usually
reduces arm load and creates more realistic technique.

---

## 15. Curriculum

Training should proceed through stages. Each stage should have explicit success
criteria before moving on.

### Stage 0: Environment Validation

Goal: make sure MoonBoard problems load, seed, render, and terminate correctly.

Tasks:

- load problems from `moonboard_data/`,
- create train/validation/test splits,
- reset random problems,
- seed start positions,
- render in the web viewer,
- verify only official route holds are attachable.

Success criteria:

- 100 random problems reset without crashing,
- all starts and finishes are correctly marked,
- off-route grip attempts fail,
- viewer shows correct route masks.

### Stage 1: Balance on Start Holds

Goal: teach or validate stable starting positions.

The climber begins on start holds and must remain stable for a short duration.

Success criteria:

- high percentage of sampled problems maintain contact without falling,
- grip forces remain within capacity,
- no severe body intersections.

### Stage 2: Single-Limb Reach

Goal: learn one controlled release/reach/grip cycle.

The climber starts stable and receives reward for moving one limb to a nearby
official route hold.

Success criteria:

- limb reaches target region smoothly,
- grip closes near hold,
- body remains stable,
- new hold is loaded without immediate slip.

### Stage 3: Short Route Fragments

Goal: climb two-to-four-move fragments sampled from real MoonBoard problems.

Success criteria:

- reliable progress across fragments,
- low fall rate,
- low slip rate,
- visible continuous movement in viewer.

### Stage 4: Full Problems at Easier Angles

Goal: complete full problems with reduced wall difficulty.

Use wall-angle curriculum:

```text
0° → 10° → 20° → 30° → 40°
```

MoonBoard's normal 40° overhang is saved for later stages.

Success criteria:

- completion rate improves at each angle,
- stable transfer from easier angles to harder angles.

### Stage 5: Full 40° MoonBoard Problems

Goal: train on the real MoonBoard angle using official route holds only.

Success criteria:

- meaningful completion rate on training problems,
- measurable success on validation problems,
- controlled 2-second finishes,
- viewer demonstrations look continuous.

### Stage 6: Dynamic Movement and Dynos

Goal: support harder moves that require momentum.

Introduce:

- dyno reach boosts,
- kinetic-energy shaping toward target holds,
- catch stability rewards,
- stricter grip-force realism.

Success criteria:

- dynamic moves can be completed without immediate falling,
- catches stabilize,
- movement remains physically plausible.

---

## 16. Training System

### 16.1 Algorithm

Use PPO first.

Reasons:

- already supported by the project,
- works for continuous action spaces,
- robust enough for early control tasks,
- compatible with CPU-first development,
- easy to replay through the existing web viewer.

### 16.2 Hardware

Primary target: CPU.

Secondary target: single NVIDIA GPU for neural-network updates if available.

Important limitation:

- MuJoCo simulation remains CPU-bound in the first version.
- GPU helps policy training but does not automatically make physics simulation
  massively parallel.

### 16.3 Parallelism

Use vectorized environments when possible, but keep the initial implementation
simple and reliable.

Recommended sequence:

1. Single environment until stable.
2. Small CPU vectorization.
3. Larger CPU vectorization if performance is acceptable.
4. GPU device for PPO if available.
5. Consider MJX or another GPU physics route only after the task design is
   proven.

### 16.4 Saved Run Metadata

Every training run should save:

- git commit hash,
- date/time,
- training config,
- reward config,
- curriculum stage,
- observation version,
- action version,
- MoonBoard data files used,
- train/validation/test split IDs,
- climber profile,
- wall angle settings,
- randomization settings,
- final model path,
- evaluation metrics.

This is necessary for reliable web replay and comparison between experiments.

---

## 17. Web Viewer Plan

The web viewer should be the primary experience for understanding what the model
learned.

### 17.1 Required Improvements

The viewer should display:

- continuous reach motion,
- current route holds,
- current hand/foot attachments,
- grip active/inactive state,
- grip force as a percentage of capacity,
- slips,
- falls,
- current reward components,
- controlled-finish timer,
- current problem metadata.

### 17.2 Smoothness

The preferred way to get smooth animation is not to fake interpolation between
snapped states. The preferred way is to run the actual MuJoCo continuous reach
simulation and stream enough intermediate poses to the browser.

If temporary smoothing is needed for user experience, it should be labeled as
visual interpolation only and should not be confused with the physics state used
for training.

### 17.3 Policy Replay

The viewer should support:

1. Loading a saved run directory.
2. Reconstructing the exact problem and config.
3. Running the policy step by step.
4. Playing the policy continuously.
5. Displaying why it failed when it fails.

This makes the viewer not only a demo but a debugging tool.

---

## 18. Evaluation Metrics

Track metrics at both episode and corpus levels.

| Metric | Why it matters |
|---|---|
| Completion rate | Primary success metric. |
| Controlled-finish rate | Ensures the finish is stable. |
| Average high point | Measures progress even on failed climbs. |
| Route holds touched | Shows whether the policy explores the route. |
| Falls per episode | Measures stability. |
| Slips per episode | Measures grip realism. |
| Invalid grip attempts | Measures whether it understands route constraints. |
| Energy per meter climbed | Measures movement efficiency. |
| Jitter score | Measures smoothness. |
| Body-intersection count | Measures physical plausibility. |
| Generalization gap | Difference between train and held-out problem performance. |

Evaluation should run on held-out MoonBoard problems from `moonboard_data/`.

---

## 19. Domain Randomization

Domain randomization should be introduced after the baseline task works.

Randomize:

- climber mass,
- wingspan,
- height,
- grip strength,
- hold friction,
- hold positivity,
- wall friction,
- small gravity variations,
- controller gains,
- initial body perturbations.

Why wait?

If randomization is introduced too early, failures become difficult to diagnose.
First prove the policy can learn the nominal task. Then use randomization to make
it robust.

---

## 20. Hold Quality Limitation

MoonBoard problem files may not contain enough hold-shape information to know
whether each hold is a jug, crimp, sloper, pinch, or poor foothold.

The first version can use defaults, but that limits realism.

Future improvement:

- add a MoonBoard hold-set annotation file,
- map each board position to hold type, size, positivity, and friction,
- allow different MoonBoard setups to have different hold metadata.

This will make grip-load and movement feasibility much more realistic.

---

## 21. Development Milestones

### Milestone 1: Planning and Interfaces

Deliverables:

- finalized observation spec,
- finalized Option B action spec,
- MoonBoard corpus split design,
- reward component list,
- viewer overlay design.

### Milestone 2: MoonBoard Corpus Environment

Deliverables:

- environment samples problems from `moonboard_data/`,
- fixed MoonBoard route tensor,
- official-route-only contact mask,
- deterministic train/validation/test split,
- reset validation tests.

### Milestone 3: Grip Actuator Semantics

Deliverables:

- grip command controls attach/release,
- off-route holds cannot attach,
- invalid grip attempts tracked,
- grip force/slip metrics exposed.

### Milestone 4: Continuous Option B Controller

Deliverables:

- single-limb reach-vector action mode,
- grip timing integrated with reaching,
- stable MuJoCo stepping,
- smooth web replay.

### Milestone 5: Reward and Curriculum

Deliverables:

- route progress reward,
- controlled 2-second finish,
- energy/jitter/style penalties,
- curriculum stages 1-3.

### Milestone 6: Full MoonBoard Training

Deliverables:

- PPO training over all training problems,
- validation evaluation,
- saved run metadata,
- web viewer policy replay.

### Milestone 7: 40° MoonBoard and Dynos

Deliverables:

- wall-angle curriculum to 40°,
- dynamic reach/dyno support,
- held-out test evaluation,
- polished viewer demos.

---

## 22. Risks and Mitigations

### Risk 1: Training Is Too Slow on CPU

Mitigation:

- start with short route fragments,
- use single-limb action mode first,
- reduce simulation horizon early,
- use vectorized CPU environments later,
- use GPU for PPO updates if available.

### Risk 2: Policy Learns to Flail

Mitigation:

- energy penalty,
- jitter penalty,
- grip spam penalty,
- controlled finish requirement,
- body-intersection penalty,
- staged curriculum.

### Risk 3: Policy Exploits Grip Constraints

Mitigation:

- force-limited grips,
- slip penalties,
- realistic hold positivity,
- attach only near official route holds,
- track grip force as percentage of capacity.

### Risk 4: Viewer Looks Discrete

Mitigation:

- stream intermediate MuJoCo poses,
- use reach mode rather than snap mode,
- separate visual interpolation from physics truth,
- add playback speed controls.

### Risk 5: MoonBoard Hold Metadata Is Too Weak

Mitigation:

- start with defaults,
- add hold-set annotation later,
- expose hold quality assumptions in run metadata.

---

## 23. Final Implementation Choice

The final execution choice is:

> Build a MoonBoard-only, route-conditioned, continuous embodied RL climber using
> MuJoCo, Option B movement-primitive control, grip actuator semantics,
> official-route-only holds, controlled 2-second finishes, PPO training, and web
> viewer replay.

This choice is made because it directly targets the hard part of climbing:
physical execution.

It avoids the main weakness of snap-based planning, which is that snapping
removes the body problem. It also avoids the main weakness of raw torque control,
which is that it may require too much training before producing useful behavior.

Option B is the right middle path:

- realistic enough to test body feasibility,
- continuous enough to produce smooth viewer motion,
- constrained enough to train on CPU-first infrastructure,
- compatible with the existing MuJoCo code,
- extensible toward full joint/torque control later.

Option A remains the long-term research path. Once Option B proves the task,
reward, observation, and MoonBoard generalization design, the project can move
toward lower-level joint-target or torque policies with a much better foundation.

---

## 24. Implementation Chunk Log

This section tracks work completed against the plan so future chunks can be
reviewed and tested independently.

### Chunk 1 — MoonBoard corpus sampling and official-route mask

Status: **completed**.

Implemented pieces:

- Added deterministic MoonBoard corpus loading from a directory or single JSON
  source.
- Added deterministic train / validation / test splitting with saved run
  manifests.
- Added a MoonBoard-generalized Gymnasium wrapper that samples a new problem on
  reset while keeping fixed 11×18 MoonBoard action and observation dimensions by
  building each sampled wall with all 198 T-nut positions.
- Added official-route-only contact masking for full-board MoonBoard episodes.
  Off-route holds remain present as fixed-index placeholders, but hand and foot
  move actions targeting them are rejected and lightly penalized.
- Wired `sim3d.train` so `--moonboard PATH` without `--problem ID` trains on the
  selected deterministic split, while `--problem ID` preserves single-problem
  training.
- Saved `moonboard_splits.json` beside generalized MoonBoard training runs so
  each run records exactly which problems were used for train / validation /
  test.

Testable boundary:

- Corpus helpers can be tested without MuJoCo training.
- `MoonboardClimbing3DEnv` can be reset and stepped independently.
- A tiny PPO smoke run can be executed against `data/moonboard/sample-problems.json`
  with a 1-step episode and 2 total PPO steps.

Next chunks:

1. Add the fixed 11×18 route-state tensor channels from Section 11 to the
   observation instead of relying only on the existing flat hold one-hot state.
2. Add the first true Option B continuous reach-vector action mode with grip and
   release commands, keeping the existing discrete move mode as a debug baseline.
3. Upgrade the finish condition from action-count frames to an explicit 2-second
   controlled timer with body-velocity stability checks.

### Chunk 1b — Ground-level start visualization and generalized replay fix

Status: **completed**.

Implemented pieces:

- Added `start_mode="ground-reach"` to start the body from its ground-level
  default pose and immediately begin continuous left/right hand reaches to the
  official start hold(s), instead of welding the body directly onto the route.
- Exposed the start mode through the training CLI, native simulator CLI, and web
  viewer start controls.
- Added full-board MoonBoard replay support so generalized split-trained models
  can be replayed with the same 198-hold action/observation dimensions they were
  trained with.

Testable boundary:

- Headless native sim can verify that `ground-reach` starts with no welded limbs
  and active reaches to the MoonBoard start hand holds.
- Tiny PPO runs now print replay commands containing `--moonboard-full-board`
  for generalized MoonBoard split policies.
- The browser viewer can start a session in either seeded mode or ground-reach
  mode via the **Start mode** dropdown.
