# Next Steps & Known Issues

> **Freshness warning.** This is a living roadmap, not a snapshot. Update
> or delete entries as you act on them — a stale roadmap is worse than no
> roadmap. Prune aggressively; don't let this drift into an audit doc that
> rots while the code moves.
>
> Last reviewed: 2026-05-26.

---

## Where we are

The sim3d/ merge landed: one canonical 3D MuJoCo env, fixed-shape (127,)
observation, `Box(25,)` continuous-joint action (21 joint targets in
[-1,1] + 4 per-limb grip intents), HWM reward, kickboard support, video
rollout callback (now wired into `train.py` by default), curriculum env,
MoonBoard train/val/test split. The env constructs, resets to a 4-grip
seed pose, and steps without NaN.

**The body is a custom 21-DOF humanoid (27 total with the free root) —
NOT the stock MuJoCo humanoid.** It is anatomically grounded (Winter mass
fractions, measured segment ratios, climbing-realistic joint limits, PD
position actuators tuned near critical damping per joint group). This is
the right accuracy/complexity balance and should be kept.

**Body fixes landed 2026-05-26** (found by viewing it in `mjpython`): the
right-side abduction/rotation hinge axes weren't mirrored (so "abduct"
adducted the right limbs across the body); `hip_flex` pointed the wrong
way for a wall-facing climber (knee could only flex 20° toward the wall vs
140° backward, forcing the "knee bends backward" contortion to reach
footholds); the feet pointed away from the wall (+Y); and the arms floated
11 cm off the torso (chest ellipsoid tapers to a point at shoulder height).
Fixed by negating the four mirrored axes + both `hip_flex` axes, flipping
the foot tip sites to −Y, lowering the shoulder mount, and adding deltoid
spheres. Symmetric commands now produce symmetric poses; the seed pose
settles stably with 0 self-intersections on the example wall.

The lever for "can it learn to climb" is **not** the body — it's the
training setup below. Do not add DOF until basic climbing trains (more DOF
= harder exploration).

What we still **don't** have:

- A policy that completes any problem with > 0% success.
- An episode long enough to climb (see 0.1 — this is the headline bug).
- A first task the agent can actually solve from random init (curriculum
  "hang" mode does not exist yet).
- A dense signal *toward the next hold* — partly addressed: the height-progress
  reward (A2, delivered) gives a smooth up-gradient, but there is still no
  per-limb reach gradient by default (the old one farmed; see A2).
- A success-rate eval harness on a held-out split.

Honest framing: the simulator is solid; the **training loop is configured
in a way that cannot produce a climb**, and the reward landscape has no
gradient out of "hang still." Phase 0 + Phase A bridge that gap.

---

## Phase 0 — train.py defaults make learning impossible (DO THESE FIRST)

These are not tuning items. With them unfixed, every run wastes GPU/CPU
hours producing flat reward, regardless of algorithm.

### 0.1 Episode length is 3.84 s in continuous mode — THE bug

`TrainConfig.max_episode_steps = 30` is passed straight into
`EnvConfig.max_steps`. In continuous-joint mode each `env.step()` advances
`sim_substeps(8) × SUBSTEPS_PER_FRAME(8) × PHYS_DT(0.002) = 0.128 s`
(≈ 7.8 Hz control). So **30 steps = 3.84 s of simulated time** — a human
takes 10–60 s to climb a MoonBoard problem. The `30` / `move_frames=24`
defaults are holdovers from `discrete-move` (where 1 step = 1 full limb
move). Fix: set `max_episode_steps` to **~800–1500** for continuous mode
(≈ 100–190 s), and stop threading `move_frames`/`move_mode` into the
continuous env config. Verify with `episode_stats.csv` `length` column.

### 0.2 MoonBoard seed pose puts the feet ABOVE the hands

Confirmed on `sample-problems.json` problem 0 (`Far from the Madding
Crowd`, starts C5/E6): `_default_seed_kwargs` returns hands on the start
holds (good) but feet on `mb_E8` (row 8) and `mb_F11` (row 11) — *above*
the hands at rows 5–6. Root cause: MoonBoard holds are all `hold_type=
"jug"`, so the `hold_type=="foothold"` foot search finds nothing and
falls back to "two lowest non-hand route holds," which for a real problem
are mid-route holds high on the wall. The kickboard exists for exactly
this (canonical low foot start) but its holds are only appended to
`hold_meta`, never to `wall.holds`, so `_default_seed_kwargs` can't see
them (confirmed: feet never land on `kb_*`). Fix: expose kickboard holds
through the `Wall` object (or have `_default_seed_kwargs` consult
`hold_meta`) and prefer kickboard/low footholds for the foot seed. Until
this is fixed, MoonBoard episodes start from a physically absurd pose.

### 0.3 Seed pose looks like a deep frog squat (naturalness, not joints)

The forearm-through-chest intersection that used to fire here is now gone
(lowering the shoulder mount fixed it — 0 intersections at the example-wall
seed). What remains is *pose quality*: `seed_pose` drops the pelvis low
(crouch branch sets `pelvis_z = 0.5·(foot_z + hand_z)`), so on the example
wall the hips flex ~100–120° and the legs splay horizontally to reach
wide-spaced footholds. The joints are anatomically correct now; the pose
just isn't a natural climbing stance. Fix: bias the seed pelvis higher
(more leg extension) and prefer narrower/lower footholds, then re-view in
`mjpython`. Verify the 0-intersection result also holds on a MoonBoard
problem (only checked the example wall so far).

### 0.4 The neutral action releases every grip

Action grip-intent semantics: `intent > 0` engages, `intent ≤ 0` releases.
SB3's Gaussian policy initializes at mean 0, `log_std=0` (std 1.0), so on
step 1 each of the 4 grips is positive ~50% of the time and the 21 joint
targets are ~N(0,1) noise → the body releases ~2 limbs and flails off the
wall almost immediately. Combined with 0.1/0.3 this means early training
is dominated by `fall (−50)` and intersection penalties. Options: (a) lower
`log_std_init` (e.g. −1.5) so the policy starts near "hold the seed pose";
(b) flip semantics so the *zero* action holds the current grips and the
agent must act to release; (c) forbid releasing a grip when it would drop
below 2 contacts. (a) is the cheapest and should be done alongside A3.

---

## Phase A — make PPO actually learn (after Phase 0)

### A1. Curriculum: "hang" → "reach-one" → "climb" — BUILT (2026-06-05)

Delivered as `EnvConfig.task_mode ∈ {hang, reach-one, climb}` +
`StagedCurriculumEnv` (auto-advance on rolling success); `--staged-curriculum`.
See CLAUDE.md → Task-stage curriculum. **reach-one is learnable** — a PPO policy
went 0%→ (stably, no collapse) on a generated wall, the project's first learned
climbing move. Getting there forced three fixes now baked in: the grip-strength
physics (one-hand stances were impossible), PPO trust-region guards (`target_kl`
etc.), and a target-aware observation.

### A1b. Chaining reaches into a climb — single move SOLVED via imitation + RSI

reach-one works **with scaffolding**: a *designated* mover hand (per-limb
deadband 0, releases freely), a *target-aware* obs (the mover's goal vector
points at the target), and a dense signed-potential reach reward. A full **climb
does not** — six distinct attempts (2026-06-06) all hit the same wall:

| Approach | Result |
|---|---|
| plain climb (strong grips) | hangs 5–6× longer, never reaches up |
| + dense `finish_approach` (B2-fixed, tip-based) | never *attempts* the reach |
| + `reach_approach` | farms (fall-and-swing) |
| lower deadband (release exploration) | random releases → falls |
| warm-start climb from the reach policy | release-and-fall (regime mismatch) |
| per-move reverse curriculum (`StagedCurriculumEnv`) | most moves unlearnable, no transfer |
| full-climb reverse curriculum (`ClimbCurriculumEnv`) | stalls on the *first* 2-row reach |

**Root cause:** in climb mode the agent must *choose* which hand to move, so
there is no designated mover and the protective deadband applies to all hands
equally — it never explores release-and-reach, even from a stance 2 rows below
the finish. The per-limb scaffolding that makes a single reach discoverable does
not exist in free-choice climbing, and it does not compose across moves (this is
the "options don't compose" problem in hierarchical RL). **No reward tweak fixes
this** — it's exploration/structure, not shaping.

**Resolved (2026-06-07) — imitation + RSI, not RL-discovery.** Following the
Babadi/Naderi/Hämäläinen method (PPO fails to *discover* long-horizon skills; the
field splits non-RL discovery from a tracking controller), the loop is now:
author a reference *without RL* → train PPO to track it with a **bounded
DeepMimic reward** (0.65 pose / 0.10 vel / 0.15 endeff / 0.10 com, ∈[0,1],
un-farmable) + **Reference State Initialization** + a **termination curriculum**
(R_min 0.75→0.50). A de-risk probe (`sim3d/probe_transitions.py`) first
established the prerequisite the chaining failures hid: the transitional stances
ARE holdable; the catastrophe was the open-loop KP-1500 reach controller
overloading anchors (release-survives-but-reach-sheds on 12/17 moves), and RSI is
faithful 16/17. New modules: `sim3d/reference.py` (Reference + bounded reward +
weight-shift authoring) and `sim3d/imitation.py` (ImitationEnv); `env.py` gains
`task_mode="imitate"` + `reset_to_reference()`; `world.py` gains `rsi()`. A single
landing move trains **0→88%** (monotonic, no collapse, no reward-hacking) — the
project's first learned move that releases, reaches, and regrips a target hold.

### A1d. THE FRONTIER — a feasible multi-move reference (CMA-ES discovery)

Multi-move authoring **infrastructure** is built and correct: `stitch_references`
(concatenate per-move refs) and `author_climb_reference` (RSI-chain per-move
authoring — start each move from the prior move's clean RSI'd end frame, so
boundaries stay smooth *and* no instability accumulates; snap to the closest
approach so marginal reaches land). The boundary-teleport and A1c-collapse failure
modes are both solved at the mechanism level (verified: boundary jump 0.09 rad,
was 0.86).

**But authoring a feasible multi-move reference by reach-rollout is blocked**, and
five approaches confirmed it's not a code bug — it's that the hand-coded reach
controller is too marginal (2026-06-07): independent+stitch → boundary teleport;
continuous → A1c collapse (grips 3→1→0); RSI-chain → moves don't land+hold; snap →
grip slips (over cap); grip-boost → reach falls short; tighter walls
(`reach_frac` 0.4) → still aborts. Every facet of the same root: on generated
walls the static reach won't close to the grip radius and the stances sit over
grip cap.

**This is exactly why the paper uses a trajectory optimizer for discovery.** Next:
build the **CMA-ES discovery stage** (Naderi 2017) — search joint-space keyframes
per move to satisfy reach + balance + grip-capacity, so references are *optimized
to be feasible* instead of hoping a marginal controller lands them. Reuses the
`Reference` container, `rsi()`, `imitation_reward`, and `holdable_fraction`.

Lesser open items: graduate grips from reference-driven (v1) to policy-controlled
(rewarded against the contact schedule); the **vetting bug** —
`feasible_reach_moves`/`feasible_climb_stances` accept a stance that merely
*doesn't fall* in N steps, so they greenlight seeds 1.3–8× over grip cap (CMA-ES
should vet true holdability); generalise beyond ONE fixed wall.

### A1c. THE DEEPER ROOT — transitional-stance instability (2026-06-06)

Tried to bootstrap via the discrete-move A* expert (BC source). Found the expert
had *also* stopped climbing — root-caused and **fixed two real bugs** (commit
6bd0c32): the reach controller's relax zeroed only the actuator gain, not the
−kp/−kv bias (a spring-to-zero that fought the reach), and `REACH_KP` was tuned
for the old softer body. Isolated reaches now land 6/6 (were 0).

**But even the fixed hand-coded expert can't reliably chain a full climb** — it
lands 1–4 moves then the body destabilises and falls, even with actuator-target
sync + a settle after each move. So the chaining wall is **not just an RL
exploration problem** — it blocks a perfect planner + hand-coded controller too.

The precise root: **the body holds *tuned* stances (the hangable seed) but not
arbitrary *mid-climb transitional* stances.** Each move lands; the *next*
transition tips it off. This is the single highest-leverage thing to fix — it
unblocks the expert (→ BC) AND RL at once.

**First balance attempt (2026-06-06) — instructive failure.** A pelvis-hold PD
(pin the pelvis at its pre-move position during a reach; `BALANCE_KP`, now 0/off)
made chaining *worse*. A move-by-move diagnosis showed the failure is
**three-factor**, not one:
1. **Grips decay** 3→2→1→0 across the sequence — the raised reach force (KP
   1500) overloads/sheds the *anchor* grips during a reach.
2. **Barn-door** — the pelvis drifts *out* from the wall (y grows each move).
3. **Frame-sensitive reaches** — 60 frames/move often don't land; 120 destabilise.
Pinning the pelvis fixes none of these and blocks the body from rising to a hold.

So the balance assist needs a smarter design: keep the COM over the **support**
(not a fixed point), *allow* vertical rise, AND coordinate reach-strength vs
grip-force so anchors don't shed mid-reach. This is a real whole-body-control
effort (or: learn it via RL with a stability reward) — the current frontier.
NOT "unrealistic motion"; specifically transitional balance + grip retention.

### A2. Dense potential-based shaping — DELIVERED (reward clean-restart, 2026-06-05)

The reward was rebuilt around a potential-based **height-progress** term:
`+60 × (com_z − prev_com_z)` per step — symmetric, telescoping, un-farmable —
plus a `+10` first-touch hold-match nudge, the `+200/−50` terminals, and the
physics gates. All the old shaping terms (reach, survival, new-high-grip,
finish-approach, HWM) are now **inert by default** but kept behind `if coeff>0`
so they can be re-added one at a time, diagnostics-driven. Per-term reward
decomposition is logged to `episode_stats.csv` (`r_*` columns); view with
`python -m sim3d.plot_reward_terms`. See CLAUDE.md → Reward Function.

If the clean signal still plateaus, that points at **exploration, not reward
shape** — the next lever is A1 (the "hang → reach-one → climb" curriculum:
an achievable first task from random init), not another shaping term. The
first shaping term to consider re-adding is `finish_approach_coeff` (with the
B2 reference-jump fixed) and only if videos show "climbs but wanders off-route."

### A3. VecNormalize + continuous-control PPO hyperparameters (NEW)

`train.py` wraps the env in a bare `DummyVecEnv`/`SubprocVecEnv` — **no
`VecNormalize`**, and PPO runs on SB3 defaults (no `gae_lambda`,
`ent_coef`, `n_epochs`, `clip_range`, `log_std_init`). The observation
contains raw world positions (pelvis up to ~3 m, com, hold offsets) and
joint velocities at very different scales — unnormalized this is a known
PPO failure mode on MuJoCo. Do: wrap in `VecNormalize(norm_obs=True,
norm_reward=True, clip_obs=10)` (and save/load its stats with the model);
bump `gamma` to ~0.997 (the effective horizon at 7.8 Hz and 0.99 is only
~13 s); raise `n_steps × n_envs` to ≥ 2048–4096; set a small `ent_coef`
(e.g. 0.001) for exploration; lower `log_std_init` per 0.4.

### A4. Behavior cloning from the discrete-move expert

`discrete-move` + the Cartesian-impedance reach controller is a working
hand-tuned expert. Generate ~10k `(obs, joint-target, grip-intent)` tuples
by running scripted betas, BC-pretrain the MlpPolicy, then PPO fine-tune.
Avoids the "random 25-D Gaussian for 1 M steps does nothing" mode. Do this
only if A1–A3 still stall — it's more work than the shaping/curriculum.

### A5. Pretrain on jug walls before MoonBoard

`data/walls/` has hand-designed walls with `jug` holds and generous
spacing (positivity 1.0, high max_force). Train there first; transfer the
checkpoint to MoonBoard crimps. Obs shape is identical, so weights port
directly. Note: jug walls already carry `foothold`-type holds, so they
avoid the 0.2 feet-above-hands seed problem — a good reason to start here.

### A6. Tighten grip engagement (proximity + alignment)

At 5 cm with a closest-hold sort, a flailing limb can engage a grip the
policy never learned to "place." Either drop `GRIP_PROXIMITY_M` to ~2 cm,
or require both proximity AND a positive dot product between the limb's
approach direction and the hold's `wall_normal` (already in `hold_meta`).

---

## Phase B — reward & observation tuning (after Phase A shows signal)

### B1. Profile the energy penalty

`−0.005 × Σ ctrl²` over 21 joints can reach ~0.1/step; a 1 cm com_z HWM
gain is only +0.05. The penalty currently outweighs the slow careful
moves we want. Drop `energy_penalty_coeff` to ~0.001, or normalize by
`nu`, or penalize post-clip torque rather than ctrl input.

### B2. Smooth the finish-distance observation

`obs[-1]` (highest gripped hand → nearest finish) jumps discontinuously
when the highest hand releases. EMA it, or always use the geometric max
of all four limb tips, to avoid destabilizing the value function.

### B3. Add an on-route / grippable flag to the K-nearest holds (NEW)

With MoonBoard `include_full_board=True` there are 198 holds and
`official_route_only=True` masks all but the current route — but the
observation's per-hold role one-hot is only `[start, mid, finish]`, so an
**off-route grey hold and an on-route mid hold look identical** to the
policy. The 8 nearest holds may be mostly ungrippable, and the agent can't
tell. Add an `is_eligible_for_this_limb` channel (or drop ineligible holds
from the K-nearest sort). High impact for the generalized MoonBoard task.

### B4. Per-hold capacity sanity check

`FOOT_FORCE_MULTIPLIER = 1.5` is a guess. Compare against real climber
power (Goddard & Neumann, or the Lattice grip dataset). Over-strong feet
let the agent discover unphysical heel-hook betas.

---

## Phase C — evaluation infrastructure (mostly missing)

### C1. Success rate, not just per-episode outcome

`episode_stats.csv` already logs an `outcome` column, so success rate is
derivable post-hoc — but there is no `sim3d/eval.py` that runs N=100 seeds
on a held-out problem and reports `success rate ± stderr`, and no periodic
eval callback. Build it and wire it into `train.py`.

### C2. Held-out validation is configured but never evaluated

`TrainConfig.moonboard_split` splits train/val/test deterministically, but
nothing loads the validation split mid-training. Add a periodic
eval-on-validation callback so we can detect overfitting to a single
problem. (Video rollout already ships by default — that earlier TODO is
done; don't re-add it.)

---

## Phase D — performance & infra

### D1. MoonboardClimbing3DEnv rebuilds the MjModel every reset

`_build_env` recompiles MJCF on every `reset` to swap the problem
(~30–100 ms). With vec-envs hitting reset on done flags this can dominate.
Cache `{problem_id: env}` and round-robin instead of rebuilding.

### D2. Grip search is O(n_holds) per intent per step

`_maybe_engage_grip` scans every hold whenever any intent is positive
(198 holds × 4 limbs × ~1000 steps × n_envs). Build a KD-tree once at
construction.

### D3. Pin imageio in requirements

`requirements.txt` should pin `imageio` + `imageio-ffmpeg` so the video
callback works out of the box instead of silently no-opping.

---

## Phase E — body model & physics (longer horizon, only after it climbs)

### E1. Spine has 1 DOF (forward lean only)

Real climbers twist and lean laterally (gaston, drop-knee, cross-through).
The single `spine_lean` hinge blocks all of these. Two more spine hinges
(lateral + rotational) unlock the most common 3D moves the policy
currently cannot perform — but they also enlarge the action space and make
exploration harder, so defer until the agent reliably climbs with 1-DOF.

### E2. No finger DOF

By design (CLAUDE.md): crimp vs jug is modeled only through grip-force
capacity, not finger flexion. Fine for now; revisit only if we later
predict per-hold grade difficulty for specific climber strengths.

---

## Things you should NOT do

- Switch to `discrete-move` as the default because continuous PPO is slow.
  That hides the problem we're solving. Keep it only as the BC expert (A4).
- Replace the custom climber with the stock Gymnasium humanoid because "it
  trains faster." It trains faster because it's tuned for locomotion, not
  climbing; switching throws away the anatomical grounding, joint limits,
  and the hold-attach machinery. The training-loop fixes (Phase 0/A), not
  the body, are what unblock learning.
- Add DOF to the body before it can climb with the current DOF. More joints
  = harder exploration; earn the complexity later (E1).
- Add `KNOWN_ISSUES.md`, `ARCH_REVIEW.md`, or per-phase trackers. This file
  is the only roadmap. Keep it pruned.

---

## Tracking

When you finish an item, **delete it from this file**, then add one or two
sentences to `CLAUDE.md` if it changed the permanent architectural
contract. Cumulative changelog lives in git; this file is for "what's next."
