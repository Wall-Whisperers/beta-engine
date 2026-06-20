# Next Steps & Known Issues

> **Freshness warning.** This is a living roadmap, not a snapshot. Update
> or delete entries as you act on them — a stale roadmap is worse than no
> roadmap. Prune aggressively; don't let this drift into an audit doc that
> rots while the code moves.
>
> Last reviewed: 2026-06-17.

---

## Where we are (2026-06-19) — discovery pipeline overhauled; HAND-move authoring now reliable, FOOT reach is the lone wall

This session fixed the reference-authoring pipeline and isolated the one real
remaining blocker. **Hand-move authoring is now reliable; the foot reach is the
only thing left.**

**Verified fixes (all in `sim3d/discover.py`, ~124 LOC):**
1. **`--auto-first-move`** — probes the feasible first-moves and picks the
   tightest-lander instead of greedy `feas[0]` (which is only 2D-vetted and is
   often unclosable: seed 14's feas[0] failed at 0.37 m). VERIFIED: hand moves
   now land **0.004–0.017 m** on the *same walls* that used to "fail." The walls
   were fine; greedy selection was the bug.
2. **Tightness-chasing restarts + `--max-gap`** — `discover_move` used to break
   on the first landing inside the loose 0.08 m radius, so restarts never bought
   tightness. Now it chases `max_gap_m` and ranks `best` by GAP (not cost, which
   the posture term dominated — that made it return a loose 0.054 m attempt over
   a found 0.017 m one). VERIFIED: RH→h_004 0.054→0.015 m.
3. **Seed-move warm-start (`seed_x0`)** — CMA gap has high run-to-run variance
   (same move: 0.013 m one run, 0.062 m the next), so the full-budget step-0
   search cold-started into a looser basin than the selector's probe already
   found, getting auto-selected moves rejected. Now step 0 warm-starts from the
   probe's `x_best`. VERIFIED: step-0 now lands 0.010 m, consistent with its
   0.019 m probe — variance-induced rejection gone.
4. **Respect the wall's `cell_size_cm`** — the CLI force-loaded every wall at
   20 cm, silently rescaling fine-grid test walls.
5. **`--foot-balance`** — capped pelvis balance assist during FOOT swings only
   (plumbed; see below — 250 N was insufficient).

**Calibration (measured): trainable landing gap is ~1–2 cm** (`ref_adaptive_s14`
trained 100% at 0.7 cm; the chain that failed had a 7.0 cm move-0).

**The lone remaining wall — FOOT REACH (was blockers #3+#4).** The open-loop
foot swing lands **~9–12 cm short of ANY upward foothold**, and this is NOT
fixable by the above:
- It is NOT foothold spacing: built a 10 cm-grid foot-test wall
  (`data/examples/footstep-fine-v1.json`); the foot still landed 9–12 cm short
  of the 10 cm-spaced footholds. The nearest reachable foot target simply isn't
  reached.
- `--foot-balance 250` did NOT close it (still 12 cm short) — so it's not (only)
  body-sag; the leg can't extend the foot far enough from a hang stance.
- This is the documented multi-week foot-move blocker. The intended real
  solution is `free_mover_imitation` on the TRAINING side (don't track the
  un-trackable recorded foot pose; reward reaching the real hold) — but that
  still needs the move to be *action-reachable*, which open-loop CMA can't
  currently author here.

### Authoring-side foot fixes — TESTED AND RULED OUT (2026-06-19)

The foot reaches a HARD CEILING of ~5–6 cm up from a hang stance in open-loop
`discover_move`, and won't close to a tight grip. Measured on the fine test
walls (`footstep-fine-v1.json` @10 cm, `footstep-fine5-v1.json` @5 cm):
- **Balance assist does NOTHING** — LF→f_003 (11.3 cm straight up): gap 7.7 cm
  at 0 N, 7.8 at 250 N, 7.9 at 500 N. Flat. The shortfall is not body-sag.
- **Finer footholds don't help** — at 5 cm spacing the foot still can't close:
  6.3 cm step → 4.9 cm gap, 11.3 → 7.1, 16.3 → 10.0 (no grip). The foot tops
  out at ~0.48 m (~5 cm above its 0.41 m start) regardless of target.
- So smaller steps + balance + spacing are all dead ends. **Open-loop CMA is the
  wrong tool for foot moves** — it can't produce the active weight-shift that
  lifts an unweighted foot.

### Concrete next step — TRAIN the foot move with free-mover (training-side, not authoring)

The closed-loop RL policy CAN do what open-loop CMA can't (active weight-shift +
capture-sphere closing). `free_mover_imitation` already exists
(`sim3d/imitation.py:154,461`; `--free-mover-imitation`): it excludes the
mover's joints/tip from the pose term and rewards reaching the REAL hold, so it
doesn't need a tight authored foot landing — only that the foot grip be inside
the 0.08 m capture sphere (it is: open-loop lands ~7.7 cm = within 8 cm).
- Author a short reference whose foot move grips (gap < 0.08, which open-loop
  CAN do) on a fine-foothold wall via the now-reliable hand pipeline
  (`--auto-first-move`) + the foot move.
- Train: `--chain --free-mover-imitation --mover-capture-coeff 0.3
  --rsi-phase-max <foot-swing-start>` (the capture_v2 recipe, foot variant).
- Headline question: can the policy close the last ~6 cm the open-loop author
  couldn't? If yes, foot moves are unblocked. If no, the foot ROM/strength under
  hang needs a body-model look (hip_flex torque cap, or a stand-up primitive).

**Do NOT** re-litigate hand-move selection/tightness (solved), retry balance
assist / finer footholds / smaller steps for foot AUTHORING (all ruled out
above), or sweep `build_tight_wall` seeds blindly.

---

## Where we are (2026-06-17) — single-move 100%, next: multi-move

**Single-move SOLVED (2026-06-17).** `capture_v2` on `ref_adaptive_s14` (RH
h_002→h_048, 33 frames, 3-frame swing) reached **100% frame-0 (40/40 eps)**
deterministically. Video: 4/4 episodes reach the regrip.
`data/runs/sim3d/imitation/capture_v2/`

**What actually fixed it — two independent blockers, both needed:**

1. **RSI gradient starvation** (the main wall): uniform RSI over 33 frames gave
   the 3-frame swing only 9% of training gradient. The policy learned the settled
   final-stance phase (85% of the reference) and never trained on the swing.
   Fix: `--rsi-phase-max 1` caps all training starts to frames [0,1] (before RH
   releases at f02), forcing 100% of episodes through the swing. Phase-avg went
   0%→0% for 163k steps, then jumped 0%→70% over 40k steps as the swing clicked.

2. **Capture sphere gradient** (`mover_capture_coeff=0.3`): `mover_reach_coeff`
   is potential-based (net-zero when stationary), so once the tip parked short
   there was no gradient to pull it into the 0.08 m grip radius. The capture term
   fires `coeff×(1−gap/R)` per step inside the sphere.

**Lesson (structural, applies to all future single-move references):**
- Always set `--rsi-phase-max` to just before the mover releases. For a
  reference where the swing starts at frame F, use `--rsi-phase-max F`.
- Always pair `--mover-reach-coeff` with `--mover-capture-coeff`.
- The 3-frame swing proved physically achievable with PD servos (CMA-ES authored
  it directly in MuJoCo with the same servo model). The policy can execute it
  once it has enough gradient.

### Multi-move chain on v6_smooth FAILED — the reference is the problem (2026-06-18)

Six chain runs (`ladder_chain_capture_v1..v6` on `ref_ladder_v6_smooth`)
**stalled at stage 2 (the first LF foot step) at 0% for 1.5M+ steps**. Stage 1
(the RH hand move) trained to ~56% phase-avg fine; the moment the curriculum
advanced to the foot step, phase-avg collapsed 16%→8%→4%→0% and never recovered.
Frame-0 episodes die at frame ~45 = exactly the LF step.

**Root cause (proven, not guessed) — `ref_ladder_v6_smooth`'s foot moves are
servo-infeasible.** New tool `sim3d.probe_footstep` RSIs to the move's start,
releases the mover, and replays the reference's own joint targets open-loop. ALL
FOUR v6 foot moves fall short of the 0.08 m grip radius (LF stage-2: **19 cm
short**; RF: 10 cm; the other LF/RF: 15/8.6 cm), in BOTH per-frame and
constant-target modes. The body cannot reproduce its own recorded foot
trajectory. The 06-17 recipe (RSI cap + capture sphere) cannot fix this — it
supplies *gradient*, but the capture sphere (8 cm) is never even entered. The
recipe was proven on a HAND move (`capture_v2`); the chain's blocker is a FOOT
move on an infeasible reference — a different failure.

**Two layered causes, both confirmed:**
1. v6's stances are scrunched (feet too high; the 2026-06-14 finding) — the leg
   genuinely lacks ROM to extend the foot to the hold from that fold.
2. **Structural, affects ALL discover-authored refs:** `discover_move` records
   the *welded-equilibrium* `qpos`, where a mid-swing weld froze the leg
   half-extended. Replaying those poses sags ~22 cm even on a wall where the
   move IS reachable (verified on a fresh ladder foot move: the discovered
   *action* reaches 0.022 m, but replaying its recorded *poses* sags 22.8 cm).
   **Implication: pose-tracking imitation is the wrong frame for foot moves.**
   `free_mover_imitation` (exclude the mover's joints/tip from the pose term;
   reward reaching the real hold) is the correct mode — it ignores the
   un-trackable recorded foot pose. The remaining requirement is only that the
   foot move be *action-reachable* from the stance, which a feasible wall gives.

### Now: train foot moves on an action-feasible reference (IN PROGRESS)

Authored `ref_feasfoot_v2` (3 moves LH→RH→LF on `data/examples/ladder-v1.json`
via `python -m sim3d.discover --adaptive --wall data/examples/ladder-v1.json
--max-moves 6 --max-evals 350`). The LF foot move is **action-feasible** (CMA
lands it at gap 0.010–0.054 m; re-confirmed reachable from its start stance).
Note: net pelvis rise is −0.05 m (lateral, "not a climb") — it is a *foot-move
test bed*, not a route. The ladder wall is where moves are reachable; the
generated tight walls (`build_tight_wall`) were too far even for move 0.

Ran `imitation/feasfoot_chain_v1` (chain + chain-rsi-at-stage-start +
free-mover + reach 50/capture 0.3/grip-bonus 20, ent 0.001, 8 envs). **RESULT
(killed at 620k): even STAGE 1 stuck at 0.0% — a NEW blocker surfaced.** Mean
episode len 43 = exactly the stage-1 boundary (`move_starts[1]`), so episodes
*reach* the stage end but the grip never closes. Cause: `ref_feasfoot_v2`'s
first move (LH→h_003) lands at gap **0.073 m** — right at the 0.08 m grip
radius. **Marginal landings (gap ≳ 0.05) don't train** — the policy can't
reliably get the tip inside 0.08 to grip. (v6's stage-1 RH landed cleaner and
reached 56%, which is why v6 at least advanced to stage 2.)

So there are TWO independent reference-quality bars, and discovery clears
neither reliably:
1. **Action-feasibility** (the foot move must be reachable from the stance) —
   `sim3d.probe_footstep` checks this. v6 fails it; the ladder wall passes.
2. **Landing tightness** (every move must land at gap ≲ 0.04, not just <0.08) —
   marginal 0.07 landings are accepted by discovery's `landed` criterion but
   are untrainable. `discover_climb_adaptive` has `max_gap_m=0.04` but the
   `--adaptive` CLI path uses `discover_move`'s `<0.08` landed bar, so it ships
   marginal moves.

### (Superseded — see the 2026-06-19 section at the top.)

The mid-investigation "four blockers" framing from this session resolved as:
#1 move-selection → FIXED (`--auto-first-move`); #2 sideways-shuffle → was an
artifact of #1 (those seeds aborted at move-0); #3 grid-quantization → NOT the
issue (10 cm footholds still landed 9–12 cm short); #4 → the real lone wall is
FOOT REACH (open-loop foot swing falls ~9–12 cm short, balance-assist 250 N
insufficient). Full detail + next steps are in the top section.

### Landed this session (2026-06-17)
- **`mover_capture_coeff`** — dense within-grip-sphere bonus; fixes the
  "last 12 cm" gradient gap that potential-based `mover_reach_coeff` can't supply.
- **Swing-aware eval** (`sim3d/imitation.py`): `eval_frame0`'s off-reference cut
  no longer penalises the reference-released limb mid-swing.
- **RL audit fixes** (commit `ebaa834`): VideoRolloutCallback obs-normalisation;
  `new_high_grip_bonus` default 75 → 0; `--play-steps` (was hardcoded 30);
  energy penalty = `Sum((ctrl-seed)^2)` not absolute `Sum(ctrl^2)`.
- **`dense-from-stances` refs retired** (`ref_overhang_dense_v1..v4`): all SAG.
  Climbing refs remain CMA-ES (`ladder_v5` +0.50, `v6_smooth` +0.28, 11 moves).

---

## Where we are (2026-06-11)

**The body + grip physics overhaul landed.** Everything below is measured,
not guessed (probes: `sim3d.probe_foot_reach`, `sim3d.probe_grip_strength`):

- **Zero-strain hold anchoring** — the slip model was reading kN-scale
  constraint-solver stretch (7× body weight on settled stances) as grip
  load; `attach_limb(anchor="tip")` + seed re-anchoring removed it. Grip
  multipliers tightened on probe evidence: hand 2.5→2.0, foot 3.0→1.5.
- **Foot moves unlocked** — the blocker was the hip_rot ±40° limit (NOT the
  hip_flex axis, which is anatomically correct): the shin couldn't orient
  wall-ward at high flexion. At ±60° (+ abduct −30..85°) CMA-ES discovered
  its first foot move (LF +0.22 m, all anchors kept). Discovery now walks
  the full LH→RH→LF→RF cycle, including in `--continue-from`.
- **Spine lateral + twist DOF added** (video-retargeting prerequisite).
  23 actuated joints; obs (131,), action (27,). All saved references
  migrated in place (`python -m sim3d.reference --migrate`); the 6-move
  reference is 100% RSI-holdable on the new body. **Old policy checkpoints
  (incl. v3) are stale — first retraining run on the new body is pending.**
- **CMA-ES hardening** — per-DOF search bounds spanning the full joint range
  (the flat ±1.6 rad box couldn't even express a high-step), restarts with
  sigma escalation, cycle-walking when a limb has no reachable hold, and a
  `--stitch` CLI replacing manual reference stitching.
- **Frame-0 success is the headline metric** — `python -m sim3d.imitation
  --eval --model … --ref …`, plus a periodic in-training frame-0 eval.
  Phase-averaged success is diagnostic only (v2: 70% phase-avg, 0/4 real).
- Historical context: v3 (`ref6moves_v3_hardened`, old body) reached 52.6%
  from frame 0 on `ref_cma9_6moves_stitched.npz` (148 frames, 6 hand moves,
  pelvis 3.29→3.49 m).

---

## Near-term: train on a new-physics reference

### N0 — RESULT (2026-06-11): the old reference is untrackable; retired

Two full runs on the migrated 6-move reference (v1 uniform RSI 2M steps →
v2 hardened, RSI cap annealed 148→0 over 2M): **frame-0 success 0/40; at
cap 0 even phase-avg fell to 0%** while r_imit held ~0.6 — the policy
tracks perfectly up to move 2's load transfer, where physics refuses.
Diagnosis (open-loop replay + multiplier sweep): the transfer builds foot
forces that scale to break ANY cap (co-contraction race), because the move
was authored under foot multiplier 3.0. **Lesson, now structural: a
reference is only trainable under the physics it was authored under.**
Discovery vets every move against live caps, so its references are feasible
by construction. (Historical 52.6% v3 was real, but on physics whose grip
forces were 7×-bodyweight artifacts. Note: `--rsi-anneal` is per-env steps —
divide by n_envs, or the cap never lands at 0.)

### N1 — RETRACTED, THEN FIXED (2026-06-11): the wiggle-in-place degeneracy

The first "100% frame-0" result (`ref3moves_newphys_v1` on
`ref_seed9_4limb.npz`) was REAL TRACKING OF A FAKE CLIMB — visual review
caught it: the reference was one 20 cm hand move plus the feet re-gripping
the holds they already stood on; **net pelvis movement −3 cm**. Greedy
adaptive discovery preferred trivial re-grips: currently-held holds weren't
excluded from candidates, and zero-strain anchors put tips a few cm below
hold centres so "own hold" passed the 2 cm upward filter. Three fixes, all
landed:
1. `_reachable_holds` excludes every currently-held hold and requires
   ≥0.08 m height gain per move;
2. `ImitationEnv` "completed" now also requires the END GRIPS to match the
   reference's final holds (window-survival alone can be gamed);
3. discovery prints `net pelvis rise` and brands `< 0.10 m` as NOT A CLIMB.
The training-loop mechanics (0→100% in 300k steps under honest physics) are
validated; the climbing content was not. Lesson for every future metric:
**watch the video before believing the number.**

### N1b. Longer real reference — IN PROGRESS

Stand-up machinery landed earlier (knees in the arm-move search, hand probe
radius 0.50→0.80 m so post-foot-step crouches can target what leg extension
reaches, cycle-walking, restarts). Mid-wall `seed_pose` starts don't work
(stances collapse on release — only bottom-up chained discovery yields
release-vetted stances). Running now: bottom-up seeds 9 + 98 with the
anti-degeneracy fixes. Aim ≥8 moves, ≥2 REAL foot moves, net rise ≥1 m.

### N2. Train on the ladder reference — chain curriculum (IN PROGRESS)

Training-loop lessons from the 11-move attempts (all structural, all fixed):
- **The imitation mesa**: on long references with quasi-static stretches,
  "hold still and sag slowly" outscores the brief critical transitions —
  full-length RSI-annealed episodes trained sagging (one move then a
  0.44 m sag read as mean len 120). Cap→0 annealing also makes the policy
  FORGET the upper moves (train with mixed sampling if annealing at all).
- **References must be smooth**: CMA-authored stand-ups without a velocity
  penalty are open-loop jerks; every run died at EXACTLY the frame a
  recorded stand-up ends. Discovery now penalises mean qvel² (0.4×) in
  moves and stands; horizons lengthened (move 40, stand 32).
- **Chain curriculum** (`--chain`): every episode starts at frame 0;
  stage k ends at move k's boundary; success = the reference's stance
  there is actually HELD (grip match — un-inflatable); ≥70% ground-start
  success unlocks stage k+1. Half the episodes RSI mid-window (a frame-0
  policy parks a foot 27 cm short forever; mid-move starts teach the
  swing) but only ground starts count toward advancement.
- **Verification discipline**: the only believable evidence is a
  FIXED-CAMERA video checked against pelvis-z numbers. Tracking-camera
  stills lie (the camera follows the body — wall motion reads as climbing).

Current run: `ladder_v6_chain2` on `ref_ladder_v6_smooth.npz` (10 smooth
moves, +0.28 m net). Stage 1 (first move from the ground, grips verified)
trains to ≥70% quickly; stage 2 (the foot step) went 0% → ~25% with
mid-window mixing. Full-route frame-0 episodes reach frame 85/552.
Headline metric unchanged: full-reference frame-0 success with end-grip
match.

---

## Medium-term: primitive library (Path B)

### M1. Author a primitive library

Once the single-wall imitation controller works (≥60% success), author
short single-move primitives covering the basic technique set:
- Left-hand reach (up-left, up-right, straight up) × 3 variants each
- Right-hand reach × 3 variants
- Left-foot step (unblocked 2026-06-11 — hips fixed)
- Right-foot step

Each primitive is a short ~30-frame clip (one move only). Author using
the existing CMA-ES pipeline — single moves are much more tractable than
multi-move chains. Target: ~20 clips total.

### M2. Multi-reference ImitationEnv

Extend `ImitationEnv` to accept a list of references, sampling a new one
each episode reset. This is the bridging step between single-wall imitation
and the full motion prior. Training on 20 primitives with uniform RSI
across all of them gives the policy a broad technique vocabulary.

### M3. Policy-controlled grips

Currently grips follow the reference contact schedule — the policy only
controls joint targets. Add grip intent output and a grip-matching reward
term. Prerequisite: M2 working.

---

## Long-term: video-based motion prior (Path C / AMP)

The goal: one policy that climbs any wall at any grade without needing a
wall-specific reference. Data source: existing climbing video. See
CLAUDE.md §"The video pipeline" for the full architectural rationale.

### V1. Pose extraction tooling

**2D feasibility CONFIRMED (2026-06-11)** on the first real video
(`data/video/moonboard/spike1/`, 9 clips V3–V7, static head-on shot):
YOLO11s-pose detects the climber in ≥99% of frames at every grade tested,
wrist confidence 0.90–0.95, ankles 0.85–0.92, drop-knee poses tracked
cleanly. See the dataset README for the full table.

**Remaining:** SMPL (3D) extraction — set up 4D-Humans or WHAM. Needs a
GPU-friendly environment (Linux/CUDA or Colab; macOS install is painful).
Verify on spike1 clips: no flipped limbs, plausible joint angles.

```bash
# 4D-Humans (recommended — best occlusion handling)
pip install git+https://github.com/shubham-goel/4D-Humans
python demo.py --video data/video/moonboard/spike1/clip01_v3a.mp4 --out smpl_output/
```

### V2. MoonBoard video dataset

**Started (2026-06-11):** `data/video/moonboard/spike1/` — 9 clips
(V3×2, V4×2, V5×2, V6×2, V7) from one static-camera video; problem names
visible in the overlay (e.g. "The Warm up Problem" / RussK, "Black
Muffler" / Koala Climbing). TODO in its README: resolve overlay names to
hold sets via `data/moonboard/`, note board version + angle.

Target: 50–100 MoonBoard send videos spanning V3–V8. YouTube and
Instagram are the sources. Organise as:
```
data/video/moonboard/<dataset>/<clip_n>.mp4   (gitignored)
data/video/moonboard/<dataset>/README.md      (tracked: timestamps, problems, QA)
data/video/moonboard/<dataset>/holds.json     ← active hold coords
```

### V3. Clip segmentation + contact labelling

For each video:
1. Run pose estimation → SMPL per frame
2. Detect move transitions: velocity peaks in wrist positions
3. Cut into individual move clips (2–5 s each)
4. Contact labelling: project known hold 3D positions into camera frame,
   match to estimated wrist/ankle positions within 15cm → gripping flag

MoonBoard holds are at exact known 3D positions (20cm grid). No camera
calibration needed if you fix the wall plane — the board is always the
same size. Use the 4 corner bolts as calibration anchors.

### V4. SMPL → 27-DOF retargeting (the hard step)

This is the main engineering investment. SMPL has 72 DOF; our body has
27. Proportions differ (real human arm span ≠ our body's).

Approach:
1. Build a joint correspondence map: SMPL joint → nearest our-body joint
2. Per frame: run MuJoCo IK to find our-body joint angles that minimise
   end-effector distance to SMPL wrist/ankle positions
3. Physics step to resolve penetrations and joint limit violations
4. Reject frames where end-effector error > 5cm (pose estimation noise)

Start with upper body only (shoulders, elbows, wrists) — hands are the
primary contact points and easiest to match. Add lower body once upper
body retargeting is validated.

### V5. Quality filtering + clip library

Filter retargeted clips:
- Reject if any joint exceeds its limit by > 5°
- Reject if body penetrates wall geometry
- Reject if retargeting error > 5cm on any active end-effector
- Reject clips shorter than 10 frames

Target: 200+ clean clips from ~50 videos. Store as `.npz` in the same
format as existing references (qpos, qvel, eef, com per frame).

### V6. AMP discriminator

Add adversarial training to the PPO loop. The discriminator takes a
`(state_t, state_{t+1})` transition pair and outputs real/fake probability.
Real = from the clip library. Fake = from the current policy rollout.

The style reward: `r_style = -log(1 - D(s, s'))` added to the task reward.
The discriminator is updated every PPO iteration with a gradient penalty
for stability (WGAN-GP or R1).

Architecture: MLP matching the policy network size. Input dim = 2 × obs_dim.

### V7. Multi-wall RL with motion prior

Train the RL policy with both rewards:
- `r_task`: height progress + hold match (from existing `config.py`)
- `r_style`: discriminator score (scale to match r_task magnitude)

Train on a curriculum of generated walls (not the MoonBoard reference wall)
so the policy generalises. The prior handles technique; RL handles
wall-specific strategy.

### V8. MoonBoard evaluation + expansion

Eval on 20 held-out MoonBoard problems (problems not seen in training).
Measure: success rate, technique quality (visual), grade range covered.

If technique gaps are visible (e.g. poor slab):
- Collect gym footage / IFSC competition footage for those technique types
- Fine-tune discriminator on new clips
- Short RL adaptation run (~200k steps) — no full retraining needed

---

## Things you should NOT do

- Switch to `discrete-move` as the default. Continuous control is the goal.
- Replace the custom climber with the stock Gymnasium humanoid.
- Add DOF to the body before it climbs reliably with the current 21 DOF.
- Add `KNOWN_ISSUES.md`, `ARCH_REVIEW.md`, or per-phase trackers. This
  file is the only roadmap. Prune it; don't append to it.

---

## Tracking

When you finish an item, **delete it from this file**, then add one or two
sentences to `CLAUDE.md` if it changed the permanent architectural
contract. Cumulative changelog lives in git; this file is for "what's next."
