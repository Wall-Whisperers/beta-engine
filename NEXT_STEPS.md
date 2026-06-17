# Next Steps & Known Issues

> **Freshness warning.** This is a living roadmap, not a snapshot. Update
> or delete entries as you act on them — a stale roadmap is worse than no
> roadmap. Prune aggressively; don't let this drift into an audit doc that
> rots while the code moves.
>
> Last reviewed: 2026-06-17.

---

## Where we are (2026-06-17) — the single-move reach gap is the blocker

**Diagnosis (measured, not guessed).** The imitation policy cannot close a
single move's reach deterministically. On a clean one-hand move
(`ref_adaptive_s14`, RH h_002 -> h_048) training reaches ~80% phase-avg but
**0% frame-0**: the deterministic hand parks **0.197 m** from the target hold
(grip radius 0.08 m) and never grips; only exploration noise closes the last
~12 cm (hence ~40% stochastic / 0% deterministic). Ruled out, each with a probe:
- NOT balance/physics — the post-release 3-grip stance holds 60 steps under
  zero action, pelvis steady;
- NOT frame-0 undertraining — RSI annealed to cap 0 (~50% frame-0 starts) is
  still 0%;
- NOT the off-reference R_min cut — removing it (R_min=0) leaves the 0.197 m
  park and 0% unchanged.

This matches the long-standing `imitation.py` note ("deterministic gap 0.23 m
while stochastic min 0.09 m"). The body is capable and stable; the wall is
**reach precision / grip acquisition at the single-move level** — upstream of
chaining AND of any motion-data / AMP work.

**Implication: do NOT invest in the video/footage pipeline (Path C) yet.** It
feeds styles to a controller that can't drive a limb the last 12 cm into a
hold. Footage becomes worth collecting only after single-move deterministic
frame-0 execution works.

### Next (data-free, cost order)
1. **Close the determinism gap** — lower `ent_coef` so the deterministic policy
   approaches the occasionally-succeeding stochastic one. Cheapest test
   (~5 min/run); says how much of the 0% is just det-vs-stochastic.
2. **Reach reward's last 12 cm** — LANDED (2026-06-17): `mover_capture_coeff`
   added to `ImitationConfig`. Fires `coeff×(1−gap/R)` per step while the mover
   tip is inside the grip radius (0.08 m) but not yet gripped. Potential-based
   `mover_reach_coeff` is net-zero once the tip is stationary; this term gives a
   gradient toward the hold centre throughout the capture sphere. Expose via
   `--mover-capture-coeff`. Try 0.2–0.5 (same scale as r_imit).
   **Recommended first run**: `--free-mover-imitation --mover-reach-coeff 50
   --mover-capture-coeff 0.3 --mover-grip-bonus 20 --ent-coef 0.001`
   (lower ent_coef + tip-capture + grip bonus together; addresses all three
   determinism-gap causes in one run on `ref_adaptive_s14`).
3. **Reference reproducibility** — refs are authored by the balance-assisted
   Cartesian reach controller but must be reproduced by the PD-servo policy;
   keep authoring within what the policy can execute or the gap recurs.

### Landed this pass
- **Swing-aware eval** (`sim3d/imitation.py`): `eval_frame0`'s off-reference cut
  no longer penalises the reference-released limb mid-swing — `ImitationEnv.step`
  frees the ref-ungripped limb when `free_mover_imitation` is set and there is no
  chain mover. Also makes the free-mover knobs function outside chain mode. (Did
  NOT move frame-0 — the reach gap, not the cut, is the wall.)
- **RL audit fixes** (commit `ebaa834`): VideoRolloutCallback obs-normalisation;
  `new_high_grip_bonus` default 75 -> 0; `--play-steps` (was hardcoded 30);
  energy penalty = `Sum((ctrl-seed)^2)` not absolute `Sum(ctrl^2)`.
- **`dense-from-stances` refs retired** (`ref_overhang_dense_v1..v4`): all SAG
  (net pelvis rise -0.65 .. +0.05) — not climbs. Climbing refs remain CMA-ES
  (`ladder_v5` +0.50, `v6_smooth` +0.28, 11 moves) and stance-keyframe.

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
