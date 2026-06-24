# Next Steps & Known Issues

> **Freshness warning.** This is a living roadmap, not a snapshot. Update
> or delete entries as you act on them — a stale roadmap is worse than no
> roadmap. Prune aggressively; don't let this drift into an audit doc that
> rots while the code moves.
>
> Last reviewed: 2026-06-22.

---

## Where we are (2026-06-22) — multi-move CLIMBING works; foot moves SOLVED; move-ORDER is the chaining key

The agent now executes **multi-move climbs** under continuous control. The
months-long foot-move blocker is fully broken, and the path from single moves to
a real route is a known recipe, not an open question.

**What works (all on `footstep-fine5-v1`, stance-keyframe refs, `--w-task 1` milestone):**
- **Single hand move** 100%; **foot high-steps** +10 cm 16/16, +15 cm 20/20.
- **2-move continuous climb** (hand→foot, bottom-up, no reset): **20/20** —
  `seqchain_v1` (video `seqchain_v1/sequential_climb.mp4`).
- **4-move per-move skills** (RH,LH hands + LF,RF feet): all **16/16** (`handsfirst_ms`).
- **4-move sequential composition** (hands-first): IN PROGRESS — `handsfirst_seq2`
  (detached: `nohup caffeinate … &`).

**The three findings that cracked it:**
1. **Foot moves are GOAL-REACHING, not imitation.** No open-loop foot trajectory is
   both gradual AND monotonic (fast CMA kick = 1-frame teleport, untrackable; any
   slow ramp detours ~0.4 m), so trajectory imitation can't author a trackable foot
   swing. Fix: `--w-task 1` (drop the pose attractor) + dense ABSOLUTE reach reward
   (`--mover-reach-abs-coeff`, full-radius pull) lets the policy discover its own
   closed-loop swing. The pose attractor at w_task=0 was a STILLNESS VALLEY that
   froze the foot at the hang for 8 runs.
2. **Composition needs on-policy landings.** Per-move skills trained from fixed
   authored stances don't chain (foot from the policy's REAL hand-landing: 0/8 vs
   16/16 from the authored stance). Fix: `--sequential-chain` — on each grip, advance
   the target to the next stance WITHOUT resetting the body, so move k+1 trains from
   move k's actual landing.
3. **MOVE ORDER is a stability constraint.** A HAND move launched from a POST-FOOT
   (asymmetric high/low feet) stance is a barn-door trap: the policy correctly refuses
   to release the loaded hand → frozen 0.000 m for 1.2M+ steps, even with dedicated
   goal-reaching. The SAME move authors + trains 16/16 from a balanced post-HAND
   stance. RULE: sequence so hand moves launch from balanced (both-feet-down) stances
   — e.g. both hands, then both feet. Never hand-from-post-foot.

**Recipe (chain N moves):** author an N-stance ref (`discover_move` per move from the
previous settled stance — hand moves cold + restarts, foot moves warm-start the leg
config hip_flex 140°/abduct 85°/knee 112°) → milestone `--w-task 1` + dense reach for
per-move skills → `--sequential-chain` warm-start to compose. macOS long runs: wrap in
`caffeinate` + `nohup … & disown` (sleep AND harness-teardown safe; see CLAUDE.md).

### Slow moves — DEFERRED (2026-06-24): velocity penalty does NOT work; needs a structural fix
Tried hard: warm-start snap policy + --vel-penalty-coeff (sum qvel², calibrated 0.2-0.25 vs snap peak ~90) over 6 runs → still 2-frame/0.03s snaps (entrenched policy eats the penalty, never explores slow control). From SCRATCH with the penalty (slow_scratch, coeff 0.15) → 0% (penalty suppresses the exploration needed to learn the move at all). CONCLUSION: a soft velocity reward can't produce slow moves here — too weak to shift a converged policy, too strong to learn from scratch. Real fix = HARD action-rate limit (clamp |action_t - action_{t-1}| in the env wrapper, or cap joint velocity) so snapping is physically impossible; OR defer to the torque/muscle phase (servo snaps won't exist there). Deferred for now.

### 3-MOVE climb COMPOSES (2026-06-24): 20/20, the composition wall is at depth 4
chain3_seq = hands-first 3-move sequential (hang→RH:h_012→LH:h_013→LF:f_047), warm-started from handsfirst_ms (per-move 16/16). Full-climb completion rose cleanly 20%→80% over 1.5M; deterministic sequential eval (start=bottom, no reset) = **20/20 full 3-move climbs**, net com rise +0.05 m, real (video chain3_seq/climb3.mp4). So: 2-move=1 composition (20/20), 3-move=2 compositions (20/20), 4-move=3 compositions (stalled ~30%). The wall is the 3rd composition. KEY PATH: build the 4-move INCREMENTALLY — warm-start the 4-move sequential from the working 3-move policy (chain3_seq) so moves 1-3 stay reliable and move 4 (RF) gets consistent gradient (it was starved before, only reached when 1-3 happened to all fire). Net rise stays small (~0.05 m) — still needs the com-rise authoring work.

### NEXT (priority order)

1. **STYLE / naturalness — the climbing is JANKY (leans back, barn-doors).** Direct
   consequence of `w_task=1`: there is NO pose/posture term during transitions, so
   reaching the hold by any contortion is optimal. Add LIGHT shaping that does NOT
   recreate the stillness valley:
   - **Anti-lean / anti-barn-door penalty** — penalize pelvis +Y (leaning off the
     wall). `discover_move`'s cost already has a `lean_back` term; port a small version
     into the milestone/sequential reward.
   - **com-RISE reward** — reward pelvis-z gain so the body pulls UP over its holds
     instead of leaning to reach (fixes style AND net-rise, see #2).
   - Optionally a small pose-attractor weight (`w_task` 0.7–0.9) applied ONLY after
     moves are learned, to tidy posture without re-freezing. Tune so success stays high.
2. **NET UPWARD RISE.** Chains raise the contacts but the pelvis barely rises
   (+0.04 m hands-first, −0.006 m original) — the body compresses/leans rather than
   ascending. `discover_move`'s cost only penalizes com-DROP, never rewards rise; add a
   com-rise term there (and/or in training). A real climb gains ~one body-segment per cycle.
3. **GENERALIZE move-order into a stance SEQUENCER.** Chains are currently hand-ordered.
   Auto-plan a stance sequence on any wall that obeys the balance rule (hands from
   balanced stances; weight-shift before a hand reach). This is the bridge from one
   hand-authored wall to arbitrary routes. (Note: the LAST limb always launches from
   the most committed stance — RF→f_048 +15 wouldn't author from the 3-limbs-up pose;
   f_046 +10 did. The sequencer should ease the final move or insert a weight-shift.)
4. **REAL MoonBoard route.** Run the full pipeline (sequencer → author → milestone →
   sequential) on an actual MoonBoard problem; measure top-out. First real-route test.
5. **Robustness:** longer chains (6+ moves); re-tighten grip caps toward ~1.2× now that
   policies weight-shift (CLAUDE.md §Grip strength); insert explicit weight-shift stances
   where balance is marginal.

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
