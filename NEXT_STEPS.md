# Next Steps & Known Issues

> **Freshness warning.** This is a living roadmap, not a snapshot. Update
> or delete entries as you act on them — a stale roadmap is worse than no
> roadmap. Prune aggressively; don't let this drift into an audit doc that
> rots while the code moves.
>
> Last reviewed: 2026-06-09.

---

## Where we are (2026-06-09)

**The imitation + CMA-ES loop is working end-to-end.**

Path taken:
- Phase 0 blockers (episode length, seed pose, grip semantics, log_std_init) — DONE.
- Potential-based reward + staged curriculum — DONE. PPO learns reach-one stably.
- Imitation + RSI infrastructure — DONE. Single-move reference trains 0→88%.
- CMA-ES discovery — DONE. `sim3d/discover.py` chains moves by probing nearest
  reachable unvisited holds. `--continue-from` starts continuation from the
  real physical end-state (not seed-pose approximation).
  Foot moves blocked by hip_flex axis; all discovery is hand-only.
- **6-move reference authored**: `ref_cma9_6moves_stitched.npz`, 148 frames,
  holds h_033→h_034→h_037→h_038→h_039→h_040, pelvis 3.29→3.49m.
- **R_min bug found and fixed**: v1 training had R_min_start=0.75 but the
  reference's worst transition gives r_imit=0.750 — barely above the threshold.
  As RSI hardened (more moves per episode), success collapsed from 13.7% peak
  → 0%. Root cause: any imperfection during a reach transition pushed r_imit
  below R_min and terminated. Fix: `--r-min-start 0.3` exposes `r_min_start`
  and `r_min_end` as CLI args; now set to 0.30 (flat).
- **Wrong-wall bug found and fixed**: `_load_wall_for_ref` was rebuilding the
  wall from the generator seed (54 holds, wrong positions) instead of the saved
  `.wall.json` (58 holds). Only 1/10 reference holds matched — catastrophic RSI
  instability. Fix: `_load_wall_for_ref` now auto-detects sibling `.wall.json`.
- **Periodic checkpointing added**: `CheckpointCallback` saves every 200k global
  steps into `checkpoints/` — runs can be warm-started if interrupted.
- **v2 training running (2026-06-09)**:
  `data/runs/sim3d/imitation/ref6moves_v2_rmin03/`
  - 2M steps, 8 envs, `--rsi-anneal 400000 --r-min-start 0.3 --r-min-end 0.3`.
  - At 811k steps: success 39–42% (tripled from 12% start), r_imit 0.665.
    Clean monotonic rise — no collapse. r_imit stabilised after early dip
    (null policy accidentally tracked well; exploration dip is normal).

---

## Immediate: evaluate v2 at 2M steps

### I1. Record a rollout video

When `ref6moves_v2_rmin03/model.zip` appears:
```bash
python -m sim3d.imitation --record ref6moves_v2_rollout.mp4 \
  --ref data/runs/sim3d/imitation/ref_cma9_6moves_stitched.npz \
  --model data/runs/sim3d/imitation/ref6moves_v2_rmin03/model.zip \
  --vecnorm data/runs/sim3d/imitation/ref6moves_v2_rmin03/vecnormalize.pkl
```
Does the policy visually execute the 6-move sequence? Are late-reference RSI
starts (easy) vs early-reference starts (hard) both succeeding?

### I2. Harden to full-chain if success ≥ 60% at 2M steps

If success plateaus below 60%, the uniform RSI is masking that early-start
episodes are undertrained. Add RSI hardening in a third run:
```bash
python -m sim3d.imitation --train \
  --ref data/runs/sim3d/imitation/ref_cma9_6moves_stitched.npz \
  --steps 2_000_000 --n-envs 8 \
  --run-id imitation/ref6moves_v3_hardened \
  --load data/runs/sim3d/imitation/ref6moves_v2_rmin03 \
  --rsi-phase-max 0 --rsi-anneal 250_000 \
  --r-min-start 0.3 --r-min-end 0.3
```

---

## Near-term: extend the reference to the top

### N1. Continue the reference past h_039/h_040

The ceiling was h_043 at z=3.7m — just beyond arm reach with feet still
at h_035/h_036 (z=2.5m). Two options:

**Option A — Foot move first** (preferred if hip_flex is fixable):
The current `hip_flex` joint axis `[-1,0,0]` rotates the leg forward, not
upward, capping foot z at ~0.87m. A foot hold at z≈2.8m would require
either: (a) raising `hip_flex` axis to something like `[0,-1,0]` (rotates
leg upward in the XZ plane), or (b) using hip_abduct + hip_rot together to
lift the foot. Check `builder.py` joint definitions before attempting.

**Option B — Different wall seed** (quickest):
Run adaptive discovery on a different seed that generates a wall where the
holds above h_039/h_040 are closer (z=3.5m, not 3.7m). Seeds with
consecutive moves 5-8 in gap≥0.45m list (from the seed scan): seeds 90,
98, 101, 118, 145, 146, 151.

**Option C — Continue from ref6moves with a new adaptive run**:
`--continue-from ref_cma9_6moves_stitched.npz` after training finishes,
targeting a wall with holds at reachable z from h_039/h_040. Note: the
continuation needs the body's FEET to advance (h_035/h_036 are at z=2.5m
and have been there since the beginning); pure hand moves can't gain height
indefinitely.

### N2. Get ≥8 moves on a single generated wall (Option B above)

Seeds 90, 98, 101, 118, 145, 146, 151 all have ≥9 gap≥0.45m moves in the
feasibility scan. Run:
```bash
python -m sim3d.discover --adaptive --seed <N> --max-evals 600 --max-moves 10 \
  --out data/runs/sim3d/imitation/ref_seed<N>_adaptive10.npz
```
Pick the seed that produces the most moves (aim for ≥8). Then train
imitation on that reference with the same schedule as above.

---

## Medium-term: primitive library (Path B)

### M1. Author a primitive library

Once the single-wall imitation controller works (≥60% success), author
short single-move primitives covering the basic technique set:
- Left-hand reach (up-left, up-right, straight up) × 3 variants each
- Right-hand reach × 3 variants
- Left-foot step (once hip_flex axis is fixed — see N1)
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

Set up 4D-Humans or WHAM locally. Verify on a short MoonBoard send video —
confirm SMPL body parameters look plausible (no flipped limbs, reasonable
joint angles). Tools are pre-trained and public; this is mostly environment
setup.

```bash
# 4D-Humans (recommended — best occlusion handling)
pip install git+https://github.com/shubham-goel/4D-Humans
python demo.py --video my_moonboard_clip.mp4 --out smpl_output/
```

### V2. MoonBoard video dataset

Collect 50–100 MoonBoard send videos spanning V3–V8. YouTube and
Instagram are the sources. Label each video with its problem ID (which
holds are active) — this is knowable from the MoonBoard problem database
in `data/moonboard/`. Organise as:
```
data/video/moonboard/<problem_id>/<clip_n>.mp4
data/video/moonboard/<problem_id>/holds.json  ← active hold coords
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
