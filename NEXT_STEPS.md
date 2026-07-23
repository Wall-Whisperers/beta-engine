# Next Steps & Known Issues

> **Freshness warning.** This is a living roadmap, not a snapshot. Update
> or delete entries as you act on them — a stale roadmap is worse than no
> roadmap. Prune aggressively; don't let this drift into an audit doc that
> rots while the code moves.
>
> Last reviewed: 2026-06-22.

---

## Where we are (2026-07-07) — RELIABLE 4-MOVE CONTINUOUS CLIMB: 20/20 deterministic ★

A full bottom-to-top **4-move climb** (RH→LH→LF→RF) now runs **20/20 deterministic**
(`eval_frame0`, un-farmable end-grip match). Checkpoint:
`data/runs/sim3d/imitation/wholebody_capture/checkpoints/model_1600000_steps.zip`
(+ matching vecnormalize). Reference `ref_chain_wholebody.npz`.

**The recipe that cracked the 4-move wall (each piece was necessary):**
1. **Centered LF landing** — `discover_move(stance_center_coeff=…)` pulls the foot
   landing's com in toward the wall (com_y 0.408→0.199) so RF launches from a
   balanced stance, not a barn-door. CLI `--foot-stance-center`.
2. **Whole-body foot moves** — `_search_dofs(whole_body=True)` recruits BOTH arms +
   the standing leg into the CMA foot-move search (was: swinging leg + spine +
   opposite hip only). A foot move is a whole-body action; this got RF gripping
   **deep (gap 0.032)** instead of at the 0.080 grip-radius edge.
3. **`--sequential-chain`** — composes moves from the previous move's real landing.
4. **DAgger** — collect the policy's real post-LF landings (`--collect-landings
   --milestone-focus-stance 3`), re-train RF from them (`--landing-bank`).
5. **`--mover-capture-radius 0.15`** — the deterministic RF was plateauing at
   0.126 m, OUTSIDE the 0.08 m capture sphere (a reward dead zone → no final pull).
   Widening the capture sphere to 0.15 m broke the plateau → RF gap 0.054 m, grips.

**Confirmed:** `wholebody_capture` FINAL CHAIN eval = **40/40 (100%)**; video
`data/runs/sim3d/imitation/wholebody_4move_climb.mp4` (real 4-move climb, verified
by watching frames).

**Video renderer FIXED** — `record_video` now takes `base_icfg` and, for a
stance-milestone policy, renders the full chain with eval_frame0-style overrides
(was building a bare config → "4 frames" bug). `--record` now works for chains.

**NET-HEIGHT (in progress).** Diagnosed: the 4 moves put holds ~0.2 m higher but
the reference only rose +0.038 m — the body REACHED up (extended limbs) instead of
RIDING up. `discover_move` optimizes tip-gap with com_rise_reward OFF. FIX (not
com_rise_reward — the balance assist pins the pelvis, the June net-rise trap):
append a **stand-up stage** (`discover_stand`, no assist) after RF that pushes com
up over the now-higher feet. Measured +0.105 m gain (all 4 anchors kept) →
`ref_chain_wholebody_rise.npz` (6 stages, **net com rise +0.141 m** vs +0.038).
`wholebody_rise` FAILED (0/40, reaches stage 5 but never completes the stand-up)
— the posture-stance stall, as predicted ([[weight-shift-stance-two-lessons]]):
a "raise+hold" stance with no reach goal is a weak imitation target. Root cause:
no dense gradient pulling the com UP (only a sparse completion bonus at tol) — the
same dead-zone that froze RF before mover_capture_radius.
`wholebody_rise2` (`--posture-rise-coeff 0.5`, 4M steps) ALSO FAILED 0/40 —
furthest stance 5.0 every episode (moves 1–4 compose fine), r_imit flat ~0.22,
com stalled ~+0.054 m of +0.141 m. The 0.5-coeff absolute pull was too weak a
bolt-on vs the attractor, the third instance of the same dead-zone pattern.

**FIX (2026-07-09, structural): GOAL-POTENTIAL milestone reward** — the default
milestone reward in `sim3d/imitation.py` is redesigned (full design note in the
`ImitationConfig` comment block, §goal_k): every stance gets an explicit GOAL
POINT (hold center for grip moves via the mover tip; the reference stance com in
3D for posture/stand-up stances — "hold this pose" becomes a reach in com-space);
a dense SIGNED POTENTIAL `goal_k·(d_prev−d_cur)` (k=100) runs from spawn to the
success tolerance with no dead zone; the pose attractor is FROZEN (latched at its
band-entry value) inside `goal_band` (0.15 m) so its saturated gradient can't
compete near the goal. Telescopes ⇒ un-farmable by oscillation; hovering at the
hold earns nothing (removes the old capture-income farm). SUPERSEDES
`--mover-reach*`, `--mover-capture*`, `--posture-rise*` (ignored with a warning
while `--goal-k` > 0; `--goal-k 0` restores legacy). Success criteria unchanged.
No regression: `wholebody_capture` 4-move chain eval through the new code =
**20/20**. `eval_frame0` now also reports `net com rise`.

**Validation trail (2026-07-09, all judged by chain eval / is_success):**
- `goalpot_rise1` (goal potential, warm-start wholebody_rise2): com now reaches
  0.065 of the goal (vs ~0.09+ stall before) but SWINGS THROUGH and drifts out —
  the hard attractor freeze inside the band removed the settling signal (pose was
  UNSATURATED there, r_imit ~0.22; saturation is a function of pose error, not
  goal distance). FIX: band latch is now a FLOOR — att = max(entry, live).
- `goalpot_rise2` (floor fix): still no holds; full-chain training perturbs
  moves 1–4 (known long-horizon fragility).
- `goalpot_rise3` (proven DAgger recipe: 200 real stance-4 landings,
  `--milestone-focus-stance 4 --milestone-focus-frac 0.6`): ~1000 FOCUSED
  stand-up attempts, ZERO successes ⇒ not exploration starvation. ROOT CAUSE
  probe: RSI'd AT the authored stance-5 keyframe with all grips welded, zero
  action, the com passively exits the 0.06 tol in 3 steps (sag +0.159 in y —
  the recorded wall-hug com relied on weld-strain authoring forces; "references
  are only trainable under the physics they were authored under", again).
  The +0.141 net-rise target was never play-feasible.
- FIX (artifact, with lineage): re-settled the stand-up keyframe under
  play-time dynamics → `..._rise_v2.npz`, then again from a REAL policy landing
  (weld anchors at real touch points shift the equilibrium ~10 cm in x) →
  `..._rise_v3.npz` (frame 244 = the settled equilibrium; parent recorded).
  Verified holdable BY CONSTRUCTION: zero-action gap 0.04–0.055 indefinitely,
  all grips kept, **net rise +0.092–0.094** (the honest ceiling of this stance).
- Criterion: `--posture-com-tol 0.06` is INSIDE the settle-transient noise floor
  (±2–3 cm wobble around equilibrium — even zero-action fails 4 consecutive
  steps); 0.08 is the measured-defensible tolerance.
- `goalpot_rise4–8` (v2/v3, DAgger bank + corridor-augmented bank, ent/budget
  tuning): net rise reliably +0.08–0.09, det stance-5 min_d dips IN-tol
  (0.061)… for exactly 1 step. Stochastic completions 3–7%, never
  consolidated. **These runs descend from wholebody_rise2 (a 0/40 parent) —
  bad-warm-start lineage, judged invalid.**
- `goalpot_cap1` (THE decisive run, per gate): v3 + corridor bank + velocity-
  augmented potential (`--goal-vel-lambda 0.15`, makes the potential bottom
  out only at the goal AT REST), warm-start `wholebody_capture@1.6M` (the
  clean 20/20 parent; probe: composes 1–4 on v3 unchanged, untrained stance-5
  already dips to 0.076). Stochastic success opened 8–9.5% and held 5–10%,
  but det chain stayed 0/12 through the 500k gate (stand-up focus also decays
  RF — the documented whack-a-mole). STOPPED at the gate.

**THE OPEN DESIGN DECISION (next session, decide before any new run):** every
policy (both lineages) swings THROUGH the tolerance in 1 step and never PARKS.
Two structural candidates, not coefficient tweaks:
1. **Posture-goal visibility (likely the real gap):** for posture stances the
   obs carries NO goal signal — all four limbs are gripped, so the goal slots
   obs[118:130] are zero and the policy cannot distinguish "target is 5 cm
   left" from "you're there"; parking on a point you can't see is memorization.
   Fix is wrapper-level via the SAME mechanism grip moves already use
   (`_patch_mover_obs`): write (goal_com − com) into a limb slot for posture
   stances. obs.py untouched, but it changes what trained policies see —
   decide, then retrain.
2. BC/ratchet alternatives: behavior-clone the corridor zero-action park, or
   train at posture_hold 2 and ratchet to 4.

**Then: generalize beyond this one wall/reference (Path A/B/C) — the real frontier.**

---

## (superseded) Where we were (2026-06-22) — multi-move CLIMBING works; foot moves SOLVED

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

### Phase-conditioning (2026-06-30 → 07-01): built, ABLATED, did NOT validate
Hypothesis: the multi-move collapse (chain4_incr 20/20→0/40 on continued training) was
catastrophic forgetting from the obs carrying no "which move" signal; `--phase-obs`
appends a move-phase scalar (131→132, DeepMimic-style) to fix it. **A clean per-move
ablation refuted it:** identical foundation + 70%-RF-oversample recipe, phase ON vs OFF —
phase OFF learned ALL FOUR moves (RF 100%); phase ON failed RF (0%) and gave NO extra
protection (phase-OFF held the easy moves at 100% too). So phase provided no benefit and
hindered the hard move. `--phase-obs` code stays (opt-in, harmless) but is NOT the fix.
See memory [[phase-conditioning-fragility-fix]].

**THE REAL WINS this session:**
1. **`--milestone-focus-stance N --milestone-focus-frac 0.6`** (oversample one weak move,
   keep others in-mix). Cracked RF — the documented hardest transition (final foot move,
   most-committed stance) — to 100% per-move. All 4 per-move skills now 100% (phase-off,
   `ladder_rf_noph`). The multi-week foot-move-DISCOVERY wall is broken at the per-move level.
2. **Reproducibility:** `train()` writes `config.json` (icfg+argv); recipes live in
   `train_chain_phase.sh` (the chain4_incr recipe was lost to shell history).

**THE REMAINING WALL — sequential composition of a foot move (distribution shift):**
LF composes fine per-move (100% from the authored stance) but stalls at 2/3 when launched
from the policy's REAL post-LH landing — ~11% stochastic, 0% deterministic, across FOUR
attempts (compose-all, phase-on ladder3, phase-off ladder3_noph, capture-0.7 v2). Neither
phase nor capture/budget tuning moves it. `--sequential-chain` (train move k from move
k-1's real landing) is the right idea but underperforms the docs' lost chain3_seq (80%).
**Recommended fix (a real build, not a knob): DAgger / re-seed** — collect the policy's
actual post-LH landing distribution and re-author/re-train LF from THAT, per the
stance-keyframe notes. This is the true long-term target for a multi-move climb.

**OUTCOME (end of 2026-07-01 session) — DAgger WORKED; 4-move blocked on RF.**
Built the DAgger loop (`--collect-landings`, `--landing-bank`, multi-bank `--landing-banks`,
`--milestone-focus-frac`, `--wall-hug-mover`; all in `sim3d/imitation.py`). Results:
- **3-move climb (RH→LH→LF): 20/20 deterministic** (`lf_dagger2`) — the composition wall is
  BROKEN. DAgger closed the LF distribution-shift (parked 0.117 m → grips at 0.059 m). ★ MILESTONE.
- **Multi-bank DAgger** (`--landing-banks 2:lf,3:rf` + `--frac 0.5`) keeps ALL moves alive
  (RH/LH/LF 100%) — solves the single-focus whack-a-mole.
- **RF (4th move) is a REACHABILITY wall**, not distribution shift: from the committed post-LF
  stance (3 limbs up, body leaning out), RF can't reach f_046 (RF-from-bank min 0.12–0.19, never
  grips). Reward-based weight-shift (global + RF-only `wall-hug`) FAILED — can't sequence "shift
  IN then reach OUT" per-step.
- **NEXT SESSION:** author a WEIGHT-SHIFT STANCE before RF via `sim3d.discover` (com over both
  feet, then release RF), then re-run multi-bank DAgger (harness is ready) to compose the 4-move.
  Judge chains by `--eval` (is_success), NOT the furthest-stance diag (undercounts the last grip).
  Best checkpoints: `lf_dagger2` (3-move 20/20), `mb_dagger2` (all-4 per-move 100%).
  See memory [[dagger-cracks-foot-composition]].

**NOT yet validated on a real run.** A (132,) phase policy can't warm-start the existing
(131,) `chain4_incr`, so two ways to test:
1. **Retrain the recipe from scratch with `--phase-obs`** (chain1→4, all warm-starts stay
   in the 132 regime), confirm it still hits 4-move 20/20, THEN push 4→5 and see if phase
   conditioning prevents the `chain5_incr` collapse. Clean but ~hours of compute.
2. **Weight-surgery (faster, NOT built yet):** lift `chain4_incr` (131-in layer) to 132
   by zero-initializing the phase-input column (behaviorally identical at load) + expand
   the VecNormalize obs_rms, then continue/extend. Tests the hypothesis on the existing
   working policy with minimal compute. Build this if the from-scratch re-run is too slow.

**RECIPE IS NOW A SCRIPT (no more shell-history loss):** `train_chain_phase.sh`
(stages 1=per-move skills, 2=compose 4-move, 3=extend to 5 — the fragility test).
Every run also writes `config.json` (full icfg + argv) to its run dir. Coeffs in the
script mirror the documented chain recipe; the exact chain4_incr flags were
unrecoverable (train.log records only the warm-start + ent_coef).

**STATUS (2026-07-01) — pivoted to the INCREMENTAL LADDER after a methodology error:**
- Phase-conditioned per-move milestone training is STABLE (first run: 85% per-move,
  clean monotonic curve). That part of the fix is validated.
- FIRST APPROACH FAILED (methodology, not phase): "train all per-move skills, then
  compose all 4 at once" (`chain4_phase_seq`) hit the documented DEPTH-4 WALL —
  full-chain completion crawled to ~3% over 1.5M+ steps; deterministic stuck at 3/4.
  The deterministic "3→2 regression" on continuation was an EXPLORATION ARTIFACT
  (stochastic full-chain completion was actually RISING 0.9%→3%), not collapse. Root
  cause: the last move never had a robust base to launch from. Compose-all-at-once is
  NOT the proven recipe.
- FIX = the PROVEN incremental ladder (`train_chain_phase.sh` rewritten): warm-start
  the N-move sequential from a WORKING (N-1)-move policy — 3-move → 4-move → 5-move,
  each rung adds ONE move to a robust base (how the non-phase chain4_incr hit 4-move
  20/20). Refs are a verified nested prefix chain: `ref_chain_3move`(RH,LH,LF) ⊂
  `ref_chain_handsfirst`(+RF) ⊂ `ref_chain_5move`(+RH→h_014).
- Also shipped: FIXED-DENOMINATOR phase (`phase_denom`=10) so the same move keeps the
  same phase across ref lengths — required for the ladder's cross-length warm-starts
  (per-ref normalization shifted them). And a `seq_eval` helper (the earlier `--eval`
  was missing `--stance-milestone --sequential-chain` → bogus "mean len 1").
- RUNNING: `ladder_ms` (per-move skills, fixed-denom). Then rung-by-rung 3→4→5,
  checking DETERMINISTIC seq_eval between each. THE TEST: does the phase ladder pass
  4→5 where the phase-blind `chain5_incr` collapsed? Superseded runs: chain4_phase_*.
- Judge by DETERMINISTIC eval, not phase-avg (sequential phase-avg = full-chain only).

**Long-term phase-encoding refinement:** `_phase_value` normalizes by per-ref
`n_stances`, so the same move shifts phase across chain lengths (move 4: 0.75 in the
5-stance ref vs 0.6 in the 6-stance). Minor (1/132 dims, ≤0.15; moves stay distinct).
For clean incremental warm-starting of longer chains, switch to a FIXED denominator
(same move = same phase regardless of total length) when re-running the recipe from
scratch for production. Don't change it mid-experiment (stages 1-2 baked in per-ref).

### Slow moves — DEFERRED (2026-06-24): velocity penalty does NOT work; needs a structural fix
Tried hard: warm-start snap policy + --vel-penalty-coeff (sum qvel², calibrated 0.2-0.25 vs snap peak ~90) over 6 runs → still 2-frame/0.03s snaps (entrenched policy eats the penalty, never explores slow control). From SCRATCH with the penalty (slow_scratch, coeff 0.15) → 0% (penalty suppresses the exploration needed to learn the move at all). CONCLUSION: a soft velocity reward can't produce slow moves here — too weak to shift a converged policy, too strong to learn from scratch. Real fix = HARD action-rate limit (clamp |action_t - action_{t-1}| in the env wrapper, or cap joint velocity) so snapping is physically impossible; OR defer to the torque/muscle phase (servo snaps won't exist there). Deferred for now.

### 5-move hit the incremental-recipe LIMIT (2026-06-24); 4-move is the clean ceiling
chain5_incr (5-move, naive sequential warm-start from working 4-move) DISRUPTED the chain: final deterministic eval RH20 LH20 but LF/RF/RH5=0, FULL 0/20 — exploration on the longer horizon broke working moves 3-4 and never recovered. So appending move N+1 via plain sequential warm-start works up to ~4 moves, then the longer-horizon training perturbs earlier moves. FIX being tried: milestone-train ALL N per-move skills on the full N-stance ref (warm-start the (N-1) policy → retains 1..N-1, learns N as a skill), THEN sequential with LOW ent to compose without disruptive exploration. If that also regresses, 4-move is the ceiling (excellent).
NIGHT ARC (2026-06-24): banked+pushed 2/3/4-move climbs all 20/20 deterministic via incremental warm-start (2→3→4). Methodology fix: judge by DETERMINISTIC eval (phase-avg understates ~2x from exploration noise — chain4_incr read 40% phase-avg but 20/20 det). Slow-moves DEFERRED (vel penalty fails both regimes, needs hard action-rate limit). Net-rise UNSOLVED (~0.06m; com_rise_reward authoring sagged with balance_cap_n 0 — needs the balance TARGET to rise during the move, not just a reward).

### 4-MOVE climb WORKS 20/20 (2026-06-24) — incremental warm-start cracks the composition wall
chain4_incr = 4-move hands-first sequential (hang→RH:h_012→LH:h_013→LF:f_047→RF:f_046), warm-started from the WORKING 3-move policy (chain3_seq). Deterministic sequential eval = **20/20 FULL 4-move** (RH/LH/LF/RF all 20/20), net rise +0.059m, real (video chain4_incr/climb4.mp4). 
KEY CORRECTION: phase-avg success badly UNDERSTATES — chain4_incr's phase-avg oscillated ~30-43% (exploration noise) but the DETERMINISTIC policy is 20/20. Judge chains by deterministic eval, not phase-avg. (The earlier "4-move stalls at 35%" reads were phase-avg; deterministic is what matters.)
PROVEN RECIPE for longer chains: build INCREMENTALLY — warm-start the N-move sequential from the working (N-1)-move policy so the new last move gets reliable gradient. 2→3→4 all reached 20/20 this way. Next: push 5,6 moves the same way. Net rise still ~0.06m (com-rise authoring is the lever — discover_move now has com_rise_reward param).

### 3-MOVE climb COMPOSES (2026-06-24): 20/20, the composition wall is at depth 4
chain3_seq = hands-first 3-move sequential (hang→RH:h_012→LH:h_013→LF:f_047), warm-started from handsfirst_ms (per-move 16/16). Full-climb completion rose cleanly 20%→80% over 1.5M; deterministic sequential eval (start=bottom, no reset) = **20/20 full 3-move climbs**, net com rise +0.05 m, real (video chain3_seq/climb3.mp4). So: 2-move=1 composition (20/20), 3-move=2 compositions (20/20), 4-move=3 compositions (stalled ~30%). The wall is the 3rd composition. KEY PATH: build the 4-move INCREMENTALLY — warm-start the 4-move sequential from the working 3-move policy (chain3_seq) so moves 1-3 stay reliable and move 4 (RF) gets consistent gradient (it was starved before, only reached when 1-3 happened to all fire). Net rise stays small (~0.05 m) — still needs the com-rise authoring work.

### OVERNIGHT 2026-06-24 wrap — 4-move climb is the verified ceiling; next steps crisp
WORKING (pushed): 2/3/4-move sequential climbs all 20/20 deterministic via incremental warm-start. Headline video: data/runs/sim3d/imitation/FINAL_4move_climb.mp4.
OPEN (research, not quick-wins): 
- 5-move: incremental warm-start disrupts the working chain past 4 (longer-horizon exploration breaks earlier moves). Try: per-move pretrain ALL N skills then compose with VERY low ent (0.004-0.006), or freeze early policy layers during compose, or DAgger each stance from real landings.
- NET RISE (~0.06m): w_task=1 ignores authored pose so authoring-side fixes (com_rise_reward, rising balance target) can't work; needs a working TRAINING-side com-rise (the --com-rise-coeff added had no effect — investigate why; may need a pull-up shaping that isn't geometry-bounded).
- SLOW MOVES: needs a hard action-rate limit in the env wrapper (clamp |action_t-action_{t-1}|); soft velocity penalty proven to fail both warm-start and from-scratch.

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

### Where this stands (2026-06-30, evening update)

**Retargeting bugs FIXED — gate cleared.** Built a headless diagnostic
renderer (`sim3d/debug_render_retarget.py`, `mujoco.Renderer` offscreen, no
mjpython needed) and used it + raw per-joint statistics to root-cause all
three reported bugs in `sim3d/retarget.py`:
1. **Left/right mirroring** — `l_knee`, `l_hip_flex`, `l_shoulder_roll`,
   `r_elbow` were each reading the mirror-opposite sign of their
   counterpart for the same physical motion (e.g. `l_knee` was 100%
   saturated negative every frame across all 5 clips while `r_knee` tracked
   0–13° cleanly). Fixed by negating the affected hinge axes in
   `_HINGE_AXES` (`sim3d/retarget.py:128`) — confirmed both legs/arms now
   move plausibly in rendered frame sheets.
2. **`spine_lean` saturation** — root cause was NOT "3x amplified real
   motion" as hypothesized; `SMPL_SPINE1` alone carries a large, almost
   entirely-negative structural offset (mean −45..−85° across all 5 clips,
   never crossing positive) that isn't physical — `spine2`+`spine3` alone
   are well-behaved on every clip. Fixed by dropping `spine1` from the
   composed spine rotation (`sim3d/retarget.py:186`). Violations per frame
   dropped from mean 4.3 → 0.5–1.8 across clips; all 5 now PASS quality.
3. **Flattened motion** — was mostly a symptom of (1)+(2): a permanently
   clipped spine and dead left limbs made every frame look similar. Fixed
   poses now show clear per-frame arm/knee variation in the render sheets.

**AMP pipeline ran end-to-end for the first time (2026-06-30).** Also fixed
two bugs blocking `build_amp.sh`: a stale `from sim3d.retarget import
make_pairs` (function actually lives in `sim3d/amp.py` itself — removed the
bad import) and a missing `--fps 10` (clips were extracted at 10fps but
`retarget_clip` defaulted to 30, silently mis-scaling qvel 3x). Saved:
`data/amp/clips/*.npz`, `data/amp/motion_library.npz`, `data/amp/disc.pt`.

**2026-07-02 audit: the frozen-disc path was a PLACEBO — online AMP now
built.** Two fatal flaws in the 06-30 setup: (1) `python -m sim3d.amp train`
uses Gaussian-noise fakes, so the boundary is "climbing vs noise" and any
coherent motion (including servo snaps) scores ~1 — as an RL term that's a
constant offset with zero style gradient (`amp_test`'s result is void);
(2) Δt mismatch — clips pair at 10 fps (0.1 s) but the env paired
consecutive control steps (0.016 s), so a real disc would separate on frame
spacing alone. **Fix (built + smoke-validated):** `--amp-online` on the
trainer — `AMPOnlineCallback` collects the policy's own (s, s+6) pairs from
env infos (stride 6 ≈ 0.096 s, matched to clips), updates the discriminator
real-vs-policy each PPO rollout (LSGAN + grad penalty, lr 3e-4), and
broadcasts weights to the workers via `env_method`; the disc normalises
inputs by library stats (buffers in the checkpoint). Smoke (12k steps,
SubprocVecEnv): d_fake 0.74→0.35, r_style(policy) 0.70→0.45, un-saturated —
a live adversarial gradient. Watch the `[AMP]` lines: healthy is d_real ≫
d_fake with r_style strictly inside (0, 0.75); pinned at either end = no
gradient. Style validation run (posture/jank vs `mb_dagger2` baseline) still
to be done — add `--amp-online --amp-coeff 0.3` to the production milestone
recipe, ideally with `--action-rate-limit` so velocity alone can't dominate
the disc's evidence.

**Still pending:** Colab clips 6–9 (5/9 done, idle-recycled mid-run — see
prior notes below; not blocking, 5 clips already gave a usable
discriminator). Once more clips land, re-run `build_amp.sh` to fold them
into a richer motion library and retrain the discriminator.

**Colab SMPL extraction — WORKING**, 5/9 spike1 clips done
(`demo_clip01..05.pkl` on Drive `MyDrive/beta_smpl/outputs/`; clips 6–9
pending, Colab idle-recycled mid-run). Use `shubham-goel/4D-Humans` fork
(NOT `brjathu` — missing `track.py`); SMPL neutral model must be placed
manually in two cache dirs (PHALP auto-download 404s); torch 2.6
`weights_only=True` needs a `sitecustomize.py` patch; `neural_renderer`
stubbed (`render.enable=False`); 10fps downsample (~8 min/clip on T4).
Master cell in `data/video/moonboard/spike1/smpl_extract_colab.ipynb` is
idempotent — re-run to resume remaining clips. Needs Colab Pro background
execution or a kept-open tab to finish (lid-closed sleep kills it).

### V1. Pose extraction tooling

**2D feasibility CONFIRMED (2026-06-11)** on the first real video
(`data/video/moonboard/spike1/`, 9 clips V3–V7, static head-on shot):
YOLO11s-pose detects the climber in ≥99% of frames at every grade tested,
wrist confidence 0.90–0.95, ankles 0.85–0.92, drop-knee poses tracked
cleanly. See the dataset README for the full table.

**3D (SMPL) extraction WORKING on Colab** — see "Where this stands" above.

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
