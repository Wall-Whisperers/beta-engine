# Next Steps & Known Issues

> **Freshness warning.** This is a living roadmap, not a snapshot. Update
> or delete entries as you act on them — a stale roadmap is worse than no
> roadmap. Prune aggressively; don't let this drift into an audit doc that
> rots while the code moves.
>
> Last reviewed: 2026-05-19.

---

## Where we are

The sim3d/ merge landed: one canonical 3D MuJoCo env, fixed-shape (127,)
observation, `Box(25,)` continuous-joint action with explicit grip intents,
HWM reward, kickboard support, video rollout callback, no more
`physics/`/`rl/`/`moonboard-rl/`. Smoke test passes; the env constructs,
resets without raising, and steps without NaN.

What we **don't** have yet:

- A trained policy that completes any MoonBoard problem with > 0% success.
- A curriculum.
- Behavior-cloning bootstrap data.
- An evaluation harness that reports success rate (we only have per-step
  reward in `episode_stats.csv`).

The honest framing: the foundation is in place, but the agent has not yet
been shown how to climb. The work below is what bridges that gap.

---

## Top priority — make PPO actually learn (Phase A)

These items unblock the first signs of life. Without them, you will burn
GPU hours watching reward stay flat.

### A1. Curriculum: start with "hang"

Train a much easier task before climbing: from a 4-point grip on the start
holds + kickboard, **survive N seconds without falling**. Action space and
observation are unchanged. Reward is just `+1 per step survived,
−50 on fall`. This teaches the policy:

- Which grip intents keep welds engaged.
- How to balance joint torques against gravity.
- The cost of body intersection (so it tucks rather than crosses itself).

A policy that solves "hang for 10 s" is a much better init for the full
climb task than random weights. Plan: add `EnvConfig.task_mode` with
values `hang | reach-one | climb` and gate the reward function on it.

### A2. Behavior cloning from the discrete-move expert

`discrete-move` still works. It uses the Cartesian-impedance reach
controller, which is a hand-tuned "expert" that succeeds on simple beta
sequences. Idea:

1. Generate ~10k (obs, joint-target, grip-intent) tuples by running
   discrete-move with a scripted beta on each MoonBoard problem.
2. BC-pretrain the PPO MlpPolicy on those tuples (Stable-Baselines3
   supports this via `imitation` library or a custom dataset).
3. Fine-tune with PPO on the full reward.

This avoids the "random 25-D Gaussian for 1M steps does nothing" failure
mode that pure-from-scratch PPO has on humanoid manipulation.

### A3. Pretrain on jug walls before MoonBoard

`data/walls/` has hand-designed walls with `jug` holds and generous
spacing. Train on those first (positivity 1.0, max_force_n high). Move
the trained checkpoint to MoonBoard crimps as a starting point. The
observation shape is the same, so weights transfer directly.

### A4. Reduce GRIP_PROXIMITY_M and / or require proximity + alignment

At 5 cm with K-nearest sort, a flailing limb can accidentally engage a
grip without the policy ever learning to "place" the limb. Either drop
proximity to 2 cm, or require both proximity AND a positive dot product
between the limb's contact normal and the hold's outward normal (use the
`wall_normal` field already in `hold_meta`).

---

## Phase B — Reward and observation tuning (after Phase A shows signal)

### B1. Profile the energy penalty

`−0.005 × Σ ctrl²` over 21 joints with [-1, 1] ranges can easily reach
~0.1 per step. HWM gains for a typical 1-cm com_z increment are
`5.0 × 0.01 = 0.05`. The energy penalty currently dominates small-
progress steps, which may discourage the slow careful moves we want.
Either:

- Drop `energy_penalty_coeff` to 0.001.
- Normalise the penalty by `nu` so it doesn't scale with body DOF.
- Switch to penalising squared torque output (post-ctrl) rather than
  ctrl input.

### B2. Smooth the finish-distance observation

`obs[-1]` is the distance from the highest gripped hand to the nearest
finish. When the agent releases its highest hand, this number jumps
discontinuously to a (lower) hand's height. Could destabilise the value
function. Options: EMA over the last few steps, or always use the
geometric max of all four limb tips.

### B3. Per-hold capacity sanity check

`FOOT_FORCE_MULTIPLIER = 1.5` is a guess. Compare against real climber
power output (Goddard & Neumann, or the Lattice grip-strength dataset).
If feet are over-strong, the agent will discover unphysical heel-hook
betas that aren't representative.

---

## Phase C — Evaluation infrastructure (mostly missing)

### C1. Success rate, not just reward

Add `success` and `n_finished_seeds` columns to `episode_stats.csv`.
Write a `sim3d/eval.py` that runs N=100 episodes with different seeds
on a held-out problem and prints `success rate ± stderr`. Wire it into
`train.py` as a periodic callback.

### C2. Video callback should ship by default

`sim3d/callbacks.py::VideoRolloutCallback` exists but is not wired into
`sim3d/train.py`. Add a `--video-freq` CLI flag (0 = disabled, > 0 =
mp4 every N steps). This is the single most useful thing for noticing
"the policy looks crazy but reward is going up" failure modes.

### C3. Held-out split is configured but unverified

`TrainConfig` has `moonboard_split=train|validation|test`. The split is
deterministic given `split_seed`, but no evaluation actually loads the
validation split mid-training. Add a periodic eval-on-validation in the
callback chain so we can detect overfitting to a single problem.

---

## Phase D — Performance and infra

### D1. MoonboardClimbing3DEnv rebuilds the MjModel every reset

`_build_env` is called from `reset` to swap the problem. Building MJCF
and compiling `MjModel.from_xml_string` takes ~30–100 ms. With ~2000
steps per episode and ~5–20 episodes/min, this is bearable; with PPO
vec-envs hitting reset on done flags, it can dominate. Caching trick:
maintain a dict `{problem_id: MoonboardClimbing3DEnv}` and round-robin
through it instead of rebuilding.

### D2. The grip search is O(n_holds) per intent per step

`_maybe_engage_grip` scans every hold in the wall on every step where
any limb intent is positive. For MoonBoard's 198 holds × 4 limbs ×
2000 steps × 8 vec-envs, that's ~12M distance checks per minute. Use a
KD-tree built once at construction.

### D3. Wire `imageio` into the default install if we ship videos

`requirements.txt` should pin `imageio` and `imageio-ffmpeg` so the
video callback works out of the box. Today it silently no-ops if the
import fails.

---

## Phase E — Body model and physics (longer horizon)

### E1. Validate the kickboard with the seed pose

The kickboard's foot-only holds sit at `±0.244 m` and `z ≈ 0.27 m`. The
default `_default_seed_kwargs` looks for foot holds in `self.wall.holds`
by `hold_type == "foothold"`. The kickboard holds are *added to
`hold_meta`* but not back-propagated into `self.wall.holds`. So the seed
pose for MoonBoard problems probably uses route holds as foot anchors,
not the kickboard. **Open question: is the kickboard ever actually
engaged at reset?** Fix: expose kickboard holds through the Wall object
or extend `_default_seed_kwargs` to consult `hold_meta`.

### E2. Custom body has wrist hinges but no finger DOF

This is by design (CLAUDE.md), but it means the policy can't model
crimp vs jug differently except through grip-force capacity. If we
later want to predict per-hold grade difficulty for specific climber
strengths, finger flexor strength becomes a missing feature.

### E3. Spine has 1 DOF (forward lean only)

Real climbers twist the torso to reach across (gaston, drop-knee). The
current spine joint blocks any such move. Two more spine hinges
(lateral and rotational) would unlock the most common 3D climbing moves
the policy currently cannot perform.

---

## Things you should NOT do

- Switch the default action mode back to `discrete-move` because PPO is
  slow to converge in continuous-joint. The discrete mode hides the
  problem we're actually trying to solve. Use it only as an expert for
  BC pretraining (A2).
- Replace the custom climber with the stock Gymnasium humanoid because
  "the stock one trains faster." It trains faster because it's tuned
  for locomotion, not climbing. Climbing reward signals are sparse; the
  body's anatomical constraints matter.
- Add `KNOWN_ISSUES.md`, `ARCH_REVIEW.md`, or per-Phase progress
  trackers. This file is the only roadmap. Keep it pruned.

---

## Tracking

When you finish an item, **delete it from this file**, then add one or two
sentences to the relevant section of `CLAUDE.md` if the item changed the
permanent architectural contract. Cumulative changelog lives in git; this
file is for "what's next."
