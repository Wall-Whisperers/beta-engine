"""Imitation training loop — RSI + bounded DeepMimic reward + termination curriculum.

This is the "learn a robust controller" half of the plan, on top of a reference
authored by ``sim3d.reference``. It replaces the reverse-curriculum / warm-start
approaches that all failed (NEXT_STEPS A1b): instead of asking PPO to *discover*
the climb, it asks PPO to *track a known-good reference*, which the field treats
as the only way long-horizon physical skills are learned.

``ImitationEnv`` wraps a ``Climbing3DEnv`` (``task_mode="imitate"``) and supplies:

* **Reference State Initialization** — every episode starts RSI'd into a random
  frame of the reference (not the bottom), so every move in the sequence gets
  on-policy gradient. This is the chaining fix: a fixed-start policy never
  reaches later states, so they never train.
* **Bounded imitation reward** — ``sim3d.reference.imitation_reward`` ∈ [0,1],
  un-farmable by oscillation (maximised only by matching the reference).
* **Termination Curriculum** — terminate when the instantaneous imitation
  reward drops below ``R_min``; anneal ``R_min`` 0.75 → 0.50 over training. Early
  on, drifting off the reference ends the episode fast (sim budget stays near the
  good trajectory); later the agent may venture further. The principled version
  of "seed high, walk down" — it hugs the *reference*, not wall geometry.

Run a smoke check / short train:
    python -m sim3d.imitation --smoke
    python -m sim3d.imitation --train --steps 60000 --n-envs 4
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import gymnasium as gym
except ImportError as e:  # pragma: no cover
    raise ImportError("sim3d.imitation requires gymnasium (pip install -r requirements.txt)") from e

from sim3d import artifact_meta as am
from sim3d import config as cfg
from sim3d.body import LIMBS, ClimberProfile
from sim3d.env import Climbing3DEnv, EnvConfig
from sim3d.reference import (ENV_SUBSTEPS, ImitationCoeffs, Reference,
                             author_weight_shift_move, imitation_reward,
                             migrate_reference_spine)


@dataclass
class ImitationConfig:
    coeffs: ImitationCoeffs = field(default_factory=ImitationCoeffs)
    w_task: float = 0.0               # weight on the task term (0 = pure imitation, v1)
    r_min_start: float = 0.75
    r_min_end: float = 0.50
    r_min_decay_steps: int = 150_000  # linear R_min anneal horizon (per-env steps)
    min_episode_frames: int = 4       # RSI start phase ≤ T − this
    settle_frames: int = 4            # RSI weld-transient settle
    inner_max_steps: int = 200        # inner-env backstop (phase-end ends it first)
    # Cap the RSI start phase so every episode includes the *active* part of the
    # move. A short single-move reference is mostly post-grip settle frames, so
    # uniform RSI would start ~80% of episodes in the trivial tail and a null
    # policy would "succeed" — masking learning. Capping to the pre-/at-reach
    # region makes success a real measure of executing the move. None = uniform
    # (the right choice for a full multi-move reference, where every phase is
    # active).
    rsi_phase_max: Optional[int] = None
    # Anneal the RSI cap from len(ref) (uniform — every move gets gradient) down
    # to rsi_phase_max (forced near-bottom — the full chain) over this many
    # per-env steps. Hardens the full climb: phase-averaged success hides that the
    # hardest start (the bottom, crossing every boundary) is undertrained under
    # pure uniform RSI; annealing focuses late training there. 0 = fixed cap.
    rsi_anneal_steps: int = 0
    # Chain curriculum: every episode starts at frame 0; stage k runs to move
    # k's boundary and SUCCEEDS only if the grips there match the reference's
    # stance (executed k real moves — un-inflatable). ≥70% rolling success
    # unlocks stage k+1. Fixes the imitation-mesa failure: on long
    # quasi-static references, "hold still and sag slowly" outscores the
    # brief critical transitions, so full-length episodes teach sagging
    # (observed: one move then a 0.44 m sag scored mean len 120). Short
    # windows put all the credit on the transition that matters.
    chain_stages: bool = False
    chain_advance_rate: float = 0.7   # rolling success to unlock next stage
    chain_window: int = 40            # episodes in the rolling window
    # Terminal bonus for a verified grip-match at a stage boundary (and at
    # full completion). Without it PPO is INDIFFERENT to stage success —
    # success and failure both end the episode at the same frame with the
    # same accumulated r_imit, so optimization drifts toward tracking-reward
    # poses that never land the grips (observed: stage-2 success 67%→21%
    # while r_imit held steady). Sparse, once per episode, requires
    # verified grips ⇒ un-farmable.
    completion_bonus: float = 10.0
    # Fraction of the completion_bonus given for correctly landing EACH
    # intermediate boundary grip during a stage-k (k>1) episode. Prevents
    # the policy from coasting past prior grip transitions — observed in
    # chain5b: stage-3 training (window 0→158) left the policy hanging by
    # one hand at frame 84 because only the frame-158 grip was checked.
    # E.g. 0.5 → 50% bonus per intermediate grip; 0.0 disables.
    intermediate_bonus_frac: float = 0.5
    # Fraction of chain episodes that RSI into the middle of the stage window
    # (vs starting at frame 0). Mid-window starts teach the move's dynamics
    # (the swing into the target); frame-0 starts are what advancement is
    # scored on. Raising this when a stage plateaus gives the stuck transition
    # more on-policy gradient without changing the advancement bar.
    chain_mid_frac: float = 0.5
    # When True, RSI for stage k starts in [0, stage_start) — i.e. BEFORE the
    # current stage's active window. Prevents the "free success" failure mode
    # where RSI at frames AFTER the mover grip auto-grips the limb from the
    # reference state, inflating phase-avg success without the policy ever
    # executing the move. Example: stage 3 RF high-step (grip fires at f90);
    # with default RSI [0, 154], 65%×64/154 = 27% of episodes start with RF
    # already on h_004 from RSI init — PPO thinks it's succeeding without
    # learning the hip-flex/abduct trajectory. With this flag, RSI is [0, 84)
    # so every episode must actually execute the high-step.
    chain_rsi_before_stage: bool = False
    # When True, mid-window RSI episodes start at EXACTLY the current stage's
    # start boundary (chain_bounds[k-2]) instead of a random frame. For stage
    # k=3 (RF high-step, boundary at f84): 65% of episodes RSI to exactly f84
    # so all mid-window gradient focuses on the RF swing. Avoids both the
    # "free grip" inflation (RSI at f90+ where reference already placed RF)
    # AND the "must do everything from f0" compound problem of chain_rsi_before_stage.
    chain_rsi_at_stage_start: bool = False
    # Dense potential-based reach bonus for the mover limb in chain mode:
    # each step rewards (prev_gap - curr_gap) × coeff when the current
    # stage's mover limb is ungripped. Breaks the "hold RF still" local
    # optimum where the policy scores r_imit≈0.50 without moving the limb
    # toward its target hold (gap stays ~0.30 m while ref reaches 0.04 m).
    # Potential-based so it can't be farmed (release = negative reward).
    # Scale: coeff ≈ 100 gives ~total_bonus≈26 for a 0.26m reach, comparable
    # to completion_bonus=40. 0.0 disables.
    mover_reach_coeff: float = 0.0
    # Per-step penalty for losing a grip that the reference maintains on a
    # NON-MOVER limb during the current stage. Fires each step the limb is
    # ungripped when the reference grip log shows it should be gripping.
    # Directly counters the "release LH during foot step" local optimum where
    # the diluted endeff signal (1/4 of w_endeff per limb) is too weak to
    # prevent slip. Scale: 5.0 = same magnitude as the physics slip penalty.
    # 0.0 disables.
    grip_retention_coeff: float = 0.0
    # When True, override the grip-intent action channels for all limbs the
    # reference keeps gripped at the current phase: force intent to +1 so the
    # policy physically CAN'T release them. The policy still learns the joint
    # trajectory; it just can't lose non-mover grips. Use when the
    # grip_retention_coeff gradient alone can't overcome an entrenched
    # "release LH" policy inherited from a prior checkpoint.
    force_non_mover_grips: bool = False
    # When True, the current stage's mover limb is EXCLUDED from the pose/endeff
    # imitation terms while it is ungripped (mid-reach). The recorded foot swing
    # is an authoring artifact the PD servos can't reproduce, so tracking it
    # fights the reach reward and parks the foot ~0.21 m short. Freeing the
    # mover lets PPO find any servo-feasible swing; the anchored limbs + CoM
    # still track, so the stance and pelvis weight-shift follow the reference.
    # Pairs with mover_reach_coeff (the only signal left on the moving limb).
    free_mover_imitation: bool = False
    # Sparse one-shot bonus the step the mover limb actually GRIPS its target
    # hold. Rewards the OUTCOME, not just gap-closing. Needed because the
    # potential-based reach reward is net-zero on retreat, so "park the foot
    # short and collect steady r_imit" is a safe high-return optimum — the
    # policy reaches only via exploration noise (observed: deterministic gap
    # 0.23 m while stochastic min 0.09 m). A big grip payoff makes committing
    # to the full swing worth the slip/fall risk. 0.0 disables.
    mover_grip_bonus: float = 0.0
    # Dense per-step bonus when the mover tip is INSIDE the grip capture sphere
    # (within GRIP_PROXIMITY_M of the target hold). The potential-based
    # mover_reach_coeff creates a gradient toward the hold from far away but is
    # net-zero once the tip stops moving — it cannot pull the tip the last few
    # cm into the 0.08 m radius. This term gives r = coeff × (1 - gap/R) per
    # step while gap < R, peaking at 1×coeff at the hold center and 0 at the
    # boundary. Unlike mover_grip_bonus (one-shot on grip), this fires every
    # step inside the radius so the optimization landscape has a gradient toward
    # commitment. 0.0 disables. Try coeff ≈ 0.2–0.5 (same scale as r_imit).
    mover_capture_coeff: float = 0.0
    # Radius (m) of the capture sphere. 0.0 → use GRIP_PROXIMITY_M (0.08), the
    # original behavior. Set larger (e.g. 0.15) when a trained mover plateaus
    # JUST outside the grip radius: the deterministic RF reach parks at ~0.126 m
    # (measured 2026-07-07), which is OUTSIDE 0.08 so the capture gradient never
    # fires — a dead zone exactly where the toe gets stuck. A wider capture
    # radius extends the strong final pull into that band to break the plateau.
    mover_capture_radius: float = 0.0

    # Dense ABSOLUTE reach pull over the full mover_reach_radius (NOT potential-
    # based): `coeff·max(0, 1 − gap/radius)` every step the mover is ungripped.
    # mover_reach_coeff is potential-based (prev−curr gap) ⇒ zero gradient from
    # a standstill, so a foot 17 cm from its hold never starts moving (the closed-
    # loop foot-move blocker, 2026-06-21). This term gives a continuous gradient
    # from rest: any motion that reduces the gap earns more cumulative reward, so
    # the policy discovers its own swing (goal-reaching, not trajectory tracking).
    # Keep coeff modest vs completion_bonus so parking just outside the grip
    # radius can't out-earn gripping: farm ≈ coeff·1·budget must stay < bonus.
    mover_reach_abs_coeff: float = 0.0
    mover_reach_radius: float = 0.25

    # ── Stance-milestone mode (the 2026-06-15 reframe) ──────────────────────
    # When True, the reference is a STANCE-KEYFRAME skeleton (settled welded
    # stances only; see discover.author_stance_reference). Each episode RSIs to
    # one stance and the policy must reach the NEXT stance — RL discovers the
    # balancing transition that discover_move (no feedback) couldn't author.
    # Reward = pose attractor to the next stance (imitation_reward against its
    # settled frame, mover freed while ungripped) + completion_bonus on a
    # verified grip-match. The R_min off-reference cut is DISABLED in this mode:
    # the target stance is deliberately far, so r_imit starts low by design;
    # termination is fall / success / budget instead.
    stance_milestone: bool = False
    milestone_budget: int = 40        # max env steps to complete one transition
    # Focus milestone RSI on ONE transition (stance c → c+1) instead of uniform over
    # all stances — puts 100% of the gradient on a single move. Used to crack a lone
    # weak move (e.g. the final RF foot move from the most-committed stance, which
    # uniform milestone under-trained to 0%) while warm-starting a foundation that
    # already knows the others. Preserves the fixed-denom phase (target_stance = c+1),
    # so the move keeps its identity and the other moves aren't overwritten. None =
    # uniform over all transitions.
    milestone_focus_stance: Optional[int] = None
    # Probability of picking the focus stance each episode (the rest sample uniformly
    # over ALL stances). 1.0 = exclusive focus, but that holds the phase input CONSTANT
    # → VecNormalize zeroes it → the policy learns to IGNORE phase and overwrites the
    # other moves (observed: RF focus cracked RF 0→100% but wiped RH/LH/LF 100→0). Use
    # <1 (e.g. 0.6) to OVERSAMPLE the hard move while keeping the others in-distribution
    # so phase stays informative and the learned moves are protected. Only used when
    # milestone_focus_stance is set.
    milestone_focus_frac: float = 1.0
    # DAgger: path to a .npz "landing bank" of the policy's OWN real states at the
    # focus stance (collected via --collect-landings). When set, milestone RSI for the
    # focus stance draws from these real landings instead of the authored stance
    # keyframe — the fix for the composition distribution-shift where a move trained
    # from its authored stance parks short when launched from the policy's real landing
    # (measured: LF reached 0.117 m, 3.7 cm short of the 0.08 grip). Only the focus
    # stance uses the bank; other stances stay on authored frames.
    rsi_landing_bank: Optional[str] = None
    # Multi-bank DAgger: "c1:path1,c2:path2" — a landing bank PER stance, so multiple
    # foot moves train from their real composition landings SIMULTANEOUSLY. Single-focus
    # DAgger specializes one foot move and degrades the other (whack-a-mole: LF-focus
    # kills RF; RF-focus decays LF). Banked stances are oversampled (milestone_focus_frac);
    # non-banked (hand) moves sample authored frames. Supersedes rsi_landing_bank when set.
    rsi_landing_banks: Optional[str] = None
    # Sequential chain (true multi-move climb): start at the bottom stance and,
    # on each grip-match, advance the target to the next stance WITHOUT reset, so
    # move k+1 trains from move k's real on-policy landing (fixes composition).
    # Success = reaching the FINAL stance. Per-transition budget still applies.
    sequential_chain: bool = False
    # Same-grip "posture" stances (weight-shift / stand-up keyframes): success
    # requires holding the com near the reference stance com, since grips alone
    # match trivially the instant such a stance becomes the target. Detected
    # automatically (stance m grips == stance m-1 grips) in ImitationEnv.__init__.
    posture_com_tol: float = 0.06      # m, com distance for a posture-stance success
    posture_hold_steps: int = 4        # consecutive in-tol steps required
    # Wrapper-level obs patch (approved 2026-07-10): in a posture stance every limb
    # is gripped, so the goal slots obs[118:130] are all zero and the policy cannot
    # perceive the com target — every probe shows it transiting the tolerance in ~1
    # step and never parking. When on, ImitationEnv._patch_mover_obs writes
    # (goal_com − com) into one goal slot for posture stances (same mechanism grip
    # moves already use; obs.py untouched, shape stays 131). It changes the obs
    # CONTRACT, so it is recorded in the checkpoint env_mode (imitate:...+postureobs)
    # and a mismatched eval/record hard-errors (see _imitate_env_mode).
    posture_goal_obs: bool = False
    # Dense com-rise pull for stand-up posture stances: coeff·max(0,1−dz/band)
    # per step, where dz = max(0, target_com_z − com_z). Gives a directed
    # gradient to raise the body (the sparse pose+com attractor alone stalls the
    # stand-up partway — the net-height fix, analog of mover_capture_radius). 0 off.
    # LEGACY: superseded by the goal potential below (active only when goal_k=0).
    posture_rise_coeff: float = 0.0
    posture_rise_band: float = 0.15    # m; height gap over which the pull ramps

    # ── Goal-potential milestone reward (2026-07-09 redesign) ───────────────
    # DESIGN NOTE — the dead-zone fix, generalized (see memory note
    # dead-zone-pattern-general-fix). Every recurring milestone failure — RF toe
    # plateauing at 0.126 m just outside the 0.08 m capture sphere, weight-shift
    # posture stances never training (0/12), the stand-up stalling at +0.054 m of
    # +0.141 m — shared one root: the bounded pose attractor SATURATES near the
    # goal (exp(-k·err²) has vanishing gradient at small err — flattest exactly
    # where the steepest is needed), while the actual goal was only a sparse
    # bonus or a weak bolt-on pull (capture sphere, posture_rise). Each symptom
    # got its own ad-hoc patch; this replaces them with one mechanism:
    #
    # 1. GOAL POINT per stance transition: the target hold center for grip moves
    #    (tracked point = the mover tip); the reference stance com (3D) for
    #    posture/stand-up stances (tracked point = the body com) — converting
    #    "hold this pose" into a reach in com-space. The 3D com goal also covers
    #    reward-side stance centering (com_y toward the wall is part of the
    #    target), so no separate centering term is stacked on.
    # 2. DENSE SIGNED POTENTIAL: r += goal_k · (d_prev − d_cur) every step of the
    #    transition, from spawn distance all the way to the success tolerance —
    #    no capture-sphere gating, no inner dead zone. It telescopes to
    #    goal_k·(d_spawn − d_final): oscillation nets zero, the episode total is
    #    bounded by goal_k·d_spawn, and — unlike the absolute capture/reach-abs
    #    pulls it replaces — hovering near the goal without gripping earns
    #    NOTHING per step, removing that farming mode outright. Per-step
    #    magnitude is physically bounded by limb/com speed (no clip: an
    #    asymmetric clip on a signed potential is farmable by slow-approach/
    #    fast-retreat cycles).
    # 3. ATTRACTOR FLOOR inside the final band (d < goal_band): the pose-
    #    attractor income is floored at its band-entry value — att =
    #    max(entry, live) — so a SATURATED attractor contributes zero gradient
    #    where the potential must own the landscape (it degenerates to a
    #    constant freeze), while an UNSATURATED one still pays for settling
    #    into the reference posture (the com potential says WHERE, the
    #    attractor says WHICH pose can hold there — a hard freeze removed that
    #    and produced swing-through-and-drift on the stand-up). Flooring fixes
    #    the near-goal RATIO without a global potential crank that would
    #    distort early transit, and without the perverse outward pull a
    #    multiplicative down-weight w(d)=d/band would create (income stays
    #    continuous at the band boundary and can never drop on entry; re-entry
    #    re-latches at the live boundary value, so band-bouncing gains
    #    nothing; pose oscillation inside the band nets zero).
    #
    # Success criteria are unchanged: grip-match for grip stances,
    # posture_com_tol/posture_hold_steps for posture stances. Wrapper-level
    # only — obs shapes and the inner-env reward are untouched.
    #
    # SUPERSEDES (ignored with a startup warning while goal_k > 0):
    # mover_reach_coeff, mover_reach_abs_coeff, mover_capture_coeff/_radius,
    # posture_rise_coeff/_band. Pass --goal-k 0 to run the legacy terms.
    # goal_k=100 is the documented reach-potential scale (a 0.26 m reach earns
    # ~26 total, below completion_bonus=40 so the verified landing still
    # dominates); goal_band=0.15 covers both observed plateau zones (RF parked
    # at 0.126 m; the stand-up stalled ~0.09 m short in com-space).
    goal_k: float = 100.0
    goal_band: float = 0.15
    # Velocity term in the POSTURE-stance goal metric: d_aug = ‖com−g‖ +
    # goal_vel_lambda·‖v_com‖ (v by finite difference; λ in seconds). The
    # position-only potential is provably NEUTRAL to in-band orbiting
    # (telescopes to zero per cycle), so a policy that swings THROUGH the
    # tolerance in 1 step pays nothing for never parking (observed: rise7/8
    # det min_d 0.061 in-tol for exactly 1 step, success stuck at ~4%
    # stochastic). With the velocity term, the potential's minimum coincides
    # exactly with the success criterion's fixed point — at the goal, AT REST —
    # so decelerating into the goal is what pays. Still telescoping ⇒ still
    # un-farmable. Grip stances keep the pure position metric (arriving fast
    # at a weld is fine; the weld does the stopping). 0 disables.
    goal_vel_lambda: float = 0.15

    # ── Phase conditioning (multi-move composition-collapse fix) ────────────
    # When True, append a single normalized move-phase scalar ∈ [0,1] to the
    # observation (obs grows 131→132). The scalar tells the policy WHICH move
    # it is on (target_stance / n_stances in milestone/sequential, chain_stage /
    # n_stages in chain mode; constant 0 for single-move dense). Without it one
    # shared MLP must handle every move from a similar-looking body state, so a
    # gradient update for a late move overwrites the representation early moves
    # depend on — the documented collapse where continued training on a working
    # 4-move chain drops it 20/20→0/40, and 5-move incremental warm-start breaks
    # moves 3-4. Phase conditioning (standard DeepMimic) lets the policy allocate
    # distinct behavior per move. OPT-IN: default False keeps the (131,) obs
    # invariant byte-identical, so existing checkpoints are untouched. A phase
    # model can only warm-start from another phase model (obs dims must match).
    phase_obs: bool = False
    # Phase is (move_index - 1) / phase_denom with a FIXED denom (not per-ref
    # n_stances), so the SAME physical move gets the SAME phase value across refs
    # of different lengths. This is what makes the incremental ladder work: warm-
    # starting a 3-move policy → 4-move ref → 5-move ref must not shift the shared
    # moves' phase (per-ref normalization did: move 3 read 0.67 on the 3-move ref
    # but 0.5 on the 4-move, undermining the warm-start). Denom 10 supports chains
    # up to 11 moves in [0,1] (clipped beyond). MoonBoard problems are ~6-12 moves.
    phase_denom: float = 10.0

    # Naturalness shaping (fixes the lean-back/barn-door jank that w_task=1 leaves
    # unpenalized). Both are gentle so they don't recreate the stillness valley:
    # lean penalizes a SPECIFIC bad direction (off the wall), com-rise REWARDS
    # motion (up) — neither rewards holding still.
    lean_penalty_coeff: float = 0.0   # × max(0, pelvis_y − start_y): discourage leaning off the wall (+Y)
    com_rise_coeff: float = 0.0       # × (com_z − prev_com_z): reward pulling the body UP (potential, un-farmable)
    # Slow the moves down. Goal-reaching with no speed cost makes the policy SNAP
    # each limb to its hold in ~2 control steps (0.03 s; a human move is ~1 s) — the
    # jerk. Penalizing mean joint-speed² makes a fast snap (high v) cost far more
    # than a slow controlled move (the quadratic does the work), so moves stretch
    # toward realistic, smoother, and land cleaner (helps long-chain composition).
    vel_penalty_coeff: float = 0.0    # × mean(qvel[6:]²)
    action_rate_limit: float = 0.0    # hard per-step |Δaction| cap on joints (env-level); slows the snap moves
    # Reward bent arms (mean elbow flexion). The bot climbs on dead-straight arms,
    # which lets the body hang BACK off the wall (the lean). Bending the arms while
    # gripping fixed holds physically pulls the body IN toward the wall, over the
    # feet ⇒ more stable, more upright. The mover arm still extends to reach (the
    # reach/grip reward dominates for it); the non-mover arms bend to pull in.
    arm_bend_coeff: float = 0.0       # × mean(elbow angle)/elbow_max
    # Pull the body IN to the wall (the real anti-lean fix). The bot hangs ~44 cm
    # off the wall on straight arms; arms are straight BECAUSE the body is far
    # (they must span the gap to the holds). Penalize com distance past a wall-hug
    # target ⇒ body comes over the feet, and the arms then bend on their own.
    wall_hug_coeff: float = 0.0       # × max(0, com_y − wall_hug_target) per step
    wall_hug_target: float = 0.16     # com_y (m) considered "in" (wall plane ≈ 0.065)
    # Apply wall_hug ONLY when this limb is the current mover (e.g. "RF"). Global
    # wall_hug helps the committed-stance foot move reach but fights the reach the
    # hand/LF moves need → degrades composition. Gating it to the RF move gives the
    # weight-shift where it's needed without disrupting the others. None = always.
    wall_hug_mover: Optional[str] = None
    # Terminal-only posture bonus: signed reward at completion = coeff*(target−mean_com_y).
    # Positive when body stayed close; negative (small) when it hung far.
    # Never applied on falls/timeouts → no incentive to fail fast.
    wall_hug_terminal_coeff: float = 0.0

    # AMP style reward: path to a trained discriminator checkpoint (.pt).
    # Empty string disables. When set, each step adds
    #   amp_coeff × disc.reward(s_t, s_{t+stride})  ∈ [0, 0.75 × amp_coeff]
    # The discriminator runs CPU-only in the env (forward pass only, no grad).
    amp_disc_path: str = ""
    amp_coeff: float = 0.5  # scale relative to completion_bonus=40; try 0.3–1.0
    # ONLINE AMP (the real adversarial loop — Peng et al. 2021). When True, the
    # trainer loads the motion library, updates the discriminator on
    # real-vs-POLICY pairs after every PPO rollout (AMPOnlineCallback), and
    # broadcasts fresh weights to the env workers; amp_disc_path is ignored.
    # The frozen-checkpoint path (amp_disc_path alone) is a PLACEBO: its
    # checkpoint was trained against Gaussian-noise fakes, so any coherent
    # motion scores ~1 and the reward is a constant offset with no gradient.
    amp_online: bool = False
    amp_library_path: str = "data/amp/motion_library.npz"
    # Control steps per AMP state pair. The library pairs are consecutive
    # 10 fps clip frames (Δt = 0.100 s); the env control step is
    # PHYS_DT 0.002 × 8 substeps = 0.016 s, so stride 6 gives Δt = 0.096 s.
    # Without matching, the discriminator separates real from fake on frame
    # spacing alone (policy pairs 6× closer in time ⇒ tiny per-pair motion)
    # and the style reward collapses to 0 everywhere — no gradient.
    amp_pair_stride: int = 6


class ImitationEnv(gym.Env):
    """RSI + bounded-imitation-reward + termination-curriculum wrapper around one
    ``Climbing3DEnv`` tracking a single ``Reference``. The reference is sampled at
    the env control rate, so phase advances 1 frame/step."""

    metadata = Climbing3DEnv.metadata

    # Posture-stance goal obs (posture_goal_obs): the limb-goal slot index that
    # carries (goal_com − com). Arbitrary but FIXED — in a posture stance all four
    # limbs are gripped so every obs[118:130] slot is zero, and the policy tells a
    # posture stance from a grip move by the all-ones grip flags (obs[58:62]), so
    # WHICH slot holds the com goal is immaterial as long as it is the same slot in
    # training, eval, and record.
    _POSTURE_GOAL_SLOT = 0   # LH

    def __init__(self, reference: Reference, wall, profile: Optional[ClimberProfile] = None,
                 imitation_config: Optional[ImitationConfig] = None,
                 env_config: Optional[EnvConfig] = None, render_mode: Optional[str] = None):
        super().__init__()
        # Transparently upgrade pre-spine references (nq 28→30, zeros for the
        # new chest joints) so existing .npz files keep working untouched.
        self.ref = migrate_reference_spine(reference)
        self.icfg = imitation_config or ImitationConfig()
        base = env_config or EnvConfig()
        base = replace(base, task_mode="imitate", max_steps=self.icfg.inner_max_steps,
                       action_rate_limit=self.icfg.action_rate_limit)
        self.env = Climbing3DEnv(wall, profile=profile, config=base, render_mode=render_mode)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        # Phase conditioning: grow the wrapper's obs by one [0,1] scalar. The
        # INNER env stays (131,) — phase is a wrapper-only concern, so obs.py
        # and the wall-agnostic invariant are untouched.
        if self.icfg.phase_obs:
            _b = self.env.observation_space
            self.observation_space = gym.spaces.Box(
                low=np.concatenate([_b.low, np.array([0.0], dtype=_b.dtype)]),
                high=np.concatenate([_b.high, np.array([1.0], dtype=_b.dtype)]),
                dtype=_b.dtype,
            )
        self._total_steps = 0
        self._phase = 0
        # Chain-curriculum state: stage k means "execute moves 0..k-1 from
        # frame 0". Window boundaries come from the reference's move_starts
        # (each marks the chained start of move k = the established end
        # stance of move k-1), plus the final frame.
        ms = list(self.ref.meta.get("move_starts") or [])
        self._chain_bounds = (ms[1:] + [len(self.ref) - 1]) if ms else [len(self.ref) - 1]
        self._chain_stage = 1
        self._chain_results: list[bool] = []
        self._intermediate_awarded: set[int] = set()  # boundary indices already rewarded this ep
        self._prev_mover_gap: float = float('inf')
        self._mover_limb: Optional[str] = None
        self._mover_hold_pos: Optional[np.ndarray] = None
        self._mover_hold_id: Optional[str] = None
        self._mover_grip_awarded: bool = False
        # Stance-milestone state: settled-frame index per stance, the stance
        # we're reaching toward, and a per-transition step budget.
        self._stance_frames = self._compute_stance_frames()
        # Stances whose grip-set equals the previous stance's are POSTURE stances
        # (weight-shift / stand-up): grip-match success is trivial for them, so
        # _step_stance additionally requires the com to hold near the reference
        # (see _stance_reached).
        self._posture_stances = {
            m for m in range(1, len(self._stance_frames))
            if self.ref.frame_grips(self._stance_frames[m])
            == self.ref.frame_grips(self._stance_frames[m - 1])
        }
        self._posture_hold = 0
        # DAgger landing bank(s): stance -> real-landing states. Multi-bank trains
        # multiple foot moves from real landings at once (avoids the single-focus
        # specialization whack-a-mole). Empty = authored-stance RSI everywhere.
        self._landing_banks: dict = {}

        def _load_bank(path):
            _d = np.load(path, allow_pickle=True)
            bank_meta, _ = am.parse_npz_meta(_d["meta"] if "meta" in _d else None)
            am.validate_landing_bank(bank_meta, wall=wall, path=path)
            return {"qpos": _d["qpos"], "qvel": _d["qvel"],
                    "grips": _d["grips"], "n": int(len(_d["qpos"]))}

        if self.icfg.rsi_landing_banks:
            for pair in self.icfg.rsi_landing_banks.split(","):
                cs, path = pair.split(":")
                self._landing_banks[int(cs)] = _load_bank(path)
                print(f"[dagger] bank stance {int(cs)}: "
                      f"{self._landing_banks[int(cs)]['n']} landings from {path}")
        elif self.icfg.rsi_landing_bank and self.icfg.milestone_focus_stance is not None:
            fs = int(self.icfg.milestone_focus_stance)
            self._landing_banks[fs] = _load_bank(self.icfg.rsi_landing_bank)
            print(f"[dagger] bank stance {fs}: {self._landing_banks[fs]['n']} "
                  f"landings from {self.icfg.rsi_landing_bank}")
        self._target_stance: int = 1
        self._milestone_step: int = 0
        self._pelvis_y0: float = 0.0
        self._prev_com_z: float = 0.0
        # Goal-potential state: previous (possibly velocity-augmented) distance
        # to the current stance goal (None = no baseline yet), previous com for
        # the finite-difference velocity, and the latched attractor value inside
        # the final band (None = outside the band).
        self._prev_goal_d: Optional[float] = None
        self._prev_goal_com: Optional[np.ndarray] = None
        self._goal_att_latch: Optional[float] = None
        if self.icfg.stance_milestone and self.icfg.goal_k > 0:
            _legacy = {k: getattr(self.icfg, k) for k in
                       ("mover_reach_coeff", "mover_reach_abs_coeff",
                        "mover_capture_coeff", "posture_rise_coeff")
                       if getattr(self.icfg, k)}
            if _legacy:
                print(f"[goal-potential] goal_k={self.icfg.goal_k} supersedes "
                      f"legacy milestone terms {sorted(_legacy)} — they are "
                      f"IGNORED. Pass --goal-k 0 to use them instead.")
        # Elbow qpos addresses + max angle, for the arm-bend (anti-lean) reward.
        _m = self.env.world.model
        self._elbow_qadr = [int(_m.jnt_qposadr[_m.joint(n).id]) for n in ("l_elbow", "r_elbow")]
        self._elbow_max = float(_m.jnt_range[_m.joint("l_elbow").id, 1]) or 2.618
        # AMP discriminator (CPU, forward-only in the env). Two modes:
        #  - amp_online: a fresh container whose weights the trainer broadcasts
        #    via set_amp_disc() at training start and after every PPO rollout
        #    (the real adversarial loop). Until the first broadcast it returns
        #    the neutral ~0.75 reward — harmless for a few steps.
        #  - amp_disc_path (legacy): a frozen checkpoint. Kept for replay of
        #    old runs; known-placebo as a training signal (see ImitationConfig).
        self._amp_disc = None
        if self.icfg.amp_online:
            from sim3d.amp import AMPDiscriminator
            self._amp_disc = AMPDiscriminator()
            self._amp_disc.eval()
        elif self.icfg.amp_disc_path:
            try:
                import torch
                from sim3d.amp import AMPDiscriminator
                ckpt = torch.load(self.icfg.amp_disc_path, map_location="cpu")
                disc = AMPDiscriminator(ckpt["input_dim"], ckpt["hidden"])
                # strict=False: pre-normalisation checkpoints lack in_mean/in_std.
                disc.load_state_dict(ckpt["state_dict"], strict=False)
                disc.eval()
                self._amp_disc = disc
                print(f"[AMP] discriminator loaded from {self.icfg.amp_disc_path}")
            except Exception as e:
                print(f"[AMP] WARNING: could not load discriminator: {e}")
        # Rolling window of encoded states; a pair spans amp_pair_stride control
        # steps so its Δt matches the 10 fps clip pairs (see ImitationConfig).
        from collections import deque
        self._amp_states: deque = deque(
            maxlen=max(1, int(self.icfg.amp_pair_stride)) + 1)

    def set_amp_disc(self, state: dict) -> None:
        """Load broadcast discriminator weights (numpy state dict from
        AMPDiscriminator.state_numpy()). Called by the trainer through
        VecEnv.env_method after each discriminator update."""
        if self._amp_disc is not None:
            self._amp_disc.load_state_numpy(state)

    def _compute_stance_frames(self) -> list[int]:
        """The settled (last-dwell) frame index of each stance, from
        move_starts. Stance m spans [move_starts[m], move_starts[m+1]); its
        settled frame is the last one in that span."""
        ms = list(self.ref.meta.get("move_starts") or [0])
        ends = [int(ms[i + 1]) - 1 for i in range(len(ms) - 1)] + [len(self.ref) - 1]
        return ends

    def _stance_mover(self, m: int) -> tuple[Optional[str], Optional[np.ndarray]]:
        """The limb whose hold differs between stance m-1 and stance m, and that
        hold's world position. Sets self._mover_hold_id as a side effect."""
        self._mover_hold_id = None
        if m < 1 or m >= len(self._stance_frames):
            return None, None
        g_prev = self.ref.frame_grips(self._stance_frames[m - 1])
        g_now = self.ref.frame_grips(self._stance_frames[m])
        for limb in LIMBS:
            if g_now.get(limb) and g_now.get(limb) != g_prev.get(limb):
                meta = self.env.world._hold_meta_by_id.get(g_now[limb])
                if meta:
                    self._mover_hold_id = g_now[limb]
                    return limb, np.array(meta["world_pos"], dtype=np.float32)
        return None, None

    def _patch_mover_obs(self, obs: np.ndarray) -> np.ndarray:
        """Replace the free mover limb's goal slot (obs[118:130]) with the vector
        to the actual mover target hold.  In imitate task_mode the inner env puts
        (nearest-reachable-hold − tip) there, which is usually the wrong hold and
        gives the policy no signal about WHERE to reach.  Patching here makes the
        obs contract identical to reach-one for the mover limb."""
        # Posture stances have no mover limb (grips == predecessor), so every
        # goal slot is zero and the policy is blind to the com target it must park
        # at. Write (goal_com − com) into a fixed slot — mirrors the reward's
        # posture goal (self.ref.com[target_frame] vs world com), so the obs the
        # policy reads matches the gradient it is graded on. Gated by
        # posture_goal_obs; recorded in env_mode so a mismatched eval/record errors.
        if (self.icfg.posture_goal_obs
                and self._target_stance in self._posture_stances):
            goal_com = np.asarray(
                self.ref.com[self._stance_frames[self._target_stance]], dtype=np.float32)
            com = np.asarray(self.env.world.com(), dtype=np.float32)
            obs = obs.copy()
            slot = 118 + self._POSTURE_GOAL_SLOT * 3
            obs[slot:slot + 3] = goal_com - com
            return obs
        if (self._mover_limb is None or self._mover_hold_pos is None
                or self.env.world.on_hold(self._mover_limb)):
            return obs
        limb_idx = LIMBS.index(self._mover_limb)
        slot = 118 + limb_idx * 3
        tip = self.env.world.limb_tip_pos(self._mover_limb)
        obs = obs.copy()
        obs[slot:slot + 3] = self._mover_hold_pos - tip
        return obs

    def _phase_value(self) -> float:
        """Normalized progress through the reference's move sequence, in [0,1].
        This is the phase-conditioning signal: which move the policy is on, so a
        gradient update for a late move doesn't overwrite the shared
        representation the early moves depend on (the composition-collapse fix).
        Constant 0 for single-move dense imitation (harmless). Uses a FIXED
        denominator (icfg.phase_denom) so the same move keeps the same phase
        across ref lengths — required for clean incremental-ladder warm-starts."""
        denom = max(1.0, float(self.icfg.phase_denom))
        if self.icfg.stance_milestone or self.icfg.sequential_chain:
            return float(np.clip((self._target_stance - 1) / denom, 0.0, 1.0))
        if self.icfg.chain_stages:
            return float(np.clip((self._chain_stage - 1) / denom, 0.0, 1.0))
        return 0.0

    def _append_phase(self, obs: np.ndarray) -> np.ndarray:
        """Append the move-phase scalar (phase_obs mode only). No-op otherwise,
        so the default (131,) obs is byte-identical. Called exactly once at each
        reset/step return so the appended dim is never doubled."""
        if not self.icfg.phase_obs:
            return obs
        return np.concatenate(
            [obs, np.array([self._phase_value()], dtype=np.float32)]
        ).astype(np.float32)

    def _compute_mover_target(self) -> tuple[Optional[str], Optional[np.ndarray]]:
        """Return (limb, hold_world_pos) for the current stage's mover limb, or (None, None).
        Side-effect: sets self._mover_hold_id to the target hold's id (or None)."""
        self._mover_hold_id = None
        if not self.icfg.chain_stages or (
                self.icfg.mover_reach_coeff == 0.0
                and not self.icfg.free_mover_imitation):
            return None, None
        stage_idx = min(self._chain_stage - 1, len(self._chain_bounds) - 1)
        bf = self._chain_bounds[stage_idx]
        prev_bf = self._chain_bounds[stage_idx - 1] if stage_idx > 0 else 0
        g_now = self.ref.frame_grips(bf)
        g_prev = self.ref.frame_grips(prev_bf)
        for limb in LIMBS:
            if g_now.get(limb) is not None and g_prev.get(limb) != g_now.get(limb):
                hold_id = g_now[limb]
                meta = self.env.world._hold_meta_by_id.get(hold_id)
                if meta:
                    self._mover_hold_id = hold_id
                    return limb, np.array(meta['world_pos'], dtype=np.float32)
        return None, None

    def r_min(self) -> float:
        frac = min(1.0, self._total_steps / max(1, self.icfg.r_min_decay_steps))
        return self.icfg.r_min_start + frac * (self.icfg.r_min_end - self.icfg.r_min_start)

    def rsi_cap(self) -> Optional[int]:
        """Current RSI start-phase cap. None = uniform. With annealing it walks
        from len(ref) (uniform) down to rsi_phase_max (forced near-bottom)."""
        tgt = self.icfg.rsi_phase_max
        if tgt is None or self.icfg.rsi_anneal_steps <= 0:
            return tgt
        frac = min(1.0, self._total_steps / self.icfg.rsi_anneal_steps)
        return int(round(len(self.ref) + frac * (tgt - len(self.ref))))

    def _grips_match(self, t: int, *, exact: bool = False) -> bool:
        """Grip-match test for reference frame ``t``. STANDARD (proximity-
        equivalent) criterion: every limb the reference grips must be gripped on
        a hold whose CENTRE is within GRIP_PROXIMITY_M (0.08 m) of the reference
        hold's centre. This makes two adjacent holds closer than the grip radius
        interchangeable — a foot on a hold 5 cm from the authored one achieves the
        same stance, so the climb is scored on physical intent, not hold-id
        identity (s1297: RF grips fd_042 vs authored fd_044, same column 5 cm
        apart). ``exact=True`` requires the identical hold id — used only for the
        logged exact-match DIAGNOSTIC, never as the success criterion."""
        ref_g = self.ref.frame_grips(min(t, len(self.ref) - 1))
        holds = self.env.world._hold_meta_by_id
        for l, h in ref_g.items():
            if h is None:
                continue
            gh = self.env.world.on_hold(l)
            if gh is None:
                return False
            if gh == h:
                continue
            if exact:
                return False
            if gh not in holds or h not in holds:
                return False
            d = float(np.linalg.norm(np.asarray(holds[gh]["world_pos"])
                                     - np.asarray(holds[h]["world_pos"])))
            if d > cfg.GRIP_PROXIMITY_M:
                return False
        return True

    def _stance_reached(self, target_frame: int) -> bool:
        """Success test for the current target stance. Grip-match everywhere;
        posture stances (same grips as predecessor) additionally require the com
        to sit within posture_com_tol of the reference for posture_hold_steps
        consecutive steps — otherwise they'd complete trivially at step 1."""
        if not self._grips_match(target_frame):
            return False
        if self._target_stance not in self._posture_stances:
            return True
        com_gap = float(np.linalg.norm(self.env.world.com() - self.ref.com[target_frame]))
        self._posture_hold = self._posture_hold + 1 if com_gap < self.icfg.posture_com_tol else 0
        return self._posture_hold >= self.icfg.posture_hold_steps

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if self.icfg.stance_milestone:
            # RSI to a uniformly-sampled stance c and reach c+1. Uniform c gives
            # every transition equal on-policy gradient (the RSI chaining fix);
            # because we RSI to the TRUE stance c, each transition is trainable
            # independently — no stage curriculum needed.
            n = len(self._stance_frames)
            # Sequential chain: always start at the BOTTOM stance and climb up,
            # advancing the target on each grip WITHOUT reset (see _step_stance).
            # This trains move k+1 from move k's ACTUAL landing (the on-policy
            # distribution) — fixing the composition gap where per-move skills
            # trained from fixed authored stances don't chain.
            if self._landing_banks and self.icfg.milestone_focus_stance is None:
                # Multi-bank: oversample the banked (foot) stances so both foot moves
                # train from real landings; otherwise sample uniformly over all moves.
                if self.np_random.random() < self.icfg.milestone_focus_frac:
                    c = int(self.np_random.choice(sorted(self._landing_banks.keys())))
                else:
                    c = int(self.np_random.integers(0, max(1, n - 1)))
            elif (self.icfg.milestone_focus_stance is not None
                    and self.np_random.random() < self.icfg.milestone_focus_frac):
                c = int(np.clip(self.icfg.milestone_focus_stance, 0, max(0, n - 2)))
            elif self.icfg.sequential_chain:
                c = 0
            else:
                c = int(self.np_random.integers(0, max(1, n - 1)))
            self._target_stance = c + 1
            self._milestone_step = 0
            self._posture_hold = 0
            self._prev_mover_gap = float("inf")
            self._prev_goal_d = None
            self._prev_goal_com = None
            self._goal_att_latch = None
            self._mover_grip_awarded = False
            self._mover_limb, self._mover_hold_pos = self._stance_mover(c + 1)
            # DAgger: for the focus stance, RSI into a COLLECTED real landing (the
            # policy's own post-move state distribution) instead of the authored
            # stance keyframe — closes the distribution-shift gap that leaves the
            # composed foot move parked short of its grip. Every other stance still
            # RSIs to the authored frame, so those moves stay trained.
            if c in self._landing_banks:
                bank = self._landing_banks[c]
                bi = int(self.np_random.integers(0, bank["n"]))
                q0, qv0 = bank["qpos"][bi], bank["qvel"][bi]
                grips0 = {LIMBS[k]: (s if s else None)
                          for k, s in enumerate(bank["grips"][bi])}
            else:
                t = self._stance_frames[c]
                q0, qv0, grips0 = self.ref.qpos[t], self.ref.qvel[t], self.ref.frame_grips(t)
            obs, info = self.env.reset_to_reference(
                q0, qv0, grips0, settle_frames=self.icfg.settle_frames,
            )
            obs = self._patch_mover_obs(obs)
            self._pelvis_y0 = float(self.env.world.pelvis_pos()[1])
            self._prev_com_z = float(self.env.world.com()[2])
            self._ep_com_y_sum = 0.0
            self._ep_com_y_n = 0
            self._amp_states.clear()
            info.update(self._info(r_imit=1.0))
            info["target_stance"] = self._target_stance
            return self._append_phase(obs), info
        if self.icfg.chain_stages:
            stage_end = self._chain_bounds[min(self._chain_stage - 1,
                                               len(self._chain_bounds) - 1)]
            # Half the episodes RSI into the middle of the stage window —
            # mid-move starts are what teach the move's dynamics (a frame-0
            # policy parks a foot 27 cm from its target forever; exploration
            # alone never finds a coordinated swing into an 8 cm window).
            # Only frame-0 episodes count toward stage advancement, so the
            # "k real moves from the ground" bar is unaffected.
            if self.np_random.random() >= self.icfg.chain_mid_frac:
                self._phase = 0
            else:
                if self.icfg.chain_rsi_at_stage_start:
                    # RSI at or near the PRIOR stage boundary. For stage 1 this
                    # is frame 0 (ground start). For stage k≥2 we spread across
                    # the free-swing window [prev_end, first_grip_frame) so that
                    # some episodes start with the mover mid-flight — the mover
                    # arrives at the target under its reference velocity and
                    # falls back through the grip window in the first few steps,
                    # giving the policy easy grip completions to bootstrap from.
                    # This is a within-stage reverse curriculum: learn "close the
                    # last few cm" first, then backprop to teach the full swing.
                    prev_end = (0 if self._chain_stage == 1
                                else self._chain_bounds[self._chain_stage - 2])
                    swing_hi = prev_end
                    if self._chain_stage >= 2:
                        # Walk forward until the first frame where ALL limbs
                        # that were free at prev_end become gripped again.
                        grips_prev = self.ref.frame_grips(prev_end)
                        free_at_start = {l for l in ("LH", "RH", "LF", "RF")
                                         if grips_prev.get(l) is None}
                        for f in range(prev_end + 1, min(prev_end + 30, stage_end)):
                            grips_f = self.ref.frame_grips(f)
                            if any(grips_f.get(l) is None for l in free_at_start):
                                swing_hi = f
                            else:
                                break
                    self._phase = int(self.np_random.integers(prev_end, swing_hi + 1))
                elif self.icfg.chain_rsi_before_stage and self._chain_stage >= 2:
                    # Restrict RSI to before the current stage's active window
                    # so RSI can't auto-grip the mover limb from the reference
                    # state (which inflates phase-avg success without the policy
                    # executing the move — observed as a 27% "free" baseline in
                    # stage-3 RF high-step training with default RSI).
                    prev_end = self._chain_bounds[self._chain_stage - 2]
                    hi0 = max(1, prev_end)
                    self._phase = int(self.np_random.integers(0, hi0))
                else:
                    hi0 = max(1, stage_end - self.icfg.min_episode_frames)
                    self._phase = int(self.np_random.integers(0, hi0))
            self._chain_from_zero = (self._phase == 0)
            self._intermediate_awarded = set()
            self._prev_mover_gap = float('inf')
            self._mover_grip_awarded = False
            self._mover_limb, self._mover_hold_pos = self._compute_mover_target()
            t = self._phase
            obs, info = self.env.reset_to_reference(
                self.ref.qpos[t], self.ref.qvel[t], self.ref.frame_grips(t),
                settle_frames=self.icfg.settle_frames,
            )
            obs = self._patch_mover_obs(obs)
            info.update(self._info(r_imit=1.0))
            info["chain_stage"] = self._chain_stage
            return self._append_phase(obs), info
        T = len(self.ref)
        hi = max(1, T - self.icfg.min_episode_frames)
        cap = self.rsi_cap()
        if cap is not None and cap + 1 < hi and self.icfg.rsi_anneal_steps > 0:
            # Mixed sampling (TRAINING only — gated on an active anneal): half
            # the episodes start within the annealed cap (extra gradient at
            # the bottom — the part only ever reached from frame 0), half stay
            # uniform over the whole reference. A pure cap→0 anneal makes late
            # moves vanish from the training distribution and the policy
            # FORGETS them (observed: r_imit declining 0.62→0.58 as the cap
            # passed 50 on an 11-move reference). Completing from the bottom
            # needs both: a strong start AND the rest of the climb intact.
            # Evaluation passes rsi_anneal_steps=0 + a fixed cap and must get
            # EXACTLY that cap — a mixture there silently dilutes the
            # frame-0 metric with easy random starts.
            if self.np_random.random() < 0.5:
                self._phase = int(self.np_random.integers(0, cap + 1))
            else:
                self._phase = int(self.np_random.integers(0, hi))
        else:
            if cap is not None:
                hi = min(hi, cap + 1)
            self._phase = int(self.np_random.integers(0, hi))
        t = self._phase
        obs, info = self.env.reset_to_reference(
            self.ref.qpos[t], self.ref.qvel[t], self.ref.frame_grips(t),
            settle_frames=self.icfg.settle_frames,
        )
        info.update(self._info(r_imit=1.0))
        return self._append_phase(obs), info

    def step(self, action):
        if self.icfg.stance_milestone:
            return self._step_stance(action)
        # v1 — drive the 4 grip intents from the reference's contacts at the
        # frame we're tracking toward (+1 hold, −1 release), so the policy learns
        # joint tracking without the std≈0.22 grip-intent noise randomly dropping
        # anchors (the NEXT_STEPS 0.4 foot-gun). The inner env's engage logic
        # no-ops on already-held anchors and releases the limb the reference
        # frees. Policy-controlled grips (rewarded for matching the contact
        # schedule) are a follow-up. The policy's own grip outputs are ignored.
        action = np.asarray(action, dtype=np.float32).copy()
        t_next = min(self._phase + 1, len(self.ref) - 1)
        ref_grips = self.ref.frame_grips(t_next)
        for i, limb in enumerate(LIMBS):
            action[len(action) - 4 + i] = 1.0 if ref_grips[limb] is not None else -1.0
        # Reference-driven grips engage the REFERENCE's hold only — without
        # this, "nearest eligible within 8 cm" re-welds a just-released limb
        # to its origin before the move can start (see env._maybe_engage_grip).
        self.env._grip_target_override = dict(ref_grips)

        # Apply the action; the inner env advances physics + handles obs, grips,
        # and the physical-fall backstop. Its reward is ignored.
        obs, _r_inner, fell, _trunc_inner, info = self.env.step(action)
        obs = self._patch_mover_obs(obs)

        self._phase += 1
        self._total_steps += 1
        t = min(self._phase, len(self.ref) - 1)
        # Free the mover from pose/endeff tracking while it's ungripped (mid-reach):
        # the recorded swing is an authoring artifact the servos can't reproduce.
        free_limb = None
        if self.icfg.free_mover_imitation:
            if (self._mover_limb is not None
                    and not self.env.world.on_hold(self._mover_limb)):
                free_limb = self._mover_limb
            elif self._mover_limb is None:
                # No chain mover (e.g. eval_frame0 runs chain_stages=False, so
                # _compute_mover_target returns None): free whichever limb is
                # currently ungripped in the ENV — it is mid-transition (swinging
                # toward its next hold, or released). Key off the env's grip state,
                # NOT the reference's grip schedule: stance-keyframe references
                # (discover --stances) flip the grip label to the TARGET hold ~1
                # frame after release, so keying off the reference stops freeing
                # the limb while it is still mid-swing — cratering r_imit and
                # cutting the headline frame-0 eval (observed: RH eval cut at
                # frame 3 though the grip physically closes at ~frame 5). Once the
                # limb grips, on_hold() is truthy and its pose is tracked again —
                # matching the chain-mover behaviour above.
                for _l in LIMBS:
                    if not self.env.world.on_hold(_l):
                        free_limb = _l
                        break
        r_imit, comp = imitation_reward(self.env.world, self.ref, t, self.icfg.coeffs,
                                        free_limb=free_limb)
        reward = (1.0 - self.icfg.w_task) * r_imit + self.icfg.w_task * self._task_reward(t)

        terminated = bool(fell)
        outcome = info.get("outcome", "")
        rmin = self.r_min()
        if not terminated and r_imit < rmin:
            terminated = True
            outcome = "off-reference"            # termination curriculum cut

        # Intermediate grip bonuses: for stage k>1, reward crossing each
        # prior boundary with the correct grips (once per boundary per episode).
        # Without this the policy can coast past earlier grip transitions while
        # tracking r_imit loosely — observed: stage-3 training left only RH
        # gripping at frame 84 because the LF-step grip was never checked.
        if (self.icfg.chain_stages and not terminated
                and self.icfg.intermediate_bonus_frac > 0
                and self._chain_stage > 1):
            ibfrac = self.icfg.intermediate_bonus_frac
            for bi in range(self._chain_stage - 1):
                bf = self._chain_bounds[bi]
                if self._phase >= bf and bi not in self._intermediate_awarded:
                    self._intermediate_awarded.add(bi)
                    if self._grips_match(bf):
                        reward += self.icfg.completion_bonus * ibfrac

        # Dense reach bonus for the current stage's mover limb: reward
        # (prev_gap - curr_gap) × coeff when the mover is ungripped.
        # Breaks "hold limb still" local optima where r_imit ≈ 0.50 is
        # achievable without the limb ever approaching its target hold.
        if (self.icfg.mover_reach_coeff > 0 and not terminated
                and self._mover_limb is not None
                and self._mover_hold_pos is not None
                and not self.env.world.on_hold(self._mover_limb)):
            tip = self.env.world.limb_tip_pos(self._mover_limb)
            curr_gap = float(np.linalg.norm(tip - self._mover_hold_pos))
            if self._prev_mover_gap < float('inf'):
                reward += self.icfg.mover_reach_coeff * (self._prev_mover_gap - curr_gap)
            self._prev_mover_gap = curr_gap

        # Sparse one-shot outcome bonus: the mover gripped its target hold.
        # Makes committing the full swing worth the risk vs parking short.
        if (self.icfg.mover_grip_bonus > 0 and not terminated
                and self._mover_limb is not None and not self._mover_grip_awarded
                and self.env.world.on_hold(self._mover_limb) == self._mover_hold_id):
            self._mover_grip_awarded = True
            reward += self.icfg.mover_grip_bonus

        # Dense capture bonus: fires every step the mover tip is within the
        # grip radius of its target hold (but hasn't gripped yet). Adds a
        # gradient the potential-based mover_reach_coeff can't supply: once
        # the tip is stationary Δdist = 0 and that reward is zero; this term
        # keeps pulling toward the hold centre throughout the capture sphere.
        if (self.icfg.mover_capture_coeff > 0 and not terminated
                and self._mover_limb is not None
                and self._mover_hold_pos is not None
                and not self.env.world.on_hold(self._mover_limb)):
            tip = self.env.world.limb_tip_pos(self._mover_limb)
            gap = float(np.linalg.norm(tip - self._mover_hold_pos))
            cap_r = self.icfg.mover_capture_radius or cfg.GRIP_PROXIMITY_M
            if gap < cap_r:
                reward += self.icfg.mover_capture_coeff * (1.0 - gap / cap_r)

        # Per-step grip-retention penalty for non-mover limbs: fires each step
        # a limb the reference keeps gripped has slipped. Directly addresses the
        # "lose LH during foot step" local optimum where the diluted endeff
        # signal (split over 4 limbs) fails to prevent stage-3 regressions.
        if self.icfg.grip_retention_coeff > 0 and not terminated:
            ref_g = self.ref.frame_grips(min(self._phase, len(self.ref) - 1))
            for limb, hold_id in ref_g.items():
                if hold_id is None:
                    continue
                if limb == self._mover_limb:
                    continue
                if self.env.world.on_hold(limb) != hold_id:
                    reward -= self.icfg.grip_retention_coeff

        # Chain curriculum: stage k's episode ends at move k's boundary;
        # success = the reference's stance there is actually held.
        if self.icfg.chain_stages and not terminated:
            stage_end = self._chain_bounds[min(self._chain_stage - 1,
                                               len(self._chain_bounds) - 1)]
            if self._phase >= stage_end:
                terminated = True
                ok = self._grips_match(stage_end)
                outcome = "completed" if ok else "end-grip-mismatch"
                if ok:
                    reward += self.icfg.completion_bonus
                if getattr(self, "_chain_from_zero", True):
                    self._chain_results.append(ok)
                    if len(self._chain_results) > self.icfg.chain_window:
                        self._chain_results.pop(0)
                    full = len(self._chain_results) >= self.icfg.chain_window
                    rate = (sum(self._chain_results) / len(self._chain_results)
                            if self._chain_results else 0.0)
                    if (full and rate >= self.icfg.chain_advance_rate
                            and self._chain_stage < len(self._chain_bounds)):
                        self._chain_stage += 1
                        self._chain_results.clear()
        if (self.icfg.chain_stages and terminated and outcome not in ("completed",)
                and getattr(self, "_chain_from_zero", True)):
            # ground-start falls / off-reference count against the rolling rate
            self._chain_results.append(False)
            if len(self._chain_results) > self.icfg.chain_window:
                self._chain_results.pop(0)

        if not terminated and self._phase >= len(self.ref) - 1:
            terminated = True
            # "Completed" must mean the climb actually happened, not merely
            # that r_imit survived the window: every limb the reference ends
            # gripped must be gripped on the SAME hold. (A degenerate
            # wiggle-in-place reference exposed that the window-survival
            # definition alone can score 100% without climbing.)
            outcome = ("completed" if self._grips_match(len(self.ref) - 1)
                       else "end-grip-mismatch")
            if outcome == "completed":
                reward += self.icfg.completion_bonus

        info["outcome"] = outcome
        info["is_success"] = (outcome == "completed")
        if self.icfg.chain_stages:
            info["chain_stage"] = self._chain_stage
        info.update(self._info(r_imit=r_imit, comp=comp, rmin=rmin))
        return self._append_phase(obs), float(reward), terminated, False, info

    def _step_stance(self, action):
        """One step of stance-milestone mode: pose-attractor toward the next
        stance keyframe; the policy discovers the balancing transition (RL owns
        balance — discover only authored the stances)."""
        action = np.asarray(action, dtype=np.float32).copy()
        n = self.env._n_act
        target_frame = self._stance_frames[self._target_stance]
        target_grips = self.ref.frame_grips(target_frame)
        mover = self._mover_limb

        # Grip driving: anchors (every limb the target stance grips except the
        # mover) are force-held; the mover releases while far and engages once
        # within proximity of its target hold. Override restricts each engage to
        # the intended hold (else "nearest within 8 cm" re-welds the mover to its
        # origin before it can move — env._maybe_engage_grip).
        override: dict = {}
        for i, limb in enumerate(LIMBS):
            hold = target_grips.get(limb)
            if limb == mover:
                gap = float("inf")
                if self._mover_hold_pos is not None:
                    gap = float(np.linalg.norm(
                        self.env.world.limb_tip_pos(limb) - self._mover_hold_pos))
                action[n + i] = 1.0 if gap < cfg.GRIP_PROXIMITY_M else -1.0
                if hold:
                    override[limb] = hold
            elif hold:
                action[n + i] = 1.0          # anchor: force-hold
                override[limb] = hold
            else:
                action[n + i] = -1.0
        self.env._grip_target_override = override

        obs, _r_inner, fell, _trunc, info = self.env.step(action)
        obs = self._patch_mover_obs(obs)
        self._milestone_step += 1
        self._total_steps += 1
        self._ep_com_y_sum += float(self.env.world.com()[1])
        self._ep_com_y_n += 1

        # Pose attractor to the NEXT stance. With --free-mover-imitation the mover
        # is freed from pose/endeff while ungripped (its swing is the policy's to
        # discover). WITHOUT the flag the mover is TRACKED toward the target
        # stance's pose — a dense exp(-k*err) pull the potential-based
        # mover_reach can't supply, needed to bootstrap a foot lift from rest when
        # the target stance pose is an authored, reachable high-step.
        free_limb = (mover if (self.icfg.free_mover_imitation and mover is not None
                               and not self.env.world.on_hold(mover)) else None)
        r_imit, comp = imitation_reward(self.env.world, self.ref, target_frame,
                                        self.icfg.coeffs, free_limb=free_limb)
        is_posture = self._target_stance in self._posture_stances
        # Posture stances keep the full pose+com attractor regardless of w_task
        # (mover is None; the stance itself is the target).
        base_att = r_imit if is_posture else (1.0 - self.icfg.w_task) * r_imit
        goal_d: Optional[float] = None

        if self.icfg.goal_k > 0:
            # Goal-potential reward (see the ImitationConfig design note): one
            # explicit goal point per stance, a signed potential active from
            # spawn to the success tolerance, and the attractor frozen (latched)
            # inside the final band so its saturated gradient can't compete.
            if is_posture:
                goal = np.asarray(self.ref.com[target_frame], dtype=np.float64)
                tracked = np.asarray(self.env.world.com(), dtype=np.float64)
            elif mover is not None and self._mover_hold_pos is not None:
                goal = np.asarray(self._mover_hold_pos, dtype=np.float64)
                tracked = np.asarray(self.env.world.limb_tip_pos(mover),
                                     dtype=np.float64)
            else:
                goal = tracked = None
            att = base_att
            if goal is not None:
                goal_d = float(np.linalg.norm(tracked - goal))
                # Posture stances: velocity-augmented metric so the potential
                # bottoms out only at the goal AT REST (see goal_vel_lambda).
                # Band gating and the success criterion stay position-based.
                if is_posture and self.icfg.goal_vel_lambda > 0:
                    if self._prev_goal_com is not None:
                        v = float(np.linalg.norm(tracked - self._prev_goal_com)) / 0.016
                        goal_pot_d = goal_d + self.icfg.goal_vel_lambda * v
                    else:
                        goal_pot_d = None   # no velocity baseline yet
                    self._prev_goal_com = tracked.copy()
                else:
                    goal_pot_d = goal_d
                if goal_d < self.icfg.goal_band:
                    # FLOOR, not a hard freeze (refined after goalpot_rise1,
                    # 2026-07-09): attractor saturation is a function of POSE
                    # error, not goal distance — a stand-up enters the com band
                    # with r_imit ~0.22, far from saturated, and a hard freeze
                    # there removes the very signal that teaches the settled,
                    # HOLDABLE posture (observed: the policy swung the com
                    # through the tolerance in a contorted pose and drifted
                    # back out; det min gap regressed 0.065→0.081 over 200k).
                    # max(entry, live): income never drops below band entry
                    # (no perverse outward pull; a genuinely saturated
                    # attractor degenerates to the constant freeze), while
                    # settling into the reference pose still pays. Pose
                    # oscillation inside the band nets zero income change.
                    if self._goal_att_latch is None:
                        self._goal_att_latch = base_att
                    att = max(self._goal_att_latch, base_att)
                else:
                    self._goal_att_latch = None
            reward = att
            if goal_d is not None and goal_pot_d is not None:
                if self._prev_goal_d is not None:
                    reward += self.icfg.goal_k * (self._prev_goal_d - goal_pot_d)
                self._prev_goal_d = goal_pot_d
        else:
            # ── Legacy milestone shaping (pre-goal-potential), via --goal-k 0 ──
            reward = base_att
            if is_posture and self.icfg.posture_rise_coeff > 0:
                # Dense com-RISE pull toward the reference stance height.
                dz = max(0.0, float(self.ref.com[target_frame][2])
                         - float(self.env.world.com()[2]))
                reward += self.icfg.posture_rise_coeff * max(
                    0.0, 1.0 - dz / self.icfg.posture_rise_band)

            # Optional dense, potential-based mover-reach shaping (net-zero on retreat).
            if (self.icfg.mover_reach_coeff > 0 and mover is not None
                    and self._mover_hold_pos is not None
                    and not self.env.world.on_hold(mover)):
                gap = float(np.linalg.norm(
                    self.env.world.limb_tip_pos(mover) - self._mover_hold_pos))
                if self._prev_mover_gap < float("inf"):
                    reward += self.icfg.mover_reach_coeff * (self._prev_mover_gap - gap)
                self._prev_mover_gap = gap
                cap_r = self.icfg.mover_capture_radius or cfg.GRIP_PROXIMITY_M
                if (self.icfg.mover_capture_coeff > 0 and gap < cap_r):
                    reward += self.icfg.mover_capture_coeff * (1.0 - gap / cap_r)

            # Dense absolute reach pull (goal-reaching): continuous gradient from
            # rest over the full radius, so the mover starts moving toward its
            # hold even when stationary. The mover is freed from pose tracking
            # (free_limb above), so THIS is its main signal.
            if (self.icfg.mover_reach_abs_coeff > 0 and mover is not None
                    and self._mover_hold_pos is not None
                    and not self.env.world.on_hold(mover)):
                gap = float(np.linalg.norm(
                    self.env.world.limb_tip_pos(mover) - self._mover_hold_pos))
                reward += self.icfg.mover_reach_abs_coeff * max(
                    0.0, 1.0 - gap / self.icfg.mover_reach_radius)

        # Naturalness shaping. Anti-lean: penalize TORSO TILT from upright — the
        # visible "leaning back" is a ~15-26 deg pelvis PITCH, not hip sag (the hips
        # actually hug the wall). upz = pelvis-up·world-up = 1-2(qx²+qy²); (1-upz) is
        # 0 upright, grows with any tilt. com-rise: reward upward com motion so the
        # body pulls UP over its holds (net ascent); potential ⇒ un-farmable by bobbing.
        if self.icfg.lean_penalty_coeff > 0:
            q = self.env.world.data.qpos[3:7]   # free-joint quat, wxyz
            upz = 1.0 - 2.0 * (float(q[1]) ** 2 + float(q[2]) ** 2)
            reward -= self.icfg.lean_penalty_coeff * max(0.0, 1.0 - upz)
        if self.icfg.com_rise_coeff > 0:
            com_z = float(self.env.world.com()[2])
            reward += self.icfg.com_rise_coeff * (com_z - self._prev_com_z)
            self._prev_com_z = com_z
        if self.icfg.vel_penalty_coeff > 0:
            # SUM (not mean) of joint-speed²: only the mover swings fast, so a mean
            # over all 23 joints dilutes it ~23× and the penalty vanishes. Sum keeps
            # the snap's spike intact; by ∫v²dt a 2-step snap costs ~15× a slow move
            # of the same reach, so this favors slow without a giant coeff.
            reward -= self.icfg.vel_penalty_coeff * float(
                np.sum(self.env.world.data.qvel[6:] ** 2))
        if self.icfg.arm_bend_coeff > 0:
            # (proxy — found to game: bends elbows without pulling in; prefer wall_hug)
            qp = self.env.world.data.qpos
            elbow = 0.5 * (float(qp[self._elbow_qadr[0]]) + float(qp[self._elbow_qadr[1]]))
            reward += self.icfg.arm_bend_coeff * max(0.0, elbow) / self._elbow_max
        if self.icfg.wall_hug_coeff > 0 and (
                self.icfg.wall_hug_mover is None
                or self._mover_limb == self.icfg.wall_hug_mover):
            # Pull the body IN to the wall (the real anti-lean fix). Penalize com
            # past the wall-hug target → body over the feet, arms bend naturally.
            # Gated to wall_hug_mover (e.g. RF) so it doesn't fight the other moves.
            reward -= self.icfg.wall_hug_coeff * max(
                0.0, float(self.env.world.com()[1]) - self.icfg.wall_hug_target)

        if self._amp_disc is not None:
            from sim3d.amp import encode_state
            self._amp_states.append(encode_state(self.env.world.data.qpos,
                                                 self.env.world.data.qvel))
            if len(self._amp_states) == self._amp_states.maxlen:
                # Pair spans amp_pair_stride control steps ⇒ Δt matches the
                # 10 fps clip pairs the discriminator's real data comes from.
                s0, s1 = self._amp_states[0], self._amp_states[-1]
                r_style = self._amp_disc.reward(s0, s1)
                reward += self.icfg.amp_coeff * r_style
                # Expose the pair so AMPOnlineCallback can train the
                # discriminator against the policy's actual motion.
                info["amp_pair"] = np.concatenate([s0, s1]).astype(np.float32)
                info["r_style"] = round(float(r_style), 3)

        terminated = False
        outcome = ""
        last_stance = len(self._stance_frames) - 1
        if fell:
            terminated, outcome = True, "fell"
        elif self._stance_reached(target_frame):
            reward += self.icfg.completion_bonus
            if self.icfg.sequential_chain and self._target_stance < last_stance:
                # Sequential chain: grip reached, but more moves remain. ADVANCE
                # the target to the next stance WITHOUT resetting the body, so the
                # next move trains from this real landing. Re-arm per-transition
                # state + re-point the mover obs at the new target.
                self._target_stance += 1
                self._milestone_step = 0
                self._posture_hold = 0
                self._prev_mover_gap = float("inf")
                self._prev_goal_d = None
                self._prev_goal_com = None
                self._goal_att_latch = None
                self._mover_grip_awarded = False
                self._mover_limb, self._mover_hold_pos = self._stance_mover(self._target_stance)
                obs = self._patch_mover_obs(obs)
                outcome = "advanced"
            else:
                # Final stance reached (or non-sequential single transition).
                terminated, outcome = True, "completed"
        elif self._milestone_step >= self.icfg.milestone_budget:
            terminated, outcome = True, "timeout"

        if outcome == "completed" and self.icfg.wall_hug_terminal_coeff > 0:
            mean_com_y = self._ep_com_y_sum / max(1, self._ep_com_y_n)
            reward += self.icfg.wall_hug_terminal_coeff * (
                self.icfg.wall_hug_target - mean_com_y)

        info["outcome"] = outcome
        info["is_success"] = (outcome == "completed")
        # Exact-hold-id match DIAGNOSTIC (not the success criterion): did the
        # completing stance also land every limb on the reference's EXACT hold?
        info["is_success_exact"] = (info["is_success"]
                                    and self._grips_match(target_frame, exact=True))
        info["target_stance"] = self._target_stance
        if goal_d is not None:
            info["goal_d"] = round(goal_d, 3)
        info.update(self._info(r_imit=r_imit, comp=comp))
        return self._append_phase(obs), float(reward), terminated, False, info

    def chain_stage(self) -> int:
        return self._chain_stage

    def _task_reward(self, t: int) -> float:
        # v1: pure imitation. (Hook for a top-out/progress term, w_task > 0.)
        return 0.0

    def _info(self, *, r_imit: float, comp: Optional[dict] = None,
              rmin: Optional[float] = None) -> dict:
        d = {"phase": self._phase, "ref_len": len(self.ref), "r_imit": round(float(r_imit), 3)}
        if rmin is not None:
            d["r_min"] = round(float(rmin), 3)
        if comp is not None:
            d.update({f"r_{k}": round(float(v), 3) for k, v in comp.items()})
        return d

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class MultiRefImitationEnv(gym.Env):
    """Train ONE policy on several (wall, reference) pairs at once (Phase 3
    multi-wall imitation). Each pair is its own ``ImitationEnv`` — different walls
    compile different MuJoCo models, so a sub-env per pair is the only correct way
    to hold them; the reference cannot be swapped inside one inner env. On each
    reset a pair is sampled UNIFORMLY and the whole episode runs in that sub-env,
    so every pair gets on-policy gradient. The obs is wall-agnostic already (K=8
    nearest holds in pelvis frame), so all sub-envs share the same obs/action
    space and one policy + one VecNormalize covers them all.

    DAgger banks live inside each sub-env (each ``ImitationEnv`` loads its OWN
    wall's bank via icfg.rsi_landing_bank(s) at construction), so banks are never
    mixed across walls. If only one wall has a bank configured, the others simply
    train without one — the per-wall wall_hash validation in
    ``am.validate_landing_bank`` hard-errors if a bank is ever pointed at the
    wrong wall.
    """

    metadata = Climbing3DEnv.metadata

    def __init__(self, pairs, imitation_config: Optional[ImitationConfig] = None,
                 render_mode: Optional[str] = None):
        super().__init__()
        if not pairs:
            raise ValueError("MultiRefImitationEnv needs at least one (ref, wall, profile) pair")
        self.icfg = imitation_config or ImitationConfig()
        self._subenvs: list[ImitationEnv] = []
        self._wall_ids: list[str] = []
        for ref, wall, profile in pairs:
            # Per-wall DAgger: a landing bank belongs to the wall it was collected
            # on. Give each sub-env only the banks whose recorded wall_id matches
            # its wall — a wall with no matching bank trains WITHOUT one (never an
            # error, never a bank from another wall). ImitationEnv's own
            # am.validate_landing_bank then passes because the wall now matches.
            sub_icfg = self._route_banks_for_wall(self.icfg, wall.wall_id)
            sub = ImitationEnv(ref, wall, profile=profile,
                               imitation_config=sub_icfg, render_mode=render_mode)
            self._subenvs.append(sub)
            self._wall_ids.append(wall.wall_id)
        self.observation_space = self._subenvs[0].observation_space
        self.action_space = self._subenvs[0].action_space
        self._active = 0

    @staticmethod
    def _bank_wall_id(path: str) -> Optional[str]:
        """The wall_id recorded in a landing bank's embedded metadata (None for a
        legacy bank without metadata — which then matches no wall and is dropped)."""
        d = np.load(path, allow_pickle=True)
        meta, _ = am.parse_npz_meta(d["meta"] if "meta" in d.files else None)
        return (meta or {}).get("wall_id")

    @classmethod
    def _route_banks_for_wall(cls, icfg: ImitationConfig, wall_id: str) -> ImitationConfig:
        """Restrict icfg's landing bank(s) to those authored on ``wall_id``. Banks
        for other walls are dropped so this wall trains without them. No-op (returns
        icfg unchanged) when no banks are configured — the common/from-scratch path."""
        changed: dict = {}
        if icfg.rsi_landing_banks:
            kept = [pair for pair in icfg.rsi_landing_banks.split(",")
                    if cls._bank_wall_id(pair.split(":", 1)[1]) == wall_id]
            changed["rsi_landing_banks"] = ",".join(kept) if kept else None
        if icfg.rsi_landing_bank:
            changed["rsi_landing_bank"] = (
                icfg.rsi_landing_bank
                if cls._bank_wall_id(icfg.rsi_landing_bank) == wall_id else None)
        return replace(icfg, **changed) if changed else icfg

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        # Uniform over pairs — every wall gets equal episode share (and thus
        # equal on-policy gradient). np_random advances each reset (seed is None
        # after the first), so the wall varies within a worker.
        self._active = int(self.np_random.integers(0, len(self._subenvs)))
        obs, info = self._subenvs[self._active].reset(seed=seed, options=options)
        info["wall_id"] = self._wall_ids[self._active]
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._subenvs[self._active].step(action)
        info["wall_id"] = self._wall_ids[self._active]
        return obs, reward, terminated, truncated, info

    # ── Delegating hooks the trainer's callbacks call through env_method ──
    def r_min(self) -> float:
        return self._subenvs[self._active].r_min()

    def rsi_cap(self) -> Optional[int]:
        return self._subenvs[self._active].rsi_cap()

    def chain_stage(self) -> int:
        return self._subenvs[self._active].chain_stage()

    def set_amp_disc(self, state: dict) -> None:
        for sub in self._subenvs:
            sub.set_amp_disc(state)

    def render(self):
        return self._subenvs[self._active].render()

    def close(self):
        for sub in self._subenvs:
            sub.close()


# ─── Evaluation ──────────────────────────────────────────────────────────────
# Frame-0 success is THE headline metric. Phase-averaged success under uniform
# RSI inflates badly (v2 reported 70% while landing 0/4 full climbs — easy
# late-reference starts dominate the average). A climb counts when it is
# executed from the bottom.

def _maybe_enable_phase_obs(model, ref, wall, profile,
                            icfg: ImitationConfig) -> ImitationConfig:
    """Auto-enable phase_obs for replay/eval when the loaded model expects one
    extra obs dim (was trained phase-conditioned) but icfg didn't ask for it.
    Makes --eval / --record foolproof if the user forgets --phase-obs — without
    this they'd hit a confusing obs shape mismatch at predict time."""
    if icfg.phase_obs:
        return icfg
    try:
        probe = ImitationEnv(ref, wall, profile, replace(icfg, phase_obs=False))
        base_dim = int(probe.observation_space.shape[0])
        probe.close()
        if int(model.observation_space.shape[0]) == base_dim + 1:
            print("[phase-obs] model expects a phase dim — enabling phase_obs for replay")
            return replace(icfg, phase_obs=True)
    except Exception as e:  # noqa: BLE001
        print(f"[phase-obs] auto-detect skipped: {e}")
    return icfg


def eval_frame0(model, ref: Reference, wall, profile,
                icfg: Optional[ImitationConfig] = None, *, vec_normalize=None,
                n_episodes: int = 20, deterministic: bool = True) -> dict:
    """Deterministic rollouts starting at reference frame 0; success = tracked
    to the end. Evaluates at the trained R_min floor (not the early-training
    start value) so the termination curriculum doesn't cut a competent policy."""
    base = icfg or ImitationConfig()
    # chain_stages must be OFF here: the eval is the FULL reference from
    # frame 0 with end-grip match — stage windows would silently turn the
    # headline metric into "stage-k success" (observed: a 100% line that
    # meant one move, not the climb).
    overrides = dict(rsi_phase_max=0, rsi_anneal_steps=0, chain_stages=False,
                     r_min_start=base.r_min_end, r_min_end=base.r_min_end)
    if base.stance_milestone:
        # Chain eval: sequential from stance 0, authored frames only. Banks /
        # focus sampling would silently turn this into a per-move average
        # (the mb_dagger4_rfhug 50%/75% mislabel — see NEXT_STEPS 2026-07-02).
        overrides.update(sequential_chain=True, rsi_landing_bank=None,
                         rsi_landing_banks=None, milestone_focus_stance=None)
    eval_icfg = replace(base, **overrides)
    mode = "chain" if base.stance_milestone else "frame0"
    env = ImitationEnv(ref, wall, profile, imitation_config=eval_icfg)
    n_succ, n_exact, lengths, com_ys, furthest_l, rises = 0, 0, [], [], [], []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done, info, n, furthest = False, {}, 0, 0
        ep_com_y = []
        com_z0 = float(env.env.world.com()[2])
        while not done:
            o = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
            action, _ = model.predict(o, deterministic=deterministic)
            obs, _r, term, trunc, info = env.step(action)
            done = term or trunc
            ep_com_y.append(float(env.env.world.com()[1]))
            furthest = max(furthest, int(info.get("target_stance", 0)))
            n += 1
        n_succ += int(info.get("is_success", False))
        n_exact += int(info.get("is_success_exact", False))
        lengths.append(n)
        furthest_l.append(furthest)
        rises.append(float(env.env.world.com()[2]) - com_z0)
        if ep_com_y:
            com_ys.append(float(np.mean(ep_com_y)))
    env.close()
    return {"success": n_succ / max(1, n_episodes), "n_succ": n_succ,
            "success_exact": n_exact / max(1, n_episodes), "n_exact": n_exact,
            "n_episodes": n_episodes, "mean_len": float(np.mean(lengths)),
            "mean_com_y": float(np.mean(com_ys)) if com_ys else float("nan"),
            "mode": mode, "mean_furthest_stance": float(np.mean(furthest_l)),
            "mean_net_rise": float(np.mean(rises))}


def eval_frame0_multi(model, pairs, icfg: Optional[ImitationConfig] = None, *,
                      vec_normalize=None, n_episodes: int = 20,
                      deterministic: bool = True) -> tuple[dict, dict]:
    """Per-wall frame-0 eval for a multi-wall policy. Runs the exact single-wall
    ``eval_frame0`` on each pair (so the same CHAIN(bottom→top) semantics apply
    per wall) and returns ``(per_wall, aggregate)``.

    A single aggregate number is NOT acceptable output for the pilot: a 100%/0%
    split (one wall carrying the policy) reads identically to 50%/50% (real
    shared learning) in the mean. ``per_wall`` is keyed by wall_id so both are
    always visible; ``aggregate`` is the pooled success over all episodes."""
    per: dict[str, dict] = {}
    for ref, wall, profile in pairs:
        per[wall.wall_id] = eval_frame0(
            model, ref, wall, profile, icfg, vec_normalize=vec_normalize,
            n_episodes=n_episodes, deterministic=deterministic)
    total_succ = sum(r["n_succ"] for r in per.values())
    total_eps = sum(r["n_episodes"] for r in per.values())
    agg = {"success": total_succ / max(1, total_eps), "n_succ": total_succ,
           "n_episodes": total_eps,
           "mode": next(iter(per.values()))["mode"] if per else "frame0"}
    return per, agg


def collect_landings(model_path: str, ref_path: str, *, focus_stance: int,
                     out_path: str, n_landings: int = 200,
                     vecnorm: Optional[str] = None, wall_json: Optional[str] = None,
                     icfg: Optional[ImitationConfig] = None) -> None:
    """DAgger collection: roll the composed policy from the bottom and snapshot the
    world state each time it REACHES ``focus_stance`` (target advances to
    focus_stance+1) — the policy's OWN landing right before the focus move. Stochastic
    rollouts give landing variety. Saves qpos/qvel/grips to a .npz bank for training
    with ``--rsi-landing-bank`` (RSI the focus move from real landings, not the
    authored stance — fixes the composition distribution-shift)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    ref = Reference.load(ref_path)
    wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
    base = icfg or ImitationConfig()
    ecfg = replace(base, stance_milestone=True, sequential_chain=True,
                   rsi_landing_bank=None, rsi_landing_banks=None,
                   milestone_focus_stance=None)
    model = PPO.load(model_path)
    ecfg = _maybe_enable_phase_obs(model, ref, wall, profile, ecfg)
    env = ImitationEnv(ref, wall, profile, ecfg)
    am.validate_checkpoint(model_path, wall=wall, obs_dim=int(env.observation_space.shape[0]),
                           action_dim=int(env.action_space.shape[0]))
    vn = None
    if vecnorm and Path(vecnorm).exists():
        vn = VecNormalize.load(vecnorm,
                               DummyVecEnv([lambda: ImitationEnv(ref, wall, profile, ecfg)]))
        vn.training = False
    W = env.env.world
    target = focus_stance + 1
    qpos_l, qvel_l, grip_l = [], [], []
    ep = 0
    while len(qpos_l) < n_landings and ep < n_landings * 5:
        ep += 1
        obs, _ = env.reset(seed=20000 + ep)
        done = False
        while not done:
            o = vn.normalize_obs(obs) if vn is not None else obs
            a, _ = model.predict(o, deterministic=False)  # stochastic → varied landings
            obs, _r, term, trunc, info = env.step(a)
            if int(info.get("target_stance", env._target_stance)) == target:
                qpos_l.append(np.array(W.data.qpos).copy())
                qvel_l.append(np.array(W.data.qvel).copy())
                grip_l.append([W.on_hold(l) or "" for l in LIMBS])
                break  # one fresh landing per episode, then re-roll for variety
            done = term or trunc
    env.close()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    bank_meta = am.build_meta(
        artifact_type="landing_bank", wall=wall, env_mode="imitate:milestone",
        parent=str(model_path), parent_eval=am.parent_eval_of(model_path),
        extra={"focus_stance": focus_stance, "ref": ref_path},
    )
    np.savez(out_path, qpos=np.array(qpos_l), qvel=np.array(qvel_l),
             grips=np.array(grip_l), focus_stance=focus_stance,
             meta=am.npz_meta_value(bank_meta))
    print(f"[dagger] collected {len(qpos_l)} landings at stance {focus_stance} "
          f"(target {target}) over {ep} episodes → {out_path}")


# ─── Training ────────────────────────────────────────────────────────────────

def _build_wall(seed: int):
    """Rebuild the exact wall the reference was authored on (suppressing the
    seed-overbrace chatter)."""
    import contextlib
    import io
    from sim3d.probe_transitions import build_wall_and_moves
    with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
        warnings.simplefilter("ignore")
        wall, profile, _feas = build_wall_and_moves(seed=seed)
    return wall, profile


def _load_wall_for_ref(ref: Reference, wall_json: Optional[str] = None,
                       ref_path: Optional[str] = None):
    """The exact wall the reference was authored/discovered on. A CMA-ES
    reference comes from a non-default ``reach_frac`` wall that can't be rebuilt
    from seed alone, so it's persisted as JSON next to the .npz.

    Resolution order:
      1. Explicit ``wall_json`` argument.
      2. Auto-detected sibling ``<ref_stem>.wall.json`` (when ``ref_path`` given).
      3. Rebuild from ``ref.wall_gen_seed`` (fallback — only correct for walls
         whose hold layout matches the generator defaults; WRONG for CMA-ES refs
         that used a different wall parametrisation).
    """
    import contextlib
    import io
    from solver.wall import load_wall

    candidates = []
    if wall_json:
        candidates.append(wall_json)
    if ref_path:
        sibling = Path(ref_path).with_suffix(".wall.json")
        if sibling.exists():
            candidates.append(str(sibling))

    for cand in candidates:
        if Path(cand).exists():
            with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
                warnings.simplefilter("ignore")
                wall = load_wall(cand)
            am.validate_reference(ref, wall, path=ref_path or cand)
            return wall, ClimberProfile()

    wall, profile = _build_wall(ref.wall_gen_seed)
    am.validate_reference(ref, wall, path=ref_path or "<rebuilt from wall_gen_seed>")
    return wall, profile


def make_env(ref_path: str, icfg: ImitationConfig, rank: int = 0,
             wall_json: Optional[str] = None):
    def _init():
        ref = Reference.load(ref_path)
        wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
        env = ImitationEnv(ref, wall, profile=profile, imitation_config=icfg)
        from stable_baselines3.common.monitor import Monitor
        return Monitor(env)
    return _init


def load_pairs(ref_paths: list[str],
               wall_jsons: Optional[list[Optional[str]]] = None) -> list[tuple]:
    """Resolve a list of ref paths into (ref, wall, profile) pairs. Each ref
    resolves its OWN wall (sibling ``<ref>.wall.json`` or embedded wall_gen_seed
    via ``_load_wall_for_ref``) — never a shared wall. ``am.validate_reference``
    inside that resolver hard-errors on a wall/metadata mismatch, so a ref pointed
    at the wrong wall fails loudly at load, not silently in training."""
    pairs = []
    for i, rp in enumerate(ref_paths):
        wj = wall_jsons[i] if (wall_jsons and i < len(wall_jsons)) else None
        ref = Reference.load(rp)
        wall, profile = _load_wall_for_ref(ref, wj, ref_path=rp)
        pairs.append((ref, wall, profile))
    return pairs


def make_multi_env(ref_paths: list[str], icfg: ImitationConfig, rank: int = 0,
                   wall_jsons: Optional[list[Optional[str]]] = None):
    def _init():
        pairs = load_pairs(ref_paths, wall_jsons)
        env = MultiRefImitationEnv(pairs, imitation_config=icfg)
        from stable_baselines3.common.monitor import Monitor
        return Monitor(env)
    return _init


def smoke(ref_path: str, icfg: Optional[ImitationConfig] = None,
          wall_json: Optional[str] = None) -> None:
    """Single-env sanity: RSI works, reward stays in [0,1], episodes end via the
    termination curriculum / phase-end, and zero-action vs random differ."""
    ref = Reference.load(ref_path)
    wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
    env = ImitationEnv(ref, wall, profile=profile, imitation_config=icfg)
    print(f"reference: {len(ref)} frames | obs {env.observation_space.shape} "
          f"action {env.action_space.shape}")
    for label, policy in (("zero", lambda: np.zeros(env.action_space.shape, dtype=np.float32)),
                          ("random", lambda: env.action_space.sample())):
        outcomes, rewards, lengths = [], [], []
        for ep in range(20):
            _o, _i = env.reset(seed=ep)
            ep_r, ep_l, done = 0.0, 0, False
            while not done:
                o, r, term, trunc, info = env.step(policy())
                # The signed goal potential is negative on retreat and its
                # per-step magnitude is bounded by physical tip/com speed —
                # a released limb whipping/catapulting under random actions
                # reaches >1 m per control step (constraint-snap pathology),
                # so this is a loose 2 m/step scale-bug check, not a physics
                # bound.
                _pot = env.icfg.goal_k * 2.0
                assert np.isfinite(r), f"non-finite reward: {r}"
                assert -_pot - 1e-6 <= r <= 1.0 + env.icfg.completion_bonus + _pot + 1e-6, \
                    f"reward out of range: {r}"
                ep_r += r
                ep_l += 1
                done = term or trunc
            outcomes.append(info["outcome"])
            rewards.append(ep_r)
            lengths.append(ep_l)
        comp = {k: outcomes.count(k) for k in set(outcomes)}
        print(f"  {label:6s}: ep_rew {np.mean(rewards):.2f}  ep_len {np.mean(lengths):.1f}  "
              f"r/step {np.mean(rewards)/max(1,np.mean(lengths)):.2f}  outcomes {comp}")
    env.close()


def _imitate_env_mode(icfg: ImitationConfig) -> str:
    """Canonical env_mode string recorded in an imitation checkpoint's metadata.
    The posture_goal_obs patch changes the obs CONTRACT (a nonzero com-goal vector
    in a goal slot during posture stances) without changing its (131,) shape, so it
    MUST be encoded here: a checkpoint trained with it, then eval'd/recorded without
    it (or vice versa), reads that slot with the opposite meaning and silently
    scores garbage. Encoding it as a suffix makes am.validate_checkpoint hard-error
    on the mismatch. Warm-start (--load) validates this NON-strictly (warns), since
    transferring weights into a new obs regime is intentional."""
    base = "milestone" if icfg.stance_milestone else "dense"
    # +proxgrip marks the proximity-equivalent grip-match success criterion (a
    # limb matches if its gripped hold's centre is within GRIP_PROXIMITY_M of the
    # reference hold's centre). Stamped so a proximity-scored checkpoint is
    # distinguishable from a pre-change exact-hold-id one — their success numbers
    # are NOT comparable, and validate_checkpoint flags a cross-criterion eval.
    return (f"imitate:{base}{'+postureobs' if icfg.posture_goal_obs else ''}"
            f"+proxgrip")


def train(ref_path: str, *, steps: int, n_envs: int, run_id: str, icfg: ImitationConfig,
          wall_json: Optional[str] = None, load_run: Optional[str] = None,
          ent_coef: float = 0.005, stop_flat_after: int = 0,
          stop_flat_evals: int = 2, stop_flat_min_stance: float = 0.0) -> None:
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    out = Path("data/runs/sim3d") / run_id
    out.mkdir(parents=True, exist_ok=True)

    # Reproducibility: persist the FULL run config. Imitation runs previously saved
    # only model.zip + a train.log that omits the flag set, so recipes were lost to
    # shell history (the exact chain4_incr command was unrecoverable). config.json
    # is the one run artifact kept in git per CLAUDE.md.
    import json
    import sys as _sys
    from dataclasses import asdict
    (out / "config.json").write_text(json.dumps({
        "ref": ref_path, "steps": steps, "n_envs": n_envs, "run_id": run_id,
        "wall_json": wall_json, "load_run": load_run, "ent_coef": ent_coef,
        "icfg": asdict(icfg), "argv": _sys.argv,
    }, indent=2, default=str))

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec = vec_cls([make_env(ref_path, icfg, i, wall_json) for i in range(n_envs)])
    # Warm-start: reuse the prior run's VecNormalize stats + policy weights so we
    # continue (and harden) a trained policy instead of relearning from scratch.
    vn_path = Path(load_run) / "vecnormalize.pkl" if load_run else None
    if vn_path is not None and vn_path.exists():
        vec = VecNormalize.load(str(vn_path), vec)
        vec.training, vec.norm_reward = True, False
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Loaded once for the periodic frame-0 eval (the headline metric — the
    # phase-averaged number is diagnostic only; see eval_frame0's docstring).
    eval_ref = Reference.load(ref_path)
    eval_wall, eval_profile = _load_wall_for_ref(eval_ref, wall_json, ref_path=ref_path)

    # Online AMP: the real adversarial loop. The trainer owns the master
    # discriminator + optimizer; envs hold forward-only replicas refreshed by
    # broadcast. Rewards for rollout k are scored by the disc trained through
    # rollout k-1 — the standard AMP schedule (Peng et al. 2021).
    amp_cb = None
    if icfg.amp_online:
        import torch
        from sim3d.amp import (AMPDiscriminator, MotionLibrary,
                               train_discriminator)
        amp_lib = MotionLibrary.load(icfg.amp_library_path)
        amp_disc = AMPDiscriminator()
        amp_disc.set_input_stats(amp_lib.pairs)
        # 3e-4: at 1e-4 the GP-regularised disc separates too slowly for a
        # ~1000-update budget (4 steps × ~250 rollouts on a 2M-step run).
        amp_opt = torch.optim.Adam(amp_disc.parameters(), lr=3e-4)
        print(f"[AMP] online: {len(amp_lib)} real pairs from "
              f"{icfg.amp_library_path}; pair stride {icfg.amp_pair_stride} "
              f"(Δt ≈ {icfg.amp_pair_stride * 0.016:.3f}s vs clip 0.100s)")

        class AMPOnlineCallback(BaseCallback):
            """Collect the policy's (s_t, s_t+stride) pairs from env infos
            during each rollout; after the rollout, update the discriminator
            real-vs-policy (LSGAN + gradient penalty) and broadcast fresh
            weights to every env worker via env_method."""
            def __init__(self):
                super().__init__()
                self._fake: list[np.ndarray] = []

            def _broadcast(self) -> None:
                self.training_env.env_method("set_amp_disc",
                                             amp_disc.state_numpy())

            def _on_training_start(self) -> None:
                self._broadcast()   # replace the envs' random-init replicas

            def _on_step(self) -> bool:
                for info in self.locals["infos"]:
                    p = info.get("amp_pair")
                    if p is not None:
                        self._fake.append(p)
                return True

            def _on_rollout_end(self) -> None:
                if len(self._fake) < 256:
                    return          # not enough policy pairs yet; keep collecting
                fake = np.stack(self._fake)
                self._fake.clear()
                stats = train_discriminator(amp_disc, amp_lib, fake,
                                            optimizer=amp_opt,
                                            n_steps=4, batch_size=256)
                self._broadcast()
                r_pol = float(np.mean(amp_disc.reward_batch(fake[:1024])))
                # Healthy adversarial training: d_real → ~0.7-0.9, d_fake →
                # ~0.1-0.3, r_style(policy) strictly between 0 and 0.75 and
                # NOT pinned at either end (pinned ⇒ no gradient either way).
                print(f"  [AMP] d_real {stats['d_real']:.2f}  "
                      f"d_fake {stats['d_fake']:.2f}  "
                      f"r_style(policy) {r_pol:.2f}  ({len(fake)} pairs)")
                torch.save({"state_dict": amp_disc.state_dict(),
                            "input_dim": int(amp_disc.in_mean.numel()),
                            "hidden": 256}, out / "amp_disc.pt")

        amp_cb = AMPOnlineCallback()

    class ProgressCallback(BaseCallback):
        """Log phase-averaged success + tracking quality per rollout, and the
        HEADLINE frame-0 success every ``eval_every`` steps. The phase-averaged
        number shows learning signal; only frame-0 counts as climbing.

        AUTOMATED FLAT-0 STOP: if ``stop_flat_after`` > 0, halt the run once the
        ★ chain eval reads 0% success for ``stop_flat_evals`` CONSECUTIVE evals
        at/after ``stop_flat_after`` steps — no manual watch needed. To avoid
        killing a healthy composition run (which sits at 0% success for a long
        time WHILE furthest-stance climbs 1→2→3→4), an eval only counts as "flat"
        when its ``mean_furthest_stance`` is ALSO below ``stop_flat_min_stance``
        (i.e. the chain isn't even composing). Set that to 0 to key on success
        alone."""
        def __init__(self, eval_every: int = 100_000, eval_episodes: int = 12,
                     stop_flat_after: int = 0, stop_flat_evals: int = 2,
                     stop_flat_min_stance: float = 0.0):
            super().__init__()
            self.ep_succ, self.ep_done = 0, 0
            self.rimit_sum, self.rimit_n = 0.0, 0
            self.eval_every = eval_every
            self.eval_episodes = eval_episodes
            self._next_eval = eval_every
            self.stop_flat_after = stop_flat_after
            self.stop_flat_evals = stop_flat_evals
            self.stop_flat_min_stance = stop_flat_min_stance
            self._flat_count = 0
            self._stop = False

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                self.rimit_sum += float(info.get("r_imit", 0.0))
                self.rimit_n += 1
                if "episode" in info:  # Monitor end-of-episode
                    self.ep_done += 1
                    self.ep_succ += int(info.get("is_success", False))
            return not self._stop        # False halts learning (flat-0 stop)

        def _on_rollout_end(self) -> None:
            sr = self.ep_succ / max(1, self.ep_done)
            rimit = self.rimit_sum / max(1, self.rimit_n)
            try:
                rmin = float(np.mean(self.training_env.env_method("r_min")))
                cap = self.training_env.env_method("rsi_cap")[0]
            except Exception:  # noqa: BLE001
                rmin, cap = float("nan"), None
            try:
                stages = self.training_env.env_method("chain_stage")
                stage_s = f"  stages {min(stages)}-{max(stages)}"
            except Exception:  # noqa: BLE001
                stage_s = ""
            cap_s = "uniform" if cap is None else f"cap{cap}"
            print(f"  [{self.num_timesteps:>7}] phase-avg success {sr*100:5.1f}%  "
                  f"({self.ep_succ}/{self.ep_done} eps)  r_imit {rimit:.3f}  "
                  f"R_min {rmin:.3f}  RSI {cap_s}{stage_s}")
            self.ep_succ, self.ep_done = 0, 0
            self.rimit_sum, self.rimit_n = 0.0, 0
            if self.num_timesteps >= self._next_eval:
                self._next_eval += self.eval_every
                res = eval_frame0(self.model, eval_ref, eval_wall, eval_profile,
                                  icfg, vec_normalize=self.model.get_vec_normalize_env(),
                                  n_episodes=self.eval_episodes)
                label = "CHAIN(bottom→top)" if res.get("mode") == "chain" else "FRAME-0"
                print(f"  [{self.num_timesteps:>7}] ★ {label} success "
                      f"{res['success']*100:5.1f}%  ({res['n_succ']}/{res['n_episodes']} "
                      f"eps, mean len {res['mean_len']:.0f}, "
                      f"mean furthest stance {res['mean_furthest_stance']:.1f}, "
                      f"net rise {res['mean_net_rise']:+.3f} m; exact-id "
                      f"{res['success_exact']*100:.0f}%)")
                # Automated flat-0 stop: only counts evals at/after stop_flat_after.
                # An eval is "flat" only if success is 0 AND (when a stance gate is
                # set) the chain isn't even composing (furthest stance below the gate)
                # — so a run climbing 1→2→3→4 toward its first completion is spared.
                if self.stop_flat_after > 0 and self.num_timesteps >= self.stop_flat_after:
                    flat = res["success"] <= 0.0 and (
                        self.stop_flat_min_stance <= 0.0
                        or res["mean_furthest_stance"] < self.stop_flat_min_stance)
                    if flat:
                        self._flat_count += 1
                        if self._flat_count >= self.stop_flat_evals:
                            print(f"  ✗ FLAT-0 STOP @ {self.num_timesteps}: chain eval "
                                  f"0% (furthest stance {res['mean_furthest_stance']:.1f}) "
                                  f"for {self._flat_count} consecutive evals ≥ "
                                  f"{self.stop_flat_after} steps — no composition signal; "
                                  f"halting.")
                            self._stop = True
                    else:
                        self._flat_count = 0

    obs_dim = int(vec.observation_space.shape[0])
    action_dim = int(vec.action_space.shape[0])
    env_mode = _imitate_env_mode(icfg)

    model_path = Path(load_run) / "model.zip" if load_run else None
    if model_path is not None and model_path.exists():
        # Warm-start: hard-check the dims/wall, but env_mode differences are only a
        # WARNING here — transferring a non-posture-obs (or different-regime) parent
        # into this run is intentional and the policy adapts. Eval/record stay strict.
        am.validate_checkpoint(model_path, wall=eval_wall, obs_dim=obs_dim,
                               action_dim=action_dim, env_mode=env_mode,
                               strict_env_mode=False)
        model = sb3.PPO.load(str(model_path), env=vec)
        model.ent_coef = ent_coef   # allow tightening exploration on warm-start
        print(f"Warm-started from {model_path}  (ent_coef={ent_coef})")
    else:
        model = sb3.PPO(
            "MlpPolicy", vec, verbose=0,
            learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=ent_coef,
            target_kl=0.03, policy_kwargs={"log_std_init": -1.5},
        )
    # Checkpoint every 200k global steps (vec-normalize stats saved alongside).
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 200_000 // n_envs),
        save_path=str(out / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,
        verbose=0,
    )
    cb_list = [ProgressCallback(stop_flat_after=stop_flat_after,
                                stop_flat_evals=stop_flat_evals,
                                stop_flat_min_stance=stop_flat_min_stance), ckpt_cb]
    if amp_cb is not None:
        cb_list.append(amp_cb)
    callbacks = CallbackList(cb_list)

    print(f"Training imitation: {steps} steps, {n_envs} envs → {out}")
    model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False)
    model.save(str(out / "model.zip"))
    vec.save(str(out / "vecnormalize.pkl"))
    print(f"Saved {out/'model.zip'}")
    res = eval_frame0(model, eval_ref, eval_wall, eval_profile, icfg,
                      vec_normalize=model.get_vec_normalize_env(), n_episodes=40)
    label = "CHAIN(bottom→top)" if res.get("mode") == "chain" else "FRAME-0"
    print(f"★ FINAL {label} success: {res['success']*100:.1f}%  "
          f"({res['n_succ']}/{res['n_episodes']} eps, "
          f"mean furthest stance {res['mean_furthest_stance']:.1f}, "
          f"net rise {res['mean_net_rise']:+.3f} m; exact-id "
          f"{res['success_exact']*100:.0f}%)")

    ckpt_meta = am.build_meta(
        artifact_type="checkpoint", wall=eval_wall, obs_dim=obs_dim, action_dim=action_dim,
        env_mode=env_mode, parent=str(model_path) if model_path else None,
        parent_eval=am.parent_eval_of(model_path),
        extra={"run_id": run_id, "ref": ref_path,
               "eval": {"frame0_success": res["success"], "n_episodes": res["n_episodes"]}},
    )
    am.write_checkpoint_meta(out / "model.zip", ckpt_meta)


def train_multi(ref_paths: list[str], *, steps: int, n_envs: int, run_id: str,
                icfg: ImitationConfig, wall_jsons: Optional[list[Optional[str]]] = None,
                load_run: Optional[str] = None, ent_coef: float = 0.005,
                guard_wall: Optional[str] = None, guard_at: int = 2_000_000,
                guard_min: float = 0.5) -> None:
    """Multi-wall imitation training: one policy on several (wall, reference)
    pairs. Mirrors ``train`` but each worker is a ``MultiRefImitationEnv`` that
    samples a pair per episode, and the periodic + final eval is PER WALL (plus
    the pooled aggregate). The checkpoint records ``wall=None`` — a multi-wall
    policy is not tied to one wall, so no single wall_hash is stamped (which would
    make eval on the other wall spuriously hard-error); obs/action dims and
    env_mode are still recorded and validated."""
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    if icfg.amp_online:
        raise NotImplementedError("online AMP is not wired into multi-wall training")

    out = Path("data/runs/sim3d") / run_id
    out.mkdir(parents=True, exist_ok=True)

    import json
    import sys as _sys
    from dataclasses import asdict
    (out / "config.json").write_text(json.dumps({
        "refs": ref_paths, "steps": steps, "n_envs": n_envs, "run_id": run_id,
        "wall_jsons": wall_jsons, "load_run": load_run, "ent_coef": ent_coef,
        "icfg": asdict(icfg), "argv": _sys.argv,
    }, indent=2, default=str))

    vec_cls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
    vec = vec_cls([make_multi_env(ref_paths, icfg, i, wall_jsons) for i in range(n_envs)])
    vn_path = Path(load_run) / "vecnormalize.pkl" if load_run else None
    if vn_path is not None and vn_path.exists():
        vec = VecNormalize.load(str(vn_path), vec)
        vec.training, vec.norm_reward = True, False
    else:
        vec = VecNormalize(vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    # Pairs loaded once in-process for the periodic per-wall eval.
    eval_pairs = load_pairs(ref_paths, wall_jsons)

    def _print_eval(tag: str, per: dict, agg: dict) -> None:
        label = "CHAIN(bottom→top)" if agg.get("mode") == "chain" else "FRAME-0"
        print(f"  {tag} ★ {label} success  AGG {agg['success']*100:5.1f}%  "
              f"({agg['n_succ']}/{agg['n_episodes']} eps)")
        for wid, r in per.items():
            print(f"      {wid:<20s} {r['success']*100:5.1f}%  "
                  f"({r['n_succ']}/{r['n_episodes']} eps, mean len {r['mean_len']:.0f}, "
                  f"furthest stance {r['mean_furthest_stance']:.1f}, "
                  f"net rise {r['mean_net_rise']:+.3f} m)")

    class MultiProgressCallback(BaseCallback):
        """Per-rollout: phase-avg success + r_imit + PER-WALL episode counts (the
        'both walls appearing in resets' health check). Every eval_every steps:
        the headline per-wall frame-0 (chain) eval, the shared VecNormalize
        pelvis-Z/com-Z running stats, and — once, at/after guard_at steps — the
        AUTOMATED mechanical stop: if the known-good ``guard_wall`` has fallen
        below guard_min, multi-wall co-training is harming it (the interference
        signal the pilot exists to detect), so training halts immediately."""
        def __init__(self, eval_every: int = 200_000, eval_episodes: int = 12,
                     guard_wall: Optional[str] = None,
                     guard_at: int = 2_000_000, guard_min: float = 0.5):
            super().__init__()
            self.ep_succ, self.ep_done = 0, 0
            self.rimit_sum, self.rimit_n = 0.0, 0
            self.wall_eps: dict[str, int] = {}
            self.eval_every = eval_every
            self.eval_episodes = eval_episodes
            self._next_eval = eval_every
            self.guard_wall = guard_wall
            self.guard_at = guard_at
            self.guard_min = guard_min
            self._guard_checked = False
            self._stop = False

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                self.rimit_sum += float(info.get("r_imit", 0.0))
                self.rimit_n += 1
                if "episode" in info:
                    self.ep_done += 1
                    self.ep_succ += int(info.get("is_success", False))
                    wid = info.get("wall_id", "?")
                    self.wall_eps[wid] = self.wall_eps.get(wid, 0) + 1
            return not self._stop        # False halts learning (the guard trip)

        def _log_vecnorm_stats(self) -> None:
            """pelvis-Z (obs[2]) + com-Z (obs[11]) running mean/std of the shared
            VecNormalize. These are ABSOLUTE-world obs channels; the two walls sit
            ~0.82 m apart in Z, so if the known-good wall decays we want to see
            whether it tracks drift/bimodality in these stats (the CONFOUND-2
            hypothesis — see memory multiwall-obs-world-coords-confound)."""
            vn = self.model.get_vec_normalize_env()
            if vn is None or not hasattr(vn, "obs_rms"):
                return
            m, v = vn.obs_rms.mean, vn.obs_rms.var
            print(f"      [vecnorm] pelvisZ mean {float(m[2]):+.3f} std {float(v[2])**0.5:.3f}"
                  f"   comZ mean {float(m[11]):+.3f} std {float(v[11])**0.5:.3f}")

        def _on_rollout_end(self) -> None:
            sr = self.ep_succ / max(1, self.ep_done)
            rimit = self.rimit_sum / max(1, self.rimit_n)
            walls = "  ".join(f"{k}:{v}" for k, v in sorted(self.wall_eps.items()))
            print(f"  [{self.num_timesteps:>7}] phase-avg success {sr*100:5.1f}%  "
                  f"({self.ep_succ}/{self.ep_done} eps)  r_imit {rimit:.3f}  "
                  f"walls[{walls}]")
            self.ep_succ, self.ep_done = 0, 0
            self.rimit_sum, self.rimit_n = 0.0, 0
            self.wall_eps = {}
            if self.num_timesteps >= self._next_eval:
                self._next_eval += self.eval_every
                per, agg = eval_frame0_multi(
                    self.model, eval_pairs, icfg,
                    vec_normalize=self.model.get_vec_normalize_env(),
                    n_episodes=self.eval_episodes)
                self._log_vecnorm_stats()
                _print_eval(f"[{self.num_timesteps:>7}]", per, agg)
                # Automated mechanical stop, evaluated once at/after guard_at.
                if (not self._guard_checked and self.guard_wall
                        and self.num_timesteps >= self.guard_at):
                    self._guard_checked = True
                    s = per.get(self.guard_wall, {}).get("success", 0.0)
                    if s < self.guard_min:
                        print(f"  ✗ MECHANICAL STOP @ {self.num_timesteps}: known-good "
                              f"wall {self.guard_wall} at {s*100:.1f}% < "
                              f"{self.guard_min*100:.0f}% — multi-wall co-training is "
                              f"HARMING it (interference). Halting the run.")
                        self._stop = True
                    else:
                        print(f"  ✓ guard @ {self.num_timesteps}: {self.guard_wall} "
                              f"{s*100:.1f}% ≥ {self.guard_min*100:.0f}% — no interference; "
                              f"continuing.")

    obs_dim = int(vec.observation_space.shape[0])
    action_dim = int(vec.action_space.shape[0])
    env_mode = _imitate_env_mode(icfg)

    model_path = Path(load_run) / "model.zip" if load_run else None
    if model_path is not None and model_path.exists():
        # Multi-wall warm-start: no single wall_hash to check (parent may be
        # single- or multi-wall), so validate dims + env_mode only.
        am.validate_checkpoint(model_path, obs_dim=obs_dim, action_dim=action_dim,
                               env_mode=env_mode, strict_env_mode=False)
        model = sb3.PPO.load(str(model_path), env=vec)
        model.ent_coef = ent_coef
        print(f"Warm-started from {model_path}  (ent_coef={ent_coef})")
    else:
        model = sb3.PPO(
            "MlpPolicy", vec, verbose=0,
            learning_rate=3e-4, n_steps=1024, batch_size=64, n_epochs=5,
            gamma=0.99, gae_lambda=0.95, clip_range=0.1, ent_coef=ent_coef,
            target_kl=0.03, policy_kwargs={"log_std_init": -1.5},
        )
    ckpt_cb = CheckpointCallback(
        save_freq=max(1, 200_000 // n_envs),
        save_path=str(out / "checkpoints"),
        name_prefix="model",
        save_vecnormalize=True,
        verbose=0,
    )
    callbacks = CallbackList([
        MultiProgressCallback(guard_wall=guard_wall, guard_at=guard_at,
                              guard_min=guard_min),
        ckpt_cb])
    if guard_wall:
        print(f"[guard] auto-stop if {guard_wall} < {guard_min*100:.0f}% at "
              f"the first eval ≥ {guard_at} steps")

    print(f"Training MULTI-WALL imitation on {len(ref_paths)} pairs: "
          f"{steps} steps, {n_envs} envs → {out}")
    for rp, (_ref, wall, _p) in zip(ref_paths, eval_pairs):
        print(f"  pair: {wall.wall_id:<20s} ← {rp}")
    model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False)
    model.save(str(out / "model.zip"))
    vec.save(str(out / "vecnormalize.pkl"))
    print(f"Saved {out/'model.zip'}")

    per, agg = eval_frame0_multi(model, eval_pairs, icfg,
                                 vec_normalize=model.get_vec_normalize_env(),
                                 n_episodes=40)
    print("★ FINAL")
    _print_eval("[  FINAL]", per, agg)

    ckpt_meta = am.build_meta(
        artifact_type="checkpoint", wall=None, obs_dim=obs_dim, action_dim=action_dim,
        env_mode=env_mode, parent=str(model_path) if model_path else None,
        parent_eval=am.parent_eval_of(model_path),
        extra={"run_id": run_id, "refs": ref_paths, "multi_wall": True,
               "eval": {"aggregate_frame0_success": agg["success"],
                        "per_wall": {k: v["success"] for k, v in per.items()},
                        "n_episodes_per_wall": 40}},
    )
    am.write_checkpoint_meta(out / "model.zip", ckpt_meta)


def record_video(model_path: str, ref_path: str, out_path: str, *,
                 vecnorm: Optional[str] = None, n_episodes: int = 4,
                 fps: int = 10, size: int = 480, wall_json: Optional[str] = None,
                 rsi_phase_max: Optional[int] = 0, r_min: float = 0.5,
                 cam_azimuth: float = 270.0, cam_elevation: float = -10.0,
                 cam_distance: float = 3.6,
                 base_icfg: Optional[ImitationConfig] = None) -> None:
    """Roll out the trained policy in its ImitationEnv and render to mp4.

    Critically applies the saved VecNormalize obs stats — without them the
    policy gets unnormalised observations and flails (which is why the standard
    browser viewer can't replay an imitation model). Starts every episode at
    phase 0 (RSI cap 0) so the clip shows the full release→reach→regrip, with a
    body-tracking camera.

    ``base_icfg`` — the training/eval config. For a stance-milestone/sequential
    policy this MUST be passed (with stance_milestone=True etc.) or the rollout
    runs in the wrong mode and each episode terminates in one frame (the "4
    frames" bug). When given a milestone config, we render the FULL chain from
    the bottom stance with the same eval-style overrides as ``eval_frame0``
    (sequential_chain on, banks/focus off) so the clip is the real climb."""
    import mujoco
    import imageio
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    ref = Reference.load(ref_path)
    wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
    # Record at the policy's TRAINED R_min floor — recording at a stricter
    # threshold than the policy was trained/verified at cuts episodes it
    # would have finished, which looks like total failure (e.g. an R_min-0.3
    # policy rendered at 0.5 lost all 4 episodes mid-climb).
    if base_icfg is not None and base_icfg.stance_milestone:
        # Full-chain render: mirror eval_frame0's overrides so the clip shows the
        # real bottom→top climb, not a per-move fragment.
        icfg = replace(base_icfg, rsi_phase_max=0, rsi_anneal_steps=0,
                       chain_stages=False, r_min_start=r_min, r_min_end=r_min,
                       sequential_chain=True, rsi_landing_bank=None,
                       rsi_landing_banks=None, milestone_focus_stance=None)
    else:
        icfg = ImitationConfig(rsi_phase_max=rsi_phase_max,
                               r_min_start=r_min, r_min_end=r_min)
    model = PPO.load(model_path)
    icfg = _maybe_enable_phase_obs(model, ref, wall, profile, icfg)
    env = ImitationEnv(ref, wall, profile, imitation_config=icfg)
    am.validate_checkpoint(model_path, wall=wall, obs_dim=int(env.observation_space.shape[0]),
                           action_dim=int(env.action_space.shape[0]),
                           env_mode=_imitate_env_mode(icfg))
    vn = None
    if vecnorm and Path(vecnorm).exists():
        vn = VecNormalize.load(vecnorm, DummyVecEnv([lambda: ImitationEnv(ref, wall, profile, icfg)]))
        vn.training = False

    m, d = env.env.world.model, env.env.world.data
    # Tame the default over-exposure so the wall isn't blown to white and the
    # coloured holds + climber read clearly.
    m.vis.headlight.diffuse[:] = 0.35
    m.vis.headlight.ambient[:] = 0.45
    m.vis.headlight.specular[:] = 0.0
    for _i in range(m.nlight):
        m.light_diffuse[_i] *= 0.35
        m.light_specular[_i] *= 0.0
    renderer = mujoco.Renderer(m, height=size, width=size)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    # Front view (az 270): the climber's back against the wall FACE, so the
    # coloured holds are visible and upward progress along them is legible. A
    # side view hides the holds (they lie flat on the face) and reads as "leaning".
    # az 225/315 give a 3/4 (45 degrees) view that shows depth off the wall.
    cam.azimuth, cam.elevation, cam.distance = cam_azimuth, cam_elevation, cam_distance

    frames, n_done, n_succ = [], 0, 0
    # Chain episodes run up to inner_max_steps (not len(ref)); cap generously.
    step_cap = max(len(ref) + 2, icfg.inner_max_steps + 2) if icfg.stance_milestone \
        else len(ref) + 2
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(step_cap):
            o = vn.normalize_obs(obs) if vn is not None else obs
            action, _ = model.predict(o, deterministic=True)
            obs, _r, term, trunc, info = env.step(action)
            com = env.env.world.com()
            cam.lookat[:] = [float(com[0]), 0.2, float(com[2])]
            renderer.update_scene(d, camera=cam)
            frames.append(renderer.render())
            if term or trunc:
                n_done += 1
                n_succ += int(info.get("is_success", False))
                break
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Wrote {out_path}  ({len(frames)} frames, {n_episodes} eps, "
          f"{n_succ}/{n_done} reached the regrip)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ref", type=str, default="data/runs/sim3d/imitation/ref_move.npz")
    ap.add_argument("--author", action="store_true", help="author the reference first")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--move-index", type=int, default=14, help="feasible move to author")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--record", type=str, default=None,
                    help="render a trained model to this mp4 (needs --model + --ref)")
    ap.add_argument("--cam-azimuth", type=float, default=270.0,
                    help="record camera azimuth deg (270=front-on, 225/315=45 deg 3/4)")
    ap.add_argument("--cam-elevation", type=float, default=-10.0,
                    help="record camera elevation deg (negative looks down)")
    ap.add_argument("--cam-distance", type=float, default=3.6,
                    help="record camera distance (m)")
    ap.add_argument("--eval", action="store_true",
                    help="frame-0 evaluation of a trained model (needs --model "
                         "+ --ref, optionally --vecnorm); the headline metric")
    ap.add_argument("--eval-episodes", type=int, default=40)
    ap.add_argument("--stop-flat-after", type=int, default=0,
                    help="single-wall AUTOMATED STOP: halt if the ★ chain eval reads "
                         "0%% for --stop-flat-evals consecutive evals at/after this "
                         "many steps (0 = disabled).")
    ap.add_argument("--stop-flat-evals", type=int, default=2,
                    help="consecutive 0%% chain evals (past --stop-flat-after) that "
                         "trigger the flat-0 stop (default 2).")
    ap.add_argument("--stop-flat-min-stance", type=float, default=0.0,
                    help="an eval counts as flat only if mean furthest stance is "
                         "below this (spares a run still composing 1→2→3→4). "
                         "0 = key on success alone (default).")
    ap.add_argument("--model", type=str, default=None, help="model.zip for --record")
    ap.add_argument("--vecnorm", type=str, default=None, help="vecnormalize.pkl for --record")
    ap.add_argument("--wall-json", type=str, default=None,
                    help="exact wall JSON (for CMA-ES refs on a non-default wall); "
                         "auto-detected as <ref>.wall.json if present")
    ap.add_argument("--refs", type=str, default=None,
                    help="MULTI-WALL: comma-separated reference .npz paths, one "
                         "policy trained on all pairs (each ref resolves its OWN "
                         "wall via sibling <ref>.wall.json). Overrides --ref for "
                         "--train and --eval; per-wall + aggregate results reported.")
    ap.add_argument("--wall-jsons", type=str, default=None,
                    help="MULTI-WALL: comma-separated wall JSONs matching --refs "
                         "positionally (default: auto-detect each ref's sibling).")
    ap.add_argument("--guard-wall", type=str, default=None,
                    help="MULTI-WALL auto-stop: wall_id of the known-good wall. At "
                         "the first eval ≥ --guard-at steps, if its per-wall success "
                         "< --guard-min the run HALTS (multi-wall co-training is "
                         "harming the known-good case — the interference signal).")
    ap.add_argument("--guard-at", type=int, default=2_000_000,
                    help="step count at which the --guard-wall check fires (default 2M).")
    ap.add_argument("--guard-min", type=float, default=0.5,
                    help="min per-wall success for --guard-wall to pass (default 0.5).")
    ap.add_argument("--steps", type=int, default=60000)
    ap.add_argument("--n-envs", type=int, default=4)
    ap.add_argument("--run-id", type=str, default="imitation/smoke")
    ap.add_argument("--load", type=str, default=None,
                    help="warm-start from a prior run dir (loads model.zip + vecnormalize.pkl)")
    ap.add_argument("--r-min-start", type=float, default=0.75,
                    help="initial R_min termination threshold (default 0.75)")
    ap.add_argument("--r-min-end", type=float, default=0.50,
                    help="final R_min after decay (default 0.50)")
    ap.add_argument("--r-min-decay", type=int, default=150_000,
                    help="per-env steps to anneal R_min start→end (default 150k)")
    ap.add_argument("--rsi-phase-max", type=int, default=None,
                    help="final RSI start-phase cap (low = forced near-bottom)")
    ap.add_argument("--rsi-anneal", type=int, default=0,
                    help="anneal the RSI cap len(ref)→rsi-phase-max over this many "
                         "per-env steps (hardens the full climb). 0 = fixed cap")
    ap.add_argument("--w-pose", type=float, default=None,
                    help="imitation reward pose weight (default 0.65). Lowering it "
                         "shifts the optimum toward end-effector/CoM fidelity — for "
                         "climbing, where the hands and feet go is the skill; exact "
                         "joint angles are style (stand-ups have redundant solutions)")
    ap.add_argument("--w-endeff", type=float, default=None)
    ap.add_argument("--w-com", type=float, default=None)
    ap.add_argument("--w-vel", type=float, default=None)
    ap.add_argument("--k-pose", type=float, default=None,
                    help="pose error sensitivity (default 12.0; lower = softer)")
    ap.add_argument("--k-com", type=float, default=None,
                    help="CoM error sensitivity (default 30.0; raise to force height tracking)")
    ap.add_argument("--k-endeff", type=float, default=None,
                    help="end-effector error sensitivity (default 40.0; raise to tighten grip retention)")
    ap.add_argument("--mover-reach-coeff", type=float, default=0.0,
                    help="dense potential-based reach bonus for the current stage's mover limb "
                         "(default 0). coeff≈100 gives ~26 total bonus for a 0.26m reach")
    ap.add_argument("--grip-retention-coeff", type=float, default=0.0,
                    help="per-step penalty subtracted when a non-mover limb loses a grip the "
                         "reference maintains (default 0). 5.0 = same scale as physics slip "
                         "penalty; directly prevents the 'release LH during foot step' optimum")
    ap.add_argument("--intermediate-bonus-frac", type=float, default=0.5,
                    help="fraction of completion_bonus awarded for each intermediate "
                         "boundary grip-match during stage k>1 episodes (default 0.5). "
                         "Prevents skipping prior moves to chase the final bonus")
    ap.add_argument("--completion-bonus", type=float, default=10.0,
                    help="terminal reward for a verified grip-match at a stage "
                         "boundary / full completion (default 10). Raise it when a "
                         "stage plateaus: parking on the prior stance scores r_imit "
                         "indefinitely, so a small landing bonus leaves PPO nearly "
                         "indifferent to committing the move")
    ap.add_argument("--chain-mid-frac", type=float, default=0.5,
                    help="fraction of chain episodes that RSI mid-stage-window "
                         "(teaches the swing); the rest start at frame 0 (scored "
                         "for advancement). Default 0.5")
    ap.add_argument("--chain", action="store_true",
                    help="chain curriculum: every episode starts at frame 0; "
                         "stage k ends at move k's boundary and succeeds only "
                         "if the reference's stance there is held; ≥70%% rolling "
                         "success unlocks stage k+1")
    ap.add_argument("--chain-rsi-before-stage", action="store_true",
                    help="RSI for stage k starts in [0, prev_stage_end) so every "
                         "episode must execute the current stage's transition. "
                         "Prevents free-success inflation from RSI auto-gripping "
                         "the mover limb from the reference state.")
    ap.add_argument("--chain-rsi-at-stage-start", action="store_true",
                    help="Mid-window RSI episodes start at EXACTLY the current stage "
                         "boundary (e.g. f84 for stage 3). Focuses 65%% of gradient "
                         "on the active transition. No free-grip inflation because the "
                         "mover limb is not yet gripped at the boundary frame.")
    ap.add_argument("--mover-grip-bonus", type=float, default=0.0,
                    help="Sparse one-shot reward the step the mover limb grips its "
                         "target hold. Rewards the reach OUTCOME so committing the "
                         "full swing beats parking the foot short. Try 20-40.")
    ap.add_argument("--mover-capture-coeff", type=float, default=0.0,
                    help="Dense per-step bonus when the mover tip is inside the grip "
                         "capture sphere (within GRIP_PROXIMITY_M = 0.08 m). Fires "
                         "coeff×(1-gap/R) each step, peaking at coeff at the hold "
                         "center. Addresses the 'last 12 cm' problem: mover_reach_coeff "
                         "is potential-based (net-zero when stationary) so has no "
                         "gradient inside the capture sphere; this term does. "
                         "Try 0.2-0.5 (same scale as r_imit). 0.0 disables.")
    ap.add_argument("--mover-reach-abs-coeff", type=float, default=0.0,
                    help="Dense ABSOLUTE reach pull over --mover-reach-radius "
                         "(milestone mode): coeff×max(0,1-gap/radius) every step the "
                         "mover is ungripped. Gradient from rest (unlike potential "
                         "mover-reach-coeff) — the closed-loop foot-move fix. Keep "
                         "small vs --completion-bonus. Try 0.1.")
    ap.add_argument("--mover-reach-radius", type=float, default=0.25,
                    help="radius (m) for --mover-reach-abs-coeff")
    ap.add_argument("--mover-capture-radius", type=float, default=0.0,
                    help="radius (m) of the capture sphere (--mover-capture-coeff). "
                         "0 → GRIP_PROXIMITY_M (0.08). Widen (e.g. 0.15) when a "
                         "trained mover plateaus just outside 0.08 (RF parks at "
                         "~0.126 m, outside the default sphere → no capture pull).")
    ap.add_argument("--w-task", type=float, default=0.0,
                    help="weight on task vs imitation reward (0=pure pose imitation). "
                         "In stance-milestone, w_task=1 drops the pose attractor "
                         "entirely → PURE goal-reaching (reach+capture+grip), removing "
                         "the pose-stillness valley that freezes the foot at the hang.")
    ap.add_argument("--stance-milestone", action="store_true",
                    help="stance-keyframe milestone mode: RSI to a stance, reach "
                         "the next one (pose attractor + grip-match); RL learns "
                         "the balancing transition. Use with a --ref authored by "
                         "`sim3d.discover --stances`.")
    ap.add_argument("--milestone-budget", type=int, default=40,
                    help="max env steps per stance transition (stance-milestone mode)")
    ap.add_argument("--milestone-focus-stance", type=int, default=None,
                    help="focus milestone RSI on ONE transition (stance c→c+1); 100%% "
                         "of gradient on a single move to crack a lone weak one (e.g. "
                         "the final RF foot move). Warm-start a foundation that knows "
                         "the others; phase preserved. None = uniform.")
    ap.add_argument("--milestone-focus-frac", type=float, default=1.0,
                    help="prob of sampling the focus stance (rest uniform over all). "
                         "1.0=exclusive (holds phase constant → wipes other moves); use "
                         "~0.6 to OVERSAMPLE the hard move while keeping the others in "
                         "the mix so phase stays informative and learned moves survive.")
    ap.add_argument("--collect-landings", action="store_true",
                    help="DAgger collect: roll the --model policy from the bottom and "
                         "snapshot its real states at --milestone-focus-stance into "
                         "--landing-bank (.npz). Then train with --rsi via --landing-bank.")
    ap.add_argument("--landing-bank", type=str, default=None,
                    help="DAgger landing bank .npz. OUTPUT of --collect-landings; INPUT to "
                         "--train (RSI the focus move from these real landings instead of "
                         "the authored stance — fixes the composition distribution-shift).")
    ap.add_argument("--n-landings", type=int, default=200,
                    help="how many real landings to collect (--collect-landings)")
    ap.add_argument("--landing-banks", type=str, default=None,
                    help="multi-bank DAgger: 'c1:path1,c2:path2' — a bank per stance so "
                         "BOTH foot moves train from real landings at once (avoids the "
                         "single-focus whack-a-mole). Banked stances oversampled by "
                         "--milestone-focus-frac; hand moves use authored frames.")
    ap.add_argument("--inner-max-steps", type=int, default=200,
                    help="inner-env episode cap. Raise for --sequential-chain with "
                         "many moves (needs > n_moves x milestone-budget).")
    ap.add_argument("--lean-penalty-coeff", type=float, default=0.0,
                    help="penalize TORSO TILT from upright (1-upz). Cleans up the "
                         "lean-back jank (a ~15-26deg pelvis pitch). 26deg≈0.10, so "
                         "try ~8-20. 0 disables.")
    ap.add_argument("--com-rise-coeff", type=float, default=0.0,
                    help="reward upward pelvis/com motion (potential-based). Makes the "
                         "body pull UP over holds (naturalness + net ascent). Try ~5-15.")
    ap.add_argument("--vel-penalty-coeff", type=float, default=0.0,
                    help="penalize SUM of joint-speed² — slows the ~2-frame snap moves "
                         "toward realistic controlled moves (smoother, cleaner landings, "
                         "better chaining). Try ~2-5 (sum, not mean). 0 disables.")
    ap.add_argument("--action-rate-limit", type=float, default=0.0,
                    help="HARD per-step cap on |Δjoint-action| (env-level). Makes 2-frame "
                         "snaps physically impossible → forces gradual controlled moves. "
                         "Try ~0.1 (full-range joint move ≈10 steps/0.16s). 0 disables.")
    ap.add_argument("--arm-bend-coeff", type=float, default=0.0,
                    help="(proxy, games easily) reward elbow flexion. Prefer --wall-hug-coeff.")
    ap.add_argument("--wall-hug-coeff", type=float, default=0.0,
                    help="penalize com distance past the wall-hug target → pulls the body IN "
                         "to the wall (fixes the straight-arm lean-back at the source; arms "
                         "bend on their own once the body is in). Try ~2-5. 0 off.")
    ap.add_argument("--wall-hug-target", type=float, default=0.16,
                    help="com_y (m) considered 'in' (wall plane ≈0.065)")
    ap.add_argument("--wall-hug-mover", type=str, default=None,
                    help="apply wall-hug ONLY when this limb is the mover (e.g. RF) — the "
                         "weight-shift for the committed-stance foot move, without disrupting "
                         "the hand/LF reaches. None = always.")
    ap.add_argument("--wall-hug-terminal-coeff", type=float, default=0.0,
                    help="signed terminal posture reward at COMPLETION only: "
                         "coeff*(target−mean_com_y). Positive when body stayed close; "
                         "small negative when it hung far. Never fires on falls/timeouts "
                         "→ no incentive to fail fast. Try ~5-15.")
    ap.add_argument("--amp-disc", type=str, default="",
                    help="path to a FROZEN AMP discriminator .pt (legacy replay "
                         "only — a frozen disc trained offline vs noise is a "
                         "constant reward offset, not a style signal; use "
                         "--amp-online for real adversarial training).")
    ap.add_argument("--amp-coeff", type=float, default=0.5,
                    help="scale for the AMP style reward (default 0.5). "
                         "Start small — style reward is in [0, 0.75], so 0.5 adds "
                         "up to 0.375/step, similar to one r_imit unit.")
    ap.add_argument("--amp-online", action="store_true",
                    help="ONLINE AMP: update the discriminator on real-vs-POLICY "
                         "pairs after every PPO rollout and broadcast weights to "
                         "the env workers. Needs --amp-library. Ignores --amp-disc.")
    ap.add_argument("--amp-library", type=str,
                    default="data/amp/motion_library.npz",
                    help="motion library .npz of real (s,s') pairs "
                         "(build with: python -m sim3d.amp build-library ...)")
    ap.add_argument("--amp-stride", type=int, default=6,
                    help="control steps per AMP pair. 6 × 0.016s ≈ the 10 fps "
                         "clip Δt (0.1s); mismatched Δt lets the discriminator "
                         "win on frame spacing alone and kills the gradient.")
    ap.add_argument("--sequential-chain", action="store_true",
                    help="true multi-move climb: start at the bottom stance and "
                         "advance the target on each grip WITHOUT reset, so each move "
                         "trains from the previous move's real landing. Success = "
                         "reaching the final stance. Use with --stance-milestone.")
    ap.add_argument("--posture-com-tol", type=float, default=0.06,
                    help="posture (same-grip) stance success: com distance (m) from "
                         "the reference stance com counted as 'holding' it.")
    ap.add_argument("--posture-hold-steps", type=int, default=4,
                    help="posture stance success: consecutive in-tol steps required.")
    ap.add_argument("--posture-goal-obs", action="store_true",
                    help="write (goal_com − com) into a goal obs slot for posture "
                         "stances so the policy can perceive the com target it must "
                         "park at (obs[118:130] are all zero when every limb is "
                         "gripped). obs.py untouched, shape stays 131; recorded in "
                         "env_mode so eval/record MUST pass the same flag.")
    ap.add_argument("--posture-rise-coeff", type=float, default=0.0,
                    help="dense com-RISE pull on stand-up posture stances: "
                         "coeff·max(0,1−dz/band) per step (dz=target_com_z−com_z). "
                         "Directed gradient to raise the body — the net-height fix "
                         "(the pose+com attractor alone stalls the stand-up partway). "
                         "Try ~0.5 (scale of r_imit). 0 off.")
    ap.add_argument("--posture-rise-band", type=float, default=0.15,
                    help="height gap (m) over which --posture-rise-coeff ramps. (legacy)")
    ap.add_argument("--goal-k", type=float, default=100.0,
                    help="goal-potential coefficient (milestone mode): k·Δdist to the "
                         "stance GOAL POINT every step (hold center for grip moves, "
                         "reference com for posture/stand-up stances), active from spawn "
                         "to the success tolerance with no dead zone; the pose attractor "
                         "is frozen inside --goal-band. DEFAULT milestone reward; "
                         "supersedes --mover-reach*/--mover-capture*/--posture-rise*. "
                         "Set 0 to use those legacy terms instead.")
    ap.add_argument("--goal-band", type=float, default=0.15,
                    help="final band (m) around the goal inside which the pose-attractor "
                         "income is latched constant, so its saturated gradient can't "
                         "fight the goal potential (default 0.15).")
    ap.add_argument("--goal-vel-lambda", type=float, default=0.15,
                    help="velocity weight (s) in the POSTURE-stance goal metric "
                         "d_aug = |com-g| + λ|v_com|: makes the potential bottom out "
                         "only at the goal AT REST, so decelerating into the goal pays "
                         "(a fast swing-through reads as 'not there'). 0 = position only.")
    ap.add_argument("--free-mover-imitation", action="store_true",
                    help="Exclude the current stage's mover limb from pose/endeff "
                         "tracking while it is ungripped (mid-reach). Lets PPO find "
                         "a servo-feasible swing instead of matching the infeasible "
                         "recorded one; anchored limbs + CoM still track. Pair with "
                         "--mover-reach-coeff (the remaining signal on the mover).")
    ap.add_argument("--ent-coef", type=float, default=0.005,
                    help="PPO entropy coefficient. Lower (e.g. 0.001) shrinks the "
                         "policy's action noise so stochastic rollout success catches "
                         "up to deterministic — needed when high exploration keeps the "
                         "chain advance gate from being met despite a good policy.")
    ap.add_argument("--phase-obs", action="store_true",
                    help="append a normalized move-phase scalar to the obs (131→132) "
                         "so the policy can tell which move it's on. Fixes multi-move "
                         "composition collapse (a late-move update no longer overwrites "
                         "early moves). Phase models only warm-start (--load) from other "
                         "phase models. --eval/--record auto-detect it from the model.")
    args = ap.parse_args()

    coeffs = ImitationCoeffs()
    if args.w_pose is not None:
        coeffs.w_pose = args.w_pose
    if args.w_endeff is not None:
        coeffs.w_endeff = args.w_endeff
    if args.w_com is not None:
        coeffs.w_com = args.w_com
    if args.w_vel is not None:
        coeffs.w_vel = args.w_vel
    if args.k_pose is not None:
        coeffs.k_pose = args.k_pose
    if args.k_com is not None:
        coeffs.k_com = args.k_com
    if args.k_endeff is not None:
        coeffs.k_endeff = args.k_endeff
    icfg = ImitationConfig(coeffs=coeffs,
                           r_min_start=args.r_min_start,
                           r_min_end=args.r_min_end,
                           r_min_decay_steps=args.r_min_decay,
                           rsi_phase_max=args.rsi_phase_max,
                           rsi_anneal_steps=args.rsi_anneal,
                           chain_stages=args.chain,
                           completion_bonus=args.completion_bonus,
                           intermediate_bonus_frac=args.intermediate_bonus_frac,
                           chain_mid_frac=args.chain_mid_frac,
                           mover_reach_coeff=args.mover_reach_coeff,
                           grip_retention_coeff=args.grip_retention_coeff,
                           chain_rsi_before_stage=args.chain_rsi_before_stage,
                           chain_rsi_at_stage_start=args.chain_rsi_at_stage_start,
                           free_mover_imitation=args.free_mover_imitation,
                           mover_grip_bonus=args.mover_grip_bonus,
                           mover_capture_coeff=args.mover_capture_coeff,
                           mover_capture_radius=args.mover_capture_radius,
                           mover_reach_abs_coeff=args.mover_reach_abs_coeff,
                           mover_reach_radius=args.mover_reach_radius,
                           w_task=args.w_task,
                           stance_milestone=args.stance_milestone,
                           sequential_chain=args.sequential_chain,
                           milestone_budget=args.milestone_budget,
                           milestone_focus_stance=args.milestone_focus_stance,
                           milestone_focus_frac=args.milestone_focus_frac,
                           rsi_landing_bank=(args.landing_bank if not args.collect_landings else None),
                           rsi_landing_banks=args.landing_banks,
                           posture_com_tol=args.posture_com_tol,
                           posture_hold_steps=args.posture_hold_steps,
                           posture_goal_obs=args.posture_goal_obs,
                           posture_rise_coeff=args.posture_rise_coeff,
                           posture_rise_band=args.posture_rise_band,
                           goal_k=args.goal_k,
                           goal_band=args.goal_band,
                           goal_vel_lambda=args.goal_vel_lambda,
                           inner_max_steps=args.inner_max_steps,
                           lean_penalty_coeff=args.lean_penalty_coeff,
                           com_rise_coeff=args.com_rise_coeff,
                           vel_penalty_coeff=args.vel_penalty_coeff,
                           action_rate_limit=args.action_rate_limit,
                           arm_bend_coeff=args.arm_bend_coeff,
                           wall_hug_coeff=args.wall_hug_coeff,
                           wall_hug_target=args.wall_hug_target,
                           wall_hug_mover=args.wall_hug_mover,
                           wall_hug_terminal_coeff=args.wall_hug_terminal_coeff,
                           amp_disc_path=args.amp_disc,
                           amp_coeff=args.amp_coeff,
                           amp_online=args.amp_online,
                           amp_library_path=args.amp_library,
                           amp_pair_stride=args.amp_stride,
                           phase_obs=args.phase_obs)

    if args.author:
        from sim3d.probe_transitions import build_wall_and_moves
        wall, profile, feas = build_wall_and_moves(seed=args.seed)
        move = feas[args.move_index]
        ref, diag = author_weight_shift_move(wall, profile, move, balance_kp=250.0,
                                             wall_gen_seed=args.seed)
        ref.save(args.ref, wall=wall, env_mode="author")
        print(f"Authored move {move['move_k']} → {args.ref}  {diag}")

    # Multi-wall: a comma-separated list of refs, each resolving its own wall.
    ref_paths = [s.strip() for s in args.refs.split(",")] if args.refs else None
    wall_jsons = ([s.strip() or None for s in args.wall_jsons.split(",")]
                  if args.wall_jsons else None)

    # Auto-detect the sibling wall JSON a CMA-ES reference saves next to itself.
    wall_json = args.wall_json
    if wall_json is None:
        sib = Path(args.ref).with_suffix(".wall.json")
        if sib.exists():
            wall_json = str(sib)

    if args.collect_landings:
        if args.milestone_focus_stance is None or not args.landing_bank:
            ap.error("--collect-landings needs --milestone-focus-stance and --landing-bank")
        collect_landings(args.model, args.ref, focus_stance=args.milestone_focus_stance,
                         out_path=args.landing_bank, n_landings=args.n_landings,
                         vecnorm=args.vecnorm, wall_json=wall_json, icfg=icfg)
    if args.eval and ref_paths:
        # Multi-wall per-wall frame-0 eval.
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        pairs = load_pairs(ref_paths, wall_jsons)
        model = PPO.load(args.model)
        # Auto-enable phase_obs off the first pair (all pairs share obs dim).
        _r0, _w0, _p0 = pairs[0]
        icfg = _maybe_enable_phase_obs(model, _r0, _w0, _p0, icfg)
        _dims_env = ImitationEnv(_r0, _w0, _p0, icfg)
        # Multi-wall checkpoint: no single wall_hash — validate dims + env_mode.
        am.validate_checkpoint(args.model,
                               obs_dim=int(_dims_env.observation_space.shape[0]),
                               action_dim=int(_dims_env.action_space.shape[0]),
                               env_mode=_imitate_env_mode(icfg))
        _dims_env.close()
        vn = None
        if args.vecnorm and Path(args.vecnorm).exists():
            vn = VecNormalize.load(
                args.vecnorm,
                DummyVecEnv([lambda: MultiRefImitationEnv(pairs, icfg)]))
            vn.training = False
        per, agg = eval_frame0_multi(model, pairs, icfg, vec_normalize=vn,
                                     n_episodes=args.eval_episodes)
        label = "CHAIN(bottom→top)" if agg.get("mode") == "chain" else "FRAME-0"
        print(f"★ MULTI-WALL {label} success  AGG {agg['success']*100:.1f}%  "
              f"({agg['n_succ']}/{agg['n_episodes']} eps)")
        for wid, r in per.items():
            print(f"    {wid:<20s} {r['success']*100:5.1f}%  "
                  f"({r['n_succ']}/{r['n_episodes']} eps, mean len {r['mean_len']:.0f}, "
                  f"furthest stance {r['mean_furthest_stance']:.1f}, "
                  f"com_y {r['mean_com_y']:.3f} m, net rise {r['mean_net_rise']:+.3f} m; "
                  f"exact-id {r['success_exact']*100:.0f}%)")
    elif args.eval:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        ref = Reference.load(args.ref)
        wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=args.ref)
        model = PPO.load(args.model)
        icfg = _maybe_enable_phase_obs(model, ref, wall, profile, icfg)
        _dims_env = ImitationEnv(ref, wall, profile, icfg)
        am.validate_checkpoint(args.model, wall=wall,
                               obs_dim=int(_dims_env.observation_space.shape[0]),
                               action_dim=int(_dims_env.action_space.shape[0]),
                               env_mode=_imitate_env_mode(icfg))
        _dims_env.close()
        vn = None
        if args.vecnorm and Path(args.vecnorm).exists():
            vn = VecNormalize.load(
                args.vecnorm,
                DummyVecEnv([lambda: ImitationEnv(ref, wall, profile, icfg)]))
            vn.training = False
        res = eval_frame0(model, ref, wall, profile, icfg, vec_normalize=vn,
                          n_episodes=args.eval_episodes)
        label = "CHAIN(bottom→top)" if res.get("mode") == "chain" else "FRAME-0"
        print(f"★ {label} success: {res['success']*100:.1f}%  "
              f"({res['n_succ']}/{res['n_episodes']} eps, mean len {res['mean_len']:.0f}, "
              f"mean furthest stance {res['mean_furthest_stance']:.1f}, "
              f"com_y {res['mean_com_y']:.3f} m, "
              f"net com rise {res['mean_net_rise']:+.3f} m)")
        print(f"    [diagnostic] exact-hold-id success: {res['success_exact']*100:.1f}%  "
              f"({res['n_exact']}/{res['n_episodes']} eps)")
    if args.record:
        record_video(args.model, args.ref, args.record, vecnorm=args.vecnorm,
                     wall_json=wall_json, r_min=args.r_min_end,
                     cam_azimuth=args.cam_azimuth, cam_elevation=args.cam_elevation,
                     cam_distance=args.cam_distance, base_icfg=icfg)
    if args.smoke:
        smoke(args.ref, icfg, wall_json)
    if args.train and ref_paths:
        train_multi(ref_paths, steps=args.steps, n_envs=args.n_envs, run_id=args.run_id,
                    icfg=icfg, wall_jsons=wall_jsons, load_run=args.load,
                    ent_coef=args.ent_coef, guard_wall=args.guard_wall,
                    guard_at=args.guard_at, guard_min=args.guard_min)
    elif args.train:
        train(args.ref, steps=args.steps, n_envs=args.n_envs, run_id=args.run_id,
              icfg=icfg, wall_json=wall_json, load_run=args.load, ent_coef=args.ent_coef,
              stop_flat_after=args.stop_flat_after, stop_flat_evals=args.stop_flat_evals,
              stop_flat_min_stance=args.stop_flat_min_stance)


if __name__ == "__main__":
    main()
