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
    # Sequential chain (true multi-move climb): start at the bottom stance and,
    # on each grip-match, advance the target to the next stance WITHOUT reset, so
    # move k+1 trains from move k's real on-policy landing (fixes composition).
    # Success = reaching the FINAL stance. Per-transition budget still applies.
    sequential_chain: bool = False

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
    wall_hug_coeff: float = 0.0       # × max(0, com_y − wall_hug_target)
    wall_hug_target: float = 0.16     # com_y (m) considered "in" (wall plane ≈ 0.065)


class ImitationEnv(gym.Env):
    """RSI + bounded-imitation-reward + termination-curriculum wrapper around one
    ``Climbing3DEnv`` tracking a single ``Reference``. The reference is sampled at
    the env control rate, so phase advances 1 frame/step."""

    metadata = Climbing3DEnv.metadata

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
        self._target_stance: int = 1
        self._milestone_step: int = 0
        self._pelvis_y0: float = 0.0
        self._prev_com_z: float = 0.0
        # Elbow qpos addresses + max angle, for the arm-bend (anti-lean) reward.
        _m = self.env.world.model
        self._elbow_qadr = [int(_m.jnt_qposadr[_m.joint(n).id]) for n in ("l_elbow", "r_elbow")]
        self._elbow_max = float(_m.jnt_range[_m.joint("l_elbow").id, 1]) or 2.618

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
        if (self._mover_limb is None or self._mover_hold_pos is None
                or self.env.world.on_hold(self._mover_limb)):
            return obs
        limb_idx = LIMBS.index(self._mover_limb)
        slot = 118 + limb_idx * 3
        tip = self.env.world.limb_tip_pos(self._mover_limb)
        obs = obs.copy()
        obs[slot:slot + 3] = self._mover_hold_pos - tip
        return obs

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

    def _grips_match(self, t: int) -> bool:
        """Every limb the reference grips at frame ``t`` is gripped on the
        SAME hold in the env."""
        ref_g = self.ref.frame_grips(min(t, len(self.ref) - 1))
        return all(self.env.world.on_hold(l) == h
                   for l, h in ref_g.items() if h is not None)

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
            c = 0 if self.icfg.sequential_chain else int(self.np_random.integers(0, max(1, n - 1)))
            self._target_stance = c + 1
            self._milestone_step = 0
            self._prev_mover_gap = float("inf")
            self._mover_grip_awarded = False
            self._mover_limb, self._mover_hold_pos = self._stance_mover(c + 1)
            t = self._stance_frames[c]
            obs, info = self.env.reset_to_reference(
                self.ref.qpos[t], self.ref.qvel[t], self.ref.frame_grips(t),
                settle_frames=self.icfg.settle_frames,
            )
            obs = self._patch_mover_obs(obs)
            self._pelvis_y0 = float(self.env.world.pelvis_pos()[1])
            self._prev_com_z = float(self.env.world.com()[2])
            info.update(self._info(r_imit=1.0))
            info["target_stance"] = self._target_stance
            return obs, info
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
            return obs, info
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
        return obs, info

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
            if gap < cfg.GRIP_PROXIMITY_M:
                reward += self.icfg.mover_capture_coeff * (1.0 - gap / cfg.GRIP_PROXIMITY_M)

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
        return obs, float(reward), terminated, False, info

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
        reward = (1.0 - self.icfg.w_task) * r_imit

        # Optional dense, potential-based mover-reach shaping (net-zero on retreat).
        if (self.icfg.mover_reach_coeff > 0 and mover is not None
                and self._mover_hold_pos is not None
                and not self.env.world.on_hold(mover)):
            gap = float(np.linalg.norm(
                self.env.world.limb_tip_pos(mover) - self._mover_hold_pos))
            if self._prev_mover_gap < float("inf"):
                reward += self.icfg.mover_reach_coeff * (self._prev_mover_gap - gap)
            self._prev_mover_gap = gap
            if (self.icfg.mover_capture_coeff > 0 and gap < cfg.GRIP_PROXIMITY_M):
                reward += self.icfg.mover_capture_coeff * (1.0 - gap / cfg.GRIP_PROXIMITY_M)

        # Dense absolute reach pull (goal-reaching): continuous gradient from rest
        # over the full radius, so the mover starts moving toward its hold even
        # when stationary (potential mover_reach gives nothing then). The mover is
        # freed from pose tracking (free_limb above), so THIS is its main signal.
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
        if self.icfg.wall_hug_coeff > 0:
            # Pull the body IN to the wall (the real anti-lean fix). Penalize com
            # past the wall-hug target → body over the feet, arms bend naturally.
            reward -= self.icfg.wall_hug_coeff * max(
                0.0, float(self.env.world.com()[1]) - self.icfg.wall_hug_target)

        terminated = False
        outcome = ""
        last_stance = len(self._stance_frames) - 1
        if fell:
            terminated, outcome = True, "fell"
        elif self._grips_match(target_frame):
            reward += self.icfg.completion_bonus
            if self.icfg.sequential_chain and self._target_stance < last_stance:
                # Sequential chain: grip reached, but more moves remain. ADVANCE
                # the target to the next stance WITHOUT resetting the body, so the
                # next move trains from this real landing. Re-arm per-transition
                # state + re-point the mover obs at the new target.
                self._target_stance += 1
                self._milestone_step = 0
                self._prev_mover_gap = float("inf")
                self._mover_grip_awarded = False
                self._mover_limb, self._mover_hold_pos = self._stance_mover(self._target_stance)
                obs = self._patch_mover_obs(obs)
                outcome = "advanced"
            else:
                # Final stance reached (or non-sequential single transition).
                terminated, outcome = True, "completed"
        elif self._milestone_step >= self.icfg.milestone_budget:
            terminated, outcome = True, "timeout"

        info["outcome"] = outcome
        info["is_success"] = (outcome == "completed")
        info["target_stance"] = self._target_stance
        info.update(self._info(r_imit=r_imit, comp=comp))
        return obs, float(reward), terminated, False, info

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


# ─── Evaluation ──────────────────────────────────────────────────────────────
# Frame-0 success is THE headline metric. Phase-averaged success under uniform
# RSI inflates badly (v2 reported 70% while landing 0/4 full climbs — easy
# late-reference starts dominate the average). A climb counts when it is
# executed from the bottom.

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
    eval_icfg = replace(base, rsi_phase_max=0, rsi_anneal_steps=0,
                        chain_stages=False,
                        r_min_start=base.r_min_end, r_min_end=base.r_min_end)
    env = ImitationEnv(ref, wall, profile, imitation_config=eval_icfg)
    n_succ, lengths = 0, []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=10_000 + ep)
        done, info, n = False, {}, 0
        while not done:
            o = vec_normalize.normalize_obs(obs) if vec_normalize is not None else obs
            action, _ = model.predict(o, deterministic=deterministic)
            obs, _r, term, trunc, info = env.step(action)
            done = term or trunc
            n += 1
        n_succ += int(info.get("is_success", False))
        lengths.append(n)
    env.close()
    return {"success": n_succ / max(1, n_episodes), "n_succ": n_succ,
            "n_episodes": n_episodes, "mean_len": float(np.mean(lengths))}


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
                return load_wall(cand), ClimberProfile()

    return _build_wall(ref.wall_gen_seed)


def make_env(ref_path: str, icfg: ImitationConfig, rank: int = 0,
             wall_json: Optional[str] = None):
    def _init():
        ref = Reference.load(ref_path)
        wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=ref_path)
        env = ImitationEnv(ref, wall, profile=profile, imitation_config=icfg)
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
                assert 0.0 <= r <= 1.0 + env.icfg.completion_bonus + 1e-6, \
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


def train(ref_path: str, *, steps: int, n_envs: int, run_id: str, icfg: ImitationConfig,
          wall_json: Optional[str] = None, load_run: Optional[str] = None,
          ent_coef: float = 0.005) -> None:
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    out = Path("data/runs/sim3d") / run_id
    out.mkdir(parents=True, exist_ok=True)

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

    class ProgressCallback(BaseCallback):
        """Log phase-averaged success + tracking quality per rollout, and the
        HEADLINE frame-0 success every ``eval_every`` steps. The phase-averaged
        number shows learning signal; only frame-0 counts as climbing."""
        def __init__(self, eval_every: int = 100_000, eval_episodes: int = 12):
            super().__init__()
            self.ep_succ, self.ep_done = 0, 0
            self.rimit_sum, self.rimit_n = 0.0, 0
            self.eval_every = eval_every
            self.eval_episodes = eval_episodes
            self._next_eval = eval_every

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                self.rimit_sum += float(info.get("r_imit", 0.0))
                self.rimit_n += 1
                if "episode" in info:  # Monitor end-of-episode
                    self.ep_done += 1
                    self.ep_succ += int(info.get("is_success", False))
            return True

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
                print(f"  [{self.num_timesteps:>7}] ★ FRAME-0 success "
                      f"{res['success']*100:5.1f}%  ({res['n_succ']}/{res['n_episodes']} "
                      f"eps, mean len {res['mean_len']:.0f})")

    model_path = Path(load_run) / "model.zip" if load_run else None
    if model_path is not None and model_path.exists():
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
    callbacks = CallbackList([ProgressCallback(), ckpt_cb])

    print(f"Training imitation: {steps} steps, {n_envs} envs → {out}")
    model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False)
    model.save(str(out / "model.zip"))
    vec.save(str(out / "vecnormalize.pkl"))
    print(f"Saved {out/'model.zip'}")
    res = eval_frame0(model, eval_ref, eval_wall, eval_profile, icfg,
                      vec_normalize=model.get_vec_normalize_env(), n_episodes=40)
    print(f"★ FINAL FRAME-0 success: {res['success']*100:.1f}%  "
          f"({res['n_succ']}/{res['n_episodes']} eps)")


def record_video(model_path: str, ref_path: str, out_path: str, *,
                 vecnorm: Optional[str] = None, n_episodes: int = 4,
                 fps: int = 10, size: int = 480, wall_json: Optional[str] = None,
                 rsi_phase_max: Optional[int] = 0, r_min: float = 0.5,
                 cam_azimuth: float = 270.0, cam_elevation: float = -10.0,
                 cam_distance: float = 3.6) -> None:
    """Roll out the trained policy in its ImitationEnv and render to mp4.

    Critically applies the saved VecNormalize obs stats — without them the
    policy gets unnormalised observations and flails (which is why the standard
    browser viewer can't replay an imitation model). Starts every episode at
    phase 0 (RSI cap 0) so the clip shows the full release→reach→regrip, with a
    body-tracking camera."""
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
    icfg = ImitationConfig(rsi_phase_max=rsi_phase_max, r_min_start=r_min, r_min_end=r_min)
    env = ImitationEnv(ref, wall, profile, imitation_config=icfg)
    model = PPO.load(model_path)
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
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep)
        for _ in range(len(ref) + 2):
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
    ap.add_argument("--model", type=str, default=None, help="model.zip for --record")
    ap.add_argument("--vecnorm", type=str, default=None, help="vecnormalize.pkl for --record")
    ap.add_argument("--wall-json", type=str, default=None,
                    help="exact wall JSON (for CMA-ES refs on a non-default wall); "
                         "auto-detected as <ref>.wall.json if present")
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
    ap.add_argument("--sequential-chain", action="store_true",
                    help="true multi-move climb: start at the bottom stance and "
                         "advance the target on each grip WITHOUT reset, so each move "
                         "trains from the previous move's real landing. Success = "
                         "reaching the final stance. Use with --stance-milestone.")
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
                           mover_reach_abs_coeff=args.mover_reach_abs_coeff,
                           mover_reach_radius=args.mover_reach_radius,
                           w_task=args.w_task,
                           stance_milestone=args.stance_milestone,
                           sequential_chain=args.sequential_chain,
                           milestone_budget=args.milestone_budget,
                           inner_max_steps=args.inner_max_steps,
                           lean_penalty_coeff=args.lean_penalty_coeff,
                           com_rise_coeff=args.com_rise_coeff,
                           vel_penalty_coeff=args.vel_penalty_coeff,
                           action_rate_limit=args.action_rate_limit,
                           arm_bend_coeff=args.arm_bend_coeff,
                           wall_hug_coeff=args.wall_hug_coeff,
                           wall_hug_target=args.wall_hug_target)

    if args.author:
        from sim3d.probe_transitions import build_wall_and_moves
        wall, profile, feas = build_wall_and_moves(seed=args.seed)
        move = feas[args.move_index]
        ref, diag = author_weight_shift_move(wall, profile, move, balance_kp=250.0,
                                             wall_gen_seed=args.seed)
        ref.save(args.ref)
        print(f"Authored move {move['move_k']} → {args.ref}  {diag}")

    # Auto-detect the sibling wall JSON a CMA-ES reference saves next to itself.
    wall_json = args.wall_json
    if wall_json is None:
        sib = Path(args.ref).with_suffix(".wall.json")
        if sib.exists():
            wall_json = str(sib)

    if args.eval:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        ref = Reference.load(args.ref)
        wall, profile = _load_wall_for_ref(ref, wall_json, ref_path=args.ref)
        model = PPO.load(args.model)
        vn = None
        if args.vecnorm and Path(args.vecnorm).exists():
            vn = VecNormalize.load(
                args.vecnorm,
                DummyVecEnv([lambda: ImitationEnv(ref, wall, profile, icfg)]))
            vn.training = False
        res = eval_frame0(model, ref, wall, profile, icfg, vec_normalize=vn,
                          n_episodes=args.eval_episodes)
        print(f"★ FRAME-0 success: {res['success']*100:.1f}%  "
              f"({res['n_succ']}/{res['n_episodes']} eps, mean len {res['mean_len']:.0f})")
    if args.record:
        record_video(args.model, args.ref, args.record, vecnorm=args.vecnorm,
                     wall_json=wall_json, r_min=args.r_min_end,
                     cam_azimuth=args.cam_azimuth, cam_elevation=args.cam_elevation,
                     cam_distance=args.cam_distance)
    if args.smoke:
        smoke(args.ref, icfg, wall_json)
    if args.train:
        train(args.ref, steps=args.steps, n_envs=args.n_envs, run_id=args.run_id,
              icfg=icfg, wall_json=wall_json, load_run=args.load, ent_coef=args.ent_coef)


if __name__ == "__main__":
    main()
