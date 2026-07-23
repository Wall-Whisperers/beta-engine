#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Phase-conditioned multi-move chain via the INCREMENTAL LADDER (the fragility fix).
#
# WHY THE LADDER (rewritten 2026-07-01): the first approach — train all per-move
# skills, then compose all 4 moves at once (sequential-chain) — hit the documented
# depth-4 wall: full-chain completion crawled to ~3% over 1.5M steps (the last move
# never got a robust base to launch from). The PROVEN recipe (chain4_incr) builds
# INCREMENTALLY: warm-start the N-move sequential from a WORKING (N-1)-move policy,
# so each rung adds ONE move to a robust base and the new move completes fast.
# 2→3→4 reached 20/20 deterministic that way.
#
# WHY PHASE (--phase-obs): a normalized move-phase scalar (obs 131→132) lets the
# policy tell which move it's on, so a late-move gradient can't overwrite earlier
# moves — the fix for the collapse that dropped the non-phase chain4_incr 20/20→0/40
# on continued training and broke chain5_incr's 4→5. Fixed-denominator phase
# (phase_denom, default 10) keeps the SAME move at the SAME phase across ref lengths
# — required so the ladder's cross-length warm-starts don't shift the shared moves.
#
# THE TEST: how deep does the phase ladder go? The non-phase ladder ceilinged at 4-5
# moves (chain5_incr collapsed). If phase holds 5-6, the fix is validated.
#
# Refs are a NESTED prefix chain (verified): ref_chain_3move (RH,LH,LF) ⊂
# ref_chain_handsfirst (+RF) ⊂ ref_chain_5move (+RH→h_014).
#
# Every run writes config.json (full icfg + argv). Judge chains by DETERMINISTIC
# eval (seq_eval), NOT phase-avg (sequential phase-avg = full-chain only; reads low
# even when 1..N-1 are solid — use scratchpad/diag_ckpt.py for furthest-stance).
#
# Usage:  ./train_chain_phase.sh [stage]    stage ∈ {ms,3,4,5,all}  (default: all)
#   ms  per-move skills   (milestone, uniform RSI, phase)          → ladder_ms
#   3   compose 3-move    (sequential, warm-start ms)              → ladder3
#   4   compose 4-move    (sequential, warm-start 3)               → ladder4
#   5   EXTEND to 5-move  (the fragility test: does phase hold?)   → ladder5
#
# macOS: a standalone `caffeinate -dimsu` should hold the machine awake (the run's
# own caffeinate dies if the run is killed). Keep on AC power. Launch detached:
#   caffeinate -dimsu nohup ./train_chain_phase.sh all > ladder.out 2>&1 &
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1   # flush progress prints live when stdout is redirected
PY="${PY:-.venv/bin/python}"
STAGE="${1:-all}"

REF3="data/runs/sim3d/imitation/ref_chain_3move.npz"        # RH,LH,LF        (4 stances)
REF4="data/runs/sim3d/imitation/ref_chain_handsfirst.npz"   # RH,LH,LF,RF     (5 stances)
REF5="data/runs/sim3d/imitation/ref_chain_5move.npz"        # +RH→h_014       (6 stances)
N_ENVS="${N_ENVS:-8}"

# Shared flags. Capture/completion/budget are set PER STAGE, because the parking
# farm depends on budget: the policy can hover in the grip sphere collecting
# (capture + reach_abs) EVERY step without gripping, so (capture+reach)*budget MUST
# stay under completion_bonus or it never grips (observed: cap0.7*budget70=49 farm +
# reach → mean len 70, 0% success). ms uses the PROVEN budget40/cap0.6/comp40
# (park (0.6+0.3)*40=36 < 40). Sequential rungs need a bigger budget for multi-move
# episodes, so raise completion to keep the margin (park (0.5+0.3)*70=56 < 80).
COMMON="--phase-obs --w-task 1 --mover-reach-abs-coeff 0.3 --mover-reach-radius 0.25"
SEQ_FLAGS="--mover-capture-coeff 0.5 --completion-bonus 80 --milestone-budget 70 --inner-max-steps 500"

# Deterministic sequential eval — the REAL chain verdict. MUST pass the mode flags
# (--stance-milestone --sequential-chain); without them --eval runs plain dense
# imitation and reports a meaningless "mean len 1".
seq_eval() {  # $1=ref  $2=run_dir
  echo "── deterministic sequential eval: $(basename "$2") ──"
  "$PY" -m sim3d.imitation --eval \
    --phase-obs --stance-milestone --sequential-chain --w-task 1 \
    --milestone-budget 70 --inner-max-steps 500 \
    --ref "$1" --model "$2/model.zip" --vecnorm "$2/vecnormalize.pkl" \
    --eval-episodes 20
}

# ── ms — per-move skills (uniform-RSI milestone; each transition in isolation) ──
# ent 0.02 to explore the swings (incl. the foot moves). Trains RH,LH,LF,RF skills.
ms() {
  "$PY" -m sim3d.imitation --train $COMMON --stance-milestone \
    --mover-capture-coeff 0.6 --completion-bonus 40 --milestone-budget 40 \
    --ref "$REF4" --ent-coef 0.02 \
    --steps 2000000 --n-envs "$N_ENVS" --run-id imitation/ladder_ms
}

# ── rf — OVERSAMPLE the lone weak move (RF: the final foot move) ───────────────
# Uniform ms under-trained RF to 0% (¼ of gradient, hardest transition: last limb from
# the most-committed stance). EXCLUSIVE focus cracked RF 0→100% but WIPED RH/LH/LF (it
# held phase constant → VecNormalize zeroed it → policy ignored phase). Fix: OVERSAMPLE
# RF (focus-frac 0.6 → ~70% RF, ~10% each other) so phase keeps varying and the learned
# moves are protected. This is also the real PHASE TEST: can the others survive heavy RF
# training? Warm-start the foundation. Verify with scratchpad/diag_permove.py.
rf() {
  "$PY" -m sim3d.imitation --train $COMMON --stance-milestone \
    --milestone-focus-stance 3 --milestone-focus-frac 0.6 \
    --mover-capture-coeff 0.6 --completion-bonus 40 --milestone-budget 40 \
    --ref "$REF4" --ent-coef 0.02 \
    --load data/runs/sim3d/imitation/ladder_ms \
    --steps 1500000 --n-envs "$N_ENVS" --run-id imitation/ladder_rf
}

# ── 3 — compose the 3-move chain (RH,LH,LF), warm-start the per-move skills ─────
# sequential-chain: start at the bottom, advance target on each grip WITHOUT reset,
# so move k+1 trains from move k's real landing. ent 0.012 (proven compose value).
ladder3() {
  "$PY" -m sim3d.imitation --train $COMMON $SEQ_FLAGS --stance-milestone --sequential-chain \
    --ref "$REF3" --ent-coef 0.012 \
    --load data/runs/sim3d/imitation/ladder_rf \
    --steps 1500000 --n-envs "$N_ENVS" --run-id imitation/ladder3
  seq_eval "$REF3" data/runs/sim3d/imitation/ladder3
}

# ── 4 — add the 4th move (RF), warm-start the WORKING 3-move policy ─────────────
ladder4() {
  "$PY" -m sim3d.imitation --train $COMMON $SEQ_FLAGS --stance-milestone --sequential-chain \
    --ref "$REF4" --ent-coef 0.012 \
    --load data/runs/sim3d/imitation/ladder3 \
    --steps 1500000 --n-envs "$N_ENVS" --run-id imitation/ladder4
  seq_eval "$REF4" data/runs/sim3d/imitation/ladder4
}

# ── 5 — EXTEND 4→5 (the fragility test): warm-start the WORKING 4-move policy ────
# This is what the phase-BLIND chain5_incr FAILED (4→5 broke moves 3-4). If phase
# holds the 5-move chain, the fix is validated.
ladder5() {
  "$PY" -m sim3d.imitation --train $COMMON $SEQ_FLAGS --stance-milestone --sequential-chain \
    --ref "$REF5" --ent-coef 0.012 \
    --load data/runs/sim3d/imitation/ladder4 \
    --steps 2000000 --n-envs "$N_ENVS" --run-id imitation/ladder5
  seq_eval "$REF5" data/runs/sim3d/imitation/ladder5
}

# ── ABLATION — the phase-conditioning PROOF: identical recipe, phase-obs OFF ────
# ladder_rf (phase ON) held RH/LH/LF at 100% under 70%-RF oversampling. If the
# phase-OFF version collapses those moves under the SAME skew, that isolates
# phase-conditioning as the cause of the protection (the fragility-fix headline).
NOPH="--w-task 1 --mover-reach-abs-coeff 0.3 --mover-reach-radius 0.25"
ms_noph() {
  "$PY" -m sim3d.imitation --train $NOPH --stance-milestone \
    --mover-capture-coeff 0.6 --completion-bonus 40 --milestone-budget 40 \
    --ref "$REF4" --ent-coef 0.02 \
    --steps 2000000 --n-envs "$N_ENVS" --run-id imitation/ladder_ms_noph
}
rf_noph() {
  "$PY" -m sim3d.imitation --train $NOPH --stance-milestone \
    --milestone-focus-stance 3 --milestone-focus-frac 0.6 \
    --mover-capture-coeff 0.6 --completion-bonus 40 --milestone-budget 40 \
    --ref "$REF4" --ent-coef 0.02 \
    --load data/runs/sim3d/imitation/ladder_ms_noph \
    --steps 1500000 --n-envs "$N_ENVS" --run-id imitation/ladder_rf_noph
}

case "$STAGE" in
  ms)  ms ;;
  rf)  rf ;;
  ms_noph) ms_noph ;;
  rf_noph) rf_noph ;;
  3)   ladder3 ;;
  4)   ladder4 ;;
  5)   ladder5 ;;
  all) ms; rf; ladder3; ladder4; ladder5 ;;
  *)   echo "unknown stage: $STAGE (use ms|rf|3|4|5|all)"; exit 1 ;;
esac
