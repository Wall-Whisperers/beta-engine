#!/usr/bin/env bash
# Watches data/runs/sim3d/imitation/wallhug/checkpoints/ for new model zips,
# evals each one, and reports success + com_y.
set -euo pipefail

# Usage: RUN_ID=imitation/wallhug_terminal EXTRA_ARGS="--wall-hug-terminal-coeff 10" bash watch_wallhug.sh
RUN_ID="${RUN_ID:-imitation/wallhug_terminal}"
RUN_DIR="data/runs/sim3d/$RUN_ID/checkpoints"
REF="${REF:-data/runs/sim3d/imitation/ref_chain_handsfirst.npz}"
EXTRA_ARGS="${EXTRA_ARGS:---wall-hug-terminal-coeff 10.0 --wall-hug-target 0.16}"
SEEN=""

source .venv/bin/activate

echo "=== wallhug watcher: $RUN_ID @ $(date) ==="
echo "Polling $RUN_DIR every 30s ..."
echo ""

while true; do
    for zip in $(ls "$RUN_DIR"/model_*_steps.zip 2>/dev/null | sort -V); do
        if [[ "$SEEN" != *"$zip"* ]]; then
            SEEN="$SEEN $zip"
            steps=$(basename "$zip" | grep -o '[0-9]*_steps' | tr -d '_steps')
            pkl="${zip/model_/model_vecnormalize_}"
            pkl="${pkl/.zip/.pkl}"
            echo "--- checkpoint $steps steps @ $(date +%H:%M:%S) ---"
            python -m sim3d.imitation --eval --stance-milestone \
                --ref "$REF" \
                --model "$zip" \
                ${pkl:+--vecnorm "$pkl"} \
                $EXTRA_ARGS \
                --eval-episodes 20 2>/dev/null || echo "  (eval failed)"
            echo ""
        fi
    done

    if ! pgrep -f "run-id $RUN_ID" > /dev/null 2>&1; then
        final="data/runs/sim3d/$RUN_ID/model.zip"
        vnfinal="data/runs/sim3d/$RUN_ID/vecnormalize.pkl"
        echo "=== FINAL MODEL @ $(date) ==="
        if [[ -f "$final" ]]; then
            python -m sim3d.imitation --eval --stance-milestone \
                --ref "$REF" \
                --model "$final" \
                ${vnfinal:+--vecnorm "$vnfinal"} \
                $EXTRA_ARGS \
                --eval-episodes 40 2>/dev/null || echo "  (eval failed)"
        else
            echo "  (no final model.zip found)"
        fi
        break
    fi

    sleep 30
done

echo "=== watcher done ==="
