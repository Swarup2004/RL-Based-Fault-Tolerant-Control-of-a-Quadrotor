#!/usr/bin/env bash
# =============================================================
# Quadrotor FTC: Training script
# Trains all 4 curriculum levels sequentially from scratch.
#
# Usage:
#   bash train.sh              -> full curriculum (Level 0 to 3)
#   bash train.sh --resume     -> resume Level 3 from included checkpoint
#
# WARNING: Running full training overwrites checkpoints/level_*.zip
# Back up the checkpoints/ folder first to preserve the pre-trained model.
# Training takes several hours on CPU, ~1 hour on a modern GPU.
# =============================================================

set -e
cd "$(dirname "$0")"

# Activate venv if it exists (created by run.sh)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

MODE="${1:-full}"

if [ "$MODE" = "--resume" ] || [ "$MODE" = "resume" ]; then
    echo ""
    echo "=========================================="
    echo " Resuming Level 3: severe fault"
    echo " (lambda 0.65-0.85, mid-episode injection)"
    echo "=========================================="
    python -u train.py --resume ./checkpoints/level_3_severe_fault --level 3 --n-envs 4
else
    echo ""
    echo "WARNING: Full training will overwrite the included checkpoint."
    echo "         Back up checkpoints/ first if you want to keep the pre-trained model."
    echo ""

    echo "=========================================="
    echo " Level 0: healthy hover (no fault)"
    echo "=========================================="
    python -u train.py --level 0 --n-envs 4

    echo ""
    echo "=========================================="
    echo " Level 1: mild fault (lambda 0.85-0.95)"
    echo "=========================================="
    python -u train.py --resume ./checkpoints/level_0_healthy --level 1 --n-envs 4

    echo ""
    echo "=========================================="
    echo " Level 2: moderate fault (lambda 0.70-0.90)"
    echo "=========================================="
    python -u train.py --resume ./checkpoints/level_1_mild_fault --level 2 --n-envs 4

    echo ""
    echo "=========================================="
    echo " Level 3: severe fault (lambda 0.65-0.85)"
    echo "=========================================="
    python -u train.py --resume ./checkpoints/level_2_moderate_fault --level 3 --n-envs 4
fi

echo ""
echo "Training complete. Checkpoints saved in ./checkpoints/"
