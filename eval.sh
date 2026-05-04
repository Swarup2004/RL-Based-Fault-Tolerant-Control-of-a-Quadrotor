#!/usr/bin/env bash
# =============================================================
# Quadrotor FTC: Evaluation script
# Evaluates the pre-trained Level 3 model over 100 episodes
# and compares against a passive PID baseline.
#
# Usage:
#   bash eval.sh               -> 100 episodes, Level 3 fault severity
#   bash eval.sh --episodes 50 -> run with 50 episodes
# =============================================================

set -e
cd "$(dirname "$0")"

# Activate venv if it exists (created by run.sh)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

EPISODES="${2:-100}"

echo ""
echo "=========================================="
echo " Evaluating: Level 3 severe fault model"
echo " Fault range: lambda in [0.65, 0.85]"
echo " Episodes: $EPISODES"
echo "=========================================="

export MPLBACKEND=Agg
python -u eval.py --model ./checkpoints/level_3_severe_fault --level 3 --episodes "$EPISODES"

echo ""
echo "Saved: eval_results.png"
