#!/usr/bin/env bash
# =============================================================
# Quadrotor FTC: Visualization scripts
#
# Usage:
#   bash viz.sh              -> trajectory plots (saves to PNG)
#   bash viz.sh compare      -> RL vs PID comparison plot (saves to PNG)
#   bash viz.sh interactive  -> interactive 3D flight viewer (requires display)
# =============================================================

set -e
cd "$(dirname "$0")"

# Activate venv if it exists (created by run.sh)
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

MODE="${1:-plot}"

case "$MODE" in
    compare)
        echo ""
        echo "=========================================="
        echo " Trajectory comparison: RL vs PID"
        echo "=========================================="
        export MPLBACKEND=Agg
        python -u visualize.py --model ./checkpoints/level_3_severe_fault --level 3 --compare --save
        echo "Saved: trajectory_compare.png"
        ;;
    interactive)
        echo ""
        echo "=========================================="
        echo " Interactive 3D flight viewer"
        echo " (requires a display; not for headless use)"
        echo "=========================================="
        if [ -z "$DISPLAY" ] && [ "$(uname)" != "Darwin" ]; then
            echo "WARNING: No DISPLAY detected. Interactive viewer requires a graphical environment."
            echo "Run this on a desktop system or with X11 forwarding."
            exit 1
        fi
        python -u interactive_viz.py --model ./checkpoints/level_3_severe_fault
        ;;
    plot|*)
        echo ""
        echo "=========================================="
        echo " Trajectory plots (single episode)"
        echo "=========================================="
        export MPLBACKEND=Agg
        python -u visualize.py --model ./checkpoints/level_3_severe_fault --level 3 --save
        echo "Saved: trajectory_episode_1.png trajectory_episode_2.png"
        ;;
esac
