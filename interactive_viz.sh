#!/usr/bin/env bash
# =============================================================
# Quadrotor FTC: Interactive 3D flight viewer
# Keyboard controls:
#   F  -- inject fault on current motor
#   1-4 -- select which motor to fault
#   R  -- reset episode
#   Space -- pause/resume
#   Q  -- quit
#
# Requires a display (not for headless/Docker use).
# Run after bash run.sh has set up the virtual environment.
# =============================================================

set -e
cd "$(dirname "$0")"

# Activate venv if it exists
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Check for display
if [ -z "$DISPLAY" ] && [ "$(uname)" != "Darwin" ]; then
    echo "ERROR: No display detected (\$DISPLAY is not set)."
    echo "This viewer requires a graphical environment."
    echo "On a desktop: just run  bash interactive_viz.sh"
    echo "Over SSH:     use X11 forwarding  ssh -X ..."
    exit 1
fi

echo ""
echo "=========================================="
echo " Interactive 3D flight viewer"
echo "=========================================="
python -u interactive_viz.py --model ./checkpoints/level_3_severe_fault
