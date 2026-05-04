#!/usr/bin/env bash
# =============================================================
# Quadrotor FTC: Automated pipeline script
# Designed for Ubuntu 22.04 (Docker). Works on Linux/macOS too.
#
# Usage:
#   bash run.sh              -> evaluate pre-trained model + generate plots
#   bash run.sh train        -> train from scratch (all 4 levels) then eval
#   bash run.sh resume       -> resume Level 3 from checkpoint then eval
#
# The pre-trained checkpoint (checkpoints/level_3_severe_fault.zip) is
# included in the repository. Default mode evaluates it without training.
#
# All outputs are saved as PNG files in the repository directory:
#   eval_results.png          -- RMSE comparison over 100 episodes
#   trajectory_compare.png    -- RL vs PID trajectory overlay
# =============================================================

set -e
cd "$(dirname "$0")"

MODE="${1:-eval}"

# ------------------------------------------------------------------
# Step 1: System packages (Ubuntu/Debian only, runs as root)
# ------------------------------------------------------------------
if command -v apt-get &>/dev/null && [ "$(id -u)" = "0" ]; then
    echo "[setup] Updating package lists and installing system dependencies..."
    apt-get update -qq
    apt-get install -y -q python3 python3-pip python3-venv
fi

# ------------------------------------------------------------------
# Step 2: Python virtual environment
# ------------------------------------------------------------------
PYTHON=$(command -v python3 2>/dev/null || command -v python 2>/dev/null)
if [ -z "$PYTHON" ]; then
    echo "ERROR: python3 not found. Install Python 3.10+."
    exit 1
fi

echo "[setup] Creating Python virtual environment..."
if [ ! -d venv ]; then
    $PYTHON -m venv venv
fi
source venv/bin/activate

echo "[setup] Python: $(python --version)"

# ------------------------------------------------------------------
# Step 3: Install Python dependencies
# ------------------------------------------------------------------
echo "[setup] Installing Python dependencies (this may take a few minutes)..."
pip install --upgrade pip -q

# Install CPU-only PyTorch (avoids the large CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cpu -q

# Install remaining dependencies
pip install "stable-baselines3>=2.0.0" "gymnasium>=0.29.0" "numpy>=1.24.0" "matplotlib>=3.7.0" -q

echo "[setup] Dependencies installed."

# Use Agg backend: no display required
export MPLBACKEND=Agg

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
run_eval() {
    echo ""
    echo "=========================================="
    echo " Evaluating pre-trained model"
    echo " 100 episodes | Level 3 | lambda=[0.65,0.85]"
    echo "=========================================="
    python -u eval.py \
        --model ./checkpoints/level_3_severe_fault \
        --level 3 \
        --episodes 100
}

run_viz() {
    echo ""
    echo "=========================================="
    echo " Generating trajectory comparison plot"
    echo "=========================================="
    python -u visualize.py \
        --model ./checkpoints/level_3_severe_fault \
        --level 3 \
        --compare \
        --save
}

train_curriculum() {
    echo ""
    echo "WARNING: Full training overwrites checkpoints. Takes several hours on CPU."
    echo ""
    echo "--- Level 0: healthy hover ---"
    python -u train.py --level 0 --n-envs 4

    echo ""
    echo "--- Level 1: mild fault (lambda 0.85-0.95) ---"
    python -u train.py --resume ./checkpoints/level_0_healthy --level 1 --n-envs 4

    echo ""
    echo "--- Level 2: moderate fault (lambda 0.70-0.90) ---"
    python -u train.py --resume ./checkpoints/level_1_mild_fault --level 2 --n-envs 4

    echo ""
    echo "--- Level 3: severe fault (lambda 0.65-0.85) ---"
    python -u train.py --resume ./checkpoints/level_2_moderate_fault --level 3 --n-envs 4
}

# ------------------------------------------------------------------
# Step 4: Run selected mode
# ------------------------------------------------------------------
case "$MODE" in
    train)
        train_curriculum
        run_eval
        run_viz
        ;;
    resume)
        echo ""
        echo "=========================================="
        echo " Resuming Level 3 training from checkpoint"
        echo "=========================================="
        python -u train.py \
            --resume ./checkpoints/level_3_severe_fault \
            --level 3 \
            --n-envs 4
        run_eval
        run_viz
        ;;
    eval|*)
        run_eval
        run_viz
        ;;
esac

echo ""
echo "============================================"
echo " Done. Outputs saved:"
echo "   eval_results.png"
echo "   trajectory_compare.png"
echo "============================================"
