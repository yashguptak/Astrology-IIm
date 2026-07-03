#!/usr/bin/env bash
#
# QLoRA SFT Training Launcher for Linux (Ubuntu GPU Server)
#
# Usage:
#   chmod +x scripts/train_linux.sh
#   ./scripts/train_linux.sh
#   ./scripts/train_linux.sh --config configs/training.yaml
#   ./scripts/train_linux.sh --resume_from_checkpoint training/checkpoints/checkpoint-100
#
# This script:
#   1. Checks GPU availability
#   2. Installs dependencies if needed
#   3. Runs training
#   4. Merges LoRA adapters (optional)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}============================================================${NC}"
echo -e "${CYAN}  ASTROLOGY LLM — Training Launcher (Linux)${NC}"
echo -e "${CYAN}============================================================${NC}"

# Parse args
CONFIG="${1:-configs/training.yaml}"
RESUME="${2:-}"
shift 2 2>/dev/null || true

# --- Helper functions ---
usage() {
    echo "Usage: $0 [--config CONFIG] [--resume CHECKPOINT] [--merge-only]"
    echo ""
    echo "  --config CONFIG       Path to training config (default: configs/training.yaml)"
    echo "  --resume CHECKPOINT   Resume from checkpoint path"
    echo "  --merge-only          Skip training, only merge LoRA adapters"
    echo "  --help                Show this help"
    exit 0
}

# Parse named args
CONFIG="configs/training.yaml"
RESUME=""
MERGE_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --resume) RESUME="$2"; shift 2 ;;
        --merge-only) MERGE_ONLY=true; shift ;;
        --help) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

# --- Step 1: Check GPU ---
echo ""
echo -e "${YELLOW}[1/4] Checking GPU availability...${NC}"

if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | tail -1)
    echo "GPU count: $GPU_COUNT"
else
    echo -e "${RED}ERROR: No NVIDIA GPU detected.${NC}"
    echo ""
    echo "Training requires a CUDA-capable GPU."
    echo ""
    echo "Options:"
    echo "  1. Install NVIDIA driver + CUDA"
    echo "     sudo apt update && sudo apt install nvidia-driver-550 nvidia-cuda-toolkit"
    echo ""
    echo "  2. Use a cloud GPU instance (RunPod, Vast.ai, Lambda Labs)"
    echo ""
    exit 1
fi

# --- Step 2: Check Python ---
echo ""
echo -e "${YELLOW}[2/4] Checking Python...${NC}"
PYTHON=$(command -v python3 || command -v python)
if [ -z "$PYTHON" ]; then
    echo -e "${RED}ERROR: Python not found. Install Python 3.10+${NC}"
    exit 1
fi
echo "Using: $($PYTHON --version)"

# --- Step 3: Install dependencies ---
echo ""
echo -e "${YELLOW}[3/4] Installing dependencies...${NC}"
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet -r requirements-gpu.txt
$PYTHON -m pip install --quiet -r requirements-local.txt

# --- Step 4: Run training ---
echo ""
echo -e "${YELLOW}[4/4] Running training...${NC}"

if [ "$MERGE_ONLY" = true ]; then
    echo "Merge-only mode..."
    $PYTHON -m src.training.merge
else
    echo "Config: $CONFIG"
    if [ -n "$RESUME" ]; then
        echo "Resuming from: $RESUME"
    fi

    $PYTHON -m src.training.train \
        --config "$CONFIG" \
        ${RESUME:+--resume_from_checkpoint "$RESUME"}

    echo ""
    echo -e "${YELLOW}Optional: Merge LoRA adapters into base model${NC}"
    echo "  Run: python -m src.training.merge"
    echo "  Or:  $0 --merge-only"
fi

echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  Done!${NC}"
echo -e "${GREEN}============================================================${NC}"
