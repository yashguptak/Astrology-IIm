#!/usr/bin/env bash
#
# QLoRA SFT Training with Docker (run on RunPod / Vast.ai / Lambda Labs)
#
# Usage:
#   bash scripts/train_docker.sh
#   bash scripts/train_docker.sh --config configs/training.yaml
#
# Builds and runs a training container with all dependencies.
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

CONFIG="${1:-configs/training.yaml}"

echo "============================================================"
echo "  ASTROLOGY LLM — Docker Training"
echo "============================================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Install Docker first."
    exit 1
fi

# Check GPU
if ! docker info 2>/dev/null | grep -q "Runtimes:.*nvidia"; then
    echo "WARNING: NVIDIA Container Toolkit not detected."
    echo "GPU may not be accessible inside container."
    echo ""
    echo "Install: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/"
fi

echo "Building training image..."
docker build \
    -f deployment/Dockerfile.training \
    -t astrology-llm-training \
    .

echo ""
echo "Running training..."
docker run --rm \
    --gpus all \
    -v "$(pwd):/workspace" \
    -v "$(pwd)/data:/workspace/data" \
    -v "$(pwd)/training:/workspace/training" \
    astrology-llm-training \
    python -m src.training.train --config "$CONFIG"

echo ""
echo "Training complete. Checkpoints saved in ./training/checkpoints/"
echo ""
echo "To merge LoRA adapters:"
echo "  docker run --rm --gpus all -v \"$(pwd):/workspace\" astrology-llm-training python -m src.training.merge"
