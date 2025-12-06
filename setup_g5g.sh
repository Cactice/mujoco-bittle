#!/bin/bash
# Setup script for AWS g5g instances (ARM64 + NVIDIA T4G GPU)
# Run this on your g5g instance after SSH'ing in
# Prerequisites: NVIDIA drivers should already be installed

set -e

echo "========================================"
echo "G5g Instance Setup for PyTorch GPU Training"
echo "========================================"
echo ""

# Check architecture
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "ERROR: This script is for ARM64 (aarch64) architecture only."
    echo "Current architecture: $ARCH"
    echo "You may be on the wrong instance type. G5g uses ARM64."
    exit 1
fi

echo "✓ Architecture: $ARCH (ARM64)"
echo ""

# Check NVIDIA GPU
echo "Checking NVIDIA GPU..."
nvidia-smi
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found. Please install Python 3.12+"
    exit 1
fi

PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo "✓ Python version: $PYTHON_VERSION"
echo ""

# Install uv if not already installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "✓ uv installed"
echo ""

# Install PyTorch for ARM64 with CUDA 12.4 support using uv
echo "Installing PyTorch for ARM64 with CUDA 12.4 support..."
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio

# Sync dependencies from pyproject.toml (except torch which we just installed)
echo ""
echo "Syncing project dependencies..."
uv sync

# Verify GPU detection
echo ""
echo "========================================"
echo "Verifying GPU Setup"
echo "========================================"
uv run python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
    print(f'GPU Count: {torch.cuda.device_count()}')
else:
    print('⚠ WARNING: CUDA not available!')
"

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Prepare assets: uv run python prepare_assets.py"
echo "2. Start training: uv run python train.py --robot bittle --timesteps 100000"
echo ""
echo "All commands should be run with 'uv run' prefix."
echo ""
