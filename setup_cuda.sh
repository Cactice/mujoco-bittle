#!/bin/bash
# Setup script for CUDA environment on EC2

set -e

echo "Setting up CUDA library path..."

# Add CUDA to library path permanently
if ! grep -q "LD_LIBRARY_PATH.*cuda" ~/.bashrc; then
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo "Added CUDA library path to ~/.bashrc"
else
    echo "CUDA library path already configured in ~/.bashrc"
fi

echo ""
echo "Setup complete!"
echo "Run: source ~/.bashrc"
echo "Then: uv sync --extra cuda"
echo "Then: uv run train.py --robot humanoid"
