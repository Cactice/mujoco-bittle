# G5g Instance GPU Setup - Quick Reference

## The Problem
- **G5g instances use ARM64 (aarch64) architecture**, not x86_64
- Your current `pyproject.toml` was configured for x86_64 PyTorch with CUDA 11.8
- **ARM64 requires different PyTorch wheels** with CUDA 12.4+ support

## The Solution

### Step 1: Connect to your instance
```bash
ssh -i "bittle.pem" ec2-user@ec2-54-83-124-94.compute-1.amazonaws.com
```

### Step 2: Upload and run the setup script
From your local machine:
```bash
scp -i "bittle.pem" setup_g5g.sh ec2-user@ec2-54-83-124-94.compute-1.amazonaws.com:~/
scp -i "bittle.pem" -r . ec2-user@ec2-54-83-124-94.compute-1.amazonaws.com:~/mujoco-bittle/
```

On the g5g instance:
```bash
cd mujoco-bittle
bash setup_g5g.sh
```

The script will:
1. Install uv if not already present
2. Install PyTorch with ARM64 CUDA 12.4 support
3. Sync all other dependencies from `pyproject.toml`

### Step 3: Verify GPU is working
```bash
# Check NVIDIA driver
nvidia-smi

# Check PyTorch can see GPU
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available(), '- Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Step 4: Train
```bash
uv run python prepare_assets.py
uv run python train.py --robot bittle --timesteps 100000
```

## Alternative: Use AWS Deep Learning AMI (Easiest)

Instead of manual setup, you can:

1. **Terminate your current instance**
2. **Launch a new g5g instance** with the "Deep Learning AMI GPU PyTorch 2.x (Amazon Linux 2023) ARM64" AMI
3. This comes with everything pre-installed
4. Just install uv and sync your dependencies:
   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   export PATH="$HOME/.local/bin:$PATH"

   # Sync dependencies
   cd mujoco-bittle
   uv sync
   ```

## Key Technical Details

### G5g Instance Specs
- **CPU**: AWS Graviton2 (ARM64/aarch64)
- **GPU**: NVIDIA T4G Tensor Core
- **CUDA**: Requires 12.4+ for ARM64 support
- **PyTorch**: Must use ARM64 wheels from PyTorch index

### Correct PyTorch Installation
```bash
# For ARM64 with CUDA 12.4 using uv
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

### Why This Matters
- x86_64 PyTorch wheels **will not work** on ARM64
- CUDA 11.8 doesn't fully support ARM64 GPUs
- You need the correct wheel for your architecture

## Troubleshooting

### GPU not detected
```bash
# Check driver
nvidia-smi

# If not found, reboot after driver installation
sudo reboot
```

### Import errors
```bash
# Verify PyTorch installation
uv run python -c "import torch; print(torch.__version__)"
```

### Still showing CPU
```bash
# Check CUDA version compatibility
uv run python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')"

# Should show CUDA 12.4 for G5g
```
