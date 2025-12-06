# MuJoCo Bittle RL Training

This project implements a Reinforcement Learning environment for the Petoi Bittle robot using MuJoCo and Gymnasium. It uses Stable Baselines3 (PPO) for training a walking policy.

## Features

- **Custom Environment**: `BittleEnv` (Gymnasium) with MuJoCo physics.
- **Asset Pipeline**: Automatically patches and converts the MIT-licensed Bittle URDF to MJCF, adding actuators, floating base, and correcting inertias.
- **Visualization**: Includes a script to visualize the trained agent with a custom environment (floor, lighting).
- **Modern Tooling**: Managed with `uv` and linted with `ruff`.

## Installation

This project uses `uv` for dependency management.

```bash
uv sync
```

## Usage

### 1. Prepare Assets
The simulation assets must be generated from the source URDF. This script creates `assets/bittle_final.xml`.

```bash
uv run python prepare_assets.py
```

### 2. Train the Agent
Train a PPO agent. The model will be saved to `ppo_bittle.zip`.

```bash
uv run python train.py
```

### 3. Visualize
Watch the trained agent (or a random one if no model exists).

```bash
uv run python play.py
```

## Project Structure

- `assets/`:
    - `bittle_esp32.urdf`: Original source URDF (MIT License).
    - `bittle_final.xml`: Generated MJCF file used for simulation.
- `envs/`: Contains `bittle_env.py`, the custom Gymnasium environment.
- `prepare_assets.py`: Script to generate simulation assets.
- `train.py`: Training script.
- `play.py`: Visualization script.

## Cloud Training (AWS/GCP)

### For x86_64 instances (g4dn, p3, etc.)
If running on a headless server:
1. Run `uv run python train.py`.
2. Download the `ppo_bittle_tensorboard/` folder to your local machine.
3. Run `tensorboard --logdir ppo_bittle_tensorboard` locally to view progress.

### For ARM64 instances (g5g with AWS Graviton2)

G5g instances use ARM64 architecture with NVIDIA T4G GPUs. You need ARM64-compatible PyTorch with CUDA support.

**Option 1: Use AWS Deep Learning AMI (Recommended)**

Launch your g5g instance with the ARM64 PyTorch GPU DLAMI, which comes pre-configured with:
- PyTorch for ARM64
- CUDA 12.4+
- cuDNN, NCCL
- NVIDIA drivers

Then install uv and sync dependencies:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Sync dependencies
cd mujoco-bittle
uv sync
```

**Option 2: Manual Setup on Amazon Linux 2023 / Ubuntu ARM64**

1. Install NVIDIA drivers and CUDA toolkit for ARM64:
```bash
# For Amazon Linux 2023
sudo yum install -y gcc kernel-devel-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/repos/rhel9/sbsa/cuda-rhel9.repo
sudo mv cuda-rhel9.repo /etc/yum.repos.d/
sudo yum clean all
sudo yum -y install cuda-toolkit-12-4

# For Ubuntu 22.04
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-4
```

2. Install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

3. Install PyTorch for ARM64 with CUDA support:
```bash
# PyTorch 2.0+ with CUDA 12.x for ARM64
uv pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
```

4. Sync other dependencies:
```bash
uv sync
```

5. Verify GPU is detected:
```bash
uv run python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

6. Run training:
```bash
uv run python train.py --robot bittle --timesteps 100000
```

**Note:** MuJoCo runs on CPU but neural network training will use the GPU, which provides significant speedup for PPO.
