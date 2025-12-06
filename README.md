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

If running on a headless server:
1. Run `uv run python train.py`.
2. Download the `ppo_bittle_tensorboard/` folder to your local machine.
3. Run `tensorboard --logdir ppo_bittle_tensorboard` locally to view progress.
