import argparse
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.bittle_env import BittleEnv
from envs.humanoid_env import HumanoidEnv
import os
import torch


def train():
    parser = argparse.ArgumentParser(description="Train MuJoCo agent")
    parser.add_argument(
        "--robot",
        type=str,
        default="bittle",
        choices=["bittle", "humanoid"],
        help="Robot to train",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000, help="Total timesteps for training"
    )
    args = parser.parse_args()

    # Create environment
    if args.robot == "bittle":
        env = BittleEnv()
        model_name = "ppo_bittle"
    else:
        env = HumanoidEnv()
        model_name = "ppo_humanoid_breakdance"

    # Check environment
    print(f"Checking environment for {args.robot}...")
    check_env(env)
    print("Environment check passed!")

    # Initialize PPO agent
    model_path = f"{model_name}.zip"
    tensorboard_log = f"./{model_name}_tensorboard/"

    # Detect GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(
            model_path, env=env, tensorboard_log=tensorboard_log, device=device
        )
        reset_num_timesteps = False
    else:
        print(f"No existing model found, creating new PPO agent for {model_name}...")
        model = PPO(
            "MlpPolicy", env, verbose=1, tensorboard_log=tensorboard_log, device=device
        )
        reset_num_timesteps = True

    # Train
    print("Starting training...")
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=reset_num_timesteps)

    # Save model
    model.save(model_name)
    print(f"Model saved to {model_name}.zip")


if __name__ == "__main__":
    train()
