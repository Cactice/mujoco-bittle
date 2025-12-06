from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from envs.bittle_env import BittleEnv
import os

def train():
    # Create environment
    env = BittleEnv()

    # Check environment
    print("Checking environment...")
    check_env(env)
    print("Environment check passed!")

    # Initialize PPO agent
    model_path = "ppo_bittle.zip"

    if os.path.exists(model_path):
        print(f"Loading existing model from {model_path}...")
        model = PPO.load(model_path, env=env, tensorboard_log="./ppo_bittle_tensorboard/")
        reset_num_timesteps = False
    else:
        print("No existing model found, creating new PPO agent...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./ppo_bittle_tensorboard/")
        reset_num_timesteps = True

    # Train
    print("Starting training...")
    model.learn(total_timesteps=100000, reset_num_timesteps=reset_num_timesteps)

    # Save model
    model.save("ppo_bittle")
    print("Model saved to ppo_bittle.zip")

if __name__ == "__main__":
    train()
