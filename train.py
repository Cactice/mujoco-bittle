import argparse
from sbx import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.bittle_env import BittleEnv
from envs.humanoid_env import HumanoidEnv
import os
import jax


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
    parser.add_argument(
        "--n-envs", type=int, default=10, help="Number of parallel environments"
    )
    parser.add_argument(
        "--save-freq", type=int, default=50000, help="Save model every N timesteps"
    )
    args = parser.parse_args()

    # Create vectorized environments
    def make_env():
        if args.robot == "bittle":
            return lambda: BittleEnv()
        else:
            return lambda: HumanoidEnv()

    if args.robot == "bittle":
        model_name = "ppo_bittle"
    else:
        model_name = "ppo_humanoid_breakdance"

    # Create vectorized environment with n parallel environments
    print(f"Creating {args.n_envs} parallel environments...")
    env = SubprocVecEnv([make_env() for _ in range(args.n_envs)])
    print(f"Vectorized environment created with {args.n_envs} parallel environments!")

    # Initialize PPO agent
    model_path = f"{model_name}.zip"
    tensorboard_log = f"./{model_name}_tensorboard/"
    checkpoint_dir = f"./checkpoints/{model_name}/"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create checkpoint callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=checkpoint_dir,
        name_prefix=model_name,
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    # Detect GPU availability (JAX)
    devices = jax.devices()
    print(f"JAX devices available: {devices}")
    jax_device = devices[0].platform
    # Convert JAX device name to format expected by stable-baselines3
    device = "cuda" if jax_device == "gpu" else "cpu"
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
    print(
        f"Checkpoints will be saved every {args.save_freq} timesteps to {checkpoint_dir}"
    )
    model.learn(
        total_timesteps=args.timesteps,
        reset_num_timesteps=reset_num_timesteps,
        callback=checkpoint_callback,
    )

    # Save model
    model.save(model_name)
    print(f"Model saved to {model_name}.zip")


if __name__ == "__main__":
    train()
