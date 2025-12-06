from sbx import PPO
from envs.bittle_env import BittleEnv
from envs.humanoid_env import HumanoidEnv
import mujoco
import time
import argparse


def play():
    parser = argparse.ArgumentParser(description="Play MuJoCo simulation")
    parser.add_argument(
        "--robot",
        type=str,
        default="bittle",
        choices=["bittle", "humanoid"],
        help="Robot to simulate",
    )
    args = parser.parse_args()

    if args.robot == "bittle":
        env = BittleEnv(render_mode="human")
        model_name = "ppo_bittle"
    else:
        env = HumanoidEnv(render_mode="human")
        model_name = "ppo_humanoid_breakdance"

    # Load model if exists, else random
    try:
        model = PPO.load(model_name)
        print(f"Loaded trained model: {model_name}")
    except Exception:
        print(f"No trained model found for {args.robot}, using random agent.")
        model = None

    obs, _ = env.reset()

    # Create viewer
    # Run simulation
    while True:
        step_start = time.time()

        if model:
            action, _ = model.predict(obs)
        else:
            action = env.action_space.sample()

        obs, reward, terminated, truncated, _ = env.step(action)
        env.render()

        if terminated or truncated:
            obs, _ = env.reset()

        # Time keeping
        time_until_next_step = env.dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


if __name__ == "__main__":
    play()
