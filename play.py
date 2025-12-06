from stable_baselines3 import PPO
from envs.bittle_env import BittleEnv
import mujoco
import time

def play():
    env = BittleEnv(render_mode="human")

    # Load model if exists, else random
    try:
        model = PPO.load("ppo_bittle")
        print("Loaded trained model.")
    except Exception:
        print("No trained model found, using random agent.")
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
