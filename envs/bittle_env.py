import gymnasium as gym
import numpy as np
import mujoco
import os

class BittleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Load the fixed URDF
        urdf_path = os.path.join(os.path.dirname(__file__), "../assets/bittle_final.xml")
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer
        self.mujoco_renderer = MujocoRenderer(self.model, self.data)

        # Action space: 8 joints
        # New joint names from bittle_esp32.urdf (order matches XML definition)
        self.joint_names = [
            "shoulder_left", "shoulder_right",
            "hip_right", "hip_left",
            "elbow_left", "elbow_right",
            "knee_right", "knee_left"
        ]

        # Define action space (joint positions)
        # Using a normalized action space [-1, 1] which we map to joint limits
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(8,), dtype=np.float32)

        # Observation space:
        # - Joint positions (8)
        # - Joint velocities (8)
        # - Base orientation (4 - quaternion)
        # - Base angular velocity (3)
        # Total: 23
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)



        # Initial pose: slightly above ground
        self.data.qpos[2] = 0.1  # z-height

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        # Map action [-1, 1] to joint limits
        # For simplicity, let's assume a range of [-1, 1] radians for now,
        # or use the limits from the model if we wanted to be precise.
        # The URDF has limits approx -1.5 to 1.5.

        ctrl = action * 1.5
        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        # Compute reward
        # 1. Forward velocity reward
        forward_vel = self.data.qvel[0] # x-velocity

        # 2. Stability reward (penalty for large roll/pitch)
        # qpos[3:7] is quaternion (w, x, y, z)
        # Simple penalty for being upside down or tilted too much
        # This is a placeholder.

        # 3. Energy penalty
        energy = np.sum(np.square(action))

        reward = forward_vel - 0.01 * energy

        # Check termination
        # Terminate if z-height is too low (fallen)
        terminated = bool(self.data.qpos[2] < 0.05)
        truncated = False

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # Gather observations
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        # qpos: [x, y, z, qw, qx, qy, qz, joint1, ... joint8]
        # qvel: [vx, vy, vz, wx, wy, wz, jvel1, ... jvel8]

        # We want: joint positions (8), joint velocities (8), base orientation (4), base ang vel (3)

        # Joint positions start at index 7 (0-2 pos, 3-6 quat)
        joint_pos = qpos[7:]
        joint_vel = qvel[6:]

        base_quat = qpos[3:7]
        base_ang_vel = qvel[3:6]

        return np.concatenate([joint_pos, joint_vel, base_quat, base_ang_vel]).astype(np.float32)

    def render(self):
        if self.render_mode == "human":
            return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
