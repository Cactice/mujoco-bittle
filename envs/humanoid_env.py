import gymnasium as gym
import numpy as np
import mujoco
import os


class HumanoidEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()

        # Load the humanoid XML
        urdf_path = os.path.join(
            os.path.dirname(__file__), "../assets/humanoid_final.xml"
        )
        self.model = mujoco.MjModel.from_xml_path(urdf_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode

        from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

        self.mujoco_renderer = MujocoRenderer(self.model, self.data)

        # Action space: 14 joints (including waist)
        self.joint_names = [
            "waist_joint",
            "torso_to_head_joint",
            "torso_to_right_shoulder_joint",
            "right_elbow_joint",
            "right_wrist_joint",
            "torso_to_left_shoulder_joint",
            "left_elbow_joint",
            "left_wrist_joint",
            "torso_to_right_hip_joint",
            "right_knee_joint",
            "right_ankle_joint",
            "torso_to_left_hip_joint",
            "left_knee_joint",
            "left_ankle_joint",
        ]

        # Define action space (joint positions)
        # Using a normalized action space [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(14,), dtype=np.float32
        )

        # Observation space:
        # - Joint positions (14)
        # - Joint velocities (14)
        # - Base orientation (4 - quaternion)
        # - Base angular velocity (3)
        # Total: 35
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(35,), dtype=np.float32
        )

        self.frame_skip = 5
        self.dt = self.model.opt.timestep * self.frame_skip

        # Identify leg geoms for breakdance task
        self.leg_geom_ids = set()
        leg_body_names = [
            "right_upper_leg_link",
            "right_lower_leg_link",
            "right_foot_link",
            "left_upper_leg_link",
            "left_lower_leg_link",
            "left_foot_link",
        ]
        for name in leg_body_names:
            try:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                if body_id != -1:
                    for geom_id in range(self.model.ngeom):
                        if self.model.geom_bodyid[geom_id] == body_id:
                            self.leg_geom_ids.add(geom_id)
            except Exception:
                pass

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)

        # Randomize initial body orientation (upside down with some variation)
        # Base orientation: 180 deg around Y (head down)
        # Add random tilt around X and Y axes
        tilt_x = self.np_random.uniform(-0.2, 0.2)  # Random tilt in radians
        tilt_y = self.np_random.uniform(-0.2, 0.2)

        # Create quaternion for upside-down + random tilt
        # Start with 180 deg rotation around Y: [0, 0, 1, 0] (w, x, y, z format varies)
        # MuJoCo uses [w, x, y, z] format
        base_angle = np.pi + tilt_y  # 180 deg + random Y tilt
        quat_y = [np.cos(base_angle / 2), 0, np.sin(base_angle / 2), 0]
        quat_x = [np.cos(tilt_x / 2), np.sin(tilt_x / 2), 0, 0]

        # Multiply quaternions (quat_y * quat_x)
        w1, x1, y1, z1 = quat_y
        w2, x2, y2, z2 = quat_x
        self.data.qpos[3:7] = [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]

        self.data.qpos[2] = 1.00  # Slightly above ground to avoid initial penetration

        # Randomize joint positions slightly
        num_joints = len(self.joint_names)
        self.data.qpos[7 : 7 + num_joints] += self.np_random.uniform(
            -1.0, 1.0, size=num_joints
        )

        # Randomize initial angular velocity for rotation (spin around local Z-axis)
        # qvel[3:6] is angular velocity (wx, wy, wz in local body frame)
        self.data.qvel[5] = self.np_random.uniform(8.0, 12.0)  # Random spin velocity

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        # Map action [-1, 1] to joint limits
        # Simple scaling for now
        ctrl = action * 1.0
        self.data.ctrl[:] = ctrl

        # Step simulation
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        terminated = False
        truncated = False

        # Reward: maximize z-axis angular velocity (spin)
        # qvel[5] is wz (world frame? No, local frame usually, but base is free joint)
        # qvel for free joint is in local frame of the body?
        # Actually, for free joint, qvel is 6D: 3 linear (world), 3 angular (local body frame).
        # If body is upside down, local Z is pointing down in world.
        # So spinning around local Z is what we want (headspin).
        spin_reward = np.abs(self.data.qvel[5])
        reward = spin_reward

        # Constraint: legs must not touch floor
        # Check contacts
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            g1 = contact.geom1
            g2 = contact.geom2

            # Check if one geom is a leg and the other is floor (or anything else really, but mostly floor)
            # Usually floor has a specific ID or name, but checking if it's NOT a robot part is safer.
            # But here, we just want to ensure legs don't touch *anything* external.
            # Since the only external thing is floor, checking if one is leg is enough?
            # No, self-collision (leg touching leg) is allowed.
            # So we need to check if one is leg and the other is NOT a robot part.
            # Or simpler: check if one is leg and the other is the floor geom.
            # Floor geom name is "floor".

            is_leg_contact = (g1 in self.leg_geom_ids) or (g2 in self.leg_geom_ids)

            if is_leg_contact:
                # Check if the other geom is floor
                other_geom = g2 if g1 in self.leg_geom_ids else g1
                other_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom
                )

                if other_name == "floor":
                    terminated = True
                    reward -= 100  # Penalty
                    break

        # Also terminate if head leaves ground? Maybe not strictly required, but "breakdance" implies headspin.
        # But user only said "without legs touching floor".
        # If it flies away, that's also technically satisfying constraints but maybe not "breakdance".
        # Let's add a penalty for flying too high? Or just let it be.

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        # Gather observations
        qpos = self.data.qpos.flat.copy()
        qvel = self.data.qvel.flat.copy()

        # qpos: [x, y, z, qw, qx, qy, qz, joint1, ... joint13]
        # qvel: [vx, vy, vz, wx, wy, wz, jvel1, ... jvel13]

        # Joint positions start at index 7 (0-2 pos, 3-6 quat)
        joint_pos = qpos[7:]
        joint_vel = qvel[6:]

        base_quat = qpos[3:7]
        base_ang_vel = qvel[3:6]

        return np.concatenate([joint_pos, joint_vel, base_quat, base_ang_vel]).astype(
            np.float32
        )

    def render(self):
        if self.render_mode == "human":
            return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if self.mujoco_renderer is not None:
            self.mujoco_renderer.close()
