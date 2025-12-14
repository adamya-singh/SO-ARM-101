"""
SO-101 Gymnasium Environment for LeRobot Integration

A Gymnasium-compatible wrapper for the SO-101 robot arm MuJoCo simulation,
designed for teleoperation data collection and SmolVLA training.
"""

import os
import numpy as np
import gymnasium
from gymnasium import spaces
import mujoco
import mujoco.viewer

from so101_mujoco_utils import (
    get_camera_observation,
    get_robot_state,
    send_position_command,
    convert_to_dictionary,
    reset_env,
    set_initial_pose,
    compute_reward,
    reset_reward_state,
)


class SO101PickPlaceEnv(gymnasium.Env):
    """
    Gymnasium environment for SO-101 robot arm pick-and-place task.
    
    Observation space matches SmolVLA format:
    - observation.images.camera1: Top-down camera (256x256 RGB)
    - observation.images.camera2: Wrist camera (256x256 RGB)
    - observation.images.camera3: Side camera (256x256 RGB)
    - observation.state: Joint positions (6,) in radians
    
    Action space:
    - 6 joint positions in radians [shoulder_pan, shoulder_lift, elbow_flex,
      wrist_flex, wrist_roll, gripper]
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 30,
    }
    
    def __init__(
        self,
        render_mode=None,
        image_size=256,
        max_episode_steps=500,
        randomize_block=True,
        block_dist_range=(0.1, 0.3),
        block_angle_range=(-60, 60),
    ):
        """
        Initialize the SO-101 pick-and-place environment.
        
        Args:
            render_mode: "human" for viewer, "rgb_array" for image, None for no render
            image_size: Size of camera images (default 256x256)
            max_episode_steps: Maximum steps per episode before truncation
            randomize_block: Whether to randomize block position on reset
            block_dist_range: (min, max) distance from arm base in meters (default 0.1-0.3m)
            block_angle_range: (min, max) angle from center in degrees (default -60 to +60)
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.image_size = image_size
        self.max_episode_steps = max_episode_steps
        self.randomize_block = randomize_block
        self.block_dist_range = block_dist_range
        self.block_angle_range = block_angle_range
        
        # Load MuJoCo model
        model_path = os.path.join(os.path.dirname(__file__), "model", "scene.xml")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Renderer for camera observations (lazy init to support headless)
        self._renderer = None
        
        # Viewer for human rendering (lazy init)
        self.viewer = None
        
        # Default starting position (in degrees, matching existing code)
        self.starting_position = {
            'shoulder_pan': 0.06,
            'shoulder_lift': -100.21,
            'elbow_flex': 89.95,
            'wrist_flex': 66.46,
            'wrist_roll': 5.96,
            'gripper': 1.0,
        }
        
        # Joint limits from MJCF (in radians)
        self.joint_limits_low = np.array(
            [-1.9198621771937616, -1.7453292519943224, -1.69, -1.6580628494556928, -2.7438472969992493, -0.17453297762778586],
            dtype=np.float32
        )
        self.joint_limits_high = np.array(
            [1.9198621771937634, 1.7453292519943366, 1.69, 1.6580627293335335, 2.841206309382605, 1.7453291995659765],
            dtype=np.float32
        )
        
        # Define action space (joint positions in radians)
        self.action_space = spaces.Box(
            low=self.joint_limits_low,
            high=self.joint_limits_high,
            dtype=np.float32
        )
        
        # Define observation space matching SmolVLA format
        self.observation_space = spaces.Dict({
            "observation.images.camera1": spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            ),
            "observation.images.camera2": spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            ),
            "observation.images.camera3": spaces.Box(
                low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
            ),
            "observation.state": spaces.Box(
                low=self.joint_limits_low,
                high=self.joint_limits_high,
                dtype=np.float32
            ),
        })
        
        # Episode tracking
        self._step_count = 0
    
    @property
    def renderer(self):
        """Lazy initialization of renderer to support headless testing."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        return self._renderer
        
    def _get_obs(self):
        """
        Get current observation in SmolVLA format.
        
        Returns:
            dict with camera images and robot state
        """
        # Get camera images
        camera1_image = get_camera_observation(self.renderer, self.data, camera_name="camera_up")
        camera2_image = get_camera_observation(self.renderer, self.data, camera_name="wrist_camera")
        camera3_image = get_camera_observation(self.renderer, self.data, camera_name="camera_side")
        
        # Get robot state (joint positions in radians)
        state = get_robot_state(self.data).astype(np.float32)
        
        return {
            "observation.images.camera1": camera1_image,
            "observation.images.camera2": camera2_image,
            "observation.images.camera3": camera3_image,
            "observation.state": state,
        }
    
    def _get_info(self):
        """Get additional info about current state."""
        # Get gripper and block positions for debugging/logging
        gripper_pos = self.data.site("gripperframe").xpos.copy()
        block_pos = self.data.body("red_block").xpos.copy()
        distance = np.linalg.norm(gripper_pos - block_pos)
        
        return {
            "gripper_pos": gripper_pos,
            "block_pos": block_pos,
            "distance_to_block": distance,
            "block_height": block_pos[2],
            "step_count": self._step_count,
        }
    
    def reset(self, seed=None, options=None):
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., specific block position)
            
        Returns:
            observation: Initial observation
            info: Additional info dict
        """
        super().reset(seed=seed)
        
        # Determine block position
        if options is not None and "block_pos" in options:
            block_pos = options["block_pos"]
        elif self.randomize_block:
            # Randomize block position using polar coordinates
            # Distance from arm base (at origin)
            distance = self.np_random.uniform(*self.block_dist_range)
            # Angle from Y-axis (front of arm), in radians
            angle_deg = self.np_random.uniform(*self.block_angle_range)
            angle_rad = np.deg2rad(angle_deg)
            # Convert to cartesian (x = distance * sin(angle), y = distance * cos(angle))
            block_x = distance * np.sin(angle_rad)
            block_y = distance * np.cos(angle_rad)
            block_pos = (block_x, block_y, 0.0125)
        else:
            # Default position (directly in front at max distance)
            block_pos = (0, 0.3, 0.0125)
        
        # Reset environment
        reset_env(self.model, self.data, self.starting_position, block_pos)
        reset_reward_state()
        
        # Step physics a few times to settle
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
        
        # Reset episode counter
        self._step_count = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action):
        """
        Execute one environment step.
        
        Args:
            action: Joint positions (6,) in radians
            
        Returns:
            observation: New observation
            reward: Step reward
            terminated: True if episode ended (success)
            truncated: True if episode truncated (max steps)
            info: Additional info dict
        """
        # Clip action to valid range
        action = np.clip(action, self.joint_limits_low, self.joint_limits_high)
        
        # Convert action (radians) to position dict (degrees) for existing utils
        action_dict = convert_to_dictionary(action)
        
        # Apply action
        send_position_command(self.data, action_dict)
        
        # Step physics multiple times for stability
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        # Compute reward
        reward, success = compute_reward(self.model, self.data)
        
        # Update step count
        self._step_count += 1
        
        # Check termination conditions
        terminated = success  # Episode ends on success (block lifted)
        truncated = self._step_count >= self.max_episode_steps
        
        observation = self._get_obs()
        info = self._get_info()
        info["success"] = success
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        Returns:
            For rgb_array mode: RGB image array
            For human mode: None (renders to viewer window)
        """
        if self.render_mode == "rgb_array":
            # Return top camera view
            return get_camera_observation(self.renderer, self.data, camera_name="camera_up")
        
        elif self.render_mode == "human":
            # Launch or update viewer
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
    
    def close(self):
        """Clean up resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# Note: Environment registration is done in envs/__init__.py
# Import that package to register: `import envs`
# Or register manually with:
#   gymnasium.register(
#       id="SO101PickPlace-v0",
#       entry_point="so101_gym_env:SO101PickPlaceEnv",
#       max_episode_steps=500,
#   )

