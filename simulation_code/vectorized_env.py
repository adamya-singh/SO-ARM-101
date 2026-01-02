"""
Vectorized MuJoCo Environment for Parallel RL Training

Manages N parallel MuJoCo environments for batched policy inference,
enabling efficient GPU utilization on A100 and similar hardware.

Usage:
    vec_env = VectorizedMuJoCoEnv(num_envs=16, model_path='model/scene.xml', ...)
    vec_env.reset_all()
    
    for step in range(max_steps):
        obs = vec_env.get_batched_observations(device)
        actions = policy(obs)  # Batched inference
        rewards, dones = vec_env.step_all(actions)
"""

import numpy as np
import torch
import mujoco

from so101_mujoco_utils import (
    set_initial_pose,
    send_position_command,
    convert_to_list,
    convert_to_dictionary,
    normalize_state_for_smolvla,
    check_gripper_block_contact,
    check_block_gripped_with_force,
    get_floor_contact_force,
)


class VectorizedMuJoCoEnv:
    """
    Vectorized MuJoCo environment for parallel RL training.

    Manages N independent simulation states sharing a single model,
    enabling batched observations for efficient GPU inference.

    Args:
        num_envs: Number of parallel environments
        model_path: Path to MuJoCo XML model file
        starting_position: Dict of joint positions in degrees
        block_pos: Initial (x, y, z) position of the block
        lift_threshold: Height threshold for successful lift
        preprocessor: Optional PolicyProcessorPipeline for state normalization
        model_type: "smolvla" or "pi0" - for future model-specific handling
    """

    def __init__(
        self,
        num_envs: int,
        model_path: str,
        starting_position: dict,
        block_pos: tuple = (0, 0.3, 0.0125),
        lift_threshold: float = 0.08,
        preprocessor=None,
        model_type: str = "smolvla",
    ):
        self.num_envs = num_envs
        self.starting_position = starting_position
        self.block_pos = block_pos
        self.lift_threshold = lift_threshold
        self.preprocessor = preprocessor
        self.model_type = model_type
        
        # Single model shared across all environments (memory efficient)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        
        # Separate data and renderer per environment
        self.datas = [mujoco.MjData(self.model) for _ in range(num_envs)]
        self.renderers = [mujoco.Renderer(self.model, height=256, width=256) for _ in range(num_envs)]
        
        # Per-environment state tracking for rewards
        self.prev_gripper_pos = [None] * num_envs
        self.prev_block_pos = [None] * num_envs
        self.initial_block_pos = [None] * num_envs
        
        # Episode status
        self.dones = np.zeros(num_envs, dtype=bool)
        self.episode_steps = np.zeros(num_envs, dtype=int)
        
        print(f"[VectorizedEnv] Created {num_envs} parallel environments (model_type={model_type})")
    
    def reset_all(self):
        """Reset all environments to starting state."""
        for i in range(self.num_envs):
            self._reset_env(i)
        self.dones[:] = False
        self.episode_steps[:] = 0
    
    def reset_done_envs(self):
        """Reset only environments that are done (for continuous training)."""
        for i in range(self.num_envs):
            if self.dones[i]:
                self._reset_env(i)
        self.dones[:] = False
    
    def _reset_env(self, env_idx: int):
        """Reset a single environment."""
        d = self.datas[env_idx]
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, d)
        
        # Set robot to starting pose
        set_initial_pose(d, self.starting_position)
        
        # Reset block position
        d.qpos[6:9] = self.block_pos
        d.qpos[9:13] = [1, 0, 0, 0]  # Quaternion (upright)
        d.qvel[:] = 0
        
        # Forward kinematics
        mujoco.mj_forward(self.model, d)
        
        # Reset reward tracking state
        self.prev_gripper_pos[env_idx] = None
        self.prev_block_pos[env_idx] = None
        self.initial_block_pos[env_idx] = None
        self.episode_steps[env_idx] = 0
    
    def get_batched_observations(self, device: torch.device) -> dict:
        """
        Get observations from all environments as batched tensors.
        
        Returns:
            dict with batched observation tensors:
                - observation.images.camera1: (N, 3, 256, 256) top camera
                - observation.images.camera2: (N, 3, 256, 256) wrist camera
                - observation.images.camera3: (N, 3, 256, 256) side camera
                - observation.state: (N, 6) joint positions (normalized)
        """
        batch_top = []
        batch_wrist = []
        batch_side = []
        batch_state = []
        
        for i in range(self.num_envs):
            d = self.datas[i]
            renderer = self.renderers[i]
            
            # Render all three cameras
            renderer.update_scene(d, camera="camera_up")
            rgb_top = renderer.render().copy()
            
            renderer.update_scene(d, camera="wrist_camera")
            rgb_wrist = renderer.render().copy()
            
            renderer.update_scene(d, camera="camera_side")
            rgb_side = renderer.render().copy()
            
            # Get robot state (radians)
            state = d.qpos[:6].copy()
            
            batch_top.append(rgb_top)
            batch_wrist.append(rgb_wrist)
            batch_side.append(rgb_side)
            batch_state.append(state)
        
        # Stack into batched tensors
        # Images: (N, H, W, C) -> (N, C, H, W), normalized to [0, 1]
        batch_top = torch.from_numpy(np.stack(batch_top)).float().permute(0, 3, 1, 2) / 255.0
        batch_wrist = torch.from_numpy(np.stack(batch_wrist)).float().permute(0, 3, 1, 2) / 255.0
        batch_side = torch.from_numpy(np.stack(batch_side)).float().permute(0, 3, 1, 2) / 255.0
        
        # State: normalize for SmolVLA using preprocessor (MuJoCo radians -> physical -> normalized)
        batch_state_np = np.stack(batch_state)
        batch_state_normalized = np.stack([
            normalize_state_for_smolvla(s, preprocessor=self.preprocessor) for s in batch_state_np
        ])
        batch_state = torch.from_numpy(batch_state_normalized).float()
        
        return {
            "observation.images.camera1": batch_top.to(device),
            "observation.images.camera2": batch_wrist.to(device),
            "observation.images.camera3": batch_side.to(device),
            "observation.state": batch_state.to(device),
        }
    
    def step_all(self, actions_radians: np.ndarray, steps_per_action: int = 10) -> tuple:
        """
        Step all environments with given actions.
        
        Args:
            actions_radians: (N, 6) array of actions in radians
            steps_per_action: Number of physics steps per action
        
        Returns:
            rewards: (N,) array of rewards
            dones: (N,) boolean array indicating episode termination
        """
        rewards = np.zeros(self.num_envs)
        
        for i in range(self.num_envs):
            if self.dones[i]:
                # Skip done environments
                continue
            
            d = self.datas[i]
            action = actions_radians[i]
            
            # Convert to dictionary and execute
            action_dict = convert_to_dictionary(action)
            
            for _ in range(steps_per_action):
                send_position_command(d, action_dict)
                mujoco.mj_step(self.model, d)
            
            # Compute reward
            reward, done = self._compute_reward(i)
            rewards[i] = reward
            self.dones[i] = done
            self.episode_steps[i] += 1
        
        return rewards, self.dones.copy()
    
    def _compute_reward(self, env_idx: int) -> tuple:
        """
        SIMPLIFIED reward: Just negative distance from end effector to block.

        This is a minimal reward to verify the learning pipeline works.
        The reward is simply: -distance (so closer = higher reward = less negative)

        Range: approximately -0.5 (far) to 0.0 (touching block)
        """
        d = self.datas[env_idx]

        # Get positions
        gripper_pos = d.site("gripperframe").xpos.copy()
        block_pos = d.body("red_block").xpos.copy()

        # Simple distance reward: closer = better (less negative)
        distance = np.linalg.norm(gripper_pos - block_pos)
        reward = -distance

        # Check if block is lifted (for episode termination only, not reward)
        lifted = block_pos[2] > self.lift_threshold

        return reward, lifted
    
    def step_all_chunk(self, action_chunks: np.ndarray, steps_per_action: int = 10) -> tuple:
        """
        Execute full action chunk for each environment.
        
        This executes ALL actions in the chunk sequentially, accumulating rewards.
        Much more efficient than querying the policy for each action.
        
        Args:
            action_chunks: (N, chunk_size, 6) array of actions in radians
            steps_per_action: Number of physics steps per action
        
        Returns:
            total_rewards: (N,) array of accumulated rewards over all chunk actions
            dones: (N,) boolean array indicating episode termination
        """
        num_envs, chunk_size, _ = action_chunks.shape
        total_rewards = np.zeros(num_envs)
        
        for action_idx in range(chunk_size):
            # Get actions for this timestep across all environments
            actions_radians = action_chunks[:, action_idx, :]
            
            for i in range(num_envs):
                if self.dones[i]:
                    # Skip done environments
                    continue
                
                d = self.datas[i]
                action = actions_radians[i]
                
                # Convert to dictionary and execute
                action_dict = convert_to_dictionary(action)
                
                for _ in range(steps_per_action):
                    send_position_command(d, action_dict)
                    mujoco.mj_step(self.model, d)
                
                # Compute reward for this step
                reward, done = self._compute_reward(i)
                total_rewards[i] += reward
                self.dones[i] = done
                self.episode_steps[i] += 1
        
        return total_rewards, self.dones.copy()
    
    def get_episode_steps(self) -> np.ndarray:
        """Get current step count for each environment."""
        return self.episode_steps.copy()
    
    def close(self):
        """Clean up renderers."""
        for renderer in self.renderers:
            renderer.close()
        self.renderers = []

