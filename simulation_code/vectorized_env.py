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
    normalize_state_for_vla,
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
        contact_bonus: Bonus reward while gripper contacts block
        height_alignment_bonus: Bonus reward when gripper is above block (top-down approach)
        grasp_bonus: Bonus reward when both sides of gripper squeeze block
        lift_bonus: Bonus reward when block is lifted above threshold
        lift_bonus_threshold: Height (meters) to trigger lift bonus
        sustained_contact_threshold: Frames of continuous contact before bonus triggers
        sustained_contact_bonus: Extra reward per step after sustained threshold reached
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
        contact_bonus: float = 0.1,
        height_alignment_bonus: float = 0.05,
        grasp_bonus: float = 0.15,
        lift_bonus: float = 0.2,
        lift_bonus_threshold: float = 0.04,
        sustained_contact_threshold: int = 5,
        sustained_contact_bonus: float = 0.2,
        preprocessor=None,
        model_type: str = "smolvla",
    ):
        self.num_envs = num_envs
        self.starting_position = starting_position
        self.block_pos = block_pos
        self.lift_threshold = lift_threshold
        self.contact_bonus = contact_bonus
        self.height_alignment_bonus = height_alignment_bonus
        self.grasp_bonus = grasp_bonus
        self.lift_bonus = lift_bonus
        self.lift_bonus_threshold = lift_bonus_threshold
        self.sustained_contact_threshold = sustained_contact_threshold
        self.sustained_contact_bonus = sustained_contact_bonus
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
        
        # Sustained contact tracking (per environment)
        self.consecutive_contact = np.zeros(num_envs, dtype=int)
        
        print(f"[VectorizedEnv] Created {num_envs} parallel environments (model_type={model_type})")
    
    def reset_all(self):
        """Reset all environments to starting state."""
        for i in range(self.num_envs):
            self._reset_env(i)
        self.dones[:] = False
        self.episode_steps[:] = 0
        self.consecutive_contact[:] = 0
    
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
        self.consecutive_contact[env_idx] = 0
    
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
        
        # State: normalize based on model type (SmolVLA uses hardcoded, Pi0 uses preprocessor)
        batch_state_np = np.stack(batch_state)
        batch_state_normalized = np.stack([
            normalize_state_for_vla(s, self.model_type, self.preprocessor) for s in batch_state_np
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
            contacts: (N,) int array of contact counts (0 or 1 per step)
            grasps: (N,) int array of grasp counts (0 or 1 per step)
            sustained_contacts: (N,) int array of sustained contact counts (0 or 1 per step)
            height_alignments: (N,) int array of height alignment counts (0 or 1 per step)
        """
        rewards = np.zeros(self.num_envs)
        contacts = np.zeros(self.num_envs, dtype=int)
        grasps = np.zeros(self.num_envs, dtype=int)
        sustained_contacts = np.zeros(self.num_envs, dtype=int)
        height_alignments = np.zeros(self.num_envs, dtype=int)
        
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
            reward, done, contacted, gripped, sustained, height_aligned, block_lifted = self._compute_reward(i)
            rewards[i] = reward
            contacts[i] = int(contacted)
            grasps[i] = int(gripped)
            sustained_contacts[i] = int(sustained)
            height_alignments[i] = int(height_aligned)
            self.dones[i] = done
            self.episode_steps[i] += 1
        
        return rewards, self.dones.copy(), contacts, grasps, sustained_contacts, height_alignments
    
    def _compute_reward(self, env_idx: int) -> tuple:
        """
        Reward with distance penalty + contact bonus + sustained contact + height alignment + grasp + lift.
        
        Components:
        - Distance: -distance (range: -0.5 to 0.0)
        - Contact bonus: +contact_bonus when gripper touches block
        - Sustained contact: +sustained_contact_bonus after threshold consecutive frames
        - Height alignment: +height_alignment_bonus when gripper is above block and close horizontally
        - Grasp bonus: +grasp_bonus when both sides of gripper squeeze block
        - Lift bonus: +lift_bonus when block is elevated above lift_bonus_threshold
        
        Total range per step: ~-0.5 to +0.70
        
        Returns:
            reward: float, the computed reward
            done: bool, whether block is lifted above terminal threshold
            contacted: bool, whether gripper is touching block
            gripped: bool, whether both gripper sides are squeezing block
            sustained: bool, whether contact has been sustained above threshold
            height_aligned: bool, whether gripper is above block and close horizontally
            block_lifted: bool, whether block is elevated above lift_bonus_threshold
        """
        d = self.datas[env_idx]

        # Get positions
        gripper_pos = d.site("gripperframe").xpos.copy()
        block_pos = d.body("red_block").xpos.copy()

        # Distance reward: closer = better (less negative)
        distance = np.linalg.norm(gripper_pos - block_pos)
        reward = -distance

        # Height alignment bonus: reward gripper being above block when close horizontally
        # Encourages top-down approach rather than sideways bumping
        horizontal_dist = np.linalg.norm(gripper_pos[:2] - block_pos[:2])
        height_above = gripper_pos[2] - block_pos[2]
        height_aligned = horizontal_dist < 0.1 and height_above > 0.02
        if height_aligned:
            reward += self.height_alignment_bonus

        # Contact bonus: positive signal while touching
        contacted = check_gripper_block_contact(self.model, d, "red_block")
        sustained = False
        if contacted:
            reward += self.contact_bonus
            # Track consecutive contact for sustained bonus
            self.consecutive_contact[env_idx] += 1
            if self.consecutive_contact[env_idx] >= self.sustained_contact_threshold:
                reward += self.sustained_contact_bonus
                sustained = True
        else:
            # Reset consecutive contact counter on contact loss
            self.consecutive_contact[env_idx] = 0

        # Grasp bonus: reward when both sides of gripper squeeze block
        gripped, _ = check_block_gripped_with_force(self.model, d, "red_block")
        if gripped:
            reward += self.grasp_bonus

        # Lift bonus: reward when block is elevated above threshold
        block_lifted = block_pos[2] > self.lift_bonus_threshold
        if block_lifted:
            reward += self.lift_bonus

        # Check if block is lifted high enough for episode termination
        done = block_pos[2] > self.lift_threshold

        return reward, done, contacted, gripped, sustained, height_aligned, block_lifted
    
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
            total_contacts: (N,) int array of contact counts over chunk
            total_grasps: (N,) int array of grasp counts over chunk
            total_sustained: (N,) int array of sustained contact counts over chunk
            total_height_aligned: (N,) int array of height alignment counts over chunk
        """
        num_envs, chunk_size, _ = action_chunks.shape
        total_rewards = np.zeros(num_envs)
        total_contacts = np.zeros(num_envs, dtype=int)
        total_grasps = np.zeros(num_envs, dtype=int)
        total_sustained = np.zeros(num_envs, dtype=int)
        total_height_aligned = np.zeros(num_envs, dtype=int)
        
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
                reward, done, contacted, gripped, sustained, height_aligned = self._compute_reward(i)
                total_rewards[i] += reward
                total_contacts[i] += int(contacted)
                total_grasps[i] += int(gripped)
                total_sustained[i] += int(sustained)
                total_height_aligned[i] += int(height_aligned)
                self.dones[i] = done
                self.episode_steps[i] += 1
        
        return total_rewards, self.dones.copy(), total_contacts, total_grasps, total_sustained, total_height_aligned
    
    def get_episode_steps(self) -> np.ndarray:
        """Get current step count for each environment."""
        return self.episode_steps.copy()
    
    def close(self):
        """Clean up renderers."""
        for renderer in self.renderers:
            renderer.close()
        self.renderers = []

