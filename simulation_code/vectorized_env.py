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
    create_reward_state_tracker,
    compute_pickup_reward_from_state,
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
        instruction: Task text used by processor-backed SmolVLA normalization
        block_pos: Initial (x, y, z) position of the block
        lift_threshold: Height threshold for successful lift
        distance_penalty_scale: Base scale for distance penalty
        horizontal_progress_scale: Reward scale for horizontal progress toward the block
        vertical_approach_scale: Reward scale for entering the grasp-height corridor
        approach_closeness_scale: Static pre-contact closeness shaping
        alignment_reward_cap: Max gated alignment reward before contact
        near_contact_bonus: Dense bridge reward for final pre-contact approach
        contact_entry_bonus: One-time reward when contact begins
        contact_persistence_reward: Per-step reward while contact is maintained
        hover_stall_threshold: Steps in the grasp corridor before stall penalty applies
        hover_penalty: Penalty for stalling in the corridor without contact
        bilateral_grasp_bonus: Reward for bilateral grasp activation
        grasp_persistence_reward: Per-step reward while grasp persists
        slip_penalty_contact: Penalty for losing sustained contact
        slip_penalty_grasp: Penalty for losing a grasp
        block_displacement_penalty_scale: Penalty scale for pushing the block away without lift
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
        instruction: str,
        block_pos: tuple = (0, 0.3, 0.0125),
        randomize_block: bool = False,
        block_dist_range: tuple = (0.1, 0.3),
        block_angle_range: tuple = (-60, 60),
        lift_threshold: float = 0.08,
        distance_penalty_scale: float = 0.4,
        horizontal_progress_scale: float = 0.08,
        vertical_approach_scale: float = 0.04,
        approach_closeness_scale: float = 0.015,
        alignment_reward_cap: float = 0.025,
        near_contact_bonus: float = 0.08,
        contact_entry_bonus: float = 0.30,
        contact_persistence_reward: float = 0.09,
        hover_stall_threshold: int = 8,
        hover_penalty: float = -0.01,
        bilateral_grasp_bonus: float = 0.50,
        grasp_persistence_reward: float = 0.15,
        slip_penalty_contact: float = -0.03,
        slip_penalty_grasp: float = -0.08,
        block_displacement_penalty_scale: float = 0.12,
        lift_bonus: float = 0.35,
        lift_bonus_threshold: float = 0.04,
        sustained_contact_threshold: int = 5,
        sustained_contact_bonus: float = 0.2,
        preprocessor=None,
        model_type: str = "smolvla",
    ):
        self.num_envs = num_envs
        self.starting_position = starting_position
        self.instruction = instruction
        self.block_pos = block_pos
        self.randomize_block = randomize_block
        self.block_dist_range = block_dist_range
        self.block_angle_range = block_angle_range
        self.lift_threshold = lift_threshold
        self.distance_penalty_scale = distance_penalty_scale
        self.horizontal_progress_scale = horizontal_progress_scale
        self.vertical_approach_scale = vertical_approach_scale
        self.approach_closeness_scale = approach_closeness_scale
        self.alignment_reward_cap = alignment_reward_cap
        self.near_contact_bonus = near_contact_bonus
        self.contact_entry_bonus = contact_entry_bonus
        self.contact_persistence_reward = contact_persistence_reward
        self.hover_stall_threshold = hover_stall_threshold
        self.hover_penalty = hover_penalty
        self.bilateral_grasp_bonus = bilateral_grasp_bonus
        self.grasp_persistence_reward = grasp_persistence_reward
        self.slip_penalty_contact = slip_penalty_contact
        self.slip_penalty_grasp = slip_penalty_grasp
        self.block_displacement_penalty_scale = block_displacement_penalty_scale
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
        self.reward_states = [create_reward_state_tracker() for _ in range(num_envs)]
        self.reset_rng = np.random.default_rng()
        
        # Episode status
        self.dones = np.zeros(num_envs, dtype=bool)
        self.episode_steps = np.zeros(num_envs, dtype=int)
        
        # Sustained contact tracking (per environment)
        self.consecutive_contact = np.zeros(num_envs, dtype=int)
        
        print(f"[VectorizedEnv] Created {num_envs} parallel environments (model_type={model_type})")
    
    def _sample_block_pos(self) -> tuple[float, float, float]:
        if not self.randomize_block:
            return tuple(self.block_pos)
        distance = self.reset_rng.uniform(*self.block_dist_range)
        angle_deg = self.reset_rng.uniform(*self.block_angle_range)
        angle_rad = np.deg2rad(angle_deg)
        block_x = distance * np.sin(angle_rad)
        block_y = distance * np.cos(angle_rad)
        return (block_x, block_y, self.block_pos[2])

    def reset_all(self, block_positions: list[tuple[float, float, float]] | None = None):
        """Reset all environments to starting state."""
        for i in range(self.num_envs):
            block_pos = None if block_positions is None else block_positions[i]
            self._reset_env(i, block_pos=block_pos)
        self.dones[:] = False
        self.episode_steps[:] = 0
        self.consecutive_contact[:] = 0
    
    def reset_done_envs(self):
        """Reset only environments that are done (for continuous training)."""
        for i in range(self.num_envs):
            if self.dones[i]:
                self._reset_env(i)
        self.dones[:] = False

    def _reset_env(self, env_idx: int, block_pos: tuple[float, float, float] | None = None):
        """Reset a single environment."""
        d = self.datas[env_idx]
        
        # Reset MuJoCo state
        mujoco.mj_resetData(self.model, d)
        
        # Set robot to starting pose
        set_initial_pose(d, self.starting_position)
        
        # Reset block position
        if block_pos is None:
            block_pos = self._sample_block_pos()
        d.qpos[6:9] = block_pos
        d.qpos[9:13] = [1, 0, 0, 0]  # Quaternion (upright)
        d.qvel[:] = 0
        
        # Forward kinematics
        mujoco.mj_forward(self.model, d)
        
        # Reset reward tracking state
        self.reward_states[env_idx] = create_reward_state_tracker()
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
        
        # State: normalize based on model type after MuJoCo->physical frame conversion.
        batch_state_np = np.stack(batch_state)
        batch_state_normalized = np.stack([
            normalize_state_for_vla(
                s,
                self.model_type,
                self.preprocessor,
                instruction=self.instruction,
            )
            for s in batch_state_np
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
            reward, done, contacted, gripped, sustained, height_aligned, *_ = self._compute_reward(i)
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
        Compute the staged pickup reward for one environment.
        
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
        reward, done, metrics = compute_pickup_reward_from_state(
            self.model,
            d,
            self.reward_states[env_idx],
            block_name="red_block",
            lift_threshold=self.lift_threshold,
            distance_penalty_scale=self.distance_penalty_scale,
            horizontal_progress_scale=self.horizontal_progress_scale,
            vertical_approach_scale=self.vertical_approach_scale,
            approach_closeness_scale=self.approach_closeness_scale,
            alignment_reward_cap=self.alignment_reward_cap,
            near_contact_bonus=self.near_contact_bonus,
            contact_entry_bonus=self.contact_entry_bonus,
            contact_persistence_reward=self.contact_persistence_reward,
            hover_stall_threshold=self.hover_stall_threshold,
            hover_penalty=self.hover_penalty,
            bilateral_grasp_bonus=self.bilateral_grasp_bonus,
            grasp_persistence_reward=self.grasp_persistence_reward,
            slip_penalty_contact=self.slip_penalty_contact,
            slip_penalty_grasp=self.slip_penalty_grasp,
            block_displacement_penalty_scale=self.block_displacement_penalty_scale,
            lift_bonus=self.lift_bonus,
            lift_bonus_threshold=self.lift_bonus_threshold,
            sustained_contact_threshold=self.sustained_contact_threshold,
            sustained_contact_bonus=self.sustained_contact_bonus,
        )
        self.consecutive_contact[env_idx] = self.reward_states[env_idx]["consecutive_contact"]
        return (
            reward,
            done,
            metrics["contacted"],
            metrics["gripped"],
            metrics["sustained"],
            metrics["height_aligned"],
            metrics["block_lifted"],
            metrics["contact_entry"],
            metrics["grasp_persistent"],
            metrics["lift_progress"],
            metrics["hover_stall"],
            metrics["slip_count"],
            metrics["block_displacement"],
            metrics["approach_reward"],
            metrics["alignment_reward"],
            metrics["near_contact"],
            metrics["contact_after_alignment"],
            metrics["horizontal_progress"],
            metrics["vertical_approach"],
            metrics["contact_loss_count"],
            metrics["grasp_loss_count"],
        )
    
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
            total_contact_entries: (N,) int array of contact-entry transition counts over chunk
            total_grasp_persistent: (N,) int array of persistent-grasp counts over chunk
            total_lift_progress: (N,) float array of cumulative lift progress over chunk
            total_hover_stall: (N,) int array of hover-stall counts over chunk
            total_slips: (N,) int array of slip events over chunk
            total_block_displacement: (N,) float array of cumulative block displacement over chunk
        """
        num_envs, chunk_size, _ = action_chunks.shape
        total_rewards = np.zeros(num_envs)
        total_contacts = np.zeros(num_envs, dtype=int)
        total_grasps = np.zeros(num_envs, dtype=int)
        total_sustained = np.zeros(num_envs, dtype=int)
        total_height_aligned = np.zeros(num_envs, dtype=int)
        total_contact_entries = np.zeros(num_envs, dtype=int)
        total_grasp_persistent = np.zeros(num_envs, dtype=int)
        total_lift_progress = np.zeros(num_envs, dtype=float)
        total_hover_stall = np.zeros(num_envs, dtype=int)
        total_slips = np.zeros(num_envs, dtype=int)
        total_block_displacement = np.zeros(num_envs, dtype=float)
        total_approach_reward = np.zeros(num_envs, dtype=float)
        total_alignment_reward = np.zeros(num_envs, dtype=float)
        total_near_contact = np.zeros(num_envs, dtype=int)
        total_contact_after_alignment = np.zeros(num_envs, dtype=int)
        total_horizontal_progress = np.zeros(num_envs, dtype=float)
        total_vertical_approach = np.zeros(num_envs, dtype=float)
        total_contact_losses = np.zeros(num_envs, dtype=int)
        total_grasp_losses = np.zeros(num_envs, dtype=int)
        
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
                (
                    reward,
                    done,
                    contacted,
                    gripped,
                    sustained,
                    height_aligned,
                    block_lifted,
                    contact_entry,
                    grasp_persistent,
                    lift_progress,
                    hover_stall,
                    slip_count,
                    block_displacement,
                    approach_reward,
                    alignment_reward,
                    near_contact,
                    contact_after_alignment,
                    horizontal_progress,
                    vertical_approach,
                    contact_loss_count,
                    grasp_loss_count,
                ) = self._compute_reward(i)
                total_rewards[i] += reward
                total_contacts[i] += int(contacted)
                total_grasps[i] += int(gripped)
                total_sustained[i] += int(sustained)
                total_height_aligned[i] += int(height_aligned)
                total_contact_entries[i] += int(contact_entry)
                total_grasp_persistent[i] += int(grasp_persistent)
                total_lift_progress[i] += float(lift_progress)
                total_hover_stall[i] += int(hover_stall)
                total_slips[i] += int(slip_count)
                total_block_displacement[i] += float(block_displacement)
                total_approach_reward[i] += float(approach_reward)
                total_alignment_reward[i] += float(alignment_reward)
                total_near_contact[i] += int(near_contact)
                total_contact_after_alignment[i] += int(contact_after_alignment)
                total_horizontal_progress[i] += float(horizontal_progress)
                total_vertical_approach[i] += float(vertical_approach)
                total_contact_losses[i] += int(contact_loss_count)
                total_grasp_losses[i] += int(grasp_loss_count)
                self.dones[i] = done
                self.episode_steps[i] += 1
        
        return (
            total_rewards,
            self.dones.copy(),
            total_contacts,
            total_grasps,
            total_sustained,
            total_height_aligned,
            total_contact_entries,
            total_grasp_persistent,
            total_lift_progress,
            total_hover_stall,
            total_slips,
            total_block_displacement,
            total_approach_reward,
            total_alignment_reward,
            total_near_contact,
            total_contact_after_alignment,
            total_horizontal_progress,
            total_vertical_approach,
            total_contact_losses,
            total_grasp_losses,
        )
    
    def get_episode_steps(self) -> np.ndarray:
        """Get current step count for each environment."""
        return self.episode_steps.copy()
    
    def close(self):
        """Clean up renderers."""
        for renderer in self.renderers:
            renderer.close()
        self.renderers = []
