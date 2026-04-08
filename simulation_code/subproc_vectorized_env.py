"""
Subprocess-based Vectorized MuJoCo Environment for Parallel RL Training

Uses multiprocessing to parallelize CPU rendering across multiple processes.
Each worker process has its own MuJoCo model, data, and renderer, enabling
true parallel rendering on multi-core CPUs.

This is an alternative to VectorizedMuJoCoEnv that trades memory for speed
by running each environment in a separate process.

Usage:
    vec_env = SubprocMuJoCoEnv(num_envs=4, model_path='model/scene.xml', ...)
    vec_env.reset_all()
    
    for step in range(max_steps):
        obs = vec_env.get_batched_observations(device)  # Parallel rendering!
        actions = policy(obs)
        rewards, dones = vec_env.step_all(actions)
    
    vec_env.close()
"""

import multiprocessing as mp
import numpy as np
import torch
import traceback


def _worker(
    remote,
    parent_remote,
    model_path,
    starting_position,
    block_pos,
    randomize_block,
    block_dist_range,
    block_angle_range,
    lift_threshold,
    distance_penalty_scale,
    horizontal_progress_scale,
    vertical_approach_scale,
    approach_closeness_scale,
    alignment_reward_cap,
    near_contact_bonus,
    contact_entry_bonus,
    contact_persistence_reward,
    hover_stall_threshold,
    hover_penalty,
    bilateral_grasp_bonus,
    grasp_persistence_reward,
    slip_penalty_contact,
    slip_penalty_grasp,
    block_displacement_penalty_scale,
    lift_bonus,
    lift_bonus_threshold,
    sustained_contact_threshold,
    sustained_contact_bonus,
    worker_idx,
):
    """
    Worker process function that runs an independent MuJoCo environment.
    
    Each worker has its own model, data, and renderer to enable parallel rendering.
    Communicates with the main process via Pipe.
    
    Commands:
        - ('reset', None): Reset environment, returns 'ok'
        - ('get_obs', None): Render and return observations
        - ('step', (action, steps_per_action)): Step physics, return (reward, done)
        - ('close', None): Exit the worker loop
    """
    # Close the parent end in the child process
    parent_remote.close()
    
    # Setup MuJoCo rendering backend BEFORE importing mujoco
    # This is critical for headless rendering in subprocess
    import os
    import sys
    
    # On Linux, try to use EGL for headless rendering
    if sys.platform != 'darwin' and 'MUJOCO_GL' not in os.environ:
        try:
            import ctypes
            ctypes.CDLL('libEGL.so.1')
            os.environ['MUJOCO_GL'] = 'egl'
        except OSError:
            try:
                ctypes.CDLL('libOSMesa.so')
                os.environ['MUJOCO_GL'] = 'osmesa'
            except OSError:
                pass
    
    # Now import mujoco after setting up rendering
    import mujoco
    
    # Import utility functions
    from so101_mujoco_utils import (
        set_initial_pose,
        send_position_command,
        convert_to_dictionary,
        create_reward_state_tracker,
        compute_pickup_reward_from_state,
    )
    
    # Initialize MuJoCo model, data, and renderer
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=256, width=256)
    
    # Per-environment state tracking for rewards
    reward_state = create_reward_state_tracker()
    reset_rng = np.random.default_rng(seed=worker_idx)

    def sample_block_pos():
        if not randomize_block:
            return block_pos
        distance = reset_rng.uniform(*block_dist_range)
        angle_deg = reset_rng.uniform(*block_angle_range)
        angle_rad = np.deg2rad(angle_deg)
        block_x = distance * np.sin(angle_rad)
        block_y = distance * np.cos(angle_rad)
        return (block_x, block_y, block_pos[2])

    def reset_env(reset_block_pos=None):
        """Reset the environment to starting state."""
        nonlocal reward_state
        
        mujoco.mj_resetData(model, data)
        set_initial_pose(data, starting_position)
        
        # Reset block position
        if reset_block_pos is None:
            reset_block_pos = sample_block_pos()
        data.qpos[6:9] = reset_block_pos
        data.qpos[9:13] = [1, 0, 0, 0]  # Quaternion (upright)
        data.qvel[:] = 0
        
        mujoco.mj_forward(model, data)
        
        # Reset reward tracking
        reward_state = create_reward_state_tracker()
    
    def get_observation():
        """Render all cameras and return observations."""
        # Render all three cameras
        renderer.update_scene(data, camera="camera_up")
        rgb_top = renderer.render().copy()
        
        renderer.update_scene(data, camera="wrist_camera")
        rgb_wrist = renderer.render().copy()
        
        renderer.update_scene(data, camera="camera_side")
        rgb_side = renderer.render().copy()
        
        # Get robot state (radians)
        state = data.qpos[:6].copy()
        
        return (rgb_top, rgb_wrist, rgb_side, state)
    
    def step_physics(action_radians, steps_per_action):
        """Step the physics simulation and compute reward.
        
        Returns:
            reward: float, the computed reward
            done: bool, whether block is lifted above terminal threshold
            contacted: bool, whether gripper is touching block
            gripped: bool, whether both gripper sides are squeezing block
            sustained: bool, whether contact has been sustained above threshold
            height_aligned: bool, whether gripper is above block and close horizontally
            block_lifted: bool, whether block is elevated above lift_bonus_threshold
        """
        action_dict = convert_to_dictionary(action_radians)

        for _ in range(steps_per_action):
            send_position_command(data, action_dict)
            mujoco.mj_step(model, data)

        reward, done, metrics = compute_pickup_reward_from_state(
            model,
            data,
            reward_state,
            block_name="red_block",
            lift_threshold=lift_threshold,
            distance_penalty_scale=distance_penalty_scale,
            horizontal_progress_scale=horizontal_progress_scale,
            vertical_approach_scale=vertical_approach_scale,
            approach_closeness_scale=approach_closeness_scale,
            alignment_reward_cap=alignment_reward_cap,
            near_contact_bonus=near_contact_bonus,
            contact_entry_bonus=contact_entry_bonus,
            contact_persistence_reward=contact_persistence_reward,
            hover_stall_threshold=hover_stall_threshold,
            hover_penalty=hover_penalty,
            bilateral_grasp_bonus=bilateral_grasp_bonus,
            grasp_persistence_reward=grasp_persistence_reward,
            slip_penalty_contact=slip_penalty_contact,
            slip_penalty_grasp=slip_penalty_grasp,
            block_displacement_penalty_scale=block_displacement_penalty_scale,
            lift_bonus=lift_bonus,
            lift_bonus_threshold=lift_bonus_threshold,
            sustained_contact_threshold=sustained_contact_threshold,
            sustained_contact_bonus=sustained_contact_bonus,
        )
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
    
    # Main worker loop
    try:
        while True:
            cmd, payload = remote.recv()
            
            if cmd == 'reset':
                reset_env(payload)
                remote.send('ok')
            
            elif cmd == 'get_obs':
                obs = get_observation()
                remote.send(obs)
            
            elif cmd == 'step':
                action_radians, steps_per_action = payload
                remote.send(step_physics(action_radians, steps_per_action))
            
            elif cmd == 'close':
                renderer.close()
                remote.close()
                break
            
            else:
                raise ValueError(f"Unknown command: {cmd}")
    
    except Exception as e:
        traceback.print_exc()
        remote.send(('error', str(e)))
        remote.close()


class SubprocMuJoCoEnv:
    """
    Subprocess-based vectorized MuJoCo environment for parallel RL training.

    Each environment runs in its own process, enabling true parallel rendering
    across multiple CPU cores. This trades memory (each process loads the model)
    for rendering speed.

    Args:
        num_envs: Number of parallel environments (one process each)
        model_path: Path to MuJoCo XML model file
        starting_position: Dict of joint positions in degrees
        instruction: Task text used by processor-backed SmolVLA normalization
        block_pos: Initial (x, y, z) position of the block
        lift_threshold: Height threshold for successful lift
        distance_penalty_scale: Base distance penalty scale
        horizontal_progress_scale: Reward scale for horizontal approach progress
        vertical_approach_scale: Reward scale for vertical approach progress
        approach_closeness_scale: Static pre-contact closeness shaping
        alignment_reward_cap: Max alignment reward before contact
        near_contact_bonus: Dense reward in the final pre-contact corridor
        contact_entry_bonus: One-time bonus when contact begins
        contact_persistence_reward: Per-step reward while contact persists
        hover_stall_threshold: Steps before hover-stall penalty applies
        hover_penalty: Penalty for stalling in the corridor without touching
        bilateral_grasp_bonus: Reward for bilateral grasp activation
        grasp_persistence_reward: Per-step reward while grasp persists
        slip_penalty_contact: Penalty for losing sustained contact
        slip_penalty_grasp: Penalty for losing a grasp
        block_displacement_penalty_scale: Penalty scale for sideways block motion without lift
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
        self.model_path = model_path
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
        
        # Episode status tracking (maintained in main process)
        self.dones = np.zeros(num_envs, dtype=bool)
        self.episode_steps = np.zeros(num_envs, dtype=int)
        
        # Use spawn context for macOS compatibility
        ctx = mp.get_context('spawn')
        
        # Create pipes for communication
        self.parent_conns = []
        self.child_conns = []
        self.processes = []
        
        print(f"[SubprocVecEnv] Starting {num_envs} worker processes...")
        
        for i in range(num_envs):
            parent_conn, child_conn = ctx.Pipe()
            self.parent_conns.append(parent_conn)
            self.child_conns.append(child_conn)
            
            process = ctx.Process(
                target=_worker,
                args=(
                    child_conn,
                    parent_conn,
                    model_path,
                    starting_position,
                    block_pos,
                    randomize_block,
                    block_dist_range,
                    block_angle_range,
                    lift_threshold,
                    distance_penalty_scale,
                    horizontal_progress_scale,
                    vertical_approach_scale,
                    approach_closeness_scale,
                    alignment_reward_cap,
                    near_contact_bonus,
                    contact_entry_bonus,
                    contact_persistence_reward,
                    hover_stall_threshold,
                    hover_penalty,
                    bilateral_grasp_bonus,
                    grasp_persistence_reward,
                    slip_penalty_contact,
                    slip_penalty_grasp,
                    block_displacement_penalty_scale,
                    lift_bonus,
                    lift_bonus_threshold,
                    sustained_contact_threshold,
                    sustained_contact_bonus,
                    i,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            
            # Close child end in parent process
            child_conn.close()
        
        print(f"[SubprocVecEnv] Created {num_envs} parallel environments")
    
    def reset_all(self, block_positions: list[tuple[float, float, float]] | None = None):
        """Reset all environments to starting state."""
        # Send reset command to all workers
        for i, conn in enumerate(self.parent_conns):
            payload = None if block_positions is None else block_positions[i]
            conn.send(('reset', payload))
        
        # Wait for all resets to complete
        for conn in self.parent_conns:
            result = conn.recv()
            if result != 'ok':
                raise RuntimeError(f"Worker reset failed: {result}")
        
        self.dones[:] = False
        self.episode_steps[:] = 0
    
    def get_batched_observations(self, device: torch.device) -> dict:
        """
        Get observations from all environments as batched tensors.
        
        This is the key parallelization point - all workers render simultaneously!
        
        Returns:
            dict with batched observation tensors:
                - observation.images.camera1: (N, 3, 256, 256) top camera
                - observation.images.camera2: (N, 3, 256, 256) wrist camera
                - observation.images.camera3: (N, 3, 256, 256) side camera
                - observation.state: (N, 6) joint positions (normalized)
        """
        # Import here to avoid circular imports in workers
        from so101_mujoco_utils import normalize_state_for_vla
        
        # Send get_obs command to ALL workers simultaneously (non-blocking sends)
        for conn in self.parent_conns:
            conn.send(('get_obs', None))
        
        # Collect results - workers render in parallel while we wait!
        observations = []
        for conn in self.parent_conns:
            obs = conn.recv()
            if isinstance(obs, tuple) and len(obs) == 2 and obs[0] == 'error':
                raise RuntimeError(f"Worker error: {obs[1]}")
            observations.append(obs)
        
        # Unpack observations: each is (rgb_top, rgb_wrist, rgb_side, state)
        batch_top = []
        batch_wrist = []
        batch_side = []
        batch_state = []
        
        for rgb_top, rgb_wrist, rgb_side, state in observations:
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
        
        Note: Physics stepping is also parallelized across workers.
        
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
        contact_entries = np.zeros(self.num_envs, dtype=int)
        grasp_persistent = np.zeros(self.num_envs, dtype=int)
        lift_progress = np.zeros(self.num_envs, dtype=float)
        hover_stall = np.zeros(self.num_envs, dtype=int)
        slips = np.zeros(self.num_envs, dtype=int)
        block_displacement = np.zeros(self.num_envs, dtype=float)
        approach_reward = np.zeros(self.num_envs, dtype=float)
        alignment_reward = np.zeros(self.num_envs, dtype=float)
        near_contact = np.zeros(self.num_envs, dtype=int)
        contact_after_alignment = np.zeros(self.num_envs, dtype=int)
        horizontal_progress = np.zeros(self.num_envs, dtype=float)
        vertical_approach = np.zeros(self.num_envs, dtype=float)
        contact_loss_count = np.zeros(self.num_envs, dtype=int)
        grasp_loss_count = np.zeros(self.num_envs, dtype=int)
        
        # Send step commands to all workers (skip done environments)
        active_indices = []
        for i, conn in enumerate(self.parent_conns):
            if not self.dones[i]:
                conn.send(('step', (actions_radians[i], steps_per_action)))
                active_indices.append(i)
        
        # Collect results from active workers
        for i in active_indices:
            result = self.parent_conns[i].recv()
            if isinstance(result, tuple) and len(result) == 21:
                (
                    reward,
                    done,
                    contacted,
                    gripped,
                    sustained,
                    height_aligned,
                    block_lifted,
                    step_contact_entry,
                    step_grasp_persistent,
                    step_lift_progress,
                    step_hover_stall,
                    step_slip_count,
                    step_block_displacement,
                    step_approach_reward,
                    step_alignment_reward,
                    step_near_contact,
                    step_contact_after_alignment,
                    step_horizontal_progress,
                    step_vertical_approach,
                    step_contact_loss_count,
                    step_grasp_loss_count,
                ) = result
                rewards[i] = reward
                contacts[i] = int(contacted)
                grasps[i] = int(gripped)
                sustained_contacts[i] = int(sustained)
                height_alignments[i] = int(height_aligned)
                contact_entries[i] = int(step_contact_entry)
                grasp_persistent[i] = int(step_grasp_persistent)
                lift_progress[i] = float(step_lift_progress)
                hover_stall[i] = int(step_hover_stall)
                slips[i] = int(step_slip_count)
                block_displacement[i] = float(step_block_displacement)
                approach_reward[i] = float(step_approach_reward)
                alignment_reward[i] = float(step_alignment_reward)
                near_contact[i] = int(step_near_contact)
                contact_after_alignment[i] = int(step_contact_after_alignment)
                horizontal_progress[i] = float(step_horizontal_progress)
                vertical_approach[i] = float(step_vertical_approach)
                contact_loss_count[i] = int(step_contact_loss_count)
                grasp_loss_count[i] = int(step_grasp_loss_count)
                self.dones[i] = done
                self.episode_steps[i] += 1
            elif isinstance(result, tuple) and len(result) == 2:
                if result[0] == 'error':
                    raise RuntimeError(f"Worker {i} error: {result[1]}")
        
        return (
            rewards,
            self.dones.copy(),
            contacts,
            grasps,
            sustained_contacts,
            height_alignments,
            contact_entries,
            grasp_persistent,
            lift_progress,
            hover_stall,
            slips,
            block_displacement,
            approach_reward,
            alignment_reward,
            near_contact,
            contact_after_alignment,
            horizontal_progress,
            vertical_approach,
            contact_loss_count,
            grasp_loss_count,
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
            total_approach_reward: (N,) float array of cumulative approach shaping over chunk
            total_alignment_reward: (N,) float array of cumulative alignment reward over chunk
            total_near_contact: (N,) int array of near-contact counts over chunk
            total_contact_after_alignment: (N,) int array of aligned contact-entry counts over chunk
            total_horizontal_progress: (N,) float array of cumulative horizontal progress over chunk
            total_vertical_approach: (N,) float array of cumulative vertical approach progress over chunk
            total_contact_losses: (N,) int array of sustained-contact loss events
            total_grasp_losses: (N,) int array of grasp loss events
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
        
        # Execute each action in the chunk sequentially across all envs
        for action_idx in range(chunk_size):
            # Get actions for this timestep across all environments
            actions_radians = action_chunks[:, action_idx, :]
            
            # Step all active environments
            step_results = self.step_all(actions_radians, steps_per_action)
            (
                step_rewards,
                _,
                step_contacts,
                step_grasps,
                step_sustained,
                step_height_aligned,
                step_contact_entries,
                step_grasp_persistent,
                step_lift_progress,
                step_hover_stall,
                step_slips,
                step_block_displacement,
                step_approach_reward,
                step_alignment_reward,
                step_near_contact,
                step_contact_after_alignment,
                step_horizontal_progress,
                step_vertical_approach,
                step_contact_losses,
                step_grasp_losses,
            ) = step_results
            
            # Accumulate rewards and contacts
            total_rewards += step_rewards
            total_contacts += step_contacts
            total_grasps += step_grasps
            total_sustained += step_sustained
            total_height_aligned += step_height_aligned
            total_contact_entries += step_contact_entries
            total_grasp_persistent += step_grasp_persistent
            total_lift_progress += step_lift_progress
            total_hover_stall += step_hover_stall
            total_slips += step_slips
            total_block_displacement += step_block_displacement
            total_approach_reward += step_approach_reward
            total_alignment_reward += step_alignment_reward
            total_near_contact += step_near_contact
            total_contact_after_alignment += step_contact_after_alignment
            total_horizontal_progress += step_horizontal_progress
            total_vertical_approach += step_vertical_approach
            total_contact_losses += step_contact_losses
            total_grasp_losses += step_grasp_losses
            
            # If all environments are done, stop early
            if self.dones.all():
                break
        
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
        """Clean up all worker processes."""
        print("[SubprocVecEnv] Closing worker processes...")
        
        # Send close command to all workers
        for conn in self.parent_conns:
            try:
                conn.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass  # Worker may have already exited
        
        # Wait for processes to finish with timeout
        for i, process in enumerate(self.processes):
            process.join(timeout=5.0)
            if process.is_alive():
                print(f"  Worker {i} did not exit cleanly, terminating...")
                process.terminate()
                process.join(timeout=1.0)
        
        # Close parent connections
        for conn in self.parent_conns:
            conn.close()
        
        self.processes = []
        self.parent_conns = []
        print("[SubprocVecEnv] All workers closed")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        if hasattr(self, 'processes') and self.processes:
            self.close()
