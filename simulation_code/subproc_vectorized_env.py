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
    lift_threshold,
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
        normalize_state_for_smolvla,
        check_gripper_block_contact,
        check_block_gripped_with_force,
        get_floor_contact_force,
    )
    
    # Initialize MuJoCo model, data, and renderer
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=256, width=256)
    
    # Per-environment state tracking for rewards
    prev_gripper_pos = None
    prev_block_pos = None
    initial_block_pos = None
    
    def reset_env():
        """Reset the environment to starting state."""
        nonlocal prev_gripper_pos, prev_block_pos, initial_block_pos
        
        mujoco.mj_resetData(model, data)
        set_initial_pose(data, starting_position)
        
        # Reset block position
        data.qpos[6:9] = block_pos
        data.qpos[9:13] = [1, 0, 0, 0]  # Quaternion (upright)
        data.qvel[:] = 0
        
        mujoco.mj_forward(model, data)
        
        # Reset reward tracking
        prev_gripper_pos = None
        prev_block_pos = None
        initial_block_pos = None
    
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
        """Step the physics simulation and compute reward."""
        nonlocal prev_gripper_pos, prev_block_pos, initial_block_pos
        
        action_dict = convert_to_dictionary(action_radians)
        
        for _ in range(steps_per_action):
            send_position_command(data, action_dict)
            mujoco.mj_step(model, data)
        
        # Compute reward
        gripper_pos = data.site("gripperframe").xpos.copy()
        block_pos_current = data.body("red_block").xpos.copy()
        
        # Initialize tracking on first call
        if initial_block_pos is None:
            initial_block_pos = block_pos_current.copy()
        
        distance = np.linalg.norm(gripper_pos - initial_block_pos)
        
        reward = 0.0
        
        # 1. Linear distance penalty
        reward += -2.0 * distance
        
        # 2. Approach velocity bonus
        if prev_gripper_pos is not None and prev_block_pos is not None:
            prev_distance = np.linalg.norm(prev_gripper_pos - prev_block_pos)
            distance_delta = prev_distance - distance
            reward += 5.0 * distance_delta
        
        # ===== PHASE 1: Keep only basic rewards =====
        
        # 3. Close proximity bonus
        if distance < 0.05:
            reward += 1.0  # increased from 0.5
        
        # ===== PHASE 2: Uncomment after approach behavior works =====
        # (Commented out for Phase 1 - validate approach learning first)
        
        # # 4. Block height bonus
        # initial_block_z = 0.0125
        # height_gain = max(0, block_pos_current[2] - initial_block_z)
        # reward += 20.0 * height_gain
        
        # # 5. Contact bonus
        # if check_gripper_block_contact(model, data, "red_block"):
        #     reward += 3.0
        
        # # 6. Grip bonus
        # is_gripped, _ = check_block_gripped_with_force(model, data, "red_block")
        # if is_gripped:
        #     reward += 5.0
        
        # ===== PHASE 3: Uncomment after contact/grip learning works =====
        # (Commented out for Phase 1)
        
        # # 7. Success bonus
        # lifted = block_pos_current[2] > lift_threshold
        # if lifted:
        #     reward += 50.0
        
        # ===== PHASE 4: Add penalties last (volatile) =====
        # (Commented out for Phase 1)
        
        # # 8. Block displacement penalty
        # if block_pos_current[2] < 0.05:
        #     displacement = np.linalg.norm(block_pos_current[:2] - initial_block_pos[:2])
        #     threshold = 0.05
        #     if displacement > threshold:
        #         excess = displacement - threshold
        #         reward += -5.0 * (np.exp(10.0 * excess) - 1)
        
        # # 9. Floor contact penalty
        # floor_force = get_floor_contact_force(model, data)
        # if floor_force > 0:
        #     raw_penalty = -1.0 * np.exp(floor_force)
        #     reward += max(raw_penalty, -50.0)
        
        # ===== END PHASE SECTIONS =====
        
        # Update tracking state
        prev_gripper_pos = gripper_pos.copy()
        prev_block_pos = block_pos_current.copy()
        
        # Phase 1: Never done - let episodes run full length to learn approach
        lifted = False
        
        return reward, lifted
    
    # Main worker loop
    try:
        while True:
            cmd, payload = remote.recv()
            
            if cmd == 'reset':
                reset_env()
                remote.send('ok')
            
            elif cmd == 'get_obs':
                obs = get_observation()
                remote.send(obs)
            
            elif cmd == 'step':
                action_radians, steps_per_action = payload
                reward, done = step_physics(action_radians, steps_per_action)
                remote.send((reward, done))
            
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
        block_pos: Initial (x, y, z) position of the block
        lift_threshold: Height threshold for successful lift
    """
    
    def __init__(
        self,
        num_envs: int,
        model_path: str,
        starting_position: dict,
        block_pos: tuple = (0, 0.3, 0.0125),
        lift_threshold: float = 0.08,
    ):
        self.num_envs = num_envs
        self.model_path = model_path
        self.starting_position = starting_position
        self.block_pos = block_pos
        self.lift_threshold = lift_threshold
        
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
                    lift_threshold,
                    i,
                ),
                daemon=True,
            )
            process.start()
            self.processes.append(process)
            
            # Close child end in parent process
            child_conn.close()
        
        print(f"[SubprocVecEnv] Created {num_envs} parallel environments")
    
    def reset_all(self):
        """Reset all environments to starting state."""
        # Send reset command to all workers
        for conn in self.parent_conns:
            conn.send(('reset', None))
        
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
        from so101_mujoco_utils import normalize_state_for_smolvla
        
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
        
        # State: normalize for SmolVLA (radians -> degrees -> normalized)
        batch_state_np = np.stack(batch_state)
        batch_state_normalized = np.stack([
            normalize_state_for_smolvla(s) for s in batch_state_np
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
        """
        rewards = np.zeros(self.num_envs)
        
        # Send step commands to all workers (skip done environments)
        active_indices = []
        for i, conn in enumerate(self.parent_conns):
            if not self.dones[i]:
                conn.send(('step', (actions_radians[i], steps_per_action)))
                active_indices.append(i)
        
        # Collect results from active workers
        for i in active_indices:
            result = self.parent_conns[i].recv()
            if isinstance(result, tuple) and len(result) == 2:
                if result[0] == 'error':
                    raise RuntimeError(f"Worker {i} error: {result[1]}")
                reward, done = result
                rewards[i] = reward
                self.dones[i] = done
                self.episode_steps[i] += 1
        
        return rewards, self.dones.copy()
    
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

