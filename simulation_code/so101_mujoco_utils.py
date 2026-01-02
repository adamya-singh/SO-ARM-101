import time
import mujoco
import numpy as np
import torch
from typing import Optional, Tuple, Any

# ===== MuJoCo to Physical Robot Coordinate Offset =====
# Physical Robot Position = MuJoCo Position (radians converted to physical units) + OFFSET
# This allows sim-trained policies to deploy on physical robots with different zero-points
MUJOCO_TO_PHYSICAL_OFFSET = np.array([
    0.0,    # shoulder_pan
    0.0,    # shoulder_lift
    0.0,    # elbow_flex
    0.0,    # wrist_flex
    0.0,    # wrist_roll
    0.0,    # gripper
])

# ===== Cached Processors =====
# Global cache for loaded processors to avoid reloading
_cached_preprocessor = None
_cached_postprocessor = None
_cached_pretrained_path = None


def load_smolvla_processors(pretrained_path: str = "lerobot/smolvla_base", policy_config=None) -> Tuple[Any, Any]:
    """
    Load preprocessor and postprocessor from a SmolVLA model repository.
    
    Uses LeRobot's PolicyProcessorPipeline to load normalization stats
    directly from the model. If Hub files don't exist, falls back to creating
    default processors from the policy config (identity normalization for RL training).
    
    Args:
        pretrained_path: HuggingFace model path or local checkpoint
        policy_config: Optional SmolVLAConfig for creating default processors
        
    Returns:
        (preprocessor, postprocessor) tuple for normalizing inputs and denormalizing outputs
        
    Raises:
        ImportError: If LeRobot's PolicyProcessorPipeline cannot be imported
        RuntimeError: If processors cannot be loaded and no policy_config provided
    """
    global _cached_preprocessor, _cached_postprocessor, _cached_pretrained_path
    
    # Return cached processors if already loaded for this path
    if _cached_pretrained_path == pretrained_path and _cached_preprocessor is not None:
        return _cached_preprocessor, _cached_postprocessor
    
    # Try to import PolicyProcessorPipeline
    try:
        from lerobot.processor import PolicyProcessorPipeline
    except ImportError as e:
        raise ImportError(
            f"Could not import PolicyProcessorPipeline from LeRobot.\n"
            f"Make sure you have the latest LeRobot installed with processor support.\n"
            f"Try: pip install --upgrade lerobot\n"
            f"Or check if your LeRobot fork has the PolicyProcessorPipeline class.\n"
            f"Original error: {e}"
        ) from e
    
    print(f"[SmolVLA] Loading processors from {pretrained_path}...")
    
    preprocessor = None
    postprocessor = None
    hub_load_failed = False
    
    # Try to load from HuggingFace Hub first
    try:
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path, config_filename="preprocessor_config.json"
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            pretrained_path, config_filename="postprocessor_config.json"
        )
        print(f"[SmolVLA] Processors loaded from Hub successfully!")
    except (FileNotFoundError, Exception) as e:
        error_msg = str(e)
        if "404" in error_msg or "Not Found" in error_msg or "Could not find" in error_msg:
            print(f"[SmolVLA] Processor configs not found on Hub (this is normal for older models)")
            hub_load_failed = True
        else:
            # Re-raise unexpected errors
            raise RuntimeError(
                f"Failed to load processors from '{pretrained_path}'.\n"
                f"Original error: {e}"
            ) from e
    
    # Fall back to creating default processors from policy config
    if hub_load_failed:
        if policy_config is None:
            raise RuntimeError(
                f"Could not load processors from Hub and no policy_config provided.\n"
                f"For RL training, pass policy_config=base_policy.config to load_smolvla_processors()."
            )
        
        try:
            from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors
            print(f"[SmolVLA] Creating default processors from policy config (identity normalization)...")
            print(f"[SmolVLA] Note: No dataset stats - using pass-through normalization for RL training")
            preprocessor, postprocessor = make_smolvla_pre_post_processors(
                config=policy_config,
                dataset_stats=None,  # Identity normalization - data passes through unchanged
            )
            print(f"[SmolVLA] Default processors created successfully!")
        except Exception as e:
            raise RuntimeError(
                f"Failed to create default processors from policy config.\n"
                f"Original error: {e}"
            ) from e
    
    # Cache for future use
    _cached_preprocessor = preprocessor
    _cached_postprocessor = postprocessor
    _cached_pretrained_path = pretrained_path
    
    # Print loaded stats for verification
    if hasattr(preprocessor, 'steps'):
        for step in preprocessor.steps:
            if hasattr(step, 'stats') and step.stats:
                print(f"[SmolVLA] Preprocessor has normalization stats")
                break
        else:
            print(f"[SmolVLA] Preprocessor using identity normalization (no stats)")
    
    return preprocessor, postprocessor


def mujoco_to_physical_state(state_radians: np.ndarray) -> np.ndarray:
    """
    Convert MuJoCo joint state (radians) to physical robot frame.
    
    This handles coordinate frame conversion WITHOUT normalization.
    The physical frame is what the SmolVLA model expects as input.
    
    Args:
        state_radians: numpy array of joint positions in radians (6,)
        
    Returns:
        state in physical robot frame (same units as training data)
    """
    # The smolvla_base model expects states in radians (based on LeRobot convention)
    # Apply offset for any zero-point differences between MuJoCo and physical robot
    physical_state = state_radians + np.radians(MUJOCO_TO_PHYSICAL_OFFSET)
    return physical_state


def physical_to_mujoco_action(physical_action: np.ndarray) -> np.ndarray:
    """
    Convert action from physical robot frame to MuJoCo frame (radians).
    
    This handles coordinate frame conversion WITHOUT normalization.
    
    Args:
        physical_action: numpy array of actions in physical robot frame (6,)
        
    Returns:
        action in MuJoCo radians
    """
    # Remove offset and keep in radians
    mujoco_action = physical_action - np.radians(MUJOCO_TO_PHYSICAL_OFFSET)
    return mujoco_action

def convert_to_dictionary(qpos):
    return {
            'shoulder_pan': qpos[0]*180.0/3.14159,  # convert radians(mujoco) to degrees(SO101)
            'shoulder_lift': qpos[1]*180.0/3.14159,
            'elbow_flex': qpos[2]*180.0/3.14159,
            'wrist_flex': qpos[3]*180.0/3.14159,
            'wrist_roll': qpos[4]*180.0/3.14159,
            'gripper': qpos[5]*100/3.14159,
        }

def convert_to_list(dictionary):
    return [
            dictionary['shoulder_pan']*3.14159/180.0,   # convert degrees(SO101) to radians(mujoco)
            dictionary['shoulder_lift']*3.14159/180.0,
            dictionary['elbow_flex']*3.14159/180.0,
            dictionary['wrist_flex']*3.14159/180.0,
            dictionary['wrist_roll']*3.14159/180.0,
            dictionary['gripper']*3.14159/100.0,
        ]

def set_initial_pose(d, position_dict):
    pos = convert_to_list(position_dict)
    d.qpos[:6] = pos  # Only set the first 6 elements (robot joints)

def send_position_command(d, position_dict):
    pos = convert_to_list(position_dict)
    d.ctrl = pos

def move_to_pose(m, d, viewer, desired_position, duration):
    start_time = time.time()
    starting_pose = d.qpos[:6].copy()  # Only get robot joints
    starting_pose = convert_to_dictionary(starting_pose)

    while True:
        t = time.time() - start_time
        if t > duration:
            break

        # interpolation factor [0,1] (make sure it doesn't exceed 1)
        alpha = min(t / duration, 1)

        # interpolate each joint
        position_dict = {}
        for joint in desired_position:
            p0 = starting_pose[joint]
            pf = desired_position[joint]
            position_dict[joint] = (1 - alpha) * p0 + alpha * pf
        
        # send command to sim
        send_position_command(d, position_dict)
        mujoco.mj_step(m, d)

        # pick up changes to the physics state, apply peturbations, update options from GUI
        viewer.sync()

def hold_position(m, d, viewer, duration):
    current_pos = d.qpos[:6].copy()  # Only get robot joints
    current_pos_dict = convert_to_dictionary(current_pos)

    start_time = time.time()
    while True:
        t = time.time() - start_time
        if t > duration:
            break
        send_position_command(d, current_pos_dict)
        mujoco.mj_step(m, d)
        viewer.sync()

# ===== SmolVLA Helper Functions =====

def normalize_state_for_smolvla(state_radians: np.ndarray, preprocessor) -> np.ndarray:
    """
    Normalize robot state for SmolVLA input.
    
    Pipeline:
    1. MuJoCo radians -> physical robot frame (apply offset)
    2. Apply preprocessor normalization
    
    Args:
        state_radians: numpy array of joint positions in radians (6,)
        preprocessor: PolicyProcessorPipeline for normalization (required)
    
    Returns:
        normalized state as numpy array (6,)
        
    Raises:
        ValueError: If preprocessor is None
        RuntimeError: If preprocessor fails to process the state
    """
    if preprocessor is None:
        raise ValueError(
            "preprocessor is required for normalize_state_for_smolvla().\n"
            "Load processors using: preprocessor, postprocessor = load_smolvla_processors(pretrained_path)"
        )
    
    # Step 1: Convert to physical robot frame
    physical_state = mujoco_to_physical_state(state_radians)
    
    # Step 2: Apply preprocessor normalization
    try:
        obs_dict = {"observation.state": torch.from_numpy(physical_state).float().unsqueeze(0)}
        processed = preprocessor(obs_dict)
        normalized = processed["observation.state"].squeeze(0).numpy()
        return normalized
    except Exception as e:
        raise RuntimeError(
            f"Preprocessor failed to normalize state.\n"
            f"Input state: {state_radians}\n"
            f"Physical state: {physical_state}\n"
            f"Original error: {e}"
        ) from e


def unnormalize_action_from_smolvla(action_normalized: np.ndarray, postprocessor) -> np.ndarray:
    """
    Unnormalize action output from SmolVLA.
    
    Pipeline:
    1. Apply postprocessor denormalization
    2. Physical robot frame -> MuJoCo radians (remove offset)
    
    Args:
        action_normalized: numpy array of normalized actions (6,) or (chunk_size, 6)
        postprocessor: PolicyProcessorPipeline for denormalization (required)
    
    Returns:
        action in MuJoCo radians as numpy array
        
    Raises:
        ValueError: If postprocessor is None
        RuntimeError: If postprocessor fails to process the action
    """
    if postprocessor is None:
        raise ValueError(
            "postprocessor is required for unnormalize_action_from_smolvla().\n"
            "Load processors using: preprocessor, postprocessor = load_smolvla_processors(pretrained_path)"
        )
    
    # Handle both single action and action chunks
    original_shape = action_normalized.shape
    is_chunk = len(original_shape) > 1
    
    if is_chunk:
        # Process each action in the chunk
        results = []
        for action in action_normalized:
            result = _unnormalize_single_action(action, postprocessor)
            results.append(result)
        return np.stack(results)
    else:
        return _unnormalize_single_action(action_normalized, postprocessor)


def _unnormalize_single_action(action_normalized: np.ndarray, postprocessor) -> np.ndarray:
    """Helper to unnormalize a single action."""
    # Apply postprocessor denormalization
    try:
        action_dict = {"action": torch.from_numpy(action_normalized).float().unsqueeze(0)}
        processed = postprocessor(action_dict)
        physical_action = processed["action"].squeeze(0).numpy()
    except Exception as e:
        raise RuntimeError(
            f"Postprocessor failed to denormalize action.\n"
            f"Input action: {action_normalized}\n"
            f"Original error: {e}"
        ) from e
    
    # Convert from physical robot frame to MuJoCo radians
    mujoco_action = physical_to_mujoco_action(physical_action)
    return mujoco_action


def get_camera_observation(renderer, d, camera_name="wrist_camera"):
    """Render a camera and return RGB image as numpy array."""
    renderer.update_scene(d, camera=camera_name)
    rgb_array = renderer.render()
    return rgb_array

def get_robot_state(d):
    """Get current joint positions only."""
    # SmolVLA base expects 6-dim state (joint positions only, no velocities)
    state = d.qpos[:6].copy()
    return state

def prepare_observation(rgb_image_top, rgb_image_wrist, rgb_image_side, robot_state, instruction, device, policy=None, preprocessor=None, debug=False):
    """
    Prepare observation dict for SmolVLA policy with multiple cameras.
    Format based on LeRobot conventions with SmolVLA standardized camera naming.
    
    Args:
        rgb_image_top: numpy array of shape (H, W, C) from top camera with values in [0, 255]
        rgb_image_wrist: numpy array of shape (H, W, C) from wrist camera with values in [0, 255]
        rgb_image_side: numpy array of shape (H, W, C) from side camera with values in [0, 255]
        robot_state: numpy array of robot state in RADIANS (from MuJoCo)
        instruction: string with task instruction
        device: torch device (cuda, mps, or cpu)
        policy: SmolVLA policy object (needed for tokenization)
        preprocessor: Optional PolicyProcessorPipeline for state normalization
    
    Returns:
        observation: dict with images and state tensors using SmolVLA standard keys
    """
    # Convert top camera image to torch tensor and normalize
    # Expected format: (C, H, W) with values in [0, 1]
    image_top_tensor = torch.from_numpy(rgb_image_top).float() / 255.0
    # Transpose from (H, W, C) to (C, H, W)
    image_top_tensor = image_top_tensor.permute(2, 0, 1)
    # Add batch dimension
    image_top_tensor = image_top_tensor.unsqueeze(0)
    # Move to device
    image_top_tensor = image_top_tensor.to(device)
    
    # Convert wrist camera image to torch tensor and normalize
    image_wrist_tensor = torch.from_numpy(rgb_image_wrist).float() / 255.0
    image_wrist_tensor = image_wrist_tensor.permute(2, 0, 1)
    image_wrist_tensor = image_wrist_tensor.unsqueeze(0)
    image_wrist_tensor = image_wrist_tensor.to(device)
    
    # Convert side camera image to torch tensor and normalize
    image_side_tensor = torch.from_numpy(rgb_image_side).float() / 255.0
    image_side_tensor = image_side_tensor.permute(2, 0, 1)
    image_side_tensor = image_side_tensor.unsqueeze(0)
    image_side_tensor = image_side_tensor.to(device)
    
    # Normalize robot state for SmolVLA using preprocessor if available
    normalized_state = normalize_state_for_smolvla(robot_state, preprocessor=preprocessor)
    state_tensor = torch.from_numpy(normalized_state).float().unsqueeze(0).to(device)
    
    # Tokenize the instruction if policy is provided
    if policy is not None and hasattr(policy, 'tokenizer'):
        # Tokenize the instruction
        tokens = policy.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # Move tokens to device and get input_ids and attention_mask
        language_tokens = tokens['input_ids'].to(device)
        # Convert attention_mask to boolean type
        attention_mask = tokens['attention_mask'].bool().to(device)
    else:
        # Fallback: create dummy tensors if no tokenizer available
        language_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
        attention_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
    
    # Observation dictionary with all three cameras
    # Using camera1/camera2/camera3 keys that the pretrained model expects
    # camera1: Top-down view (corresponds to OBS_IMAGE_1 in documentation)
    # camera2: Wrist-mounted view (corresponds to OBS_IMAGE_2 in documentation)
    # camera3: Side view (corresponds to OBS_IMAGE_3 in documentation)
    observation = {
        "observation.images.camera1": image_top_tensor,
        "observation.images.camera2": image_wrist_tensor,
        "observation.images.camera3": image_side_tensor,
        "observation.state": state_tensor,
        "observation.language.tokens": language_tokens,
        "observation.language.attention_mask": attention_mask,
    }
    
    # DEBUG: Check if tokenizer worked (only if debug=True)
    if debug:
        print(f"\n[Observation Preparation Debug]")
        print(f"  Instruction: '{instruction}'")
        print(f"  Policy provided: {policy is not None}")
        print(f"  Preprocessor provided: {preprocessor is not None}")
        if policy is not None:
            print(f"  Has tokenizer: {hasattr(policy, 'tokenizer')}")
            if hasattr(policy, 'tokenizer'):
                print(f"  Token shape: {language_tokens.shape}")
                print(f"  First 15 tokens: {language_tokens[0][:15].tolist()}")
                print(f"  Attention mask shape: {attention_mask.shape}")
            else:
                print(f"  WARNING: Policy has no tokenizer! Using dummy tokens.")
        print(f"  Image camera1 (top) shape: {image_top_tensor.shape}, range: [{image_top_tensor.min():.3f}, {image_top_tensor.max():.3f}]")
        print(f"  Image camera2 (wrist) shape: {image_wrist_tensor.shape}, range: [{image_wrist_tensor.min():.3f}, {image_wrist_tensor.max():.3f}]")
        print(f"  Image camera3 (side) shape: {image_side_tensor.shape}, range: [{image_side_tensor.min():.3f}, {image_side_tensor.max():.3f}]")
        print(f"  State shape: {state_tensor.shape}")
        print(f"  Raw state (radians): {robot_state.tolist()}")
        print(f"  Physical state (with offset): {mujoco_to_physical_state(robot_state).tolist()}")
        print(f"  Normalized state: {normalized_state.tolist()}")
    
    return observation

# ===== RL Training Helper Functions =====

# Store previous positions for velocity-based rewards
_prev_gripper_pos = None
_prev_block_pos = None
_initial_block_pos = None  # Track where block started (to avoid reward hacking)


def check_gripper_block_contact(m, d, block_name="red_block"):
    """
    Check if gripper or jaw is touching the block.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        block_name: Name of the block body
        
    Returns:
        bool: True if gripper/jaw is in contact with the block
    """
    block_body_id = m.body(block_name).id
    gripper_body_id = m.body("gripper").id
    jaw_body_id = m.body("moving_jaw_so101_v1").id
    
    for i in range(d.ncon):
        contact = d.contact[i]
        geom1_body = m.geom_bodyid[contact.geom1]
        geom2_body = m.geom_bodyid[contact.geom2]
        
        bodies = {geom1_body, geom2_body}
        if block_body_id in bodies:
            if gripper_body_id in bodies or jaw_body_id in bodies:
                return True
    return False


def check_block_gripped_with_force(m, d, block_name="red_block", min_force=0.1):
    """
    Check if block is gripped by analyzing contact forces from both gripper parts.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        block_name: Name of the block body
        min_force: Minimum force threshold (Newtons) to consider as contact
        
    Returns:
        gripped: bool - True if block is being squeezed by both sides
        grip_force: float - magnitude of the weaker grip force (N)
    """
    block_body_id = m.body(block_name).id
    gripper_body_id = m.body("gripper").id
    jaw_body_id = m.body("moving_jaw_so101_v1").id
    
    gripper_force_mag = 0.0
    jaw_force_mag = 0.0
    
    for i in range(d.ncon):
        contact = d.contact[i]
        geom1_body = m.geom_bodyid[contact.geom1]
        geom2_body = m.geom_bodyid[contact.geom2]
        
        bodies = {geom1_body, geom2_body}
        if block_body_id in bodies:
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, wrench)
            force_mag = np.linalg.norm(wrench[:3])
            
            if gripper_body_id in bodies:
                gripper_force_mag += force_mag
            if jaw_body_id in bodies:
                jaw_force_mag += force_mag
    
    # Gripped if both sides have sufficient force
    if gripper_force_mag > min_force and jaw_force_mag > min_force:
        return True, min(gripper_force_mag, jaw_force_mag)
    
    return False, 0.0


def get_floor_contact_force(m, d, floor_geom_name="floor"):
    """
    Measure total contact force between robot and floor.
    
    Returns:
        force_magnitude: float - total force magnitude (Newtons)
    """
    floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, floor_geom_name)
    
    total_force = np.zeros(3)
    
    for i in range(d.ncon):
        contact = d.contact[i]
        if contact.geom1 == floor_geom_id or contact.geom2 == floor_geom_id:
            # Skip floor-block contacts (we only care about robot-floor)
            block_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "red_block_geom")
            if contact.geom1 == block_geom_id or contact.geom2 == block_geom_id:
                continue
            
            wrench = np.zeros(6)
            mujoco.mj_contactForce(m, d, i, wrench)
            total_force += wrench[:3]
    
    return np.linalg.norm(total_force)


def compute_reward(m, d, block_name="red_block", lift_threshold=0.08):
    """
    SIMPLIFIED reward: Just negative distance from end effector to block.

    This is a minimal reward to verify the learning pipeline works.
    The reward is simply: -distance (so closer = higher reward = less negative)

    Range: approximately -0.5 (far) to 0.0 (touching block)

    Returns:
        reward: float - negative distance to block
        done: bool - True if block is lifted above threshold (for episode termination)
    """
    # Get gripper position (end effector)
    gripper_pos = d.site("gripperframe").xpos.copy()

    # Get block position (current, not initial)
    block_pos = d.body(block_name).xpos.copy()

    # Simple distance reward: closer = better (less negative)
    distance = np.linalg.norm(gripper_pos - block_pos)
    reward = -distance

    # Check if block is lifted (for episode termination only, not reward)
    lifted = block_pos[2] > lift_threshold

    return reward, lifted


def reset_reward_state():
    """Reset the reward state (call at episode start)."""
    global _prev_gripper_pos, _prev_block_pos, _initial_block_pos
    _prev_gripper_pos = None
    _prev_block_pos = None
    _initial_block_pos = None


def reset_env(m, d, starting_position, block_pos=(0, 0.3, 0.0125)):
    """
    Reset robot and block to initial positions.
    
    Args:
        m: MuJoCo model
        d: MuJoCo data
        starting_position: dict with joint positions in degrees
        block_pos: tuple (x, y, z) for block initial position
    """
    # Reset all state
    mujoco.mj_resetData(m, d)
    
    # Set robot to starting pose
    set_initial_pose(d, starting_position)
    
    # Reset block position (qpos indices after robot joints)
    # Block has freejoint: 3 pos + 4 quat = 7 DOF
    # Robot has 6 joints, so block starts at index 6
    d.qpos[6:9] = block_pos  # position (x, y, z)
    d.qpos[9:13] = [1, 0, 0, 0]  # quaternion (w, x, y, z) - upright
    
    # Zero out velocities
    d.qvel[:] = 0
    
    # Forward kinematics to update positions
    mujoco.mj_forward(m, d)