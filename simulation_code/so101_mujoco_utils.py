import time
import mujoco
import numpy as np
import torch

# ===== SmolVLA Normalization Stats =====
# These are the mean/std values from the SO-100 training data (in DEGREES)
# Used to normalize state inputs and unnormalize action outputs
SMOLVLA_STATE_MEAN = np.array([1.596, 119.944, 109.770, 56.706, -27.423, 12.003])
SMOLVLA_STATE_STD = np.array([26.392, 52.411, 49.854, 36.998, 59.360, 19.040])
SMOLVLA_ACTION_MEAN = np.array([1.596, 119.944, 109.770, 56.706, -27.423, 12.003])
SMOLVLA_ACTION_STD = np.array([26.392, 52.411, 49.854, 36.998, 59.360, 19.040])

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

def normalize_state_for_smolvla(state_radians):
    """
    Normalize robot state for SmolVLA input.
    
    SmolVLA was trained with states in DEGREES, normalized with mean/std.
    This function converts radians -> degrees -> normalized.
    
    Args:
        state_radians: numpy array of joint positions in radians (6,)
    
    Returns:
        normalized state as numpy array (6,)
    """
    # Convert radians to degrees
    state_degrees = np.degrees(state_radians)
    # Normalize using training stats: (x - mean) / std
    normalized = (state_degrees - SMOLVLA_STATE_MEAN) / SMOLVLA_STATE_STD
    return normalized


def unnormalize_action_from_smolvla(action_normalized):
    """
    Unnormalize action output from SmolVLA.
    
    SmolVLA outputs normalized actions that need to be converted:
    normalized -> degrees -> radians for MuJoCo.
    
    Args:
        action_normalized: numpy array of normalized actions (6,)
    
    Returns:
        action in radians as numpy array (6,)
    """
    # Unnormalize: action_degrees = normalized * std + mean
    action_degrees = action_normalized * SMOLVLA_ACTION_STD + SMOLVLA_ACTION_MEAN
    # Convert degrees to radians for MuJoCo
    action_radians = np.radians(action_degrees)
    return action_radians


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

def prepare_observation(rgb_image_top, rgb_image_wrist, rgb_image_side, robot_state, instruction, device, policy=None, debug=False):
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
    
    # Normalize robot state for SmolVLA (radians -> degrees -> normalized)
    normalized_state = normalize_state_for_smolvla(robot_state)
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
        if policy is not None:
            print(f"  Has tokenizer: {hasattr(policy, 'tokenizer')}")
            if hasattr(policy, 'tokenizer'):
                print(f"  Token shape: {language_tokens.shape}")
                print(f"  First 15 tokens: {language_tokens[0][:15].tolist()}")
                print(f"  Attention mask shape: {attention_mask.shape}")
            else:
                print(f"  ⚠️  WARNING: Policy has no tokenizer! Using dummy tokens.")
        print(f"  Image camera1 (top) shape: {image_top_tensor.shape}, range: [{image_top_tensor.min():.3f}, {image_top_tensor.max():.3f}]")
        print(f"  Image camera2 (wrist) shape: {image_wrist_tensor.shape}, range: [{image_wrist_tensor.min():.3f}, {image_wrist_tensor.max():.3f}]")
        print(f"  Image camera3 (side) shape: {image_side_tensor.shape}, range: [{image_side_tensor.min():.3f}, {image_side_tensor.max():.3f}]")
        print(f"  State shape: {state_tensor.shape}")
        print(f"  Raw state (radians): {robot_state.tolist()}")
        print(f"  Raw state (degrees): {np.degrees(robot_state).tolist()}")
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
    Compute shaped reward for pick-up task.
    
    Reward components:
    1. Linear distance penalty - consistent gradient signal at all distances
    2. Approach velocity bonus - reward for moving toward block
    3. Block height bonus - reward for lifting block
    4. Close proximity bonus - extra reward when very close
    5. Success bonus - large reward when block is lifted above threshold
    6. Block displacement penalty - exponential penalty for knocking block away (>5cm)
    7. Contact bonus - reward for gripper touching the block
    8. Grip bonus - additional reward for block squeezed between both gripper parts
    9. Floor contact penalty - exponential penalty for pressing against the floor
    
    Note: Distance is measured to the INITIAL block position, not current.
    This prevents the robot from learning to avoid the block (to not knock it away).
    
    Returns:
        reward: float - shaped reward
        done: bool - True if block is lifted above threshold
    """
    global _prev_gripper_pos, _prev_block_pos, _initial_block_pos
    
    # Get gripper position (end effector)
    gripper_pos = d.site("gripperframe").xpos.copy()
    
    # Get block position
    block_pos = d.body(block_name).xpos.copy()
    
    # Store initial block position on first call of episode
    if _initial_block_pos is None:
        _initial_block_pos = block_pos.copy()
    
    # Current distance to INITIAL block position (not current position)
    # This encourages going to where the block started, not chasing it if knocked
    distance = np.linalg.norm(gripper_pos - _initial_block_pos)
    
    reward = 0.0
    
    # 1. Linear distance penalty (consistent gradient at all distances)
    # Negative reward proportional to distance - always pushes toward block
    # At distance=0: penalty=0, at distance=0.3: penalty=-0.6, at distance=0.5: penalty=-1.0
    distance_penalty = -2.0 * distance
    reward += distance_penalty
    
    # 2. Approach velocity bonus (reward moving toward block)
    if _prev_gripper_pos is not None:
        prev_distance = np.linalg.norm(_prev_gripper_pos - _initial_block_pos)
        distance_delta = prev_distance - distance  # positive if getting closer
        approach_reward = 5.0 * distance_delta  # scale factor (increased from 4.0)
        reward += approach_reward
    
    # ===== PHASE 1: Keep only these basic rewards =====
    
    # 3. Close proximity bonus (extra reward when very close)
    if distance < 0.05:
        reward += 1.0  # bonus for being within 5cm (increased from 0.5)
    
    # 4. Block height bonus (reward lifting block above initial z=0.025)
    initial_block_z = 0.0125  # Half-size block
    height_gain = max(0, block_pos[2] - initial_block_z)
    height_reward = 20.0 * height_gain  # stronger reward for any lifting
    reward += height_reward
    
    # 5. Contact bonus (gripper touching the block!)
    if check_gripper_block_contact(m, d, block_name):
        reward += 3.0  # Bonus for making contact
    
    # 6. Grip bonus (block squeezed between both gripper parts!)
    is_gripped, grip_force = check_block_gripped_with_force(m, d, block_name)
    if is_gripped:
        reward += 5.0  # Additional bonus for actual grip
    
    # 7. Success bonus (block lifted above threshold)
    lifted = block_pos[2] > lift_threshold
    if lifted:
        reward += 50.0  # large success bonus
    
    # 8. Block displacement penalty - DISABLED for initial training
    # if block_pos[2] < 0.05:
    #     displacement = np.linalg.norm(block_pos[:2] - _initial_block_pos[:2])  # XY only
    #     threshold = 0.05  # 5cm tolerance
    #     if displacement > threshold:
    #         excess = displacement - threshold
    #         displacement_penalty = -5.0 * (np.exp(10.0 * excess) - 1)
    #         reward += displacement_penalty
    
    # 9. Floor contact penalty - DISABLED for initial training
    # floor_force = get_floor_contact_force(m, d)
    # if floor_force > 0:
    #     # Exponential scaling but capped to prevent reward explosion
    #     # At 1N: -2.7, at 5N: -50 (capped), at 10N+: -50 (capped)
    #     raw_penalty = -1.0 * np.exp(floor_force)
    #     floor_penalty = max(raw_penalty, -50.0)  # Cap at -50 (same magnitude as success bonus)
    #     reward += floor_penalty
    
    # Update previous positions for next step
    _prev_gripper_pos = gripper_pos.copy()
    _prev_block_pos = block_pos.copy()
    
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