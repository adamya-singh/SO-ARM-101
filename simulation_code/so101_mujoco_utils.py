import time
import mujoco
import numpy as np
import torch

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

def get_camera_observation(renderer, d, camera_name="wrist_camera"):
    """Render a camera and return RGB image as numpy array."""
    renderer.update_scene(d, camera=camera_name)
    rgb_array = renderer.render()
    return rgb_array

def get_robot_state(d):
    """Get current joint positions and velocities."""
    # qpos: joint positions (only first 6 are robot joints)
    # qvel: joint velocities (only first 6 are robot joints)
    state = np.concatenate([d.qpos[:6].copy(), d.qvel[:6].copy()])
    return state

def prepare_observation(rgb_image_top, rgb_image_side, robot_state, instruction, device, policy=None):
    """
    Prepare observation dict for SmolVLA policy with multiple cameras.
    Format based on LeRobot conventions.
    
    Args:
        rgb_image_top: numpy array of shape (H, W, C) from top camera with values in [0, 255]
        rgb_image_side: numpy array of shape (H, W, C) from side camera with values in [0, 255]
        robot_state: numpy array of robot state (positions + velocities)
        instruction: string with task instruction
        device: torch device (cuda, mps, or cpu)
        policy: SmolVLA policy object (needed for tokenization)
    
    Returns:
        observation: dict with images and state tensors
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
    
    # Convert side camera image to torch tensor and normalize
    image_side_tensor = torch.from_numpy(rgb_image_side).float() / 255.0
    image_side_tensor = image_side_tensor.permute(2, 0, 1)
    image_side_tensor = image_side_tensor.unsqueeze(0)
    image_side_tensor = image_side_tensor.to(device)
    
    # Convert robot state to torch tensor
    state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0).to(device)
    
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
    
    # Observation dictionary with both cameras
    # Using "camera1" and "camera2" keys to match the trained model's expected format
    observation = {
        "observation.images.camera1": image_top_tensor,
        "observation.images.camera2": image_side_tensor,
        "observation.state": state_tensor,
        "observation.language.tokens": language_tokens,
        "observation.language.attention_mask": attention_mask,
    }
    
    return observation