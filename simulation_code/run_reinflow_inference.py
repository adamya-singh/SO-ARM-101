"""
Run inference with trained ReinFlow checkpoint.

This script loads a vanilla SmolVLA policy and applies the trained
action_out_proj weights from a ReinFlow checkpoint for deterministic inference.

Usage:
    python run_reinflow_inference.py
"""

import os
import time
import mujoco
import mujoco.viewer
import torch
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from so101_mujoco_utils import *

# ===== Configuration =====
CHECKPOINT_PATH = "reinflow_checkpoint.pt"
INSTRUCTION = "pick up the block"

m = mujoco.MjModel.from_xml_path('model/scene.xml')
d = mujoco.MjData(m)

# ===== SmolVLA Setup =====
# Check for device availability (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")

# Set up camera renderer for offscreen rendering at target resolution
renderer = mujoco.Renderer(m, height=256, width=256)

# Load SmolVLA policy
print("Loading SmolVLA policy...")
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy.to(device)
policy.eval()
print("SmolVLA policy loaded successfully!")

# Load trained weights from ReinFlow checkpoint
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    if 'action_out_proj' in checkpoint:
        policy.model.action_out_proj.load_state_dict(checkpoint['action_out_proj'])
        print(f"Loaded action head from {CHECKPOINT_PATH} (episode {checkpoint.get('episode', '?')})")
    if 'action_time_mlp_out' in checkpoint:
        policy.model.action_time_mlp_out.load_state_dict(checkpoint['action_time_mlp_out'])
        print(f"  Also loaded action_time_mlp_out weights")
else:
    print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
    print("  Running with base SmolVLA weights (untrained)")

# Load and attach tokenizer if missing
print("\nChecking tokenizer...")
if not hasattr(policy, 'tokenizer') or policy.tokenizer is None:
    print("Policy missing tokenizer. Loading SmolVLM2 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        policy.tokenizer = tokenizer
        print(f"Tokenizer loaded successfully: {type(tokenizer).__name__}")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        print("  Policy will use dummy tokens (language instructions will be ignored)")
else:
    print(f"Policy already has tokenizer: {type(policy.tokenizer).__name__}")

# ===== End SmolVLA Setup =====

starting_position = {
    'shoulder_pan': 0.06,  # degrees
    'shoulder_lift': -100.21,
    'elbow_flex': 89.95,
    'wrist_flex': 66.46,
    'wrist_roll': 5.96,
    'gripper': 1.0,  # 0-100 range for open and closed
}

set_initial_pose(d, starting_position)

# Policy control frequency settings
STEPS_PER_POLICY_UPDATE = 10  # Run policy every 10 physics steps
policy_step_counter = 0
policy_inference_count = 0
last_action_dict = starting_position.copy()

with mujoco.viewer.launch_passive(m, d) as viewer:
    start = time.time()
    hold_position(m, d, viewer, 2)  # hold starting position for 2 seconds

    print(f"\nStarting ReinFlow inference with instruction: '{INSTRUCTION}'")
    print(f"Running policy inference every {STEPS_PER_POLICY_UPDATE} physics steps")
    print(f"Debug output will be shown for first 3 policy inferences only")
    print("Press Ctrl+C to stop\n")

    while viewer.is_running() and time.time() - start < 300:
        step_start = time.time()

        # Only run policy inference every N steps to improve performance
        if policy_step_counter % STEPS_PER_POLICY_UPDATE == 0:
            try:
                policy_inference_count += 1
                DEBUG_THIS_ITERATION = (policy_inference_count <= 3)

                # Get all three camera observations (top, wrist, and side for SmolVLA)
                rgb_image_top = get_camera_observation(renderer, d, camera_name="camera_up")
                rgb_image_wrist = get_camera_observation(renderer, d, camera_name="wrist_camera")
                rgb_image_side = get_camera_observation(renderer, d, camera_name="camera_side")

                if DEBUG_THIS_ITERATION:
                    print(f"\n{'='*60}")
                    print(f"POLICY INFERENCE #{policy_inference_count}")
                    print(f"{'='*60}")
                    print(f"\n[Camera Debug]")
                    print(f"  Top camera shape: {rgb_image_top.shape}, dtype: {rgb_image_top.dtype}")
                    print(f"  Wrist camera shape: {rgb_image_wrist.shape}, dtype: {rgb_image_wrist.dtype}")
                    print(f"  Side camera shape: {rgb_image_side.shape}, dtype: {rgb_image_side.dtype}")

                # Get robot state
                robot_state = get_robot_state(d)

                # Prepare observation for policy
                observation = prepare_observation(rgb_image_top, rgb_image_wrist, rgb_image_side, robot_state, INSTRUCTION, device, policy, debug=DEBUG_THIS_ITERATION)

                if DEBUG_THIS_ITERATION:
                    print(f"\n[Observation Structure Debug]")
                    print(f"  Observation keys: {list(observation.keys())}")

                with torch.no_grad():
                    try:
                        if DEBUG_THIS_ITERATION:
                            print(f"\n[Policy Call Debug]")
                            print(f"  Attempting: policy.select_action(observation)")
                        action = policy.select_action(observation)
                        if DEBUG_THIS_ITERATION:
                            print(f"  select_action() succeeded")
                    except AttributeError as e:
                        if DEBUG_THIS_ITERATION:
                            print(f"  select_action() failed: {e}")
                            print(f"  Attempting: policy(observation)")
                        action = policy(observation)
                        if DEBUG_THIS_ITERATION:
                            print(f"  policy() call succeeded")
                    except Exception as e:
                        print(f"  Policy call failed with exception: {type(e).__name__}: {e}")
                        raise

                # Convert action to numpy if it's a tensor
                if torch.is_tensor(action):
                    action = action.cpu().numpy().squeeze()

                if DEBUG_THIS_ITERATION:
                    print(f"\n[Action Output Debug]")
                    print(f"  Raw normalized action: {action}")
                    print(f"  Action shape: {action.shape}")
                    print(f"  Normalized range: min={action.min():.4f}, max={action.max():.4f}")

                # Unnormalize action from SmolVLA output (normalized -> degrees -> radians)
                action_radians = unnormalize_action_from_smolvla(action)
                
                # Convert action to degrees dict for SO101
                last_action_dict = convert_to_dictionary(action_radians)

                if DEBUG_THIS_ITERATION:
                    print(f"  Unnormalized (radians): {action_radians}")
                    print(f"  Converted to degrees: {last_action_dict}")
                    print(f"\n{'='*60}\n")

                if policy_inference_count == 3:
                    print(f"\n{'='*60}")
                    print(f"Debug output complete. Policy continuing to run silently...")
                    print(f"{'='*60}\n")

            except Exception as e:
                print(f"Error in control loop: {e}")
                pass

        # Apply the action
        send_position_command(d, last_action_dict)
        policy_step_counter += 1

        # Step the physics simulation
        mujoco.mj_step(m, d)
        viewer.sync()

        # Timing
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

