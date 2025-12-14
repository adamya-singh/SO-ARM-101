import time
import mujoco
import mujoco.viewer
import numpy as np
import torch
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
#from so101_mujoco_utils import set_initial_pose, send_position_command
from so101_mujoco_utils import *

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
#policy = SmolVLAPolicy.from_pretrained("adamyathegreat/my_smolvla_pickplace")
policy.to(device)
policy.eval()
print("SmolVLA policy loaded successfully!")

# === SMOLVLA CONFIG INSPECTION ===
print("\n=== SMOLVLA CONFIG INSPECTION ===")
print(f"Image features: {policy.config.image_features}")
print(f"Action feature: {policy.config.action_feature}")
print(f"Max state dim: {policy.config.max_state_dim}")
print(f"Max action dim: {policy.config.max_action_dim}")
print(f"Chunk size: {policy.config.chunk_size}")
print(f"N action steps: {policy.config.n_action_steps}")
print(f"Adapt to pi_aloha: {policy.config.adapt_to_pi_aloha}")
print(f"Empty cameras: {policy.config.empty_cameras}")
print("\nFull config dict:")
for key, value in vars(policy.config).items():
    if not key.startswith('_'):
        print(f"  {key}: {value}")
print("=================================\n")

# === NORMALIZATION STATS ===
print("=== NORMALIZATION STATS ===")
if hasattr(policy, 'normalize_inputs'):
    print(f"Has normalize_inputs: True")
if hasattr(policy, 'unnormalize_outputs'):
    print(f"Has unnormalize_outputs: True")
if hasattr(policy, 'stats'):
    print(f"Stats: {policy.stats}")
if hasattr(policy, 'dataset_stats'):
    print(f"Dataset stats: {policy.dataset_stats}")
# Check for normalization in the config
print("Normalization-related attributes:")
for attr in dir(policy):
    if 'norm' in attr.lower() or 'stat' in attr.lower():
        try:
            val = getattr(policy, attr, 'N/A')
            if not callable(val):
                print(f"  {attr}: {val}")
        except:
            pass
print("===========================\n")

# Load and attach tokenizer if missing
print("\nChecking tokenizer...")
if not hasattr(policy, 'tokenizer') or policy.tokenizer is None:
    print("Policy missing tokenizer. Loading SmolVLM2 tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        policy.tokenizer = tokenizer
        print(f"✓ Tokenizer loaded successfully: {type(tokenizer).__name__}")
        
        # Test tokenizer
        test_tokens = tokenizer("pick up the red block", return_tensors="pt")
        print(f"✓ Tokenizer test: input shape {test_tokens['input_ids'].shape}, sample tokens: {test_tokens['input_ids'][0][:10].tolist()}")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        print("  Policy will use dummy tokens (language instructions will be ignored)")
else:
    print(f"✓ Policy already has tokenizer: {type(policy.tokenizer).__name__}")

# DEBUG: Check policy attributes
print("\n=== POLICY INSPECTION ===")
print(f"Policy type: {type(policy)}")
print(f"Policy has 'tokenizer': {hasattr(policy, 'tokenizer')}")
if hasattr(policy, 'tokenizer') and policy.tokenizer is not None:
    print(f"  Tokenizer type: {type(policy.tokenizer).__name__}")
print(f"Policy has 'select_action': {hasattr(policy, 'select_action')}")
print(f"Policy has 'forward': {hasattr(policy, 'forward')}")
if hasattr(policy, 'config'):
    print(f"Policy config keys: {policy.config.keys() if hasattr(policy.config, 'keys') else 'N/A'}")
# Check what the expected input keys are
if hasattr(policy, 'expected_image_keys'):
    print(f"Expected image keys: {policy.expected_image_keys}")
if hasattr(policy, 'input_shapes'):
    print(f"Input shapes: {policy.input_shapes}")
print("========================\n")

# Task instruction for SmolVLA
INSTRUCTION = "pick up the red block"

# ===== End SmolVLA Setup =====

starting_position = {
	'shoulder_pan': 0.06, #degrees
        'shoulder_lift': -100.21,
        'elbow_flex': 89.95,
        'wrist_flex': 66.46,
        'wrist_roll': 5.96,
        'gripper': 1.0,  #0-100 range for open and closed
}

all_zeros_position = {
    'shoulder_pan': 0.0,   # in degrees
    'shoulder_lift': 0.0,
    'elbow_flex': 0.0,
    'wrist_flex': 0.0,
    'wrist_roll': 0.0,
    'gripper': 0.0           # 0-100 range
}

set_initial_pose(d, starting_position) #set initial pose before starting sim viewer

# Policy control frequency settings
STEPS_PER_POLICY_UPDATE = 10  # Run policy every 10 physics steps
policy_step_counter = 0
policy_inference_count = 0  # Track number of policy inferences
last_action_dict = starting_position.copy()  # Initialize with starting pose

with mujoco.viewer.launch_passive(m, d) as viewer:
    #close the viewer automatically after 30 wall-seconds
    start = time.time()
    hold_position(m, d, viewer, 2) #hold starting position for 2 seconds

    print(f"\nStarting SmolVLA control with instruction: '{INSTRUCTION}'")
    print(f"Running policy inference every {STEPS_PER_POLICY_UPDATE} physics steps")
    print(f"Debug output will be shown for first 3 policy inferences only")
    print("Press Ctrl+C to stop\n")

    while viewer.is_running() and time.time() - start < 300:
        step_start = time.time()

        # ===== Old motion code (commented out) =====
        # move_to_pose(m, d, viewer, all_zeros_position, 2)
        # move_to_pose(m, d, viewer, starting_position, 2)
        # ===== End old motion code =====

        # ===== SmolVLA Control Loop =====
        # Only run policy inference every N steps to improve performance
        if policy_step_counter % STEPS_PER_POLICY_UPDATE == 0:
            try:
                policy_inference_count += 1
                DEBUG_THIS_ITERATION = (policy_inference_count <= 3)  # Only debug first 3 inferences
                
                # Get all three camera observations (top, wrist, and side for SmolVLA)
                rgb_image_top = get_camera_observation(renderer, d, camera_name="camera_up")
                rgb_image_wrist = get_camera_observation(renderer, d, camera_name="wrist_camera")
                rgb_image_side = get_camera_observation(renderer, d, camera_name="camera_side")
                
                # DEBUG: Check camera images (first inference only)
                if DEBUG_THIS_ITERATION:
                    print(f"\n{'='*60}")
                    print(f"POLICY INFERENCE #{policy_inference_count}")
                    print(f"{'='*60}")
                    print(f"\n[Camera Debug]")
                    print(f"  Top camera shape: {rgb_image_top.shape}, dtype: {rgb_image_top.dtype}")
                    print(f"  Top camera range: [{rgb_image_top.min()}, {rgb_image_top.max()}]")
                    print(f"  Wrist camera shape: {rgb_image_wrist.shape}, dtype: {rgb_image_wrist.dtype}")
                    print(f"  Wrist camera range: [{rgb_image_wrist.min()}, {rgb_image_wrist.max()}]")
                    print(f"  Side camera shape: {rgb_image_side.shape}, dtype: {rgb_image_side.dtype}")
                    print(f"  Side camera range: [{rgb_image_side.min()}, {rgb_image_side.max()}]")
                
                # Get robot state
                robot_state = get_robot_state(d)
                
                # Prepare observation for policy (includes tokenized instruction)
                observation = prepare_observation(rgb_image_top, rgb_image_wrist, rgb_image_side, robot_state, INSTRUCTION, device, policy, debug=DEBUG_THIS_ITERATION)
                
                # DEBUG: Verify observation structure matches policy expectations
                if DEBUG_THIS_ITERATION:
                    print(f"\n[Observation Structure Debug]")
                    print(f"  Observation keys: {list(observation.keys())}")
                    for key, value in observation.items():
                        if torch.is_tensor(value):
                            print(f"  {key}: shape={value.shape}, dtype={value.dtype}, device={value.device}")
                
                # Get action from SmolVLA policy
                if DEBUG_THIS_ITERATION:
                    print(f"\n[Policy Call Debug]")
                    print(f"  Calling policy inference...")
                    print(f"  Policy type: {type(policy)}")
                    print(f"  Policy has select_action: {hasattr(policy, 'select_action')}")
                
                with torch.no_grad():
                    # Try calling the policy - the exact method may vary
                    # Common patterns: policy.select_action(), policy(), or policy.generate()
                    try:
                        if DEBUG_THIS_ITERATION:
                            print(f"  Attempting: policy.select_action(observation)")
                        action = policy.select_action(observation)
                        if DEBUG_THIS_ITERATION:
                            print(f"  ✓ select_action() succeeded")
                    except AttributeError as e:
                        # If select_action doesn't exist, try calling the policy directly
                        if DEBUG_THIS_ITERATION:
                            print(f"  ✗ select_action() failed: {e}")
                            print(f"  Attempting: policy(observation)")
                        action = policy(observation)
                        if DEBUG_THIS_ITERATION:
                            print(f"  ✓ policy() call succeeded")
                    except Exception as e:
                        print(f"  ✗ Policy call failed with exception: {type(e).__name__}: {e}")
                        raise
                
                # Convert action to numpy if it's a tensor
                if torch.is_tensor(action):
                    action = action.cpu().numpy().squeeze()
                
                # DEBUG: Print action values to verify range
                if DEBUG_THIS_ITERATION:
                    print(f"\n[Action Output Debug]")
                    print(f"  Raw normalized action: {action}")
                    print(f"  Action shape: {action.shape}")
                    print(f"  Normalized range: min={action.min():.4f}, max={action.max():.4f}")
                    print(f"  Normalized per joint: {[f'{a:.4f}' for a in action]}")
                
                # Unnormalize action from SmolVLA output
                # SmolVLA outputs normalized actions (trained on degrees with mean/std normalization)
                # This converts: normalized -> degrees -> radians
                action_radians = unnormalize_action_from_smolvla(action)
                
                if DEBUG_THIS_ITERATION:
                    print(f"  Unnormalized (radians): {[f'{a:.4f}' for a in action_radians]}")
                    print(f"  Unnormalized (degrees): {[f'{np.degrees(a):.2f}' for a in action_radians]}")
                
                # Convert action from radians to degrees dict (SO101 format) using utility function
                last_action_dict = convert_to_dictionary(action_radians)
                
                # DEBUG: Print converted values
                if DEBUG_THIS_ITERATION:
                    print(f"  Final action dict (degrees): {last_action_dict}")
                    print(f"  Current robot qpos (radians): {d.qpos[:6]}")
                    print(f"\n{'='*60}\n")
                
                # After 3rd inference, print final message
                if policy_inference_count == 3:
                    print(f"\n{'='*60}")
                    print(f"Debug output complete. Policy continuing to run silently...")
                    print(f"{'='*60}\n")
                
            except Exception as e:
                print(f"Error in SmolVLA control loop: {e}")
                # On error, keep using the last action
                pass
        
        # Apply the last action (either newly computed or from previous inference)
        send_position_command(d, last_action_dict)
        policy_step_counter += 1
        # ===== End SmolVLA Control Loop =====

        # Step the physics simulation
        mujoco.mj_step(m, d)

        # pick up changes to the physics state, apply peturbations, update options from GUI
        viewer.sync()

        # rudimentary time keeping, will drift relative to wall clock
        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
