"""
Run inference with trained ReinFlow checkpoint.

This script loads a VLA policy (SmolVLA or Pi0) and applies the trained
weights from a ReinFlow checkpoint for deterministic inference.

Auto-detects model type from checkpoint metadata, or can be specified manually.

Usage:
    python run_reinflow_inference.py
    
    # Specify model type manually:
    python run_reinflow_inference.py --model-type pi0 --checkpoint reinflow_pi0_checkpoint.pt
    
    # Headless mode (for Colab/SSH):
    MUJOCO_GL=osmesa python run_reinflow_inference.py
"""

import os
import time
import argparse

# Setup headless rendering BEFORE importing mujoco
from mujoco_rendering import setup_mujoco_rendering
setup_mujoco_rendering()

import mujoco
import mujoco.viewer
import torch
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from so101_mujoco_utils import *
from reinflow_smolvla import detect_model_type_from_checkpoint

# ===== Configuration =====
parser = argparse.ArgumentParser(description='Run ReinFlow inference with trained checkpoint')
parser.add_argument('--checkpoint', type=str, default='reinflow_checkpoint.pt',
                    help='Path to ReinFlow checkpoint')
parser.add_argument('--model-type', type=str, choices=['smolvla', 'pi0', 'auto'], default='auto',
                    help='Model type (auto-detect from checkpoint by default)')
parser.add_argument('--pretrained', type=str, default=None,
                    help='Path to pretrained model (auto-selected based on model type if not specified)')
parser.add_argument('--no-quantize', action='store_true',
                    help='Disable 4-bit quantization for Pi0 on MPS (uses more memory)')
args, _ = parser.parse_known_args()

CHECKPOINT_PATH = args.checkpoint
INSTRUCTION = "pick up the block"

# Auto-detect model type from checkpoint
if args.model_type == 'auto' and os.path.exists(CHECKPOINT_PATH):
    MODEL_TYPE = detect_model_type_from_checkpoint(CHECKPOINT_PATH)
    print(f"Auto-detected model type: {MODEL_TYPE}")
else:
    MODEL_TYPE = args.model_type if args.model_type != 'auto' else 'smolvla'

# Set pretrained path based on model type
if args.pretrained:
    PRETRAINED_PATH = args.pretrained
elif MODEL_TYPE == "pi0":
    PRETRAINED_PATH = "lerobot/pi0"
else:
    PRETRAINED_PATH = "lerobot/smolvla_base"

m = mujoco.MjModel.from_xml_path('model/scene.xml')
d = mujoco.MjData(m)

# ===== VLA Setup =====
# Check for device availability (MPS for Apple Silicon, CUDA for NVIDIA, else CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Apple Silicon GPU)")
    if MODEL_TYPE == "pi0":
        print("  WARNING: Pi0 on MPS may have limited support.")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")
    if MODEL_TYPE == "pi0":
        print("  WARNING: Pi0 on CPU will be very slow.")

# Set up camera renderer for offscreen rendering at target resolution
renderer = mujoco.Renderer(m, height=256, width=256)

# Load policy based on model type
print(f"\n=== Loading {MODEL_TYPE.upper()} Policy ===")
if MODEL_TYPE == "pi0":
    from pi0_quantization import load_pi0_quantized, should_quantize_pi0
    
    # Use quantized loading for Pi0 (especially on MPS)
    use_quantization = not args.no_quantize and should_quantize_pi0(device)
    
    if use_quantization:
        print(f"Loading Pi0 policy from {PRETRAINED_PATH} with 4-bit quantization...")
    else:
        print(f"Loading Pi0 policy from {PRETRAINED_PATH}...")
    
    policy, was_quantized = load_pi0_quantized(
        PRETRAINED_PATH,
        device=str(device),
        quantize=not args.no_quantize,
        verbose=True
    )
    
    if was_quantized:
        print("Pi0 policy loaded with INT4 quantization!")
    else:
        print("Pi0 policy loaded successfully!")
else:
    print(f"Loading SmolVLA policy from {PRETRAINED_PATH}...")
    policy = SmolVLAPolicy.from_pretrained(PRETRAINED_PATH)
    policy.to(device)
    policy.eval()
    print("SmolVLA policy loaded successfully!")

# Load processors for normalization/denormalization
print("Loading processors...")
preprocessor, postprocessor = load_vla_processors(MODEL_TYPE, PRETRAINED_PATH, policy_config=policy.config)
print("Processors loaded successfully!")

# Load trained weights from ReinFlow checkpoint
if os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    
    # Load action head weights
    if 'action_out_proj' in checkpoint:
        policy.model.action_out_proj.load_state_dict(checkpoint['action_out_proj'])
        print(f"Loaded action head from {CHECKPOINT_PATH} (episode {checkpoint.get('episode', '?')})")
    
    # Load time MLP weights
    if 'action_time_mlp_out' in checkpoint:
        policy.model.action_time_mlp_out.load_state_dict(checkpoint['action_time_mlp_out'])
        print(f"  Also loaded action_time_mlp_out weights")
    if 'action_time_mlp_in' in checkpoint:
        policy.model.action_time_mlp_in.load_state_dict(checkpoint['action_time_mlp_in'])
        print(f"  Also loaded action_time_mlp_in weights")
    
    # Load expert transformer weights (if full expert was trained)
    if MODEL_TYPE == "pi0" and 'gemma_expert' in checkpoint:
        policy.model.paligemma_with_expert.gemma_expert.load_state_dict(checkpoint['gemma_expert'])
        print(f"  Also loaded gemma_expert weights")
    elif MODEL_TYPE == "smolvla" and 'expert' in checkpoint:
        policy.model.vlm_with_expert.expert.load_state_dict(checkpoint['expert'])
        print(f"  Also loaded expert transformer weights")
    
    # Load other trained components
    if 'action_in_proj' in checkpoint:
        policy.model.action_in_proj.load_state_dict(checkpoint['action_in_proj'])
        print(f"  Also loaded action_in_proj weights")
    if 'state_proj' in checkpoint:
        policy.model.state_proj.load_state_dict(checkpoint['state_proj'])
        print(f"  Also loaded state_proj weights")
else:
    print(f"WARNING: Checkpoint not found at {CHECKPOINT_PATH}")
    print(f"  Running with base {MODEL_TYPE.upper()} weights (untrained)")

# Load and attach tokenizer if missing
print("\nChecking tokenizer...")
if not hasattr(policy, 'tokenizer') or policy.tokenizer is None:
    if MODEL_TYPE == "pi0":
        print("Loading PaliGemma tokenizer for Pi0...")
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224", trust_remote_code=True)
            policy.tokenizer = tokenizer
            print(f"Tokenizer loaded successfully: {type(tokenizer).__name__}")
        except Exception as e:
            print(f"Failed to load tokenizer: {e}")
            print("  Policy will use dummy tokens")
    else:
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

# ===== End VLA Setup =====

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

                # Prepare observation for policy (with preprocessor for state normalization)
                observation = prepare_observation(rgb_image_top, rgb_image_wrist, rgb_image_side, robot_state, INSTRUCTION, device, policy, preprocessor=preprocessor, debug=DEBUG_THIS_ITERATION)

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

                # Unnormalize action based on model type (SmolVLA uses hardcoded, Pi0 uses postprocessor)
                # This converts: normalized -> physical -> MuJoCo radians
                action_radians = unnormalize_action_for_vla(action, MODEL_TYPE, postprocessor)
                
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

