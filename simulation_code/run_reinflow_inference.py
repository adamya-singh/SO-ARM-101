"""
Run inference with a ReinFlow-trained policy checkpoint.

This script strictly evaluates the ReinFlow policy that was trained with
stochastic denoising. It does not fall back to plain base-model inference.
Use `run_mujoco_simulation.py` for raw pretrained SmolVLA/Pi0 inference.

Auto-detects model type from checkpoint metadata, or can be specified manually.

Usage:
    python run_reinflow_inference.py --checkpoint reinflow_checkpoint.pt

    # Specify model type manually:
    python run_reinflow_inference.py --model-type pi0 --checkpoint reinflow_pi0_checkpoint.pt

    # Headless mode (for Colab/SSH):
    MUJOCO_GL=osmesa python run_reinflow_inference.py --checkpoint reinflow_checkpoint.pt
"""

import os
import time
import argparse
import numpy as np

# Setup headless rendering BEFORE importing mujoco
from mujoco_rendering import setup_mujoco_rendering
setup_mujoco_rendering()

import mujoco
import mujoco.viewer
import torch

from so101_mujoco_utils import (
    hold_position,
    set_initial_pose,
    send_position_command,
    get_camera_observation,
    get_robot_state,
    convert_to_dictionary,
    unnormalize_action_for_vla,
    prepare_observation,
)
from reinflow_smolvla import (
    setup_reinflow_policy,
    setup_reinflow_pi0_policy,
    load_reinflow_checkpoint,
    load_reinflow_pi0_checkpoint,
    detect_model_type_from_checkpoint,
    prepare_observation_for_reinflow,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Run ReinFlow inference with a trained checkpoint')
    parser.add_argument('--checkpoint', type=str, default='reinflow_checkpoint.pt',
                        help='Path to ReinFlow checkpoint (required to evaluate trained policy)')
    parser.add_argument('--model-type', type=str, choices=['smolvla', 'pi0', 'auto'], default='auto',
                        help='Model type (auto-detect from checkpoint by default)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained base model (auto-selected based on model type if not specified)')
    parser.add_argument('--no-quantize', action='store_true',
                        help='Deprecated: ignored by the strict ReinFlow wrapper path')
    return parser.parse_args()


def build_reinflow_observation(model_type, rl_policy, preprocessor, rgb_top, rgb_wrist, rgb_side, robot_state,
                               instruction, device, debug=False):
    """Build the observation dict expected by the ReinFlow wrapper."""
    if model_type == "smolvla":
        return prepare_observation_for_reinflow(
            rgb_top,
            rgb_wrist,
            rgb_side,
            robot_state,
            instruction,
            device,
            rl_policy,
        )

    # Pi0 uses the generic processor-based path, but tokenization should happen
    # against the base policy owned by the ReinFlow wrapper.
    return prepare_observation(
        rgb_top,
        rgb_wrist,
        rgb_side,
        robot_state,
        instruction,
        device,
        policy=rl_policy.base,
        preprocessor=preprocessor,
        model_type="pi0",
        debug=debug,
    )


args = parse_args()

CHECKPOINT_PATH = args.checkpoint
INSTRUCTION = "pick up the block"

if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(
        f"ReinFlow checkpoint not found at {CHECKPOINT_PATH}. "
        f"This script only evaluates trained ReinFlow policies."
    )

# Auto-detect model type from checkpoint
if args.model_type == 'auto':
    MODEL_TYPE = detect_model_type_from_checkpoint(CHECKPOINT_PATH)
    print(f"Auto-detected model type: {MODEL_TYPE}")
else:
    MODEL_TYPE = args.model_type

# Set pretrained path based on model type
if args.pretrained:
    PRETRAINED_PATH = args.pretrained
elif MODEL_TYPE == "pi0":
    PRETRAINED_PATH = "lerobot/pi0"
else:
    PRETRAINED_PATH = "lerobot/smolvla_base"

m = mujoco.MjModel.from_xml_path('model/scene.xml')
d = mujoco.MjData(m)

# ===== ReinFlow Policy Setup =====
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using device: {device} (Apple Silicon GPU)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using device: {device}")
else:
    device = torch.device("cpu")
    print(f"Using device: {device}")

if args.no_quantize:
    print("WARNING: --no-quantize is ignored by the strict ReinFlow inference path.")

renderer = mujoco.Renderer(m, height=256, width=256)

print(f"\n=== Loading {MODEL_TYPE.upper()} ReinFlow Policy ===")
if MODEL_TYPE == "pi0":
    rl_policy, preprocessor, postprocessor = setup_reinflow_pi0_policy(
        pretrained_path=PRETRAINED_PATH,
        device=str(device),
        train_action_head=True,
        train_time_mlp=True,
        train_full_expert=True,
        train_noise_head=True,
        train_critic=True,
    )
    start_episode, _ = load_reinflow_pi0_checkpoint(rl_policy, CHECKPOINT_PATH, str(device))
else:
    rl_policy = setup_reinflow_policy(
        pretrained_path=PRETRAINED_PATH,
        device=str(device),
        train_action_head=True,
        train_time_mlp=True,
        train_full_expert=True,
        train_noise_head=True,
        train_critic=True,
    )
    preprocessor = None
    postprocessor = None
    start_episode, _ = load_reinflow_checkpoint(rl_policy, CHECKPOINT_PATH, str(device))

rl_policy.eval()
checkpoint_episode = max(0, start_episode - 1)

print(f"Loaded ReinFlow checkpoint from saved episode {checkpoint_episode}")
if MODEL_TYPE == "pi0":
    sigma_stats = rl_policy.get_sigma_stats()
    print(f"Restored ReinFlow sigma bounds: [{sigma_stats['sigma_min']}, {sigma_stats['sigma_max']}]")
else:
    sigma_stats = rl_policy.get_sigma_stats()
    print(f"Restored ReinFlow sigma bounds: [{sigma_stats['sigma_min']}, {sigma_stats['sigma_max']}]")
print("ReinFlow policy ready for evaluation.")

# ===== End ReinFlow Setup =====

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

    print(f"\nStarting ReinFlow evaluation with instruction: '{INSTRUCTION}'")
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

                robot_state = get_robot_state(d)
                observation = build_reinflow_observation(
                    MODEL_TYPE,
                    rl_policy,
                    preprocessor,
                    rgb_image_top,
                    rgb_image_wrist,
                    rgb_image_side,
                    robot_state,
                    INSTRUCTION,
                    device,
                    debug=DEBUG_THIS_ITERATION,
                )

                if DEBUG_THIS_ITERATION:
                    print(f"\n[Observation Structure Debug]")
                    print(f"  Observation keys: {list(observation.keys())}")

                with torch.no_grad():
                    if DEBUG_THIS_ITERATION:
                        print(f"\n[Policy Call Debug]")
                        print(f"  Using ReinFlow wrapper select_action()")
                    action = rl_policy.select_action(observation)

                if torch.is_tensor(action):
                    action = action.detach().cpu().numpy().squeeze()

                if DEBUG_THIS_ITERATION:
                    print(f"\n[Action Output Debug]")
                    print(f"  Raw normalized action: {action}")
                    print(f"  Action shape: {action.shape}")
                    print(f"  Normalized range: min={action.min():.4f}, max={action.max():.4f}")

                action_radians = unnormalize_action_for_vla(action, MODEL_TYPE, postprocessor)

                JOINT_LIMITS_LOW = np.array([-1.92, -1.745, -1.69, -1.658, -2.744, -0.175])
                JOINT_LIMITS_HIGH = np.array([1.92, 1.745, 1.69, 1.658, 2.841, 1.745])
                action_radians = np.clip(action_radians, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)

                last_action_dict = convert_to_dictionary(action_radians)

                if DEBUG_THIS_ITERATION:
                    print(f"  Unnormalized (radians): {action_radians}")
                    print(f"  Converted to degrees: {last_action_dict}")
                    print(f"\n{'='*60}\n")

                if policy_inference_count == 3:
                    print(f"\n{'='*60}")
                    print(f"Debug output complete. ReinFlow policy continuing to run silently...")
                    print(f"{'='*60}\n")

            except Exception as e:
                print(f"Error in ReinFlow control loop: {e}")
                raise

        send_position_command(d, last_action_dict)
        policy_step_counter += 1

        mujoco.mj_step(m, d)
        viewer.sync()

        time_until_next_step = m.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
