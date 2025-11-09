import time
import mujoco
import mujoco.viewer
import torch
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
#policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
policy = SmolVLAPolicy.from_pretrained("adamyathegreat/my_smolvla_pickplace")
policy.to(device)
policy.eval()
print("SmolVLA policy loaded successfully!")

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
last_action_dict = starting_position.copy()  # Initialize with starting pose

with mujoco.viewer.launch_passive(m, d) as viewer:
    #close the viewer automatically after 30 wall-seconds
    start = time.time()
    hold_position(m, d, viewer, 2) #hold starting position for 2 seconds

    print(f"\nStarting SmolVLA control with instruction: '{INSTRUCTION}'")
    print(f"Running policy inference every {STEPS_PER_POLICY_UPDATE} physics steps")
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
                # Get camera observation
                rgb_image = get_camera_observation(renderer, d, camera_name="camera_up")
                
                # Get robot state
                robot_state = get_robot_state(d)
                
                # Prepare observation for policy (includes tokenized instruction)
                observation = prepare_observation(rgb_image, robot_state, INSTRUCTION, device, policy)
                
                # Get action from SmolVLA policy
                with torch.no_grad():
                    # Try calling the policy - the exact method may vary
                    # Common patterns: policy.select_action(), policy(), or policy.generate()
                    try:
                        action = policy.select_action(observation)
                    except AttributeError:
                        # If select_action doesn't exist, try calling the policy directly
                        action = policy(observation)
                
                # Convert action to numpy if it's a tensor
                if torch.is_tensor(action):
                    action = action.cpu().numpy().squeeze()
                
                # Map action to robot control using utility functions
                # Assuming action is 6D: [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
                # SmolVLA actions are likely in radians (MuJoCo format)
                
                # Convert action from radians (MuJoCo) to degrees (SO101 format) using utility function
                last_action_dict = convert_to_dictionary(action)
                
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
