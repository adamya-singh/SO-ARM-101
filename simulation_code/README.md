# SO-ARM-101 MuJoCo Simulation

MuJoCo simulation environment for the SO-ARM-101 robot arm with SmolVLA policy inference, teleoperation data collection, and ReinFlow RL training.

## Quick Start

```bash
# 1. Activate environment
conda activate lerobot

# 2. Run basic simulation viewer
python example_run_mujoco_sim.py

# 3. Run SmolVLA (base) policy inference
python run_mujoco_simulation.py

# 4. Record teleoperation demonstrations
python record_dataset.py --input keyboard --output_dir ./datasets/my_dataset --task "pick up the red block"

# 5. Train with ReinFlow RL
python train_reinflow.py
```

---

## Project Structure

```
SO-ARM-101/                         # Repository root
├── smolvla_modifications/          # Modified LeRobot files for ReinFlow
│   └── lerobot-src-lerobot-policies-smolvla-modeling_smolvla.py
│
└── simulation_code/                # Main code directory
    ├── model/                      # MuJoCo model files
    │   ├── scene.xml               # Main scene (robot + block + cameras)
    │   ├── so101_new_calib.xml     # Calibrated SO-101 robot MJCF
    │   ├── assets/                 # STL meshes for robot parts
    │   └── old/                    # Archived model versions
    │
    ├── datasets/                   # Recorded demonstration datasets
    │   ├── so101_pickplace/        # 50 episodes (randomized block)
    │   └── so101_pickplace_fixed/  # 50 episodes (fixed block position)
    │
    ├── envs/                       # Gymnasium environment registration
    │   └── __init__.py
    │
    ├── # === CORE LIBRARIES ===
    ├── so101_gym_env.py            # Gymnasium environment wrapper
    ├── so101_mujoco_utils.py       # Utility functions (reward, reset, etc.)
    ├── lerobot_dataset_writer.py   # LeRobotDataset v3 format writer
    │
    ├── # === SIMULATION SCRIPTS ===
    ├── example_run_mujoco_sim.py   # Basic MuJoCo viewer (no policy)
    ├── run_mujoco_simulation_startingpose.py  # Hold robot at home position
    ├── run_mujoco_simulation.py    # Run SmolVLA policy inference
    │
    ├── # === TELEOPERATION ===
    ├── teleop_keyboard.py          # Keyboard control handler
    ├── teleop_gamepad.py           # DualShock 4 controller handler
    ├── teleop_physical_arm.py      # Physical SO-101 arm mirroring
    ├── record_dataset.py           # Main recording script
    │
    ├── # === RL TRAINING ===
    ├── reinflow_smolvla.py         # ReinFlow wrapper for SmolVLA
    ├── train_reinflow.py           # Full ReinFlow training script
    │
    ├── # === DEPRECATED ===
    ├── old-training-scripts/       # Failed training attempts
    │   └── train_reinflow_lite-DIDNTWORK.py
    └── old-checkpoints/            # Old checkpoints
```

---

## Setup & Installation

### Prerequisites

```bash
# Create conda environment
conda create -n lerobot python=3.10
conda activate lerobot
```

### Install LeRobot (Editable Mode - Required for ReinFlow)

ReinFlow training requires a modified version of SmolVLA. You must install LeRobot from source in editable mode:

```bash
# Clone LeRobot repository (place it alongside this repo)
cd /path/to/your/projects
git clone https://github.com/huggingface/lerobot.git
cd lerobot

# Install in editable mode with SmolVLA support
pip install -e ".[smolvla]"
```

### Apply ReinFlow Modifications to SmolVLA

We provide a modified `modeling_smolvla.py` that adds the `sample_actions_reinflow()` method required for ReinFlow training. Copy it to your LeRobot installation:

```bash
# From this repo's root directory (SO-ARM-101/):
cp smolvla_modifications/lerobot-src-lerobot-policies-smolvla-modeling_smolvla.py \
   /path/to/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

Or from the `simulation_code/` directory:

```bash
cp ../smolvla_modifications/lerobot-src-lerobot-policies-smolvla-modeling_smolvla.py \
   /path/to/lerobot/src/lerobot/policies/smolvla/modeling_smolvla.py
```

> **Note**: The modification adds a single method `sample_actions_reinflow()` to the `VLAFlowMatching` class (lines 749-830). This method injects learnable noise at each denoising step for ReinFlow RL training.

### Additional Dependencies

```bash
pip install mujoco pynput pygame pyarrow imageio
```

### Verify Installation

```bash
cd simulation_code

# Check basic imports
python -c "import mujoco; import lerobot; print('OK')"

# Verify ReinFlow modification is applied
python -c "from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching; print('sample_actions_reinflow' in dir(VLAFlowMatching))"
# Should print: True
```

---

## Running the Simulation

### Basic MuJoCo Viewer

Opens the simulation with no policy control:

```bash
python example_run_mujoco_sim.py
```

### Hold Starting Pose

Holds the robot at the home position:

```bash
python run_mujoco_simulation_startingpose.py
```

### SmolVLA Policy Inference

Runs the SmolVLA model to control the robot:

```bash
python run_mujoco_simulation.py
```

The script loads either `lerobot/smolvla_base` (pretrained) or a fine-tuned checkpoint and executes the policy in a loop.

---

## Teleoperation & Data Collection

### Input Modes

| Mode | Flag | Description |
|------|------|-------------|
| Keyboard | `--input keyboard` | Arrow keys, WASD for control |
| Gamepad | `--input gamepad` | DualShock 4 / Xbox controller |
| Physical Arm | `--input physical` | Mirror a real SO-101 arm |

### Recording Demonstrations

```bash
# Keyboard teleoperation
python record_dataset.py \
    --input keyboard \
    --output_dir ./datasets/my_dataset \
    --task "pick up the red block" \
    --num_episodes 50 \
    --fps 10

# Gamepad teleoperation
python record_dataset.py \
    --input gamepad \
    --output_dir ./datasets/my_dataset \
    --task "pick up the red block" \
    --num_episodes 50

# Physical arm mirroring (requires mjpython on macOS)
mjpython record_dataset.py \
    --input physical \
    --port /dev/tty.usbmodem5A680096011 \
    --output_dir ./datasets/my_dataset \
    --task "pick up the red block" \
    --num_episodes 50
```

### Keyboard Controls

```
Movement:
  ↑/↓     Shoulder pan forward/backward
  ←/→     Shoulder pan left/right
  W/S     Shoulder lift up/down
  I/K     Elbow extend/retract

Wrist:
  Z/X     Wrist flex down/up
  A/D     Wrist roll left/right

Gripper:
  Q       Open gripper
  E       Close gripper

Episode Control:
  Space   Start/stop recording
  R       Reset environment
  ESC     Quit
```

### Gamepad Controls (DualShock 4)

```
Left Stick      Shoulder pan (X) / lift (Y)
Right Stick     Wrist roll (X) / flex (Y)
L1/R1           Elbow retract/extend
L2/R2           Gripper close/open
X (Cross)       Start/stop recording
Circle          Reset environment
Options         Quit
```

### Physical Arm Controls

```
Move arm        Simulation mirrors movements
Space           Start/stop recording
R               Reset simulation
ESC             Quit
```

---

## Training

### ReinFlow RL Training (Recommended)

ReinFlow is a flow-based RL method that injects learnable noise at each denoising step, enabling proper policy gradient training:

```bash
# Start training
python train_reinflow.py

# Resume from checkpoint
python train_reinflow.py --resume reinflow_checkpoint.pt

# Headless (faster, no visualization)
python train_reinflow.py --no-render

# Custom settings
python train_reinflow.py --episodes 50000 --lr 1e-4
```

**Key features:**
- Injects learnable noise at each of 10 denoising steps
- Computes exact log-probabilities for REINFORCE
- Optionally trains action head (~23K params)

### Fine-tuning SmolVLA (Behavioral Cloning)

Use LeRobot's training script for supervised fine-tuning:

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=your_hf_username/your_dataset \
    --batch_size=64 \
    --steps=20000 \
    --output_dir=outputs/train/my_smolvla
```

---

## Gymnasium Environment

### Registration

```python
import gymnasium
import envs  # Registers environments

# Create environment
env = gymnasium.make("SO101PickPlace-v0")

# Variants:
# - SO101PickPlace-v0: Random block position
# - SO101PickPlaceNoRandom-v0: Fixed block position
# - SO101PickPlaceHuman-v0: With viewer, longer episodes
```

### Observation Space

```python
{
    "observation.images.camera1": (256, 256, 3),  # Top-down camera
    "observation.images.camera2": (256, 256, 3),  # Wrist camera
    "observation.images.camera3": (256, 256, 3),  # Side camera
    "observation.state": (6,),                     # Joint positions (radians)
}
```

### Action Space

```python
# 6 joint positions in radians:
# [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]
action_space = Box(low=joint_limits_low, high=joint_limits_high, shape=(6,))
```

### Reward Function

The reward function in `so101_mujoco_utils.py` includes:

| Component | Weight | Description |
|-----------|--------|-------------|
| Distance penalty | -2.0 | Linear penalty for gripper-to-block distance |
| Approach bonus | +5.0 | Reward for moving toward block |
| Height bonus | +20.0 | Reward for lifting block |
| Proximity bonus | +0.5 | Bonus when within 5cm of block |
| Success bonus | +50.0 | Large reward when block lifted above threshold |

---

## Model Files

### Scene Structure (`model/scene.xml`)

```xml
<mujoco model="scene">
    <include file="so101_new_calib.xml" />  <!-- Robot definition -->
    
    <!-- Red block target object -->
    <body name="red_block" pos="0 0.3 0.0125">
        <freejoint/>
        <geom type="box" size="0.0125 0.0125 0.0125" rgba="1 0 0 1"/>
    </body>
</mujoco>
```

### Cameras

| Camera | Name | Purpose |
|--------|------|---------|
| Top-down | `camera_up` | Bird's eye view |
| Wrist | `wrist_camera` | End-effector mounted |
| Side | `camera_side` | Profile view |

### Robot Joints

| Joint | Index | Description |
|-------|-------|-------------|
| shoulder_pan | 0 | Base rotation |
| shoulder_lift | 1 | Shoulder pitch |
| elbow_flex | 2 | Elbow pitch |
| wrist_flex | 3 | Wrist pitch |
| wrist_roll | 4 | Wrist rotation |
| gripper | 5 | Gripper open/close |

---

## Scripts Reference

| Script | Description | Example |
|--------|-------------|---------|
| `example_run_mujoco_sim.py` | Basic MuJoCo viewer | `python example_run_mujoco_sim.py` |
| `run_mujoco_simulation_startingpose.py` | Hold home position | `python run_mujoco_simulation_startingpose.py` |
| `run_mujoco_simulation.py` | SmolVLA policy inference | `python run_mujoco_simulation.py` |
| `record_dataset.py` | Record demonstrations | `python record_dataset.py --input keyboard` |
| `train_reinflow.py` | ReinFlow RL training | `python train_reinflow.py` |
| `teleop_keyboard.py` | Test keyboard input | `python teleop_keyboard.py` |
| `teleop_gamepad.py` | Test gamepad input | `python teleop_gamepad.py` |

---

## Troubleshooting

### MuJoCo Renderer Error on macOS

If you get `CGLError: invalid CoreGraphics connection`:
- Run with `mjpython` instead of `python` for scripts that need rendering
- Or use `--no-render` flag for training scripts

### Controller Not Detected

```bash
# Test gamepad detection
python teleop_gamepad.py

# May need to pair via Bluetooth settings first
```

### SmolVLA Model Download

First run downloads ~2GB model. Ensure network access:
```bash
python -c "from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy; SmolVLAPolicy.from_pretrained('lerobot/smolvla_base')"
```

---

## License

Apache 2.0 (following LeRobot licensing)

