# SO-ARM-101 MuJoCo Simulation

MuJoCo simulation environment for the SO-ARM-101 robot arm with **SmolVLA (450M)** and **Pi0 (3.3B)** policy inference, teleoperation data collection, and ReinFlow RL training.

## Quick Start

```bash
# 1. Activate environment
conda activate lerobot

# 2. Run basic simulation viewer
python example_run_mujoco_sim.py

# 3. Run VLA policy inference (SmolVLA default)
python run_mujoco_simulation.py

# 3b. Run with Pi0 model (requires [pi] extras)
python run_mujoco_simulation.py --model-type pi0

# 4. Record teleoperation demonstrations
python record_dataset.py --input keyboard --output_dir ./datasets/my_dataset --task "pick up the red block"

# 5a. Train with ReinFlow RL (SmolVLA - default)
python train_reinflow.py

# 5b. Train with ReinFlow RL (Pi0 - 3.3B model, requires 24GB+ VRAM)
python train_reinflow.py --model-type pi0

# 6. Run inference with trained checkpoint (auto-detects model type)
python run_reinflow_inference.py --checkpoint reinflow_checkpoint.pt

# 7. View training statistics from checkpoint
python print_checkpoint_stats.py
```

---

## Model Selection

This codebase supports two VLA models for policy inference and ReinFlow training:

| Model | Parameters | VRAM Required | Best For |
|-------|------------|---------------|----------|
| SmolVLA | 450M | ~4GB | Fast training, limited GPU, quick iteration |
| Pi0 | 3.3B | ~24GB (CUDA)<br>~3-4GB (MPS with 4-bit) | Higher quality, A100/RTX 4090, production<br>Apple Silicon inference with quantization |

### Choosing a Model

- **SmolVLA (default)**: Recommended for most users. Fast training, works on consumer GPUs.
- **Pi0**: Use when you have access to high-end GPUs (A100, RTX 4090) and want better performance.

> **Important**: SmolVLA and Pi0 require different versions of the `transformers` library and cannot be installed in the same environment. See Installation section below.

---

## Project Structure

```
SO-ARM-101/                         # Repository root
├── smolvla_modifications/          # Reference copy of modified LeRobot files (already in fork)
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
    ├── so101_mujoco_utils.py       # Utility functions (reward, reset, normalization)
    ├── lerobot_dataset_writer.py   # LeRobotDataset v3 format writer
    │
    ├── # === VLA POLICY INTERFACE ===
    ├── vla_policy_interface.py     # Abstract interface for VLA models (SmolVLA/Pi0)
    ├── smolvla_adapter.py          # SmolVLA adapter for ReinFlow
    ├── pi0_adapter.py              # Pi0 adapter for ReinFlow (adds noise_mlp)
    ├── pi0_quantization.py         # 4-bit quantization utilities for Pi0 on MPS
    │
    ├── # === SIMULATION SCRIPTS ===
    ├── example_run_mujoco_sim.py   # Basic MuJoCo viewer (no policy)
    ├── run_mujoco_simulation_startingpose.py  # Hold robot at home position
    ├── run_mujoco_simulation.py    # Run VLA policy inference (--model-type flag)
    │
    ├── # === TELEOPERATION ===
    ├── teleop_keyboard.py          # Keyboard control handler
    ├── teleop_gamepad.py           # DualShock 4 controller handler
    ├── teleop_physical_arm.py      # Physical SO-101 arm mirroring
    ├── record_dataset.py           # Main recording script
    │
    ├── # === RL TRAINING ===
    ├── reinflow_smolvla.py         # ReinFlow wrapper for SmolVLA and Pi0
    ├── train_reinflow.py           # Full ReinFlow training script (--model-type flag)
    ├── run_reinflow_inference.py   # Run inference with trained checkpoint
    ├── print_checkpoint_stats.py   # View training stats from checkpoint
    ├── reinflow_checkpoint.pt      # Trained SmolVLA checkpoint file
    ├── reinflow_pi0_checkpoint.pt  # Trained Pi0 checkpoint file (if trained)
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

### Install LeRobot Fork (Editable Mode - Required for ReinFlow)

ReinFlow training requires a modified version of LeRobot. Install our fork which includes the necessary `sample_actions_reinflow()` method:

```bash
# Clone our LeRobot fork (place it alongside this repo)
cd /path/to/your/projects
git clone https://github.com/adamyasingh/lerobot-fork.git
cd lerobot-fork
```

#### For SmolVLA Only (Default - Recommended)

```bash
# Install with SmolVLA support
pip install -e ".[smolvla]" --no-cache-dir
```

#### For Pi0 Support

Pi0 requires a custom transformers fork with LeRobot-specific patches:

```bash
# Install with Pi0 support (uses custom transformers branch)
pip install -e ".[pi]" --no-cache-dir
```

> **Warning**: The `[pi]` and `[smolvla]` extras use different versions of `transformers` and **cannot be installed together**. Use separate conda environments if you need both:
> ```bash
> conda create -n lerobot-smolvla python=3.10  # For SmolVLA
> conda create -n lerobot-pi0 python=3.10      # For Pi0
> ```

### Additional Dependencies

```bash
pip install mujoco pynput pygame pyarrow imageio
```

**For Pi0 on Apple Silicon (MPS):**
```bash
# Optional: Enables 4-bit quantization to reduce memory usage (~3-4GB vs ~6.6GB)
pip install optimum-quanto
```

> **Note**: When running Pi0 on MPS devices, 4-bit quantization is automatically enabled if `optimum-quanto` is installed. Without it, the model falls back to float16 precision.

### Verify Installation

```bash
cd simulation_code

# Check basic imports
python -c "import mujoco; import lerobot; print('OK')"

# Verify SmolVLA ReinFlow modification is applied
python -c "from lerobot.policies.smolvla.modeling_smolvla import VLAFlowMatching; print('sample_actions_reinflow' in dir(VLAFlowMatching))"
# Should print: True

# Verify Pi0 is available (only if [pi] extras installed)
python -c "from lerobot.policies.pi0.modeling_pi0 import PI0Policy; print('PI0Policy available')"
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

### VLA Policy Inference

Runs a VLA model to control the robot:

```bash
# SmolVLA (default)
python run_mujoco_simulation.py

# Pi0 (requires [pi] extras)
python run_mujoco_simulation.py --model-type pi0

# Pi0 on Apple Silicon (MPS) - automatically uses 4-bit quantization
# Reduces memory from ~6.6GB (float16) to ~3-4GB (INT4 weights)
python run_mujoco_simulation.py --model-type pi0

# Disable quantization if needed (uses more memory)
python run_mujoco_simulation.py --model-type pi0 --no-quantize
```

The script loads either `lerobot/smolvla_base` or `lerobot/pi0` (pretrained) and executes the policy in a loop. On MPS devices, Pi0 automatically applies 4-bit weight quantization when `optimum-quanto` is installed.

### ReinFlow Trained Policy Inference

Runs inference using the trained ReinFlow checkpoint:

```bash
# Auto-detect model type from checkpoint
python run_reinflow_inference.py

# Specify checkpoint explicitly
python run_reinflow_inference.py --checkpoint reinflow_checkpoint.pt

# Force model type (overrides auto-detection)
python run_reinflow_inference.py --model-type pi0 --checkpoint reinflow_pi0_checkpoint.pt
```

This script loads the base VLA policy and applies the trained weights from the checkpoint. Debug output is shown for the first 3 policy inferences.

### View Training Statistics

Print training statistics from a saved checkpoint:

```bash
python print_checkpoint_stats.py
```

Outputs episode rewards, sigma values, and training progress summary.

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

### ReinFlow RL Training

ReinFlow is a flow-based RL method that injects learnable noise at each denoising step, enabling proper policy gradient training.

#### SmolVLA Training (Default - Recommended)

```bash
# Start training (with viewer)
python train_reinflow.py

# Resume from checkpoint
python train_reinflow.py --resume reinflow_checkpoint.pt

# Headless mode (for Colab/SSH, no visualization)
python train_reinflow.py --no-render --headless

# Parallel mode with 8 environments (optimized for A100 GPU)
python train_reinflow.py --parallel-envs 8 --no-render --headless

# Custom settings
python train_reinflow.py --episodes 50000 --policy-lr 1e-5
```

#### Pi0 Training (Requires 24GB+ VRAM)

```bash
# Pi0 training with automatic memory optimizations
python train_reinflow.py --model-type pi0 --no-render --headless

# Pi0 parallel mode (reduced envs due to model size)
python train_reinflow.py --model-type pi0 --parallel-envs 2 --no-render --headless

# Resume Pi0 checkpoint
python train_reinflow.py --model-type pi0 --resume reinflow_pi0_checkpoint.pt
```

#### Training Settings Comparison

| Setting | SmolVLA | Pi0 |
|---------|---------|-----|
| Batch size | 8 | 2 (auto-adjusted) |
| Gradient accumulation | 15 | 30 (auto-adjusted) |
| Policy learning rate | 5e-6 | 2.5e-6 (auto-adjusted) |
| Gradient checkpointing | No | Yes (auto-enabled) |
| Parallel environments | 8-16 | 2-4 |
| VRAM required | ~4GB | ~24GB |
| Checkpoint file | `reinflow_checkpoint.pt` | `reinflow_pi0_checkpoint.pt` |

**Key features:**
- Injects learnable noise at each denoising step
- Computes exact log-probabilities for PPO
- Supports both SmolVLA and Pi0 through adapter interface
- Pi0 automatically enables gradient checkpointing and memory optimizations
- Auto-detects model type when resuming from checkpoint

### Fine-tuning VLA Models (Behavioral Cloning)

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

# For VLA inference, enable normalization:
from so101_gym_env import SO101PickPlaceEnv
env = SO101PickPlaceEnv(smolvla_normalize=True)
# This normalizes state outputs and unnormalizes action inputs
# to match VLA expected format (degrees with mean/std normalization)
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
| Distance penalty | -2.0 | Linear penalty for gripper-to-initial-block distance |
| Approach bonus | +5.0 | Reward for moving toward block |
| Height bonus | +20.0 | Reward for lifting block |
| Proximity bonus | +0.5 | Bonus when within 5cm of block |
| Success bonus | +50.0 | Large reward when block lifted above threshold |
| Displacement penalty | -5.0 × exp | Exponential penalty for knocking block >5cm away |

> **Note**: Distance is measured to the *initial* block position, not current. This prevents the robot from learning to avoid the block (to not knock it away).

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
| `run_mujoco_simulation.py` | VLA policy inference | `python run_mujoco_simulation.py --model-type smolvla` |
| `run_reinflow_inference.py` | Inference with trained checkpoint | `python run_reinflow_inference.py --checkpoint reinflow_checkpoint.pt` |
| `record_dataset.py` | Record demonstrations | `python record_dataset.py --input keyboard` |
| `train_reinflow.py` | ReinFlow RL training | `python train_reinflow.py --model-type pi0` |
| `print_checkpoint_stats.py` | View training stats | `python print_checkpoint_stats.py` |
| `teleop_keyboard.py` | Test keyboard input | `python teleop_keyboard.py` |
| `teleop_gamepad.py` | Test gamepad input | `python teleop_gamepad.py` |

---

## Running in Google Colab / Headless Environments

For headless environments (Colab, SSH sessions, cloud VMs), MuJoCo needs a software rendering backend since there's no display.

### Setup in Colab

```bash
# Install headless rendering dependencies
!apt-get update && apt-get install -y libgl1-mesa-glx libosmesa6-dev libglfw3

# Clone the repo
!git clone https://github.com/your-username/SO-ARM-101.git
%cd SO-ARM-101/simulation_code

# Install Python dependencies
!pip install mujoco torch transformers
```

### Running Training Headless

```bash
# SmolVLA - Sequential mode (best for most cases)
python train_reinflow.py --no-render --headless

# SmolVLA - Parallel mode with 10 environments (for A100 GPU)
python train_reinflow.py --parallel-envs 10 --no-render --headless

# Pi0 - Parallel mode (reduced envs for memory)
python train_reinflow.py --model-type pi0 --parallel-envs 2 --no-render --headless

# Or set environment variable directly
MUJOCO_GL=osmesa python train_reinflow.py --no-render
```

### How It Works

The scripts automatically detect the best rendering backend:

1. **Native OpenGL** - Used when a display is available (default on desktop)
2. **EGL** - Headless GPU rendering (preferred on Colab with GPU runtime)
3. **OSMesa** - Software rendering (fallback, works anywhere)

The `mujoco_rendering.py` module handles this detection automatically. You can also force a specific backend:

```python
import os
os.environ['MUJOCO_GL'] = 'egl'  # or 'osmesa'
# Must be set BEFORE importing mujoco
```

### Verifying Headless Rendering

```python
import os
os.environ['MUJOCO_GL'] = 'osmesa'
import mujoco

model = mujoco.MjModel.from_xml_string('<mujoco><worldbody/></mujoco>')
data = mujoco.MjData(model)
renderer = mujoco.Renderer(model, 256, 256)
renderer.update_scene(data)
img = renderer.render()
print(f"Rendered image shape: {img.shape}")  # Should print (256, 256, 3)
```

---

## Troubleshooting

### MuJoCo Renderer Error on macOS

If you get `CGLError: invalid CoreGraphics connection`:
- Run with `mjpython` instead of `python` for scripts that need rendering
- Or use `--no-render` flag for training scripts

### Headless Rendering Errors

If you get rendering errors in headless mode:

```bash
# Install OSMesa
sudo apt-get install libosmesa6-dev

# Verify it works
MUJOCO_GL=osmesa python -c "import mujoco; print('OK')"
```

For Colab with GPU, EGL should work automatically. If not, fall back to OSMesa.

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

### Pi0: "Incorrect transformer version" Error

This error means you need the Pi0-specific transformers fork:

```bash
# Install Pi0 extras (includes custom transformers)
cd /path/to/lerobot-fork
pip install -e ".[pi]"
```

> **Note**: You cannot have both `[smolvla]` and `[pi]` extras installed in the same environment. Use separate conda environments.

### Pi0: Out of Memory (OOM) Errors

Pi0 is a 3.3B parameter model and requires significant VRAM:

1. **Reduce parallel environments**: Use `--parallel-envs 2` instead of 8
2. **Verify gradient checkpointing**: It's auto-enabled for Pi0, but you can check logs
3. **Use CUDA**: Pi0 is not recommended on MPS (Apple Silicon) or CPU
4. **Minimum VRAM**: 24GB (A100, RTX 4090)

```bash
# Minimal memory usage for Pi0
python train_reinflow.py --model-type pi0 --parallel-envs 1 --no-render --headless
```

### Pi0: MPS Warning on Apple Silicon

**Inference**: Pi0 now supports 4-bit weight quantization on Apple Silicon (MPS) devices, reducing memory usage from ~6.6GB (float16) to ~3-4GB. Quantization is automatically enabled when running on MPS. Install `optimum-quanto` for best results:

```bash
pip install optimum-quanto
```

**Training**: Pi0 training on MPS is not recommended. The model is optimized for CUDA and training will be slow or have compatibility issues. For best results, use an NVIDIA GPU with 24GB+ VRAM.

---

## License

Apache 2.0 (following LeRobot licensing)
