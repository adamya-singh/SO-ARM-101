# Single-Arm Dataset Recording Script - Technical Specification

## Overview

A Python script for recording imitation learning datasets using a single SO-101 follower arm (moved by hand) and a USB wrist camera. Outputs datasets in **LeRobot v3.0 native format**, fully compatible with SmolVLA fine-tuning.

### Key Difference from Official Recording

| Aspect | Official LeRobot | This Script |
|--------|------------------|-------------|
| Arms | Leader + Follower | Single Follower only |
| Control | Leader arm drives follower | Human moves follower by hand |
| Action source | Leader joint positions | Follower joint positions (action = state) |
| Camera source | Attached to follower | USB camera on follower wrist |

---

## Requirements Summary

### Hardware
- **Robot**: SO-101 follower arm with 6x STS3215 servos (IDs 1-6)
- **Camera**: USB wrist camera, native 1920x1080 resolution
- **Connection**: USB serial for arm, USB for camera

### Software Dependencies
- LeRobot library (for dataset API, motor bus, camera abstraction)
- Match LeRobot's Python version and dependency requirements
- Rerun for live preview visualization

---

## Detailed Specifications

### 1. Dataset Format

**Format**: LeRobot v3.0 native (multiple episodes per file)

**Implementation**: Use LeRobot's built-in `LeRobotDataset.create()` API
- Guarantees format compatibility
- Auto-handles chunking, metadata, video encoding
- Produces identical output to official `lerobot-record`

**Directory Structure** (produced by LeRobot API):
```
datasets/{dataset_name}/
├── meta/
│   ├── info.json           # Schema, fps, features
│   ├── stats.json          # Normalization statistics
│   ├── tasks.jsonl         # Task descriptions
│   └── episodes/
│       └── chunk-000/
│           └── episodes.parquet
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── videos/
    └── observation.images.wrist/
        └── chunk-000/
            └── file-000.mp4
```

### 2. Data Schema

**Observations**:
- `observation.state`: `float32[6]` - Joint positions in radians
  - `[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper]`
- `observation.images.wrist`: `uint8[256, 256, 3]` - Resized camera frame

**Actions**:
- `action`: `float32[6]` - Same as `observation.state`
  - Rationale: With single arm, the "action" is the position the arm is currently in

**Metadata per frame**:
- `frame_index`: int64
- `episode_index`: int64
- `timestamp`: float64 (seconds since episode start)
- `task_index`: int64

### 3. Camera Configuration

**Camera Count**: 1 (wrist camera only, expandable later)

**Camera Naming**: SmolVLA style
- `observation.images.wrist` (primary)
- Future: `observation.images.top`, `observation.images.side`

**Image Processing**:
- Native capture: 1920x1080
- Resize to: 256x256 (default, configurable)
- Resize happens before saving to dataset

**Camera Library**: LeRobot camera API (`lerobot.cameras.opencv.OpenCVCamera`)

**Device Selection**: Specified in config file (e.g., device index 0 or path)

### 4. Arm Configuration

**Calibration**: Required
- Script checks for LeRobot calibration file on startup
- If missing, exits with instructions to run `lerobot-calibrate`
- Motor config read from calibration file

**Robot ID**: Specified in config file
- Used for calibration file lookup
- Example: `"my_so101_follower"`

**Torque**: Automatically disabled on startup
- User moves arm freely by hand
- User is responsible for holding arm against gravity

### 5. Recording Workflow

**Episode Control** (keyboard):
- `SPACE`: Start/stop recording
- `R`: Discard current episode (if recording)
- `ESC`: Discard current episode, finalize dataset, quit

**Frame Synchronization**: Fixed rate sampling
- Sample camera + arm at configured FPS intervals
- Small timing jitter acceptable

**Session Flow**:
1. Script starts, loads config
2. Connects to arm (disables torque) and camera
3. Opens Rerun preview window
4. User presses SPACE to start episode
5. Records frames at configured FPS
6. User presses SPACE to end episode (auto-saves)
7. Repeat 4-6 for more episodes
8. User presses ESC to finalize and quit

**Progress Display**: Terminal status line
- Single updating line: `Recording episode 3 | 45 frames | 4.5s`

### 6. Live Preview

**Framework**: Rerun

**Content**: Camera feed only
- Live video stream from wrist camera
- Minimal CPU overhead

### 7. Configuration

**Format**: JSON

**Location**: Passed via CLI argument

**Example `config.json`**:
```json
{
  "dataset_name": "so101_pickplace_v1",
  "task_description": "pick up the red cube and place it in the box",
  "output_dir": "./datasets",

  "robot": {
    "id": "my_so101_follower",
    "port": "/dev/tty.usbmodem58760431541"
  },

  "camera": {
    "name": "wrist",
    "device": 0,
    "capture_width": 1920,
    "capture_height": 1080,
    "target_width": 256,
    "target_height": 256
  },

  "recording": {
    "fps": 30
  },

  "hub": {
    "push_to_hub": false,
    "repo_id": null
  }
}
```

### 8. Resume Support

**Behavior**: Auto-detect and append
- If output directory contains valid existing dataset, append new episodes
- Episode indices continue from last existing episode
- If directory doesn't exist or is empty, create new dataset

### 9. Error Handling

**Camera Disconnect**: Discard current episode, continue session
- Print warning, allow recording more episodes

**Arm Disconnect**: Discard current episode, continue session
- Print warning, attempt reconnection

**Startup Validation**: None (fail fast)
- Just try to use devices
- Errors surface immediately on first use

### 10. Statistics Computation

**When**: On finalize (when user quits)

**What**:
- Mean, std, min, max for `observation.state`
- Mean, std, min, max for `action`
- Standard image normalization for camera features

**Output**: Saved to `meta/stats.json`

### 11. Script Invocation

**Command**:
```bash
python record_single_arm.py --config config.json
```

**Arguments**:
- `--config`: Path to JSON config file (required)

**Output Location**: `{output_dir}/{dataset_name}/`
- Default output_dir: `./datasets`

### 12. Hub Upload

**Behavior**: Optional, configured in config file
- If `hub.push_to_hub` is true and `hub.repo_id` is set, upload after finalize
- Otherwise, save locally only

---

## File Structure

```
imitation-learning/
├── record_single_arm.py      # Main recording script
├── config.json               # Example configuration
└── single-arm-record-script-SPEC.md  # This specification
```

---

## Implementation Notes

### LeRobot API Usage

```python
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.cameras.opencv import OpenCVCamera, OpenCVCameraConfig
from lerobot.motors.feetech import FeetechMotorsBus

# Create dataset
dataset = LeRobotDataset.create(
    repo_id=config["dataset_name"],
    fps=config["recording"]["fps"],
    features=dataset_features,
    robot_type="so101",
)

# Recording loop
dataset.add_frame({
    "observation.state": joint_positions,
    "observation.images.wrist": camera_frame,
    "action": joint_positions,  # Same as state for single-arm
})

# Finalize
dataset.save_episode()
dataset.finalize()
```

### Motor Bus Configuration

```python
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus

motors = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

bus = FeetechMotorsBus(port=config["robot"]["port"], motors=motors)
bus.connect()
bus.disable_torque()  # Allow free movement
```

---

## Interview Q&A Summary

| Category | Decision |
|----------|----------|
| Action data | action = observation.state (same values) |
| Dataset writer | LeRobot API (not custom) |
| Calibration | Required before running |
| Camera count | 1 (wrist only) |
| Camera naming | SmolVLA style (observation.images.wrist) |
| Config format | JSON |
| Frame sync | Fixed rate sampling |
| Preview | Rerun, camera feed only |
| Keyboard controls | SPACE, R, ESC |
| Arm drift | No special handling |
| Script invocation | Direct Python script |
| Output location | ./datasets/ subdirectory |
| Image resize | 256x256 default |
| Video settings | LeRobot defaults |
| Progress display | Terminal status line |
| Resume | Auto-detect and append |
| Stats | Compute on finalize |
| Quit behavior | Discard current + finalize |
| Motor config | Read from calibration |
| Camera device | Config file path |
| Dry run | Not supported |
| Camera library | LeRobot camera API |
| Dependencies | Match LeRobot |
| Startup validation | None (fail fast) |
| Robot ID | Config file |
| Camera required | Yes |
