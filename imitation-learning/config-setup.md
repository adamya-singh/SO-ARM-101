# config.json Setup Guide

This guide walks you through configuring `config.json` before running `record_single_arm.py`.

## Prerequisites

Before starting, ensure you have:
- LeRobot installed with feetech extras: `pip install lerobot[feetech]`
- SO-101 arm calibration completed (see LeRobot calibration docs)
- USB camera connected and working

---

## Required Configuration Steps

### 1. Find Your Robot Port

The `robot.port` field specifies the USB serial port for your SO-101 arm.

**macOS:**
```bash
ls /dev/tty.usb*
```
Example output: `/dev/tty.usbmodem58760431541`

**Linux:**
```bash
ls /dev/ttyUSB*
# or
ls /dev/ttyACM*
```
Example output: `/dev/ttyUSB0` or `/dev/ttyACM0`

Update your config:
```json
"robot": {
  "port": "/dev/tty.usbmodem58760431541"
}
```

### 2. Set Your Robot ID

The `robot.id` must match your calibration file name.

Find your calibration file:
```bash
ls ~/.cache/huggingface/lerobot/calibration/robots/so101_follower/
```

The ID is the filename without `.json`. For example, if you see `my_so101_follower.json`, use:
```json
"robot": {
  "id": "my_so101_follower",
  "port": "/dev/tty.usbmodem58760431541"
}
```

### 3. Configure Your Camera

The `camera.device` is the camera index OpenCV uses to access your camera.

**Find your camera index:**
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0:', cap.isOpened()); cap.release()"
python -c "import cv2; cap = cv2.VideoCapture(1); print('Camera 1:', cap.isOpened()); cap.release()"
```

- **0** = Usually the built-in webcam (laptops)
- **1+** = External USB cameras

Test your camera opens correctly, then set:
```json
"camera": {
  "device": 0
}
```

---

## User Choice Fields

### 4. Dataset Name

Choose a descriptive name for your dataset folder:
```json
"dataset_name": "so101_pickplace_v1"
```

### 5. Task Description

Describe the task in natural language. This is stored as metadata:
```json
"task_description": "pick up the cube and place it in the box"
```

### 6. Output Directory

Where datasets are saved (default is `./datasets`):
```json
"output_dir": "./datasets"
```

### 7. Recording FPS

Frame rate for recording (default: 30):
```json
"recording": {
  "fps": 30
}
```

---

## Optional: HuggingFace Hub Upload

To upload your dataset to HuggingFace Hub after recording:

### 8. Enable Hub Upload
```json
"hub": {
  "push_to_hub": true,
  "repo_id": "your-username/dataset-name"
}
```

Set `push_to_hub` to `false` to keep data local only.

---

## Complete Example

```json
{
  "dataset_name": "so101_pickplace_v1",
  "task_description": "pick up the cube and place it in the box",
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

---

## Troubleshooting

**Robot not found:**
- Check USB cable connection
- Try a different USB port
- Re-run the port detection command

**Camera not opening:**
- Ensure no other app is using the camera
- Try different device indices (0, 1, 2...)
- Check camera permissions in System Settings (macOS)

**Calibration file not found:**
- Run LeRobot calibration: `python -m lerobot.calibrate --robot.type=so101_follower`
- Verify the file exists in `~/.cache/huggingface/lerobot/calibration/robots/so101_follower/`
