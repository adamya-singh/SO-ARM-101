#!/usr/bin/env python3
"""
Single-Arm Dataset Recording Script for SO-101

Records imitation learning datasets using a single SO-101 follower arm
(moved by hand) and a USB wrist camera. Outputs datasets in LeRobot v3.0
format compatible with SmolVLA fine-tuning.

Key Difference from Official Recording:
    - Single arm only (no leader/follower setup)
    - Human moves follower arm by hand
    - action = observation.state (same values)

Usage:
    python record_single_arm.py --config config.json

Controls:
    SPACE: Start/stop recording an episode
    R: Discard current episode (if recording)
    ESC: Discard current episode, finalize dataset, quit

Requirements:
    - Calibrated SO-101 arm (run: lerobot-calibrate --robot.type=so101_follower)
    - USB camera mounted on wrist
    - LeRobot with feetech extras: pip install lerobot[feetech]
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import cv2
import numpy as np
import rerun as rr
from pynput import keyboard

# LeRobot imports
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import delete_episodes
from lerobot.datasets.utils import (
    DEFAULT_FEATURES,
    build_dataset_frame,
    combine_feature_dicts,
    hw_to_dataset_features,
)
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.utils.constants import ACTION, OBS_STR


def check_calibration(robot_id: str, robot_type: str) -> Path:
    """
    Check if calibration file exists for the robot.

    Args:
        robot_id: Robot identifier used during calibration
        robot_type: Robot type folder under calibration/robots (e.g., so100_follower)

    Returns:
        Path to calibration file

    Raises:
        SystemExit: If calibration file not found
    """
    cal_dir = (
        Path.home() / ".cache" / "huggingface" / "lerobot"
        / "calibration" / "robots" / robot_type
    )

    # Try robot_id.json first, then fallback patterns
    candidates = [
        cal_dir / f"{robot_id}.json",
        cal_dir / "so101_follower_1.json",
        cal_dir / "None.json",
    ]

    for cal_path in candidates:
        if cal_path.exists():
            print(f"Found calibration file: {cal_path}")
            return cal_path

    print("\n" + "=" * 60)
    print("ERROR: Calibration file not found!")
    print("=" * 60)
    print(f"\nLooked in: {cal_dir}")
    print(f"Expected file: {robot_id}.json")
    print("\nPlease calibrate your arm first:")
    print(f"  lerobot-calibrate --robot.type={robot_type} --robot.id={robot_id}")
    print("=" * 60 + "\n")
    sys.exit(1)


def load_calibration(cal_path: Path) -> Dict[str, MotorCalibration]:
    """Load calibration data from JSON file."""
    with open(cal_path, 'r') as f:
        cal_data = json.load(f)

    calibration = {}
    for motor_name, motor_cal in cal_data.items():
        calibration[motor_name] = MotorCalibration(
            id=motor_cal['id'],
            drive_mode=motor_cal['drive_mode'],
            homing_offset=motor_cal['homing_offset'],
            range_min=motor_cal['range_min'],
            range_max=motor_cal['range_max'],
        )

    return calibration


class SingleArmRecorder:
    """
    Records imitation learning datasets from a single SO-101 arm.

    Connects to the arm and camera, records episodes with keyboard control,
    and saves datasets in LeRobot v3.0 format.
    """

    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the recorder with configuration.

        Args:
            config: Configuration dictionary from JSON file
        """
        self.config = config

        # Extract config values
        self.dataset_name = config["dataset_name"]
        self.task_description = config["task_description"]
        self.output_dir = Path(config["output_dir"])
        self.robot_id = config["robot"]["id"]
        self.robot_type = config["robot"].get("type", "so101_follower")
        self.robot_port = config["robot"]["port"]
        self.fps = config["recording"]["fps"]
        self.frame_duration = 1.0 / self.fps

        # Camera config
        cam_cfg = config["camera"]
        self.camera_name = cam_cfg.get("name", "wrist")
        self.camera_device = cam_cfg["device"]
        self.capture_size = (cam_cfg["capture_width"], cam_cfg["capture_height"])
        self.target_size = (cam_cfg["target_width"], cam_cfg["target_height"])

        # Hub config
        self.push_to_hub = config["hub"]["push_to_hub"]
        self.hub_repo_id = config["hub"]["repo_id"]

        # Dataset config (optional overrides)
        dataset_cfg = config.get("dataset", {})
        self.use_videos = dataset_cfg.get("use_videos", True)
        self.vcodec = dataset_cfg.get("vcodec", "libsvtav1")
        self.video_encoding_batch_size = dataset_cfg.get("video_encoding_batch_size", 1)
        self.image_writer_processes = dataset_cfg.get("image_writer_processes", 0)
        self.image_writer_threads = dataset_cfg.get("image_writer_threads", 0)

        # Hardware handles
        self.bus: Optional[FeetechMotorsBus] = None
        self.camera: Optional[cv2.VideoCapture] = None

        # Recording state
        self.recording = False
        self.running = False

        # Keyboard state
        self._lock = Lock()
        self._space_pressed = False
        self._r_pressed = False
        self._esc_pressed = False
        self._keyboard_listener: Optional[keyboard.Listener] = None
        self._gripper_open_step = False
        self._gripper_close_step = False
        self._gripper_target: Optional[float] = None
        self._wrist_roll_neg_step = False
        self._wrist_roll_pos_step = False
        self._wrist_roll_target: Optional[float] = None
        self._d_pressed = False

        # Dataset
        self.dataset: Optional[LeRobotDataset] = None
        self._dataset_initialized = False
        self.dataset_root: Optional[Path] = None
        self.repo_id: Optional[str] = None

        # Build dataset features to match native LeRobot format
        joint_features = {name: float for name in self.JOINT_NAMES}
        obs_hw_features = {
            **joint_features,
            self.camera_name: (self.target_size[1], self.target_size[0], 3),
        }
        obs_features = hw_to_dataset_features(obs_hw_features, prefix=OBS_STR, use_video=self.use_videos)
        action_features = hw_to_dataset_features(joint_features, prefix=ACTION, use_video=self.use_videos)
        self.dataset_features = combine_feature_dicts(action_features, obs_features)

    def _init_dataset(self):
        """Initialize or resume the dataset."""
        dataset_root = self.output_dir / self.dataset_name
        repo_id = self.hub_repo_id or self.dataset_name

        if dataset_root.exists() and not (dataset_root / "meta" / "info.json").exists():
            raise RuntimeError(
                f"Dataset folder exists but is not a valid LeRobot dataset: {dataset_root}"
            )

        if (dataset_root / "meta" / "info.json").exists():
            print(f"Resuming existing dataset at {dataset_root}")
            self.dataset = LeRobotDataset(
                repo_id=repo_id,
                root=dataset_root,
                batch_encoding_size=self.video_encoding_batch_size,
                vcodec=self.vcodec,
            )
            if self.image_writer_processes or self.image_writer_threads:
                self.dataset.start_image_writer(
                    num_processes=self.image_writer_processes,
                    num_threads=self.image_writer_threads,
                )
            self._validate_resume_dataset()
        else:
            print(f"Creating new dataset at {dataset_root}")
            self.dataset = LeRobotDataset.create(
                repo_id=repo_id,
                fps=self.fps,
                features=self.dataset_features,
                root=dataset_root,
                robot_type=self.robot_type,
                use_videos=self.use_videos,
                image_writer_processes=self.image_writer_processes,
                image_writer_threads=self.image_writer_threads,
                batch_encoding_size=self.video_encoding_batch_size,
                vcodec=self.vcodec,
            )

        self.dataset_root = dataset_root
        self.repo_id = repo_id
        self._dataset_initialized = True

    def _validate_resume_dataset(self) -> None:
        """Check that an existing dataset matches current recording settings."""
        if self.dataset is None:
            return
        expected_features = {**self.dataset_features, **DEFAULT_FEATURES}
        if self._normalize_features(self.dataset.features) != self._normalize_features(
            expected_features
        ):
            raise ValueError(
                "Existing dataset features do not match current recording setup. "
                "Please use a new dataset name or update the config."
            )
        if self.dataset.fps != self.fps:
            raise ValueError(
                f"Existing dataset fps ({self.dataset.fps}) != configured fps ({self.fps})."
            )
        if self.dataset.meta.robot_type and self.dataset.meta.robot_type != self.robot_type:
            raise ValueError(
                f"Existing dataset robot_type ({self.dataset.meta.robot_type}) != configured robot_type ({self.robot_type})."
            )

    @staticmethod
    def _normalize_features(features: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Normalize feature specs for comparison across serialization formats."""
        normalized: Dict[str, Dict[str, Any]] = {}
        for key, feature in features.items():
            if not isinstance(feature, dict):
                normalized[key] = feature
                continue
            norm = dict(feature)
            norm.pop("info", None)
            shape = norm.get("shape")
            if isinstance(shape, list):
                norm["shape"] = tuple(shape)
            normalized[key] = norm
        return normalized

    def connect(self):
        """Connect to the arm and camera, disable torque."""
        # Check calibration first
        cal_path = check_calibration(self.robot_id, self.robot_type)
        calibration = load_calibration(cal_path)

        # Create motor configuration
        motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }

        # Connect to motor bus
        print(f"Connecting to arm on {self.robot_port}...")
        self.bus = FeetechMotorsBus(
            port=self.robot_port,
            motors=motors,
            calibration=calibration,
        )
        self.bus.connect()

        # Disable torque for free movement, then enable torque only on gripper
        print("Disabling torque - arm can now be moved freely by hand")
        self.bus.disable_torque()
        self.bus.write("Operating_Mode", "gripper", OperatingMode.POSITION.value)
        self.bus.enable_torque("gripper")
        print("Gripper torque enabled for keyboard control")

        # Connect to camera
        print(f"Connecting to camera (device {self.camera_device})...")
        self.camera = cv2.VideoCapture(self.camera_device)

        if not self.camera.isOpened():
            raise RuntimeError(f"Failed to open camera device {self.camera_device}")

        # Set capture resolution
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_size[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_size[1])

        # Verify camera is working
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read from camera")

        actual_w = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Camera connected: {actual_w}x{actual_h}")

        # Initialize gripper target to current position (normalized 0-100)
        self._gripper_target = float(self.bus.read("Present_Position", "gripper"))
        # Initialize wrist roll target to current position (normalized -100 to 100)
        self._wrist_roll_target = float(self.bus.read("Present_Position", "wrist_roll"))

    def start_preview(self):
        """Initialize Rerun preview window."""
        rr.init("SO-101 Recording", spawn=True)
        rr.log("info/task", rr.TextDocument(f"Task: {self.task_description}"))

    def _read_arm_state(self) -> np.ndarray:
        """Read current joint positions from the arm."""
        pos_dict = self.bus.sync_read("Present_Position")

        # Get positions in order
        positions = np.array([
            pos_dict["shoulder_pan"],
            pos_dict["shoulder_lift"],
            pos_dict["elbow_flex"],
            pos_dict["wrist_flex"],
            pos_dict["wrist_roll"],
            pos_dict["gripper"],
        ], dtype=np.float32)

        # Convert from normalized (-100 to 100) to radians
        positions_rad = np.zeros(6, dtype=np.float32)
        for i in range(5):
            positions_rad[i] = positions[i] / 100.0 * np.pi
        # Gripper is 0-100, map to 0 to ~1.7 radians
        positions_rad[5] = positions[5] / 100.0 * 1.7

        return positions_rad

    def _read_camera(self) -> np.ndarray:
        """Read and resize camera frame."""
        ret, frame = self.camera.read()
        if not ret:
            raise RuntimeError("Failed to read camera frame")

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to target size
        frame_resized = cv2.resize(
            frame_rgb,
            (self.target_size[0], self.target_size[1]),
            interpolation=cv2.INTER_LINEAR
        )

        return frame_resized

    def start_episode(self):
        """Begin recording a new episode."""
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized")
        self.dataset.clear_episode_buffer(delete_images=True)
        self.recording = True

    def _state_dict(self, state: np.ndarray) -> Dict[str, float]:
        return {name: float(state[i]) for i, name in enumerate(self.JOINT_NAMES)}

    def add_frame(self, state: np.ndarray, image: np.ndarray):
        """Add a frame to the current episode using LeRobotDataset."""
        if self.dataset is None:
            raise RuntimeError("Dataset not initialized")

        values = self._state_dict(state)
        values[self.camera_name] = image

        observation_frame = build_dataset_frame(self.dataset_features, values, prefix=OBS_STR)
        action_frame = build_dataset_frame(self.dataset_features, values, prefix=ACTION)

        frame = {
            **observation_frame,
            **action_frame,
            "task": self.task_description,
        }
        self.dataset.add_frame(frame)

    def end_episode(self):
        """Save the current episode to disk."""
        if self.dataset is None:
            return

        num_frames = int(self.dataset.episode_buffer.get("size", 0))
        if num_frames == 0:
            print("No frames to save")
            self.recording = False
            return

        episode_idx = self.dataset.num_episodes
        self.dataset.save_episode()
        self.recording = False

        print(
            f"\r✓  Saved episode {episode_idx + 1} ({num_frames} frames, {num_frames/self.fps:.1f}s)\033[K"
        )

    def discard_episode(self):
        """Discard the current in-progress episode."""
        if self.recording:
            num_frames = 0
            if self.dataset is not None:
                num_frames = int(self.dataset.episode_buffer.get("size", 0))
                self.dataset.clear_episode_buffer(delete_images=True)
            print(f"\r✗  Discarded episode ({num_frames} frames)\033[K")
        self.recording = False

    def finalize(self):
        """Finalize dataset writers and optionally push to hub."""
        if self.dataset is None:
            return

        if self.dataset.num_episodes == 0:
            print("No episodes recorded!")
            return

        print("\nFinalizing dataset...")
        self.dataset.finalize()

        print("\nDataset finalized:")
        print(f"  Episodes: {self.dataset.num_episodes}")
        print(f"  Total frames: {self.dataset.meta.total_frames}")
        if self.dataset_root is not None:
            print(f"  Output: {self.dataset_root}")

        if self.push_to_hub and self.repo_id:
            print(f"\nPushing to hub: {self.repo_id}")
            try:
                self.dataset.push_to_hub()
                print("Successfully pushed to hub!")
            except Exception as e:
                print(f"Failed to push to hub: {e}")

    def _on_key_press(self, key):
        """Handle keyboard input."""
        with self._lock:
            try:
                if key == keyboard.Key.space:
                    self._space_pressed = True
                elif key == keyboard.Key.esc:
                    self._esc_pressed = True
                elif hasattr(key, 'char') and key.char == 'r':
                    self._r_pressed = True
                elif hasattr(key, 'char') and key.char == '1':
                    self._gripper_open_step = True
                elif hasattr(key, 'char') and key.char == '2':
                    self._gripper_close_step = True
                elif hasattr(key, 'char') and key.char == '3':
                    self._wrist_roll_neg_step = True
                elif hasattr(key, 'char') and key.char == '4':
                    self._wrist_roll_pos_step = True
                elif hasattr(key, 'char') and key.char == 'd':
                    self._d_pressed = True
            except AttributeError:
                pass

    def _check_space(self) -> bool:
        """Check and clear space flag."""
        with self._lock:
            if self._space_pressed:
                self._space_pressed = False
                return True
        return False

    def _check_r(self) -> bool:
        """Check and clear R flag."""
        with self._lock:
            if self._r_pressed:
                self._r_pressed = False
                return True
        return False

    def _check_esc(self) -> bool:
        """Check ESC flag."""
        with self._lock:
            return self._esc_pressed

    def _check_gripper_step(self) -> int:
        """Check and clear gripper step flags."""
        with self._lock:
            if self._gripper_open_step:
                self._gripper_open_step = False
                return 1
            if self._gripper_close_step:
                self._gripper_close_step = False
                return -1
        return 0

    def _check_wrist_roll_step(self) -> int:
        """Check and clear wrist roll step flags."""
        with self._lock:
            if self._wrist_roll_neg_step:
                self._wrist_roll_neg_step = False
                return -1
            if self._wrist_roll_pos_step:
                self._wrist_roll_pos_step = False
                return 1
        return 0

    def _check_delete(self) -> bool:
        """Check and clear delete flag."""
        with self._lock:
            if self._d_pressed:
                self._d_pressed = False
                return True
        return False

    def _parse_episode_ranges(self, text: str) -> list[int]:
        """Parse ranges like '0-3,7,10-12' into sorted unique indices."""
        if not text.strip():
            return []
        indices: set[int] = set()
        parts = [p.strip() for p in text.split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                start_str, end_str = [s.strip() for s in part.split("-", 1)]
                if not start_str or not end_str:
                    raise ValueError(f"Invalid range: '{part}'")
                start = int(start_str)
                end = int(end_str)
                if end < start:
                    raise ValueError(f"Invalid range (end < start): '{part}'")
                indices.update(range(start, end + 1))
            else:
                indices.add(int(part))
        return sorted(indices)

    def _delete_episodes_interactive(self) -> None:
        """Prompt for episode ranges and delete in-place."""
        if self.recording:
            print("\nStop recording before deleting episodes.")
            return
        if self.dataset is None or self.dataset_root is None or self.repo_id is None:
            print("\nDataset not initialized.")
            return
        total_eps = self.dataset.meta.total_episodes
        if total_eps == 0:
            print("\nNo episodes to delete.")
            return

        print(f"\nDelete episodes from 0 to {total_eps - 1}.")
        ranges = input("Enter episode ranges (e.g., 0-3,7,10-12), or blank to cancel: ").strip()
        if ranges and ranges[0] in ("d", "D"):
            ranges = ranges[1:].strip()
        if not ranges:
            print("Delete canceled.")
            return
        try:
            indices = self._parse_episode_ranges(ranges)
        except ValueError as e:
            print(f"Invalid range: {e}")
            return

        if not indices:
            print("No indices provided.")
            return
        invalid = [i for i in indices if i < 0 or i >= total_eps]
        if invalid:
            print(f"Invalid episode indices: {invalid}")
            return
        if len(indices) >= total_eps:
            print("Cannot delete all episodes.")
            return

        confirm = input(f"Delete episodes {indices}? (y/N): ").strip().lower()
        if confirm != "y":
            print("Delete canceled.")
            return

        print("\nFinalizing current dataset...")
        self.dataset.finalize()

        old_path = Path(str(self.dataset_root) + "_old")
        if old_path.exists():
            shutil.rmtree(old_path)
        shutil.move(str(self.dataset_root), str(old_path))

        print(f"Deleting episodes from {self.repo_id} (in-place)...")
        src_dataset = LeRobotDataset(self.repo_id, root=old_path)
        delete_episodes(
            src_dataset,
            episode_indices=indices,
            output_dir=self.dataset_root,
            repo_id=self.repo_id,
        )

        self.dataset = LeRobotDataset(
            repo_id=self.repo_id,
            root=self.dataset_root,
            batch_encoding_size=self.video_encoding_batch_size,
            vcodec=self.vcodec,
        )
        if self.image_writer_processes or self.image_writer_threads:
            self.dataset.start_image_writer(
                num_processes=self.image_writer_processes,
                num_threads=self.image_writer_threads,
            )

        print(
            f"Deleted episodes. New dataset episodes: {self.dataset.meta.total_episodes}. "
            f"Old dataset moved to {old_path}."
        )

    def run(self):
        """Main recording loop."""
        print("\n" + "=" * 60)
        print("SO-101 Single-Arm Recording")
        print("=" * 60)
        print(f"Task: {self.task_description}")
        print(f"Dataset: {self.dataset_name}")
        print(f"FPS: {self.fps}")
        print(f"Output: {self.output_dir / self.dataset_name}")
        print("=" * 60)
        print("\nControls:")
        print("  SPACE: Start/stop recording")
        print("  R: Discard current episode")
        print("  1: Open gripper 25%")
        print("  2: Close gripper 25%")
        print("  3: Wrist roll -10%")
        print("  4: Wrist roll +10%")
        print("  D: Delete episodes (when paused)")
        print("  ESC: Finalize and quit")
        print("=" * 60 + "\n")

        # Initialize dataset
        self._init_dataset()

        # Connect hardware
        self.connect()

        # Start preview
        self.start_preview()

        # Start keyboard listener
        self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press)
        self._keyboard_listener.start()

        self.running = True

        try:
            while self.running:
                frame_start = time.time()

                # Check keyboard
                if self._check_esc():
                    self.discard_episode()
                    break

                if self._check_r():
                    self.discard_episode()

                if self._check_delete():
                    self._delete_episodes_interactive()

                if self._check_space():
                    if self.recording:
                        self.end_episode()
                    else:
                        self.start_episode()

                gripper_step = self._check_gripper_step()
                if gripper_step != 0 and self._gripper_target is not None:
                    step = 25.0 * gripper_step
                    self._gripper_target = max(0.0, min(100.0, self._gripper_target + step))
                    try:
                        self.bus.write("Goal_Position", "gripper", self._gripper_target)
                    except Exception as e:
                        print(f"\nGripper command error: {e}")

                wrist_roll_step = self._check_wrist_roll_step()
                if wrist_roll_step != 0 and self._wrist_roll_target is not None:
                    step = 10.0 * wrist_roll_step
                    self._wrist_roll_target = max(-100.0, min(100.0, self._wrist_roll_target + step))
                    try:
                        self.bus.write("Goal_Position", "wrist_roll", self._wrist_roll_target)
                    except Exception as e:
                        print(f"\nWrist roll command error: {e}")

                # Read arm and camera
                try:
                    state = self._read_arm_state()
                    image = self._read_camera()
                except Exception as e:
                    print(f"\nDevice error: {e}")
                    if self.recording:
                        self.discard_episode()
                    continue

                # Add frame if recording
                if self.recording:
                    self.add_frame(state, image)

                    # Update status line (overwrite in place)
                    n_frames = 0
                    if self.dataset is not None:
                        n_frames = int(self.dataset.episode_buffer.get("size", 0))
                    elapsed = n_frames / self.fps
                    next_ep = self.dataset.num_episodes + 1 if self.dataset else 1
                    status = f"\r⏺  Episode {next_ep} | {n_frames} frames | {elapsed:.1f}s"
                    print(f"{status}\033[K", end="", flush=True)

                # Update Rerun preview
                rr.log("camera/wrist", rr.Image(image))

                if self.recording:
                    rr.log("status", rr.TextDocument(
                        f"RECORDING - Episode {self.dataset.num_episodes + 1} - "
                        f"{int(self.dataset.episode_buffer.get('size', 0))} frames"
                    ))
                else:
                    rr.log("status", rr.TextDocument(
                        f"PAUSED - {self.dataset.num_episodes if self.dataset else 0} episodes saved"
                    ))

                # Maintain frame rate
                elapsed = time.time() - frame_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)

        except KeyboardInterrupt:
            print("\n\nInterrupted.")
            self.discard_episode()

        finally:
            self._cleanup()

    def _cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")

        self.running = False

        if self._keyboard_listener:
            self._keyboard_listener.stop()

        if self.camera:
            self.camera.release()

        if self.bus:
            self.bus.disconnect()

        # Finalize dataset
        if self._dataset_initialized:
            self.finalize()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from JSON file."""
    path = Path(config_path)

    if not path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    with open(path, 'r') as f:
        config = json.load(f)

    # Validate required fields
    required = [
        "dataset_name", "task_description", "output_dir",
        "robot", "camera", "recording", "hub"
    ]

    for field in required:
        if field not in config:
            print(f"Error: Missing required config field: {field}")
            sys.exit(1)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Record single-arm imitation learning dataset"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to JSON configuration file"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Create and run recorder
    recorder = SingleArmRecorder(config)
    recorder.run()


if __name__ == "__main__":
    main()
