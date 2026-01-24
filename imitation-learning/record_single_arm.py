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
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


def check_calibration(robot_id: str) -> Path:
    """
    Check if calibration file exists for the robot.

    Args:
        robot_id: Robot identifier used during calibration

    Returns:
        Path to calibration file

    Raises:
        SystemExit: If calibration file not found
    """
    cal_dir = (
        Path.home() / ".cache" / "huggingface" / "lerobot"
        / "calibration" / "robots" / "so101_follower"
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
    print(f"  lerobot-calibrate --robot.type=so101_follower --robot.id={robot_id}")
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
        self.robot_port = config["robot"]["port"]
        self.fps = config["recording"]["fps"]
        self.frame_duration = 1.0 / self.fps

        # Camera config
        cam_cfg = config["camera"]
        self.camera_device = cam_cfg["device"]
        self.capture_size = (cam_cfg["capture_width"], cam_cfg["capture_height"])
        self.target_size = (cam_cfg["target_width"], cam_cfg["target_height"])

        # Hub config
        self.push_to_hub = config["hub"]["push_to_hub"]
        self.hub_repo_id = config["hub"]["repo_id"]

        # Hardware handles
        self.bus: Optional[FeetechMotorsBus] = None
        self.camera: Optional[cv2.VideoCapture] = None

        # Recording state
        self.recording = False
        self.episode_frames: list = []
        self.episode_start_time = 0.0
        self.episodes_saved = 0
        self.running = False

        # Keyboard state
        self._lock = Lock()
        self._space_pressed = False
        self._r_pressed = False
        self._esc_pressed = False
        self._keyboard_listener: Optional[keyboard.Listener] = None

        # Dataset writer (lazy import to avoid issues if not using LeRobot dataset API)
        self.dataset = None
        self._dataset_initialized = False

    def _init_dataset(self):
        """Initialize or resume the dataset."""
        dataset_path = self.output_dir / self.dataset_name

        # Check if resuming existing dataset
        if (dataset_path / "meta" / "info.json").exists():
            print(f"Resuming existing dataset at {dataset_path}")
            self._init_resume_dataset(dataset_path)
        else:
            print(f"Creating new dataset at {dataset_path}")
            self._init_new_dataset(dataset_path)

        self._dataset_initialized = True

    def _init_new_dataset(self, dataset_path: Path):
        """Initialize a new dataset using custom writer (LeRobot v3 format)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        # Create directory structure
        dataset_path.mkdir(parents=True, exist_ok=True)
        (dataset_path / "meta").mkdir(exist_ok=True)
        (dataset_path / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (dataset_path / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        (dataset_path / "videos" / "observation.images.wrist" / "chunk-000").mkdir(parents=True, exist_ok=True)

        # Write tasks.parquet
        tasks_data = {"task_index": [0], "task": [self.task_description]}
        pq.write_table(pa.table(tasks_data), dataset_path / "meta" / "tasks.parquet")

        self.dataset_path = dataset_path
        self.episodes_saved = 0

    def _init_resume_dataset(self, dataset_path: Path):
        """Resume an existing dataset."""
        import pyarrow.parquet as pq

        self.dataset_path = dataset_path

        # Find next episode index
        data_dir = dataset_path / "data" / "chunk-000"
        existing = list(data_dir.glob("episode_*.parquet"))

        if existing:
            indices = []
            for f in existing:
                try:
                    idx = int(f.stem.split("_")[1])
                    indices.append(idx)
                except (IndexError, ValueError):
                    continue
            self.episodes_saved = max(indices) + 1 if indices else 0
        else:
            self.episodes_saved = 0

        print(f"Found {self.episodes_saved} existing episodes")

    def connect(self):
        """Connect to the arm and camera, disable torque."""
        # Check calibration first
        cal_path = check_calibration(self.robot_id)
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

        # Disable torque for free movement (including gripper - can be positioned by hand)
        print("Disabling torque - arm and gripper can now be moved freely by hand")
        self.bus.disable_torque()

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
        self.episode_frames = []
        self.episode_start_time = time.time()
        self.recording = True

    def add_frame(self, state: np.ndarray, image: np.ndarray, timestamp: float):
        """Add a frame to the current episode."""
        self.episode_frames.append({
            "observation.state": state.copy(),
            "observation.images.wrist": image.copy(),
            "action": state.copy(),  # Action = state for single-arm
            "timestamp": timestamp,
        })

    def end_episode(self):
        """Save the current episode to disk."""
        if not self.episode_frames:
            print("No frames to save")
            self.recording = False
            return

        import imageio
        import pyarrow as pa
        import pyarrow.parquet as pq

        num_frames = len(self.episode_frames)
        episode_idx = self.episodes_saved

        # Save video
        video_path = (
            self.dataset_path / "videos" / "observation.images.wrist"
            / "chunk-000" / f"episode_{episode_idx:06d}.mp4"
        )

        writer = imageio.get_writer(
            str(video_path),
            fps=self.fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p',
        )

        for frame in self.episode_frames:
            writer.append_data(frame["observation.images.wrist"])
        writer.close()

        # Save parquet data
        schema = pa.schema([
            ('frame_index', pa.int64()),
            ('episode_index', pa.int64()),
            ('timestamp', pa.float64()),
            ('task_index', pa.int64()),
            ('observation.state', pa.list_(pa.float32(), 6)),
            ('action', pa.list_(pa.float32(), 6)),
        ])

        arrays = [
            pa.array(list(range(num_frames)), type=pa.int64()),
            pa.array([episode_idx] * num_frames, type=pa.int64()),
            pa.array([f["timestamp"] for f in self.episode_frames], type=pa.float64()),
            pa.array([0] * num_frames, type=pa.int64()),
            pa.array([f["observation.state"].tolist() for f in self.episode_frames],
                     type=pa.list_(pa.float32(), 6)),
            pa.array([f["action"].tolist() for f in self.episode_frames],
                     type=pa.list_(pa.float32(), 6)),
        ]

        table = pa.table(dict(zip(schema.names, arrays)), schema=schema)
        parquet_path = (
            self.dataset_path / "data" / "chunk-000"
            / f"episode_{episode_idx:06d}.parquet"
        )
        pq.write_table(table, parquet_path)

        self.episodes_saved += 1
        self.recording = False
        self.episode_frames = []

        print(f"\r✓  Saved episode {episode_idx + 1} ({num_frames} frames, {num_frames/self.fps:.1f}s)\033[K")

    def discard_episode(self):
        """Discard the current in-progress episode."""
        if self.recording:
            print(f"\r✗  Discarded episode ({len(self.episode_frames)} frames)\033[K")
        self.recording = False
        self.episode_frames = []

    def finalize(self):
        """Compute stats and write final metadata."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        if self.episodes_saved == 0:
            print("No episodes recorded!")
            return

        print("\nFinalizing dataset...")

        # Read all episode data to compute stats
        all_states = []
        all_actions = []
        episode_metadata = []
        total_frames = 0

        data_dir = self.dataset_path / "data" / "chunk-000"
        for ep_idx in range(self.episodes_saved):
            parquet_path = data_dir / f"episode_{ep_idx:06d}.parquet"
            if not parquet_path.exists():
                continue

            table = pq.read_table(parquet_path)
            df = table.to_pandas()

            num_frames = len(df)
            length_s = num_frames / self.fps

            episode_metadata.append({
                'episode_index': ep_idx,
                'num_frames': num_frames,
                'length_s': length_s,
                'task_index': 0,
            })

            for state in df['observation.state']:
                all_states.append(np.array(state, dtype=np.float32))
            for action in df['action']:
                all_actions.append(np.array(action, dtype=np.float32))

            total_frames += num_frames

        # Compute statistics
        states_arr = np.array(all_states)
        actions_arr = np.array(all_actions)

        stats = {
            'observation.state': {
                'mean': states_arr.mean(axis=0).tolist(),
                'std': states_arr.std(axis=0).tolist(),
                'min': states_arr.min(axis=0).tolist(),
                'max': states_arr.max(axis=0).tolist(),
            },
            'action': {
                'mean': actions_arr.mean(axis=0).tolist(),
                'std': actions_arr.std(axis=0).tolist(),
                'min': actions_arr.min(axis=0).tolist(),
                'max': actions_arr.max(axis=0).tolist(),
            },
            # ImageNet normalization for camera
            'observation.images.wrist': {
                'mean': [[[0.485]], [[0.456]], [[0.406]]],
                'std': [[[0.229]], [[0.224]], [[0.225]]],
                'min': [[[0.0]], [[0.0]], [[0.0]]],
                'max': [[[1.0]], [[1.0]], [[1.0]]],
            },
        }

        with open(self.dataset_path / "meta" / "stats.json", 'w') as f:
            json.dump(stats, f, indent=2)

        # Write info.json
        info = {
            'codebase_version': '3.0',
            'robot_type': 'so101',
            'fps': self.fps,
            'total_episodes': self.episodes_saved,
            'total_frames': total_frames,
            'total_tasks': 1,
            'total_videos': self.episodes_saved,
            'total_chunks': 1,
            'chunks_size': self.episodes_saved,
            'data_path': 'data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet',
            'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/episode_{file_index:06d}.mp4',
            'features': {
                'timestamp': {'dtype': 'float64', 'shape': [1]},
                'frame_index': {'dtype': 'int64', 'shape': [1]},
                'episode_index': {'dtype': 'int64', 'shape': [1]},
                'task_index': {'dtype': 'int64', 'shape': [1]},
                'observation.images.wrist': {
                    'dtype': 'video',
                    'shape': [self.target_size[1], self.target_size[0], 3],
                    'names': ['height', 'width', 'channels'],
                    'video_info': {
                        'video.fps': self.fps,
                        'video.codec': 'libx264',
                        'video.pix_fmt': 'yuv420p',
                    },
                },
                'observation.state': {
                    'dtype': 'float32',
                    'shape': [6],
                    'names': self.JOINT_NAMES,
                },
                'action': {
                    'dtype': 'float32',
                    'shape': [6],
                    'names': self.JOINT_NAMES,
                },
            },
            'repo_id': self.dataset_name,
        }

        with open(self.dataset_path / "meta" / "info.json", 'w') as f:
            json.dump(info, f, indent=2)

        # Write episodes.parquet
        dataset_from = []
        dataset_to = []
        current_idx = 0
        for ep in episode_metadata:
            dataset_from.append(current_idx)
            current_idx += ep['num_frames']
            dataset_to.append(current_idx)

        num_eps = len(episode_metadata)
        episodes_data = {
            'episode_index': [ep['episode_index'] for ep in episode_metadata],
            'num_frames': [ep['num_frames'] for ep in episode_metadata],
            'length_s': [ep['length_s'] for ep in episode_metadata],
            'task_index': [ep['task_index'] for ep in episode_metadata],
            'dataset_from_index': dataset_from,
            'dataset_to_index': dataset_to,
            'videos/observation.images.wrist/from_timestamp': [0.0] * num_eps,
            'videos/observation.images.wrist/to_timestamp': [ep['length_s'] for ep in episode_metadata],
            'videos/observation.images.wrist/chunk_index': [0] * num_eps,
            'videos/observation.images.wrist/file_index': [ep['episode_index'] for ep in episode_metadata],
        }

        pq.write_table(
            pa.table(episodes_data),
            self.dataset_path / "meta" / "episodes" / "chunk-000" / "episodes.parquet"
        )

        print(f"\nDataset finalized:")
        print(f"  Episodes: {self.episodes_saved}")
        print(f"  Total frames: {total_frames}")
        print(f"  Output: {self.dataset_path}")

        # Push to hub if configured
        if self.push_to_hub and self.hub_repo_id:
            print(f"\nPushing to hub: {self.hub_repo_id}")
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                api.upload_folder(
                    folder_path=str(self.dataset_path),
                    repo_id=self.hub_repo_id,
                    repo_type="dataset",
                )
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

                if self._check_space():
                    if self.recording:
                        self.end_episode()
                    else:
                        self.start_episode()

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
                    timestamp = time.time() - self.episode_start_time
                    self.add_frame(state, image, timestamp)

                    # Update status line (overwrite in place)
                    n_frames = len(self.episode_frames)
                    elapsed = n_frames / self.fps
                    status = f"\r⏺  Episode {self.episodes_saved + 1} | {n_frames} frames | {elapsed:.1f}s"
                    print(f"{status}\033[K", end="", flush=True)

                # Update Rerun preview
                rr.log("camera/wrist", rr.Image(image))

                if self.recording:
                    rr.log("status", rr.TextDocument(
                        f"RECORDING - Episode {self.episodes_saved + 1} - "
                        f"{len(self.episode_frames)} frames"
                    ))
                else:
                    rr.log("status", rr.TextDocument(
                        f"PAUSED - {self.episodes_saved} episodes saved"
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
