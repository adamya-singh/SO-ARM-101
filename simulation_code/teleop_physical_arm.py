"""
Physical Arm Teleoperation Handler for SO-101 Robot

Reads joint positions from a physical SO-101 arm via USB and mirrors
them to the MuJoCo simulation for demonstration recording.

Requirements:
    - LeRobot with feetech extras: pip install -e ".[feetech]"
    - Calibrated SO-101 arm (run: python -m lerobot.calibrate --robot.type=so101_follower)

Usage:
    The physical arm acts as a "leader" - move it by hand and the simulation follows.
    
    Episode Control (via keyboard):
        Space: Start/stop episode recording
        R: Reset simulation
        ESC: Quit
"""

import numpy as np
import time
from typing import Optional, Dict, Any
from pathlib import Path
from threading import Thread, Lock
from pynput import keyboard

# LeRobot imports for Feetech motor communication
from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus


class PhysicalArmTeleop:
    """
    Physical arm teleoperation controller for SO-101 robot.
    
    Reads joint positions from the physical arm and provides them
    in a format compatible with the simulation environment.
    """
    
    # Joint names in order
    JOINT_NAMES = [
        "shoulder_pan",
        "shoulder_lift", 
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "gripper",
    ]
    
    def __init__(
        self,
        port: str = "/dev/tty.usbmodem5A680096011",
        calibration_dir: Optional[str] = None,
    ):
        """
        Initialize physical arm teleop controller.
        
        Args:
            port: USB port for the SO-101 arm
            calibration_dir: Directory containing calibration file (optional)
        """
        self.port = port
        self.calibration_dir = calibration_dir
        
        # Motor bus (initialized on start)
        self.bus: Optional[FeetechMotorsBus] = None
        
        # Position tracking
        self._current_positions: np.ndarray = np.zeros(6, dtype=np.float32)
        self._previous_positions: np.ndarray = np.zeros(6, dtype=np.float32)
        self._lock = Lock()
        
        # State flags
        self._episode_toggle = False
        self._reset_requested = False
        self._quit_requested = False
        
        # Keyboard listener for episode control
        self._keyboard_listener: Optional[keyboard.Listener] = None
        
        # Reading thread
        self._running = False
        self._read_thread: Optional[Thread] = None
    
    def _load_calibration(self) -> Optional[Dict[str, Any]]:
        """Load calibration from file if available."""
        from lerobot.motors import MotorCalibration
        
        # LeRobot stores calibration at this path
        cal_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / "calibration" / "robots" / "so101_follower"
        cal_path = cal_dir / "so101_follower_1.json"
        
        if not cal_path.exists():
            # Try None.json as fallback
            cal_path = cal_dir / "None.json"
        
        if not cal_path.exists():
            print(f"Warning: No calibration file found in {cal_dir}")
            print("Run: python -m lerobot.calibrate --robot.type=so101_follower --robot.port=YOUR_PORT")
            return None
        
        try:
            import json
            with open(cal_path, 'r') as f:
                cal_data = json.load(f)
            
            # Convert to MotorCalibration format expected by FeetechMotorsBus
            calibration = {}
            for motor_name, motor_cal in cal_data.items():
                calibration[motor_name] = MotorCalibration(
                    id=motor_cal['id'],
                    drive_mode=motor_cal['drive_mode'],
                    homing_offset=motor_cal['homing_offset'],
                    range_min=motor_cal['range_min'],
                    range_max=motor_cal['range_max'],
                )
            
            print(f"Loaded calibration from {cal_path}")
            return calibration
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return None
    
    def start(self):
        """Start the physical arm connection and reading."""
        print(f"Connecting to SO-101 arm on {self.port}...")
        
        # Create motor configuration
        # Using RANGE_M100_100 for normalized output (-100 to 100 range)
        # We'll convert to radians ourselves
        motors = {
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        }
        
        # Initialize motor bus
        calibration = self._load_calibration()
        self.bus = FeetechMotorsBus(
            port=self.port,
            motors=motors,
            calibration=calibration,
        )
        
        # Connect to the bus
        self.bus.connect()
        
        # Disable torque so arm can be moved freely by hand
        print("Disabling torque - you can now move the arm freely...")
        self.bus.disable_torque()
        
        # Start keyboard listener for episode control
        self._keyboard_listener = keyboard.Listener(
            on_press=self._on_key_press,
        )
        self._keyboard_listener.start()
        
        # Start reading thread
        self._running = True
        self._read_thread = Thread(target=self._reading_loop, daemon=True)
        self._read_thread.start()
        
        # Initial position read
        time.sleep(0.1)
        self._read_positions()
        self._previous_positions = self._current_positions.copy()
        
        print("\nPhysical arm teleop started. Controls:")
        print("  Move the physical arm to control the simulation")
        print("  Space: Start/stop episode recording")
        print("  R: Reset simulation")
        print("  ESC: Quit")
    
    def stop(self):
        """Stop the physical arm connection."""
        self._running = False
        
        if self._read_thread is not None:
            self._read_thread.join(timeout=1.0)
            self._read_thread = None
        
        if self._keyboard_listener is not None:
            self._keyboard_listener.stop()
            self._keyboard_listener = None
        
        if self.bus is not None:
            self.bus.disconnect()
            self.bus = None
        
        print("Physical arm teleop stopped.")
    
    def _on_key_press(self, key):
        """Handle keyboard input for episode control."""
        try:
            if key == keyboard.Key.space:
                self._episode_toggle = True
            elif key.char == 'r':
                self._reset_requested = True
            elif key == keyboard.Key.esc:
                self._quit_requested = True
        except AttributeError:
            if key == keyboard.Key.esc:
                self._quit_requested = True
    
    def _reading_loop(self):
        """Background thread for reading arm positions."""
        while self._running:
            try:
                self._read_positions()
                time.sleep(0.02)  # 50 Hz reading
            except Exception as e:
                print(f"Error reading positions: {e}")
                time.sleep(0.1)
    
    def _read_positions(self):
        """Read current positions from the physical arm."""
        if self.bus is None:
            return
        
        try:
            # Read all motor positions
            pos_dict = self.bus.sync_read("Present_Position")
            
            # Convert to numpy array in correct order
            positions = np.array([
                pos_dict["shoulder_pan"],
                pos_dict["shoulder_lift"],
                pos_dict["elbow_flex"],
                pos_dict["wrist_flex"],
                pos_dict["wrist_roll"],
                pos_dict["gripper"],
            ], dtype=np.float32)
            
            # Convert from normalized range to radians
            # Motors return -100 to 100 (normalized range)
            # Convert to radians: -100 to 100 maps to approximately -pi to pi
            positions_rad = np.zeros(6, dtype=np.float32)
            for i in range(5):  # First 5 joints
                positions_rad[i] = positions[i] / 100.0 * np.pi
            # Gripper is 0-100, map to 0 to ~1.7 radians
            positions_rad[5] = positions[5] / 100.0 * 1.7
            
            with self._lock:
                self._previous_positions = self._current_positions.copy()
                self._current_positions = positions_rad
                
        except Exception as e:
            print(f"Error reading positions: {e}")
    
    def get_joint_positions(self) -> np.ndarray:
        """
        Get current joint positions in radians.
        
        Returns:
            Array of shape (6,) with joint positions in radians
        """
        with self._lock:
            return self._current_positions.copy()
    
    def get_action_delta(self) -> np.ndarray:
        """
        Get joint position deltas since last call.
        
        For compatibility with other teleop interfaces.
        
        Returns:
            Array of shape (6,) with joint position deltas in radians
        """
        with self._lock:
            delta = self._current_positions - self._previous_positions
            self._previous_positions = self._current_positions.copy()
            return delta
    
    def check_episode_toggle(self) -> bool:
        """Check and clear episode toggle flag."""
        if self._episode_toggle:
            self._episode_toggle = False
            return True
        return False
    
    def check_reset_requested(self) -> bool:
        """Check and clear reset flag."""
        if self._reset_requested:
            self._reset_requested = False
            return True
        return False
    
    def check_quit_requested(self) -> bool:
        """Check quit flag."""
        return self._quit_requested
    
    def set_callbacks(self, **kwargs):
        """For API compatibility with other teleop interfaces."""
        pass


def test_physical_arm_teleop():
    """Test the physical arm teleop controller."""
    print("="*60)
    print("Physical Arm Teleop Test")
    print("="*60)
    print("\nMake sure your SO-101 arm is connected via USB.")
    print("Press ESC to quit.\n")
    
    teleop = PhysicalArmTeleop()
    
    try:
        teleop.start()
        
        while not teleop.check_quit_requested():
            positions = teleop.get_joint_positions()
            
            # Print positions
            pos_str = ", ".join([f"{p:+.3f}" for p in positions])
            print(f"\rPositions (rad): [{pos_str}]", end="", flush=True)
            
            if teleop.check_episode_toggle():
                print("\n>>> Episode toggle (Space pressed)")
            
            if teleop.check_reset_requested():
                print("\n>>> Reset requested (R pressed)")
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        teleop.stop()


if __name__ == "__main__":
    test_physical_arm_teleop()

