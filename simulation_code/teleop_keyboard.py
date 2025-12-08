"""
Keyboard Teleoperation Handler for SO-101 Robot

Maps keyboard inputs to joint position deltas for controlling
the SO-101 robot arm in MuJoCo simulation.

Key Mappings:
    Movement (End-effector approximation):
        Arrow Up/Down: Move forward/backward (shoulder_pan)
        Arrow Left/Right: Move left/right (shoulder_pan)
        W/S: Move up/down (shoulder_lift + elbow_flex)
        
    Wrist Control:
        A/D: Wrist roll left/right
        Z/X: Wrist flex up/down
        
    Gripper:
        Q: Open gripper
        E: Close gripper
        
    Episode Control:
        Space: Start/stop recording episode
        R: Reset environment
        ESC: Quit
"""

import numpy as np
from pynput import keyboard
from threading import Lock
from typing import Dict, Set, Optional, Callable


class KeyboardTeleop:
    """
    Keyboard teleoperation controller for SO-101 robot.
    
    Captures keyboard input and converts to joint position deltas.
    Thread-safe for use with async keyboard listener.
    """
    
    # Joint indices
    SHOULDER_PAN = 0
    SHOULDER_LIFT = 1
    ELBOW_FLEX = 2
    WRIST_FLEX = 3
    WRIST_ROLL = 4
    GRIPPER = 5
    
    def __init__(
        self,
        delta_scale: float = 0.05,
        gripper_scale: float = 0.1,
    ):
        """
        Initialize keyboard teleop controller.
        
        Args:
            delta_scale: Scale factor for joint position deltas (radians)
            gripper_scale: Scale factor for gripper movement
        """
        self.delta_scale = delta_scale
        self.gripper_scale = gripper_scale
        
        # Currently pressed keys
        self._pressed_keys: Set[str] = set()
        self._lock = Lock()
        
        # State flags
        self._episode_toggle = False  # True when space was just pressed
        self._reset_requested = False
        self._quit_requested = False
        
        # Callbacks
        self._on_episode_toggle: Optional[Callable] = None
        self._on_reset: Optional[Callable] = None
        self._on_quit: Optional[Callable] = None
        
        # Keyboard listener
        self._listener: Optional[keyboard.Listener] = None
        
        # Key mappings (key -> (joint_index, delta_multiplier))
        self._key_mappings = {
            # Forward/backward motion via shoulder pan
            'up': (self.SHOULDER_PAN, -1.0),
            'down': (self.SHOULDER_PAN, 1.0),
            
            # Left/right motion via shoulder pan (combined with lift for arc)
            'left': (self.SHOULDER_PAN, 1.0),
            'right': (self.SHOULDER_PAN, -1.0),
            
            # Up/down motion via shoulder lift
            'w': (self.SHOULDER_LIFT, -1.0),  # Lift up (negative = up)
            's': (self.SHOULDER_LIFT, 1.0),   # Lower down
            
            # Elbow flex for reach adjustment
            'i': (self.ELBOW_FLEX, 1.0),   # Extend
            'k': (self.ELBOW_FLEX, -1.0),  # Retract
            
            # Wrist flex
            'z': (self.WRIST_FLEX, 1.0),   # Flex down
            'x': (self.WRIST_FLEX, -1.0),  # Flex up
            
            # Wrist roll
            'a': (self.WRIST_ROLL, 1.0),   # Roll left
            'd': (self.WRIST_ROLL, -1.0),  # Roll right
            
            # Gripper
            'q': (self.GRIPPER, 1.0),   # Open
            'e': (self.GRIPPER, -1.0),  # Close
        }
    
    def start(self):
        """Start the keyboard listener."""
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release
        )
        self._listener.start()
        print("Keyboard teleop started. Controls:")
        print("  Arrow keys: Move shoulder pan")
        print("  W/S: Shoulder lift up/down")
        print("  I/K: Elbow extend/retract")
        print("  Z/X: Wrist flex down/up")
        print("  A/D: Wrist roll left/right")
        print("  Q/E: Gripper open/close")
        print("  Space: Start/stop episode recording")
        print("  R: Reset environment")
        print("  ESC: Quit")
    
    def stop(self):
        """Stop the keyboard listener."""
        if self._listener is not None:
            self._listener.stop()
            self._listener = None
    
    def _normalize_key(self, key) -> Optional[str]:
        """Convert pynput key to string."""
        try:
            # Regular character keys
            return key.char.lower()
        except AttributeError:
            # Special keys
            if key == keyboard.Key.up:
                return 'up'
            elif key == keyboard.Key.down:
                return 'down'
            elif key == keyboard.Key.left:
                return 'left'
            elif key == keyboard.Key.right:
                return 'right'
            elif key == keyboard.Key.space:
                return 'space'
            elif key == keyboard.Key.esc:
                return 'esc'
            else:
                return None
    
    def _on_press(self, key):
        """Handle key press event."""
        key_str = self._normalize_key(key)
        if key_str is None:
            return
        
        with self._lock:
            # Handle special keys
            if key_str == 'space':
                self._episode_toggle = True
                if self._on_episode_toggle:
                    self._on_episode_toggle()
            elif key_str == 'r':
                self._reset_requested = True
                if self._on_reset:
                    self._on_reset()
            elif key_str == 'esc':
                self._quit_requested = True
                if self._on_quit:
                    self._on_quit()
            else:
                # Regular movement keys
                self._pressed_keys.add(key_str)
    
    def _on_release(self, key):
        """Handle key release event."""
        key_str = self._normalize_key(key)
        if key_str is None:
            return
        
        with self._lock:
            self._pressed_keys.discard(key_str)
    
    def get_action_delta(self) -> np.ndarray:
        """
        Get current joint position deltas based on pressed keys.
        
        Returns:
            Array of shape (6,) with joint position deltas in radians
        """
        delta = np.zeros(6, dtype=np.float32)
        
        with self._lock:
            for key in self._pressed_keys:
                if key in self._key_mappings:
                    joint_idx, multiplier = self._key_mappings[key]
                    
                    # Use different scale for gripper
                    if joint_idx == self.GRIPPER:
                        scale = self.gripper_scale
                    else:
                        scale = self.delta_scale
                    
                    delta[joint_idx] += multiplier * scale
        
        return delta
    
    def check_episode_toggle(self) -> bool:
        """Check and clear episode toggle flag."""
        with self._lock:
            result = self._episode_toggle
            self._episode_toggle = False
            return result
    
    def check_reset_requested(self) -> bool:
        """Check and clear reset flag."""
        with self._lock:
            result = self._reset_requested
            self._reset_requested = False
            return result
    
    def check_quit_requested(self) -> bool:
        """Check quit flag (doesn't clear it)."""
        with self._lock:
            return self._quit_requested
    
    def set_callbacks(
        self,
        on_episode_toggle: Optional[Callable] = None,
        on_reset: Optional[Callable] = None,
        on_quit: Optional[Callable] = None
    ):
        """Set callback functions for special keys."""
        self._on_episode_toggle = on_episode_toggle
        self._on_reset = on_reset
        self._on_quit = on_quit


def test_keyboard_teleop():
    """Test the keyboard teleop controller."""
    import time
    
    teleop = KeyboardTeleop()
    teleop.start()
    
    print("\nTest mode: Press keys to see deltas. Press ESC to quit.\n")
    
    try:
        while not teleop.check_quit_requested():
            delta = teleop.get_action_delta()
            
            # Only print if there's movement
            if np.any(delta != 0):
                print(f"Delta: {delta}")
            
            if teleop.check_episode_toggle():
                print(">>> Episode toggle!")
            
            if teleop.check_reset_requested():
                print(">>> Reset requested!")
            
            time.sleep(0.1)
    finally:
        teleop.stop()
        print("\nKeyboard teleop stopped.")


if __name__ == "__main__":
    test_keyboard_teleop()

