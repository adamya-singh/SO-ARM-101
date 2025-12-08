"""
Gamepad Teleoperation Handler for SO-101 Robot

Uses pygame in a separate process to read DualShock 4 controller input,
communicating with the main process via shared memory to avoid conflicts
with MuJoCo's main thread requirements.

Controller Mapping (DualShock 4):
    Left Stick:
        X axis → Shoulder pan (left/right)
        Y axis → Shoulder lift (up/down)
    
    Right Stick:
        X axis → Wrist roll
        Y axis → Wrist flex
    
    Bumpers:
        L1 → Elbow retract
        R1 → Elbow extend
    
    Triggers:
        L2 → Close gripper (analog)
        R2 → Open gripper (analog)
    
    Buttons:
        X (Cross) → Toggle episode recording
        Circle → Reset environment
        Options → Quit
"""

import numpy as np
import multiprocessing
from multiprocessing import Process, Array, Value
import time
from typing import Optional, Callable


def _pygame_input_loop(
    axes: Array,
    buttons: Array,
    running: Value,
    dead_zone: float,
):
    """
    Pygame input loop running in a separate process.
    
    Reads controller input and updates shared memory arrays.
    
    Args:
        axes: Shared array for axis values [lx, ly, rx, ry, l2, r2]
        buttons: Shared array for button states [episode_toggle, reset, quit, l1, r1]
        running: Shared value to signal when to stop
        dead_zone: Minimum axis value to register (ignore drift)
    """
    import pygame
    
    pygame.init()
    pygame.joystick.init()
    
    # Wait for controller
    while pygame.joystick.get_count() == 0 and running.value:
        print("Waiting for controller...")
        time.sleep(1.0)
        pygame.joystick.quit()
        pygame.joystick.init()
    
    if not running.value:
        return
    
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    print(f"Controller connected: {joystick.get_name()}")
    print(f"  Axes: {joystick.get_numaxes()}")
    print(f"  Buttons: {joystick.get_numbuttons()}")
    
    # DualShock 4 axis mapping (may vary by OS/driver)
    # Typical mapping:
    #   Axis 0: Left stick X
    #   Axis 1: Left stick Y
    #   Axis 2: Right stick X (or L2 on some systems)
    #   Axis 3: Right stick Y (or R2 on some systems)
    #   Axis 4: L2 trigger
    #   Axis 5: R2 trigger
    
    # DualShock 4 button mapping on macOS (discovered via testing):
    #   Button 0: Cross (X)
    #   Button 1: Circle
    #   Button 2: Square
    #   Button 3: Triangle
    #   Button 4: Share
    #   Button 6: Options
    #   Button 9: L1
    #   Button 10: R1
    
    # Track previous button states for edge detection
    prev_cross = False
    prev_circle = False
    prev_options = False
    
    try:
        while running.value:
            pygame.event.pump()
            
            # Read axes with dead zone
            def apply_dead_zone(val):
                if abs(val) < dead_zone:
                    return 0.0
                return val
            
            # Read stick axes
            num_axes = joystick.get_numaxes()
            
            # Left stick
            if num_axes > 0:
                axes[0] = apply_dead_zone(joystick.get_axis(0))  # Left X
            if num_axes > 1:
                axes[1] = apply_dead_zone(joystick.get_axis(1))  # Left Y
            
            # Right stick
            if num_axes > 2:
                axes[2] = apply_dead_zone(joystick.get_axis(2))  # Right X
            if num_axes > 3:
                axes[3] = apply_dead_zone(joystick.get_axis(3))  # Right Y
            
            # Triggers (L2/R2) - may be axes 4,5 or 2,5 depending on driver
            if num_axes > 4:
                # Triggers typically range from -1 (released) to 1 (pressed)
                # Normalize to 0-1 range
                l2_raw = joystick.get_axis(4)
                axes[4] = max(0, (l2_raw + 1) / 2)  # L2: 0 to 1
            if num_axes > 5:
                r2_raw = joystick.get_axis(5)
                axes[5] = max(0, (r2_raw + 1) / 2)  # R2: 0 to 1
            
            # Read buttons with edge detection (only trigger on press, not hold)
            num_buttons = joystick.get_numbuttons()
            
            # Cross button (X) - episode toggle
            if num_buttons > 0:
                cross = joystick.get_button(0)
                if cross and not prev_cross:
                    buttons[0] = 1  # Signal episode toggle
                prev_cross = cross
            
            # Circle button - reset
            if num_buttons > 1:
                circle = joystick.get_button(1)
                if circle and not prev_circle:
                    buttons[1] = 1  # Signal reset
                prev_circle = circle
            
            # Options button - quit (button 6 on macOS)
            if num_buttons > 6:
                options = joystick.get_button(6)
                if options and not prev_options:
                    buttons[2] = 1  # Signal quit
                prev_options = options
            
            # L1/R1 buttons (held, not edge-detected) - buttons 9/10 on macOS
            if num_buttons > 9:
                buttons[3] = joystick.get_button(9)  # L1
            if num_buttons > 10:
                buttons[4] = joystick.get_button(10)  # R1
            
            # Small delay to prevent busy-waiting
            time.sleep(0.01)  # 100 Hz polling
            
    except Exception as e:
        print(f"Pygame input loop error: {e}")
    finally:
        pygame.quit()


class GamepadTeleop:
    """
    Gamepad teleoperation controller for SO-101 robot.
    
    Runs pygame in a separate process and communicates via shared memory
    to avoid conflicts with MuJoCo's main thread requirements.
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
        dead_zone: float = 0.15,
        stick_scale: float = 0.06,
        trigger_scale: float = 0.08,
        button_scale: float = 0.04,
    ):
        """
        Initialize gamepad teleop controller.
        
        Args:
            dead_zone: Minimum stick value to register (0-1)
            stick_scale: Scale factor for stick-controlled joints (radians per step)
            trigger_scale: Scale factor for trigger-controlled gripper
            button_scale: Scale factor for bumper-controlled elbow
        """
        self.dead_zone = dead_zone
        self.stick_scale = stick_scale
        self.trigger_scale = trigger_scale
        self.button_scale = button_scale
        
        # Shared memory for axes: [left_x, left_y, right_x, right_y, l2, r2]
        self._axes = Array('f', 6)
        
        # Shared memory for buttons: [episode_toggle, reset, quit, l1, r1]
        self._buttons = Array('i', 5)
        
        # Running flag
        self._running = Value('i', 1)
        
        # Input process
        self._process: Optional[Process] = None
    
    def start(self):
        """Start the gamepad input process."""
        if self._process is not None and self._process.is_alive():
            return
        
        self._running.value = 1
        
        # Reset shared memory
        for i in range(6):
            self._axes[i] = 0.0
        for i in range(5):
            self._buttons[i] = 0
        
        # Start input process
        self._process = Process(
            target=_pygame_input_loop,
            args=(self._axes, self._buttons, self._running, self.dead_zone),
            daemon=True,
        )
        self._process.start()
        
        print("\nGamepad teleop started. Controls:")
        print("  Left Stick: Shoulder pan (X) / lift (Y)")
        print("  Right Stick: Wrist roll (X) / flex (Y)")
        print("  L1/R1: Elbow retract/extend")
        print("  L2/R2: Gripper close/open")
        print("  X (Cross): Start/stop episode recording")
        print("  Circle: Reset environment")
        print("  Options: Quit")
        
        # Give process time to initialize
        time.sleep(0.5)
    
    def stop(self):
        """Stop the gamepad input process."""
        self._running.value = 0
        
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None
    
    def get_action_delta(self) -> np.ndarray:
        """
        Get current joint position deltas based on controller input.
        
        Returns:
            Array of shape (6,) with joint position deltas in radians
        """
        delta = np.zeros(6, dtype=np.float32)
        
        # Read axes from shared memory
        left_x = self._axes[0]
        left_y = self._axes[1]
        right_x = self._axes[2]
        right_y = self._axes[3]
        l2 = self._axes[4]
        r2 = self._axes[5]
        
        # Read bumpers
        l1 = self._buttons[3]
        r1 = self._buttons[4]
        
        # Map to joint deltas with exponential scaling for fine control
        def scale_input(val, scale):
            # Apply exponential curve for finer control near center
            sign = np.sign(val)
            magnitude = abs(val) ** 1.5  # Exponential curve
            return sign * magnitude * scale
        
        # Left stick → Shoulder
        delta[self.SHOULDER_PAN] = scale_input(left_x, self.stick_scale)
        delta[self.SHOULDER_LIFT] = scale_input(left_y, self.stick_scale)
        
        # Right stick → Wrist
        delta[self.WRIST_ROLL] = scale_input(right_x, self.stick_scale)
        delta[self.WRIST_FLEX] = scale_input(right_y, self.stick_scale)
        
        # Bumpers → Elbow
        if l1:
            delta[self.ELBOW_FLEX] -= self.button_scale  # Retract
        if r1:
            delta[self.ELBOW_FLEX] += self.button_scale  # Extend
        
        # Triggers → Gripper
        # L2 closes, R2 opens
        gripper_delta = (r2 - l2) * self.trigger_scale
        delta[self.GRIPPER] = gripper_delta
        
        return delta
    
    def check_episode_toggle(self) -> bool:
        """Check and clear episode toggle flag."""
        if self._buttons[0]:
            self._buttons[0] = 0
            return True
        return False
    
    def check_reset_requested(self) -> bool:
        """Check and clear reset flag."""
        if self._buttons[1]:
            self._buttons[1] = 0
            return True
        return False
    
    def check_quit_requested(self) -> bool:
        """Check quit flag."""
        return bool(self._buttons[2])
    
    def set_callbacks(
        self,
        on_episode_toggle: Optional[Callable] = None,
        on_reset: Optional[Callable] = None,
        on_quit: Optional[Callable] = None
    ):
        """Set callback functions (for API compatibility with keyboard teleop)."""
        # Callbacks not used in gamepad implementation (polling instead)
        pass


def test_gamepad_teleop():
    """Test the gamepad teleop controller."""
    print("="*60)
    print("Gamepad Teleop Test")
    print("="*60)
    print("\nConnect your DualShock 4 controller via Bluetooth.")
    print("Press Options to quit.\n")
    
    teleop = GamepadTeleop()
    teleop.start()
    
    try:
        while not teleop.check_quit_requested():
            delta = teleop.get_action_delta()
            
            # Print delta if there's significant movement
            if np.any(np.abs(delta) > 0.001):
                print(f"Delta: [{', '.join(f'{d:+.4f}' for d in delta)}]")
            
            if teleop.check_episode_toggle():
                print(">>> Episode toggle (X pressed)")
            
            if teleop.check_reset_requested():
                print(">>> Reset requested (Circle pressed)")
            
            time.sleep(0.05)  # 20 Hz display
            
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        teleop.stop()
        print("Gamepad teleop stopped.")


if __name__ == "__main__":
    test_gamepad_teleop()

