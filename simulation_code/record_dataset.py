#!/usr/bin/env python3
"""
SO-101 Teleoperation Dataset Recording Script

Records teleoperation demonstrations in LeRobotDataset v3 format
for SmolVLA fine-tuning.

Usage:
    # With keyboard (default):
    python record_dataset.py \
        --output_dir ./datasets/so101_pickplace \
        --task "pick up the red block" \
        --num_episodes 50 \
        --fps 10

    # With gamepad (DualShock 4):
    python record_dataset.py \
        --input gamepad \
        --output_dir ./datasets/so101_pickplace \
        --task "pick up the red block" \
        --num_episodes 50

    # With physical SO-101 arm:
    mjpython record_dataset.py \
        --input physical \
        --port /dev/tty.usbmodem5A680096011 \
        --output_dir ./datasets/so101_pickplace \
        --task "pick up the red block" \
        --num_episodes 50

Keyboard Controls:
    Arrow keys: Move shoulder pan
    W/S: Shoulder lift up/down
    I/K: Elbow extend/retract
    Z/X: Wrist flex down/up
    A/D: Wrist roll left/right
    Q/E: Gripper open/close
    Space: Start/stop episode recording
    R: Reset environment
    ESC: Quit

Gamepad Controls (DualShock 4):
    Left Stick: Shoulder pan (X) / lift (Y)
    Right Stick: Wrist roll (X) / flex (Y)
    L1/R1: Elbow retract/extend
    L2/R2: Gripper close/open
    X (Cross): Start/stop episode recording
    Circle: Reset environment
    Options: Quit

Physical Arm Controls:
    Move the physical arm by hand - simulation mirrors movements
    Space: Start/stop episode recording
    R: Reset simulation
    ESC: Quit
"""

import argparse
import time
import sys
import os
import numpy as np

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from so101_gym_env import SO101PickPlaceEnv
from teleop_keyboard import KeyboardTeleop
from teleop_gamepad import GamepadTeleop
from teleop_physical_arm import PhysicalArmTeleop
from lerobot_dataset_writer import LeRobotDatasetWriter


class TeleopRecorder:
    """
    Main class for teleoperation recording.
    
    Combines the gym environment, teleop controller (keyboard or gamepad),
    and dataset writer to record demonstrations.
    """
    
    def __init__(
        self,
        output_dir: str,
        task: str,
        num_episodes: int = 50,
        fps: int = 10,
        input_type: str = "keyboard",
        delta_scale: float = 0.05,
        max_episode_steps: int = 10000,
        randomize_block: bool = False,
        port: str = "/dev/tty.usbmodem5A680096011",
    ):
        """
        Initialize the teleoperation recorder.
        
        Args:
            output_dir: Directory to save the dataset
            task: Task description
            num_episodes: Target number of episodes to record
            fps: Recording frame rate
            input_type: "keyboard", "gamepad", or "physical"
            delta_scale: Scale for input-to-joint deltas
            max_episode_steps: Max steps before auto-truncation (default 10000 = ~16min at 10fps)
            randomize_block: Whether to randomize block position each episode
            port: USB port for physical arm (only used when input_type="physical")
        """
        self.output_dir = output_dir
        self.task = task
        self.num_episodes = num_episodes
        self.fps = fps
        self.frame_duration = 1.0 / fps
        self.input_type = input_type
        self.use_absolute_positions = (input_type == "physical")
        
        # Create environment with human rendering
        print("Initializing SO-101 environment...")
        self.env = SO101PickPlaceEnv(
            render_mode='human',
            randomize_block=randomize_block,
            max_episode_steps=max_episode_steps,  # Long episodes for teleoperation
        )
        
        # Create teleop controller based on input type
        if input_type == "gamepad":
            print("Initializing gamepad controller...")
            self.teleop = GamepadTeleop(
                stick_scale=delta_scale,
                trigger_scale=delta_scale * 1.5,
                button_scale=delta_scale * 0.8,
            )
        elif input_type == "physical":
            print(f"Initializing physical arm controller on {port}...")
            self.teleop = PhysicalArmTeleop(port=port)
        else:
            print("Initializing keyboard controller...")
            self.teleop = KeyboardTeleop(delta_scale=delta_scale)
        
        # Create dataset writer
        print(f"Initializing dataset writer at {output_dir}...")
        self.writer = LeRobotDatasetWriter(
            output_dir=output_dir,
            task_description=task,
            fps=fps,
        )
        
        # State tracking
        self.recording = False
        self.episode_start_time = 0.0
        self.current_action = None
        self.episodes_recorded = 0
        self.quit_requested = False
    
    def run(self):
        """Main recording loop."""
        print("\n" + "="*60)
        print("SO-101 Teleoperation Recording")
        print("="*60)
        print(f"Task: {self.task}")
        print(f"Target episodes: {self.num_episodes}")
        print(f"Recording FPS: {self.fps}")
        print(f"Input: {self.input_type}")
        print(f"Randomize block: {self.env.randomize_block}")
        print(f"Output: {self.output_dir}")
        print("="*60)
        
        if self.input_type == "gamepad":
            print("\nPress X (Cross) to start recording an episode.")
            print("Press Options to quit.\n")
        elif self.input_type == "physical":
            print("\nMove the physical arm - simulation will follow.")
            print("Press SPACE to start recording an episode.")
            print("Press ESC to quit.\n")
        else:
            print("\nPress SPACE to start recording an episode.")
            print("Press ESC to quit.\n")
        
        # Reset environment
        obs, info = self.env.reset()
        self.current_action = obs['observation.state'].copy()
        
        # Start keyboard listener
        self.teleop.start()
        
        try:
            while not self.quit_requested and self.episodes_recorded < self.num_episodes:
                loop_start = time.time()
                
                # Check for special key presses
                if self.teleop.check_quit_requested():
                    self.quit_requested = True
                    break
                
                if self.teleop.check_reset_requested():
                    self._handle_reset()
                
                if self.teleop.check_episode_toggle():
                    self._handle_episode_toggle()
                
                # Get action from teleop controller
                if self.use_absolute_positions:
                    # Physical arm provides absolute joint positions
                    self.current_action = np.clip(
                        self.teleop.get_joint_positions(),
                        self.env.joint_limits_low,
                        self.env.joint_limits_high
                    )
                else:
                    # Keyboard/gamepad provides delta positions
                    delta = self.teleop.get_action_delta()
                    self.current_action = np.clip(
                        self.current_action + delta,
                        self.env.joint_limits_low,
                        self.env.joint_limits_high
                    )
                
                # Step environment
                obs, reward, terminated, truncated, info = self.env.step(self.current_action)
                self.env.render()
                
                # Record frame if we're recording
                if self.recording:
                    timestamp = time.time() - self.episode_start_time
                    self.writer.add_frame(obs, self.current_action, timestamp)
                    
                    # Check for success
                    if info.get('success', False):
                        print("\n>>> Block lifted! Episode successful!")
                        self._end_episode(success=True)
                
                # Handle episode termination/truncation
                if terminated or truncated:
                    if self.recording:
                        self._end_episode(success=terminated)
                    obs, info = self.env.reset()
                    self.current_action = obs['observation.state'].copy()
                
                # Maintain frame rate
                elapsed = time.time() - loop_start
                if elapsed < self.frame_duration:
                    time.sleep(self.frame_duration - elapsed)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        
        finally:
            self._cleanup()
    
    def _handle_episode_toggle(self):
        """Handle space bar press to toggle recording."""
        if not self.recording:
            self._start_episode()
        else:
            self._end_episode(success=True)  # Assume success if manually ended
    
    def _start_episode(self):
        """Start recording a new episode."""
        # Reset environment for new episode
        obs, info = self.env.reset()
        self.current_action = obs['observation.state'].copy()
        
        # Start recording
        self.writer.start_episode()
        self.episode_start_time = time.time()
        self.recording = True
        
        print(f"\n>>> Recording episode {self.episodes_recorded + 1}/{self.num_episodes}...")
        print("    Press SPACE to stop recording.")
    
    def _end_episode(self, success: bool = True):
        """End the current episode."""
        if not self.recording:
            return
        
        num_frames = self.writer.end_episode(success=success)
        self.recording = False
        self.episodes_recorded += 1
        
        print(f"    Episode {self.episodes_recorded}/{self.num_episodes} saved ({num_frames} frames)")
        
        if self.episodes_recorded < self.num_episodes:
            print("\n>>> Press SPACE to start next episode.")
    
    def _handle_reset(self):
        """Handle reset request."""
        print("\n>>> Resetting environment...")
        
        # If recording, discard current episode
        if self.recording:
            print("    Discarding current episode.")
            self.recording = False
            # Don't call end_episode to avoid saving
            self.writer.current_frames = []
            for writer in self.writer._video_writers.values():
                writer.close()
            self.writer._video_writers = {}
        
        # Reset environment
        obs, info = self.env.reset()
        self.current_action = obs['observation.state'].copy()
        print("    Environment reset. Press SPACE to start recording.")
    
    def _cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        
        # End any in-progress recording
        if self.recording:
            self._end_episode(success=False)
        
        # Finalize dataset
        if self.episodes_recorded > 0:
            self.writer.finalize()
        
        # Stop keyboard listener
        self.teleop.stop()
        
        # Close environment
        self.env.close()
        
        print(f"\nRecording complete!")
        print(f"Episodes recorded: {self.episodes_recorded}")
        print(f"Dataset saved to: {self.output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Record teleoperation demonstrations for SO-101 robot"
    )
    parser.add_argument(
        "--input",
        type=str,
        choices=["keyboard", "gamepad", "physical"],
        default="keyboard",
        help="Input device: 'keyboard', 'gamepad' (DualShock 4), or 'physical' (SO-101 arm)"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/tty.usbmodem5A680096011",
        help="USB port for physical arm (only used with --input physical)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./datasets/so101_pickplace",
        help="Directory to save the dataset"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="pick up the red block",
        help="Task description"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=50,
        help="Number of episodes to record"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Recording frame rate"
    )
    parser.add_argument(
        "--delta_scale",
        type=float,
        default=0.05,
        help="Scale factor for joint deltas (sensitivity)"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=10000,
        help="Max steps per episode before auto-reset (default 10000 = ~16min at 10fps)"
    )
    parser.add_argument(
        "--randomize",
        action="store_true",
        help="Randomize block position each episode (default: fixed position)"
    )
    
    args = parser.parse_args()
    
    # Create recorder and run
    recorder = TeleopRecorder(
        output_dir=args.output_dir,
        task=args.task,
        num_episodes=args.num_episodes,
        fps=args.fps,
        input_type=args.input,
        delta_scale=args.delta_scale,
        max_episode_steps=args.max_steps,
        randomize_block=args.randomize,
        port=args.port,
    )
    recorder.run()


if __name__ == "__main__":
    main()

