"""
SO-101 Gymnasium Environments Package

This package registers SO-101 robot arm environments with Gymnasium,
enabling integration with LeRobot for teleoperation data collection
and SmolVLA training.

Usage:
    import gymnasium
    import envs  # This registers the environments
    
    env = gymnasium.make("SO101PickPlace-v0")
"""

import gymnasium

# Import the environment class
import sys
import os

# Add parent directory to path to import so101_gym_env
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from so101_gym_env import SO101PickPlaceEnv

# Register environments
gymnasium.register(
    id="SO101PickPlace-v0",
    entry_point="so101_gym_env:SO101PickPlaceEnv",
    max_episode_steps=500,
)

# Also register with different configurations
gymnasium.register(
    id="SO101PickPlaceNoRandom-v0",
    entry_point="so101_gym_env:SO101PickPlaceEnv",
    max_episode_steps=500,
    kwargs={"randomize_block": False},
)

gymnasium.register(
    id="SO101PickPlaceHuman-v0",
    entry_point="so101_gym_env:SO101PickPlaceEnv",
    max_episode_steps=1000,  # Longer episodes for teleoperation
    kwargs={"render_mode": "human", "randomize_block": False},
)

__all__ = ["SO101PickPlaceEnv"]

