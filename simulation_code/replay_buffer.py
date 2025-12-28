"""
ReinFlow Replay Buffer for Off-Policy Training

Stores denoising trajectories, sigmas, observations, and rewards for
off-policy ReinFlow training as specified in the paper.

Each entry contains:
- trajectory: List of K+1 tensors [a^0, a^1, ..., a^K] (intermediate denoised actions)
- sigmas: List of K tensors (sigma used at each denoising step)
- observation: Dict of observation tensors
- reward: Scalar reward for the action chunk
- done: Boolean done flag

Reference: https://reinflow.github.io/
"""

import torch
import numpy as np
from collections import deque
import random
from typing import Dict, List, Tuple, Any


class ReinFlowReplayBuffer:
    """
    Replay buffer storing denoising trajectories for off-policy ReinFlow training.
    
    This enables:
    1. Off-policy learning (sample from buffer, not just current rollout)
    2. Multiple gradient updates per environment step
    3. Better sample efficiency
    
    Args:
        capacity: Maximum number of transitions to store
    """
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        trajectory: List[torch.Tensor],
        sigmas: List[torch.Tensor],
        observation: Dict[str, torch.Tensor],
        reward: float,
        done: bool,
    ):
        """
        Store a transition in the buffer.
        
        Args:
            trajectory: List of K+1 tensors [a^0, a^1, ..., a^K]
            sigmas: List of K tensors (sigma used at each step)
            observation: Dict of observation tensors
            reward: Scalar reward for the chunk
            done: Boolean done flag
        """
        # Detach and move to CPU for storage efficiency
        traj_cpu = [a.detach().cpu() for a in trajectory]
        sigmas_cpu = [s.detach().cpu() for s in sigmas]
        obs_cpu = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v 
                   for k, v in observation.items()}
        
        self.buffer.append({
            'trajectory': traj_cpu,
            'sigmas': sigmas_cpu,
            'observation': obs_cpu,
            'reward': float(reward),
            'done': bool(done),
        })
    
    def sample(self, batch_size: int, device: str = 'cpu') -> Tuple[
        List[List[torch.Tensor]],  # trajectories
        List[List[torch.Tensor]],  # sigmas
        Dict[str, torch.Tensor],   # observations
        torch.Tensor,              # rewards
        torch.Tensor,              # dones
    ]:
        """
        Sample a batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Device to move tensors to
        
        Returns:
            trajectories: List of B trajectories, each is list of K+1 tensors
            sigmas: List of B sigma lists, each is list of K tensors
            observations: Dict of batched observation tensors
            rewards: (B,) tensor of rewards
            dones: (B,) tensor of done flags
        """
        actual_batch_size = min(batch_size, len(self.buffer))
        batch = random.sample(list(self.buffer), actual_batch_size)
        
        # Collate trajectories and sigmas (keep as list of lists for flexibility)
        trajectories = [[t.to(device) for t in item['trajectory']] for item in batch]
        sigmas = [[s.to(device) for s in item['sigmas']] for item in batch]
        
        # Batch rewards and dones
        rewards = torch.tensor([item['reward'] for item in batch], 
                               device=device, dtype=torch.float32)
        dones = torch.tensor([item['done'] for item in batch], 
                             device=device, dtype=torch.bool)
        
        # Batch observations - stack along batch dimension
        observations = {}
        for key in batch[0]['observation']:
            tensors = [item['observation'][key] for item in batch]
            if isinstance(tensors[0], torch.Tensor):
                observations[key] = torch.stack(tensors).to(device)
            else:
                # Handle non-tensor observations (shouldn't happen but be safe)
                observations[key] = tensors
        
        return trajectories, sigmas, observations, rewards, dones
    
    def sample_recent(self, batch_size: int, device: str = 'cpu') -> Tuple[
        List[List[torch.Tensor]],
        List[List[torch.Tensor]],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Sample from the most recent transitions (for on-policy bias).
        
        Useful for warm-starting or when you want more on-policy samples.
        """
        actual_batch_size = min(batch_size, len(self.buffer))
        # Take from the end (most recent)
        batch = list(self.buffer)[-actual_batch_size:]
        
        trajectories = [[t.to(device) for t in item['trajectory']] for item in batch]
        sigmas = [[s.to(device) for s in item['sigmas']] for item in batch]
        rewards = torch.tensor([item['reward'] for item in batch], 
                               device=device, dtype=torch.float32)
        dones = torch.tensor([item['done'] for item in batch], 
                             device=device, dtype=torch.bool)
        
        observations = {}
        for key in batch[0]['observation']:
            tensors = [item['observation'][key] for item in batch]
            if isinstance(tensors[0], torch.Tensor):
                observations[key] = torch.stack(tensors).to(device)
            else:
                observations[key] = tensors
        
        return trajectories, sigmas, observations, rewards, dones
    
    def clear(self):
        """Clear all transitions from the buffer."""
        self.buffer.clear()
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples for training."""
        return len(self.buffer) >= min_size
    
    def get_stats(self) -> Dict[str, Any]:
        """Get buffer statistics for logging."""
        if len(self.buffer) == 0:
            return {
                'size': 0,
                'capacity': self.capacity,
                'utilization': 0.0,
            }
        
        rewards = [item['reward'] for item in self.buffer]
        return {
            'size': len(self.buffer),
            'capacity': self.capacity,
            'utilization': len(self.buffer) / self.capacity,
            'reward_mean': np.mean(rewards),
            'reward_std': np.std(rewards),
            'reward_min': np.min(rewards),
            'reward_max': np.max(rewards),
        }

