"""Privileged value function: the critic sees simulator state the actor never does (asymmetric actor-critic).

Input is ``ChunkEnv.privileged()`` (joints and velocities, cube pose and velocities, contact flags, square
placement, progress, milestones): the legacy ACT-PPO ``PrivilegedCritic`` idea, on the v2 measurements.
Used only to train; a deployed policy never needs it.
"""

from __future__ import annotations


def make_critic(input_dim: int, hidden: int = 256):
    import torch

    class PrivilegedCritic(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = torch.nn.LayerNorm(input_dim)
            self.net = torch.nn.Sequential(torch.nn.Linear(input_dim, hidden), torch.nn.SiLU(), torch.nn.Linear(hidden, hidden),
                                           torch.nn.SiLU(), torch.nn.Linear(hidden, 1))

        def forward(self, x):
            return self.net(self.norm(x)).squeeze(-1)

    return PrivilegedCritic()


__all__ = ["make_critic"]
