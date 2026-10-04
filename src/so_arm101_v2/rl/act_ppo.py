"""Gaussian chunk policy over ACT, ported from simulation_code/train_act_in_sim.py ``ACTGaussianPPOPolicy``.

The action is the ACT chunk (latent zero, eval mode: no dropout) plus independent Gaussian noise with a
learnable per-joint standard deviation, in the quantile-normalised residual units ACT was fine-tuned in.
The log-probability sums the 90 x 6 executed dimensions. The legacy policy started at log std -2 in its own
normalisation; here the start is a probe result (the widest noise whose rollouts keep zero safety frames on
the monitoring suite), and the std is clamped to [std_min, std_max].
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


class GaussianChunkPolicy:
    def __init__(self, model: Any, *, std_init: float, std_min: float = 1e-3, std_max: float = 0.5, micro: int = 64) -> None:
        import torch
        if model.kind != "act":
            raise ValueError("GaussianChunkPolicy wraps an ACT model")
        self.model, self.micro = model, int(micro)
        self.device = model.device
        self.std_min, self.std_max = float(std_min), float(std_max)
        model.eval()
        self.heads = model.set_scope("rl_heads")
        self.log_std = torch.nn.Parameter(torch.full((6,), math.log(float(std_init)), device=self.device))
        self.torch = torch

    def named_parameters(self) -> list[tuple[str, Any]]:
        return list(self.heads) + [("log_std", self.log_std)]

    def std(self) -> Any:
        return self.log_std.exp().clamp(self.std_min, self.std_max)

    def act(self, frames: np.ndarray, states: np.ndarray, *, deterministic: bool = False, seed: int | None = None) -> dict[str, Any]:
        torch = self.torch
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.no_grad():
            mean = self.model.mean(frames, states).float()
            if deterministic:
                action = mean
                logp = torch.zeros(mean.shape[0], device=self.device)
            else:
                std = self.std()
                eps = torch.randn(mean.shape, generator=generator, device=self.device)
                action = mean + std * eps
                logp = (-(eps ** 2) / 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(dim=(1, 2))
            residual = self.model.stats.unnorm_action(action).cpu().numpy()
            action_np = action.cpu().numpy()
        return dict(residual=residual, records=[dict(action=action_np[i]) for i in range(action_np.shape[0])], logp=logp.cpu().numpy(),
                    sigma_mean=float(self.std().mean()))

    def logprob(self, frames: np.ndarray, states: np.ndarray, records: list[dict[str, Any]]) -> tuple[Any, dict[str, float]]:
        torch = self.torch
        std = self.std()
        outs = []
        for start in range(0, len(records), self.micro):
            sl = slice(start, start + self.micro)
            mean = self.model.mean(frames[sl], states[sl]).float()
            action = torch.as_tensor(np.stack([r["action"] for r in records[sl]]), dtype=torch.float32, device=self.device)
            outs.append((-((action - mean) ** 2) / (2 * std ** 2) - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(dim=(1, 2)))
        s = std.detach()
        return torch.cat(outs), dict(sigma_mean=float(s.mean()), sigma_min=float(s.min()), sigma_max=float(s.max()))

    def state_dict(self) -> dict[str, Any]:
        return dict(log_std=self.log_std.detach().cpu(), std_min=self.std_min, std_max=self.std_max)

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        with self.torch.no_grad():
            self.log_std.copy_(payload["log_std"].to(self.device))


__all__ = ["GaussianChunkPolicy"]
