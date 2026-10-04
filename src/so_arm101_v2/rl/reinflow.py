"""ReinFlow over a flow-matching policy (pi0.5 or SmolVLA), ported from simulation_code/reinflow_smolvla.py.

A deterministic flow policy integrates its velocity field from noise in K Euler steps. ReinFlow makes each
step a Gaussian transition: x_{k+1} = x_k + dt * v(x_k, t_k) + sigma(x_k, t_k) * eps, with sigma from a small
noise head on the expert's features, bounded by tanh to [sigma_min, sigma_max]. The whole chain then has an
exact log-probability (a sum of K Gaussians), which is what PPO needs (Zhang et al. 2025, "ReinFlow").

What carries over from the legacy stack (notes/*-fix.md):
- one sampler for rollouts and for log-probs (reinflow-inference-sampler-fix);
- sigma bounds scaled for a large action chunk (sigma-scaling-bug-fix: the paper's 0.08-0.16 was for small
  action spaces; the legacy SmolVLA run used 0.25-0.50 on 50 x 6). Here the bounds are a probe result: the
  widest range whose noisy rollouts keep zero safety frames on the monitoring suite;
- the base model in eval mode (kl-divergence-bug-fix);
- only stable heads train (README legacy status: ``rl_stable_heads``), plus the noise head.
Noise enters the 6 real action channels only; the padded channels follow the deterministic step, so a
recomputation from the stored chain reproduces the same velocities.

Evaluation uses the model's plain Euler sampler with the same K (noise off), which is what the fine-tune
baseline is measured with.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import numpy as np

from so_arm101_v2.learning.pi05 import ACTION_DIM


def make_noise_head(feature_dim: int):
    import torch

    class NoiseHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = torch.nn.Sequential(torch.nn.Linear(feature_dim, 256), torch.nn.SiLU(), torch.nn.Linear(256, 128), torch.nn.SiLU(),
                                           torch.nn.Linear(128, ACTION_DIM))
            torch.nn.init.zeros_(self.net[-1].weight)
            torch.nn.init.zeros_(self.net[-1].bias)    # sigma starts at the middle of its range

        def forward(self, features):
            return self.net(features)

    return NoiseHead()


class ReinFlowPolicy:
    def __init__(self, model: Any, *, steps: int, sigma_min: float, sigma_max: float, micro: int = 8, amp: bool = False) -> None:
        import torch
        if not model.flow:
            raise ValueError("ReinFlow needs a flow-matching model (pi0.5 or SmolVLA)")
        if not 0.0 < sigma_min < sigma_max:
            raise ValueError("need 0 < sigma_min < sigma_max")
        self.model, self.steps, self.sigma_min, self.sigma_max = model, int(steps), float(sigma_min), float(sigma_max)
        self.micro, self.amp = int(micro), bool(amp)
        self.device = model.device
        model.eval()                                   # dropout off for good (the KL fix)
        self.heads = model.set_scope("rl_heads")
        feature_dim = int(model.out_proj.in_features)
        self.noise_head = make_noise_head(feature_dim).to(self.device)
        self.torch = torch

    # ------------------------------------------------------------------ parameters
    def named_parameters(self) -> list[tuple[str, Any]]:
        return list(self.heads) + [(f"noise_head.{n}", p) for n, p in self.noise_head.named_parameters()]

    def _autocast(self):
        return self.torch.autocast("cuda", dtype=self.torch.bfloat16) if self.amp else contextlib.nullcontext()

    def sigma(self, features: Any) -> Any:
        raw = self.noise_head(features.float())
        return self.sigma_min + (self.sigma_max - self.sigma_min) * (self.torch.tanh(raw) + 1.0) / 2.0

    # ------------------------------------------------------------------ chain
    def _chain_logprob(self, ctx: dict[str, Any], chain: Any) -> tuple[Any, Any]:
        """log p of a stored chain (B, K+1, H, W) under the current parameters, and the mean sigma per sample."""
        torch = self.torch
        bsize = chain.shape[0]
        dt = -1.0 / self.steps
        logp = torch.zeros(bsize, device=self.device, dtype=torch.float32)
        sigma_mean = torch.zeros(bsize, device=self.device, dtype=torch.float32)
        for k in range(self.steps):
            t = torch.full((bsize,), 1.0 + k * dt, dtype=torch.float32, device=self.device)
            x_k = chain[:, k]
            with self._autocast():
                v, features = self.model.velocity(ctx, x_k, t)
            mu = (x_k + dt * v.float())[..., :ACTION_DIM]
            sigma = self.sigma(features)
            target = chain[:, k + 1, :, :ACTION_DIM]
            logp = logp + (-((target - mu) ** 2) / (2 * sigma ** 2) - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(dim=(1, 2))
            sigma_mean = sigma_mean + sigma.mean(dim=(1, 2)) / self.steps
        return logp, sigma_mean

    def act(self, frames: np.ndarray, states: np.ndarray, *, deterministic: bool = False, seed: int | None = None) -> dict[str, Any]:
        """Sample a chunk for each environment: residuals (B, H, 6) in normalised-act units, chains, log-probs."""
        torch = self.torch
        generator = None if seed is None else torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.no_grad():
            with self._autocast():
                ctx = self.model.prefix(frames, states)
            bsize = ctx["bsize"]
            width = self.model.action_width
            from so_arm101_v2.learning.vla_models import HORIZON
            x = torch.randn((bsize, HORIZON, width), generator=generator, device=self.device, dtype=torch.float32)
            chain = [x]
            logp = torch.zeros(bsize, device=self.device)
            sigmas = []
            dt = -1.0 / self.steps
            for k in range(self.steps):
                t = torch.full((bsize,), 1.0 + k * dt, dtype=torch.float32, device=self.device)
                with self._autocast():
                    v, features = self.model.velocity(ctx, x, t)
                mean = x + dt * v.float()
                if deterministic:
                    x = mean
                else:
                    sigma = self.sigma(features)
                    eps = torch.randn((bsize, HORIZON, ACTION_DIM), generator=generator, device=self.device)
                    x = mean.clone()
                    x[..., :ACTION_DIM] = mean[..., :ACTION_DIM] + sigma * eps
                    logp = logp + (-(eps ** 2) / 2 - torch.log(sigma) - 0.5 * math.log(2 * math.pi)).sum(dim=(1, 2))
                    sigmas.append(sigma.mean().item())
                chain.append(x)
            residual = self.model.stats.unnorm_action(x[..., :ACTION_DIM]).cpu().numpy()
            chain_np = torch.stack(chain, dim=1).cpu().numpy()
        return dict(residual=residual, records=[dict(chain=chain_np[i]) for i in range(bsize)], logp=logp.cpu().numpy(),
                    sigma_mean=float(np.mean(sigmas)) if sigmas else 0.0)

    def logprob(self, frames: np.ndarray, states: np.ndarray, records: list[dict[str, Any]]) -> tuple[Any, dict[str, float]]:
        """(B,) log-probs with grad, microbatched (the deterministic path PPO uses for both old and new)."""
        torch = self.torch
        outs, sig = [], []
        for start in range(0, len(records), self.micro):
            sl = slice(start, start + self.micro)
            with torch.no_grad(), self._autocast():
                ctx = self.model.prefix(frames[sl], states[sl])
            chain = torch.as_tensor(np.stack([r["chain"] for r in records[sl]]), dtype=torch.float32, device=self.device)
            logp, sigma_mean = self._chain_logprob(ctx, chain)
            outs.append(logp); sig.append(sigma_mean.detach())
        sigma_all = torch.cat(sig)
        return torch.cat(outs), dict(sigma_mean=float(sigma_all.mean()), sigma_min=float(sigma_all.min()), sigma_max=float(sigma_all.max()))

    def state_dict(self) -> dict[str, Any]:
        return dict(noise_head=self.noise_head.state_dict(), steps=self.steps, sigma_min=self.sigma_min, sigma_max=self.sigma_max)

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.noise_head.load_state_dict(payload["noise_head"])


__all__ = ["ReinFlowPolicy", "make_noise_head"]
