#!/usr/bin/env python3
"""
Experimental online PPO fine-tuning for an ACT policy in SO-101 MuJoCo.

The policy is initialized from a supervised ACT checkpoint trained by
`train_act_on_data.py`. The actor sees only the same wrist camera and 6D state
schema as the physical dataset; the critic uses privileged MuJoCo state.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INIT_CHECKPOINT = SCRIPT_DIR / "outputs" / "train" / "act_so101_physical" / "checkpoints" / "last" / "pretrained_model"

np = None
torch = None
nn = None
F = None
wandb = None
SO101PickPlaceEnv = None


def load_training_dependencies() -> None:
    """Import heavy dependencies only after CLI parsing so `--help` is cheap."""
    global np, torch, nn, F, wandb, SO101PickPlaceEnv
    try:
        import numpy as _np
        import torch as _torch
        import torch.nn as _nn
        import torch.nn.functional as _F
    except ImportError as exc:
        raise RuntimeError(
            "ACT sim training requires the simulation/LeRobot Python environment. "
            "Install the requirements or activate the environment that contains numpy, torch, and mujoco."
        ) from exc

    from mujoco_rendering import setup_mujoco_rendering

    setup_mujoco_rendering()
    try:
        import wandb as _wandb
    except ImportError:  # pragma: no cover - optional dependency
        _wandb = None

    from so101_gym_env import SO101PickPlaceEnv as _SO101PickPlaceEnv

    np = _np
    torch = _torch
    nn = _nn
    F = _F
    wandb = _wandb
    SO101PickPlaceEnv = _SO101PickPlaceEnv


@dataclass
class RolloutBatch:
    observations: list[dict[str, torch.Tensor]]
    critic_obs: list[torch.Tensor]
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor


class PrivilegedCritic(nn.Module if nn is not None else object):
    """Small value network over joint state plus privileged MuJoCo task state."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class ACTGaussianPPOPolicy(nn.Module if nn is not None else object):
    """
    PPO wrapper around a LeRobot ACT policy.

    The wrapper requires a differentiable ACT forward path. If the installed
    LeRobot version only exposes no-grad inference, training fails explicitly.
    """

    def __init__(self, act_policy: nn.Module, action_dim: int, chunk_size: int, log_std_init: float):
        super().__init__()
        self.act_policy = act_policy
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

    def _extract_action_tensor(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("action", "actions", "action_pred", "pred_action", "pred_actions"):
                value = output.get(key)
                if isinstance(value, torch.Tensor):
                    return value
        if hasattr(output, "action") and isinstance(output.action, torch.Tensor):
            return output.action
        raise RuntimeError(
            "Could not find a differentiable action tensor in ACT forward output. "
            "Expected a tensor or dict key such as 'action', 'actions', or 'action_pred'."
        )

    def _predict_action_chunk_with_grad(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
        if not hasattr(self.act_policy, "model"):
            output = self.act_policy(observation)
            return self._extract_action_tensor(output)

        batch = dict(observation)
        image_features = getattr(getattr(self.act_policy, "config", None), "image_features", None)
        if image_features:
            batch["observation.images"] = [batch[key] for key in image_features]

        was_training = self.act_policy.model.training
        self.act_policy.model.eval()
        try:
            return self.act_policy.model(batch)[0]
        finally:
            self.act_policy.model.train(was_training)

    def mean_chunk(self, observation: dict[str, torch.Tensor], require_grad: bool = True) -> torch.Tensor:
        mean = self._predict_action_chunk_with_grad(observation)
        if require_grad and not mean.requires_grad:
            raise RuntimeError(
                "ACT action output does not require gradients. Use a LeRobot ACT version whose forward path "
                "returns differentiable action means for training."
            )

        if mean.ndim == 2:
            mean = mean.unsqueeze(1)
        if mean.ndim != 3:
            raise RuntimeError(f"Expected ACT action output shape (B, T, A) or (B, A), got {tuple(mean.shape)}")
        if mean.shape[-1] != self.action_dim:
            raise RuntimeError(f"Expected action_dim={self.action_dim}, got {mean.shape[-1]}")

        if mean.shape[1] < self.chunk_size:
            pad = mean[:, -1:, :].expand(-1, self.chunk_size - mean.shape[1], -1)
            mean = torch.cat([mean, pad], dim=1)
        return mean[:, : self.chunk_size, :]

    def distribution(self, observation: dict[str, torch.Tensor]) -> torch.distributions.Normal:
        mean = self.mean_chunk(observation)
        std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
        return torch.distributions.Normal(mean, std)

    def sample(self, observation: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        dist = self.distribution(observation)
        actions = dist.rsample()
        log_probs = dist.log_prob(actions).sum(dim=(1, 2))
        return actions, log_probs

    def log_prob(self, observation: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
        return self.distribution(observation).log_prob(actions).sum(dim=(1, 2))


def define_model_classes() -> None:
    """Bind torch-backed module classes after lazy dependency loading."""
    global PrivilegedCritic, ACTGaussianPPOPolicy

    class _PrivilegedCritic(nn.Module):
        """Small value network over joint state plus privileged MuJoCo task state."""

        def __init__(self, input_dim: int = 16, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    class _ACTGaussianPPOPolicy(nn.Module):
        """
        PPO wrapper around a LeRobot ACT policy.

        The wrapper requires a differentiable ACT forward path. If the installed
        LeRobot version only exposes no-grad inference, training fails explicitly.
        """

        def __init__(self, act_policy: nn.Module, action_dim: int, chunk_size: int, log_std_init: float):
            super().__init__()
            self.act_policy = act_policy
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))

        def _extract_action_tensor(self, output: Any) -> torch.Tensor:
            if isinstance(output, torch.Tensor):
                return output
            if isinstance(output, dict):
                for key in ("action", "actions", "action_pred", "pred_action", "pred_actions"):
                    value = output.get(key)
                    if isinstance(value, torch.Tensor):
                        return value
            if hasattr(output, "action") and isinstance(output.action, torch.Tensor):
                return output.action
            raise RuntimeError(
                "Could not find a differentiable action tensor in ACT forward output. "
                "Expected a tensor or dict key such as 'action', 'actions', or 'action_pred'."
            )

        def _predict_action_chunk_with_grad(self, observation: dict[str, torch.Tensor]) -> torch.Tensor:
            if not hasattr(self.act_policy, "model"):
                output = self.act_policy(observation)
                return self._extract_action_tensor(output)

            batch = dict(observation)
            image_features = getattr(getattr(self.act_policy, "config", None), "image_features", None)
            if image_features:
                batch["observation.images"] = [batch[key] for key in image_features]

            was_training = self.act_policy.model.training
            self.act_policy.model.eval()
            try:
                return self.act_policy.model(batch)[0]
            finally:
                self.act_policy.model.train(was_training)

        def mean_chunk(self, observation: dict[str, torch.Tensor], require_grad: bool = True) -> torch.Tensor:
            mean = self._predict_action_chunk_with_grad(observation)
            if require_grad and not mean.requires_grad:
                raise RuntimeError(
                    "ACT action output does not require gradients. Use a LeRobot ACT version whose forward path "
                    "returns differentiable action means for training."
                )

            if mean.ndim == 2:
                mean = mean.unsqueeze(1)
            if mean.ndim != 3:
                raise RuntimeError(f"Expected ACT action output shape (B, T, A) or (B, A), got {tuple(mean.shape)}")
            if mean.shape[-1] != self.action_dim:
                raise RuntimeError(f"Expected action_dim={self.action_dim}, got {mean.shape[-1]}")

            if mean.shape[1] < self.chunk_size:
                pad = mean[:, -1:, :].expand(-1, self.chunk_size - mean.shape[1], -1)
                mean = torch.cat([mean, pad], dim=1)
            return mean[:, : self.chunk_size, :]

        def distribution(self, observation: dict[str, torch.Tensor]) -> torch.distributions.Normal:
            mean = self.mean_chunk(observation)
            std = self.log_std.exp().view(1, 1, -1).expand_as(mean)
            return torch.distributions.Normal(mean, std)

        def sample(self, observation: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            dist = self.distribution(observation)
            actions = dist.rsample()
            log_probs = dist.log_prob(actions).sum(dim=(1, 2))
            return actions, log_probs

        def log_prob(self, observation: dict[str, torch.Tensor], actions: torch.Tensor) -> torch.Tensor:
            return self.distribution(observation).log_prob(actions).sum(dim=(1, 2))

    PrivilegedCritic = _PrivilegedCritic
    ACTGaussianPPOPolicy = _ACTGaussianPPOPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EXPERIMENTAL: PPO fine-tune ACT chunks in SO-101 MuJoCo. "
            "Use train_sim_baseline.py first to verify reward/action conventions."
        )
    )
    parser.add_argument(
        "--experimental-act-ppo",
        action="store_true",
        help="Required safety opt-in for the old high-dimensional ACT-chunk PPO path.",
    )
    parser.add_argument("--init-checkpoint", type=Path, default=DEFAULT_INIT_CHECKPOINT)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps-per-episode", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--steps-per-action", type=int, default=1)
    parser.add_argument("--policy-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=8)
    parser.add_argument("--randomize-block-reset", action="store_true")
    parser.add_argument(
        "--curriculum-fixed-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use easier fixed block pose by default; pass --no-curriculum-fixed-block for the old farther pose.",
    )
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--checkpoint-path", type=Path, default=SCRIPT_DIR / "act_sim_ppo_checkpoint.pt")
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser.parse_args()


def load_act_policy(checkpoint: Path, device: torch.device) -> nn.Module:
    try:
        try:
            from lerobot.policies.act.policy_act import ACTPolicy
        except ImportError:
            from lerobot.policies.act.modeling_act import ACTPolicy
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed, so ACTPolicy cannot be loaded.\n"
            "Run `train_act_on_data.py --dry-run` to validate data, then install LeRobot before training."
        ) from exc

    checkpoint = checkpoint.resolve()
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"ACT checkpoint not found: {checkpoint}\n"
            "Run train_act_on_data.py first or pass --init-checkpoint."
        )
    policy = ACTPolicy.from_pretrained(str(checkpoint))
    policy.to(device)
    policy.train()
    return policy


def adapt_sim_observation(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Map SO101PickPlaceEnv output to the physical ACT dataset schema."""
    wrist = obs["observation.images.camera2"]
    if wrist.dtype != np.float32:
        wrist_tensor = torch.from_numpy(wrist).float() / 255.0
    else:
        wrist_tensor = torch.from_numpy(wrist).float()
    wrist_tensor = wrist_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    state_tensor = torch.from_numpy(obs["observation.state"].astype(np.float32)).unsqueeze(0).to(device)
    return {
        "observation.images.wrist": wrist_tensor,
        "observation.state": state_tensor,
    }


def privileged_critic_obs(obs: dict[str, np.ndarray], info: dict, device: torch.device) -> torch.Tensor:
    state = obs["observation.state"].astype(np.float32)
    gripper = np.asarray(info["gripper_pos"], dtype=np.float32)
    block = np.asarray(info["block_pos"], dtype=np.float32)
    features = np.concatenate(
        [
            state,
            gripper,
            block,
            np.asarray(
                [
                    info["distance_to_block"],
                    info["block_height"],
                    float(info.get("success", False)),
                    float(info.get("step_count", 0)) / 500.0,
                ],
                dtype=np.float32,
            ),
        ]
    )
    return torch.from_numpy(features).to(device)


def step_action(env: SO101PickPlaceEnv, action: np.ndarray, repeats: int) -> tuple[dict, float, bool, bool, dict]:
    total_reward = 0.0
    last_obs = None
    last_info = None
    terminated = False
    truncated = False
    for _ in range(repeats):
        last_obs, reward, terminated, truncated, last_info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    return last_obs, total_reward, terminated, truncated, last_info


def compute_returns_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.tensor(0.0, device=rewards.device)
    next_value = torch.tensor(0.0, device=rewards.device)
    for t in reversed(range(rewards.numel())):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantages[t] = last_gae
        next_value = values[t]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return returns, advantages


def collect_rollout(
    env: SO101PickPlaceEnv,
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[RolloutBatch, dict]:
    obs, info = env.reset()
    observations = []
    critic_observations = []
    actions = []
    log_probs = []
    rewards = []
    dones = []
    values = []
    metrics = {
        "episode_return": 0.0,
        "success": 0.0,
        "contact_steps": 0,
        "grasp_steps": 0,
        "lift_steps": 0,
        "steps": 0,
    }

    for _ in range(args.max_steps_per_episode):
        policy_obs = adapt_sim_observation(obs, device)
        critic_obs = privileged_critic_obs(obs, info, device)
        with torch.no_grad():
            value = critic(critic_obs.unsqueeze(0)).squeeze(0)
        action_chunk, log_prob = policy.sample(policy_obs)
        action_chunk_np = action_chunk.squeeze(0).detach().cpu().numpy()

        chunk_reward = 0.0
        chunk_done = False
        for action in action_chunk_np:
            clipped_action = np.clip(action, env.joint_limits_low, env.joint_limits_high)
            obs, reward, terminated, truncated, info = step_action(env, clipped_action, args.steps_per_action)
            chunk_reward += reward
            metrics["steps"] += 1
            if terminated or truncated:
                chunk_done = True
                break

        observations.append(policy_obs)
        critic_observations.append(critic_obs)
        actions.append(action_chunk.squeeze(0).detach())
        log_probs.append(log_prob.squeeze(0).detach())
        rewards.append(chunk_reward)
        dones.append(float(chunk_done))
        values.append(value.detach())
        metrics["episode_return"] += chunk_reward
        metrics["success"] = float(info.get("success", False))
        metrics["contact_steps"] += int(info.get("contacted", False))
        metrics["grasp_steps"] += int(info.get("gripped", False))
        metrics["lift_steps"] += int(info.get("block_lifted", False))

        if chunk_done:
            break

    batch = RolloutBatch(
        observations=observations,
        critic_obs=critic_observations,
        actions=torch.stack(actions).to(device),
        old_log_probs=torch.stack(log_probs).to(device),
        rewards=torch.tensor(rewards, device=device, dtype=torch.float32),
        dones=torch.tensor(dones, device=device, dtype=torch.float32),
        values=torch.stack(values).to(device).float(),
    )
    return batch, metrics


def update_ppo(
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    args: argparse.Namespace,
) -> dict:
    returns, advantages = compute_returns_advantages(
        rollout.rewards,
        rollout.dones,
        rollout.values,
        args.gamma,
        args.gae_lambda,
    )
    num_samples = rollout.rewards.numel()
    indices = torch.arange(num_samples, device=rollout.rewards.device)
    stats = {"policy_loss": 0.0, "critic_loss": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
    updates = 0

    for _ in range(args.ppo_epochs):
        perm = indices[torch.randperm(num_samples, device=indices.device)]
        for start in range(0, num_samples, args.minibatch_size):
            mb = perm[start : start + args.minibatch_size]
            new_log_probs = torch.stack([
                policy.log_prob(rollout.observations[int(i.item())], rollout.actions[int(i.item())].unsqueeze(0)).squeeze(0)
                for i in mb
            ])
            ratio = torch.exp(torch.clamp(new_log_probs - rollout.old_log_probs[mb], -20.0, 20.0))
            unclipped = ratio * advantages[mb]
            clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages[mb]
            policy_loss = -torch.min(unclipped, clipped).mean()

            value_pred = critic(torch.stack([rollout.critic_obs[int(i.item())] for i in mb]))
            critic_loss = F.mse_loss(value_pred, returns[mb])

            policy_optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            policy_optimizer.step()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
            critic_optimizer.step()

            with torch.no_grad():
                approx_kl = (rollout.old_log_probs[mb] - new_log_probs).mean().item()
                clip_fraction = ((ratio - 1.0).abs() > args.clip_epsilon).float().mean().item()
            stats["policy_loss"] += policy_loss.item()
            stats["critic_loss"] += critic_loss.item()
            stats["approx_kl"] += approx_kl
            stats["clip_fraction"] += clip_fraction
            updates += 1

    return {key: value / max(1, updates) for key, value in stats.items()}


def save_checkpoint(
    path: Path,
    episode: int,
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "act_policy": policy.act_policy.state_dict(),
            "log_std": policy.log_std.detach().cpu(),
            "critic": critic.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "config": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    checkpoint = torch.load(path, map_location=device)
    policy.act_policy.load_state_dict(checkpoint["act_policy"], strict=False)
    policy.log_std.data.copy_(checkpoint["log_std"].to(device))
    critic.load_state_dict(checkpoint["critic"])
    policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    return int(checkpoint.get("episode", 0)) + 1


def evaluate(env: SO101PickPlaceEnv, policy: ACTGaussianPPOPolicy, device: torch.device, args: argparse.Namespace) -> dict:
    was_training = policy.training
    policy.eval()
    returns = []
    successes = []
    with torch.no_grad():
        for _ in range(args.eval_episodes):
            obs, info = env.reset()
            total_reward = 0.0
            success = False
            for _ in range(args.max_steps_per_episode):
                policy_obs = adapt_sim_observation(obs, device)
                mean_chunk = policy.mean_chunk(policy_obs, require_grad=False).squeeze(0).detach().cpu().numpy()
                done = False
                for action in mean_chunk:
                    clipped_action = np.clip(action, env.joint_limits_low, env.joint_limits_high)
                    obs, reward, terminated, truncated, info = step_action(env, clipped_action, args.steps_per_action)
                    total_reward += reward
                    done = terminated or truncated
                    success = success or bool(info.get("success", False))
                    if done:
                        break
                if done:
                    break
            returns.append(total_reward)
            successes.append(float(success))
    if was_training:
        policy.train()
    return {
        "eval/return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval/success_rate": float(np.mean(successes)) if successes else 0.0,
    }


def main() -> int:
    args = parse_args()
    if not args.experimental_act_ppo:
        print(
            "Refusing to launch ACT-chunk PPO without --experimental-act-ppo.\n"
            "This path previously completed but produced 0% success and unhealthy PPO clipping/KL metrics.\n"
            "Run train_sim_baseline.py first to verify the MuJoCo reward and action contract.",
            file=sys.stderr,
        )
        return 2
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
    load_training_dependencies()
    define_model_classes()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) != "cuda":
        print("Warning: CUDA is not available; ACT PPO will be slow.", file=sys.stderr)

    act_policy = load_act_policy(args.init_checkpoint, device)
    policy = ACTGaussianPPOPolicy(act_policy, action_dim=6, chunk_size=args.chunk_size, log_std_init=args.log_std_init).to(device)
    critic = PrivilegedCritic(input_dim=16).to(device)
    policy_optimizer = torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    start_episode = 0
    if args.resume is not None:
        start_episode = load_checkpoint(args.resume, policy, critic, policy_optimizer, critic_optimizer, device)

    block_pos = (0.0, 0.24, 0.0125) if args.curriculum_fixed_block else (0.0, 0.3, 0.0125)
    env = SO101PickPlaceEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_steps_per_episode,
        randomize_block=args.randomize_block_reset,
        task_instruction="pick up the block",
    )
    if not args.randomize_block_reset:
        # SO101PickPlaceEnv supports exact block placement via reset options but not
        # constructor block_pos. Monkey-patch only the nonrandom default used on reset.
        env.randomize_block = False
        original_reset = env.reset

        def reset_with_block(*reset_args, **reset_kwargs):
            options = reset_kwargs.pop("options", None) or {}
            options.setdefault("block_pos", block_pos)
            return original_reset(*reset_args, options=options, **reset_kwargs)

        env.reset = reset_with_block

    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        wandb.init(project="act-so101-sim-ppo", config=vars(args))

    print("Starting ACT PPO fine-tuning in SO-101 MuJoCo")
    print(f"  init_checkpoint: {args.init_checkpoint}")
    print(f"  checkpoint_path: {args.checkpoint_path}")
    print(f"  actor observation: observation.images.wrist + observation.state")
    print(f"  critic observation: privileged MuJoCo features")

    try:
        for episode in range(start_episode, args.episodes):
            start = time.time()
            rollout, rollout_metrics = collect_rollout(env, policy, critic, device, args)
            ppo_metrics = update_ppo(policy, critic, policy_optimizer, critic_optimizer, rollout, args)
            metrics = {
                "episode": episode,
                "rollout/return": rollout_metrics["episode_return"],
                "rollout/success": rollout_metrics["success"],
                "rollout/steps": rollout_metrics["steps"],
                "rollout/contact_steps": rollout_metrics["contact_steps"],
                "rollout/grasp_steps": rollout_metrics["grasp_steps"],
                "rollout/lift_steps": rollout_metrics["lift_steps"],
                "train/log_std_mean": policy.log_std.mean().item(),
                "time/episode_seconds": time.time() - start,
                **{f"train/{k}": v for k, v in ppo_metrics.items()},
            }
            if episode % 10 == 0:
                metrics.update(evaluate(env, policy, device, args))
                save_checkpoint(args.checkpoint_path, episode, policy, critic, policy_optimizer, critic_optimizer, args)
            if use_wandb:
                wandb.log(metrics, step=episode)
            print(
                f"[{episode:05d}] return={metrics['rollout/return']:.3f} "
                f"success={metrics['rollout/success']:.0f} "
                f"policy_loss={metrics['train/policy_loss']:.4f} "
                f"critic_loss={metrics['train/critic_loss']:.4f}"
            )
    finally:
        save_checkpoint(args.checkpoint_path, min(args.episodes, max(start_episode, args.episodes - 1)), policy, critic, policy_optimizer, critic_optimizer, args)
        env.close()
        if use_wandb:
            wandb.finish()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ACT sim training failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
