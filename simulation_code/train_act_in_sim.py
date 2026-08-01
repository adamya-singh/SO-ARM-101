#!/usr/bin/env python3
"""
Experimental online PPO fine-tuning for an ACT policy in SO-101 MuJoCo.

The policy is initialized from a supervised ACT checkpoint trained by
`train_act_on_data.py`. The actor sees only the same wrist camera and 6D state
schema as the physical dataset; the critic uses privileged MuJoCo state.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PosixPath
from typing import Any

from act_coordinate_utils import (
    MUJOCO_JOINT_HIGH,
    MUJOCO_JOINT_LOW,
    act_to_mujoco_qpos_torch,
    clip_mujoco_qpos,
    mujoco_qpos_to_act,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_INIT_CHECKPOINT = (
    SCRIPT_DIR
    / "outputs"
    / "train"
    / "act_so101_lead3_30_b32_20260709_161056"
    / "checkpoints"
    / "026020"
    / "pretrained_model"
)

ACT_REWARD_PROFILES: dict[str, dict[str, float | int | bool]] = {
    "baseline": {},
    "jaw_quality": {
        "pregrasp_alignment_reward_scale": 0.16,
        "aligned_close_reward": 0.14,
        "jaw_centered_contact_reward": 0.16,
        "recent_jaw_centered_contact_window": 20,
    },
    "jaw_quality_rebalanced": {
        "pregrasp_alignment_reward_scale": 0.16,
        "aligned_close_reward": 0.14,
        "jaw_centered_contact_reward": 0.16,
        "recent_jaw_centered_contact_window": 20,
        "bilateral_grasp_bonus": 0.8,
        "grasp_persistence_reward": 0.25,
        "side_push_penalty": -0.06,
        "block_displacement_penalty_scale": 0.45,
    },
    "strict_transition": {
        "strict_transition": True,
        "strict_grasp_required_steps": 5,
        "pregrasp_alignment_reward_scale": 0.0,
        "aligned_close_reward": 0.0,
        "jaw_centered_contact_reward": 0.0,
        "contact_persistence_reward": 0.0,
        "bilateral_grasp_bonus": 1.10,
        "grasp_persistence_reward": 0.10,
        "grasped_vertical_lift_reward_scale": 40.0,
        "grasped_vertical_lift_reward_cap": 0.40,
        "micro_lift_bonus": 0.80,
        "lift_bonus": 2.0,
        "success_lift_bonus": 6.0,
        "pregrasp_potential_scale": 0.20,
        "interior_contact_potential_scale": 0.30,
        "bilateral_opposition_potential_scale": 0.50,
        "corner_only_contact_penalty": -0.08,
    },
}


def resolve_reward_profile(name: str) -> dict[str, float | int | bool]:
    """Return an isolated reward-override mapping for a named ACT PPO profile."""
    try:
        return dict(ACT_REWARD_PROFILES[name])
    except KeyError as exc:
        choices = ", ".join(sorted(ACT_REWARD_PROFILES))
        raise ValueError(f"Unknown reward profile {name!r}; choose one of: {choices}") from exc

np = None
torch = None
nn = None
F = None
wandb = None
SO101PickPlaceEnv = None


def git_revision() -> str:
    """Return the source revision used for an experiment, including dirty state."""
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_DIR,
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        ).stdout.strip()
        return f"{revision}-dirty" if dirty else revision
    except Exception:
        return "unknown"


def simulator_model_hash() -> str:
    """Hash the production arm XML recorded in checkpoints and manifests."""
    path = SCRIPT_DIR / "model" / "so101_new_calib.xml"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configure_actor_train_scope(policy: Any, scope: str) -> list[str]:
    """Freeze ACT parameters outside a declared fine-tuning scope."""
    for parameter in policy.act_policy.parameters():
        parameter.requires_grad_(False)
    prefixes = {
        "all": ("",),
        "decoder_head": (
            "model.decoder.",
            "model.decoder_pos_embed.",
            "model.action_head.",
        ),
        "action_head": ("model.action_head.",),
    }[scope]
    for name, parameter in policy.act_policy.named_parameters():
        if any(name.startswith(prefix) for prefix in prefixes):
            parameter.requires_grad_(True)
    policy.log_std.requires_grad_(True)
    trainable = [name for name, parameter in policy.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError(f"actor train scope {scope!r} selected no parameters")
    return trainable


def set_optimizer_lr(optimizer: Any, learning_rate: float) -> None:
    """Apply the requested effective LR after restoring Adam state."""
    for group in optimizer.param_groups:
        group["lr"] = float(learning_rate)


def load_branch_checkpoint(path: Path, policy: Any, critic: Any, device: Any) -> dict[str, Any]:
    """Load weights for a new branch without inheriting optimizer/counter state."""
    torch.serialization.add_safe_globals([PosixPath])
    checkpoint = torch.load(path, map_location=device)
    policy.act_policy.load_state_dict(checkpoint["act_policy"], strict=False)
    policy.log_std.data.copy_(checkpoint["log_std"].to(device))
    critic.load_state_dict(checkpoint["critic"])
    return checkpoint


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


def seed_training(seed: int | None) -> None:
    """Seed the main PPO process; workers are seeded independently."""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


@dataclass
class RolloutBatch:
    observations: dict[str, torch.Tensor]
    critic_obs: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    num_envs: int


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

    def __init__(
        self,
        act_policy: nn.Module,
        action_dim: int,
        chunk_size: int,
        log_std_init: float,
        normalization_stats: dict[str, torch.Tensor],
    ):
        super().__init__()
        self.act_policy = act_policy
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
        for name, value in normalization_stats.items():
            self.register_buffer(name, value.detach().clone(), persistent=False)

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

        batch = {
            **observation,
            "observation.state": (
                observation["observation.state"] - self.state_mean
            ) / self.state_std,
            "observation.images.wrist": (
                observation["observation.images.wrist"] - self.image_mean
            ) / self.image_std,
        }
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
        normalized_mean = mean[:, : self.chunk_size, :]
        act_mean = normalized_mean * self.action_std + self.action_mean
        return act_to_mujoco_qpos_torch(act_mean)

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

        def __init__(
            self,
            act_policy: nn.Module,
            action_dim: int,
            chunk_size: int,
            log_std_init: float,
            normalization_stats: dict[str, torch.Tensor],
        ):
            super().__init__()
            self.act_policy = act_policy
            self.action_dim = action_dim
            self.chunk_size = chunk_size
            self.log_std = nn.Parameter(torch.full((action_dim,), float(log_std_init)))
            for name, value in normalization_stats.items():
                self.register_buffer(name, value.detach().clone(), persistent=False)

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

            batch = {
                **observation,
                "observation.state": (
                    observation["observation.state"] - self.state_mean
                ) / self.state_std,
                "observation.images.wrist": (
                    observation["observation.images.wrist"] - self.image_mean
                ) / self.image_std,
            }
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
            normalized_mean = mean[:, : self.chunk_size, :]
            act_mean = normalized_mean * self.action_std + self.action_mean
            return act_to_mujoco_qpos_torch(act_mean)

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
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--steps-per-action", type=int, default=1)
    parser.add_argument("--policy-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument(
        "--reward-profile",
        choices=tuple(ACT_REWARD_PROFILES),
        default="baseline",
        help="Named staged-pickup reward configuration; baseline preserves current reward defaults.",
    )
    parser.add_argument(
        "--reset-curriculum", choices=("none", "mixed", "adaptive"), default="none"
    )
    parser.add_argument("--reset-bank", type=Path, default=None)
    parser.add_argument(
        "--adaptive-stage",
        choices=("lift", "grasp", "full", "normal"),
        default="lift",
        help="Reset mixture used when --reset-curriculum adaptive.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Base RNG seed for training and environment workers.")
    parser.add_argument(
        "--resume-log-std-offset",
        type=float,
        default=0.0,
        help=(
            "Additive offset applied to the checkpoint log_std after --resume. "
            "Use negative values to continue with lower exploration noise."
        ),
    )
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--minibatch-size", type=int, default=32)
    parser.add_argument("--parallel-envs", type=int, default=8)
    parser.add_argument("--rollout-chunks-per-env", type=int, default=4)
    parser.add_argument("--randomize-block-reset", action="store_true")
    parser.add_argument(
        "--randomize-appearance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Vary scene lighting and surface brightness on reset (enabled by default).",
    )
    parser.add_argument("--block-dist-range", type=float, nargs=2, default=(0.22, 0.26), metavar=("MIN", "MAX"))
    parser.add_argument("--block-angle-range", type=float, nargs=2, default=(-10.0, 10.0), metavar=("MIN", "MAX"))
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
    parser.add_argument(
        "--branch-from",
        type=Path,
        default=None,
        help="Load policy/critic weights from a PPO checkpoint but start fresh optimizers and counters.",
    )
    parser.add_argument(
        "--actor-train-scope",
        choices=("all", "decoder_head", "action_head"),
        default="all",
    )
    parser.add_argument("--reference-anchor-coef", type=float, default=0.0)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--campaign-generation", type=int, default=-1)
    parser.add_argument("--campaign-condition", default="")
    parser.add_argument("--lineage-parent", default="")
    parser.add_argument(
        "--branch-log-std-offset",
        type=float,
        default=0.0,
        help="Global log-standard-deviation offset applied only with --branch-from.",
    )
    parser.add_argument(
        "--gripper-log-std-offset",
        type=float,
        default=0.0,
        help="Gripper-only log-standard-deviation offset applied only with --branch-from.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=SCRIPT_DIR / "act_sim_ppo_checkpoint.pt")
    parser.add_argument("--metrics-jsonl", type=Path, default=None)
    parser.add_argument("--wandb-url-path", type=Path, default=None)
    parser.add_argument(
        "--snapshot-every",
        type=int,
        default=0,
        help="Save numbered checkpoint snapshots every N completed training iterations after start/resume.",
    )
    parser.add_argument("--eval-episodes", type=int, default=0)
    args = parser.parse_args()
    if args.block_dist_range[0] > args.block_dist_range[1]:
        raise ValueError("--block-dist-range MIN must be <= MAX")
    if args.block_angle_range[0] > args.block_angle_range[1]:
        raise ValueError("--block-angle-range MIN must be <= MAX")
    if args.snapshot_every < 0:
        raise ValueError("--snapshot-every must be >= 0")
    if args.reset_curriculum != "none" and args.reset_bank is None:
        raise ValueError("--reset-bank is required for mixed/adaptive curricula")
    if args.resume is not None and args.branch_from is not None:
        raise ValueError("--resume and --branch-from are mutually exclusive")
    if args.branch_from is None and (
        args.branch_log_std_offset != 0.0 or args.gripper_log_std_offset != 0.0
    ):
        raise ValueError("branch log-std offsets require --branch-from")
    if args.reference_anchor_coef < 0.0 or args.entropy_coef < 0.0:
        raise ValueError("anchor and entropy coefficients must be non-negative")
    if args.reset_bank is not None and not args.reset_bank.is_file():
        raise FileNotFoundError(args.reset_bank)
    for checkpoint_arg in (args.resume, args.branch_from):
        if checkpoint_arg is not None and not checkpoint_arg.is_file():
            raise FileNotFoundError(checkpoint_arg)
    args.reward_kwargs = resolve_reward_profile(args.reward_profile)
    return args


def reset_stage_probabilities(curriculum: str, progress: int, adaptive_stage: str) -> dict[str, float]:
    if curriculum == "none":
        return {"normal": 1.0, "pregrasp": 0.0, "grasped": 0.0}
    if curriculum == "mixed":
        if progress < 30:
            return {"normal": 0.25, "pregrasp": 0.35, "grasped": 0.40}
        if progress < 60:
            return {"normal": 0.50, "pregrasp": 0.35, "grasped": 0.15}
        return {"normal": 0.75, "pregrasp": 0.25, "grasped": 0.0}
    mixes = {
        "lift": {"normal": 0.0, "pregrasp": 0.30, "grasped": 0.70},
        "grasp": {"normal": 0.20, "pregrasp": 0.60, "grasped": 0.20},
        "full": {"normal": 0.75, "pregrasp": 0.25, "grasped": 0.0},
        "normal": {"normal": 1.0, "pregrasp": 0.0, "grasped": 0.0},
    }
    return mixes[adaptive_stage]


def load_reset_bank(path: Path | None) -> dict[str, list[dict[str, Any]]]:
    if path is None:
        return {"pregrasp": [], "grasped": []}
    payload = json.loads(path.read_text())
    if int(payload.get("schema_version", 0)) != 2:
        raise ValueError("Reset bank must use preload-aware schema_version 2")
    bank = {stage: list(payload.get(stage, [])) for stage in ("pregrasp", "grasped")}
    for stage, states in bank.items():
        if len(states) < 32:
            raise ValueError(f"Reset bank requires at least 32 validated {stage} states; found {len(states)}")
        for index, state in enumerate(states):
            for key in ("robot_qpos", "robot_ctrl", "grasp_ctrl", "lift_ctrl", "block_pos"):
                if key not in state:
                    raise ValueError(f"Reset bank {stage}[{index}] is missing {key}")
            if not bool(state.get("proof", {}).get("valid", False)):
                raise ValueError(f"Reset bank {stage}[{index}] lacks a valid dynamic proof")
    return bank


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


def load_act_normalization_stats(checkpoint: Path, device: torch.device) -> dict[str, torch.Tensor]:
    """Load differentiable ACT normalization constants from the checkpoint."""
    try:
        from safetensors.torch import load_file
    except ImportError as exc:
        raise RuntimeError("safetensors is required to load ACT normalization statistics") from exc

    stats_path = checkpoint.resolve() / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    if not stats_path.is_file():
        raise FileNotFoundError(f"ACT normalization statistics not found: {stats_path}")
    values = load_file(str(stats_path), device=str(device))
    required = {
        "state_mean": "observation.state.mean",
        "state_std": "observation.state.std",
        "image_mean": "observation.images.wrist.mean",
        "image_std": "observation.images.wrist.std",
        "action_mean": "action.mean",
        "action_std": "action.std",
    }
    missing = [key for key in required.values() if key not in values]
    if missing:
        raise KeyError(f"ACT normalization statistics are missing: {', '.join(missing)}")
    return {
        name: values[key].to(device=device, dtype=torch.float32)
        for name, key in required.items()
    }


def adapt_sim_observation(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    """Map SO101PickPlaceEnv output to the physical ACT dataset schema."""
    wrist = obs["observation.images.camera2"]
    if wrist.dtype != np.float32:
        wrist_tensor = torch.from_numpy(wrist).float() / 255.0
    else:
        wrist_tensor = torch.from_numpy(wrist).float()
    wrist_tensor = wrist_tensor.permute(2, 0, 1).unsqueeze(0).to(device)
    act_state = mujoco_qpos_to_act(obs["observation.state"].astype(np.float32))
    state_tensor = torch.from_numpy(act_state).unsqueeze(0).to(device)
    return {
        "observation.images.wrist": wrist_tensor,
        "observation.state": state_tensor,
    }


def adapt_sim_observation_batch(wrist_images: np.ndarray, states: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """Map batched SO101PickPlaceEnv wrist images/states to the physical ACT dataset schema."""
    wrist_tensor = torch.from_numpy(wrist_images).float()
    if wrist_tensor.max() > 1.0:
        wrist_tensor = wrist_tensor / 255.0
    if wrist_tensor.ndim != 4:
        raise ValueError(f"Expected batched wrist images with shape (B,H,W,C), got {tuple(wrist_tensor.shape)}")
    wrist_tensor = wrist_tensor.permute(0, 3, 1, 2).contiguous().to(device)
    act_states = mujoco_qpos_to_act(states.astype(np.float32))
    state_tensor = torch.from_numpy(act_states).to(device)
    return {
        "observation.images.wrist": wrist_tensor,
        "observation.state": state_tensor,
    }


def flatten_observation_batches(observation_batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "observation.images.wrist": torch.cat([obs["observation.images.wrist"] for obs in observation_batches], dim=0),
        "observation.state": torch.cat([obs["observation.state"] for obs in observation_batches], dim=0),
    }


def select_observation_batch(observations: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "observation.images.wrist": observations["observation.images.wrist"][indices],
        "observation.state": observations["observation.state"][indices],
    }


def privileged_critic_obs(obs: dict[str, np.ndarray], info: dict, device: torch.device, max_steps: int = 500) -> torch.Tensor:
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
                    float(info.get("step_count", 0)) / float(max_steps),
                ],
                dtype=np.float32,
            ),
        ]
    )
    return torch.from_numpy(features).to(device)


def build_critic_features(obs: dict[str, np.ndarray], info: dict[str, Any], max_steps: int) -> np.ndarray:
    state = obs["observation.state"].astype(np.float32)
    gripper = np.asarray(info["gripper_pos"], dtype=np.float32)
    block = np.asarray(info["block_pos"], dtype=np.float32)
    return np.concatenate(
        [
            state,
            gripper,
            block,
            np.asarray(
                [
                    info["distance_to_block"],
                    info["block_height"],
                    float(info.get("success", False)),
                    float(info.get("step_count", 0)) / float(max_steps),
                ],
                dtype=np.float32,
            ),
        ]
    )


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


REWARD_COMPONENT_KEYS = [
    "distance_penalty",
    "far_from_block_penalty",
    "moving_away_penalty",
    "gripper_closing_reward",
    "closing_contact_bonus",
    "pregrasp_hover_penalty",
    "disengaged_stall_penalty",
    "post_attempt_commitment_reward",
    "escape_posture_penalty",
    "post_attempt_escape_penalty",
    "early_close_penalty",
    "approach_open_gripper_reward",
    "timed_close_reward",
    "pregrasp_alignment_score",
    "jaw_centering_score",
    "jaw_axis_alignment",
    "jaw_gap_width",
    "jaw_lateral_error",
    "jaw_depth_error",
    "pregrasp_face_axis_alignment",
    "pregrasp_face_guidance_score",
    "pregrasp_face_height_error",
    "pregrasp_face_depth_error",
    "pregrasp_face_lateral_error",
    "face_axis_guidance_progress",
    "interior_face_contact_score",
    "face_contact_quality",
    "bilateral_opposition_quality",
    "pregrasp_alignment_reward",
    "aligned_close_reward",
    "jaw_centered_contact_reward",
    "misaligned_close_penalty",
    "misaligned_lift_penalty",
    "side_push_penalty",
    "lift_grasp_ready",
    "micro_lift_ready",
    "micro_lifted",
    "recent_pregrasp_aligned_steps",
    "recent_jaw_centered_contact_steps",
    "near_contact_reward",
    "contact_persistence_reward",
    "contact_stall_penalty",
    "grasp_reward",
    "grasp_persistence_reward",
    "grasp_lift_motion_reward",
    "lift_progress_reward",
    "micro_lift_bonus",
    "grasped_vertical_lift_reward",
    "lift_side_push_penalty",
    "lift_bonus_reward",
    "success_lift_bonus",
    "block_displacement_penalty",
    "grip_force",
]


def empty_reward_components() -> dict[str, float]:
    return {key: 0.0 for key in REWARD_COMPONENT_KEYS}


def add_reward_components(accumulator: dict[str, float], info: dict[str, Any]) -> None:
    for key in REWARD_COMPONENT_KEYS:
        accumulator[key] += float(info.get(key, 0.0))


def _act_env_worker(remote, parent_remote, worker_config: dict[str, Any]) -> None:
    global np
    parent_remote.close()
    if worker_config.get("headless", False):
        os.environ.setdefault("MUJOCO_GL", "egl")
    try:
        import numpy as _np
        from so101_gym_env import SO101PickPlaceEnv as WorkerEnv

        np = _np
        worker_seed = worker_config["seed"]
        if worker_seed is not None:
            random.seed(worker_seed)
            np.random.seed(worker_seed)
        env = WorkerEnv(
            render_mode=None,
            max_episode_steps=worker_config["max_steps_per_episode"],
            randomize_block=worker_config["randomize_block_reset"],
            block_dist_range=worker_config["block_dist_range"],
            block_angle_range=worker_config["block_angle_range"],
            task_instruction="pick up the block",
            randomize_appearance=worker_config["randomize_appearance"],
            reward_kwargs=worker_config["reward_kwargs"],
        )
        block_pos = worker_config["block_pos"]
        reset_bank = worker_config["reset_bank"]
        reset_rng = random.Random(worker_seed)
        training_progress = 0
        max_steps = int(worker_config["max_steps_per_episode"])
        done = True
        obs = None
        info = None
        first_reset = True

        def reset_env():
            nonlocal done, obs, info, first_reset
            probabilities = reset_stage_probabilities(
                worker_config["reset_curriculum"], training_progress, worker_config["adaptive_stage"]
            )
            stage = reset_rng.choices(
                ("normal", "pregrasp", "grasped"),
                weights=tuple(probabilities[name] for name in ("normal", "pregrasp", "grasped")),
                k=1,
            )[0]
            options = {} if worker_config["randomize_block_reset"] else {"block_pos": block_pos}
            if stage != "normal":
                options["reset_state"] = reset_rng.choice(reset_bank[stage])
            obs, info = env.reset(seed=worker_seed if first_reset else None, options=options)
            first_reset = False
            done = False

        def current_payload():
            nonlocal done
            if done:
                reset_env()
            return {
                "wrist": obs["observation.images.camera2"],
                "state": obs["observation.state"].astype(np.float32),
                "critic": build_critic_features(obs, info, max_steps),
            }

        reset_env()
        while True:
            cmd, payload = remote.recv()
            if cmd == "get_obs":
                remote.send(current_payload())
            elif cmd == "step_chunk":
                action_chunk, steps_per_action = payload
                total_reward = 0.0
                metrics = {
                    "steps": 0,
                    "success": 0.0,
                    "contact_steps": 0,
                    "grasp_steps": 0,
                    "force_grasp_steps": 0,
                    "face_grasp_steps": 0,
                    "force_only_grasp_steps": 0,
                    "interior_face_contact_steps": 0,
                    "corner_only_contact_steps": 0,
                    "fixed_interior_face_contact_steps": 0,
                    "moving_interior_face_contact_steps": 0,
                    "bilateral_interior_face_contact_steps": 0,
                    "face_corner_rejection_steps": 0,
                    "face_alignment_sum": 0.0,
                    "face_opposition_sum": 0.0,
                    "face_jaw_axis_alignment_sum": 0.0,
                    "face_diagnostic_steps": 0,
                    "lift_steps": 0,
                    "max_block_height_gain": 0.0,
                    "max_strict_grasp_streak": 0,
                    "strict_success_steps": 0,
                    "stage_steps": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
                    "stage_strict_success_steps": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
                    "stage_max_strict_grasp_streak": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
                    "episodes_completed": 0,
                    "action_clip_count": 0,
                    "action_value_count": 0,
                    "reward_components": empty_reward_components(),
                }
                for action in action_chunk:
                    clipped_action, clip_mask = clip_mujoco_qpos(action)
                    metrics["action_clip_count"] += int(clip_mask.sum())
                    metrics["action_value_count"] += int(clip_mask.size)
                    for _ in range(int(steps_per_action)):
                        obs, reward, terminated, truncated, info = env.step(clipped_action)
                        total_reward += float(reward)
                        metrics["steps"] += 1
                        metrics["success"] = max(metrics["success"], float(info.get("success", False)))
                        metrics["contact_steps"] += int(bool(info.get("contacted", False)))
                        metrics["grasp_steps"] += int(bool(info.get("gripped", False)))
                        metrics["force_grasp_steps"] += int(bool(info.get("force_gripped", False)))
                        metrics["face_grasp_steps"] += int(bool(info.get("face_gripped", False)))
                        metrics["force_only_grasp_steps"] += int(
                            bool(info.get("force_only_grasp", False))
                        )
                        metrics["interior_face_contact_steps"] += int(
                            bool(info.get("interior_face_contact", False))
                        )
                        metrics["corner_only_contact_steps"] += int(
                            bool(info.get("corner_only_contact", False))
                        )
                        metrics["fixed_interior_face_contact_steps"] += int(
                            bool(info.get("fixed_interior_face_contact", False))
                        )
                        metrics["moving_interior_face_contact_steps"] += int(
                            bool(info.get("moving_interior_face_contact", False))
                        )
                        metrics["bilateral_interior_face_contact_steps"] += int(
                            bool(info.get("bilateral_interior_face_contact", False))
                        )
                        metrics["face_corner_rejection_steps"] += int(
                            bool(info.get("face_corner_rejection", False))
                        )
                        metrics["face_alignment_sum"] += float(info.get("face_alignment", 0.0))
                        metrics["face_opposition_sum"] += float(info.get("face_opposition", 0.0))
                        metrics["face_jaw_axis_alignment_sum"] += float(
                            info.get("face_jaw_axis_alignment", 0.0)
                        )
                        metrics["face_diagnostic_steps"] += 1
                        metrics["lift_steps"] += int(bool(info.get("block_lifted", False)))
                        metrics["max_block_height_gain"] = max(
                            metrics["max_block_height_gain"],
                            float(info.get("block_height_gain", 0.0)),
                        )
                        reset_stage = str(info.get("reset_stage", "normal"))
                        metrics["stage_steps"][reset_stage] += 1
                        strict_success = int(bool(info.get("strict_lift_success", False)))
                        metrics["strict_success_steps"] += strict_success
                        metrics["stage_strict_success_steps"][reset_stage] += strict_success
                        metrics["max_strict_grasp_streak"] = max(
                            metrics["max_strict_grasp_streak"],
                            int(info.get("strict_grasp_streak", 0)),
                        )
                        metrics["stage_max_strict_grasp_streak"][reset_stage] = max(
                            metrics["stage_max_strict_grasp_streak"][reset_stage],
                            int(info.get("strict_grasp_streak", 0)),
                        )
                        add_reward_components(metrics["reward_components"], info)
                        done = bool(terminated or truncated)
                        if done:
                            metrics["episodes_completed"] += 1
                            break
                    if done:
                        break
                remote.send(
                    {
                        "reward": total_reward,
                        "done": float(done),
                        "metrics": metrics,
                    }
                )
            elif cmd == "set_progress":
                training_progress = int(payload)
                remote.send({"ok": True})
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise ValueError(f"Unknown worker command: {cmd}")
    except Exception as exc:
        try:
            remote.send({"error": repr(exc)})
        finally:
            remote.close()


class ACTSubprocVecEnv:
    def __init__(self, num_envs: int, args: argparse.Namespace, block_pos: tuple[float, float, float]):
        self.num_envs = int(num_envs)
        self.closed = False
        ctx = mp.get_context("spawn")
        worker_config = {
            "headless": bool(args.headless),
            "max_steps_per_episode": int(args.max_steps_per_episode),
            "randomize_block_reset": bool(args.randomize_block_reset),
            "randomize_appearance": bool(args.randomize_appearance),
            "block_dist_range": tuple(float(v) for v in args.block_dist_range),
            "block_angle_range": tuple(float(v) for v in args.block_angle_range),
            "block_pos": tuple(float(v) for v in block_pos),
            "reward_kwargs": dict(args.reward_kwargs),
            "reset_curriculum": args.reset_curriculum,
            "adaptive_stage": args.adaptive_stage,
            "reset_bank": load_reset_bank(args.reset_bank),
        }
        self.parent_conns = []
        self.processes = []
        for worker_idx in range(self.num_envs):
            worker_config_for_process = {
                **worker_config,
                "seed": None if args.seed is None else int(args.seed) + worker_idx,
            }
            parent_conn, child_conn = ctx.Pipe()
            process = ctx.Process(
                target=_act_env_worker,
                args=(child_conn, parent_conn, worker_config_for_process),
                daemon=True,
                name=f"act-env-{worker_idx}",
            )
            process.start()
            child_conn.close()
            self.parent_conns.append(parent_conn)
            self.processes.append(process)

    def set_training_progress(self, progress: int) -> None:
        for conn in self.parent_conns:
            conn.send(("set_progress", int(progress)))
        payloads = [conn.recv() for conn in self.parent_conns]
        self._raise_worker_errors(payloads)

    def get_obs(self) -> dict[str, np.ndarray]:
        for conn in self.parent_conns:
            conn.send(("get_obs", None))
        payloads = [conn.recv() for conn in self.parent_conns]
        self._raise_worker_errors(payloads)
        return {
            "wrist": np.stack([payload["wrist"] for payload in payloads]),
            "state": np.stack([payload["state"] for payload in payloads]).astype(np.float32),
            "critic": np.stack([payload["critic"] for payload in payloads]).astype(np.float32),
        }

    def step_chunks(self, action_chunks: np.ndarray, steps_per_action: int) -> dict[str, Any]:
        for conn, action_chunk in zip(self.parent_conns, action_chunks):
            conn.send(("step_chunk", (action_chunk.astype(np.float32), int(steps_per_action))))
        payloads = [conn.recv() for conn in self.parent_conns]
        self._raise_worker_errors(payloads)
        metrics = {
            "steps": 0,
            "success": 0.0,
            "contact_steps": 0,
            "grasp_steps": 0,
            "force_grasp_steps": 0,
            "face_grasp_steps": 0,
            "force_only_grasp_steps": 0,
            "interior_face_contact_steps": 0,
            "corner_only_contact_steps": 0,
            "fixed_interior_face_contact_steps": 0,
            "moving_interior_face_contact_steps": 0,
            "bilateral_interior_face_contact_steps": 0,
            "face_corner_rejection_steps": 0,
            "face_alignment_sum": 0.0,
            "face_opposition_sum": 0.0,
            "face_jaw_axis_alignment_sum": 0.0,
            "face_diagnostic_steps": 0,
            "lift_steps": 0,
            "max_block_height_gain": 0.0,
            "max_strict_grasp_streak": 0,
            "strict_success_steps": 0,
            "stage_steps": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
            "stage_strict_success_steps": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
            "stage_max_strict_grasp_streak": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
            "episodes_completed": 0,
            "action_clip_count": 0,
            "action_value_count": 0,
            "reward_components": empty_reward_components(),
        }
        for payload in payloads:
            for key in (
                "steps",
                "success",
                "contact_steps",
                "grasp_steps",
                "force_grasp_steps",
                "face_grasp_steps",
                "force_only_grasp_steps",
                "interior_face_contact_steps",
                "corner_only_contact_steps",
                "fixed_interior_face_contact_steps",
                "moving_interior_face_contact_steps",
                "bilateral_interior_face_contact_steps",
                "face_corner_rejection_steps",
                "face_alignment_sum",
                "face_opposition_sum",
                "face_jaw_axis_alignment_sum",
                "face_diagnostic_steps",
                "lift_steps",
                "episodes_completed",
                "action_clip_count",
                "action_value_count",
            ):
                metrics[key] += payload["metrics"][key]
            metrics["max_block_height_gain"] = max(
                metrics["max_block_height_gain"],
                float(payload["metrics"]["max_block_height_gain"]),
            )
            metrics["max_strict_grasp_streak"] = max(
                metrics["max_strict_grasp_streak"], payload["metrics"]["max_strict_grasp_streak"]
            )
            metrics["strict_success_steps"] += payload["metrics"]["strict_success_steps"]
            for stage in ("normal", "pregrasp", "grasped"):
                metrics["stage_steps"][stage] += payload["metrics"]["stage_steps"][stage]
                metrics["stage_strict_success_steps"][stage] += payload["metrics"]["stage_strict_success_steps"][stage]
                metrics["stage_max_strict_grasp_streak"][stage] = max(
                    metrics["stage_max_strict_grasp_streak"][stage],
                    payload["metrics"]["stage_max_strict_grasp_streak"][stage],
                )
            for key in REWARD_COMPONENT_KEYS:
                metrics["reward_components"][key] += payload["metrics"]["reward_components"][key]
        return {
            "rewards": np.asarray([payload["reward"] for payload in payloads], dtype=np.float32),
            "dones": np.asarray([payload["done"] for payload in payloads], dtype=np.float32),
            "metrics": metrics,
        }

    def _raise_worker_errors(self, payloads: list[Any]) -> None:
        for payload in payloads:
            if isinstance(payload, dict) and "error" in payload:
                raise RuntimeError(f"ACT env worker failed: {payload['error']}")

    def close(self) -> None:
        if self.closed:
            return
        for conn in self.parent_conns:
            try:
                conn.send(("close", None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.terminate()
        self.closed = True


def compute_returns_advantages(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    num_envs: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    if num_envs <= 1:
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

    chunks = rewards.numel() // num_envs
    reward_grid = rewards.view(chunks, num_envs)
    done_grid = dones.view(chunks, num_envs)
    value_grid = values.view(chunks, num_envs)
    advantage_grid = torch.zeros_like(reward_grid)
    last_gae = torch.zeros(num_envs, device=rewards.device)
    next_value = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(chunks)):
        nonterminal = 1.0 - done_grid[t]
        delta = reward_grid[t] + gamma * next_value * nonterminal - value_grid[t]
        last_gae = delta + gamma * gae_lambda * nonterminal * last_gae
        advantage_grid[t] = last_gae
        next_value = value_grid[t]
    returns = (advantage_grid + value_grid).reshape(-1)
    advantages = advantage_grid.reshape(-1)
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
    observation_batches = []
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
        "force_grasp_steps": 0,
        "face_grasp_steps": 0,
        "force_only_grasp_steps": 0,
        "interior_face_contact_steps": 0,
        "corner_only_contact_steps": 0,
        "fixed_interior_face_contact_steps": 0,
        "moving_interior_face_contact_steps": 0,
        "bilateral_interior_face_contact_steps": 0,
        "face_corner_rejection_steps": 0,
        "face_alignment_sum": 0.0,
        "face_opposition_sum": 0.0,
        "face_jaw_axis_alignment_sum": 0.0,
        "face_diagnostic_steps": 0,
        "lift_steps": 0,
        "max_block_height_gain": 0.0,
        "max_strict_grasp_streak": 0,
        "strict_success_steps": 0,
        "stage_steps": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
        "stage_strict_success_steps": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
        "stage_max_strict_grasp_streak": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
        "steps": 0,
        "episodes_completed": 0,
        "policy_forward_sec": 0.0,
        "env_step_sec": 0.0,
        "action_clip_count": 0,
        "action_value_count": 0,
        "reward_components": empty_reward_components(),
    }

    chunks_per_env = max(1, int(args.rollout_chunks_per_env))
    for _ in range(chunks_per_env):
        policy_obs = adapt_sim_observation(obs, device)
        critic_obs = privileged_critic_obs(obs, info, device, args.max_steps_per_episode)
        with torch.no_grad():
            value = critic(critic_obs.unsqueeze(0)).squeeze(0)
        forward_start = time.time()
        action_chunk, log_prob = policy.sample(policy_obs)
        metrics["policy_forward_sec"] += time.time() - forward_start
        action_chunk_np = action_chunk.squeeze(0).detach().cpu().numpy()

        chunk_reward = 0.0
        chunk_done = False
        env_start = time.time()
        for action in action_chunk_np:
            clipped_action, clip_mask = clip_mujoco_qpos(action)
            metrics["action_clip_count"] += int(clip_mask.sum())
            metrics["action_value_count"] += int(clip_mask.size)
            obs, reward, terminated, truncated, info = step_action(env, clipped_action, args.steps_per_action)
            chunk_reward += reward
            metrics["steps"] += 1
            add_reward_components(metrics["reward_components"], info)
            if terminated or truncated:
                chunk_done = True
                break
        metrics["env_step_sec"] += time.time() - env_start

        observation_batches.append(policy_obs)
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
        metrics["force_grasp_steps"] += int(info.get("force_gripped", False))
        metrics["face_grasp_steps"] += int(info.get("face_gripped", False))
        metrics["force_only_grasp_steps"] += int(info.get("force_only_grasp", False))
        metrics["interior_face_contact_steps"] += int(
            info.get("interior_face_contact", False)
        )
        metrics["corner_only_contact_steps"] += int(
            info.get("corner_only_contact", False)
        )
        metrics["fixed_interior_face_contact_steps"] += int(
            info.get("fixed_interior_face_contact", False)
        )
        metrics["moving_interior_face_contact_steps"] += int(
            info.get("moving_interior_face_contact", False)
        )
        metrics["bilateral_interior_face_contact_steps"] += int(
            info.get("bilateral_interior_face_contact", False)
        )
        metrics["face_corner_rejection_steps"] += int(info.get("face_corner_rejection", False))
        metrics["face_alignment_sum"] += float(info.get("face_alignment", 0.0))
        metrics["face_opposition_sum"] += float(info.get("face_opposition", 0.0))
        metrics["face_jaw_axis_alignment_sum"] += float(
            info.get("face_jaw_axis_alignment", 0.0)
        )
        metrics["face_diagnostic_steps"] += 1
        metrics["lift_steps"] += int(info.get("block_lifted", False))
        metrics["max_block_height_gain"] = max(
            metrics["max_block_height_gain"],
            float(info.get("block_height_gain", 0.0)),
        )
        metrics["max_strict_grasp_streak"] = max(
            metrics["max_strict_grasp_streak"], int(info.get("strict_grasp_streak", 0))
        )
        reset_stage = str(info.get("reset_stage", "normal"))
        strict_success = int(bool(info.get("strict_lift_success", False)))
        metrics["strict_success_steps"] += strict_success
        metrics["stage_steps"][reset_stage] += 1
        metrics["stage_strict_success_steps"][reset_stage] += strict_success
        metrics["stage_max_strict_grasp_streak"][reset_stage] = max(
            metrics["stage_max_strict_grasp_streak"][reset_stage],
            int(info.get("strict_grasp_streak", 0)),
        )

        if chunk_done:
            metrics["episodes_completed"] += 1
            break

    batch = RolloutBatch(
        observations=flatten_observation_batches(observation_batches),
        critic_obs=torch.stack(critic_observations).to(device),
        actions=torch.stack(actions).to(device),
        old_log_probs=torch.stack(log_probs).to(device),
        rewards=torch.tensor(rewards, device=device, dtype=torch.float32),
        dones=torch.tensor(dones, device=device, dtype=torch.float32),
        values=torch.stack(values).to(device).float(),
        num_envs=1,
    )
    return batch, metrics


def collect_parallel_rollout(
    env: ACTSubprocVecEnv,
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[RolloutBatch, dict]:
    observation_batches = []
    critic_batches = []
    action_batches = []
    log_prob_batches = []
    reward_batches = []
    done_batches = []
    value_batches = []
    metrics = {
        "episode_return": 0.0,
        "success": 0.0,
        "contact_steps": 0,
        "grasp_steps": 0,
        "force_grasp_steps": 0,
        "face_grasp_steps": 0,
        "force_only_grasp_steps": 0,
        "interior_face_contact_steps": 0,
        "corner_only_contact_steps": 0,
        "fixed_interior_face_contact_steps": 0,
        "moving_interior_face_contact_steps": 0,
        "bilateral_interior_face_contact_steps": 0,
        "face_corner_rejection_steps": 0,
        "face_alignment_sum": 0.0,
        "face_opposition_sum": 0.0,
        "face_jaw_axis_alignment_sum": 0.0,
        "face_diagnostic_steps": 0,
        "lift_steps": 0,
        "max_block_height_gain": 0.0,
        "max_strict_grasp_streak": 0,
        "strict_success_steps": 0,
        "stage_steps": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
        "stage_strict_success_steps": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
        "stage_max_strict_grasp_streak": {stage: 0 for stage in ("normal", "pregrasp", "grasped")},
        "steps": 0,
        "episodes_completed": 0,
        "policy_forward_sec": 0.0,
        "env_step_sec": 0.0,
        "action_clip_count": 0,
        "action_value_count": 0,
        "reward_components": empty_reward_components(),
    }

    for _ in range(max(1, int(args.rollout_chunks_per_env))):
        obs_payload = env.get_obs()
        policy_obs = adapt_sim_observation_batch(obs_payload["wrist"], obs_payload["state"], device)
        critic_obs = torch.from_numpy(obs_payload["critic"]).float().to(device)
        with torch.no_grad():
            values = critic(critic_obs)

        forward_start = time.time()
        action_chunks, log_probs = policy.sample(policy_obs)
        metrics["policy_forward_sec"] += time.time() - forward_start

        env_start = time.time()
        step_payload = env.step_chunks(action_chunks.detach().cpu().numpy(), args.steps_per_action)
        metrics["env_step_sec"] += time.time() - env_start

        observation_batches.append(policy_obs)
        critic_batches.append(critic_obs)
        action_batches.append(action_chunks.detach())
        log_prob_batches.append(log_probs.detach())
        reward_batches.append(torch.from_numpy(step_payload["rewards"]).float().to(device))
        done_batches.append(torch.from_numpy(step_payload["dones"]).float().to(device))
        value_batches.append(values.detach())

        step_metrics = step_payload["metrics"]
        metrics["episode_return"] += float(step_payload["rewards"].sum())
        metrics["success"] += float(step_metrics["success"])
        metrics["contact_steps"] += int(step_metrics["contact_steps"])
        metrics["grasp_steps"] += int(step_metrics["grasp_steps"])
        metrics["force_grasp_steps"] += int(step_metrics["force_grasp_steps"])
        metrics["face_grasp_steps"] += int(step_metrics["face_grasp_steps"])
        metrics["force_only_grasp_steps"] += int(step_metrics["force_only_grasp_steps"])
        metrics["interior_face_contact_steps"] += int(
            step_metrics["interior_face_contact_steps"]
        )
        metrics["corner_only_contact_steps"] += int(
            step_metrics["corner_only_contact_steps"]
        )
        metrics["fixed_interior_face_contact_steps"] += int(
            step_metrics["fixed_interior_face_contact_steps"]
        )
        metrics["moving_interior_face_contact_steps"] += int(
            step_metrics["moving_interior_face_contact_steps"]
        )
        metrics["bilateral_interior_face_contact_steps"] += int(
            step_metrics["bilateral_interior_face_contact_steps"]
        )
        metrics["face_corner_rejection_steps"] += int(step_metrics["face_corner_rejection_steps"])
        metrics["face_alignment_sum"] += float(step_metrics["face_alignment_sum"])
        metrics["face_opposition_sum"] += float(step_metrics["face_opposition_sum"])
        metrics["face_jaw_axis_alignment_sum"] += float(
            step_metrics["face_jaw_axis_alignment_sum"]
        )
        metrics["face_diagnostic_steps"] += int(step_metrics["face_diagnostic_steps"])
        metrics["lift_steps"] += int(step_metrics["lift_steps"])
        metrics["max_block_height_gain"] = max(
            metrics["max_block_height_gain"],
            float(step_metrics["max_block_height_gain"]),
        )
        metrics["max_strict_grasp_streak"] = max(
            metrics["max_strict_grasp_streak"], int(step_metrics["max_strict_grasp_streak"])
        )
        metrics["strict_success_steps"] += int(step_metrics["strict_success_steps"])
        for stage in ("normal", "pregrasp", "grasped"):
            metrics["stage_steps"][stage] += int(step_metrics["stage_steps"][stage])
            metrics["stage_strict_success_steps"][stage] += int(
                step_metrics["stage_strict_success_steps"][stage]
            )
            metrics["stage_max_strict_grasp_streak"][stage] = max(
                metrics["stage_max_strict_grasp_streak"][stage],
                int(step_metrics["stage_max_strict_grasp_streak"][stage]),
            )
        metrics["steps"] += int(step_metrics["steps"])
        metrics["episodes_completed"] += int(step_metrics["episodes_completed"])
        metrics["action_clip_count"] += int(step_metrics["action_clip_count"])
        metrics["action_value_count"] += int(step_metrics["action_value_count"])
        for key in REWARD_COMPONENT_KEYS:
            metrics["reward_components"][key] += float(step_metrics["reward_components"][key])

    num_envs = max(1, env.num_envs)
    metrics["episode_return"] /= float(num_envs)
    metrics["success"] /= float(num_envs)
    return (
        RolloutBatch(
            observations=flatten_observation_batches(observation_batches),
            critic_obs=torch.cat(critic_batches, dim=0),
            actions=torch.cat(action_batches, dim=0),
            old_log_probs=torch.cat(log_prob_batches, dim=0),
            rewards=torch.cat(reward_batches, dim=0),
            dones=torch.cat(done_batches, dim=0),
            values=torch.cat(value_batches, dim=0).float(),
            num_envs=env.num_envs,
        ),
        metrics,
    )


def update_ppo(
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    rollout: RolloutBatch,
    args: argparse.Namespace,
    reference_policy: ACTGaussianPPOPolicy | None = None,
) -> dict:
    returns, advantages = compute_returns_advantages(
        rollout.rewards,
        rollout.dones,
        rollout.values,
        args.gamma,
        args.gae_lambda,
        rollout.num_envs,
    )
    num_samples = rollout.rewards.numel()
    indices = torch.arange(num_samples, device=rollout.rewards.device)
    stats = {
        "policy_loss": 0.0,
        "surrogate_loss": 0.0,
        "anchor_loss": 0.0,
        "entropy": 0.0,
        "critic_loss": 0.0,
        "approx_kl": 0.0,
        "clip_fraction": 0.0,
    }
    updates = 0
    update_start = time.time()

    for _ in range(args.ppo_epochs):
        perm = indices[torch.randperm(num_samples, device=indices.device)]
        for start in range(0, num_samples, args.minibatch_size):
            mb = perm[start : start + args.minibatch_size]
            observations = select_observation_batch(rollout.observations, mb)
            distribution = policy.distribution(observations)
            new_log_probs = distribution.log_prob(rollout.actions[mb]).sum(dim=(1, 2))
            ratio = torch.exp(torch.clamp(new_log_probs - rollout.old_log_probs[mb], -20.0, 20.0))
            unclipped = ratio * advantages[mb]
            clipped = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages[mb]
            surrogate_loss = -torch.min(unclipped, clipped).mean()
            entropy = distribution.entropy().mean()
            anchor_loss = torch.zeros((), dtype=surrogate_loss.dtype, device=surrogate_loss.device)
            if reference_policy is not None and args.reference_anchor_coef > 0.0:
                with torch.no_grad():
                    reference_mean = reference_policy.mean_chunk(observations, require_grad=False)
                joint_range = torch.as_tensor(
                    MUJOCO_JOINT_HIGH - MUJOCO_JOINT_LOW,
                    dtype=distribution.mean.dtype,
                    device=distribution.mean.device,
                ).view(1, 1, -1)
                anchor_loss = ((distribution.mean - reference_mean) / joint_range).square().mean()
            policy_loss = (
                surrogate_loss
                + float(args.reference_anchor_coef) * anchor_loss
                - float(args.entropy_coef) * entropy
            )

            value_pred = critic(rollout.critic_obs[mb])
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
            stats["surrogate_loss"] += surrogate_loss.item()
            stats["anchor_loss"] += anchor_loss.item()
            stats["entropy"] += entropy.item()
            stats["critic_loss"] += critic_loss.item()
            stats["approx_kl"] += approx_kl
            stats["clip_fraction"] += clip_fraction
            updates += 1

    averaged = {key: value / max(1, updates) for key, value in stats.items()}
    averaged["ppo_update_sec"] = time.time() - update_start
    averaged["num_samples"] = float(num_samples)
    return averaged


def save_checkpoint(
    path: Path,
    episode: int,
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    total_chunks: int = 0,
    total_env_steps: int = 0,
    reference_policy: ACTGaussianPPOPolicy | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
            "episode": episode,
            "curriculum_progress": episode,
            "curriculum_stage": args.adaptive_stage if args.reset_curriculum == "adaptive" else args.reset_curriculum,
            "total_chunks": total_chunks,
            "total_env_steps": total_env_steps,
            "act_policy": policy.act_policy.state_dict(),
            "log_std": policy.log_std.detach().cpu(),
            "critic": critic.state_dict(),
            "policy_optimizer": policy_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
            "config": vars(args),
            "metadata": {
                "simulator_model_sha256": simulator_model_hash(),
                "trainable_actor_parameters": [
                    name for name, parameter in policy.named_parameters() if parameter.requires_grad
                ],
                "effective_policy_lrs": [float(group["lr"]) for group in policy_optimizer.param_groups],
                "effective_critic_lrs": [float(group["lr"]) for group in critic_optimizer.param_groups],
                "lineage_parent": args.lineage_parent
                or str(args.branch_from or args.resume or args.init_checkpoint),
                "campaign_generation": int(args.campaign_generation),
                "campaign_condition": args.campaign_condition,
                "reset_configuration": {
                    "reset_curriculum": args.reset_curriculum,
                    "randomize_block_reset": bool(args.randomize_block_reset),
                    "curriculum_fixed_block": bool(args.curriculum_fixed_block),
                },
                "reward_profile": args.reward_profile,
            },
        }
    if reference_policy is not None:
        payload["reference_act_policy"] = reference_policy.act_policy.state_dict()
    torch.save(payload, path)


def checkpoint_snapshot_path(path: Path, episode: int) -> Path:
    """Return a sibling checkpoint path marked with the completed episode."""
    return path.with_name(f"{path.stem}_ep{episode:04d}{path.suffix}")


def load_checkpoint(
    path: Path,
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[int, int, int, dict[str, Any]]:
    torch.serialization.add_safe_globals([PosixPath])
    checkpoint = torch.load(path, map_location=device)
    policy.act_policy.load_state_dict(checkpoint["act_policy"], strict=False)
    policy.log_std.data.copy_(checkpoint["log_std"].to(device))
    critic.load_state_dict(checkpoint["critic"])
    policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
    critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
    return (
        int(checkpoint.get("episode", 0)) + 1,
        int(checkpoint.get("total_chunks", 0)),
        int(checkpoint.get("total_env_steps", 0)),
        checkpoint,
    )


def gpu_snapshot(device: torch.device) -> dict[str, float]:
    metrics = {
        "gpu/util_percent": 0.0,
        "gpu/memory_util_percent": 0.0,
        "gpu/temp_c": 0.0,
        "gpu/power_w": 0.0,
        "gpu/memory_allocated_mb": 0.0,
        "gpu/memory_reserved_mb": 0.0,
    }
    if device.type == "cuda":
        metrics["gpu/memory_allocated_mb"] = float(torch.cuda.memory_allocated(device) / (1024 * 1024))
        metrics["gpu/memory_reserved_mb"] = float(torch.cuda.memory_reserved(device) / (1024 * 1024))
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,utilization.memory,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0 and result.stdout.strip():
            values = [float(item.strip()) for item in result.stdout.strip().splitlines()[0].split(",")]
            metrics["gpu/util_percent"] = values[0]
            metrics["gpu/memory_util_percent"] = values[1]
            metrics["gpu/temp_c"] = values[2]
            metrics["gpu/power_w"] = values[3]
    except Exception:
        pass
    return metrics


def block_position(args: argparse.Namespace) -> tuple[float, float, float]:
    return (0.0, 0.24, 0.0125) if args.curriculum_fixed_block else (0.0, 0.3, 0.0125)


def make_sequential_env(args: argparse.Namespace, block_pos: tuple[float, float, float]) -> SO101PickPlaceEnv:
    env = SO101PickPlaceEnv(
        render_mode=None if args.no_render else "human",
        max_episode_steps=args.max_steps_per_episode,
        randomize_block=args.randomize_block_reset,
        block_dist_range=tuple(float(v) for v in args.block_dist_range),
        block_angle_range=tuple(float(v) for v in args.block_angle_range),
        task_instruction="pick up the block",
        randomize_appearance=args.randomize_appearance,
        reward_kwargs=dict(args.reward_kwargs),
    )
    if not args.randomize_block_reset:
        env.randomize_block = False
        original_reset = env.reset

        def reset_with_block(*reset_args, **reset_kwargs):
            options = reset_kwargs.pop("options", None) or {}
            options.setdefault("block_pos", block_pos)
            return original_reset(*reset_args, options=options, **reset_kwargs)

        env.reset = reset_with_block
    if args.seed is not None:
        env.reset(seed=int(args.seed))
    return env


def make_training_env(args: argparse.Namespace, block_pos: tuple[float, float, float]) -> SO101PickPlaceEnv | ACTSubprocVecEnv:
    if args.parallel_envs <= 1:
        return make_sequential_env(args, block_pos)
    return ACTSubprocVecEnv(args.parallel_envs, args, block_pos)


def close_training_env(env: Any) -> None:
    if env is not None:
        env.close()


def run_train_iteration(
    env: SO101PickPlaceEnv | ACTSubprocVecEnv,
    policy: ACTGaussianPPOPolicy,
    critic: PrivilegedCritic,
    policy_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    device: torch.device,
    args: argparse.Namespace,
    progress: int = 0,
    reference_policy: ACTGaussianPPOPolicy | None = None,
) -> tuple[dict[str, float], RolloutBatch]:
    iteration_start = time.time()
    if isinstance(env, ACTSubprocVecEnv):
        env.set_training_progress(progress)
        rollout, rollout_metrics = collect_parallel_rollout(env, policy, critic, device, args)
    else:
        rollout, rollout_metrics = collect_rollout(env, policy, critic, device, args)
    ppo_metrics = update_ppo(
        policy,
        critic,
        policy_optimizer,
        critic_optimizer,
        rollout,
        args,
        reference_policy=reference_policy,
    )
    elapsed = max(time.time() - iteration_start, 1e-9)
    chunks = int(rollout.rewards.numel())
    env_steps = int(rollout_metrics["steps"])
    metrics = {
        "rollout/return": float(rollout_metrics["episode_return"]),
        "rollout/success": float(rollout_metrics["success"]),
        "rollout/steps": float(env_steps),
        "rollout/chunks": float(chunks),
        "rollout/episodes_completed": float(rollout_metrics["episodes_completed"]),
        "rollout/contact_steps": float(rollout_metrics["contact_steps"]),
        "rollout/grasp_steps": float(rollout_metrics["grasp_steps"]),
        "rollout/force_grasp_steps": float(rollout_metrics["force_grasp_steps"]),
        "rollout/face_grasp_steps": float(rollout_metrics["face_grasp_steps"]),
        "rollout/force_only_grasp_steps": float(
            rollout_metrics["force_only_grasp_steps"]
        ),
        "rollout/interior_face_contact_steps": float(
            rollout_metrics["interior_face_contact_steps"]
        ),
        "rollout/corner_only_contact_steps": float(
            rollout_metrics["corner_only_contact_steps"]
        ),
        "rollout/fixed_interior_face_contact_steps": float(
            rollout_metrics["fixed_interior_face_contact_steps"]
        ),
        "rollout/moving_interior_face_contact_steps": float(
            rollout_metrics["moving_interior_face_contact_steps"]
        ),
        "rollout/bilateral_interior_face_contact_steps": float(
            rollout_metrics["bilateral_interior_face_contact_steps"]
        ),
        "rollout/face_corner_rejection_steps": float(
            rollout_metrics["face_corner_rejection_steps"]
        ),
        "rollout/mean_face_alignment": float(
            rollout_metrics["face_alignment_sum"]
            / max(1, rollout_metrics["face_diagnostic_steps"])
        ),
        "rollout/mean_face_opposition": float(
            rollout_metrics["face_opposition_sum"]
            / max(1, rollout_metrics["face_diagnostic_steps"])
        ),
        "rollout/mean_face_jaw_axis_alignment": float(
            rollout_metrics["face_jaw_axis_alignment_sum"]
            / max(1, rollout_metrics["face_diagnostic_steps"])
        ),
        "rollout/lift_steps": float(rollout_metrics["lift_steps"]),
        "rollout/max_block_height_gain": float(rollout_metrics["max_block_height_gain"]),
        "rollout/max_strict_grasp_streak": float(rollout_metrics["max_strict_grasp_streak"]),
        "rollout/strict_success_steps": float(rollout_metrics["strict_success_steps"]),
        "rollout/action_clip_rate": float(
            rollout_metrics["action_clip_count"] / max(1, rollout_metrics["action_value_count"])
        ),
        "throughput/env_steps_per_sec": float(env_steps / elapsed),
        "throughput/chunks_per_sec": float(chunks / elapsed),
        "time/iteration_seconds": elapsed,
        "time/policy_forward_sec": float(rollout_metrics["policy_forward_sec"]),
        "time/env_step_sec": float(rollout_metrics["env_step_sec"]),
        "train/log_std_mean": float(policy.log_std.mean().item()),
        **{f"train/{key}": float(value) for key, value in ppo_metrics.items()},
        **gpu_snapshot(device),
    }
    metrics.update(
        {
            f"rollout/reward_components/{key}": float(value)
            for key, value in rollout_metrics["reward_components"].items()
        }
    )
    for stage in ("normal", "pregrasp", "grasped"):
        metrics[f"rollout/reset_stage/{stage}_steps"] = float(rollout_metrics["stage_steps"][stage])
        metrics[f"rollout/reset_stage/{stage}_strict_success_steps"] = float(
            rollout_metrics["stage_strict_success_steps"][stage]
        )
        metrics[f"rollout/reset_stage/{stage}_max_strict_grasp_streak"] = float(
            rollout_metrics["stage_max_strict_grasp_streak"][stage]
        )
    return metrics, rollout


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
                    clipped_action, _clip_mask = clip_mujoco_qpos(action)
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
    if args.parallel_envs > 1 and not args.no_render:
        print("Forcing --no-render because subprocess parallel training cannot use the human viewer.")
        args.no_render = True
    load_training_dependencies()
    define_model_classes()
    seed_training(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if str(device) != "cuda":
        print("Warning: CUDA is not available; ACT PPO will be slow.", file=sys.stderr)

    act_policy = load_act_policy(args.init_checkpoint, device)
    normalization_stats = load_act_normalization_stats(args.init_checkpoint, device)
    policy = ACTGaussianPPOPolicy(
        act_policy,
        action_dim=6,
        chunk_size=args.chunk_size,
        log_std_init=args.log_std_init,
        normalization_stats=normalization_stats,
    ).to(device)
    critic = PrivilegedCritic(input_dim=16).to(device)

    start_episode = 0
    total_chunks = 0
    total_env_steps = 0
    restored_checkpoint = None
    if args.branch_from is not None:
        restored_checkpoint = load_branch_checkpoint(args.branch_from, policy, critic, device)
        policy.log_std.data.add_(float(args.branch_log_std_offset))
        policy.log_std.data[-1].add_(float(args.gripper_log_std_offset))

    trainable_actor_parameters = configure_actor_train_scope(policy, args.actor_train_scope)
    policy_optimizer = torch.optim.Adam(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.policy_lr,
    )
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    if args.resume is not None:
        start_episode, total_chunks, total_env_steps, restored_checkpoint = load_checkpoint(
            args.resume, policy, critic, policy_optimizer, critic_optimizer, device
        )
        set_optimizer_lr(policy_optimizer, args.policy_lr)
        set_optimizer_lr(critic_optimizer, args.critic_lr)
        if args.resume_log_std_offset != 0.0:
            policy.log_std.data.add_(float(args.resume_log_std_offset))
            print(
                f"Applied resume log_std offset {args.resume_log_std_offset:+.4f}; "
                f"log_std_mean={policy.log_std.mean().item():.4f}"
            )

    reference_policy = None
    if args.reference_anchor_coef > 0.0:
        reference_policy = copy.deepcopy(policy).to(device).eval()
        if args.resume is not None:
            reference_state = (restored_checkpoint or {}).get("reference_act_policy")
            if reference_state is None:
                raise ValueError(
                    "anchored exact resume requires reference_act_policy in the checkpoint"
                )
            reference_policy.act_policy.load_state_dict(reference_state, strict=False)
        for parameter in reference_policy.parameters():
            parameter.requires_grad_(False)

    block_pos = block_position(args)
    env = make_training_env(args, block_pos)
    eval_env = make_sequential_env(args, block_pos) if args.eval_episodes > 0 else None

    use_wandb = not args.no_wandb and wandb is not None
    if use_wandb:
        wandb.init(
            project="act-so101-sim-ppo",
            config={
                **vars(args),
                "git_revision": git_revision(),
                "init_checkpoint_resolved": str(args.init_checkpoint.resolve()),
            },
        )
        if args.wandb_url_path is not None:
            args.wandb_url_path.parent.mkdir(parents=True, exist_ok=True)
            args.wandb_url_path.write_text(str(wandb.run.url or "") + "\n")

    print("Starting ACT PPO fine-tuning in SO-101 MuJoCo")
    print(f"  init_checkpoint: {args.init_checkpoint}")
    print(f"  checkpoint_path: {args.checkpoint_path}")
    print(f"  actor observation: observation.images.wrist + observation.state")
    print(f"  critic observation: privileged MuJoCo features")
    print(f"  parallel_envs: {args.parallel_envs}")
    print(f"  rollout_chunks_per_env: {args.rollout_chunks_per_env}")
    print(f"  minibatch_size: {args.minibatch_size}")
    print(f"  seed: {args.seed}")
    print(f"  randomize_appearance: {args.randomize_appearance}")
    print(f"  reward_profile: {args.reward_profile}")
    print(f"  reward_kwargs: {args.reward_kwargs}")
    print(f"  actor_train_scope: {args.actor_train_scope}")
    print(f"  trainable_actor_parameters: {len(trainable_actor_parameters)}")
    print(f"  reference_anchor_coef: {args.reference_anchor_coef}")
    print(f"  entropy_coef: {args.entropy_coef}")
    print(f"  simulator_model_sha256: {simulator_model_hash()}")
    print(f"  git_revision: {git_revision()}")

    last_completed_episode = start_episode - 1
    try:
        for episode in range(start_episode, args.episodes):
            metrics, rollout = run_train_iteration(
                env,
                policy,
                critic,
                policy_optimizer,
                critic_optimizer,
                device,
                args,
                progress=episode,
                reference_policy=reference_policy,
            )
            total_chunks += int(rollout.rewards.numel())
            total_env_steps += int(metrics["rollout/steps"])
            last_completed_episode = episode
            metrics["episode"] = episode
            metrics["total/chunks"] = total_chunks
            metrics["total/env_steps"] = total_env_steps
            if episode % 10 == 0:
                if eval_env is not None:
                    metrics.update(evaluate(eval_env, policy, device, args))
                save_checkpoint(
                    args.checkpoint_path,
                    episode,
                    policy,
                    critic,
                    policy_optimizer,
                    critic_optimizer,
                    args,
                    total_chunks=total_chunks,
                    total_env_steps=total_env_steps,
                    reference_policy=reference_policy,
                )
            completed_iterations = episode - start_episode + 1
            if args.snapshot_every and completed_iterations % args.snapshot_every == 0:
                save_checkpoint(
                    checkpoint_snapshot_path(args.checkpoint_path, episode),
                    episode,
                    policy,
                    critic,
                    policy_optimizer,
                    critic_optimizer,
                    args,
                    total_chunks=total_chunks,
                    total_env_steps=total_env_steps,
                    reference_policy=reference_policy,
                )
            if use_wandb:
                wandb.log(metrics, step=episode)
            if args.metrics_jsonl is not None:
                args.metrics_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with args.metrics_jsonl.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(metrics, sort_keys=True) + "\n")
            print(
                f"[{episode:05d}] return={metrics['rollout/return']:.3f} "
                f"success={metrics['rollout/success']:.0f} "
                f"policy_loss={metrics['train/policy_loss']:.4f} "
                f"critic_loss={metrics['train/critic_loss']:.4f} "
                f"env_steps/s={metrics['throughput/env_steps_per_sec']:.1f} "
                f"gpu={metrics['gpu/util_percent']:.0f}%"
            )
    finally:
        save_checkpoint(
            args.checkpoint_path,
            max(0, last_completed_episode),
            policy,
            critic,
            policy_optimizer,
            critic_optimizer,
            args,
            total_chunks=total_chunks,
            total_env_steps=total_env_steps,
            reference_policy=reference_policy,
        )
        close_training_env(env)
        close_training_env(eval_env)
        if use_wandb:
            wandb.finish()

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ACT sim training failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
