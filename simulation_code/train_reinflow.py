"""
ReinFlow Training Script for VLA Models (On-Policy Actor-Critic)

Full ReinFlow-style flow-based RL training that:
1. Uses a NOISE NETWORK σ_θ'(t, a, o) conditioned on time, action, and observation
2. Collects trajectories and updates ON-POLICY (no replay buffer)
3. Uses actor-critic with learned value function baseline
4. Computes exact per-step log-probabilities with gradients through both networks

Reference: https://reinflow.github.io/

Supports:
- SmolVLA (450M parameters) - default, fast training
- Pi0 (3.3B parameters) - requires more memory, use --model-type pi0

Key features:
- ON-POLICY training (collect rollout -> compute loss -> discard data)
- Actor-Critic architecture with learned V(s) baseline
- Gradients flow through both velocity AND noise networks
- Fresh sigma computation at training time (not stored)

Usage:
    conda activate lerobot
    python train_reinflow.py
    
    # Train with Pi0 model:
    python train_reinflow.py --model-type pi0 --parallel-envs 2

    # Resume from checkpoint:
    python train_reinflow.py --resume reinflow_checkpoint.pt
    
    # Headless mode (for Colab/SSH):
    python train_reinflow.py --no-render --headless
    
    # Parallel mode for A100 GPU (8 environments):
    python train_reinflow.py --parallel-envs 8 --no-render --headless
"""

import os
import sys
import time
import math
import argparse
from dataclasses import dataclass
from collections import deque

# Setup headless rendering BEFORE importing mujoco
from mujoco_rendering import setup_mujoco_rendering
setup_mujoco_rendering()

import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

# SmolVLA imports
from reinflow_smolvla import (
    ReinFlowSmolVLA,
    setup_reinflow_policy,
    prepare_observation_for_reinflow,
    prepare_batched_observation,
    compute_trajectory_log_probs_onpolicy,
    compute_actor_critic_loss,
    compute_ppo_loss,
    compute_gae,
    compute_returns,
    save_reinflow_checkpoint,
    load_reinflow_checkpoint,
    # Pi0 support
    ReinFlowPi0,
    setup_reinflow_pi0_policy,
    compute_trajectory_log_probs_onpolicy_pi0,
    save_reinflow_pi0_checkpoint,
    load_reinflow_pi0_checkpoint,
    detect_model_type_from_checkpoint,
)
from so101_mujoco_utils import (
    set_initial_pose,
    send_position_command,
    convert_to_dictionary,
    get_camera_observation,
    get_robot_state,
    compute_reward,
    reset_env,
    reset_reward_state,
    unnormalize_action_for_vla,
    load_vla_processors,
)


# ===== Training Configuration =====

class TrainingConfig:
    """
    Configuration for ReinFlow RL training (On-Policy Actor-Critic).
    
    HYPERPARAMETER SCALING FOR SMOLVLA's CHUNK SIZE:
    ================================================
    SmolVLA outputs action chunks of size 50 (vs typical 4-8 in ReinFlow paper).
    This 6-12x increase in action dimensionality (300 dims vs 24-48) affects several quantities:
    
    1. LOG PROBABILITY SCALE:
       - Log prob formula sums over all dimensions: Σ(-0.5 * diff²/σ²)
       - With 6x more dimensions, log probs are ~6x more negative
       - This is expected, not a bug
    
    2. KL DIVERGENCE SCALE:
       - KL is computed from log prob differences
       - With larger log probs, KL values are naturally larger
       - Paper's target_kl=0.01 → our target_kl=0.05-0.1 (scaled ~6x)
    
    3. GRADIENT MAGNITUDE:
       - Gradients accumulate over 300 output dimensions vs 48
       - Effective gradient signal is ~6x stronger
       - Paper's policy_lr=4.5e-5 → our policy_lr=3e-7 (scaled down aggressively for stable reward growth)
    
    4. PARAMETERS THAT NEED ADJUSTMENT FOR HIGH DIMS:
       - clip_epsilon: While ratio-based, high-dim actions cause more ratio drift. Use 0.15-0.2.
       - gae_lambda, gamma: Reward-based, independent of action dims
       
       NOTE: sigma_min/sigma_max DO need scaling! See notes/sigma-scaling-bug-fix.md.
       Sigma scales as √(D_new/D_old) to maintain stable log probability variance.
    
    Reference: ReinFlow paper Table 7b (visual manipulation settings)
    """
    
    # Model selection
    model_type = "smolvla"  # "smolvla" (450M) or "pi0" (3.3B)
    
    # Environment
    model_path = 'model/scene.xml'
    instruction = "pick up the block"
    
    # Robot starting pose (degrees)
    starting_position = {
        'shoulder_pan': 0.06,
        'shoulder_lift': -100.21,
        'elbow_flex': 89.95,
        'wrist_flex': 66.46,
        'wrist_roll': 5.96,
        'gripper': 1.0,
    }
    
    # Training hyperparameters
    num_episodes = 20000
    max_steps_per_episode = 150
    gamma = 0.999  # Discount factor (paper uses 0.99 for state tasks)
    # SCALED FOR CHUNK SIZE 50: Paper uses 4.5e-5 for chunks of 4-8. With 6x more dims,
    # gradients are ~6x stronger. RL reward growth is currently more stable with a smaller actor step.
    policy_lr = 0.0000003  # 3e-7 - reduce post-update KL while preserving PPO correctness fixes
    critic_lr = 0.0001   # Critic learning rate (can be higher, doesn't scale with action dims)
    grad_clip_norm = 0.25  # Gradient clipping for stability
    
    # ReinFlow specific
    num_denoising_steps = 1  # Paper uses K=4 for most tasks (smolvla default was 10)
    chunks_per_episode = 3   # How many chunks to execute per episode (fresh obs between each)
    
    # ReinFlow noise bounds (paper Table 7b - visual manipulation)
    # SCALED FOR CHUNK SIZE 50: Sigma must scale as √(D_smolvla / D_paper) ≈ √(300/28) ≈ 3.3×
    # Paper uses [0.05, 0.14] for ~28 dims → SmolVLA needs [0.16, 0.46] for 300 dims
    # Using slightly higher values to ensure stable log probabilities
    sigma_min = 0.25  # Scaled from paper's ~0.08 (0.08 × 3.3 ≈ 0.26)
    sigma_max = 0.50  # Scaled from paper's ~0.14 (0.14 × 3.3 ≈ 0.46)
    
    # Noise decay schedule (paper Appendix D)
    noise_decay_start = 1.0 #paper says no decay for visual tasks #0.35    # Hold sigma_max for 35% of training
    noise_decay_ratio = 0.7     # Decay to 0.3*sigma_min + 0.7*sigma_max
    
    # Entropy regularization (paper Section 4.4 - visual tasks)
    entropy_coeff = 0.0  # Paper uses 0.00-0.01 for visual manipulation
    
    # Critic warmup (paper Appendix D.2)
    critic_warmup_iters = 30  # Paper uses 2-5 iterations
    
    # What to train
    train_action_head = True   # Train action_out_proj (velocity head)
    train_time_mlp = True      # Train time MLP
    train_full_expert = False  # Full-expert RL remains available, but is no longer the default
    trainable_scope = "rl_stable_heads"  # Stable RL subset: action_in/out, time MLP in/out, noise_mlp
    train_noise_head = True    # Train noise_mlp (σ_θ' network) - always True for ReinFlow
    train_critic = True        # Train critic network for actor-critic
    
    # Policy execution
    steps_per_action = 10  # Physics steps per policy action
    
    # Reward
    lift_threshold = 0.08
    distance_penalty_scale = 0.4
    horizontal_progress_scale = 0.12
    vertical_approach_scale = 0.05
    approach_closeness_scale = 0.035
    alignment_reward_cap = 0.035
    near_contact_bonus = 0.03
    contact_entry_bonus = 0.18
    contact_persistence_reward = 0.045
    hover_stall_threshold = 8
    hover_penalty = -0.01
    bilateral_grasp_bonus = 0.30
    grasp_persistence_reward = 0.08
    slip_penalty_contact = -0.03
    slip_penalty_grasp = -0.08
    block_displacement_penalty_scale = 0.08
    lift_bonus = 0.2  # Bonus when block is lifted above threshold
    lift_bonus_threshold = 0.04  # Height (meters) to trigger lift bonus (lower than terminal)
    sustained_contact_threshold = 5   # Frames of continuous contact before bonus triggers
    sustained_contact_bonus = 0.2     # Extra reward per step after threshold reached
    
    # Logging
    log_interval = 1
    save_interval = 10
    
    # Rendering
    render = False  # Set False for faster training
    
    # Checkpointing
    checkpoint_path = "reinflow_checkpoint.pt"
    pretrained_path = "lerobot/smolvla_base"
    finetuned_smolvla_path = "adamyathegreat/so101_pickplace_v1_smolvla"
    
    # Parallelization
    num_parallel_envs = 1  # SmolVLA on a 14.6 GB GPU is most reliable at <= 5 envs.
    use_subproc_env = False
    
    # Weights & Biases
    wandb_project = "reinflow-smolvla"
    wandb_enabled = True
    
    # PPO Hyperparameters (paper Table 7b - visual manipulation)
    # Note: Some values scaled for SmolVLA's chunk_size=50 (see docstring above)
    num_ppo_epochs = 2           # Keep PPO updates shallow; this setup usually destabilizes by epoch 1.
    minibatch_size = 8           # Mini-batch size for PPO updates
    clip_epsilon = 0.05          # Reverted to 0.05 for stability (0.15 caused KL explosion at 4.5k eps)
    value_clip_epsilon = 0.2     # Clip range for value function (0 to disable)
    gae_lambda = 0.95            # GAE lambda parameter
    # SCALED FOR CHUNK SIZE 50: Paper uses 0.01 for chunks of 4-8. With 6x more dims,
    # KL values are naturally ~6x larger, so we scale target_kl accordingly (0.05-0.1)
    target_kl = 0.1             # KL threshold for early stopping (scaled ~6x from paper's 0.01)
    
    # Gradient accumulation (paper Appendix D)
    gradient_accumulation_steps = 15  # Paper uses 15 for visual tasks
    
    # Learning rate warmup (paper Table 9b)
    lr_warmup_iterations = 10  # Paper uses 10 for PickPlaceCan, 25 for NutAssemblySquare

    # PPO correctness / stability guards
    logprob_eval_microbatch_size = 1
    critic_backprop_into_policy = False
    
    # Pi0-specific settings (used when model_type="pi0")
    pi0_gradient_checkpointing = True  # Required for 3.3B model to fit in 24GB
    pi0_pretrained_path = "lerobot/pi0"  # Default Pi0 model path


def get_noise_bounds(episode: int, total_episodes: int, config) -> tuple:
    """
    Get current noise bounds with decay schedule (paper Appendix D).
    
    The paper holds sigma_max constant for the first portion of training,
    then decays it linearly to improve precision in later stages.
    
    Args:
        episode: Current episode number
        total_episodes: Total number of episodes
        config: TrainingConfig with sigma_min, sigma_max, noise_decay_start, noise_decay_ratio
        
    Returns:
        (sigma_min, sigma_max): Current noise bounds
    """
    progress = episode / total_episodes if total_episodes > 0 else 0.0
    
    if progress < config.noise_decay_start:
        # Hold at initial bounds for first portion of training
        return config.sigma_min, config.sigma_max
    else:
        # Decay sigma_max linearly
        decay_progress = (progress - config.noise_decay_start) / (1.0 - config.noise_decay_start)
        # Decay from sigma_max to sigma_max * noise_decay_ratio
        new_sigma_max = config.sigma_max * (1.0 - decay_progress * (1.0 - config.noise_decay_ratio))
        return config.sigma_min, max(new_sigma_max, config.sigma_min)


def _get_reward_kwargs(config) -> dict:
    return {
        "lift_threshold": config.lift_threshold,
        "distance_penalty_scale": config.distance_penalty_scale,
        "horizontal_progress_scale": config.horizontal_progress_scale,
        "vertical_approach_scale": config.vertical_approach_scale,
        "approach_closeness_scale": config.approach_closeness_scale,
        "alignment_reward_cap": config.alignment_reward_cap,
        "near_contact_bonus": config.near_contact_bonus,
        "contact_entry_bonus": config.contact_entry_bonus,
        "contact_persistence_reward": config.contact_persistence_reward,
        "hover_stall_threshold": config.hover_stall_threshold,
        "hover_penalty": config.hover_penalty,
        "bilateral_grasp_bonus": config.bilateral_grasp_bonus,
        "grasp_persistence_reward": config.grasp_persistence_reward,
        "slip_penalty_contact": config.slip_penalty_contact,
        "slip_penalty_grasp": config.slip_penalty_grasp,
        "block_displacement_penalty_scale": config.block_displacement_penalty_scale,
        "lift_bonus": config.lift_bonus,
        "lift_bonus_threshold": config.lift_bonus_threshold,
        "sustained_contact_threshold": config.sustained_contact_threshold,
        "sustained_contact_bonus": config.sustained_contact_bonus,
    }


@dataclass
class ParallelRolloutBatch:
    """Env-major chunk rollouts before flattening for PPO."""
    trajectories: list
    observations: list
    rewards: np.ndarray
    dones: np.ndarray
    valid: np.ndarray


def _validate_parallel_rollout_masks(valid_mask: np.ndarray, done_mask: np.ndarray) -> None:
    """Assert env-major rollout invariants used by parallel PPO."""
    num_envs, max_chunks = valid_mask.shape
    for env_idx in range(num_envs):
        seen_invalid = False
        seen_done = False
        for chunk_idx in range(max_chunks):
            is_valid = bool(valid_mask[env_idx, chunk_idx])
            is_done = bool(done_mask[env_idx, chunk_idx])

            if not is_valid:
                seen_invalid = True
                continue

            if seen_invalid:
                raise AssertionError(
                    f"Env {env_idx} has a valid rollout slot after an invalid slot at chunk {chunk_idx}"
                )
            if seen_done:
                raise AssertionError(
                    f"Env {env_idx} has a valid rollout slot after a terminal chunk at chunk {chunk_idx}"
                )
            if is_done:
                seen_done = True


def _flatten_valid_observations(observation_grid: list, valid_mask: np.ndarray) -> dict:
    """Flatten env-major observations into a batch using only valid chunk slots."""
    flat_observations = []
    num_envs, max_chunks = valid_mask.shape
    for env_idx in range(num_envs):
        for chunk_idx in range(max_chunks):
            if valid_mask[env_idx, chunk_idx]:
                flat_observations.append(observation_grid[env_idx][chunk_idx])

    if not flat_observations:
        return {}

    return {
        key: torch.cat([obs[key] for obs in flat_observations], dim=0)
        for key in flat_observations[0].keys()
    }


def _compute_policy_log_probs(policy, trajectories, observations, return_sigmas: bool = False):
    """Dispatch log-prob evaluation through the policy-specific deterministic path."""
    if isinstance(policy, ReinFlowPi0):
        return compute_trajectory_log_probs_onpolicy_pi0(
            policy, trajectories, observations, return_sigmas=return_sigmas
        )
    return compute_trajectory_log_probs_onpolicy(
        policy, trajectories, observations, return_sigmas=return_sigmas
    )


def _default_logprob_shift_metrics() -> dict:
    return {
        "logprob_abs_mean": 0.0,
        "logprob_abs_max": 0.0,
        "kl_div": 0.0,
        "ratio_mean": 1.0,
        "ratio_std": 0.0,
        "ratio_min": 1.0,
        "ratio_max": 1.0,
        "clip_fraction": 0.0,
    }


def _compute_logprob_shift_metrics(
    reference_log_probs: torch.Tensor,
    current_log_probs: torch.Tensor,
    clip_epsilon: float = 0.0,
) -> dict:
    """Summarize how much two log-prob evaluations differ."""
    logprob_diff = current_log_probs - reference_log_probs
    log_ratio = torch.clamp(logprob_diff, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    metrics = {
        "logprob_abs_mean": logprob_diff.abs().mean().item(),
        "logprob_abs_max": logprob_diff.abs().max().item(),
        "kl_div": ((ratio - 1.0) - log_ratio).mean().item(),
        "ratio_mean": ratio.mean().item(),
        "ratio_std": ratio.std().item() if ratio.numel() > 1 else 0.0,
        "ratio_min": ratio.min().item(),
        "ratio_max": ratio.max().item(),
        "clip_fraction": ((ratio - 1.0).abs() > clip_epsilon).float().mean().item() if clip_epsilon > 0 else 0.0,
    }
    return metrics


def _aggregate_logprob_shift_metrics(metric_history: list[dict]) -> dict:
    if not metric_history:
        return _default_logprob_shift_metrics()
    return {
        "logprob_abs_mean": float(np.mean([m["logprob_abs_mean"] for m in metric_history])),
        "logprob_abs_max": float(np.max([m["logprob_abs_max"] for m in metric_history])),
        "kl_div": float(np.mean([m["kl_div"] for m in metric_history])),
        "ratio_mean": float(np.mean([m["ratio_mean"] for m in metric_history])),
        "ratio_std": float(np.mean([m["ratio_std"] for m in metric_history])),
        "ratio_min": float(np.min([m["ratio_min"] for m in metric_history])),
        "ratio_max": float(np.max([m["ratio_max"] for m in metric_history])),
        "clip_fraction": float(np.mean([m["clip_fraction"] for m in metric_history])),
    }


def _compute_actor_delta_metrics(optimizer: torch.optim.Optimizer) -> dict:
    """Estimate the actor parameter update magnitude from Adam's internal step tensors."""
    total_sq = 0.0
    max_abs = 0.0
    has_state = False

    for group in optimizer.param_groups:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group.get("weight_decay", 0.0)
        amsgrad = group.get("amsgrad", False)

        for param in group["params"]:
            state = optimizer.state.get(param, {})
            if "exp_avg" not in state or "exp_avg_sq" not in state or "step" not in state:
                continue

            step_value = state["step"].item() if isinstance(state["step"], torch.Tensor) else state["step"]
            if step_value <= 0:
                continue

            bias_correction1 = 1.0 - beta1 ** step_value
            bias_correction2 = 1.0 - beta2 ** step_value
            if bias_correction1 == 0.0 or bias_correction2 == 0.0:
                continue

            exp_avg = state["exp_avg"]
            exp_avg_sq = state["max_exp_avg_sq"] if amsgrad and "max_exp_avg_sq" in state else state["exp_avg_sq"]
            denom = exp_avg_sq.sqrt() / math.sqrt(bias_correction2)
            denom = denom.add(eps)

            step_tensor = exp_avg / denom
            if weight_decay != 0.0:
                step_tensor = step_tensor + weight_decay * param.detach()
            step_tensor = step_tensor * (lr / bias_correction1)

            total_sq += step_tensor.pow(2).sum().item()
            max_abs = max(max_abs, step_tensor.abs().max().item())
            has_state = True

    if not has_state:
        return {"delta_l2": 0.0, "delta_max_abs": 0.0}
    return {"delta_l2": math.sqrt(total_sq), "delta_max_abs": max_abs}


def _update_ema(previous: float | None, value: float, window: int = 20) -> float:
    alpha = 2.0 / (window + 1.0)
    if previous is None:
        return value
    return alpha * value + (1.0 - alpha) * previous


def _validate_pre_update_logprob_invariant(metrics: dict) -> None:
    """Fail loudly if PPO old/new log-prob evaluation is inconsistent before any update."""
    kl_tol = 1e-6
    ratio_mean_tol = 1e-4
    if metrics["kl_div"] > kl_tol or abs(metrics["ratio_mean"] - 1.0) > ratio_mean_tol:
        raise RuntimeError(
            "Pre-update PPO log-prob invariant failed: "
            f"kl={metrics['kl_div']:.8f}, ratio_mean={metrics['ratio_mean']:.8f}, "
            f"logprob_abs_mean={metrics['logprob_abs_mean']:.8f}, "
            f"logprob_abs_max={metrics['logprob_abs_max']:.8f}. "
            "Old and recomputed log probabilities must match before the first optimizer step."
        )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReinFlow Training for VLA Models (On-Policy Actor-Critic)')
    parser.add_argument('--model-type', type=str, choices=['smolvla', 'pi0'], default='smolvla',
                        help='Model type: smolvla (450M, default) or pi0 (3.3B)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable visualization for faster training')
    parser.add_argument('--headless', action='store_true',
                        help='Force headless rendering (EGL/OSMesa) for Colab/SSH')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train')
    parser.add_argument('--policy-lr', type=float, default=None,
                        help='Policy learning rate')
    parser.add_argument('--critic-lr', type=float, default=None,
                        help='Critic learning rate')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Initial policy source for a fresh RL run (HF repo or local LeRobot model/checkpoint)')
    parser.add_argument('--start-from-finetuned', action='store_true',
                        help='Start a fresh SmolVLA RL run from the canonical finetuned SmolVLA policy')
    parser.add_argument('--parallel-envs', type=int, default=None,
                        help='Number of parallel environments (default: 1 for sequential, use 8-16 for A100)')
    parser.add_argument('--subproc', action='store_true',
                        help='Use subprocess-based parallel rendering')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--full-expert', action='store_true',
                        help='Train entire Action Expert (~100M params)')
    parser.add_argument('--no-full-expert', action='store_true',
                        help='Disable full expert training')
    parser.add_argument('--no-gradient-checkpointing', action='store_true',
                        help='Disable gradient checkpointing for Pi0 (not recommended)')
    return parser.parse_args()


def _get_trainable_scope_label(config) -> str:
    """Return a human-readable summary of the ReinFlow trainable scope."""
    if getattr(config, "model_type", "smolvla") == "smolvla" and getattr(config, "trainable_scope", None) == "rl_stable_heads":
        return "rl-stable-heads"
    if config.train_full_expert:
        return "full-expert"
    components = []
    if config.train_action_head:
        components.append("action-head")
    if config.train_time_mlp:
        components.append("time-mlp")
    if config.train_noise_head:
        components.append("noise-head")
    if config.train_critic:
        components.append("critic")
    return ",".join(components) if components else "frozen"


def _print_smolvla_startup_summary(config, rl_policy, start_mode: str, resume_path: str | None) -> None:
    """Log SmolVLA startup semantics and normalization state."""
    print(f"  [ReinFlow] Startup mode: {start_mode}")
    if resume_path is not None:
        print(f"  [ReinFlow] Resume checkpoint: {resume_path}")
        print("  [ReinFlow] Ignoring fresh-start --pretrained/--start-from-finetuned weights for model restoration")
    print(f"  [ReinFlow] Policy source: {getattr(rl_policy, 'base_policy_source', None)}")
    print(
        f"  [ReinFlow] Policy source type: "
        f"{getattr(rl_policy, 'base_policy_source_type', 'unknown')}"
    )
    print(f"  [ReinFlow] Normalization mode: {getattr(rl_policy, 'normalization_mode', 'unknown')}")
    print(f"  [ReinFlow] State/action frame: {getattr(rl_policy, 'state_action_frame', 'unknown')}")
    print(f"  [ReinFlow] Image schema: {getattr(rl_policy, 'expected_image_keys', [])}")
    print(f"  [ReinFlow] Trainable scope: {_get_trainable_scope_label(config)}")


# ===== Parallel Training Loop (for A100) =====

def train_parallel(config, args, device):
    """
    Parallel ON-POLICY PPO training loop with GAE, mini-batching, and KL early stopping.
    
    Key features (paper-faithful):
    1. PPO clipped surrogate objective
    2. GAE (Generalized Advantage Estimation) for better credit assignment
    3. Multiple epochs with mini-batching for sample efficiency
    4. KL divergence monitoring with early stopping
    5. Learning rate scheduling
    
    Supports both SmolVLA and Pi0 models.
    """
    num_envs = config.num_parallel_envs
    
    # Setup ReinFlow policy with critic and processors FIRST (needed for env)
    print("\n" + "="*60)
    print(f"Setting up ReinFlow {config.model_type.upper()} (PPO, On-Policy)")
    print("="*60)
    
    # Select appropriate setup function based on model type.
    preprocessor = None
    postprocessor = None
    
    if config.model_type == "pi0":
        rl_policy, preprocessor, postprocessor = setup_reinflow_pi0_policy(
            pretrained_path=config.pretrained_path,
            device=str(device),
            num_steps=config.num_denoising_steps,
            train_action_head=config.train_action_head,
            train_time_mlp=config.train_time_mlp,
            train_full_expert=config.train_full_expert,
            train_noise_head=config.train_noise_head,
            train_critic=config.train_critic,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            gradient_checkpointing=config.pi0_gradient_checkpointing,
        )
        # Pi0 sigma is set in adapter during creation
        print(f"  [ReinFlow] Sigma bounds: [{config.sigma_min}, {config.sigma_max}]")
    else:
        rl_policy, preprocessor, postprocessor = setup_reinflow_policy(
            pretrained_path=config.pretrained_path,
            device=str(device),
            num_steps=config.num_denoising_steps,
            train_action_head=config.train_action_head,
            train_time_mlp=config.train_time_mlp,
            train_full_expert=config.train_full_expert,
            trainable_scope=config.trainable_scope,
            train_noise_head=config.train_noise_head,
            train_critic=config.train_critic,
        )
        rl_policy.base.model.sigma_min = config.sigma_min
        rl_policy.base.model.sigma_max = config.sigma_max
        print(f"  [ReinFlow] Sigma bounds: [{config.sigma_min}, {config.sigma_max}]")
    
    # Choose environment implementation.
    # The env layer always owns MuJoCo<->physical frame conversion; model-specific
    # processors only affect normalization/denormalization.
    if config.use_subproc_env:
        from subproc_vectorized_env import SubprocMuJoCoEnv
        print(f"\n[Parallel Mode - SUBPROC] Running {num_envs} environments in separate processes")
        vec_env = SubprocMuJoCoEnv(
            num_envs=num_envs,
            model_path=config.model_path,
            starting_position=config.starting_position,
            instruction=config.instruction,
            model_type=config.model_type,
            preprocessor=preprocessor,  # Processor-backed normalization when available
            **_get_reward_kwargs(config),
        )
    else:
        from vectorized_env import VectorizedMuJoCoEnv
        print(f"\n[Parallel Mode] Running {num_envs} environments in parallel")
        vec_env = VectorizedMuJoCoEnv(
            num_envs=num_envs,
            model_path=config.model_path,
            starting_position=config.starting_position,
            instruction=config.instruction,
            model_type=config.model_type,
            preprocessor=preprocessor,  # Processor-backed normalization when available
            **_get_reward_kwargs(config),
        )
    
    # Load checkpoint if resuming
    start_episode = 0
    episode_rewards_history = []
    wandb_run_id = None
    
    checkpoint_to_load = args.resume if args else None
    if checkpoint_to_load is None and os.path.exists(config.checkpoint_path):
        checkpoint_to_load = config.checkpoint_path
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        # Use model-appropriate checkpoint loader
        if config.model_type == "pi0":
            start_episode, wandb_run_id = load_reinflow_pi0_checkpoint(rl_policy, checkpoint_to_load, str(device))
        else:
            start_episode, wandb_run_id = load_reinflow_checkpoint(rl_policy, checkpoint_to_load, str(device))

    if config.model_type == "smolvla":
        _print_smolvla_startup_summary(
            config,
            rl_policy,
            start_mode="resume-rl" if checkpoint_to_load and os.path.exists(checkpoint_to_load) else "fresh-rl",
            resume_path=checkpoint_to_load if checkpoint_to_load and os.path.exists(checkpoint_to_load) else None,
        )

    rl_policy.logprob_eval_microbatch_size = config.logprob_eval_microbatch_size
    rl_policy.critic_backprop_into_policy = config.critic_backprop_into_policy
    
    # Initialize wandb with PPO config
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            id=wandb_run_id,
            resume="allow",
            config={
                "model_type": config.model_type,
                "policy_lr": config.policy_lr,
                "critic_lr": config.critic_lr,
                "gamma": config.gamma,
                "gae_lambda": config.gae_lambda,
                "clip_epsilon": config.clip_epsilon,
                "num_ppo_epochs": config.num_ppo_epochs,
                "minibatch_size": config.minibatch_size,
                "target_kl": config.target_kl,
                "logprob_eval_microbatch_size": config.logprob_eval_microbatch_size,
                "critic_backprop_into_policy": config.critic_backprop_into_policy,
                "trainable_scope": _get_trainable_scope_label(config),
                "actor_lr_start": config.policy_lr,
                "actor_lr_end": config.policy_lr * 0.1,
                "distance_penalty_scale": config.distance_penalty_scale,
                "horizontal_progress_scale": config.horizontal_progress_scale,
                "vertical_approach_scale": config.vertical_approach_scale,
                "approach_closeness_scale": config.approach_closeness_scale,
                "alignment_reward_cap": config.alignment_reward_cap,
                "near_contact_bonus": config.near_contact_bonus,
                "contact_entry_bonus": config.contact_entry_bonus,
                "contact_persistence_reward": config.contact_persistence_reward,
                "hover_stall_threshold": config.hover_stall_threshold,
                "hover_penalty": config.hover_penalty,
                "bilateral_grasp_bonus": config.bilateral_grasp_bonus,
                "grasp_persistence_reward": config.grasp_persistence_reward,
                "slip_penalty_contact": config.slip_penalty_contact,
                "slip_penalty_grasp": config.slip_penalty_grasp,
                "block_displacement_penalty_scale": config.block_displacement_penalty_scale,
                "num_denoising_steps": config.num_denoising_steps,
                "chunks_per_episode": config.chunks_per_episode,
                "train_action_head": config.train_action_head,
                "train_time_mlp": config.train_time_mlp,
                "train_full_expert": config.train_full_expert,
                "train_noise_head": config.train_noise_head,
                "train_critic": config.train_critic,
                "num_parallel_envs": config.num_parallel_envs,
                "training_mode": "ppo-on-policy",
                "gradient_checkpointing": config.pi0_gradient_checkpointing if config.model_type == "pi0" else False,
                "base_policy_source": getattr(rl_policy, "base_policy_source", config.pretrained_path),
                "base_policy_source_type": getattr(rl_policy, "base_policy_source_type", "unknown"),
                "normalization_mode": getattr(rl_policy, "normalization_mode", "unknown"),
                "state_action_frame": getattr(rl_policy, "state_action_frame", "unknown"),
            },
        )
        wandb_run_id = wandb.run.id
    
    # Separate optimizers for actor and critic
    policy_params = list(rl_policy.get_trainable_params())
    critic_params = list(rl_policy.get_critic_params())
    
    policy_optimizer = torch.optim.Adam(policy_params, lr=config.policy_lr)
    critic_optimizer = torch.optim.Adam(critic_params, lr=config.critic_lr)
    
    # Learning rate schedulers with warmup (paper Table 9b)
    total_batches = config.num_episodes // num_envs
    
    # Policy scheduler: linear warmup + cosine annealing
    policy_warmup = torch.optim.lr_scheduler.LinearLR(
        policy_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.lr_warmup_iterations
    )
    policy_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer,
        T_max=max(1, total_batches - config.lr_warmup_iterations),
        eta_min=config.policy_lr * 0.1
    )
    policy_scheduler = torch.optim.lr_scheduler.SequentialLR(
        policy_optimizer,
        schedulers=[policy_warmup, policy_cosine],
        milestones=[config.lr_warmup_iterations]
    )
    
    # Critic scheduler: linear warmup + cosine annealing
    critic_warmup = torch.optim.lr_scheduler.LinearLR(
        critic_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.lr_warmup_iterations
    )
    critic_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer,
        T_max=max(1, total_batches - config.lr_warmup_iterations),
        eta_min=config.critic_lr * 0.1
    )
    critic_scheduler = torch.optim.lr_scheduler.SequentialLR(
        critic_optimizer,
        schedulers=[critic_warmup, critic_cosine],
        milestones=[config.lr_warmup_iterations]
    )
    
    print(f"\n{'='*60}")
    print(f"Starting ReinFlow Training (PPO, Parallel)")
    print(f"{'='*60}")
    print(f"Instruction: '{config.instruction}'")
    print(f"Parallel environments: {num_envs}")
    print(f"Chunks per episode: {config.chunks_per_episode}")
    print(f"Denoising steps: {config.num_denoising_steps}")
    print(f"Policy LR: {config.policy_lr} -> {config.policy_lr * 0.1}")
    print(f"Critic LR: {config.critic_lr} -> {config.critic_lr * 0.1}")
    print(f"LR warmup iterations: {config.lr_warmup_iterations}")
    print(f"PPO epochs: {config.num_ppo_epochs}")
    print(f"Mini-batch size: {config.minibatch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.minibatch_size * config.gradient_accumulation_steps}")
    print(f"Clip epsilon: {config.clip_epsilon}")
    print(f"GAE lambda: {config.gae_lambda}")
    print(f"Target KL: {config.target_kl}")
    print(f"Entropy coeff: {config.entropy_coeff}")
    print(f"Critic warmup iters: {config.critic_warmup_iters}")
    print(f"Training mode: PPO ON-POLICY")
    print(f"{'='*60}\n")
    if config.model_type == "smolvla" and num_envs > 5:
        print("  [ReinFlow] Warning: SmolVLA on a 14.6 GB GPU is typically stable at <= 5 parallel envs.")
    
    total_episodes = start_episode  # Initialize with checkpoint episode count when resuming
    reward_ema20 = None
    recent_epoch1_early_stops = deque(maxlen=5)
    epoch1_warning_active = False
    
    # ===== CRITIC WARMUP (paper Appendix D.2) =====
    # Train critic for a few iterations before updating actor
    # This ensures value estimates are reasonable before policy gradients
    # SKIP if resuming from checkpoint (critic already trained)
    if config.critic_warmup_iters > 0 and start_episode == 0:
        print(f"\n[Critic Warmup] Training critic for {config.critic_warmup_iters} iterations...")
        for warmup_iter in range(config.critic_warmup_iters):
            vec_env.reset_all()
            
            # Collect one episode worth of data
            warmup_observations = []
            warmup_rewards = []
            
            for chunk_idx in range(config.chunks_per_episode):
                obs_dict = vec_env.get_batched_observations(device)
                observation = prepare_batched_observation(
                    obs_dict, config.instruction, device, rl_policy, num_envs
                )
                warmup_observations.append(observation)
                
                with torch.no_grad():
                    action_chunks, _, _ = rl_policy.forward_batched_with_trajectory(observation)
                action_chunks_np = action_chunks.detach().cpu().numpy()
                # Unnormalize actions based on model type
                action_chunks_radians = np.stack([
                    np.stack([unnormalize_action_for_vla(a, config.model_type, postprocessor) for a in chunk])
                    for chunk in action_chunks_np
                ])
                
                chunk_rewards, dones, *_ = vec_env.step_all_chunk(
                    action_chunks_radians, config.steps_per_action
                )
                warmup_rewards.extend(chunk_rewards.tolist())
                
                if dones.all():
                    break
            
            # Only update critic (not actor)
            if len(warmup_observations) > 0:
                # Stack observations and compute critic loss
                obs_batch = warmup_observations[0]  # Use first observation
                rewards_tensor = torch.tensor(warmup_rewards[:num_envs], device=device, dtype=torch.float32)
                
                values = rl_policy.get_value(obs_batch)
                critic_loss = F.mse_loss(values, rewards_tensor)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_params, max_norm=config.grad_clip_norm)
                critic_optimizer.step()
                
                print(f"  [Warmup {warmup_iter+1}/{config.critic_warmup_iters}] Critic loss: {critic_loss.item():.4f}")
        
        print(f"[Critic Warmup] Complete!\n")
    elif start_episode > 0:
        print(f"\n[Critic Warmup] Skipping warmup (resuming from episode {start_episode})\n")
    
    try:
        for episode_batch in range(config.num_episodes // num_envs):
            batch_start_time = time.time()
            
            # Update noise bounds with decay schedule (paper Appendix D)
            current_episode = episode_batch * num_envs
            sigma_min, sigma_max = get_noise_bounds(current_episode, config.num_episodes, config)
            rl_policy.base.model.sigma_min = sigma_min
            rl_policy.base.model.sigma_max = sigma_max
            
            # Reset all environments
            vec_env.reset_all()
            episode_rewards = np.zeros(num_envs)
            episode_contacts = np.zeros(num_envs, dtype=int)
            episode_grasps = np.zeros(num_envs, dtype=int)
            episode_sustained = np.zeros(num_envs, dtype=int)
            episode_height_aligned = np.zeros(num_envs, dtype=int)
            episode_contact_entries = np.zeros(num_envs, dtype=int)
            episode_grasp_persistent = np.zeros(num_envs, dtype=int)
            episode_hover_stall = np.zeros(num_envs, dtype=int)
            episode_slips = np.zeros(num_envs, dtype=int)
            episode_lift_progress = np.zeros(num_envs, dtype=float)
            episode_block_displacement = np.zeros(num_envs, dtype=float)
            episode_approach_reward = np.zeros(num_envs, dtype=float)
            episode_alignment_reward = np.zeros(num_envs, dtype=float)
            episode_near_contact = np.zeros(num_envs, dtype=int)
            episode_contact_after_alignment = np.zeros(num_envs, dtype=int)
            episode_horizontal_progress = np.zeros(num_envs, dtype=float)
            episode_vertical_approach = np.zeros(num_envs, dtype=float)
            episode_contact_losses = np.zeros(num_envs, dtype=int)
            episode_grasp_losses = np.zeros(num_envs, dtype=int)
            
            # Collect data for this batch (on-policy), preserving env/chunk identity.
            rollout = ParallelRolloutBatch(
                trajectories=[[None for _ in range(config.chunks_per_episode)] for _ in range(num_envs)],
                observations=[[None for _ in range(config.chunks_per_episode)] for _ in range(num_envs)],
                rewards=np.zeros((num_envs, config.chunks_per_episode), dtype=np.float32),
                dones=np.zeros((num_envs, config.chunks_per_episode), dtype=bool),
                valid=np.zeros((num_envs, config.chunks_per_episode), dtype=bool),
            )
            
            # Execute multiple chunks per episode (getting fresh observations!)
            for chunk_idx in range(config.chunks_per_episode):
                active_envs = ~vec_env.dones.copy()
                if not active_envs.any():
                    break

                # Get FRESH observation for this chunk
                obs_dict = vec_env.get_batched_observations(device)
                observation = prepare_batched_observation(
                    obs_dict, config.instruction, device, rl_policy, num_envs
                )
                
                # Forward pass with trajectory storage
                action_chunks, trajectory, sigmas = rl_policy.forward_batched_with_trajectory(observation)
                
                
                # Unnormalize actions for execution based on model type
                action_chunks_np = action_chunks.detach().cpu().numpy()
                action_chunks_radians = np.stack([
                    np.stack([unnormalize_action_for_vla(a, config.model_type, postprocessor) for a in chunk])
                    for chunk in action_chunks_np
                ])
                
                # Execute chunk and get rewards
                (
                    chunk_rewards,
                    dones,
                    chunk_contacts,
                    chunk_grasps,
                    chunk_sustained,
                    chunk_height_aligned,
                    chunk_contact_entries,
                    chunk_grasp_persistent,
                    chunk_lift_progress,
                    chunk_hover_stall,
                    chunk_slips,
                    chunk_block_displacement,
                    chunk_approach_reward,
                    chunk_alignment_reward,
                    chunk_near_contact,
                    chunk_contact_after_alignment,
                    chunk_horizontal_progress,
                    chunk_vertical_approach,
                    chunk_contact_losses,
                    chunk_grasp_losses,
                ) = vec_env.step_all_chunk(
                    action_chunks_radians, config.steps_per_action
                )
                episode_rewards += chunk_rewards
                episode_contacts += chunk_contacts
                episode_grasps += chunk_grasps
                episode_sustained += chunk_sustained
                episode_height_aligned += chunk_height_aligned
                episode_contact_entries += chunk_contact_entries
                episode_grasp_persistent += chunk_grasp_persistent
                episode_lift_progress += chunk_lift_progress
                episode_hover_stall += chunk_hover_stall
                episode_slips += chunk_slips
                episode_block_displacement += chunk_block_displacement
                episode_approach_reward += chunk_approach_reward
                episode_alignment_reward += chunk_alignment_reward
                episode_near_contact += chunk_near_contact
                episode_contact_after_alignment += chunk_contact_after_alignment
                episode_horizontal_progress += chunk_horizontal_progress
                episode_vertical_approach += chunk_vertical_approach
                episode_contact_losses += chunk_contact_losses
                episode_grasp_losses += chunk_grasp_losses
                
                # Store data for on-policy update, indexed by [env][chunk].
                # trajectory is list of K+1 tensors, each (num_envs, chunk, action_dim)
                traj_tensor = torch.stack(trajectory, dim=1).detach()  # (num_envs, K+1, chunk, action)

                for i in range(num_envs):
                    if not active_envs[i]:
                        continue
                    rollout.trajectories[i][chunk_idx] = traj_tensor[i]
                    rollout.observations[i][chunk_idx] = {k: v[i:i+1] for k, v in observation.items()}
                    rollout.rewards[i, chunk_idx] = chunk_rewards[i]
                    rollout.dones[i, chunk_idx] = bool(dones[i])
                    rollout.valid[i, chunk_idx] = True
                
                # Early termination if all done
                if dones.all():
                    break
            
            # Track episode rewards
            episode_rewards_history.extend(episode_rewards.tolist())
            total_episodes += num_envs
            
            # ===== PPO UPDATE WITH MINI-BATCHING =====
            if rollout.valid.any():
                _validate_parallel_rollout_masks(rollout.valid, rollout.dones)

                # Batch value inference for all valid rollout observations.
                value_observations = _flatten_valid_observations(rollout.observations, rollout.valid)
                if not value_observations:
                    raise AssertionError("Parallel rollout has valid slots but no flattened observations")

                with torch.no_grad():
                    flat_values = rl_policy.get_value(value_observations)

                    if (~vec_env.dones).any():
                        final_obs_dict = vec_env.get_batched_observations(device)
                        final_observation = prepare_batched_observation(
                            final_obs_dict, config.instruction, device, rl_policy, num_envs
                        )
                        final_bootstrap_values = rl_policy.get_value(final_observation)
                    else:
                        final_bootstrap_values = torch.zeros(num_envs, device=device, dtype=torch.float32)

                value_grid = torch.zeros((num_envs, config.chunks_per_episode), device=device, dtype=torch.float32)
                next_value_grid = torch.zeros_like(value_grid)
                advantage_grid = torch.zeros_like(value_grid)
                return_grid = torch.zeros_like(value_grid)
                source_grid = [["invalid" for _ in range(config.chunks_per_episode)] for _ in range(num_envs)]
                rewards_grid = torch.from_numpy(rollout.rewards).to(device=device, dtype=torch.float32)
                dones_grid = torch.from_numpy(rollout.dones.astype(np.float32)).to(device=device, dtype=torch.float32)

                flat_index_map = []
                flat_cursor = 0
                for env_idx in range(num_envs):
                    for chunk_idx in range(config.chunks_per_episode):
                        if not rollout.valid[env_idx, chunk_idx]:
                            continue
                        value_grid[env_idx, chunk_idx] = flat_values[flat_cursor]
                        flat_index_map.append((env_idx, chunk_idx))
                        flat_cursor += 1

                if flat_cursor != int(rollout.valid.sum()):
                    raise AssertionError(
                        f"Flattened sample count mismatch: got {flat_cursor}, expected {int(rollout.valid.sum())}"
                    )

                # Build next-values and run GAE per environment trajectory.
                for env_idx in range(num_envs):
                    valid_indices = np.flatnonzero(rollout.valid[env_idx])
                    if len(valid_indices) == 0:
                        continue

                    for pos, chunk_idx in enumerate(valid_indices):
                        chunk_idx = int(chunk_idx)
                        if rollout.dones[env_idx, chunk_idx]:
                            next_value_grid[env_idx, chunk_idx] = 0.0
                            source_grid[env_idx][chunk_idx] = "terminal"
                        elif pos + 1 < len(valid_indices):
                            next_chunk_idx = int(valid_indices[pos + 1])
                            next_value_grid[env_idx, chunk_idx] = value_grid[env_idx, next_chunk_idx]
                            source_grid[env_idx][chunk_idx] = f"env:{env_idx}:chunk:{next_chunk_idx}"
                        else:
                            next_value_grid[env_idx, chunk_idx] = final_bootstrap_values[env_idx]
                            source_grid[env_idx][chunk_idx] = "post_rollout"

                    env_rewards = rewards_grid[env_idx, valid_indices]
                    env_values = value_grid[env_idx, valid_indices]
                    env_next_values = next_value_grid[env_idx, valid_indices]
                    env_dones = dones_grid[env_idx, valid_indices]

                    env_advantages, env_returns = compute_gae(
                        env_rewards,
                        env_values,
                        env_next_values,
                        env_dones,
                        gamma=config.gamma,
                        gae_lambda=config.gae_lambda,
                    )
                    advantage_grid[env_idx, valid_indices] = env_advantages
                    return_grid[env_idx, valid_indices] = env_returns

                # Validate bootstrap sources stay within the same env trajectory.
                for env_idx, chunk_idx in flat_index_map:
                    source = source_grid[env_idx][chunk_idx]
                    if rollout.dones[env_idx, chunk_idx]:
                        if source != "terminal":
                            raise AssertionError(
                                f"Env {env_idx} chunk {chunk_idx} should bootstrap from terminal zero, got {source}"
                            )
                    elif source not in ("post_rollout", f"env:{env_idx}:chunk:{chunk_idx + 1}"):
                        raise AssertionError(
                            f"Env {env_idx} chunk {chunk_idx} bootstrapped from invalid source {source}"
                        )

                # Flatten valid rollout tensors for PPO.
                all_trajectories = torch.stack(
                    [rollout.trajectories[env_idx][chunk_idx] for env_idx, chunk_idx in flat_index_map],
                    dim=0,
                )
                all_observations = _flatten_valid_observations(rollout.observations, rollout.valid)
                if not all_observations:
                    raise AssertionError("Parallel rollout flattening produced no PPO observations")

                all_rewards = torch.stack([rewards_grid[env_idx, chunk_idx] for env_idx, chunk_idx in flat_index_map])
                all_dones = torch.stack([dones_grid[env_idx, chunk_idx] for env_idx, chunk_idx in flat_index_map])
                all_values = torch.stack([value_grid[env_idx, chunk_idx] for env_idx, chunk_idx in flat_index_map])
                all_next_values = torch.stack([next_value_grid[env_idx, chunk_idx] for env_idx, chunk_idx in flat_index_map])
                advantages = torch.stack([advantage_grid[env_idx, chunk_idx] for env_idx, chunk_idx in flat_index_map])
                returns = torch.stack([return_grid[env_idx, chunk_idx] for env_idx, chunk_idx in flat_index_map])
                old_values = all_values.clone()
                batch_size = all_trajectories.shape[0]

                # Normalize advantages after per-env GAE is complete.
                if batch_size > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Compute old log probabilities (detached for PPO ratio).
                with torch.no_grad():
                    old_log_probs = _compute_policy_log_probs(rl_policy, all_trajectories, all_observations)
                    recomputed_log_probs = _compute_policy_log_probs(rl_policy, all_trajectories, all_observations)
                pre_update_metrics = _compute_logprob_shift_metrics(
                    old_log_probs, recomputed_log_probs, clip_epsilon=config.clip_epsilon
                )
                _validate_pre_update_logprob_invariant(pre_update_metrics)
                
                # PPO epochs with mini-batching
                kl_early_stop = False
                epoch_policy_losses = []
                epoch_critic_losses = []
                post_update_metric_history = []
                
                # Gradient norm tracking for diagnostics
                last_policy_grad_norm = 0.0
                last_critic_grad_norm = 0.0
                policy_grad_clipped = False
                critic_grad_clipped = False
                actor_delta_l2_history = []
                actor_delta_max_abs_history = []
                epoch1_early_stop = False
                
                for epoch in range(config.num_ppo_epochs):
                    if kl_early_stop:
                        break
                    
                    # Shuffle indices for mini-batching
                    indices = torch.randperm(batch_size, device=device)
                    
                    # Gradient accumulation setup (paper Appendix D)
                    accumulation_counter = 0
                    policy_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    
                    for start in range(0, batch_size, config.minibatch_size):
                        end = min(start + config.minibatch_size, batch_size)
                        mb_indices = indices[start:end]
                        
                        
                        # Get mini-batch data
                        mb_trajectories = all_trajectories[mb_indices]
                        mb_observations = {k: v[mb_indices] for k, v in all_observations.items()}
                        mb_advantages = advantages[mb_indices]
                        mb_returns = returns[mb_indices]
                        mb_old_log_probs = old_log_probs[mb_indices]
                        mb_old_values = old_values[mb_indices]
                        
                        # Compute PPO loss with entropy regularization
                        policy_loss, critic_loss, loss_info = compute_ppo_loss(
                            rl_policy,
                            mb_trajectories,
                            mb_observations,
                            mb_old_log_probs,
                            mb_advantages,
                            mb_returns,
                            clip_epsilon=config.clip_epsilon,
                            value_clip_epsilon=config.value_clip_epsilon,
                            old_values=mb_old_values,
                            entropy_coeff=config.entropy_coeff,
                        )
                        
                        # Combine losses and do single backward pass to avoid graph issues
                        # (policy_loss and critic_loss share computation graph through observations)
                        total_loss = policy_loss + critic_loss
                        
                        # Scale loss by accumulation steps for gradient averaging
                        scaled_loss = total_loss / config.gradient_accumulation_steps
                        scaled_loss.backward()
                        
                        accumulation_counter += 1
                        epoch_policy_losses.append(policy_loss.item())
                        epoch_critic_losses.append(critic_loss.item())
                        
                        # Step optimizer only after accumulating enough gradients
                        if accumulation_counter % config.gradient_accumulation_steps == 0:
                            # Capture gradient norms before clipping for diagnostics
                            last_critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, max_norm=config.grad_clip_norm).item()
                            critic_grad_clipped = last_critic_grad_norm > config.grad_clip_norm
                            critic_optimizer.step()
                            last_policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy_params, max_norm=config.grad_clip_norm).item()
                            policy_grad_clipped = last_policy_grad_norm > config.grad_clip_norm
                            policy_optimizer.step()
                            actor_delta_metrics = _compute_actor_delta_metrics(policy_optimizer)
                            actor_delta_l2_history.append(actor_delta_metrics["delta_l2"])
                            actor_delta_max_abs_history.append(actor_delta_metrics["delta_max_abs"])
                            
                            policy_optimizer.zero_grad()
                            critic_optimizer.zero_grad()

                            with torch.no_grad():
                                updated_log_probs = _compute_policy_log_probs(
                                    rl_policy, mb_trajectories, mb_observations
                                )
                            post_update_metrics = _compute_logprob_shift_metrics(
                                mb_old_log_probs,
                                updated_log_probs,
                                clip_epsilon=config.clip_epsilon,
                            )
                            post_update_metric_history.append(post_update_metrics)

                            if post_update_metrics['kl_div'] > config.target_kl * 1.5:
                                print(
                                    f"  [KL Early Stop] Epoch {epoch+1}, "
                                    f"post-update KL={post_update_metrics['kl_div']:.4f} > {config.target_kl * 1.5:.4f}"
                                )
                                if epoch == 0:
                                    epoch1_early_stop = True
                                kl_early_stop = True
                                break
                    
                    # Handle remaining gradients if batch doesn't divide evenly
                    if accumulation_counter % config.gradient_accumulation_steps != 0:
                        last_critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, max_norm=config.grad_clip_norm).item()
                        critic_grad_clipped = last_critic_grad_norm > config.grad_clip_norm
                        critic_optimizer.step()
                        last_policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy_params, max_norm=config.grad_clip_norm).item()
                        policy_grad_clipped = last_policy_grad_norm > config.grad_clip_norm
                        policy_optimizer.step()
                        actor_delta_metrics = _compute_actor_delta_metrics(policy_optimizer)
                        actor_delta_l2_history.append(actor_delta_metrics["delta_l2"])
                        actor_delta_max_abs_history.append(actor_delta_metrics["delta_max_abs"])

                        with torch.no_grad():
                            updated_log_probs = _compute_policy_log_probs(
                                rl_policy, mb_trajectories, mb_observations
                            )
                        post_update_metrics = _compute_logprob_shift_metrics(
                            mb_old_log_probs,
                            updated_log_probs,
                            clip_epsilon=config.clip_epsilon,
                        )
                        post_update_metric_history.append(post_update_metrics)

                        if post_update_metrics['kl_div'] > config.target_kl * 1.5:
                            print(
                                f"  [KL Early Stop] Epoch {epoch+1}, "
                                f"post-update KL={post_update_metrics['kl_div']:.4f} > {config.target_kl * 1.5:.4f}"
                            )
                            if epoch == 0:
                                epoch1_early_stop = True
                            kl_early_stop = True
                
                # Step LR schedulers
                policy_scheduler.step()
                critic_scheduler.step()
                
                # Aggregate loss info
                avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0.0
                avg_critic_loss = np.mean(epoch_critic_losses) if epoch_critic_losses else 0.0
                post_update_summary = _aggregate_logprob_shift_metrics(post_update_metric_history)
                avg_kl_div = post_update_summary["kl_div"]
                avg_actor_delta_l2 = float(np.mean(actor_delta_l2_history)) if actor_delta_l2_history else 0.0
                max_actor_delta_abs = float(np.max(actor_delta_max_abs_history)) if actor_delta_max_abs_history else 0.0
                
            else:
                loss_info = {'advantage_mean': 0, 'value_mean': 0, 'log_prob_mean': 0, 'kl_div': 0, 'clip_fraction': 0,
                             'ratio_mean': 0, 'ratio_std': 0, 'ratio_min': 0, 'ratio_max': 0,
                             'advantage_std': 0, 'old_log_prob_mean': 0, 'return_mean': 0}
                pre_update_metrics = _default_logprob_shift_metrics()
                post_update_summary = _default_logprob_shift_metrics()
                avg_policy_loss = 0.0
                avg_critic_loss = 0.0
                avg_kl_div = 0.0
                last_policy_grad_norm = 0.0
                last_critic_grad_norm = 0.0
                policy_grad_clipped = False
                critic_grad_clipped = False
                avg_actor_delta_l2 = 0.0
                max_actor_delta_abs = 0.0
                epoch1_early_stop = False
                # Set defaults for metrics computed from tensors
                all_rewards = torch.tensor([0.0], device=device)
                all_values = torch.tensor([0.0], device=device)
                returns = torch.tensor([0.0], device=device)
                advantages = torch.tensor([0.0], device=device)
                action_chunks = torch.tensor([[[0.0]]], device=device)
            
            batch_time = time.time() - batch_start_time
            
            # Logging
            avg_reward = np.mean(episode_rewards)
            reward_ema20 = _update_ema(reward_ema20, avg_reward, window=20)
            recent_epoch1_early_stops.append(1 if epoch1_early_stop else 0)
            should_warn_epoch1 = len(recent_epoch1_early_stops) == 5 and sum(recent_epoch1_early_stops) > 3
            if should_warn_epoch1 and not epoch1_warning_active:
                print("  [Training Warning] More than 3 of the last 5 batches early-stopped in epoch 1. Actor updates are still too aggressive for reward growth.")
            epoch1_warning_active = should_warn_epoch1
            current_lr = policy_scheduler.get_last_lr()[0]
            print(f"Batch {episode_batch+1:4d} ({total_episodes:5d} eps) | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"EMA20: {reward_ema20:7.2f} | "
                  f"KL: {avg_kl_div:.4f} | "
                  f"RatioMax: {post_update_summary['ratio_max']:.3f} | "
                  f"ClipFrac: {post_update_summary['clip_fraction']:.2f} | "
                  f"LR: {current_lr:.2e} | "
                  f"Time: {batch_time:.1f}s")
            
            # Compute additional diagnostic metrics for wandb
            with torch.no_grad():
                # Critic diagnostics
                returns_var = returns.var() + 1e-8
                explained_var = (1 - (returns - all_values).var() / returns_var).item()
                
                # Action diagnostics (from last forward pass)
                action_saturation_low = (action_chunks < -0.95).float().mean().item()
                action_saturation_high = (action_chunks > 0.95).float().mean().item()
                
                # Log prob per dimension (action_dim=6, chunk_size=50)
                log_prob_per_dim = loss_info.get('log_prob_mean', 0) / (6 * 50)
                log_prob_drift = loss_info.get('log_prob_mean', 0) - loss_info.get('old_log_prob_mean', 0)
            
            # Log to wandb
            if config.wandb_enabled:
                log_dict = {
                    # Basic info
                    "batch": episode_batch + 1,
                    "episodes_total": total_episodes,
                    "time/batch_seconds": batch_time,
                    
                    # Reward metrics (5)
                    "reward/batch_avg": avg_reward,
                    "reward/ema20": reward_ema20,
                    "reward/batch_min": np.min(episode_rewards),
                    "reward/batch_max": np.max(episode_rewards),
                    "reward/std": all_rewards.std().item(),
                    "reward/positive_fraction": (all_rewards > 0).float().mean().item(),
                    
                    # Contact metrics (3)
                    "reward/contact_count_avg": episode_contacts.mean(),
                    "reward/contact_count_max": episode_contacts.max(),
                    "reward/contact_rate": episode_contacts.sum() / (num_envs * config.chunks_per_episode * 50),
                    # Grasp metrics (3)
                    "reward/grasp_count_avg": episode_grasps.mean(),
                    "reward/grasp_count_max": episode_grasps.max(),
                    "reward/grasp_rate": episode_grasps.sum() / (num_envs * config.chunks_per_episode * 50),
                    # Sustained contact metrics (3)
                    "reward/sustained_count_avg": episode_sustained.mean(),
                    "reward/sustained_count_max": episode_sustained.max(),
                    "reward/sustained_contact_rate": episode_sustained.sum() / (num_envs * config.chunks_per_episode * 50),
                    # Height alignment metrics (3)
                    "reward/height_align_count_avg": episode_height_aligned.mean(),
                    "reward/height_align_count_max": episode_height_aligned.max(),
                    "reward/height_align_rate": episode_height_aligned.sum() / (num_envs * config.chunks_per_episode * 50),
                    "reward/contact_entry_count_avg": episode_contact_entries.mean(),
                    "reward/contact_entry_rate": episode_contact_entries.sum() / (num_envs * config.chunks_per_episode * 50),
                    "reward/grasp_persistence_count_avg": episode_grasp_persistent.mean(),
                    "reward/grasp_persistence_rate": episode_grasp_persistent.sum() / (num_envs * config.chunks_per_episode * 50),
                    "reward/approach_reward_mean": episode_approach_reward.mean() / max(1, config.chunks_per_episode * 50),
                    "reward/alignment_reward_mean": episode_alignment_reward.mean() / max(1, config.chunks_per_episode * 50),
                    "reward/near_contact_count_avg": episode_near_contact.mean(),
                    "reward/near_contact_rate": episode_near_contact.sum() / (num_envs * config.chunks_per_episode * 50),
                    "reward/contact_after_alignment_count_avg": episode_contact_after_alignment.mean(),
                    "reward/contact_after_alignment_rate": episode_contact_after_alignment.sum() / (num_envs * config.chunks_per_episode * 50),
                    "reward/horizontal_progress_mean": episode_horizontal_progress.mean() / max(1, config.chunks_per_episode * 50),
                    "reward/vertical_approach_mean": episode_vertical_approach.mean() / max(1, config.chunks_per_episode * 50),
                    "reward/lift_progress_mean": episode_lift_progress.mean() / max(1, config.chunks_per_episode * 50),
                    "reward/hover_stall_count_avg": episode_hover_stall.mean(),
                    "reward/hover_stall_rate": episode_hover_stall.sum() / (num_envs * config.chunks_per_episode * 50),
                    "reward/slip_count_avg": episode_slips.mean(),
                    "reward/slip_count_total": episode_slips.sum(),
                    "reward/contact_loss_count_avg": episode_contact_losses.mean(),
                    "reward/contact_loss_count_total": episode_contact_losses.sum(),
                    "reward/grasp_loss_count_avg": episode_grasp_losses.mean(),
                    "reward/grasp_loss_count_total": episode_grasp_losses.sum(),
                    "reward/block_displacement_mean": episode_block_displacement.mean() / max(1, config.chunks_per_episode * 50),
                    
                    # Loss metrics
                    "loss/policy": avg_policy_loss,
                    "loss/critic": avg_critic_loss,
                    "loss/entropy": loss_info.get('entropy', 0),
                    
                    # Critic/Value metrics (7)
                    "critic/value_mean": loss_info.get('value_mean', 0),
                    "critic/value_std": all_values.std().item(),
                    "critic/value_min": all_values.min().item(),
                    "critic/value_max": all_values.max().item(),
                    "critic/return_mean": loss_info.get('return_mean', 0),
                    "critic/return_std": returns.std().item(),
                    "critic/explained_variance": explained_var,
                    
                    # Advantage metrics (4)
                    "advantage/mean": loss_info.get('advantage_mean', 0),
                    "advantage/std": loss_info.get('advantage_std', 0),
                    "advantage/min": advantages.min().item(),
                    "advantage/max": advantages.max().item(),
                    
                    # PPO ratio metrics (4)
                    "ppo/ratio_mean": loss_info.get('ratio_mean', 0),
                    "ppo/ratio_std": loss_info.get('ratio_std', 0),
                    "ppo/ratio_min": loss_info.get('ratio_min', 0),
                    "ppo/ratio_max": loss_info.get('ratio_max', 0),
                    
                    # Log probability metrics (4)
                    "logprob/new_mean": loss_info.get('log_prob_mean', 0),
                    "logprob/old_mean": loss_info.get('old_log_prob_mean', 0),
                    "logprob/per_dimension": log_prob_per_dim,
                    "logprob/drift": log_prob_drift,
                    "debug/pre_update_kl": pre_update_metrics.get('kl_div', 0.0),
                    "debug/pre_update_ratio_mean": pre_update_metrics.get('ratio_mean', 1.0),
                    "debug/pre_update_logprob_abs_mean": pre_update_metrics.get('logprob_abs_mean', 0.0),
                    "debug/pre_update_logprob_abs_max": pre_update_metrics.get('logprob_abs_max', 0.0),
                    
                    # Action metrics (4)
                    "actions/mean": action_chunks.mean().item(),
                    "actions/std": action_chunks.std().item(),
                    "actions/saturation_low": action_saturation_low,
                    "actions/saturation_high": action_saturation_high,
                    
                    # Gradient metrics (4)
                    "gradients/policy_norm": last_policy_grad_norm,
                    "gradients/actor_norm": last_policy_grad_norm,
                    "gradients/critic_norm": last_critic_grad_norm,
                    "gradients/policy_clipped": float(policy_grad_clipped),
                    "gradients/actor_clipped": float(policy_grad_clipped),
                    "gradients/critic_clipped": float(critic_grad_clipped),
                    "updates/actor_delta_l2": avg_actor_delta_l2,
                    "updates/actor_delta_max_abs": max_actor_delta_abs,
                    
                    # Training dynamics
                    "training/kl_divergence": avg_kl_div,
                    "training/post_update_kl": avg_kl_div,
                    "training/post_update_ratio_mean": post_update_summary.get('ratio_mean', 1.0),
                    "training/post_update_ratio_max": post_update_summary.get('ratio_max', 1.0),
                    "training/post_update_clip_fraction": post_update_summary.get('clip_fraction', 0.0),
                    "training/post_update_logprob_abs_mean": post_update_summary.get('logprob_abs_mean', 0.0),
                    "training/clip_fraction": post_update_summary.get('clip_fraction', 0.0),
                    "training/learning_rate": current_lr,
                    "training/effective_batch_size": config.minibatch_size * config.gradient_accumulation_steps,
                    "training/sigma_min": sigma_min,
                    "training/sigma_max": sigma_max,
                    "training/epoch1_early_stop": float(epoch1_early_stop),
                    "training/recent_epoch1_early_stop_count": float(sum(recent_epoch1_early_stops)),
                }
                wandb.log(log_dict)
            
            # Save checkpoint periodically
            if (episode_batch + 1) % (config.save_interval // num_envs + 1) == 0:
                if config.model_type == "pi0":
                    save_reinflow_pi0_checkpoint(
                        rl_policy, total_episodes - 1, episode_rewards_history, 
                        config.checkpoint_path, wandb_run_id
                    )
                else:
                    save_reinflow_checkpoint(
                        rl_policy, total_episodes - 1, episode_rewards_history, 
                        config.checkpoint_path, wandb_run_id
                    )
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        vec_env.close()
        if config.wandb_enabled:
            wandb.finish()
    
    # Save final checkpoint
    if config.model_type == "pi0":
        save_reinflow_pi0_checkpoint(
            rl_policy, total_episodes - 1, episode_rewards_history, 
            config.checkpoint_path, wandb_run_id
        )
    else:
        save_reinflow_checkpoint(
            rl_policy, total_episodes - 1, episode_rewards_history, 
            config.checkpoint_path, wandb_run_id
        )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total episodes: {len(episode_rewards_history)}")
    if episode_rewards_history:
        print(f"Best reward: {max(episode_rewards_history):.2f}")
        print(f"Final avg reward (last 100): {np.mean(episode_rewards_history[-100:]):.2f}")
    
    return rl_policy, episode_rewards_history


# ===== Sequential Training Loop (for M1/CPU) =====

def train_sequential(config, args, device):
    """
    Sequential ON-POLICY PPO training loop with GAE, mini-batching, and KL early stopping.
    
    Key features (paper-faithful):
    1. PPO clipped surrogate objective
    2. GAE (Generalized Advantage Estimation) for better credit assignment
    3. Multiple epochs with mini-batching for sample efficiency
    4. KL divergence monitoring with early stopping
    5. Learning rate scheduling
    
    Supports both SmolVLA and Pi0 models.
    """
    # Load MuJoCo environment
    print(f"\nLoading MuJoCo model from {config.model_path}")
    m = mujoco.MjModel.from_xml_path(config.model_path)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=256, width=256)
    
    # Setup ReinFlow policy with critic and processors
    print("\n" + "="*60)
    print(f"Setting up ReinFlow {config.model_type.upper()} (PPO, On-Policy)")
    print("="*60)
    
    # Select appropriate setup function based on model type.
    preprocessor = None
    postprocessor = None
    
    if config.model_type == "pi0":
        rl_policy, preprocessor, postprocessor = setup_reinflow_pi0_policy(
            pretrained_path=config.pretrained_path,
            device=str(device),
            num_steps=config.num_denoising_steps,
            train_action_head=config.train_action_head,
            train_time_mlp=config.train_time_mlp,
            train_full_expert=config.train_full_expert,
            trainable_scope=config.trainable_scope,
            train_noise_head=config.train_noise_head,
            train_critic=config.train_critic,
            sigma_min=config.sigma_min,
            sigma_max=config.sigma_max,
            gradient_checkpointing=config.pi0_gradient_checkpointing,
        )
        print(f"  [ReinFlow] Sigma bounds: [{config.sigma_min}, {config.sigma_max}]")
    else:
        rl_policy, preprocessor, postprocessor = setup_reinflow_policy(
            pretrained_path=config.pretrained_path,
            device=str(device),
            num_steps=config.num_denoising_steps,
            train_action_head=config.train_action_head,
            train_time_mlp=config.train_time_mlp,
            train_full_expert=config.train_full_expert,
            train_noise_head=config.train_noise_head,
            train_critic=config.train_critic,
        )
        rl_policy.base.model.sigma_min = config.sigma_min
        rl_policy.base.model.sigma_max = config.sigma_max
        print(f"  [ReinFlow] Sigma bounds: [{config.sigma_min}, {config.sigma_max}]")
    
    # Load checkpoint if resuming
    start_episode = 0
    episode_rewards_history = []
    wandb_run_id = None
    
    checkpoint_to_load = args.resume if args else None
    if checkpoint_to_load is None and os.path.exists(config.checkpoint_path):
        checkpoint_to_load = config.checkpoint_path
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        # Use model-appropriate checkpoint loader
        if config.model_type == "pi0":
            start_episode, wandb_run_id = load_reinflow_pi0_checkpoint(rl_policy, checkpoint_to_load, str(device))
        else:
            start_episode, wandb_run_id = load_reinflow_checkpoint(rl_policy, checkpoint_to_load, str(device))

    if config.model_type == "smolvla":
        _print_smolvla_startup_summary(
            config,
            rl_policy,
            start_mode="resume-rl" if checkpoint_to_load and os.path.exists(checkpoint_to_load) else "fresh-rl",
            resume_path=checkpoint_to_load if checkpoint_to_load and os.path.exists(checkpoint_to_load) else None,
        )

    rl_policy.logprob_eval_microbatch_size = config.logprob_eval_microbatch_size
    rl_policy.critic_backprop_into_policy = config.critic_backprop_into_policy
    
    # Initialize wandb with PPO config
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            id=wandb_run_id,
            resume="allow",
            config={
                "model_type": config.model_type,
                "policy_lr": config.policy_lr,
                "critic_lr": config.critic_lr,
                "gamma": config.gamma,
                "gae_lambda": config.gae_lambda,
                "clip_epsilon": config.clip_epsilon,
                "num_ppo_epochs": config.num_ppo_epochs,
                "minibatch_size": config.minibatch_size,
                "target_kl": config.target_kl,
                "logprob_eval_microbatch_size": config.logprob_eval_microbatch_size,
                "critic_backprop_into_policy": config.critic_backprop_into_policy,
                "trainable_scope": _get_trainable_scope_label(config),
                "actor_lr_start": config.policy_lr,
                "actor_lr_end": config.policy_lr * 0.1,
                "distance_penalty_scale": config.distance_penalty_scale,
                "horizontal_progress_scale": config.horizontal_progress_scale,
                "vertical_approach_scale": config.vertical_approach_scale,
                "approach_closeness_scale": config.approach_closeness_scale,
                "alignment_reward_cap": config.alignment_reward_cap,
                "near_contact_bonus": config.near_contact_bonus,
                "contact_entry_bonus": config.contact_entry_bonus,
                "contact_persistence_reward": config.contact_persistence_reward,
                "hover_stall_threshold": config.hover_stall_threshold,
                "hover_penalty": config.hover_penalty,
                "bilateral_grasp_bonus": config.bilateral_grasp_bonus,
                "grasp_persistence_reward": config.grasp_persistence_reward,
                "slip_penalty_contact": config.slip_penalty_contact,
                "slip_penalty_grasp": config.slip_penalty_grasp,
                "block_displacement_penalty_scale": config.block_displacement_penalty_scale,
                "num_denoising_steps": config.num_denoising_steps,
                "chunks_per_episode": config.chunks_per_episode,
                "train_action_head": config.train_action_head,
                "train_time_mlp": config.train_time_mlp,
                "train_full_expert": config.train_full_expert,
                "train_noise_head": config.train_noise_head,
                "train_critic": config.train_critic,
                "training_mode": "ppo-on-policy",
                "gradient_checkpointing": config.pi0_gradient_checkpointing if config.model_type == "pi0" else False,
                "base_policy_source": getattr(rl_policy, "base_policy_source", config.pretrained_path),
                "base_policy_source_type": getattr(rl_policy, "base_policy_source_type", "unknown"),
                "normalization_mode": getattr(rl_policy, "normalization_mode", "unknown"),
                "state_action_frame": getattr(rl_policy, "state_action_frame", "unknown"),
            },
        )
        wandb_run_id = wandb.run.id
    
    reward_ema20 = None
    recent_epoch1_early_stops = deque(maxlen=5)
    epoch1_warning_active = False
    
    # Separate optimizers for actor and critic (allows different learning rates)
    policy_params = list(rl_policy.get_trainable_params())
    critic_params = list(rl_policy.get_critic_params())
    
    policy_optimizer = torch.optim.Adam(policy_params, lr=config.policy_lr)
    critic_optimizer = torch.optim.Adam(critic_params, lr=config.critic_lr)
    
    # Learning rate schedulers with warmup (paper Table 9b)
    total_batches = config.num_episodes
    
    # Policy scheduler: linear warmup + cosine annealing
    policy_warmup = torch.optim.lr_scheduler.LinearLR(
        policy_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.lr_warmup_iterations
    )
    policy_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        policy_optimizer,
        T_max=max(1, total_batches - config.lr_warmup_iterations),
        eta_min=config.policy_lr * 0.1
    )
    policy_scheduler = torch.optim.lr_scheduler.SequentialLR(
        policy_optimizer,
        schedulers=[policy_warmup, policy_cosine],
        milestones=[config.lr_warmup_iterations]
    )
    
    # Critic scheduler: linear warmup + cosine annealing
    critic_warmup = torch.optim.lr_scheduler.LinearLR(
        critic_optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config.lr_warmup_iterations
    )
    critic_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        critic_optimizer,
        T_max=max(1, total_batches - config.lr_warmup_iterations),
        eta_min=config.critic_lr * 0.1
    )
    critic_scheduler = torch.optim.lr_scheduler.SequentialLR(
        critic_optimizer,
        schedulers=[critic_warmup, critic_cosine],
        milestones=[config.lr_warmup_iterations]
    )
    
    print(f"\n{'='*60}")
    print(f"Starting ReinFlow Training (PPO, Sequential)")
    print(f"{'='*60}")
    print(f"Instruction: '{config.instruction}'")
    print(f"Episodes: {config.num_episodes}")
    print(f"Chunks per episode: {config.chunks_per_episode}")
    print(f"Denoising steps: {config.num_denoising_steps}")
    print(f"Policy LR: {config.policy_lr} -> {config.policy_lr * 0.1}")
    print(f"Critic LR: {config.critic_lr} -> {config.critic_lr * 0.1}")
    print(f"LR warmup iterations: {config.lr_warmup_iterations}")
    print(f"PPO epochs: {config.num_ppo_epochs}")
    print(f"Mini-batch size: {config.minibatch_size}")
    print(f"Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"Effective batch size: {config.minibatch_size * config.gradient_accumulation_steps}")
    print(f"Clip epsilon: {config.clip_epsilon}")
    print(f"GAE lambda: {config.gae_lambda}")
    print(f"Target KL: {config.target_kl}")
    print(f"Entropy coeff: {config.entropy_coeff}")
    print(f"Critic warmup iters: {config.critic_warmup_iters}")
    print(f"Training mode: PPO ON-POLICY")
    print(f"{'='*60}\n")
    
    # Optional viewer
    viewer = None
    if config.render:
        viewer = mujoco.viewer.launch_passive(m, d)
    
    # ===== CRITIC WARMUP (paper Appendix D.2) =====
    # Train critic for a few iterations before updating actor
    # SKIP if resuming from checkpoint (critic already trained)
    if config.critic_warmup_iters > 0 and start_episode == 0:
        print(f"\n[Critic Warmup] Training critic for {config.critic_warmup_iters} iterations...")
        for warmup_iter in range(config.critic_warmup_iters):
            reset_env(m, d, config.starting_position)
            reset_reward_state()
            
            # Collect one episode worth of data
            warmup_rewards = []
            warmup_done = False
            warmup_observation = None
            
            for chunk_idx in range(config.chunks_per_episode):
                if warmup_done:
                    break
                    
                rgb_top = get_camera_observation(renderer, d, camera_name="camera_up")
                rgb_wrist = get_camera_observation(renderer, d, camera_name="wrist_camera")
                rgb_side = get_camera_observation(renderer, d, camera_name="camera_side")
                robot_state = get_robot_state(d)
                
                warmup_observation = prepare_observation_for_reinflow(
                    rgb_top, rgb_wrist, rgb_side, robot_state,
                    config.instruction, device, rl_policy
                )
                
                with torch.no_grad():
                    action_chunk, _, _ = rl_policy.forward_with_trajectory(warmup_observation)
                
                # Execute chunk and collect rewards
                chunk_reward = 0.0
                chunk_size = action_chunk.shape[1]
                
                for action_idx in range(chunk_size):
                    if warmup_done:
                        break
                    
                    action = action_chunk[0, action_idx]
                    action_np = action.detach().cpu().numpy()
                    # Model denormalization stays separate from the robot frame conversion.
                    action_radians = unnormalize_action_for_vla(action_np, config.model_type, postprocessor)
                    action_dict = convert_to_dictionary(action_radians)
                    
                    # Execute action
                    for _ in range(config.steps_per_action):
                        send_position_command(d, action_dict)
                        mujoco.mj_step(m, d)
                        if viewer is not None:
                            viewer.sync()
                    
                    # Get reward
                    reward, warmup_done, *_ = compute_reward(m, d, **_get_reward_kwargs(config))
                    chunk_reward += reward
                
                warmup_rewards.append(chunk_reward)
            
            # Only update critic (not actor)
            if len(warmup_rewards) > 0 and warmup_observation is not None:
                total_reward = sum(warmup_rewards)
                value = rl_policy.get_value(warmup_observation)
                target = torch.tensor([total_reward], device=device, dtype=torch.float32)
                critic_loss = F.mse_loss(value, target)
                
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic_params, max_norm=config.grad_clip_norm)
                critic_optimizer.step()
                
                print(f"  [Warmup {warmup_iter+1}/{config.critic_warmup_iters}] Critic loss: {critic_loss.item():.4f}, Reward: {total_reward:.4f}")
        
        print(f"[Critic Warmup] Complete!\n")
    elif start_episode > 0:
        print(f"\n[Critic Warmup] Skipping warmup (resuming from episode {start_episode})\n")
    
    try:
        for episode in range(start_episode, config.num_episodes):
            episode_start_time = time.time()
            
            # Update noise bounds with decay schedule (paper Appendix D)
            sigma_min, sigma_max = get_noise_bounds(episode, config.num_episodes, config)
            rl_policy.base.model.sigma_min = sigma_min
            rl_policy.base.model.sigma_max = sigma_max
            
            # Reset environment
            reset_env(m, d, config.starting_position)
            reset_reward_state()
            
            episode_reward = 0.0
            done = False
            
            # Track reward components for this episode
            episode_contacts = 0
            episode_grasps = 0
            episode_sustained = 0
            episode_height_aligned = 0
            episode_contact_entries = 0
            episode_grasp_persistent = 0
            episode_hover_stall = 0
            episode_slips = 0
            episode_lift_progress = 0.0
            episode_block_displacement = 0.0
            episode_approach_reward = 0.0
            episode_alignment_reward = 0.0
            episode_near_contact = 0
            episode_contact_after_alignment = 0
            episode_horizontal_progress = 0.0
            episode_vertical_approach = 0.0
            episode_contact_losses = 0
            episode_grasp_losses = 0
            
            # Collect data for this episode (on-policy)
            episode_trajectories = []  # List of trajectories for each chunk
            episode_observations = []  # List of observations for each chunk
            episode_chunk_rewards = []  # Reward for each chunk
            episode_dones = []  # Done flags for GAE
            
            # Execute multiple chunks per episode (getting fresh observations!)
            for chunk_idx in range(config.chunks_per_episode):
                if done:
                    break
                
                # Get FRESH observation for this chunk
                rgb_top = get_camera_observation(renderer, d, camera_name="camera_up")
                rgb_wrist = get_camera_observation(renderer, d, camera_name="wrist_camera")
                rgb_side = get_camera_observation(renderer, d, camera_name="camera_side")
                robot_state = get_robot_state(d)
                
                observation = prepare_observation_for_reinflow(
                    rgb_top, rgb_wrist, rgb_side, robot_state,
                    config.instruction, device, rl_policy
                )
                
                # Forward pass with trajectory storage
                action_chunk, trajectory, sigmas = rl_policy.forward_with_trajectory(observation)
                
                # Execute chunk and collect rewards
                chunk_reward = 0.0
                chunk_size = action_chunk.shape[1]  # (batch=1, chunk_size, action_dim)
                
                for action_idx in range(chunk_size):
                    if done:
                        break
                    
                    action = action_chunk[0, action_idx]
                    action_np = action.detach().cpu().numpy()
                    # Unnormalize actions based on model type
                    action_radians = unnormalize_action_for_vla(action_np, config.model_type, postprocessor)
                    action_dict = convert_to_dictionary(action_radians)
                    
                    # Execute action
                    for _ in range(config.steps_per_action):
                        send_position_command(d, action_dict)
                        mujoco.mj_step(m, d)
                        if viewer is not None:
                            viewer.sync()
                    
                    # Get reward with component flags
                    (
                        reward,
                        done,
                        contacted,
                        gripped,
                        sustained,
                        height_aligned,
                        block_lifted,
                        contact_entry,
                        grasp_persistent,
                        lift_progress,
                        hover_stall,
                        slip_count,
                        block_displacement,
                        approach_reward,
                        alignment_reward,
                        near_contact,
                        contact_after_alignment,
                        horizontal_progress,
                        vertical_approach,
                        contact_loss_count,
                        grasp_loss_count,
                    ) = compute_reward(m, d, **_get_reward_kwargs(config))
                    chunk_reward += reward
                    
                    # Track reward components
                    episode_contacts += int(contacted)
                    episode_grasps += int(gripped)
                    episode_sustained += int(sustained)
                    episode_height_aligned += int(height_aligned)
                    episode_contact_entries += int(contact_entry)
                    episode_grasp_persistent += int(grasp_persistent)
                    episode_hover_stall += int(hover_stall)
                    episode_slips += int(slip_count)
                    episode_lift_progress += float(lift_progress)
                    episode_block_displacement += float(block_displacement)
                    episode_approach_reward += float(approach_reward)
                    episode_alignment_reward += float(alignment_reward)
                    episode_near_contact += int(near_contact)
                    episode_contact_after_alignment += int(contact_after_alignment)
                    episode_horizontal_progress += float(horizontal_progress)
                    episode_vertical_approach += float(vertical_approach)
                    episode_contact_losses += int(contact_loss_count)
                    episode_grasp_losses += int(grasp_loss_count)
                    
                    if done:
                        print(f"  Episode {episode+1}: SUCCESS! Block lifted at chunk {chunk_idx+1}, action {action_idx+1}")
                
                episode_reward += chunk_reward
                
                # Store data for on-policy update (trajectory without batch dim)
                # Stack trajectory into tensor: (K+1, chunk_size, action_dim)
                # DETACH to prevent graph issues - we only need the values, gradients flow through compute_ppo_loss
                traj_tensor = torch.stack(trajectory, dim=1)[0].detach()  # Remove batch dim -> (K+1, chunk, action)
                episode_trajectories.append(traj_tensor)
                episode_observations.append(observation)
                episode_chunk_rewards.append(chunk_reward)
                episode_dones.append(float(done))
            
            # Track episode reward
            episode_rewards_history.append(episode_reward)
            
            # ===== PPO UPDATE WITH MINI-BATCHING =====
            # Only update if we collected at least one chunk
            if len(episode_trajectories) > 0:
                # Stack all chunk data into batches
                # Each trajectory: (K+1, chunk_size, action_dim) -> stack to (num_chunks, K+1, chunk, action)
                batch_trajectories = torch.stack(episode_trajectories, dim=0)  # (num_chunks, K+1, chunk, action)
                batch_size = batch_trajectories.shape[0]
                
                # Stack observations - need to handle dict
                batch_observations = {}
                for key in episode_observations[0].keys():
                    batch_observations[key] = torch.stack([obs[key].squeeze(0) for obs in episode_observations], dim=0)
                
                # Rewards and dones tensors
                batch_rewards = torch.tensor(episode_chunk_rewards, device=device, dtype=torch.float32)
                batch_dones = torch.tensor(episode_dones, device=device, dtype=torch.float32)
                
                # Compute values for GAE
                with torch.no_grad():
                    batch_values = rl_policy.get_value(batch_observations)
                    # For next values, use current values shifted (simplified for chunk-level)
                    batch_next_values = torch.zeros_like(batch_values)
                    batch_next_values[:-1] = batch_values[1:]
                    batch_next_values = batch_next_values * (1.0 - batch_dones)
                
                # Compute GAE advantages and returns
                advantages, returns = compute_gae(
                    batch_rewards, batch_values, batch_next_values, batch_dones,
                    gamma=config.gamma, gae_lambda=config.gae_lambda
                )
                
                # Normalize advantages
                if batch_size > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Compute old log probabilities (detached for PPO ratio)
                with torch.no_grad():
                    old_log_probs = _compute_policy_log_probs(rl_policy, batch_trajectories, batch_observations)
                    recomputed_log_probs = _compute_policy_log_probs(rl_policy, batch_trajectories, batch_observations)
                    old_values = batch_values.clone()
                pre_update_metrics = _compute_logprob_shift_metrics(
                    old_log_probs, recomputed_log_probs, clip_epsilon=config.clip_epsilon
                )
                _validate_pre_update_logprob_invariant(pre_update_metrics)
                
                # PPO epochs with mini-batching
                kl_early_stop = False
                epoch_policy_losses = []
                epoch_critic_losses = []
                post_update_metric_history = []
                
                # Gradient norm tracking for diagnostics
                last_policy_grad_norm = 0.0
                last_critic_grad_norm = 0.0
                policy_grad_clipped = False
                critic_grad_clipped = False
                actor_delta_l2_history = []
                actor_delta_max_abs_history = []
                epoch1_early_stop = False
                
                for ppo_epoch in range(config.num_ppo_epochs):
                    if kl_early_stop:
                        break
                    
                    # Shuffle indices for mini-batching
                    indices = torch.randperm(batch_size, device=device)
                    
                    # Gradient accumulation setup (paper Appendix D)
                    accumulation_counter = 0
                    policy_optimizer.zero_grad()
                    critic_optimizer.zero_grad()
                    
                    for start in range(0, batch_size, config.minibatch_size):
                        end = min(start + config.minibatch_size, batch_size)
                        mb_indices = indices[start:end]
                        
                        # Get mini-batch data
                        mb_trajectories = batch_trajectories[mb_indices]
                        mb_observations = {k: v[mb_indices] for k, v in batch_observations.items()}
                        mb_advantages = advantages[mb_indices]
                        mb_returns = returns[mb_indices]
                        mb_old_log_probs = old_log_probs[mb_indices]
                        mb_old_values = old_values[mb_indices]
                        
                        # Compute PPO loss with entropy regularization
                        policy_loss, critic_loss, loss_info = compute_ppo_loss(
                            rl_policy,
                            mb_trajectories,
                            mb_observations,
                            mb_old_log_probs,
                            mb_advantages,
                            mb_returns,
                            clip_epsilon=config.clip_epsilon,
                            value_clip_epsilon=config.value_clip_epsilon,
                            old_values=mb_old_values,
                            entropy_coeff=config.entropy_coeff,
                        )
                        
                        # Combine losses and do single backward pass to avoid graph issues
                        # (policy_loss and critic_loss share computation graph through observations)
                        total_loss = policy_loss + critic_loss
                        
                        # Scale loss by accumulation steps for gradient averaging
                        scaled_loss = total_loss / config.gradient_accumulation_steps
                        scaled_loss.backward()
                        
                        accumulation_counter += 1
                        epoch_policy_losses.append(policy_loss.item())
                        epoch_critic_losses.append(critic_loss.item())
                        
                        # Step optimizer only after accumulating enough gradients
                        if accumulation_counter % config.gradient_accumulation_steps == 0:
                            # Capture gradient norms before clipping for diagnostics
                            last_critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, max_norm=config.grad_clip_norm).item()
                            critic_grad_clipped = last_critic_grad_norm > config.grad_clip_norm
                            critic_optimizer.step()
                            last_policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy_params, max_norm=config.grad_clip_norm).item()
                            policy_grad_clipped = last_policy_grad_norm > config.grad_clip_norm
                            policy_optimizer.step()
                            actor_delta_metrics = _compute_actor_delta_metrics(policy_optimizer)
                            actor_delta_l2_history.append(actor_delta_metrics["delta_l2"])
                            actor_delta_max_abs_history.append(actor_delta_metrics["delta_max_abs"])
                            
                            policy_optimizer.zero_grad()
                            critic_optimizer.zero_grad()

                            with torch.no_grad():
                                updated_log_probs = _compute_policy_log_probs(
                                    rl_policy, mb_trajectories, mb_observations
                                )
                            post_update_metrics = _compute_logprob_shift_metrics(
                                mb_old_log_probs,
                                updated_log_probs,
                                clip_epsilon=config.clip_epsilon,
                            )
                            post_update_metric_history.append(post_update_metrics)

                            if post_update_metrics['kl_div'] > config.target_kl * 1.5:
                                print(
                                    f"  [KL Early Stop] Epoch {ppo_epoch+1}, "
                                    f"post-update KL={post_update_metrics['kl_div']:.4f} > {config.target_kl * 1.5:.4f}"
                                )
                                if ppo_epoch == 0:
                                    epoch1_early_stop = True
                                kl_early_stop = True
                                break
                    
                    # Handle remaining gradients if batch doesn't divide evenly
                    if accumulation_counter % config.gradient_accumulation_steps != 0:
                        last_critic_grad_norm = torch.nn.utils.clip_grad_norm_(critic_params, max_norm=config.grad_clip_norm).item()
                        critic_grad_clipped = last_critic_grad_norm > config.grad_clip_norm
                        critic_optimizer.step()
                        last_policy_grad_norm = torch.nn.utils.clip_grad_norm_(policy_params, max_norm=config.grad_clip_norm).item()
                        policy_grad_clipped = last_policy_grad_norm > config.grad_clip_norm
                        policy_optimizer.step()
                        actor_delta_metrics = _compute_actor_delta_metrics(policy_optimizer)
                        actor_delta_l2_history.append(actor_delta_metrics["delta_l2"])
                        actor_delta_max_abs_history.append(actor_delta_metrics["delta_max_abs"])

                        with torch.no_grad():
                            updated_log_probs = _compute_policy_log_probs(
                                rl_policy, mb_trajectories, mb_observations
                            )
                        post_update_metrics = _compute_logprob_shift_metrics(
                            mb_old_log_probs,
                            updated_log_probs,
                            clip_epsilon=config.clip_epsilon,
                        )
                        post_update_metric_history.append(post_update_metrics)

                        if post_update_metrics['kl_div'] > config.target_kl * 1.5:
                            print(
                                f"  [KL Early Stop] Epoch {ppo_epoch+1}, "
                                f"post-update KL={post_update_metrics['kl_div']:.4f} > {config.target_kl * 1.5:.4f}"
                            )
                            if ppo_epoch == 0:
                                epoch1_early_stop = True
                            kl_early_stop = True
                
                # Step LR schedulers
                policy_scheduler.step()
                critic_scheduler.step()
                
                # Aggregate loss info
                avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0.0
                avg_critic_loss = np.mean(epoch_critic_losses) if epoch_critic_losses else 0.0
                post_update_summary = _aggregate_logprob_shift_metrics(post_update_metric_history)
                avg_kl_div = post_update_summary["kl_div"]
                avg_actor_delta_l2 = float(np.mean(actor_delta_l2_history)) if actor_delta_l2_history else 0.0
                max_actor_delta_abs = float(np.max(actor_delta_max_abs_history)) if actor_delta_max_abs_history else 0.0
                
            else:
                loss_info = {'advantage_mean': 0, 'value_mean': 0, 'log_prob_mean': 0, 'kl_div': 0,
                             'ratio_mean': 0, 'ratio_std': 0, 'ratio_min': 0, 'ratio_max': 0,
                             'advantage_std': 0, 'old_log_prob_mean': 0, 'return_mean': 0, 'clip_fraction': 0}
                pre_update_metrics = _default_logprob_shift_metrics()
                post_update_summary = _default_logprob_shift_metrics()
                avg_policy_loss = 0.0
                avg_critic_loss = 0.0
                avg_kl_div = 0.0
                last_policy_grad_norm = 0.0
                last_critic_grad_norm = 0.0
                policy_grad_clipped = False
                critic_grad_clipped = False
                avg_actor_delta_l2 = 0.0
                max_actor_delta_abs = 0.0
                epoch1_early_stop = False
                # Set defaults for metrics computed from tensors
                batch_rewards = torch.tensor([0.0], device=device)
                batch_values = torch.tensor([0.0], device=device)
                returns = torch.tensor([0.0], device=device)
                advantages = torch.tensor([0.0], device=device)
                action_chunk = torch.tensor([[[0.0]]], device=device)
            
            episode_time = time.time() - episode_start_time
            
            # Logging
            if (episode + 1) % config.log_interval == 0:
                avg_reward = np.mean(episode_rewards_history[-min(100, len(episode_rewards_history)):])
                reward_ema20 = _update_ema(reward_ema20, episode_reward, window=20)
                recent_epoch1_early_stops.append(1 if epoch1_early_stop else 0)
                should_warn_epoch1 = len(recent_epoch1_early_stops) == 5 and sum(recent_epoch1_early_stops) > 3
                if should_warn_epoch1 and not epoch1_warning_active:
                    print("  [Training Warning] More than 3 of the last 5 batches early-stopped in epoch 1. Actor updates are still too aggressive for reward growth.")
                epoch1_warning_active = should_warn_epoch1
                current_lr = policy_scheduler.get_last_lr()[0]
                print(f"Episode {episode+1:5d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"EMA20: {reward_ema20:7.2f} | "
                      f"Avg: {avg_reward:7.2f} | "
                      f"KL: {avg_kl_div:.4f} | "
                      f"RatioMax: {post_update_summary['ratio_max']:.3f} | "
                      f"ClipFrac: {post_update_summary['clip_fraction']:.2f} | "
                      f"LR: {current_lr:.2e} | "
                      f"Time: {episode_time:.1f}s")
                
                # Compute additional diagnostic metrics for wandb
                with torch.no_grad():
                    # Critic diagnostics
                    returns_var = returns.var() + 1e-8
                    explained_var = (1 - (returns - batch_values).var() / returns_var).item()
                    
                    # Action diagnostics (from last forward pass)
                    action_saturation_low = (action_chunk < -0.95).float().mean().item()
                    action_saturation_high = (action_chunk > 0.95).float().mean().item()
                    
                    # Log prob per dimension (action_dim=6, chunk_size=50)
                    log_prob_per_dim = loss_info.get('log_prob_mean', 0) / (6 * 50)
                    log_prob_drift = loss_info.get('log_prob_mean', 0) - loss_info.get('old_log_prob_mean', 0)
                
                # Compute total steps in episode for rate calculation
                total_steps = config.chunks_per_episode * 50  # chunk_size=50
                
                if config.wandb_enabled:
                    wandb.log({
                        # Basic info
                        "episode": episode + 1,
                        "time/episode_seconds": episode_time,
                        
                        # Reward metrics (4)
                        "reward/episode": episode_reward,
                        "reward/ema20": reward_ema20,
                        "reward/avg": avg_reward,
                        "reward/std": batch_rewards.std().item(),
                        "reward/positive_fraction": (batch_rewards > 0).float().mean().item(),
                        
                        # Contact metrics (3)
                        "reward/contact_count": episode_contacts,
                        "reward/contact_rate": episode_contacts / total_steps,
                        # Grasp metrics (2)
                        "reward/grasp_count": episode_grasps,
                        "reward/grasp_rate": episode_grasps / total_steps,
                        # Sustained contact metrics (2)
                        "reward/sustained_count": episode_sustained,
                        "reward/sustained_contact_rate": episode_sustained / total_steps,
                        # Height alignment metrics (2)
                        "reward/height_align_count": episode_height_aligned,
                        "reward/height_align_rate": episode_height_aligned / total_steps,
                        "reward/contact_entry_count": episode_contact_entries,
                        "reward/contact_entry_rate": episode_contact_entries / total_steps,
                        "reward/grasp_persistence_count": episode_grasp_persistent,
                        "reward/grasp_persistence_rate": episode_grasp_persistent / total_steps,
                        "reward/approach_reward_mean": episode_approach_reward / total_steps,
                        "reward/alignment_reward_mean": episode_alignment_reward / total_steps,
                        "reward/near_contact_count": episode_near_contact,
                        "reward/near_contact_rate": episode_near_contact / total_steps,
                        "reward/contact_after_alignment_count": episode_contact_after_alignment,
                        "reward/contact_after_alignment_rate": episode_contact_after_alignment / total_steps,
                        "reward/horizontal_progress_mean": episode_horizontal_progress / total_steps,
                        "reward/vertical_approach_mean": episode_vertical_approach / total_steps,
                        "reward/lift_progress_mean": episode_lift_progress / total_steps,
                        "reward/hover_stall_count": episode_hover_stall,
                        "reward/hover_stall_rate": episode_hover_stall / total_steps,
                        "reward/slip_count": episode_slips,
                        "reward/contact_loss_count": episode_contact_losses,
                        "reward/grasp_loss_count": episode_grasp_losses,
                        "reward/block_displacement_mean": episode_block_displacement / total_steps,
                        
                        # Loss metrics
                        "loss/policy": avg_policy_loss,
                        "loss/critic": avg_critic_loss,
                        "loss/entropy": loss_info.get('entropy', 0),
                        
                        # Critic/Value metrics (7)
                        "critic/value_mean": loss_info.get('value_mean', 0),
                        "critic/value_std": batch_values.std().item(),
                        "critic/value_min": batch_values.min().item(),
                        "critic/value_max": batch_values.max().item(),
                        "critic/return_mean": loss_info.get('return_mean', 0),
                        "critic/return_std": returns.std().item(),
                        "critic/explained_variance": explained_var,
                        
                        # Advantage metrics (4)
                        "advantage/mean": loss_info.get('advantage_mean', 0),
                        "advantage/std": loss_info.get('advantage_std', 0),
                        "advantage/min": advantages.min().item(),
                        "advantage/max": advantages.max().item(),
                        
                        # PPO ratio metrics (4)
                        "ppo/ratio_mean": loss_info.get('ratio_mean', 0),
                        "ppo/ratio_std": loss_info.get('ratio_std', 0),
                        "ppo/ratio_min": loss_info.get('ratio_min', 0),
                        "ppo/ratio_max": loss_info.get('ratio_max', 0),
                        
                        # Log probability metrics (4)
                        "logprob/new_mean": loss_info.get('log_prob_mean', 0),
                        "logprob/old_mean": loss_info.get('old_log_prob_mean', 0),
                        "logprob/per_dimension": log_prob_per_dim,
                        "logprob/drift": log_prob_drift,
                        "debug/pre_update_kl": pre_update_metrics.get('kl_div', 0.0),
                        "debug/pre_update_ratio_mean": pre_update_metrics.get('ratio_mean', 1.0),
                        "debug/pre_update_logprob_abs_mean": pre_update_metrics.get('logprob_abs_mean', 0.0),
                        "debug/pre_update_logprob_abs_max": pre_update_metrics.get('logprob_abs_max', 0.0),
                        
                        # Action metrics (4)
                        "actions/mean": action_chunk.mean().item(),
                        "actions/std": action_chunk.std().item(),
                        "actions/saturation_low": action_saturation_low,
                        "actions/saturation_high": action_saturation_high,
                        
                        # Gradient metrics (4)
                        "gradients/policy_norm": last_policy_grad_norm,
                        "gradients/actor_norm": last_policy_grad_norm,
                        "gradients/critic_norm": last_critic_grad_norm,
                        "gradients/policy_clipped": float(policy_grad_clipped),
                        "gradients/actor_clipped": float(policy_grad_clipped),
                        "gradients/critic_clipped": float(critic_grad_clipped),
                        "updates/actor_delta_l2": avg_actor_delta_l2,
                        "updates/actor_delta_max_abs": max_actor_delta_abs,
                        
                        # Training dynamics
                        "training/kl_divergence": avg_kl_div,
                        "training/post_update_kl": avg_kl_div,
                        "training/post_update_ratio_mean": post_update_summary.get('ratio_mean', 1.0),
                        "training/post_update_ratio_max": post_update_summary.get('ratio_max', 1.0),
                        "training/post_update_clip_fraction": post_update_summary.get('clip_fraction', 0.0),
                        "training/post_update_logprob_abs_mean": post_update_summary.get('logprob_abs_mean', 0.0),
                        "training/clip_fraction": post_update_summary.get('clip_fraction', 0.0),
                        "training/learning_rate": current_lr,
                        "training/effective_batch_size": config.minibatch_size * config.gradient_accumulation_steps,
                        "training/sigma_min": sigma_min,
                        "training/sigma_max": sigma_max,
                        "training/epoch1_early_stop": float(epoch1_early_stop),
                        "training/recent_epoch1_early_stop_count": float(sum(recent_epoch1_early_stops)),
                    })
            
            # Save checkpoint periodically
            if (episode + 1) % config.save_interval == 0:
                if config.model_type == "pi0":
                    save_reinflow_pi0_checkpoint(
                        rl_policy, episode, episode_rewards_history, 
                        config.checkpoint_path, wandb_run_id
                    )
                else:
                    save_reinflow_checkpoint(
                        rl_policy, episode, episode_rewards_history, 
                        config.checkpoint_path, wandb_run_id
                    )
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        if viewer is not None:
            viewer.close()
        if config.wandb_enabled:
            wandb.finish()
    
    # Save final checkpoint
    if config.model_type == "pi0":
        save_reinflow_pi0_checkpoint(
            rl_policy, episode, episode_rewards_history, 
            config.checkpoint_path, wandb_run_id
        )
    else:
        save_reinflow_checkpoint(
            rl_policy, episode, episode_rewards_history, 
            config.checkpoint_path, wandb_run_id
        )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Total episodes: {len(episode_rewards_history)}")
    if episode_rewards_history:
        print(f"Best reward: {max(episode_rewards_history):.2f}")
        print(f"Final avg reward (last 100): {np.mean(episode_rewards_history[-100:]):.2f}")
    
    return rl_policy, episode_rewards_history


# ===== Main Entry Point =====

def train(config=None, args=None):
    """
    Main training entry point - dispatches to parallel or sequential training.
    Uses on-policy actor-critic training.
    
    Supports both SmolVLA (default) and Pi0 models.
    """
    if config is None:
        config = TrainingConfig()
    
    # Apply command line overrides
    if args is not None:
        # Model type selection
        if hasattr(args, 'model_type') and args.model_type is not None:
            config.model_type = args.model_type
        
        if args.no_render:
            config.render = False
        if args.headless:
            config.render = False
        if args.episodes is not None:
            config.num_episodes = args.episodes
        if args.policy_lr is not None:
            config.policy_lr = args.policy_lr
        if args.critic_lr is not None:
            config.critic_lr = args.critic_lr
        if (
            hasattr(args, 'start_from_finetuned')
            and args.start_from_finetuned
            and args.pretrained is None
            and config.model_type == "smolvla"
        ):
            config.pretrained_path = config.finetuned_smolvla_path
        if args.pretrained is not None:
            config.pretrained_path = args.pretrained
        if args.parallel_envs is not None:
            config.num_parallel_envs = args.parallel_envs
        if args.subproc:
            config.use_subproc_env = True
        if args.no_wandb:
            config.wandb_enabled = False
        if args.full_expert:
            config.train_full_expert = True
            config.trainable_scope = "full-expert"
        if args.no_full_expert:
            config.train_full_expert = False
            config.trainable_scope = "rl_stable_heads"
        if hasattr(args, 'no_gradient_checkpointing') and args.no_gradient_checkpointing:
            config.pi0_gradient_checkpointing = False

    if args is not None and getattr(args, 'resume', None) is not None:
        print("[ReinFlow] --resume specified: ReinFlow RL checkpoint state will take precedence over fresh-start weights")
    
    # Apply Pi0-specific defaults if using Pi0 model
    if config.model_type == "pi0":
        print("\n" + "="*60)
        print("Pi0 Model Selected - Applying Pi0-specific settings")
        print("="*60)
        
        # Use Pi0 pretrained path if not overridden
        if config.pretrained_path == "lerobot/smolvla_base":
            config.pretrained_path = config.pi0_pretrained_path
        
        # Reduce batch size and increase accumulation for 3.3B model
        if config.minibatch_size > 2:
            print(f"  [Pi0] Reducing minibatch_size: {config.minibatch_size} -> 2")
            config.minibatch_size = 2
        
        if config.gradient_accumulation_steps < 30:
            print(f"  [Pi0] Increasing gradient_accumulation_steps: {config.gradient_accumulation_steps} -> 30")
            config.gradient_accumulation_steps = 30
        
        # Reduce learning rate for larger model
        if config.policy_lr > 2.5e-6:
            print(f"  [Pi0] Reducing policy_lr: {config.policy_lr} -> 2.5e-6")
            config.policy_lr = 2.5e-6
        
        # Reduce parallel envs for memory
        if config.num_parallel_envs > 4:
            print(f"  [Pi0] Reducing num_parallel_envs: {config.num_parallel_envs} -> 4")
            config.num_parallel_envs = 4
        
        # Update wandb project name
        config.wandb_project = "reinflow-pi0"
        
        # Update checkpoint path
        if config.checkpoint_path == "reinflow_checkpoint.pt":
            config.checkpoint_path = "reinflow_pi0_checkpoint.pt"
        
        print(f"  [Pi0] Gradient checkpointing: {config.pi0_gradient_checkpointing}")
        print("="*60 + "\n")
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Apple Silicon GPU)")
        if config.model_type == "pi0":
            print("  WARNING: Pi0 on MPS may have limited support. CUDA recommended.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
        if config.model_type == "pi0":
            print("  WARNING: Pi0 on CPU will be very slow. CUDA strongly recommended.")
    
    # Dispatch to appropriate training loop
    if config.num_parallel_envs > 1:
        if not torch.cuda.is_available():
            print("WARNING: Parallel mode is optimized for CUDA. "
                  "On M1/CPU, sequential mode is usually faster.")
        return train_parallel(config, args, device)
    else:
        return train_sequential(config, args, device)


if __name__ == "__main__":
    args = parse_args()
    train(args=args)
