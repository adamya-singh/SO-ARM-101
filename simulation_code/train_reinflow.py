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
import argparse

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
       - Paper's policy_lr=4.5e-5 → our policy_lr=5e-7 (scaled down ~100x for stability)
    
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
    # gradients are ~6x stronger. Increased from 1e-6 to 3e-6 after addressing clip fraction issue.
    policy_lr = 0.000003  # 3e-6 - can increase since clip_epsilon=0.15 protects against large updates
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
    train_full_expert = True   # Train entire Action Expert (~100M params)
    train_noise_head = True    # Train noise_mlp (σ_θ' network) - always True for ReinFlow
    train_critic = True        # Train critic network for actor-critic
    
    # Policy execution
    steps_per_action = 10  # Physics steps per policy action
    
    # Reward
    lift_threshold = 0.08
    contact_bonus = 0.1   # Bonus reward while gripper contacts block
    height_alignment_bonus = 0.05  # Bonus when gripper is above block (top-down approach)
    grasp_bonus = 0.15  # Bonus when both sides of gripper squeeze block
    
    # Logging
    log_interval = 1
    save_interval = 10
    
    # Rendering
    render = False  # Set False for faster training
    
    # Checkpointing
    checkpoint_path = "reinflow_checkpoint.pt"
    pretrained_path = "lerobot/smolvla_base"
    
    # Parallelization
    num_parallel_envs = 1
    use_subproc_env = False
    
    # Weights & Biases
    wandb_project = "reinflow-smolvla"
    wandb_enabled = True
    
    # PPO Hyperparameters (paper Table 7b - visual manipulation)
    # Note: Some values scaled for SmolVLA's chunk_size=50 (see docstring above)
    num_ppo_epochs = 5           # Reduced from 10 to prevent ratio drift over epochs
    minibatch_size = 8           # Mini-batch size for PPO updates
    clip_epsilon = 0.15          # Increased from 0.05 to reduce 83% clip fraction (paper: 0.1-0.2)
    value_clip_epsilon = 0.2     # Clip range for value function (0 to disable)
    gae_lambda = 0.95            # GAE lambda parameter
    # SCALED FOR CHUNK SIZE 50: Paper uses 0.01 for chunks of 4-8. With 6x more dims,
    # KL values are naturally ~6x larger, so we scale target_kl accordingly (0.05-0.1)
    target_kl = 0.1             # KL threshold for early stopping (scaled ~6x from paper's 0.01)
    
    # Gradient accumulation (paper Appendix D)
    gradient_accumulation_steps = 15  # Paper uses 15 for visual tasks
    
    # Learning rate warmup (paper Table 9b)
    lr_warmup_iterations = 10  # Paper uses 10 for PickPlaceCan, 25 for NutAssemblySquare
    
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
                        help='Path to pretrained VLA model (SmolVLA or Pi0)')
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
    
    # Select appropriate setup function based on model type
    # SmolVLA uses hardcoded normalization (no processors needed)
    # Pi0 uses processor-based normalization
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
        # SmolVLA - no processors needed (uses hardcoded normalization)
        rl_policy = setup_reinflow_policy(
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
    
    # Choose environment implementation
    # SmolVLA uses hardcoded normalization (no preprocessor needed)
    # Pi0 uses processor-based normalization (pass preprocessor)
    if config.use_subproc_env:
        from subproc_vectorized_env import SubprocMuJoCoEnv
        print(f"\n[Parallel Mode - SUBPROC] Running {num_envs} environments in separate processes")
        vec_env = SubprocMuJoCoEnv(
            num_envs=num_envs,
            model_path=config.model_path,
            starting_position=config.starting_position,
            lift_threshold=config.lift_threshold,
            contact_bonus=config.contact_bonus,
            height_alignment_bonus=config.height_alignment_bonus,
            grasp_bonus=config.grasp_bonus,
            model_type=config.model_type,
            preprocessor=preprocessor,  # None for SmolVLA, actual preprocessor for Pi0
        )
    else:
        from vectorized_env import VectorizedMuJoCoEnv
        print(f"\n[Parallel Mode] Running {num_envs} environments in parallel")
        vec_env = VectorizedMuJoCoEnv(
            num_envs=num_envs,
            model_path=config.model_path,
            starting_position=config.starting_position,
            lift_threshold=config.lift_threshold,
            contact_bonus=config.contact_bonus,
            height_alignment_bonus=config.height_alignment_bonus,
            grasp_bonus=config.grasp_bonus,
            model_type=config.model_type,
            preprocessor=preprocessor,  # None for SmolVLA, actual preprocessor for Pi0
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
    
    total_episodes = 0
    
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
                
                action_chunks, _, _ = rl_policy.forward_batched_with_trajectory(observation)
                action_chunks_np = action_chunks.detach().cpu().numpy()
                # Unnormalize actions based on model type
                action_chunks_radians = np.stack([
                    np.stack([unnormalize_action_for_vla(a, config.model_type, postprocessor) for a in chunk])
                    for chunk in action_chunks_np
                ])
                
                chunk_rewards, dones, _ = vec_env.step_all_chunk(
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
            
            # Collect data for this batch (on-policy)
            batch_trajectories = []  # List of trajectories across all envs and chunks
            batch_observations = []  # List of observations
            batch_chunk_rewards = []  # Rewards for each chunk
            batch_dones = []  # Done flags for GAE
            
            # Execute multiple chunks per episode (getting fresh observations!)
            for chunk_idx in range(config.chunks_per_episode):
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
                chunk_rewards, dones, chunk_contacts = vec_env.step_all_chunk(
                    action_chunks_radians, config.steps_per_action
                )
                episode_rewards += chunk_rewards
                episode_contacts += chunk_contacts
                
                # Store data for on-policy update (one entry per environment)
                # trajectory is list of K+1 tensors, each (num_envs, chunk, action_dim)
                # Stack to (num_envs, K+1, chunk, action_dim)
                # DETACH to prevent graph issues - we only need the values, gradients flow through compute_ppo_loss
                traj_tensor = torch.stack(trajectory, dim=1).detach()  # (num_envs, K+1, chunk, action)
                
                for i in range(num_envs):
                    batch_trajectories.append(traj_tensor[i])  # (K+1, chunk, action)
                    # Store observation for env i
                    obs_i = {k: v[i:i+1] for k, v in observation.items()}  # Keep batch dim
                    batch_observations.append(obs_i)
                    batch_chunk_rewards.append(chunk_rewards[i])
                    batch_dones.append(float(dones[i]))
                
                # Early termination if all done
                if dones.all():
                    break
            
            # Track episode rewards
            episode_rewards_history.extend(episode_rewards.tolist())
            total_episodes += num_envs
            
            # ===== PPO UPDATE WITH MINI-BATCHING =====
            if len(batch_trajectories) > 0:
                # Stack all trajectories: (total_chunks, K+1, chunk, action)
                all_trajectories = torch.stack(batch_trajectories, dim=0)
                batch_size = all_trajectories.shape[0]
                
                
                # Stack observations
                all_observations = {}
                for key in batch_observations[0].keys():
                    all_observations[key] = torch.cat([obs[key] for obs in batch_observations], dim=0)
                
                # Rewards and dones tensors
                all_rewards = torch.tensor(batch_chunk_rewards, device=device, dtype=torch.float32)
                all_dones = torch.tensor(batch_dones, device=device, dtype=torch.float32)
                
                # Compute values for GAE
                with torch.no_grad():
                    all_values = rl_policy.get_value(all_observations)
                    # For next values, use current values shifted (simplified for chunk-level)
                    # In practice, for terminal states, next_value = 0
                    all_next_values = torch.zeros_like(all_values)
                    all_next_values[:-1] = all_values[1:]
                    all_next_values = all_next_values * (1.0 - all_dones)  # Zero out terminal states
                
                # Compute GAE advantages and returns
                advantages, returns = compute_gae(
                    all_rewards, all_values, all_next_values, all_dones,
                    gamma=config.gamma, gae_lambda=config.gae_lambda
                )
                
                # Normalize advantages
                if batch_size > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Compute old log probabilities (detached for PPO ratio)
                with torch.no_grad():
                    old_log_probs = compute_trajectory_log_probs_onpolicy(
                        rl_policy, all_trajectories, all_observations
                    )
                    old_values = all_values.clone()
                
                # PPO epochs with mini-batching
                kl_early_stop = False
                epoch_policy_losses = []
                epoch_critic_losses = []
                epoch_kl_divs = []
                
                # Gradient norm tracking for diagnostics
                last_policy_grad_norm = 0.0
                last_critic_grad_norm = 0.0
                policy_grad_clipped = False
                critic_grad_clipped = False
                
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
                        
                        # Record KL before checking for early stop
                        epoch_kl_divs.append(loss_info['kl_div'])
                        
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
                            
                            policy_optimizer.zero_grad()
                            critic_optimizer.zero_grad()
                        
                        # Check KL AFTER backward - early stop for remaining mini-batches/epochs
                        if loss_info['kl_div'] > config.target_kl * 1.5:
                            print(f"  [KL Early Stop] Epoch {epoch+1}, KL={loss_info['kl_div']:.4f} > {config.target_kl * 1.5:.4f}")
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
                
                # Step LR schedulers
                policy_scheduler.step()
                critic_scheduler.step()
                
                # Aggregate loss info
                avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0.0
                avg_critic_loss = np.mean(epoch_critic_losses) if epoch_critic_losses else 0.0
                avg_kl_div = np.mean(epoch_kl_divs) if epoch_kl_divs else 0.0
                
            else:
                loss_info = {'advantage_mean': 0, 'value_mean': 0, 'log_prob_mean': 0, 'kl_div': 0, 'clip_fraction': 0,
                             'ratio_mean': 0, 'ratio_std': 0, 'ratio_min': 0, 'ratio_max': 0,
                             'advantage_std': 0, 'old_log_prob_mean': 0, 'return_mean': 0}
                avg_policy_loss = 0.0
                avg_critic_loss = 0.0
                avg_kl_div = 0.0
                last_policy_grad_norm = 0.0
                last_critic_grad_norm = 0.0
                policy_grad_clipped = False
                critic_grad_clipped = False
                # Set defaults for metrics computed from tensors
                all_rewards = torch.tensor([0.0], device=device)
                all_values = torch.tensor([0.0], device=device)
                returns = torch.tensor([0.0], device=device)
                advantages = torch.tensor([0.0], device=device)
                action_chunks = torch.tensor([[[0.0]]], device=device)
            
            batch_time = time.time() - batch_start_time
            
            # Logging
            avg_reward = np.mean(episode_rewards)
            current_lr = policy_scheduler.get_last_lr()[0]
            print(f"Batch {episode_batch+1:4d} ({total_episodes:5d} eps) | "
                  f"Reward: {avg_reward:7.2f} | "
                  f"KL: {avg_kl_div:.4f} | "
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
                    "reward/batch_min": np.min(episode_rewards),
                    "reward/batch_max": np.max(episode_rewards),
                    "reward/std": all_rewards.std().item(),
                    "reward/positive_fraction": (all_rewards > 0).float().mean().item(),
                    
                    # Contact metrics (3)
                    "reward/contact_count_avg": episode_contacts.mean(),
                    "reward/contact_count_max": episode_contacts.max(),
                    "reward/contact_rate": episode_contacts.sum() / (num_envs * config.chunks_per_episode * 50),
                    
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
                    
                    # Action metrics (4)
                    "actions/mean": action_chunks.mean().item(),
                    "actions/std": action_chunks.std().item(),
                    "actions/saturation_low": action_saturation_low,
                    "actions/saturation_high": action_saturation_high,
                    
                    # Gradient metrics (4)
                    "gradients/policy_norm": last_policy_grad_norm,
                    "gradients/critic_norm": last_critic_grad_norm,
                    "gradients/policy_clipped": float(policy_grad_clipped),
                    "gradients/critic_clipped": float(critic_grad_clipped),
                    
                    # Training dynamics
                    "training/kl_divergence": avg_kl_div,
                    "training/clip_fraction": loss_info.get('clip_fraction', 0),
                    "training/learning_rate": current_lr,
                    "training/effective_batch_size": config.minibatch_size * config.gradient_accumulation_steps,
                    "training/sigma_min": sigma_min,
                    "training/sigma_max": sigma_max,
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
    
    # Select appropriate setup function based on model type
    # SmolVLA uses hardcoded normalization (no processors needed)
    # Pi0 uses processor-based normalization
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
        print(f"  [ReinFlow] Sigma bounds: [{config.sigma_min}, {config.sigma_max}]")
    else:
        # SmolVLA - no processors needed (uses hardcoded normalization)
        rl_policy = setup_reinflow_policy(
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
                "num_denoising_steps": config.num_denoising_steps,
                "chunks_per_episode": config.chunks_per_episode,
                "train_action_head": config.train_action_head,
                "train_time_mlp": config.train_time_mlp,
                "train_full_expert": config.train_full_expert,
                "train_noise_head": config.train_noise_head,
                "train_critic": config.train_critic,
                "training_mode": "ppo-on-policy",
                "gradient_checkpointing": config.pi0_gradient_checkpointing if config.model_type == "pi0" else False,
            },
        )
        wandb_run_id = wandb.run.id
    
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
                
                action_chunk, _, _ = rl_policy.forward_with_trajectory(warmup_observation)
                
                # Execute chunk and collect rewards
                chunk_reward = 0.0
                chunk_size = action_chunk.shape[1]
                
                for action_idx in range(chunk_size):
                    if warmup_done:
                        break
                    
                    action = action_chunk[0, action_idx]
                    action_np = action.detach().cpu().numpy()
                    # SmolVLA uses hardcoded normalization (no postprocessor)
                    action_radians = unnormalize_action_for_vla(action_np, config.model_type, postprocessor)
                    action_dict = convert_to_dictionary(action_radians)
                    
                    # Execute action
                    for _ in range(config.steps_per_action):
                        send_position_command(d, action_dict)
                        mujoco.mj_step(m, d)
                        if viewer is not None:
                            viewer.sync()
                    
                    # Get reward
                    reward, warmup_done = compute_reward(m, d, lift_threshold=config.lift_threshold)
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
                    
                    # Get reward
                    reward, done = compute_reward(m, d, lift_threshold=config.lift_threshold)
                    chunk_reward += reward
                    
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
                    old_log_probs = compute_trajectory_log_probs_onpolicy(
                        rl_policy, batch_trajectories, batch_observations
                    )
                    old_values = batch_values.clone()
                
                # PPO epochs with mini-batching
                kl_early_stop = False
                epoch_policy_losses = []
                epoch_critic_losses = []
                epoch_kl_divs = []
                
                # Gradient norm tracking for diagnostics
                last_policy_grad_norm = 0.0
                last_critic_grad_norm = 0.0
                policy_grad_clipped = False
                critic_grad_clipped = False
                
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
                        
                        # Record KL before checking for early stop
                        epoch_kl_divs.append(loss_info['kl_div'])
                        
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
                            
                            policy_optimizer.zero_grad()
                            critic_optimizer.zero_grad()
                        
                        # Check KL AFTER backward - early stop for remaining mini-batches/epochs
                        if loss_info['kl_div'] > config.target_kl * 1.5:
                            print(f"  [KL Early Stop] Epoch {ppo_epoch+1}, KL={loss_info['kl_div']:.4f} > {config.target_kl * 1.5:.4f}")
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
                
                # Step LR schedulers
                policy_scheduler.step()
                critic_scheduler.step()
                
                # Aggregate loss info
                avg_policy_loss = np.mean(epoch_policy_losses) if epoch_policy_losses else 0.0
                avg_critic_loss = np.mean(epoch_critic_losses) if epoch_critic_losses else 0.0
                avg_kl_div = np.mean(epoch_kl_divs) if epoch_kl_divs else 0.0
                
            else:
                loss_info = {'advantage_mean': 0, 'value_mean': 0, 'log_prob_mean': 0, 'kl_div': 0,
                             'ratio_mean': 0, 'ratio_std': 0, 'ratio_min': 0, 'ratio_max': 0,
                             'advantage_std': 0, 'old_log_prob_mean': 0, 'return_mean': 0, 'clip_fraction': 0}
                avg_policy_loss = 0.0
                avg_critic_loss = 0.0
                avg_kl_div = 0.0
                last_policy_grad_norm = 0.0
                last_critic_grad_norm = 0.0
                policy_grad_clipped = False
                critic_grad_clipped = False
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
                current_lr = policy_scheduler.get_last_lr()[0]
                print(f"Episode {episode+1:5d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg: {avg_reward:7.2f} | "
                      f"KL: {avg_kl_div:.4f} | "
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
                
                if config.wandb_enabled:
                    wandb.log({
                        # Basic info
                        "episode": episode + 1,
                        "time/episode_seconds": episode_time,
                        
                        # Reward metrics (5)
                        "reward/episode": episode_reward,
                        "reward/avg": avg_reward,
                        "reward/std": batch_rewards.std().item(),
                        "reward/positive_fraction": (batch_rewards > 0).float().mean().item(),
                        
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
                        
                        # Action metrics (4)
                        "actions/mean": action_chunk.mean().item(),
                        "actions/std": action_chunk.std().item(),
                        "actions/saturation_low": action_saturation_low,
                        "actions/saturation_high": action_saturation_high,
                        
                        # Gradient metrics (4)
                        "gradients/policy_norm": last_policy_grad_norm,
                        "gradients/critic_norm": last_critic_grad_norm,
                        "gradients/policy_clipped": float(policy_grad_clipped),
                        "gradients/critic_clipped": float(critic_grad_clipped),
                        
                        # Training dynamics
                        "training/kl_divergence": avg_kl_div,
                        "training/clip_fraction": loss_info.get('clip_fraction', 0),
                        "training/learning_rate": current_lr,
                        "training/effective_batch_size": config.minibatch_size * config.gradient_accumulation_steps,
                        "training/sigma_min": sigma_min,
                        "training/sigma_max": sigma_max,
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
        if args.no_full_expert:
            config.train_full_expert = False
        if hasattr(args, 'no_gradient_checkpointing') and args.no_gradient_checkpointing:
            config.pi0_gradient_checkpointing = False
    
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
