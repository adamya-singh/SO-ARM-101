"""
ReinFlow Training Script for SmolVLA

Full ReinFlow-style flow-based RL training that:
1. Injects learnable noise at each of the 10 denoising steps
2. Computes exact log-probabilities through the Markov chain
3. Uses REINFORCE policy gradient for fine-tuning

This is the CORRECT way to do RL with flow-matching policies, unlike
the "ReinFlow-lite" approach which only added noise at the output.

Usage:
    conda activate lerobot
    python train_reinflow.py

    # Resume from checkpoint:
    python train_reinflow.py --resume reinflow_checkpoint.pt
    
    # Headless mode (for Colab/SSH):
    python train_reinflow.py --no-render --headless
    
    # Parallel mode for A100 GPU (8 environments):
    python train_reinflow.py --parallel-envs 8 --no-render --headless
    
    # Parallel mode with subprocess-based rendering (true CPU parallelism):
    python train_reinflow.py --parallel-envs 4 --subproc --no-render --headless
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
import numpy as np
import wandb

from reinflow_smolvla import (
    ReinFlowSmolVLA,
    setup_reinflow_policy,
    prepare_observation_for_reinflow,
    prepare_batched_observation,
    compute_returns,
    save_reinflow_checkpoint,
    load_reinflow_checkpoint,
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
    unnormalize_action_from_smolvla,
)


# ===== Training Configuration =====

class TrainingConfig:
    """Configuration for ReinFlow RL training."""
    
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
    num_episodes = 3000
    max_steps_per_episode = 50
    gamma = 0.95  # Discount factor
    lr = 0.00003  # Lower LR for full expert training (was 0.0003)
    grad_clip_norm = 0.5  # Tighter clipping for stability (was 1.0)
    batch_size = 30  # Number of episodes to accumulate before gradient update
    
    # ReinFlow specific
    num_denoising_steps = 10  # Must match SmolVLA config
    init_log_sigma = -0.7    # Initial noise scale (exp(-0.7) ≈ 0.5, good exploration)
    entropy_coef = 0.01       # Reduced for full expert (was 0.05)
    
    # What to train
    train_action_head = True   # Train action_out_proj (23K params) - ignored if train_full_expert=True
    train_time_mlp = True      # Ignored if train_full_expert=True
    train_full_expert = True   # Train entire Action Expert (~100M params)
    
    # Policy execution
    steps_per_action = 10  # Physics steps per policy action
    
    # Reward
    lift_threshold = 0.08
    
    # Logging
    log_interval = 1
    save_interval = 10
    
    # Rendering
    render = False  # Set False for faster training
    
    # Checkpointing
    checkpoint_path = "reinflow_checkpoint.pt"
    pretrained_path = "lerobot/smolvla_base"
    
    # Parallelization (A100 optimization)
    # Set >1 to run multiple environments in parallel for GPU efficiency
    # Default 1 = sequential mode (best for M1 Mac)
    num_parallel_envs = 1
    
    # Use subprocess-based environment for parallel CPU rendering
    # When True, each environment runs in a separate process for true parallel rendering
    # This can significantly speed up training on multi-core CPUs
    use_subproc_env = False
    
    # Weights & Biases
    wandb_project = "reinflow-smolvla"
    wandb_enabled = True


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReinFlow Training for SmolVLA')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable visualization for faster training')
    parser.add_argument('--headless', action='store_true',
                        help='Force headless rendering (EGL/OSMesa) for Colab/SSH')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained SmolVLA model')
    parser.add_argument('--parallel-envs', type=int, default=None,
                        help='Number of parallel environments (default: 1 for sequential, use 8-16 for A100)')
    parser.add_argument('--subproc', action='store_true',
                        help='Use subprocess-based parallel rendering (true parallelism across CPU cores)')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable Weights & Biases logging')
    parser.add_argument('--full-expert', action='store_true',
                        help='Train entire Action Expert (~100M params) instead of just output layers')
    parser.add_argument('--no-full-expert', action='store_true',
                        help='Disable full expert training (train only output layers ~540K params)')
    return parser.parse_args()


# ===== Parallel Training Loop (for A100) =====

def train_parallel(config, args, device):
    """
    Parallel training loop using vectorized environments.
    
    Runs N environments in parallel, batching observations for efficient
    GPU inference. Best for A100/CUDA where batch processing is fast.
    
    With --subproc flag, uses subprocess-based parallelism for true parallel
    CPU rendering across multiple cores.
    """
    num_envs = config.num_parallel_envs
    
    # Choose environment implementation based on config
    if config.use_subproc_env:
        from subproc_vectorized_env import SubprocMuJoCoEnv
        print(f"\n[Parallel Mode - SUBPROC] Running {num_envs} environments in separate processes")
        print("  (True parallel CPU rendering enabled)")
        vec_env = SubprocMuJoCoEnv(
            num_envs=num_envs,
            model_path=config.model_path,
            starting_position=config.starting_position,
            lift_threshold=config.lift_threshold,
        )
    else:
        from vectorized_env import VectorizedMuJoCoEnv
        print(f"\n[Parallel Mode] Running {num_envs} environments in parallel")
        vec_env = VectorizedMuJoCoEnv(
            num_envs=num_envs,
            model_path=config.model_path,
            starting_position=config.starting_position,
            lift_threshold=config.lift_threshold,
        )
    
    # Setup ReinFlow policy
    print("\n" + "="*60)
    print("Setting up ReinFlow SmolVLA")
    print("="*60)
    
    rl_policy = setup_reinflow_policy(
        pretrained_path=config.pretrained_path,
        device=str(device),
        num_steps=config.num_denoising_steps,
        init_log_sigma=config.init_log_sigma,
        train_action_head=config.train_action_head,
        train_time_mlp=config.train_time_mlp,
        train_full_expert=config.train_full_expert,
    )
    
    # Load checkpoint if resuming
    start_episode = 0
    episode_rewards_history = []
    wandb_run_id = None
    
    checkpoint_to_load = args.resume if args else None
    if checkpoint_to_load is None and os.path.exists(config.checkpoint_path):
        checkpoint_to_load = config.checkpoint_path
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        start_episode, wandb_run_id = load_reinflow_checkpoint(rl_policy, checkpoint_to_load, str(device))
    
    # Initialize wandb AFTER loading checkpoint so we can resume the same run
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            id=wandb_run_id,
            resume="allow",
            config={
                "lr": config.lr,
                "gamma": config.gamma,
                "batch_size": config.batch_size,
                "num_denoising_steps": config.num_denoising_steps,
                "init_log_sigma": config.init_log_sigma,
                "entropy_coef": config.entropy_coef,
                "max_steps_per_episode": config.max_steps_per_episode,
                "train_action_head": config.train_action_head,
                "train_time_mlp": config.train_time_mlp,
                "train_full_expert": config.train_full_expert,
                "num_parallel_envs": config.num_parallel_envs,
                "use_subproc_env": config.use_subproc_env,
            },
        )
        wandb_run_id = wandb.run.id  # Update to current run ID for saving
    
    # Optimizer
    trainable_params = rl_policy.get_trainable_params()
    optimizer = torch.optim.Adam(trainable_params, lr=config.lr)
    
    env_mode = "PARALLEL MODE - SUBPROC" if config.use_subproc_env else "PARALLEL MODE"
    print(f"\n{'='*60}")
    print(f"Starting ReinFlow Training ({env_mode})")
    print(f"{'='*60}")
    print(f"Instruction: '{config.instruction}'")
    print(f"Parallel environments: {num_envs}")
    print(f"Subproc rendering: {config.use_subproc_env}")
    print(f"Episodes per batch: {num_envs} (each batch = {num_envs} parallel episodes)")
    print(f"Total batches: {config.num_episodes // num_envs}")
    print(f"Max steps per episode: {config.max_steps_per_episode}")
    print(f"Denoising steps: {config.num_denoising_steps}")
    print(f"Learning rate: {config.lr}")
    print(f"Gradient clip norm: {config.grad_clip_norm}")
    print(f"Initial sigmas: {rl_policy.get_sigmas().data.cpu().numpy()}")
    print(f"{'='*60}\n")
    
    # Calculate number of batches
    num_batches = config.num_episodes // num_envs
    total_episodes = 0
    
    # Running baseline for variance reduction (key for REINFORCE!)
    # Initialize as None - will be set from first batch's mean
    baseline = None
    
    try:
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            
            # Reset all environments
            vec_env.reset_all()
            
            # Get batched observations from all environments (initial state)
            obs_dict = vec_env.get_batched_observations(device)
            
            # Add language tokens
            observation = prepare_batched_observation(
                obs_dict, config.instruction, device, rl_policy, num_envs
            )
            
            # SINGLE BATCHED INFERENCE - get FULL action chunks for all N environments!
            # action_chunks: (N, chunk_size, action_dim) - all 50 actions per env
            # log_probs: (N,) - one log prob per env for the entire chunk
            action_chunks, log_probs = rl_policy.forward_batched_chunks(observation)
            
            # Unnormalize all actions in chunks (batched)
            action_chunks_np = action_chunks.detach().cpu().numpy()
            # action_chunks_np is (N, chunk_size, action_dim)
            action_chunks_radians = np.stack([
                np.stack([unnormalize_action_from_smolvla(a) for a in chunk])
                for chunk in action_chunks_np
            ])
            
            # Execute ALL 50 actions for each environment, accumulating rewards
            # This is ~50x more efficient than querying policy each step!
            total_rewards, dones = vec_env.step_all_chunk(
                action_chunks_radians, config.steps_per_action
            )
            
            # Simplified: one log_prob and one total_reward per environment
            batch_log_probs = log_probs  # (N,) tensor
            batch_total_rewards = total_rewards.tolist()  # List of N floats
            
            # For REINFORCE, we treat total_reward as the return for the chunk
            all_log_probs_tensor = batch_log_probs
            all_returns_tensor = torch.tensor(batch_total_rewards, device=device, dtype=torch.float32)
            
            # Update running baseline (exponential moving average)
            batch_mean = all_returns_tensor.mean().item()
            if baseline is None:
                baseline = batch_mean  # Initialize from first batch!
            else:
                baseline = 0.9 * baseline + 0.1 * batch_mean  # Faster adaptation
            
            # Compute advantages using baseline (key for REINFORCE variance reduction!)
            # This tells the policy "is this better or worse than average?"
            advantages = all_returns_tensor - baseline
            advantages = torch.clamp(advantages, -10.0, 10.0)  # Clip outliers!
            # Do NOT normalize - magnitude matters for learning rate scaling
            
            # Policy gradient loss
            policy_loss = -(advantages * all_log_probs_tensor).mean()
            
            # Entropy bonus
            entropy_bonus = config.entropy_coef * rl_policy.log_sigmas.mean()
            
            # Total loss
            loss = policy_loss - entropy_bonus
            
            # Update policy
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip_norm)
            optimizer.step()
            
            # Clamp sigma to [0.1, 1.0] - allows natural learning within bounds
            with torch.no_grad():
                rl_policy.log_sigmas.clamp_(min=-2.3, max=0.0)
            
            # Track episode rewards
            episode_rewards_history.extend(batch_total_rewards)
            total_episodes += num_envs
            
            batch_time = time.time() - batch_start_time
            
            # Logging
            avg_reward = np.mean(batch_total_rewards)
            current_sigmas = rl_policy.get_sigmas().data.cpu().numpy()
            print(f"Batch {batch_idx+1:4d} ({total_episodes:5d} eps) | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Baseline: {baseline:8.2f} | "
                  f"Loss: {loss.item():8.4f} | "
                  f"σ_mean: {current_sigmas.mean():.4f} | "
                  f"Time: {batch_time:.1f}s ({batch_time/num_envs:.2f}s/ep)")
            
            # Log to Weights & Biases
            if config.wandb_enabled:
                wandb.log({
                    "batch": batch_idx + 1,
                    "episodes_total": total_episodes,
                    "reward/batch_avg": avg_reward,
                    "reward/batch_min": np.min(batch_total_rewards),
                    "reward/batch_max": np.max(batch_total_rewards),
                    "reward/baseline": baseline,
                    "loss/policy": loss.item(),
                    "exploration/sigma_mean": current_sigmas.mean(),
                    "time/batch_seconds": batch_time,
                    "time/per_episode": batch_time / num_envs,
                })
            
            # Save checkpoint periodically
            if (batch_idx + 1) % (config.save_interval // num_envs + 1) == 0:
                save_reinflow_checkpoint(
                    rl_policy, total_episodes - 1, episode_rewards_history, config.checkpoint_path, wandb_run_id
                )
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        vec_env.close()
        if config.wandb_enabled:
            wandb.finish()
    
    # Save final checkpoint
    save_reinflow_checkpoint(
        rl_policy, total_episodes - 1, episode_rewards_history, config.checkpoint_path, wandb_run_id
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final sigmas: {rl_policy.get_sigmas().data.cpu().numpy()}")
    print(f"Total episodes: {len(episode_rewards_history)}")
    if episode_rewards_history:
        print(f"Best reward: {max(episode_rewards_history):.2f}")
        print(f"Final avg reward (last 100): {np.mean(episode_rewards_history[-100:]):.2f}")
    
    return rl_policy, episode_rewards_history


# ===== Sequential Training Loop (for M1/CPU) =====

def train_sequential(config, args, device):
    """
    Sequential training loop for ReinFlow (original implementation).
    
    Best for M1 Mac and CPU where single-sample inference is efficient.
    
    Key differences from ReinFlow-lite:
    1. Noise injected at EACH denoising step (not just output)
    2. Log-probabilities computed through Markov chain
    3. Exact REINFORCE gradient (not approximation)
    """
    # Load MuJoCo environment
    print(f"\nLoading MuJoCo model from {config.model_path}")
    m = mujoco.MjModel.from_xml_path(config.model_path)
    d = mujoco.MjData(m)
    renderer = mujoco.Renderer(m, height=256, width=256)
    
    # Setup ReinFlow policy
    print("\n" + "="*60)
    print("Setting up ReinFlow SmolVLA")
    print("="*60)
    
    rl_policy = setup_reinflow_policy(
        pretrained_path=config.pretrained_path,
        device=str(device),
        num_steps=config.num_denoising_steps,
        init_log_sigma=config.init_log_sigma,
        train_action_head=config.train_action_head,
        train_time_mlp=config.train_time_mlp,
        train_full_expert=config.train_full_expert,
    )
    
    # Load checkpoint if resuming or if specified
    start_episode = 0
    episode_rewards_history = []
    wandb_run_id = None
    
    checkpoint_to_load = args.resume if args else None
    if checkpoint_to_load is None and os.path.exists(config.checkpoint_path):
        checkpoint_to_load = config.checkpoint_path
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        start_episode, wandb_run_id = load_reinflow_checkpoint(rl_policy, checkpoint_to_load, str(device))
    
    # Initialize wandb AFTER loading checkpoint so we can resume the same run
    if config.wandb_enabled:
        wandb.init(
            project=config.wandb_project,
            id=wandb_run_id,
            resume="allow",
            config={
                "lr": config.lr,
                "gamma": config.gamma,
                "batch_size": config.batch_size,
                "num_denoising_steps": config.num_denoising_steps,
                "init_log_sigma": config.init_log_sigma,
                "entropy_coef": config.entropy_coef,
                "max_steps_per_episode": config.max_steps_per_episode,
                "train_action_head": config.train_action_head,
                "train_time_mlp": config.train_time_mlp,
                "train_full_expert": config.train_full_expert,
            },
        )
        wandb_run_id = wandb.run.id  # Update to current run ID for saving
    
    # Optimizer with all trainable parameters
    trainable_params = rl_policy.get_trainable_params()
    optimizer = torch.optim.Adam(trainable_params, lr=config.lr)
    
    print(f"\n{'='*60}")
    print(f"Starting ReinFlow Training")
    print(f"{'='*60}")
    print(f"Instruction: '{config.instruction}'")
    print(f"Episodes: {config.num_episodes}")
    print(f"Max steps per episode: {config.max_steps_per_episode}")
    print(f"Denoising steps: {config.num_denoising_steps}")
    print(f"Learning rate: {config.lr}")
    print(f"Batch size: {config.batch_size} episodes per update")
    print(f"Gradient clip norm: {config.grad_clip_norm}")
    print(f"Training action head: {config.train_action_head}")
    print(f"Training time MLP: {config.train_time_mlp}")
    print(f"Entropy coefficient: {config.entropy_coef}")
    print(f"Initial sigmas: {rl_policy.get_sigmas().data.cpu().numpy()}")
    print(f"{'='*60}\n")
    
    # Optional viewer for visualization
    viewer = None
    if config.render:
        viewer = mujoco.viewer.launch_passive(m, d)
    
    try:
        # Batch storage for accumulating multiple episodes
        batch_log_probs = []
        batch_returns = []
        batch_episode_rewards = []  # Track rewards for each episode in batch
        batch_start_time = time.time()
        
        # Running baseline for variance reduction (key for REINFORCE!)
        # Initialize as None - will be set from first batch's mean
        baseline = None
        
        for episode in range(start_episode, config.num_episodes):
            episode_start_time = time.time()
            
            # Reset environment and reward state
            reset_env(m, d, config.starting_position)
            reset_reward_state()
            
            # Get initial observation
            rgb_top = get_camera_observation(renderer, d, camera_name="camera_up")
            rgb_wrist = get_camera_observation(renderer, d, camera_name="wrist_camera")
            rgb_side = get_camera_observation(renderer, d, camera_name="camera_side")
            robot_state = get_robot_state(d)
            
            # Prepare observation for ReinFlow policy
            observation = prepare_observation_for_reinflow(
                rgb_top, rgb_wrist, rgb_side, robot_state,
                config.instruction, device, rl_policy
            )
            
            # Get FULL action chunk from ReinFlow policy (with noise at each denoising step!)
            # action: first action, log_prob: log prob for entire chunk, action_chunk: all 50 actions
            _, log_prob, action_chunk = rl_policy(observation)
            
            # Execute ALL actions in the chunk, accumulating rewards
            total_reward = 0.0
            done = False
            chunk_size = action_chunk.shape[0]  # Should be 50
            
            for action_idx in range(chunk_size):
                if done:
                    break
                    
                # Get action for this timestep
                action = action_chunk[action_idx]
                action_np = action.detach().cpu().numpy()
                action_radians = unnormalize_action_from_smolvla(action_np)
                action_dict = convert_to_dictionary(action_radians)
                
                # Execute action for multiple physics steps
                for _ in range(config.steps_per_action):
                    send_position_command(d, action_dict)
                    mujoco.mj_step(m, d)
                    if viewer is not None:
                        viewer.sync()
                
                # Get reward for this step
                reward, done = compute_reward(m, d, lift_threshold=config.lift_threshold)
                total_reward += reward
                
                if done:
                    print(f"  Episode {episode+1}: SUCCESS! Block lifted at action {action_idx+1}/{chunk_size}")
            
            # Simplified: one log_prob and one total_reward per episode
            # Add to batch (single values, not lists)
            batch_log_probs.append(log_prob)
            batch_returns.append(torch.tensor(total_reward, device=device, dtype=torch.float32))
            
            # Track episode reward
            episode_rewards_history.append(total_reward)
            batch_episode_rewards.append(total_reward)
            
            # Only update policy every batch_size episodes
            if (episode + 1) % config.batch_size == 0:
                # Stack all episodes in batch (each is a single value now)
                all_log_probs = torch.stack(batch_log_probs)
                all_returns = torch.stack(batch_returns)
                
                # Update running baseline (exponential moving average)
                batch_mean = all_returns.mean().item()
                if baseline is None:
                    baseline = batch_mean  # Initialize from first batch!
                else:
                    baseline = 0.9 * baseline + 0.1 * batch_mean  # Faster adaptation
                
                # Compute advantages using baseline (key for REINFORCE variance reduction!)
                advantages = all_returns - baseline
                advantages = torch.clamp(advantages, -10.0, 10.0)  # Clip outliers!
                # Do NOT normalize - magnitude matters for learning rate scaling
                
                # Policy gradient loss: -E[A(s,a) * log π(a|s)]
                policy_loss = -(advantages * all_log_probs).mean()
                
                # Entropy bonus: encourage exploration by penalizing low sigma
                entropy_bonus = config.entropy_coef * rl_policy.log_sigmas.mean()
                
                # Total loss (subtract entropy bonus to encourage higher sigma)
                loss = policy_loss - entropy_bonus
                
                # Update policy
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip_norm)
                
                optimizer.step()
                
                # Clamp sigma to [0.1, 1.0] - allows natural learning within bounds
                with torch.no_grad():
                    rl_policy.log_sigmas.clamp_(min=-2.3, max=0.0)
                
                # Clear batch
                batch_log_probs = []
                batch_returns = []
                batch_episode_rewards = []
                batch_start_time = time.time()
            else:
                # No update yet, set loss to None for logging
                loss = None
            
            episode_time = time.time() - episode_start_time
            
            # Logging
            if (episode + 1) % config.log_interval == 0 or episode == 0:
                avg_reward = np.mean(episode_rewards_history[-config.log_interval:])
                current_sigmas = rl_policy.get_sigmas().data.cpu().numpy()
                loss_str = f"{loss.item():8.4f}" if loss is not None else "    N/A "
                baseline_str = f"{baseline:8.2f}" if baseline is not None else "    N/A "
                print(f"Episode {episode+1:5d} | "
                      f"Reward: {total_reward:8.2f} | "
                      f"Avg: {avg_reward:8.2f} | "
                      f"Baseline: {baseline_str} | "
                      f"Loss: {loss_str} | "
                      f"σ_mean: {current_sigmas.mean():.4f} | "
                      f"Time: {episode_time:.1f}s")
                
                # Log to Weights & Biases
                if config.wandb_enabled:
                    wandb.log({
                        "episode": episode + 1,
                        "reward/episode": total_reward,
                        "reward/avg": avg_reward,
                        "reward/baseline": baseline,
                        "loss/policy": loss.item() if loss is not None else None,
                        "exploration/sigma_mean": current_sigmas.mean(),
                        "exploration/sigma_min": current_sigmas.min(),
                        "exploration/sigma_max": current_sigmas.max(),
                        "time/episode_seconds": episode_time,
                        "steps/episode": chunk_size,  # Full chunk executed
                    })
            
            # Print 10-episode average every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_10 = np.mean(episode_rewards_history[-10:])
                print(f"  >>> Last 10 episodes avg: {avg_10:.2f}")
            
            # Save checkpoint periodically
            if (episode + 1) % config.save_interval == 0:
                save_reinflow_checkpoint(
                    rl_policy, episode, episode_rewards_history, config.checkpoint_path, wandb_run_id
                )
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        if viewer is not None:
            viewer.close()
        if config.wandb_enabled:
            wandb.finish()
    
    # Save final checkpoint
    save_reinflow_checkpoint(
        rl_policy, episode, episode_rewards_history, config.checkpoint_path, wandb_run_id
    )
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final sigmas: {rl_policy.get_sigmas().data.cpu().numpy()}")
    print(f"Total episodes: {len(episode_rewards_history)}")
    if episode_rewards_history:
        print(f"Best reward: {max(episode_rewards_history):.2f}")
        print(f"Final avg reward (last 100): {np.mean(episode_rewards_history[-100:]):.2f}")
    
    return rl_policy, episode_rewards_history


# ===== Main Entry Point =====

def train(config=None, args=None):
    """
    Main training entry point - dispatches to parallel or sequential training.
    
    Automatically chooses parallel mode on CUDA with --parallel-envs flag,
    otherwise uses sequential mode (best for M1/CPU).
    """
    if config is None:
        config = TrainingConfig()
    
    # Apply command line overrides
    if args is not None:
        if args.no_render:
            config.render = False
        if args.headless:
            config.render = False
        if args.episodes is not None:
            config.num_episodes = args.episodes
        if args.lr is not None:
            config.lr = args.lr
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
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"Using device: {device} (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: {device}")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")
    
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

