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
    num_episodes = 20000
    max_steps_per_episode = 50
    gamma = 0.99  # Discount factor
    lr = 5e-3
    grad_clip_norm = 1.0
    batch_size = 20  # Number of episodes to accumulate before gradient update
    
    # ReinFlow specific
    num_denoising_steps = 10  # Must match SmolVLA config
    init_log_sigma = -0.7    # Initial noise scale (exp(-1) ≈ 0.37, more exploration)
    entropy_coef = 0.0001      # Entropy bonus to prevent sigma collapse
    
    # What to train
    train_action_head = True   # Train action_out_proj (23K params)
    train_time_mlp = False     # Also train action_time_mlp_out (519K params)
    
    # Policy execution
    steps_per_action = 10  # Physics steps per policy action
    
    # Reward
    lift_threshold = 0.08
    
    # Logging
    log_interval = 1
    save_interval = 10
    
    # Rendering
    render = True  # Set False for faster training
    
    # Checkpointing
    checkpoint_path = "reinflow_checkpoint.pt"
    pretrained_path = "lerobot/smolvla_base"
    
    # Parallelization (A100 optimization)
    # Set >1 to run multiple environments in parallel for GPU efficiency
    # Default 1 = sequential mode (best for M1 Mac)
    num_parallel_envs = 1


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
    return parser.parse_args()


# ===== Parallel Training Loop (for A100) =====

def train_parallel(config, args, device):
    """
    Parallel training loop using vectorized environments.
    
    Runs N environments in parallel, batching observations for efficient
    GPU inference. Best for A100/CUDA where batch processing is fast.
    """
    from vectorized_env import VectorizedMuJoCoEnv
    
    num_envs = config.num_parallel_envs
    print(f"\n[Parallel Mode] Running {num_envs} environments in parallel")
    
    # Create vectorized environment
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
    )
    
    # Load checkpoint if resuming
    start_episode = 0
    episode_rewards_history = []
    
    checkpoint_to_load = args.resume if args else None
    if checkpoint_to_load is None and os.path.exists(config.checkpoint_path):
        checkpoint_to_load = config.checkpoint_path
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        start_episode = load_reinflow_checkpoint(rl_policy, checkpoint_to_load, str(device))
    
    # Optimizer
    trainable_params = rl_policy.get_trainable_params()
    optimizer = torch.optim.Adam(trainable_params, lr=config.lr)
    
    print(f"\n{'='*60}")
    print(f"Starting ReinFlow Training (PARALLEL MODE)")
    print(f"{'='*60}")
    print(f"Instruction: '{config.instruction}'")
    print(f"Parallel environments: {num_envs}")
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
    
    try:
        for batch_idx in range(num_batches):
            batch_start_time = time.time()
            
            # Reset all environments
            vec_env.reset_all()
            
            # Storage for all environments: list of lists
            # all_log_probs[env_idx] = [log_prob_step0, log_prob_step1, ...]
            all_log_probs = [[] for _ in range(num_envs)]
            all_rewards = [[] for _ in range(num_envs)]
            
            # Track which envs are still running
            active_envs = np.ones(num_envs, dtype=bool)
            
            # Run episode steps
            for step in range(config.max_steps_per_episode):
                # Get batched observations from all environments
                obs_dict = vec_env.get_batched_observations(device)
                
                # Add language tokens
                observation = prepare_batched_observation(
                    obs_dict, config.instruction, device, rl_policy, num_envs
                )
                
                # BATCHED INFERENCE - single GPU call for all N environments!
                actions, log_probs = rl_policy.forward_batched(observation)
                
                # Unnormalize actions (batched)
                actions_np = actions.detach().cpu().numpy()
                actions_radians = np.stack([
                    unnormalize_action_from_smolvla(a) for a in actions_np
                ])
                
                # Step all environments
                rewards, dones = vec_env.step_all(actions_radians, config.steps_per_action)
                
                # Store results for active environments
                for i in range(num_envs):
                    if active_envs[i]:
                        all_log_probs[i].append(log_probs[i])
                        all_rewards[i].append(rewards[i])
                        if dones[i]:
                            active_envs[i] = False
                
                # Check if all environments are done
                if not active_envs.any():
                    break
            
            # Compute returns and losses for all environments
            batch_log_probs = []
            batch_returns = []
            batch_total_rewards = []
            
            for i in range(num_envs):
                if len(all_rewards[i]) > 0:
                    # Compute returns
                    returns = compute_returns(all_rewards[i], gamma=config.gamma)
                    returns = torch.tensor(returns, device=device, dtype=torch.float32)
                    
                    # Stack log probs
                    log_probs = torch.stack(all_log_probs[i])
                    
                    batch_log_probs.append(log_probs)
                    batch_returns.append(returns)
                    batch_total_rewards.append(sum(all_rewards[i]))
            
            # Concatenate all environments
            all_log_probs_tensor = torch.cat(batch_log_probs)
            all_returns_tensor = torch.cat(batch_returns)
            
            # Normalize advantages
            advantages = all_returns_tensor - all_returns_tensor.mean()
            if all_returns_tensor.std() > 1e-8:
                advantages = advantages / (all_returns_tensor.std() + 1e-8)
            
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
            
            # Track episode rewards
            episode_rewards_history.extend(batch_total_rewards)
            total_episodes += num_envs
            
            batch_time = time.time() - batch_start_time
            
            # Logging
            avg_reward = np.mean(batch_total_rewards)
            current_sigmas = rl_policy.get_sigmas().data.cpu().numpy()
            print(f"Batch {batch_idx+1:4d} ({total_episodes:5d} eps) | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Loss: {loss.item():8.4f} | "
                  f"σ_mean: {current_sigmas.mean():.4f} | "
                  f"Time: {batch_time:.1f}s ({batch_time/num_envs:.2f}s/ep)")
            
            # Save checkpoint periodically
            if (batch_idx + 1) % (config.save_interval // num_envs + 1) == 0:
                save_reinflow_checkpoint(
                    rl_policy, total_episodes - 1, episode_rewards_history, config.checkpoint_path
                )
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        vec_env.close()
    
    # Save final checkpoint
    save_reinflow_checkpoint(
        rl_policy, total_episodes - 1, episode_rewards_history, config.checkpoint_path
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
    )
    
    # Load checkpoint if resuming or if specified
    start_episode = 0
    episode_rewards_history = []
    
    checkpoint_to_load = args.resume if args else None
    if checkpoint_to_load is None and os.path.exists(config.checkpoint_path):
        checkpoint_to_load = config.checkpoint_path
    
    if checkpoint_to_load and os.path.exists(checkpoint_to_load):
        start_episode = load_reinflow_checkpoint(rl_policy, checkpoint_to_load, str(device))
    
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
        
        for episode in range(start_episode, config.num_episodes):
            episode_start_time = time.time()
            
            # Reset environment and reward state
            reset_env(m, d, config.starting_position)
            reset_reward_state()
            
            # Episode storage
            episode_log_probs = []
            episode_rewards = []
            
            # Run episode
            for step in range(config.max_steps_per_episode):
                # Get all three camera observations (top, wrist, and side for SmolVLA)
                rgb_top = get_camera_observation(renderer, d, camera_name="camera_up")
                rgb_wrist = get_camera_observation(renderer, d, camera_name="wrist_camera")
                rgb_side = get_camera_observation(renderer, d, camera_name="camera_side")
                robot_state = get_robot_state(d)
                
                # Prepare observation for ReinFlow policy
                observation = prepare_observation_for_reinflow(
                    rgb_top, rgb_wrist, rgb_side, robot_state,
                    config.instruction, device, rl_policy
                )
                
                # Get action from ReinFlow policy (with noise at each denoising step!)
                action, log_prob, _ = rl_policy(observation)
                
                # Convert to numpy and unnormalize (normalized -> degrees -> radians)
                action_np = action.detach().cpu().numpy()
                action_radians = unnormalize_action_from_smolvla(action_np)
                action_dict = convert_to_dictionary(action_radians)
                
                # Execute action for multiple physics steps
                for _ in range(config.steps_per_action):
                    send_position_command(d, action_dict)
                    mujoco.mj_step(m, d)
                    if viewer is not None:
                        viewer.sync()
                
                # Get reward
                reward, done = compute_reward(m, d, lift_threshold=config.lift_threshold)
                
                # Store for policy gradient
                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)
                
                if done:
                    print(f"  Episode {episode+1}: SUCCESS! Block lifted at step {step+1}")
                    break
            
            # Compute returns (rewards-to-go) for this episode
            returns = compute_returns(episode_rewards, gamma=config.gamma)
            returns = torch.tensor(returns, device=device, dtype=torch.float32)
            
            # Stack log probs for this episode
            log_probs = torch.stack(episode_log_probs)
            
            # Add to batch
            batch_log_probs.append(log_probs)
            batch_returns.append(returns)
            
            # Track episode reward
            total_reward = sum(episode_rewards)
            episode_rewards_history.append(total_reward)
            batch_episode_rewards.append(total_reward)
            
            # Only update policy every batch_size episodes
            if (episode + 1) % config.batch_size == 0:
                # Concatenate all episodes in batch
                all_log_probs = torch.cat(batch_log_probs)
                all_returns = torch.cat(batch_returns)
                
                # Normalize advantages across entire batch
                advantages = all_returns - all_returns.mean()
                if all_returns.std() > 1e-8:
                    advantages = advantages / (all_returns.std() + 1e-8)
                
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
                print(f"Episode {episode+1:5d} | "
                      f"Reward: {total_reward:8.2f} | "
                      f"Avg: {avg_reward:8.2f} | "
                      f"Loss: {loss_str} | "
                      f"σ_mean: {current_sigmas.mean():.4f} | "
                      f"Time: {episode_time:.1f}s")
            
            # Print 10-episode average every 10 episodes
            if (episode + 1) % 10 == 0:
                avg_10 = np.mean(episode_rewards_history[-10:])
                print(f"  >>> Last 10 episodes avg: {avg_10:.2f}")
            
            # Save checkpoint periodically
            if (episode + 1) % config.save_interval == 0:
                save_reinflow_checkpoint(
                    rl_policy, episode, episode_rewards_history, config.checkpoint_path
                )
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        if viewer is not None:
            viewer.close()
    
    # Save final checkpoint
    save_reinflow_checkpoint(
        rl_policy, episode, episode_rewards_history, config.checkpoint_path
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

