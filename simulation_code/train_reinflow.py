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
"""

import os
import sys
import time
import argparse
import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np

from reinflow_smolvla import (
    ReinFlowSmolVLA,
    setup_reinflow_policy,
    prepare_observation_for_reinflow,
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
    lr = 1e-3
    grad_clip_norm = 1.0
    
    # ReinFlow specific
    num_denoising_steps = 10  # Must match SmolVLA config
    init_log_sigma = -1.0    # Initial noise scale (exp(-1) ≈ 0.37, more exploration)
    entropy_coef = 0.0001      # Entropy bonus to prevent sigma collapse
    
    # What to train
    train_action_head = True   # Train action_out_proj (23K params)
    train_time_mlp = False     # Also train action_time_mlp_out (519K params)
    
    # Policy execution
    steps_per_action = 10  # Physics steps per policy action
    
    # Reward
    lift_threshold = 0.08
    
    # Logging
    log_interval = 10
    save_interval = 100
    
    # Rendering
    render = True  # Set False for faster training
    
    # Checkpointing
    checkpoint_path = "reinflow_checkpoint.pt"
    pretrained_path = "lerobot/smolvla_base"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='ReinFlow Training for SmolVLA')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable visualization for faster training')
    parser.add_argument('--episodes', type=int, default=None,
                        help='Number of episodes to train')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pretrained SmolVLA model')
    return parser.parse_args()


# ===== Main Training Loop =====

def train(config=None, args=None):
    """
    Main training loop for ReinFlow.
    
    Key differences from ReinFlow-lite:
    1. Noise injected at EACH denoising step (not just output)
    2. Log-probabilities computed through Markov chain
    3. Exact REINFORCE gradient (not approximation)
    """
    if config is None:
        config = TrainingConfig()
    
    # Apply command line overrides
    if args is not None:
        if args.no_render:
            config.render = False
        if args.episodes is not None:
            config.num_episodes = args.episodes
        if args.lr is not None:
            config.lr = args.lr
        if args.pretrained is not None:
            config.pretrained_path = args.pretrained
    
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
                # Get observations
                rgb_top = get_camera_observation(renderer, d, camera_name="camera_up")
                rgb_wrist = get_camera_observation(renderer, d, camera_name="wrist_camera")
                robot_state = get_robot_state(d)
                
                # Prepare observation for ReinFlow policy
                observation = prepare_observation_for_reinflow(
                    rgb_top, rgb_wrist, robot_state,
                    config.instruction, device, rl_policy
                )
                
                # Get action from ReinFlow policy (with noise at each denoising step!)
                action, log_prob, _ = rl_policy(observation)
                
                # Convert to numpy and execute
                action_np = action.detach().cpu().numpy()
                action_dict = convert_to_dictionary(action_np)
                
                # Execute action for multiple physics steps
                for _ in range(config.steps_per_action):
                    send_position_command(d, action_dict)
                    mujoco.mj_step(m, d)
                    if viewer is not None:
                        viewer.sync()
                
                # Get reward
                reward, done = compute_reward(d, lift_threshold=config.lift_threshold)
                
                # Store for policy gradient
                episode_log_probs.append(log_prob)
                episode_rewards.append(reward)
                
                if done:
                    print(f"  Episode {episode+1}: SUCCESS! Block lifted at step {step+1}")
                    break
            
            # Compute returns (rewards-to-go)
            returns = compute_returns(episode_rewards, gamma=config.gamma)
            returns = torch.tensor(returns, device=device, dtype=torch.float32)
            
            # Stack log probs
            log_probs = torch.stack(episode_log_probs)
            
            # Normalize advantages (returns - baseline)
            advantages = returns - returns.mean()
            if returns.std() > 1e-8:
                advantages = advantages / (returns.std() + 1e-8)
            
            # Policy gradient loss: -E[A(s,a) * log π(a|s)]
            policy_loss = -(advantages * log_probs).mean()
            
            # Entropy bonus: encourage exploration by penalizing low sigma
            # Higher log_sigma = higher entropy = more exploration
            entropy_bonus = config.entropy_coef * rl_policy.log_sigmas.mean()
            
            # Total loss (subtract entropy bonus to encourage higher sigma)
            loss = policy_loss - entropy_bonus
            
            # Update policy
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip_norm)
            
            optimizer.step()
            
            # Track episode reward
            total_reward = sum(episode_rewards)
            episode_rewards_history.append(total_reward)
            
            episode_time = time.time() - episode_start_time
            
            # Logging
            if (episode + 1) % config.log_interval == 0 or episode == 0:
                avg_reward = np.mean(episode_rewards_history[-config.log_interval:])
                current_sigmas = rl_policy.get_sigmas().data.cpu().numpy()
                print(f"Episode {episode+1:5d} | "
                      f"Reward: {total_reward:8.2f} | "
                      f"Avg: {avg_reward:8.2f} | "
                      f"Loss: {loss.item():8.4f} | "
                      f"σ_mean: {current_sigmas.mean():.4f} | "
                      f"Time: {episode_time:.1f}s")
            
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


if __name__ == "__main__":
    args = parse_args()
    train(args=args)

