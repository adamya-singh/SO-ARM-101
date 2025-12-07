"""
ReinFlow-lite: Gaussian Policy Wrapper for SmolVLA

This script implements a simplified reinforcement learning approach that:
1. Wraps SmolVLA's deterministic output as the mean of a Gaussian distribution
2. Adds learnable log-std parameters for exploration
3. Uses REINFORCE policy gradient to fine-tune the exploration parameters

Usage:
    conda activate lerobot
    python train_reinflow_lite.py
"""

import os
import time
import mujoco
import mujoco.viewer
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from so101_mujoco_utils import (
    set_initial_pose,
    send_position_command,
    convert_to_dictionary,
    get_camera_observation,
    get_robot_state,
    prepare_observation,
    compute_reward,
    reset_env,
    reset_reward_state,
)


# ===== Gaussian Policy Wrapper =====

class GaussianSmolVLA(nn.Module):
    """
    Wraps SmolVLA to output a Gaussian distribution over actions.
    
    The base SmolVLA policy output becomes the mean (mu), and we add
    learnable log-std parameters for each action dimension.
    
    Args:
        base_policy: SmolVLAPolicy instance
        act_dim: Number of action dimensions (default: 6 for SO-ARM-101)
        init_log_std: Initial log standard deviation (default: -2.0)
        device: Torch device
        train_action_head: If True, unfreeze action_out_proj (23K params)
        train_time_mlp: If True, also unfreeze action_time_mlp_out (519K params)
    """
    
    def __init__(self, base_policy, act_dim=6, init_log_std=-1.0, device='cpu',
                 train_action_head=True, train_time_mlp=False):
        super().__init__()
        self.base = base_policy
        self.device = device
        self.train_action_head = train_action_head
        self.train_time_mlp = train_time_mlp
        
        # Freeze entire base policy first
        for p in self.base.parameters():
            p.requires_grad = False
        
        # Selectively unfreeze action output projection head
        if train_action_head:
            for p in self.base.model.action_out_proj.parameters():
                p.requires_grad = True
            print(f"  Unfroze action_out_proj: {sum(p.numel() for p in self.base.model.action_out_proj.parameters()):,} params")
        
        # Optionally unfreeze time MLP output layer
        if train_time_mlp:
            for p in self.base.model.action_time_mlp_out.parameters():
                p.requires_grad = True
            print(f"  Unfroze action_time_mlp_out: {sum(p.numel() for p in self.base.model.action_time_mlp_out.parameters()):,} params")
        
        # Learnable per-joint log standard deviation
        # Initialize with small std for stability (exp(-2) ≈ 0.135)
        self.log_std = nn.Parameter(
            torch.full((act_dim,), init_log_std, device=device)
        )
        
        # Count total trainable params
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  Total trainable parameters: {total_trainable:,}")
    
    def forward(self, observation):
        """
        Forward pass: get mean from SmolVLA, sample from Gaussian.
        
        Args:
            observation: dict with observation tensors
            
        Returns:
            action: sampled action tensor (act_dim,)
            log_prob: log probability of the action (scalar)
            mu: mean action from SmolVLA (act_dim,)
            std: standard deviation (act_dim,)
        """
        # Get action from SmolVLA as mean
        # No torch.no_grad() - we need gradients to flow through the action head!
        # The backbone is frozen via requires_grad=False, so only unfrozen params get gradients
        mu = self.base.select_action(observation)
        
        # Handle tensor shape - ensure it's (act_dim,)
        if mu.dim() > 1:
            mu = mu.squeeze(0)
        
        # Move mu to same device as log_std and ensure float32
        mu = mu.to(self.device).float()
        
        # Build Gaussian distribution
        std = self.log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        
        # Reparameterized sample (allows gradient flow through std and mu)
        action = dist.rsample()
        
        # Compute log probability (sum over action dimensions)
        log_prob = dist.log_prob(action).sum(-1)
        
        return action, log_prob, mu, std


# ===== Helper Functions =====

def compute_returns(rewards, gamma=0.95):
    """
    Compute discounted returns (rewards-to-go).
    
    Args:
        rewards: list of rewards for each timestep
        gamma: discount factor
        
    Returns:
        list of discounted returns
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def setup_smolvla(device):
    """
    Load and configure SmolVLA policy.
    
    Returns:
        policy: SmolVLAPolicy instance
    """
    print("Loading SmolVLA policy...")
    policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
    policy.to(device)
    policy.eval()
    print("SmolVLA policy loaded successfully!")
    
    # Load tokenizer if missing
    if not hasattr(policy, 'tokenizer') or policy.tokenizer is None:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        policy.tokenizer = tokenizer
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")
    
    return policy


# ===== Training Configuration =====

class TrainingConfig:
    """Configuration for RL training."""
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
    num_episodes = 100000
    max_steps_per_episode = 50
    gamma = 0.95
    lr = 3e-4
    grad_clip_norm = 1.0  # Max gradient norm for clipping
    
    # What to train
    train_action_head = True   # Train action_out_proj (23K params)
    train_time_mlp = False     # Also train action_time_mlp_out (519K params)
    
    # Policy execution
    steps_per_action = 10  # Physics steps per policy action
    
    # Reward
    lift_threshold = 0.08
    
    # Logging
    log_interval = 10
    
    # Rendering
    render = True  # Set False for faster training
    #render = False


# ===== Main Training Loop =====

def train(config=None):
    """
    Main training loop for ReinFlow-lite.
    """
    if config is None:
        config = TrainingConfig()
    
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
    
    # Load SmolVLA and wrap with Gaussian policy
    base_policy = setup_smolvla(device)
    print(f"\nSetting up GaussianSmolVLA wrapper...")
    print(f"  train_action_head: {config.train_action_head}")
    print(f"  train_time_mlp: {config.train_time_mlp}")
    rl_policy = GaussianSmolVLA(
        base_policy, 
        act_dim=6, 
        device=device,
        train_action_head=config.train_action_head,
        train_time_mlp=config.train_time_mlp
    )
    
    # Load checkpoint if it exists
    checkpoint_path = "reinflow_lite_checkpoint.pt"
    start_episode = 0
    if os.path.exists(checkpoint_path):
        print(f"\nLoading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        rl_policy.log_std.data = checkpoint['log_std'].to(device)
        if config.train_action_head and 'action_out_proj' in checkpoint:
            rl_policy.base.model.action_out_proj.load_state_dict(checkpoint['action_out_proj'])
        if config.train_time_mlp and 'action_time_mlp_out' in checkpoint:
            rl_policy.base.model.action_time_mlp_out.load_state_dict(checkpoint['action_time_mlp_out'])
        if 'episode' in checkpoint:
            start_episode = checkpoint['episode'] + 1
        print(f"  Resuming from episode {start_episode}")
        print(f"  Loaded log_std: {rl_policy.log_std.data.cpu().numpy()}")
    
    # Build list of trainable parameters
    trainable_params = [rl_policy.log_std]
    if config.train_action_head:
        trainable_params.extend(rl_policy.base.model.action_out_proj.parameters())
    if config.train_time_mlp:
        trainable_params.extend(rl_policy.base.model.action_time_mlp_out.parameters())
    
    # Optimizer with all trainable parameters
    optimizer = torch.optim.Adam(trainable_params, lr=config.lr)
    
    # Training stats
    episode_rewards_history = []
    
    print(f"\n{'='*60}")
    print(f"Starting ReinFlow-lite Training")
    print(f"{'='*60}")
    print(f"Instruction: '{config.instruction}'")
    print(f"Episodes: {config.num_episodes}")
    print(f"Max steps per episode: {config.max_steps_per_episode}")
    print(f"Learning rate: {config.lr}")
    print(f"Gradient clip norm: {config.grad_clip_norm}")
    print(f"Training action head: {config.train_action_head}")
    print(f"Training time MLP: {config.train_time_mlp}")
    print(f"Initial log_std: {rl_policy.log_std.data.cpu().numpy()}")
    print(f"Initial std: {rl_policy.log_std.exp().data.cpu().numpy()}")
    print(f"{'='*60}\n")
    
    # Optional viewer for visualization
    viewer = None
    if config.render:
        viewer = mujoco.viewer.launch_passive(m, d)
    
    try:
        for episode in range(start_episode, config.num_episodes):
            # Reset environment and reward state
            reset_env(m, d, config.starting_position)
            reset_reward_state()  # Reset velocity tracking for approach reward
            
            # Episode storage
            episode_log_probs = []
            episode_rewards = []
            
            # Run episode
            for step in range(config.max_steps_per_episode):
                # Get observations
                rgb_top = get_camera_observation(renderer, d, camera_name="camera_up")
                rgb_wrist = get_camera_observation(renderer, d, camera_name="wrist_camera")
                robot_state = get_robot_state(d)
                
                # Prepare observation dict for policy
                observation = prepare_observation(
                    rgb_top, rgb_wrist, robot_state,
                    config.instruction, device, base_policy, debug=False
                )
                
                # Get action from Gaussian policy
                action, log_prob, mu, std = rl_policy(observation)
                
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
            
            # Compute returns
            returns = compute_returns(episode_rewards, gamma=config.gamma)
            returns = torch.tensor(returns, device=device, dtype=torch.float32)
            
            # Stack log probs
            log_probs = torch.stack(episode_log_probs)
            
            # Normalize advantages (returns - baseline)
            advantages = returns - returns.mean()
            if returns.std() > 1e-8:
                advantages = advantages / (returns.std() + 1e-8)
            
            # Policy gradient loss
            loss = -(advantages * log_probs).mean()
            
            # Update policy
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=config.grad_clip_norm)
            
            optimizer.step()
            
            # Track episode reward
            total_reward = sum(episode_rewards)
            episode_rewards_history.append(total_reward)
            
            # Logging
            if (episode + 1) % config.log_interval == 0 or episode == 0:
                avg_reward = np.mean(episode_rewards_history[-config.log_interval:])
                current_std = rl_policy.log_std.exp().data.cpu().numpy()
                print(f"Episode {episode+1:4d} | "
                      f"Reward: {total_reward:8.2f} | "
                      f"Avg Reward: {avg_reward:8.2f} | "
                      f"Loss: {loss.item():8.4f} | "
                      f"Std: {current_std.mean():.4f}")
    
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
    
    finally:
        if viewer is not None:
            viewer.close()
    
    # Save trained parameters
    save_path = "reinflow_lite_checkpoint.pt"
    checkpoint = {
        'log_std': rl_policy.log_std.data,
        'episode': episode,
        'episode_rewards': episode_rewards_history,
        'config': vars(config),
    }
    # Save action head weights if trained
    if config.train_action_head:
        checkpoint['action_out_proj'] = rl_policy.base.model.action_out_proj.state_dict()
    if config.train_time_mlp:
        checkpoint['action_time_mlp_out'] = rl_policy.base.model.action_time_mlp_out.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"\nCheckpoint saved to {save_path}")
    
    
    return rl_policy, episode_rewards_history


if __name__ == "__main__":
    train()

