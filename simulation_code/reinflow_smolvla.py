"""
ReinFlow SmolVLA: Full Flow-Based RL Fine-Tuning

This module implements ReinFlow-style reinforcement learning for SmolVLA by:
1. Injecting learnable noise at each denoising step (converting ODE → SDE)
2. Computing exact log-probabilities through the Markov chain
3. Using REINFORCE policy gradient for fine-tuning

Key difference from "ReinFlow-lite":
- ReinFlow-lite: Added noise AFTER the full denoising (single Gaussian at output)
- Full ReinFlow: Adds noise AT EACH denoising step (stochastic Markov chain)

This enables:
- Multimodal action distributions (preserves flow structure)
- Exact log-probability computation (tractable density)
- Per-step noise scale learning (adaptive exploration)
"""

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE


class ReinFlowSmolVLA(nn.Module):
    """
    ReinFlow wrapper around SmolVLA for RL fine-tuning.
    
    Injects learnable noise at each of the K denoising steps, converting the
    deterministic flow-matching ODE into a stochastic SDE with tractable
    log-probabilities for policy gradient training.
    
    Args:
        base_policy: Pre-trained SmolVLAPolicy instance
        num_steps: Number of denoising steps (default: 10, matches SmolVLA config)
        init_log_sigma: Initial log(σ) for noise scales (default: -2.0 → σ ≈ 0.135)
        train_action_head: If True, unfreeze action_out_proj (23K params)
        train_time_mlp: If True, also unfreeze action_time_mlp_out (519K params)
        device: Torch device
    """
    
    def __init__(
        self,
        base_policy: SmolVLAPolicy,
        num_steps: int = 10,
        init_log_sigma: float = -2.0,
        train_action_head: bool = True,
        train_time_mlp: bool = False,
        device: str = 'cpu'
    ):
        super().__init__()
        self.base = base_policy
        self.num_steps = num_steps
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
            print(f"  [ReinFlow] Unfroze action_out_proj: "
                  f"{sum(p.numel() for p in self.base.model.action_out_proj.parameters()):,} params")
        
        # Optionally unfreeze time MLP output layer
        if train_time_mlp:
            for p in self.base.model.action_time_mlp_out.parameters():
                p.requires_grad = True
            print(f"  [ReinFlow] Unfroze action_time_mlp_out: "
                  f"{sum(p.numel() for p in self.base.model.action_time_mlp_out.parameters()):,} params")
        
        # Learnable noise scales - one per denoising step
        # This is the core ReinFlow innovation: stochastic noise at each step
        # Initialize with small std for stability (exp(-2) ≈ 0.135)
        self.log_sigmas = nn.Parameter(
            torch.full((num_steps,), init_log_sigma, device=device)
        )
        
        # Count total trainable params
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [ReinFlow] Total trainable parameters: {total_trainable:,}")
        print(f"  [ReinFlow] Initial sigmas: {self.log_sigmas.exp().data.cpu().numpy()}")
    
    def get_trainable_params(self):
        """Return list of all trainable parameters for optimizer."""
        params = [self.log_sigmas]
        if self.train_action_head:
            params.extend(self.base.model.action_out_proj.parameters())
        if self.train_time_mlp:
            params.extend(self.base.model.action_time_mlp_out.parameters())
        return params
    
    def forward(self, observation: dict) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass: sample action from stochastic flow with log probability.
        
        Args:
            observation: dict with observation tensors (images, state, language)
            
        Returns:
            action: (action_dim,) sampled action for execution
            log_prob: scalar log probability for policy gradient
            action_chunk: (chunk_size, action_dim) full action chunk
        """
        # Prepare inputs using SmolVLA's preprocessing
        images, img_masks = self.base.prepare_images(observation)
        state = self.base.prepare_state(observation)
        lang_tokens = observation[OBS_LANGUAGE_TOKENS]
        lang_masks = observation[OBS_LANGUAGE_ATTENTION_MASK]
        
        # Call ReinFlow sampling (with noise injection at each step)
        action_chunk, log_prob = self.base.model.sample_actions_reinflow(
            images, img_masks, lang_tokens, lang_masks, state,
            log_sigmas=self.log_sigmas
        )
        
        # Unpad actions to original dimension
        original_action_dim = self.base.config.action_feature.shape[0]
        action_chunk = action_chunk[:, :, :original_action_dim]
        
        # Return first action from chunk for single-step execution
        # action_chunk is (batch=1, chunk_size, action_dim)
        action = action_chunk[0, 0, :]  # First timestep of first batch
        
        return action, log_prob[0], action_chunk[0]
    
    def select_action(self, observation: dict) -> tuple[Tensor, Tensor]:
        """
        Convenience method matching SmolVLA's interface.
        
        Returns:
            action: (action_dim,) action to execute
            log_prob: scalar log probability
        """
        action, log_prob, _ = self.forward(observation)
        return action, log_prob
    
    def get_sigmas(self) -> Tensor:
        """Return current noise scales (for logging)."""
        return self.log_sigmas.exp()


def setup_reinflow_policy(
    pretrained_path: str = "lerobot/smolvla_base",
    device: str = None,
    num_steps: int = 10,
    init_log_sigma: float = -2.0,
    train_action_head: bool = True,
    train_time_mlp: bool = False,
) -> ReinFlowSmolVLA:
    """
    Load SmolVLA and wrap with ReinFlow for RL training.
    
    Args:
        pretrained_path: HuggingFace model path or local checkpoint
        device: Torch device (auto-detected if None)
        num_steps: Number of denoising steps
        init_log_sigma: Initial log noise scale
        train_action_head: Whether to train action output projection
        train_time_mlp: Whether to train time MLP
    
    Returns:
        ReinFlowSmolVLA policy ready for training
    """
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"[ReinFlow] Loading SmolVLA from {pretrained_path}...")
    print(f"[ReinFlow] Using device: {device}")
    
    # Load base policy
    base_policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    base_policy.to(device)
    base_policy.eval()
    
    # Load tokenizer if missing
    if not hasattr(base_policy, 'tokenizer') or base_policy.tokenizer is None:
        print("[ReinFlow] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        base_policy.tokenizer = tokenizer
    
    print("[ReinFlow] SmolVLA loaded successfully!")
    print(f"[ReinFlow] Setting up ReinFlow wrapper with {num_steps} denoising steps...")
    
    # Wrap with ReinFlow
    reinflow_policy = ReinFlowSmolVLA(
        base_policy=base_policy,
        num_steps=num_steps,
        init_log_sigma=init_log_sigma,
        train_action_head=train_action_head,
        train_time_mlp=train_time_mlp,
        device=device,
    )
    
    return reinflow_policy


def prepare_observation_for_reinflow(
    rgb_image_top,
    rgb_image_wrist,
    robot_state,
    instruction: str,
    device,
    policy: ReinFlowSmolVLA,
):
    """
    Prepare observation dict for ReinFlow policy.
    
    This is a convenience function that handles image preprocessing and
    tokenization, matching the format expected by SmolVLA.
    
    Args:
        rgb_image_top: (H, W, C) numpy array from top camera [0, 255]
        rgb_image_wrist: (H, W, C) numpy array from wrist camera [0, 255]
        robot_state: (6,) numpy array of joint positions
        instruction: Task instruction string
        device: Torch device
        policy: ReinFlowSmolVLA policy (for tokenizer access)
    
    Returns:
        observation: dict ready for policy.forward()
    """
    import numpy as np
    
    # Convert top camera image to tensor
    image_top_tensor = torch.from_numpy(rgb_image_top).float() / 255.0
    image_top_tensor = image_top_tensor.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
    image_top_tensor = image_top_tensor.unsqueeze(0).to(device)
    
    # Convert wrist camera image to tensor
    image_wrist_tensor = torch.from_numpy(rgb_image_wrist).float() / 255.0
    image_wrist_tensor = image_wrist_tensor.permute(2, 0, 1)
    image_wrist_tensor = image_wrist_tensor.unsqueeze(0).to(device)
    
    # Convert robot state to tensor
    state_tensor = torch.from_numpy(robot_state).float().unsqueeze(0).to(device)
    
    # Tokenize instruction
    if hasattr(policy.base, 'tokenizer') and policy.base.tokenizer is not None:
        tokens = policy.base.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        language_tokens = tokens['input_ids'].to(device)
        attention_mask = tokens['attention_mask'].bool().to(device)
    else:
        # Fallback dummy tokens
        language_tokens = torch.zeros((1, 1), dtype=torch.long, device=device)
        attention_mask = torch.ones((1, 1), dtype=torch.bool, device=device)
    
    observation = {
        "observation.images.camera1": image_top_tensor,
        "observation.images.camera2": image_wrist_tensor,
        OBS_STATE: state_tensor,
        OBS_LANGUAGE_TOKENS: language_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: attention_mask,
    }
    
    return observation


def compute_returns(rewards: list, gamma: float = 0.99) -> list:
    """
    Compute discounted returns (rewards-to-go).
    
    Args:
        rewards: List of rewards for each timestep
        gamma: Discount factor
        
    Returns:
        List of discounted returns
    """
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def save_reinflow_checkpoint(
    policy: ReinFlowSmolVLA,
    episode: int,
    episode_rewards: list,
    save_path: str = "reinflow_checkpoint.pt"
):
    """Save ReinFlow training checkpoint."""
    checkpoint = {
        'log_sigmas': policy.log_sigmas.data.cpu(),
        'episode': episode,
        'episode_rewards': episode_rewards,
        'num_steps': policy.num_steps,
        'train_action_head': policy.train_action_head,
        'train_time_mlp': policy.train_time_mlp,
    }
    
    # Save action head weights if trained
    if policy.train_action_head:
        checkpoint['action_out_proj'] = policy.base.model.action_out_proj.state_dict()
    if policy.train_time_mlp:
        checkpoint['action_time_mlp_out'] = policy.base.model.action_time_mlp_out.state_dict()
    
    torch.save(checkpoint, save_path)
    print(f"[ReinFlow] Checkpoint saved to {save_path}")


def load_reinflow_checkpoint(
    policy: ReinFlowSmolVLA,
    checkpoint_path: str,
    device: str = 'cpu'
) -> int:
    """
    Load ReinFlow checkpoint.
    
    Returns:
        Starting episode number
    """
    print(f"[ReinFlow] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    policy.log_sigmas.data = checkpoint['log_sigmas'].to(device)
    
    if policy.train_action_head and 'action_out_proj' in checkpoint:
        policy.base.model.action_out_proj.load_state_dict(checkpoint['action_out_proj'])
    
    if policy.train_time_mlp and 'action_time_mlp_out' in checkpoint:
        policy.base.model.action_time_mlp_out.load_state_dict(checkpoint['action_time_mlp_out'])
    
    start_episode = checkpoint.get('episode', 0) + 1
    print(f"[ReinFlow] Resuming from episode {start_episode}")
    print(f"[ReinFlow] Loaded sigmas: {policy.log_sigmas.exp().data.cpu().numpy()}")
    
    return start_episode

