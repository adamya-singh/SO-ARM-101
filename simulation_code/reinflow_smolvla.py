"""
ReinFlow SmolVLA: Full Flow-Based RL Fine-Tuning

This module implements ReinFlow-style reinforcement learning for SmolVLA by:
1. Using a learnable NOISE NETWORK σ_θ'(t, a, o) conditioned on time, action, and observation
2. Storing full denoising trajectories [a^0, a^1, ..., a^K] for proper log-prob computation
3. Computing exact per-step log-probabilities through the Markov chain
4. Using REINFORCE policy gradient for fine-tuning

Reference: https://reinflow.github.io/

Key changes from previous implementation:
- REMOVED: Scalar log_sigmas parameter (was just K numbers)
- ADDED: Noise network that shares features with velocity head
- ADDED: Trajectory storage for replay buffer
- ADDED: Per-step log probability computation
"""

import math
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK, OBS_STATE
from so101_mujoco_utils import normalize_state_for_smolvla
import torch.nn.functional as F


def compute_entropy_regularization(sigmas: List[Tensor], K: int) -> Tensor:
    """
    Compute per-symbol entropy rate for ReinFlow (paper Section 4.4).
    
    R_h = -1/(K+1) * E[h(a^0, ..., a^K | o, θ̄)]
    
    For Gaussian noise at each step:
    h(N(μ, σ²)) = 0.5 * log(2πe * σ²) = 0.5 * (1 + log(2π) + 2*log(σ))
    
    Args:
        sigmas: List of K tensors, each (batch, chunk_size, action_dim)
        K: Number of denoising steps
        
    Returns:
        entropy: Scalar tensor representing average entropy (higher = more exploration)
    """
    if len(sigmas) == 0:
        return torch.tensor(0.0)
    
    device = sigmas[0].device
    total_entropy = torch.tensor(0.0, device=device)
    
    for sigma in sigmas:
        # Entropy of multivariate Gaussian with diagonal covariance
        # h(N(μ, Σ)) = 0.5 * d * (1 + log(2π)) + 0.5 * log(det(Σ))
        # For diagonal Σ: log(det(Σ)) = sum(log(σ_i²)) = 2 * sum(log(σ_i))
        d = sigma.shape[-1] * sigma.shape[-2]  # action_dim * chunk_size
        
        # Per-sample entropy: 0.5 * d * (1 + log(2π)) + sum(log(σ))
        log_sigma_sum = torch.log(sigma + 1e-8).sum(dim=(-1, -2))  # (batch,)
        entropy_per_sample = 0.5 * d * (1 + math.log(2 * math.pi)) + log_sigma_sum
        total_entropy = total_entropy + entropy_per_sample.mean()
    
    # Normalize by (K+1) as per paper (per-symbol entropy rate)
    return total_entropy / (K + 1)


class ReinFlowCritic(nn.Module):
    """
    Value function V(s) for actor-critic ReinFlow training.
    
    Takes observation features from the VLM prefix encoding and outputs
    a scalar value estimate. This enables proper variance reduction
    through learned baselines rather than simple running mean.
    
    Args:
        input_size: Size of the input features (from VLM hidden state)
        hidden_size: Size of hidden layers
    """
    
    def __init__(self, input_size: int = 1024, hidden_size: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 1)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Compute value estimate from observation features.
        
        Args:
            features: (batch_size, feature_dim) tensor from VLM encoding
            
        Returns:
            values: (batch_size,) tensor of value estimates
        """
        return self.net(features).squeeze(-1)


class ReinFlowSmolVLA(nn.Module):
    """
    ReinFlow wrapper around SmolVLA for RL fine-tuning.
    
    Uses a learnable noise NETWORK (not scalar sigmas!) that shares features with
    the velocity head. The noise is conditioned on observation, time, and current
    action through the shared Expert transformer features.
    
    Args:
        base_policy: Pre-trained SmolVLAPolicy instance
        num_steps: Number of denoising steps (default: 10, matches SmolVLA config)
        train_action_head: If True, unfreeze action_out_proj
        train_time_mlp: If True, also unfreeze action_time_mlp_out
        train_full_expert: If True, train entire Action Expert (~100M params)
        train_noise_head: If True, train the noise output projection (always True for ReinFlow)
        device: Torch device
    """
    
    def __init__(
        self,
        base_policy: SmolVLAPolicy,
        num_steps: int = 10,
        train_action_head: bool = True,
        train_time_mlp: bool = False,
        train_full_expert: bool = False,
        train_noise_head: bool = True,  # New: always train noise head for ReinFlow
        train_critic: bool = True,  # New: train critic for actor-critic
        device: str = 'cpu'
    ):
        super().__init__()
        self.base = base_policy
        self.num_steps = num_steps
        self.device = device
        self.train_action_head = train_action_head
        self.train_time_mlp = train_time_mlp
        self.train_full_expert = train_full_expert
        self.train_noise_head = train_noise_head
        self.train_critic = train_critic
        
        # Create critic network for actor-critic training
        # Input size is VLM hidden size (text_config.hidden_size)
        vlm_hidden_size = self.base.model.vlm_with_expert.config.text_config.hidden_size
        self.critic = ReinFlowCritic(input_size=vlm_hidden_size, hidden_size=512)
        self.critic.to(device)
        print(f"  [ReinFlow] Created critic network with input size {vlm_hidden_size}")
        critic_params = sum(p.numel() for p in self.critic.parameters())
        print(f"  [ReinFlow] Critic parameters: {critic_params:,}")
        
        # Freeze entire base policy first
        for p in self.base.parameters():
            p.requires_grad = False
        
        if train_full_expert:
            # Unfreeze ALL Action Expert components (~100M params total)
            print("  [ReinFlow] Training FULL Action Expert")
            
            # 1. Expert transformer layers (the big one! ~95M params)
            if hasattr(self.base.model.vlm_with_expert, 'expert'):
                for p in self.base.model.vlm_with_expert.expert.parameters():
                    p.requires_grad = True
                expert_params = sum(p.numel() for p in self.base.model.vlm_with_expert.expert.parameters())
                print(f"  [ReinFlow] Unfroze expert transformer: {expert_params:,} params")
            
            # 2. Action input projection
            for p in self.base.model.action_in_proj.parameters():
                p.requires_grad = True
            print(f"  [ReinFlow] Unfroze action_in_proj: "
                  f"{sum(p.numel() for p in self.base.model.action_in_proj.parameters()):,} params")
            
            # 3. Action output projection (velocity head)
            for p in self.base.model.action_out_proj.parameters():
                p.requires_grad = True
            print(f"  [ReinFlow] Unfroze action_out_proj: "
                  f"{sum(p.numel() for p in self.base.model.action_out_proj.parameters()):,} params")
            
            # 4. Time MLPs (both in and out)
            for p in self.base.model.action_time_mlp_in.parameters():
                p.requires_grad = True
            for p in self.base.model.action_time_mlp_out.parameters():
                p.requires_grad = True
            time_mlp_params = (sum(p.numel() for p in self.base.model.action_time_mlp_in.parameters()) +
                              sum(p.numel() for p in self.base.model.action_time_mlp_out.parameters()))
            print(f"  [ReinFlow] Unfroze action_time_mlp_in/out: {time_mlp_params:,} params")
            
            # 5. State projection
            for p in self.base.model.state_proj.parameters():
                p.requires_grad = True
            print(f"  [ReinFlow] Unfroze state_proj: "
                  f"{sum(p.numel() for p in self.base.model.state_proj.parameters()):,} params")
        
        else:
            # Original selective unfreezing (lightweight training)
            if train_action_head:
                for p in self.base.model.action_out_proj.parameters():
                    p.requires_grad = True
                print(f"  [ReinFlow] Unfroze action_out_proj: "
                      f"{sum(p.numel() for p in self.base.model.action_out_proj.parameters()):,} params")
            
            if train_time_mlp:
                for p in self.base.model.action_time_mlp_out.parameters():
                    p.requires_grad = True
                print(f"  [ReinFlow] Unfroze action_time_mlp_out: "
                      f"{sum(p.numel() for p in self.base.model.action_time_mlp_out.parameters()):,} params")
        
        # ReinFlow: Always train the noise head (this is the σ_θ' network)
        if train_noise_head:
            for p in self.base.model.noise_mlp.parameters():
                p.requires_grad = True
            print(f"  [ReinFlow] Unfroze noise_mlp (σ_θ'): "
                  f"{sum(p.numel() for p in self.base.model.noise_mlp.parameters()):,} params")
        
        # Count total trainable params
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [ReinFlow] Total trainable parameters: {total_trainable:,}")
    
    def get_trainable_params(self) -> List[torch.nn.Parameter]:
        """
        Return list of all trainable POLICY parameters for optimizer.
        
        Note: Critic parameters are separate - use get_critic_params() for those.
        This allows using different learning rates for actor vs critic.
        """
        params = []
        
        if self.train_full_expert:
            # All Action Expert components
            if hasattr(self.base.model.vlm_with_expert, 'expert'):
                params.extend(self.base.model.vlm_with_expert.expert.parameters())
            params.extend(self.base.model.action_in_proj.parameters())
            params.extend(self.base.model.action_out_proj.parameters())
            params.extend(self.base.model.action_time_mlp_in.parameters())
            params.extend(self.base.model.action_time_mlp_out.parameters())
            params.extend(self.base.model.state_proj.parameters())
        else:
            # Selective unfreezing
            if self.train_action_head:
                params.extend(self.base.model.action_out_proj.parameters())
            if self.train_time_mlp:
                params.extend(self.base.model.action_time_mlp_out.parameters())
        
        # Always include noise head for ReinFlow
        if self.train_noise_head:
            params.extend(self.base.model.noise_mlp.parameters())
        
        return params
    
    def get_all_trainable_params(self) -> List[torch.nn.Parameter]:
        """Return all trainable parameters (policy + critic) for single optimizer."""
        params = self.get_trainable_params()
        if self.train_critic:
            params.extend(self.critic.parameters())
        return params
    
    def forward_with_trajectory(self, observation: dict) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Forward pass returning action chunk, full trajectory, and sigmas.
        
        This is the core ReinFlow forward pass that:
        1. Runs denoising with noise injection at each step
        2. Stores the full trajectory [a^0, ..., a^K] for replay buffer
        3. Returns sigmas used at each step for log-prob computation
        
        Args:
            observation: dict with observation tensors (images, state, language)
            
        Returns:
            action_chunk: (batch, chunk_size, action_dim) final actions
            trajectory: List of K+1 tensors [a^0, a^1, ..., a^K]
            sigmas: List of K tensors, sigma used at each step
        """
        # Prepare inputs using SmolVLA's preprocessing
        images, img_masks = self.base.prepare_images(observation)
        state = self.base.prepare_state(observation)
        lang_tokens = observation[OBS_LANGUAGE_TOKENS]
        lang_masks = observation[OBS_LANGUAGE_ATTENTION_MASK]
        
        # Call ReinFlow sampling (with noise network at each step)
        action_chunk, trajectory, sigmas = self.base.model.sample_actions_reinflow(
            images, img_masks, lang_tokens, lang_masks, state
        )
        
        # Unpad actions to original dimension (for execution)
        original_action_dim = self.base.config.action_feature.shape[0]
        action_chunk = action_chunk[:, :, :original_action_dim]
        
        # Keep trajectory in padded space (needed for denoise_step in log prob computation)
        # Only unpad sigmas since they're not used in on-policy mode anyway
        sigmas = [s[:, :, :original_action_dim] for s in sigmas]
        
        return action_chunk, trajectory, sigmas
    
    def forward_batched_with_trajectory(self, observation: dict) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Batched forward pass returning action chunks, trajectories, and sigmas.
        
        Same as forward_with_trajectory but handles batch dimension explicitly.
        
        Args:
            observation: dict with batched observation tensors
            
        Returns:
            action_chunks: (N, chunk_size, action_dim) final actions for N envs
            trajectory: List of K+1 tensors, each (N, chunk_size, action_dim)
            sigmas: List of K tensors, each (N, chunk_size, action_dim)
        """
        return self.forward_with_trajectory(observation)
    
    def forward(self, observation: dict) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Forward pass for single environment (batch size 1).
        
        Returns:
            action: (action_dim,) first action for execution
            trajectory: Full denoising trajectory
            sigmas: Sigmas used at each step
        """
        action_chunk, trajectory, sigmas = self.forward_with_trajectory(observation)
        # Return first action from first batch
        return action_chunk[0, 0, :], trajectory, sigmas
    
    def select_action(self, observation: dict) -> Tensor:
        """
        Convenience method to just get the action (no trajectory).
        
        Returns:
            action: (action_dim,) action to execute
        """
        action, _, _ = self.forward(observation)
        return action
    
    def get_sigma_stats(self) -> Dict[str, float]:
        """Return statistics about current sigma bounds (for logging)."""
        return {
            'sigma_min': self.base.model.sigma_min,
            'sigma_max': self.base.model.sigma_max,
        }
    
    def extract_observation_features(self, observation: dict) -> Tensor:
        """
        Extract observation features from VLM prefix encoding for critic.
        
        Uses the embedded prefix features directly (images, language, state)
        without running through the full VLM forward pass. This matches
        the ReinFlow paper's approach where "the critic only receives
        features from time and condition."
        
        Args:
            observation: dict with observation tensors (images, state, language)
            
        Returns:
            features: (batch_size, hidden_size) tensor of observation features
        """
        # Prepare inputs using SmolVLA's preprocessing
        images, img_masks = self.base.prepare_images(observation)
        state = self.base.prepare_state(observation)
        lang_tokens = observation[OBS_LANGUAGE_TOKENS]
        lang_masks = observation[OBS_LANGUAGE_ATTENTION_MASK]
        
        # Embed prefix (images, language, state)
        prefix_embs, prefix_pad_masks, _ = self.base.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        
        # Pool over sequence dimension - use mean of valid (non-padded) tokens
        # prefix_pad_masks: (batch, seq_len) - True where valid
        mask_expanded = prefix_pad_masks.unsqueeze(-1).float()  # (batch, seq, 1)
        pooled = (prefix_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        return pooled  # (batch_size, hidden_size)
    
    def get_value(self, observation: dict) -> Tensor:
        """
        Get value estimate V(s) for the given observation.
        
        Args:
            observation: dict with observation tensors
            
        Returns:
            value: (batch_size,) tensor of value estimates
        """
        features = self.extract_observation_features(observation)
        return self.critic(features)
    
    def get_critic_params(self) -> List[torch.nn.Parameter]:
        """Return list of critic parameters for separate optimizer."""
        return list(self.critic.parameters())


def compute_trajectory_log_probs(
    policy: ReinFlowSmolVLA,
    trajectories: List[List[Tensor]],
    sigmas: List[List[Tensor]],
    observations: Dict[str, Tensor],
) -> Tensor:
    """
    Compute sum of log probabilities for stored denoising trajectories.
    
    This implements the ReinFlow paper's equation:
    log π = Σ_{k=0}^{K-1} log N(a^{k+1} | μ_k, σ_k²)
    
    where μ_k = a^k + v_θ(t_k, a^k, o) * Δt
        
        Args:
        policy: ReinFlowSmolVLA policy
        trajectories: List of B trajectories, each is list of K+1 tensors
        sigmas: List of B sigma lists, each is list of K tensors
        observations: Dict of batched observation tensors
            
        Returns:
        log_probs: (B,) tensor of summed log probabilities
    """
    batch_size = len(trajectories)
    num_steps = len(trajectories[0]) - 1  # K steps (trajectory has K+1 elements)
    device = trajectories[0][0].device
    
    # Prepare observation features (embed prefix and cache KV)
    images, img_masks = policy.base.prepare_images(observations)
    state = policy.base.prepare_state(observations)
    lang_tokens = observations[OBS_LANGUAGE_TOKENS]
    lang_masks = observations[OBS_LANGUAGE_ATTENTION_MASK]
    
    # Embed prefix
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.base.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    
    # Cache KV
    _, past_key_values = policy.base.model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=policy.base.model.config.use_cache,
        fill_kv_cache=True,
    )
    
    dt = -1.0 / num_steps
    total_log_probs = torch.zeros(batch_size, device=device)
    
    for k in range(num_steps):
        t_k = 1.0 + k * dt
        t_k_tensor = torch.tensor(t_k, device=device).expand(batch_size)
        
        # Stack trajectories and sigmas for this step
        a_k = torch.stack([traj[k] for traj in trajectories])  # (B, chunk, action_dim)
        a_k_next = torch.stack([traj[k + 1] for traj in trajectories])  # (B, chunk, action_dim)
        sigma_k = torch.stack([sig[k] for sig in sigmas])  # (B, chunk, action_dim)
        
        # Get velocity from current policy (re-run forward for gradients)
        v_k = policy.base.model.denoise_step(
            prefix_pad_masks, past_key_values, a_k, t_k_tensor, return_sigma=False
        )
        
        # Mean of transition: μ_k = a^k + v_θ(t_k, a^k, o) * Δt
        mu_k = a_k + dt * v_k
        
        # Log probability: log N(a^{k+1} | μ_k, σ_k²)
        # = -0.5 * [||a^{k+1} - μ_k||² / σ_k² + d * log(2π) + 2 * d * log(σ_k)]
        diff = a_k_next - mu_k
        d = diff.shape[-1] * diff.shape[-2]  # action_dim * chunk_size
        
        # Compute log prob carefully to handle per-element sigmas
        log_prob_k = -0.5 * (
            (diff ** 2 / (sigma_k ** 2 + 1e-8)).sum(dim=(-1, -2)) +
            d * math.log(2 * math.pi) +
            2 * torch.log(sigma_k + 1e-8).sum(dim=(-1, -2))
        )
        
        total_log_probs = total_log_probs + log_prob_k
    
    return total_log_probs


def compute_trajectory_log_probs_onpolicy(
    policy: ReinFlowSmolVLA,
    trajectories: List[Tensor],
    observations: Dict[str, Tensor],
    return_sigmas: bool = False,
) -> Tensor:
    """
    On-policy log probability computation - recomputes BOTH velocity AND sigma.
    
    This is the correct implementation for on-policy training where we don't
    store sigmas. Instead, we recompute everything from the current policy,
    enabling proper gradient flow through both the velocity and noise networks.
    
    Key difference from compute_trajectory_log_probs:
    - Does NOT use stored sigmas
    - Recomputes sigma_k from current policy at each step (return_sigma=True)
    - Gradients flow through both velocity head AND noise head
    
    Args:
        policy: ReinFlowSmolVLA policy
        trajectories: List of K+1 tensors [a^0, a^1, ..., a^K] for single trajectory
                     OR batched tensor of shape (batch, K+1, chunk, action_dim)
        observations: Dict of observation tensors (can be batched)
        return_sigmas: If True, also return list of sigmas for entropy computation
            
    Returns:
        log_probs: (batch_size,) tensor of summed log probabilities
        sigmas: (optional) List of K sigma tensors if return_sigmas=True
    """
    # Handle both list and tensor formats
    if isinstance(trajectories, list):
        # List of K+1 tensors - stack them
        trajectory_tensor = torch.stack(trajectories, dim=1)  # (batch, K+1, chunk, action_dim)
    else:
        trajectory_tensor = trajectories
    
    batch_size = trajectory_tensor.shape[0]
    num_steps = trajectory_tensor.shape[1] - 1  # K steps
    device = trajectory_tensor.device
    
    # Get original action dim for slicing (trajectory is in padded space)
    original_action_dim = policy.base.config.action_feature.shape[0]
    
    # Prepare observation features (embed prefix and cache KV)
    images, img_masks = policy.base.prepare_images(observations)
    state = policy.base.prepare_state(observations)
    lang_tokens = observations[OBS_LANGUAGE_TOKENS]
    lang_masks = observations[OBS_LANGUAGE_ATTENTION_MASK]
    
    # Embed prefix
    prefix_embs, prefix_pad_masks, prefix_att_masks = policy.base.model.embed_prefix(
        images, img_masks, lang_tokens, lang_masks, state=state
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    
    # Cache KV
    _, past_key_values = policy.base.model.vlm_with_expert.forward(
        attention_mask=prefix_att_2d_masks,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=policy.base.model.config.use_cache,
        fill_kv_cache=True,
    )
    
    dt = -1.0 / num_steps
    total_log_probs = torch.zeros(batch_size, device=device)
    collected_sigmas = []  # For entropy regularization
    
    for k in range(num_steps):
        t_k = 1.0 + k * dt
        t_k_tensor = torch.tensor(t_k, device=device).expand(batch_size)
        
        # Get trajectory states at step k and k+1
        a_k = trajectory_tensor[:, k]  # (batch, chunk, action_dim)
        a_k_next = trajectory_tensor[:, k + 1]  # (batch, chunk, action_dim)
        
        # KEY DIFFERENCE: Get BOTH velocity AND sigma from current policy
        # This enables gradient flow through the noise network!
        v_k, sigma_k = policy.base.model.denoise_step(
            prefix_pad_masks, past_key_values, a_k, t_k_tensor, return_sigma=True
        )
        
        # Collect sigma for entropy computation (sliced to original dims)
        if return_sigmas:
            collected_sigmas.append(sigma_k[:, :, :original_action_dim])
        
        # Slice to original action dims for log prob computation
        # (trajectory is in padded space for denoise_step, but we only compute
        # log prob on real action dimensions)
        a_k_slice = a_k[:, :, :original_action_dim]
        a_k_next_slice = a_k_next[:, :, :original_action_dim]
        v_k_slice = v_k[:, :, :original_action_dim]
        sigma_k_slice = sigma_k[:, :, :original_action_dim]
        
        # Mean of transition: μ_k = a^k + v_θ(t_k, a^k, o) * Δt
        mu_k = a_k_slice + dt * v_k_slice
        
        # Log probability: log N(a^{k+1} | μ_k, σ_k²)
        # = -0.5 * [||a^{k+1} - μ_k||² / σ_k² + d * log(2π) + 2 * log(σ_k)]
        diff = a_k_next_slice - mu_k
        chunk_size = a_k.shape[-2]
        d = original_action_dim * chunk_size  # real action_dim * chunk_size
        
        # Compute log prob with gradients through sigma
        log_prob_k = -0.5 * (
            (diff ** 2 / (sigma_k_slice ** 2 + 1e-8)).sum(dim=(-1, -2)) +
            d * math.log(2 * math.pi) +
            2 * torch.log(sigma_k_slice + 1e-8).sum(dim=(-1, -2))
        )
        
        total_log_probs = total_log_probs + log_prob_k
    
    if return_sigmas:
        return total_log_probs, collected_sigmas
    return total_log_probs


def compute_actor_critic_loss(
    policy: ReinFlowSmolVLA,
    trajectories: List[Tensor],
    observations: Dict[str, Tensor],
    rewards: Tensor,
    gamma: float = 0.95,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """
    Compute actor-critic loss for on-policy ReinFlow training.
    
    DEPRECATED: Use compute_ppo_loss for paper-faithful training.
    
    This implements the full actor-critic update:
    - Critic loss: MSE between value estimates and returns
    - Policy loss: -advantage * log_prob (REINFORCE with baseline)
    
    Args:
        policy: ReinFlowSmolVLA policy with critic
        trajectories: List of K+1 action tensors [a^0, ..., a^K]
        observations: Dict of observation tensors
        rewards: (batch_size,) tensor of rewards (one per trajectory)
        gamma: Discount factor for return computation
        
    Returns:
        policy_loss: Scalar policy gradient loss
        critic_loss: Scalar critic MSE loss
        info: Dict with additional metrics for logging
    """
    batch_size = rewards.shape[0]
    device = rewards.device
    
    # For single-step rewards (chunk-level), returns = rewards
    # If we had multi-step rewards, we'd compute discounted returns here
    returns = rewards  # (batch_size,)
    
    # Get value estimates from critic
    with torch.no_grad():
        values_for_advantage = policy.get_value(observations)
    
    # Compute advantages: A = R - V(s)
    advantages = returns - values_for_advantage
    
    # Normalize advantages for stability
    if batch_size > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # Clamp advantages to prevent extreme updates
    advantages = torch.clamp(advantages, -10.0, 10.0)
    
    # Compute log probabilities with gradients through both velocity and sigma
    log_probs = compute_trajectory_log_probs_onpolicy(policy, trajectories, observations)
    
    # Policy loss: -E[A * log π]
    policy_loss = -(advantages.detach() * log_probs).mean()
    
    # Critic loss: MSE(V(s), R)
    # Need fresh forward pass for critic gradients
    values = policy.get_value(observations)
    critic_loss = F.mse_loss(values, returns)
    
    # Info for logging
    info = {
        'advantage_mean': advantages.mean().item(),
        'advantage_std': advantages.std().item() if batch_size > 1 else 0.0,
        'log_prob_mean': log_probs.mean().item(),
        'value_mean': values.mean().item(),
        'return_mean': returns.mean().item(),
    }
    
    return policy_loss, critic_loss, info


def compute_ppo_loss(
    policy: ReinFlowSmolVLA,
    trajectories: Tensor,
    observations: Dict[str, Tensor],
    old_log_probs: Tensor,
    advantages: Tensor,
    returns: Tensor,
    clip_epsilon: float = 0.2,
    value_clip_epsilon: float = 0.2,
    old_values: Optional[Tensor] = None,
    entropy_coeff: float = 0.0,
) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """
    Compute PPO clipped surrogate objective for ReinFlow training.
    
    This implements the paper-faithful PPO update:
    - Policy loss: clipped surrogate objective
    - Critic loss: MSE between value estimates and returns (optionally clipped)
    - KL divergence monitoring for early stopping
    - Entropy regularization for exploration (paper Section 4.4)
    
    Args:
        policy: ReinFlowSmolVLA policy with critic
        trajectories: (batch, K+1, chunk, action_dim) tensor of denoising trajectories
        observations: Dict of observation tensors (batched)
        old_log_probs: (batch_size,) tensor of log probs from behavior policy (detached)
        advantages: (batch_size,) tensor of GAE advantages (pre-normalized)
        returns: (batch_size,) tensor of target returns for value function
        clip_epsilon: PPO clip range for policy ratio
        value_clip_epsilon: Clip range for value function (0 to disable)
        old_values: (batch_size,) tensor of old value estimates (for value clipping)
        entropy_coeff: Coefficient for entropy regularization (paper uses 0.03)
        
    Returns:
        policy_loss: Scalar PPO clipped policy loss
        critic_loss: Scalar critic MSE loss
        info: Dict with metrics including KL divergence
    """
    batch_size = advantages.shape[0]
    device = advantages.device
    num_steps = trajectories.shape[1] - 1  # K steps
    
    # Compute new log probs with gradients through both velocity and noise networks
    # Also get sigmas for entropy regularization
    if entropy_coeff > 0:
        new_log_probs, sigmas = compute_trajectory_log_probs_onpolicy(
            policy, trajectories, observations, return_sigmas=True
        )
    else:
        new_log_probs = compute_trajectory_log_probs_onpolicy(
            policy, trajectories, observations, return_sigmas=False
        )
        sigmas = []
    
    # Probability ratio: r(θ) = π_θ(a|s) / π_θ_old(a|s)
    log_ratio_raw = new_log_probs - old_log_probs
    # Clamp log_ratio to prevent numerical overflow (standard PPO stabilization)
    log_ratio = torch.clamp(log_ratio_raw, -20.0, 20.0)
    ratio = torch.exp(log_ratio)
    
    # Clipped surrogate objective
    # L^CLIP = min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    policy_loss_unclipped = ratio * advantages
    policy_loss_clipped = clipped_ratio * advantages
    policy_loss = -torch.min(policy_loss_unclipped, policy_loss_clipped).mean()
    
    # Entropy regularization (paper Section 4.4)
    # Higher entropy = more exploration
    entropy_bonus = torch.tensor(0.0, device=device)
    if entropy_coeff > 0 and len(sigmas) > 0:
        entropy_bonus = compute_entropy_regularization(sigmas, num_steps)
        # Subtract entropy_coeff * entropy_bonus to encourage higher entropy
        # (since we're minimizing loss, and higher entropy is better)
        policy_loss = policy_loss - entropy_coeff * entropy_bonus
    
    # Value loss with optional clipping
    values = policy.get_value(observations)
    if old_values is not None and value_clip_epsilon > 0:
        # Clipped value loss to prevent large value updates
        values_clipped = old_values + torch.clamp(
            values - old_values, -value_clip_epsilon, value_clip_epsilon
        )
        critic_loss_unclipped = (values - returns) ** 2
        critic_loss_clipped = (values_clipped - returns) ** 2
        critic_loss = 0.5 * torch.max(critic_loss_unclipped, critic_loss_clipped).mean()
    else:
        critic_loss = 0.5 * F.mse_loss(values, returns)
    
    # KL divergence approximation for monitoring and early stopping
    # Using the approximation: KL ≈ (r - 1) - log(r) = exp(log_ratio) - 1 - log_ratio
    # Or simpler: KL ≈ 0.5 * (log_ratio)^2 for small changes
    with torch.no_grad():
        approx_kl = ((ratio - 1) - log_ratio).mean().item()
        # Also compute simple KL approximation
        kl_simple = (old_log_probs - new_log_probs).mean().item()
        
        # Clip fraction: how often the ratio was clipped
        clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()
    
    # Info for logging
    info = {
        'kl_div': approx_kl,
        'kl_simple': kl_simple,
        'clip_fraction': clip_fraction,
        'ratio_mean': ratio.mean().item(),
        'ratio_std': ratio.std().item() if batch_size > 1 else 0.0,
        'advantage_mean': advantages.mean().item(),
        'advantage_std': advantages.std().item() if batch_size > 1 else 0.0,
        'log_prob_mean': new_log_probs.mean().item(),
        'old_log_prob_mean': old_log_probs.mean().item(),
        'value_mean': values.mean().item(),
        'return_mean': returns.mean().item(),
        'entropy': entropy_bonus.item() if isinstance(entropy_bonus, Tensor) else entropy_bonus,
    }
    
    return policy_loss, critic_loss, info


def setup_reinflow_policy(
    pretrained_path: str = "lerobot/smolvla_base",
    device: str = None,
    num_steps: int = 10,
    train_action_head: bool = True,
    train_time_mlp: bool = False,
    train_full_expert: bool = False,
    train_noise_head: bool = True,
    train_critic: bool = True,
) -> ReinFlowSmolVLA:
    """
    Load SmolVLA and wrap with ReinFlow for RL training.
    
    Args:
        pretrained_path: HuggingFace model path or local checkpoint
        device: Torch device (auto-detected if None)
        num_steps: Number of denoising steps
        train_action_head: Whether to train action output projection
        train_time_mlp: Whether to train time MLP
        train_full_expert: Whether to train entire Action Expert (~100M params)
        train_noise_head: Whether to train noise output projection (default True)
        train_critic: Whether to train critic network (default True for actor-critic)
    
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
    print(f"[ReinFlow] Using NOISE NETWORK (not scalar sigmas)")
    print(f"[ReinFlow] Actor-Critic mode: {train_critic}")
    
    # Wrap with ReinFlow
    reinflow_policy = ReinFlowSmolVLA(
        base_policy=base_policy,
        num_steps=num_steps,
        train_action_head=train_action_head,
        train_time_mlp=train_time_mlp,
        train_full_expert=train_full_expert,
        train_noise_head=train_noise_head,
        train_critic=train_critic,
        device=device,
    )
    
    return reinflow_policy


def prepare_observation_for_reinflow(
    rgb_image_top,
    rgb_image_wrist,
    rgb_image_side,
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
        rgb_image_side: (H, W, C) numpy array from side camera [0, 255]
        robot_state: (6,) numpy array of joint positions
        instruction: Task instruction string
        device: Torch device
        policy: ReinFlowSmolVLA policy (for tokenizer access)
    
    Returns:
        observation: dict ready for policy.forward()
    """
    # Convert top camera image to tensor
    image_top_tensor = torch.from_numpy(rgb_image_top).float() / 255.0
    image_top_tensor = image_top_tensor.permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
    image_top_tensor = image_top_tensor.unsqueeze(0).to(device)
    
    # Convert wrist camera image to tensor
    image_wrist_tensor = torch.from_numpy(rgb_image_wrist).float() / 255.0
    image_wrist_tensor = image_wrist_tensor.permute(2, 0, 1)
    image_wrist_tensor = image_wrist_tensor.unsqueeze(0).to(device)
    
    # Convert side camera image to tensor
    image_side_tensor = torch.from_numpy(rgb_image_side).float() / 255.0
    image_side_tensor = image_side_tensor.permute(2, 0, 1)
    image_side_tensor = image_side_tensor.unsqueeze(0).to(device)
    
    # Normalize robot state for SmolVLA (radians -> degrees -> normalized)
    normalized_state = normalize_state_for_smolvla(robot_state)
    state_tensor = torch.from_numpy(normalized_state).float().unsqueeze(0).to(device)
    
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
        "observation.images.camera3": image_side_tensor,
        OBS_STATE: state_tensor,
        OBS_LANGUAGE_TOKENS: language_tokens,
        OBS_LANGUAGE_ATTENTION_MASK: attention_mask,
    }
    
    return observation


def prepare_batched_observation(
    obs_dict: dict,
    instruction: str,
    device,
    policy: ReinFlowSmolVLA,
    num_envs: int,
):
    """
    Add language tokens to batched observation from VectorizedMuJoCoEnv.
    
    The VectorizedMuJoCoEnv already provides batched images and states,
    this function adds the tokenized instruction repeated for each env.
    
    Args:
        obs_dict: Dict from vec_env.get_batched_observations() with:
            - observation.images.camera1: (N, C, H, W)
            - observation.images.camera2: (N, C, H, W)
            - observation.images.camera3: (N, C, H, W)
            - observation.state: (N, 6) already normalized
        instruction: Task instruction string
        device: Torch device
        policy: ReinFlowSmolVLA policy (for tokenizer access)
        num_envs: Number of parallel environments
    
    Returns:
        observation: dict ready for policy.forward_batched()
    """
    # Tokenize instruction once
    if hasattr(policy.base, 'tokenizer') and policy.base.tokenizer is not None:
        tokens = policy.base.tokenizer(
            instruction,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        # Repeat for all environments in batch
        language_tokens = tokens['input_ids'].repeat(num_envs, 1).to(device)
        attention_mask = tokens['attention_mask'].bool().repeat(num_envs, 1).to(device)
    else:
        # Fallback dummy tokens
        language_tokens = torch.zeros((num_envs, 1), dtype=torch.long, device=device)
        attention_mask = torch.ones((num_envs, 1), dtype=torch.bool, device=device)
    
    observation = {
        "observation.images.camera1": obs_dict["observation.images.camera1"],
        "observation.images.camera2": obs_dict["observation.images.camera2"],
        "observation.images.camera3": obs_dict["observation.images.camera3"],
        OBS_STATE: obs_dict["observation.state"],
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


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    dones: Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> Tuple[Tensor, Tensor]:
    """
    Compute Generalized Advantage Estimation (GAE).
    
    GAE provides better credit assignment by balancing bias and variance
    in advantage estimates using a lambda parameter.
    
    Args:
        rewards: (batch_size,) tensor of rewards
        values: (batch_size,) tensor of value estimates V(s)
        next_values: (batch_size,) tensor of next state values V(s')
        dones: (batch_size,) tensor of done flags (1 if terminal, 0 otherwise)
        gamma: Discount factor
        gae_lambda: GAE lambda parameter (0=TD(0), 1=MC)
        
    Returns:
        advantages: (batch_size,) tensor of GAE advantages
        returns: (batch_size,) tensor of target returns for value function
    """
    batch_size = rewards.shape[0]
    device = rewards.device
    
    advantages = torch.zeros(batch_size, device=device, dtype=rewards.dtype)
    gae = 0.0
    
    # Compute GAE in reverse order
    for t in reversed(range(batch_size)):
        # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done_t) - V(s_t)
        delta = rewards[t] + gamma * next_values[t] * (1.0 - dones[t]) - values[t]
        # GAE: A_t = δ_t + γ * λ * (1 - done_t) * A_{t+1}
        gae = delta + gamma * gae_lambda * (1.0 - dones[t]) * gae
        advantages[t] = gae
    
    # Returns are advantages + values (for value function training)
    returns = advantages + values
    
    return advantages, returns


def save_reinflow_checkpoint(
    policy: ReinFlowSmolVLA,
    episode: int,
    episode_rewards: list,
    save_path: str = "reinflow_checkpoint.pt",
    wandb_run_id: str = None
):
    """Save ReinFlow training checkpoint including critic."""
    checkpoint = {
        'episode': episode,
        'episode_rewards': episode_rewards,
        'num_steps': policy.num_steps,
        'train_action_head': policy.train_action_head,
        'train_time_mlp': policy.train_time_mlp,
        'train_full_expert': policy.train_full_expert,
        'train_noise_head': policy.train_noise_head,
        'train_critic': policy.train_critic,
    }
    
    # Save wandb run ID for resuming
    if wandb_run_id is not None:
        checkpoint['wandb_run_id'] = wandb_run_id
    
    # Always save noise head (core of ReinFlow)
    checkpoint['noise_mlp'] = policy.base.model.noise_mlp.state_dict()
    
    # Save critic network (new for actor-critic)
    checkpoint['critic'] = policy.critic.state_dict()
    
    if policy.train_full_expert:
        # Save ALL Action Expert component weights
        if hasattr(policy.base.model.vlm_with_expert, 'expert'):
            checkpoint['expert'] = policy.base.model.vlm_with_expert.expert.state_dict()
        checkpoint['action_in_proj'] = policy.base.model.action_in_proj.state_dict()
        checkpoint['action_out_proj'] = policy.base.model.action_out_proj.state_dict()
        checkpoint['action_time_mlp_in'] = policy.base.model.action_time_mlp_in.state_dict()
        checkpoint['action_time_mlp_out'] = policy.base.model.action_time_mlp_out.state_dict()
        checkpoint['state_proj'] = policy.base.model.state_proj.state_dict()
    else:
        # Save selective weights (original behavior)
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
) -> Tuple[int, Optional[str]]:
    """
    Load ReinFlow checkpoint including critic.
    
    Returns:
        Tuple of (starting_episode, wandb_run_id or None)
    """
    print(f"[ReinFlow] Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load noise head (core of ReinFlow)
    if 'noise_mlp' in checkpoint:
        policy.base.model.noise_mlp.load_state_dict(checkpoint['noise_mlp'])
        print("  [ReinFlow] Loaded noise_mlp weights")
    elif 'noise_out_proj' in checkpoint:
        # Legacy support for old checkpoints with linear noise layer
        print("  [ReinFlow] Warning: Old checkpoint with linear noise layer, skipping (architecture changed to MLP)")
    
    # Load critic network (new for actor-critic)
    if 'critic' in checkpoint:
        policy.critic.load_state_dict(checkpoint['critic'])
        print("  [ReinFlow] Loaded critic weights")
    
    if policy.train_full_expert:
        # Load ALL Action Expert component weights
        if hasattr(policy.base.model.vlm_with_expert, 'expert') and 'expert' in checkpoint:
            policy.base.model.vlm_with_expert.expert.load_state_dict(checkpoint['expert'])
            print("  [ReinFlow] Loaded expert transformer weights")
        if 'action_in_proj' in checkpoint:
            policy.base.model.action_in_proj.load_state_dict(checkpoint['action_in_proj'])
        if 'action_out_proj' in checkpoint:
            policy.base.model.action_out_proj.load_state_dict(checkpoint['action_out_proj'])
        if 'action_time_mlp_in' in checkpoint:
            policy.base.model.action_time_mlp_in.load_state_dict(checkpoint['action_time_mlp_in'])
        if 'action_time_mlp_out' in checkpoint:
            policy.base.model.action_time_mlp_out.load_state_dict(checkpoint['action_time_mlp_out'])
        if 'state_proj' in checkpoint:
            policy.base.model.state_proj.load_state_dict(checkpoint['state_proj'])
        print("  [ReinFlow] Loaded full Action Expert weights")
    else:
        # Load selective weights (original behavior)
        if policy.train_action_head and 'action_out_proj' in checkpoint:
            policy.base.model.action_out_proj.load_state_dict(checkpoint['action_out_proj'])
        
        if policy.train_time_mlp and 'action_time_mlp_out' in checkpoint:
            policy.base.model.action_time_mlp_out.load_state_dict(checkpoint['action_time_mlp_out'])
    
    start_episode = checkpoint.get('episode', 0) + 1
    wandb_run_id = checkpoint.get('wandb_run_id', None)
    
    print(f"[ReinFlow] Resuming from episode {start_episode}")
    if wandb_run_id:
        print(f"[ReinFlow] Will resume wandb run: {wandb_run_id}")
    
    return start_episode, wandb_run_id
