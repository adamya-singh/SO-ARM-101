"""
Pi0 Adapter for ReinFlow Training

Adapter adapting PI0Policy to the VLAPolicyInterface.
Pi0 requires more adaptation than SmolVLA because:
1. It doesn't have a built-in noise_mlp (we add one)
2. State is embedded in suffix, so denoise_step needs state parameter
3. Need to implement sample_actions_reinflow from scratch

Key characteristics of Pi0:
- State is embedded in suffix (via embed_suffix(state, x_t, t))
- denoise_step() DOES need state parameter
- No built-in noise_mlp - we add one matching SmolVLA's architecture
- VLM hidden size: 2048 (PaliGemma Gemma 2B)
- Expert hidden size: 2048 (Gemma 2B action expert)
- Supports gradient checkpointing for memory efficiency
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from vla_policy_interface import VLAPolicyInterface, ReinFlowCapableMixin
from lerobot.policies.pi0.modeling_pi0 import PI0Policy, make_att_2d_masks
from lerobot.utils.constants import OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK


class Pi0Adapter(nn.Module, VLAPolicyInterface, ReinFlowCapableMixin):
    """
    Adapter wrapping PI0Policy for the VLAPolicyInterface.
    
    Pi0 requires more adaptation than SmolVLA:
    - We add a noise_mlp network for ReinFlow (Pi0 doesn't have one)
    - State must be cached and passed to denoise_step
    - We implement sample_actions_reinflow using the mixin
    
    Args:
        base_policy: Pre-loaded PI0Policy instance
        sigma_min: Minimum sigma bound for noise network
        sigma_max: Maximum sigma bound for noise network
    """
    
    def __init__(
        self,
        base_policy: PI0Policy,
        sigma_min: float = 0.25,  # Scaled for high-dim actions: √(D/28) ≈ 3.3x
        sigma_max: float = 0.50,  # See notes/sigma-scaling-bug-fix.md
    ):
        super().__init__()
        self.base = base_policy
        self._device = next(base_policy.parameters()).device
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        
        # Get expert hidden size from Pi0's action_out_proj
        expert_hidden = base_policy.model.action_out_proj.in_features
        max_action_dim = base_policy.config.max_action_dim
        
        # Add noise MLP matching SmolVLA's architecture
        # This is the σ_θ'(t, a, o) network for ReinFlow
        self.noise_mlp = nn.Sequential(
            nn.Linear(expert_hidden, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, max_action_dim)
        )
        
        # Move noise_mlp to same device as base policy
        self.noise_mlp.to(self._device)
        
        print(f"[Pi0Adapter] Added noise_mlp with input size {expert_hidden}")
        print(f"[Pi0Adapter] Sigma bounds: [{sigma_min}, {sigma_max}]")
    
    @property
    def vlm_hidden_size(self) -> int:
        """Pi0 uses PaliGemma with 2048 hidden size (Gemma 2B)."""
        return self.base.model.paligemma_with_expert.paligemma.config.text_config.hidden_size
    
    @property
    def expert_hidden_size(self) -> int:
        """Return Pi0's action expert hidden dimension."""
        return self.base.model.action_out_proj.in_features
    
    @property
    def chunk_size(self) -> int:
        return self.base.config.chunk_size
    
    @property
    def max_action_dim(self) -> int:
        return self.base.config.max_action_dim
    
    @property
    def original_action_dim(self) -> int:
        return self.base.config.output_features['action'].shape[0]
    
    @property
    def num_denoising_steps(self) -> int:
        return self.base.config.num_inference_steps
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def noise_network(self) -> nn.Module:
        """Return the added noise_mlp."""
        return self.noise_mlp
    
    @property
    def sigma_min(self) -> float:
        return self._sigma_min
    
    @property
    def sigma_max(self) -> float:
        return self._sigma_max
    
    def set_sigma_bounds(self, sigma_min: float, sigma_max: float) -> None:
        """Set sigma bounds."""
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
    
    def prepare_images(self, observation: Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Prepare images using Pi0's preprocessing.
        
        Pi0's _preprocess_images handles:
        - Resizing with padding to image_resolution
        - Normalization from [0,1] to [-1,1] for SigLIP
        """
        return self.base._preprocess_images(observation)
    
    def prepare_state(self, observation: Dict[str, Tensor]) -> Tensor:
        """Delegate to Pi0's prepare_state (handles padding)."""
        return self.base.prepare_state(observation)
    
    def cache_prefix(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> Dict[str, Any]:
        """
        Cache prefix embeddings and KV for Pi0.
        
        IMPORTANT: Pi0 embeds state in suffix, NOT prefix.
        So we cache the state here for use in denoise_step.
        """
        # Embed prefix (images + language only, no state for Pi0)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.base.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # Create 2D attention masks and position IDs
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # Prepare 4D attention mask for transformer
        prefix_att_2d_masks_4d = self.base.model._prepare_attention_masks_4d(prefix_att_2d_masks)
        
        # Set attention implementation to eager for KV caching
        self.base.model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        
        # Cache KV through forward pass
        _, past_key_values = self.base.model.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        
        return {
            'prefix_pad_masks': prefix_pad_masks,
            'past_key_values': past_key_values,
            'prefix_embs': prefix_embs,
            'prefix_att_masks': prefix_att_masks,
            # IMPORTANT: Pi0 needs state for denoise_step
            'state': state,
        }
    
    def denoise_step_reinflow(
        self,
        cached_data: Dict[str, Any],
        x_t: Tensor,
        timestep: Tensor,
        return_sigma: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply one denoising step for Pi0 with optional sigma computation.
        
        Pi0's denoise_step requires state parameter (state is in suffix, not prefix).
        We compute sigma using our added noise_mlp.
        """
        prefix_pad_masks = cached_data['prefix_pad_masks']
        past_key_values = cached_data['past_key_values']
        state = cached_data['state']  # Pi0 needs state here!
        
        # Pi0's denoise_step signature: (state, prefix_pad_masks, past_key_values, x_t, timestep)
        # But we need to get the suffix_out for sigma computation
        # So we'll inline the denoise_step logic here
        
        # Embed suffix (state + noisy actions + time)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.base.model.embed_suffix(
            state, x_t, timestep
        )
        
        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        
        # Create attention masks
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        
        # Position IDs
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        
        # Prepare 4D attention mask
        full_att_2d_masks_4d = self.base.model._prepare_attention_masks_4d(full_att_2d_masks)
        
        # Set attention implementation
        self.base.model.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        
        # Forward through expert with cached KV
        outputs_embeds, _ = self.base.model.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        
        # Get suffix output
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.base.config.chunk_size:]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        # Compute velocity
        v_t = self.base.model.action_out_proj(suffix_out)
        
        if return_sigma:
            # Compute sigma using our added noise_mlp
            sigma_raw = self.noise_mlp(suffix_out)
            # Tanh bounding: differentiable mapping to [sigma_min, sigma_max]
            sigma_t = self._sigma_min + (self._sigma_max - self._sigma_min) * (torch.tanh(sigma_raw) + 1) / 2
            return v_t, sigma_t
        
        return v_t, None
    
    def sample_actions_reinflow(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
        noise: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        ReinFlow-style sampling for Pi0.
        
        Uses the mixin's denoising loop implementation.
        """
        bsize = state.shape[0]
        device = state.device
        
        # Sample initial noise if not provided
        if noise is None:
            actions_shape = (bsize, self.chunk_size, self.max_action_dim)
            noise = torch.normal(
                mean=0.0, std=1.0, size=actions_shape,
                dtype=torch.float32, device=device
            )
        
        # Cache prefix (and state for Pi0)
        cached_data = self.cache_prefix(
            images, img_masks, lang_tokens, lang_masks, state
        )
        
        # Run denoising loop using mixin
        x_t, trajectory, sigmas_used = self._run_reinflow_denoising_loop(
            cached_data, noise, self.num_denoising_steps
        )
        
        return x_t, trajectory, sigmas_used
    
    def extract_observation_features(self, observation: Dict[str, Tensor]) -> Tensor:
        """
        Extract observation features for critic network.
        
        Uses Pi0's prefix embedding with mean pooling.
        """
        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)
        lang_tokens = observation[OBS_LANGUAGE_TOKENS]
        lang_masks = observation[OBS_LANGUAGE_ATTENTION_MASK]
        
        # Embed prefix (images + language only for Pi0)
        prefix_embs, prefix_pad_masks, _ = self.base.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        
        # Pool over sequence dimension - mean of valid tokens
        mask_expanded = prefix_pad_masks.unsqueeze(-1).float()
        pooled = (prefix_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        return pooled
    
    def get_trainable_components(self) -> Dict[str, nn.Module]:
        """
        Return dict of Pi0's trainable components.
        
        Includes our added noise_mlp.
        """
        model = self.base.model
        components = {
            'noise_mlp': self.noise_mlp,  # Our added noise network
            'action_out_proj': model.action_out_proj,
            'action_time_mlp_out': model.action_time_mlp_out,
            'action_time_mlp_in': model.action_time_mlp_in,
            'action_in_proj': model.action_in_proj,
            'state_proj': model.state_proj,
        }
        
        # Add expert transformer
        components['gemma_expert'] = model.paligemma_with_expert.gemma_expert
        
        return components
    
    def get_base_policy(self) -> nn.Module:
        """Return the underlying PI0Policy."""
        return self.base
    
    def tokenize_instruction(self, instruction: str, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Tokenize instruction using Pi0's PaliGemma tokenizer.
        
        Note: Pi0 uses a different tokenizer than SmolVLA.
        """
        from transformers import AutoTokenizer
        
        # Pi0 uses PaliGemma tokenizer
        if not hasattr(self, '_tokenizer') or self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                "google/paligemma-3b-pt-224",
                trust_remote_code=True
            )
        
        tokens = self._tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self.base.config.tokenizer_max_length,
            truncation=True
        )
        return tokens['input_ids'].to(device), tokens['attention_mask'].bool().to(device)
    
    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        self.base.model.gradient_checkpointing_enable()
        print("[Pi0Adapter] Enabled gradient checkpointing")
    
    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing."""
        self.base.model.gradient_checkpointing_disable()
        print("[Pi0Adapter] Disabled gradient checkpointing")


def create_pi0_adapter(
    pretrained_path: str = "lerobot/pi0",
    device: str = None,
    sigma_min: float = 0.08,
    sigma_max: float = 0.16,
    gradient_checkpointing: bool = True,
    dtype: str = "bfloat16",
) -> Pi0Adapter:
    """
    Factory function to create a Pi0Adapter.
    
    Args:
        pretrained_path: HuggingFace model path or local checkpoint
        device: Target device (auto-detected if None)
        sigma_min: Minimum sigma bound for noise network
        sigma_max: Maximum sigma bound for noise network
        gradient_checkpointing: Enable gradient checkpointing (recommended for 3.3B model)
        dtype: Model dtype ("bfloat16" or "float32")
        
    Returns:
        Pi0Adapter instance
    """
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    print(f"[Pi0Adapter] Loading Pi0 from {pretrained_path}...")
    print(f"[Pi0Adapter] Using device: {device}")
    print(f"[Pi0Adapter] Using dtype: {dtype}")
    
    # Load base policy
    # Note: PI0Policy.from_pretrained handles weight loading and remapping
    base_policy = PI0Policy.from_pretrained(pretrained_path)
    base_policy.to(device)
    base_policy.eval()
    
    # Create adapter
    adapter = Pi0Adapter(base_policy, sigma_min=sigma_min, sigma_max=sigma_max)
    
    # Enable gradient checkpointing if requested (recommended for 3.3B model)
    if gradient_checkpointing:
        adapter.enable_gradient_checkpointing()
    
    print(f"[Pi0Adapter] Created adapter with VLM hidden size: {adapter.vlm_hidden_size}")
    print(f"[Pi0Adapter] Expert hidden size: {adapter.expert_hidden_size}")
    
    return adapter

