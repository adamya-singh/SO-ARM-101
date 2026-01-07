"""
SmolVLA Adapter for ReinFlow Training

Thin wrapper adapting SmolVLAPolicy to the VLAPolicyInterface.
SmolVLA already has most ReinFlow functionality built in (noise_mlp,
sample_actions_reinflow), so this adapter mainly delegates to the
existing implementation.

Key characteristics of SmolVLA:
- State is embedded in prefix (via embed_prefix(state=state))
- denoise_step() does NOT need state parameter
- Has built-in noise_mlp for ReinFlow
- VLM hidden size: 960
"""

from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor

from vla_policy_interface import VLAPolicyInterface, ReinFlowCapableMixin
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy, make_att_2d_masks
from lerobot.utils.constants import OBS_STATE


class SmolVLAAdapter(VLAPolicyInterface, ReinFlowCapableMixin):
    """
    Adapter wrapping SmolVLAPolicy for the VLAPolicyInterface.
    
    SmolVLA already has ReinFlow support built-in, so this adapter
    is a thin wrapper that delegates most functionality to the
    underlying SmolVLAPolicy.
    
    Args:
        base_policy: Pre-loaded SmolVLAPolicy instance
    """
    
    def __init__(self, base_policy: SmolVLAPolicy):
        self.base = base_policy
        self._device = next(base_policy.parameters()).device
    
    @property
    def vlm_hidden_size(self) -> int:
        """SmolVLA uses SmolVLM2-500M with 960 hidden size."""
        return self.base.model.vlm_with_expert.config.text_config.hidden_size
    
    @property
    def expert_hidden_size(self) -> int:
        """Return the action expert's hidden dimension."""
        return self.base.model.vlm_with_expert.expert_hidden_size
    
    @property
    def chunk_size(self) -> int:
        return self.base.config.chunk_size
    
    @property
    def max_action_dim(self) -> int:
        return self.base.config.max_action_dim
    
    @property
    def original_action_dim(self) -> int:
        return self.base.config.action_feature.shape[0]
    
    @property
    def num_denoising_steps(self) -> int:
        return self.base.config.num_steps
    
    @property
    def device(self) -> torch.device:
        return self._device
    
    @property
    def noise_network(self) -> nn.Module:
        """SmolVLA has built-in noise_mlp."""
        return self.base.model.noise_mlp
    
    @property
    def sigma_min(self) -> float:
        return self.base.model.sigma_min
    
    @property
    def sigma_max(self) -> float:
        return self.base.model.sigma_max
    
    def set_sigma_bounds(self, sigma_min: float, sigma_max: float) -> None:
        """Set sigma bounds on the underlying model."""
        self.base.model.sigma_min = sigma_min
        self.base.model.sigma_max = sigma_max
    
    def prepare_images(self, observation: Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """Delegate to SmolVLA's prepare_images."""
        return self.base.prepare_images(observation)
    
    def prepare_state(self, observation: Dict[str, Tensor]) -> Tensor:
        """Delegate to SmolVLA's prepare_state."""
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
        Cache prefix embeddings and KV for SmolVLA.
        
        SmolVLA embeds state in the prefix, so we include it here.
        """
        # Embed prefix with state (SmolVLA-specific: state goes in prefix)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.base.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        
        # Create 2D attention masks and position IDs
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        
        # Cache KV through forward pass
        _, past_key_values = self.base.model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=self.base.model.config.use_cache,
            fill_kv_cache=True,
        )
        
        return {
            'prefix_pad_masks': prefix_pad_masks,
            'past_key_values': past_key_values,
            'prefix_embs': prefix_embs,
            'prefix_att_masks': prefix_att_masks,
            # SmolVLA doesn't need state in denoise_step, but we store for consistency
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
        Apply one denoising step using SmolVLA's built-in method.
        
        SmolVLA's denoise_step already supports return_sigma for ReinFlow.
        """
        prefix_pad_masks = cached_data['prefix_pad_masks']
        past_key_values = cached_data['past_key_values']
        
        if return_sigma:
            v_t, sigma_t = self.base.model.denoise_step(
                prefix_pad_masks, past_key_values, x_t, timestep, return_sigma=True
            )
            return v_t, sigma_t
        else:
            v_t = self.base.model.denoise_step(
                prefix_pad_masks, past_key_values, x_t, timestep, return_sigma=False
            )
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
        Delegate to SmolVLA's built-in sample_actions_reinflow.
        
        SmolVLA already has this method implemented.
        """
        return self.base.model.sample_actions_reinflow(
            images, img_masks, lang_tokens, lang_masks, state, noise
        )
    
    def extract_observation_features(self, observation: Dict[str, Tensor]) -> Tensor:
        """
        Extract observation features for critic network.
        
        Uses SmolVLA's prefix embedding with mean pooling over valid tokens.
        """
        images, img_masks = self.prepare_images(observation)
        state = self.prepare_state(observation)
        
        from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
        lang_tokens = observation[OBS_LANGUAGE_TOKENS]
        lang_masks = observation[OBS_LANGUAGE_ATTENTION_MASK]
        
        # Embed prefix (with state for SmolVLA)
        prefix_embs, prefix_pad_masks, _ = self.base.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        
        # Pool over sequence dimension - mean of valid (non-padded) tokens
        mask_expanded = prefix_pad_masks.unsqueeze(-1).float()
        pooled = (prefix_embs * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
        
        return pooled
    
    def get_trainable_components(self) -> Dict[str, nn.Module]:
        """
        Return dict of SmolVLA's trainable components.
        """
        model = self.base.model
        components = {
            'noise_mlp': model.noise_mlp,
            'action_out_proj': model.action_out_proj,
            'action_time_mlp_out': model.action_time_mlp_out,
            'action_time_mlp_in': model.action_time_mlp_in,
            'action_in_proj': model.action_in_proj,
            'state_proj': model.state_proj,
        }
        
        # Add expert transformer if it exists
        if hasattr(model.vlm_with_expert, 'expert'):
            components['expert'] = model.vlm_with_expert.expert
        
        return components
    
    def get_base_policy(self) -> nn.Module:
        """Return the underlying SmolVLAPolicy."""
        return self.base
    
    def tokenize_instruction(self, instruction: str, device: torch.device) -> Tuple[Tensor, Tensor]:
        """Tokenize instruction using SmolVLA's tokenizer."""
        if hasattr(self.base, 'tokenizer') and self.base.tokenizer is not None:
            tokens = self.base.tokenizer(
                instruction,
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            return tokens['input_ids'].to(device), tokens['attention_mask'].bool().to(device)
        else:
            # Fallback dummy tokens
            return (
                torch.zeros((1, 1), dtype=torch.long, device=device),
                torch.ones((1, 1), dtype=torch.bool, device=device)
            )


def create_smolvla_adapter(
    pretrained_path: str = "lerobot/smolvla_base",
    device: str = None,
) -> SmolVLAAdapter:
    """
    Factory function to create a SmolVLAAdapter.
    
    Args:
        pretrained_path: HuggingFace model path or local checkpoint
        device: Target device (auto-detected if None)
        
    Returns:
        SmolVLAAdapter instance
    """
    from transformers import AutoTokenizer
    
    # Auto-detect device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    
    print(f"[SmolVLAAdapter] Loading SmolVLA from {pretrained_path}...")
    print(f"[SmolVLAAdapter] Using device: {device}")
    
    # Load base policy
    base_policy = SmolVLAPolicy.from_pretrained(pretrained_path)
    base_policy.to(device)
    base_policy.eval()
    
    # Load tokenizer if missing
    if not hasattr(base_policy, 'tokenizer') or base_policy.tokenizer is None:
        print("[SmolVLAAdapter] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
        base_policy.tokenizer = tokenizer
    
    adapter = SmolVLAAdapter(base_policy)
    print(f"[SmolVLAAdapter] Created adapter with VLM hidden size: {adapter.vlm_hidden_size}")
    
    return adapter

