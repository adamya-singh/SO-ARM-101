"""
VLA Policy Interface for ReinFlow Training

Abstract interface that allows ReinFlow to work with different VLA models
(SmolVLA, Pi0) through a unified API. This handles the architectural
differences between models, particularly around state embedding and
denoising step signatures.

Key differences handled:
- SmolVLA: state embedded in prefix, denoise_step doesn't need state
- Pi0: state embedded in suffix, denoise_step needs state parameter
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn
from torch import Tensor


class VLAPolicyInterface(ABC):
    """
    Abstract interface for VLA policies to be used with ReinFlow.
    
    This interface abstracts away the differences between SmolVLA and Pi0
    architectures, providing a unified API for:
    - Prefix embedding and caching
    - Denoising steps with sigma computation
    - Trajectory sampling for ReinFlow
    """
    
    @property
    @abstractmethod
    def vlm_hidden_size(self) -> int:
        """Return the VLM's hidden dimension size (for critic network input)."""
        pass
    
    @property
    @abstractmethod
    def expert_hidden_size(self) -> int:
        """Return the action expert's hidden dimension size."""
        pass
    
    @property
    @abstractmethod
    def chunk_size(self) -> int:
        """Return the action chunk size."""
        pass
    
    @property
    @abstractmethod
    def max_action_dim(self) -> int:
        """Return the maximum action dimension (padded)."""
        pass
    
    @property
    @abstractmethod
    def original_action_dim(self) -> int:
        """Return the original (unpadded) action dimension."""
        pass
    
    @property
    @abstractmethod
    def num_denoising_steps(self) -> int:
        """Return the number of denoising steps (K)."""
        pass
    
    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Return the device the model is on."""
        pass
    
    @property
    @abstractmethod
    def noise_network(self) -> nn.Module:
        """Return the noise network (sigma_theta')."""
        pass
    
    @property
    @abstractmethod
    def sigma_min(self) -> float:
        """Return minimum sigma bound."""
        pass
    
    @property
    @abstractmethod
    def sigma_max(self) -> float:
        """Return maximum sigma bound."""
        pass
    
    @abstractmethod
    def set_sigma_bounds(self, sigma_min: float, sigma_max: float) -> None:
        """Set the sigma bounds for noise network."""
        pass
    
    @abstractmethod
    def prepare_images(self, observation: Dict[str, Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        """
        Prepare images from observation dict for the model.
        
        Args:
            observation: Dict containing image tensors
            
        Returns:
            images: List of image tensors
            img_masks: List of image mask tensors
        """
        pass
    
    @abstractmethod
    def prepare_state(self, observation: Dict[str, Tensor]) -> Tensor:
        """
        Prepare state tensor from observation dict.
        
        Args:
            observation: Dict containing state tensor
            
        Returns:
            state: Prepared state tensor (padded if needed)
        """
        pass
    
    @abstractmethod
    def cache_prefix(
        self,
        images: List[Tensor],
        img_masks: List[Tensor],
        lang_tokens: Tensor,
        lang_masks: Tensor,
        state: Tensor,
    ) -> Dict[str, Any]:
        """
        Compute and cache prefix embeddings and KV cache.
        
        This method handles the architectural difference between SmolVLA
        (state in prefix) and Pi0 (state in suffix) by returning all
        necessary cached data for subsequent denoise steps.
        
        Args:
            images: List of image tensors
            img_masks: List of image mask tensors
            lang_tokens: Language token tensor
            lang_masks: Language attention mask
            state: Robot state tensor
            
        Returns:
            cached_data: Dict containing:
                - prefix_pad_masks: Padding masks for prefix
                - past_key_values: Cached KV from prefix encoding
                - state: State tensor (for Pi0 which needs it in denoise_step)
                - prefix_embs: Prefix embeddings (for critic feature extraction)
        """
        pass
    
    @abstractmethod
    def denoise_step_reinflow(
        self,
        cached_data: Dict[str, Any],
        x_t: Tensor,
        timestep: Tensor,
        return_sigma: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Apply one denoising step with optional sigma computation.
        
        This abstracts the different denoise_step signatures:
        - SmolVLA: denoise_step(prefix_pad_masks, past_key_values, x_t, timestep)
        - Pi0: denoise_step(state, prefix_pad_masks, past_key_values, x_t, timestep)
        
        Args:
            cached_data: Dict from cache_prefix()
            x_t: Current noisy actions (batch, chunk_size, action_dim)
            timestep: Current timestep tensor
            return_sigma: If True, also compute and return sigma
            
        Returns:
            v_t: Velocity prediction (batch, chunk_size, action_dim)
            sigma_t: (optional) Noise std from noise network if return_sigma=True
        """
        pass
    
    @abstractmethod
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
        ReinFlow-style sampling with noise injection at each denoising step.
        
        Args:
            images: List of image tensors
            img_masks: List of image mask tensors
            lang_tokens: Language token tensor
            lang_masks: Language attention mask
            state: Robot state tensor
            noise: Optional initial noise
            
        Returns:
            actions: (batch, chunk_size, action_dim) final sampled actions
            trajectory: List of K+1 tensors [a^0, a^1, ..., a^K]
            sigmas_used: List of K tensors, sigma used at each step
        """
        pass
    
    @abstractmethod
    def extract_observation_features(self, observation: Dict[str, Tensor]) -> Tensor:
        """
        Extract observation features for critic network.
        
        Args:
            observation: Dict with observation tensors
            
        Returns:
            features: (batch_size, vlm_hidden_size) tensor
        """
        pass
    
    @abstractmethod
    def get_trainable_components(self) -> Dict[str, nn.Module]:
        """
        Return dict of trainable component names to modules.
        
        Used for selective unfreezing and optimizer setup.
        
        Returns:
            Dict mapping component names to nn.Module instances
        """
        pass
    
    @abstractmethod
    def get_base_policy(self) -> nn.Module:
        """Return the underlying base policy (SmolVLAPolicy or PI0Policy)."""
        pass
    
    @abstractmethod
    def tokenize_instruction(self, instruction: str, device: torch.device) -> Tuple[Tensor, Tensor]:
        """
        Tokenize a language instruction.
        
        Args:
            instruction: Text instruction string
            device: Target device for tensors
            
        Returns:
            tokens: Token IDs tensor
            attention_mask: Attention mask tensor
        """
        pass


class ReinFlowCapableMixin:
    """
    Mixin providing common ReinFlow functionality.
    
    This mixin provides the denoising loop logic that's shared between
    adapters, using the abstract methods defined in VLAPolicyInterface.
    """
    
    def _run_reinflow_denoising_loop(
        self: VLAPolicyInterface,
        cached_data: Dict[str, Any],
        noise: Tensor,
        num_steps: int,
    ) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Run the ReinFlow denoising loop with noise injection.
        
        Args:
            cached_data: Cached prefix data from cache_prefix()
            noise: Initial noise tensor (batch, chunk_size, action_dim)
            num_steps: Number of denoising steps
            
        Returns:
            x_t: Final denoised actions
            trajectory: List of K+1 action tensors
            sigmas_used: List of K sigma tensors
        """
        device = noise.device
        bsize = noise.shape[0]
        
        dt = -1.0 / num_steps
        dt_tensor = torch.tensor(dt, dtype=torch.float32, device=device)
        
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        
        # Store full denoising trajectory and sigmas
        trajectory = [x_t.clone()]
        sigmas_used = []
        
        while time >= -dt_tensor / 2:
            expanded_time = time.expand(bsize)
            
            # Get velocity and sigma from denoise step
            v_t, sigma_t = self.denoise_step_reinflow(
                cached_data, x_t, expanded_time, return_sigma=True
            )
            
            # Deterministic Euler step = mean of transition
            mu = x_t + dt_tensor * v_t
            
            # Inject noise from noise network (ReinFlow core)
            eps = torch.randn_like(x_t)
            x_next = mu + sigma_t * eps
            
            # Clip to prevent noise from interrupting path too violently
            x_next = torch.clamp(x_next, -1.0, 1.0)
            
            # Store trajectory and sigma
            trajectory.append(x_next.clone())
            sigmas_used.append(sigma_t.clone())
            
            x_t = x_next
            time = time + dt_tensor
        
        return x_t, trajectory, sigmas_used

