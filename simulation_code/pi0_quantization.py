"""
Pi0 4-bit Quantization Utilities for MPS (Apple Silicon)

This module provides utilities to load Pi0 with 4-bit weight quantization
on MPS devices to reduce memory usage and enable inference on M1/M2 MacBooks.

Uses HuggingFace's optimum-quanto library for INT4 weight quantization.
Falls back to float16 if quantization fails.
"""

import torch
from typing import Optional, Tuple


def is_mps_device(device: torch.device) -> bool:
    """Check if the device is MPS (Apple Silicon GPU)."""
    if isinstance(device, str):
        return device == "mps"
    return device.type == "mps"


def should_quantize_pi0(device: torch.device, force_quantize: bool = False) -> bool:
    """
    Check if we should quantize Pi0.
    
    By default, quantizes on MPS devices to reduce memory usage.
    Can be forced on other devices with force_quantize=True.
    
    Args:
        device: The target device
        force_quantize: Force quantization regardless of device
        
    Returns:
        True if quantization should be applied
    """
    if force_quantize:
        return True
    return is_mps_device(device)


def _check_quanto_available() -> bool:
    """Check if optimum-quanto is available."""
    try:
        # Try new import path (quanto >= 0.3)
        import quanto
        return True
    except ImportError:
        pass
    try:
        # Try old import path (optimum-quanto < 0.3)
        from optimum import quanto
        return True
    except ImportError:
        return False


def _get_quanto_imports():
    """Get quanto imports, handling different version import paths."""
    try:
        # Try new import path (quanto >= 0.3)
        from quanto import quantize, qint4, freeze
        return quantize, qint4, freeze
    except ImportError:
        # Fall back to old import path (optimum-quanto < 0.3)
        from optimum.quanto import quantize, qint4, freeze
        return quantize, qint4, freeze


def _quantize_model_weights(model: torch.nn.Module) -> torch.nn.Module:
    """
    Apply INT4 weight quantization to model using quanto.
    
    Args:
        model: The PyTorch model to quantize
        
    Returns:
        Quantized model
    """
    quantize, qint4, _ = _get_quanto_imports()
    
    # Quantize all linear layers to INT4 weights
    quantize(model, weights=qint4)
    
    return model


def load_pi0_quantized(
    pretrained_path: str,
    device: str = None,
    quantize: bool = True,
    verbose: bool = True,
) -> Tuple["PI0Policy", bool]:
    """
    Load Pi0 policy with optional 4-bit quantization.
    
    On MPS devices, applies INT4 weight quantization to reduce memory usage.
    Falls back to float16 if quantization is unavailable or fails.
    
    Args:
        pretrained_path: HuggingFace model path or local checkpoint
        device: Target device (auto-detected if None)
        quantize: Whether to attempt quantization (default True)
        verbose: Print status messages
        
    Returns:
        Tuple of (policy, was_quantized) where was_quantized indicates
        if INT4 quantization was successfully applied
    """
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    
    # Auto-detect device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    device_obj = torch.device(device)
    should_quantize = quantize and should_quantize_pi0(device_obj)
    was_quantized = False
    
    if verbose:
        print(f"[Pi0Quantization] Loading Pi0 from {pretrained_path}")
        print(f"[Pi0Quantization] Target device: {device}")
    
    # Check if quanto is available for quantization
    quanto_available = _check_quanto_available()
    
    if should_quantize and not quanto_available:
        print("[Pi0Quantization] WARNING: optimum-quanto not installed.")
        print("[Pi0Quantization] Install with: pip install optimum-quanto")
        print("[Pi0Quantization] Falling back to float16 precision.")
        should_quantize = False
    
    if should_quantize:
        try:
            if verbose:
                print("[Pi0Quantization] Applying INT4 weight quantization...")
            
            # Load model first (on CPU to avoid memory issues during quantization)
            policy = PI0Policy.from_pretrained(pretrained_path)
            
            # Apply quantization to the underlying model
            _quantize_model_weights(policy.model)
            
            # Freeze quantized weights (required after quanto quantize)
            _, _, freeze = _get_quanto_imports()
            freeze(policy.model)
            
            # Move to target device
            policy.to(device)
            policy.eval()
            
            was_quantized = True
            if verbose:
                print("[Pi0Quantization] INT4 quantization applied successfully!")
                _print_memory_stats(device)
                
        except Exception as e:
            print(f"[Pi0Quantization] WARNING: Quantization failed: {e}")
            print("[Pi0Quantization] Falling back to float16 precision.")
            
            # Reload without quantization
            policy = PI0Policy.from_pretrained(pretrained_path)
            
            # Use float16 on MPS for memory efficiency
            if is_mps_device(device_obj):
                policy = policy.to(dtype=torch.float16)
            
            policy.to(device)
            policy.eval()
            was_quantized = False
    else:
        # Standard loading without quantization
        if verbose:
            print("[Pi0Quantization] Loading without quantization...")
        
        policy = PI0Policy.from_pretrained(pretrained_path)
        
        # Use float16 on MPS for memory efficiency even without quantization
        if is_mps_device(device_obj):
            if verbose:
                print("[Pi0Quantization] Using float16 for MPS device")
            policy = policy.to(dtype=torch.float16)
        
        policy.to(device)
        policy.eval()
    
    if verbose:
        print("[Pi0Quantization] Pi0 policy loaded successfully!")
    
    return policy, was_quantized


def _print_memory_stats(device: str):
    """Print memory usage statistics."""
    if device == "mps":
        # MPS memory stats (if available in PyTorch version)
        try:
            if hasattr(torch.mps, 'current_allocated_memory'):
                allocated = torch.mps.current_allocated_memory() / 1024**3
                print(f"[Pi0Quantization] MPS memory allocated: {allocated:.2f} GB")
        except Exception:
            pass
    elif device == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"[Pi0Quantization] CUDA memory: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


def estimate_memory_savings():
    """
    Estimate memory savings from INT4 quantization.
    
    Pi0 model size breakdown:
    - PaliGemma (SigLIP + Gemma 2B): ~2.9B parameters
    - Action Expert (Gemma 2B variant): ~300M parameters  
    - Total: ~3.3B parameters
    
    Memory estimates:
    - float32: ~13.2 GB
    - float16: ~6.6 GB
    - INT4 (weights only): ~2-3 GB (weights) + activations in float16
    """
    print("\n=== Pi0 Memory Estimation ===")
    print("Model: ~3.3B parameters")
    print("")
    print("Memory usage estimates:")
    print("  float32: ~13.2 GB")
    print("  float16: ~6.6 GB")
    print("  INT4 weights + float16 activations: ~3-4 GB")
    print("")
    print("Note: Actual usage depends on batch size and sequence length.")
    print("================================\n")

