# Debugging the KL Divergence Explosion in Parallel ReinFlow Training

**Date:** January 2, 2026  
**Issue:** Astronomically high KL divergence (~300 million) during PPO training with parallel environments  
**Time to Debug:** ~2 hours of deep code analysis

---

## The Problem

### Command Run

```bash
python train_reinflow.py --parallel-envs 10 --no-render --headless
```

### Training Configuration

The training was configured with these hyperparameters (printed at startup):

```
============================================================
Starting ReinFlow Training (PPO, Parallel)
============================================================
Instruction: 'pick up the block'
Parallel environments: 10
Chunks per episode: 3
Denoising steps: 1
Policy LR: 5e-06 -> 5.000000000000001e-07
Critic LR: 0.0003 -> 2.9999999999999997e-05
LR warmup iterations: 10
PPO epochs: 10
Mini-batch size: 8
Gradient accumulation: 15
Effective batch size: 120
Clip epsilon: 0.0005
GAE lambda: 0.95
Target KL: 100.0
Entropy coeff: 0.0
Critic warmup iters: 2
Training mode: PPO ON-POLICY
============================================================
```

### The Error

Immediately after critic warmup, every batch showed KL values in the hundreds of millions:

```
[Critic Warmup] Complete!

  [KL Early Stop] Epoch 1, KL=303228256.0000 > 150.0000
Batch    1 (   10 eps) | Reward:  -60.44 | KL: 303228256.0000 | LR: 9.50e-07 | Time: 2.7s
  [KL Early Stop] Epoch 1, KL=60731376.0000 > 150.0000
Batch    2 (   20 eps) | Reward:  -61.68 | KL: 60731376.0000 | LR: 1.40e-06 | Time: 2.5s
  [KL Early Stop] Epoch 1, KL=320451456.0000 > 150.0000
Batch    3 (   30 eps) | Reward:  -62.14 | KL: 320451456.0000 | LR: 1.85e-06 | Time: 2.5s
  [KL Early Stop] Epoch 1, KL=363878752.0000 > 150.0000
Batch    4 (   40 eps) | Reward:  -60.66 | KL: 363878752.0000 | LR: 2.30e-06 | Time: 2.5s
  [KL Early Stop] Epoch 1, KL=181936960.0000 > 150.0000
Batch    5 (   50 eps) | Reward:  -60.24 | KL: 181936960.0000 | LR: 2.75e-06 | Time: 2.5s
...
```

**Key observations:**
- KL values ranged from 60 million to 485 million
- Early stopping triggered on Epoch 1 (first PPO epoch) — before ANY weight updates
- Target KL was 100.0, but actual KL exceeded by 6+ orders of magnitude
- Normal PPO KL values should be 0.001 to 0.1

---

## Investigation Process

### Step 1: Understanding PPO and KL Divergence

**PPO (Proximal Policy Optimization)** is a policy gradient algorithm that constrains how much the policy can change in a single update. It does this by:

1. Collecting trajectories with the current policy
2. Computing "old" log probabilities for the actions taken
3. During optimization, computing "new" log probabilities
4. Clipping the ratio `π_new / π_old` to prevent large updates

The **KL divergence** measures how different the new policy is from the old policy. If KL is too large, PPO triggers early stopping.

### Step 2: Tracing the KL Calculation

Located the KL calculation in `reinflow_smolvla.py`, function `compute_ppo_loss()`:

```python
# Lines 741-780 in reinflow_smolvla.py
log_ratio_raw = new_log_probs - old_log_probs
# Clamp log_ratio to prevent numerical overflow (standard PPO stabilization)
log_ratio = torch.clamp(log_ratio_raw, -20.0, 20.0)
ratio = torch.exp(log_ratio)

# ... later ...

# KL divergence approximation for monitoring and early stopping
# Using the approximation: KL ≈ (r - 1) - log(r)
with torch.no_grad():
    approx_kl = ((ratio - 1) - log_ratio).mean().item()
```

**Math check:** For KL ≈ 300 million:
- The `log_ratio` must be hitting the +20 clamp
- `exp(20) ≈ 485,165,195`
- `KL = (485M - 1) - 20 ≈ 485 million`

This confirmed: `new_log_probs` were **much higher** than `old_log_probs` (by more than 20 in log space).

### Step 3: Tracing Log Probability Computation

Found two places where log probabilities are computed:

**1. Before PPO epochs (train_reinflow.py, lines 610-615):**
```python
# Compute old log probabilities (detached for PPO ratio)
with torch.no_grad():
    old_log_probs = compute_trajectory_log_probs_onpolicy(
        rl_policy, all_trajectories, all_observations
    )
    old_values = all_values.clone()
```

**2. During each mini-batch (compute_ppo_loss, lines 731-739):**
```python
if entropy_coeff > 0:
    new_log_probs, sigmas = compute_trajectory_log_probs_onpolicy(
        policy, trajectories, observations, return_sigmas=True
    )
else:
    new_log_probs = compute_trajectory_log_probs_onpolicy(
        policy, trajectories, observations, return_sigmas=False
    )
```

### Step 4: The Critical Realization

The KL explosion happened on **Epoch 1, before any gradient updates**. The training loop is:

```
1. Collect trajectories
2. Compute old_log_probs (ONCE)
3. For each PPO epoch:
   3a. For each mini-batch:
       - Compute new_log_probs
       - Compute KL  ← EXPLODED HERE on first mini-batch!
       - Compute loss
       - Backward pass
       - (optimizer.step only after gradient accumulation)
```

On the very first mini-batch of epoch 1, **no optimizer.step() had been called yet**. The policy weights were identical for old and new log prob computations.

**Yet the KL was 300 million.**

This meant: same weights, same inputs → different outputs. **Non-determinism.**

### Step 5: Hunting for Non-Determinism

Systematically searched for sources of randomness:

| Source | Found? | Notes |
|--------|--------|-------|
| `torch.randn()` in log prob computation | No | Only used in trajectory sampling, not log prob |
| Batch normalization | No | Transformers use LayerNorm (per-sample) |
| Random data augmentation | No | Not used |
| **Dropout** | **YES** | HuggingFace transformers have dropout layers |

### Step 6: Understanding the Log Probability Formula

The `compute_trajectory_log_probs_onpolicy` function computes:

```python
# For each denoising step k:
# mu_k = a_k + dt * v_k  (where v_k is velocity from neural network)
# log p(a_{k+1} | mu_k, sigma_k) = log N(a_{k+1}; mu_k, sigma_k^2)

diff = a_k_next_slice - mu_k
log_prob_k = -0.5 * (
    (diff ** 2 / (sigma_k_slice ** 2 + 1e-8)).sum(dim=(-1, -2)) +
    d * math.log(2 * math.pi) +
    2 * torch.log(sigma_k_slice + 1e-8).sum(dim=(-1, -2))
)
```

The key term is `diff = a_{k+1} - mu_k`. If the velocity prediction `v_k` changes (due to dropout), then `mu_k` changes, and the log probability changes dramatically.

With sigma ≈ 0.12 and chunk_size × action_dim = 300:
- If diff ≈ 0.01 (velocity consistent): log_prob ≈ -360
- If diff ≈ 0.5 (velocity inconsistent): log_prob ≈ -2240
- Difference: ~1880 in log space → ratio of `exp(1880)` → clamped to `exp(20)` → massive KL

---

## Root Cause: Dropout in Training Mode

### What is Dropout?

Dropout is a regularization technique invented by Hinton et al. (2014). During training, each neuron is randomly "dropped" (set to zero) with probability p (typically 0.1-0.5). This:
- Prevents co-adaptation of neurons
- Acts as an ensemble method
- Reduces overfitting

**Critical behavior difference:**
- **Training mode (`model.train()`)**: Dropout active, outputs are stochastic
- **Eval mode (`model.eval()`)**: Dropout disabled, outputs are deterministic

```
Training Mode (dropout ON):
Forward pass 1: [neuron1, 0,       neuron3, 0,       neuron5] → Output A
Forward pass 2: [0,       neuron2, 0,       neuron4, neuron5] → Output B
                                                              ↑ DIFFERENT!

Eval Mode (dropout OFF):
Forward pass 1: [neuron1, neuron2, neuron3, neuron4, neuron5] → Output C
Forward pass 2: [neuron1, neuron2, neuron3, neuron4, neuron5] → Output C
                                                              ↑ SAME!
```

### The Model Architecture

```
ReinFlowSmolVLA (nn.Module - our training wrapper)
│
├── base = SmolVLAPolicy (~500M parameters)
│   │
│   └── model = VLAFlowMatching
│       │
│       ├── vlm_with_expert = SmolVLMWithExpertModel
│       │   │
│       │   ├── vlm = SmolVLM2 (HuggingFace transformer)
│       │   │         ↑ CONTAINS DROPOUT in attention and MLP layers
│       │   │
│       │   └── expert = Action Expert transformer
│       │             ↑ ALSO CONTAINS DROPOUT
│       │
│       ├── action_out_proj (Linear - velocity output)
│       ├── noise_mlp (MLP - sigma output for ReinFlow)
│       └── state_proj, action_in_proj, etc.
│
└── critic = ReinFlowCritic (~620K parameters)
    └── net = MLP with LayerNorm (no dropout)
```

### The Setup Code (What Was Supposed to Happen)

In `reinflow_smolvla.py`, function `setup_reinflow_policy()`:

```python
# Line 846-849
base_policy = SmolVLAPolicy.from_pretrained(pretrained_path)
base_policy.to(device)
base_policy.eval()  # ← This SHOULD disable dropout
```

Then the wrapper is created:

```python
# Lines 859-867
reinflow_policy = ReinFlowSmolVLA(
    base_policy,
    num_steps=num_steps,
    train_action_head=train_action_head,
    train_time_mlp=train_time_mlp,
    train_full_expert=train_full_expert,
    train_noise_head=train_noise_head,
    train_critic=train_critic,
    device=device
)
```

### What Actually Went Wrong

The `ReinFlowSmolVLA` class extends `nn.Module`:

```python
class ReinFlowSmolVLA(nn.Module):
    def __init__(self, base_policy, ...):
        super().__init__()  # ← nn.Module defaults to training=True
        self.base = base_policy  # ← base_policy was in eval mode
        self.critic = ReinFlowCritic(...)
        ...
```

**PyTorch's module hierarchy behavior:**

When you assign an `nn.Module` to an attribute of another `nn.Module`, it becomes a **submodule**. When you call `.train()` on the parent, it recursively calls `.train()` on ALL submodules.

```python
# What happens when anything calls rl_policy.train():
rl_policy.train()
# Internally does:
#   self.training = True
#   for child in self.children():
#       child.train()  # ← This sets base_policy.training = True!
```

Even though `base_policy.eval()` was called during setup, if ANYTHING later called `rl_policy.train()`, it would override the eval mode on the base policy.

**The smoking gun:** PyTorch's default behavior. New `nn.Module` instances start with `training=True`. Even without an explicit `.train()` call, the base policy might have been affected.

### Timeline of the Bug

```
1. base_policy = SmolVLAPolicy.from_pretrained(...)
   base_policy.training = True (default)

2. base_policy.eval()
   base_policy.training = False ← CORRECT

3. reinflow_policy = ReinFlowSmolVLA(base_policy, ...)
   reinflow_policy.training = True (default for new nn.Module)
   reinflow_policy.base = base_policy
   reinflow_policy.base.training = False ← STILL CORRECT

4. [Somewhere in training loop or PyTorch internals]
   reinflow_policy.train() or similar
   reinflow_policy.training = True
   reinflow_policy.base.training = True ← BUG! Dropout now active!

5. compute old_log_probs
   Forward pass with RANDOM DROPOUT MASK A
   old_log_probs = [...]

6. compute new_log_probs  
   Forward pass with RANDOM DROPOUT MASK B
   new_log_probs = [...] ← DIFFERENT VALUES!

7. KL = huge because log probs differ by >20
```

### Why Parallel Mode Made It Worse

With sequential training (1 environment), the issue might be masked because:
- Smaller batch sizes
- Fewer forward passes
- Dropout variance might average out

With parallel training (10 environments):
- Batch size = 30 (10 envs × 3 chunks)
- Mini-batches of size 8 selected from 30
- Each mini-batch requires a fresh forward pass
- More forward passes = more dropout randomness
- Variance accumulates → more obvious KL explosion

---

## The Fix

### Solution: Override `train()` Method

Added to `ReinFlowSmolVLA` class in `reinflow_smolvla.py` (line 273):

```python
def train(self, mode=True):
    """
    Override train to keep base policy in eval mode.
    
    The base SmolVLA model has dropout layers that must remain disabled
    for deterministic forward passes during PPO updates. Only the critic
    and newly added parameters (noise_mlp) should be in training mode.
    """
    # Set wrapper module's training flag
    self.training = mode
    # Critic can be in training mode
    self.critic.train(mode)
    # Base policy MUST stay in eval mode (no dropout)
    self.base.eval()
    return self
```

### Why This Works

Now, regardless of what calls `.train()` on the wrapper:
1. The wrapper's `training` flag is set (allowing gradient computation)
2. The critic is put in training mode (doesn't matter, it has no dropout)
3. The base policy is **forced back to eval mode** (no dropout)

```
After fix:
1. rl_policy.train() is called
2. rl_policy.training = True
3. rl_policy.critic.training = True  
4. rl_policy.base.training = False ← FORCED BY OUR OVERRIDE

5. compute old_log_probs
   Forward pass with NO DROPOUT
   old_log_probs = [X, Y, Z, ...]

6. compute new_log_probs
   Forward pass with NO DROPOUT
   new_log_probs = [X, Y, Z, ...] ← SAME VALUES!

7. KL ≈ 0 (before any weight updates)
```

### Alternative Fixes Considered

1. **Explicit `base.eval()` calls before each log prob computation**
   - Pros: Simple
   - Cons: Easy to forget, not self-documenting, multiple call sites

2. **Using `torch.inference_mode()` context**
   - Pros: Disables dropout and gradients
   - Cons: Can't use for `new_log_probs` (need gradients)

3. **Modifying HuggingFace model config to disable dropout**
   - Pros: Permanent fix
   - Cons: Requires model reload, affects all uses

**The `train()` override is the cleanest solution** because:
- Self-documenting (clear docstring explaining why)
- Automatic (no need to remember manual calls)
- Follows PyTorch conventions
- Localized change (single method in one file)

---

## Verification

After applying the fix, expected behavior:

```
Batch    1 (   10 eps) | Reward:  -60.44 | KL: 0.0012 | LR: 9.50e-07
Batch    2 (   20 eps) | Reward:  -61.68 | KL: 0.0018 | LR: 1.40e-06
Batch    3 (   30 eps) | Reward:  -62.14 | KL: 0.0025 | LR: 1.85e-06
...
```

KL values should now be in the normal range (0.001 - 0.1) and only increase as actual weight updates occur.

---

## Key Lessons Learned

### 1. PPO Requires Deterministic Policy Evaluation

The importance ratio `r(θ) = π_θ(a|s) / π_θ_old(a|s)` is the core of PPO. If this ratio is noisy due to stochastic forward passes, the algorithm breaks down. The ratio should only change when the policy weights change.

### 2. Eval Mode is Critical for Inference in Training Loops

Even during "training," any inference-like forward passes (computing log probs, values, etc.) should use eval mode to ensure determinism. Only the backward pass needs training mode.

### 3. PyTorch's Module Hierarchy Has Hidden Gotchas

When wrapping models:
- Submodules inherit training mode from parents
- `model.train()` propagates to ALL submodules
- You need explicit overrides to break this propagation

### 4. Extreme Values Indicate Fundamental Issues

KL values in the millions aren't a hyperparameter problem. They indicate:
- Non-deterministic evaluation
- Numerical instability
- Data corruption
- Architecture bugs

Normal debugging (adjusting learning rates, batch sizes) won't help.

### 5. Debug with First Principles

The breakthrough came from asking: "What could possibly cause two identical forward passes to give different results?"

Not:
- "What hyperparameters should I tune?"
- "Is my learning rate too high?"
- "Is my batch size wrong?"

But:
- "What sources of randomness exist in a neural network forward pass?"
- "Is the model in training or eval mode?"
- "Are there any stochastic layers?"

### 6. Read the PyTorch Documentation on train() and eval()

From PyTorch docs:
> "This has any effect only on certain modules. See documentations of particular modules for details of their behaviors in training/evaluation mode, if they are affected, e.g. Dropout, BatchNorm, etc."

Key modules affected:
- `nn.Dropout` - zeros random neurons in training mode
- `nn.BatchNorm*` - uses running stats in eval mode
- `nn.LayerNorm` - NOT affected (no batch statistics)

---

## Code References

### Files Modified

| File | Change |
|------|--------|
| `simulation_code/reinflow_smolvla.py` | Added `train()` method override at line 273 |

### Key Functions Involved

| Function | File | Purpose |
|----------|------|---------|
| `compute_trajectory_log_probs_onpolicy()` | `reinflow_smolvla.py:496` | Computes log probs for trajectories |
| `compute_ppo_loss()` | `reinflow_smolvla.py:687` | PPO loss with KL calculation |
| `train_parallel()` | `train_reinflow.py:254` | Main parallel training loop |
| `setup_reinflow_policy()` | `reinflow_smolvla.py:806` | Policy initialization |

---

## Related Concepts

- **PPO (Proximal Policy Optimization)**: Policy gradient algorithm that constrains updates using clipped surrogate objective and KL divergence monitoring
- **Dropout**: Regularization that randomly zeros neurons during training
- **ReinFlow**: Flow matching approach for robot learning using denoising trajectories
- **SmolVLA**: Small Vision-Language-Action model (~500M params) from HuggingFace
- **KL Divergence**: Measure of difference between probability distributions
- **Eval vs Training Mode**: PyTorch modes affecting dropout, batchnorm, etc.

---

## Appendix: The Log Probability Formula

For ReinFlow, actions are generated through a denoising process. The log probability of a trajectory is:

```
log π(trajectory) = Σ_{k=0}^{K-1} log N(a^{k+1} | μ_k, σ_k²)
```

Where:
- `a^k` is the action at denoising step k
- `μ_k = a^k + Δt · v_θ(a^k, o, t_k)` is the predicted mean
- `σ_k = σ_θ(a^k, o, t_k)` is the predicted standard deviation
- `v_θ` is the velocity network (affected by dropout!)
- `σ_θ` is the noise network

The Gaussian log probability is:
```
log N(x | μ, σ²) = -½ [||x - μ||² / σ² + d·log(2π) + 2·log(σ)]
```

When dropout changes `v_θ` output, `μ_k` changes, and `||a^{k+1} - μ_k||²` can change dramatically, causing huge swings in log probability.
