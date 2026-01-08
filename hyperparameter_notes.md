# ReinFlow Training Hyperparameter Reference

This document catalogs all training, model, environment, and normalization parameters for ReinFlow-based VLA fine-tuning, organized by tunability and impact.

**Last Updated**: January 6, 2026  
**Reference**: [ReinFlow Paper](https://reinflow.github.io/) | [SmolVLA](https://huggingface.co/lerobot/smolvla_base)

---

## Table of Contents

1. [Tier 1: Critical Parameters](#tier-1-critical---frequently-tuned-parameters)
2. [Tier 2: Important Parameters](#tier-2-important---occasionally-tuned-parameters)
3. [Tier 3: Architecture Parameters](#tier-3-modelarchitecture---set-once-parameters)
4. [Tier 4: Environment Constants](#tier-4-environmentconstants---rarely-changed)
5. [Scaling Relationships](#scaling-relationships-smolvla-vs-reinflow-paper)

---

## Tier 1: Critical - Frequently Tuned Parameters

These parameters have the highest impact on training dynamics and are adjusted most often during experimentation.

### Learning Rates

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `policy_lr` | `3e-6` | `4.5e-5` | Increased from 1e-6 after addressing clip fraction issue. With clip_epsilon=0.15 protecting against large updates, higher LR accelerates learning. |
| `critic_lr` | `1e-4` | N/A | Critic learning rate can be higher than policy LR since it doesn't directly affect action distribution stability. Value function converges faster with higher LR. |

**Source**: `train_reinflow.py` (TrainingConfig)

### PPO Clipping

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `clip_epsilon` | `0.15` | `0.001-0.2` | Increased from 0.05 after observing 83% clip fraction. Higher-dimensional action spaces cause more ratio drift, requiring wider clip range. |
| `target_kl` | `0.1` | `0.01` | Scaled ~10x from paper because KL divergence naturally scales with action dimensionality. With 300 dims (vs ~48), KL values are ~6x larger. Early stopping threshold for PPO epochs. |

**Source**: `train_reinflow.py` (TrainingConfig)

### ReinFlow Noise Bounds

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `sigma_min` | `0.25` | `~0.05-0.08` | Minimum noise for exploration. Scaled as √(D_smolvla/D_paper) ≈ √(300/28) ≈ 3.3x from paper values. Ensures stable log probability computation. |
| `sigma_max` | `0.50` | `~0.14` | Maximum noise during early training. Same scaling factor as sigma_min. Higher values = more exploration but less precision. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Training Scale

| Parameter | Current Value | Default | Rationale |
|-----------|---------------|---------|-----------|
| `num_episodes` | `20000` | `20000` | Total training episodes. More episodes = better convergence but longer training time. Adjust based on task complexity and available compute. |
| `num_parallel_envs` | `1` | `1` | Number of parallel MuJoCo environments. Set to 8-16 on A100 GPUs for faster data collection. Sequential mode (1) is often faster on M1/CPU due to parallelization overhead. |

**Source**: `train_reinflow.py` (TrainingConfig)

---

## Tier 2: Important - Occasionally Tuned Parameters

Parameters you may adjust for specific experiments or hardware constraints.

### PPO Algorithm

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `num_ppo_epochs` | `5` | `10` | Reduced from 10 to prevent ratio drift. Each epoch makes old_log_probs more stale, increasing clipping. |
| `minibatch_size` | `8` | Varies | Mini-batch size for PPO updates. Smaller = more gradient updates per batch but noisier gradients. Adjust based on GPU memory. |
| `gae_lambda` | `0.95` | `0.95` | GAE (Generalized Advantage Estimation) lambda parameter. Controls bias-variance tradeoff. 0.95-0.99 is standard. Higher = less bias, more variance. |
| `gradient_accumulation_steps` | `15` | `15` | Paper uses 15 for visual tasks. Accumulates gradients over multiple mini-batches before optimizer step. Effective batch size = minibatch_size × gradient_accumulation_steps. |
| `value_clip_epsilon` | `0.2` | N/A | Clip range for value function updates. Set to 0 to disable clipping. Prevents large value function updates that can destabilize training. |

**Source**: `train_reinflow.py` (TrainingConfig)

### ReinFlow Specific

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `num_denoising_steps` | `1` | `4` | Number of denoising steps (K) in flow matching. Paper uses K=4 for most tasks. SmolVLA default was 10, but K=1 works well and is faster. Fewer steps = faster inference but potentially less precise actions. |
| `chunks_per_episode` | `3` | N/A | How many action chunks to execute per episode before policy update. Each chunk gets fresh observation. More chunks = longer episodes but more data per update. |
| `noise_decay_start` | `1.0` | `0.35` | Fraction of training to hold sigma_max constant before decay. Paper recommends 0.35, but set to 1.0 (no decay) for visual tasks per paper guidance. |
| `noise_decay_ratio` | `0.7` | `0.7` | Final sigma_max = sigma_max × noise_decay_ratio. Controls how much noise decreases by end of training. Lower = more precision in later training. |
| `entropy_coeff` | `0.0` | `0.0-0.01` | Entropy regularization coefficient (paper Section 4.4). Higher values encourage more exploration. Paper uses 0.0 for visual manipulation tasks. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Training Stability

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `grad_clip_norm` | `0.25` | N/A | Maximum gradient norm for clipping. Prevents exploding gradients. Lower values = more conservative updates but may slow learning. |
| `lr_warmup_iterations` | `10` | `10-25` | Number of iterations for linear LR warmup. Paper uses 10 for PickPlaceCan, 25 for NutAssemblySquare. Warmup prevents early training instability. |
| `critic_warmup_iters` | `30` | `2-5` | Number of critic-only updates before joint training (paper Appendix D.2). Ensures value estimates are reasonable before policy gradients are computed. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Reward and Discount

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `gamma` | `0.999` | `0.99` | Discount factor for future rewards. Higher values weight future rewards more heavily. 0.999 for longer-horizon tasks, 0.99 for shorter tasks. |
| `max_steps_per_episode` | `150` | N/A | Maximum physics steps per episode before forced termination. Prevents infinite episodes. Adjust based on task complexity. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Reward Shaping

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `contact_bonus` | `0.1` | N/A | Positive reward when gripper contacts block. Provides clear gradient signal for contact-seeking behavior. Value chosen to exceed distance penalty when close (~0.03) but not dominate when far. |
| `height_alignment_bonus` | `0.05` | N/A | Positive reward when gripper is above block and close horizontally (within 10cm horizontal, 2cm+ above). Encourages top-down grasping approach rather than sideways bumping. |
| `lift_threshold` | `0.08` | N/A | Block height (meters) for episode termination. 8cm ensures block is clearly lifted, not just nudged. |

**Reward Formula**:
```
reward = -distance + (height_alignment_bonus if aligned_above else 0) + (contact_bonus if touching else 0)
```

**Alignment Conditions**:
- `horizontal_dist < 0.1` (within 10cm horizontally)
- `height_above > 0.02` (at least 2cm above block)

**Component Ranges**:

| Component | Per-step Range | Per-episode (150 actions) |
|-----------|----------------|---------------------------|
| Distance penalty | -0.5 to 0.0 | -75 to 0 |
| Height alignment bonus | 0.0 or +0.05 | 0 to +7.5 |
| Contact bonus | 0.0 or +0.1 | 0 to +15 |
| **Net (no contact, no align)** | -0.5 to 0.0 | ~-30 |
| **Net (aligned, no contact)** | -0.45 to +0.05 | ~-20 |
| **Net (with contact)** | -0.4 to +0.15 | ~+15 |

**Source**: `so101_mujoco_utils.py` (compute_reward), `train_reinflow.py` (TrainingConfig)

---

## Tier 3: Model/Architecture - Set Once Parameters

Parameters that define the model architecture or are inherited from pretrained models. These are typically set once and rarely changed.

### Trainable Components

| Parameter | Current Value | Rationale |
|-----------|---------------|-----------|
| `train_action_head` | `True` | Train the action_out_proj (velocity head). Essential for adapting pretrained policy to new task. |
| `train_time_mlp` | `True` | Train the time embedding MLP. Helps adapt flow matching to new action distributions. |
| `train_full_expert` | `True` | Train entire Action Expert (~100M params). More expressive but requires more compute. Set False to only train output heads. |
| `train_noise_head` | `True` | Train noise_mlp (σ_θ' network). Always True for ReinFlow since this is the core innovation enabling RL fine-tuning. |
| `train_critic` | `True` | Train critic network for actor-critic. Required for PPO-style training with value function baseline. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Critic Network Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `input_size` | `960` | Size of input features from VLM hidden state. Must match SmolVLA's VLM output dimension (text_config.hidden_size). |
| `hidden_size` | `512` | Size of hidden layers in critic MLP. Smaller than input for parameter efficiency. |
| `architecture` | `Linear(960→512) → ReLU → Linear(512→256) → ReLU → Linear(256→1)` | Simple MLP with two hidden layers. Outputs scalar value estimate. |
| `total_params` | `~620K` | Lightweight compared to 450M policy. Fast to train and doesn't bottleneck. |

**Source**: `reinflow_smolvla.py` (ReinFlowCritic class)

### Noise Network Architecture (σ_θ')

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `architecture` | `Linear(expert_hidden→256) → SiLU → Linear(256→128) → SiLU → Linear(128→max_action_dim)` | Shares features with velocity head. SiLU activation for smooth gradients. |
| `output_bounding` | `σ_min + (σ_max - σ_min) × (tanh(raw) + 1) / 2` | Tanh bounding ensures differentiable mapping to [σ_min, σ_max] range. |

**Source**: `pi0_adapter.py`, `reinflow_smolvla.py` (noise_mlp)

### Model Selection

| Parameter | SmolVLA Value | Pi0 Value | Rationale |
|-----------|---------------|-----------|-----------|
| `model_type` | `"smolvla"` | `"pi0"` | SmolVLA (450M params) is default for fast training. Pi0 (3.3B params) requires more memory but may be more capable. |
| `pretrained_path` | `"lerobot/smolvla_base"` | `"lerobot/pi0"` | HuggingFace model paths. Can also use local checkpoints. |
| `pi0_gradient_checkpointing` | N/A | `True` | Required for Pi0 to fit in 24GB GPU memory. Trades compute for memory by recomputing activations during backward pass. |

**Source**: `train_reinflow.py` (TrainingConfig)

### SmolVLA Fixed Architecture (Inherited from Pretrained)

| Parameter | Value | Notes |
|-----------|-------|-------|
| `chunk_size` | `50` | Number of actions per output chunk. Fixed by pretrained model. This is 6-12x larger than typical RL settings (4-8), which affects hyperparameter scaling. |
| `action_dim` | `6` | Degrees of freedom: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper. |
| `total_action_dims` | `300` | chunk_size × action_dim = 50 × 6. This is the dimensionality that affects log probability scaling. |
| `max_action_dim` | `7` | Maximum action dimension (padded). Used internally for batching. |
| `vlm_hidden_size` | `960` | SmolVLM2-500M hidden dimension. Used for critic input projection. |
| `expert_hidden_size` | `720` | Action Expert hidden dimension (vlm_hidden_size × expert_width_multiplier). Used for noise MLP input. |

**Source**: Inherited from `lerobot/smolvla_base` pretrained model

---

## Tier 4: Environment/Constants - Rarely Changed

Constants and environment settings that define the physical simulation. These are derived from the robot/task and should rarely be modified.

### SmolVLA Normalization Statistics

These values are the mean/std from SO-100 training data, used for z-score normalization. **Do not change** unless retraining SmolVLA from scratch.

| Parameter | Value | Units | Notes |
|-----------|-------|-------|-------|
| `SMOLVLA_STATE_MEAN` | `[1.596, 119.944, 109.770, 56.706, -27.423, 12.003]` | degrees | Mean joint positions in physical robot frame |
| `SMOLVLA_STATE_STD` | `[26.392, 52.411, 49.854, 36.998, 59.360, 19.040]` | degrees | Std dev of joint positions |
| `SMOLVLA_ACTION_MEAN` | `[1.596, 119.944, 109.770, 56.706, -27.423, 12.003]` | degrees | Same as state (position control) |
| `SMOLVLA_ACTION_STD` | `[26.392, 52.411, 49.854, 36.998, 59.360, 19.040]` | degrees | Same as state |

**Normalization Formula**: `normalized = (value - mean) / std`

**Source**: `so101_mujoco_utils.py`

### MuJoCo to Physical Robot Coordinate Offsets

These offsets convert MuJoCo's calibrated coordinates (where calibration pose = 0°) to SmolVLA's absolute servo frame.

| Joint | Offset (degrees) | Explanation |
|-------|------------------|-------------|
| `shoulder_pan` | `0.0` | Near zero offset (training mean ≈ 1.6°) |
| `shoulder_lift` | `120.0` | Calibration pose ≈ 120° in absolute frame |
| `elbow_flex` | `110.0` | Calibration pose ≈ 110° in absolute frame |
| `wrist_flex` | `57.0` | Calibration pose ≈ 57° in absolute frame |
| `wrist_roll` | `-27.0` | Calibration pose ≈ -27° in absolute frame |
| `gripper` | `12.0` | Calibration pose ≈ 12° in absolute frame |

**Formula**: `Physical Robot Position = MuJoCo Position (rad→deg) + OFFSET`

**Source**: `so101_mujoco_utils.py`

### Robot Starting Position

Default joint positions for episode reset (in degrees, MuJoCo frame):

| Joint | Value (degrees) | Notes |
|-------|-----------------|-------|
| `shoulder_pan` | `0.06` | Arm pointing forward |
| `shoulder_lift` | `-100.21` | Arm raised |
| `elbow_flex` | `89.95` | Elbow bent ~90° |
| `wrist_flex` | `66.46` | Wrist angled down toward table |
| `wrist_roll` | `5.96` | Minimal roll |
| `gripper` | `1.0` | Gripper open |

**Source**: `train_reinflow.py` (TrainingConfig.starting_position)

### Environment Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `model_path` | `'model/scene.xml'` | Path to MuJoCo MJCF scene file containing robot and objects. |
| `instruction` | `"pick up the block"` | Natural language task instruction passed to VLA model. |
| `steps_per_action` | `10` | Number of physics simulation steps per policy action. Higher = smoother but slower execution. |
| `block_pos` | `(0, 0.3, 0.0125)` | Initial (x, y, z) position of the red block in meters. y=0.3 is directly in front of robot. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Camera Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `image_size` | `256 × 256` | Resolution for all camera observations. SmolVLA expects this size. |
| `camera_up` | Top-down view | Primary observation camera, mounted above workspace. |
| `wrist_camera` | End-effector view | Mounted on robot wrist for close-up manipulation view. |
| `camera_side` | Side view | Third camera for additional perspective (SmolVLA uses 3 cameras). |

**Source**: `train_reinflow.py`, camera names in MJCF

### Joint Limits (from MJCF)

| Joint | Min (rad) | Max (rad) | Min (deg) | Max (deg) |
|-------|-----------|-----------|-----------|-----------|
| `shoulder_pan` | `-1.920` | `1.920` | `-110°` | `110°` |
| `shoulder_lift` | `-1.745` | `1.745` | `-100°` | `100°` |
| `elbow_flex` | `-1.690` | `1.690` | `-97°` | `97°` |
| `wrist_flex` | `-1.658` | `1.658` | `-95°` | `95°` |
| `wrist_roll` | `-2.744` | `2.841` | `-157°` | `163°` |
| `gripper` | `-0.175` | `1.745` | `-10°` | `100°` |

**Source**: `so101_gym_env.py`

### Logging and Checkpointing

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `log_interval` | `1` | Log metrics every N episodes. Set to 1 for detailed tracking. |
| `save_interval` | `10` | Save checkpoint every N episodes. Balance between safety and disk I/O. |
| `wandb_project` | `"reinflow-smolvla"` | Weights & Biases project name for experiment tracking. |
| `wandb_enabled` | `True` | Enable W&B logging. Set False for offline training or debugging. |
| `checkpoint_path` | `"reinflow_checkpoint.pt"` | Path for saving/loading training checkpoints. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Rendering

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `render` | `False` | Disable MuJoCo viewer for faster training. Set True for debugging. |
| `use_subproc_env` | `False` | Use subprocess-based parallel environments. Required for true parallelism but adds overhead. |

**Source**: `train_reinflow.py` (TrainingConfig)

---

## Scaling Relationships: SmolVLA vs ReinFlow Paper

SmolVLA outputs action chunks of size 50 (vs typical 4-8 in the ReinFlow paper). This 6-12x increase in action dimensionality affects several quantities that must be scaled accordingly.

### Dimensionality Comparison

| Metric | ReinFlow Paper | SmolVLA | Ratio |
|--------|----------------|---------|-------|
| Chunk size | 4-8 | 50 | ~6-12x |
| Action dim | 6-7 | 6 | 1x |
| Total dims (D) | 24-56 (~28 typical) | 300 | ~6-10x |

### 1. Log Probability Scale

**Formula**: `log p = Σ(-0.5 × diff²/σ²)` summed over all dimensions

**Scaling**: With 6x more dimensions, log probs are ~6x more negative.

| Setting | Typical log prob |
|---------|------------------|
| Paper (D≈28) | `-50 to -200` |
| SmolVLA (D=300) | `-300 to -1200` |

**Impact**: This is expected behavior, not a bug. Larger negative log probs don't affect the algorithm since PPO uses ratios.

### 2. KL Divergence Scale

**Formula**: `KL ≈ (r - 1) - log(r)` where `r = exp(new_log_prob - old_log_prob)`

**Scaling**: KL is computed from log prob differences. With larger-magnitude log probs, small percentage changes create larger absolute differences.

| Parameter | Paper Value | SmolVLA Value | Scaling Factor |
|-----------|-------------|---------------|----------------|
| `target_kl` | 0.01 | 0.1 | ~10x |

**Rationale**: We scale target_kl ~10x to account for the naturally larger KL values with higher dimensionality.

### 3. Policy Learning Rate Scale

**Formula**: Gradients accumulate over all output dimensions.

**Scaling**: With 6x more dimensions, gradient magnitude is ~6x larger. Additionally, larger log probs create larger gradient signals.

| Parameter | Paper Value | SmolVLA Value | Scaling Factor |
|-----------|-------------|---------------|----------------|
| `policy_lr` | 4.5e-5 | 1e-6 | ~50x reduction |

**Rationale**: Reduced ~50x for training stability. The 50x factor accounts for the ~6x dimension increase.

### 4. Sigma (Noise) Bounds Scale

**Formula**: For stable log probabilities, sigma must scale with dimensionality.

**Scaling**: `σ_smolvla = σ_paper × √(D_smolvla / D_paper) ≈ σ_paper × √(300/28) ≈ σ_paper × 3.3`

| Parameter | Paper Value | SmolVLA Value | Scaling Factor |
|-----------|-------------|---------------|----------------|
| `sigma_min` | 0.05-0.08 | 0.25 | ~3.3x |
| `sigma_max` | 0.14 | 0.50 | ~3.3x |

**Rationale**: Sigma scales as √D to maintain similar log probability magnitudes per denoising step.

### 5. Parameters That DON'T Need Scaling

These parameters are **scale-invariant** and can use paper values directly:

| Parameter | Why No Scaling |
|-----------|----------------|
| `gae_lambda` | Reward-based, independent of action dimensions |
| `gamma` | Reward-based discount factor |
| `gradient_accumulation_steps` | Batch size multiplier |

**Note**: `clip_epsilon` and `num_ppo_epochs` were previously listed here but actually DO need adjustment for high-dimensional action spaces due to ratio drift.

---

## Quick Reference: Current Values

```python
# Tier 1: Critical
policy_lr = 3e-6
critic_lr = 1e-4
clip_epsilon = 0.15
target_kl = 0.1
sigma_min = 0.25
sigma_max = 0.50
num_episodes = 20000
num_parallel_envs = 1
num_ppo_epochs = 5

# Tier 2: Important
minibatch_size = 8
gae_lambda = 0.95
gradient_accumulation_steps = 15
num_denoising_steps = 1
chunks_per_episode = 3
gamma = 0.999
grad_clip_norm = 0.25

# Tier 3: Architecture
train_full_expert = True
model_type = "smolvla"
chunk_size = 50  # fixed
action_dim = 6   # fixed

# Tier 4: Environment
lift_threshold = 0.08
contact_bonus = 0.1
height_alignment_bonus = 0.05
steps_per_action = 10
image_size = 256
```

---

## Changelog

> **Note for AI assistants**: When updating hyperparameters, always document *why* the change was made (e.g., "after 17k episodes showed no learning") in addition to the new value. This paper trail helps track what was tried, what worked/didn't work, and prevents repeating failed experiments. Good documentation accelerates learning.

| Date | Changes |
|------|---------|
| 2026-01-08 | Added `height_alignment_bonus = 0.05` reward shaping parameter; encourages top-down grasping approach by rewarding gripper being above block when close horizontally |
| 2026-01-08 | Updated clip_epsilon (0.05→0.15), policy_lr (1e-6→3e-6), num_ppo_epochs (10→5) to address 83% clip fraction observed in 2.3k episode run |
| 2026-01-07 | Added `contact_bonus = 0.1` reward shaping parameter; added Reward Shaping documentation section with formula and component ranges |
| 2026-01-07 | Increased `policy_lr` from 5e-7 to 1e-6 after 17k episodes showed no reward improvement (wandb run 71) |
| 2026-01-06 | Initial comprehensive documentation |

---

## References

1. [ReinFlow Paper](https://reinflow.github.io/) - Table 7b (visual manipulation hyperparameters)
2. [ReinFlow Paper](https://reinflow.github.io/) - Appendix D (noise decay schedule)
3. [SmolVLA HuggingFace](https://huggingface.co/lerobot/smolvla_base) - Model architecture
4. `train_reinflow.py` (TrainingConfig class) - Detailed hyperparameter comments
5. `notes/kl-divergence-bug-fix.md` - Dropout/eval mode debugging notes


