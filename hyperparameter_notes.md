# ReinFlow Training Hyperparameter Reference

This document catalogs all training, model, environment, and normalization parameters for ReinFlow-based VLA fine-tuning, organized by tunability and impact.

**Last Updated**: April 8, 2026  
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
| `policy_lr` | `3e-7` | `4.5e-5` | Reduced again in April 2026 after PPO correctness was fixed but real post-update KL was still too high. The current default prioritizes reward growth through smaller actor steps over maximum update aggressiveness. |
| `critic_lr` | `1e-4` | N/A | Critic learning rate can be higher than policy LR since it doesn't directly affect action distribution stability. Value function converges faster with higher LR. |

**Source**: `train_reinflow.py` (TrainingConfig)

### PPO Clipping

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `clip_epsilon` | `0.05` | `0.001-0.2` | Reverted to 0.05 for stability. 0.15 removed PPO's protection against large updates, leading to KL explosion at ~4.5k episodes. 83% clip fraction was actually correct behavior. |
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
| `num_parallel_envs` | `1` | `1` | Number of parallel MuJoCo environments. On a 14.6 GB GPU, SmolVLA is currently treated as a `<= 5` env workload. Larger values may still fit on larger GPUs, but `5` is the practical ceiling for the current single-GPU research setup. |

**Source**: `train_reinflow.py` (TrainingConfig)

---

## Tier 2: Important - Occasionally Tuned Parameters

Parameters you may adjust for specific experiments or hardware constraints.

### PPO Algorithm

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `num_ppo_epochs` | `2` | `10` | Reduced again after the April 2026 PPO correctness fix. In this setup, useful updates usually happen in epoch 1; extra epochs mostly increase stale-log-prob drift and trigger post-update KL early-stop. |
| `minibatch_size` | `8` | Varies | Mini-batch size for PPO updates. Smaller = more gradient updates per batch but noisier gradients. Adjust based on GPU memory. |
| `gae_lambda` | `0.95` | `0.95` | GAE (Generalized Advantage Estimation) lambda parameter. Controls bias-variance tradeoff. 0.95-0.99 is standard. Higher = less bias, more variance. |
| `gradient_accumulation_steps` | `15` | `15` | Paper uses 15 for visual tasks. Accumulates gradients over multiple mini-batches before optimizer step. Effective batch size = minibatch_size × gradient_accumulation_steps. |
| `value_clip_epsilon` | `0.2` | N/A | Clip range for value function updates. Set to 0 to disable clipping. Prevents large value function updates that can destabilize training. |
| `logprob_eval_microbatch_size` | `1` | N/A | Forces deterministic trajectory log-prob evaluation independent of batch composition. This is a correctness guard, not a throughput optimization. The value `1` is intentionally conservative. |

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
| `critic_backprop_into_policy` | `False` | N/A | Detaches critic input features from the shared actor conditioning path. This prevents critic loss from indirectly moving trainable policy conditioning components such as `state_proj`. |

**Source**: `train_reinflow.py` (TrainingConfig)

### PPO Correctness / Diagnostic Invariants

These are not ordinary tuning knobs. They are runtime semantics added in April 2026 after discovering that KL could spike before any real optimizer step.

| Metric / Behavior | Current Meaning | Why it exists |
|-------------------|-----------------|---------------|
| `debug/pre_update_kl` | KL between cached `old_log_probs` and an immediate recomputation with identical weights | Must be ~0 before the first optimizer step. If not, log-prob evaluation is inconsistent. |
| `debug/pre_update_ratio_mean` | Mean PPO ratio from the same no-update recomputation | Must be ~1 before the first optimizer step. |
| `debug/pre_update_logprob_abs_mean` | Mean absolute drift in log-prob before any update | Helps diagnose subtle nondeterminism or batch-composition dependence. |
| `debug/pre_update_logprob_abs_max` | Max absolute drift in log-prob before any update | Highlights worst-case sample instability. |
| `training/post_update_kl` | KL after an actual optimizer step | This is the KL that now drives PPO early-stop. |
| `training/post_update_ratio_max` | Worst PPO ratio on the minibatch after the optimizer step | Useful for detecting true actor overshoot after a real update. |
| `training/post_update_clip_fraction` | Clip fraction recomputed after the optimizer step | Preferred over the pre-step PPO loss metric when judging update aggressiveness. |
| `updates/actor_delta_l2` | Estimated actor parameter update magnitude per optimizer step | Tracks actual actor movement instead of only gradient size. |
| `updates/actor_delta_max_abs` | Maximum absolute actor parameter update on a step | Highlights rare but dangerous large parameter jumps. |

**Important**:
- The trainer now treats **pre-update** old/new log-prob agreement as an invariant.
- PPO early-stop should be interpreted using `training/post_update_kl`, not the pre-update diagnostics.
- For reward-focused debugging, use `training/post_update_kl`, `training/post_update_ratio_max`, `training/post_update_clip_fraction`, and `reward/ema20` together.

### Reward and Discount

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `gamma` | `0.999` | `0.99` | Discount factor for future rewards. Higher values weight future rewards more heavily. 0.999 for longer-horizon tasks, 0.99 for shorter tasks. |
| `max_steps_per_episode` | `150` | N/A | Maximum physics steps per episode before forced termination. Prevents infinite episodes. Adjust based on task complexity. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Reward Shaping

| Parameter | Current Value | Paper Value | Rationale |
|-----------|---------------|-------------|-----------|
| `distance_penalty_scale` | `0.4` | N/A | The base distance term is now scaled down from raw `-distance` so progress rewards can dominate when the arm moves correctly. |
| `horizontal_progress_scale` | `0.12` | N/A | Rewards actual horizontal improvement toward the block, not just static proximity. |
| `vertical_approach_scale` | `0.05` | N/A | Rewards entering and improving within the grasp-height corridor before contact. |
| `approach_closeness_scale` | `0.035` | N/A | Static support term for being near the block before contact. Smaller than the progress terms. |
| `alignment_reward_cap` | `0.035` | N/A | Capped alignment reward inside the widened pre-grasp corridor. Higher than the first anti-hover pass to recover useful approach shaping, but still capped to prevent the old hover exploit. |
| `near_contact_bonus` | `0.03` | N/A | Dense reward in the final pre-contact corridor. This is the bridge between “near the block” and actual first touch. |
| `contact_entry_bonus` | `0.18` (+`0.02` when aligned/close) | N/A | One-time bonus when contact begins. Rewards crossing the touch transition rather than only sitting near the block. |
| `contact_persistence_reward` | `0.045` per step | N/A | Small per-step contact reward. Encourages staying in contact without letting contact dominate the return. |
| `sustained_contact_threshold` | `5` | N/A | Number of consecutive contact steps before the sustained-contact state is considered active. |
| `sustained_contact_bonus` | `0.2` configured, applied as `min(0.06, sustained_contact_bonus * 0.25)` | N/A | Sustained contact now exists as a capped continuation reward rather than the old large additive term. |
| `hover_stall_threshold` | `8` | N/A | The hover penalty now waits longer before firing so the anti-hover logic does not starve legitimate pre-contact exploration. |
| `hover_penalty` | `-0.01` after hover stall | N/A | Penalizes genuine stalling in the grasp corridor without contact. Reduced from the first anti-hover pass to avoid over-correction. |
| `bilateral_grasp_bonus` | `0.30` | N/A | Stronger than the older grasp reward. Activates when both gripper sides are meaningfully squeezing the block. |
| `grasp_persistence_reward` | `0.08` per step | N/A | Rewards keeping the grasp instead of only achieving the initial squeeze. |
| `lift_progress_reward` | `min(0.4, 5.0 * block_height_gain)` | N/A | Dense lift reward based on block height above the initial pose. Replaced the old binary-only lift shaping. |
| `lift_bonus` | `max(lift_bonus, 0.25)` once lifted above threshold | N/A | Completion bonus once the block is clearly lifted above `lift_bonus_threshold`. |
| `lift_bonus_threshold` | `0.04` | N/A | Height threshold for the lift completion bonus. Still below terminal success so partial lifts are rewarded before episode end. |
| `lift_threshold` | `0.08` | N/A | Terminal success threshold for ending the episode. |
| `slip_penalty_contact` | `-0.03` | N/A | Applied only after losing sustained contact, not after every exploratory touch. This keeps first-touch exploration cheap. |
| `slip_penalty_grasp` | `-0.08` | N/A | Applied after losing a real grasp. Larger than contact-loss penalty because grasp regression is a more meaningful failure. |
| `block_displacement_penalty_scale` | `0.08` | N/A | Penalizes knocking the block sideways without meaningful lift or hold. Disabled once the block is clearly being lifted or held. |

**Current reward structure**:
```text
reward = -distance_penalty_scale * distance
       + approach_reward
       + alignment_reward
       + near_contact_bonus
       + contact_entry_bonus
       + contact_persistence_reward
       + sustained_contact_continuation
        + bilateral_grasp_bonus
        + grasp_persistence_reward
       + lift_progress_reward
       + lift_completion_bonus
       - hover_penalty
       - slip_penalty
       - block_displacement_penalty
```

**Important semantic changes**:
- The older additive reward should now be treated as historical context, not the current implementation.
- The first April anti-hover redesign is also historical context. The current reward is a second-pass hybrid that restores approach/contact shaping without reverting to the old local optimum.
- Alignment is still gated and capped so hovering cannot dominate the return, but the corridor is now wider and the reward is paired with explicit progress terms.
- Lift reward now uses **height gain above the initial block pose**, not only a binary threshold.
- Contact and grasp use both **transition** rewards and **persistence** rewards.
- The reward is now designed to push the behavior sequence `approach -> align -> near-contact -> touch -> hold -> squeeze -> lift`.

**Key diagnostics to interpret reward learning**:
- `reward/approach_reward_mean`
- `reward/alignment_reward_mean`
- `reward/near_contact_rate`
- `reward/contact_after_alignment_rate`
- `reward/horizontal_progress_mean`
- `reward/vertical_approach_mean`
- `reward/contact_entry_rate`
- `reward/grasp_persistence_rate`
- `reward/lift_progress_mean`
- `reward/hover_stall_rate`
- `reward/block_displacement_mean`
- slip metrics:
  - `reward/slip_count` in sequential runs
  - `reward/slip_count_total` and `reward/slip_count_avg` in parallel runs
  - `reward/contact_loss_count*`
  - `reward/grasp_loss_count*`

**How to interpret the current metrics**:
- Low `reward/approach_reward_mean`, `reward/horizontal_progress_mean`, or `reward/vertical_approach_mean` means the policy is not meaningfully reaching the block.
- High `reward/near_contact_rate` but low `reward/contact_entry_rate` means the last-touch bridge is still failing.
- Contact without persistence or grasp means the policy can touch but cannot yet hold or squeeze.
- Improved scalar reward without movement in these manipulation metrics should not be treated as real pickup progress.

**Source**: `so101_mujoco_utils.py` (compute_reward), `train_reinflow.py` (TrainingConfig)

---

## Tier 3: Model/Architecture - Set Once Parameters

Parameters that define the model architecture or are inherited from pretrained models. These are typically set once and rarely changed.

### Trainable Components

| Parameter | Current Value | Rationale |
|-----------|---------------|-----------|
| `train_action_head` | `True` | Train the action_out_proj (velocity head). Essential for adapting pretrained policy to new task. |
| `train_time_mlp` | `True` | Train the time embedding MLP. Helps adapt flow matching to new action distributions. |
| `train_full_expert` | `False` | Full-expert RL remains available, but it is no longer the default because the current objective is stable reward growth rather than maximum update capacity. |
| `trainable_scope` | `rl_stable_heads` | Default SmolVLA RL scope. Trains `action_in_proj`, `action_out_proj`, `action_time_mlp_in`, `action_time_mlp_out`, and `noise_mlp`, while leaving `state_proj` frozen. |
| `train_noise_head` | `True` | Train noise_mlp (σ_θ' network). Always True for ReinFlow since this is the core innovation enabling RL fine-tuning. |
| `train_critic` | `True` | Train critic network for actor-critic. Required for PPO-style training with value function baseline. |
| `critic_backprop_into_policy` | `False` | Critic still reads policy-derived features, but by default those features are detached before the critic head so value learning does not push the actor through shared conditioning paths. |

**Source**: `train_reinflow.py` (TrainingConfig)

### Critic Network Architecture

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `input_size` | `960` | Size of input features from VLM hidden state. Must match SmolVLA's VLM output dimension (text_config.hidden_size). |
| `hidden_size` | `512` | Size of hidden layers in critic MLP. Smaller than input for parameter efficiency. |
| `architecture` | `Linear(960→512) → ReLU → Linear(512→256) → ReLU → Linear(256→1)` | Simple MLP with two hidden layers. Outputs scalar value estimate. |
| `total_params` | `~620K` | Lightweight compared to 450M policy. Fast to train and doesn't bottleneck. |
| `feature_detach` | `Enabled by default` | Critic consumes pooled observation features, but these are detached before the critic MLP unless explicitly overridden. |

**Source**: `reinflow_smolvla.py` (ReinFlowCritic class)

### Warmup / Memory Notes

- Critic warmup actor sampling now runs under `torch.no_grad()`.
- This change was made after an April 2026 OOM on a 14.6 GB GPU with `--parallel-envs 10`.
- The practical runtime guidance for this repo is:
  - SmolVLA + ReinFlow + headless EGL + subprocess vectorization: start from `--parallel-envs 5`
  - only exceed that on hardware with clearly larger memory headroom

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
policy_lr = 3e-7
critic_lr = 1e-4
clip_epsilon = 0.05
target_kl = 0.1
sigma_min = 0.25
sigma_max = 0.50
num_episodes = 20000
num_parallel_envs = 1
num_ppo_epochs = 2

# Tier 2: Important
minibatch_size = 8
gae_lambda = 0.95
gradient_accumulation_steps = 15
num_denoising_steps = 1
chunks_per_episode = 3
gamma = 0.999
grad_clip_norm = 0.25

# Tier 3: Architecture
train_full_expert = False
trainable_scope = "rl_stable_heads"
model_type = "smolvla"
chunk_size = 50  # fixed
action_dim = 6   # fixed

# Tier 4: Environment
lift_threshold = 0.08
contact_bonus = 0.1
sustained_contact_threshold = 5
sustained_contact_bonus = 0.2
height_alignment_bonus = 0.05
grasp_bonus = 0.15
lift_bonus = 0.2
lift_bonus_threshold = 0.04
steps_per_action = 10
image_size = 256
```

---

## Changelog

> **Note for AI assistants**: When updating hyperparameters, always document *why* the change was made (e.g., "after 17k episodes showed no learning") in addition to the new value. This paper trail helps track what was tried, what worked/didn't work, and prevents repeating failed experiments. Good documentation accelerates learning.

| Date | Changes |
|------|---------|
| 2026-01-08 | REVERTED policy_lr (3e-6→1e-6) and clip_epsilon (0.15→0.05). Analysis showed pre-515b18d run had stable KL (0.01-0.03) for 740 eps with zero early stops. Post-commit runs collapsed at ~4.5k eps. The 83% clip fraction was PPO correctly constraining updates. |
| 2026-01-08 | REVERTED `recompute_old_log_probs` - the fix prevented cumulative learning across PPO epochs. 900-ep test showed worse rewards (-46 vs -8) and no grasps despite "healthier" KL metrics. Original staleness was a symptom of learning, not the root cause. |
| 2026-01-08 | Added `lift_bonus = 0.2` and `lift_bonus_threshold = 0.04` reward shaping; rewards when block is elevated above 4cm. Provides dense reward for lifting before episode terminates at 8cm. Reward progression: align → contact → grasp → lift |
| 2026-01-08 | Added sustained contact reward: `sustained_contact_threshold = 5` (frames before bonus), `sustained_contact_bonus = 0.2` (extra reward per step). Creates reward gradient: approach → touch → hold → grasp. Added wandb metrics: grasp_rate, grasp_count_avg, sustained_contact_rate |
| 2026-01-08 | Added `grasp_bonus = 0.15` reward shaping parameter; rewards when both sides of gripper squeeze block (bilateral force > 0.1N), bridging gap between contact and lift |
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
