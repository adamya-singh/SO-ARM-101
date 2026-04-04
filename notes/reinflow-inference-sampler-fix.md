# Debugging the ReinFlow Inference Sampler Mismatch

**Date:** April 4, 2026  
**Issue:** Training optimized the ReinFlow stochastic sampler, but inference evaluated plain base-model action selection  
**Time to Debug:** ~1 hour of code-path tracing across training, checkpoint loading, and inference

---

## The Problem

### Simple English Version

The training code and the inference code were not evaluating the same policy.

Training used the ReinFlow wrapper, which defines a stochastic denoising-based action sampler and a PPO-compatible log-probability path.

Inference, however, loaded the base model and called plain `select_action()`.

So the repo could train policy A and evaluate policy B.

That is not a minor script bug. In RL, the policy is not just the neural network weights. It is the weights plus the sampling procedure that turns observations into actions.

### The Old Mismatch

The training path used ReinFlow-specific setup and action sampling:

```python
rl_policy = setup_reinflow_policy(...)
action = rl_policy.select_action(observation)
```

or, for Pi0:

```python
rl_policy, preprocessor, postprocessor = setup_reinflow_pi0_policy(...)
action = rl_policy.select_action(observation)
```

But the old inference path manually loaded base-model components and then called plain policy inference:

```python
policy = SmolVLAPolicy.from_pretrained(...)
policy.model.action_out_proj.load_state_dict(...)
action = policy.select_action(observation)
```

That bypassed the ReinFlow wrapper's stochastic denoising semantics.

### Why This Was a Conceptual Failure

ReinFlow is not just "SmolVLA with a few updated weights."

It changes the action-generation process itself:

- adds a learned noise head
- defines a denoising trajectory
- samples actions through the ReinFlow wrapper
- computes PPO log-probabilities for that same stochastic process

If inference does not use that same sampler, then evaluation is not measuring the trained RL policy.

### Additional Inconsistency: Incomplete Checkpoint Semantics

The old SmolVLA checkpoint path also did not persist and restore ReinFlow sigma bounds.

That meant even if the right wrapper were loaded later, the sampler configuration could drift from what training actually used.

So there were really two mismatches:

1. wrong policy object at inference time
2. incomplete restoration of sampler configuration

---

## Task

1. Trace the exact policy object used during training
2. Trace the exact policy object and sampler used during inference
3. Remove any fallback path that silently evaluates plain base-model inference from a ReinFlow checkpoint script
4. Ensure checkpoint loading restores the full ReinFlow sampling semantics, including sigma bounds

---

## Action

### Step 1: Compare the Training and Inference Control Paths

The first step was to compare:

- the training setup path in the ReinFlow trainer
- the checkpoint save/load utilities
- the simulation inference script

That made the mismatch obvious:

- training went through `setup_reinflow_policy(...)` or `setup_reinflow_pi0_policy(...)`
- training called wrapper `select_action(...)`
- inference rebuilt a base policy manually and called base `select_action(...)`

So the policy-gradient training path and the deployment/evaluation path were not aligned.

### Step 2: Clarify the Intended Responsibility of the Script

The right design decision was to make `run_reinflow_inference.py` strict ReinFlow evaluation.

That means:

- if you want base pretrained inference, use `run_mujoco_simulation.py`
- if you want ReinFlow checkpoint evaluation, use `run_reinflow_inference.py`

The old "missing checkpoint -> silently fall back to base model" behavior was removed because it hid the exact bug this note is about.

### Step 3: Rebuild Inference Around the Wrapper

The inference script was refactored so it now:

- requires a checkpoint to exist
- auto-detects model type from checkpoint metadata when requested
- constructs the proper ReinFlow wrapper
- restores checkpoint state through wrapper loaders
- calls `rl_policy.select_action(observation)` directly

For SmolVLA:

```python
rl_policy = setup_reinflow_policy(...)
start_episode, _ = load_reinflow_checkpoint(rl_policy, checkpoint_path, device)
action = rl_policy.select_action(observation)
```

For Pi0:

```python
rl_policy, preprocessor, postprocessor = setup_reinflow_pi0_policy(...)
start_episode, _ = load_reinflow_pi0_checkpoint(rl_policy, checkpoint_path, device)
action = rl_policy.select_action(observation)
```

This makes the evaluation path match the object PPO actually trained.

### Step 4: Fix Observation Construction So It Matches Wrapper Expectations

The next issue was observation formatting.

SmolVLA ReinFlow inference now uses:

```python
prepare_observation_for_reinflow(...)
```

instead of the generic base-policy path.

Pi0 keeps the processor-based path, but it is now built around the ReinFlow wrapper's returned preprocessor/postprocessor and base policy handle.

That keeps tokenization and normalization aligned with the wrapper that actually owns the policy.

### Step 5: Restore SmolVLA Sigma Bounds From Checkpoint

SmolVLA ReinFlow checkpoints were extended to save:

- `model_type`
- `sigma_min`
- `sigma_max`

and the loader now restores those values when present.

For older checkpoints that do not contain sigma metadata, the loader falls back to the wrapper setup defaults and logs that fallback explicitly.

That makes checkpoint semantics auditable instead of implicit.

---

## Result

### The Inference Script Now Evaluates the Actual Trained Policy

`run_reinflow_inference.py` is now ReinFlow-only.

It does not pretend to evaluate a ReinFlow checkpoint while secretly running plain base-model action selection.

The inference path now consistently does all of the following:

- requires the checkpoint to exist
- loads the ReinFlow wrapper
- restores checkpoint weights through wrapper loaders
- restores SmolVLA sigma bounds when available
- builds wrapper-correct observations
- routes action selection through `rl_policy.select_action(...)`

### Verification Performed

The fix was validated with lightweight checks:

1. **Compile check**
   - `python -m py_compile simulation_code/run_reinflow_inference.py simulation_code/reinflow_smolvla.py`

2. **Startup-path audit**
   - startup logging now reports the saved checkpoint episode
   - startup logging now reports restored ReinFlow sigma bounds

3. **Sampler-path audit**
   - the inference script now reaches wrapper `select_action()`
   - the old base-policy fallback path was removed from this script

This does not prove task success in simulation.

It proves that evaluation now measures the same policy object and sampler semantics that PPO trained.

---

## Key Lessons Learned

### 1. In RL, the Policy Includes the Sampler

A policy is not only a set of weights.

It is the full mapping from observation to sampled action.

If training and inference use different samplers, they are different policies in the RL sense.

### 2. Checkpoint Loading Must Restore Behavior, Not Just Tensors

Restoring a few trainable layers is not enough if the runtime behavior depends on additional sampler configuration such as sigma bounds.

Behavioral equivalence requires restoring the full action-generation setup.

### 3. Silent Fallbacks Can Hide Conceptual Bugs

The old "checkpoint missing or awkward path -> run base policy anyway" behavior made it too easy to think ReinFlow evaluation was happening when it was not.

For research code, strictness is often more honest than convenience.

### 4. Observation Helpers Are Part of Policy Identity Too

If the wrapper expects a particular observation construction path, that path is part of the executable policy contract.

Tokenizer access, processor use, and normalization should all follow the wrapper, not an older base-model shortcut.

### 5. The Core Lesson

Evaluation must use the same policy object and sampling semantics that PPO trained.
