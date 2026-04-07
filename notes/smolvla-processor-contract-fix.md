# SmolVLA Processor-Contract Hardening

**Date:** April 7, 2026  
**Issue:** Processor-backed SmolVLA artifacts were loaded successfully, but the simulation helper layer only partially honored the LeRobot processor contract

## Problem

Two separate failures appeared when running ReinFlow training from the processor-backed SmolVLA repo `adamyathegreat/so101_pickplace_v1_smolvla`:

1. **State normalization silently degraded to legacy stats**
   - The SmolVLA preprocessor path raised a `'task'` error
   - The helper code caught that exception and fell back to the old hardcoded normalization path
   - So training printed `processor-backed`, but the state path was not fully using the processor contract

2. **Action denormalization crashed**
   - The simulation helper called `postprocessor(action_tensor)` directly
   - For the loaded artifact, the pipeline expected the action-specific processor path rather than a generic tensor input to `__call__`
   - This produced `ValueError: EnvTransition must be a dictionary. Got Tensor`

## Root Cause

This was not a coordinate-frame rollback.

The April coordinate fix was still correct: processor-backed SmolVLA uses calibrated radians, while the legacy fallback uses hardcoded servo-frame stats plus `MUJOCO_TO_PHYSICAL_OFFSET`.

The real bug was that the repo’s simulation abstraction layer had drifted from the processor contract expected by LeRobot:

- processor-backed SmolVLA state normalization needed task text in the batch contract
- processor-backed action denormalization was safer through the action-specific API than through a raw `postprocessor(tensor)` assumption

## Fix

The simulation helper layer was hardened rather than bypassed:

- added a shared action postprocessor helper in `simulation_code/so101_mujoco_utils.py`
- prefer `process_action(...)` when available, with direct `__call__` only as a compatibility fallback
- normalize postprocessor outputs to a consistent `(action_dim,)` shape
- require task text for processor-backed SmolVLA state normalization
- thread instruction text through the vectorized and single-env normalization paths

## Intended Invariant

After this fix:

- **Processor-backed SmolVLA path**
  - input state is calibrated MuJoCo radians
  - task text is present in the preprocessor batch
  - the processor handles model normalization / denormalization
  - simulation code does not silently fall back unless processor invocation actually fails

- **Legacy SmolVLA fallback**
  - hardcoded SmolVLA stats and `MUJOCO_TO_PHYSICAL_OFFSET` remain available
  - this path is fallback-only and should not override a working processor-backed artifact

## Why This Note Exists

This bug is related to SmolVLA normalization, but it is different from the original coordinate-frame mismatch documented in `notes/smolvla-coordinate-fix.md`.

That earlier note explains **which coordinate frame is correct**.

This note explains **how the processor-backed implementation contract was accidentally violated even after the frame choice was correct**.
