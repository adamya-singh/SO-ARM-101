# Debugging the Trajectory-Identity Bug in Parallel ReinFlow GAE

**Date:** April 4, 2026  
**Issue:** Parallel PPO/GAE bootstrapping mixed trajectories across environments, corrupting value targets and advantages  
**Time to Debug:** ~1 hour of trainer/code-path analysis and rollout indexing checks

---

## The Problem

### Simple English Version

In parallel training, each MuJoCo environment is its own story.

If environment 0 reaches toward the block and environment 1 misses it, PPO must estimate returns for each story using that same story's next state.

The old parallel trainer broke that rule. It flattened chunk samples into one global list, then built `next_values` by shifting that list forward by one slot. That meant one environment's chunk could bootstrap from another environment's next chunk.

So instead of:

```text
env0 chunk0 -> env0 chunk1
env1 chunk0 -> env1 chunk1
```

the code could do:

```text
env0 chunk0 -> env1 chunk0
env1 chunk0 -> env0 chunk1
```

That destroys trajectory identity.

### Where It Happened Conceptually

The old logic collected samples in chunk-major flattened order:

```python
all_trajectories.append(trajectory)
all_rewards.append(reward)
all_dones.append(done)
all_observations[key].append(obs[key])
```

and then later constructed bootstrap targets with a global shift:

```python
all_next_values = torch.zeros_like(all_values)
all_next_values[:-1] = all_values[1:]
all_next_values = all_next_values * (1.0 - all_dones)
```

That shift assumes adjacent flattened samples belong to the same trajectory.

In parallel chunked rollouts, they do not.

### Tiny 2-Environment / 2-Chunk Example

Assume the rollout buffer was appended in this order:

```text
index 0 = env0 chunk0
index 1 = env1 chunk0
index 2 = env0 chunk1
index 3 = env1 chunk1
```

With the old shift:

```text
next_value(index 0) = value(index 1) = value(env1 chunk0)   <-- wrong
next_value(index 1) = value(index 2) = value(env0 chunk1)   <-- wrong
next_value(index 2) = value(index 3) = value(env1 chunk1)   <-- wrong
```

The correct bootstrap mapping is:

```text
next_value(env0 chunk0) = value(env0 chunk1)
next_value(env1 chunk0) = value(env1 chunk1)
next_value(env0 chunk1) = 0 or same-env post-rollout bootstrap
next_value(env1 chunk1) = 0 or same-env post-rollout bootstrap
```

This is not a small indexing detail. It changes the RL target itself.

---

## Task

1. Identify whether parallel PPO was preserving per-environment trajectory structure
2. Determine whether GAE and returns were being computed on real trajectories or on a mixed flattened list
3. Refactor the rollout path so `next_value`, GAE, and returns are computed per environment
4. Add assertions that would catch this failure mode if it ever reappears

---

## Action

### Step 1: Trace the Parallel Rollout Data Shape

The first step was to inspect how `train_parallel()` stored rollouts.

The critical finding was that trajectory data was accumulated into flat Python lists during rollout collection rather than being kept in a `[env, chunk]` structure through the value/GAE stage.

That made it possible to preserve sample count while silently losing trajectory identity.

### Step 2: Inspect Bootstrap and GAE Construction

The next step was to inspect how value targets were built after rollout collection.

The old code effectively treated the entire flattened batch as one long sequence:

```text
sample 0 -> sample 1 -> sample 2 -> sample 3 -> ...
```

But PPO/GAE needed:

```text
env0: chunk0 -> chunk1 -> chunk2
env1: chunk0 -> chunk1
env2: chunk0 -> terminal
```

This mismatch meant:

- critic targets used the wrong future
- TD errors were wrong
- GAE recursion crossed environment boundaries
- PPO advantages were computed from corrupted returns

### Step 3: Restructure the Buffer Around Real Trajectories

The fix was to keep rollout data env-major until after value targets are complete.

The trainer now stores:

- trajectories as `[env][chunk]`
- observations as `[env][chunk]`
- rewards as `[num_envs, max_chunks]`
- dones as `[num_envs, max_chunks]`
- valid mask as `[num_envs, max_chunks]`

If an environment ends early, its later chunk slots stay invalid instead of being compacted away.

That preserves the real rollout topology.

### Step 4: Build Next Values Only From the Same Environment

After value inference on all valid observations, values are reshaped back to `[env, chunk]`.

Then bootstrap targets are built with only three legal cases:

1. If `(env, t)` is terminal, `next_value = 0`
2. If `(env, t+1)` is valid, `next_value = value(env, t+1)`
3. If rollout truncation ended collection while the env was still alive, bootstrap from that same env's post-rollout observation

Critically, there is no legal path where `(env_a, t)` bootstraps from `(env_b, t')`.

### Step 5: Compute GAE Per Environment, Then Flatten

GAE is now run independently on each environment's valid chunk sequence.

Only after:

- values are aligned per env
- next values are aligned per env
- returns are computed per env
- advantages are computed per env

do we flatten valid `(env, chunk)` entries into PPO minibatches.

That restores the correct semantics:

one PPO sample still equals one full action chunk, but now its advantage belongs to the right trajectory.

### Step 6: Add Invariants So the Bug Cannot Hide

The parallel trainer now checks invariants that directly target this bug class:

- no valid slot may appear after the first invalid slot within an environment
- if a chunk is terminal, later slots in that env must be invalid
- flattened sample count must equal `valid.sum()`
- a chunk's bootstrap source must be either the same env's next chunk or the same env's post-rollout observation

These are lightweight checks, but they encode the actual algorithmic contract.

---

## Result

### Correct Semantics Restored

The parallel trainer now computes PPO targets on actual per-environment trajectories instead of on a mixed global sequence.

That means:

- critic targets correspond to the right future
- GAE no longer leaks across environments
- advantages reflect same-env returns
- flattening is only a minibatch packaging step, not part of trajectory semantics

### Verification Performed

The fix was checked with targeted regression scenarios:

1. **Synthetic indexing regression**
   - Built a tiny 2-env / 2-chunk case with distinct values
   - Confirmed `next_value(env0, chunk0)` uses `value(env0, chunk1)`, not `value(env1, chunk0)`

2. **Early termination handling**
   - Confirmed an env that terminates after chunk 0 gets zero bootstrap and no later valid slots

3. **Truncated-but-not-done bootstrap**
   - Confirmed the final collected chunk can bootstrap from the same env's post-rollout observation when rollout truncates without termination

4. **`num_envs=1` sanity alignment**
   - Confirmed the parallel logic reduces cleanly to sequential semantics when only one environment is active

The implementation also passed Python compilation after the trainer refactor.

---

## Key Lessons Learned

### 1. Trajectory Identity Is Part of the RL Algorithm

For chunked RL with parallel environments, trajectory structure is not a convenience-layer detail.

It is part of the mathematical definition of the target.

### 2. Flattening Too Early Can Silently Change the Method

It is easy to keep sample counts, tensor shapes, and PPO loops looking "correct" while actually changing what the algorithm computes.

That is especially dangerous in generative/chunked action settings, where a sample already represents a more complex object than a single low-dimensional action.

### 3. Valid Masks Matter as Much as Values

Once environments terminate at different times, invalid padded slots are unavoidable.

The correct question is not "can we flatten this?" but "what semantics are preserved before flattening?"

### 4. Assertions Should Encode the Intended Math

The most useful assertions here were not generic shape checks.

They were semantic checks:

- no resurrection after invalid
- no chunks after terminal
- no cross-env bootstrap

Those assertions defend the RL objective itself.

### 5. The Core Lesson

For chunked RL with parallel environments, trajectory identity is part of the algorithm, not a bookkeeping detail.
