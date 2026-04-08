# Reward Redesign: Contact, Grasp, and Lift Progress

## Why the reward changed

By April 2026, PPO stability was no longer the main failure mode. W&B showed that the trainer could run with sane post-update KL and without immediate early-stop, but the policy was still not learning meaningful pickup behavior.

The key pattern was:

- `reward/height_align_rate` stayed high
- `reward/contact_rate` remained near zero
- `reward/grasp_rate` stayed at zero
- `reward/sustained_contact_rate` stayed at zero
- `reward/ema20` improved only slightly, then flattened

That combination meant the agent had found a local optimum: hover above the block in a geometrically "good" pose, collect alignment reward, and avoid the risk of actual contact.

## What was wrong with the older reward

The older reward was effectively additive:

```text
reward = -distance
       + height_alignment_bonus
       + contact_bonus
       + sustained_contact_bonus
       + grasp_bonus
       + lift_bonus
```

That shaping was useful early in the project, but it had two structural problems once PPO was stable enough to exploit it:

1. **Alignment was too easy to collect repeatedly.**  
   If the gripper could stay above the block without committing to contact, the policy could get paid for safe hovering.

2. **Contact and grasp were too sparse relative to alignment.**  
   The policy did not need to cross the risky transition from alignment into touch and squeeze to preserve a decent shaped return.

In practice, that meant the reward favored pre-grasp posture more than actual manipulation.

## New reward design

The first April reward redesign in [`simulation_code/so101_mujoco_utils.py`](/Users/adamyasingh/dev/SO-ARM-101/mujoco/SO-ARM-101/simulation_code/so101_mujoco_utils.py) was phase-aware:

1. **Approach reward**
   Dense shaping from horizontal closeness and vertical pre-grasp alignment, active only before contact.

2. **Gated alignment reward**
   Small capped reward that only applies when the gripper is close horizontally, near a grasp height, and still moving into the grasp instead of stalling.

3. **Contact entry and persistence**
   A one-time bonus when contact begins, followed by a smaller persistence reward while contact is maintained.

4. **Bilateral grasp and grasp persistence**
   A stronger reward when both gripper sides are meaningfully squeezing the block, plus an extra persistence term if the grasp continues.

5. **Lift progress and completion**
   Dense reward based on block height gain relative to the initial block pose, plus a completion bonus once the block is clearly lifted.

6. **Failure-mode penalties**
   Penalties for hover stalls, slips after contact/grasp, and horizontal block displacement without lift.

## Intended behavior shift

The redesigned reward is meant to push the policy through the full pickup sequence:

```text
approach -> align -> touch -> hold -> squeeze -> lift
```

The important change is that alignment is now a gateway, not a local optimum. The policy should not be able to collect most of its return by hovering above the block without touch.

## Second April 2026 setback

After the first anti-hover redesign, PPO stayed healthy, but the next live run revealed a new failure mode:

- the older stable run still had better `reward/ema20`
- the newer run had much weaker `reward/height_align_rate`
- `reward/positive_fraction` dropped to zero
- grasp remained flat

That meant the anti-hover fix had worked, but it had also removed too much of the pre-contact shaping. The policy was no longer getting paid enough for approach and alignment progress to reach contact reliably.

In other words:

- first failure mode: too much reward for hovering above the block
- second failure mode: too little reward for getting back to the block after removing the hover exploit

## Current reward rebalancing

The current reward is therefore a hybrid progress-aware pickup reward, not just the first anti-hover design.

It keeps the anti-hover structure, but adds back stronger pre-contact signals:

1. **Scaled distance penalty**
   The base penalty is now `-distance_penalty_scale * distance`, not raw `-distance`.

2. **Progress-aware approach shaping**
   Horizontal progress reward, vertical approach reward, and static approach-closeness shaping are active before contact.

3. **Wider gated alignment corridor**
   Alignment is still capped and controlled, but the corridor is wider than the first anti-hover version so the arm can earn useful pre-contact reward again.

4. **Near-contact bridge**
   A dense `near_contact_bonus` rewards the final pre-contact corridor, so the policy can learn the last step before touch.

5. **Less punitive exploratory contact**
   Contact-loss penalties now apply only after losing sustained contact. Grasp-loss penalties apply only after losing a real grasp. Single exploratory touches are no longer punished like failures.

6. **Same downstream manipulation phases**
   Contact-entry, contact persistence, bilateral grasp, grasp persistence, dense lift progress, lift completion, and block-displacement penalties remain part of the reward.

## Metrics to watch

For current runs, reward improvement should be interpreted together with:

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

These should be read alongside the PPO stability metrics:

- `debug/pre_update_kl`
- `training/post_update_kl`
- `training/post_update_ratio_max`
- `training/post_update_clip_fraction`
- `reward/ema20`

Interpretation:

- low approach/progress metrics means the policy is not reaching the block
- high `reward/near_contact_rate` but low contact metrics means the last-touch bridge is failing
- contact without persistence or grasp means the policy can touch but not hold/squeeze
- improved reward without these manipulation metrics moving is not real task progress

## Acceptance criteria for future runs

The reward redesign should be considered successful only if:

- contact clearly emerges early in training
- grasp persistence or sustained contact becomes nonzero
- `reward/hover_stall_rate` trends down instead of staying high
- approach and near-contact metrics improve together with reward
- reward gains are explained by contact, grasp, and lift metrics rather than alignment alone

If reward improves while contact and grasp remain flat, that should be treated as another reward-topology failure rather than genuine task progress.
