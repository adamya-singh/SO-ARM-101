# 2026-07-11 Lead-3 Grasp-Focused PPO Ablation

## Goal

Start fresh from the lead-3 supervised `026020` checkpoint and determine which
optimizer, exploration, and grasp-quality reward settings most reliably turn
the policy's useful approach behavior into jaw-centered grasp and lift.

The study is staged. Phase one screens six settings for 100 updates at seeds
17 and 71. After comparing complete W&B curves, phase two replicates the top
two settings for 190 updates at seeds 83 and 101 and evaluates fixed milestone
snapshots. Final confirmation uses 30 deterministic fixed, 30 stochastic fixed,
and 30 stochastic narrow-block episodes per finalist.

## Reward Profiles

- `baseline`: no overrides; numerically preserves the current reward defaults.
- `jaw_quality`: raises positive pregrasp alignment, aligned-close, and
  jaw-centered-contact shaping and extends centered-contact memory.
- `jaw_quality_rebalanced`: keeps the positive quality changes but reduces
  generic bilateral/persistent-grasp reward and modestly strengthens side-push
  and displacement penalties.

The profiles are selected with `train_act_in_sim.py --reward-profile ...` and
are stored in W&B and checkpoint configuration.

## Launchers

- Phase one: `simulation_code/queue_act_grasp_phase1_20260711.sh`
- Phase two: `simulation_code/queue_act_grasp_phase2_20260711.sh`
- Final evaluation: `simulation_code/queue_act_grasp_final_eval_20260711.sh`

Every launcher validates required artifacts, limits task spooler to one GPU
job, and supports `DRY_RUN=1`. Phase two requires `WINNER1` and `WINNER2` IDs;
final evaluation requires two checkpoint paths and their reward profiles.

## Selection Rules

At the phase-one gate, rank profiles by cross-seed success/lift presence,
worst-seed lift and micro-lift counts, micro-lift-to-grasp conversion, centered
contact/close quality, side pushing, displacement, action clipping, and PPO
health. Raw grasp duration or return alone cannot select a winner.

The independent evaluator writes JSON containing per-episode and aggregate
sustained-grasp rate, strict-lift rate, success, height gain, displacement, and
action clipping. The final setting is chosen by sustained-grasp episode rate,
then strict lift/success, with height, displacement, and clipping as
tie-breakers.

## Phase-One Results

All 12 runs completed 100 updates. The figures below were aggregated from all
100 W&B history rows in each run, not from only the final summary. A
success-bearing worker chunk is recovered from the logged mean success rate
across the 12 parallel workers. These are on-policy training-rollout metrics;
they are not independent checkpoint evaluations.

Common conditions were the lead-3 supervised `026020` initialization, fixed
appearance and block placement, actor learning rate `1e-6`, critic learning
rate `5e-5`, and initial log standard deviation `-2.0`, except for the change
named in each setting:

- A: unchanged baseline and the control for all comparisons.
- B: actor learning rate reduced to `5e-7`.
- C: lower exploration, with initial log standard deviation `-2.356675`
  (action standard deviation about `0.095` instead of `0.135`).
- D: baseline optimizer and exploration with the `jaw_quality` reward profile.
- E: baseline optimizer and exploration with the
  `jaw_quality_rebalanced` reward profile.
- F: the `jaw_quality` profile combined with C's lower exploration.

| Setting | Seed | W&B | Success chunks | Lift steps | Micro lifts | Grasp steps | Maximum height gain |
|---|---:|---|---:|---:|---:|---:|---:|
| A | 17 | [`kdwzzqh7`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/kdwzzqh7) | 2 | 34 | 511 | 1,457 | 11.4mm |
| A | 71 | [`5x3y7onh`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/5x3y7onh) | 1 | 22 | 274 | 652 | 10.2mm |
| B | 17 | [`svxstwyd`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/svxstwyd) | 3 | 16 | 71 | 266 | 12.4mm |
| B | 71 | [`yeyzhlqb`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/yeyzhlqb) | 1 | 6 | 60 | 173 | 10.5mm |
| C | 17 | [`bx0nm3q9`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/bx0nm3q9) | 1 | 5 | 163 | 392 | 10.1mm |
| C | 71 | [`rkn4swif`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/rkn4swif) | 5 | 38 | 2,448 | 7,849 | 10.4mm |
| D | 17 | [`n2l20yhk`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/n2l20yhk) | 2 | 15 | 251 | 448 | 10.3mm |
| D | 71 | [`pjph7x1l`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/pjph7x1l) | 2 | 9 | 67 | 156 | 10.7mm |
| E | 17 | [`ejh08us4`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/ejh08us4) | 1 | 14 | 64 | 270 | 10.6mm |
| E | 71 | [`kima1cyb`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/kima1cyb) | 2 | 27 | 1,019 | 2,776 | 10.9mm |
| F | 17 | [`r3aznu1j`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/r3aznu1j) | 2 | 3 | 70 | 205 | 11.2mm |
| F | 71 | [`u8fliart`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/u8fliart) | 1 | 22 | 1,161 | 5,004 | 10.5mm |

### Interpretation

- A had the strongest worst-seed lift, micro-lift, and grasp counts. Both seeds
  improved late, so A is the most reproducible phase-one setting in this
  two-seed sample.
- B produced much less grasp and lift activity at both seeds. Its isolated
  seed-17 success and 12.4mm height maximum do not outweigh the consistently
  slower learning, but they show that it was not completely inactive.
- C seed 71 was the strongest individual training run by grasp, micro-lift,
  lift, success, and late return. C seed 17 had only five lift steps. C
  therefore has the highest observed upside but no evidence yet that the
  improvement reliably comes from lower exploration rather than seed
  variation.
- D increased the logged aligned-close reward, as intended, but reduced grasp,
  micro-lift, and lift counts relative to matched A seeds. Its two success
  chunks at each seed are encouraging but too sparse to establish an
  improvement.
- E also improved aligned-close measurements. Seed 71 developed strong late
  grasp and micro-lift activity, while seed 17 did not. It is a plausible
  reward-profile candidate, but the two seeds do not establish a consistent
  advantage over A.
- F behaved similarly to C in its seed sensitivity. Seed 71 accumulated many
  grasp steps but only matched A seed 71's 22 lift steps, while seed 17 had
  three lift steps. This suggests poor grasp-to-lift conversion, not a clear
  improvement.

Reward-component magnitudes are not directly comparable between profiles when
the profile changes their weights. Return and raw grasp duration were therefore
not used alone to name a winner. Approximate KL and action clipping remained
small in every run, so there is no obvious PPO-instability explanation for the
cross-seed differences.

### Provisional Checkpoint Candidates

- Most reproducible setting: A. Its strongest training-associated snapshot is
  [`A seed 17 ep0099`](../simulation_code/outputs/train/act_lead3_grasp_phase1_20260711/A_baseline_lr1em6_stdm2.0_s17/act_sim_ppo_checkpoint_ep0099.pt).
  Updates 90-99 contained 14 lift steps, 267 micro lifts, 798 grasp steps, and
  two success-bearing chunks.
- Strongest individual run: C seed 71. Its provisional snapshot is
  [`ep0089`](../simulation_code/outputs/train/act_lead3_grasp_phase1_20260711/C_baseline_lr1em6_stdm2.356675_s71/act_sim_ppo_checkpoint_ep0089.pt).
  Updates 80-89 contained nine lift steps, 507 micro lifts, 1,502 grasp steps,
  and two success-bearing chunks. The following ten updates had higher grasp
  duration and return but only two lift steps and no success-bearing chunks.

These labels are provisional. A rollout window measures the policies used to
collect training data around a snapshot; it does not directly evaluate the
saved checkpoint. Neither candidate has yet passed the planned deterministic
and stochastic independent evaluation, so this phase does not establish a
new final policy.
