# 2026-07-10 Corrected-Coordinate ACT PPO Sweep

## Goal

Re-establish ACT post-training results after fixing the ACT dataset-coordinate
to MuJoCo mechanical-coordinate adapter and differentiable checkpoint
normalization contract. All runs start from supervised checkpoints; pre-fix PPO
checkpoints are not resumed.

The balanced experiment compares:

- lead-3 versus the old unshifted corrected ACT pretrain;
- actor learning rates `1e-6` versus `3e-6`; and
- seeds `11`, `29`, and `47`.

Launcher:
[`simulation_code/queue_act_coordinate_lr_sweep_20260710.sh`](../simulation_code/queue_act_coordinate_lr_sweep_20260710.sh)

W&B group: `act-posttrain-coordinate-lr-sweep-20260710`

Output root:
`simulation_code/outputs/train/act_posttrain_coordinate_lr_sweep_20260710/`

## Shared Settings

- 190 fresh PPO updates per primary run; snapshots every 10 updates
- fixed appearance and fixed curriculum block pose
- actor LR selected by the matrix; critic LR `5e-5`; initial log standard deviation `-2`
- 12 parallel environments; 2 rollout chunks per environment
- chunk size 30; 150 maximum episode steps; one simulator step per action
- minibatch size 64; one PPO epoch; no in-loop evaluation
- EGL headless rendering and W&B enabled
- one task-spooler slot and a shared hard deadline eight hours after `START_AT`

The seed-29 condition order is reversed to reduce correlation between condition
and wall-clock order. A lead-3, `1e-6`, seed-71 overflow run is queued after the
balanced matrix and runs only if time remains before the shared deadline.

## Run Matrix

| Order | Run | Pretrain | Actor LR | Seed |
|---:|---|---|---:|---:|
| 1 | `lead3_lr1e6_s11` | lead-3 | `1e-6` | 11 |
| 2 | `old_lr1e6_s11` | old corrected | `1e-6` | 11 |
| 3 | `lead3_lr3e6_s11` | lead-3 | `3e-6` | 11 |
| 4 | `old_lr3e6_s11` | old corrected | `3e-6` | 11 |
| 5 | `old_lr3e6_s29` | old corrected | `3e-6` | 29 |
| 6 | `lead3_lr3e6_s29` | lead-3 | `3e-6` | 29 |
| 7 | `old_lr1e6_s29` | old corrected | `1e-6` | 29 |
| 8 | `lead3_lr1e6_s29` | lead-3 | `1e-6` | 29 |
| 9 | `lead3_lr1e6_s47` | lead-3 | `1e-6` | 47 |
| 10 | `old_lr1e6_s47` | old corrected | `1e-6` | 47 |
| 11 | `lead3_lr3e6_s47` | lead-3 | `3e-6` | 47 |
| 12 | `old_lr3e6_s47` | old corrected | `3e-6` | 47 |

## Analysis Contract

Compare best and final 20-update windows for strict success-bearing chunks,
lift steps, micro-lifts, maximum height gain, contact/grasp quality, jaw
centering, aligned closure, side pushing, block displacement, KL, clip fraction,
losses, log standard deviation, action clipping, throughput, and worker errors.

Use matched seed-level differences for pretrain, actor-LR, and interaction
effects. Require directional agreement in at least two of the three primary
seeds before recommending a condition. Select at most three snapshots from the
winning condition and evaluate them after training; do not rank policies from
training return alone.

## Completion Status

All 12 primary jobs completed 190 PPO updates successfully on 2026-07-10. The
seed-71 overflow job used the final minute of the eight-hour window and was
interrupted at the shared deadline as designed; it is not included in the
balanced analysis.

`Success chunks` below is the sum of `rollout/success * 12`. It counts worker
rollout chunks containing a success event and is not an independently measured
episode success rate. All other counts are sums across the 190 training
updates. Peak height is the largest height gain observed in any training
rollout.

## Results

| Run | Success chunks | Lift steps | Micro lifts | Peak height | Contact steps | Grasp steps |
|---|---:|---:|---:|---:|---:|---:|
| `lead3_lr1e6_s11` | 6 | 46 | 629 | 11.79mm | 23,692 | 2,439 |
| `old_lr1e6_s11` | 16 | 96 | 1,806 | 12.75mm | 32,501 | 5,345 |
| `lead3_lr3e6_s11` | 22 | 108 | 1,496 | 12.56mm | 24,600 | 4,805 |
| `old_lr3e6_s11` | 1 | 12 | 815 | 10.37mm | 25,753 | 5,726 |
| `old_lr3e6_s29` | 2 | 6 | 367 | 10.33mm | 45,813 | 11,988 |
| `lead3_lr3e6_s29` | 0 | 13 | 257 | 11.59mm | 28,752 | 4,942 |
| `old_lr1e6_s29` | 14 | 67 | 2,450 | 12.51mm | 31,795 | 6,828 |
| `lead3_lr1e6_s29` | 2 | 34 | 51 | 11.85mm | 10,431 | 314 |
| `lead3_lr1e6_s47` | 5 | 34 | 584 | 11.50mm | 25,672 | 3,199 |
| `old_lr1e6_s47` | 11 | 53 | 646 | 12.74mm | 24,587 | 2,592 |
| `lead3_lr3e6_s47` | 3 | 9 | 615 | 11.36mm | 34,418 | 6,059 |
| `old_lr3e6_s47` | 11 | 41 | 980 | 11.84mm | 26,717 | 4,027 |

Condition totals across the three primary seeds:

| Pretrain | Actor LR | Success chunks | Lift steps | Micro lifts | Mean per-seed peak height |
|---|---:|---:|---:|---:|---:|
| old corrected | `1e-6` | **41** | **216** | **4,902** | **12.66mm** |
| lead-3 | `1e-6` | 13 | 114 | 1,264 | 11.71mm |
| lead-3 | `3e-6` | 25 | 130 | 2,368 | 11.84mm |
| old corrected | `3e-6` | 14 | 59 | 2,162 | 10.85mm |

For old corrected plus `1e-6`, the best 20-update windows contained 6, 4,
and 5 success chunks for seeds 11, 29, and 47. The corresponding final
20-update windows contained 6, 3, and 0. This condition therefore remained
strong through the end for seeds 11 and 29 but had already faded for seed 47.
The final checkpoint should not automatically be preferred over an earlier
snapshot.

## Interpretation

- At `1e-6`, old corrected exceeded lead-3 on success chunks, lift steps,
  micro lifts, and peak height in every matched seed. This is the only factor
  comparison that agrees directionally across all three seeds.
- Increasing the old-corrected actor LR to `3e-6` increased aggregate contact
  and grasp steps, but reduced success chunks, lift steps, and peak height in
  all three seeds. More interaction at the higher LR did not translate into
  better pickup behavior and may include pushing or poorly aligned contact.
- For lead-3, `3e-6` was much stronger at seed 11 but weaker on success/lift at
  seeds 29 and 47. Its aggregate improvement is therefore seed-driven rather
  than a consistent LR effect.
- PPO health did not show an optimizer blow-up: recorded PPO clip fraction was
  zero, approximate KL stayed below about `0.01`, and all workers completed.
  Stochastic action clipping still occurred in some runs, reaching roughly
  8% in the most affected run, but old corrected plus `1e-6` stayed at or
  below roughly 1%. This is separate from the eliminated systematic
  coordinate-contract clipping.

The training-rollout evidence selects old corrected plus `1e-6` as the
**provisional evaluation condition**. It does not establish an independently
validated policy, a fixed-block episode success rate, narrow-placement
generalization, or sim-to-real performance.

## Provisional Evaluation Candidates

- `old_lr1e6_s11/act_sim_ppo_checkpoint_ep0189.pt`
- `old_lr1e6_s29/act_sim_ppo_checkpoint_ep0129.pt`
- `old_lr1e6_s47/act_sim_ppo_checkpoint_ep0089.pt`

These snapshots are near strong success/lift windows and cover all three
training seeds. They remain candidates until the planned deterministic and
stochastic fixed-block evaluations, followed by narrow-block evaluation, are
completed.

### Live inspection note

User live-simulation inspection of
`old_lr1e6_s11/act_sim_ppo_checkpoint_ep0189.pt` with stochastic actions found
that the policy interacts with the block and makes limited grasp attempts. This
is encouraging qualitative evidence that the checkpoint retained useful
contact behavior, but it is not a measured grasp rate, lift result, or success
rate. The checkpoint remains provisional pending the planned controlled
evaluation.

## 2026-07-11 Paired Seed-5 Follow-up

A subsequent controlled pair used the same fixed scene, actor LR `1e-6`, seed
`5`, and 200-update budget for no-lead and lead-3. Only the supervised pretrain
changed. Full W&B histories, rather than final rows, gave:

| Pretrain | Success chunks | Lift steps | Micro lifts | Grasp steps | Peak height |
|---|---:|---:|---:|---:|---:|
| no-lead | 1 | 26 | 758 | 4,010 | **11.29mm** |
| lead-3 | **7** | **33** | **1,166** | **7,647** | 10.43mm |

Lead-3's strongest window was updates 22-41; the saved episode-39 snapshot is
the corresponding live-inspection candidate. Both runs weakened after their
best windows, so neither final checkpoint is automatically preferred. PPO
health remained stable, although lead-3 stochastic action clipping peaked at
about 11.9% versus 1.8% for no-lead.

User side-by-side live stochastic inspection found the policies broadly
similar, with lead-3 making more grasp attempts and looking slightly better.
Based on this paired result and live behavior, lead-3 remains the practical
default pretrain for current PPO follow-up. This is an engineering choice, not
a resolved universal lead ablation: the earlier three-seed `1e-6` training
comparison favored no-lead on aggregate rollout metrics.
