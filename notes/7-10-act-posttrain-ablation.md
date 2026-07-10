# 2026-07-10 ACT Post-train PPO Ablation

## Goal

Run a controlled overnight study after the camera, initial-pose, scene-appearance,
reward-telemetry, and default-pretrain updates. The experiment is intended to
answer three questions:

1. Does the new action-lead-3 supervised pretrain improve PPO learning compared
   with the previous unshifted corrected pretrain?
2. Does per-reset lighting and surface-brightness randomization help or hurt PPO
   learning?
3. Can the best new setup learn with narrow block-pose randomization instead of
   overfitting to one fixed cube position?

Seeds `11` and `29` repeat the important comparisons. Treat an effect as
credible only if both seeds agree directionally. This is an ablation study, not
ten attempts at the same policy.

W&B group: `act-posttrain-ablation-20260710`

Queue launcher:
[`simulation_code/queue_act_posttrain_ablation_20260710.sh`](../simulation_code/queue_act_posttrain_ablation_20260710.sh)

Output root:
`simulation_code/outputs/train/act_posttrain_ablation_20260710/`

## Compared Pretrains

- New/default: `act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model`.
  This uses a three-step action lead.
- Old/corrected: `act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model`.
  This is the previous corrected but unshifted ACT base.

June PPO checkpoints are deliberately not resumed because the simulation and
supervised initialization have materially changed.

## Shared Settings

- 200 fresh PPO updates per run; snapshots every 10 updates
- actor LR `1e-6`; critic LR `5e-5`; initial log standard deviation `-2`
- 12 parallel environments; 2 rollout chunks per environment
- chunk size 30; 150 maximum episode steps; one simulator step per action
- minibatch size 64; one PPO epoch; no in-loop evaluation
- EGL headless rendering; W&B enabled
- main-process and worker RNGs seeded from the listed base seed

Fixed-appearance runs retain the updated nominal black-mat, white-cube scene but
disable reset-to-reset lighting and brightness variation. Narrow-block runs use
distance `0.22-0.26m` and angle `-10 to +10 degrees`.

## Run Matrix

All ten jobs completed successfully on 2026-07-10.

| # | Run | Pretrain | Appearance | Block reset | Seed | W&B | Status/result |
|---:|---|---|---|---|---:|---|---|
| 1 | `new_app_fixed_s11` | lead-3 | randomized | fixed | 11 | [`zx9eo09z`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/zx9eo09z) | Completed; no strict lift or success. |
| 2 | `old_app_fixed_s11` | old corrected | randomized | fixed | 11 | [`vqcxl2y1`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/vqcxl2y1) | Completed; effectively no interaction. |
| 3 | `new_noapp_fixed_s11` | lead-3 | fixed | fixed | 11 | [`mxxzvpsf`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/mxxzvpsf) | Completed; more contact/height, no strict lift. |
| 4 | `old_noapp_fixed_s11` | old corrected | fixed | fixed | 11 | [`8w49ekbe`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/8w49ekbe) | Completed; one success-bearing worker chunk. |
| 5 | `new_app_fixed_s29` | lead-3 | randomized | fixed | 29 | [`6o3f4l1a`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/6o3f4l1a) | Completed; eight strict-lift steps, no success. |
| 6 | `old_app_fixed_s29` | old corrected | randomized | fixed | 29 | [`l0tmvx59`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/l0tmvx59) | Completed; weak contact, no strict lift. |
| 7 | `new_noapp_fixed_s29` | lead-3 | fixed | fixed | 29 | [`8yyilf1l`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/8yyilf1l) | Completed; four success-bearing worker chunks and 28 lift steps. |
| 8 | `old_noapp_fixed_s29` | old corrected | fixed | fixed | 29 | [`qm5kywyk`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/qm5kywyk) | Completed; four success-bearing worker chunks and 28 lift steps. |
| 9 | `new_app_narrow_s11` | lead-3 | randomized | narrow randomized | 11 | [`u97hd43j`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/u97hd43j) | Completed; no strict lift or success. |
| 10 | `new_app_narrow_s29` | lead-3 | randomized | narrow randomized | 29 | [`7vxgqnrq`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/7vxgqnrq) | Completed; three success-bearing worker chunks and 47 lift steps. |

## Aggregate Training Results

Counts cover all 200 updates. `Success chunks` is the sum of the fractional
12-environment rollout metric multiplied by 12. It counts worker chunks in
which success was observed; it is not an independently measured episode
success rate.

| Run | Success chunks | Lift steps | Micro lifts | Max height gain | Mean contact steps/update |
|---|---:|---:|---:|---:|---:|
| `new_app_fixed_s11` | 0 | 0 | 0 | 8.9mm | 0.2 |
| `old_app_fixed_s11` | 0 | 0 | 0 | 0.2mm | 0.0 |
| `new_noapp_fixed_s11` | 0 | 0 | 0 | 8.5mm | 3.5 |
| `old_noapp_fixed_s11` | 1 | 2 | 6 | 10.7mm | 31.2 |
| `new_app_fixed_s29` | 0 | 8 | 16 | 9.6mm | 36.6 |
| `old_app_fixed_s29` | 0 | 0 | 2 | 8.8mm | 5.8 |
| `new_noapp_fixed_s29` | 4 | 28 | 546 | 12.8mm | 114.9 |
| `old_noapp_fixed_s29` | 4 | 28 | 634 | 11.1mm | 142.2 |
| `new_app_narrow_s11` | 0 | 0 | 0 | 8.7mm | 1.4 |
| `new_app_narrow_s29` | 3 | 47 | 193 | 11.7mm | 56.0 |

## Morning Analysis

For every run, record both the best 20-update rolling window and the final 20
updates:

- primary: success, lift steps, micro-lift count, and maximum block-height gain
- grasp quality: contact/grasp steps, jaw centering/alignment, aligned closure,
  and side-push penalties
- PPO health: approximate KL, clip fraction, policy/critic loss, log standard
  deviation, and throughput

Make these comparisons:

1. New versus old pretrain within each matched fixed-block seed and appearance.
2. Randomized versus fixed appearance within each matched pretrain and seed.
3. Narrow randomized block placement versus the matched lead-3 randomized-
   appearance fixed-block runs.
4. Report a conclusion only when seeds 11 and 29 agree directionally.

## Checkpoint Selection and Evaluation

Select up to three snapshots from the best condition. Prefer snapshots nearest
strict lift events; otherwise use the highest rolling height gain with sustained
grasp and low side-push penalty.

After the queue finishes, evaluate each candidate for 20 fixed-seed episodes on
both fixed and narrow-randomized block placement. Do not select a policy from
training return alone.

## Findings and Limitations

- Within these two seeds, fixed-appearance runs had higher aggregate contact
  and lift metrics than their appearance-randomized counterparts for both
  pretrains. This measures PPO learning in the tested simulator distribution;
  it does not show whether appearance randomization helps or hurts sim-to-real
  transfer.
- With randomized appearance, lead-3 exceeded the old pretrain on aggregate
  interaction metrics at both seeds. With fixed appearance, the comparison was
  mixed: the old pretrain was stronger at seed 11, while the two seed-29 runs
  had the same total lift-step count and different secondary metrics. The sweep
  therefore does not identify an overall pretrain winner.
- The narrow-block comparison disagreed across seeds: seed 29 produced 47 lift
  steps and three success-bearing worker chunks, while seed 11 produced no
  strict lift or success signal. Generalization to narrow block placement is
  not established.
- Seed 29 produced substantially more interaction than seed 11 in most matched
  conditions. Two seeds are insufficient to estimate the variance or attach
  uncertainty bounds to the factor effects.
- Clip fraction was logged as zero throughout and approximate KL remained below
  about `0.007`. These observations rule out a recorded clipping/KL spike, but
  they do not by themselves establish that the optimizer or update scale was
  ideal.
- Success, lift, contact, and grasp values above are training-rollout metrics,
  not independent evaluation rates. High-contact runs also had larger negative
  side-push terms, so contact and height gain cannot be interpreted as clean
  grasp-and-lift behavior without rollout evaluation.

Using the predeclared rule of requiring strict-lift/success signal at both
seeds, the old corrected pretrain with fixed appearance and fixed block was
chosen as a **provisional evaluation condition**. This is a selection for
follow-up, not evidence that it is the best policy.

Provisional snapshots selected for evaluation:

- `old_noapp_fixed_s11/act_sim_ppo_checkpoint_ep0129.pt`
- `old_noapp_fixed_s29/act_sim_ppo_checkpoint_ep0099.pt`
- `old_noapp_fixed_s29/act_sim_ppo_checkpoint_ep0159.pt`

The planned snapshot tournament is incomplete. A deterministic fixed-block
screen of the seed-11 episode-129 snapshot was manually interrupted after 13 of
20 episodes: it recorded zero strict lifts/successes, and one episode had four
grasp-count steps. The other provisional snapshots and narrow-placement cases
have not been independently evaluated. No checkpoint from this sweep should be
called a validated winner yet.

Suggested follow-up (interpretation, not a sweep result): use fixed appearance
and fixed block to compare the two pretrains across more seeds, complete
independent snapshot evaluation, and only then test appearance or block-pose
curricula.
