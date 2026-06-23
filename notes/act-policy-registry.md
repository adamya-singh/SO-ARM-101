# ACT Policy Registry

Canonical local index of ACT policy-producing runs. Detailed training notes stay
in the linked history files; this file is the quick lookup for lineage, W&B
runs, checkpoints, and observed behavior.

Scope: ACT policy runs only. Pure smoke tests, benchmarks, and throughput sweeps
are omitted unless they produced a checkpoint that became a policy used later.

## ACT on Data

Supervised ACT policies trained from physical demonstration data with
`simulation_code/train_act_on_data.py`.

| Date | Policy / run | Dataset | W&B run | Steps | Final output policy | Status / behavior |
|---|---|---|---|---:|---|---|
| 2026-06-19 23:51 | `act_so101_physical` | `so101_pickplace_v1` | [`n9a49t3l`](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/n9a49t3l) | 100000 configured | Not recorded | Stopped/archived early; valid run, but underused GPU and had excessive raw step budget. |
| 2026-06-20 00:17 | `act_so101_physical` | `so101_pickplace_v1` | [`76pa7tet`](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/76pa7tet) | 6505 configured | Intermediate checkpoints `001627`, `003254` | Interrupted/resumed; higher throughput than baseline. |
| 2026-06-20 02:53 | `act_so101_physical` resumed | `so101_pickplace_v1` | Not recorded | 6505 | [`006505/pretrained_model`](../simulation_code/outputs/train/act_so101_physical/checkpoints/006505/pretrained_model) | Completed old `100/100` ACT config; poor behavior, barely moved and drifted downward. |
| 2026-06-21 16:08 | `act_so101_corrected_30_20260621_160830` | `so101_pickplace_v1` | [`8pfniqe2`](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/8pfniqe2) | 13010 configured | None | Failed immediately with CUDA OOM. |
| 2026-06-21 16:09 | `act_so101_corrected_30_b32_20260621_160923` | `so101_pickplace_v1` | [`08v5hbn1`](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/08v5hbn1) | 26020 | [`026020/pretrained_model`](../simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model) | Current best supervised base policy; improved approach and gripper timing. |

## ACT in Sim

PPO post-training runs from supervised ACT base policies with
`simulation_code/train_act_in_sim.py`.

| Date | Policy / run | Base ACT policy | W&B run | Updates / env steps | Final output policy | Status / behavior |
|---|---|---|---|---:|---|---|
| 2026-06-22 22:46 | `act_sim_ppo_corrected026020_fast_noeval_20260622_224633` | [`act_so101_corrected_30_b32_20260621_160923 / 026020`](../simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model) | [`ub7ged1m`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/ub7ged1m) | 50 / 5000 implied | [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_corrected026020_fast_noeval_20260622_224633/act_sim_ppo_checkpoint.pt) | 0 success; no contact, grasp, or lift steps in final summary. |
| 2026-06-23 01:31 | `act_sim_ppo_checkpoint` | [`act_so101_corrected_30_b32_20260621_160923 / 026020`](../simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model) | [`ck6lfw46`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/ck6lfw46) | 1000 / 600000 | [`act_sim_ppo_checkpoint.pt`](../simulation_code/act_sim_ppo_checkpoint.pt) | Better approach/contact confidence and gripper angling, but crashes into the block and stalls; 0 success. |

## How to Update

For each new ACT on data run, add one row with:

- date/time training started
- policy or output run name
- dataset name
- W&B run ID/link, or `Not recorded`
- configured or completed step count
- final checkpoint link, or `None` if no policy exists
- short behavior/status summary

For each new ACT in sim run, add one row with:

- date/time training started
- PPO policy or checkpoint name
- base ACT policy name and checkpoint link
- W&B run ID/link, or `Not recorded`
- PPO updates and env steps
- final `.pt` checkpoint link
- short behavior/status summary, including success/contact/grasp/lift outcome when known

Related detailed notes:

- [ACT training workflow](act-training.md)
- [ACT on data history](train-act-on-data-history.md)
- [ACT sim PPO throughput sweep 2026-06-22](act-sim-ppo-throughput-sweep-20260622.md)
- [ACT sim PPO throughput sweep 2026-06-23](act-sim-ppo-throughput-sweep-20260623_012249.md)
