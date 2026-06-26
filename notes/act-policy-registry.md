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
| 2026-06-23 13:27 | `act_sim_ppo_contact_stall_v2` | [`act_so101_corrected_30_b32_20260621_160923 / 026020`](../simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model) | [`1db3iq53`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/1db3iq53) | 300 / 180000 | [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_contact_stall_v2/act_sim_ppo_checkpoint.pt) | W&B return improved to near zero, but live sim behavior regressed badly: arm retracts into a sitting position and stays there, with occasional slight gripper spinning; 0 success/contact/grasp/lift in final rollout. |
| 2026-06-24 11:28 | `act_sim_ppo_workspace_engagement_v3` | [`act_so101_corrected_30_b32_20260621_160923 / 026020`](../simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model) | [`g1jtc81c`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/g1jtc81c) | 300 / 180000 | [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_workspace_engagement_v3/act_sim_ppo_checkpoint.pt) | Better workspace engagement: arm moves immediately toward the block and hovers slightly away, but does not attempt grasp; final rollout had 0 success/lift/grasp and 190 contact steps, with best run showing grasp attempts but no lift. |
| 2026-06-24 19:29 | `act_sim_ppo_grasp_transition_v4` | [`act_so101_corrected_30_b32_20260621_160923 / 026020`](../simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model) | [`0ufccyej`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/0ufccyej), resumed as [`kqky26x8`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/kqky26x8) | 300 / 180000 | [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_grasp_transition_v4/act_sim_ppo_checkpoint.pt) | Regressed: learned gripper-closing reward but no contact/grasp/lift; live sim returns to full arm retraction into sitting position and stays there. |
| 2026-06-25 10:07 | `act_sim_ppo_contact_commit_v5` | [`act_so101_corrected_30_b32_20260621_160923 / 026020`](../simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model) | [`skfhaguf`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/skfhaguf) | 300 / 180000 | [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_contact_commit_v5/act_sim_ppo_checkpoint.pt) | Regressed to a new local minimum: arm scrunches up with forearm straight up and gripper facing outward around 90 degrees; final rollout had 0 success/grasp/lift and 21 contact steps. |
| 2026-06-25 12:44 | `act_sim_ppo_contact_commit_v5_ep150` episode-90 snapshot | [`act_so101_corrected_30_b32_20260621_160923 / 026020`](../simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model) | [`7aw545ha`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/7aw545ha) | 90 / 54720 | [`act_sim_ppo_checkpoint_snapshot_20260625_130447.pt`](../simulation_code/outputs/train/act_sim_ppo_contact_commit_v5_ep150/act_sim_ppo_checkpoint_snapshot_20260625_130447.pt) | Promising checkpoint: good approach and visible grip attempt from fixed start, then scrunches up, turns far left, and stays away; from alternate starts it goes directly to the far-left scrunched posture. |
| 2026-06-25 14:33 | `act_sim_ppo_preserve_grip_v6` | [`act_sim_ppo_contact_commit_v5_ep150` episode-90 snapshot](../simulation_code/outputs/train/act_sim_ppo_contact_commit_v5_ep150/act_sim_ppo_checkpoint_snapshot_20260625_130447.pt) | [`ulu2z9qq`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/ulu2z9qq) | 164 / 99120 | [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_preserve_grip_v6/act_sim_ppo_checkpoint.pt); snapshots [`ep0115`](../simulation_code/outputs/train/act_sim_ppo_preserve_grip_v6/act_sim_ppo_checkpoint_ep0115.pt), [`ep0140`](../simulation_code/outputs/train/act_sim_ppo_preserve_grip_v6/act_sim_ppo_checkpoint_ep0140.pt) | Promising continuation: approach/grip-attempt behavior preserved and retreat left is slower after attempted grasp, but policy closes gripper immediately at episode start and keeps it closed, preventing a real grasp; final W&B summary still 0 success with 188 contact steps. |
| 2026-06-25 22:23 | `act_sim_ppo_timed_gripper_v7` paused | [`act_sim_ppo_preserve_grip_v6`](../simulation_code/outputs/train/act_sim_ppo_preserve_grip_v6/act_sim_ppo_checkpoint.pt) | [`ipxywbv8`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/ipxywbv8) | 673 / 404640 | [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_timed_gripper_v7/act_sim_ppo_checkpoint.pt); snapshots [`ep0264`](../simulation_code/outputs/train/act_sim_ppo_timed_gripper_v7/act_sim_ppo_checkpoint_ep0264.pt), [`ep0364`](../simulation_code/outputs/train/act_sim_ppo_timed_gripper_v7/act_sim_ppo_checkpoint_ep0364.pt), [`ep0464`](../simulation_code/outputs/train/act_sim_ppo_timed_gripper_v7/act_sim_ppo_checkpoint_ep0464.pt), [`ep0564`](../simulation_code/outputs/train/act_sim_ppo_timed_gripper_v7/act_sim_ppo_checkpoint_ep0564.pt), [`ep0664`](../simulation_code/outputs/train/act_sim_ppo_timed_gripper_v7/act_sim_ppo_checkpoint_ep0664.pt) | Paused overnight run: still closes gripper immediately, approaches and stops just before the block with gripper pointed down, avoids meaningful interaction and barely touches; improvement is that it no longer quickly scrunches/retreats to a corner. |
| 2026-06-26 00:45 | `act_sim_ppo_pregrasp_guard_v8_overnight` | [`act_sim_ppo_preserve_grip_v6`](../simulation_code/outputs/train/act_sim_ppo_preserve_grip_v6/act_sim_ppo_checkpoint.pt) | Not recorded | ~256 / ~73728 | [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_pregrasp_guard_v8_overnight/act_sim_ppo_checkpoint.pt) | Rejected: live MuJoCo inference showed severe degradation; the arm waves around in the air instead of making a meaningful approach/contact/grasp attempt. Do not continue from this checkpoint; future experiments should restart from the v5 episode-90 snapshot. |
| 2026-06-26 04:31 | `act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17` | [`act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13 / ep0255`](../simulation_code/outputs/train/act_sim_ppo_r11ep240_gain_9p5mm_low_lr_r13/act_sim_ppo_checkpoint_ep0255.pt) | [`st01r7h2`](https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/st01r7h2) | 899 / not recorded | Best live-inspected snapshot [`ep0875`](../simulation_code/outputs/train/act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17/act_sim_ppo_checkpoint_ep0875.pt); final [`act_sim_ppo_checkpoint.pt`](../simulation_code/outputs/train/act_sim_ppo_r13ep255_gain_9mm_1cm_overnight_r17/act_sim_ppo_checkpoint.pt) | Best snapshot so far for visible lift intent: tries to grasp and begins to lift, but cannot align the gripper around the block well enough for a proper grasp. W&B had sparse 1cm curriculum successes near updates `869`-`871`; final checkpoint regressed, so prefer `ep0875` for handoff. Next work should target gripper/block alignment before raising lift thresholds. |

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
