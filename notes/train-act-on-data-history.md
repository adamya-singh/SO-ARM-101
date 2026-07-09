# train_act_on_data.py Training History

This log tracks offline ACT training runs launched through
`simulation_code/train_act_on_data.py`. It is meant to preserve the practical
training decisions, failures, and checkpoint paths so future runs do not repeat
known bad settings.

## Current Recommendation

Use the corrected ACT profile:

```bash
cd /home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code

python3 train_act_on_data.py \
  --profile corrected-act \
  --num-workers 0
```

As of the latest script version, `--profile corrected-act` resolves to:

- `batch_size=32`
- `policy.chunk_size=30`
- `policy.n_action_steps=30`
- `policy.action_lead_steps=3`
- `policy.use_amp=true`
- `target_epochs=20`
- `steps=26020`
- `save_freq=6505`
- `num_workers=0`

The batch-size default is intentionally `32`: the corrected batch-64 attempt
OOMed immediately on the RTX 3090, while batch 32 completed.

Current supervised base preference after live MuJoCo inspection is the lead-3
checkpoint from `act_so101_lead3_30_b32_20260709_161056`. Compared with the old
best corrected `30/30` checkpoint, it appears more confident, moves a little
faster, and tries to interact with the block more. This is a qualitative
pre-PPO observation, not a solved pickup result.

## Dataset Contract

Dataset:

`/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/imitation-learning/datasets/so101_pickplace_v1`

Confirmed preflight facts:

- 100 episodes
- 41,631 frames
- 30 FPS
- 6D `observation.state`
- 6D `action`
- wrist image key: `observation.images.wrist`
- same-frame `action == observation.state`
- future deltas are meaningful, so ACT can still train future state chunks

The same-frame equality is important: the model is learning future trajectory
chunks built by LeRobot ACT, not a same-frame delta-action command stream. New
corrected ACT runs should use an action lead so the first target starts at a
future follower pose instead of repeating the current pose.

## Run History

| Date/time | Run | Command/profile | Result | Notes |
|---|---|---|---|---|
| 2026-06-19 23:51 | `act_so101_physical` | baseline defaults, `batch_size=8`, `steps=100000` | stopped/archived early | Valid run, but underused GPU and had excessive raw step budget. W&B: [n9a49t3l](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/n9a49t3l). |
| 2026-06-20 00:17 | `act_so101_physical` | fast profile, `batch_size=128`, old ACT `chunk_size=100`, `n_action_steps=100`, `steps=6505` | interrupted/resumed | Higher throughput than baseline. Saved checkpoints at `001627` and `003254` before resume work. W&B: [76pa7tet](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/76pa7tet). |
| 2026-06-20 02:53 | `act_so101_physical` | resumed fast run, `batch_size=128`, old ACT `100/100` | completed | Final checkpoint `checkpoints/006505/pretrained_model`. Physical behavior was poor: barely moved and drifted downward. Final config used `chunk_size=100`, `n_action_steps=100`. |
| 2026-06-21 16:08 | `act_so101_corrected_30_20260621_160830` | corrected ACT, `batch_size=64`, `chunk_size=30`, `n_action_steps=30`, `steps=13010` | failed immediately | CUDA OOM shortly after training started. W&B: [8pfniqe2](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/8pfniqe2). |
| 2026-06-21 16:09 | `act_so101_corrected_30_b32_20260621_160923` | corrected ACT, `batch_size=32`, `chunk_size=30`, `n_action_steps=30`, `steps=26020` | completed | Best run so far. Final loss about `0.048` at ~20 epochs. Sim/physical qualitative behavior improved: arm moves toward block and opens/closes gripper around it. W&B: [08v5hbn1](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/08v5hbn1). |
| 2026-07-09 16:10 | `act_so101_lead3_30_b32_20260709_161056` | corrected ACT, `batch_size=32`, `chunk_size=30`, `n_action_steps=30`, `action_lead_steps=3`, `steps=26020` | completed | New preferred supervised base after live sim comparison. Final loss about `0.048` at ~20 epochs. Compared with the old best pretrain, it appears more sure of itself, moves a little faster, and tries to interact with the block more. W&B: [sl2zf3jl](https://wandb.ai/7adamyasingh-rutgers-university/lerobot/runs/sl2zf3jl). |

## Important Checkpoints

Old poor-behavior checkpoint:

```text
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_physical/checkpoints/006505/pretrained_model
```

Latest corrected checkpoint:

```text
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model
```

Latest lead-3 corrected checkpoint:

```text
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model
```

Intermediate corrected checkpoints to compare behaviorally:

```text
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/006505/pretrained_model
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/013010/pretrained_model
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/019515/pretrained_model
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model
```

## Lessons Learned

- `chunk_size=100` and `n_action_steps=100` produced overly smooth, weak
  behavior on this 100-episode dataset.
- `chunk_size=30` and `n_action_steps=30` produced much better approach and
  gripper timing.
- `action_lead_steps=3` improved live sim confidence versus the old unshifted
  corrected pretrain: faster movement and more willingness to interact with the
  block.
- Batch 64 with corrected ACT OOMed despite apparent VRAM headroom; batch 32 is
  the stable default.
- More epochs should not be the first lever. First compare saved checkpoints by
  task behavior and collect targeted additional demos for the failure modes.
- Loss alone is not enough. Track reach, gripper timing, grasp, lift, and
  success rate.

## Action-Lead Sweep

For follower-only kinesthetic recordings, compare these next supervised ACT
pretraining runs before choosing the next base checkpoint:

```bash
python3 train_act_on_data.py --profile corrected-act --action-lead-steps 1 \
  --output-dir outputs/train/act_so101_lead1_30_b32 \
  --job-name act_so101_lead1_30_b32 --num-workers 0

python3 train_act_on_data.py --profile corrected-act --action-lead-steps 3 \
  --output-dir outputs/train/act_so101_lead3_30_b32 \
  --job-name act_so101_lead3_30_b32 --num-workers 0

python3 train_act_on_data.py --profile corrected-act --action-lead-steps 5 \
  --output-dir outputs/train/act_so101_lead5_30_b32 \
  --job-name act_so101_lead5_30_b32 --num-workers 0
```

## Next Evaluation Commands

Headless sim evaluation for the latest corrected checkpoint:

```bash
cd /home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code

python3 run_act_sim_inference.py \
  --checkpoint /home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model \
  --episodes 10 \
  --headless
```

Live MuJoCo viewer:

```bash
python3 run_act_sim_inference.py \
  --checkpoint /home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model \
  --episodes 1 \
  --render
```
