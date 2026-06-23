# ACT Sim PPO Throughput Sweep - 2026-06-22

Goal: choose practical high-throughput settings for `simulation_code/train_act_in_sim.py`
when initializing from the corrected offline ACT checkpoint:

```text
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model
```

The script is still an experimental ACT-chunk PPO path and requires
`--experimental-act-ppo`. The previous long ACT PPO attempt produced 0% success,
so these settings optimize runtime throughput only; task success still needs to
be judged from the real training run.

## Sweep Setup

- Environment: `/home/win10ubuntu/miniforge3/envs/lerobot`
- Headless MuJoCo: `MUJOCO_GL=egl`
- W&B disabled during sweep
- Render disabled
- Base policy fixed to corrected `026020` ACT checkpoint
- Horizon: `--max-steps-per-episode 100`
- Action repeat: `--steps-per-action 1`
- Sustained sweep used `--episodes 10` and `--eval-episodes 0`
- Sweep artifacts:

```text
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_sim_ppo_throughput_sweep_20260622_223701
```

## Short Sweep

| Setting | Wall time | Episodes | Seconds / episode | Last logged result |
|---|---:|---:|---:|---|
| `chunk8_mb1_ep1` | 33.31s | 3 | 11.10 | return `-7.111`, success `0` |
| `chunk15_mb1_ep1` | 28.95s | 3 | 9.65 | return `-11.764`, success `0` |
| `chunk30_mb1_ep1` | 29.57s | 3 | 9.86 | return `-0.967`, success `0` |
| `chunk30_mb2_ep1` | 31.21s | 3 | 10.40 | return `-8.788`, success `0` |
| `chunk30_mb4_ep1` | 30.98s | 3 | 10.33 | return `-4.264`, success `0` |
| `chunk30_mb4_ep2` | 30.63s | 3 | 10.21 | return `-0.316`, success `0` |

## Sustained Sweep

| Setting | Wall time | Episodes | Seconds / episode | Last logged result |
|---|---:|---:|---:|---|
| `sustained_chunk15_mb1_ep1` | 73.72s | 10 | 7.37 | return `-12.732`, success `0` |
| `sustained_chunk30_mb1_ep1` | 69.86s | 10 | 6.99 | return `-13.879`, success `0` |
| `sustained_chunk30_mb4_ep2` | 63.73s | 10 | 6.37 | return `-9.478`, success `0` |

## Winner

Use:

- `--chunk-size 30`
- `--minibatch-size 4`
- `--ppo-epochs 2`
- `--steps-per-action 1`
- `--max-steps-per-episode 100`
- `--eval-episodes 0` during the throughput-oriented training run
- Disable rendering and use EGL headless rendering

Rationale: `chunk_size=30` matches the corrected offline ACT policy horizon and
reduces expensive ACT forward passes. `minibatch_size=4` with `ppo_epochs=2`
was the fastest sustained setting tested, at about `6.37 s/episode`.
The sweep disabled in-loop eval, so the production throughput run should also
set `--eval-episodes 0`; run separate evaluation afterward to avoid slowing
training.

## Launch Command

```bash
cd /home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code

MUJOCO_GL=egl /home/win10ubuntu/miniforge3/envs/lerobot/bin/python train_act_in_sim.py \
  --experimental-act-ppo \
  --init-checkpoint /home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model \
  --episodes 50 \
  --max-steps-per-episode 100 \
  --chunk-size 30 \
  --steps-per-action 1 \
  --ppo-epochs 2 \
  --minibatch-size 4 \
  --eval-episodes 0 \
  --no-render \
  --headless \
  --checkpoint-path /home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/<run_id>/act_sim_ppo_checkpoint.pt
```

## Completed Run

Run launched with the winning throughput settings:

```text
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_sim_ppo_corrected026020_fast_noeval_20260622_224633
```

W&B:

```text
https://wandb.ai/7adamyasingh-rutgers-university/act-so101-sim-ppo/runs/ub7ged1m
```

Result summary from the 50-episode run:

- Final episode: `49`
- Final rollout return: `-1.85816`
- Final rollout success: `0`
- Final rollout contact/grasp/lift steps: `0 / 0 / 0`
- Final logged episode time: `4.68287s`
- Checkpoint:

```text
/home/win10ubuntu/dev/robotic-arm/SO-ARM-101/simulation_code/outputs/train/act_sim_ppo_corrected026020_fast_noeval_20260622_224633/act_sim_ppo_checkpoint.pt
```
