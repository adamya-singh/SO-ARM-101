# Retry tranche: legacy experiments on the fixed simulator (2026-10-04)

Live status page: https://claude.ai/artifact/Dt7rfwHp88iE4d1e8VPZQi (rows come from
`tools/tranche_status.py`). W&B: project `so-arm101-v2-scaling`, tag `retry-tranche-20261004`, groups `act`,
`smolvla`, `rl-pi05`, `rl-smolvla`, `rl-act`, `rerun-entries-03-07`. Saved views are made with
`tools/wandb_workspace.py --family {act,smolvla,rl,tranche}`.

## Why

From January to July, a strict grasp of the cube was physically impossible in every simulator configuration:
- MuJoCo's convex hulls filled the jaw mouth.
- The jaw-tip sites sat 15–27 mm off the blades.
- The jaws closed like scissors.

So every RL and VLA result from that period measured a task no policy could solve. That includes ReinFlow on
SmolVLA (best 2.6% grasp rate after 21,650 episodes), the Pi0 ReinFlow path, ACT + sim PPO (corner pinches only),
and the ACT and SmolVLA fine-tunes.

The fixes landed in August and September:
- the Menagerie arm model, on 08-01
- MuJoCo pinned to 3.9, on 08-03
- the lens-matched camera and measured joint zeros, from 09-06 to 09-09

Research-log entries 03–07 used the new arm model but the stray MuJoCo 3.11 and were never re-run.

## What the user decided (2026-10-04)

- **What "retry" means:** port the methods to the current bench task (20 mm cube, put back on the square, wrist
  camera). The old napkin task is not rebuilt; the old fine-tuned checkpoints are gone anyway.
- **RL targets:** ReinFlow on pi0.5 (starting from `pi05_lora16_10k_20261002`, W&B c07me9rl), ReinFlow on SmolVLA,
  and Gaussian PPO on ACT. Pi0 is dropped, since it was never trained in the old stack either.
- **Budget:** a fixed number of training episodes, the same for every RL run.
- **Order of work:** every code change, probe and rehearsal first, then the whole tranche is queued at once.

## Experiments and what each answer would mean

| id | question | if it works |
|---|---|---|
| ft-act | Does ACT, trained like pi0.5, place the cube on unseen placements? | It matches or beats the CNN (6/30), so the June–July ACT failures were the simulator. |
| ft-smolvla | Does SmolVLA (450M) learn the task as well as pi0.5 (3B)? | It reaches pi0.5's held-out error and 12/30, so a 7x smaller VLA would do. |
| rl-pi05 | Does closed-loop RL fix the pi0.5 fine-tune's failures (grasp quality, ~10 mm scatter)? | It beats 12/30 at execute 90 (and 15/30 at execute 30), with smaller closure error. |
| rl-smolvla | Does January–April ReinFlow work once grasping is possible? | It beats the SmolVLA fine-tune held-out. |
| rl-act | Does the June–July ACT PPO lane work once grasping is possible? | It beats the ACT fine-tune held-out. |
| rerun-03..07 | Do entries 03–07 reach the same verdict under MuJoCo 3.9? | Same verdict: the entry stands. A different verdict: the entry is corrected. |

**Reading the numbers.**
- **Held-out set:** 10 unseen placements × 3 rollouts. The three repeats are bit-identical, so this is
  effectively 10 poses.
- **Baselines:** CNN-4800 6/30. The pi0.5 fine-tune 12/30 at execute 90 and 15/30 at execute 30.
- **Held-out start-90 error:** CNN 3.6e-4, pi0.5 2.5e-4, working fixed-square policy 2.5e-6.
- **Closure:** tolerance 8 mm; working policy 1.7 mm.
- **Judge by pooled counts.** One run's count of 30 swings by about 9 across seeds (entry 34). Read a single
  RL run against its own starting policy on the same 30 rollouts, and against the monitoring curve, not as a
  precise rate.

## Design (same conditions for every method)

### Fine-tunes (`tools/vla_finetune.py`)

- **Recipe:** the pi0.5 recipe. 4800-placement capture (stride 30) plus 2400-placement capture (stride 18), shift
  12, the 90-step residual chunk target, pi0.5's quantile normalisation.
- **Logged during training:** the held-out start-90 curve.
- **Evaluation:** 30 rollouts at execute 90 and 30, with closure in mm.
- **SmolVLA:** starts from `lerobot/smolvla_base`. The VLM is frozen; the expert, projections and state
  projection train, under bf16 autocast.
- **ACT:** an ImageNet ResNet18 with a VAE, chunk 90, L1 + KL, backbone at 1e-5.

### RL (`tools/rl_train.py`, `src/so_arm101_v2/rl/`)

- **Environment:** `ChunkEnv` copies the evaluator's step sequence; `test_rl.py` checks it reproduces it step for
  step.
  - The frame is rendered only at chunk boundaries.
  - An invalidated episode stops at once.
  - Workers are spawned without torch.
- **Reward** (`rl/reward.py`):
  - +1 on contract success, −1 on a safety invalidation.
  - Potential shaping of at most 0.3, built only from contact milestones: strict grasp acquired, pickup complete,
    closeness of the cube to the square after the pickup.
  - Nothing pays for approach or alignment. The April reward did, and PPO learned to hover.
- **Scenarios:** 1600 teacher-screened poses on seeds 30–37 for training, and 20 monitoring poses on seed 21.
  - Both sets are disjoint from the held-out seed 8 and the capture seeds 12 and 14; the 7210 reserved poses are
    checked.
  - Every method walks the same fixed order.
  - Index: `artifacts/.../retry_tranche_20261004/rl_suites/rl_suites.json`, order sha256 78b3e1e4….
- **PPO** (`rl/ppo.py`), carrying the legacy fixes:
  - GAE per episode (the trajectory-identity fix).
  - The denominator recomputed through the update's own path; the gap to the rollout is logged as a consistency
    check.
  - Early stop on post-update KL.
  - The base model always in eval mode (the dropout KL fix).
  - One sampler for rollouts and log-probs (the inference-sampler fix).
  - Only stable heads train.
  - New: a KL-adaptive actor learning rate. The SmolVLA and ACT starting policies do not exist until the queue
    trains them, and a rehearsal ACT update reached KL ~1500.
- **ReinFlow** (`rl/reinflow.py`):
  - Noise head on the expert's features (via a hook on `action_out_proj`, so lerobot-fork is untouched).
  - tanh-bounded sigma; noise only in the 6 real action channels.
  - Trainable: action in/out projections, time MLPs and the noise head.
- **ACT PPO** (`rl/act_ppo.py`): per-joint Gaussian on the deterministic chunk; the decoder and action head train.
- **Schedule:** execute 90 (the CNN runs' decision points), about 6 decisions per episode.
- **Monitoring:** every 480 episodes, the monitoring suite with noise off.
- **Final evaluation:** the same 30 held-out rollouts as every placement run, with noise off.
- **Abort guards:** a non-finite loss, KL above 10× target three times in a row, or monitoring at 0 three times
  in a row while the start was above 0.

### Speed

- **Rollouts:** render only at chunk boundaries (`rollout.py`, test-guarded). Measured 33 s → 8 s per episode.
- **RL collection:** subprocess envs, with one batched GPU forward per group of boundaries that overlaps the
  simulation.
- **Probes:** written under `retry_tranche_20261004/probes/`; results are below.

## Probes and the episode budget

(filled in from the probes before queueing)

## Queue

(filled in when queued)

## Deviations from the plan file

- `Pi05ChunkPolicy` was not re-based on `ResidualChunkPolicy`. It carries the live servo-lag compensation that
  another session is changing today. The generic policy reproduces its sampling and clamp exactly, and both read
  the same checkpoint.
- pi0.5's RL scope is the stable heads (action projections and time MLPs), not the expert LoRA. This is the same
  scope as SmolVLA's, so the two ReinFlow runs differ only in the base model.
