# pi0.5 and SmolVLA on the recipe that works for ACT (2026-11-22)

Status: both fine-tunes done; RL on SmolVLA queued behind them (tsp 263 rehearsal, 264 run). Simulation only.

## Problem and how we know

On 2026-10-02/04 pi0.5 scored 15/30 and SmolVLA 2/30 (0/30 at execute 30). Those runs used the SO-100 scene, the
unrandomized 4800 + 2400 captures and a 25 mm lift. Since then the ACT recipe changed in ways that never reached the
VLAs: SO-101 gripper (wider grasp tolerance), measured servo model, physics v2 (servo gains soft to stiff), 100 mm
lift, slow release, execute 30. ACT on that recipe scores 21 to 24 of 30 and lifts on the real arm.

## Experiment

Same data and scene as the tuned ACT baseline of 2026-11-18: `offset_prevcmd_20261118/capture_v2` (7150 placements,
`bench_physics_v2`), scene `so101_servo_lift100_rel30`, the same 10 held-out poses x 3, execute 30.
`simulation_code/queue_vla_best_sim_20261122.sh`, output `experiments/vla_best_sim_20261122`.

- SmolVLA: 30,000 steps, batch 32, frozen VLM (the October settings).
- pi0.5: LoRA rank 16, 5,000 steps (October: 10,000). Cut to fit the night; it is half-trained.

## Outcome

| Policy | Held-out loss (start 90) | Successes | Pickups | Closure, lateral median | Failures |
|---|---|---|---|---|---|
| ACT, tuned baseline (seeds 202 / 101) | 3.4e-4 / 3.2e-4 | 21 / 24 of 30 | 24 / 27 | 5.2 mm | |
| SmolVLA | 1.6e-3 | **14 of 30** | 17 | 13.2 mm | 10 safety invalidations, 3 not settled, 3 pickup incomplete |
| pi0.5, 5,000 steps | 7.5e-4 | **7 of 30** | 16 | 6.8 mm | 12 safety invalidations, 9 not settled, 2 pickup incomplete |

- SmolVLA went from 2/30 (0/30 at execute 30) to 14/30 with the same model settings. The recipe, not the model, was
  what held it back. Its closing error is still 13 mm, the same as in October, so the gain is the SO-101's grasp
  tolerance and the task changes, not better placement.
- pi0.5 picks the cube up about as often as SmolVLA (16 against 17) with half the closing error (6.8 mm), but loses
  9 episodes after the pickup ("not settled") and 12 to safety invalidations. Its held-out loss was still falling
  (1.1e-3 at 4,000 steps, 7.5e-4 at 5,000; October reached 2.5e-4 at 10,000), so this is not its ceiling.
- Neither reaches ACT. ACT has 5x the samples seen (5.1 M against 0.96 M for SmolVLA, 0.16 M for pi0.5).

## What went wrong on the way

- The `lerobot-pi05` env's editable installs point at the old checkout path; pi0.5 tools need
  `PYTHONPATH=$ROOT/src:~/projects/lerobot-fork/src` (first rehearsal failed on `import lerobot`).
- A pi0.5 rehearsal run beside the SmolVLA job on a second spooler socket filled the GPU and slowed SmolVLA from 75
  to 3 samples/s for about 15 minutes. CPU experiments run at the same time also cost both jobs about 15% throughput.

## Next

- RL on SmolVLA (ReinFlow PPO, 20,000 episodes, execute 30, physics v2 per episode): queued, runs the evening of 11-22.
- pi0.5 deserves the full 10,000 steps before RL: resume from this checkpoint's settings, about 7 more hours.
- Read the pi0.5 "not settled" episodes: 9 losses after a good pickup points at the release or the set-down.
