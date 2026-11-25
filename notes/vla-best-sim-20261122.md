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

<!-- written by OpenClaw from the queue's files, 20261125T184055; not reviewed -->
## Results (2026-11-25)

### Setup and terms for a new intern

A **policy** is the neural network that looks at the robot's camera images and joint angles and outputs joint commands to pick up a small cube and set it back down on a target square. Instead of predicting one time step at a time, our policies predict an **action chunk** of 90 future steps (`chunk_horizon: 90`) and execute the first 30 steps before looking at the camera again and replanning (**execute 30**). **Chunk start 0** is the opening survey move from the home pose, **chunk start 90** is the first descent toward the cube where visual localization matters most, and **chunk starts 180, 270, 360** cover lifting, carrying, setting down, and retreating.

**Offline loss** is the mean squared error (MSE) between the policy's predicted action chunk and the scripted teacher's action chunk on recorded demonstrations; **held-out** data means 10 cube placements (`pose_000` to `pose_009`) that the policy never trained on. In task units, the root-mean-square (**RMS**) joint angle error in degrees is reported as `heldout_start_90_deg_rms`. In closed-loop simulation rollouts (10 held-out poses with 3 repeats each, 30 rollouts in total), **lateral closure error** (`lateral_median_mm`) is the horizontal distance in millimetres between the gripper jaws and the cube center just before the jaws close; the newer **SO-101** gripper has an `8.0 mm` grasp tolerance (`grasp_tolerance_mm: 8.0`). A rollout can end in four ways:
1. **success**: the arm picks up the cube, lifts it, places it back inside the target region, releases it, retreats, and the cube settles stationary on the table (`settled: true`).
2. **safety_invalidation**: the simulator stops or disqualifies the rollout because the arm hits a joint limit (`limiting_frames`), exceeds a relative step bound, or collides unsafely with the table or cube (`unsafe_contact_frames`).
3. **pickup_incomplete**: the arm never completes a valid lift of the cube.
4. **not_settled**: the arm completes the pickup (`pickup_completed: true`), enters the placement region, releases the cube, and retreats, but the cube does not satisfy the settled-in-target check before the 480-action timeout (`settled: false`, `settled_frames: 0`).

This experiment tested whether two large pretrained **Vision-Language-Action (VLA)** models, **SmolVLA** (`450,267,776` parameters, `99,880,992` trainable with a frozen vision-language backbone) and **pi0.5** (`3,380,841,744` parameters, `457,506,336` trainable via **Low-Rank Adaptation / LoRA** rank 16 across `285` modules), can match the smaller **Action Chunking with Transformers (ACT)** baseline when given the exact same tuned SO-101 recipe (`offset_prevcmd_20261118/capture_v2`, `7,150` placements with `bench_physics_v2` soft-to-stiff servo randomization, scene `so101_servo_lift100_rel30` with a `100 mm` lift and slow release, and `execute 30`), and whether online **Reinforcement Learning (RL)** using **ReinFlow Proximal Policy Optimization (PPO)** (`20,000` episodes on the output head `out_head`) closes SmolVLA's remaining gap.

---

### 1. Headline comparison against baselines

| Policy | Samples or episodes | Held-out loss (start 90) | Start-90 RMS error (deg) | Successes (execute 30) | Pickups | Closure lateral median (mm) | Within 8.0 mm tolerance | Rollout failure breakdown |
|---|---|---|---|---|---|---|---|---|
| Fixed-square working policy (baseline) | single square | 2.5e-6 | | | | 1.7 mm | | |
| CNN, 4,800 placements (`cnn4800` baseline) | | 3.6e-4 | | 6 of 30 (best pooled 0.367) | | 5.1 mm | | |
| ACT, tuned baseline (seeds 202 / 101) | 5.1 M samples | 3.4e-4 / 3.2e-4 | | 21 / 24 of 30 | 24 / 27 | 5.2 mm | | |
| SmolVLA, October recipe (SO-100, 25 mm lift, baseline) | 0.96 M (30,000 steps) | | | 0 of 30 (2 of 30 at exec 90) | | 13 mm | | |
| pi0.5, October recipe (SO-100, 25 mm lift, baseline) | 0.32 M (10,000 steps) | 2.5e-4 | | 15 of 30 (12 of 30 at exec 90) | | | | |
| **SmolVLA fine-tune (ours, 30,000 steps)** | 0.96 M samples | 1.6e-3 (`0.001614`) | 7.23 | **14 of 30** | 17 | 13.2 mm (`13.15`) | 9 of 30 | 10 safety invalidation, 3 not settled, 3 pickup incomplete |
| **pi0.5 fine-tune (ours, 5,000 steps)** | 0.16 M samples | 7.5e-4 (`0.000751`) | 4.93 | **7 of 30** | 16 | 6.8 mm (`6.77`) | 3 of 30 | 12 safety invalidation, 9 not settled, 2 pickup incomplete |
| **SmolVLA + RL (ours, 20,000 PPO episodes)** | +20,000 episodes | | | **19 of 30** | 20 | 9.7 mm (`9.69`) | 12 of 30 | 10 safety invalidation, 0 not settled, 1 pickup incomplete |

- **SmolVLA fine-tune (`14 of 30`):** Moving SmolVLA from the October SO-100 setup (`0 of 30` at execute 30, `2 of 30` at execute 90) to the tuned SO-101 recipe raised its success count to `14 of 30` with the same model hyperparameters (`30,000` steps, batch `32`). Because its lateral closure error median remained `13.2 mm` (`13.154369001689311 mm`), identical to October's `13 mm`, the gain from `0/30` to `14/30` came from the wider SO-101 gripper tolerance, the `100 mm` lift with slow release, and replanning every 30 steps rather than tighter initial placement.
- **SmolVLA + RL (`19 of 30`):** Running `20,000` episodes (`334` iterations across `12` parallel environments, `60` episodes per iteration) of ReinFlow PPO on `out_head` starting from `smolvla_ft` raised held-out successes from `14 of 30` to `19 of 30` and pickups from `17` to `20`. RL cut the median lateral closure error from `13.15 mm` to `9.69 mm` (`9.694346355616375 mm`), increased the number of rollouts closing within the `8.0 mm` grasp tolerance from `9 of 30` to `12 of 30`, cut median yaw error from `3.36 deg` (`3.35818082924051`) to `2.11 deg` (`2.1124944061295494`), eliminated all `3` `not_settled` failures, and reduced `pickup_incomplete` from `3` to `1`.
- **pi0.5 fine-tune (`7 of 30` at `5,000` steps):** With only `5,000` steps (`0.16 M` samples, cut from October's `10,000` steps to fit the overnight window), `pi0.5` achieved half the start-90 loss of 30,000-step SmolVLA (`0.000751` vs `0.001614`) and nearly half the median lateral closure error (`6.77 mm` vs `13.15 mm`), picking up the cube `16` times (against `17` for `SmolVLA` fine-tune). However, it converted only `7` of those `16` pickups into full task successes because `9` rollouts failed on `not_settled` at the end of the episode.
- **Comparison with ACT (`21 / 24 of 30`):** Neither VLA fine-tune nor `SmolVLA + RL` beat the tuned ACT baseline (`21` and `24 of 30` successes, `24` and `27` pickups, `5.2 mm` closure median, `3.4e-4 / 3.2e-4` start-90 loss). `SmolVLA + RL` (`19 of 30`) came within `2` rollouts of ACT seed 202 (`21 of 30`), while seeing `0.96 M` supervised samples plus `20,000` RL episodes against ACT's `5.1 M` supervised samples.

---

### 2. Offline loss curves and task-stage decomposition

| Policy and step | Held-out start 0 (survey) | Held-out start 90 (descent) | Train start 90 | Held-out / train ratio (start 90) | Held-out starts 180, 270, 360 | Start-90 RMS (deg) | Poses under 2.5e-4 threshold |
|---|---|---|---|---|---|---|---|
| Fixed-square working policy (baseline) | | 2.5e-6 | | | | | |
| CNN, 4,800 placements (baseline) | | 3.6e-4 (`0.00036`) | | | | | |
| ACT, tuned baseline (seeds 202 / 101) | | 3.4e-4 / 3.2e-4 | | | | | |
| SmolVLA, step 0 | 0.097881 | 0.353728 | 0.358140 | 0.99 | 0.177254 | 107.05 | 0 of 10 |
| SmolVLA, step 5,000 | 0.000555 | 0.005347 | 0.005284 | 1.01 | 0.000536 | 13.16 | 0 of 10 |
| SmolVLA, step 10,000 | 0.000368 | 0.004624 | 0.004034 | 1.15 | 0.000309 | 12.24 | 0 of 10 |
| SmolVLA, step 15,000 | 0.000267 | 0.003105 | 0.003407 | 0.91 | 0.000247 | 10.03 | 0 of 10 |
| SmolVLA, step 20,000 | 0.000261 | 0.002682 | 0.002562 | 1.05 | 0.000169 | 9.32 | 0 of 10 |
| SmolVLA, step 25,000 | 0.000259 | 0.002379 | 0.002327 | 1.02 | 0.000169 | 8.78 | 0 of 10 |
| **SmolVLA, step 30,000 (final)** | **0.000232** | **0.001614** | **0.002068** | **0.78** | **0.000174** | **7.23** | **1 of 10** |
| pi0.5, step 0 | 0.024103 | 0.144996 | 0.137959 | 1.05 | 0.064773 | 68.54 | 0 of 10 |
| pi0.5, step 1,000 | 0.000489 | 0.006313 | 0.004782 | 1.32 | 0.000531 | 14.30 | 0 of 10 |
| pi0.5, step 2,000 | 0.000437 | 0.002998 | 0.004274 | 0.70 | 0.000286 | 9.86 | 0 of 10 |
| pi0.5, step 3,000 | 0.000329 | 0.001125 | 0.002448 | 0.46 | 0.000267 | 6.04 | 0 of 10 |
| pi0.5, step 4,000 | 0.000240 | 0.001089 | 0.002155 | 0.51 | 0.000214 | 5.94 | 1 of 10 |
| **pi0.5, step 5,000 (final)** | **0.000290** | **0.000751** | **0.001703** | **0.44** | **0.000206** | **4.93** | **1 of 10** |

- **Error concentrates at chunk start 90:** Both models learn the survey move (`heldout_start_0` = `0.000232` and `0.000290`) and the post-grasp lift/carry/set-down chunks (`heldout_starts_180_270_360` = `0.000174` and `0.000206`) to low error. The error is `3.6x` to `9.3x` larger at chunk start 90, where the policy must locate the randomized cube and aim the descent.
- **No memorization of the 7,150 training placements:** At step 30,000 for SmolVLA and step 5,000 for pi0.5, the held-out to train ratio at start 90 is `0.78` and `0.44` (near or below 1, because training samples include `random_shift: 12` and randomized `bench_physics_v2` trajectories). Neither model is overfitting the training set.
- **pi0.5 is still learning rapidly at step 5,000:** Between step 4,000 and step 5,000, `pi0.5` held-out start-90 loss dropped from `0.001089` (`1.1e-3`) to `0.000751` (`7.5e-4`), and train start-90 loss dropped from `0.002155` to `0.001703`. In October, `pi0.5` reached `2.5e-4` at `10,000` steps. Stopping at `5,000` steps left `pi0.5` half-trained.

---

### 3. Closed-loop approach correction and closure geometry (`execute 30`)

| Policy | Step 179 lateral median / RMS (mm) | Step 250 closure lateral median / RMS (mm) | RMS correction 179 to 250 (mm) | Closure max lateral (mm) | Vertical gap `dz` median (mm) | Yaw error median (deg) | Within 8.0 mm / not descended |
|---|---|---|---|---|---|---|---|
| Fixed-square working policy (baseline) | | 1.7 / | | | | | |
| CNN, 4,800 placements (baseline) | | 5.1 / | | | | | |
| ACT, tuned baseline | | 5.2 / | | | | | |
| **SmolVLA fine-tune (ours)** | 19.40 / 48.60 | 13.15 / 19.10 | 29.50 | 35.83 | 2.60 | 3.36 | 9 / 6 |
| **pi0.5 fine-tune (ours)** | not logged | 6.77 / 17.12 | not logged | 49.66 | 8.97 | 3.29 | 3 / 24 |
| **SmolVLA + RL (ours)** | 16.90 / 48.10 | 9.69 / 17.41 | 30.69 | 34.03 | 2.94 | 2.11 | 12 / 9 |

- **Replanning at execute 30 actively corrects approach error:** For `SmolVLA`, replanning every 30 steps reduces lateral RMS error by `29.50 mm` (`48.60 mm` at step 179 down to `19.10 mm` at step 250) in the fine-tune, and by `30.69 mm` (`48.10 mm` down to `17.41 mm`) after RL.
- **Why `pi0.5` had only 3 of 30 `within_tolerance` despite a `6.77 mm` lateral median:** The `within_tolerance` counter requires both lateral error under `8.0 mm` and full descent at the nominal closure frame (`not_descended` = false). For `pi0.5`, `24 of 30` rollouts were flagged `not_descended` at the closure check with a median vertical gap `dz_median_mm` of `8.97 mm` (versus `2.60 mm` for `SmolVLA` fine-tune and `2.94 mm` for `SmolVLA + RL`), even though `pi0.5` went on to pick up the cube in `16 of 30` rollouts.

---

### 4. Per-pose breakdown and rollout failure analysis (`pose_000` to `pose_009`)

| Pose | pi0.5 start-90 loss | SmolVLA start-90 loss | pi0.5 closure `lat` / `dz` (mm) | SmolVLA FT closure `lat` / `dz` (mm) | SmolVLA RL closure `lat` / `dz` (mm) | pi0.5 rollouts (3 repeats) | SmolVLA FT rollouts (3 repeats) | SmolVLA RL rollouts (3 repeats) |
|---|---|---|---|---|---|---|---|---|
| `pose_000` | 0.002103 | 0.001924 | 49.66 / 42.93 | 35.83 / 19.42 | 33.63 / 21.43 | 0/3 (3 safety: 56 limit frames) | 0/3 (3 safety: 87 limit frames) | 0/3 (3 safety: 87 limit frames) |
| `pose_001` | 0.000316 | 0.000488 | 4.57 / 7.02 | 8.84 / 2.82 | 4.06 / 2.76 | 0/3 (3 not settled, lifted 82.4 mm) | **3/3 success** | **3/3 success** |
| `pose_002` | 0.000248 | 0.000965 | 11.01 / 32.44 | 5.79 / 1.36 | 3.90 / 3.34 | 0/3 (3 safety: 36 limit frames) | **3/3 success** | **3/3 success** |
| `pose_003` | 0.000313 | 0.000377 | 9.24 to 9.70 / 8.45 to 9.68 | 17.08 to 20.23 / 2.38 to 4.60 | 15.46 to 19.14 / 1.50 to 12.04 | 0/3 (3 not settled, lifted 83.8 to 84.1 mm) | 0/3 (1 safety: 15 contact; 2 incomplete) | 0/3 (2 safety: 45, 44 contact; 1 incomplete) |
| `pose_004` | 0.001465 | 0.004581 | 6.77 / 7.12 | 7.44 / 1.93 | 2.51 / 2.89 | **3/3 success** | 0/3 (3 not settled, lifted 90.3 mm) | **3/3 success** |
| `pose_005` | 0.000453 | 0.003517 | 5.40 to 6.08 / 3.73 to 4.10 | 14.94 to 16.12 / 0.96 to 3.04 | 9.02 to 15.48 / 1.96 to 6.11 | **3/3 success** | 0/3 (2 safety: 27, 5 contact; 1 incomplete) | **1/3 success** (2 safety: 26, 4 contact) |
| `pose_006` | 0.000253 | 0.000253 | 6.26 to 7.48 / 7.92 to 9.16 | 12.13 to 12.69 / 1.35 to 1.70 | 8.61 to 11.94 / 1.78 to 2.33 | **1/3 success** (2 incomplete, lifted 85.5 to 85.8 mm) | **3/3 success** | **3/3 success** |
| `pose_007` | 0.000848 | 0.000175 | 3.72 to 3.77 / 10.39 to 10.59 | 13.62 to 15.45 / 3.42 to 3.72 | 10.00 to 10.07 / 2.99 to 3.13 | 0/3 (3 safety: 31 to 32 contact frames) | **2/3 success** (1 safety: 2 contact frames) | **3/3 success** |
| `pose_008` | 0.000721 | 0.003041 | 2.14 / 12.41 | 0.82 to 5.56 / 0.86 to 1.55 | 6.76 / 0.78 | 0/3 (3 safety: 30 contact frames) | **3/3 success** | **3/3 success** |
| `pose_009` | 0.000788 | 0.000814 | 8.77 to 9.55 / 2.70 to 3.32 | 34.68 to 35.35 / 10.87 to 11.10 | 33.65 to 34.03 / 11.63 to 12.19 | 0/3 (3 not settled, lifted 82.4 to 82.9 mm) | 0/3 (3 safety: 141 to 144 contact frames) | 0/3 (3 safety: 143 to 147 contact frames) |

Reading `evaluation.json` across all 30 rollouts for each model explains the exact failure mechanisms:
1. **What happened in `pi0.5`'s 9 `not_settled` failures (`pose_001`, `pose_003`, `pose_009`):** In all 9 rollouts, `evaluation.json` records `pickup_completed: true` (maximum height gain `0.08239 m` to `0.08408 m`), `entered_placement_region: true`, `released: true` (`final_gripper_act` `0.8476` to `0.8505`), and `retreated: true`, with the cube resting stationary on the table (`final_linear_speed_m_s` at most `2.37e-6 m/s`, `final_cube_position` z = `0.01086 m` to `0.01089 m`). However, all 9 had `settled: false` and `settled_frames: 0` up to the `480`-action timeout (`timed_out: true`) because the final cube xy coordinates (`[-0.1327, 0.1733]` on `pose_001`, `[0.0496, 0.2344]` on `pose_003`, `[0.1049, 0.1470]` on `pose_009`) landed outside the settled-in-square tolerance around `square_center_xy` (`[-0.1352, 0.1921]`, `[0.0570, 0.2522]`, `[0.1196, 0.1683]`). The same failure happened to `SmolVLA` fine-tune on `pose_004` (`final_cube_position` xy `[-0.1648, 0.1191]` against `square_center_xy` `[-0.1723, 0.1364]`), which RL fixed on all 3 repeats (`[-0.1699, 0.1289]`).
2. **Two distinct causes of `safety_invalidation`:**
   - **Joint-limit violations (`limiting_frames`):** On `pose_000`, all three policies hit joint limits (`56` limiting frames per rollout for `pi0.5`, `87` for `SmolVLA` fine-tune and `SmolVLA + RL`) and never descend (`maximum_height_gain_m: 0.0`). `pi0.5` also hit `36` limiting frames per rollout on `pose_002` despite `pose_002` having its lowest offline start-90 loss (`0.000248`).
   - **Unsafe table or cube contact (`unsafe_contact_frames`):** On `pose_007` and `pose_008`, `pi0.5` had very small lateral closure error (`3.72` to `3.77 mm` and `2.14 mm`) but large vertical closure error (`dz` = `10.39` to `10.59 mm` and `12.41 mm`) and accumulated `31` to `32` and `30` unsafe contact frames per rollout. On `pose_009`, both `SmolVLA` fine-tune and `SmolVLA + RL` missed laterally by `33.65` to `35.35 mm` and accumulated `141` to `147` unsafe contact frames per rollout.
3. **Where RL helped `SmolVLA` and where it could not:** Because ReinFlow PPO only updated `out_head` (`scope: "out_head"`) with a small clipping range (`clip: 0.05`, `target_kl: 0.1`), it refined trajectories that were already near the cube: `pose_004` went from `0/3` (`not_settled`, lateral error `7.44 mm`) to `3/3` (`success`, lateral error `2.51 mm`), `pose_007` went from `2/3` to `3/3` (lateral error `15.01 mm` down to `10.05 mm`), and `pose_005` went from `0/3` to `1/3` (plus repeat 1, which completed pickup with `96.1 mm` lift and `38` settled frames before 4 unsafe contact frames invalidated it). RL could not fix the gross reach/collision failures on `pose_000`, `pose_003`, or `pose_009`.

---

### 5. RL progression on the 20-pose monitor suite (`rl_smolvla`)

| RL episodes completed | Success rate (`20` poses) | Pickup rate | Grasp rate | Invalidation rate | Mean return | Mean max lift (mm) | Closure lateral median (mm) | Within 8.0 mm rate | Violation counts (contact / rel limit / cmd bound) |
|---|---|---|---|---|---|---|---|---|---|
| 0 (`start_monitor`, baseline) | 0.50 | 0.50 | 0.55 | 0.40 | 0.251 | 51.37 | 7.41 | 0.40 | 4.0 / 3.0 / 1.0 |
| 1,920 | 0.55 | 0.65 | 0.65 | 0.35 | 0.385 | 66.42 | 9.83 | 0.25 | 2.0 / 3.0 / 2.0 |
| 4,320 | 0.25 | 0.30 | 0.30 | 0.60 | -0.262 | 29.70 | 13.00 | 0.25 | 10.0 / 2.0 / 0.0 |
| 6,720 | 0.55 | 0.55 | 0.60 | 0.25 | 0.467 | 57.57 | 9.91 | 0.35 | 2.0 / 2.0 / 1.0 |
| 13,440 | 0.45 | 0.60 | 0.70 | 0.30 | 0.334 | 64.99 | 8.08 | 0.40 | 3.0 / 2.0 / 1.0 |
| 15,840 | 0.45 | 0.45 | 0.50 | 0.45 | 0.137 | 48.06 | 6.68 | 0.40 | 5.0 / 3.0 / 1.0 |
| 16,800 | 0.50 | 0.65 | 0.70 | 0.35 | 0.344 | 67.74 | 9.27 | 0.30 | 4.0 / 1.0 / 2.0 |
| **20,000 (final)** | **0.60** | **0.65** | **0.70** | **0.30** | **0.496** | **65.12** | **7.14** | **0.50** | **3.0 / 2.0 / 1.0** |

---

### 6. What held and what failed

No separate pre-registered predictions file was recorded before this run. Evaluating the expectations written in the experiment note:
- **Held (the tuned ACT recipe lifts SmolVLA from near-zero to working):** SmolVLA rose from `2/30` (`0/30` at execute 30) in October to `14/30` on the SO-101 recipe without changing model settings.
- **Held (SmolVLA's fine-tune gain is task and control tolerance, not better single-shot placement):** Its median lateral closure error remained `13.2 mm` (`13.15 mm`), the same as October's `13 mm`.
- **Held (ReinFlow PPO RL improves SmolVLA):** 20,000 RL episodes raised held-out successes from `14/30` to `19/30`, monitor success rate from `0.50` to `0.60`, pickups from `17/30` to `20/30`, and cut median lateral closure error from `13.15 mm` to `9.69 mm`.
- **Held (pi0.5 localizes better per sample than SmolVLA, and 5,000 steps is not its ceiling):** With one-sixth the samples (`0.16 M` vs `0.96 M`), `pi0.5` reached half the start-90 loss (`7.5e-4` vs `1.6e-3`) and nearly half the median lateral closure error (`6.77 mm` vs `13.15 mm`), with its loss still dropping (`1.1e-3` at step 4,000 to `7.5e-4` at step 5,000).
- **Failed (neither VLA beats the tuned ACT baseline):** ACT scores `21/30` and `24/30` (`5.2 mm` closure median, `3.4e-4 / 3.2e-4` start-90 loss), ahead of `SmolVLA + RL` (`19/30`), `SmolVLA` fine-tune (`14/30`), and 5,000-step `pi0.5` (`7/30`).
- **Failed (pi0.5 at 5,000 steps underperformed on full task completion at `7/30`):** Despite `16/30` pickups, `pi0.5` lost `9` rollouts to `not_settled` (setting the cube down slightly outside the settled-in-square tolerance) and `12` rollouts to `safety_invalidation`.
- **Failed (output-head RL did not reduce held-out safety invalidations, `10/30` before and `10/30` after RL):** Updating only `out_head` could not stop the arm from hitting joint limits on `pose_000` (`3/3`) or table contact on `pose_009` (`3/3`) and `pose_003` (`2/3`).

---

### 7. What this means for the decision, limits, and what to run next

- **Decision:** Keep the tuned **ACT** policy (`21` to `24 of 30`) as the primary policy for the real SO-101 arm right now. Do not replace ACT with `SmolVLA` (`14/30` fine-tune, `19/30` after RL) while `10 of 30` held-out rollouts still trigger safety invalidations. Before judging `pi0.5` or spending RL episodes on it, finish its supervised training to `10,000` steps.
- **Limits of this evidence:**
  1. **Simulation only:** None of the three Experiment 56 checkpoints (`smolvla_ft`, `pi05_ft`, `rl_smolvla`) was tested on the real arm or on the recorded real-frame gate.
  2. **Unequal sample budgets:** ACT trained on `5.1 M` samples (`160,000` steps), `SmolVLA` on `0.96 M` samples (`30,000` steps) plus `20,000` RL episodes, and `pi0.5` on `0.16 M` samples (`5,000` steps).
  3. **Single training seed (`seed: 202`) and 10 held-out poses (`30` rollouts):** On `10` poses x `3` repeats, rollout counts have multi-rollout variance across seeds and checkpoints (as seen in the 20-pose RL monitor history oscillating between `0.25` and `0.60`).
  4. **Missing per-frame telemetry and job-queue exit logs in the analysis bundle:** The `evaluation.json` files record final cube positions and frame counts (`limiting_frames`, `unsafe_contact_frames`, `settled_frames`), but the per-rollout `telemetry/*.json` files and the original task-spooler table for job IDs `256` to `264` (`tsp.txt` only lists jobs `0` to `8` after a queue reset) were not included in `in/`, so frame-by-frame trajectory traces during set-down could not be inspected directly.
- **Ordered next steps:**
  1. **Resume `pi0.5` LoRA fine-tuning from `5,000` to `10,000` steps (`0.32 M` samples):** Watch `heldout_start_90` (target: under `3.4e-4`, matching ACT and October's `2.5e-4`), `dz_median_mm` at closure (target: under `4.0 mm` from `8.97 mm`), and whether the `9` `not_settled` rollouts on `pose_001`, `pose_003`, and `pose_009` land inside the target square and settle.
  2. **Inspect the per-rollout telemetry (`telemetry/pi05_exec30.pose_001.repeat0.json`, `pose_003`, `pose_009`) and `pi05_ft/progress.json`:** Check the exact set-down trajectory and verify why `pi05_ft/progress.json` logged `"step": 4000` alongside the step-5,000 `final` evaluation block.
  3. **Unfreeze more than `out_head` or add a joint-limit/contact penalty shaping before further RL on `SmolVLA`:** Watch the `10 of 30` held-out `safety_invalidation` rollouts (`pose_000`, `pose_003`, `pose_009`), where `out_head`-only PPO left `limiting_frames` (`87` on `pose_000`) and `unsafe_contact_frames` (`143` to `147` on `pose_009`) unchanged.
