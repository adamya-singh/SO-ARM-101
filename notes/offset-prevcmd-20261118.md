# Joint-offset randomization and previous-command input, against a tuned baseline (2026-11-18)

Finished 2026-11-22; the results are in the last section of this note. Queue: `simulation_code/queue_offset_prevcmd_20261118.sh` (tsp 150-165,
then the ablations it queues itself), output `experiments/offset_prevcmd_20261118/`, gated on
`simulation_code/rehearse_offset_prevcmd_20261118.sh` (tsp 150). The first queue (105-149) was withdrawn: its
rehearsal (106) failed on the batch loader's fixed 7-column state buffer (fixed), and the user asked for a
hyperparameter sweep of the baseline first.

Reference (tsp 105): the existing high-lift ACT under real_like_offset_v1 scores 7/30, 19 pickups, 9.4 mm lateral
median, against 8/30, 20 pickups, 7.8 mm under real_like_v1. A +-3 deg joint-zero error costs a policy that never
trained on it about one pickup in 30 and 1.6 mm.

## Tuned baseline: two-stage sweep (user, 2026-11-18)

Stage 1: eight settings, 15k steps, seed 202, on capture A: learning rate {3e-5, 1e-4, 3e-4}, backbone rate
{1e-5, 3e-5, 1e-4}, KL weight {10, 1}; ranked by held-out loss at chunk start 90 (`tools/sweep_select.py`).
Stage 2: the winner and the current settings (1e-4, 1e-5, 10) at 40k steps; the lower held-out loss becomes
`sweep/tuned.json` and that run is `base_s202`. The ablations below are then queued with those settings. Training
length is read from the full-length held-out curves, not swept.

## Why

- The arm's joint zeros were set by eye; the model's jaw height is 10-35 mm off at some poses (2026-10-04 contact,
  episode 10). Randomizing the zero error in training should make the policy take the cube's position from the image
  and not from the joint readings.
- Physics-v2 policies cannot tell a soft servo from a stiff one from a single frame (analysis 2026-11-09); the
  pre-registered lever is the previous commands as an input.

## What was built

- **Joint offsets** (`contracts/physics.py` `joint_offset_deg`): per episode, a fixed offset uniform in +-3 deg on each
  of the five arm joints, own draw stream. The policy's frame is shifted (the state it reads, the commands it sends and
  the capture records); the teacher, the gate and the physics stay in the true frame (`adapter.joint_offset_act`,
  `apply_policy_command`, `PrivilegedStagedController._act_from_plan`). Residual targets (command minus state) are
  unchanged by construction; only the state input stops being a reliable position. Regimes `bench_physics_v3`
  (v2 + offsets) and `real_like_offset_v1` (real_like_v1 + offsets). Older regimes keep their identity and draws.
  Teacher under real_like_offset_v1: 27/30, the same as real_like_v1.
- **Previous-command input** (`vla_finetune.py --prev-commands K`, ACT only): the state gains, for k = 1..K, the
  command sent k actions ago minus the current joints (`CaptureSource.state`, `residual_policy.lead_features`,
  `VLAConfig.extra["prev_commands"]`). K = 3 here. The physical runner does not pass the history yet: a
  prev-command checkpoint cannot run live until it does.
- **Tuned baseline**: the live high-lift ACT recipe with its known defect fixed. `BenchConfig.teacher_release_rows`
  (scene `bench_pick_replace_v1_so101_servo_lift100_rel30`, XML identical to lift100): the gripper opens over 30
  actions instead of 18, taken out of the hold at the top of the lift. Teacher clean 10/10 (lift100: 9/10),
  real_like_v1 27/30.

## Design

Two by two, seeds 202 and 101 each, ACT fine-tune 40k steps, 7200 placements, suite seed 17 (captures A and B hold
the same placements and physics draws; B adds the offsets), execute 30:

| arm | capture | input |
|---|---|---|
| base | A: bench_physics_v2 | state |
| prev | A | state + 3 previous commands |
| off | B: bench_physics_v3 | state |
| offprev | B | state + 3 previous commands |

Cells (slow-release scene): servo, real_like_v1, real_like_offset_v1, bench_physics_v2. Reference: the existing
high-lift ACT under real_like_offset_v1 (tsp 105).

Read it as: pooled over the two seeds (60 rollouts per cell), on successes, pickups and lateral error in mm.
- Offsets help if `off` beats `base` under real_like_offset_v1 without losing under real_like_v1.
- Previous commands help if `prev` beats `base` under bench_physics_v2 (the cell where servo identity matters).
- The baseline is sound if `base` under real_like_v1 is near its pickup count (the release artifact is gone).

## Results (finished 2026-11-22, written 2026-11-25)

Every number below is read from `experiments/offset_prevcmd_20261118/`: `matrix.json` (successes),
`evals/<run>_exec30__<cell>/result.json` (pickups, lateral median), `<run>/heldout_curve.jsonl` last line (held-out
loss), `sweep/winner_stage1.json` and `sweep/tuned.json` (sweep). Losses are rounded to three figures and
millimetres to one decimal. Sums over the two seeds are added by hand, because the `pooled` block of `matrix.json`
is empty.

Words used here. A *rollout* is one simulated attempt at the task. Each "of 30" is 10 held-out placements (cube
positions the policy never trained on) run 3 times. A *success* is a full pick, lift and replace with no safety
frame. A *pickup* is a rollout where the cube was lifted, whether or not the rest went well. The *lateral median* is
the median sideways distance between the jaws and the cube just before closing (grasp tolerance is 8 mm). The
*held-out loss* is the prediction error on the held-out placements, measured for the plan that starts at action 90.
A *cell* is one servo setting used at evaluation: `servo` (the measured servo model), `real_like_v1` (soft servos
like the real arm), `real_like_offset_v1` (the same plus joint-zero errors), `bench_physics_v2` (servos of random
strength).

### Sweep outcome

Stage 1 (15,000 steps, seed 202, ranked by held-out loss): the winner was learning rate 3e-4, backbone rate 1e-4,
KL weight 10, with loss 0.000382, 24/30 successes, 27 pickups, 4.6 mm. The current settings (1e-4, 1e-5, 10) came
fifth of eight: 0.000508, 21/30, 21 pickups, 7.2 mm. Two settings scored 27/30 (1e-4, 1e-4 with KL 1 and with
KL 10) but ranked third and fourth on loss, and loss was the rule.

Stage 2 (40,000 steps): the current settings reached 0.000339, 21/30, 24 pickups, 5.6 mm. The stage-1 winner
diverged: its held-out loss was 0.000964 at step 2,500, 0.164 at step 7,500 and 0.173 at the end, with 0/30,
0 pickups, 136.1 mm. So the current settings were kept, and that run is `base_s202`.

### The 2 x 2

Successes of 30, pickups in brackets, lateral median after the semicolon. Baseline rows first.

| arm | seed | servo | real_like_v1 | real_like_offset_v1 | bench_physics_v2 |
|---|---|---|---|---|---|
| base | 202 | 21 (24); 5.2 mm | 17 (18); 6.5 mm | 15 (17); 7.6 mm | 14 (18); 9.9 mm |
| base | 101 | 24 (27); 5.2 mm | 18 (18); 10.5 mm | 19 (20); 6.6 mm | 16 (18); 8.7 mm |
| prev | 202 | 24 (24); 4.5 mm | 13 (19); 5.3 mm | 14 (19); 6.8 mm | 14 (16); 8.6 mm |
| prev | 101 | 21 (24); 4.7 mm | 17 (22); 5.3 mm | 19 (23); 4.7 mm | 13 (16); 8.1 mm |
| off | 202 | 16 (22); 4.1 mm | 15 (17); 4.3 mm | 14 (17); 5.4 mm | 13 (14); 10.7 mm |
| off | 101 | 21 (27); 3.4 mm | 19 (21); 4.5 mm | 16 (18); 4.2 mm | 10 (15); 10.2 mm |
| offprev | 202 | 21 (21); 6.3 mm | 15 (20); 5.4 mm | 18 (21); 4.9 mm | 15 (17); 6.2 mm |
| offprev | 101 | 18 (21); 5.5 mm | 13 (18); 4.2 mm | 16 (20); 4.4 mm | 17 (19); 6.4 mm |

Summed over the two seeds (of 60), successes with pickups in brackets:

| arm | servo | real_like_v1 | real_like_offset_v1 | bench_physics_v2 |
|---|---|---|---|---|
| base | 45 (51) | 35 (36) | 34 (37) | 30 (36) |
| prev | 45 (48) | 30 (41) | 33 (42) | 27 (32) |
| off | 37 (49) | 34 (38) | 30 (35) | 23 (29) |
| offprev | 39 (42) | 28 (38) | 34 (41) | 32 (36) |

Reference (old high-lift ACT, old fast-release scene): 7/30, 19 pickups, 9.4 mm under real_like_offset_v1; 8/30,
20 pickups, 7.8 mm under real_like_v1 (`high_lift_20261112/evals/lift_act_ft_exec30__servo_real_like_v1`).

### Held-out loss (step 40,000, chunk start 90)

| arm | seed 202 | seed 101 |
|---|---|---|
| base | 0.000339 | 0.000324 |
| prev | 0.000261 | 0.000245 |
| off | 0.000359 | 0.000284 |
| offprev | 0.000260 | 0.000235 |

The two arms with previous commands have the lowest loss on both seeds. `off` is measured on capture B's held-out
set and `base` on capture A's, so those two rows are not on exactly the same data.

### The three reading rules

1. **Offsets help if `off` beats `base` under real_like_offset_v1 without losing under real_like_v1: FAILED.**
   Under real_like_offset_v1, `off` has 14 + 16 = 30 successes of 60 against 15 + 19 = 34 for `base`, and
   17 + 18 = 35 pickups against 17 + 20 = 37. Only the lateral median is better: 5.4 and 4.2 mm against 7.6 and
   6.6 mm. Under real_like_v1, `off` has 15 + 19 = 34 against 17 + 18 = 35 (pickups 38 against 36; lateral 4.3 and
   4.5 mm against 6.5 and 10.5 mm), so it does not lose much there, but the first half of the rule is not met.
2. **Previous commands help if `prev` beats `base` under bench_physics_v2: FAILED.** `prev` has 14 + 13 = 27
   successes of 60 against 14 + 16 = 30, and 16 + 16 = 32 pickups against 18 + 18 = 36. Lateral median 8.6 and
   8.1 mm against 9.9 and 8.7 mm.
3. **The baseline is sound if `base` under real_like_v1 is near its pickup count: HELD.** Seed 202: 17 successes
   from 18 pickups. Seed 101: 18 from 18. The old policy had 8 successes from 20 pickups in this cell. The slow
   release removed the gap between lifting the cube and finishing the task.

### What it means

- Neither change earned its place on the measure the note chose. Training with joint offsets did not give more
  successes when offsets are present, and the last three commands did not give more successes under random servos.
- The reference already said there was little to win from offsets: the old policy lost 1 success, 1 pickup and
  1.6 mm to them. The new baseline, which never saw offsets, scores 34 of 60 with them and 35 of 60 without.
- The `off` arm does place the jaws closer to the cube sideways than `base` in three of the four cells, on both
  seeds (not in bench_physics_v2).
  That did not turn into successes here. It may matter on the real arm, where the zeros are wrong. This experiment
  does not test that.
- Previous commands lower the held-out loss on both seeds but not the success count. A lower loss was not a better
  policy in this run.
- Both changes together (`offprev`) are level with `base` in the two test cells: 34 against 34 under
  real_like_offset_v1 and 32 against 30 under bench_physics_v2, with the lowest bench_physics_v2 lateral median
  (6.2 and 6.4 mm). It is behind in the other two cells (39 against 45, and 28 against 35).
- What the experiment did deliver is the tuned baseline: the slow release took real_like_v1 from 8/30 to 17 and
  18 of 30.

### Limits

- Two seeds per arm. The two `base` seeds differ by 3, 1, 4 and 2 successes of 30 across the four cells (in the
  column order above). The gaps in the two test cells are 4 of 60 (rule 1) and 3 of 60 (rule 2). Two seeds cannot
  separate the arms.
- The same checkpoint scored differently when it was evaluated twice in the `servo` cell: once at the end of
  training (`<run>/result.json`, `rollouts`) and once in the matrix (`evals/`). Same scene hash and suite name in
  both identity blocks. `base_s101`: 21/30 with 24 pickups, then 24/30 with 27. `prev_s101`: 24/30, then 21/30.
  `off_s202`: 15/30 with 21 pickups, then 16/30 with 22. The other five runs gave the same counts. So one
  evaluation is itself uncertain by up to 3 of 30. I did not find the cause.
- Every 30 is 10 placements run 3 times, so it is closer to 10 independent tries than to 30.
- Almost every failure is a `safety_invalidation` (the rollout is cancelled because the safety gate flagged a
  frame), for example 13 and 12 of the `base` failures under real_like_v1. The counts say more about safety frames
  than about missed grasps. The limiter-frame analysis done for the high-lift run was not repeated here.
- Both captures hold 7,150 placements, not the 7,200 in the design (`capture_v2/progress.json`,
  `capture_v3/progress.json`).
- The sweep used one seed. The stage-1 winner diverged at 40,000 steps, so its 15,000-step rank said nothing about
  the full-length run. The current settings were fifth of eight at 15,000 steps.
- Simulation only. No policy from this experiment ran on the real arm. The physical runner still does not pass the
  command history, so `prev` and `offprev` cannot run live.
- No pooled lateral median is given, because a median of two medians is not a median of 60 rollouts.

### What to run next

- Nothing more is needed to close this experiment. Keep `base` as the recipe (it is already the data and scene of
  the 2026-11-22 VLA fine-tunes).
- Do not adopt `off` or `prev` alone. If the lower sideways error of the `off` arm is worth chasing for the real
  arm, the cheap first step is to find out why one checkpoint gives two different scores in the same cell, then
  add seeds (at least a third for `base` and `off`), before any live run.
- `offprev` is the only arm that matches `base` in both test cells. Trying it live needs the runner to pass the
  command history first.
