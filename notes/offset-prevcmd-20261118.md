# Joint-offset randomization and previous-command input, against a tuned baseline (2026-11-18)

Queued, no results yet. Queue: `simulation_code/queue_offset_prevcmd_20261118.sh` (tsp 105-149), output
`experiments/offset_prevcmd_20261118/`, gated on `simulation_code/rehearse_offset_prevcmd_20261118.sh` (tsp 106).

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
