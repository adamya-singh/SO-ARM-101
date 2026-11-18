# Feature probes: testing the recipe on checkpoints we already have (2026-11-18)

Status: built, tested, queued (see "Queue"). No learned policy has been probed yet; everything under
"Predictions" was written before the first learned-policy result.

## What is being tested

The recipe says: a policy trained by imitation learns only the features its data forces, so for each
feature we want (a) a data intervention that forces it and (b) a probe that says whether it is there.
That is three separate claims, and each can fail on its own:

1. the intervention produces the feature;
2. the probe detects the feature;
3. the feature is what buys success under the matching change.

A success count cannot test any of them. A policy that looks at later frames and one that only
remembers the first frame both succeed when nothing changes after the first frame. A probe changes the
world in the middle of an episode, where the two must behave differently, and compares the episode
with the same episode left alone. The simulator is deterministic, so the pair differs only by the change.

## The probes

Code: `src/so_arm101_v2/simulation/probes.py`, tool `tools/feature_probes.py`, tests `tests/test_probes.py`.

| Probe | What changes, and when | Feature | Reading |
|---|---|---|---|
| `shift_placement` | Square and cube move 20 mm together at action 100, after the survey frame (action 90) | Reach follows later frames | Follow gain |
| `shift_cube` | The cube alone moves 10 mm on its square at action 100 | Follows the cube, not the square | Follow gain |
| `drop_cube` | The cube is put back on its square once it has risen 15 mm | Notices an empty gripper, retries | Regrasp rate |
| `move_square` | The square moves 20 mm once the cube has risen 15 mm | Set-down follows the square | Follow gain |
| `swap_elbow` | At action 100 the elbow servo becomes the other kind (measured soft kp 6.5, or nominal stiff kp 50) | Copes with a servo that changed | Pickup retention |

- **Follow gain** is the displacement of the jaws at action 250 (just before they close), or of the
  cube at the end for `move_square`, between the probed and the unprobed episode, projected on the
  applied shift and divided by its length. 0 means the change was ignored, 1 means it was followed.
- **Pickup retention** is pickups under the probe over pickups in the unprobed episodes.
- **Reaction** is reported for every probe: the largest change in any commanded joint after the
  change, against the unprobed episode. Zero means the policy did exactly what it would have done anyway.

Verdict thresholds, fixed in `tools/feature_probes.py`:

- follow gain: present at 0.5 or more with the 95% interval above 0.25, absent at 0.2 or less;
- regrasp rate: present at 0.5 or more, absent at 0.05 or less;
- pickup retention: present at 0.8 or more, absent at 0.5 or less.

## What makes the measurement sound

- **Reference controls.** The scripted teacher is open-loop, so its follow gain must be 0. The same
  teacher planned with the change known in advance must read 1 and still succeed. If they do not, the
  probe is wrong and no policy result is read. First check on the old 10-pose suite, nominal SO-101
  scene: plain teacher +0.08 and 0/10 successes under `shift_placement`, informed teacher +0.99 and 10/10;
  `move_square` 0.00 against 0.99.
- **Teacher-admitted displacements.** On the first 100-placement run some 20 mm shifts landed where
  the teacher has no IK solution. A shift plan per scene (`feature_probes.py plan`) now keeps, for each
  placement, the first candidate direction the informed teacher completes; every policy on that scene
  gets the same vector. Placements with no admissible candidate are left out of that probe.
- **Enough placements.** The suites used so far are 10 poses run 3 times, which is 10 samples. The
  probe suite is 100 distinct teacher-screened placements (suite seed 31, unused elsewhere), one
  episode each. Gains carry a bootstrap interval over placements; CNN results are pooled over 3 seeds.
- **Same loop as the certified evaluation.** The probe loop is `_run_pick_place_rollout` with the
  change applied before the frame of its action is read; a test pins the unprobed loop to it. It keeps
  stepping after the task evaluator finishes so a late retry is still seen.

## Checkpoints

| Name | Checkpoint | Intervention it carries |
|---|---|---|
| `base_act`, `base_cnn_s*` | SO-101 benchmark ACT; CNN seeds 202 / 101 / 303 | none (baseline) |
| `pert_act`, `pert_cnn_s*` | perturbed-teacher ACT and CNNs (2026-11-08) | teacher recentres at 20 mm after a shifted descent |
| `physv2_act`, `physv2_cnn_s*` | physics v2 ACT and CNNs | per-joint servo gains randomized, soft to stiff |
| `lift_act`, `fair_cnn_s*` | high-lift ACT, fair-comparison CNNs | physics v2 plus the 100 mm lift (current live candidates) |

ACT runs at execute 90 and 30, the CNNs at 90 (the fair CNNs also at 30). Scenes: nominal SO-101,
measured-servo SO-101, and the high-lift servo scene for the last row.

## Predictions (registered before the runs)

Machine-readable copy: `notes/feature-probes-20261118-predictions.json`; the report prints held or failed.

| Probe | Policy | Prediction | Why |
|---|---|---|---|
| `shift_placement` | baseline CNN | absent | 2026-09-18: the action-180 chunk corrects nothing |
| `shift_placement` | perturbed CNN | present | its data contains the correction |
| `shift_placement` | physics v2 CNN | absent | nothing in its data forces a second look |
| `shift_placement` | baseline ACT, execute 90 | absent or partial | no evidence either way |
| `shift_placement` | perturbed ACT, execute 90 | present | trained to recentre |
| `shift_placement` | ACT, execute 30 | partial or present | reads four frames before closing |
| `swap_elbow`, nominal scene | baseline ACT | absent | 0 to 12 of 30 on the soft servo scene |
| `swap_elbow`, nominal scene | baseline CNN | absent or partial | 9 of 30 soft against 15 of 30 nominal |
| `swap_elbow`, both scenes | physics v2 policies | present | 18 to 24 of 30 on soft and stiff elbows |
| `drop_cube` | every policy | absent | no training episode contains a failed grasp |
| `move_square` | every policy | absent | no intervention targets it |

Link 3 is tested across cells: under `shift_placement`, pickup retention should rise with follow gain
(rank correlation above 0.5 over the learned-policy cells), with gain 0.5 or more keeping at least 70%
of the pickups and gain 0.2 or less keeping at most 30%.

What would count against the recipe:

- a perturbed-teacher policy with gain 0.2 or less (the intervention did not produce the feature);
- a baseline CNN with gain 0.5 or more (the feature needs no intervention);
- high gain without retained pickups, or retained pickups without gain (the feature is not what
  buys success);
- any policy that regrasps after `drop_cube` or follows `move_square` (features appear that no data
  forced; good news for the policy, bad for the claim that data alone decides).

## Limits

- `swap_elbow` measures coping with a changed servo, not identifying it. A policy can cope by being
  robust. Identification needs the previous-command policies (tsp 150-165), probed the same way after.
- On the measured-servo scene the plain teacher keeps 81 of 100 successes when the elbow stiffens: its
  servo compensation was computed for the soft elbow. That is the open-loop reference for that cell.
- `drop_cube` and `move_square` have no intervention to compare, so they can only show absence here.
  They become tests of the recipe once a retry capture exists.
- Simulation only. The real-arm version is the same probes by hand (move the cube after the survey frame).

## Queue

`simulation_code/queue_feature_probes_20261118.sh`, output `experiments/feature_probes_20261118`:

| Stage | Content | Where in the queue |
|---|---|---|
| plan | shift plans for the three scenes (CPU) | front |
| rehearsal | 3 placements through teacher, one CNN, one ACT cell, and the report; gates the rest | front |
| controls | teacher plain and informed on three scenes (CPU) | front |
| headline | `shift_placement` and `swap_elbow` for baseline, perturbed and physics v2 policies, nominal scene (about 2.5 h) | front |
| rest | all other probes and the servo and high-lift scenes (about 6 h) | back |

Read `experiments/feature_probes_20261118/matrix.json` (or the report's printed table) when the headline
stage finishes: controls first, then the predictions, then link 3.
