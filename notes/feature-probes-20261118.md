# Feature probes: testing the recipe on checkpoints we already have (2026-11-18)

Status: finished. All stages ran between 2026-11-18 and 2026-11-21; see "Results of the full matrix" at the end.
Everything under "Predictions" was written before the first learned-policy result.

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

## Results of the full matrix (read 2026-11-25)

Written with the `experiment-analyze-explain` method (measure, decide, record, explain). Everything in this
section is new; the text above is unchanged except the status line.

Sources. `matrix.json` is the report file of this experiment (written 2026-11-21 16:42). Each cell folder under
`cells/` holds, per probe, `rows_<probe>.json` (one line of outcome per episode) and `trace_<probe>.npz` (jaw,
cube and command at every action). A number marked "matrix" is copied from `matrix.json`. A number marked
"recomputed" was made on 2026-11-25 from the stored rows and traces with the repository's own functions
(`probes.paired_metrics`, `feature_probes.verdict`, and the pooling of `cmd_report`). No simulation was run. Where
`matrix.json` has the same number, the recomputation matches it exactly. The recomputed numbers are in no file
yet: `matrix.json` needs to be regenerated (see "What we do next").

### The setup, in one paragraph

A policy is the network that drives the arm. A probe changes the world in the middle of an episode and compares
with the same episode left alone (the "unprobed" episode). Follow gain is how far something moved between the two
episodes, along the change, as a fraction of the change: 0 means ignored, 1 means followed. Before the run we wrote
down 55 predictions of the form "this policy on this scene will show (or not show) this skill". The headline
stage read 13 of them and reported "12 of 13 held". This section reads all 55. The decision that hinges on it:
can we keep planning on "a skill exists only where a data change was aimed at it" (experiment 53 was built on
that), or do some skills come with the standard data?

### Step 1. Were all 55 predictions read?

*Why.* A score of "12 of 13" says nothing about the 42 predictions outside it.

*What we did.* Counted the `predictions` list of `matrix.json`, then looked at why some entries have no verdict.

*What it says.*

| Reading | Predictions | Held | Failed | Not read |
|---|---|---|---|---|
| Headline stage (notebook, 2026-11-19) (baseline) | 13 | 12 | 1 | 0 |
| `matrix.json` as it stands | 55 | 26 | 11 | 18 |
| All 55, the 18 recomputed from stored rows and traces | 55 | 32 | 23 | 0 |

The 18 unread ones are `drop_cube` and `move_square` for the 9 policy groups on the nominal scene. Nothing is
missing and no control invalidates them. The episodes ran on 2026-11-19 (their `rows_*.json` and `trace_*.npz`
files are in the cell folders). Their numbers were then dropped from each cell's `result.json`. Every nominal-scene
`result.json` carries the time 2026-11-20 03:44 and lists only `shift_placement`, `swap_elbow` and `shift_and_swap`.
That is the probe list experiment 53 passes for the same cell folders
(`queue_recursive_generalization_20261119.sh`, stage `combo`), and `feature_probes.py run` writes `result.json`
with only the probes named in that call. The report then had no verdict to compare.

The 23 failures by probe:

| Probe | Predictions | Held | Failed | What the failures are |
|---|---|---|---|---|
| `shift_placement` | 8 | 7 | 1 | baseline ACT at execute 30 was predicted to follow partly and does not (+0.06). Known since the headline. |
| `swap_elbow` | 9 | 9 | 0 | |
| `drop_cube` | 19 | 16 | 3 | three nominal-scene groups read "partial". Step 4 shows this is the probe, not a retry. |
| `move_square` | 19 | 0 | 19 | 18 groups follow the square. 1 group has nothing to compare and the report counts it as failed. |

The earlier pass said "10 of the 11 failures are `move_square`". That is right as counted, but one of the 10 is
the baseline ACT at execute 30 on the servo scene. It picks up 0 of 100 unprobed, so there is no pair to compare
and its verdict is "n/a". It is unreadable, not a failure of the prediction. On the same cell `drop_cube` "held"
on 1 drop, which is as empty.

All 55, with the numbers behind each ("matrix" or "recomputed" in the last column; CNN rows are pooled over 3 seeds):

| Probe | Scene | Policy | Predicted | Observed | Numbers | Result | Source |
|---|---|---|---|---|---|---|---|
| `shift_placement` | nominal | baseline ACT, execute 90 | absent or partial | absent | gain +0.04, pickups 14 of 87 | held | matrix |
| `shift_placement` | nominal | baseline ACT, execute 30 | partial or present | absent | gain +0.06, pickups 15 of 87 | FAILED | matrix |
| `shift_placement` | nominal | baseline CNN, execute 90 | absent | absent | gain +0.01, pickups 46 of 227 | held | matrix |
| `shift_placement` | nominal | perturbed ACT, execute 90 | present | present | gain +0.73, pickups 56 of 77 | held | matrix |
| `shift_placement` | nominal | perturbed ACT, execute 30 | partial or present | present | gain +0.92, pickups 9 of 7 | held | matrix |
| `shift_placement` | nominal | perturbed CNN, execute 90 | present | present | gain +0.55, pickups 139 of 246 | held | matrix |
| `shift_placement` | nominal | physics v2 CNN, execute 90 | absent | absent | gain +0.11, pickups 53 of 202 | held | matrix |
| `shift_placement` | high-lift | fair CNN, execute 90 | absent | absent | gain +0.11, pickups 45 of 151 | held | matrix |
| `swap_elbow` | nominal | baseline ACT, execute 90 | absent | absent | pickups 35 of 87 | held | matrix |
| `swap_elbow` | nominal | baseline ACT, execute 30 | absent | absent | pickups 2 of 87 | held | matrix |
| `swap_elbow` | nominal | baseline CNN, execute 90 | absent or partial | absent | pickups 68 of 227 | held | matrix |
| `swap_elbow` | nominal | physics v2 ACT, execute 90 | present | present | pickups 62 of 73 | held | matrix |
| `swap_elbow` | nominal | physics v2 ACT, execute 30 | present | present | pickups 59 of 67 | held | matrix |
| `swap_elbow` | nominal | physics v2 CNN, execute 90 | present | present | pickups 190 of 202 | held | matrix |
| `swap_elbow` | servo | physics v2 ACT, execute 90 | present | present | pickups 71 of 63 | held | matrix |
| `swap_elbow` | servo | physics v2 ACT, execute 30 | present | present | pickups 85 of 82 | held | matrix |
| `swap_elbow` | servo | physics v2 CNN, execute 90 | present | present | pickups 200 of 169 | held | matrix |
| `drop_cube` | nominal | baseline ACT, execute 90 | absent | absent | 2 of 94 drops | held | recomputed |
| `drop_cube` | nominal | baseline ACT, execute 30 | absent | partial | 5 of 94 drops | FAILED | recomputed |
| `drop_cube` | nominal | baseline CNN, execute 90 | absent | absent | 1 of 241 drops | held | recomputed |
| `drop_cube` | nominal | perturbed ACT, execute 90 | absent | partial | 6 of 87 drops | FAILED | recomputed |
| `drop_cube` | nominal | perturbed ACT, execute 30 | absent | absent | 0 of 7 drops | held | recomputed |
| `drop_cube` | nominal | perturbed CNN, execute 90 | absent | partial | 31 of 274 drops | FAILED | recomputed |
| `drop_cube` | nominal | physics v2 ACT, execute 90 | absent | absent | 0 of 75 drops | held | recomputed |
| `drop_cube` | nominal | physics v2 ACT, execute 30 | absent | absent | 0 of 75 drops | held | recomputed |
| `drop_cube` | nominal | physics v2 CNN, execute 90 | absent | absent | 0 of 210 drops | held | recomputed |
| `drop_cube` | servo | baseline ACT, execute 90 | absent | absent | 0 of 64 drops | held | matrix |
| `drop_cube` | servo | baseline ACT, execute 30 | absent | absent | 0 of 1 drops | held | matrix |
| `drop_cube` | servo | baseline CNN, execute 90 | absent | absent | 0 of 154 drops | held | matrix |
| `drop_cube` | servo | physics v2 ACT, execute 90 | absent | absent | 0 of 69 drops | held | matrix |
| `drop_cube` | servo | physics v2 ACT, execute 30 | absent | absent | 0 of 90 drops | held | matrix |
| `drop_cube` | servo | physics v2 CNN, execute 90 | absent | absent | 0 of 178 drops | held | matrix |
| `drop_cube` | high-lift | fair CNN, execute 90 | absent | absent | 0 of 165 drops | held | matrix |
| `drop_cube` | high-lift | fair CNN, execute 30 | absent | absent | 0 of 211 drops | held | matrix |
| `drop_cube` | high-lift | high-lift ACT, execute 90 | absent | absent | 0 of 72 drops | held | matrix |
| `drop_cube` | high-lift | high-lift ACT, execute 30 | absent | absent | 0 of 89 drops | held | matrix |
| `move_square` | nominal | baseline ACT, execute 90 | absent | present | gain +0.71, on moved square 83 of 87 | FAILED | recomputed |
| `move_square` | nominal | baseline ACT, execute 30 | absent | present | gain +0.74, on moved square 83 of 87 | FAILED | recomputed |
| `move_square` | nominal | baseline CNN, execute 90 | absent | present | gain +0.51, on moved square 165 of 227 | FAILED | recomputed |
| `move_square` | nominal | perturbed ACT, execute 90 | absent | present | gain +0.68, on moved square 63 of 77 | FAILED | recomputed |
| `move_square` | nominal | perturbed ACT, execute 30 | absent | present | gain +0.85, on moved square 7 of 7 | FAILED | recomputed |
| `move_square` | nominal | perturbed CNN, execute 90 | absent | present | gain +0.52, on moved square 175 of 246 | FAILED | recomputed |
| `move_square` | nominal | physics v2 ACT, execute 90 | absent | present | gain +0.93, on moved square 72 of 73 | FAILED | recomputed |
| `move_square` | nominal | physics v2 ACT, execute 30 | absent | present | gain +0.96, on moved square 64 of 66 | FAILED | recomputed |
| `move_square` | nominal | physics v2 CNN, execute 90 | absent | present | gain +0.71, on moved square 178 of 201 | FAILED | recomputed |
| `move_square` | servo | baseline ACT, execute 90 | absent | partial | gain +0.44, on moved square 12 of 20 | FAILED | matrix |
| `move_square` | servo | baseline ACT, execute 30 | absent | n/a | nothing to compare (0 pickups unprobed) | counted FAILED, unreadable | matrix |
| `move_square` | servo | baseline CNN, execute 90 | absent | partial | gain +0.31, on moved square 14 of 32 | FAILED | matrix |
| `move_square` | servo | physics v2 ACT, execute 90 | absent | present | gain +0.75, on moved square 57 of 63 | FAILED | matrix |
| `move_square` | servo | physics v2 ACT, execute 30 | absent | present | gain +0.81, on moved square 77 of 82 | FAILED | matrix |
| `move_square` | servo | physics v2 CNN, execute 90 | absent | present | gain +0.51, on moved square 120 of 169 | FAILED | matrix |
| `move_square` | high-lift | fair CNN, execute 90 | absent | partial | gain +0.22, on moved square 57 of 151 | FAILED | matrix |
| `move_square` | high-lift | fair CNN, execute 30 | absent | present | gain +0.51, on moved square 123 of 188 | FAILED | matrix |
| `move_square` | high-lift | high-lift ACT, execute 90 | absent | present | gain +0.81, on moved square 57 of 64 | FAILED | matrix |
| `move_square` | high-lift | high-lift ACT, execute 30 | absent | present | gain +0.87, on moved square 72 of 72 | FAILED | matrix |

"On moved square" counts the compared episodes whose cube ends inside the square at its new place. A ratio above 1
in "pickups" (9 of 7, 71 of 63) means the probe produced pickups in episodes that had none unprobed; it is a ratio
of counts, not a share of episodes.

*Conclusion.* The headline was a fair reading of its 13. The full record is 32 held and 23 failed, and 19 of the
23 are one probe. That probe needs its own look before anything is concluded from it.

### Step 2. `move_square`: do the policies follow the square, or does the probe invent it?

*Why.* The note said in advance that a policy following `move_square` counts against "data alone decides". Before
accepting that, rule out the ways the number could be produced without the policy doing anything.

*What we did.* Read the scoring code, the reference controls, and the per-episode traces of every cell.

How the number is made (`probes.py`, `paired_metrics`). The square moves 20 mm when the cube has risen 15 mm.
Only the square moves; the cube is in the jaws. The gain is the cube's final position in the probed episode minus
its final position in the unprobed episode, projected on the 20 mm move and divided by 20 mm. Only episodes where
the probe fired and the unprobed episode had a pickup are compared.

*What it says.*

1. The reference controls pass on all three scenes (matrix). The plain teacher is the scripted controller that
   never looks again, so it is the zero baseline. The informed teacher is told the move in advance.

| Control | Scene | Follow gain (mean, median) | Cube ends on the moved square | Commands changed after the move |
|---|---|---|---|---|
| plain teacher (baseline, must read 0) | nominal | -0.00, -0.00 | 1 of 100 | 0.000 |
| plain teacher (baseline, must read 0) | servo | -0.00, -0.00 | 0 of 100 | 0.000 |
| plain teacher (baseline, must read 0) | high-lift | -0.00, -0.00 | 3 of 85 | 0.000 |
| informed teacher (must read 1) | nominal | +0.99, +0.99 | 100 of 100 | 0.313 |
| informed teacher (must read 1) | servo | +1.02, +1.02 | 100 of 100 | 0.358 |
| informed teacher (must read 1) | high-lift | +1.02, +1.01 | 85 of 85 | 0.360 |

   The zero control is read on the median since commit 099f3c9. That change does not matter for this probe: mean
   and median are both -0.00.

2. "The unprobed episode already puts the cube where the moved square is": ruled out. The gain is a difference
   between two episodes of the same placement, so where the unprobed cube lands cancels. And a cube set down at the
   old place is not on the moved square: the plain teacher ends there in 1 of 100, 0 of 100 and 3 of 85.

3. "A cube still in the jaws, or scatter at release, makes the number": ruled out (recomputed from traces). In the
   compared episodes the cube's final height differs between the two episodes by a median of 0.0 mm and at most
   1.0 mm in every cell, so both cubes are resting. Scatter has no direction: it would average to 0 along the move.
   The sideways part of the displacement averages between -0.04 and +0.06 of the move in every cell. The part along
   the move is +0.19 to +0.96, with the share of episodes that read below 0 between 0.00 and 0.15 (0.15 is a
   13-episode cell; cells with 45 or more episodes are at 0.04 or less).

4. The jaws move, and only after the next camera frame is read (recomputed from traces). The table is the mean
   jaw displacement along the move, as a fraction of the move. "Execute 90" policies read a frame every 90 actions
   (270, 360), "execute 30" policies every 30. The square moves at about action 306 to 325.

| Policy, scene | Square moves at (median) | Next frame read | Jaws at action 330 | 360 | 390 | 479 (end) |
|---|---|---|---|---|---|---|
| plain teacher, nominal (baseline; commands identical, jaws within 0.07 mm all episode) | 320 | never | 0.00 | 0.00 | 0.00 | 0.00 |
| baseline CNN seed 101, nominal, execute 90 | 320 | 360 | 0.00 | 0.00 | 0.47 | 0.49 |
| baseline ACT, nominal, execute 90 | 320 | 360 | 0.00 | 0.00 | 0.66 | 0.65 |
| physics v2 ACT, nominal, execute 90 | 316 | 360 | -0.00 | 0.00 | 0.91 | 0.73 |
| fair CNN seed 303, high-lift, execute 30 | 307 | 330 | 0.01 | 0.34 | 0.45 | 0.48 |
| high-lift ACT, high-lift, execute 30 | 306 | 330 | 0.00 | 0.56 | 0.83 | 0.91 |

5. The same checkpoint follows more when it reads more frames: the fair CNN reads +0.22 at execute 90 and +0.51 at
   execute 30 (matrix).
6. The learned policies do not read the square's position from the simulator. The ACT wrapper discards the
   simulator handle (`residual_policy.py`, `del adapter`); the policies get the camera frame and the joint state.
7. The probed episodes mostly still succeed, which the open-loop baseline cannot do: baseline ACT on the nominal
   scene 83 of 87 successes under the probe (recomputed), plain teacher 1 of 100 (matrix).
8. It is in every learned policy, the baselines included, and it is strongest in ACT:

| Policy group | Nominal scene (recomputed) | Servo scene (matrix) | High-lift scene (matrix) |
|---|---|---|---|
| plain teacher (baseline) | -0.00 | -0.00 | -0.00 |
| baseline CNN | +0.51 | +0.31 (32 episodes) | |
| baseline ACT, execute 90 / 30 | +0.71 / +0.74 | +0.44 (20 episodes) / n/a | |
| perturbed CNN | +0.52 | | |
| perturbed ACT, execute 90 / 30 | +0.68 / +0.85 (7 episodes) | | |
| physics v2 CNN | +0.71 | +0.51 | |
| physics v2 ACT, execute 90 / 30 | +0.93 / +0.96 | +0.75 / +0.81 | |
| fair CNN, execute 90 / 30 | | | +0.22 / +0.51 |
| high-lift ACT, execute 90 / 30 | | | +0.81 / +0.87 |

*Conclusion.*

- What the files prove. The following is real behaviour, not a product of the score. The policies carry the cube
  toward the moved square and set it down there. The open-loop baseline reads 0 on the same probe. The motion
  starts only after the next frame is read.
- What the files only suggest. Why the policies have it. The prediction said "absent, no intervention targets it".
  A possible reason it was wrong: every policy here was trained on placements drawn at random, the teacher always
  sets the cube down on the square, and the square is in view while the cube is carried. So the ordinary data may
  already reward "set down where the square is now". No file tests this. Every checkpoint in the matrix has
  random placements, so there is no policy without it to compare.
- One limit found on the way. The unprobed and probed episodes of a learned policy are not identical before the
  change, as the note assumes. Commands differ by a median of 0 but by up to 0.10 command units in the worst
  episode of a cell, and the jaws by up to 8.6 mm in the worst pair (recomputed; the teacher pairs are identical
  to 0.000). The cause was not looked for. It has no direction, so it does not make a gain: the mean jaw
  displacement along the move is 0.00 until the next frame read.
- The measurement that would settle the cause: probe one checkpoint again with `move_square`, and at the first
  frame read after the move give the policy the frame from the unprobed episode. If the gain falls from about
  +0.9 to 0.2 or less, the camera frame of the square is the cause, and nothing else is. Whether the data forces
  it then needs one training run with the square hidden in the frames after the lift.

### Step 3. Link 3: is the skill what buys success?

*Why.* A policy could follow the cube and still not pick it up. The registered test has two halves.

*What we did.* Read `link3` in `matrix.json`: one point per learned-policy cell under `shift_placement` (cube and
square move 20 mm at action 100). 32 cells have a point. One cell (baseline ACT, execute 30, servo scene) has no
pickups unprobed and is left out.

Registered: (a) rank correlation between follow gain and pickups kept above 0.5; (b) a gain of 0.5 or more keeps at
least 70% of pickups, and a gain of 0.2 or less keeps at most 30%.

*What it says.*

| Test | Cells | Result |
|---|---|---|
| plain teacher, gain +0.02 (baseline: no following at all) | 1 | keeps 12 of 100 |
| informed teacher, gain +0.98 (baseline: full following) | 1 | keeps 100 of 100 |
| (a) rank correlation, headline stage, nominal scene | 15 | +0.90, held |
| (a) rank correlation, all cells | 32 | +0.77, held |
| (b) gain 0.5 or more keeps at least 70% | 11 | 8 meet it, 3 do not |
| (b) gain 0.2 or less keeps at most 30% | 18 | 11 meet it, 7 do not |

The 3 high-gain cells under 70% are the three perturbed-CNN seeds:

| Cell | Follow gain | Pickups kept |
|---|---|---|
| plain teacher, nominal (baseline) | +0.02 | 12 of 100 (0.12) |
| perturbed CNN seed 101 | +0.60 | 49 of 83 (0.59) |
| perturbed CNN seed 202 | +0.54 | 45 of 81 (0.56) |
| perturbed CNN seed 303 | +0.51 | 45 of 82 (0.55) |
| perturbed CNN, pooled | +0.55 | 139 of 246 (0.57) |
| perturbed ACT, execute 90 (meets it) | +0.73 | 56 of 77 (0.73) |

The 7 low-gain cells over 30% keep 0.32 to 0.50: the three baseline CNN seeds on the servo scene (6 of 13, 3 of 9,
5 of 10), two fair CNN seeds (18 of 52, 15 of 45), and physics v2 CNN seed 101 on both scenes (20 of 63, 19 of 56).

*Conclusion.*

- Half (a) holds. Half (b) does not hold as registered.
- What the correlation says: across cells, the ones that follow more keep more pickups.
- What it does not say. It is a ranking, so it says nothing about the 70% and 30% levels. The cells are not
  independent: three seeds of one recipe, the same checkpoint at execute 90 and 30, and the same 100 placements
  count as separate points. It compares different policies, so anything else that differs between them (ACT or
  CNN, how often they read a frame) rides along. It does not show that following is what keeps a given pickup.
- A reading the traces suggest (recomputed, not registered). Per episode, a gain of 0.5 is not enough. Among the
  perturbed CNN's episodes with gain 0.5 or more, pickups are kept in 28 of 50, 22 of 43 and 22 of 41. Half of a
  20 mm move still leaves up to 10 mm, and the median distance left is 9.1 to 9.7 mm where the pickup is kept
  against 11.7 to 14.2 mm where it is lost. Two execute-30 ACT cells (high-lift ACT, and physics v2 ACT on
  the servo scene) leave a median of 2.0 and 2.6 mm where the pickup is kept, and keep 59 of 67 and 72 of 79. Distance left in millimetres looks like the better yardstick than a gain threshold.

### Step 4. `drop_cube`: three "partial" verdicts that are not retries

*Why.* A retry after a dropped cube would be a second skill nobody trained. Three nominal-scene groups read
"partial" (recomputed): perturbed CNN 31 of 274 drops, perturbed ACT at execute 90 6 of 87, baseline ACT at
execute 30 5 of 94.

*What we did.* The probe puts the cube back at its start once it has risen 15 mm. It counts a regrasp when the
cube is 10 mm up again, from 10 actions after the drop. Read the height and gripper traces of every counted episode.

*What it says.*

| Group, nominal scene | Counted regrasps | Gripper reopened before the cube rose again | Cube already 10 mm up at the first counted action |
|---|---|---|---|
| plain teacher (baseline) | 0 of 100 | | |
| perturbed CNN, 3 seeds | 31 of 274 | 0 of 31 | 30 of 31 |
| perturbed ACT, execute 90 | 6 of 87 | 0 of 6 | 6 of 6 |
| baseline ACT, execute 30 | 5 of 94 | 0 of 5 | 5 of 5 |

*Conclusion.* In no counted episode do the jaws open again (the gripper reading changes by at most 0.018 on a
range of 0 to about 1 between the drop and the counted rise). The cube is back above 10 mm ten actions after it was
put down, in jaws that stayed closed. That is the closed jaws taking the cube along again, not a policy noticing an
empty gripper. Why this happens on the nominal scene and never on the other two was not determined. The honest
reading of `drop_cube` is: no policy retries anywhere, and the probe needs a stricter count (a regrasp requires
the jaws to reopen).

### Step 5. Things the full matrix shows that were not registered

No prediction covers these, so they are observations, not tests.

| Observation | Without the look data | With the look data (perturbed teacher) | Baseline |
|---|---|---|---|
| Follows cube and square at execute 30 (`shift_placement`, matrix) | physics v2 ACT +0.87 (57 of 67); high-lift ACT +0.61 (61 of 72); fair CNN +0.64 (165 of 188) | perturbed ACT +0.92 (9 of 7) | baseline ACT +0.06 (15 of 87) |
| The same checkpoints at execute 90 (matrix) | physics v2 ACT +0.38; high-lift ACT +0.30; fair CNN +0.11 | perturbed ACT +0.73 | baseline ACT +0.04 |
| Follows the cube alone, nominal scene (`shift_cube`, recomputed) | physics v2 CNN +0.06; physics v2 ACT +0.16 / +0.86 (execute 90 / 30) | perturbed CNN +0.65; perturbed ACT +0.91 / +0.97 | baseline CNN +0.02; baseline ACT +0.02 / +0.09; plain teacher +0.02 |

- Every policy family trained with random servo physics follows a moved cube when it replans every 30 actions,
  with no look data. The baseline does not. So the "motor" data change brings some of the "look" skill with it.
- On the servo scene `swap_elbow` reads "present" for the baseline policies with ratios far above 1 (baseline CNN
  186 of 32). There the swap makes the elbow stiff, which is the elbow those policies were trained on. It is not a
  skill and nothing was registered for it.

### What we conclude, and what we do next

What stands:

- A targeted data change produces its targeted skill, and the baselines lack it. Look: +0.55 and +0.73 against
  +0.01 and +0.04. Motor: 190 of 202 and 62 of 73 against 68 of 227 and 35 of 87. All 17 `shift_placement` and
  `swap_elbow` predictions but one held.
- The probes read 0 and 1 where they must on all three scenes.
- Following and kept pickups rise together across cells (+0.77 over 32).

What does not stand:

- "Each data change creates its own skill, and a skill exists only where a change was aimed at it." Set-down
  following is in every learned policy with no change aimed at it (+0.51 to +0.96 on the nominal scene, baseline 0).
  Looking at execute 30 comes with the motor data. By the rule written above ("What would count against the
  recipe") this counts against the claim that data alone decides, unless the ordinary data turns out to force it.
- The registered link 3 thresholds.

The honest status of the experiment is mixed: the two aimed tests worked, the "nothing else appears" half failed.

For experiment 53 (`notes/recursive-generalization-20261119.md`). Its own results use `shift_placement` and
`swap_elbow` on ACT, whose controls pass and whose numbers are unchanged. They stand. Three things in it need a
caveat. (1) Its first sentence says the probes "showed that one data change gives one feature". Read all the way,
they showed that an aimed change gives its feature, and that features also appear unaimed. (2) Its reading names
"set-down on a moved square" as a feature that would have to be built on an earlier one. Every policy already has
it. (3) Its `combo` stage is what removed three probes from the nominal-scene `result.json` files of this
experiment.

Next, in order:

1. Regenerate the record. Make `feature_probes.py run` keep the metrics of probes already on disk, rewrite the 15
   nominal-scene `result.json` files from the stored rows and traces, run the report, commit. CPU only.
2. Settle the cause of `move_square` with the frozen-frame control described in step 2. The number to watch is the
   follow gain of one ACT checkpoint under it; 0.2 or less says the frame of the square is the cause.
3. Fix `drop_cube` (count a regrasp only when the jaws reopen) before any retry capture is judged with it.
4. Restate link 3 in millimetres left at closure before registering it again.

### One lesson

A prediction that failed in all 18 groups where it could be read is not noise, and it sat unread for five days because the report
printed "not run" for half of it and the headline had already been told. Read every registered prediction, and
when a tool re-runs part of a cell, check that it did not drop what was there.
