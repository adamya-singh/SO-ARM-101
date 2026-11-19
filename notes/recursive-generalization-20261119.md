# Do the features stack, combine and build on each other? (2026-11-19)

Status: built, tested, queued (see "Queue"). Everything under "Predictions" was written before any cell ran.

## Why this follows the feature probes

The feature probes (`notes/feature-probes-20261118.md`) showed that one data change gives one feature, that a
probe detects it, and that the feature tracks success. Headline, nominal SO-101 scene, 100 placements:

| ACT policy | Follow gain when cube and square shift 20 mm | Pickups kept when the elbow goes soft |
|---|---|---|
| baseline | +0.04 (execute 90), +0.06 (execute 30) | 35 of 87, 2 of 87 |
| perturbed teacher ("look") | +0.73 | 26 of 77 |
| physics v2 ("motor") | +0.38, +0.87 | 62 of 73, 59 of 67 |

That is the foundation, not the claim. Planning a curriculum for recursive generalization needs three more things
to be true, and none of them was tested:

1. **Features stack.** One policy trained on both data changes has both features.
2. **Features combine.** That policy handles both changes in one episode, which no training episode contains.
3. **Order matters.** A policy that already has one feature learns the next from less data than a policy that
   has neither.

## The three experiments

All use ACT with the retry-tranche settings, the probes of 2026-11-18 and the same 100 placements and shift plans.
Queue script `simulation_code/queue_recursive_generalization_20261119.sh`, report
`tools/recursive_generalization_report.py`, output `experiments/recursive_generalization_20261119`.

### 1. Stacking

| Run | Training data | Steps |
|---|---|---|
| `both_split` | first 3600 perturbed-teacher placements + first 3600 physics v2 placements | 40,000 |
| `both_full` | all 7200 + all 7143 | 40,000 |

`both_split` has the 7200-placement budget of the three existing policies, so the four form a 2 by 2 (neither, look
only, motor only, both). `both_full` gives each change its full amount, to tell "half the data was too little" from
"the two changes interfere". Each is probed with `shift_placement` and `swap_elbow` at execute 90 and 30, on the
nominal and the measured-servo scene.

### 2. Combination

New probe `shift_and_swap`: at action 100 the cube and square move 20 mm and the elbow servo becomes the other
kind. The two data changes sit in different training episodes (different captures), so no policy has seen the two
together.

- Its follow gain is measured against the `swap_elbow` episode, not the unprobed one, so it is the effect of the
  shift on an arm whose elbow has already changed.
- Its verdict is the weaker of the two readings (follow gain, pickups kept).
- The report prints the pickups kept next to the product of the two single-probe retentions, which is what two
  features that do not interact would keep.
- It runs first on the existing policies (baseline, look only, motor only, ACT and CNN) and then on the two runs above.
- Reference controls: plain teacher must read gain 0, the teacher told the shift must read 1. Its successes are not
  a control here, because the teacher's plan does not survive the changed elbow.

### 3. Order

Start from a finished ACT checkpoint, train 5000 further steps on the first N placements of the other data change,
N = 300, 1200, 4800, and probe. N = 0 is the start checkpoint itself.

| Direction | Feature being added | New data | Start without the other feature | Start with it | Seeds |
|---|---|---|---|---|---|
| `motor` | coping with a changed elbow (`swap_elbow`) | physics v2 | baseline ACT | perturbed-teacher ACT | 202, 101 |
| `look` | following the shift (`shift_placement`) | perturbed teacher | baseline ACT | physics v2 ACT | 202 |

- The two starts of a direction had the same amount of training (40,000 steps on 7200 placements), so the
  comparison is "has the other feature" against "does not", not "more training" against "less".
- The run starts from the checkpoint's weights and its normalisation statistics (`--init-checkpoint`, new) and
  sees only the first N placements (`--episode-limits`, new).
- The report also prints what happened to the feature the start already had.

## Predictions (registered before the runs)

Machine-readable: `notes/recursive-generalization-20261119-predictions.json`.

| Experiment | Cell | Prediction |
|---|---|---|
| 1 | `both_split`, execute 90 | follows the shift (present or partial), copes with the elbow (present) |
| 1 | `both_split` execute 30, `both_full` both | present on both probes |
| 2 | `both_split`, `both_full` | `shift_and_swap` present or partial |
| 2 | baseline, perturbed-teacher (ACT and every CNN seed), physics v2 CNN | absent |
| 2 | physics v2 ACT | absent or partial at execute 90, present or partial at execute 30 |
| 3 | `motor` | the perturbed-teacher start keeps at least 0.10 more of its pickups, mean over N and seeds (low confidence) |
| 3 | `look` | the physics v2 start reads at least 0.10 more follow gain, mean over N (moderate confidence) |

What would count against recursive generalization for this pair of features:

- `both_split` and `both_full` both lack one of the features: the changes interfere instead of stacking.
- Both features present alone but `shift_and_swap` keeps clearly fewer pickups than the independent product: the
  features do not combine.
- The start with the other feature leads by less than 0.10 in both directions: the two features are learned
  independently, and the order of a curriculum would not matter for them.

## Limits

- One training seed for experiments 1 and 2, and for the `look` direction of 3. The probe noise is small (100
  placements) but the seed-to-seed spread of ACT on these probes is not measured outside `motor`.
- Two features only. A positive result supports stacking and order for this pair, not for a ladder of features.
- The physics v2 data was captured on the measured-servo scene and the perturbed-teacher data on the nominal scene,
  so "motor" data also differs in the base servo model.
- In experiment 3 the new data is not mixed with the old, so the start's own feature may fade; the report shows it.
- Simulation only.

## Queue

About 22 hours of GPU time at the back of the shared queue, one job at a time:

| Stage | Content |
|---|---|
| rehearsal | 40 training steps from a checkpoint on two limited captures, the new probe on 3 placements, the report; gates everything (moved to the front, about 5 minutes) |
| combo | `shift_and_swap` on the existing policies and the teacher controls |
| `both_split` | trainer, then its probe cells |
| order | 18 trainers, each followed by its probe cell: `motor` seed 202, `look`, `motor` seed 101 |
| `both_full` | trainer, then its probe cells |
| report | the three tables, the predictions held or failed |

Because `shift_and_swap` is now one of the default probes, the feature probes' remaining stage (tsp 170) also runs
it on every cell it visits.
