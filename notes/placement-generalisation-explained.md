# Placement generalisation, explained for a new intern (2026-09-10 to 2026-09-18)

Written with the experiment-analyze-explain skill (Part 4). It covers the
scaling ladder, random-shift augmentation, the steps / placements / seeds
follow-ups, the 4800-placement run and the closing analysis. The running log
with every table is `notes/vision-rung-notebook.md`; evidence paths are at
the end. An earlier version of this story contained a wrong conclusion; it
is told here as part of the story because the mistake is instructive.

## The setup

A robot arm has a camera on its wrist. A small neural network looks at the
camera image and outputs the next 90 joint commands at once. We call those
90 commands a **chunk**. The network produces a new chunk at steps 0, 90,
180, 270 and so on, each from the image at that step. The task is to pick a
20 mm cube off a square target, lift it, and put it back.

We train by imitation. A scripted **teacher** that knows the cube's exact
position solves the task in a simulator, and the network learns to copy it
from images alone. The **loss** is the mean squared difference between the
network's chunk and the teacher's, in normalised joint units. A **placement**
is one position and angle of the square and cube on the table. **Held-out**
means ten placements the network never trained on. **Closed loop** means we
let the network drive the simulated arm and count successes out of 30
(ten held-out placements, three tries each).

The chunk that matters most starts at step 90. At that moment the arm is in
a **survey pose**, looking down at the whole table, and the network has to
read where the cube is and plan the descent. The cube is 12 to 25 pixels
wide in that image.

The question: a policy that works 30 of 30 when the square never moves
scored 3 of 30 once the square could be anywhere. Why, and what fixes it?

## The measurements

### 1. Train versus held-out loss

*Why.* A model can fail because it learned too little or because it learned
the wrong thing. Comparing its error on training scenes with its error on
new scenes separates the two.

*What we did.* We ran the teacher on the ten held-out placements and scored
the network on those frames, at the step-90 chunk.

*What it says.*

| | training loss | held-out loss | ratio |
|---|---|---|---|
| placement policy, 1200 placements | 8e-7 | 4e-3 | 5000x |

*Conclusion.* A ratio of thousands is **memorisation**. The network stored
an image-to-answer table for its training scenes instead of learning the
rule "cube at this pixel means move there".

### 2. The scaling ladder: more data, a bigger network, longer training

*Why.* The textbook cure for memorisation is more data. The other usual
suspects are model size and training length. Each costs GPU time, so we
tested all three cheaply before committing to any.

*What we did.* Fifteen models: 300 to 2400 placements, three encoder sizes
from 0.8M to 5M parameters, 60k to 480k training steps. Same ten held-out
placements for all.

*What it says.* Held-out loss stayed between 2e-3 and 8e-3 everywhere.
Bigger networks memorised harder. Longer training changed nothing.

*Conclusion.* None of the three standard axes helps this architecture.

### 3. A dumb baseline: nearest-neighbour lookup

*Why.* If the training set does not cover the table densely enough, no
learner can do well. A lookup table tells you what the data alone is worth.

*What we did.* For each held-out placement we found the closest training
placement (2 to 9 mm away) and used its teacher chunk as the prediction.

*What it says.* The lookup beat every network by 5x on average and by 300x
on the worst placement.

*Conclusion.* The data is dense enough. The network cannot read the cube's
position out of the image. It is a perception problem, not a scale problem.

### 4. Random-shift augmentation

*Why.* If the network memorises exact pixel patterns, remove the patterns.

*What we did.* Every time a training image is used, we slide it by a random
amount of up to 12 pixels and leave the label unchanged. No image ever
looks the same twice, so the only thing worth learning is where the cube is
relative to the scene. Same network, same data, same training length.

*What it says.*

| shift | held-out / training ratio | closed loop |
|---|---|---|
| 0 px | 5250x | 0 of 30 |
| 4 px | 168x | 9 of 30 |
| 12 px | 15x | 9 of 30 |

*Conclusion.* Memorisation largely gone from one change to the data loader.

### 5. Watching held-out loss during training

*Why.* A single number at the end cannot tell "never learned" from "learned,
then forgot". So every run now scores the held-out set every 5000 steps.

*What we did.* Re-tested the two axes the ladder had ruled out, one at a
time, now with augmentation on.

*What it says.*

| change | ratio at the end | closed loop | the curve |
|---|---|---|---|
| 480k steps instead of 120k | 77x | 0 of 30 | held-out flat from 150k, training loss kept falling |
| 2400 placements instead of 1200 | 3x to 6x | 15, 6, 12 of 30 over three seeds | ratio low for the whole run |
| 4800 placements | 1.6x | 6 of 30 | held-out tracks training throughout |

*Conclusion.* More steps bring memorisation back: a 12 px shift has only
625 distinct offsets, and with enough passes the network learns those too.
More placements keep it away. At 4800 the two losses move together, so
memorisation is solved.

### 6. Seeds

*Why.* We were about to rank recipes by closed-loop counts. We needed to
know how much those counts move by chance.

*What we did.* Trained the 2400-placement recipe three times, changing only
the random seed.

*What it says.* Losses agreed (3.4e-4 to 5.8e-4). Closed-loop counts were
15, 6 and 12 of 30.

*Conclusion.* Thirty rollouts cannot rank two recipes that differ by less
than about ten successes. Pool rollouts over seeds, and trust the loss more
than a single count.

### 7. Comparing with something that works (the step we skipped)

*Why.* A number means nothing alone. 3e-4 looked small next to the 4e-3 we
started from.

*What we did.* Looked up the same loss for the older policy that works 30
of 30, and converted both to physical units. One normalised unit is pi
radians, so the joint error is the square root of the loss times 180 degrees.

*What it says.*

| policy | loss | joint error | error at the gripper |
|---|---|---|---|
| working policy, fixed square | 2.5e-6 | 0.3 degrees | about 1.7 mm |
| our best placement policies | 3e-4 to 6e-4 | about 3 degrees | about 10 mm |

*Conclusion.* We were still ten times too imprecise. Before doing this
comparison we had written that the loss had "stopped predicting success".
That was wrong. The loss was predicting failure, correctly, and we had
misread it because we compared it with our own past instead of with a
system that works.

### 8. Reading the failed rollouts, in millimetres

*Why.* A loss averaged over 90 steps and 6 joints is a long way from "did
the gripper close around the cube". We wanted the deciding quantity itself.

*What we did.* From the logged joint angles we computed where the gripper
was just before closing, relative to the cube, in every rollout. The working
policy's gripper position served as the target (it varies by under 2 mm).

*What it says.* Rollouts landing within 8 mm of the target succeeded 13
times out of 20. Beyond 8 mm, 3 out of 22. Our policies scatter by about 10
mm (11 mm with 1200 placements, 8 mm with 4800). The gripper angle was fine
in every run.

*Conclusion.* A 10 mm scatter against an 8 mm tolerance gives a success rate
of a third to a half. That is exactly the 6 to 15 of 30 we had been puzzling
over. Data improves it, but slowly.

### 9. Is the error a bias or a scatter?

*Why.* If predictions are pulled toward the middle of the table, the network
is under-trained and more training fixes it. If they scatter evenly, it
cannot see well enough and more training will not help.

*What we did.* Regressed gripper position on target position.

*What it says.* Slopes of 0.95 to 1.00. No pull toward the middle.

*Conclusion.* Scatter. One distant look at a cube 12 to 25 pixels wide is
good to about 10 mm and no better.

### 10. Does the policy use its second look?

*Why.* At step 180 the camera is 46 mm above the cube and the cube fills
the image. Fixing a 10 mm error from there should be easy.

*What we did.* Compared the sideways error just before that chunk (step
179) with the error just before the gripper closes (step 250).

*What it says.*

| | sideways error |
|---|---|
| before the second chunk | 11.4 mm |
| after it | 11.2 mm |

In 18 of 60 rollouts the second chunk did not even finish the descent.
Those 18 contain no successes and every collision.

*Conclusion.* The second chunk corrects nothing. The reason is in the
training data. The teacher is always perfectly centred at step 180, so
every training image at that step shows the cube exactly under the gripper,
with the label "go straight down". The network has never been shown an
off-centre view together with the move that fixes it. Tested offline, that
chunk scores almost zero error, because offline it is tested on the
teacher's centred images too. The test cannot see the problem. In effect
the policy is steering blind after its first look.

## What we conclude, and what we are doing

1. Random shift stays on for every placement policy. 120k steps is the
   budget. 2400 to 4800 placements is the data range.
2. The next work is to make the second look count. We will deliberately
   push the teacher off course during the first descent in the training
   episodes, by up to about 15 mm, and record how it corrects. Then the
   step-180 chunk learns to centre on the cube from the close-up view.
3. We judge that by the sideways error before and after the second chunk
   (target under 4 mm after), by how many rollouts fail to descend (target
   zero), and by closed-loop counts pooled over seeds.
4. A separate network that locates the square from the survey image comes
   after that. It would attack the same single distant look, which is the
   hard way to get millimetres.

## The lesson to keep

An imitation learner is tested offline only on situations the expert got
into. In closed loop it gets into its own situations, slightly wrong ones,
and if the training data contains no recoveries it has nothing to say there.
So: convert your loss into the units of the task, compare it with a system
that works, and measure the failing thing itself in the closed loop. We had
fifteen careful loss tables. The answer was in one afternoon of reading
rollouts in millimetres.

## Evidence

- `notes/vision-rung-notebook.md`, entries from "Scaling ladder (2026-09-11)"
  to "Tranche analysis (2026-09-18)".
- `artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/`:
  `scaling_ladder_20260910`, `augmentation_20260912`,
  `augmentation_shift12_480k_20260912`, `augmentation_shift12_2400_*`,
  `augmentation_shift12_4800_20260912`, `tranche_analysis_20260918`.
- W&B project `so-arm101-v2-scaling`, group `augmentation`.
