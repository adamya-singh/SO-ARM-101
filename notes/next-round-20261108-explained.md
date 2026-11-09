# Random servos and training on mistakes, explained for a new intern (2026-11-08 to 2026-11-09)

Written with the experiment-analyze-explain skill (Part 4). The tables with every number are in
`notes/vision-rung-notebook.md`, "Items 3 and 4 of the next round". The scripts and their outputs are in
`artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/analysis_20261109/`.

## The setup

**The task.** A small robot arm (the SO-101) has a camera on its wrist. It must pick a 20 mm cube off a square target,
lift it and put it back. We train in a simulator by imitation:
- A scripted **teacher**, which knows exactly where the cube is, solves the task.
- A neural network learns to copy the teacher from camera images and the arm's joint angles.

**The networks.** Each one outputs a **chunk**: the next 90 joint commands at once.
- The **CNN** always runs all 90 commands before it looks again.
- The **ACT** (a transformer policy) can also **re-plan every 30 steps**: it looks again after 30 commands and replaces
  the rest of the chunk. We call this "exec30"; running the full 90 is "exec90".

**The servos.** These are the motors in the joints. In the default simulator they are **stiff**: a joint goes almost
exactly where it is told. We measured the real arm's elbow, and it is **soft**: it sags and lags. Nobody knows yet
whether the real arm sits nearer the stiff end or the soft end (reading its gain settings will tell us).

**How we score.** **Held-out** means ten table placements the network never trained on, tried three times each, so
scores are out of 30.

**The two experiments:**
1. **Item 3, random physics.** Every training episode draws its own servo stiffness, damping and friction, anywhere
   from soft to stiff, plus random delays. The hope is one policy that works on any servo.
2. **Item 4, training on mistakes.** The teacher deliberately aims up to 15 mm wrong on its first descent. It then
   corrects itself on the second look. The hope is that the network learns to correct its own aim.

## Measurement 1: does the random-physics policy work everywhere?

*Why.* The previous attempt (step 2) randomized only around the soft servo. It scored 3/30 on the stiff simulator. If
this one also specialises, randomizing more widely is not the answer.

*What we did.* We ran every policy on six versions of the simulator with `tools/eval_policy.py`. We also ran the
teacher on exactly the same 30 rollouts, which gives the best score available.

*What it says.* ACT, re-planning every 30 steps:

| | Stiff | Soft | Soft + delays | Fully random | Stiff elbow | Low friction |
|---|---|---|---|---|---|---|
| Teacher (best possible) | 30 | 30 | 27 | 25 | 30 | 30 |
| Trained on random physics | 24 | 18 | 19 | 11 | 24 | 26 |
| Trained on the stiff servo only | 25 | 0 | 0 | 2 | 0 | 0 |

The policy trained only on the stiff servo falls to zero as soon as the servo changes. The random-physics policy stays
between 18 and 26 on every fixed version, and costs only 1 point on the stiff one.

*Conclusion.*
- **Ruled in:** randomizing across the whole range works. We had set the target before the run: 15/30 on both the
  stiff and the soft-plus-delays versions. It scored 24 and 19.
- **Ruled out:** that wide randomization makes the policy hedge so much that it is bad everywhere.

## Measurement 2: why is "fully random" still only 11/30?

*Why.* In this version, every one of the 30 tries has its own random servo. A low score could mean the tries are
impossible. It could also mean the policy cannot work out which servo it has. Those two need different fixes.

*What we did.*
- We compared the policy with the teacher on the same 30 draws.
- We listed which safety rule ended each failed try.
- We correlated each draw's servo settings with how often it failed (`v2_cell_draws.py`, `safety_breakdown.py`).

*What it says.*
- The teacher solves 25 of the 30 draws, so most of them are possible.
- Almost every policy failure is a **relative-limit hold**. This safety rule stops a command that is more than 20
  units (on a ±100 scale) away from where the joint actually is.
- The holds happen mostly on the elbow during the descent.
- They happen most on draws where the elbow is slow: a high ratio of damping to stiffness, correlation +0.64. Long
  delays come next, at +0.41.

*Conclusion.* The policy runs its 90 commands without feedback. When the elbow is slow, the commands race ahead of the
arm until the safety rule trips. The policy cannot know the elbow is slow, because nothing it sees records how the
elbow has been responding. The next fix is to give it its own recent commands, so the lag becomes visible. This is a
correlation, not yet proven by an experiment.

## Measurement 3: did reinforcement learning help?

*Why.* After imitation, we ran PPO, a reinforcement-learning method, for 40,000 simulated episodes (5.8 GPU hours).
If it helps, it stays in the recipe.

*What we did.* We compared the policy before and after PPO in every version of the simulator, and read PPO's own
training log.

*What it says.*
- During training, success rose from 29% to 35%.
- The relative-limit holds did not fall at all.
- On held-out tests, PPO is within 3 points of the policy before PPO everywhere.

*Conclusion.* PPO does not fix the failure that matters. It leaves the default recipe for now.

## Measurement 4: does training on mistakes help?

*Why.* On the older arm model (the SO-100), this recipe corrected the aim, but that arm's gripper did not reward
better aim. The SO-101's gripper does.

*What we did.* We trained three CNNs with different random seeds on the mistake data and three on normal data. For
each, we measured the sideways miss just before the gripper closes, using `tools/correction_report.py`.

*What it says.* Three seeds each:

| | Mistakes data | Normal data |
|---|---|---|
| Successes, out of 90 | 74 | 59 |
| Sideways miss (mm) | 6.4 | 8.5 |
| Never went down for the cube | 4 | 22 |

*Conclusion.* For the CNN, training on mistakes works. The second look now removes about 3 mm of aim error, against
1 mm before.

## Measurement 5: why did the ACT trained on mistakes fail?

*Why.* The same data gave the ACT 15/30 at exec90 and 0/30 at exec30, against 27 and 25 on normal data. A bad
algorithm and a bad dataset need different fixes.

*What we did.*
- We scored the ACT's error separately at each point where it re-plans: step 0, 30, 60 and so on
  (`act_per_start.py`).
- We plotted the gripper's height over time (`height_profile.py`).

*What it says.*
- The error is normal at every re-planning point except steps 120 and 150. There it is about 1,500 times larger: 33°
  per joint against 0.8°.
- Those are exactly the points the training set never contains. The training code drops re-planning points inside
  the deliberate mistake, because what the teacher does there is half-finished and has no correct answer to copy.
- Every episode in this dataset contains a mistake, so steps 120 and 150 vanish completely.
- In the simulator, the ACT re-plans at 120 and 150, descends too slowly, and stops about 35 mm above the cube.

*Conclusion.* This is a hole in the data, not a weakness of the ACT. Mixing in normal episodes fills the hole. The CNN
never re-plans at 120 or 150, so it never hit the problem.

## What we conclude and what we do next

1. **Real arm.** Try the random-physics ACT, re-planning every 30 steps. First read the real servos' gain settings,
   which takes about 2 minutes and is read-only. Then pass the real-camera-frame check, which first has to be extended
   to support ACT.
2. **Training on mistakes, for the ACT.** Re-train with normal episodes mixed in. Judge it by the error at steps
   120/150, and by 25/30 at both exec90 and exec30.
3. **Fully random servos.** Give the policy its recent commands as an input. Judge it against the teacher's 25/30.
4. **Drop PPO** from the default recipe for now.

## One lesson to carry

**When a model fails, check whether the situation it failed in exists in its training data.**
- A dataset can be large, 7,200 episodes here, and still contain no example at all of the exact moment the model is
  asked about.
- One average loss hides that hole. Breaking the loss down by "where in the task" found it in minutes.
- The general habit: list the situations the model meets when deployed, then check that each one appears in training.
