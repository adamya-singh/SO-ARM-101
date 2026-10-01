# 06. What if it learns from its own mistakes? (Aug 2-3, 2026)

**TL;DR:** I recorded 4,818 rows of the script fixing the clone's actual
mistakes, and the small network couldn't even fit them.

**Problem.** The clone visits a whole range of off-path states, and so far it
had been shown 8 (entry 05).

*How we know:* velocity inputs could tell the states apart and it still failed
0/3. It needed coverage, not features.

**Experiment.** DAgger (dataset aggregation): let the clone drive, and at 11
points where it drifts (steps 31, 74, 194 and the 8 phases from before), hand
control to the script from the clone's exact state and record the full
correction. That's 11 × 438 = 4,818 new rows. Then retrain the same model once.

*What we hope to learn:* if it fits, run it in the sim. The rule written down
beforehand: if it can't fit, stop and record why.

**Outcome.** The capture worked perfectly: all 11 corrections succeeded with
zero safety events, twice, bit for bit. The training didn't. It stalled at an
error of 1.1e-4, 110x above the bar, so it never got a real try:

- The worst error was at step 31, the first correction. There the clone's state
  is only a hair off the normal one, but the right answer is a whole different
  correction. Two near-identical inputs, two answers, and the network can only
  split the difference.
- It was also a 68k-parameter network asked to absorb 10x more rows on the same
  budget.

My first plan, rushing the corrections to fit the time limit, was physically
impossible: a firm grasp takes about 160 steps to settle and can't be sped up.

*Note:* captured on the wrong MuJoCo version (entry 09). The capture was redone
on the right one, the retrain wasn't.

**Takeaway:** the idea was right; the model couldn't hold it. I closed the
step-by-step clone and moved on to predicting chunks of actions.
