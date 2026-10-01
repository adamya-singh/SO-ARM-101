# 33. Without memorization, do steps or placements help now? (Sep 12, 2026)

**TL;DR:** Longer training brought memorization right back, while more
placements kept it away and pushed closed loop to 15/30.

**Problem.** Shift 12 stopped the memorizing but stalled at 9/30. The ladder
(entry 31) had said steps and data don't help, but that was measured while the
network was memorizing.

*How we know:* every ladder model had held-out error 400x to 50,000x its
training error, so the ladder only told us about memorizers. I didn't know what
scales once that's fixed.

**Experiment.** Two runs, one change each from the shift-12 model: 480k steps
instead of 120k, and 2400 placements instead of 1200. New this time: I scored
held-out error every 5000 steps during training, not just at the end.

*What we hope to learn:* whichever change lowers held-out error while keeping it
close to training error is where to spend compute.

**Outcome.** Opposite answers:

- More steps: held-out error flattened at 1e-3 from step 150k while training
  error kept falling, and the gap grew from 7x to 77x. Closed loop: 0/30. A 12
  pixel shift only gives 625 different offsets per image, so with enough passes
  it memorizes those too. Like re-reading the same shuffled deck until you know
  every shuffle.
- More placements: the gap stayed between 1x and 5x the whole run, the worst
  placement from the ladder dropped from 1.8e-2 to 1.8e-4, and closed loop hit
  15/30 (the same data without the shift got 6/30).

**Takeaway:** a scaling result only holds in the regime you measured it in.
Fix memorization, then ask the scaling question again, and watch held-out error
during training so you see when it turns.
