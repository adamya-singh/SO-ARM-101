# 35. Does doubling the placements again help? (Sep 13, 2026)

**TL;DR:** Memorization was fully gone (held-out error only 1.6x training), but
closed loop fell to 6/30, and I misread why.

**Problem.** Placements were the axis that paid (entry 33), so the obvious next
step was more of them.

*How we know:* going from 1200 to 2400 placements took the held-out gap from 15x
to 3-6x and the pooled score to 33/90. I didn't know if that trend kept going.

**Experiment.** Capture 4800 new placements (about 11 hours of CPU), and train
the same shift-12 recipe on them. To make it fit on disk, the capture stored only
every 30th camera frame: 15 GB instead of about 450 GB, since training never
reads the rest.

*What we hope to learn:* if held-out error drops below the best 2400 seed
(3.4e-4), data still pays. If not, something else is limiting.

**Outcome.** Held-out error 3.6e-4, inside the 2400 band. Closed loop 6/30, and
half the placements caught the cube by an edge:

- The gap closed to 1.6x, so the network now learns the rule instead of the
  examples. But training error itself was 2x higher and still falling. With
  twice the scenes and the same steps, each one gets seen half as often. Like a
  class twice the size with the same hours of teaching.
- I wrote that the loss had "stopped predicting success" because it looked low
  enough. That was wrong (entry 36): it was still 150x above the policy that
  actually works.

**Takeaway:** when the metric improves and the outcome doesn't, don't decide the
metric broke until you've compared it against a system that works.
