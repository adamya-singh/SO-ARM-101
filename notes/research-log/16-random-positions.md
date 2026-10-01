# 16. Does it work on cube positions it's never seen? (Aug 6, 2026)

**TL;DR:** Trained on 25 random cube positions, the state policy solved 6 of 10
new positions perfectly (18/30). The vision policy solved none.

**Problem.** Everything so far trained and tested on the same 5 starts, so I
couldn't tell solving the task from memorizing 5 starts.

*How we know:* entry 10, where moving the cube 1.5 mm broke the policy.

**Experiment.** Generate random cube positions, train on 25, and test on 10
different positions, 3 tries each. A position only counts if the script can do
the whole episode cleanly. Getting that rule right took two failed versions:
with no screening the script's planner failed outright, and screening only the
plan still let 6 of 35 positions fail when actually run.

*What we hope to learn:* whether the policy interpolates between positions or
only knows the ones it saw.

**Outcome.** The state policy got 18/30, and it was all-or-nothing: 6 positions
went 3/3 with zero bad frames, 4 went 0/3, each failing a different way. Vision
got 0/30 with 848 safety frames.

- Two of the failures sat at the near edge of the area, where 25 random samples
  left gaps. It interpolates fine where it has neighbours.
- "The plan works" isn't "the episode works." Like checking a route on a map
  versus actually driving it.

**Takeaway:** screen your test set with the same standard as your expert, and
read results per position. In a deterministic sim, each position is a pass or a
fail, not a percentage.
