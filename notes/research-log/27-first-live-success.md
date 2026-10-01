# 27. Does it work now? (Sep 10, 2026)

**TL;DR:** Yes, once: the real arm grasped the cube, lifted it 18.6 mm, and put
it back within 3 mm of where it started.

**Problem.** Every real attempt so far had missed or aborted (entries 24 and 26).

*How we know:* the fixes from entry 26 were in, and the cube read 27 mm beyond
the task pose, inside the widened 35 mm tolerance.

**Experiment.** Episode 10: the randomized-look policy, with the speed-limited
safety rule, from the start pose.

*What we hope to learn:* whether the whole chain works end to end on hardware.

**Outcome.** All 480 actions at 30 Hz, zero holds, zero timing misses:

- The gripper stopped at 6 units. Closing on nothing goes down to 0.5, so it was
  holding the cube.
- It lifted 18.6 mm, held about 4.7 s, released, and retreated, and the cube
  ended within 3 mm of its start.

It worked with the cube read 20-27 mm off, which suggests the placement reading
carries a few degrees of camera tilt.

**Takeaway:** one success proves it's possible, not that it's learned. The next
question is always whether it repeats.
