# 24. Does it work on the real arm? (Sep 9, 2026)

**TL;DR:** The runner and safety system worked, but the policy's very first move
on the real camera image was wrong, because it had only ever seen one look of
the scene.

**Problem.** Every result so far was in simulation.

*How we know:* 30/30 in sim says nothing about the real arm until you try it.

**Experiment.** Run the lens-matched policy on the real arm with the new
physical runner: drive to the start pose, check everything, then run the policy
at 30 Hz with every safety check on.

*What we hope to learn:* if it works, great. If not, which part breaks.

**Outcome.** Attempt 1: the elbow stalled 4 units short of the start pose under
gravity and the approach timed out. I let the approach overshoot slightly.
Attempt 2 reached the start in 0.6 s and ran 72 steps at 30 Hz with no timing
misses, then aborted: the first chunk drove the shoulder below its safe floor.
In the sim, the first chunk just holds still.

- It's brittle to how the scene looks. It trained on one rendered look (dark,
  flat background, flat lighting, an exact towel), and the real frame was
  brighter, textured and different. Even adding +25 brightness to a sim frame
  broke it. Like recognizing a friend only from one photo.
- The servos read 5.4 V. I first called that a fault, but it's normal for the
  stock 5 V adapter under load.

Next: randomize how the scene looks during training.

**Takeaway:** a sim score is a hypothesis about the real world, and the first
real frame is the test. Check the policy on a recorded real frame before you
move the arm.
