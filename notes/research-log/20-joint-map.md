# 20. Are the sim's joint angles the real arm's joint angles? (Sep 7, 2026)

**TL;DR:** No: the wrist roll was a quarter turn off and other joints were off
by up to 30 degrees, though my first measurement made it look even worse than
it was.

**Problem.** In the live viewer, the sim gripper was visibly rotated compared to
the real one, and the wrist roll was a quarter turn off.

*How we know:* I could see it. The June joint map was never measured; it assumed
each motor's calibrated range lined up with the model's joint range.

**Experiment.** With torque off, read the raw motor encoders in two poses: the
arm resting under gravity, and the model's all-zero pose held by hand.

*What we hope to learn:* the true zero and scale of every joint.

**Outcome.** The new map, built from the encoder counts (4096 per turn), showed
ranges off by up to 37% and three zeros about 90 degrees off: shoulder lift,
elbow and wrist roll. I moved the script to a near-vertical approach and
re-certified it 15/15.

- Two of those three were wrong. My hand-held "zero pose" had the upper arm
  raised and the elbow bent, so the next day's calibration (entry 21) moved
  them back. Against the final map, the June map had the wrist roll 94 degrees
  off, the elbow about 14, the wrist flex 12-23, the shoulder lift only 2.5, and
  the pan's scale up to 30 degrees off at the ends of its travel.
- So the old map turned commanded poses into slightly different real poses,
  and one very different wrist. Like a map with one street labeled wrong: most
  routes still work, until you need that street.

Every August result came from that map, so its poses don't match the real arm.

**Takeaway:** measure the mapping between your hardware and your model, and
treat your own reference pose as a measurement that can be wrong too.
