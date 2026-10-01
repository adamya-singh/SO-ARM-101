# 26. Why does it miss the real cube? (Sep 10, 2026)

**TL;DR:** The plan ran fine but the jaws closed beside the cube, because the
cube sat 20-40 mm off from where the task assumed, and a strict safety rule kept
freezing the arm.

**Problem.** The randomized-look policy ran on the real arm and missed.

*How we know:* attempt 4 ran 427 steps at 30 Hz with no timing misses, and the
gripper closed on nothing. The same policy in sim succeeds.

**Experiment.** Map the real start frame onto the table through the calibrated
lens (back-projection) to measure where the cube actually was. Then refuse to
run unless the cube is close enough, and print how to move it.

*What we hope to learn:* whether the miss is the cube's placement, the camera,
or the joint map.

**Outcome.** The towel's near edge was right where the square's centre should
be, so the cube was about 40 mm farther than intended and 11 mm to the side.
Training only covered ±10 mm. Attempt 5 failed the real-frame check (the cube had
been knocked off the towel), and the new placement check refused 6-8 (46, 18
and 25 mm off). Attempt 9 ran, but the lift outran the servo:

- The safety rule froze the arm whenever a command jumped too far ahead of the
  servo, while the plan kept moving on. Like a driver who brakes every time the
  car lags the pedal. Fix: on a speed-only violation, send the speed-limited
  command and keep going. Range limits still stop it.

Later I found the cube reading shifts by about 25 mm with the arm's pose, so the
40 mm isn't exact.

**Takeaway:** when a real run fails, measure the world before blaming the model,
and test your safety rules against real motor lag.
