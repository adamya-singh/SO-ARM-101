# 22. Does the recipe learn the bench task from pixels? (Sep 8, 2026)

**TL;DR:** 27/30 on new cube positions with zero safety frames, and 0/30 with a
black image, so this time it really used the camera.

**Problem.** This was the first real evidence on the bench task. Everything in
August ran on the old task with the wrong arm and camera.

*How we know:* entries 20-21 found the joint map and lens were both wrong. I
didn't know if the August recipe would carry over.

**Experiment.** One run, written down beforehand: 400 screened cube positions
(±10 mm around the task pose), seed 202, 120k steps, the vision policy with the
gripper clip, on the calibrated scene. Test on 10 new positions, 3 tries each,
and again with black images.

*What we hope to learn:* if it learns, head toward the real arm. If not, more
data or a second seed before changing the recipe.

**Outcome.** Normal start 3/3, new positions 27/30, zero safety frames. With
black images, 0/30 and 2,055 bad frames. All 3 failures were one position (6 mm
left and 6 mm short), where it lifted the cube 28 mm without a strict grasp.

- Unlike entry 15, blanking the camera kills it, because the cube actually
  moves now. The image is the only place the answer is.

One catch: the sim rendered a perfect pinhole while the real lens bends the
image, so this policy would need real frames undistorted first. That got fixed
the next day.

**Takeaway:** with a calibrated sim and randomized positions, the August recipe
carried over to the new task in one run.
