# 18. Can the script do the real bench task? (Sep 6, 2026)

**TL;DR:** The script couldn't make a strict grasp on the new 20 mm cube, and it
turned out the grasp checker had been measuring the wrong axis all along.

**Problem.** After a month off, I moved to the real bench task: a 20 mm cube on
a 2 inch white square, 8.5 inches in front of the arm. Pick it up, put it back.
The script lifted the cube about 29 mm but never registered a strict grasp.

*How we know:* the checker scored the jaw alignment at 0.888 against a threshold
of 0.906 (25 degrees). I didn't know if this was tuning or geometry.

**Experiment.** Work out, for each of the four pad positions along the jaws,
where it pinches a 20 mm cube, and how the checker decides which way the jaws
close.

*What we hope to learn:* if it's tuning, sweep parameters. If it's geometry,
move the grasp or fix the checker.

**Outcome.** Geometry, twice:

- The checker measured the closing direction as the line between two tip
  markers, which sit 18.6 mm apart vertically. That line is 25 degrees off the
  real closing direction, right at the limit, so pads 2-4 could never pass. Like
  checking a door is shut with a line from the handle to the hinge.
- Moving the grasp to the outermost pad passed 15/15. Then, with my sign-off,
  the checker switched to the true closing direction (between the two pad
  normals). The old 25 mm task's results didn't change, but it turns out its
  grasps had passed the old checker by only 0.002.

**Takeaway:** check the checker. A metric that's almost right can quietly decide
which solutions are allowed to exist.
