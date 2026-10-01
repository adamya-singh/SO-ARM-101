# 29. Can it find a cube anywhere on the table? (Sep 10, 2026)

**TL;DR:** No: trained on 400 random placements it got 3/30 on new ones, because
it memorized the placements instead of learning to read where the cube is.

**Problem.** The policy needed to handle the cube anywhere, not one spot
(entry 28).

*How we know:* real grasps landed in the trained spot no matter where the cube
was.

**Experiment.** Put the square and cube anywhere in the rectangle at any angle,
add a raised "survey" pose so the camera can see the whole area, and certify the
script 30/30 on it. Train on 400 placements with the same recipe and test on 10
new ones. Then compare training and test error on each chunk.

*What we hope to learn:* if it works across the table, go back to the arm.

**Outcome.** Normal start 0/3, new placements 3/30 (all from one placement). I
first called it underfit. The error breakdown said otherwise:

| chunk | training error | test error | ratio |
|---|---|---|---|
| first descent (from the survey frame) | 1.4e-5 | 1.9e-3 | **136x** |
| the others | | | 3-4x |

- It fits 400 survey images without learning "cube here, so go there." The image
  encoder had 8k parameters, looking at a cube 12-25 pixels wide.

**Takeaway:** compare training and test error on the exact step that matters.
"Underfit" and "memorized" need opposite fixes, and I almost picked the wrong
one.
