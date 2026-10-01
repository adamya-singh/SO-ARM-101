# 36. Why does a policy that looks fine on paper still miss? (Sep 18, 2026)

**TL;DR:** Measured in millimetres, the arm lands about 10 mm off against an
8 mm tolerance, and its second look at the cube fixes none of it.

**Problem.** Every placement policy scored somewhere from 6 to 15 out of 30, and
the failures were cubes caught by an edge. I had concluded the loss had stopped
predicting success.

*How we know it was wrong:* the policy that works 30/30 on a fixed target has a
loss of 2.5e-6. These had 3e-4 to 6e-4, 150x worse. In real units that's about
0.3 degrees of joint error versus 3, or roughly 1.7 mm versus 10 mm at the jaw.

**Experiment.** For every rollout, compute where the jaw was just before
closing, relative to the cube, using the working policy's jaw position as the
target. Then check if the error is a bias or a scatter, and whether the second
chunk (predicted from the close-up view at step 180) shrinks it.

*What we hope to learn:* how big the grasp tolerance is, how far off we are, and
which part of the policy to fix.

**Outcome.**

- Tolerance is about 8 mm. Within 8 mm, 13 of 20 tries succeeded; beyond it, 3
  of 22. The policies scatter by about 10 mm (8 mm at 4800 placements), which is
  exactly a third to a half succeeding.
- It's scatter, not bias (slope 0.95 to 1.00). One distant look at a cube 12 to
  25 pixels wide is just good to about 10 mm.
- The second look corrects nothing: 11.4 mm before that chunk, 11.2 mm after.
  In training the script is always perfectly centred at step 180, so the network
  never saw an off-centre view with its fix. It's entry 02's covariate shift
  again, hidden one chunk later.

Next: push the script off course on purpose during training, so the second look
learns to centre on the cube.

**Takeaway:** convert your loss into real units, compare it to something that
works, and measure the failing thing directly. A model trained only on perfect
runs can't fix mistakes, no matter how good the loss looks.
