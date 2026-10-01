# 10. Is it a policy or a recording? (Aug 4, 2026)

**TL;DR:** Moving the cube 1.5 mm broke the first policy. Retraining on 5 cube
positions made it almost work everywhere, but not quite anywhere.

**Problem.** The policy from entry 08 learned from one recording of one cube
position. It might just be replaying it.

*How we know:* its margins were razor thin: its gripper command came within
0.0007 of the limit, and it used 74% of the fastest allowed step. I didn't know
how it handled anything new.

**Experiment.** Run it with the cube shifted ±1.5 mm in x and y (5 starts × 3
tries), plus starts from 8 states mid-task. Written down beforehand: if it only
works at the normal start, record the script on all 5 starts (2,250 rows) and
retrain once with the exact same recipe.

*What we hope to learn:* if it passes, move on to vision. If only the normal
start works, it's a recording.

**Outcome.** It's a recording. Normal start 3/3, every shifted start failed
with 43 to 734 safety frames per try. The retrain flipped the picture:

- It got 0/15 again, but in all 5 starts it grasped the cube, lifted it 42-45
  mm, carried it and let go, and it finished 7 of 8 mid-task starts. It missed
  on 1-2 clipped frames, or by holding the grasp 24-26 frames when 30 are
  required.
- It's underfit, not broken: 5x the data, same 30k steps, and its error came out
  8x higher. Like studying five chapters in the time you used to spend on one.

**Takeaway:** test a policy away from its training start before celebrating.
"Works in one place" and "almost works everywhere" are different failures with
different fixes, and this one pointed at more training.
