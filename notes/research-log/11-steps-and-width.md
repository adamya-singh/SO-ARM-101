# 11. Do more steps or a wider network finish the job? (Aug 6, 2026)

**TL;DR:** The best combination got 9/15, and every clipped or limited frame
turned out to be the gripper misbehaving after the script's recording ran out.

**Problem.** The 5-start policy almost worked everywhere but scored 0/15
(entry 10). It looked underfit.

*How we know:* its training error was 8x higher than the one-start memorizer's
on the same 30k steps. I didn't know if steps or size was the bottleneck.

**Experiment.** A 2×2: 3x the steps (90k), 2x the width (512), or both. Judged
on the 15-try suite.

*What we hope to learn:* whichever change closes the gap is where the budget
goes.

**Outcome.** Steps 3/15, width 3/15, both 9/15. Every failure was a safety
violation after the whole task had basically gone right. The best run in closed
loop had the worst training error.

So I went frame by frame through every clipped or limited command:

- All of them were the gripper, and 27 of 33 came after step 450. The script's
  recording is 450 steps long but the task runs for 480, and the progress clock
  stops at the end of the recording. So the last chunk is pure improvisation.
  Like a recipe that ends three steps before the dish is done.
- The other 6 were at release, where the gripper opens fast.

**Takeaway:** before turning more knobs, find the exact frames that fail. It
told me the problem wasn't capacity, it was a mismatch between how long the
demo was and how long the task was. And training error didn't even rank the
runs correctly.
