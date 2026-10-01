# 13. What if the recording covers the whole task? (Aug 6, 2026)

**TL;DR:** I extended the script's recording to all 480 steps and it fixed
nothing, because that failure was already gone. The real problem was the
gripper squeezing past its limit.

**Problem.** My plan blamed the leftover failures on the recording being 450
steps while the task runs 480.

*How we know:* in entry 11, 27 of 33 bad frames came after step 450. I didn't
re-check that after entry 12.

**Experiment.** Re-record the script for the full 480 steps by holding still at
the end, and retrain three seeds on it.

*What we hope to learn:* my written prediction was 15/15 on all three seeds.

**Outcome.** 9, 9 and 12 out of 15. There were zero bad frames past step 450,
but the cosine runs in entry 12 already had zero there. I ran the planned next
step without reading my own latest failure analysis.

The actual failures were all the gripper, and the frames showed why:

- The script grips by commanding the gripper to within 0.0035 of its closed
  limit, and the policy overshoots below the limit, as far as −0.02. Like aiming
  for the very edge of a table: a tiny overshoot and you're off it.
- That overshoot is bigger than any safety margin I could build into the
  training targets, so retraining can't absorb it.

**Takeaway:** re-read the newest failure analysis before running the step your
plan says comes next. Plans go stale faster than you'd think.
