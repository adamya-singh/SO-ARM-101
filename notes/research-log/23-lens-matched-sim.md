# 23. Can the sim see through the real lens? (Sep 9, 2026)

**TL;DR:** I made the sim render through the real lens instead of straightening
real images, and the same recipe got 30/30.

**Problem.** The real camera bends the image (barrel distortion), the sim was a
perfect pinhole, and my first lens fit only held near the centre.

*How we know:* the fit was only valid out to 929 px from centre, and the frame's
corners are 1,101 px out. The policy sees the whole frame, edges included.

**Experiment.** Capture 17 more checkerboard views at the edges and corners (46
total), with a live preview window telling me where to hold the phone, and fit
a richer 8-coefficient lens model. Then make the sim render wide and resample
through that lens, so its images look like the real camera's. Same training
recipe.

*What we hope to learn:* whether the model covers the full frame, and whether a
policy trained on lens-matched images needs no fixing at deploy time.

**Outcome.** 1.65 px error out to the corners, field of view 44.85 degrees. The
run got 3/3 and 30/30 with zero safety frames (black images 0/30), and the
position that failed in entry 22 now went 3/3. End to end it took 94 minutes
instead of 5.5 hours, thanks to caching frames in RAM.

- Matching the sim to the camera means nothing extra at deploy time. Like
  learning to read someone's handwriting instead of making them type.

One seed, so 27 to 30 is suggestive, not proven. And this policy failed on the
real arm (entry 24).

**Takeaway:** make the sim's observation match the real sensor exactly, flaws
included.
