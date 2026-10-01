# 28. Can it do it again? (Sep 10, 2026)

**TL;DR:** No: the repeat closed beside the cube, and every real grasp landed in
about the same spot wherever the cube was, so the policy had no tolerance for
placement.

**Problem.** Episode 10 was a single success.

*How we know:* one run proves possible, not reliable (entry 27).

**Experiment.** Run it again from the start pose a few more times (episodes
11-13).

*What we hope to learn:* how repeatable it is, and what changes when it fails.

**Outcome.** The placement check refused 11 and 12, but the same cube read 300
mm away from one arm pose and 325 mm from another, so I made the check advisory.
Episode 13 ran all 480 steps with zero holds, in brighter daylight, and closed
beside the cube.

| run | outcome | where the jaws closed (x, y mm) |
|---|---|---|
| episode 10 | success | (8, 290) |
| episode 13 | miss | (1, 284) |
| episode 4 | miss | (−13, 283) |
| sim | | (−1, 288) |

- Every real grasp landed within about 10 mm of the sim's grasp point while the
  cube sat about 20 mm beyond it. It trained on ±10 mm, so it goes to roughly
  where it trained. Like a claw machine that always drops in the same spot.
- The real jaws also close about 15 mm higher than the sim's, every time.

Next: let the square and cube go anywhere in a 14 × 10 inch rectangle, at any
angle, during training.

**Takeaway:** the spread across repeats tells you what the model actually
learned. Here it learned a location, not how to look for the cube.
