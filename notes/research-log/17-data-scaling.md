# 17. Is it a data problem? (Aug 7-8, 2026)

**TL;DR:** More random positions alone took the state policy to a perfect 30/30,
and vision reached 24-30/30 once training time grew with the data.

**Problem.** At 25 positions, the state policy got 18/30 and vision 0/30.

*How we know:* the state policy failed exactly where the 25 positions left gaps
(entry 16). I didn't know if vision would follow.

**Experiment.** Same recipe, same 10 test positions, training on 50, 100, 200
and 400 positions. Vision first at a fixed 20k steps, then at 400 positions with
120k steps on 3 seeds.

*What we hope to learn:* if it's a data problem, the curve climbs.

**Outcome.**

| positions | state | vision (20k steps) |
|---|---|---|
| 50 | 27/30 | 6/30 |
| 100 | 27/30 | 14/30 |
| 200 | 27/30 | 9/30 |
| 400 | **30/30**, 0 bad frames | 18/30, 642 bad frames |

Vision at 400 positions and 120k steps: 24, 30 and 24 out of 30 across seeds.

- Fixed steps with more data means fewer passes over each example, and vision
  needs more of them. Like doubling the reading list without more study time.
- Undertrained vision is unsafe vision: 642 bad frames at 400 positions and 20k
  steps.

*Note:* all of August ran on a sim whose joint map and camera turned out to be
wrong (entries 20-21), so these numbers don't transfer to the real arm. The
lessons do.

**Takeaway:** scale training steps with the data. And one 30/30 seed with
others at 24 is a promising recipe, not a solved one.
