# 12. Does a better learning-rate schedule fix it? (Aug 6, 2026)

**TL;DR:** Decaying the learning rate cut bad frames 17x and got 12/15, but
bigger budgets made it worse and seeds scattered 12, 12 and 3.

**Problem.** The best policy got 9/15, failing on small gripper overshoots.

*How we know:* 210 safety frames across the 15 tries (entry 11). I didn't know
if better optimization would polish them away.

**Experiment.** Three stages, rules written down beforehand: cosine
learning-rate decay to a small floor versus a fixed rate; then bigger budgets
(150k steps, 300k steps, or width 1024); then two more seeds of the best one.

*What we hope to learn:* if decay polishes the overshoot away and it holds
across seeds, this rung is done.

**Outcome.** Cosine got 12/15 with 12 safety frames, down from 210. The rest
didn't hold up:

- Bigger budgets went backwards (9, 9 and 3 out of 15). The schedule stretches
  with the budget, so a longer run spends longer at a high learning rate and
  ends less polished.
- Seeds came out 12, 12 and 3. Which start failed moved around too: the normal
  start failed while all the shifted ones passed.

Cosine helps because it takes big steps early and tiny ones at the end to
settle. Like parking: fast down the street, slow into the spot.

**Takeaway:** a 12 vs 3 spread across seeds means you're standing on a knife
edge. Don't celebrate one seed, and don't assume a bigger budget is a better
one when the schedule depends on it.
