# 14. Can we just clip the gripper? (Aug 6, 2026)

**TL;DR:** Clipping the gripper command to its limits at run time, with no
retraining, gave the first 15/15 with zero safety frames in the project, on two
of three seeds.

**Problem.** The policies overshot the gripper past its closed limit (entry 13).

*How we know:* for two of the seeds, every single violation was the gripper
going below its floor. I didn't know if fixing just that would be enough.

**Experiment.** Re-run the same three trained policies, but clip the gripper
output to its exact limits before sending it. Only the gripper: clipping the
other joints trips rounding errors. Written down beforehand: seeds 101 and 202
should pass, and 303 might not, because its 3 bad frames per try were the
gripper opening too fast, not going past a limit.

*What we hope to learn:* whether this one narrow failure is the whole story.

**Outcome.** Exactly as predicted:

| seed | result | safety frames |
|---|---|---|
| 101 | 15/15 | 0 |
| 202 | 15/15 | 0 |
| 303 | 12/15 | 9 |

- The clip removes any out-of-range command by construction, so the over-squeeze
  just disappears. It does nothing about speed, though. Like a fence: it stops
  you leaving the yard, not running too fast inside it.

The clip became part of every policy after this.

**Takeaway:** once a failure is narrow and pinned down, the cheapest fix at run
time can beat any amount of retraining. And writing the partial outcome down
first means you know it's real when it shows up.
