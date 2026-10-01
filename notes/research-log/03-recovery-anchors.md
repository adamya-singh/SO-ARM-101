# 03. Can a few recovery examples teach it to recover? (Aug 2, 2026)

**TL;DR:** I added 8 examples of the script recovering from small mistakes, and
the clone got worse: it failed all 15 tries and started slamming into its joint
limits.

**Problem.** The clone from entry 02 drifts off the script's path and has no
idea how to get back.

*How we know:* it failed 3/3, its joints were half a degree off by step 74, and
the same commands replayed on the script's states worked fine. I didn't know if
showing it a few recoveries would be enough.

**Experiment.** At 8 points across the task (approach, contact, closure, lift,
carry, place, release and so on), nudge the arm half a degree off course, record
what the script does from there, and add those 8 rows to the 450 training rows.
Same model, same training.

*What we hope to learn:* if a few labeled recoveries teach correction, the clone
should pass. If not, sparse examples aren't enough.

**Outcome.** It still hit the memorization bar, then failed 0/15 (5 cube
positions, 3 tries each), every try killed by safety violations:

- On the normal start, 291 commands per try got clipped at the joint limits,
  and it lifted the cube 5 mm.
- Even when I started it exactly at the recovery states it was trained on, it
  only finished 1 of 8 (the old clone finished 3).

Eight points is nothing next to a continuous path. To fit them, the network
bends hard between them, and those bends become huge commands. Like teaching
someone to drive with 8 photos of near-crashes.

*Note:* these numbers came from the wrong MuJoCo version (entry 09) and were
never re-run; the conclusion didn't depend on the exact counts.

**Takeaway:** a handful of recovery examples doesn't teach recovery, and it can
break what already worked. The memorization bar couldn't see any of it.
