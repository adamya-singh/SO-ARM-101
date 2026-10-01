# 04. What if the recovery examples count for less? (Aug 2, 2026)

**TL;DR:** Turning down the weight of the 8 recovery rows didn't help: every
version that trained still failed 0/3 on safety.

**Problem.** The 8 recovery rows broke the clone's safety (entry 03).

*How we know:* on the normal start, 291 clipped commands and 13 unsafe contacts
per try. I didn't know if a lighter touch would keep it safe while still
teaching something.

**Experiment.** Retrain with the same 8 rows counting 0.10, 0.25 and 0.50 as
much as a normal row in the loss, everything else frozen. I also wrote a static
check that scans the model's commands for anything out of bounds.

*What we hope to learn:* if one weight keeps the normal start safe, use it. If
none do, sparse recovery rows are a dead end.

**Outcome.** None worked:

| weight | memorization bar | normal start | per try |
|---|---|---|---|
| 0.10 | passed | 0/3 | 119 clipped, 36 limited, 3 unsafe |
| 0.25 | missed | not run | |
| 0.50 | passed | 0/3 | 178 clipped, 175 limited |

- My static check was useless: even the model with no recovery rows at all
  failed it. Like a smoke alarm that goes off when you make toast. It became a
  log line, and a real 3/3 run in the sim became the first safety check I
  trusted.

*Note:* also run on the wrong MuJoCo version (entry 09), not re-run.

**Takeaway:** if a dial doesn't fix the problem at any setting, the idea is
wrong, not the setting. And test your checker on a known-good case before you
trust it.
