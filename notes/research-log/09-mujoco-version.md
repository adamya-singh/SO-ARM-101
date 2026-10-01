# 09. Why did the same run give different numbers? (Aug 3, 2026)

**TL;DR:** A stray pip install silently swapped the physics engine mid-project,
so 34 results came from a different simulator than the rest.

**Problem.** I was speeding up the evaluation code, and to check nothing
changed, I re-ran old results expecting identical bytes. Some didn't match.

*How we know:* the Aug 1 preflight only reproduced with one environment setting,
and everything from Aug 2-3 only reproduced without it. I didn't know which
runs were which, or why.

**Experiment.** Re-run every result into a copy and compare it byte for byte,
under each setting, to sort the runs into two groups and find the cause.

*What we hope to learn:* exactly which experiments are affected, and whether
any decision rests on bad numbers.

**Outcome.** On Aug 2 at 19:03, a `pip install --user` put MuJoCo 3.11 in my
user folder, and Python checks that folder before the project's environment, so
it quietly replaced the pinned 3.9:

- The two versions differ in the last decimal place on step 1, and once the
  jaw touches the cube that grows into a visibly different run. Like two
  calculators that round slightly differently.
- 34 results were affected, covering entries 03-08. None of my checks could
  see it, because they hashed the inputs, not the simulator.

Fix: pinned 3.9.0, turned off the user folder in the environment, and added a
check that refuses to run on any other version. The 34 results went into
quarantine. I re-ran what later work depended on (entry 08's gate: same
result). Entries 03-07 were not re-run, so their numbers are 3.11 numbers.

**Takeaway:** pin your simulator version in code, not just in a requirements
file, and re-run old results bit for bit now and then. That's the only check
that catches this.
