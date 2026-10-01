# 05. Is the network just missing information? (Aug 2, 2026)

**TL;DR:** Giving it velocity, contact flags and recent history didn't help:
all three versions memorized fast and failed 0/3.

**Problem.** The clone only sees joint positions, the cube's position and a
progress clock. No velocities, no idea if it's touching the cube. Maybe it can't
tell a recovery state from a normal one.

*How we know:* every clone so far failed 0/3, and its input list is missing all
of that. I didn't know if any of it mattered.

**Experiment.** Add inputs in three steps, same data and model each time:
velocities (26 inputs), plus contact flags (29), plus the previous two frames
(57). Three tries each from the normal start.

*What we hope to learn:* if the right input lets it tell states apart, closed
loop should improve. If all three fail, missing inputs aren't the problem on
their own.

**Outcome.** All three hit the memorization bar in under 2,600 steps, then
failed 0/3. None even grasped the cube:

- The contact flags and history were identical between each normal state and
  its recovery state, because each nudge started from the same previous moment.
  So they added zero new information. Like asking a witness who was looking the
  other way.
- Velocities did tell the states apart, and it still failed. The real gap was
  coverage: the policy wanders through a continuous range of states, and it had
  seen 8.

*Note:* also on the wrong MuJoCo version (entry 09), not re-run.

**Takeaway:** before adding an input, check that it actually differs between
the cases you want the model to separate.
