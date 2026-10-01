# 07. What if it decides less often? (Aug 3, 2026)

**TL;DR:** Predicting 90 actions at a time got the first learned strict grasp
in the project, and then the arm drove itself into its joint limits.

**Problem.** Four fixes to the step-by-step clone failed (entries 03-06). It
makes 450 decisions per episode, and every one adds a little error.

*How we know:* every version failed 0/3, with joint error building from step 1.
I didn't know if fewer decisions would help.

**Experiment.** Action chunking (the idea behind ACT): read the state once,
output the next H commands, and run them all without looking again. I tried
H = 10, 30 and 90 on the original 450 rows. To pass: 3/3 with zero safety
violations.

*What we hope to learn:* H = 1 is the failing clone and H = 450 is just
replaying the script, which works. Somewhere in between might pass.

**Outcome.** None passed, but H = 90 got the first learned strict grasp and
lifted the cube 60.7 mm through every lift milestone. Then it failed on safety:
150 clipped, 82 limited and 20 unsafe frames per try.

- Chunks help because 5 decisions drift far less than 450. Like navigating by a
  few landmarks instead of second-guessing every step.
- Near the end of a blind chunk, the network asks for things it never saw in
  training: the gripper below its floor from step 281, the wrist to −5.2 rad
  against a ±3.14 limit. Every training command was safe, and the loss never
  punishes an impossible one.

*Note:* on the wrong MuJoCo version (entry 09), not re-run; entry 08 was.

**Takeaway:** changing the architecture fixed what four data fixes couldn't.
And a loss on good examples tells you nothing about what the model does outside
them.
