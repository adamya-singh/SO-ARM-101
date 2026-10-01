# 01. Can the task even be solved? (Aug 1, 2026)

**TL;DR:** The simulator made the grasp physically impossible, so seven months
of RL had been training on a broken task.

**Problem.** I spent seven months using RL to train policies (SmolVLA, ACT) to
pick up a 25 mm cube in sim, and none of them could do it. Every policy just
pinched the cube by a corner and dropped it.

*How we know:* the grasp detector checks for a strict grasp (both jaw pads flat
on opposite faces of the cube), and across 12,000-episode runs it never fired
once. What we didn't know was whether the learning was broken or the sim was.

**Experiment.** Take learning out of the picture. I wrote a privileged
controller, basically a script that cheats by reading the cube's exact position
from the sim, and had it try 15 pickups.

*What we hope to learn:* if the cheating script can do it, the task is fine and
the problem is learning. If it can't, no policy ever could have.

**Outcome.** The cheating script failed too. Turns out the sim itself made the
grasp impossible:

- MuJoCo collides meshes using their convex hull (think shrink-wrap). The jaw is
  C-shaped, so the hull filled in the gripper's mouth with ~15 mm of invisible
  solid right where the cube goes.
- The jaws only close parallel at a 2-7 mm gap. At 25 mm they make a ~30 degree
  wedge that squeezes the cube out like a watermelon seed.

I switched to the MuJoCo Menagerie arm model, which lots of people already use
and has properly built collision shapes. One controller bug was left, and I
found it by watching the sim: the fixed jaw was coming down on top of the cube
instead of beside it. After fixing that, the script got 15/15 strict pickups.

**Takeaway:** seven months of RL results were measuring a physics bug, not
learning. Before training anything, prove the task is solvable with a script
that cheats.
