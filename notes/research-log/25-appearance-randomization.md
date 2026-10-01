# 25. Can it handle how the real scene looks? (Sep 10, 2026)

**TL;DR:** Randomizing lighting, colors, textures and camera details kept 30/30
in sim and passed a check on real recorded frames.

**Problem.** On the real start frame the policy panicked (entry 24).

*How we know:* on the real frame its first chunk moved the shoulder 25 units and
hit 23 safety holds. On the sim frame: 0.5 units.

**Experiment.** Give every training episode a random look: lights and shadows,
colors and tints, speckled ground, a towel of random size and angle, small
camera jitter, and random brightness, contrast, noise and blur. Same recipe. New
rule before touching the arm: the policy's first chunk on recorded real frames
has to stay still.

*What we hope to learn:* if it passes the real-frame check and keeps its sim
score, it's ready for the arm.

**Outcome.** Normal start 3/3, new positions 30/30 (29/30 under random looks).
On the real frames: shoulder moved 0.7 units, elbow 0.6, zero holds, and it
passed all 16 brightness and contrast edits.

- It can't rely on any particular look, so it keys on what stays the same: the
  cube and the square. Like recognizing a friend in any lighting because you've
  seen them in lots of it.
- Side effect: with a black image it still got 12/30, so it also leans more on
  its joint angles.

This became the policy that ran on the arm.

**Takeaway:** when real frames differ from sim, randomize everything that isn't
the task, and gate on recorded real frames before touching hardware.
