# 32. Can we make memorizing impossible? (Sep 12, 2026)

**TL;DR:** Sliding every training image a few pixels at random cut memorization
350x and took closed loop from 0/30 to 9/30.

**Problem.** The network was memorizing exact images instead of learning where
the cube is.

*How we know:* at 1200 placements its held-out error was 5250x its training
error, it scored 0/30 on new placements, and a lookup table beat it (entry 31).
I didn't know if the network could localize the cube at all.

**Experiment.** Random-shift augmentation (the DrQ trick): every time a training
image is used, pad it and crop it back at a random offset of up to 4 or 12
pixels, with the same label. Nothing else changed.

*What we hope to learn:* if it's memorizing exact pixels, removing that fixed
identity forces it to learn where the cube sits in the scene, so the gap should
collapse. If nothing changes, the network can't localize and I need a different
model.

**Outcome.** The gap collapsed:

| shift | held-out / training error | closed loop |
|---|---|---|
| 0 px | 5250x | 0/30 |
| 4 px | 168x | 9/30 |
| 12 px | 15x | 9/30 |

- Why it works: a memorized lookup needs the same pixels every time, and now it
  never gets them. Like flashcards that get shuffled and reworded every pass, so
  the only way through is learning the concept.
- The failures changed kind. At 0 px the arm collided or missed the cube
  completely. At 12 px it found the cube but often caught it by an edge and
  dropped it at release.

**Takeaway:** when a model memorizes, make memorizing impossible instead of
adding capacity. One change to the data loader beat 15 scaling runs.
