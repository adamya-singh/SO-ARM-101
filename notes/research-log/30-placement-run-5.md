# 30. Does 3x the placements fix it? (Sep 10, 2026)

**TL;DR:** No: 1200 placements and twice the training got the same 3/30, and the
memorization gap grew to 417x.

**Problem.** Run 4 memorized its 400 placements (entry 29).

*How we know:* its test error on the first descent chunk was 136x its training
error. The usual fix for memorizing is more data.

**Experiment.** 1200 placements, 240k steps, same recipe and test set.

*What we hope to learn:* if data is the problem, the gap shrinks.

**Outcome.** Same 3/30, with the same mix of failures. Training error on that
chunk halved, to 7e-6. Test error stayed at 2.9e-3, so the gap went from 136x to
417x. The survey move itself still worked on real frames.

- It memorized 1200 images as easily as 400. Like handing a bigger flashcard
  deck to someone who memorizes cards.

No live trial with this one. Next: a proper scaling ladder to find out which
knob, if any, helps.

**Takeaway:** when more data makes the gap bigger, data isn't the bottleneck.
