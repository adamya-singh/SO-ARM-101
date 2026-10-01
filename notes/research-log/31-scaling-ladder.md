# 31. Do more data, a bigger network, or longer training fix it? (Sep 11, 2026)

**TL;DR:** None of them did, and a dumb nearest-neighbour lookup beat every
network, so the problem was reading the image, not scale.

**Problem.** Once the cube could be anywhere on the table, the vision policy
fell apart. It memorized its training placements instead of learning where the
cube is.

*How we know:* on 10 placements it never trained on, its error on the first
descent chunk was 136x its training error with 400 placements, and 417x with
1200, and closed loop was 3/30 both times (entries 29-30). I didn't know which
knob to turn.

**Experiment.** A scaling ladder, Chinchilla-style: 15 models from one
2400-placement dataset, sweeping placements (300 to 2400), encoder size (0.8M
to 5M parameters) and training steps (60k to 480k), all scored on the same 10
held-out placements.

*What we hope to learn:* if one curve bends down, that's where to spend GPU. If
they're all flat, scale isn't the problem.

**Outcome.** All flat. Held-out error stayed between 2e-3 and 7.5e-3 while
training error hit 1e-7, and the best model got 9/30:

- More data helped barely. On the fitted curve, reaching a usable error would
  take about 250x the placements.
- Bigger networks memorized harder. Like handing a student who memorizes
  answers a bigger notebook.

Then the real test: for each held-out placement, I just reused the recorded
answer from its nearest training placement (2 to 9 mm away). That lookup table
beat every network, 5x on average and 300x on the worst placement. The data was
dense enough. The network just couldn't read the cube's position from the image.

**Takeaway:** before scaling anything, check a dumb baseline. If a lookup table
beats your model, more data won't save it; the model isn't extracting what's
already in the input.
