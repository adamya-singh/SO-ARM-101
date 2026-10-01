# 34. How much of 15/30 is luck? (Sep 12, 2026)

**TL;DR:** The loss reproduced across seeds, but the closed-loop score didn't:
15, 6 and 12 out of 30 for the exact same recipe.

**Problem.** I was about to rank recipes by closed-loop score out of 30, and one
result made me suspicious.

*How we know:* the 480k-step model scored 0/30 with the same per-placement error
as the shift-4 model that scored 9/30. Same quality on paper, 9 successes apart.

**Experiment.** Retrain the 2400-placement, shift-12 recipe twice more, changing
only the random seed (initialization, batch order, shift offsets), and evaluate
all three the same way.

*What we hope to learn:* if the scores agree, 15/30 is real and I can rank
recipes by it. If they spread, one run's score can't rank anything.

**Outcome.** The loss agreed (held-out 5.7e-4, 5.8e-4, 3.4e-4, all under 6x
training error). The scores didn't: 15, 6 and 12 out of 30.

- It's really 10 samples, not 30. The sim is deterministic, so all 3 tries at a
  placement end the same way. A score out of 30 is a score out of 10 placements,
  times 3. Like grading a student on a 10-question quiz.
- Which placements succeed moves between seeds. Five were won by at least two
  seeds, and four were lost by all three.

From here on, a recipe gets judged by its held-out loss across seeds and its
closed-loop score pooled over seeds (33/90 for this one).

**Takeaway:** measure your noise before you rank anything. Retraining with a
different seed is the cheapest way to find out how big a difference has to be
before it means something.
