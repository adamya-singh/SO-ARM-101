# 08. How do we stop it asking for impossible moves? (Aug 3, 2026)

**TL;DR:** Training on noisy inputs with a penalty for impossible commands gave
the first learned policy to pass: 3/3, full pick and place, zero safety frames.

**Problem.** The 90-step chunk policy could grasp, but it saturated: it asked
for commands past the arm's limits.

*How we know:* 150 clipped, 82 limited and 20 unsafe frames per try (entry 07).
I didn't know which fix would work, or if the correction data would help here.

**Experiment.** Five versions of the 90-step policy, all run to the end instead
of stopping at the first pass: a hard decoder that physically can't output
out-of-bounds commands; a soft penalty (add noise to the inputs while training,
and penalize any predicted command outside the safe range or moving too fast);
the correction data alone; and two combinations.

*What we hope to learn:* running all five tells me which ingredient matters,
not just whether something passes.

**Outcome.** The soft penalty alone passed: 3/3, full pick and place through
release and retreat, 32.4 mm lift, zero safety frames, about 10 minutes of
training.

- The noise shows the network what to do slightly off its training states, and
  the penalty keeps those answers in bounds. Like practicing a route with
  guardrails on both sides.
- The hard decoder never clipped and failed anyway: the safety check compares
  against the arm's measured position, and servo lag tripped it. I'd written
  that risk down beforehand.
- The correction data hurt every version it was in. Over 90 steps, nearly
  identical states have totally different futures.

After the MuJoCo mix-up (entry 09) I re-ran this on the right version: same
winner, byte-identical checkpoint.

**Takeaway:** run every arm of an ablation. Five answers taught way more than
one pass, and a soft penalty beat a hard guarantee.
