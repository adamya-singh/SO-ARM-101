# 02. Can a small network copy the cheating script? (Aug 2, 2026)

**TL;DR:** A network copied the script almost perfectly on paper, then failed
every real try, because tiny errors piled up until the arm was somewhere new.

**Problem.** After entry 01 the task was solvable, but only by a script that
cheats. I needed a learned policy, and the old ones were useless.

*How we know:* I re-ran every older policy on the fixed sim, 165 rollouts total,
and none of them even reached the cube. I didn't know if learning could work
here at all.

**Experiment.** Behaviour cloning, basically copying an expert: record the
script once (450 steps), train a small MLP to output the same command from the
same state until it's almost exact, then let it drive the arm 3 times.

*What we hope to learn:* if a near-perfect copy succeeds, the pipeline works and
I can scale up. If it fails, I need to know why.

**Outcome.** It hit the "basically memorized" bar (training error under 1e-6)
and still failed all 3 tries, lifting the cube 0.16 mm:

- Small errors compound. The copy is off by a hair every step, and by step 74
  its joints are half a degree off. Think of a steering wheel that's slightly
  off: fine for a second, in a ditch after a mile.
- Off the recording, it's guessing. It only saw states the script visited, so
  at contact it's somewhere new and never closes on the cube. Like memorizing an
  answer key and then getting a question with one number changed.

To prove it, I fed the network the script's exact recorded states and replayed
its commands: full pick and place. The outputs were fine. The drift was the
problem.

**Takeaway:** a perfect fit on training data says nothing about what happens
once the model's own mistakes change its inputs (called covariate shift). A
policy that only copied a perfect path has never seen a mistake, so it can't
fix one.
