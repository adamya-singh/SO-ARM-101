---
name: experiment-analyze-explain
description: Diagnose an ML result (an underperforming run, a surprising number, a "should we scale data, model or steps?" question) with a fixed sequence of measurements, decide what to change next, record it, and explain the whole chain to a new intern in plain technical English. Invoke when the user asks why a run behaved as it did, what to scale, or calls /experiment-analyze-explain; invoke on your own when a run underperforms its baseline or a result would be worth an intern's lesson.
---

# Experiment, analyze, explain

A run gave a number. Before changing anything, find out *why* the number is what it is, using
measurements that separate the usual suspects (not enough training, not enough capacity, not
enough data, an irreducible ambiguity in the task, a bug). Then decide, record, and teach.

The process below is the one that diagnosed the 2026-09-10 placement run in this repository
(`notes/vision-rung-notebook.md`, "Analysis of run 4"): training loss looked fine, closed-loop
success was 3/30, and a single held-out measurement showed a 136x generalisation gap at the one
chunk that mattered. Every step exists because skipping it once cost a wrong conclusion.

## Part 1: measure (do all of these; each answers a different question)

1. **Frame the decision, not the curiosity.** Write one sentence: "The result is X; the decision
   that hinges on it is Y (train longer / bigger network / more data / change the task / fix a
   bug)." Every measurement below must bear on Y. If no decision hinges on it, stop.
2. **Establish the baseline.** Find a run that *worked* under a comparable recipe and compute
   every number you are about to compute for the failing run on that baseline too. A number
   without a baseline is not evidence. In this repo: `experiments/<run>/training.jsonl`,
   `evaluation_summary.json`, `models/vision_h90/*/report.json`.
3. **Loss curves over training, compared at the same steps.** Read the logged loss at fixed
   fractions of the budget (e.g. 10k / 60k / 120k) and the ratio of improvement in the second
   half. Distinguish "still learning" from "the schedule is annealing": under a cosine or decaying
   learning rate the last stretch always drifts down a little (1.2-1.4x here). A run that is far
   above the baseline at *every* stage and improves less in the second half is heading to a higher
   floor, not lagging.
4. **Decompose the error along the task's structure.** Group the evaluation error by the part of
   the task each sample belongs to (here: chunk start step, i.e. the survey move, the first
   descent chunk at step 90, later boundaries, the tail; or region of the workspace; or object
   class). Compare each group against the baseline's group. The failure usually concentrates in
   one place, and that place is the skill the new setting added.
5. **Train versus held-out on the same metric.** This is the decisive measurement. Build a dataset
   the model never trained on (here: run the teacher on the held-out suite with `store_frames`,
   see `tools/heldout_fit.py`) and score the checkpoint on it with the same decomposition as
   step 4. Read the ratio held-out / train per group:
   - ratio near 1, both low: the model generalises; look elsewhere (evaluation, gate, hardware).
   - ratio near 1, both high: capacity or an irreducible ambiguity in the inputs (the target is not
     predictable from what the model sees); more steps will not fix it, a better encoder or
     better inputs might.
   - ratio far above 1 (10x or more): memorisation of the training scenes; more *data* first;
     training longer makes it worse; a bigger head alone makes it worse.
6. **Look at the model where the error concentrates.** Count parameters per component (encoder
   versus head), relate the encoder's receptive field / stride to the size of the object it must
   localise (a 12-25 px cube on 8 px patches covers 2-3 patches), and note which part would find
   memorisation cheaper than learning the rule.
7. **Check for ambiguity in the data itself.** Are there training samples whose target cannot be
   inferred from the input (e.g. chunk starts whose frame does not show the object yet)? They put
   a floor on the loss and teach the model to hedge; consider restricting the sample set to the
   starts that occur at inference (here: multiples of the chunk length).
8. **Rule out the pipeline.** One cheap parity check per changed component (e.g. the GPU frame
   store is bitwise identical to the host path by test; the stride keeps every inference-time
   start in the training set). State explicitly which changes were proven neutral.

## Part 2: decide

Write the decision as an ordered list of levers with the evidence for the order, e.g.
"data first (136x held-out gap), encoder second (8k-parameter trunk, 2-3 patches per object),
steps last (only once held-out tracks train)". Name the single number that will judge the next
run (here: held-out loss at chunk start 90) and the threshold that flips the decision. Prefer
one change per run unless GPU hours make a combined run clearly cheaper; then say which change
you are confounding and why that is acceptable.

## Part 3: record

- Notebook entry (`notes/vision-rung-notebook.md`): the tables, the ratios, the decision, the
  number to watch. Runbook status if the plan changed. Memory file for the project state.
- Keep the analysis artifacts (held-out capture manifest, `heldout_fit.json`) force-added under
  `artifacts/` so the numbers are reproducible; commit.

## Part 4: explain to a new intern

Write for someone bright but new to the project and to ML practice. Structure:

1. **The setup, in one paragraph**: what is being trained, from what data, what "loss" means
   here, and the question being asked.
2. **One section per measurement step**, each with exactly these parts:
   - *Why* (what question this step answers and what would change depending on the answer),
   - *What we did* (one or two sentences, name the tool or file),
   - *What it says* (the table or the numbers, then the reading in words),
   - *Conclusion* (what is now ruled in or out).
3. **What we conclude overall, and what we are doing**: the ordered levers and the number to
   watch.
4. **One general lesson** the intern can carry to other projects (e.g. "a low training loss on
   its own tells you almost nothing; always check data the model did not train on, and break the
   number down by the part of the task that matters").

Style rules: define every term the first time (chunk, survey pose, held-out, memorisation); one
idea per sentence; numbers in tables, not prose; no jargon as decoration; say what a result
*rules out* as well as what it suggests; never present the training loss as evidence of
generalisation; never claim a cause you did not measure.

## Repository pointers

- Loss logs: `experiments/<run>/training.jsonl` (step, loss, steps_per_second).
- Closed-loop results: `experiments/<run>/evaluation_summary.json` (successes per suite,
  `success_by_region` for placement runs), per-rollout `evaluation.json` (failure categories,
  safety frames, height gain).
- Train vs held-out at chunk boundaries: `tools/heldout_fit.py --checkpoint <model.pt>
  --heldout-manifest <capture manifest>`; make the held-out capture with
  `capture_oracle_demonstrations(..., store_frames=True)` on the held-out suite.
- Real-frame gate (sim-to-real check on recorded frames): `tools/check_policy_on_real_frames.py`.
- Network: `src/so_arm101_v2/learning/vision.py` (`build_vision_chunked_model`, encoders v1/v2).
