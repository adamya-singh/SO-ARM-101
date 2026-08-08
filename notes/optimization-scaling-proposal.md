# Optimization-Scaling Tranche (Proposal, pre-registered before any run)

## Question

The broader-evaluation tranche (`broader_evaluations/68c56c66d2dd3bb9`,
status `starts_not_resolved`) established that retraining the frozen H=90
noise-penalty recipe on the five-scenario 2,250-row oracle capture produces a
near-complete generalizer: strict grasp, full lift ladder, carry, and release
in all five scenarios and 7/8 anchor handoffs, failing only the 30-frame
strict-hold window and a handful of envelope frames. Offline telemetry pins
the residual gap as underfit at the frozen budget (`1.94e-5` MSE at 30k steps
on 5x data versus `2.37e-6` for the single-scenario memorizer). The recorded
next step is: scale the optimization budget (steps and/or the now-earned
width increase) to the 2,250-row dataset — same recipe, same gates, nothing
else.

Does scaling the optimization budget close the last-mile gap, and is the
binding factor steps, width, or both?

## Design (2x2 factorial, one cell already immutable)

All candidates train on the immutable five-scenario capture
`oracle/fixed_pick_place_v3/3f8eb5499eb18128` (2,250 rows, 5 episodes,
content sha `5d7401d8e662609f10f14d7cb4e23726e0a4ac011164cd5af1ab868f8a6b6c51`)
with the frozen recipe unchanged except the two scaled factors:
chunk_horizon 90, saturation_mode `noise_penalty_v1`, decoder_eta 0.9,
margin_act 0.002, noise_sigma 0.05, penalty_weight 1.0, seed 101, lr 1e-3,
full-batch Adam, capture-wide normalization, phase_state features,
episode-bounded chunk lookahead.

| Cell | Width | Steps | Status |
| --- | ---: | ---: | --- |
| baseline (control) | 256 | 30,000 | already run; immutable in `68c56c66d2dd3bb9` (retrained record) |
| `steps90k` | 256 | 90,000 | this tranche |
| `width512` | 512 | 30,000 | this tranche |
| `width512_steps90k` | 512 | 90,000 | this tranche |

Steps scale is 3x, not the 5x data-parity value, because single-threaded
deterministic training at width 512 measures ~0.29 s/step: 150k steps would
not reliably fit the validated overnight compute window alongside the other
arms. 90k at width 512 (~9 h) does. This constraint is recorded here before
any run.

Width 512 requires widening the chunked-clone width allowlist (128/256 ->
128/256/512). The oracle-distillation (tiny-lane) allowlist is closed and is
not touched.

## Gates (identical to the broader tranche)

- **Stage A** — full `fixed_pick_place_v3` suite: 5 scenarios x 3 repeats =
  15 rollouts, deterministic. Pass: 15/15 success, zero
  clip/limit/nonfinite/unsafe frames, no invalidation.
- **Margins** — `pick_place_margin_analyzer_v1` on every Stage A evaluation.
  Telemetry only; never gates.
- **Stage B** — 8 recovery-anchor handoffs (nominal-scenario oracle prefix,
  recovery set `d52b460b4ab3a683`). Recorded always; never triggers
  retraining or retries.

**All three candidates run to completion regardless of earlier results**
(factorial attribution; no early stop). Offline MSE remains telemetry.

## Promotion rule and terminal statuses

Promotion order (minimal change first): `steps90k`, then `width512`, then
`width512_steps90k`. The promoted candidate is the first in that order whose
Stage A passes.

| Status | Meaning / next action |
| --- | --- |
| `promoted_<id>_robust` | Stage A and Stage B pass -> write the vision-rung proposal |
| `promoted_<id>_starts_only` | Stage A passes, Stage B fails -> new proposal targeting mid-trajectory recovery |
| `scaling_not_resolved` | no candidate passes Stage A -> stop; new proposal |

No retraining branch, no retry budget: every arm is defined up front.

## Pre-registered caveats

- The baseline cell was trained by the same trainer at the same recipe; its
  Stage A/B evidence lives in the broader tranche report and is referenced,
  not re-run.
- If both single-factor arms pass, the factorial still attributes: promotion
  takes `steps90k` (minimal change), and the width effect is documented from
  the other cells' telemetry.
- Anchor handoffs replay the nominal-scenario prefix only, as before.
- Trainings for the three arms run as independent single-threaded processes
  in parallel; each run digest is content-addressed and immutable, so
  parallelism cannot mix identities.

## Environment

Pinned mujoco 3.9.0 lane (`PYTHONNOUSERSITE=1`, conda hook + CLI version
guard). Inputs: capture `3f8eb5499eb18128`, nominal oracle
`9164a76699186c34`, recovery set `d52b460b4ab3a683`, the passing v3
preflight, baseline tranche `68c56c66d2dd3bb9`.

## Artifacts

Tranche report: `artifacts/so_arm101_v2/oracle_distillation/
scaling_gates/<digest16>/report.json` (immutable; embeds candidate training
report hashes, Stage A/B evidence, margin hashes, promotion trace). Stage A
evaluations under `scaling_stage_a_evaluations/`; margins under
`margin_analyses/`; handoffs under `policy_anchor_evaluations/`.
Implementation: `src/so_arm101_v2/simulation/scaling.py`, CLI
`run-scaling-gate`, covered by `tests/test_scaling_gate.py`.

## Addendum (2026-08-05 23:41 EDT): launch aborted before any artifact; re-run under numerics regime v2

The first launch of this tranche (2026-08-05 ~23:02, session b09b89d0,
`run-scaling-gate --workers 10`, parent PID 3094012 with three training
children 3094028/29/30) was killed at 23:41 after ~39 minutes, **before any
artifact existed**: no `scaling_gates/` directory, no new `models/` digests
(verified by scan). The atomic immutable writers guarantee nothing torn was
left behind; the trainings write only at completion, so there was nothing to
salvage and nothing to clean up.

Reason: the project adopted **numerics regime v2** (GPU + torch.compile
training by default — see `notes/numerics-regime-v2.md`). Spending the
remaining ~8-10 hours of single-threaded CPU on a tranche about to be
re-executed under v2 in roughly an hour was not compute-justified. The
aborted run would have produced *valid legacy-lineage* digests — this abort
is economic, not corrective.

Commitment: the identical pre-registration re-executes under regime v2 once
the v2 acceptance gate passes — same candidates, same promotion order
(`steps90k` → `width512` → `width512_steps90k`), same Stage A/B rules, same
90k-step budget, no retry budget. Only the execution regime (and therefore
the run and tranche digests, via the conditional top-level `numerics`
identity key) changes. Queued as task-spooler job `nrv2-scaling-rerun`.

The ~0.29 s/step single-thread cost model above remains the historical
record of why 90k (not 150k) was chosen; 90k is retained — revisiting the
budget would be a design change requiring a new proposal. The 256/30k
control remains the legacy-trained broader-tranche cell (referenced, never
re-run); promotion is absolute Stage-A pass/fail, so the cross-regime
control is context only.

## Result (2026-08-06): `scaling_not_resolved` under numerics regime v2

Executed as `scaling_gates/b672196e85a945aa` (regime `cuda_inductor_v2`, see
`notes/numerics-regime-v2.md`; ~35 min wall clock). No arm passed Stage A —
every failure is `safety_invalidation`, never a milestone shortfall — but the
optimization budget scales cleanly toward the bar:

| Arm | Stage A clean successes | Failure mode |
| --- | --- | --- |
| `steps90k` (256/90k) | 3/15 | safety invalidation, full milestone chains |
| `width512` (512/30k) | 3/15 | same |
| `width512_steps90k` (512/90k) | **9/15** | same |

A frame-level failure analysis of the best arm's six failing rollouts —
plots and violating-frame tables in `notes/scaling-failure-frame-analysis.md` —
localizes every violation to the **gripper**, with 27/33 findings past the
450-action teacher horizon (extrapolation spike at action ~451) and the rest
in the release/retreat stages.

Terminal status per pre-registration: stop; the next proposal owns the
remaining gap (the last-mile envelope frames the broader tranche identified,
now at 9/15 rather than 0/15). Training cost is no longer a constraint on
that proposal's design space.
