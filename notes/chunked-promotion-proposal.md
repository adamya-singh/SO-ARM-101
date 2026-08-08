# Tiny-Lane Closure and Chunked-Policy Promotion (Proposal)

## Decision being made

The `phase_state` tiny-model lane is closed as a diagnostic instrument. Its
purpose — making failures legible cheaply — is fulfilled: it exposed the
coordinate-contract bug, the impossible legacy simulator, the retreat
pathology, and a replay-proven diagnosis of the terminal failure mode
(per-step feedback compounding under covariate shift). Four controlled
interventions on it (sparse anchors, loss-weight ablation, richer
observability, full correction augmentation) are rejected on immutable
evidence, with diminishing information per tranche.

The diagnosed failure mode has a known architectural treatment — action
chunking, which reduces the number of feedback decisions per episode — and
the rebuild plan's Phase 8 anticipated exactly this promotion. All
infrastructure (proven simulator, contracts, oracle and correction capture,
closed-loop evaluation gates) is model-agnostic and carries forward.

**Retired with this proposal:** the near-exact offline memorization gate
(`1e-6` normalized MSE). It was a pipeline-validation tool; it is proven
non-predictive of closed-loop success (the nominal-only clone passed it and
failed 0/3) and unreachable under useful data volumes (the correction gate's
documented floor). Offline metrics remain recorded as telemetry; the first
physically meaningful promotion gate is deterministic nominal MuJoCo 3/3 with
zero safety frames, as established by the recovery-weight ablation.

## Experiment 1: diagnostic closed-loop probe of the correction checkpoint

One pre-registered, explicitly **non-promoting** evaluation of the
offline-blocked correction-augmented checkpoint
(`models/phase_state/9e1bb18f724aa989`), to close the tiny lane with an
answer to the only untested question: does correction training move
closed-loop behavior at all?

- Setup: three deterministic nominal `fixed_pick_place_v3` rollouts, the same
  suite restriction as the correction gate. The checkpoint is loaded through
  a dedicated diagnostic wrapper that validates hash integrity but bypasses
  the promotion-eligibility gate; its artifacts carry a non-promoting claim.
- Baseline reference: the inducing clone's immutable evaluation
  (`clone_evaluations/943cf536710e3d84`): milestone frontier
  `{reach, first_contact, released, retreated}`, maximum cube height gain
  `0.000159 m`, zero safety frames, `pickup_incomplete`.
- Pre-registered decision rule:
  - `behavior_moved` iff any probe rollout achieves a milestone outside the
    baseline frontier, or the probe's maximum cube height gain exceeds the
    baseline's by at least `0.005 m`.
  - otherwise `behavior_unchanged`.
  - `safety_regressed` is flagged independently iff any
    clip/limit/nonfinite/unsafe frame occurs (baseline has zero).
- Use of the answer (pre-registered): `behavior_moved` licenses including the
  correction dataset in a *future* chunked-policy data rung;
  `behavior_unchanged` defers correction data pending a redesign that
  addresses the documented near-nominal label competition. Neither outcome
  promotes this checkpoint or reopens tiny-model training.

## Experiment 2: chunked-action promotion rung (Phase 8)

Hypothesis: executing H-action chunks between observations reduces feedback
compounding enough for a learned policy to pass nominal closed-loop control.
The fixed-action replay (H = 450, zero feedback) succeeds; the reactive clone
(H = 1) fails; this rung searches the interval from the reactive end.

Fixed design, one hypothesis (the chunk horizon):

- Data: the unchanged 450-row nominal oracle capture
  (`oracle/fixed_pick_place_v3/9164a76699186c34`) only. Correction data is
  deliberately excluded from this rung (single hypothesis; its use is decided
  by Experiment 1's pre-registered rule).
- Features: the same 10 `phase_state` inputs with unchanged nominal-only
  normalization and the deployment progress clock.
- Targets: for row `t`, the next `H` absolute executed ACT commands
  (normalized), as a residual on the current pose so the zero-initialized
  head predicts "hold" — rows past 449 pad by repeating the final command.
- Model: the same two-hidden-layer width-256 MLP with a zero-initialized
  head, output dimension `6H`. Optimizer: full-batch Adam, fixed `1e-3`,
  30,000 steps, seed 101.
- Deployment: the policy queries privileged state once per chunk, then
  executes its `H` commands open-loop through the unchanged safety path;
  progress uses `min(index, 449)/449`.
- Candidate order, ascending (most reactive first), stop at first pass:
  `H = 10`, `H = 30`, `H = 90`.
- Offline telemetry is recorded (loss trace, per-offset errors, zero-init and
  finiteness checks) but does not gate. Promotion gate: deterministic nominal
  MuJoCo 3/3 with zero clip/limit/nonfinite/unsafe frames.
- Statuses: `passed_h{H}` (earliest passing horizon) or
  `closed_loop_not_resolved` (all three trained and evaluated, none passes).
  Nonfinite training or violated invariants invalidate the run rather than
  producing a status.

## Result: Experiment 1 (2026-08-03)

Status **`behavior_moved`**, with **`safety_regressed`** flagged. All three
deterministic probe rollouts of the blocked correction checkpoint achieved
`cube_supported`, `lift_5mm`, and `lift_10mm` — none in the baseline
frontier — with a maximum cube height gain of `0.01847 m` versus the
baseline's `0.000159 m` (116x, far above the 0.005 m margin). This is the
deepest closed-loop task progress any learned policy has shown in this
repository. The cost: 326 clipped, 43 limited, and 27 unsafe-contact frames
per rollout versus the baseline's zero, reproducing the command-saturation
signature of every correction/recovery-augmented tiny policy.

Immutable report: `artifacts/so_arm101_v2/oracle_distillation/
correction_probes/63e2717b530a2cf0/report.json`.

Pre-registered consequence: correction data is **authorized** for a future
chunked-policy data rung (Experiment 2 remains nominal-only as registered);
any such rung must treat the demonstrated safety regression as a first-class
gate, not a footnote. The probe promotes nothing.

## Result: Experiment 2 (2026-08-03)

Status **`closed_loop_not_resolved`**: all three horizons trained, all three
evaluated, none passed the zero-safety nominal 3/3 gate. Every failure is
`safety_invalidation` — command saturation, not task indifference:

| H | Offline MSE (telemetry) | Max height gain | Deepest milestone | Safety per rollout |
| ---: | ---: | ---: | --- | --- |
| 10 | `4.02e-7` | `5.4 mm` | `lift_5mm` | 165 clipped / 39 limited / 6 unsafe |
| 30 | `2.13e-6` | `1.6 mm` | `bilateral_interior_contact` | 128 clipped / 71 limited / 0 unsafe |
| 90 | `2.33e-6` | **`60.7 mm`** | **`strict_grasp_acquired` + `lift_20mm`** | 150 clipped / 82 limited / 20 unsafe |

All rollouts deterministic across three repeats. The H=90 policy is the
first learned policy in this repository to acquire a certified strict
bilateral grasp and lift the cube through every lift milestone — nominal-only
training data, no corrections — before failing on safety saturation during
the carry. Immutable decision: `artifacts/so_arm101_v2/oracle_distillation/
chunked_gates/591f0686e94d27fd/report.json`.

Combined reading of Experiments 1 and 2: both levers move task behavior
(corrections: 18.5 mm with a reactive policy; chunking: 60.7 mm with a strict
grasp), and both fail the same way — saturated, safety-invalidating commands.
The binding constraint for the next proposal is command saturation. Per the
pre-registered stopping rule, addressing it is a new proposal, not tuning
inside this rung.

**Recommended next proposal (direction only; not yet registered):**
correction-augmented H=90 chunked training with a saturation-aware objective.
Rationale: the H=90 architecture grasps; the correction data is authorized by
Experiment 1's rule; and every training command is in-envelope, so saturation
is pure extrapolation error the current loss never sees — penalizing or
clip-weighting out-of-envelope predicted commands attacks the observed
failure directly. Promotion gate unchanged (nominal MuJoCo 3/3, zero safety
frames; broader suite and anchor handoffs for a passing policy only).
Fallback levers, in order: temporal ensembling of overlapping chunks, then
capacity. That proposal must be written with its own pre-registered
candidates, gates, and stopping rules before anything runs.

## Stopping and invalidation

- Experiment 1 is a single evaluation; it cannot iterate.
- Experiment 2 trains at most three candidates and stops at the first pass;
  no horizon outside the pre-registered ladder, no capacity, optimizer, or
  data changes inside this rung.
- On `passed_h{H}`: the authorized follow-up is broader evaluation of the
  passing policy only (five-scenario suite, then anchor handoffs).
- On `closed_loop_not_resolved`: stop and write a new proposal (candidate
  directions: correction-augmented chunks per Experiment 1's rule, wider
  chunk model, or ACT-style temporal ensembling), rather than tuning inside
  this rung.
- Vision, physical deployment, and RL remain unauthorized.


## Addendum (2026-08-04): mujoco environment standardization

Some artifacts cited above were produced between 2026-08-02 19:03 and
2026-08-03 13:41, when a `pip --user` mujoco 3.11.0 silently shadowed the
env's pinned 3.9.0 and forked simulation numerics. The project standardized
on mujoco 3.9.0 (`PYTHONNOUSERSITE=1`, enforced by a conda activation hook
and a version guard in `so-arm101-v2-sim`); the affected artifacts were moved
unmodified to `artifacts/so_arm101_v2/quarantine_mujoco_3_11_0/` (same
relative paths, every report still self-verifies). Where a decision needed to
stay active it was re-established under 3.9.0: corrections re-captured as
`corrections/53ad45590cfb60c0` (4,818 rows, all 11 sites accepted), recovery
re-captured as `recovery/d52b460b4ab3a683`, and the saturation gate re-run as
`saturation_gates/46de62c4f6d1b78f` — identical attribution
(`promoted_noise_penalty_only`, all evaluations deterministic) and the
byte-identical promoted checkpoint. This addendum records the relocation and
re-establishment; the pre-registered text above is unchanged. Details:
`notes/parallel-execution-infrastructure.md`.
