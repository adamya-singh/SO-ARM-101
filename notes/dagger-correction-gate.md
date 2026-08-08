# DAgger Correction Decision Gate

## Decision

The gate completed with terminal status **`blocked_offline`**. With 4,818
policy-induced oracle correction rows added to the 450 nominal rows, the
unchanged width-256 `phase_state` trainer stopped at a nominal-only loss floor
of `1.1059e-4` normalized MSE and `0.1921` maximum ACT error against the
unchanged `1e-6` / `0.01` gates after the full 30,000-step schedule. Per the
gate's pre-registered `blocked_offline_policy`, this is a terminal stop with
the loss floor documented: no closed-loop evaluation was authorized, and no
gate relaxation, capacity, or optimizer compensation is permitted inside this
tranche.

Immutable decision:
`artifacts/so_arm101_v2/oracle_distillation/correction_gates/
faf13e95b8caa2e8/report.json`.

## Capture design (amended before any immutable artifact was written)

The originally planned capture required timeline-preserving corrections: a
correction starting at induced step `k` had to finish inside the remaining
episode budget by compressing the privileged controller's stage schedule.
That design failed at all 11 sites, and the diagnostics localized why:

- Even the nominal oracle acquires its strict grasp only at action 364 and
  pickup success at 389 against the immutable 450-action pickup deadline
  (source: the nominal capture's own event log). The jaw squeeze consolidates
  over ~160 actions of physics that do not compress.
- Compressed schedules either tripped the gripper delta limiter (the
  documented >=16-action release floor) or ran the grasp chain past the
  deadline. Only `k=31` ever succeeded, and only with a hand-retuned
  480-horizon schedule; every `k >= 70` variant failed.

The amended design, adopted after explicit user approval: run the replan at
**nominal speed** and validate it as a **fresh sub-episode** under the
unchanged `fixed_cube_pick_place_v3` contract (its own 450-action pickup
deadline and 480-action budget starting at the correction's first action).
Correction rows keep the deployment clock: `progress = min(index, 449)/449`,
which saturates at 1.0 exactly as the deployed policy computes it. The
timeline-compression infeasibility evidence is retained in the session
diagnostics and summarized here; `compress_boundaries` /
`reset_from_state` remain in `simulation/privileged.py` as the tested,
rejected mechanism.

## Dataset

All 11 predefined sites were accepted (the 8 established recovery-anchor
phase indices plus divergence onsets 31, 74, 194), each captured twice with
bitwise-identical rows, each completing the full pick-and-place sub-episode
in 438 actions with zero clip/limit/nonfinite/unsafe frames and labels within
1e-6 ACT through the unchanged safety path. Clone prefixes were replayed
exactly and were safety-clean; planner state purity was verified on raw
`qpos`/`qvel` (derived-state refresh is recorded as telemetry).

Manifest: `artifacts/so_arm101_v2/oracle_distillation/corrections/
7feed472071fd725/manifest.json` (4,818 rows, content-addressed, provenance
pinned to the inducing checkpoint `d5f96d397bd9b915` and the nominal oracle
capture `9164a76699186c34`).

## Controlled training result

Fixed, identical to the established control: `phase_state` 10 inputs, width
256, seed 101, full-batch Adam with the `decay_10k_20k` schedule, 30,000-step
ceiling, delta target, nominal-only normalization and gating, equal per-row
loss weight 1.0. Only the training rows changed (450 -> 5,268).

| Measurement | Nominal-only control (`d5f96d397bd9b915`) | With corrections (`9e1bb18f724aa989`) | Gate |
| --- | ---: | ---: | ---: |
| Nominal normalized MSE | `9.999720305e-7` (passed at step 28,718) | `1.1058597738e-4` (floor `1.1004e-4` at step 29,900) | `<=1e-6` |
| Nominal maximum ACT error | `0.007196128` | `0.1921165586` | `<=0.01` |
| Training-command safety violations | 0 | 0 | 0 |
| Correction-block MSE / max error | n/a | `5.109e-5` / `0.1184` | non-gating |

The worst nominal error is **row 31, wrist_flex — the first correction site**.
The clone's induced state at action 31 differs from nominal by only ~0.005
ACT, so the drift-onset correction episodes place near-identical feature
vectors next to nominal rows with different labels. Competing supervision at
nearly-aliased features is the natural reading of the localized floor, but it
is recorded here as a hypothesis; the residual analysis machinery, not this
note, is the instrument for establishing it.

## Interpretation and boundaries

- This result does **not** show that policy-induced corrections cannot fix
  feedback control: no correction-augmented policy ever reached a closed-loop
  evaluation. It shows that the fixed 68k-parameter model cannot satisfy the
  near-exact nominal memorization gate while also absorbing 10.7x additional
  correction rows under the frozen optimization budget.
- The capture method itself is proven: the privileged controller can generate
  contract-certified, safety-clean correction demonstrations from every
  policy-induced state family the clone visits, including post-contact ones.
- The pre-registered stop honors the plan's discipline: near-exact nominal
  fit was retained as the promotion prerequisite, and the experiment reports
  a loss floor instead of quietly relaxing thresholds.

## Status after this tranche

Sparse-anchor distillation, richer-observability schemas, and now
equal-weight full-correction augmentation at fixed capacity have all been
rejected on immutable evidence. The follow-up decision is recorded in
`notes/chunked-promotion-proposal.md`: the tiny-model lane is closed, one
pre-registered non-promoting diagnostic probe of this tranche's blocked
checkpoint answers whether correction training moved closed-loop behavior at
all, and the project promotes to a Phase-8 chunked-action rung. Vision,
physical deployment, and RL remain unauthorized.


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
