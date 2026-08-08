# Saturation-Attribution Gate (Proposal, pre-registered before any run)

## Problem

Both promotion-era experiments moved task behavior decisively (correction
probe: 18.5 mm lift; chunked H=90: first learned strict grasp, 60.7 mm lift)
and both failed the same way: saturated commands. H=90 telemetry localizes
it — gripper requests below its effective floor from action 281, delta-limiter
trips from 361 at the start of the final blind chunk, grasp lost at 363, then
runaway out-of-envelope extrapolation (wrist_flex −5.2 rad against a −3.14
bound). Every training command is deeply feasible (≥0.004 ACT bound margin,
steps 6.3x under the delta cap), so saturation is pure off-distribution
extrapolation error that the plain MSE objective never sees.

## Candidates (user-approved factorial; all share H=90, width 256, seed 101, Adam 1e-3, 30k full-batch steps)

| ID | Data | Saturation mechanism |
| --- | --- | --- |
| `feasible_decoder_only` | nominal 450 | chained-bounded decoder |
| `corrections_and_feasible_decoder` | nominal + 4,818 correction rows | chained-bounded decoder |
| `noise_penalty_only` | nominal 450 | noise-augmented hinge penalty |
| `corrections_and_noise_penalty` | nominal + corrections | noise-augmented hinge penalty |
| `corrections_only` | nominal + corrections | none |

The unmitigated nominal-only H=90 baseline exists at
`chunked_gates/591f0686e94d27fd` and is referenced, not re-run.

Mechanisms, fixed parameters:

- **Chained-bounded decoder** (`feasible_chain_v1`): each of the 90 commands
  is a tanh-bounded step (≤ 0.9 of the per-step delta cap) from the previous
  command, clamped into the effective safe ACT box (intersection of the ACT
  dataset box and the mujoco-envelope pullback, `effective_safe_act_bounds`)
  shrunk by a 0.002 ACT margin. Near-identity on-distribution (targets are
  6.3x under the delta cap and 0.004 inside the box); incapable of emitting a
  bound-violating request off-distribution. Shared verbatim between training
  loss and deployment. Zero-init raw output holds the current pose.
- **Noise-augmented hinge penalty** (`noise_penalty_v1`): decoder unchanged;
  each training step adds Gaussian noise (σ = 0.05) to the 10-dim features
  and penalizes (weight 1.0) implied commands outside the margin-shrunk box
  or with steps beyond 0.9 of the delta cap. Softer, no hard guarantee;
  included at the user's request for mechanism attribution.

Correction data enters exactly as in the reactive lane: nominal-only
normalization statistics, stored deployment-clock progress, targets built
per-episode (padding repeats each episode's own final command; the lookahead
never crosses an episode boundary).

## Promotion rule (pre-registered)

All five candidates train and evaluate — no early stop; attribution is the
point. Pass rule unchanged: deterministic nominal MuJoCo 3/3 with zero
clip/limit/nonfinite/unsafe frames. Among passers, promote the earliest in
the candidate order above: decoder family first (hard feasibility guarantee),
then penalty family, then unmitigated corrections; within a family,
single-ingredient before combined. Offline metrics are telemetry only. Status
is `promoted_{candidate_id}` or `closed_loop_not_resolved`.

On promotion, the only authorized follow-up is broader evaluation of the
promoted policy (five-scenario suite, then anchor handoffs). On
`closed_loop_not_resolved`: stop and write a new proposal (re-planning
cadence / smaller final chunk are the named directions); no tuning inside
this tranche.

## Stopping / invalidation

- Hard invalidation (run void): nonfinite loss or predictions; raw-head
  zero-init violation; correction provenance mismatch.
- Telemetry-only, never gates: decoder `clamp_active_fraction > 0` or
  `tanh_activation_max_abs > 0.999` at convergence; penalty floor values;
  offline MSE regressions.
- Known residual risk, recorded not fixed: the runtime delta limiter compares
  against the *measured* pose, so servo lag can still trip it under the
  feasible decoder (eta = 0.9 is headroom, not proof). Decoder candidates
  failing *only* via limiter frames would itself be this tranche's
  diagnostic finding.
- Single pre-registered values for eta / margin / sigma / weight; no new
  horizons; vision, RL, physical deployment remain unauthorized.

## Result (2026-08-03): `promoted_noise_penalty_only` — FIRST PROMOTED POLICY

The `noise_penalty_only` candidate — H=90 chunked clone, nominal 450 rows,
unchanged decoder, trained with the noise-augmented feasibility hinge —
**passed the promotion gate**: three deterministic nominal rollouts, all
completing the full pick-and-place (strict grasp, every lift milestone,
placement, settle, retreat; 32.4 mm peak lift) with **zero**
clip/limit/nonfinite/unsafe frames. It is the first learned policy in this
repository to pass closed-loop promotion. Its converged penalty term is
0.0 — the network learned to keep implied commands feasible across a noise
neighborhood of every training state, which evidently covered the states the
policy actually visits.

Full attribution (all five ran to completion):

| Candidate | Result | Failure signature |
| --- | --- | --- |
| `feasible_decoder_only` | failed | 0 clipped (guarantee held) but 71 limiter + 5 unsafe frames; decode chain 4.2% clamped at convergence, fit degraded (max err 0.087) |
| `corrections_and_feasible_decoder` | failed | 262 limiter + 13 unsafe frames; chain saturated (tanh max 1.0), max err 1.08 |
| **`noise_penalty_only`** | **PASSED 3/3, zero safety** | — |
| `corrections_and_noise_penalty` | failed | safe (zero safety frames) but task-inert: 0.3 mm lift, `pickup_incomplete` |
| `corrections_only` | failed | 184 clipped + 136 limiter frames, 0.2 mm lift |

Established findings:

1. **Noise-augmented feasibility regularization alone resolves command
   saturation** at H=90 — and it was also the cheapest candidate to train
   (~10 minutes). The hard-decoder alternative eliminated bound-clipping
   exactly as designed but exposed the pre-registered residual risk: the
   runtime delta limiter compares against the *measured* (servo-lagged)
   pose, and the chain's partial saturation additionally degraded offline
   fit. The soft mechanism beat the hard guarantee.
2. **Correction data actively hurts chunked training.** All three
   correction-bearing candidates regressed (offline max ACT errors
   0.86-1.08 versus 0.046 for the winner): correction episodes place
   near-nominal states next to entirely different 90-step futures, a far
   more severe label conflict at chunk scale than the row-level conflict
   documented in the reactive lane. The correction dataset remains valid
   evidence and a valid dataset; it is the wrong supplement for this
   architecture.
3. The promoted checkpoint is `models/chunked_h90/2b6195d619ab531b`
   (checkpoint sha `f6c59b8e25762202...`); its passing evaluation is
   `saturation_gate_evaluations/2f18514f298cda22`; gate decision:
   `saturation_gates/1a78ec8affead704/report.json`.

Per the pre-registration, the only authorized follow-up is broader
evaluation of the promoted policy: the five-scenario v3 suite, then
recovery-anchor handoffs. Vision, RL, and physical deployment remain gated
behind those.

## Artifacts

Gate: `artifacts/so_arm101_v2/oracle_distillation/saturation_gates/<digest16>/report.json`
(immutable, with a per-candidate `attribution` block). Models under
`models/chunked_h90/<digest16>/`; evaluations under
`saturation_gate_evaluations/`. Inputs: nominal oracle capture
`9164a76699186c34`, correction dataset `7feed472071fd725`, the passing v3
preflight. Implementation is covered by `tests/test_saturation_gate.py`
(11 tests; suite total 149).


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

## Addendum (2026-08-06): re-validated under numerics regime v2

This gate served as the pre-registered acceptance gate for the numerics-
regime-v2 flip (GPU + torch.compile training by default; see
`notes/numerics-regime-v2.md`). Re-executed under v2 as
`saturation_gates/32f8972f7996b5f6`: status `promoted_noise_penalty_only`
with per-candidate attribution identical to `46de62c4f6d1b78f`. The legacy
gate remains the authoritative promotion record; the v2 gate is the
active-regime confirmation. The pre-registered text above is unchanged.
