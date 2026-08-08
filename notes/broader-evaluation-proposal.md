# Broader-Evaluation Tranche (Proposal, pre-registered before any run)

## Question

The promoted policy (H=90 chunked clone, noise-penalty trained, checkpoint
`models/chunked_h90/2b6195d619ab531b`, promoted by
`saturation_gates/1a78ec8affead704`) passed nominal MuJoCo 3/3 with zero
safety frames — but it trained on one episode of one scenario and observes
the world about five times per episode. Is it a policy or a recording?

## Structure (single tranche, automatic single-retry branch, user-approved)

- **Stage A** — full `fixed_pick_place_v3` suite: 5 scenarios (nominal plus
  cube x/y ±1.5 mm) × 3 repeats = 15 rollouts, 480-action budget. Pass:
  deterministic, 15/15 success without invalidation, zero
  clip/limit/nonfinite/unsafe frames.
- **Margins** — `pick_place_margin_analyzer_v1` runs on every Stage A
  evaluation (pass or fail): per-joint envelope headroom, per-step
  delta-cap usage, strict-grasp streak slack versus the 30-frame hold,
  napkin footprint and support margins. Telemetry only; never gates.
- **Stage B** — 8 recovery-anchor handoffs
  (`policy_anchor_handoff_evaluation_v1`, nominal-scenario oracle prefix,
  chunked student from the anchor). Pass: 8/8 success, zero student safety
  counts. Recorded always; **never triggers retraining** — the contingent
  intervention targets start-state generalization, and spending the single
  retry on an un-hypothesized mid-trajectory fix would muddy attribution.
- **Contingent branch (at most once)** — trigger = memorization signature in
  Stage A: every nominal rollout passes with zero safety counts, at least
  one shifted-start rollout fails, and no nonfinite frame appears anywhere.
  Then: capture the five-scenario oracle dataset
  (`capture_oracle_demonstrations(scenario="all")`; must yield exactly 2,250
  rows across 5 successful episodes or the tranche aborts), retrain the
  exact frozen winning recipe (H=90, `noise_penalty_v1`, eta 0.9, margin
  0.002, sigma 0.05, weight 1.0, seed 101, width 256, lr 1e-3, 30k steps;
  chunk lookahead bounded per episode), and re-run Stage A + margins +
  Stage B for the retrained policy. No second retry.

## Terminal statuses → next action

| Status | Next action |
| --- | --- |
| `promoted_policy_robust` / `retrained_policy_robust` | write the vision-rung proposal |
| `*_passes_starts_fails_handoffs` | stop; new proposal targeting mid-trajectory recovery |
| `starts_not_resolved` | stop; new proposal |

`starts_not_resolved` covers: Stage A failure without the signature,
capture or retrain failure, retrained Stage A failure, or nonfinite frames
anywhere (which also forbids the branch).

## Pre-registered caveats

- Anchor handoffs replay the nominal-scenario prefix only; Stage B says
  nothing about shifted-start mid-trajectory states.
- A retrained policy is a different policy (new manifest lineage,
  capture-wide normalization, new digest); statuses name it `retrained_*`
  and the report records both checkpoints.
- The trainer's episode-boundary fix (chunk lookahead never crosses an
  episode boundary) is proven byte-identical for all existing
  single-episode artifacts by `tests/test_broader_evaluation.py`.

## Artifacts

Tranche report: `artifacts/so_arm101_v2/oracle_distillation/
broader_evaluations/<digest16>/report.json` (immutable; embeds stage
evidence, margin and anchor report hashes, branch trace). Stage A
evaluations under `broader_stage_a_evaluations/`; margins under
`margin_analyses/`; handoffs under `policy_anchor_evaluations/`. Inputs:
checkpoint `2b6195d619ab531b`, gate `1a78ec8affead704`, nominal oracle
`9164a76699186c34`, recovery set `0570ec8c0d37002f`, the passing v3
preflight. Implementation covered by `tests/test_broader_evaluation.py`
(suite total 157).


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

When this tranche is (re)launched, use gate `46de62c4f6d1b78f` and recovery
set `d52b460b4ab3a683` as inputs, under the pinned environment; the partial
outputs of the first (3.11-regime) launch are quarantined.

## Result (2026-08-04): `starts_not_resolved` — memorizer confirmed, retrain trades memorization for near-complete generalization

Tranche report: `broader_evaluations/68c56c66d2dd3bb9` (pinned 3.9.0,
`--workers 10`).

**Stage A, promoted policy: the memorization signature fired exactly as
pre-registered.** Nominal 3/3 perfect (zero safety frames); all four ±1.5 mm
shifted starts fail by `safety_invalidation` with heavy saturation (43-734
safety frames). The margins instrument shows why nominal was always fragile:
minimum envelope headroom `0.0007` ACT and peak delta usage 74% of the cap —
the pass lived on razor margins. Stage B: handoffs fail (recorded, per
protocol non-triggering).

**Contingent branch (single retry, consumed):** the five-scenario capture
succeeded (2,250 rows, oracle solves all five starts), and the frozen recipe
retrained on it. The retrained policy **fails Stage A everywhere — including
nominal — but in a completely different way**: it achieves strict grasp,
every lift milestone (42-45 mm), carry, and release in *all five* scenarios,
completes 7/8 anchor handoffs (sole failure: `lift`, by one clipped frame),
and its worst scenario has 18 safety frames versus the memorizer's 734. It
fails only at the last mile: the 30-frame strict-hold window and a handful of
envelope frames (headroom −0.002 ACT). Offline telemetry confirms underfit,
not concept failure: `1.94e-5` MSE at the fixed 30k steps versus `2.37e-6`
for the single-scenario winner — five times the data, the same optimization
budget, eight times the residual.

**Established findings:**
1. The promoted policy was a trajectory memorizer; single-scenario promotion
   evidence does not transfer even 1.5 mm.
2. Multi-scenario oracle data converts the same recipe into a broad
   generalizer that nearly completes everywhere and recovers from 7/8
   mid-task handoffs — at the cost of per-trajectory polish under the frozen
   optimization budget.
3. The remaining gap is quantitatively an optimization/capacity shortfall
   (grasp-hold slack and single-digit safety frames), not a data or
   architecture gap.

Terminal status `starts_not_resolved` per pre-registration; the retry budget
is spent. The next proposal should scale the optimization budget (steps
and/or the now-arguably-earned width increase) to the 2,250-row dataset with
the same recipe and gates — and nothing else. Vision, RL, and physical
deployment remain unauthorized.
