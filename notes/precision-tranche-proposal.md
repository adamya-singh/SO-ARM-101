# Precision Tranche (Proposal, pre-registered before any run)

## Question

The optimization-scaling tranche (`notes/optimization-scaling-proposal.md`,
`scaling_gates/b672196e85a945aa`, status `scaling_not_resolved`) left its best
arm — `width512_steps90k` — at 9/15 Stage A clean successes, with **every**
failure a `safety_invalidation` carrying the full milestone chain: the policy
completes the task behaviorally and grazes the safety envelope by small
margins. The failure-frame analysis (`notes/scaling-failure-frame-analysis.md`,
step 1 of this tranche) localizes the violations before any experiment runs.

Question: do (a) a learning-rate decay schedule, (b) a larger optimization
budget (steps and/or width), or (c) both, remove the residual envelope-grazing
frames — and does the answer replicate across seeds?

## Design (three stages, executed in order, docs updated between stages)

Immutable inputs (identical to the scaling tranche): five-scenario capture
`3f8eb5499eb18128` (2,250 rows), nominal oracle `9164a76699186c34`, recovery
set `d52b460b4ab3a683`, the passing deterministic v3 preflight, and the
scaling gate report `b672196e85a945aa` (its `width512_steps90k` cell is the
immutable control: 9/15, referenced, never re-run). Frozen recipe: identical
to the scaling tranche (`FROZEN_RECIPE`); stage cells override only the
registered factors below. Numerics regime v2 (`notes/numerics-regime-v2.md`).

**Stage "schedule"** — one cell:

| Cell | Width | Steps | LR schedule |
| --- | ---: | ---: | --- |
| control (referenced) | 512 | 90,000 | fixed 1e-3 |
| `cosine_w512_90k` | 512 | 90,000 | `cosine_floor_v1` (cosine 1e-3 → 1e-5 over max_steps) |

Selection rule (pre-stated, mechanical): the budget stage inherits
`cosine_floor_v1` **iff** the cosine cell strictly improves on the control —
more Stage-A clean successes, or equal successes with fewer total safety
frames (clip+limit+nonfinite+unsafe summed over 15 rollouts). Any tie keeps
`fixed` (smaller change).

**Stage "budget"** — three cells in registered minimal-change order, all under
the selected schedule:

| Cell | Width | Steps |
| --- | ---: | ---: |
| `steps150k` | 512 | 150,000 |
| `steps300k` | 512 | 300,000 |
| `width1024` | 1,024 | 90,000 |

Width 1024 requires widening the chunked-clone width allowlist
(128/256/512 → 128/256/512/1024); the closed tiny-lane allowlist is not
touched. All cells run to completion (factorial attribution; no early stop).

**Stage "seeds"** — seeds 202 and 303 of the best cell so far; the best
cell's own seed-101 record is the control. Best-cell rule (pre-stated):
most Stage-A clean successes, then fewest total safety frames, then earliest
in the registered change order (`control`, `cosine_w512_90k`, `steps150k`,
`steps300k`, `width1024`).

Compute justification (measured GPU regime-v2 costs, recorded before any
run): schedule ≈ 15 min (7-min training + Stage A + margins + Stage B);
budget ≈ 75 min (11.5 + 23 + ~10 min trainings + three eval blocks);
seeds ≈ 25 min. The ~0.29 s/step CPU constraint that capped the scaling
tranche at 90k is void under regime v2; evals, not training, now dominate.

## Gates (identical to the scaling tranche)

- **Stage A:** 15/15 deterministic closed-loop rollouts on the five-scenario
  v3 suite with zero clip/limit/nonfinite/unsafe frames (`STAGE_A_PASS_RULE`
  imported from the scaling module, byte-identical).
- **Margins:** the pick-place margin analysis recorded per cell (telemetry).
- **Stage B:** the eight anchor handoffs recorded per cell; never gates.

**All registered cells in a stage run to completion regardless of earlier
results.** Offline MSE remains telemetry.

## Promotion rule and terminal statuses

| Stage | Status | Meaning / next action |
| --- | --- | --- |
| schedule | `schedule_selected_cosine_floor_v1_stage_a_passed` | cosine cell 15/15 zero-safety; budget stage still runs, under cosine |
| schedule | `schedule_selected_cosine_floor_v1` | strict improvement without a full pass; budget stage under cosine |
| schedule | `schedule_selected_fixed` | no strict improvement; budget stage under fixed |
| budget | `budget_promoted_<cell_id>` | first cell in registered order passing Stage A 15/15 zero-safety; seeds stage runs on it |
| budget | `budget_not_resolved` | no cell passes; seeds stage runs on the best cell (dispersion evidence for the next proposal) |
| seeds | `seeds_robust` | seeds 202 and 303 both pass Stage A 15/15 zero-safety and the best cell's own record passed; tranche resolved |
| seeds | `seeds_not_resolved` | any seed fails (or the best cell never passed); stop; next proposal per the caveat below |

No retraining branch, no retry budget: every cell is defined up front. Stage
reports chain cryptographically: the budget identity embeds the schedule
report's content hash; seeds embeds budget's.

## Pre-registered caveats

- **Horizon-mismatch caveat (mandatory interpretation bound):** the
  failure-frame analysis shows the violating frames concentrate in actions
  451-480 — past the 450-action teacher horizon (the v3 contract runs 480
  actions; the progress feature clamps at `min(action, 449)/449`), in the
  chunk predicted entirely beyond the teacher's demonstration, plus
  release-phase delta-limiter frames (the privileged teacher itself needs
  ≥16 actions to open without tripping the limiter). LR decay and budget
  scaling are **not expected to fix that extrapolation regime**; this tranche
  measures whether they move the envelope-grazing margin. If failures remain
  confined to past-horizon/release frames across all cells, the terminal
  `next_action` is a **new proposal aligning the teacher horizon with the
  480-action contract** (extended teacher tail or explicit hold-tail
  targets) — a teacher/contract change explicitly out of scope here.
- The seed sweep tests replication of the *best cell*, not of every cell;
  single-seed conclusions elsewhere in this tranche inherit that limitation.
- The control cell is a cross-stage constant: 9/15 with 33 total safety
  frames, immutable in the scaling gate report.
- `policy_id` becomes seed-aware (derived from the checkpoint's stored
  config seed) so seed cells are distinguishable in reports and telemetry
  stems; every existing artifact's id string is unchanged (all stored seeds
  are 101).

## Environment

Pinned mujoco 3.9.0 evaluation lane (`PYTHONNOUSERSITE=1`, conda hook + CLI
version guard). Training under numerics regime v2 (`PINNED_NUMERICS_V2`:
RTX 3090, torch 2.7.1+cu126, CUDA 12.6, driver 610.47, inductor "default",
TF32 off, deterministic algorithms, `cpu_generator_v1` noise stream);
`--legacy-numerics` reproduces the legacy lane byte-for-byte. Inputs by
digest: capture `3f8eb5499eb18128`, oracle `9164a76699186c34`, recovery
`d52b460b4ab3a683`, scaling gate `b672196e85a945aa`, the passing v3
preflight.

## Artifacts

- Stage reports: `artifacts/so_arm101_v2/oracle_distillation/precision_gates/<digest16>/report.json`
- Stage A evaluations: `.../precision_stage_a_evaluations/<digest16>/` (suite ids `fixed_pick_place_v3.precision_<stage>_<cell>`)
- Trainings: content-addressed under `.../models/chunked_h90/<digest16>/` (regime-v2 identities)
- Failure-frame analysis: `notes/scaling-failure-frame-analysis.md` + `notes/images/scaling-failure-*.png` + JSON mirror
- Implementation: `src/so_arm101_v2/simulation/precision.py`, `tools/analyze_safety_frames.py`, CLI `so-arm101-v2-sim run-precision-stage --stage {schedule,budget,seeds}`, tests `tests/test_precision_gate.py`, queue `simulation_code/queue_precision_tranche_20260806.sh` (tsp labels `precision-{analysis,schedule,budget,seeds}`)

## Result (stage schedule, 2026-08-06): `schedule_selected_cosine_floor_v1`

Report: `precision_gates/f0a6169e8c0f05e7` (regime `cuda_inductor_v2`, ~13 min).
The cosine cell strictly improves the control on both pre-stated metrics:

| Cell | Stage A clean successes | Total safety frames |
| --- | ---: | ---: |
| control (fixed LR) | 9/15 | 210 |
| `cosine_w512_90k` | **12/15** | **12** |

Per the selection rule the budget stage inherits `cosine_floor_v1`. Frame
analysis of the remaining failures (`notes/precision-schedule-failure-analysis.md`):
still gripper-only, but the **past-horizon extrapolation spike is gone**
(0/4 findings past action 450, vs 27/33 under fixed LR) — the low terminal
learning rate visibly tames the chunk-6 extrapolation. The residual four
violating frames sit in nominal's grasp/release phases; notably the failing
scenario set *moved* (nominal now fails 0/3 clean while all four ±1.5 mm
shifts pass), consistent with grazing noise around a much smaller violation
mass rather than a structural failure. Offline MSE 1.13e-7 (telemetry).

## Result (stage budget, 2026-08-06): `budget_not_resolved`

Report: `precision_gates/0cb6ded4f78a0a68` (~55 min, all three cells under
`cosine_floor_v1`). **The budget curve bent** — no cell matches the schedule
stage's 90k-cosine cell:

| Cell | Stage A clean successes | Total safety frames |
| --- | ---: | ---: |
| `cosine_w512_90k` (schedule stage) | **12/15** | **12** |
| `steps150k` | 9/15 | 33 |
| `steps300k` | 9/15 | 15 |
| `width1024` | 3/15 | 66 |

Interpretation (telemetry): `cosine_floor_v1` is budget-relative, so larger
step budgets hold the learning rate high for longer in absolute terms — the
150k/300k cells spend most of training at a hotter LR than the 90k cell ever
does, and width 1024 regresses sharply (matching the width-512/30k pattern
from the scaling tranche: more capacity without proportional optimization
polish). The optimization sweet spot on this data is the 90k cosine cell.

**Recorded defect, caught before the seeds stage ran:** the budget report's
derived `best_cell` field reads `steps300k` because the implementation
ranked only {control + budget cells}, omitting the schedule stage's cell —
contradicting the pre-registered `BEST_CELL_RULE`, whose registered cell
order explicitly includes `cosine_w512_90k`. The rule string (embedded in
every stage identity) is authoritative: the code was corrected the same day
(selection now spans control + schedule + budget cells; pinned by
`test_best_cell_selection_spans_all_registered_stages`), and the seeds stage
recomputes the selection from the hash-chained stage reports rather than
trusting the stored field. Applying the rule correctly selects
**`cosine_w512_90k`** (12/15, 12 frames) as the best cell. The budget
report itself is immutable and stands as written, with this addendum as the
correction record.

## Result (stage seeds, 2026-08-06): `seeds_not_resolved` — tranche terminal

Report: `precision_gates/6d2c95a8f577c95b` (~25 min; best cell
`cosine_w512_90k`, recomputed per BEST_CELL_RULE over the full registered
set). Seed dispersion is decisive:

| Seed | Stage A clean successes | Total safety frames | Failing scenarios |
| --- | ---: | ---: | --- |
| 101 (best-cell record) | 12/15 | 12 | nominal |
| 202 | 12/15 | 6 | cube_x_plus |
| 303 | **3/15** | 63 | four of five scenarios |

**Tranche conclusion:** cosine LR decay is a real, large improvement (it
removed the past-horizon extrapolation spike and cut violation mass ~17x),
but the remaining envelope-grazing is a **seed lottery around a structural
boundary** — which scenario fails changes per seed, budget scaling bends the
wrong way, and no cell reaches 15/15 zero-safety. Per the pre-registered
horizon-mismatch caveat, the terminal `next_action` is
`stop_and_write_teacher_horizon_alignment_proposal`: align the 450-action
teacher with the 480-action contract (extended teacher tail or explicit
hold-tail targets) so the final chunk is trained rather than extrapolated,
and adopt `cosine_floor_v1` as the established schedule going in.
