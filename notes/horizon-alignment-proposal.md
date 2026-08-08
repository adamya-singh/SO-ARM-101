# Horizon-Alignment Tranche (Proposal, pre-registered before any run)

## Question

The precision tranche (`notes/precision-tranche-proposal.md`, terminal status
`seeds_not_resolved`, next action
`stop_and_write_teacher_horizon_alignment_proposal`) established that the
residual envelope-grazing failures are structural: the v3 contract runs 480
actions but the privileged teacher demonstrates only 450, so the student's
final chunk is extrapolated, never trained — and the frame analyses localize
essentially all violations to that regime. Optimization levers are exhausted
(cosine LR removes the extrapolation spike but seeds disperse 12/15, 12/15,
3/15).

Question: with the teacher extended to **hold still after full retreat**
through action 480 and the students retrained on a 480-action capture under
the established recipe, does the aligned recipe reach 15/15 zero-safety
Stage A — and does it replicate across three seeds?

**Prediction on record:** 15/15 with zero safety frames on all three seeds.

## Design

**Teacher hold-tail** (`simulation/privileged.py`): the hold tail past
action 450 is the stage-loop **fall-through** in `predict()`, which returns
the clamped retreat pose bit-exactly on every call; the recorded stage
schedule is untouched. **Verifiable invariant (pinned by
`test_teacher_hold_tail_is_behavior_neutral`): the schedule is unchanged for
all actions ≤ 450, and every action in 451-480 emits exactly the action-450
command.** Every existing artifact (preflight, captures, corrections,
recovery) consumed only actions ≤ 450 and is therefore unaffected.

*Amendment (2026-08-06, before the gate ran):* the design first tried an
explicit retreat→retreat hold stage (`boundaries … 450, 480`). The
invariant test caught that minimum-jerk interpolation of identical endpoints
wobbles the output by **one ulp** — and the same wobble is baked into the
*recorded* hold stages (closed-closed, lift-lift, released-released), so
`predict()` cannot special-case identical endpoints without forking legacy
capture bits. The explicit stage was reverted in favor of the (exact)
fall-through. A first 480 capture made under the wobble-stage code
(`oracle/fixed_pick_place_v3/5642bb3f8c3072f1`, 2,400 rows, zero safety
counts — incidentally the first physical proof that the 451-480 tail is
safe) is **superseded** by the fall-through re-capture recorded below and is
not used by the gate.

**Capture** (`simulation/oracle.py`): `capture_oracle_demonstrations` gains
`teacher_horizon` (keyword, default 450 — the default call's identity bytes,
and hence every legacy collection digest, are unchanged). One new capture:
five-scenario, `--teacher-horizon 480` → **2,400 rows** (5 × 480), progress
clock `action_index/479`, per-episode `rows: 480`. The 30 tail rows record
the physical servo settle onto the retreat pose (constant requested command,
decaying deltas) — explicit hold-tail targets, not degenerate padding.

**Cells** — one recipe, three seeds, all run to completion:

| Cell | Width | Steps | LR schedule | Seed |
| --- | ---: | ---: | --- | ---: |
| `seed101` | 512 | 90,000 | `cosine_floor_v1` | 101 |
| `seed202` | 512 | 90,000 | `cosine_floor_v1` | 202 |
| `seed303` | 512 | 90,000 | `cosine_floor_v1` | 303 |

The recipe is `FROZEN_RECIPE` + the precision tranche's established
overrides; the only new factor is the training data (480 capture). Students
load with `teacher_horizon = 480` from the checkpoint; the deployed progress
feature `min(action, 479)/479` matches the capture clock exactly and never
saturates within the contract.

Compute (regime v2): capture ~10 min; 3 trainings × ~7 min; 3 × (Stage A +
margins + Stage B) ≈ 15-25 min. Total ≈ 1-1.5 h.

**Capture record (2026-08-06):** fall-through capture
`oracle/fixed_pick_place_v3/9f54b9e66c884855` — 2,400 rows (5 × 480), zero
safety counts on every episode, progress clock `action_index/479`, and the
tail's `requested_act` verified **bit-constant** over actions 451-480 (the
exact-hold invariant holds in the physical data). This is the gate's
training input. The 451-480 tail is hereby physically proven safe.

## Gates (identical to prior tranches)

- **Stage A:** 15/15 deterministic five-scenario rollouts, zero
  clip/limit/nonfinite/unsafe frames (`STAGE_A_PASS_RULE`, byte-identical
  import), per seed.
- **Margins:** recorded per seed (telemetry).
- **Stage B:** eight anchor handoffs per seed, using the **legacy 450-row
  nominal oracle (`9164a76699186c34`) and recovery set (`d52b460b4ab3a683`)**
  — valid because the teacher is bit-identical for actions ≤ 450. Never
  gates.

All three seed cells run to completion regardless of earlier results.
Offline MSE remains telemetry.

## Promotion rule and terminal statuses

| Status | Meaning / next action |
| --- | --- |
| `horizon_promoted_robust` | **all three seeds** pass Stage A 15/15 zero-safety → the first multi-seed-robust recipe in the repo; next: the vision-rung proposal |
| `horizon_not_resolved` | any seed fails → frame-level failure analysis (`tools/analyze_safety_frames.py --teacher-horizon 480`) on every failing cell; stop; new proposal |

The ≥3-seed requirement is the promotion rule itself, calibrated against the
measured precision-tranche dispersion (12/15, 12/15, 3/15). No retry budget.

## Pre-registered caveats

- **The 451-480 tail is physically unexercised** until this capture: no
  recorded artifact has ever executed the teacher past action 450 (preflight
  episodes terminate at success, action 438). The capture's own gates
  arbitrate — episode success, **zero safety counts over all 480 actions**,
  and the `|requested − executed| ≤ 1e-6` label-purity check. A tail-induced
  safety event fails the capture loudly and terminates the tranche with that
  finding.
- Stage B measures handoffs on the legacy 450-action timeline (student
  stops at action 450 of the handoff episode); it is telemetry either way.
- The correction/recovery/observability capture pipelines retain their
  450-horizon loops; re-running those captures at 480 is future work,
  documented here as out of scope.
- Single change discipline: relative to the precision tranche's best cell,
  the only new factor is the aligned training data; schedule/width/steps are
  carried unchanged.

## Environment

Pinned mujoco 3.9.0 evaluation lane (`PYTHONNOUSERSITE=1`, conda hook + CLI
guard). Training under numerics regime v2 (`PINNED_NUMERICS_V2`; see
`notes/numerics-regime-v2.md`). Input digests: new 480 capture (recorded in
the Result section below on creation), nominal oracle `9164a76699186c34`,
recovery `d52b460b4ab3a683`, precision seeds report
`precision_gates/6d2c95a8f577c95b` (lineage), the passing v3 preflight.

## Artifacts

- Gate report: `artifacts/so_arm101_v2/oracle_distillation/horizon_gates/<digest16>/report.json`
- Stage A evaluations: `.../horizon_stage_a_evaluations/<digest16>/` (suite ids `fixed_pick_place_v3.horizon_seed<N>`)
- Capture: `.../oracle/fixed_pick_place_v3/<digest16>/` (2,400 rows, teacher_horizon 480)
- Implementation: `src/so_arm101_v2/simulation/horizon.py`, CLI
  `so-arm101-v2-sim run-horizon-gate`, capture flag
  `capture-oracle --teacher-horizon`, tests `tests/test_horizon_gate.py`,
  queue `simulation_code/queue_horizon_tranche_20260806.sh`
  (tsp labels `horizon-capture`, `horizon-gate`)

## Result (2026-08-06): `horizon_not_resolved` — prediction failed, mechanism advanced

Report: `horizon_gates/704b7a9574e559db` (~50 min). The 15/15 × 3 prediction
did **not** hold:

| Seed | Stage A clean successes | Safety frames | Handoffs | Failing scenarios |
| --- | ---: | ---: | ---: | --- |
| 101 | 9/15 | 24 | 5/8 | nominal, cube_y_minus |
| 202 | 9/15 | 48 | 7/8 | nominal, cube_x_minus |
| 303 | 12/15 | 33 | 5/8 | cube_y_minus |

**What the alignment did fix:** frame analyses of all three cells
(`notes/horizon-seed{101,202,303}-failure-analysis.md`) show **zero
past-horizon violations in any seed** — the extrapolation failure mode is
structurally eliminated, exactly as designed, and the physically-captured
hold tail is safe and bit-constant.

**What remains (the new binding constraint):** all violations are still the
gripper, now concentrated **in-distribution**: 22/35 findings in `set_down`,
6 in `traverse`, 7 in release — and the signed excesses show the policy
crossing the gripper's **floor bound (0.000472), not the 1.7 open bound**:
requested values dive to −0.002…−0.02, i.e. the student squeezes *harder
than fully closed*. The teacher generates grip force by commanding the
gripper essentially at the floor (its recorded closest approach to the
envelope is ~0.0035 ACT), so imitation overshoot around that wall-hugging
target crosses below zero. Notably the overshoot magnitude (up to 0.02)
exceeds the maximum feasible target margin (< 0.0035), so target-clipping
alone cannot absorb it. Per pre-registration: stop; the next proposal should
target the **gripper floor overshoot** (e.g. a gripper-channel output clamp
in the policy — the absolute-bound, single-channel form of the feasibility
decoder, which avoids the delta-chain lag that sank the full decoder — or a
teacher variant commanding a shallower squeeze), not horizon, budget, or
schedule — those levers are now all measured and exhausted.
