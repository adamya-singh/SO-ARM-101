# Gripper-Clamp Gate (Proposal, pre-registered before any run)

## Question

The horizon-alignment tranche (`notes/horizon-alignment-proposal.md`,
terminal `horizon_not_resolved`, gate `horizon_gates/704b7a9574e559db`)
localized the last failure mode: the three seed policies request gripper
positions **below the floor bound** (0.000472) by −0.002…−0.02 while
squeezing during `set_down`/`traverse` — the teacher commands the gripper at
the floor for grip force, and imitation overshoot crosses it. The
`noise_penalty_v1` recipe penalizes this at training time but enforces
nothing at inference.

Question: does clamping the **gripper channel to its exact effective bounds
at the policy output** — an evaluation-time architecture variant requiring
no retraining — turn the three existing horizon checkpoints into
15/15 zero-safety policies?

**Prediction on record:** seeds 101 and 202 pass (their violations are all
`envelope_clip` on the gripper, which the clamp removes by construction).
Overall `clamp_promoted_robust` **iff seed 303's three `delta_limit` frames
(actions 413-415) dissolve** — see the risk below. Primary prediction: all
three pass.

## Design

**Variant** (`simulation/chunked.py`): `ChunkedClonePolicy` gains
`clamp_channels: tuple[int, ...] = ()`. At the per-action buffer pop — the
single choke point where commands leave the policy — each listed channel is
clipped to `effective_safe_act_bounds()` (cached at load). The gate uses
`clamp_channels = (5,)` (gripper only). `policy_id` appends
`.gripper_clamp_v1` when exactly the gripper is clamped. Delivered through
the standard `PolicySpec` options mechanism (picklable; spawn-worker safe).

**Legality (verified empirically before registration):** the envelope test
in `contracts/physical.py` uses strict inequality, and requesting exactly
`low[5]` (4.7194745e-4) or exactly `high[5]` (1.7) produces all-False
clip masks. **Scope is gripper-only by necessity:** clamping other channels
to their raw bounds trips `mujoco_clip_mask` on `shoulder_lift`/`elbow_flex`
via float32 round-off in the coordinate pullback; a general clamp would need
an inward margin and is not registered here.

**Precedent distinction:** the failed `feasible_chain_v1` decoder was a
recursive per-step **delta** chain in normalized space whose clamping
propagated through the chunk (and lagged the measured pose). This variant is
a stateless per-action **absolute** clamp on one channel, applied after
decode. They share nothing but the word "clamp".

**Gate** (eval-only; no training): `simulation/clamp.py`, the
`run_correction_probe` pattern at `horizon.py` cell structure. Cites
`horizon_gates/704b7a9574e559db` by content hash
(`7c7784de46ff48e9…`); validates each cell's checkpoint bytes
(`checkpoint_sha256`) and training report hash against the cited report:

| Cell | Checkpoint (models/chunked_h90/…) | checkpoint_sha256 (prefix) |
| --- | --- | --- |
| `seed101` | `67a8d95efa2ca3f1/model.pt` | `146dbb285f2238b5` |
| `seed202` | `4055a2537851dd0f/model.pt` | `3f8d941bd63178f1` |
| `seed303` | `5668b555861db8ca/model.pt` | `75c0933b626e2a77` |

Per cell: Stage A (15/15 deterministic five-scenario rollouts, zero
clip/limit/nonfinite/unsafe frames — the imported `STAGE_A_PASS_RULE`),
margin analysis (telemetry), Stage B anchor handoffs (telemetry). Suite ids
`fixed_pick_place_v3.clamp_seed<N>`; evaluations under
`clamp_stage_a_evaluations/<digest16>`; report under
`clamp_gates/<digest16>`.

**Statuses:** `clamp_promoted_robust` (all three pass Stage A) → rung 3
closes; the vision rung begins either way. `clamp_not_resolved` (any seed
fails) → frame analysis on every failing cell, findings documented — and
**the vision rung begins anyway** (pre-registered: no further rung-3 tuning
regardless of outcome; the clamp result simply decides whether the vision
rung inherits a robust state-policy baseline or a near-robust one).

## Pre-registered risk

Seed 303's failure set includes **three `delta_limit` frames** (actions
413-415, excess up to +0.035). The delta test applies the envelope clip
*before* measuring the step (`|clip(requested) − current| > cap`), so an
output clamp does not remove those frames directly — they can only vanish
through the second-order trajectory change the clamp induces (the executed
gripper value shifts by ~2e-4 rad on formerly-clipped frames, and the
rollouts genuinely differ from there). If seed 303 fails on exactly those
frames, that is the expected partial outcome, not an anomaly.

## Environment

Pinned mujoco 3.9.0 evaluation lane; no training, so the numerics regime is
irrelevant to the cells (evaluation is CPU as always). Inputs by digest:
horizon gate report `7c7784de46ff48e9…`, the three checkpoints above, the
passing v3 preflight (`0330cc10…`).

## Artifacts

- Report: `artifacts/so_arm101_v2/oracle_distillation/clamp_gates/<digest16>/report.json`
- Implementation: `simulation/clamp.py`, `ChunkedClonePolicy.clamp_channels`,
  `policy_specs.py` option plumbing, CLI `so-arm101-v2-sim run-clamp-gate`,
  tests `tests/test_clamp_gate.py`,
  queue `simulation_code/queue_clamp_gate_20260806.sh` (tsp label `clamp-gate`)

## Result (2026-08-06): `clamp_not_resolved` — the prediction held in detail

Report: `clamp_gates/f63104b1f82cadfc` (~14 min, eval-only).

| Cell | Stage A | Successes | Safety frames | Handoffs |
| --- | --- | ---: | ---: | ---: |
| seed101 | **PASS** | **15/15** | **0** | 5/8 |
| seed202 | **PASS** | **15/15** | **0** | 7/8 |
| seed303 | fail | 12/15 | 9 | 5/8 |

Seeds 101 and 202 are the **first policies in this repository to pass Stage
A 15/15 with zero safety frames**. Seed 303 fails exactly as pre-registered:
frame analysis (`notes/clamp-seed303-failure-analysis.md`) shows its only
violations are the three pre-identified `delta_limit` frames (actions
413-415, release stage, excess up to +0.035) — the gripper opens faster than
the per-step cap, which an absolute output clamp cannot remove, and the
clamp's second-order trajectory change did not dissolve them.

Per pre-registration: `clamp_not_resolved` under the strict all-three rule;
**the vision rung starts anyway**. The gripper clamp is validated as an
architecture element (it eliminated 100% of envelope violations at zero
training cost) and should be carried into the vision lane. The one residual
defect — single-seed release-speed overshoot, 3 frames — is recorded as an
open item; the natural candidate fix (a single-channel per-step rate limit
at the policy output) is noted for a future exploratory pass, distinct from
the failed multi-channel delta-chain decoder.
