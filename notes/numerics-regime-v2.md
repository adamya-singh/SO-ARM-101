# Numerics Regime v2: GPU + torch.compile training by default (2026-08-05)

Canonical record of the training-numerics regime change. Cross-links:
`notes/parallel-execution-infrastructure.md` (the mujoco 3.9.0 regime template
and the previously deferred GPU/compile items), `notes/optimization-scaling-proposal.md`
(the in-flight run handled under this change).

## 1. Decision and rationale

Training moves from single-threaded eager CPU to **GPU (RTX 3090) +
`torch.compile` (inductor), by default** — no flags needed to run optimally.
The pre-registered ladder (README "Next Steps") said the compute/determinism
policy would be "decided explicitly at the vision rung"; it was decided
**early, at the scaling rung**, because the 90k-step scaling arms made CPU
economics binding (~9 h per arm) and the decision could be made empirically
here (bitwise GPU determinism proven, promotion re-validated behaviorally)
rather than deferred. Measured on the 2,250-row capture, width 256:
cpu-eager ≈ 95.5 ms/step vs gpu-compile ≈ 4.0 ms/step (~24x); a 90k-step
training drops from hours to ~6 minutes.

## 2. Regime definition

A training run's numerics regime is either **legacy** (cpu-eager; the lane
every pre-v2 digest was produced under) or a **`NumericsSpec`** fingerprint
(`src/so_arm101_v2/learning/numerics.py`). The fingerprint enters the
training identity and every gate identity as a **conditionally absent**
top-level `numerics` key: absent = legacy (all existing digests byte-valid
forever), present = v2 (its own digest space). Identity fields:

`regime, device, gpu, torch, cuda, driver, compile, compile_mode, tf32,
deterministic_algorithms, cublas_workspace, noise_stream`

Pinned v2 environment (`PINNED_NUMERICS_V2`): RTX 3090, torch 2.7.1+cu126,
CUDA 12.6, **driver 610.47 (identity-bearing: a driver update forks v2
digests loudly, by design)**, inductor `mode="default"` (never max-autotune —
autotune selects kernels by measured runtime, which is not bit-reproducible),
TF32 off, `torch.use_deterministic_algorithms(True)`,
`CUBLAS_WORKSPACE_CONFIG=:4096:8`, noise stream `cpu_generator_v1` (noise-
penalty augmentation draws from a dedicated seeded CPU generator and is
uploaded to the device, so the noise bits are device-invariant).

What compiles: the step-loss functions (`_chunked_step_loss`,
`_oracle_step_loss`) — both model forwards plus the whole penalty/loss chain
fuse into one inductor graph; optimizer and all host syncs stay eager.
Compiled kernels cache in `~/.cache/so_arm101_v2/torchinductor` (WSL2 wipes
/tmp).

## 3. Enforcement (two ways, mujoco-pin template)

1. `require_numerics` fires inside `apply_numerics` at every v2 training:
   exact-match on torch / CUDA / GPU name / driver (via nvidia-smi;
   unresolvable driver is an error). One enforcement point covers both CLIs
   and every library caller.
2. The legacy lane is only reachable explicitly: `--legacy-numerics` (CLI) or
   `numerics=None` (library) or `SO_ARM101_V2_NUMERICS=legacy` (env; the test
   suite pins this in `tests/conftest.py`). Nothing silently re-runs legacy.

## 4. What is NOT forked — and no quarantine

- **Closed-loop evaluation stays CPU-pinned** (`ChunkedClonePolicy` etc.;
  `TorchCheckpointPolicy` default flipped from "auto" to "cpu", closing a
  latent GPU path in the evidence lane). All simulation/rollout lineage,
  including the mujoco 3.9.0 pin, is untouched.
- **No quarantine.** Unlike the mujoco fork (old artifacts irreproducible
  under the pin → moved), v2 invalidates nothing: every existing digest still
  reproduces bit-for-bit under `--legacy-numerics`. The flip changes only
  *future* digests.
- Proposals audited and unaffected (complete, legacy-valid, never re-executed
  under v2): chunked-promotion, bounded-observability, dagger-correction,
  broader-evaluation. The saturation gate is re-validated (rung d); the
  scaling tranche re-executes under v2 (see its proposal addendum).
- The tiny-model acceptance lane (`tiny_model.py`) stays legacy-CPU forever.

## 5. Pre-registered validation ladder

Written before rungs b-f ran. Pass criteria are binding; the rollback row
states what happens on failure.

| Rung | What | Pass criterion |
|---|---|---|
| a | Full pytest suite under the pin | all green, legacy digest pins (`test_saturation_gate.py:201` family) unmodified — **PASSED (194 tests)** |
| b1 | cpu-compile re-train of stored `models/phase_state/2215e6361027023e` | state-dict bitwise + loss-trace float comparison vs stored; documentation-only outcome |
| b2 | cpu-compile re-train of the promoted `models/chunked_h90/2b6195d619ab531b` (w256/450 rows/30k) | same; either outcome documented (skipped if b1 already diverges — the question is answered) |
| c | Two separate-process gpu-compile trainings of the promoted recipe | **byte-identical state dicts + identical report floats** — hard gate for GPU default |
| d | **Acceptance gate**: saturation gate re-run under v2, same inputs as `saturation_gates/46de62c4f6d1b78f` | status `promoted_noise_penalty_only` AND identical per-candidate attribution. Pre-registered expectation: the v2 promoted checkpoint will NOT be byte-identical to `2b6195d…` — acceptance is behavioral, exactly like the mujoco 3.9.0 re-validation |
| e | Benchmark matrix {cpu-eager, cpu-compile, gpu-eager, gpu-compile} x {256, 512} | numbers recorded below; default = fastest config that passed c |
| f | Scaling tranche re-run under v2 (`nrv2-scaling-rerun`) | its own pre-registered terminal statuses (not a regime criterion) |

Rollback (cheap by construction — only conditionally-present keys were added):
rung c any byte differs between the two runs → GPU stays opt-in (default
falls back to cpu-compile if b proved bitwise, else full legacy); gpu-eager
deterministic but gpu-compile not → default gpu-eager. Rung d attribution
differs → **abort the default flip**, v2 becomes an opt-in research lane and
the scaling re-run is re-queued as the original legacy CPU run. Any
immutability-guard divergence on a *legacy* artifact at any point → stop
everything (mujoco-incident-class investigation).

## 6. In-flight run handling

The scaling tranche's first (legacy) launch was killed pre-artifact on
2026-08-05 23:41 — see the dated addendum in
`notes/optimization-scaling-proposal.md`. Re-run: tsp label `nrv2-scaling-rerun`.

## 7. Results

(filled as rungs complete)

| Rung | Outcome |
|---|---|
| a | PASSED — 194 tests green under `PYTHONNOUSERSITE=1`; legacy digest pins untouched |
| b1 | **DIVERGED** — cpu-compile does not reproduce legacy bytes: the early-stop fired at a different step (stored: 7577), so weights, loss trace, and metrics all differ. Consequence: **inductor fusion changes CPU bits; the legacy lane remains cpu-eager only**, and cpu-compile is a v2-only regime (`cpu_inductor_v2`). No rollback implication (documentation-only rung). |
| b2 | **skipped as redundant** — b1 answered the fusion-changes-bits question decisively; a 2 h re-train of the promoted artifact under cpu-compile could only re-confirm divergence. Legacy reproduction of `2b6195d…` stays `--legacy-numerics` cpu-eager. |
| c | **PASSED — BITWISE PARITY.** Two separate task-spooler processes trained the promoted recipe (w256/450 rows/30k steps) under gpu-compile: byte-identical state dicts, identical steps/mse/max-act-error/loss-trace. ~2.5 min per run including compile. The hard gate for the GPU default is cleared. |
| d | **PASSED — regime v2 accepted.** Saturation gate re-run under v2: `saturation_gates/32f8972f7996b5f6`, status `promoted_noise_penalty_only`, per-candidate attribution identical to the legacy gate `46de62c4f6d1b78f` (only `noise_penalty_only` passes; all five evaluations deterministic). As pre-registered, no v2 checkpoint is byte-identical to its legacy counterpart — acceptance is behavioral. Five GPU trainings + parallel CPU evals completed in ~19 min end to end (the legacy equivalent took ~9.5 h of training alone). |
| e | **Full matrix (2,250-row capture, 50 warmup + 200 measured steps):** cpu-eager 97.4/198.4 ms-per-step (w256/w512), cpu-compile 89.1/192.4, gpu-eager 6.87/7.01, **gpu-compile 4.13/4.57 → 90k steps in ~6-7 min (≈43x vs cpu-eager at w512)**. gpu-compile is the fastest config that passed rung c → confirmed as the default. JSON: `outputs/benchmarks/numerics_regime_v2/matrix_20260806.json`. |
| f | **COMPLETE — `scaling_gates/b672196e85a945aa`, status `scaling_not_resolved`** (science result, no regime implication). All three arms fail Stage A on `safety_invalidation`; clean successes scale monotonically: steps90k 3/15, width512 3/15, width512_steps90k **9/15** with full milestone chains. Per pre-registration: stop, write a new proposal. Wall clock ~35 min under v2 (vs the aborted legacy launch's projected ~9 h). |

The full catalogue of environment switches this lane reads — including
this one — lives in `notes/environment-switches.md`.
