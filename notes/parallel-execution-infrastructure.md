# Parallel execution infrastructure + the MuJoCo environment fork (2026-08-03)

Written after implementing the "free speedups" infrastructure tranche (tranche paused for it).
Two things are documented here: the new `--workers` parallelism, and a **critical
environment-integrity finding** discovered while verifying it.

## 1. CRITICAL: user-site mujoco 3.11.0 forked all simulation numerics on Aug 2, 19:03

On **2026-08-02 at 19:03**, mujoco 3.11.0 (plus torch 2.7.1, pillow, pyarrow) was installed
into `~/.local/lib/python3.10/site-packages`. Python user-site shadows the `lerobot` conda
env's mujoco **3.9.0**, so every simulation run after that moment silently used 3.11.0.
Physics trajectories differ: divergence seeds at the last-ulp level in derived contact
quantities at action 1 and grows chaotically (up to ~2e-2 relative in `robot_qpos`).

Verified by copy-and-rerun with the immutability guards as the comparator:

| Artifact | Created | Reproduces with `PYTHONNOUSERSITE=1` (mujoco 3.9.0) | Reproduces without (3.11.0) |
|---|---|---|---|
| preflight `fixed_pick_place_v3` (`02decb93…` telemetry) | Aug 1 20:34 | **yes, bit-for-bit** | no |
| clone eval `e67ba4433d3aa98c` | Aug 2 19:16 | no | **yes, bit-for-bit** |
| saturation gate `1a78ec8affead704` + its 5 evals | Aug 3 02:23–11:59 | no | **yes, bit-for-bit** |

Consequences:
- The saturation gate (promoted `noise_penalty_only`) ran under 3.11.0 while consuming the
  3.9.0-era preflight, validated by hash only — the evidence chain mixes two numeric regimes
  and no gate could detect it.
- The paused broader-evaluation tranche is running under 3.11.0.
- The env's torch is 2.7.1+cu126 with or without user-site, so **training numerics are not
  forked** — only MuJoCo.

**Decision (2026-08-03): standardized on mujoco 3.9.0 via `PYTHONNOUSERSITE=1` — the
pre-switch stored lineage stays valid.** Actions taken:

- Pin enforced two ways: a conda activation hook
  (`~/miniforge3/envs/lerobot/etc/conda/activate.d/so_arm101_pin_mujoco.sh` exports
  `PYTHONNOUSERSITE=1`) and a hard guard in `so-arm101-v2-sim`
  (`EXPECTED_MUJOCO_VERSION = "3.9.0"` in `simulation/cli.py` — every sim command refuses
  to run under any other mujoco). Direct `python` invocations that bypass conda activation
  are caught by the CLI guard.
- All 34 post-switch sim-derived artifact directories were moved (not deleted) to
  `artifacts/so_arm101_v2/quarantine_mujoco_3_11_0/` preserving relative paths — see the
  README there. This includes the 3.11-regime saturation gate `1a78ec8affead704`
  (`promoted_noise_penalty_only`), the corrections capture `7feed472071fd725`, the recovery
  manifest, the observability/correction/chunked gate reports and evaluations, and the
  paused tranche's partial outputs.
- Training artifacts (`models/…`) were kept — torch 2.7.1 is identical in both regimes. The
  three correction-using saturation candidates were, however, trained on the 3.11-regime
  correction capture, so they are superseded by 3.9.0 retrains.
- Re-captured under 3.9.0: corrections → `corrections/53ad45590cfb60c0` (4818 rows, all 11
  sites accepted), recovery → `recovery/d52b460b4ab3a683` (8 rows).
- The saturation gate was re-run under 3.9.0 with the new correction manifest
  (`--workers 10`: two cached valid trainings reused, three correction candidates retrained
  in parallel, all evals parallel). **Result: `saturation_gates/46de62c4f6d1b78f`, status
  `promoted_noise_penalty_only` — identical attribution to the quarantined 3.11 gate (only
  `noise_penalty_only` passes; all five evaluations deterministic) and the byte-identical
  promoted checkpoint (`models/chunked_h90/2b6195d619ab531b`,
  sha `f6c59b8e…`).** The promotion decision therefore survives the regime change; the
  3.11 gate `1a78ec8affead704` is historical only. The three fresh correction retrains all
  ran to the full 30,000-step cap (~2–9.5 h CPU each, concurrent).
- The paused broader-evaluation tranche ran under 3.11.0 and its partial outputs are
  quarantined; it must be relaunched fresh under the pin, against the new gate report and
  the new recovery manifest, once the 3.9.0 gate reports.
- `pytest` previously resolved only from user-site (so test runs silently used the shadowed
  packages too); it is now installed in the lerobot env itself, and the full suite (180
  tests) passes under `PYTHONNOUSERSITE=1` — run tests with the pin from now on.

Also note: the code changes below were proven bitwise-neutral — the pre-change code
(reverse-patched) and post-change code produce identical telemetry under the same mujoco,
and the post-change code reproduces the Aug 1 preflight bit-for-bit under 3.9.0.

## 2. What was added (defaults unchanged — `workers=1`, video on, byte-stable)

- **Atomic immutable writes** — `write_immutable_bytes` in `data/_serialization.py`
  (mkstemp + fsync + `os.link` create-exclusive publish; concurrent identical writers both
  succeed, divergent writers get the usual `FileExistsError`, crashes leave no torn files).
  All seven copy-pasted `_write_immutable_bytes` helpers plus the HTML and `model.pt` raw
  writes now delegate to it.
- **`PolicySpec` registry** (`simulation/policy_specs.py`) — picklable policy construction
  (`kind` + checkpoint path + options) replacing lambda factories, so spawn workers can
  rebuild policies. Bare callables still work sequentially; `workers > 1` with a callable
  raises.
- **Parallel rollouts** — `evaluate_closed_loop(..., workers=N)` and
  `run_simulation_preflight(..., workers=N)` fan independent rollouts out to a spawn-context
  `ProcessPoolExecutor`; results are collected in submission order so `evaluation.json` is
  byte-identical to the sequential fan-out; workers pin `torch.set_num_threads(1)`.
- **Parallel saturation gate** — `run_saturation_gate(..., workers=N)` runs the 5 always-run
  candidates (train + eval) in parallel workers; budget split
  `candidate_workers = min(5, N)`, `eval_workers = N // candidate_workers`. Ladder gates
  (chunked horizon, observability) stay sequential by decision — early-stop semantics
  preserved — but their inner evals accept `workers`.
- **Honest `--no-video`** — policies expose `requires_pixels`; the adapter renderer is now
  lazy and `observation(render_pixels=False)` skips the wrist render, so pixel-free policies
  with video off do **zero** rendering and never create a GL context. With video on (the
  default everywhere) behavior is unchanged.
- **CLI** — `--workers N` on `preflight`, `evaluate`, `evaluate-clone`, and every gate.
  Recommended on this 16-core box: `--workers 15` with `--no-video`, `--workers 10` with video.

## 3. Parity evidence (each run under the artifact's native mujoco regime)

Protocol: copy the original telemetry into a scratch tree, delete only the derived reports,
re-run the parallel implementation into the copy. The immutability guards fail loudly on any
byte difference in telemetry; reports are compared field-by-field with absolute-path fields
normalized (reports embed resolved paths, so cross-directory hash comparison is impossible).

| Run | Workers | Result | Wall clock |
|---|---|---|---|
| preflight (15 rollouts, video) | 8 | bitwise telemetry + report match | **1m10s** (vs 3m37s sequential) |
| evaluate-clone `e67ba443` (15 rollouts) | 6, thread-pinned | bitwise telemetry; report matches except video-path fields (original ran `--no-video`) — proves 1-thread workers reproduce default-thread artifacts | 1m16s |
| saturation gate, evals only (trainings cached) | 15 | bitwise: gate report + all 5 eval reports + all telemetry | **1m16s** |
| saturation gate, full from-scratch trainings | 15 | **bitwise**: all 5 candidate `model.pt` checkpoints and training reports byte-identical to the sequential originals; gate report identical after path normalization; exit 0 | ~7.5h wall (5 concurrent 30k-step trainings; sequential originals took ~9.5h just for training) |

Unit tests: `tests/test_serialization_atomic.py`, `tests/test_policy_specs.py`,
`tests/test_rollout_parallel.py` (real MuJoCo spawn-worker parity), plus parallel
orchestration tests appended to `tests/test_saturation_gate.py`. Full suite: 180 tests pass.

## 4. Deliberately deferred (digest-coupled, needs its own pre-registered parity tranche)

> **Addendum (2026-08-06):** the `torch.compile` and GPU/float-tolerance items
> below are now resolved through their own pre-registered validation ladder —
> **numerics regime v2** (GPU + inductor training by default, legacy cpu-eager
> preserved for byte-exact reproduction). See `notes/numerics-regime-v2.md`.
> The per-step metrics-pass trim remains deferred (still digest-coupled), and
> speculative parallel ladder gates remain declined.

- `torch.compile` of the training step (adopt only on bitwise parity).
- Trimming the per-step post-optimizer inference pass in `oracle_distillation.py` /
  `tiny_model.py` — it feeds the early-stop check and prefix-parity assertion, so removing it
  changes stopping steps and forks run digests. **Not** a free win as previously assumed.
- GPU / float-tolerance policy.
- Speculative parallel execution of early-stop ladder gates.


## 5. Process-parallel oracle capture (2026-09-09)

`capture_oracle_demonstrations(..., workers=)` (CLI `capture-oracle --workers`,
`tools/run_bench_pipeline.py --workers`, which also forwards to the preflights and
the closed-loop evaluations that used to hard-code `workers=1`). Each scenario
is an independent deterministic episode (fresh adapter and controller, no RNG,
no cross-scenario state), so scenarios fan out over a spawn pool exactly like
rollouts: results are assembled in scenario order, and frames are written by
each worker into its own slot `[i·H, (i+1)·H)` of the single preallocated
memmap; skipped scenarios are compacted forward in ascending order, which
reproduces the sequential running-row layout byte for byte. The overview
(`camera_side`) render is now skipped when no video is recorded; it only ever
fed the video writer.

Parity evidence (`tests/test_oracle_capture_parallel.py`): `demonstrations.npz`
bytes, episodes, labels and every manifest field except the frames block are
identical between `workers=1` and `workers=2`; the video toggle is likewise
neutral for arrays and episodes.

**Rendered frames are not byte-reproducible, and never were.** Measured
2026-09-09 on the legacy v3 scene: two *sequential* captures in the same
process differ on 72 of 900 frames, two parallel captures on 30, sequential vs
parallel on 22, always by at most one grey level on a handful of pixels (EGL
rasterization jitter). Physics arrays are bit-identical in every pairing. So
`frames.sha256` and hence `collection_digest` of a frames capture are a record
of what was captured, not a reproducibility claim; the tests compare frames
with that tolerance (max |Δ| ≤ 1, < 0.1 % of pixels).

Expected wall-clock for the bench capture (400 × 480 steps, lens path ≈ 65
ms/step sequential): ~10 workers → ~15–20 min instead of ~3.5 h.

### Training-side measurements (same day)

Raw random-read ceiling of the 37.7 GB frames sidecar on this WSL disk: 5.7
batches/s at 1 thread, 10.0 at 6, 12.8 at 12 (161 MB/s), 10.5 at 24. The GPU
step with data already on the device is 5.3 ms (w256, batch 64); host-to-device
plus conversion of a uint8 batch is 1.9 ms. So the vision loop was disk-bound
at ~10 steps/s and the prefetch/upload knobs alone gave 9.5 → 10.6 steps/s.
The in-RAM lossless frame cache (`SO_ARM101_V2_FRAME_CACHE`, see
`notes/environment-switches.md`) removes the disk from the loop: 57.8 steps/s
measured over steps 1500–3000 on the real data, bit-identical results.
