# Environment switches

Every environment variable that changes how the `so_arm101_v2` lane runs, in
one place. Two rules govern the whole table: **nothing here enters an
identity payload**, so no switch can move a content digest or invalidate a
legacy artifact; and anything that would change *results* rather than
*speed* says so explicitly.

## Required for every run

| Variable | Value | Why |
| --- | --- | --- |
| `PYTHONNOUSERSITE` | `1` | Keeps the conda env's pinned mujoco 3.9.0 ahead of a `pip --user` install. A shadowed mujoco 3.11.0 once forked the sim numerics and tainted 34 artifacts (`notes/parallel-execution-infrastructure.md`); the `so-arm101-v2-sim` CLI now refuses any version other than `EXPECTED_MUJOCO_VERSION` (`simulation/cli.py:276`), but this variable is what stops the mismatch happening in the first place. |
| `MUJOCO_GL` | `egl` | Headless GPU rendering. Required for wrist-camera capture and any evaluation that records video on this machine. |
| `PYTHONUNBUFFERED` | `1` | Makes progress lines reach the task-spooler log immediately instead of at process exit. Convenience only. |

The queue scripts under `simulation_code/queue_*.sh` export all three; set
them yourself when running a command by hand.

## Numerics

| Variable | Default | Effect |
| --- | --- | --- |
| `SO_ARM101_V2_NUMERICS` | `auto` | `legacy` forces the CPU-eager regime (`resolve_default_numerics`, `learning/numerics.py:116`); any other value, including unset, resolves to the pinned GPU regime `PINNED_NUMERICS_V2` when CUDA is live and legacy otherwise. |

**This one changes results, by design.** Regime v2 and legacy produce
different (both reproducible) numbers, which is why the regime is
fingerprinted into training identities — a run's digest already tells you
which regime produced it. Use `legacy` to reproduce pre-2026-08-06 digests
bit for bit, or to train without a GPU. See `notes/numerics-regime-v2.md`.

Two variables are *set for you* by `apply_numerics` via `setdefault` and
should be left alone: `CUBLAS_WORKSPACE_CONFIG` (`:4096:8`, required for
deterministic cuBLAS) and `TORCHINDUCTOR_CACHE_DIR`
(`~/.cache/so_arm101_v2/torchinductor`). Presetting either wins over the
default, and overriding the cuBLAS one silently costs determinism.

## Vision training throughput

Frame reads are the only per-step disk I/O in vision training. Once a frames
sidecar outgrows the page cache — the 38 GB n400 sidecar against 32 GB of RAM
— those reads dominate wall time, so they run on a small reader pool while
index draws and every float operation stay on the training thread in step
order (`learning/vision.py:253`).

| Variable | Default | Effect |
| --- | --- | --- |
| `SO_ARM101_V2_PREFETCH` | `1` | `0` restores the synchronous read path. |
| `SO_ARM101_V2_PREFETCH_DEPTH` | `8` | Batches read ahead. A buffered batch costs `batch_size × 192 KiB` (one 256×256×3 uint8 frame per row), so the defaults hold about 100 MB at `batch_size=64`. |
| `SO_ARM101_V2_PREFETCH_WORKERS` | `6` | Reader threads; this is what raises the effective disk queue depth. |

**These change speed only.** Prefetched and synchronous training are bitwise
identical — `tests/test_vision_lane.py` pins the same run digest, the same
loss trace, and the same checkpoint sha across the two paths — so the kill
switch exists for debugging, not for correctness. Measured 3.2x on the n400
sidecar.

## Experiment tracking

Tracking is opt-in per invocation (`--wandb-project` on
`tools/run_randomized_v1.py`) and observational: it records the loss curve
and evaluation summaries and touches nothing that is hashed.

| Variable | Read by | Effect |
| --- | --- | --- |
| `WANDB_NAME` | the driver, passed to `wandb.init` | Run name. Unset means wandb picks one. |
| `WANDB_RUN_GROUP` | wandb | Groups a sweep's runs, e.g. `vision-n400-120k-20260807`. |
| `WANDB_SILENT` | wandb | `true` keeps wandb's banner out of the job log. |

Credentials come from `~/.netrc`; no key belongs in a script. Current
project: `so-arm101-v2-scaling`.
