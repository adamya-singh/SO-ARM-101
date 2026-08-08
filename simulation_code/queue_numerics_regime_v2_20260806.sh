#!/usr/bin/env bash
# Numerics-regime-v2 validation ladder (notes/numerics-regime-v2.md, section 5).
# Rungs c (GPU determinism) and e (benchmark matrix) enqueue immediately;
# rung d (saturation re-validation) and rung f (scaling re-run) are enqueued
# by the operator only after the preceding rung's pass criteria are verified.
set -euo pipefail

TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
PY="/home/win10ubuntu/miniforge3/envs/lerobot/bin/python"
PROJ="/home/win10ubuntu/dev/robotic-arm/SO-ARM-101"
SCRATCH="${NRV2_SCRATCH:-/tmp/claude-1000/-home-win10ubuntu-dev-robotic-arm/9cb90642-a478-4b90-b812-018e2f957613/scratchpad/nrv2}"
PROBE="$SCRATCH/nrv2_parity_probe.py"
ORACLE_DIST="$PROJ/artifacts/so_arm101_v2/oracle_distillation"
DRY_RUN="${DRY_RUN:-0}"

[[ -x "$TSP" ]] || { echo "Missing task spooler: $TSP" >&2; exit 1; }
[[ -f "$PROBE" ]] || { echo "Missing probe script: $PROBE" >&2; exit 1; }

enqueue() {
    local label="$1"; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY %s: %s\n' "$label" "$*"
    else
        "$TSP" -N 1 -L "$label" "$@"
    fi
}

if [[ "$DRY_RUN" != "1" ]]; then "$TSP" -S 1; fi   # one GPU: serialize the lane

STAGE="${1:-c}"

if [[ "$STAGE" == "c" ]]; then
    enqueue nrv2-gpu-det-a env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
        "$PY" -u "$PROBE" run --lane chunked \
        --config-from "$ORACLE_DIST/models/chunked_h90/2b6195d619ab531b/report.json" \
        --manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json" \
        --numerics gpu-compile --output-dir "$SCRATCH/c_gpu_a"
    enqueue nrv2-gpu-det-b env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
        "$PY" -u "$PROBE" run --lane chunked \
        --config-from "$ORACLE_DIST/models/chunked_h90/2b6195d619ab531b/report.json" \
        --manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json" \
        --numerics gpu-compile --output-dir "$SCRATCH/c_gpu_b"
    enqueue nrv2-bench-matrix env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
        "$PY" -u "$PROJ/tools/benchmark_training_numerics.py" \
        --manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/3f8eb5499eb18128/manifest.json" \
        --json "$PROJ/outputs/benchmarks/numerics_regime_v2/matrix_20260806.json"
elif [[ "$STAGE" == "d" ]]; then
    enqueue nrv2-saturation-revalidation env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
        "$PY" -u -m so_arm101_v2.simulation.cli run-saturation-gate \
        --mujoco-model "$PROJ/simulation_code/model/menagerie_so_arm100/scene_v2.xml" \
        --output-dir "$ORACLE_DIST" \
        --oracle-manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json" \
        --correction-manifest "$ORACLE_DIST/corrections/53ad45590cfb60c0/manifest.json" \
        --preflight-report "$PROJ/artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"
elif [[ "$STAGE" == "f" ]]; then
    enqueue nrv2-scaling-rerun env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
        "$PY" -u -m so_arm101_v2.simulation.cli run-scaling-gate \
        --mujoco-model "$PROJ/simulation_code/model/menagerie_so_arm100/scene_v2.xml" \
        --output-dir "$ORACLE_DIST" \
        --capture-manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/3f8eb5499eb18128/manifest.json" \
        --oracle-manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json" \
        --recovery-manifest "$ORACLE_DIST/recovery/d52b460b4ab3a683/manifest.json" \
        --baseline-tranche-report "$ORACLE_DIST/broader_evaluations/68c56c66d2dd3bb9/report.json" \
        --preflight-report "$PROJ/artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"
else
    echo "unknown stage: $STAGE (use c, d, or f)" >&2; exit 1
fi

"$TSP"
