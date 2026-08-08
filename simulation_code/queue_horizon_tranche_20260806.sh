#!/usr/bin/env bash
# Horizon-alignment tranche (notes/horizon-alignment-proposal.md).
# Stages: capture | gate <capture-manifest-path>
set -euo pipefail

TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
PY="/home/win10ubuntu/miniforge3/envs/lerobot/bin/python"
PROJ="/home/win10ubuntu/dev/robotic-arm/SO-ARM-101"
ORACLE_DIST="$PROJ/artifacts/so_arm101_v2/oracle_distillation"
DRY_RUN="${DRY_RUN:-0}"

[[ -x "$TSP" ]] || { echo "Missing task spooler: $TSP" >&2; exit 1; }

enqueue() {
    local label="$1"; shift
    if [[ "$DRY_RUN" == "1" ]]; then
        printf 'DRY %s: %s\n' "$label" "$*"
    else
        "$TSP" -N 1 -L "$label" "$@"
    fi
}

if [[ "$DRY_RUN" != "1" ]]; then "$TSP" -S 1; fi

STAGE="${1:-capture}"

case "$STAGE" in
capture)
    enqueue horizon-capture env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
        "$PY" -u -m so_arm101_v2.simulation.cli capture-oracle \
        --mujoco-model "$PROJ/simulation_code/model/menagerie_so_arm100/scene_v2.xml" \
        --output-dir "$ORACLE_DIST" \
        --suite fixed_pick_place_v3 --scenario all --teacher-horizon 480 \
        --preflight-report "$PROJ/artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"
    ;;
gate)
    [[ $# -ge 2 ]] || { echo "gate stage requires the 480 capture manifest path" >&2; exit 1; }
    enqueue horizon-gate env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
        "$PY" -u -m so_arm101_v2.simulation.cli run-horizon-gate \
        --mujoco-model "$PROJ/simulation_code/model/menagerie_so_arm100/scene_v2.xml" \
        --output-dir "$ORACLE_DIST" \
        --capture-manifest "$2" \
        --oracle-manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json" \
        --recovery-manifest "$ORACLE_DIST/recovery/d52b460b4ab3a683/manifest.json" \
        --precision-report "$ORACLE_DIST/precision_gates/6d2c95a8f577c95b/report.json" \
        --preflight-report "$PROJ/artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"
    ;;
*)
    echo "unknown stage: $STAGE (use capture, or gate <manifest>)" >&2; exit 1
    ;;
esac

"$TSP"
