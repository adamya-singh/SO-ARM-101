#!/usr/bin/env bash
# Gripper-clamp gate (notes/gripper-clamp-proposal.md): eval-only, no training.
set -euo pipefail

TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
PY="/home/win10ubuntu/miniforge3/envs/lerobot/bin/python"
PROJ="/home/win10ubuntu/dev/robotic-arm/SO-ARM-101"
ORACLE_DIST="$PROJ/artifacts/so_arm101_v2/oracle_distillation"
DRY_RUN="${DRY_RUN:-0}"

[[ -x "$TSP" ]] || { echo "Missing task spooler: $TSP" >&2; exit 1; }

if [[ "$DRY_RUN" == "1" ]]; then
    ECHO="echo DRY:"
else
    "$TSP" -S 1
    ECHO=""
fi

$ECHO "$TSP" -N 1 -L clamp-gate env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
    "$PY" -u -m so_arm101_v2.simulation.cli run-clamp-gate \
    --mujoco-model "$PROJ/simulation_code/model/menagerie_so_arm100/scene_v2.xml" \
    --output-dir "$ORACLE_DIST" \
    --horizon-gate-report "$ORACLE_DIST/horizon_gates/704b7a9574e559db/report.json" \
    --oracle-manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json" \
    --recovery-manifest "$ORACLE_DIST/recovery/d52b460b4ab3a683/manifest.json" \
    --preflight-report "$PROJ/artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"

"$TSP"
