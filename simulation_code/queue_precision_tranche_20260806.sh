#!/usr/bin/env bash
# Precision tranche (notes/precision-tranche-proposal.md): staged tsp jobs.
# Stages: analysis | schedule | budget <schedule-report> | seeds <budget-report>
# Operator advances stages after verifying the prior stage's report and
# updating docs (same idiom as queue_numerics_regime_v2_20260806.sh).
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

common_stage_args=(
    --mujoco-model "$PROJ/simulation_code/model/menagerie_so_arm100/scene_v2.xml"
    --output-dir "$ORACLE_DIST"
    --capture-manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/3f8eb5499eb18128/manifest.json"
    --oracle-manifest "$ORACLE_DIST/oracle/fixed_pick_place_v3/9164a76699186c34/manifest.json"
    --recovery-manifest "$ORACLE_DIST/recovery/d52b460b4ab3a683/manifest.json"
    --scaling-gate-report "$ORACLE_DIST/scaling_gates/b672196e85a945aa/report.json"
    --preflight-report "$PROJ/artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"
)

if [[ "$DRY_RUN" != "1" ]]; then "$TSP" -S 1; fi

STAGE="${1:-analysis}"

case "$STAGE" in
analysis)
    enqueue precision-analysis env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
        "$PY" -u "$PROJ/tools/analyze_safety_frames.py" \
        --evaluation "$ORACLE_DIST/scaling_stage_a_evaluations/59286035eef0f008/policies/fixed_pick_place_v3.scaling_width512_steps90k/evaluation.json" \
        --notes-dir "$PROJ/notes"
    ;;
schedule)
    enqueue precision-schedule env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
        "$PY" -u -m so_arm101_v2.simulation.cli run-precision-stage --stage schedule \
        "${common_stage_args[@]}"
    ;;
budget)
    [[ $# -ge 2 ]] || { echo "budget stage requires the schedule report path" >&2; exit 1; }
    enqueue precision-budget env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
        "$PY" -u -m so_arm101_v2.simulation.cli run-precision-stage --stage budget \
        "${common_stage_args[@]}" --prior-stage-report "$2"
    ;;
seeds)
    [[ $# -ge 2 ]] || { echo "seeds stage requires the budget report path" >&2; exit 1; }
    enqueue precision-seeds env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
        "$PY" -u -m so_arm101_v2.simulation.cli run-precision-stage --stage seeds \
        "${common_stage_args[@]}" --prior-stage-report "$2"
    ;;
*)
    echo "unknown stage: $STAGE (use analysis, schedule, budget <report>, seeds <report>)" >&2
    exit 1
    ;;
esac

"$TSP"
