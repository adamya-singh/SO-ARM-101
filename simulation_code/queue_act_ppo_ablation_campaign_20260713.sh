#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PY="/home/win10ubuntu/miniforge3/envs/lerobot/bin/python"
TSP="/home/win10ubuntu/.local/bin/tsp"
RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/outputs/train/act_ppo_original_model_open_ended_20260713}"
MODEL="${SCRIPT_DIR}/model/so101_new_calib.xml"
EXPECTED_SHA="ac5254b0283e342ba2499c0b560f47b958deb833f291092728bad7c278f5d167"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -x "$TSP" ]] || { echo "Missing Task Spooler: $TSP" >&2; exit 1; }
[[ "$(sha256sum "$MODEL" | awk '{print $1}')" == "$EXPECTED_SHA" ]] || {
    echo "Refusing launch: production SO-101 XML is not the restored original." >&2
    exit 1
}

mkdir -p "$RUN_DIR"
"$TSP" -S 1

command=(
    env MUJOCO_GL=egl PYTHONUNBUFFERED=1
    "$PY" -u "$SCRIPT_DIR/run_act_ppo_ablation_campaign.py"
    --output-dir "$RUN_DIR"
)

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    printf '%q ' "${command[@]}"
    printf '\n'
    exit 0
fi

"$TSP" -L act_ppo_original_model_open_ended_20260713 "${command[@]}"
