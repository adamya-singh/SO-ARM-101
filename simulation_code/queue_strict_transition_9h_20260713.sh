#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TSP="/home/win10ubuntu/.local/bin/tsp"
PYTHON="/home/win10ubuntu/miniforge3/envs/lerobot/bin/python"
OUTPUT="$ROOT/outputs/train/act_strict_transition_9h_20260713"
RESET_BANK="$ROOT/artifacts/strict_transition/reset_bank.json"

mkdir -p "$OUTPUT"
exec "$TSP" -L act_strict_transition_9h_20260713 \
  env MUJOCO_GL=egl PYTHONUNBUFFERED=1 \
  "$PYTHON" "$ROOT/simulation_code/run_strict_transition_9h_pipeline.py" \
  --output-dir "$OUTPUT" \
  --reset-bank "$RESET_BANK" \
  --training-seconds 32400
