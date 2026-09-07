#!/usr/bin/env bash
# Bench lift-and-replace (2026-09-06): enqueue the gated pipeline on the persistent tsp queue.
# Usage: queue_bench_pipeline.sh <verification.json> <experiment-output-dir> [--stop-after preflight|screen|capture|train]
#        queue_bench_pipeline.sh --rehearsal <experiment-output-dir under .../rehearsal/> [--stop-after ...]
# The pipeline itself orders its gates (camera review -> certification -> screened suites -> capture -> train -> eval)
# and refuses to continue past a failed gate, so one queue job is the unit of dependency. The job's tsp output file
# is the launch log; progress.json in the experiment dir is what tools/bench_health_check.py reads.
set -u
ROOT=/home/win10ubuntu/dev/robotic-arm/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl"
MODEL=$ROOT/simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
mkdir -p "$(dirname "$LOG")"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

if [ "${1:-}" = "--rehearsal" ]; then
  OUT=$2; shift 2
  J=$(tsp -L bench-rehearsal $ENVV $PY -u $ROOT/tools/run_bench_pipeline.py --model "$MODEL" --output-dir "$OUT" --rehearsal "$@")
  note "queued REHEARSAL job $J -> $OUT (tsp -o $J for the log)"
else
  VERIFICATION=$1; OUT=$2; shift 2
  [ -f "$VERIFICATION" ] || { note "verification record not found: $VERIFICATION"; exit 1; }
  J=$(tsp -L bench-pipeline $ENVV $PY -u $ROOT/tools/run_bench_pipeline.py --model "$MODEL" --verification "$VERIFICATION" --output-dir "$OUT" "$@")
  note "queued bench pipeline job $J -> $OUT (verification $VERIFICATION)"
  note "health check: $PY $ROOT/tools/bench_health_check.py --experiment-dir $OUT  (first check 20 min after training_clock.json appears)"
fi
echo "$J"
