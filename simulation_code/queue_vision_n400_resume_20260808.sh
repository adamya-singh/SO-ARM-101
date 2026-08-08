#!/usr/bin/env bash
# Resume vision n400 seeds 202/303 after the deterministic-prefetch change
# (bitwise-identical training, parallel disk reads). Same driver args as
# queue_vision_n400_20260807.sh — digests unchanged.
set -u
ROOT=/home/win10ubuntu/dev/robotic-arm/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
MODEL=$ROOT/simulation_code/model/menagerie_so_arm100/scene_v2.xml
HELDOUT=$ROOT/artifacts/so_arm101_v2/suites/10e6a56077e91ce4/suite.json
MANIFEST=$(cat "$ROOT/artifacts/so_arm101_v2/randomized_scaling/n400.manifest.path")
LOG=$ROOT/artifacts/so_arm101_v2/randomized_scaling/VISION_N400_LOG.txt
note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

note "=== resume: seeds 202/303 with deterministic prefetch"
declare -A JOBS
for SEED in 202 303; do
  JOBS[$SEED]=$(tsp -L "vision-n400-s$SEED" env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl \
    WANDB_SILENT=true WANDB_RUN_GROUP=vision-n400-120k-20260807 WANDB_NAME=vision-n400-s$SEED-120k \
    $PY -u $ROOT/tools/run_randomized_v1.py \
    --capture-manifest "$MANIFEST" --heldout-suite-path "$HELDOUT" \
    --mujoco-model "$MODEL" --skip-state --vision-max-steps 120000 --seed "$SEED" \
    --wandb-project so-arm101-v2-scaling \
    --output-dir $ROOT/artifacts/so_arm101_v2/randomized_scaling/n400_vision120k)
  note "queued seed $SEED as tsp job ${JOBS[$SEED]}"
done

for SEED in 202 303; do
  J=${JOBS[$SEED]}
  ( while true; do
      OUT=$(tsp -o "$J" 2>/dev/null) || { sleep 5; continue; }
      [ -n "$OUT" ] && [ -f "$OUT" ] && grep -m1 "^WANDB_URL" "$OUT" >/dev/null 2>&1 && {
        echo "[seed $SEED] $(grep -m1 '^WANDB_URL' "$OUT")" | tee -a "$LOG"; break; }
      STATE=$(tsp -s "$J" 2>/dev/null)
      [ "$STATE" = "finished" ] && break
      sleep 10
    done ) &
  URLWATCH=$!
  tsp -w "$J"
  wait "$URLWATCH" 2>/dev/null
  grep -h -E "^(SUMMARY|VISION TRAINED)" "$(tsp -o "$J")" | sed "s/^/[seed $SEED] /" | tee -a "$LOG"
  grep -q "^SUMMARY" "$(tsp -o "$J")" || {
    note "seed $SEED FAILED"; tail -6 "$(tsp -o "$J")" >> "$LOG"; }
done
note "VISION N400 SWEEP FINISHED (prefetch)"
