#!/usr/bin/env bash
# n400 tier retry after streaming-publish fix (suite + preflight already proven).
set -u
ROOT=/home/win10ubuntu/dev/robotic-arm/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl"
MODEL=$ROOT/simulation_code/model/menagerie_so_arm100/scene_v2.xml
SUITE_PATH=$ROOT/artifacts/so_arm101_v2/suites/c9c2b540e09c9188/suite.json
SUITE_ID=random_pick_place_v3_seed12_n400_e3125a69
HELDOUT=$ROOT/artifacts/so_arm101_v2/suites/10e6a56077e91ce4/suite.json
SUMMARY=$ROOT/artifacts/so_arm101_v2/randomized_scaling/SWEEP_LOG.txt
note() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

note "=== tier n400 RETRY (streaming frames publish)"
J=$(tsp -L scale-n400-capture-retry $ENVV $PY -u -m so_arm101_v2.simulation.cli capture-oracle \
  --mujoco-model "$MODEL" --output-dir $ROOT/artifacts/so_arm101_v2/oracle_distillation \
  --suite fixed_pick_place_v3 --suite-path "$SUITE_PATH" --scenario all \
  --teacher-horizon 480 --store-frames --skip-failed-scenarios --no-video \
  --preflight-report $ROOT/artifacts/so_arm101_v2/simulation/preflight/$SUITE_ID/evaluation.json)
tsp -w "$J"
MANIFEST=$(grep -m1 -o '/home[^ ]*manifest\.json' "$(tsp -o "$J")")
[ -n "$MANIFEST" ] && [ -f "$MANIFEST" ] \
  || { note "tier n400: capture FAILED (retry)"; tail -4 "$(tsp -o "$J")" >> "$SUMMARY"; exit 1; }
note "tier n400: capture done -> $MANIFEST"
echo "$MANIFEST" > "$ROOT/artifacts/so_arm101_v2/randomized_scaling/n400.manifest.path"

J=$(tsp -L scale-n400-train $ENVV $PY -u $ROOT/tools/run_randomized_v1.py \
  --capture-manifest "$MANIFEST" --heldout-suite-path "$HELDOUT" \
  --mujoco-model "$MODEL" \
  --output-dir $ROOT/artifacts/so_arm101_v2/randomized_scaling/n400)
tsp -w "$J"
grep -h -E "^(SUMMARY|STATE TRAINED|VISION TRAINED)" "$(tsp -o "$J")" | sed 's/^/[n400] /' | tee -a "$SUMMARY"
grep -q "^SUMMARY" "$(tsp -o "$J")" || { note "tier n400: train/eval FAILED"; exit 1; }
note "tier n400: complete"
note "SCALING SWEEP FINISHED (with n400)"
