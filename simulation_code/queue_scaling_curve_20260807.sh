#!/usr/bin/env bash
# Data-scaling curve (2026-08-07): held-out generalization vs training-set size.
# Frozen benchmark: suites/10e6a56077e91ce4 (random_pick_place_v3_seed8_n10_9d8c330a, 30 rollouts).
# Tiers: n50(seed9) -> n100(seed10) -> n200(seed11) -> vision-60k probe on n200 -> n400(seed12, bonus).
# Ascending order so the curve fills smallest-first if the 15h window closes early.
set -u
ROOT=/home/win10ubuntu/dev/robotic-arm/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl"
MODEL=$ROOT/simulation_code/model/menagerie_so_arm100/scene_v2.xml
HELDOUT=$ROOT/artifacts/so_arm101_v2/suites/10e6a56077e91ce4/suite.json
SUMMARY=$ROOT/artifacts/so_arm101_v2/randomized_scaling/SWEEP_LOG.txt
mkdir -p "$(dirname "$SUMMARY")"
note() { echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY"; }

run_tier() {
  local SEED=$1 COUNT=$2 TIER=$3
  note "=== tier $TIER: generating suite (seed $SEED, $COUNT scenarios, full-episode screen)"
  local GEN
  GEN=$($ENVV $PY $ROOT/tools/generate_random_suite.py \
    --generator-seed "$SEED" --count "$COUNT" --repeats 1 \
    --screen-model "$MODEL") || { note "tier $TIER: generation FAILED"; return 1; }
  local SUITE_PATH SUITE_ID
  SUITE_PATH=$ROOT/$(echo "$GEN" | sed -n 1p)
  SUITE_ID=$(echo "$GEN" | sed -n 2p | sed 's/suite_id=\([^ ]*\).*/\1/')
  note "tier $TIER: suite $SUITE_ID at $SUITE_PATH"

  local J
  J=$(tsp -L "scale-$TIER-preflight" $ENVV $PY -u -m so_arm101_v2.simulation.cli preflight \
    --mujoco-model "$MODEL" --output-dir $ROOT/artifacts/so_arm101_v2/simulation \
    --suite fixed_pick_place_v3 --suite-path "$SUITE_PATH" --no-video)
  tsp -w "$J"
  grep -q "environment_proven=true" "$(tsp -o "$J")" \
    || { note "tier $TIER: preflight NOT proven — skipping tier"; tail -3 "$(tsp -o "$J")" >> "$SUMMARY"; return 1; }
  note "tier $TIER: preflight proven"

  J=$(tsp -L "scale-$TIER-capture" $ENVV $PY -u -m so_arm101_v2.simulation.cli capture-oracle \
    --mujoco-model "$MODEL" --output-dir $ROOT/artifacts/so_arm101_v2/oracle_distillation \
    --suite fixed_pick_place_v3 --suite-path "$SUITE_PATH" --scenario all \
    --teacher-horizon 480 --store-frames --skip-failed-scenarios --no-video \
    --preflight-report $ROOT/artifacts/so_arm101_v2/simulation/preflight/$SUITE_ID/evaluation.json)
  tsp -w "$J"
  local MANIFEST
  MANIFEST=$(grep -m1 -o '/home[^ ]*manifest\.json' "$(tsp -o "$J")")
  [ -n "$MANIFEST" ] && [ -f "$MANIFEST" ] \
    || { note "tier $TIER: capture FAILED"; tail -3 "$(tsp -o "$J")" >> "$SUMMARY"; return 1; }
  note "tier $TIER: capture done -> $MANIFEST"
  echo "$MANIFEST" > "$ROOT/artifacts/so_arm101_v2/randomized_scaling/$TIER.manifest.path"

  J=$(tsp -L "scale-$TIER-train" $ENVV $PY -u $ROOT/tools/run_randomized_v1.py \
    --capture-manifest "$MANIFEST" --heldout-suite-path "$HELDOUT" \
    --mujoco-model "$MODEL" \
    --output-dir $ROOT/artifacts/so_arm101_v2/randomized_scaling/$TIER)
  tsp -w "$J"
  grep -h -E "^(SUMMARY|STATE TRAINED|VISION TRAINED)" "$(tsp -o "$J")" \
    | sed "s/^/[$TIER] /" | tee -a "$SUMMARY"
  grep -q "^SUMMARY" "$(tsp -o "$J")" || { note "tier $TIER: train/eval FAILED"; return 1; }
  note "tier $TIER: complete"
}

run_tier 9 50 n50
run_tier 10 100 n100
run_tier 11 200 n200

# Vision compute probe: same n200 data, 60k steps (3x), vision only.
if [ -f "$ROOT/artifacts/so_arm101_v2/randomized_scaling/n200.manifest.path" ]; then
  MANIFEST=$(cat "$ROOT/artifacts/so_arm101_v2/randomized_scaling/n200.manifest.path")
  note "=== probe: vision 60k steps on n200"
  J=$(tsp -L scale-probe-vision60k $ENVV $PY -u $ROOT/tools/run_randomized_v1.py \
    --capture-manifest "$MANIFEST" --heldout-suite-path "$HELDOUT" \
    --mujoco-model "$MODEL" --skip-state --vision-max-steps 60000 \
    --output-dir $ROOT/artifacts/so_arm101_v2/randomized_scaling/n200_vision60k)
  tsp -w "$J"
  grep -h -E "^(SUMMARY|VISION TRAINED)" "$(tsp -o "$J")" | sed 's/^/[probe60k] /' | tee -a "$SUMMARY"
else
  note "probe skipped: no n200 manifest"
fi

# Bonus tier if the window allows.
run_tier 12 400 n400

note "SCALING SWEEP FINISHED"
