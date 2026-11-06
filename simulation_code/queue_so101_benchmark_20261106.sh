#!/usr/bin/env bash
# SO-101 benchmark (2026-11-06, notes/handoff-20261105.md step 3; manifest tools/tranche_20261106.json).
# For each Menagerie SO-101 scene -- bench_pick_replace_v1_so101 (the bench's dynamics) and
# bench_pick_replace_v1_so101_menagerie (Menagerie's as shipped) -- the SO-100 baselines' exact recipe:
#   1. capture 4800 placements (suite seed 14, stride 30), after re-capturing the ladder's 10 held-out poses on the scene
#   2. capture 2400 placements (suite seed 12, stride 18; the ACT baseline trained on both captures)
#   3. ACT fine-tune, retry-tranche act_ft_args, on both captures (SO-100: 20/30 at execute 90)
#   4. CNN seed 202, the 4800 shift-12 recipe (SO-100: 6/30)
# Each trainer evaluates on its own scene's held-out poses. Then 5. tools/so101_benchmark_report.py.
# One job at a time on the shared tsp queue.
# Usage: [GATE_JOB=<tsp id>] queue_so101_benchmark_20261106.sh [--dry-run]
#   GATE_JOB: a tsp job (the last rehearsal) both scenes' first capture depends on, so a failed rehearsal skips the benchmark.
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
# The lerobot fork moved with the repo; the conda envs' editable .pth files still name the old path.
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/so101_benchmark_20261106
LADDER=$E/scaling_ladder_20260910
RETRY=$E/retry_tranche_20261004
TRANCHE=so101-benchmark-20261106
BUDGET=$RETRY/budget.json
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
b() { $PY -c "import json,sys; print(json.load(open('$BUDGET'))$1)"; }
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
mkdir -p "$T"
HSUITE=$(cat $LADDER/heldout_suite.path)
JOBS=()
for S in so101 so101_menagerie; do
  SCENE=$ROOT/simulation_code/model/bench_pick_replace_v1_$S/scene_bench_pick_replace_v1.xml
  D=$T/$S
  GATE=(); [ -n "${GATE_JOB:-}" ] && GATE=(-D "$GATE_JOB")
  C1=$(q "so101-$S-capture4800" "${GATE[@]}" $ENVV $PY -u $ROOT/tools/capture_placements.py --model $SCENE --output-dir $D/cap4800 --count 4800 \
        --suite-seed 14 --frame-row-stride 30 --heldout-suite $HSUITE --heldout-from $LADDER --workers 12)
  C2=$(q "so101-$S-capture2400" -D "$C1" $ENVV $PY -u $ROOT/tools/capture_placements.py --model $SCENE --output-dir $D/cap2400 --count 2400 \
        --suite-seed 12 --frame-row-stride 18 --heldout-from $D/cap4800 --workers 12)
  A=$(q "so101-$S-act-ft" -D "$C2" $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --scene $SCENE --output-dir $D/act_ft \
        --data-dirs $D/cap4800 $D/cap2400 --run-name "so101-bench-act-ft-$S" --tranche $TRANCHE $(b "['act_ft_args']"))
  N=$(q "so101-$S-cnn-s202" -D "$C1" $ENVV $PY -u $ROOT/tools/augmentation_run.py --model $SCENE --ladder-dir $D/cap4800 --output-dir $D/cnn_s202 \
        --shifts 12 --episodes 4800 --encoder v2 --hidden-width 512 --steps 120000 --frame-stride 30 --no-baseline --seed 202 \
        --run-name "so101-bench-cnn-4800-s202-$S" --workers 8)
  JOBS+=("$S: capture4800 $C1, capture2400 $C2, act-ft $A, cnn $N")
done
R=$(q so101-report $ENVV $PY -u $ROOT/tools/so101_benchmark_report.py --root $T --out $T/benchmark.json)
note "SO-101 benchmark queued: ${JOBS[*]}; report $R"
