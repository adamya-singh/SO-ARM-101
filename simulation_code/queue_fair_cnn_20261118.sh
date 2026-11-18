#!/usr/bin/env bash
# Fair CNN vs ACT comparison (2026-11-18). The best CNN so far (physics v2, seed 303) trained on 4800 of the 7200
# physics-v2 placements with the 31 mm teacher lift and could only run whole 90-step chunks; the live high-lift ACT
# trained on all 7200 placements of the 100 mm lift capture and re-plans every 30 steps. This trains the CNN recipe
# (v2 encoder, width 512, 120k steps, shift 12, frame stride 30) on the SAME capture the high-lift ACT used, all 7149
# accepted placements, seeds 202/101/303, and evaluates it in the high-lift ACT's cells at execute 90 and execute 30
# (VisionChunkedPolicy execute, new 2026-11-18). Compare against experiments/high_lift_20261112/evals.
#   simulation_code/queue_fair_cnn_20261118.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/fair_cnn_20261118
CAP=$E/high_lift_20261112/capture7200
NOMINAL=$ROOT/simulation_code/model/bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml
LIFT=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100/scene_bench_pick_replace_v1.xml
HS101=$E/so101_benchmark_20261106/so101/cap4800/heldout_suite.path
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then [ -n "$2" ] && dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
grep -q '"status": "complete"' "$CAP/progress.json" || { echo "high-lift capture incomplete: $CAP" >&2; exit 1; }
mkdir -p "$T"
N=$($PY -c "import json; print(json.load(open('$CAP/progress.json'))['episodes'])")   # 7149 accepted of 7200 drawn: the ACT's full set

cnn() {  # cnn <seed>
  [ -f "$T/cnn_s$1/augmentation.json" ] && { echo ""; return; }
  q "fair-cnn-s$1" $ENVV $PY -u $ROOT/tools/augmentation_run.py --model $LIFT --ladder-dir $CAP --output-dir $T/cnn_s$1 \
    --shifts 12 --episodes $N --encoder v2 --hidden-width 512 --steps 120000 --frame-stride 30 --no-baseline --seed $1 \
    --run-name "fair-cnn-lift100-all-s$1" --workers 8
}
declare -A J
for s in 202 101 303; do J[$s]=$(cnn $s); done

EVALS=()
for s in 202 101 303; do
  d=(); [ -n "${J[$s]}" ] && d=(-D "${J[$s]}")
  for x in 90 30; do
    tag=fair_cnn_s$s; [ $x != 90 ] && tag=${tag}_exec$x
    for cell in nominal servo servo_real_like_v1 servo_bench_physics_v2; do
      [ -f "$T/evals/${tag}__$cell/result.json" ] && continue
      scene=$LIFT; physics=none
      case $cell in nominal) scene=$NOMINAL ;; servo) ;; servo_*) physics=${cell#servo_} ;; esac
      EVALS+=("$(q "fair-eval-$tag-$cell" "${d[@]}" bash -c "exec $ENVV $PY -u $ROOT/tools/eval_policy.py --kind vision_chunked \
        --checkpoint \$(ls $T/cnn_s$s/runs/shift12/models/vision_h90/*/model.pt | head -1) --scene $scene --suite $HS101 \
        --physics $physics --execute $x --workers 8 --output-dir $T/evals/${tag}__$cell")")
    done
  done
done
R=$(q fair-cnn-report $ENVV $PY -u $ROOT/tools/eval_matrix_report.py --evals $T/evals --out $T/matrix.json)
note "fair CNN comparison queued: cnn s202 ${J[202]:-done}, s101 ${J[101]:-done}, s303 ${J[303]:-done}; ${#EVALS[@]} evals ${EVALS[0]:-}..${EVALS[-1]:-}; report $R"
