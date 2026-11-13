#!/usr/bin/env bash
# High-lift teacher on the SO-101 servo scene (2026-11-12). On the real arm the physics v2 ACT policy's 31 mm lift reads
# as a wrist curl (12-16 mm rise; notes/vision-rung-notebook.md, "Live physics v2 ACT"), so the user asked for a retrain
# that lifts much higher. Scene bench_pick_replace_v1_so101_servo_lift100 = the servo scene + teacher_lift_m 0.10 (same
# XML; high-lift stage schedule, lift pose solved loosely). Teacher on the 10 held-out poses: 9/10 clean, 27/30
# real_like_v1, 25/30 bench_physics_v2 (31 mm teacher: 10/10, 27/30, 25/30).
#   1. capture 7200 placements, suite seed 17, stride 30, --physics bench_physics_v2 --skip-failed (the item-3 recipe)
#   2. ACT fine-tune on all of them, retry config (evaluates itself at execute 90 and 30)
#   3. held-out cells at execute 30 and 90: nominal | servo (lift scene) | servo+real_like_v1 | servo+bench_physics_v2
#   4. tools/eval_matrix_report.py -> matrix.json
# Judge against physics v2 ACT at execute 30: 24 nominal, 19 real_like. No CNN, no PPO (dropped 2026-11-09).
# One job at a time on the shared tsp queue; resumable. Usage: queue_high_lift_20261112.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/high_lift_20261112
CAP=$T/capture7200
LADDER=$E/scaling_ladder_20260910
RETRY=$E/retry_tranche_20261004
NOMINAL=$ROOT/simulation_code/model/bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml
LIFT=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100/scene_bench_pick_replace_v1.xml
HS101=$E/so101_benchmark_20261106/so101/cap4800/heldout_suite.path
TRANCHE=high-lift-20261112
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
ACT_ARGS=$($PY -c "import json; print(json.load(open('$RETRY/budget.json'))['act_ft_args'])")
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then [ -n "$2" ] && dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
mkdir -p "$T"
[ -f "$LIFT" ] || { echo "missing $LIFT (prepare_bench_scene.py --teacher-lift-m 0.10)" >&2; exit 1; }
J1=""; grep -q '"status": "complete"' "$CAP/progress.json" 2>/dev/null || J1=$(q lift-capture -D "" $ENVV $PY -u $ROOT/tools/capture_placements.py \
      --model $LIFT --output-dir $CAP --count 7200 --suite-seed 17 --frame-row-stride 30 --physics bench_physics_v2 --skip-failed \
      --heldout-suite $(cat $LADDER/heldout_suite.path) --heldout-from $LADDER --workers 10)
J2=""; [ -f "$T/act_ft/result.json" ] || J2=$(q lift-act-ft -D "$J1" $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --scene $LIFT \
      --output-dir $T/act_ft --data-dirs $CAP --run-name high-lift-act-ft --tranche $TRANCHE $ACT_ARGS)
EVALS=()
for x in 30 90; do
  tag=lift_act_ft; [ $x != 90 ] && tag=lift_act_ft_exec$x
  for cell in nominal servo servo_real_like_v1 servo_bench_physics_v2; do
    [ -f "$T/evals/${tag}__$cell/result.json" ] && continue
    scene=$LIFT; physics=none
    case $cell in nominal) scene=$NOMINAL ;; servo) ;; servo_*) physics=${cell#servo_} ;; esac
    EVALS+=("$(q "lift-eval-$tag-$cell" -D "$J2" $ENVV $PY -u $ROOT/tools/eval_policy.py --kind vla --checkpoint $T/act_ft/model/act.pt \
      --scene $scene --suite $HS101 --physics $physics --execute $x --output-dir $T/evals/${tag}__$cell)")
  done
done
J3=$(q lift-report $ENVV $PY -u $ROOT/tools/eval_matrix_report.py --evals $T/evals --out $T/matrix.json)
note "high-lift tranche queued: capture ${J1:-done}; act-ft ${J2:-done}; ${#EVALS[@]} evals ${EVALS[0]:-}..${EVALS[-1]:-}; report $J3"
