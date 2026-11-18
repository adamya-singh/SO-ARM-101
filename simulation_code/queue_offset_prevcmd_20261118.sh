#!/usr/bin/env bash
# Joint-offset randomization and previous-command input, against a tuned baseline (2026-11-18).
# Baseline = the live high-lift ACT recipe (ACT fine-tune, 40k steps, execute 30, 7200 placements, bench_physics_v2,
# 100 mm teacher lift) with its one known defect fixed: the gripper release takes 30 actions instead of 18
# (scene bench_pick_replace_v1_so101_servo_lift100_rel30), so real_like evaluations are no longer invalidated after
# a good pickup. Two by two, two seeds each (202, 101), so every arm is judged on 60 pooled rollouts per cell:
#   base      capture A (bench_physics_v2)                          the tuned baseline
#   prev      capture A, --prev-commands 3                          previous-command input
#   off       capture B (bench_physics_v3 = v2 + joint offsets 3 deg)   joint-offset randomization
#   offprev   capture B, --prev-commands 3                          both
# Captures use suite seed 17 like the high-lift capture, so A and B hold the same placements and physics draws and
# differ only in the offsets. Evaluation at execute 30 on the slow-release scene: servo (no randomization),
# real_like_v1, real_like_offset_v1 (the unmeasured-joint-zero stand-in), bench_physics_v2.
# Reference: the existing high-lift ACT under real_like_offset_v1 (how much the offsets cost a policy never trained on them).
#   simulation_code/queue_offset_prevcmd_20261118.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/offset_prevcmd_20261118
LADDER=$E/scaling_ladder_20260910
RETRY=$E/retry_tranche_20261004
LIFT=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100/scene_bench_pick_replace_v1.xml
REL30=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100_rel30/scene_bench_pick_replace_v1.xml
HS101=$E/so101_benchmark_20261106/so101/cap4800/heldout_suite.path
TRANCHE=offset-prevcmd-20261118
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
# CUDA_LAUNCH_BLOCKING: the high-lift fine-tune died twice with "CUDA error: unknown error" and ran clean with it (2026-11-13).
ACT_ARGS="$($PY -c "import json; print(json.load(open('$RETRY/budget.json'))['act_ft_args'])") --execs 30"
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then [ -n "$2" ] && dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
[ -f "$REL30" ] || { echo "missing $REL30 (prepare_bench_scene.py --teacher-release-rows 30)" >&2; exit 1; }
mkdir -p "$T"

# Reference and rehearsal first; both are short and are moved to the head of the queue.
REF=""; [ -f "$T/evals/ref_lift_act_ft_exec30__servo_real_like_offset_v1/result.json" ] || REF=$(q offprev-ref-eval $ENVV $PY -u $ROOT/tools/eval_policy.py --kind vla \
  --checkpoint $E/high_lift_20261112/act_ft/model/act.pt --scene $LIFT --suite $HS101 --physics real_like_offset_v1 --execute 30 \
  --output-dir $T/evals/ref_lift_act_ft_exec30__servo_real_like_offset_v1)
GATE=$(q offprev-rehearsal bash $ROOT/simulation_code/rehearse_offset_prevcmd_20261118.sh)
if [ $DRY = 0 ]; then tsp -u "$GATE"; [ -n "$REF" ] && tsp -u "$REF"; fi

capture() {  # capture <dir> <regime>
  grep -q '"status": "complete"' "$T/$1/progress.json" 2>/dev/null && { echo ""; return; }
  q "offprev-$1" -D "$GATE" $ENVV $PY -u $ROOT/tools/capture_placements.py --model $REL30 --output-dir $T/$1 --count 7200 --suite-seed 17 \
    --frame-row-stride 30 --physics $2 --skip-failed --heldout-suite $(cat $LADDER/heldout_suite.path) --heldout-from $LADDER --workers 10
}
CA=$(capture capture_v2 bench_physics_v2); CB=$(capture capture_v3 bench_physics_v3)
dep() { [ -n "$1" ] && echo "$1" || echo "$GATE"; }

EVALS=(); TRAIN=()
for seed in 202 101; do
  for arm in base prev off offprev; do
    cap=capture_v2; cj=$CA; case $arm in off*) cap=capture_v3; cj=$CB ;; esac
    extra=""; case $arm in *prev) extra="--prev-commands 3" ;; esac
    name=${arm}_s$seed
    J=""; [ -f "$T/$name/result.json" ] || J=$(q "offprev-act-$name" -D "$(dep "$cj")" env CUDA_LAUNCH_BLOCKING=1 ${ENVV#env } $PY -u $ROOT/tools/vla_finetune.py \
      --model act --scene $REL30 --output-dir $T/$name --data-dirs $T/$cap --seed $seed $extra --run-name "offprev-$name" --tranche $TRANCHE $ACT_ARGS)
    TRAIN+=("${J:-done}")
    for cell in servo servo_real_like_v1 servo_real_like_offset_v1 servo_bench_physics_v2; do
      [ -f "$T/evals/${name}_exec30__$cell/result.json" ] && continue
      physics=none; case $cell in servo_*) physics=${cell#servo_} ;; esac
      EVALS+=("$(q "offprev-eval-$name-$cell" -D "$(dep "$J")" $ENVV $PY -u $ROOT/tools/eval_policy.py --kind vla --checkpoint $T/$name/model/act.pt \
        --scene $REL30 --suite $HS101 --physics $physics --execute 30 --output-dir $T/evals/${name}_exec30__$cell)")
    done
  done
done
R=$(q offprev-report $ENVV $PY -u $ROOT/tools/eval_matrix_report.py --evals $T/evals --out $T/matrix.json)
note "offset/prev-command tranche queued: reference eval ${REF:-done}; rehearsal gate $GATE; captures v2 ${CA:-done}, v3 ${CB:-done}; ACT ${TRAIN[*]}; ${#EVALS[@]} evals ${EVALS[0]:-}..${EVALS[-1]:-}; report $R"
