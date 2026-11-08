#!/usr/bin/env bash
# Step 1 again, on the SO-101 (2026-11-08, item 4 of the steps 1-3 analysis; notes/vision-rung-notebook.md 2026-11-07).
# Scene bench_pick_replace_v1_so101 (bench dynamics, nominal actuators), so it compares with the SO-101 benchmark.
#   1. perturbed-teacher capture, 7200 placements, suite seed 15, stride 30, offsets 15 mm, recentre at 20 mm (the SO-101
#      scan admits only 20 mm: 77/80; 8 and 12 mm hit the cube), held-out poses from the benchmark's SO-101 capture
#   2-4. CNN seeds 202 / 101 / 303 on the first 4800 (the CNN 4800 recipe)
#   5-6. SO-101 baseline CNN seeds 101 and 303 on the benchmark's unperturbed cap4800 (only seed 202 existed: 15/30)
#   7. ACT fine-tune on the perturbed 7200 ONLY (no unperturbed mix, so the hover timing is not ambiguous), retry config;
#      it evaluates itself at execute 90 and 30
#   8. tools/correction_report.py, CNN seeds pooled -> correction.json
# Pre-registered: CNN pooled step-250 lateral rms below the SO-101 baseline pool's and pooled success above it; ACT at least
# 25/30 at both execute 90 and 30 (SO-101 baseline 27 / 25).
# One job at a time on the shared tsp queue; resumable (finished steps are not queued again).
# Usage: [GATE_JOB=<tsp id>] queue_perturbed_so101_20261108.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/perturbed_so101_20261108
CAP=$T/capture7200
B=$E/so101_benchmark_20261106/so101
SO101=$ROOT/simulation_code/model/bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml
RETRY=$E/retry_tranche_20261004
TRANCHE=perturbed-so101-20261108
BUDGET=$RETRY/budget.json
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
b() { $PY -c "import json,sys; print(json.load(open('$BUDGET'))$1)"; }
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then [ -n "$2" ] && dep=(-D "$2"); shift 2; fi   # an empty id = that job already finished earlier
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
mkdir -p "$T"
# SO-101 capture workers: 10, not 12 (2026-11-08): an SO-101 worker holds about 1.1 GB and the capture's main process
# grew past 5.6 GB; both 7200 captures were OOM-killed overnight under 15 GB (04:44, 05:42).
GATE="${GATE_JOB:-}"

J1=""; grep -q '"status": "complete"' "$CAP/progress.json" 2>/dev/null || J1=$(q so101p-capture -D "$GATE" $ENVV $PY -u $ROOT/tools/capture_placements.py \
      --model $SO101 --output-dir $CAP --count 7200 --suite-seed 15 --frame-row-stride 30 --perturb-teacher-mm 15 --teacher-descend-mm 20 \
      --heldout-from $B/cap4800 --workers 10)
[ -z "$J1" ] && J1="$GATE"
cnn() {  # cnn <label> <ladder dir> <seed> <dep>
  [ -f "$T/$1/augmentation.json" ] && { echo ""; return; }
  q "so101p-$1" -D "$4" $ENVV $PY -u $ROOT/tools/augmentation_run.py --model $SO101 --ladder-dir $2 --output-dir $T/$1 \
    --shifts 12 --episodes 4800 --encoder v2 --hidden-width 512 --steps 120000 --frame-stride 30 --no-baseline --seed $3 \
    --run-name "perturbed-so101-$1" --workers 8
}
J2=$(cnn cnn_s202 $CAP 202 "$J1"); J3=$(cnn cnn_s101 $CAP 101 "$J1"); J4=$(cnn cnn_s303 $CAP 303 "$J1")
J5=$(cnn base_cnn_s101 $B/cap4800 101 "$GATE"); J6=$(cnn base_cnn_s303 $B/cap4800 303 "$GATE")
J7=""; [ -f "$T/act_ft/result.json" ] || J7=$(q so101p-act-ft -D "$J1" $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --scene $SO101 \
      --output-dir $T/act_ft --data-dirs $CAP --run-name perturbed-so101-act-ft --tranche $TRANCHE $(b "['act_ft_args']"))
# The report has no dependency: queued last on a one-slot queue it runs after every job above, and reports what finished.
J8=$(q so101p-report $ENVV $PY -u $ROOT/tools/correction_report.py --model $SO101 --out $T/correction.json \
      --eval cnn_s202=$T/cnn_s202/evaluations --eval cnn_s101=$T/cnn_s101/evaluations --eval cnn_s303=$T/cnn_s303/evaluations \
      --eval base_cnn_s202=$B/cnn_s202/evaluations --eval base_cnn_s101=$T/base_cnn_s101/evaluations --eval base_cnn_s303=$T/base_cnn_s303/evaluations \
      --eval act_ft_exec90=$T/act_ft/evaluations/exec90 --eval act_ft_exec30=$T/act_ft/evaluations/exec30 \
      --eval base_act_ft_exec90=$B/act_ft/evaluations/exec90 --eval base_act_ft_exec30=$B/act_ft/evaluations/exec30 \
      --pool cnn=cnn_ --pool base_cnn=base_cnn_)
note "perturbed SO-101 tranche (item 4) queued: capture ${J1:-done}; cnn s202 $J2, s101 $J3, s303 $J4; base cnn s101 $J5, s303 $J6; act-ft $J7; report $J8; gate ${GATE:-none}"
