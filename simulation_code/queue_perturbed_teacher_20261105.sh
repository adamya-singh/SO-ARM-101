#!/usr/bin/env bash
# Perturbed-teacher tranche (2026-11-05, notes/handoff-20261105.md step 1; manifest tools/tranche_20261105.json).
#   1. capture 4800 perturbed placements (suite seed 15, stride 30; offset uniform in a 15 mm disk, recentre at 20 mm)
#   2-4. CNN from scratch on it, seeds 202 / 101 / 303 (the 4800 shift-12 recipe: v2, width 512, 120k, frame stride 30)
#   5. ACT fine-tune, retry-tranche config, on perturbed 4800 + unperturbed ladder 2400 (baseline: same recipe, 20/30)
#   6. ACT PPO from 5, actor lr 2e-7 (baseline 24/30)
#   7. correction report: step 179 vs 250 lateral rms for every run and the baselines, CNN seeds pooled
# One job at a time on the shared tsp queue (single slot); 2-5 need 1 (tsp -D), 6 needs 5. 7 runs last regardless.
# Usage: queue_perturbed_teacher_20261105.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
# The lerobot fork moved with the repo (~/dev/robotic-arm -> ~/projects); the conda envs' editable .pth files still name
# the old path, so the fork goes on PYTHONPATH here.
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/perturbed_teacher_20261105
CAP=$T/capture4800
RETRY=$E/retry_tranche_20261004
BUDGET=$RETRY/budget.json
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
b() { $PY -c "import json,sys; print(json.load(open('$BUDGET'))$1)"; }
N=$(b "['episodes']")
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
mkdir -p "$T"

J1=$(q perturbed-capture $ENVV $PY -u $ROOT/tools/capture_placements.py --output-dir $CAP --count 4800 --suite-seed 15 \
      --frame-row-stride 30 --perturb-teacher-mm 15 --teacher-descend-mm 20 --heldout-from $E/scaling_ladder_20260910 --workers 8)
vision() {  # vision <seed>
  q "perturbed-vision-s$1" -D "$J1" $ENVV $PY -u $ROOT/tools/augmentation_run.py --ladder-dir $CAP --output-dir $T/vision_s$1 \
    --shifts 12 --episodes 4800 --encoder v2 --hidden-width 512 --steps 120000 --frame-stride 30 --no-baseline --seed $1 \
    --run-name "perturbed-teacher-cnn-4800-s$1" --workers 8
}
J2=$(vision 202)
J3=$(vision 101)
J4=$(vision 303)
J5=$(q perturbed-act-ft -D "$J1" $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --output-dir $T/act_ft \
      --data-dirs $CAP $E/scaling_ladder_20260910 --run-name perturbed-teacher-act-ft --tranche perturbed-teacher-20261105 $(b "['act_ft_args']"))
J6=$(q perturbed-rl-act -D "$J5" $ENVV $PY -u $ROOT/tools/rl_train.py --model act --checkpoint $T/act_ft/model/act.pt \
      --output-dir $T/rl_act_lr2e-7 --episodes $N --actor-lr 2e-7 --run-name perturbed-teacher-act-ppo --tranche perturbed-teacher-20261105 $(b "['rl_act_args']"))
J7=$(q perturbed-report $ENVV $PY -u $ROOT/tools/correction_report.py --out $T/correction.json \
      --eval cnn_s202=$T/vision_s202/evaluations --eval cnn_s101=$T/vision_s101/evaluations --eval cnn_s303=$T/vision_s303/evaluations \
      --eval act_ft=$T/act_ft/evaluations/exec90 --eval act_ppo=$T/rl_act_lr2e-7/evaluations/exec90 \
      --eval base_cnn2400_s202=$E/augmentation_shift12_2400_20260912/evaluations \
      --eval base_cnn2400_s101=$E/augmentation_shift12_2400_seed101_20260912/evaluations \
      --eval base_cnn2400_s303=$E/augmentation_shift12_2400_seed303_20260912/evaluations \
      --eval base_cnn4800_s202=$E/augmentation_shift12_4800_20260912/evaluations/shift12 \
      --eval base_act_ft=$RETRY/act_ft/evaluations/exec90 --eval base_act_ppo=$RETRY/rl_act_lr2e-7/evaluations/exec90 \
      --pool cnn=cnn_ --pool base_cnn2400=base_cnn2400)
note "perturbed-teacher tranche queued: capture $J1; cnn s202 $J2, s101 $J3, s303 $J4 (-D $J1); act-ft $J5 (-D $J1); rl-act $J6 (-D $J5); report $J7; N=$N"
