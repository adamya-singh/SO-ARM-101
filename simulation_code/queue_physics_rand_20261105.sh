#!/usr/bin/env bash
# Physics-randomization tranche (2026-11-05, notes/handoff-20261105.md step 2; manifest tools/tranche_20261105b.json).
#   1. capture on the measured-servo scene: the ladder's 10 held-out poses (scene's own physics), then 7200 placements,
#      suite seed 16, stride 30, every placement under its own bench_physics_v1 draw (compensated teacher)
#   2-4. CNN from scratch, seeds 202 / 101 / 303, on the first 4800 (the CNN 4800 baseline's recipe)
#   5. ACT fine-tune on all 7200 (the ACT baseline's data size), retry-tranche config
#   6. ACT PPO from 5, actor lr 2e-7, every training episode under its own draw
#   7. evaluation matrix: held-out 10 poses x 3, execute 90, cells nominal | servo | servo+real_like_v1 | servo+bench_physics_v1,
#      for the new policies, the step-1 (perturbed-teacher) policies and the baselines (CNN 4800, ACT FT, ACT PPO)
#   8. report: tools/eval_matrix_report.py -> matrix.json
# One job at a time on the shared tsp queue. Trainers need 1 (tsp -D); each eval needs its own checkpoint's job.
# Usage: queue_physics_rand_20261105.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
# The lerobot fork moved with the repo (~/dev/robotic-arm -> ~/projects); the conda envs' editable .pth files still name
# the old path, so the fork goes on PYTHONPATH here.
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/physics_rand_20261105
CAP=$T/capture7200
P1=$E/perturbed_teacher_20261105
RETRY=$E/retry_tranche_20261004
NOMINAL=$ROOT/simulation_code/model/bench_pick_replace_v1/scene_bench_pick_replace_v1.xml
SERVO=$ROOT/simulation_code/model/bench_pick_replace_v1_servo/scene_bench_pick_replace_v1.xml
TRANCHE=physics-randomization-20261105
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
HSUITE=$(cat $E/scaling_ladder_20260910/heldout_suite.path)

J1=$(q physics-capture $ENVV $PY -u $ROOT/tools/capture_placements.py --model $SERVO --output-dir $CAP --count 7200 --suite-seed 16 \
      --frame-row-stride 30 --physics bench_physics_v1 --heldout-suite $HSUITE --heldout-from $E/scaling_ladder_20260910 --workers 12)
cnn() {  # cnn <seed>
  q "physics-cnn-s$1" -D "$J1" $ENVV $PY -u $ROOT/tools/augmentation_run.py --model $SERVO --ladder-dir $CAP --output-dir $T/cnn_s$1 \
    --shifts 12 --episodes 4800 --encoder v2 --hidden-width 512 --steps 120000 --frame-stride 30 --no-baseline --seed $1 \
    --run-name "physics-rand-cnn-4800-s$1" --workers 8
}
J2=$(cnn 202); J3=$(cnn 101); J4=$(cnn 303)
J5=$(q physics-act-ft -D "$J1" $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --scene $SERVO --output-dir $T/act_ft \
      --data-dirs $CAP --run-name physics-rand-act-ft --tranche $TRANCHE $(b "['act_ft_args']"))
J6=$(q physics-rl-act -D "$J5" $ENVV $PY -u $ROOT/tools/rl_train.py --model act --checkpoint $T/act_ft/model/act.pt --scene $SERVO \
      --physics bench_physics_v1 --output-dir $T/rl_act_lr2e-7 --episodes $N --actor-lr 2e-7 --run-name physics-rand-act-ppo \
      --tranche $TRANCHE $(b "['rl_act_args']"))

# Evaluation matrix. A CNN checkpoint's path carries its run digest, so it is resolved when the job runs.
EVALS=()
ev() {  # ev <name> <dep job or -> <kind> <checkpoint or glob> <cells...>
  local name=$1 dep=$2 kind=$3 ckpt=$4; shift 4
  local d=(); [ "$dep" != "-" ] && d=(-D "$dep")
  for cell in "$@"; do
    local scene=$SERVO physics=${cell#servo_}
    [ "$cell" = "nominal" ] && { scene=$NOMINAL; physics=none; }
    [ "$cell" = "servo" ] && physics=none
    EVALS+=("$(q "physics-eval-$name-$cell" "${d[@]}" bash -c "exec $ENVV $PY -u $ROOT/tools/eval_policy.py --kind $kind \
      --checkpoint \$(ls $ckpt | head -1) --scene $scene --physics $physics --output-dir $T/evals/${name}__$cell")")
  done
}
ALL=(nominal servo servo_real_like_v1 servo_bench_physics_v1)
SERVO_CELLS=(servo servo_real_like_v1 servo_bench_physics_v1)
for s in 202 101 303; do ev "physics_cnn_s$s" "$(eval echo \$J$(( s == 202 ? 2 : s == 101 ? 3 : 4 )))" vision_chunked "$T/cnn_s$s/runs/shift12/models/vision_h90/*/model.pt" "${ALL[@]}"; done
ev physics_act_ft "$J5" vla "$T/act_ft/model/act.pt" "${ALL[@]}"
ev physics_act_ppo "$J6" vla "$T/rl_act_lr2e-7/model/act_rl.pt" "${ALL[@]}"
ev base_cnn4800_s202 - vision_chunked "$E/augmentation_shift12_4800_20260912/runs/shift12/models/vision_h90/*/model.pt" "${SERVO_CELLS[@]}"
ev base_act_ft - vla "$RETRY/act_ft/model/act.pt" "${SERVO_CELLS[@]}"
ev base_act_ppo - vla "$RETRY/rl_act_lr2e-7/model/act_rl.pt" "${SERVO_CELLS[@]}"
for s in 202 101 303; do ev "perturbed_cnn_s$s" - vision_chunked "$P1/vision_s$s/runs/shift12/models/vision_h90/*/model.pt" "${SERVO_CELLS[@]}"; done
ev perturbed_act_ft - vla "$P1/act_ft/model/act.pt" "${SERVO_CELLS[@]}"
ev perturbed_act_ppo - vla "$P1/rl_act_lr2e-7/model/act_rl.pt" "${SERVO_CELLS[@]}"
J8=$(q physics-report $ENVV $PY -u $ROOT/tools/eval_matrix_report.py --evals $T/evals --out $T/matrix.json)
note "physics-randomization tranche queued: capture $J1; cnn s202 $J2, s101 $J3, s303 $J4 (-D $J1); act-ft $J5 (-D $J1); rl-act $J6 (-D $J5); ${#EVALS[@]} evals ${EVALS[0]}..${EVALS[-1]}; report $J8; N=$N"
