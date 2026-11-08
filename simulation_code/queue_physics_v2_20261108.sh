#!/usr/bin/env bash
# Physics regime v2 on the SO-101 (2026-11-08, item 3 of the steps 1-3 analysis; notes/vision-rung-notebook.md 2026-11-07).
# Step 2's recipe on the SO-101 measured-servo scene (bench_pick_replace_v1_so101_servo, servo_model_20261108_so101), but
# every placement draws from bench_physics_v2, whose per-joint ranges contain both the soft servo and the stiff nominal
# actuators (teacher scan: 263/300 admitted, worst quintile fails 23%).
#   1. capture 7200 placements, suite seed 17, stride 30, --physics bench_physics_v2 --skip-failed; held-out poses
#      re-captured on the scene (suite rebased onto the SO-101)
#   2-4. CNN seeds 202 / 101 / 303 on the first 4800
#   5. ACT fine-tune on all 7200, retry config (evaluates itself at execute 90 and 30)
#   6. ACT PPO from 5, actor lr 2e-7, every training episode under its own v2 draw, RL pool rebased onto the SO-101
#   7. evaluation matrix, held-out 10 poses x 3, at execute 90 and (ACT) 30, cells:
#      nominal (bench_pick_replace_v1_so101) | servo | servo+real_like_v1 | servo+bench_physics_v2 |
#      stiff_elbow, low_friction (the 2026-11-07 diagnostics rebuilt on the SO-101 servo scene, next_round_20261108/scenes)
#      for the new policies and the SO-101 benchmark's ACT FT and CNN s202 (trained on the stiff scene)
#   8. tools/eval_matrix_report.py -> matrix.json
# Pre-registered: pass if success is at least 15/30 on BOTH SO-101 nominal and servo+real_like_v1 at execute 30 (ACT).
# One job at a time on the shared tsp queue; resumable (finished captures, trainers and eval cells are not queued again).
# Usage: [GATE_JOB=<tsp id>] queue_physics_v2_20261108.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/physics_v2_20261108
CAP=$T/capture7200
B=$E/so101_benchmark_20261106/so101
NR=$E/next_round_20261108
LADDER=$E/scaling_ladder_20260910
RETRY=$E/retry_tranche_20261004
NOMINAL=$ROOT/simulation_code/model/bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml
SERVO=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo/scene_bench_pick_replace_v1.xml
HS101=$B/cap4800/heldout_suite.path          # the ladder's 10 held-out poses rebased onto the SO-101 (benchmark capture)
RLS=$NR/rl_suites_so101/rl_suites.json
TRANCHE=physics-v2-20261108
BUDGET=$RETRY/budget.json
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
b() { $PY -c "import json,sys; print(json.load(open('$BUDGET'))$1)"; }
N=$(b "['episodes']")
# PPO env workers sized to the host (2026-11-08): an SO-101 worker holds about 0.74 GB (its 335k-face meshes in the
# software renderer, vs 0.40 GB on the SO-100), so 20 workers + the trainer need about 18 GB. Under WSL's default
# 15 GB they were OOM-killed (rehearsals 173, 251); 12 fit (about 12.5 GB). Only the parallelism changes.
MEM_GB=$(awk '/MemTotal/{print int($2/1048576)}' /proc/meminfo)
RL_ENVS=20; [ "$MEM_GB" -lt 22 ] && RL_ENVS=12
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then [ -n "$2" ] && dep=(-D "$2"); shift 2; fi   # an empty id = that job already finished earlier
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
mkdir -p "$T"
for f in "$NR/scenes/so101_servo_stiff_elbow/scene_bench_pick_replace_v1.xml" "$NR/scenes/so101_servo_low_friction/scene_bench_pick_replace_v1.xml" "$RLS" "$HS101"; do
  [ -f "$f" ] || { echo "missing $f (build the diagnostic scenes and the rebased RL pool first; see notes/bench-pick-replace-v1.md)" >&2; exit 1; }
done
# SO-101 capture workers: 10, not 12 (2026-11-08): an SO-101 worker holds about 1.1 GB and the capture's main process
# grew past 5.6 GB; both 7200 captures were OOM-killed overnight under 15 GB (04:44, 05:42).
GATE="${GATE_JOB:-}"

J1=""; grep -q '"status": "complete"' "$CAP/progress.json" 2>/dev/null || J1=$(q physv2-capture -D "$GATE" $ENVV $PY -u $ROOT/tools/capture_placements.py \
      --model $SERVO --output-dir $CAP --count 7200 --suite-seed 17 --frame-row-stride 30 --physics bench_physics_v2 --skip-failed \
      --heldout-suite $(cat $LADDER/heldout_suite.path) --heldout-from $LADDER --workers 10)
[ -z "$J1" ] && J1="$GATE"
cnn() {  # cnn <seed>
  [ -f "$T/cnn_s$1/augmentation.json" ] && { echo ""; return; }
  q "physv2-cnn-s$1" -D "$J1" $ENVV $PY -u $ROOT/tools/augmentation_run.py --model $SERVO --ladder-dir $CAP --output-dir $T/cnn_s$1 \
    --shifts 12 --episodes 4800 --encoder v2 --hidden-width 512 --steps 120000 --frame-stride 30 --no-baseline --seed $1 \
    --run-name "physics-v2-cnn-4800-s$1" --workers 8
}
J2=$(cnn 202); J3=$(cnn 101); J4=$(cnn 303)
J5=""; [ -f "$T/act_ft/result.json" ] || J5=$(q physv2-act-ft -D "$J1" $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --scene $SERVO \
      --output-dir $T/act_ft --data-dirs $CAP --run-name physics-v2-act-ft --tranche $TRANCHE $(b "['act_ft_args']"))
J6=""; [ -f "$T/rl_act_lr2e-7/result.json" ] || J6=$(q physv2-rl-act -D "$J5" $ENVV $PY -u $ROOT/tools/rl_train.py --model act \
      --checkpoint $T/act_ft/model/act.pt --scene $SERVO --physics bench_physics_v2 --rl-suites $RLS --heldout-suite $HS101 \
      --output-dir $T/rl_act_lr2e-7 --episodes $N --actor-lr 2e-7 --run-name physics-v2-act-ppo --tranche $TRANCHE $(b "['rl_act_args']") --envs $RL_ENVS)

# Evaluation matrix. A CNN checkpoint's path carries its run digest, so it is resolved when the job runs.
EVALS=()
ev() {  # ev <name> <dep job or -> <kind> <checkpoint or glob> <execute> <cells...>
  local name=$1 dep=$2 kind=$3 ckpt=$4 execute=$5; shift 5
  local d=(); [ "$dep" != "-" ] && d=(-D "$dep")
  local tag=$name; [ "$execute" != 90 ] && tag=${name}_exec$execute
  for cell in "$@"; do
    [ -f "$T/evals/${tag}__$cell/result.json" ] && continue
    local scene=$SERVO physics=none
    case $cell in
      nominal) scene=$NOMINAL ;;
      servo) ;;
      servo_*) physics=${cell#servo_} ;;
      stiff_elbow|low_friction) scene=$NR/scenes/so101_servo_$cell/scene_bench_pick_replace_v1.xml ;;
      *) echo "unknown cell $cell" >&2; exit 1 ;;
    esac
    EVALS+=("$(q "physv2-eval-$tag-$cell" "${d[@]}" bash -c "exec $ENVV $PY -u $ROOT/tools/eval_policy.py --kind $kind \
      --checkpoint \$(ls $ckpt | head -1) --scene $scene --suite $HS101 --physics $physics --execute $execute --output-dir $T/evals/${tag}__$cell")")
  done
}
CELLS=(nominal servo servo_real_like_v1 servo_bench_physics_v2 stiff_elbow low_friction)
dep() { [ -n "$1" ] && echo "$1" || echo -; }   # a trainer that already finished has no job to wait on
for s in 202 101 303; do ev "physv2_cnn_s$s" "$(dep "$(eval echo \$J$(( s == 202 ? 2 : s == 101 ? 3 : 4 )))")" vision_chunked \
  "$T/cnn_s$s/runs/shift12/models/vision_h90/*/model.pt" 90 "${CELLS[@]}"; done
for x in 90 30; do
  ev physv2_act_ft "$(dep "$J5")" vla "$T/act_ft/model/act.pt" $x "${CELLS[@]}"
  ev physv2_act_ppo "$(dep "$J6")" vla "$T/rl_act_lr2e-7/model/act_rl.pt" $x "${CELLS[@]}"
  ev so101_act_ft - vla "$B/act_ft/model/act.pt" $x "${CELLS[@]}"
done
ev so101_cnn_s202 - vision_chunked "$B/cnn_s202/runs/shift12/models/vision_h90/*/model.pt" 90 "${CELLS[@]}"
J8=$(q physv2-report $ENVV $PY -u $ROOT/tools/eval_matrix_report.py --evals $T/evals --out $T/matrix.json)
note "physics v2 SO-101 tranche (item 3) queued: capture ${J1:-done}; cnn s202 $J2, s101 $J3, s303 $J4; act-ft $J5; rl-act $J6; ${#EVALS[@]} evals ${EVALS[0]:-}..${EVALS[-1]:-}; report $J8; N=$N; rl envs $RL_ENVS; gate ${GATE:-none}"
