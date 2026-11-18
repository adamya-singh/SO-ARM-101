#!/usr/bin/env bash
# Joint-offset randomization and previous-command input, against a TUNED baseline (2026-11-18).
#
# Baseline recipe = the live high-lift ACT (ACT fine-tune, execute 30, 7200 placements, bench_physics_v2, 100 mm
# teacher lift) on the slow-release scene (bench_pick_replace_v1_so101_servo_lift100_rel30: the gripper opens over 30
# actions instead of 18, so real_like evaluations are no longer invalidated after a good pickup), with its optimizer
# settings chosen by a two-stage sweep on the baseline data (user, 2026-11-18):
#   stage 1  8 settings x 15k steps, seed 202: learning rate {3e-5, 1e-4, 3e-4}, backbone rate {1e-5, 3e-5, 1e-4},
#            KL weight {10, 1}; ranked by held-out loss at chunk start 90 (tools/sweep_select.py) -> sweep/winner_stage1.json
#   stage 2  the stage-1 winner and the current settings (1e-4, 1e-5, 10), each 40k steps, seed 202; the lower
#            held-out loss is the tuned baseline -> sweep/tuned.json. (If the winner IS the current settings, one run.)
#   stage 3  (this script with --ablations, queued by the last stage-2 job) the two-by-two below with the tuned settings.
# Training length is read from the full-length runs' held-out curves (logged every 2500 steps), not swept.
#
# Two by two, seeds 202 and 101 (60 pooled rollouts per arm per cell); the stage-2 run that won is base_s202:
#   base      capture A (bench_physics_v2)                              the tuned baseline
#   prev      capture A, --prev-commands 3                              previous-command input
#   off       capture B (bench_physics_v3 = v2 + joint offsets 3 deg)   joint-offset randomization
#   offprev   capture B, --prev-commands 3                              both
# Captures use suite seed 17 like the high-lift capture, so A and B hold the same placements and physics draws and
# differ only in the offsets. Evaluation at execute 30 on the slow-release scene: servo (no randomization),
# real_like_v1, real_like_offset_v1 (the unmeasured-joint-zero stand-in), bench_physics_v2.
# Reference (done): the existing high-lift ACT under real_like_offset_v1.
#   simulation_code/queue_offset_prevcmd_20261118.sh [--dry-run] [--ablations]
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
# CUDA_LAUNCH_BLOCKING: the high-lift fine-tune died twice with "CUDA error: unknown error" and ran clean with it (2026-11-13).
TRAINENV="env CUDA_LAUNCH_BLOCKING=1 ${ENVV#env }"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/offset_prevcmd_20261118
SW=$T/sweep
LADDER=$E/scaling_ladder_20260910
RETRY=$E/retry_tranche_20261004
LIFT=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100/scene_bench_pick_replace_v1.xml
REL30=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100_rel30/scene_bench_pick_replace_v1.xml
HS101=$E/so101_benchmark_20261106/so101/cap4800/heldout_suite.path
TRANCHE=offset-prevcmd-20261118
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; ABL=0
for a in "$@"; do case $a in --dry-run) DRY=1 ;; --ablations) ABL=1 ;; *) echo "unknown argument $a" >&2; exit 1 ;; esac; done
# budget.json's ACT arguments (batch 128, amp, 40k steps, eval every 2500); --steps is overridden per run below.
ACT_ARGS="$($PY -c "import json; print(json.load(open('$RETRY/budget.json'))['act_ft_args'])") --execs 30"
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then [ -n "$2" ] && dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
act() {  # act <label> <dep> <output dir> <capture dir> <seed> <extra args...>
  local label=$1 dep=$2 out=$3 cap=$4 seed=$5; shift 5
  [ -f "$out/result.json" ] && { echo ""; return; }
  q "$label" -D "$dep" $TRAINENV $PY -u $ROOT/tools/vla_finetune.py --model act --scene $REL30 --output-dir $out --data-dirs $cap --seed $seed \
    --run-name "${label#offprev-}" --tranche $TRANCHE $ACT_ARGS "$@"
}
[ -f "$REL30" ] || { echo "missing $REL30 (prepare_bench_scene.py --teacher-release-rows 30)" >&2; exit 1; }
mkdir -p "$T" "$SW"

if [ $ABL = 0 ]; then
  # ---- reference, rehearsal gate, captures, sweep stages 1 and 2 ----
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
  DA=${CA:-$GATE}
  S1=(); RUNS=()
  while read -r name lr bb kl; do
    S1+=("$(act "offprev-sweep-$name" "$DA" "$SW/s1_$name" "$T/capture_v2" 202 --steps 15000 --lr $lr --backbone-lr $bb --kl-weight $kl)")
    RUNS+=(--run "$name=$SW/s1_$name")
  done <<'GRID'
lr1e-4_bb1e-5_kl10 1e-4 1e-5 10
lr3e-5_bb1e-5_kl10 3e-5 1e-5 10
lr3e-4_bb1e-5_kl10 3e-4 1e-5 10
lr1e-4_bb1e-4_kl10 1e-4 1e-4 10
lr3e-4_bb3e-5_kl10 3e-4 3e-5 10
lr3e-4_bb1e-4_kl10 3e-4 1e-4 10
lr1e-4_bb1e-5_kl1 1e-4 1e-5 1
lr1e-4_bb1e-4_kl1 1e-4 1e-4 1
GRID
  SEL1=$(q offprev-sweep-select1 $ENVV $PY -u $ROOT/tools/sweep_select.py "${RUNS[@]}" --out $SW/winner_stage1.json)
  F1=$(act offprev-sweep-full-current "$SEL1" "$SW/full_current_s202" "$T/capture_v2" 202 --steps 40000 --lr 1e-4 --backbone-lr 1e-5 --kl-weight 10)
  F2=$(q offprev-sweep-full-winner -D "$SEL1" bash $ROOT/simulation_code/sweep_full_run_20261118.sh)
  SEL2=$(q offprev-sweep-select2 $ENVV $PY -u $ROOT/tools/sweep_select.py --run current=$SW/full_current_s202 --run winner=$SW/full_winner_s202 --out $SW/tuned.json)
  NEXT=$(q offprev-queue-ablations -D "$SEL2" bash $ROOT/simulation_code/queue_offset_prevcmd_20261118.sh --ablations)
  note "offset/prev-command tranche with a tuned baseline queued: reference ${REF:-done}; rehearsal gate $GATE; captures v2 ${CA:-done}, v3 ${CB:-done}; sweep stage 1 ${S1[*]}, select $SEL1; stage 2 current ${F1:-done}, winner $F2, select $SEL2; ablations queued by $NEXT"
  exit 0
fi

# ---- stage 3: the two-by-two with the tuned settings ----
[ -f "$SW/tuned.json" ] || { echo "missing $SW/tuned.json (sweep stage 2 has not finished)" >&2; exit 1; }
read -r LR BB KL BASE <<<"$($PY -c "import json; w=json.load(open('$SW/tuned.json'))['winner']; print(w['learning_rate'], w['backbone_learning_rate'], w['kl_weight'], w['dir'])")"
TUNED="--steps 40000 --lr $LR --backbone-lr $BB --kl-weight $KL"
[ -e "$T/base_s202" ] || { [ $DRY = 1 ] || ln -s "$BASE" "$T/base_s202"; }      # the stage-2 run that won is the baseline's seed 202
EVALS=(); TRAIN=()
for seed in 202 101; do
  for arm in base prev off offprev; do
    cap=capture_v2; case $arm in off*) cap=capture_v3 ;; esac
    extra=(); case $arm in *prev) extra=(--prev-commands 3) ;; esac
    name=${arm}_s$seed
    J=$(act "offprev-act-$name" "" "$T/$name" "$T/$cap" $seed $TUNED "${extra[@]}")
    TRAIN+=("${J:-done}")
    for cell in servo servo_real_like_v1 servo_real_like_offset_v1 servo_bench_physics_v2; do
      [ -f "$T/evals/${name}_exec30__$cell/result.json" ] && continue
      physics=none; case $cell in servo_*) physics=${cell#servo_} ;; esac
      EVALS+=("$(q "offprev-eval-$name-$cell" -D "$J" $ENVV $PY -u $ROOT/tools/eval_policy.py --kind vla --checkpoint $T/$name/model/act.pt \
        --scene $REL30 --suite $HS101 --physics $physics --execute 30 --output-dir $T/evals/${name}_exec30__$cell)")
    done
  done
done
R=$(q offprev-report $ENVV $PY -u $ROOT/tools/eval_matrix_report.py --evals $T/evals --out $T/matrix.json)
note "offset/prev-command ablations queued with the tuned settings ($TUNED; base_s202 -> $BASE): ACT ${TRAIN[*]}; ${#EVALS[@]} evals ${EVALS[0]:-}..${EVALS[-1]:-}; report $R"
