#!/usr/bin/env bash
# Counterfactual feature probes on finished checkpoints (2026-11-18, notes/feature-probes-20261118.md). No training.
# Each cell is one policy on one scene: the unprobed control and the probes on 100 teacher-screened held-out placements
# (tools/feature_probes.py). Stages, each one tsp job:
#   plan       per scene, the displacement every policy gets (teacher-admitted)            CPU
#   rehearsal  3 placements through the teacher, a CNN and an ACT cell, then the report     gate for everything below
#   controls   scripted teacher, plain and informed, on the three scenes                    CPU, must read 0 and 1
#   headline   the two intervention-vs-baseline comparisons on the nominal SO-101 scene:
#                shift_placement: perturbed-teacher policies vs their baselines
#                swap_elbow:      physics-v2 policies vs their baselines
#   rest       every other probe and scene (nominal, measured servo, high-lift)
# plan/rehearsal/controls/headline are moved to the front of the queue (about 3 hours); rest stays at the back.
#   simulation_code/queue_feature_probes_20261118.sh [--dry-run]          queue the stages
#   simulation_code/queue_feature_probes_20261118.sh --stage <name>       run one stage (what the tsp jobs call)
set -uo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/feature_probes_20261118
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
TOOL="$ENVV $PY -u $ROOT/tools/feature_probes.py"
SELF=$ROOT/simulation_code/queue_feature_probes_20261118.sh
scene() { echo "$ROOT/simulation_code/model/bench_pick_replace_v1_$1/scene_bench_pick_replace_v1.xml"; }
cnn() { ls "$1"/runs/shift12/models/vision_h90/*/model.pt | head -1; }

# name -> "kind checkpoint execute"
declare -A P
P[base_act]="vla $E/so101_benchmark_20261106/so101/act_ft/model/act.pt 90"
P[pert_act]="vla $E/perturbed_so101_20261108/act_ft/model/act.pt 90"
P[physv2_act]="vla $E/physics_v2_20261108/act_ft/model/act.pt 90"
P[lift_act]="vla $E/high_lift_20261112/act_ft/model/act.pt 90"
for n in base_act pert_act physv2_act lift_act; do read -r k c _ <<<"${P[$n]}"; P[${n}_exec30]="$k $c 30"; done
P[base_cnn_s202]="vision_chunked $(cnn $E/so101_benchmark_20261106/so101/cnn_s202) 90"
for s in 101 303; do P[base_cnn_s$s]="vision_chunked $(cnn $E/perturbed_so101_20261108/base_cnn_s$s) 90"; done
for s in 202 101 303; do
  P[pert_cnn_s$s]="vision_chunked $(cnn $E/perturbed_so101_20261108/cnn_s$s) 90"
  P[physv2_cnn_s$s]="vision_chunked $(cnn $E/physics_v2_20261108/cnn_s$s) 90"
  P[fair_cnn_s$s]="vision_chunked $(cnn $E/fair_cnn_20261118/cnn_s$s) 90"
  P[fair_cnn_s${s}_exec30]="vision_chunked $(cnn $E/fair_cnn_20261118/cnn_s$s) 30"
done
BASE="base_act base_act_exec30 base_cnn_s202 base_cnn_s101 base_cnn_s303"
PERT="pert_act pert_act_exec30 pert_cnn_s202 pert_cnn_s101 pert_cnn_s303"
PHYS="physv2_act physv2_act_exec30 physv2_cnn_s202 physv2_cnn_s101 physv2_cnn_s303"
LIFT="lift_act_exec30 lift_act fair_cnn_s202 fair_cnn_s101 fair_cnn_s303 fair_cnn_s202_exec30 fair_cnn_s101_exec30 fair_cnn_s303_exec30"

FAILED=0
cell() {  # cell <root> <policy> <scene> [extra args...]   (probes already done in that cell are loaded, not re-run)
  local root=$1 name=$2 sc=$3; shift 3
  local kind ckpt exec
  if [ "$name" = teacher ] || [ "$name" = teacher_informed ]; then kind=$name; ckpt=; exec=90
  else read -r kind ckpt exec <<<"${P[$name]}"; fi
  $TOOL run --kind "$kind" ${ckpt:+--checkpoint "$ckpt"} --execute "$exec" --scene "$(scene $sc)" --plan "$T/plans/$sc.json" \
    --workers 10 --output-dir "$root/cells/${name}__$sc" "$@" || { echo "CELL FAILED ${name}__$sc" >&2; FAILED=1; }
}

stage() {
  case $1 in
    plan)
      [ -f "$T/probe_suite.path" ] || $TOOL suite --scene "$(scene so101)" --workers 10 || exit 1
      for sc in so101 so101_servo so101_servo_lift100; do $TOOL plan --scene "$(scene $sc)" --workers 10 || FAILED=1; done ;;
    rehearsal)
      R=$T/rehearsal; rm -rf "$R"
      cell "$R" teacher so101 --limit 3
      cell "$R" teacher_informed so101 --limit 3
      cell "$R" base_cnn_s202 so101 --limit 3
      cell "$R" physv2_act_exec30 so101_servo --limit 3
      [ $FAILED = 0 ] && { $TOOL report --root "$R" || FAILED=1; } ;;
    controls)
      for sc in so101 so101_servo so101_servo_lift100; do cell "$T" teacher $sc; cell "$T" teacher_informed $sc; done
      $TOOL report --root "$T" || FAILED=1 ;;
    headline)
      for n in $BASE $PERT $PHYS; do cell "$T" $n so101 --probes shift_placement swap_elbow; done
      $TOOL report --root "$T" || FAILED=1 ;;
    rest)
      for n in $BASE $PERT $PHYS; do cell "$T" $n so101; done
      for n in $BASE $PHYS; do cell "$T" $n so101_servo; done
      for n in $LIFT; do cell "$T" $n so101_servo_lift100; done
      $TOOL report --root "$T" || FAILED=1 ;;
    *) echo "unknown stage $1" >&2; exit 2 ;;
  esac
  exit $FAILED
}

if [ "${1:-}" = "--stage" ]; then stage "$2"; fi
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
for n in "${!P[@]}"; do read -r _ c _ <<<"${P[$n]}"; [ -f "$c" ] || { echo "missing checkpoint for $n: $c" >&2; exit 1; }; done
q() {   # q <label> [-D id] -- stage
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} bash $SELF --stage $1" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" bash "$SELF" --stage "$1"
}
mkdir -p "$T"
PLAN=$(q probe-plan plan)
GATE=$(q probe-rehearsal -D "$PLAN" rehearsal)
CTRL=$(q probe-controls -D "$GATE" controls)
HEAD=$(q probe-headline -D "$GATE" headline)
REST=$(q probe-rest -D "$GATE" rest)
if [ $DRY = 0 ]; then for j in $HEAD $CTRL $GATE $PLAN; do tsp -u "$j"; done; fi     # last moved = first to run
MSG="feature probes queued: plan $PLAN, rehearsal gate $GATE, controls $CTRL, headline $HEAD (these four moved to the front), rest $REST (back of the queue)"
if [ $DRY = 1 ]; then echo "(dry run) $MSG"; else echo "[$(date '+%F %T')] $MSG" | tee -a "$LOG"; fi
