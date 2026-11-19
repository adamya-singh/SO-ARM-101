#!/usr/bin/env bash
# Do the features stack, combine and build on each other? (2026-11-19, notes/recursive-generalization-20261119.md)
# The feature probes (2026-11-18) showed one data change gives one feature. This tranche tests the three further claims
# a curriculum rests on, with ACT, read by the same probes on the same 100 placements (tools/feature_probes.py):
#   1 stacking      one policy trained on both data changes has both features
#                     both_split  3600 perturbed-teacher + 3600 physics-v2 placements (the 7200 budget of the single-change runs)
#                     both_full   7200 + 7143 (each change at its full amount; queued last)
#   2 combination   the cube shifts AND the elbow changes in one episode (probe shift_and_swap), which no training
#                   episode contains; on the existing policies first, then on the two above
#   3 order         a policy that already has one feature learns the other from less data than one that has neither:
#                   5000 further steps on the first N placements of the other change, N = 300 / 1200 / 4800, from
#                     motor  baseline ACT vs perturbed-teacher ACT, on physics-v2 data   (seeds 202, 101)  -> swap_elbow
#                     look   baseline ACT vs physics-v2 ACT, on perturbed-teacher data   (seed 202)        -> shift_placement
# Every trainer is one tsp job and its probe cells another that depends on it; a rehearsal gates all of them.
# Resumable: finished trainers and finished probes are not run again.
#   simulation_code/queue_recursive_generalization_20261119.sh [--dry-run]
#   simulation_code/queue_recursive_generalization_20261119.sh --stage <rehearsal|combo|cells NAME|report>
set -uo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/recursive_generalization_20261119
PT=$E/feature_probes_20261118                 # the probe suite, the shift plans and the cells of the existing policies
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
SELF=$ROOT/simulation_code/queue_recursive_generalization_20261119.sh
TOOL="$ENVV $PY -u $ROOT/tools/feature_probes.py"
TRAIN="$ENVV $PY -u $ROOT/tools/vla_finetune.py --model act"
REPORT="$ENVV $PY -u $ROOT/tools/recursive_generalization_report.py --root $T --probe-root $PT"
TRANCHE=recursive-generalization-20261119
ACT_ARGS=$($PY -c "import json; print(json.load(open('$E/retry_tranche_20261004/budget.json'))['act_ft_args'])")
scene() { echo "$ROOT/simulation_code/model/bench_pick_replace_v1_$1/scene_bench_pick_replace_v1.xml"; }
PCAP=$E/perturbed_so101_20261108/capture7200  # perturbed teacher, nominal SO-101 scene
VCAP=$E/physics_v2_20261108/capture7200       # physics v2 draws, measured-servo SO-101 scene
declare -A START=([base]=$E/so101_benchmark_20261106/so101/act_ft/model/act.pt [pert]=$E/perturbed_so101_20261108/act_ft/model/act.pt
                  [physv2]=$E/physics_v2_20261108/act_ft/model/act.pt)
STEPS=5000
SIZES="300 1200 4800"

FAILED=0
cell() {  # cell <root> <cell name> <kind> <checkpoint or -> <execute> <scene> [extra args...]
  local root=$1 name=$2 kind=$3 ckpt=$4 exec=$5 sc=$6; shift 6
  [ "$ckpt" = - ] && ckpt=
  $TOOL run --kind "$kind" ${ckpt:+--checkpoint "$ckpt"} --execute "$exec" --scene "$(scene $sc)" --plan "$PT/plans/$sc.json" \
    --workers 10 --output-dir "$root/cells/${name}__$sc" "$@" || { echo "CELL FAILED ${name}__$sc" >&2; FAILED=1; }
}
cnn() { ls "$1"/runs/shift12/models/vision_h90/*/model.pt | head -1; }

stage() {
  case $1 in
    rehearsal)   # the two new trainer switches and the new probe, end to end, on a few steps and 3 placements
      R=$T/rehearsal; rm -rf "$R"
      $TRAIN --scene "$(scene so101_servo)" --output-dir "$R/train" --data-dirs $VCAP $PCAP --episode-limits 40 40 --init-checkpoint "${START[pert]}" \
        --rehearsal --rollout-limit 1 --execs 90 --tranche $TRANCHE $ACT_ARGS --steps 40 --eval-every 40 || exit 1
      cell "$R" rehearsal vla "$R/train/model/act.pt" 90 so101 --limit 3 --probes shift_placement shift_and_swap
      cell "$R" teacher_informed teacher_informed - 90 so101 --limit 3 --probes shift_and_swap
      [ $FAILED = 0 ] && { $ENVV $PY -u $ROOT/tools/recursive_generalization_report.py --root "$R" --probe-root "$R" || FAILED=1; } ;;
    combo)       # experiment 2 on what exists: the new probe joins the cells the feature probes already hold
      for k in teacher teacher_informed; do cell "$PT" $k $k - 90 so101 --probes swap_elbow shift_and_swap; done
      for n in base pert physv2; do for x in 90 30; do
        tag=${n}_act; [ $x = 30 ] && tag=${n}_act_exec30
        cell "$PT" $tag vla "${START[$n]}" $x so101 --probes shift_placement swap_elbow shift_and_swap
      done; done
      for s in 202 101 303; do
        b=$E/perturbed_so101_20261108/base_cnn_s$s; [ $s = 202 ] && b=$E/so101_benchmark_20261106/so101/cnn_s202
        cell "$PT" base_cnn_s$s vision_chunked "$(cnn $b)" 90 so101 --probes shift_placement swap_elbow shift_and_swap
        cell "$PT" pert_cnn_s$s vision_chunked "$(cnn $E/perturbed_so101_20261108/cnn_s$s)" 90 so101 --probes shift_placement swap_elbow shift_and_swap
        cell "$PT" physv2_cnn_s$s vision_chunked "$(cnn $E/physics_v2_20261108/cnn_s$s)" 90 so101 --probes shift_placement swap_elbow shift_and_swap
      done
      $REPORT || FAILED=1 ;;
    cells)       # cells <run name>: probe one trained policy
      name=$2; ckpt=$T/$name/model/act.pt
      [ -f "$ckpt" ] || { echo "no checkpoint for $name" >&2; exit 1; }
      case $name in
        both_*) for x in 90 30; do
                  tag=$name; [ $x = 30 ] && tag=${name}_exec30
                  cell "$T" $tag vla "$ckpt" $x so101 --probes shift_placement swap_elbow shift_and_swap
                  cell "$T" $tag vla "$ckpt" $x so101_servo --probes shift_placement swap_elbow shift_and_swap
                done ;;
        *)      cell "$T" $name vla "$ckpt" 90 so101 --probes shift_placement swap_elbow ;;
      esac
      $REPORT || FAILED=1 ;;
    report) $REPORT || FAILED=1 ;;
    *) echo "unknown stage $1" >&2; exit 2 ;;
  esac
  exit $FAILED
}

if [ "${1:-}" = "--stage" ]; then shift; stage "$@"; fi
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
for f in "${START[@]}" "$PCAP/train_manifest.path" "$VCAP/train_manifest.path" "$PT/probe_suite.path" "$PT/plans/so101.json" "$PT/plans/so101_servo.json"; do
  [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }
done
q() {   # q <label> [-D id] command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then [ -n "$2" ] && dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
mkdir -p "$T"
JOBS=()
run() {   # run <name> <trainer args...>: the trainer (unless it finished) and then its probe cells
  local name=$1; shift
  local j=$GATE
  [ -f "$T/$name/result.json" ] || j=$(q "rg-train-$name" -D "$GATE" $TRAIN --output-dir "$T/$name" --run-name "rg-$name" --tranche $TRANCHE $ACT_ARGS "$@")
  JOBS+=("$name:$j/$(q "rg-cells-$name" -D "$j" bash "$SELF" --stage cells "$name")")
}
curriculum() {   # curriculum <direction> <start> <data dir> <scene> <seed>
  for n in $SIZES; do
    run "cur_$1_from_$2_n${n}_s$5" --scene "$(scene $4)" --data-dirs "$3" --episode-limits "$n" --init-checkpoint "${START[$2]}" \
      --seed "$5" --steps $STEPS --execs 90
  done
}
GATE=$(q rg-rehearsal bash "$SELF" --stage rehearsal)
COMBO=$(q rg-combo -D "$GATE" bash "$SELF" --stage combo)
run both_split --scene "$(scene so101)" --data-dirs $PCAP $VCAP --episode-limits 3600 3600
for start in base pert; do curriculum motor $start $VCAP so101_servo 202; done
for start in base physv2; do curriculum look $start $PCAP so101 202; done
for start in base pert; do curriculum motor $start $VCAP so101_servo 101; done
run both_full --scene "$(scene so101)" --data-dirs $PCAP $VCAP
LAST=$(q rg-report bash "$SELF" --stage report)
MSG="recursive-generalization tranche queued: rehearsal gate $GATE, combined probe on existing policies $COMBO, ${#JOBS[@]} trainers with cells (${JOBS[0]} .. ${JOBS[-1]}), report $LAST"
if [ $DRY = 1 ]; then echo "(dry run) $MSG"; else echo "[$(date '+%F %T')] $MSG" | tee -a "$LOG"; fi
