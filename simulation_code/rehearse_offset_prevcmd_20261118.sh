#!/usr/bin/env bash
# Rehearsal for the joint-offset / previous-command tranche (2026-11-18): toy-sized runs of every new path, under
# rehearsal/ (nothing in it may be quoted). Runs its steps in order and fails on the first error; the tranche's queue
# (queue_offset_prevcmd_20261118.sh) is gated on it.
#   1 capture, 6 placements, slow-release scene, bench_physics_v3 (joint offsets)   2 ACT 20 steps with --prev-commands 3
#   3 ACT 20 steps with the sweep's flags, then the selector   4 eval_policy of the step-2 checkpoint under real_like_offset_v1 at execute 30
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
R=$E/offset_prevcmd_20261118/rehearsal
LADDER=$E/scaling_ladder_20260910
REL30=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100_rel30/scene_bench_pick_replace_v1.xml
HS101=$E/so101_benchmark_20261106/so101/cap4800/heldout_suite.path
rm -rf "$R"; mkdir -p "$R"
$ENVV $PY -u $ROOT/tools/capture_placements.py --model $REL30 --output-dir $R/capture --count 6 --suite-seed 99 --frame-row-stride 30 \
  --physics bench_physics_v3 --skip-failed --heldout-suite $(cat $LADDER/heldout_suite.path) --heldout-from $LADDER --workers 10
$ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --rehearsal --scene $REL30 --output-dir $R/act --data-dirs $R/capture --prev-commands 3 \
  --steps 20 --batch 8 --micro 8 --eval-every 10 --scratch-every 10 --rollout-limit 1 --execs 30 --tranche offset-prevcmd-20261118-rehearsal
# the sweep's path: the tuning flags, the selector and the full-length helper (which must skip a winner equal to the current settings)
$ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --rehearsal --scene $REL30 --output-dir $R/act_sweep --data-dirs $R/capture --lr 3e-4 --backbone-lr 1e-4 \
  --kl-weight 1 --steps 20 --batch 8 --micro 8 --eval-every 10 --scratch-every 10 --rollout-limit 1 --execs 30 --tranche offset-prevcmd-20261118-rehearsal
$ENVV $PY -u $ROOT/tools/sweep_select.py --run sweep=$R/act_sweep --run absent=$R/absent --out $R/winner.json
grep -q '"kl_weight": 1.0' $R/winner.json
$ENVV $PY -u $ROOT/tools/eval_policy.py --kind vla --checkpoint $R/act/model/act.pt --scene $REL30 --suite $HS101 \
  --physics real_like_offset_v1 --execute 30 --output-dir $R/evals/act_exec30__servo_real_like_offset_v1
echo "rehearsal complete"
