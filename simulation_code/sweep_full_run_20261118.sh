#!/usr/bin/env bash
# Sweep stage 2 (queue_offset_prevcmd_20261118.sh): train the stage-1 winner's settings for the full 40k steps, seed 202.
# Exits 0 without training when the winner is the current settings (1e-4, 1e-5, 10): full_current_s202 is that run.
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/offset_prevcmd_20261118
REL30=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100_rel30/scene_bench_pick_replace_v1.xml
read -r LR BB KL <<<"$($PY -c "import json; w=json.load(open('$T/sweep/winner_stage1.json'))['winner']; print(w['learning_rate'], w['backbone_learning_rate'], w['kl_weight'])")"
if $PY -c "import sys; sys.exit(0 if (float('$LR'), float('$BB'), float('$KL')) == (1e-4, 1e-5, 10.0) else 1)"; then
  echo "stage-1 winner is the current settings; full_current_s202 is the tuned run"; exit 0
fi
[ -f "$T/sweep/full_winner_s202/result.json" ] && exit 0
ACT_ARGS="$($PY -c "import json; print(json.load(open('$E/retry_tranche_20261004/budget.json'))['act_ft_args'])") --execs 30"
exec env CUDA_LAUNCH_BLOCKING=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src \
  $PY -u $ROOT/tools/vla_finetune.py --model act --scene $REL30 --output-dir $T/sweep/full_winner_s202 --data-dirs $T/capture_v2 --seed 202 \
  --run-name sweep-full-winner --tranche offset-prevcmd-20261118 $ACT_ARGS --steps 40000 --lr $LR --backbone-lr $BB --kl-weight $KL
