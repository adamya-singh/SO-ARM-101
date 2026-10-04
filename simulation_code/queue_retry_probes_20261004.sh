#!/usr/bin/env bash
# Retry tranche (2026-10-04), readiness step: throughput / step-time / noise probes, each on the GPU alone.
# The GPU is shared with other sessions' tsp jobs; a probe that runs beside another 3B model measures the
# contention, not the code (2026-10-04: a pi0.5 probe beside a lag-stress run made no progress in 31 min).
# Results land in .../retry_tranche_20261004/probes/<name>/{probe,noise_probe}.json; the budget is set from them.
set -euo pipefail
ROOT=/home/win10ubuntu/dev/robotic-arm/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
PY05=/home/win10ubuntu/miniforge3/envs/lerobot-pi05/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src"
EXP=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
P=$EXP/retry_tranche_20261004/probes
R=$EXP/rehearsal/retry_tranche
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
mkdir -p $P
J1=$(tsp -L retry-probe-rl-pi05 $ENVV $PY05 -u $ROOT/tools/rl_train.py --model pi05 --checkpoint $EXP/pi05_lora16_10k_20261002/model/pi05.pt \
      --output-dir $P/rl_pi05 --no-wandb --probe --probe-envs 8 14 --noise-probe --envs 14 --noise-levels 0.02 0.05 0.1 0.2)
J2=$(tsp -L retry-probe-ft-act $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --output-dir $P/ft_act --probe --probe-micro 32 64 128)
J3=$(tsp -L retry-probe-ft-smolvla $ENVV $PY -u $ROOT/tools/vla_finetune.py --model smolvla --output-dir $P/ft_smolvla --probe --probe-micro 8 16 32)
J4=$(tsp -L retry-probe-smolvla-rehearsal $ENVV $PY -u $ROOT/tools/vla_finetune.py --model smolvla --output-dir $R/smolvla_ft --rehearsal \
      --steps 20 --eval-every 10 --scratch-every 10 --rollout-limit 1 --execs 90 --rl-steps 4)
J5=$(tsp -L retry-probe-rl-smolvla -D $J4 $ENVV $PY -u $ROOT/tools/rl_train.py --model smolvla --checkpoint $R/smolvla_ft/model/smolvla.pt \
      --output-dir $P/rl_smolvla --no-wandb --probe --probe-envs 8 14)
J6=$(tsp -L retry-probe-rl-act $ENVV $PY -u $ROOT/tools/rl_train.py --model act --checkpoint $R/act_ft/model/act.pt \
      --output-dir $P/rl_act --no-wandb --probe --probe-envs 8 14)
echo "[$(date '+%F %T')] retry-tranche probes queued: rl-pi05 $J1, ft-act $J2, ft-smolvla $J3, smolvla rehearsal $J4, rl-smolvla $J5, rl-act $J6" | tee -a $LOG
