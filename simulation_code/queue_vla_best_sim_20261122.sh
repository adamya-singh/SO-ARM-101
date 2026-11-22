#!/usr/bin/env bash
# pi0.5 and SmolVLA on the recipe that works for ACT (2026-11-22, notes/vla-best-sim-20261122.md).
# The 2026-10-02/04 VLA runs (pi0.5 15/30, SmolVLA 2/30) used the SO-100 scene, the unrandomized 4800+2400 captures
# and a 25 mm lift. Since then: SO-101 gripper (grasp tolerance), measured servo model, physics v2 (soft..stiff
# servos), 100 mm lift, slow release, execute 30. Same data and scene as the tuned ACT baseline of 2026-11-18
# (offset_prevcmd capture_v2, scene so101_servo_lift100_rel30: ACT 21-24/30 at execute 30), same 10 held-out poses x 3.
#   rehearsals (gate each trainer)  ->  pi0.5 LoRA 16, 6000 steps  ->  SmolVLA 30000 steps
# RL on a fine-tune that works is queued separately once its result is in (tools/rl_train.py).
set -uo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
PY05=/home/win10ubuntu/miniforge3/envs/lerobot-pi05/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
ENV05="$ENVV"     # the lerobot-pi05 env's editable .pth files still point at the old checkout: both packages come from PYTHONPATH
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$E/vla_best_sim_20261122
CAP=$E/offset_prevcmd_20261118/capture_v2
SCENE=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo_lift100_rel30/scene_bench_pick_replace_v1.xml
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
TRANCHE=vla-best-sim-20261122
mkdir -p "$T"
R1=$(tsp -L vla-rehearse-pi05 bash -c "rm -rf $T/rehearsal/pi05; exec $ENV05 $PY05 -u $ROOT/tools/pi05_finetune.py --output-dir $T/rehearsal/pi05 --data-dirs $CAP --model $SCENE --rehearsal --steps 8 --eval-every 8 --rollout-limit 1 --execs 30")
R2=$(tsp -L vla-rehearse-smolvla bash -c "rm -rf $T/rehearsal/smolvla; exec $ENVV $PY -u $ROOT/tools/vla_finetune.py --model smolvla --scene $SCENE --output-dir $T/rehearsal/smolvla --data-dirs $CAP --rehearsal --rollout-limit 1 --execs 30 --tranche $TRANCHE --batch 32 --micro 32 --steps 20 --eval-every 20 --loader-threads 6")
J1=$(tsp -L vla-pi05-ft -D $R1 $ENV05 $PY05 -u $ROOT/tools/pi05_finetune.py --output-dir $T/pi05_ft --data-dirs $CAP --model $SCENE --steps 6000 --eval-every 1000 --execs 30 --run-name vla-best-sim-pi05-lora16-6k)
J2=$(tsp -L vla-smolvla-ft -D $R2 $ENVV $PY -u $ROOT/tools/vla_finetune.py --model smolvla --scene $SCENE --output-dir $T/smolvla_ft --data-dirs $CAP --execs 30 --tranche $TRANCHE --run-name vla-best-sim-smolvla-ft --batch 32 --micro 32 --steps 30000 --eval-every 2500 --scratch-every 2500 --loader-threads 6)
echo "[$(date '+%F %T')] VLA best-sim tranche queued: rehearsals pi0.5 $R1, SmolVLA $R2; pi0.5 fine-tune $J1; SmolVLA fine-tune $J2" | tee -a "$LOG"
