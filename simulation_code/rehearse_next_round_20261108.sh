#!/usr/bin/env bash
# Rehearsals for items 3 and 4 (2026-11-08): toy-sized runs of every new path on the SO-101 scenes, under rehearsal/
# directories (nothing in them may be quoted). Prints the last job id; pass it as GATE_JOB to both queue scripts.
#   R1 perturbed capture on the SO-101 (item 4)            R2 physics-v2 capture on the SO-101 servo scene (item 3)
#   R3 CNN 500 steps on R2                                 R4 ACT 20 steps on R2, both execute rates
#   R5 PPO 120 episodes from R4 under bench_physics_v2 on the rebased RL pool
#   R6/R7 eval_policy cells: R4 on servo+bench_physics_v2 at execute 30, R3 on the stiff-elbow diagnostic scene
set -euo pipefail
ROOT=/home/win10ubuntu/projects/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src:/home/win10ubuntu/projects/lerobot-fork/src"
E=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
NR=$E/next_round_20261108
R=$NR/rehearsal
B=$E/so101_benchmark_20261106/so101
LADDER=$E/scaling_ladder_20260910
SO101=$ROOT/simulation_code/model/bench_pick_replace_v1_so101/scene_bench_pick_replace_v1.xml
SERVO=$ROOT/simulation_code/model/bench_pick_replace_v1_so101_servo/scene_bench_pick_replace_v1.xml
HS101=$B/cap4800/heldout_suite.path
TR=next-round-20261108-rehearsal
mkdir -p "$R"
R1=$(tsp -L rehearse-so101p-capture $ENVV $PY -u $ROOT/tools/capture_placements.py --model $SO101 --output-dir $R/capture_perturbed --count 6 \
     --suite-seed 99 --frame-row-stride 30 --perturb-teacher-mm 15 --teacher-descend-mm 20 --heldout-from $B/cap4800 --workers 6)
R2=$(tsp -L rehearse-physv2-capture -D $R1 $ENVV $PY -u $ROOT/tools/capture_placements.py --model $SERVO --output-dir $R/capture_physv2 --count 6 \
     --suite-seed 99 --frame-row-stride 30 --physics bench_physics_v2 --skip-failed --heldout-suite $(cat $LADDER/heldout_suite.path) \
     --heldout-from $LADDER --workers 10)
R3=$(tsp -L rehearse-physv2-cnn -D $R2 $ENVV $PY -u $ROOT/tools/augmentation_run.py --rehearsal --model $SERVO --ladder-dir $R/capture_physv2 \
     --output-dir $R/cnn --shifts 12 --episodes 6 --frame-stride 30 --no-baseline --seed 202 --steps 500)
R4=$(tsp -L rehearse-physv2-act -D $R3 $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --rehearsal --scene $SERVO --output-dir $R/act \
     --data-dirs $R/capture_physv2 --steps 20 --batch 8 --micro 8 --eval-every 10 --scratch-every 10 --rollout-limit 1 --execs 90 30 --tranche $TR)
R5=$(tsp -L rehearse-physv2-rl -D $R4 $ENVV $PY -u $ROOT/tools/rl_train.py --model act --rehearsal --checkpoint $R/act/model/act.pt --scene $SERVO \
     --physics bench_physics_v2 --rl-suites $NR/rl_suites_so101/rl_suites.json --heldout-suite $HS101 --output-dir $R/rl --episodes 120 \
     --rollout-limit 1 --execs 90 30 --actor-lr 2e-7 --envs 20 --groups 2 --episodes-per-iter 60 --watchdog 30 --render cpu --tranche $TR)
R6=$(tsp -L rehearse-physv2-eval-act -D $R5 $ENVV $PY -u $ROOT/tools/eval_policy.py --kind vla --checkpoint $R/act/model/act.pt --scene $SERVO \
     --suite $HS101 --physics bench_physics_v2 --execute 30 --output-dir $R/evals/act_exec30__servo_bench_physics_v2)
R7=$(tsp -L rehearse-physv2-eval-cnn -D $R6 bash -c "exec $ENVV $PY -u $ROOT/tools/eval_policy.py --kind vision_chunked \
     --checkpoint \$(ls $R/cnn/runs/shift12/models/vision_h90/*/model.pt | head -1) --scene $NR/scenes/so101_servo_stiff_elbow/scene_bench_pick_replace_v1.xml \
     --suite $HS101 --physics none --output-dir $R/evals/cnn__stiff_elbow")
echo "[$(date '+%F %T')] next-round rehearsals queued: R1 $R1 .. R7 $R7" >> $ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
echo $R7
