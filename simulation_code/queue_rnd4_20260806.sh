#!/usr/bin/env bash
# rnd4 chain: full-episode-screened suites -> preflights -> frame capture -> generalization loop
set -u
ROOT=/home/win10ubuntu/dev/robotic-arm/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl"
MODEL=$ROOT/simulation_code/model/menagerie_so_arm100/scene_v2.xml
TRAIN_SUITE=$ROOT/artifacts/so_arm101_v2/suites/e7663bab273d7514/suite.json
EVAL_SUITE=$ROOT/artifacts/so_arm101_v2/suites/10e6a56077e91ce4/suite.json
TRAIN_ID=random_pick_place_v3_seed7_n25_2b99d8ab

J1=$(tsp -L rnd4-preflight-train $ENVV $PY -u -m so_arm101_v2.simulation.cli preflight --mujoco-model $MODEL --output-dir $ROOT/artifacts/so_arm101_v2/simulation --suite fixed_pick_place_v3 --suite-path $TRAIN_SUITE --no-video)
J2=$(tsp -L rnd4-preflight-eval $ENVV $PY -u -m so_arm101_v2.simulation.cli preflight --mujoco-model $MODEL --output-dir $ROOT/artifacts/so_arm101_v2/simulation --suite fixed_pick_place_v3 --suite-path $EVAL_SUITE --no-video)
J3=$(tsp -L rnd4-capture $ENVV $PY -u -m so_arm101_v2.simulation.cli capture-oracle --mujoco-model $MODEL --output-dir $ROOT/artifacts/so_arm101_v2/oracle_distillation --suite fixed_pick_place_v3 --suite-path $TRAIN_SUITE --scenario all --teacher-horizon 480 --store-frames --skip-failed-scenarios --no-video --preflight-report $ROOT/artifacts/so_arm101_v2/simulation/preflight/$TRAIN_ID/evaluation.json)
echo "chain queued through capture job $J3"
tsp -w "$J3"
for L in rnd4-preflight-train rnd4-preflight-eval rnd4-capture; do
  echo "== $L"
  case $L in
    rnd4-preflight-train) ID=$J1 ;;
    rnd4-preflight-eval) ID=$J2 ;;
    rnd4-capture) ID=$J3 ;;
  esac
  tail -4 "$(tsp -o "$ID")"
done
echo "RND4 CAPTURE DONE"
