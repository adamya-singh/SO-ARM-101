#!/usr/bin/env bash
# Retry tranche, second half (2026-10-05): what the 10-04 queue did not finish (notes/retry-tranche-20261004.md).
#   1-3. the three finished policies, live in the MuJoCo viewer on the held-out placements, recorded to MP4
#   4.   SmolVLA fine-tune, alone   (10-05 00:33 it was OOM-killed beside ACT PPO's 20 env workers in 15 GB of RAM)
#   5.   ACT PPO again, actor lr 2e-7 (10-05: KL 17 on the first update at 1e-5; KL scaled ~lr^2, so 2e-7 -> ~0.007)
#   6.   SmolVLA ReinFlow, only if 4 succeeded (tsp -D; the old wait loop polled forever when the fine-tune died)
# Everything runs one at a time on the shared tsp queue, so no two jobs compete for RAM.
# Usage: queue_retry_remaining_20261005.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/dev/robotic-arm/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
PY05=/home/win10ubuntu/miniforge3/envs/lerobot-pi05/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src"
VIEWENV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 DISPLAY=${DISPLAY:-:0} WAYLAND_DISPLAY=${WAYLAND_DISPLAY:-wayland-0} XDG_RUNTIME_DIR=${XDG_RUNTIME_DIR:-/run/user/1000}"
T=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments/retry_tranche_20261004
V=$T/videos
BUDGET=$T/budget.json
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
b() { $PY -c "import json,sys; print(json.load(open('$BUDGET'))$1)"; }
N=$(b "['episodes']")
note() { if [ $DRY = 1 ]; then echo "(dry run) $*"; else echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; fi; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
mkdir -p "$V"
view() {  # view <label> <kind> <python> <checkpoint> <name>
  q "view-$5" $VIEWENV "$3" "$ROOT/tools/view_policy_live.py" --kind "$2" --checkpoint "$4" --label "$1" --execute 90 --once \
    --record "$V/$5.mp4" --summary "$V/$5.json"
}
J1=$(view "pi0.5 ReinFlow (40k ep)" pi05 "$PY05" "$T/rl_pi05/model/pi05_rl.pt" pi05_reinflow)
J2=$(view "ACT fine-tune" act "$PY" "$T/act_ft/model/act.pt" act_finetune)
J3=$(view "ACT PPO (aborted at 780 ep)" act "$PY" "$T/rl_act/model/act_rl.pt" act_ppo_aborted)
J4=$(q retry-smolvla-ft $ENVV $PY -u $ROOT/tools/vla_finetune.py --model smolvla --output-dir $T/smolvla_ft $(b "['smolvla_ft_args']"))
J5=$(q retry-rl-act-lr2e-7 $ENVV $PY -u $ROOT/tools/rl_train.py --model act --checkpoint $T/act_ft/model/act.pt \
      --output-dir $T/rl_act_lr2e-7 --episodes $N --actor-lr 2e-7 $(b "['rl_act_args']"))
J6=$(q retry-rl-smolvla -D "$J4" $ENVV $PY -u $ROOT/tools/rl_train.py --model smolvla --checkpoint $T/smolvla_ft/model/smolvla.pt \
      --output-dir $T/rl_smolvla --episodes $N $(b "['rl_smolvla_args']"))
note "retry tranche remainder queued: views $J1 $J2 $J3, smolvla-ft $J4, rl-act-lr2e-7 $J5, rl-smolvla $J6 (-D $J4); N=$N"
