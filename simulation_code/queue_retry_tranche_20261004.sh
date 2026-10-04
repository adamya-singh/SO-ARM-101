#!/usr/bin/env bash
# Retry tranche (2026-10-04): queue every job at once on the shared tsp queue (notes/retry-tranche-20261004.md).
# Main queue: pi0.5 ReinFlow -> ACT PPO (after the ACT fine-tune) -> SmolVLA ReinFlow (after the SmolVLA fine-tune)
#             -> research-log entries 03-07 under MuJoCo 3.9 -> final status/view.
# Fine-tune queue (runs at the same time, GPU-bound): ACT fine-tune -> SmolVLA fine-tune.
# The equal RL episode budget and each run's settings come from budget.json, written from the probes.
# Usage: queue_retry_tranche_20261004.sh [--dry-run]
set -euo pipefail
ROOT=/home/win10ubuntu/dev/robotic-arm/SO-ARM-101
PY=/home/win10ubuntu/miniforge3/envs/lerobot/bin/python
PY05=/home/win10ubuntu/miniforge3/envs/lerobot-pi05/bin/python
ENVV="env PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MUJOCO_GL=egl PYTHONPATH=$ROOT/src"
EXP=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/experiments
T=$EXP/retry_tranche_20261004
BUDGET=$T/budget.json
LOG=$ROOT/artifacts/so_arm101_v2/bench_pick_replace_v1/QUEUE_LOG.txt
DRY=0; [ "${1:-}" = "--dry-run" ] && DRY=1
[ -f "$BUDGET" ] || { echo "missing $BUDGET (write it from the probes first)"; exit 1; }
[ -f "$T/rl_suites/rl_suites.json" ] || { echo "missing RL suites"; exit 1; }
b() { $PY -c "import json,sys; print(json.load(open('$BUDGET'))$1)"; }
N=$(b "['episodes']")
note() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
q() {   # q <label> [-D id] -- command...
  local label=$1; shift
  local dep=()
  if [ "$1" = "-D" ]; then dep=(-D "$2"); shift 2; fi
  if [ $DRY = 1 ]; then echo "tsp -L $label ${dep[*]} $*" >&2; echo 0; return; fi
  tsp -L "$label" "${dep[@]}" "$@"
}
if [ $DRY = 0 ] && tsp -l | awk 'NR>1 && ($2=="running" || $2=="queued")' | grep -q .; then
  note "warning: the shared tsp queue already has running/queued jobs; the tranche runs after them"
fi

# Two queues share the GPU: RL is CPU-bound (16 sim workers) and leaves the GPU idle >50 % of the time, the fine-tunes
# are GPU-bound and need ~2 cores. Fine-tunes run on their own queue (TS_SOCKET=$FTQ) beside the RL runs; an RL run
# that starts from a fine-tune waits for its result.json (and exits if the fine-tune failed).
FTQ=/tmp/tsp-retry-finetunes
wait_for() {  # wait_for <fine-tune dir>: block until result.json, fail if the fine-tune failed
  echo "until [ -f $1/result.json ]; do grep -q '\"status\": \"failed\"' $1/progress.json 2>/dev/null && exit 1; sleep 60; done"
}
J3=$(q retry-rl-pi05 $ENVV $PY05 -u $ROOT/tools/rl_train.py --model pi05 --checkpoint $EXP/pi05_lora16_10k_20261002/model/pi05.pt \
      --output-dir $T/rl_pi05 --episodes $N $(b "['rl_pi05_args']"))
J1=$(TS_SOCKET=$FTQ q retry-act-ft $ENVV $PY -u $ROOT/tools/vla_finetune.py --model act --output-dir $T/act_ft $(b "['act_ft_args']"))
J2=$(TS_SOCKET=$FTQ q retry-smolvla-ft $ENVV $PY -u $ROOT/tools/vla_finetune.py --model smolvla --output-dir $T/smolvla_ft $(b "['smolvla_ft_args']"))
J5=$(q retry-rl-act bash -c "$(wait_for $T/act_ft); exec $ENVV $PY -u $ROOT/tools/rl_train.py --model act --checkpoint $T/act_ft/model/act.pt \
      --output-dir $T/rl_act --episodes $N $(b "['rl_act_args']")")
J4=$(q retry-rl-smolvla bash -c "$(wait_for $T/smolvla_ft); exec $ENVV $PY -u $ROOT/tools/rl_train.py --model smolvla --checkpoint $T/smolvla_ft/model/smolvla.pt \
      --output-dir $T/rl_smolvla --episodes $N $(b "['rl_smolvla_args']")")
J6=$(q retry-rerun-03-07 bash -c "$ENVV $PY -u $ROOT/tools/rerun_entries_03_07.py --root $T/rerun_03_07 --legacy-numerics --wandb; \
      rc=\$?; $ENVV $PY $ROOT/tools/compare_quarantine.py --root $T/rerun_03_07; exit \$rc")
J7=$(q retry-final bash -c "$PY $ROOT/tools/tranche_status.py; $ENVV $PY05 $ROOT/tools/wandb_workspace.py --family tranche > $T/tranche_view_url.txt 2>&1 || true")
note "retry tranche queued: rl-pi05 $J3, rl-act $J5, rl-smolvla $J4, rerun-03-07 $J6, final $J7 (main queue); act-ft $J1, smolvla-ft $J2 (TS_SOCKET=$FTQ); N=$N"
