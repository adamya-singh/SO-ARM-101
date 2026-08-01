#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/win10ubuntu/miniforge3/envs/lerobot/bin/python}"
TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
DRY_RUN="${DRY_RUN:-0}"
source "${SCRIPT_DIR}/act_policy_mode.sh"
GROUP="${WANDB_RUN_GROUP:-act-lead3-grasp-phase1-20260711}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/train/act_lead3_grasp_phase1_20260711}"
PRETRAIN="${SCRIPT_DIR}/outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model"
NORMALIZER="policy_preprocessor_step_3_normalizer_processor.safetensors"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -x "$TSP" ]] || { echo "Missing task spooler: $TSP" >&2; exit 1; }
[[ -f "$PRETRAIN/model.safetensors" ]] || { echo "Missing model: $PRETRAIN" >&2; exit 1; }
[[ -f "$PRETRAIN/$NORMALIZER" ]] || { echo "Missing normalizer: $PRETRAIN/$NORMALIZER" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
act_print_policy_mode
act_configure_queue "$TSP" "$DRY_RUN"

enqueue() {
    local id="$1" lr="$2" log_std="$3" profile="$4" seed="$5"
    local name="${id}_${profile}_lr${lr//-/m}_std${log_std//-/m}_s${seed}"
    local run_dir="${OUTPUT_ROOT}/${name}"
    local command_file="${run_dir}/command.sh"
    mkdir -p "$run_dir"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        printf 'cd %q\n' "$SCRIPT_DIR"
        printf 'export MUJOCO_GL=egl WANDB_DIR=%q WANDB_RUN_GROUP=%q WANDB_NAME=%q PYTHONHASHSEED=%q\n' \
            "${run_dir}/wandb" "$GROUP" "$name" "$seed"
        printf 'exec %q -u train_act_in_sim.py' "$PY"
        printf ' --experimental-act-ppo --init-checkpoint %q' "$PRETRAIN"
        printf ' --reward-profile %q --policy-lr %q --critic-lr 5e-5 --log-std-init %q' "$profile" "$lr" "$log_std"
        act_append_trainer_resource_args
        printf ' --chunk-size 30 --max-steps-per-episode 150 --steps-per-action 1'
        printf ' --episodes 100 --snapshot-every 10 --eval-episodes 0 --seed %q' "$seed"
        printf ' --checkpoint-path %q' "${run_dir}/act_sim_ppo_checkpoint.pt"
        printf ' --no-randomize-appearance --curriculum-fixed-block --headless --no-render'
        printf ' > %q 2>&1\n' "${run_dir}/train.log"
    } > "$command_file"
    chmod +x "$command_file"
    act_enqueue_training "$TSP" "$DRY_RUN" "$name" "$command_file"
}

enqueue_seed_forward() {
    local seed="$1"
    enqueue A 1e-6 -2.0 baseline "$seed"
    enqueue B 5e-7 -2.0 baseline "$seed"
    enqueue C 1e-6 -2.356675 baseline "$seed"
    enqueue D 1e-6 -2.0 jaw_quality "$seed"
    enqueue E 1e-6 -2.0 jaw_quality_rebalanced "$seed"
    enqueue F 1e-6 -2.356675 jaw_quality "$seed"
}

enqueue_seed_reverse() {
    local seed="$1"
    enqueue F 1e-6 -2.356675 jaw_quality "$seed"
    enqueue E 1e-6 -2.0 jaw_quality_rebalanced "$seed"
    enqueue D 1e-6 -2.0 jaw_quality "$seed"
    enqueue C 1e-6 -2.356675 baseline "$seed"
    enqueue B 5e-7 -2.0 baseline "$seed"
    enqueue A 1e-6 -2.0 baseline "$seed"
}

enqueue_seed_forward 17
enqueue_seed_reverse 71

echo "Phase-one output root: $OUTPUT_ROOT"
if [[ "$DRY_RUN" != "1" ]]; then "$TSP"; fi
