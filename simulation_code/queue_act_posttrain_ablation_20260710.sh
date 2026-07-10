#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/win10ubuntu/miniforge3/envs/lerobot/bin/python}"
TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
START_AT="${START_AT:-2026-07-10 00:00:00}"
GROUP="${WANDB_RUN_GROUP:-act-posttrain-ablation-20260710}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/train/act_posttrain_ablation_20260710}"

NEW_PRETRAIN="${SCRIPT_DIR}/outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model"
OLD_PRETRAIN="${SCRIPT_DIR}/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model"

[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 1; }
[[ -x "$TSP" ]] || { echo "Task Spooler is not executable: $TSP" >&2; exit 1; }
[[ -f "$NEW_PRETRAIN/model.safetensors" ]] || { echo "Missing new pretrain: $NEW_PRETRAIN" >&2; exit 1; }
[[ -f "$OLD_PRETRAIN/model.safetensors" ]] || { echo "Missing old pretrain: $OLD_PRETRAIN" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
"$TSP" -S 1

enqueue_run() {
    local run_name="$1"
    local checkpoint="$2"
    local appearance="$3"
    local block_mode="$4"
    local seed="$5"
    local wait_for_start="$6"
    local run_dir="${OUTPUT_ROOT}/${run_name}"
    local command_file="${run_dir}/command.sh"

    mkdir -p "$run_dir"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        if [[ "$wait_for_start" == "yes" ]]; then
            printf 'start_epoch=$(date -d %q +%%s)\n' "$START_AT"
            echo 'while (( $(date +%s) < start_epoch )); do sleep 30; done'
        fi
        printf 'cd %q\n' "$SCRIPT_DIR"
        printf 'export MUJOCO_GL=egl WANDB_DIR=%q WANDB_RUN_GROUP=%q WANDB_NAME=%q PYTHONHASHSEED=%q\n' \
            "${run_dir}/wandb" "$GROUP" "$run_name" "$seed"
        printf 'exec %q -u train_act_in_sim.py' "$PY"
        printf ' --experimental-act-ppo --init-checkpoint %q' "$checkpoint"
        printf ' --parallel-envs 12 --rollout-chunks-per-env 2 --minibatch-size 64 --ppo-epochs 1'
        printf ' --chunk-size 30 --max-steps-per-episode 150 --steps-per-action 1'
        printf ' --policy-lr 1e-6 --critic-lr 5e-5 --log-std-init -2'
        printf ' --episodes 200 --snapshot-every 10 --eval-episodes 0 --seed %q' "$seed"
        printf ' --checkpoint-path %q' "${run_dir}/act_sim_ppo_checkpoint.pt"
        if [[ "$appearance" == "fixed" ]]; then
            printf ' --no-randomize-appearance'
        else
            printf ' --randomize-appearance'
        fi
        if [[ "$block_mode" == "narrow" ]]; then
            printf ' --randomize-block-reset --block-dist-range 0.22 0.26 --block-angle-range -10 10'
        else
            printf ' --curriculum-fixed-block'
        fi
        printf ' --headless --no-render > %q 2>&1\n' "${run_dir}/train.log"
    } > "$command_file"
    chmod +x "$command_file"
    "$TSP" -L "$run_name" "$command_file"
}

enqueue_run new_app_fixed_s11    "$NEW_PRETRAIN" randomized fixed  11 yes
enqueue_run old_app_fixed_s11    "$OLD_PRETRAIN" randomized fixed  11 no
enqueue_run new_noapp_fixed_s11  "$NEW_PRETRAIN" fixed      fixed  11 no
enqueue_run old_noapp_fixed_s11  "$OLD_PRETRAIN" fixed      fixed  11 no
enqueue_run new_app_fixed_s29    "$NEW_PRETRAIN" randomized fixed  29 no
enqueue_run old_app_fixed_s29    "$OLD_PRETRAIN" randomized fixed  29 no
enqueue_run new_noapp_fixed_s29  "$NEW_PRETRAIN" fixed      fixed  29 no
enqueue_run old_noapp_fixed_s29  "$OLD_PRETRAIN" fixed      fixed  29 no
enqueue_run new_app_narrow_s11   "$NEW_PRETRAIN" randomized narrow 11 no
enqueue_run new_app_narrow_s29   "$NEW_PRETRAIN" randomized narrow 29 no

"$TSP"
