#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/win10ubuntu/miniforge3/envs/lerobot/bin/python}"
TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
DRY_RUN="${DRY_RUN:-0}"
source "${SCRIPT_DIR}/act_policy_mode.sh"
TIMEOUT="${TIMEOUT:-/usr/bin/timeout}"
START_AT="${START_AT:-$(date '+%Y-%m-%d %H:%M:%S')}"
START_EPOCH="$(date -d "$START_AT" +%s)"
END_EPOCH="${END_EPOCH:-$((START_EPOCH + 8 * 60 * 60))}"
END_AT="$(date -d "@$END_EPOCH" '+%Y-%m-%d %H:%M:%S %z')"
GROUP="${WANDB_RUN_GROUP:-act-posttrain-coordinate-lr-sweep-20260710}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/train/act_posttrain_coordinate_lr_sweep_20260710}"

LEAD3_PRETRAIN="${SCRIPT_DIR}/outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model"
OLD_PRETRAIN="${SCRIPT_DIR}/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model"
NORMALIZER_FILE="policy_preprocessor_step_3_normalizer_processor.safetensors"

[[ -x "$PY" ]] || { echo "Python is not executable: $PY" >&2; exit 1; }
[[ -x "$TSP" ]] || { echo "Task Spooler is not executable: $TSP" >&2; exit 1; }
[[ -x "$TIMEOUT" ]] || { echo "timeout is not executable: $TIMEOUT" >&2; exit 1; }
for checkpoint in "$LEAD3_PRETRAIN" "$OLD_PRETRAIN"; do
    [[ -f "$checkpoint/model.safetensors" ]] || {
        echo "Missing pretrained model: $checkpoint/model.safetensors" >&2
        exit 1
    }
    [[ -f "$checkpoint/$NORMALIZER_FILE" ]] || {
        echo "Missing ACT normalization statistics: $checkpoint/$NORMALIZER_FILE" >&2
        exit 1
    }
done
if (( END_EPOCH <= START_EPOCH )); then
    echo "END_EPOCH must be later than START_AT" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT"
cat > "${OUTPUT_ROOT}/sweep_metadata.txt" <<EOF
start_at=$START_AT
start_epoch=$START_EPOCH
end_at=$END_AT
end_epoch=$END_EPOCH
wandb_group=$GROUP
act_policy_mode=$ACT_POLICY_MODE
concurrency=$ACT_QUEUE_SLOTS
parallel_envs=$ACT_PARALLEL_ENVS
rollout_chunks_per_env=$ACT_ROLLOUT_CHUNKS_PER_ENV
lead3_pretrain=$LEAD3_PRETRAIN
old_pretrain=$OLD_PRETRAIN
EOF

act_print_policy_mode
act_configure_queue "$TSP" "$DRY_RUN"

enqueue_run() {
    local run_name="$1"
    local checkpoint="$2"
    local policy_lr="$3"
    local seed="$4"
    local episodes="$5"
    local run_dir="${OUTPUT_ROOT}/${run_name}"
    local command_file="${run_dir}/command.sh"

    mkdir -p "$run_dir"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        printf 'START_EPOCH=%q\n' "$START_EPOCH"
        printf 'END_EPOCH=%q\n' "$END_EPOCH"
        echo 'while (( $(date +%s) < START_EPOCH )); do sleep 5; done'
        echo 'remaining=$((END_EPOCH - $(date +%s)))'
        echo 'if (( remaining <= 0 )); then echo "Shared eight-hour deadline already reached; skipping."; exit 0; fi'
        printf 'cd %q\n' "$SCRIPT_DIR"
        printf 'export MUJOCO_GL=egl WANDB_DIR=%q WANDB_RUN_GROUP=%q WANDB_NAME=%q PYTHONHASHSEED=%q\n' \
            "${run_dir}/wandb" "$GROUP" "$run_name" "$seed"
        echo 'set +e'
        printf '%q --foreground --signal=INT --kill-after=60s "${remaining}s" ' "$TIMEOUT"
        printf '%q -u train_act_in_sim.py' "$PY"
        printf ' --experimental-act-ppo --init-checkpoint %q' "$checkpoint"
        act_append_trainer_resource_args
        printf ' --chunk-size 30 --max-steps-per-episode 150 --steps-per-action 1'
        printf ' --policy-lr %q --critic-lr 5e-5 --log-std-init -2' "$policy_lr"
        printf ' --episodes %q --snapshot-every 10 --eval-episodes 0 --seed %q' "$episodes" "$seed"
        printf ' --checkpoint-path %q' "${run_dir}/act_sim_ppo_checkpoint.pt"
        printf ' --no-randomize-appearance --curriculum-fixed-block --headless --no-render'
        printf ' > %q 2>&1\n' "${run_dir}/train.log"
        echo 'status=$?'
        echo 'set -e'
        echo 'if (( status == 124 || status == 130 )); then'
        echo '  echo "Stopped at the shared eight-hour deadline after a graceful interrupt." >> train.log'
        echo '  exit 0'
        echo 'fi'
        echo 'exit "$status"'
    } > "$command_file"
    chmod +x "$command_file"
    act_enqueue_training "$TSP" "$DRY_RUN" "$run_name" "$command_file"
}

# Balanced 2 pretrains x 2 actor learning rates x 3 seeds study.
enqueue_run lead3_lr1e6_s11 "$LEAD3_PRETRAIN" 1e-6 11 190
enqueue_run old_lr1e6_s11   "$OLD_PRETRAIN"   1e-6 11 190
enqueue_run lead3_lr3e6_s11 "$LEAD3_PRETRAIN" 3e-6 11 190
enqueue_run old_lr3e6_s11   "$OLD_PRETRAIN"   3e-6 11 190

# Reverse condition order for seed 29 to reduce wall-clock-order confounding.
enqueue_run old_lr3e6_s29   "$OLD_PRETRAIN"   3e-6 29 190
enqueue_run lead3_lr3e6_s29 "$LEAD3_PRETRAIN" 3e-6 29 190
enqueue_run old_lr1e6_s29   "$OLD_PRETRAIN"   1e-6 29 190
enqueue_run lead3_lr1e6_s29 "$LEAD3_PRETRAIN" 1e-6 29 190

enqueue_run lead3_lr1e6_s47 "$LEAD3_PRETRAIN" 1e-6 47 190
enqueue_run old_lr1e6_s47   "$OLD_PRETRAIN"   1e-6 47 190
enqueue_run lead3_lr3e6_s47 "$LEAD3_PRETRAIN" 3e-6 47 190
enqueue_run old_lr3e6_s47   "$OLD_PRETRAIN"   3e-6 47 190

# Starts only if the balanced matrix leaves time. The same deadline bounds it.
enqueue_run overflow_lead3_lr1e6_s71 "$LEAD3_PRETRAIN" 1e-6 71 190

echo "Queued corrected-coordinate ACT PPO sweep."
echo "Start: $START_AT"
echo "Hard deadline: $END_AT"
echo "Output root: $OUTPUT_ROOT"
if [[ "$DRY_RUN" != "1" ]]; then "$TSP"; fi
