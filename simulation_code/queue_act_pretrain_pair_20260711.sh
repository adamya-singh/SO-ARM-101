#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/win10ubuntu/miniforge3/envs/lerobot/bin/python}"
TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
DRY_RUN="${DRY_RUN:-0}"
source "${SCRIPT_DIR}/act_policy_mode.sh"
GROUP="${WANDB_RUN_GROUP:-act-pretrain-paired-ablation-20260711}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/train/act_pretrain_paired_ablation_20260711_s5}"
NOLEAD="${SCRIPT_DIR}/outputs/train/act_so101_corrected_30_b32_20260621_160923/checkpoints/026020/pretrained_model"
LEAD3="${SCRIPT_DIR}/outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model"
NORMALIZER="policy_preprocessor_step_3_normalizer_processor.safetensors"

[[ -x "$PY" ]] || { echo "Missing Python: $PY" >&2; exit 1; }
[[ -x "$TSP" ]] || { echo "Missing task spooler: $TSP" >&2; exit 1; }
for checkpoint in "$NOLEAD" "$LEAD3"; do
    [[ -f "$checkpoint/model.safetensors" ]] || { echo "Missing model: $checkpoint" >&2; exit 1; }
    [[ -f "$checkpoint/$NORMALIZER" ]] || { echo "Missing normalizer: $checkpoint/$NORMALIZER" >&2; exit 1; }
done

mkdir -p "$OUTPUT_ROOT"
act_print_policy_mode
act_configure_queue "$TSP" "$DRY_RUN"

enqueue() {
    local name="$1"
    local checkpoint="$2"
    local run_dir="${OUTPUT_ROOT}/${name}"
    local command_file="${run_dir}/command.sh"
    mkdir -p "$run_dir"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        printf 'cd %q\n' "$SCRIPT_DIR"
        printf 'export MUJOCO_GL=egl WANDB_DIR=%q WANDB_RUN_GROUP=%q WANDB_NAME=%q PYTHONHASHSEED=5\n' \
            "${run_dir}/wandb" "$GROUP" "$name"
        printf 'exec %q -u train_act_in_sim.py' "$PY"
        printf ' --experimental-act-ppo --init-checkpoint %q' "$checkpoint"
        act_append_trainer_resource_args
        printf ' --chunk-size 30 --max-steps-per-episode 150 --steps-per-action 1'
        printf ' --policy-lr 1e-6 --critic-lr 5e-5 --log-std-init -2'
        printf ' --episodes 200 --snapshot-every 10 --eval-episodes 0 --seed 5'
        printf ' --checkpoint-path %q' "${run_dir}/act_sim_ppo_checkpoint.pt"
        printf ' --no-randomize-appearance --curriculum-fixed-block --headless --no-render'
        printf ' > %q 2>&1\n' "${run_dir}/train.log"
    } > "$command_file"
    chmod +x "$command_file"
    act_enqueue_training "$TSP" "$DRY_RUN" "$name" "$command_file"
}

enqueue nolead_lr1e6_s5 "$NOLEAD"
enqueue lead3_lr1e6_s5 "$LEAD3"

echo "Queued paired pretrain ablation in $OUTPUT_ROOT"
if [[ "$DRY_RUN" != "1" ]]; then "$TSP"; fi
