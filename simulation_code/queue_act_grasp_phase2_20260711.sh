#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/win10ubuntu/miniforge3/envs/lerobot/bin/python}"
TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
DRY_RUN="${DRY_RUN:-0}"
source "${SCRIPT_DIR}/act_policy_mode.sh"
WINNER1="${WINNER1:?Set WINNER1 to one of A B C D E F}"
WINNER2="${WINNER2:?Set WINNER2 to one of A B C D E F}"
GROUP="${WANDB_RUN_GROUP:-act-lead3-grasp-phase2-20260711}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/train/act_lead3_grasp_phase2_20260711}"
PRETRAIN="${SCRIPT_DIR}/outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model"

declare -A LR=( [A]=1e-6 [B]=5e-7 [C]=1e-6 [D]=1e-6 [E]=1e-6 [F]=1e-6 )
declare -A STD=( [A]=-2.0 [B]=-2.0 [C]=-2.356675 [D]=-2.0 [E]=-2.0 [F]=-2.356675 )
declare -A PROFILE=( [A]=baseline [B]=baseline [C]=baseline [D]=jaw_quality [E]=jaw_quality_rebalanced [F]=jaw_quality )

for winner in "$WINNER1" "$WINNER2"; do
    [[ -n "${LR[$winner]:-}" ]] || { echo "Unknown winner ID: $winner" >&2; exit 1; }
done
[[ "$WINNER1" != "$WINNER2" ]] || { echo "WINNER1 and WINNER2 must differ" >&2; exit 1; }
[[ -x "$PY" && -x "$TSP" && -f "$PRETRAIN/model.safetensors" ]] || { echo "Missing runtime or pretrain" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT"
act_print_policy_mode
act_configure_queue "$TSP" "$DRY_RUN"

enqueue_train() {
    local id="$1" seed="$2" lr="${LR[$1]}" log_std="${STD[$1]}" profile="${PROFILE[$1]}"
    local name="${id}_${profile}_s${seed}"
    local run_dir="${OUTPUT_ROOT}/${name}"
    local command_file="${run_dir}/command.sh"
    mkdir -p "$run_dir"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        printf 'cd %q\n' "$SCRIPT_DIR"
        printf 'export MUJOCO_GL=egl WANDB_DIR=%q WANDB_RUN_GROUP=%q WANDB_NAME=%q PYTHONHASHSEED=%q\n' "${run_dir}/wandb" "$GROUP" "$name" "$seed"
        printf 'exec %q -u train_act_in_sim.py --experimental-act-ppo --init-checkpoint %q' "$PY" "$PRETRAIN"
        printf ' --reward-profile %q --policy-lr %q --critic-lr 5e-5 --log-std-init %q' "$profile" "$lr" "$log_std"
        act_append_trainer_resource_args
        printf ' --chunk-size 30 --max-steps-per-episode 150 --steps-per-action 1'
        printf ' --episodes 190 --snapshot-every 10 --eval-episodes 0 --seed %q' "$seed"
        printf ' --checkpoint-path %q --no-randomize-appearance --curriculum-fixed-block --headless --no-render' "${run_dir}/act_sim_ppo_checkpoint.pt"
        printf ' > %q 2>&1\n' "${run_dir}/train.log"
    } > "$command_file"
    chmod +x "$command_file"
    act_enqueue_training "$TSP" "$DRY_RUN" "$name" "$command_file"
}

enqueue_eval() {
    local id="$1" seed="$2" profile="${PROFILE[$1]}"
    local name="${id}_${profile}_s${seed}"
    local run_dir="${OUTPUT_ROOT}/${name}"
    local eval_dir="${run_dir}/evaluation"
    local command_file="${eval_dir}/command.sh"
    mkdir -p "$eval_dir"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        printf 'cd %q\n' "$SCRIPT_DIR"
        echo 'export MUJOCO_GL=egl'
        for ep in 0039 0079 0119 0159 0189; do
            for mode in deterministic stochastic; do
                printf '%q -u run_act_ppo_sim_inference.py --resume %q --init-checkpoint %q' "$PY" "${run_dir}/act_sim_ppo_checkpoint_ep${ep}.pt" "$PRETRAIN"
                printf ' --reward-profile %q --episodes 5 --max-steps-per-episode 150 --chunk-size 30 --steps-per-action 1' "$profile"
                printf ' --headless --no-randomize-appearance --curriculum-fixed-block --seed 2606'
                [[ "$mode" == "stochastic" ]] && printf ' --stochastic'
                printf ' --output-json %q > %q 2>&1\n' "${eval_dir}/ep${ep}_${mode}.json" "${eval_dir}/ep${ep}_${mode}.log"
            done
        done
    } > "$command_file"
    chmod +x "$command_file"
    act_enqueue_exclusive "$TSP" "$DRY_RUN" "eval_${name}" "$command_file"
}

enqueue_train "$WINNER1" 83
enqueue_train "$WINNER2" 83
enqueue_train "$WINNER2" 101
enqueue_train "$WINNER1" 101
enqueue_eval "$WINNER1" 83
enqueue_eval "$WINNER2" 83
enqueue_eval "$WINNER2" 101
enqueue_eval "$WINNER1" 101

echo "Phase-two output root: $OUTPUT_ROOT"
if [[ "$DRY_RUN" != "1" ]]; then "$TSP"; fi
