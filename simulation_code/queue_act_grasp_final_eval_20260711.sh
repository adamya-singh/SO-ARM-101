#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-/home/win10ubuntu/miniforge3/envs/lerobot/bin/python}"
TSP="${TSP:-/home/win10ubuntu/.local/bin/tsp}"
DRY_RUN="${DRY_RUN:-0}"
FINALIST1="${FINALIST1:?Set FINALIST1 to the first .pt checkpoint}"
FINALIST2="${FINALIST2:?Set FINALIST2 to the second .pt checkpoint}"
PROFILE1="${PROFILE1:?Set PROFILE1 to the first reward profile}"
PROFILE2="${PROFILE2:?Set PROFILE2 to the second reward profile}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/outputs/eval/act_lead3_grasp_finalists_20260711}"
PRETRAIN="${SCRIPT_DIR}/outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model"

[[ -x "$PY" && -x "$TSP" && -f "$PRETRAIN/model.safetensors" ]] || { echo "Missing runtime or pretrain" >&2; exit 1; }
for checkpoint in "$FINALIST1" "$FINALIST2"; do [[ -f "$checkpoint" ]] || { echo "Missing finalist: $checkpoint" >&2; exit 1; }; done
for profile in "$PROFILE1" "$PROFILE2"; do
    "$PY" -c 'import sys; from train_act_in_sim import resolve_reward_profile; resolve_reward_profile(sys.argv[1])' "$profile"
done

mkdir -p "$OUTPUT_ROOT"
if [[ "$DRY_RUN" != "1" ]]; then "$TSP" -S 1; fi

enqueue_finalist() {
    local label="$1" checkpoint="$2" profile="$3"
    local run_dir="${OUTPUT_ROOT}/${label}"
    local command_file="${run_dir}/command.sh"
    mkdir -p "$run_dir"
    {
        echo '#!/usr/bin/env bash'
        echo 'set -euo pipefail'
        printf 'cd %q\n' "$SCRIPT_DIR"
        echo 'export MUJOCO_GL=egl'
        for mode in deterministic_fixed stochastic_fixed stochastic_narrow; do
            printf '%q -u run_act_ppo_sim_inference.py --resume %q --init-checkpoint %q' "$PY" "$checkpoint" "$PRETRAIN"
            printf ' --reward-profile %q --episodes 30 --max-steps-per-episode 150 --chunk-size 30 --steps-per-action 1' "$profile"
            printf ' --headless --no-randomize-appearance --curriculum-fixed-block --seed 76011'
            [[ "$mode" != "deterministic_fixed" ]] && printf ' --stochastic'
            [[ "$mode" == "stochastic_narrow" ]] && printf ' --randomize-block-reset --block-dist-range 0.22 0.26 --block-angle-range -10 10'
            printf ' --output-json %q > %q 2>&1\n' "${run_dir}/${mode}.json" "${run_dir}/${mode}.log"
        done
    } > "$command_file"
    chmod +x "$command_file"
    if [[ "$DRY_RUN" == "1" ]]; then echo "$label: $command_file"; else "$TSP" -L "$label" "$command_file"; fi
}

enqueue_finalist finalist1 "$FINALIST1" "$PROFILE1"
enqueue_finalist finalist2 "$FINALIST2" "$PROFILE2"

echo "Final evaluation output root: $OUTPUT_ROOT"
if [[ "$DRY_RUN" != "1" ]]; then "$TSP"; fi
