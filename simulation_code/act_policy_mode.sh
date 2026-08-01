#!/usr/bin/env bash

# Shared resource presets for ACT PPO queue scripts.
# The caller is expected to enable `set -euo pipefail` before sourcing this file.

ACT_POLICY_MODE="${ACT_POLICY_MODE:-single}"
ACT_MINIBATCH_SIZE=64
ACT_PPO_EPOCHS=1

case "$ACT_POLICY_MODE" in
    single)
        ACT_QUEUE_SLOTS=1
        ACT_PARALLEL_ENVS=12
        ACT_ROLLOUT_CHUNKS_PER_ENV=2
        ;;
    dual)
        ACT_QUEUE_SLOTS=2
        ACT_PARALLEL_ENVS=6
        ACT_ROLLOUT_CHUNKS_PER_ENV=4
        ;;
    *)
        echo "Invalid ACT_POLICY_MODE='$ACT_POLICY_MODE'; expected 'single' or 'dual'." >&2
        return 2 2>/dev/null || exit 2
        ;;
esac

ACT_SAMPLES_PER_UPDATE=$((ACT_PARALLEL_ENVS * ACT_ROLLOUT_CHUNKS_PER_ENV))
ACT_TRAINER_RESOURCE_ARGS=(
    --parallel-envs "$ACT_PARALLEL_ENVS"
    --rollout-chunks-per-env "$ACT_ROLLOUT_CHUNKS_PER_ENV"
    --minibatch-size "$ACT_MINIBATCH_SIZE"
    --ppo-epochs "$ACT_PPO_EPOCHS"
)

act_print_policy_mode() {
    printf 'ACT policy mode: mode=%s concurrency=%s envs_per_policy=%s chunks_per_env=%s samples_per_update=%s\n' \
        "$ACT_POLICY_MODE" "$ACT_QUEUE_SLOTS" "$ACT_PARALLEL_ENVS" \
        "$ACT_ROLLOUT_CHUNKS_PER_ENV" "$ACT_SAMPLES_PER_UPDATE"
}

act_configure_queue() {
    local tsp="$1" dry_run="${2:-0}"
    if [[ "$dry_run" != "1" ]]; then
        "$tsp" -S "$ACT_QUEUE_SLOTS"
    fi
}

act_append_trainer_resource_args() {
    printf ' %q' "${ACT_TRAINER_RESOURCE_ARGS[@]}"
}

act_enqueue_training() {
    local tsp="$1" dry_run="$2" label="$3" command_file="$4"
    if [[ "$dry_run" == "1" ]]; then
        printf '%s: %s\n' "$label" "$command_file"
    else
        "$tsp" -N 1 -L "$label" "$command_file"
    fi
}

act_enqueue_exclusive() {
    local tsp="$1" dry_run="$2" label="$3" command_file="$4"
    if [[ "$dry_run" == "1" ]]; then
        printf '%s (exclusive_slots=%s): %s\n' "$label" "$ACT_QUEUE_SLOTS" "$command_file"
    else
        "$tsp" -N "$ACT_QUEUE_SLOTS" -L "$label" "$command_file"
    fi
}
