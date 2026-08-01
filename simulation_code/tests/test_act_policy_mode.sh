#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HELPER="${SCRIPT_DIR}/act_policy_mode.sh"

fail() {
    echo "FAIL: $*" >&2
    exit 1
}

assert_mode() {
    local mode="$1" expected_slots="$2" expected_envs="$3" expected_chunks="$4"
    local output
    output="$({ ACT_POLICY_MODE="$mode" source "$HELPER"; printf '%s %s %s %s\n' \
        "$ACT_QUEUE_SLOTS" "$ACT_PARALLEL_ENVS" "$ACT_ROLLOUT_CHUNKS_PER_ENV" "$ACT_SAMPLES_PER_UPDATE"; })"
    [[ "$output" == "$expected_slots $expected_envs $expected_chunks 24" ]] || \
        fail "$mode resolved to '$output'"
}

assert_mode single 1 12 2
assert_mode dual 2 6 4

default_output="$({ unset ACT_POLICY_MODE; source "$HELPER"; printf '%s %s %s\n' \
    "$ACT_POLICY_MODE" "$ACT_PARALLEL_ENVS" "$ACT_ROLLOUT_CHUNKS_PER_ENV"; })"
[[ "$default_output" == "single 12 2" ]] || fail "default mode resolved to '$default_output'"

if ACT_POLICY_MODE=invalid bash -c 'source "$1"' _ "$HELPER" 2>/dev/null; then
    fail "invalid mode was accepted"
fi

tmp_dir="$(mktemp -d)"
trap 'rm -rf "$tmp_dir"' EXIT
fake_tsp="${tmp_dir}/tsp"
calls="${tmp_dir}/calls"
cat > "$fake_tsp" <<'EOF'
#!/usr/bin/env bash
printf '%q ' "$@" >> "$CALLS"
printf '\n' >> "$CALLS"
EOF
chmod +x "$fake_tsp"

CALLS="$calls" ACT_POLICY_MODE=dual bash -c '
    set -euo pipefail
    source "$1"
    act_configure_queue "$2" 0
    act_enqueue_training "$2" 0 train /tmp/train.sh
    act_enqueue_exclusive "$2" 0 eval /tmp/eval.sh
' _ "$HELPER" "$fake_tsp"

mapfile -t recorded < "$calls"
[[ "${recorded[0]}" == "-S 2 " ]] || fail "dual queue capacity call was '${recorded[0]}'"
[[ "${recorded[1]}" == "-N 1 -L train /tmp/train.sh " ]] || fail "training enqueue was '${recorded[1]}'"
[[ "${recorded[2]}" == "-N 2 -L eval /tmp/eval.sh " ]] || fail "exclusive enqueue was '${recorded[2]}'"

training_queues=(
    "${SCRIPT_DIR}/queue_act_coordinate_lr_sweep_20260710.sh"
    "${SCRIPT_DIR}/queue_act_grasp_phase1_20260711.sh"
    "${SCRIPT_DIR}/queue_act_grasp_phase2_20260711.sh"
    "${SCRIPT_DIR}/queue_act_posttrain_ablation_20260710.sh"
    "${SCRIPT_DIR}/queue_act_pretrain_pair_20260711.sh"
)
for queue in "${training_queues[@]}"; do
    grep -Fq 'source "${SCRIPT_DIR}/act_policy_mode.sh"' "$queue" || \
        fail "$(basename "$queue") does not load the shared mode"
    grep -Fq 'act_append_trainer_resource_args' "$queue" || \
        fail "$(basename "$queue") does not use shared trainer arguments"
    if grep -Eq -- '--parallel-envs (6|12)|--rollout-chunks-per-env (2|4)' "$queue"; then
        fail "$(basename "$queue") contains duplicated resource settings"
    fi
done

grep -Fq 'act_enqueue_exclusive "$TSP" "$DRY_RUN" "eval_${name}"' \
    "${SCRIPT_DIR}/queue_act_grasp_phase2_20260711.sh" || \
    fail "mixed phase-two queue does not serialize evaluation"

echo "ACT policy mode shell tests passed."
