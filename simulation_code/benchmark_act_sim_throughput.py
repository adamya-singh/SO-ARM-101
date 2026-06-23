#!/usr/bin/env python3
"""Discover high-throughput settings for ACT PPO training in SO-101 MuJoCo."""

from __future__ import annotations

import argparse
import copy
import os
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import train_act_in_sim as train


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
NOTES_DIR = PROJECT_DIR / "notes"
CORRECTED_INIT_CHECKPOINT = (
    SCRIPT_DIR
    / "outputs"
    / "train"
    / "act_so101_corrected_30_b32_20260621_160923"
    / "checkpoints"
    / "026020"
    / "pretrained_model"
)


@dataclass
class TrialResult:
    parallel_envs: int
    rollout_chunks_per_env: int
    minibatch_size: int
    ppo_epochs: int
    status: str
    env_steps_per_sec: float = 0.0
    chunks_per_sec: float = 0.0
    gpu_util_percent: float = 0.0
    gpu_power_w: float = 0.0
    gpu_memory_reserved_mb: float = 0.0
    iteration_seconds: float = 0.0
    policy_forward_sec: float = 0.0
    env_step_sec: float = 0.0
    ppo_update_sec: float = 0.0
    rollout_return: float = 0.0
    success: float = 0.0
    contact_steps: float = 0.0
    grasp_steps: float = 0.0
    lift_steps: float = 0.0
    error: str = ""


def parse_csv_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("expected at least one integer")
    if any(item < 1 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def default_init_checkpoint() -> Path:
    return CORRECTED_INIT_CHECKPOINT if CORRECTED_INIT_CHECKPOINT.exists() else train.DEFAULT_INIT_CHECKPOINT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ACT-in-sim PPO throughput and write the best training settings to a local note. "
            "This is separate from train_act_in_sim.py so normal training runs stay fixed and reproducible."
        )
    )
    parser.add_argument("--init-checkpoint", type=Path, default=default_init_checkpoint())
    parser.add_argument("--sweep-envs", type=parse_csv_ints, default=parse_csv_ints("1,2,4,8,12"))
    parser.add_argument("--sweep-rollout-chunks", type=parse_csv_ints, default=parse_csv_ints("2,4,8"))
    parser.add_argument("--sweep-minibatches", type=parse_csv_ints, default=parse_csv_ints("16,32,64"))
    parser.add_argument("--sweep-ppo-epochs", type=parse_csv_ints, default=parse_csv_ints("1,2"))
    parser.add_argument("--target-gpu-util", type=float, default=70.0)
    parser.add_argument("--benchmark-iterations", type=int, default=2)
    parser.add_argument("--warmup-iterations", type=int, default=1)
    parser.add_argument("--min-env-step-improvement", type=float, default=3.0)
    parser.add_argument("--baseline-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps-per-episode", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=30)
    parser.add_argument("--steps-per-action", type=int, default=1)
    parser.add_argument("--policy-lr", type=float, default=1e-5)
    parser.add_argument("--critic-lr", type=float, default=1e-4)
    parser.add_argument("--log-std-init", type=float, default=-2.0)
    parser.add_argument("--clip-epsilon", type=float, default=0.1)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--randomize-block-reset", action="store_true")
    parser.add_argument(
        "--curriculum-fixed-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Match train_act_in_sim.py default fixed block curriculum.",
    )
    parser.add_argument("--checkpoint-path", type=Path, default=SCRIPT_DIR / "act_sim_ppo_checkpoint.pt")
    parser.add_argument("--output-note", type=Path, default=None)
    parser.add_argument("--device", default=None, help="Override torch device, for example cuda or cpu.")
    return parser.parse_args()


def make_train_args(args: argparse.Namespace, result: TrialResult | None = None) -> argparse.Namespace:
    parallel_envs = result.parallel_envs if result is not None else args.baseline_envs
    rollout_chunks_per_env = result.rollout_chunks_per_env if result is not None else min(args.sweep_rollout_chunks)
    minibatch_size = result.minibatch_size if result is not None else min(args.sweep_minibatches)
    ppo_epochs = result.ppo_epochs if result is not None else min(args.sweep_ppo_epochs)
    return argparse.Namespace(
        experimental_act_ppo=True,
        init_checkpoint=args.init_checkpoint,
        episodes=args.episodes,
        max_steps_per_episode=args.max_steps_per_episode,
        chunk_size=args.chunk_size,
        steps_per_action=args.steps_per_action,
        policy_lr=args.policy_lr,
        critic_lr=args.critic_lr,
        log_std_init=args.log_std_init,
        clip_epsilon=args.clip_epsilon,
        gae_lambda=args.gae_lambda,
        gamma=args.gamma,
        ppo_epochs=ppo_epochs,
        minibatch_size=minibatch_size,
        parallel_envs=parallel_envs,
        rollout_chunks_per_env=rollout_chunks_per_env,
        randomize_block_reset=args.randomize_block_reset,
        curriculum_fixed_block=args.curriculum_fixed_block,
        no_render=True,
        headless=True,
        no_wandb=True,
        resume=None,
        checkpoint_path=args.checkpoint_path,
        eval_episodes=0,
    )


def clone_state(module: Any) -> dict[str, Any]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def restore_state(module: Any, state: dict[str, Any], device: Any) -> None:
    module.load_state_dict({key: value.to(device) for key, value in state.items()})


def average_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = sorted({key for metric in metrics for key in metric})
    averaged: dict[str, float] = {}
    for key in keys:
        values = [float(metric[key]) for metric in metrics if key in metric]
        if values:
            averaged[key] = sum(values) / len(values)
    return averaged


def clear_cuda_cache(device: Any) -> None:
    if getattr(device, "type", None) == "cuda":
        train.torch.cuda.empty_cache()


def benchmark_trial(
    candidate: TrialResult,
    args: argparse.Namespace,
    device: Any,
    policy: Any,
    critic: Any,
    policy_optimizer: Any,
    critic_optimizer: Any,
    base_policy_state: dict[str, Any],
    base_critic_state: dict[str, Any],
    base_policy_optimizer_state: dict[str, Any],
    base_critic_optimizer_state: dict[str, Any],
) -> TrialResult:
    train_args = make_train_args(args, candidate)
    env = None
    try:
        restore_state(policy, base_policy_state, device)
        restore_state(critic, base_critic_state, device)
        policy_optimizer.load_state_dict(copy.deepcopy(base_policy_optimizer_state))
        critic_optimizer.load_state_dict(copy.deepcopy(base_critic_optimizer_state))
        clear_cuda_cache(device)

        block_pos = train.block_position(train_args)
        env = train.make_training_env(train_args, block_pos)
        measured: list[dict[str, float]] = []
        total_iterations = max(0, args.warmup_iterations) + max(1, args.benchmark_iterations)

        for iteration in range(total_iterations):
            metrics, _rollout = train.run_train_iteration(
                env,
                policy,
                critic,
                policy_optimizer,
                critic_optimizer,
                device,
                train_args,
            )
            if getattr(device, "type", None) == "cuda":
                train.torch.cuda.synchronize(device)
            if iteration >= args.warmup_iterations:
                measured.append(metrics)

        avg = average_metrics(measured)
        candidate.status = "ok"
        candidate.env_steps_per_sec = avg.get("throughput/env_steps_per_sec", 0.0)
        candidate.chunks_per_sec = avg.get("throughput/chunks_per_sec", 0.0)
        candidate.gpu_util_percent = avg.get("gpu/util_percent", 0.0)
        candidate.gpu_power_w = avg.get("gpu/power_w", 0.0)
        candidate.gpu_memory_reserved_mb = avg.get("gpu/memory_reserved_mb", 0.0)
        candidate.iteration_seconds = avg.get("time/iteration_seconds", 0.0)
        candidate.policy_forward_sec = avg.get("time/policy_forward_sec", 0.0)
        candidate.env_step_sec = avg.get("time/env_step_sec", 0.0)
        candidate.ppo_update_sec = avg.get("train/ppo_update_sec", 0.0)
        candidate.rollout_return = avg.get("rollout/return", 0.0)
        candidate.success = avg.get("rollout/success", 0.0)
        candidate.contact_steps = avg.get("rollout/contact_steps", 0.0)
        candidate.grasp_steps = avg.get("rollout/grasp_steps", 0.0)
        candidate.lift_steps = avg.get("rollout/lift_steps", 0.0)
    except RuntimeError as exc:
        lowered = str(exc).lower()
        candidate.status = "oom" if "out of memory" in lowered else "failed"
        candidate.error = str(exc).splitlines()[0][:240]
        clear_cuda_cache(device)
    except Exception as exc:
        candidate.status = "failed"
        candidate.error = str(exc).splitlines()[0][:240]
        clear_cuda_cache(device)
    finally:
        train.close_training_env(env)
    return candidate


def rank_results(results: list[TrialResult], target_gpu_util: float) -> TrialResult | None:
    successful = [result for result in results if result.status == "ok" and result.env_steps_per_sec > 0]
    if not successful:
        return None
    gpu_qualified = [result for result in successful if result.gpu_util_percent >= target_gpu_util]
    pool = gpu_qualified or successful
    return max(pool, key=lambda item: (item.env_steps_per_sec, item.gpu_util_percent, item.chunks_per_sec))


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def training_command(args: argparse.Namespace, winner: TrialResult) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "train_act_in_sim.py"),
        "--experimental-act-ppo",
        "--init-checkpoint",
        str(args.init_checkpoint),
        "--parallel-envs",
        str(winner.parallel_envs),
        "--rollout-chunks-per-env",
        str(winner.rollout_chunks_per_env),
        "--minibatch-size",
        str(winner.minibatch_size),
        "--ppo-epochs",
        str(winner.ppo_epochs),
        "--chunk-size",
        str(args.chunk_size),
        "--max-steps-per-episode",
        str(args.max_steps_per_episode),
        "--steps-per-action",
        str(args.steps_per_action),
        "--episodes",
        str(args.episodes),
        "--checkpoint-path",
        str(args.checkpoint_path),
        "--headless",
        "--no-render",
    ]


def result_table(results: list[TrialResult]) -> str:
    lines = [
        "| envs | rollout chunks/env | minibatch | ppo epochs | status | env steps/s | chunks/s | gpu util % | power W | iter s | policy s | env s | ppo s | health |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        health = f"contact={result.contact_steps:.1f}, grasp={result.grasp_steps:.1f}, lift={result.lift_steps:.1f}"
        if result.error:
            health = result.error.replace("|", "/")
        lines.append(
            f"| {result.parallel_envs} | {result.rollout_chunks_per_env} | {result.minibatch_size} | "
            f"{result.ppo_epochs} | {result.status} | {result.env_steps_per_sec:.1f} | "
            f"{result.chunks_per_sec:.2f} | {result.gpu_util_percent:.1f} | {result.gpu_power_w:.1f} | "
            f"{result.iteration_seconds:.2f} | {result.policy_forward_sec:.2f} | {result.env_step_sec:.2f} | "
            f"{result.ppo_update_sec:.2f} | {health} |"
        )
    return "\n".join(lines)


def write_note(args: argparse.Namespace, results: list[TrialResult], winner: TrialResult | None, baseline: TrialResult | None) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    note_path = args.output_note or (NOTES_DIR / f"act-sim-ppo-throughput-sweep-{timestamp}.md")
    note_path.parent.mkdir(parents=True, exist_ok=True)

    command = shell_join(training_command(args, winner)) if winner is not None else "No successful candidate."
    baseline_rate = baseline.env_steps_per_sec if baseline and baseline.status == "ok" else 0.0
    improvement = (winner.env_steps_per_sec / baseline_rate) if winner and baseline_rate > 0 else 0.0
    target_text = "met" if winner and winner.gpu_util_percent >= args.target_gpu_util else "not met"
    min_improvement_text = (
        "met" if improvement >= args.min_env_step_improvement else f"not met ({improvement:.2f}x)"
    )

    content = [
        "# ACT Sim PPO Throughput Sweep",
        "",
        f"- Created: {timestamp}",
        f"- Init checkpoint: `{args.init_checkpoint}`",
        f"- Target GPU utilization: `{args.target_gpu_util:.1f}%` ({target_text})",
        f"- Minimum sequential improvement target: `{args.min_env_step_improvement:.1f}x` ({min_improvement_text})",
        f"- Warmup iterations per candidate: `{args.warmup_iterations}`",
        f"- Measured iterations per candidate: `{args.benchmark_iterations}`",
        "",
        "## Winner",
        "",
    ]
    if winner is None:
        content.append("No candidate completed successfully.")
    else:
        content.extend(
            [
                f"- `parallel_envs={winner.parallel_envs}`",
                f"- `rollout_chunks_per_env={winner.rollout_chunks_per_env}`",
                f"- `minibatch_size={winner.minibatch_size}`",
                f"- `ppo_epochs={winner.ppo_epochs}`",
                f"- env steps/s: `{winner.env_steps_per_sec:.1f}`",
                f"- chunks/s: `{winner.chunks_per_sec:.2f}`",
                f"- GPU util: `{winner.gpu_util_percent:.1f}%`",
                f"- GPU power: `{winner.gpu_power_w:.1f} W`",
                "",
                "## Training Command",
                "",
                "```bash",
                f"MUJOCO_GL=egl {command}",
                "```",
            ]
        )

    content.extend(["", "## Results", "", result_table(results), ""])
    note_path.write_text("\n".join(content), encoding="utf-8")
    return note_path


def main() -> int:
    args = parse_args()
    args.init_checkpoint = args.init_checkpoint.resolve()
    if args.output_note is not None:
        args.output_note = args.output_note.resolve()
    os.environ.setdefault("MUJOCO_GL", "egl")

    train.load_training_dependencies()
    train.define_model_classes()
    if args.device is not None:
        device = train.torch.device(args.device)
    else:
        device = train.torch.device("cuda" if train.torch.cuda.is_available() else "cpu")

    act_policy = train.load_act_policy(args.init_checkpoint, device)
    policy = train.ACTGaussianPPOPolicy(
        act_policy,
        action_dim=6,
        chunk_size=args.chunk_size,
        log_std_init=args.log_std_init,
    ).to(device)
    critic = train.PrivilegedCritic(input_dim=16).to(device)
    policy_optimizer = train.torch.optim.Adam(policy.parameters(), lr=args.policy_lr)
    critic_optimizer = train.torch.optim.Adam(critic.parameters(), lr=args.critic_lr)

    base_policy_state = clone_state(policy)
    base_critic_state = clone_state(critic)
    base_policy_optimizer_state = copy.deepcopy(policy_optimizer.state_dict())
    base_critic_optimizer_state = copy.deepcopy(critic_optimizer.state_dict())

    candidates = [
        TrialResult(envs, rollout_chunks, minibatch, ppo_epochs, "pending")
        for envs in args.sweep_envs
        for rollout_chunks in args.sweep_rollout_chunks
        for minibatch in args.sweep_minibatches
        for ppo_epochs in args.sweep_ppo_epochs
    ]
    baseline: TrialResult | None = None
    results: list[TrialResult] = []

    print(f"Benchmarking {len(candidates)} ACT sim PPO throughput candidates on {device}")
    print(f"init_checkpoint={args.init_checkpoint}")
    start = time.time()
    for index, candidate in enumerate(candidates, start=1):
        print(
            f"[{index:03d}/{len(candidates):03d}] envs={candidate.parallel_envs} "
            f"rollout={candidate.rollout_chunks_per_env} mb={candidate.minibatch_size} "
            f"epochs={candidate.ppo_epochs}",
            flush=True,
        )
        result = benchmark_trial(
            candidate,
            args,
            device,
            policy,
            critic,
            policy_optimizer,
            critic_optimizer,
            base_policy_state,
            base_critic_state,
            base_policy_optimizer_state,
            base_critic_optimizer_state,
        )
        results.append(result)
        if (
            result.parallel_envs == args.baseline_envs
            and result.rollout_chunks_per_env == min(args.sweep_rollout_chunks)
            and result.minibatch_size == min(args.sweep_minibatches)
            and result.ppo_epochs == min(args.sweep_ppo_epochs)
        ):
            baseline = result
        print(
            f"  status={result.status} env_steps/s={result.env_steps_per_sec:.1f} "
            f"chunks/s={result.chunks_per_sec:.2f} gpu={result.gpu_util_percent:.1f}% "
            f"power={result.gpu_power_w:.1f}W",
            flush=True,
        )

    winner = rank_results(results, args.target_gpu_util)
    note_path = write_note(args, results, winner, baseline)
    elapsed = time.time() - start
    print(f"note={note_path}")
    print(f"elapsed_s={elapsed:.1f}")
    if winner is None:
        print("No successful throughput candidate found.", file=sys.stderr)
        return 1

    command = shell_join(training_command(args, winner))
    print(
        "best="
        f"parallel_envs={winner.parallel_envs} "
        f"rollout_chunks_per_env={winner.rollout_chunks_per_env} "
        f"minibatch_size={winner.minibatch_size} "
        f"ppo_epochs={winner.ppo_epochs} "
        f"env_steps_per_sec={winner.env_steps_per_sec:.1f} "
        f"gpu_util={winner.gpu_util_percent:.1f}%"
    )
    print(f"run: MUJOCO_GL=egl {command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
