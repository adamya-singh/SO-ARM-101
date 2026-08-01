#!/usr/bin/env python3
"""Resumable open-ended ACT PPO ablation campaign on the original SO-101 model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import signal
import subprocess
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PYTHON = Path("/home/win10ubuntu/miniforge3/envs/lerobot/bin/python")
MODEL_XML = SCRIPT_DIR / "model" / "so101_new_calib.xml"
EXPECTED_MODEL_SHA256 = "ac5254b0283e342ba2499c0b560f47b958deb833f291092728bad7c278f5d167"
INIT_CHECKPOINT = (
    SCRIPT_DIR
    / "outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model"
)
BRANCH_CHECKPOINT = (
    SCRIPT_DIR
    / "outputs/train/act_lead3_grasp_phase1_20260711/"
    "A_baseline_lr1em6_stdm2.0_s17/act_sim_ppo_checkpoint_ep0149.pt"
)
PREFLIGHT_BASELINE = (
    PROJECT_DIR
    / "outputs/eval/act_ppo_ep0149_original_model_strict_baseline_20260713/stochastic20.json"
)
SCREEN_SEEDS = (17, 71)
REPLICATION_SEEDS = (83, 101)
STOP_REQUESTED = False
ACTIVE_PROCESS: subprocess.Popen[str] | None = None


BASE_CONFIG: dict[str, Any] = {
    "policy_lr": 1e-6,
    "critic_lr": 5e-5,
    "log_std_offset": 0.0,
    "gripper_log_std_offset": 0.0,
    "actor_train_scope": "all",
    "reference_anchor_coef": 0.0,
    "entropy_coef": 0.0,
    "clip_epsilon": 0.1,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ppo_epochs": 1,
    "rollout_chunks_per_env": 2,
}


def variant(name: str, **overrides: Any) -> dict[str, Any]:
    config = dict(BASE_CONFIG)
    config.update(overrides)
    return {"name": name, "config": config}


GENERATION_ONE = [
    variant("control"),
    variant("actor_lr_3e7", policy_lr=3e-7),
    variant("actor_lr_6e7", policy_lr=6e-7),
    variant("actor_lr_2e6", policy_lr=2e-6),
    variant("critic_lr_2e5", critic_lr=2e-5),
    variant("critic_lr_1e4", critic_lr=1e-4),
    variant("scope_decoder_head", actor_train_scope="decoder_head"),
    variant("scope_action_head", actor_train_scope="action_head"),
    variant("noise_minus_035", log_std_offset=-0.35),
    variant("noise_plus_035", log_std_offset=0.35),
    variant("gripper_noise_plus_040", gripper_log_std_offset=0.40),
    variant("rollout_chunks_4", rollout_chunks_per_env=4),
    variant("ppo_epochs_2", ppo_epochs=2),
    variant("clip_005", clip_epsilon=0.05),
    variant("clip_020", clip_epsilon=0.20),
    variant("anchor_001", reference_anchor_coef=0.01),
    variant("anchor_010", reference_anchor_coef=0.10),
    variant("entropy_1e3", entropy_coef=1e-3),
    variant("gamma_0995", gamma=0.995),
    variant("gae_098", gae_lambda=0.98),
]


SEARCH_SPACE = {
    "policy_lr": (3e-7, 6e-7, 1e-6, 2e-6),
    "critic_lr": (2e-5, 5e-5, 1e-4),
    "log_std_offset": (-0.35, 0.0, 0.35),
    "gripper_log_std_offset": (0.0, 0.40),
    "actor_train_scope": ("all", "decoder_head", "action_head"),
    "reference_anchor_coef": (0.0, 0.01, 0.10),
    "entropy_coef": (0.0, 1e-3),
    "clip_epsilon": (0.05, 0.1, 0.2),
    "gamma": (0.99, 0.995),
    "gae_lambda": (0.95, 0.98),
    "ppo_epochs": (1, 2),
    "rollout_chunks_per_env": (2, 4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one update per actor scope and exit.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_invariants() -> None:
    if sha256(MODEL_XML) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("production SO-101 model XML differs from the restored original")
    text = MODEL_XML.read_text(encoding="utf-8")
    forbidden = ("fixed_jaw_contact_pad", "moving_jaw_contact_pad")
    if any(token in text for token in forbidden):
        raise RuntimeError("artificial fingertip pad geometry found in production model")
    if not INIT_CHECKPOINT.is_dir() or not BRANCH_CHECKPOINT.is_file():
        raise FileNotFoundError("required ACT initialization or ep0149 branch checkpoint is missing")


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_state(output_dir: Path) -> dict[str, Any]:
    path = output_dir / "campaign_state.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "status": "running",
        "model_sha256": EXPECTED_MODEL_SHA256,
        "branch_checkpoint": str(BRANCH_CHECKPOINT),
        "init_checkpoint": str(INIT_CHECKPOINT),
        "qualification_threshold": 0.20,
        "excluded_jobs": {
            "28": "invalid approximate reset-bank experiment",
            "29": "reverted simulator collision-model experiment",
        },
        "training_seconds": 0.0,
        "trials": {},
        "promotions": [],
        "evaluations": {},
        "champion": None,
        "qualified": False,
        "cycle": 0,
        "used_signatures": [],
    }


def save_state(output_dir: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    atomic_json(output_dir / "campaign_state.json", state)
    manifest = {
        "status": state["status"],
        "qualified": state["qualified"],
        "champion": state["champion"],
        "training_seconds": state["training_seconds"],
        "model_sha256": state["model_sha256"],
        "branch_checkpoint": state["branch_checkpoint"],
        "qualification": "both confirmation seeds >= 0.20 strict success over 30 stochastic episodes",
        "excluded_jobs": state["excluded_jobs"],
        "promotion_history": state["promotions"],
        "updated_at": state["updated_at"],
    }
    atomic_json(output_dir / "champion_manifest.json", manifest)


def handle_signal(signum: int, _frame: Any) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True
    if ACTIVE_PROCESS is not None and ACTIVE_PROCESS.poll() is None:
        ACTIVE_PROCESS.send_signal(signal.SIGINT)


def run_command(command: list[str], log_path: Path, dry_run: bool) -> tuple[int, float]:
    global ACTIVE_PROCESS
    verify_invariants()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        print("DRY RUN:", " ".join(command))
        return 0, 0.0
    started = time.monotonic()
    environment = dict(os.environ)
    environment.update(
        MUJOCO_GL="egl",
        PYTHONUNBUFFERED="1",
        WANDB_RUN_GROUP="act-ppo-original-model-open-ended",
        WANDB_NAME=log_path.parent.name,
    )
    with log_path.open("a", encoding="utf-8") as stream:
        ACTIVE_PROCESS = subprocess.Popen(
            command,
            cwd=SCRIPT_DIR,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return_code = ACTIVE_PROCESS.wait()
    ACTIVE_PROCESS = None
    elapsed = time.monotonic() - started
    verify_invariants()
    return return_code, elapsed


def config_signature(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


def config_cli(config: dict[str, Any], branching: bool) -> list[str]:
    result = [
        "--policy-lr", str(config["policy_lr"]),
        "--critic-lr", str(config["critic_lr"]),
        "--actor-train-scope", str(config["actor_train_scope"]),
        "--reference-anchor-coef", str(config["reference_anchor_coef"]),
        "--entropy-coef", str(config["entropy_coef"]),
        "--clip-epsilon", str(config["clip_epsilon"]),
        "--gamma", str(config["gamma"]),
        "--gae-lambda", str(config["gae_lambda"]),
        "--ppo-epochs", str(config["ppo_epochs"]),
        "--rollout-chunks-per-env", str(config["rollout_chunks_per_env"]),
    ]
    if branching:
        result += [
            "--branch-log-std-offset", str(config["log_std_offset"]),
            "--gripper-log-std-offset", str(config["gripper_log_std_offset"]),
        ]
    return result


def training_command(
    run_dir: Path,
    config: dict[str, Any],
    seed: int,
    episodes: int,
    generation: int,
    condition: str,
    parent: Path,
    resume: bool = False,
) -> list[str]:
    command = [
        str(PYTHON), "-u", str(SCRIPT_DIR / "train_act_in_sim.py"),
        "--experimental-act-ppo", "--headless", "--no-render",
        "--init-checkpoint", str(INIT_CHECKPOINT),
        "--resume" if resume else "--branch-from", str(parent),
        "--episodes", str(episodes), "--max-steps-per-episode", "150",
        "--chunk-size", "30", "--steps-per-action", "1",
        "--parallel-envs", "12", "--minibatch-size", "64",
        "--snapshot-every", "10", "--eval-episodes", "0",
        "--reward-profile", "baseline", "--reset-curriculum", "none",
        "--no-randomize-appearance", "--curriculum-fixed-block",
        "--seed", str(seed), "--campaign-generation", str(generation),
        "--campaign-condition", condition, "--lineage-parent", str(parent),
        "--checkpoint-path", str(run_dir / "checkpoint.pt"),
        "--metrics-jsonl", str(run_dir / "metrics.jsonl"),
        "--wandb-url-path", str(run_dir / "wandb_url.txt"),
    ]
    command += config_cli(config, branching=not resume)
    return command


def evaluation_command(
    checkpoint: Path,
    episodes: int,
    seed: int,
    output: Path,
    stochastic: bool,
) -> list[str]:
    command = [
        str(PYTHON), "-u", str(SCRIPT_DIR / "run_act_ppo_sim_inference.py"),
        "--resume", str(checkpoint), "--init-checkpoint", str(INIT_CHECKPOINT),
        "--episodes", str(episodes), "--max-steps-per-episode", "150",
        "--chunk-size", "30", "--steps-per-action", "1",
        "--reward-profile", "baseline", "--headless", "--no-randomize-appearance",
        "--curriculum-fixed-block", "--seed", str(seed), "--output-json", str(output),
    ]
    if stochastic:
        command.append("--stochastic")
    return command


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def training_window_score(rows: list[dict[str, Any]]) -> tuple[float, ...]:
    if not rows:
        return (-1.0,) * 8
    return (
        sum(float(row.get("rollout/strict_success_steps", 0.0)) for row in rows),
        max(float(row.get("rollout/max_strict_grasp_streak", 0.0)) for row in rows),
        sum(float(row.get("rollout/bilateral_interior_face_contact_steps", 0.0)) for row in rows),
        sum(float(row.get("rollout/interior_face_contact_steps", 0.0)) for row in rows),
        max(float(row.get("rollout/max_block_height_gain", 0.0)) for row in rows),
        -sum(float(row.get("rollout/corner_only_contact_steps", 0.0)) for row in rows),
        -sum(float(row.get("rollout/action_clip_rate", 0.0)) for row in rows),
        -sum(float(row.get("rollout/reward_components/block_displacement_penalty", 0.0)) for row in rows),
    )


def select_snapshot(run_dir: Path) -> Path:
    rows = read_jsonl(run_dir / "metrics.jsonl")
    snapshots = sorted(run_dir.glob("checkpoint_ep*.pt"))
    if not snapshots:
        checkpoint = run_dir / "checkpoint.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"no checkpoint found in {run_dir}")
        return checkpoint
    best = None
    for snapshot in snapshots:
        match = re.search(r"_ep(\d+)\.pt$", snapshot.name)
        episode = int(match.group(1)) if match else -1
        window = [row for row in rows if episode - 9 <= int(row.get("episode", -1)) <= episode]
        candidate = (training_window_score(window), snapshot)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1]


def evaluation_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def pair_rank(summaries: list[dict[str, Any]]) -> tuple[float, ...]:
    def values(key: str, default: float = 0.0) -> list[float]:
        return [float(summary.get(key, default)) for summary in summaries]

    strict = values("strict_success_rate")
    sustained = values("sustained_grasp_rate")
    lifts = values("strict_lift_rate")
    bilateral = values("mean_bilateral_interior_face_contact_count")
    height = values("mean_max_block_height_gain")
    corner = values("mean_corner_only_contact_count")
    displacement = values("mean_max_block_displacement")
    clipping = values("mean_action_clip_rate")
    return (
        min(strict), sum(strict) / len(strict),
        min(sustained), sum(sustained) / len(sustained),
        sum(lifts) / len(lifts), sum(bilateral) / len(bilateral),
        sum(height) / len(height),
        -sum(corner) / len(corner),
        -sum(displacement) / len(displacement),
        -sum(clipping) / len(clipping),
    )


def strictly_better(candidate: dict[str, Any], champion: dict[str, Any] | None) -> bool:
    if champion is None:
        # Negative penalty tie-breakers alone are not learning progress.
        return any(float(value) > 0.0 for value in candidate["rank"][:7])
    return tuple(candidate["rank"]) > tuple(champion["rank"])


def checkpoint_episode(path: Path) -> int:
    """Read only the episode counter needed to choose an exact-resume target."""
    command = [
        str(PYTHON), "-c",
        "import torch,sys; print(int(torch.load(sys.argv[1],map_location='cpu',weights_only=False).get('episode',0)))",
        str(path),
    ]
    return int(subprocess.check_output(command, text=True).strip())


def run_training(
    output_dir: Path,
    state: dict[str, Any],
    trial_id: str,
    config: dict[str, Any],
    seed: int,
    episodes: int,
    generation: int,
    parent: Path,
    dry_run: bool,
    resume: bool = False,
) -> Path:
    record = state["trials"].setdefault(trial_id, {})
    run_dir = output_dir / "runs" / trial_id
    checkpoint = run_dir / "checkpoint.pt"
    if record.get("status") == "complete" and checkpoint.is_file():
        return Path(record["selected_checkpoint"])
    recovering = bool(
        checkpoint.is_file() and record.get("status") in {"running", "interrupted"}
    )
    effective_parent = checkpoint if recovering else parent
    effective_resume = True if recovering else resume
    record.update(
        status="running", config=config, seed=seed, episodes=episodes,
        generation=generation, parent=str(parent), run_dir=str(run_dir),
        recovery_checkpoint=str(checkpoint) if recovering else None,
    )
    save_state(output_dir, state)
    command = training_command(
        run_dir, config, seed, episodes, generation, trial_id,
        effective_parent, effective_resume,
    )
    code, elapsed = run_command(command, run_dir / "train.log", dry_run)
    state["training_seconds"] += elapsed
    if STOP_REQUESTED:
        record["status"] = "interrupted"
        save_state(output_dir, state)
        raise KeyboardInterrupt
    if code != 0:
        record.update(status="failed", return_code=code)
        save_state(output_dir, state)
        raise RuntimeError(f"training trial {trial_id} failed with exit code {code}")
    if dry_run:
        record.update(status="dry_run", selected_checkpoint=str(checkpoint))
        save_state(output_dir, state)
        return checkpoint
    selected = select_snapshot(run_dir)
    record.update(status="complete", selected_checkpoint=str(selected), elapsed_seconds=elapsed)
    save_state(output_dir, state)
    return selected


def run_evaluation(
    output_dir: Path,
    state: dict[str, Any],
    evaluation_id: str,
    checkpoint: Path,
    episodes: int,
    seed: int,
    stochastic: bool,
    dry_run: bool,
) -> dict[str, Any]:
    record = state["evaluations"].get(evaluation_id)
    if record and record.get("status") == "complete":
        return record["summary"]
    path = output_dir / "evaluations" / f"{evaluation_id}.json"
    command = evaluation_command(checkpoint, episodes, seed, path, stochastic)
    code, elapsed = run_command(command, path.with_suffix(".log"), dry_run)
    if STOP_REQUESTED:
        raise KeyboardInterrupt
    if code != 0:
        raise RuntimeError(f"evaluation {evaluation_id} failed with exit code {code}")
    summary = {} if dry_run else evaluation_summary(path)
    state["evaluations"][evaluation_id] = {
        "status": "dry_run" if dry_run else "complete",
        "checkpoint": str(checkpoint), "episodes": episodes, "seed": seed,
        "stochastic": stochastic, "elapsed_seconds": elapsed, "summary": summary,
    }
    save_state(output_dir, state)
    return summary


def evaluate_pair(
    output_dir: Path,
    state: dict[str, Any],
    prefix: str,
    checkpoints: dict[int, Path],
    episodes: int,
    stochastic: bool,
    dry_run: bool,
) -> tuple[list[dict[str, Any]], tuple[float, ...]]:
    summaries = [
        run_evaluation(
            output_dir, state, f"{prefix}_s{seed}_{'stoch' if stochastic else 'det'}{episodes}",
            checkpoints[seed], episodes, 91000 + seed, stochastic, dry_run,
        )
        for seed in sorted(checkpoints)
    ]
    rank = pair_rank(summaries) if not dry_run else (0.0,) * 10
    return summaries, rank


def generation_one(output_dir: Path, state: dict[str, Any], dry_run: bool) -> list[dict[str, Any]]:
    results = []
    for condition in GENERATION_ONE:
        checkpoints = {}
        for seed in SCREEN_SEEDS:
            trial_id = f"g1_screen/{condition['name']}_s{seed}"
            checkpoints[seed] = run_training(
                output_dir, state, trial_id, condition["config"], seed, 40, 1,
                BRANCH_CHECKPOINT, dry_run,
            )
        summaries, rank = evaluate_pair(
            output_dir, state, f"g1_screen_{condition['name']}", checkpoints, 10, True, dry_run
        )
        results.append({**condition, "checkpoints": checkpoints, "summaries": summaries, "rank": rank})
    results.sort(key=lambda item: item["rank"], reverse=True)
    promoted = results[:6]
    state["promotions"].append({"stage": "g1_screen", "conditions": [item["name"] for item in promoted]})
    save_state(output_dir, state)

    continued = []
    for condition in promoted:
        checkpoints = {}
        for seed in SCREEN_SEEDS:
            parent = condition["checkpoints"][seed]
            trial_id = f"g1_promoted/{condition['name']}_s{seed}"
            checkpoints[seed] = run_training(
                output_dir, state, trial_id, condition["config"], seed, 120, 1,
                parent, dry_run, resume=True,
            )
        summaries, rank = evaluate_pair(
            output_dir, state, f"g1_promoted_{condition['name']}", checkpoints, 20, True, dry_run
        )
        continued.append({**condition, "checkpoints": checkpoints, "summaries": summaries, "rank": rank})
    continued.sort(key=lambda item: item["rank"], reverse=True)
    finalists = continued[:2]
    state["promotions"].append({"stage": "g1_promoted", "conditions": [item["name"] for item in finalists]})
    save_state(output_dir, state)

    replicated = []
    for condition in finalists:
        checkpoints = {}
        for seed in REPLICATION_SEEDS:
            trial_id = f"g1_replication/{condition['name']}_s{seed}"
            checkpoints[seed] = run_training(
                output_dir, state, trial_id, condition["config"], seed, 160, 1,
                BRANCH_CHECKPOINT, dry_run,
            )
        stochastic, rank = evaluate_pair(
            output_dir, state, f"g1_confirm_{condition['name']}", checkpoints, 30, True, dry_run
        )
        deterministic, _ = evaluate_pair(
            output_dir, state, f"g1_confirm_{condition['name']}", checkpoints, 30, False, dry_run
        )
        qualified = bool(
            not dry_run
            and all(float(summary.get("strict_success_rate", 0.0)) >= 0.20 for summary in stochastic)
        )
        replicated.append({
            **condition, "checkpoints": checkpoints, "summaries": stochastic,
            "deterministic": deterministic, "rank": rank, "qualified": qualified,
        })
    replicated.sort(key=lambda item: (item["qualified"], item["rank"]), reverse=True)
    return replicated


def sampled_challengers(state: dict[str, Any], count: int = 8) -> list[dict[str, Any]]:
    used = set(state["used_signatures"])
    randomizer = random.Random(29071 + int(state["cycle"]))
    challengers = []
    keys = tuple(SEARCH_SPACE)
    while len(challengers) < count:
        config = {key: randomizer.choice(SEARCH_SPACE[key]) for key in keys}
        signature = config_signature(config)
        if signature in used:
            continue
        used.add(signature)
        challengers.append({"name": f"cycle{state['cycle']:04d}_c{len(challengers):02d}", "config": config})
    state["used_signatures"] = sorted(used)
    return challengers


def update_champion(state: dict[str, Any], result: dict[str, Any]) -> None:
    candidate = {
        "condition": result["name"], "config": result["config"],
        "checkpoints": {str(seed): str(path) for seed, path in result["checkpoints"].items()},
        "rank": list(result["rank"]), "qualified": bool(result.get("qualified", False)),
        "summaries": result.get("summaries", []),
    }
    champion = state.get("champion")
    if champion is None or (
        candidate["qualified"] > bool(champion.get("qualified", False))
        or (
            candidate["qualified"] == bool(champion.get("qualified", False))
            and strictly_better(candidate, champion)
        )
    ):
        state["champion"] = candidate
        state["qualified"] = candidate["qualified"]


def additional_cycle(output_dir: Path, state: dict[str, Any], dry_run: bool) -> None:
    state["cycle"] += 1
    parent = BRANCH_CHECKPOINT
    if state.get("qualified") and state.get("champion"):
        champion_paths = state["champion"]["checkpoints"]
        parent = Path(champion_paths[sorted(champion_paths)[0]])
        config = state["champion"]["config"]
        # Exploit the independently retained champion before opening a challenger batch.
        target_episode = 100 if dry_run else checkpoint_episode(parent) + 101
        checkpoint = run_training(
            output_dir, state, f"cycle{state['cycle']:04d}/exploit", config, 17000 + state["cycle"],
            target_episode, state["cycle"] + 1, parent, dry_run, resume=True,
        )
        summaries = [
            run_evaluation(
                output_dir, state, f"cycle{state['cycle']:04d}_exploit_s{seed}_stoch30",
                checkpoint, 30, 92000 + seed, True, dry_run,
            )
            for seed in REPLICATION_SEEDS
        ]
        exploit_result = {
            "name": f"cycle{state['cycle']:04d}_exploit", "config": config,
            "checkpoints": {seed: checkpoint for seed in REPLICATION_SEEDS}, "summaries": summaries,
            "rank": pair_rank(summaries) if not dry_run else (0.0,) * 10,
            "qualified": bool(
                not dry_run
                and all(float(summary.get("strict_success_rate", 0.0)) >= 0.20 for summary in summaries)
            ),
        }
        update_champion(state, exploit_result)

    screened = []
    for challenger in sampled_challengers(state):
        checkpoints = {}
        for seed in SCREEN_SEEDS:
            checkpoints[seed] = run_training(
                output_dir, state, f"cycle{state['cycle']:04d}/screen/{challenger['name']}_s{seed}",
                challenger["config"], seed, 60, state["cycle"] + 1, parent, dry_run,
            )
        summaries, rank = evaluate_pair(
            output_dir, state, f"cycle{state['cycle']:04d}_{challenger['name']}", checkpoints,
            10, True, dry_run,
        )
        screened.append({**challenger, "checkpoints": checkpoints, "summaries": summaries, "rank": rank})
    screened.sort(key=lambda item: item["rank"], reverse=True)
    for finalist in screened[:2]:
        checkpoints = {}
        for seed in REPLICATION_SEEDS:
            checkpoints[seed] = run_training(
                output_dir, state, f"cycle{state['cycle']:04d}/replicate/{finalist['name']}_s{seed}",
                finalist["config"], seed, 160, state["cycle"] + 1, parent, dry_run,
            )
        summaries, rank = evaluate_pair(
            output_dir, state, f"cycle{state['cycle']:04d}_confirm_{finalist['name']}", checkpoints,
            30, True, dry_run,
        )
        finalist.update(
            checkpoints=checkpoints, summaries=summaries, rank=rank,
            qualified=bool(
                not dry_run
                and all(float(summary.get("strict_success_rate", 0.0)) >= 0.20 for summary in summaries)
            ),
        )
        update_champion(state, finalist)
    save_state(output_dir, state)


def smoke(output_dir: Path, state: dict[str, Any], dry_run: bool) -> None:
    for scope in ("all", "decoder_head", "action_head"):
        config = dict(BASE_CONFIG, actor_train_scope=scope)
        run_training(
            output_dir, state, f"smoke/{scope}", config, 701, 1, 0,
            BRANCH_CHECKPOINT, dry_run,
        )


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.expanduser().resolve()
    verify_invariants()
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    state = load_state(args.output_dir)
    if "ep0149_baseline_stoch20" not in state["evaluations"] and PREFLIGHT_BASELINE.is_file():
        state["evaluations"]["ep0149_baseline_stoch20"] = {
            "status": "complete",
            "checkpoint": str(BRANCH_CHECKPOINT),
            "episodes": 20,
            "seed": 90017,
            "stochastic": True,
            "elapsed_seconds": 0.0,
            "source": str(PREFLIGHT_BASELINE),
            "summary": evaluation_summary(PREFLIGHT_BASELINE),
        }
    save_state(args.output_dir, state)
    try:
        if args.smoke:
            smoke(args.output_dir, state, args.dry_run)
            state["status"] = "smoke_complete"
            save_state(args.output_dir, state)
            return 0

        run_evaluation(
            args.output_dir, state, "ep0149_baseline_stoch20", BRANCH_CHECKPOINT,
            20, 90017, True, args.dry_run,
        )
        replicated = generation_one(args.output_dir, state, args.dry_run)
        for result in replicated:
            update_champion(state, result)
        save_state(args.output_dir, state)
        if args.dry_run:
            state["status"] = "dry_run_complete"
            save_state(args.output_dir, state)
            return 0
        while not STOP_REQUESTED:
            additional_cycle(args.output_dir, state, False)
    except KeyboardInterrupt:
        state["status"] = "interrupted"
        save_state(args.output_dir, state)
        return 130
    except Exception as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        save_state(args.output_dir, state)
        raise
    state["status"] = "interrupted"
    save_state(args.output_dir, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
