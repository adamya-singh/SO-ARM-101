#!/usr/bin/env python3
"""Run the strict-grasp factorial, select a champion, and spend a 9h GPU budget."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
PYTHON = Path("/home/win10ubuntu/miniforge3/envs/lerobot/bin/python")
BASE = SCRIPT_DIR / "outputs/train/act_so101_lead3_30_b32_20260709_161056/checkpoints/026020/pretrained_model"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reset-bank", type=Path, required=True)
    parser.add_argument("--training-seconds", type=float, default=32400.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def checkpoint_episode(path: Path) -> int:
    try:
        return int(torch.load(path, map_location="cpu", weights_only=False).get("episode", -1))
    except Exception:
        return -1


class Pipeline:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.root = args.output_dir.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.root / "training_time.json"
        self.training_seconds = 0.0
        self.stop_requested = False
        self.history: list[dict] = []
        if self.ledger_path.exists():
            ledger = json.loads(self.ledger_path.read_text())
            self.training_seconds = float(ledger.get("training_seconds", 0.0))
            self.history = list(ledger.get("history", []))
        self.env = {**os.environ, "MUJOCO_GL": "egl", "WANDB_SILENT": "true"}

    def save_ledger(self) -> None:
        self.ledger_path.write_text(json.dumps({
            "training_seconds": self.training_seconds,
            "budget_seconds": self.args.training_seconds,
            "history": self.history,
        }, indent=2) + "\n")

    def train(self, name: str, reward: str, curriculum: str, seed: int, updates: int,
              resume: Path | None = None, adaptive_stage: str = "lift") -> Path:
        run = self.root / name
        run.mkdir(parents=True, exist_ok=True)
        checkpoint = run / "checkpoint.pt"
        start = checkpoint_episode(resume) + 1 if resume else 0
        target = start + updates
        command = [
            str(PYTHON), str(SCRIPT_DIR / "train_act_in_sim.py"),
            "--experimental-act-ppo", "--headless", "--no-render",
            "--init-checkpoint", str(BASE), "--checkpoint-path", str(checkpoint),
            "--metrics-jsonl", str(run / "metrics.jsonl"), "--episodes", str(target),
            "--wandb-url-path", str(run / "wandb_url.txt"),
            "--reward-profile", reward, "--reset-curriculum", curriculum,
            "--adaptive-stage", adaptive_stage, "--seed", str(seed),
            "--policy-lr", "1e-6", "--critic-lr", "5e-5", "--log-std-init", "-2",
            "--parallel-envs", "12", "--rollout-chunks-per-env", "2",
            "--snapshot-every", "10", "--no-randomize-appearance",
        ]
        if curriculum != "none":
            command += ["--reset-bank", str(self.args.reset_bank.resolve())]
        if resume:
            command += ["--resume", str(resume.resolve())]
        if self.args.dry_run:
            print("DRY RUN:", " ".join(command), flush=True)
            return checkpoint
        remaining = self.args.training_seconds - self.training_seconds
        if remaining <= 0:
            return resume or checkpoint
        log = (run / "train.log").open("a", encoding="utf-8")
        started = time.monotonic()
        process = subprocess.Popen(command, cwd=PROJECT_DIR, env=self.env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        interrupted = False
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            interrupted = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        except KeyboardInterrupt:
            interrupted = True
            self.stop_requested = True
            os.killpg(process.pid, signal.SIGINT)
            try:
                process.wait(timeout=120)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=30)
        finally:
            elapsed = min(time.monotonic() - started, remaining)
            log.close()
            self.training_seconds += elapsed
            self.history.append({
                "name": name, "reward_profile": reward, "reset_curriculum": curriculum,
                "adaptive_stage": adaptive_stage, "seed": seed, "requested_updates": updates,
                "training_seconds": elapsed, "checkpoint": str(checkpoint),
                "returncode": process.returncode, "budget_interrupt": interrupted,
            })
            self.save_ledger()
        if process.returncode not in (0, 130, -signal.SIGINT) and not interrupted:
            raise RuntimeError(f"Training run {name} failed with {process.returncode}; see {run / 'train.log'}")
        return checkpoint

    def select_snapshot(self, checkpoint: Path) -> Path:
        metrics_path = checkpoint.parent / "metrics.jsonl"
        if not metrics_path.exists():
            return checkpoint
        rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
        candidates = [row for row in rows if (int(row["episode"]) + 1) % 10 == 0]
        if not candidates:
            return checkpoint
        best = max(candidates, key=lambda row: (
            row.get("rollout/reset_stage/normal_strict_success_steps", 0.0),
            row.get("rollout/max_strict_grasp_streak", 0.0),
            row.get("rollout/face_grasp_steps", 0.0),
            row.get("rollout/max_block_height_gain", 0.0),
            -row.get("rollout/action_clip_rate", 1.0),
        ))
        snapshot = checkpoint.with_name(f"{checkpoint.stem}_ep{int(best['episode']):04d}{checkpoint.suffix}")
        return snapshot if snapshot.exists() else checkpoint

    def evaluate(self, checkpoint: Path, name: str, episodes: int, stochastic: bool,
                 reset_stage: str = "normal", seed: int = 260713, narrow: bool = False) -> dict:
        output = self.root / "evaluations" / f"{name}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(PYTHON), str(SCRIPT_DIR / "run_act_ppo_sim_inference.py"),
            "--headless", "--resume", str(checkpoint), "--init-checkpoint", str(BASE),
            "--episodes", str(episodes), "--max-steps-per-episode", "100",
            "--reward-profile", "strict_transition", "--seed", str(seed),
            "--no-randomize-appearance", "--output-json", str(output),
            "--evaluation-reset-stage", reset_stage,
        ]
        if stochastic:
            command.append("--stochastic")
        if narrow:
            command += ["--randomize-block-reset", "--block-dist-range", "0.235", "0.245",
                        "--block-angle-range", "-3", "3"]
        if reset_stage != "normal":
            command += ["--reset-bank", str(self.args.reset_bank.resolve())]
        if self.args.dry_run:
            print("DRY RUN:", " ".join(command), flush=True)
            return {"strict_success_rate": 0.0, "sustained_grasp_rate": 0.0,
                    "strict_lift_rate": 0.0, "mean_max_block_height_gain": 0.0}
        with (output.with_suffix(".log")).open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=PROJECT_DIR, env=self.env, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
        return json.loads(output.read_text())["summary"]


def rank_key(pair: list[dict]) -> tuple:
    success = [item["summary"].get("strict_success_rate", 0.0) for item in pair]
    return (
        min(success), sum(success) / len(success),
        min(item["summary"].get("sustained_grasp_rate", 0.0) for item in pair),
        min(item["summary"].get("strict_lift_rate", 0.0) for item in pair),
        min(item["summary"].get("mean_max_block_height_gain", 0.0) for item in pair),
        -max(item["summary"].get("mean_max_block_displacement", 0.0) for item in pair),
        -max(item["summary"].get("mean_action_clip_rate", 0.0) for item in pair),
    )


def precursor_key(summary: dict) -> tuple:
    return (
        float(summary.get("sustained_grasp_rate", 0.0)),
        float(summary.get("mean_bilateral_interior_face_contact_count", 0.0)),
        float(summary.get("mean_interior_face_contact_count", 0.0)),
        float(summary.get("mean_max_block_height_gain", 0.0)),
        -float(summary.get("mean_corner_only_contact_count", 0.0)),
        -float(summary.get("mean_max_block_displacement", 0.0)),
        -float(summary.get("mean_action_clip_rate", 0.0)),
    )


def strictly_better_precursor(candidate: dict, champion: dict) -> bool:
    return precursor_key(candidate) > precursor_key(champion)


def adaptive_transition(stage: str, summary: dict, full_success_streak: int) -> tuple[str, int]:
    if stage == "lift" and summary.get("strict_success_rate", 0.0) >= 0.20:
        return "grasp", 0
    if (
        stage == "grasp"
        and summary.get("sustained_grasp_rate", 0.0) >= 0.10
        and summary.get("strict_lift_rate", 0.0) > 0.0
    ):
        return "full", 0
    if stage == "full":
        streak = full_success_streak + 1 if summary.get("strict_success_rate", 0.0) > 0.0 else 0
        return ("normal", streak) if streak >= 2 else ("full", streak)
    return stage, full_success_streak


def main() -> int:
    def request_interrupt(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, request_interrupt)
    args = parse_args()
    pipe = Pipeline(args)
    if not args.dry_run:
        preflight = pipe.root / "reset_bank_preflight.json"
        subprocess.run(
            [str(PYTHON), str(SCRIPT_DIR / "validate_act_reset_bank.py"),
             "--reset-bank", str(args.reset_bank.resolve()), "--subprocess-workers", "12",
             "--output-json", str(preflight)],
            cwd=PROJECT_DIR, env=pipe.env, check=True,
        )
    conditions = [
        ("baseline_normal", "baseline", "none"),
        ("strict_normal", "strict_transition", "none"),
        ("baseline_mixed", "baseline", "mixed"),
        ("strict_mixed", "strict_transition", "mixed"),
    ]
    candidates: list[dict] = []
    for condition, reward, curriculum in conditions:
        for seed in (17, 71):
            checkpoint = pipe.train(f"ablation/{condition}_s{seed}", reward, curriculum, seed, 100)
            if not checkpoint.exists() and not args.dry_run:
                break
            selected = pipe.select_snapshot(checkpoint)
            summary = pipe.evaluate(selected, f"{condition}_s{seed}_stochastic20", 20, True)
            candidates.append({"condition": condition, "reward": reward, "curriculum": curriculum,
                               "seed": seed, "checkpoint": str(selected), "summary": summary})
            if pipe.stop_requested:
                break
        if pipe.training_seconds >= args.training_seconds or pipe.stop_requested:
            break

    if args.dry_run:
        (pipe.root / "dry_run_manifest.json").write_text(json.dumps({
            "commands_validated": True, "candidate_count": len(candidates),
            "budget_training_seconds": args.training_seconds,
        }, indent=2) + "\n")
        return 0

    grouped = {condition: [item for item in candidates if item["condition"] == condition]
               for condition, _, _ in conditions}
    qualified = [items for items in grouped.values() if len(items) == 2 and all(
        item["summary"].get("strict_success_rate", 0.0) > 0.0 and
        item["summary"].get("sustained_grasp_rate", 0.0) > 0.0 for item in items)]
    ablation_qualified = bool(qualified)
    complete_pairs = [items for items in grouped.values() if len(items) == 2]
    if qualified:
        winning_pair = max(qualified, key=rank_key)
        champion = max(winning_pair, key=lambda item: rank_key([item]))
    else:
        champion = max(candidates, key=lambda item: precursor_key(item["summary"]), default=None)

    confirmations = []
    if champion and ablation_qualified and pipe.training_seconds < args.training_seconds:
        for item in grouped[champion["condition"]]:
            ckpt = Path(item["checkpoint"])
            confirmations.append({
                "seed": item["seed"],
                "deterministic": pipe.evaluate(ckpt, f"confirm_{item['condition']}_s{item['seed']}_det30", 30, False),
                "stochastic": pipe.evaluate(ckpt, f"confirm_{item['condition']}_s{item['seed']}_sto30", 30, True),
            })

    current = Path(champion["checkpoint"]) if champion else None
    continuation_seed = 1701
    adaptive_history = []
    adaptive_stage = "lift"
    full_success_streak = 0
    adaptive_qualified = False
    while current and pipe.training_seconds < args.training_seconds and not pipe.stop_requested:
        if ablation_qualified:
            current = pipe.train(f"continuation/winner_{continuation_seed}", champion["reward"],
                                 champion["curriculum"], continuation_seed, 100, current)
        else:
            branch = pipe.train(
                f"adaptive/{adaptive_stage}_{continuation_seed}", "strict_transition",
                "adaptive", continuation_seed, 100, current, adaptive_stage,
            )
            eval_stage = "pregrasp" if adaptive_stage == "lift" else "normal"
            summary = pipe.evaluate(
                branch, f"adaptive_{adaptive_stage}_{continuation_seed}", 20, True, eval_stage
            )
            improved = strictly_better_precursor(summary, champion["summary"])
            adaptive_history.append({
                "stage": adaptive_stage, "checkpoint": str(branch), "summary": summary,
                "champion_replaced": improved,
            })
            if improved:
                current = branch
                champion = {**champion, "checkpoint": str(branch), "summary": summary}

            previous_stage = adaptive_stage
            adaptive_stage, full_success_streak = adaptive_transition(
                adaptive_stage, summary, full_success_streak
            )
            if previous_stage == "full" and adaptive_stage == "normal":
                adaptive_qualified = True
            continuation_seed += 1
        continuation_seed += 1

    final_evaluation = {}
    if current and current.exists() and not args.dry_run:
        final_evaluation = {
            "deterministic_fixed": pipe.evaluate(current, "final_deterministic_fixed30", 30, False),
            "stochastic_fixed": pipe.evaluate(current, "final_stochastic_fixed30", 30, True),
            "stochastic_narrow": pipe.evaluate(current, "final_stochastic_narrow30", 30, True,
                                                seed=260743, narrow=True),
        }
    final_strict_success = max(
        (float(summary.get("strict_success_rate", 0.0)) for summary in final_evaluation.values()),
        default=0.0,
    )
    qualification = ablation_qualified or adaptive_qualified or final_strict_success > 0.0
    manifest = {
        "schema_version": 1, "base_checkpoint": str(BASE.resolve()),
        "actual_training_seconds": pipe.training_seconds, "budget_training_seconds": args.training_seconds,
        "qualification_status": "qualified" if qualification else "unqualified_best_precursor",
        "candidates": candidates, "confirmations": confirmations,
        "adaptive_history": adaptive_history, "curriculum_history": pipe.history,
        "final_evaluation": final_evaluation, "checkpoint_path": str(current) if current else None,
        "wandb_links": sorted({
            path.read_text().strip()
            for path in pipe.root.glob("**/wandb_url.txt")
            if path.read_text().strip()
        }),
    }
    (pipe.root / "winner_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
