"""CLI for named reward-independent MuJoCo evaluation suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from so_arm101_v2.learning import load_small_model_comparison

from .rollout import (
    ConstantPosePolicy,
    CurrentPosePolicy,
    TorchCheckpointPolicy,
    evaluate_closed_loop,
    run_simulation_preflight,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="so-arm101-v2-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
        command.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/simulation"))
        command.add_argument("--no-video", action="store_true")
    evaluate = commands.choices["evaluate"]
    evaluate.add_argument("--suite", choices=("fixed_pickup_contract_v1", "fixed_pickup_recovery_probe_v1"), default="fixed_pickup_contract_v1")
    evaluate.add_argument("--comparison-report", type=Path, required=True)
    evaluate.add_argument("--preflight-report", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "preflight":
            result = run_simulation_preflight(args.mujoco_model, args.output_dir, record_video=not args.no_video)
            print(result.report_json)
            print(f"environment_proven={str(bool(result.environment_proven)).lower()}")
            return 0 if result.environment_proven else 2
        comparison = load_small_model_comparison(args.comparison_report)
        offline = json.loads(comparison.report_json.read_text(encoding="utf-8"))
        mean = np.asarray(offline["development_baselines"]["train_mean_target"], dtype=np.float32)
        policies = {
            "current_pose": CurrentPosePolicy,
            "train_mean": lambda: ConstantPosePolicy(mean),
        }
        for run in comparison.runs:
            policy_id = f"{run.kind.value}.seed{run.seed}"
            checkpoint = run.checkpoint
            policies[policy_id] = lambda checkpoint=checkpoint: TorchCheckpointPolicy(checkpoint)
            if run.kind.value == "image_state":
                policies[policy_id + ".black"] = lambda checkpoint=checkpoint: TorchCheckpointPolicy(checkpoint, black_image=True)
        preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
        result = evaluate_closed_loop(
            args.mujoco_model, args.suite, policies, args.output_dir,
            environment_proven=bool(preflight.get("environment_proven")),
            record_video=not args.no_video,
        )
        print(result.report_json)
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
