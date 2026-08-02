"""CLI for named reward-independent MuJoCo evaluation suites."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from so_arm101_v2.learning import load_small_model_comparison
from so_arm101_v2.data._serialization import content_sha256

from .rollout import (
    ConstantPosePolicy,
    CurrentPosePolicy,
    TorchCheckpointPolicy,
    evaluate_closed_loop,
    run_simulation_preflight,
)
from .oracle import capture_oracle_demonstrations
from .recovery import capture_phase_wide_recovery_examples, evaluate_recovery_anchor_starts
from .clone_policy import OracleCloneCheckpointPolicy
from .suites import load_simulation_suite


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
    preflight = commands.choices["preflight"]
    preflight.add_argument(
        "--suite",
        choices=("fixed_pickup_contract_v1", "fixed_pick_place_v3"),
        default="fixed_pickup_contract_v1",
    )
    capture = commands.add_parser("capture-oracle")
    capture.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    capture.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    capture.add_argument("--suite", choices=("fixed_pick_place_v3",), default="fixed_pick_place_v3")
    capture.add_argument("--scenario", default="nominal")
    capture.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    capture.add_argument("--no-video", action="store_true")
    recovery = commands.add_parser("capture-recovery")
    recovery.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    recovery.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    recovery.add_argument("--oracle-manifest", type=Path, required=True)
    recovery_eval = commands.add_parser("evaluate-recovery-starts")
    recovery_eval.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    recovery_eval.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    recovery_eval.add_argument("--oracle-manifest", type=Path, required=True)
    recovery_eval.add_argument("--recovery-manifest", type=Path, required=True)
    recovery_eval.add_argument("--checkpoint", type=Path, required=True)
    clone = commands.add_parser("evaluate-clone")
    clone.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    clone.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    clone.add_argument("--suite", choices=("fixed_pick_place_v3",), default="fixed_pick_place_v3")
    clone.add_argument("--scenario", choices=("nominal", "all"), default="all")
    clone.add_argument("--checkpoint", type=Path, required=True)
    clone.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    clone.add_argument("--no-video", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "evaluate-recovery-starts":
            report = evaluate_recovery_anchor_starts(
                args.mujoco_model, args.oracle_manifest, args.recovery_manifest,
                args.checkpoint, args.output_dir,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            print(report)
            print(f"passed={str(bool(payload['passed'])).lower()} anchors={len(payload['results'])}")
            return 0 if payload["passed"] else 2
        if args.command == "capture-recovery":
            result = capture_phase_wide_recovery_examples(
                args.mujoco_model, args.oracle_manifest, args.output_dir,
            )
            print(result.manifest)
            print(f"rows={result.rows}")
            return 0
        if args.command == "evaluate-clone":
            preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
            stated_preflight = preflight.get("content_sha256")
            preflight_body = dict(preflight)
            preflight_body.pop("content_sha256", None)
            if (
                stated_preflight != content_sha256(preflight_body)
                or
                preflight.get("environment_proven") is not True
                or preflight.get("deterministic") is not True
                or preflight.get("suite", {}).get("suite_id") != args.suite
            ):
                raise ValueError("clone evaluation requires a passing v3 preflight")
            suite = load_simulation_suite(args.suite)
            if args.scenario == "nominal":
                suite = replace(
                    suite,
                    suite_id=f"{suite.suite_id}.nominal",
                    scenarios=tuple(item for item in suite.scenarios if item.scenario_id == "nominal"),
                )
            checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
            probe = OracleCloneCheckpointPolicy(args.checkpoint)
            policies = {probe.policy_id: lambda: OracleCloneCheckpointPolicy(args.checkpoint)}
            evaluation_identity = content_sha256({
                "checkpoint_sha256": checkpoint_sha,
                "offline_report_content_sha256": probe.report_content_sha256,
                "oracle_manifest_content_sha256": probe.manifest_content_sha256,
                "preflight_report_content_sha256": stated_preflight,
                "suite_id": suite.suite_id,
            })
            result = evaluate_closed_loop(
                args.mujoco_model, suite, policies,
                args.output_dir / "clone_evaluations" / evaluation_identity[:16],
                environment_proven=True,
                record_video=not args.no_video,
                provenance={
                    "checkpoint_sha256": checkpoint_sha,
                    "offline_report_content_sha256": probe.report_content_sha256,
                    "oracle_manifest_content_sha256": probe.manifest_content_sha256,
                    "preflight_report_content_sha256": stated_preflight,
                },
            )
            print(result.report_json)
            expected_rollouts = len(suite.scenarios) * suite.repeats
            passed = bool(
                len(result.rollouts) == expected_rollouts
                and result.deterministic
                and all(
                    item.success
                    and not item.invalidated
                    and item.clipping_frames == 0
                    and item.limiting_frames == 0
                    and item.nonfinite_frames == 0
                    and item.unsafe_contact_frames == 0
                    for item in result.rollouts
                )
            )
            print(f"passed={str(passed).lower()} rollouts={len(result.rollouts)}")
            return 0 if passed else 2
        if args.command == "capture-oracle":
            result = capture_oracle_demonstrations(
                args.mujoco_model, args.suite, args.preflight_report, args.output_dir,
                scenario=args.scenario, record_video=not args.no_video,
            )
            print(result.manifest)
            print(f"rows={result.rows} scenarios={','.join(result.scenario_ids)}")
            return 0
        if args.command == "preflight":
            result = run_simulation_preflight(
                args.mujoco_model, args.output_dir, suite=args.suite,
                record_video=not args.no_video,
            )
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
