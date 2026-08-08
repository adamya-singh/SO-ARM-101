"""CLI for named reward-independent MuJoCo evaluation suites."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from so_arm101_v2.learning import load_small_model_comparison
from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.learning.numerics import resolve_numerics

from .policy_specs import PolicySpec
from .rollout import (
    evaluate_closed_loop,
    run_simulation_preflight,
)
from .oracle import capture_oracle_demonstrations
from .observability import (
    capture_observability_annotations,
    run_bounded_observability_gate,
)
from .recovery import (
    capture_phase_wide_recovery_examples,
    evaluate_recovery_anchor_starts,
    scan_oracle_clone_commands,
)
from .clone_policy import OracleCloneCheckpointPolicy
from .broader import run_broader_evaluation
from .scaling import run_scaling_gate
from .precision import run_precision_stage
from .chunked import run_chunked_gate, run_saturation_gate
from .margins import analyze_pick_place_margins
from .correction import (
    capture_dagger_corrections,
    run_correction_gate,
    run_correction_probe,
)
from .suites import load_simulation_suite


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="so-arm101-v2-sim")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("preflight", "evaluate"):
        command = commands.add_parser(name)
        command.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
        command.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/simulation"))
        command.add_argument("--no-video", action="store_true")
        command.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
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
    observability = commands.add_parser("capture-observability")
    observability.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    observability.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    observability.add_argument("--oracle-manifest", type=Path, required=True)
    observability.add_argument("--recovery-manifest", type=Path, required=True)
    gate = commands.add_parser("run-observability-gate")
    gate.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    gate.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    gate.add_argument("--oracle-manifest", type=Path, required=True)
    gate.add_argument("--recovery-manifest", type=Path, required=True)
    gate.add_argument("--observability-manifest", type=Path, required=True)
    gate.add_argument("--baseline-checkpoint", type=Path, required=True)
    gate.add_argument("--baseline-evaluation", type=Path, required=True)
    gate.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    gate.add_argument("--no-video", action="store_true")
    gate.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
    corrections = commands.add_parser("capture-corrections")
    corrections.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    corrections.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    corrections.add_argument("--oracle-manifest", type=Path, required=True)
    corrections.add_argument("--checkpoint", type=Path, required=True)
    correction_gate = commands.add_parser("run-correction-gate")
    correction_gate.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    correction_gate.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    correction_gate.add_argument("--oracle-manifest", type=Path, required=True)
    correction_gate.add_argument("--correction-manifest", type=Path, required=True)
    correction_gate.add_argument("--baseline-checkpoint", type=Path, required=True)
    correction_gate.add_argument("--baseline-evaluation", type=Path, required=True)
    correction_gate.add_argument("--prefix-parity-report", type=Path, required=True)
    correction_gate.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    correction_gate.add_argument("--no-video", action="store_true")
    correction_gate.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
    chunked = commands.add_parser("run-chunked-gate")
    chunked.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    chunked.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    chunked.add_argument("--oracle-manifest", type=Path, required=True)
    chunked.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    chunked.add_argument("--no-video", action="store_true")
    chunked.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="training numerics device (auto = pinned GPU regime when CUDA is live)")
    chunked.add_argument("--no-compile", action="store_true", help="disable torch.compile in the numerics regime")
    chunked.add_argument("--legacy-numerics", action="store_true", help="exact legacy CPU-eager training lane; reproduces pre-v2 digests byte-for-byte")
    chunked.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
    probe = commands.add_parser("run-correction-probe")
    probe.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    probe.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    probe.add_argument("--checkpoint", type=Path, required=True)
    probe.add_argument("--baseline-evaluation", type=Path, required=True)
    probe.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    probe.add_argument("--no-video", action="store_true")
    probe.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
    saturation = commands.add_parser("run-saturation-gate")
    saturation.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    saturation.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    saturation.add_argument("--oracle-manifest", type=Path, required=True)
    saturation.add_argument("--correction-manifest", type=Path, required=True)
    saturation.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    saturation.add_argument("--no-video", action="store_true")
    saturation.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="training numerics device (auto = pinned GPU regime when CUDA is live)")
    saturation.add_argument("--no-compile", action="store_true", help="disable torch.compile in the numerics regime")
    saturation.add_argument("--legacy-numerics", action="store_true", help="exact legacy CPU-eager training lane; reproduces pre-v2 digests byte-for-byte")
    saturation.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
    broader = commands.add_parser("run-broader-evaluation")
    broader.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    broader.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    broader.add_argument("--checkpoint", type=Path, required=True)
    broader.add_argument("--gate-report", type=Path, required=True)
    broader.add_argument("--oracle-manifest", type=Path, required=True)
    broader.add_argument("--recovery-manifest", type=Path, required=True)
    broader.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    broader.add_argument("--no-video", action="store_true")
    broader.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="training numerics device (auto = pinned GPU regime when CUDA is live)")
    broader.add_argument("--no-compile", action="store_true", help="disable torch.compile in the numerics regime")
    broader.add_argument("--legacy-numerics", action="store_true", help="exact legacy CPU-eager training lane; reproduces pre-v2 digests byte-for-byte")
    broader.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
    margins_parser = commands.add_parser("analyze-margins")
    margins_parser.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    margins_parser.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    margins_parser.add_argument("--evaluation-report", type=Path, required=True)
    scaling = commands.add_parser("run-scaling-gate")
    scaling.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    scaling.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    scaling.add_argument("--capture-manifest", type=Path, required=True)
    scaling.add_argument("--oracle-manifest", type=Path, required=True)
    scaling.add_argument("--recovery-manifest", type=Path, required=True)
    scaling.add_argument("--baseline-tranche-report", type=Path, required=True)
    scaling.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    scaling.add_argument("--no-video", action="store_true")
    scaling.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="training numerics device (auto = pinned GPU regime when CUDA is live)")
    scaling.add_argument("--no-compile", action="store_true", help="disable torch.compile in the numerics regime")
    scaling.add_argument("--legacy-numerics", action="store_true", help="exact legacy CPU-eager training lane; reproduces pre-v2 digests byte-for-byte")
    precision = commands.add_parser("run-precision-stage")
    precision.add_argument("--stage", choices=("schedule", "budget", "seeds"), required=True)
    precision.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    precision.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    precision.add_argument("--capture-manifest", type=Path, required=True)
    precision.add_argument("--oracle-manifest", type=Path, required=True)
    precision.add_argument("--recovery-manifest", type=Path, required=True)
    precision.add_argument("--scaling-gate-report", type=Path, required=True)
    precision.add_argument("--prior-stage-report", type=Path)
    precision.add_argument(
        "--preflight-report", type=Path,
        default=Path("artifacts/so_arm101_v2/simulation/preflight/fixed_pick_place_v3/evaluation.json"),
    )
    precision.add_argument("--no-video", action="store_true")
    precision.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
    precision.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto", help="training numerics device (auto = pinned GPU regime when CUDA is live)")
    precision.add_argument("--no-compile", action="store_true", help="disable torch.compile in the numerics regime")
    precision.add_argument("--legacy-numerics", action="store_true", help="exact legacy CPU-eager training lane; reproduces pre-v2 digests byte-for-byte")
    recovery_eval = commands.add_parser("evaluate-recovery-starts")
    recovery_eval.add_argument("--mujoco-model", type=Path, default=Path("simulation_code/model/menagerie_so_arm100/scene_v2.xml"))
    recovery_eval.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    recovery_eval.add_argument("--oracle-manifest", type=Path, required=True)
    recovery_eval.add_argument("--recovery-manifest", type=Path, required=True)
    recovery_eval.add_argument("--checkpoint", type=Path, required=True)
    static_scan = commands.add_parser("scan-clone-static")
    static_scan.add_argument("--output-dir", type=Path, default=Path("artifacts/so_arm101_v2/oracle_distillation"))
    static_scan.add_argument("--oracle-manifest", type=Path, required=True)
    static_scan.add_argument("--recovery-manifest", type=Path, required=True)
    static_scan.add_argument("--checkpoint", type=Path, required=True)
    static_scan.add_argument("--samples-per-path", type=int, default=101)
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
    clone.add_argument("--workers", type=int, default=None, help="process-level parallelism for independent rollouts (default: auto = min(rollouts, 10 with video, cores-2 without); pass 1 to force sequential)")
    return parser


# The evidence lineage is pinned to this exact MuJoCo build.  A shadowed
# install (e.g. pip --user mujoco 3.11.0, 2026-08-02) silently forks physics
# numerics; run with PYTHONNOUSERSITE=1 so the conda env's pin wins.  See
# notes/parallel-execution-infrastructure.md.
EXPECTED_MUJOCO_VERSION = "3.9.0"


def _resolve_cli_numerics(args: argparse.Namespace) -> Any:
    if args.legacy_numerics:
        return None
    return resolve_numerics(args.device, compile=not args.no_compile)


def _require_pinned_mujoco() -> None:
    import mujoco

    if mujoco.__version__ != EXPECTED_MUJOCO_VERSION:
        raise RuntimeError(
            f"mujoco {mujoco.__version__} (from {mujoco.__file__}) does not match "
            f"the pinned evidence version {EXPECTED_MUJOCO_VERSION}; "
            "run with PYTHONNOUSERSITE=1 (the lerobot conda activate hook sets it)"
        )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _require_pinned_mujoco()
        if args.command == "run-observability-gate":
            result = run_bounded_observability_gate(
                args.mujoco_model, args.oracle_manifest, args.recovery_manifest,
                args.observability_manifest, args.baseline_checkpoint,
                args.baseline_evaluation, args.preflight_report, args.output_dir,
                record_video=not args.no_video,
                workers=args.workers,
            )
            print(result.report_json)
            print(f"status={result.status}")
            return 0 if result.status.startswith("passed_") else 2
        if args.command == "capture-observability":
            result = capture_observability_annotations(
                args.mujoco_model, args.oracle_manifest, args.recovery_manifest,
                args.output_dir,
            )
            print(result.manifest)
            return 0
        if args.command == "scan-clone-static":
            report = scan_oracle_clone_commands(
                args.oracle_manifest, args.recovery_manifest, args.checkpoint,
                args.output_dir, samples_per_path=args.samples_per_path,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            print(report)
            print(f"passed={str(bool(payload['passed'])).lower()} samples={payload['total_samples']}")
            return 0 if payload["passed"] else 2
        if args.command == "evaluate-recovery-starts":
            report = evaluate_recovery_anchor_starts(
                args.mujoco_model, args.oracle_manifest, args.recovery_manifest,
                args.checkpoint, args.output_dir,
            )
            payload = json.loads(report.read_text(encoding="utf-8"))
            print(report)
            print(f"passed={str(bool(payload['passed'])).lower()} anchors={len(payload['results'])}")
            return 0 if payload["passed"] else 2
        if args.command == "capture-corrections":
            result = capture_dagger_corrections(
                args.mujoco_model, args.oracle_manifest, args.checkpoint, args.output_dir,
            )
            print(result.manifest)
            print(
                f"rows={result.rows} "
                f"accepted_sites={','.join(result.accepted_sites)} "
                f"rejected_sites={','.join(result.rejected_sites) or 'none'}"
            )
            return 0
        if args.command == "run-correction-gate":
            gate_result = run_correction_gate(
                args.mujoco_model, args.oracle_manifest, args.correction_manifest,
                args.baseline_checkpoint, args.baseline_evaluation,
                args.prefix_parity_report, args.preflight_report, args.output_dir,
                record_video=not args.no_video,
                workers=args.workers,
            )
            print(gate_result.report_json)
            print(f"status={gate_result.status}")
            return 0 if gate_result.status == "passed" else 2
        if args.command == "run-chunked-gate":
            chunked_result = run_chunked_gate(
                args.mujoco_model, args.oracle_manifest, args.preflight_report,
                args.output_dir, record_video=not args.no_video,
                workers=args.workers,
                numerics=_resolve_cli_numerics(args),
            )
            print(chunked_result.report_json)
            print(f"status={chunked_result.status}")
            return 0 if chunked_result.status.startswith("passed_") else 2
        if args.command == "run-correction-probe":
            probe_result = run_correction_probe(
                args.mujoco_model, args.checkpoint, args.baseline_evaluation,
                args.preflight_report, args.output_dir,
                record_video=not args.no_video,
                workers=args.workers,
            )
            print(probe_result.report_json)
            print(f"status={probe_result.status}")
            return 0
        if args.command == "run-saturation-gate":
            saturation_result = run_saturation_gate(
                args.mujoco_model, args.oracle_manifest, args.correction_manifest,
                args.preflight_report, args.output_dir,
                record_video=not args.no_video,
                workers=args.workers,
                numerics=_resolve_cli_numerics(args),
            )
            print(saturation_result.report_json)
            print(f"status={saturation_result.status}")
            return 0 if saturation_result.status.startswith("promoted_") else 2
        if args.command == "run-broader-evaluation":
            broader_result = run_broader_evaluation(
                args.mujoco_model, args.checkpoint, args.gate_report,
                args.oracle_manifest, args.recovery_manifest,
                args.preflight_report, args.output_dir,
                record_video=not args.no_video,
                workers=args.workers,
                numerics=_resolve_cli_numerics(args),
            )
            print(broader_result.report_json)
            print(f"status={broader_result.status}")
            return 0 if broader_result.status.endswith("_robust") else 2
        if args.command == "analyze-margins":
            margins_path = analyze_pick_place_margins(
                args.evaluation_report, args.output_dir,
                mujoco_model_path=args.mujoco_model,
            )
            print(margins_path)
            return 0
        if args.command == "run-scaling-gate":
            scaling_result = run_scaling_gate(
                args.mujoco_model, args.capture_manifest,
                args.oracle_manifest, args.recovery_manifest,
                args.preflight_report, args.baseline_tranche_report,
                args.output_dir,
                record_video=not args.no_video,
                workers=args.workers,
                numerics=_resolve_cli_numerics(args),
            )
            print(scaling_result.report_json)
            print(f"status={scaling_result.status}")
            return 0 if scaling_result.status.endswith("_robust") else 2
        if args.command == "run-precision-stage":
            precision_result = run_precision_stage(
                args.stage,
                args.mujoco_model, args.capture_manifest,
                args.oracle_manifest, args.recovery_manifest,
                args.preflight_report, args.scaling_gate_report,
                args.output_dir,
                prior_stage_report_path=args.prior_stage_report,
                record_video=not args.no_video,
                workers=args.workers,
                numerics=_resolve_cli_numerics(args),
            )
            print(precision_result.report_json)
            print(f"status={precision_result.status}")
            passing = (
                precision_result.status == "schedule_selected_cosine_floor_v1_stage_a_passed"
                or precision_result.status.startswith("budget_promoted_")
                or precision_result.status == "seeds_robust"
            )
            return 0 if passing else 2
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
            policies = {probe.policy_id: PolicySpec(kind="oracle_clone", checkpoint=str(args.checkpoint.resolve()))}
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
                workers=args.workers,
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
                workers=args.workers,
            )
            print(result.report_json)
            print(f"environment_proven={str(bool(result.environment_proven)).lower()}")
            return 0 if result.environment_proven else 2
        comparison = load_small_model_comparison(args.comparison_report)
        offline = json.loads(comparison.report_json.read_text(encoding="utf-8"))
        mean = np.asarray(offline["development_baselines"]["train_mean_target"], dtype=np.float32)
        policies = {
            "current_pose": PolicySpec(kind="current_pose"),
            "train_mean": PolicySpec(
                kind="constant_pose",
                options=(("target_act", tuple(float(value) for value in mean)),),
            ),
        }
        for run in comparison.runs:
            policy_id = f"{run.kind.value}.seed{run.seed}"
            checkpoint = str(run.checkpoint.resolve())
            policies[policy_id] = PolicySpec(kind="torch_checkpoint", checkpoint=checkpoint)
            if run.kind.value == "image_state":
                policies[policy_id + ".black"] = PolicySpec(
                    kind="torch_checkpoint",
                    checkpoint=checkpoint,
                    options=(("black_image", True),),
                )
        preflight = json.loads(args.preflight_report.read_text(encoding="utf-8"))
        result = evaluate_closed_loop(
            args.mujoco_model, args.suite, policies, args.output_dir,
            environment_proven=bool(preflight.get("environment_proven")),
            record_video=not args.no_video,
            workers=args.workers,
        )
        print(result.report_json)
        return 0
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["main"]
