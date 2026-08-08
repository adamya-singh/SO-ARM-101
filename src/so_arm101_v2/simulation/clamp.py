"""Gripper-clamp gate: eval-only re-run of the horizon checkpoints, clamped.

Pre-registered in ``notes/gripper-clamp-proposal.md``: the three horizon-
alignment seed checkpoints are re-evaluated with the gripper channel clamped
to its exact effective bounds at the policy output (`ChunkedClonePolicy
clamp_channels=(5,)`). No training happens; the cells are cited from the
horizon gate report by content hash and validated byte-for-byte.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Mapping

from so_arm101_v2.data._serialization import content_sha256, write_immutable_json

from .chunked import ChunkedClonePolicy
from .margins import ANALYZER_VERSION, analyze_pick_place_margins
from .observability import _load_hashed_json, _rollout_summary
from .oracle import load_oracle_demonstrations
from .policy_specs import PolicySpec
from .precision import cell_metrics
from .recovery import evaluate_policy_anchor_starts, load_oracle_recovery_examples
from .scaling import STAGE_A_PASS_RULE, STAGE_B_PASS_RULE
from .suites import load_simulation_suite

CLAMP_EXPERIMENT = "gripper_clamp_gate_v1"
CLAMP_CHANNELS = (5,)

PROMOTION_RULE = (
    "clamp_promoted_robust iff all three cited horizon seed checkpoints, "
    "evaluated with the gripper channel clamped to the exact effective "
    "envelope at the policy output, pass Stage A (15/15 deterministic "
    "five-scenario rollouts with zero clip/limit/nonfinite/unsafe frames); "
    "any failure yields clamp_not_resolved"
)


@dataclass(frozen=True)
class ClampGateResult:
    directory: Path
    report_json: Path
    status: str


def resolve_clamp_status(cells: list[Mapping[str, Any]]) -> str:
    if not cells:
        raise ValueError("clamp gate requires at least one cell")
    for cell in cells:
        if "stage_a_passed" not in cell or cell.get("stage_a_evaluation_content_sha256") is None:
            raise ValueError("clamp cell lacks evaluation evidence")
    if all(cell["stage_a_passed"] for cell in cells):
        return "clamp_promoted_robust"
    return "clamp_not_resolved"


def run_clamp_gate(
    model_path: str | Path,
    horizon_gate_report_path: str | Path,
    oracle_manifest_path: str | Path,
    recovery_manifest_path: str | Path,
    preflight_report_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
    workers: int | None = None,
) -> ClampGateResult:
    """Run the pre-registered gripper-clamp gate end to end (no training)."""
    from .rollout import evaluate_closed_loop

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    horizon_report = _load_hashed_json(
        horizon_gate_report_path, label="horizon gate report"
    )
    if (
        horizon_report.get("experiment") != "horizon_alignment_tranche_v1"
        or horizon_report.get("status") != "horizon_not_resolved"
    ):
        raise ValueError("clamp gate requires the terminal horizon gate report")
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    if oracle_manifest.get("scenario_ids") != ["nominal"]:
        raise ValueError("clamp gate requires the canonical nominal oracle capture")
    recovery_manifest, _ = load_oracle_recovery_examples(recovery_manifest_path)
    if (
        recovery_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
    ):
        raise ValueError("clamp gate manifests disagree")
    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("clamp gate requires the passing deterministic v3 preflight")

    # Cite the cells: checkpoint bytes and training reports must match the
    # horizon gate record exactly.
    cited_cells: list[dict[str, Any]] = []
    for cell in horizon_report.get("cells", []):
        training_report_path = Path(cell["training_report"])
        checkpoint = training_report_path.parent / "model.pt"
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if checkpoint_sha != cell["checkpoint_sha256"]:
            raise ValueError(
                f"checkpoint bytes for {cell['cell_id']} do not match the cited "
                "horizon gate report"
            )
        training_report = _load_hashed_json(
            training_report_path, label="cited training report"
        )
        if training_report["content_sha256"] != cell["training_report_content_sha256"]:
            raise ValueError(
                f"training report for {cell['cell_id']} does not match the cited "
                "horizon gate report"
            )
        cited_cells.append({
            "cell_id": cell["cell_id"],
            "seed": int(cell["seed"]),
            "checkpoint": checkpoint,
            "checkpoint_sha256": checkpoint_sha,
            "training_report_content_sha256": cell["training_report_content_sha256"],
        })
    if len(cited_cells) != 3:
        raise ValueError("clamp gate expects exactly the three horizon seed cells")

    suite = load_simulation_suite("fixed_pick_place_v3")
    expected_rollouts = len(suite.scenarios) * suite.repeats
    if expected_rollouts != 15:
        raise ValueError("clamp gate expects the fifteen-rollout v3 suite")

    identity: dict[str, Any] = {
        "schema_version": 1,
        "experiment": CLAMP_EXPERIMENT,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "horizon_gate_report_content_sha256": horizon_report["content_sha256"],
        "clamp_channels": list(CLAMP_CHANNELS),
        "clamp_bound": "exact_effective_safe_act_bounds",
        "cells_cited": [
            {
                "cell_id": cell["cell_id"],
                "seed": cell["seed"],
                "checkpoint_sha256": cell["checkpoint_sha256"],
                "training_report_content_sha256": cell["training_report_content_sha256"],
            }
            for cell in cited_cells
        ],
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "suite_id": suite.suite_id,
        "suite_repeats": suite.repeats,
        "expected_rollouts": expected_rollouts,
        "stage_a_pass_rule": STAGE_A_PASS_RULE,
        "stage_b_pass_rule": STAGE_B_PASS_RULE,
        "promotion_rule": PROMOTION_RULE,
        "margin_analyzer": ANALYZER_VERSION,
        "training": "none_eval_only_variant_of_cited_checkpoints",
        "retry_budget": 0,
    }
    gate_digest = content_sha256(identity)
    directory = output_dir / "clamp_gates" / gate_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="clamp gate report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable clamp gate differs: {directory}")
        return ClampGateResult(directory, report_path, str(existing["status"]))

    def _evaluate_cell(cited: Mapping[str, Any]) -> dict[str, Any]:
        cell_id = str(cited["cell_id"])
        checkpoint = Path(cited["checkpoint"])
        probe = ChunkedClonePolicy(checkpoint, clamp_channels=CLAMP_CHANNELS)
        cell_suite = replace(suite, suite_id=f"{suite.suite_id}.clamp_{cell_id}")
        evaluation_identity = content_sha256({
            "gate_digest": gate_digest,
            "stage": f"stage_a_{cell_id}",
            "checkpoint_sha256": cited["checkpoint_sha256"],
            "checkpoint_report_content_sha256": probe.report_content_sha256,
            "preflight_report_content_sha256": preflight["content_sha256"],
            "suite_id": cell_suite.suite_id,
            "clamp_channels": list(CLAMP_CHANNELS),
        })
        evaluation = evaluate_closed_loop(
            model_path,
            cell_suite,
            {probe.policy_id: PolicySpec(
                kind="chunked_clone",
                checkpoint=str(checkpoint.resolve()),
                options=(("clamp_channels", CLAMP_CHANNELS),),
            )},
            output_dir / "clamp_stage_a_evaluations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=record_video,
            workers=workers,
            provenance={
                "gate_digest": gate_digest,
                "stage": f"stage_a_{cell_id}",
                "checkpoint_sha256": cited["checkpoint_sha256"],
                "offline_report_content_sha256": probe.report_content_sha256,
                "preflight_report_content_sha256": preflight["content_sha256"],
                "clamp_channels": list(CLAMP_CHANNELS),
            },
        )
        evaluation_report = _load_hashed_json(
            evaluation.report_json, label="stage A evaluation report"
        )
        rollouts = evaluation_report.get("rollouts", [])
        stage_a_passed = bool(
            evaluation_report.get("deterministic") is True
            and len(rollouts) == expected_rollouts
            and all(
                item.get("success") is True
                and item.get("invalidated") is False
                and int(item.get("clipping_frames", -1)) == 0
                and int(item.get("limiting_frames", -1)) == 0
                and int(item.get("nonfinite_frames", -1)) == 0
                and int(item.get("unsafe_contact_frames", -1)) == 0
                for item in rollouts
            )
        )
        margins_path = analyze_pick_place_margins(
            evaluation.report_json, output_dir, mujoco_model_path=model_path,
        )
        margins_report = _load_hashed_json(margins_path, label="margin analysis report")
        anchor_path = evaluate_policy_anchor_starts(
            model_path, oracle_manifest_path, recovery_manifest_path,
            checkpoint, output_dir,
        )
        anchor_report = _load_hashed_json(anchor_path, label="anchor handoff report")
        record = {
            "cell_id": cell_id,
            "seed": int(cited["seed"]),
            "checkpoint_sha256": cited["checkpoint_sha256"],
            "checkpoint_report_content_sha256": probe.report_content_sha256,
            "policy_id": probe.policy_id,
            "stage_a_evaluation": str(evaluation.report_json.resolve()),
            "stage_a_evaluation_content_sha256": evaluation_report["content_sha256"],
            "stage_a_passed": stage_a_passed,
            "stage_a_rollouts": [_rollout_summary(item) for item in rollouts],
            "margin_report": str(margins_path.resolve()),
            "margin_report_content_sha256": margins_report["content_sha256"],
            "stage_b_report": str(anchor_path.resolve()),
            "stage_b_report_content_sha256": anchor_report["content_sha256"],
            "stage_b_passed": bool(anchor_report["passed"]),
            "stage_b_results": anchor_report["results"],
        }
        successes, frames = cell_metrics(record)
        record["stage_a_successes"] = successes
        record["total_safety_frames"] = frames
        return record

    cells = [_evaluate_cell(cited) for cited in cited_cells]
    status = resolve_clamp_status(cells)
    report: dict[str, Any] = {
        **identity,
        "gate_digest": gate_digest,
        "status": status,
        "cells": cells,
        "margins_are_telemetry_only": True,
        "handoffs_never_gate": True,
        "next_action": (
            "write_vision_rung_proposal" if status == "clamp_promoted_robust"
            else "analyze_failures_then_start_vision_rung_anyway"
        ),
        "claim": "gripper_clamp_decision_not_deployment_success",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return ClampGateResult(directory, report_path, status)


__all__ = [
    "CLAMP_CHANNELS",
    "CLAMP_EXPERIMENT",
    "ClampGateResult",
    "PROMOTION_RULE",
    "resolve_clamp_status",
    "run_clamp_gate",
]
