"""Horizon-alignment tranche: 480-action teacher, aligned recipe, three seeds.

Pre-registered in ``notes/horizon-alignment-proposal.md``: the students are
retrained on the 480-action five-scenario capture (teacher hold-tail after
full retreat) under the precision tranche's established recipe, at seeds
101/202/303, all run to completion. Promotion requires **all three seeds** to
pass Stage A — the >=3-seed standard is the rule itself, calibrated against
the precision tranche's measured dispersion (12/15, 12/15, 3/15).
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import multiprocessing
from pathlib import Path
from typing import Any, Mapping

from so_arm101_v2.data._serialization import content_sha256, write_immutable_json
from so_arm101_v2.learning.numerics import (
    AUTO,
    NumericsSpec,
    numerics_identity,
    resolve_default_numerics,
)

from .chunked import ChunkedClonePolicy
from .margins import ANALYZER_VERSION, analyze_pick_place_margins
from .observability import _load_hashed_json, _rollout_summary
from .oracle import load_oracle_demonstrations
from .policy_specs import PolicySpec
from .precision import cell_metrics
from .recovery import evaluate_policy_anchor_starts, load_oracle_recovery_examples
from .scaling import FROZEN_RECIPE, STAGE_A_PASS_RULE, STAGE_B_PASS_RULE
from .suites import load_simulation_suite

HORIZON_EXPERIMENT = "horizon_alignment_tranche_v1"
ALIGNED_TEACHER_HORIZON = 480

# The precision tranche's established recipe; the only new factor in this
# tranche is the aligned training data.
ALIGNED_RECIPE_OVERRIDES = {
    "hidden_width": 512,
    "max_steps": 90_000,
    "lr_schedule": "cosine_floor_v1",
}
HORIZON_SEEDS = (101, 202, 303)

PROMOTION_RULE = (
    "horizon_promoted_robust iff all three seeds pass Stage A (15/15 "
    "deterministic five-scenario rollouts with zero clip/limit/nonfinite/"
    "unsafe frames); any seed failing yields horizon_not_resolved"
)


@dataclass(frozen=True)
class HorizonGateResult:
    directory: Path
    report_json: Path
    status: str


def resolve_horizon_status(cells: list[Mapping[str, Any]]) -> str:
    expected = [f"seed{seed}" for seed in HORIZON_SEEDS]
    if [cell["cell_id"] for cell in cells] != expected:
        raise ValueError("horizon cells do not match the registered seed order")
    for cell in cells:
        if "stage_a_passed" not in cell or cell.get("stage_a_evaluation_content_sha256") is None:
            raise ValueError("horizon cell lacks evaluation evidence")
    if all(cell["stage_a_passed"] for cell in cells):
        return "horizon_promoted_robust"
    return "horizon_not_resolved"


def _train_horizon_cell(
    manifest_path: str, output_dir: str, seed: int,
    *, legacy_numerics: bool = False,
) -> str:
    """Train one seed cell; in-process (GPU) or spawn child (legacy)."""
    from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
    from so_arm101_v2.learning.numerics import AUTO as NUMERICS_AUTO

    result = train_chunked_clone(
        manifest_path, output_dir,
        config=ChunkedCloneConfig(
            **{**FROZEN_RECIPE, **ALIGNED_RECIPE_OVERRIDES, "seed": seed}
        ),
        numerics=None if legacy_numerics else NUMERICS_AUTO,
    )
    return str(result.report_json)


def run_horizon_gate(
    model_path: str | Path,
    capture_manifest_path: str | Path,
    oracle_manifest_path: str | Path,
    recovery_manifest_path: str | Path,
    preflight_report_path: str | Path,
    precision_report_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
    workers: int | None = None,
    numerics: NumericsSpec | None | str = AUTO,
) -> HorizonGateResult:
    """Run the pre-registered horizon-alignment tranche end to end."""
    from .rollout import evaluate_closed_loop

    if isinstance(numerics, str):
        if numerics != AUTO:
            raise ValueError(f"unknown numerics request {numerics!r}")
        numerics = resolve_default_numerics()

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    capture_manifest, _ = load_oracle_demonstrations(capture_manifest_path)
    episodes = capture_manifest.get("episodes", [])
    if (
        len(episodes) != 5
        or int(capture_manifest.get("teacher_horizon", 0)) != ALIGNED_TEACHER_HORIZON
        or sum(int(item["rows"]) for item in episodes) != 5 * ALIGNED_TEACHER_HORIZON
        or "nominal" not in capture_manifest.get("scenario_ids", [])
    ):
        raise ValueError(
            "horizon gate requires the five-scenario 480-action (2,400-row) capture"
        )
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    if oracle_manifest.get("scenario_ids") != ["nominal"]:
        raise ValueError("horizon gate requires the canonical nominal oracle capture")
    recovery_manifest, _ = load_oracle_recovery_examples(recovery_manifest_path)
    if (
        recovery_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
    ):
        raise ValueError("horizon gate manifests disagree")
    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("horizon gate requires the passing deterministic v3 preflight")
    precision_report = _load_hashed_json(
        precision_report_path, label="precision seeds report"
    )
    if (
        precision_report.get("experiment") != "precision_tranche_v1"
        or precision_report.get("stage") != "seeds"
    ):
        raise ValueError("horizon gate requires the terminal precision seeds report")

    suite = load_simulation_suite("fixed_pick_place_v3")
    expected_rollouts = len(suite.scenarios) * suite.repeats
    if expected_rollouts != 15:
        raise ValueError("horizon gate expects the fifteen-rollout v3 suite")

    identity: dict[str, Any] = {
        "schema_version": 1,
        "experiment": HORIZON_EXPERIMENT,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "capture_manifest_content_sha256": capture_manifest["content_sha256"],
        "capture_collection_digest": capture_manifest["collection_digest"],
        "teacher_horizon": ALIGNED_TEACHER_HORIZON,
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "precision_report_content_sha256": precision_report["content_sha256"],
        "frozen_recipe": dict(FROZEN_RECIPE),
        "aligned_recipe_overrides": dict(ALIGNED_RECIPE_OVERRIDES),
        "seeds_registered": list(HORIZON_SEEDS),
        "suite_id": suite.suite_id,
        "suite_repeats": suite.repeats,
        "expected_rollouts": expected_rollouts,
        "stage_a_pass_rule": STAGE_A_PASS_RULE,
        "stage_b_pass_rule": STAGE_B_PASS_RULE,
        "promotion_rule": PROMOTION_RULE,
        "margin_analyzer": ANALYZER_VERSION,
        "retry_budget": 0,
    }
    if numerics is not None:
        identity["numerics"] = numerics_identity(numerics)
    gate_digest = content_sha256(identity)
    directory = output_dir / "horizon_gates" / gate_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="horizon gate report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable horizon gate differs: {directory}")
        return HorizonGateResult(directory, report_path, str(existing["status"]))

    manifest_argument = str(Path(capture_manifest_path).resolve())
    if numerics is not None:
        training_reports = {
            seed: Path(_train_horizon_cell(
                manifest_argument, str(output_dir), seed, legacy_numerics=False,
            ))
            for seed in HORIZON_SEEDS
        }
    else:
        with ProcessPoolExecutor(
            max_workers=len(HORIZON_SEEDS),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                seed: pool.submit(
                    _train_horizon_cell, manifest_argument, str(output_dir),
                    seed, legacy_numerics=True,
                )
                for seed in HORIZON_SEEDS
            }
            training_reports = {
                seed: Path(future.result()) for seed, future in futures.items()
            }

    def _evaluate_cell(seed: int) -> dict[str, Any]:
        cell_id = f"seed{seed}"
        training_report_path = training_reports[seed]
        training_report = _load_hashed_json(
            training_report_path, label="cell training report"
        )
        checkpoint = training_report_path.parent / "model.pt"
        probe = ChunkedClonePolicy(checkpoint)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        cell_suite = replace(suite, suite_id=f"{suite.suite_id}.horizon_{cell_id}")
        evaluation_identity = content_sha256({
            "gate_digest": gate_digest,
            "stage": f"stage_a_{cell_id}",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_report_content_sha256": probe.report_content_sha256,
            "preflight_report_content_sha256": preflight["content_sha256"],
            "suite_id": cell_suite.suite_id,
        })
        evaluation = evaluate_closed_loop(
            model_path,
            cell_suite,
            {probe.policy_id: PolicySpec(kind="chunked_clone", checkpoint=str(checkpoint.resolve()))},
            output_dir / "horizon_stage_a_evaluations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=record_video,
            workers=workers,
            provenance={
                "gate_digest": gate_digest,
                "stage": f"stage_a_{cell_id}",
                "checkpoint_sha256": checkpoint_sha,
                "offline_report_content_sha256": probe.report_content_sha256,
                "preflight_report_content_sha256": preflight["content_sha256"],
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
            "seed": seed,
            "training_report": str(training_report_path.resolve()),
            "training_report_content_sha256": training_report["content_sha256"],
            "offline_normalized_mse": float(training_report["normalized_mse"]),
            "offline_max_act_error": float(training_report["max_act_error"]),
            "training_steps": int(training_report["steps"]),
            "checkpoint_sha256": checkpoint_sha,
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

    cells = [_evaluate_cell(seed) for seed in HORIZON_SEEDS]
    status = resolve_horizon_status(cells)
    report: dict[str, Any] = {
        **identity,
        "gate_digest": gate_digest,
        "status": status,
        "cells": cells,
        "margins_are_telemetry_only": True,
        "handoffs_never_gate": True,
        "offline_metrics_are_telemetry_only": True,
        "next_action": (
            "write_vision_rung_proposal" if status == "horizon_promoted_robust"
            else "analyze_failures_and_write_new_proposal"
        ),
        "claim": "horizon_alignment_decision_not_deployment_success",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return HorizonGateResult(directory, report_path, status)


__all__ = [
    "ALIGNED_RECIPE_OVERRIDES",
    "ALIGNED_TEACHER_HORIZON",
    "HORIZON_EXPERIMENT",
    "HORIZON_SEEDS",
    "HorizonGateResult",
    "PROMOTION_RULE",
    "resolve_horizon_status",
    "run_horizon_gate",
]
