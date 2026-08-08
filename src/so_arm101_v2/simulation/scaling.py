"""Optimization-scaling tranche for the five-scenario chunked recipe.

Pre-registered in ``notes/optimization-scaling-proposal.md``: a 2x2 factorial
over optimization budget (steps 30k/90k) and capacity (width 256/512) on the
immutable five-scenario 2,250-row oracle capture, with the broader tranche's
retrained 256/30k cell as the already-run control.  Every candidate runs to
completion (no early stop); Stage A gates promotion, Stage B and margins are
recorded and never gate.  Promotion takes the first passing candidate in
minimal-change order.
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
from .policy_specs import PolicySpec
from .margins import ANALYZER_VERSION, analyze_pick_place_margins
from .observability import _load_hashed_json, _rollout_summary
from .oracle import load_oracle_demonstrations
from .recovery import evaluate_policy_anchor_starts, load_oracle_recovery_examples
from .suites import load_simulation_suite

# The frozen noise-penalty recipe; candidates may change nothing but the two
# pre-registered scaled factors.
FROZEN_RECIPE = {
    "chunk_horizon": 90,
    "seed": 101,
    "learning_rate": 1e-3,
    "saturation_mode": "noise_penalty_v1",
    "decoder_eta": 0.9,
    "margin_act": 0.002,
    "noise_sigma": 0.05,
    "penalty_weight": 1.0,
}

# Minimal-change promotion order; the 256/30k cell is the immutable control
# inside the baseline broader tranche and is referenced, never re-run.
SCALING_CANDIDATES = (
    ("steps90k", {"hidden_width": 256, "max_steps": 90_000}),
    ("width512", {"hidden_width": 512, "max_steps": 30_000}),
    ("width512_steps90k", {"hidden_width": 512, "max_steps": 90_000}),
)

STAGE_A_PASS_RULE = (
    "deterministic, all fifteen scenario rollouts succeed without invalidation "
    "and with zero clip/limit/nonfinite/unsafe frames"
)
STAGE_B_PASS_RULE = "all eight anchor handoffs succeed with zero student safety counts"
PROMOTION_RULE = (
    "all candidates run to completion; the promoted candidate is the first in "
    "the pre-registered order (steps90k, width512, width512_steps90k) whose "
    "Stage A passes; Stage B and margins never gate"
)


@dataclass(frozen=True)
class ScalingGateResult:
    directory: Path
    report_json: Path
    status: str


def resolve_scaling_status(candidates: list[Mapping[str, Any]]) -> str:
    """Derive the terminal status from completed candidate records."""
    expected = [candidate_id for candidate_id, _ in SCALING_CANDIDATES]
    if [item.get("candidate_id") for item in candidates] != expected:
        raise ValueError("scaling status requires all candidates in registered order")
    for item in candidates:
        if item.get("stage_a_passed"):
            suffix = "robust" if item.get("stage_b_passed") else "starts_only"
            return f"promoted_{item['candidate_id']}_{suffix}"
    return "scaling_not_resolved"


def _train_scaling_candidate(
    manifest_path: str, output_dir: str, overrides: dict[str, Any],
    *, legacy_numerics: bool = False,
) -> str:
    """Training entry point: train one candidate, return its report path.

    Safe both in-process (regime v2, one CUDA context) and as a spawn-pool
    child (legacy lane); the numerics regime is re-resolved and re-guarded
    inside ``train_chunked_clone`` either way.
    """
    from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
    from so_arm101_v2.learning.numerics import AUTO

    result = train_chunked_clone(
        manifest_path, output_dir,
        config=ChunkedCloneConfig(**FROZEN_RECIPE, **overrides),
        numerics=None if legacy_numerics else AUTO,
    )
    return str(result.report_json)


def run_scaling_gate(
    model_path: str | Path,
    capture_manifest_path: str | Path,
    oracle_manifest_path: str | Path,
    recovery_manifest_path: str | Path,
    preflight_report_path: str | Path,
    baseline_tranche_report_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
    workers: int | None = None,
    numerics: "NumericsSpec | None | str" = AUTO,
) -> ScalingGateResult:
    """Run the pre-registered optimization-scaling tranche end to end."""
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
        or sum(int(item["rows"]) for item in episodes) != 2250
        or int(capture_manifest.get("teacher_horizon", 0)) != 450
        or "nominal" not in capture_manifest.get("scenario_ids", [])
    ):
        raise ValueError("scaling gate requires the five-scenario 2,250-row capture")
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    if (
        oracle_manifest.get("scenario_ids") != ["nominal"]
        or int(oracle_manifest.get("teacher_horizon", 0)) != 450
    ):
        raise ValueError("scaling gate requires the canonical nominal oracle capture")
    recovery_manifest, _ = load_oracle_recovery_examples(recovery_manifest_path)
    if (
        recovery_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
    ):
        raise ValueError("scaling gate manifests disagree")
    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("scaling gate requires the passing deterministic v3 preflight")
    baseline = _load_hashed_json(
        baseline_tranche_report_path, label="baseline broader tranche report"
    )
    baseline_retrained = baseline.get("retrained") or {}
    if (
        baseline.get("experiment") != "broader_evaluation_tranche_v1"
        or baseline.get("status") != "starts_not_resolved"
        or baseline_retrained.get("capture_manifest_content_sha256")
        != capture_manifest["content_sha256"]
    ):
        raise ValueError(
            "baseline tranche does not contain the 256/30k control cell for this capture"
        )

    suite = load_simulation_suite("fixed_pick_place_v3")
    expected_rollouts = len(suite.scenarios) * suite.repeats
    if expected_rollouts != 15:
        raise ValueError("scaling gate expects the fifteen-rollout v3 suite")
    identity = {
        "schema_version": 1,
        "experiment": "optimization_scaling_tranche_v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "capture_manifest_content_sha256": capture_manifest["content_sha256"],
        "capture_collection_digest": capture_manifest["collection_digest"],
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "baseline_tranche_content_sha256": baseline["content_sha256"],
        "baseline_cell": {
            "hidden_width": 256,
            "max_steps": 30_000,
            "checkpoint_sha256": baseline_retrained.get("checkpoint_sha256"),
            "stage_a_passed": baseline_retrained.get("stage_a_passed"),
            "stage_b_passed": baseline_retrained.get("stage_b_passed"),
        },
        "frozen_recipe": dict(FROZEN_RECIPE),
        "candidates_registered": [
            {"candidate_id": candidate_id, **overrides}
            for candidate_id, overrides in SCALING_CANDIDATES
        ],
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
    tranche_digest = content_sha256(identity)
    directory = output_dir / "scaling_gates" / tranche_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="scaling gate report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable scaling gate differs: {directory}")
        return ScalingGateResult(directory, report_path, str(existing["status"]))

    manifest_argument = str(Path(capture_manifest_path).resolve())
    if numerics is not None:
        # One GPU: sequential in-process trainings beat three concurrent CUDA
        # contexts, share one warm inductor cache, and each arm is minutes.
        training_reports = {
            candidate_id: Path(_train_scaling_candidate(
                manifest_argument, str(output_dir), overrides, legacy_numerics=False,
            ))
            for candidate_id, overrides in SCALING_CANDIDATES
        }
    else:
        # Legacy lane: independent single-threaded spawn processes; run
        # digests are content-addressed, so parallelism cannot mix identities.
        with ProcessPoolExecutor(
            max_workers=len(SCALING_CANDIDATES),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                candidate_id: pool.submit(
                    _train_scaling_candidate, manifest_argument, str(output_dir),
                    overrides, legacy_numerics=True,
                )
                for candidate_id, overrides in SCALING_CANDIDATES
            }
            training_reports = {
                candidate_id: Path(future.result()) for candidate_id, future in futures.items()
            }

    def _evaluate_candidate(candidate_id: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
        training_report_path = training_reports[candidate_id]
        training_report = _load_hashed_json(
            training_report_path, label="candidate training report"
        )
        checkpoint = training_report_path.parent / "model.pt"
        probe = ChunkedClonePolicy(checkpoint)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        candidate_suite = replace(suite, suite_id=f"{suite.suite_id}.scaling_{candidate_id}")
        evaluation_identity = content_sha256({
            "tranche_digest": tranche_digest,
            "stage": f"stage_a_{candidate_id}",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_report_content_sha256": probe.report_content_sha256,
            "preflight_report_content_sha256": preflight["content_sha256"],
            "suite_id": candidate_suite.suite_id,
        })
        evaluation = evaluate_closed_loop(
            model_path,
            candidate_suite,
            {probe.policy_id: PolicySpec(kind="chunked_clone", checkpoint=str(checkpoint.resolve()))},
            output_dir / "scaling_stage_a_evaluations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=record_video,
            workers=workers,
            provenance={
                "tranche_digest": tranche_digest,
                "stage": f"stage_a_{candidate_id}",
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
        return {
            "candidate_id": candidate_id,
            **{name: overrides[name] for name in ("hidden_width", "max_steps")},
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

    candidates = [
        _evaluate_candidate(candidate_id, overrides)
        for candidate_id, overrides in SCALING_CANDIDATES
    ]
    status = resolve_scaling_status(candidates)
    next_actions = {
        "scaling_not_resolved": "stop_and_write_new_proposal",
    }
    next_action = next_actions.get(
        status,
        "write_next_tranche_proposal_vision_rung"
        if status.endswith("_robust")
        else "stop_and_write_new_proposal_targeting_mid_trajectory_recovery",
    )
    report = {
        **identity,
        "tranche_digest": tranche_digest,
        "status": status,
        "next_action": next_action,
        "candidates": candidates,
        "margins_are_telemetry_only": True,
        "handoffs_never_gate": True,
        "offline_metrics_are_telemetry_only": True,
        "claim": "optimization_scaling_decision_not_deployment_success",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return ScalingGateResult(directory, report_path, status)


__all__ = [
    "FROZEN_RECIPE",
    "SCALING_CANDIDATES",
    "ScalingGateResult",
    "resolve_scaling_status",
    "run_scaling_gate",
]
