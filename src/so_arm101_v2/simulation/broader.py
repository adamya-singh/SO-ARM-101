"""Broader-evaluation tranche for a promoted chunked policy.

Stage A runs the full five-scenario ``fixed_pick_place_v3`` suite; Stage B
runs the eight recovery-anchor handoffs; margins are always analyzed and
never gate.  If Stage A fails with the pre-registered memorization signature
(nominal passes cleanly, at least one shifted start fails, everything
finite), the tranche automatically captures the five-scenario oracle
dataset, retrains the frozen winning recipe once, and re-runs both stages
for the retrained policy.  One retry, ever.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Callable, Mapping

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
from .oracle import capture_oracle_demonstrations, load_oracle_demonstrations
from .recovery import evaluate_policy_anchor_starts, load_oracle_recovery_examples
from .suites import load_simulation_suite

# The exact recipe promoted by the saturation gate; the contingent retrain
# may change nothing but the dataset.
# The config dict itself is frozen and test-pinned.  Numerics-regime-v2 runs
# fork the retrain digest and the tranche digest via the conditional top-level
# "numerics" identity key, never via this dict; legacy tranche artifacts stay
# addressable forever with --legacy-numerics.
FROZEN_RETRAIN_CONFIG = {
    "chunk_horizon": 90,
    "seed": 101,
    "hidden_width": 256,
    "learning_rate": 1e-3,
    "max_steps": 30_000,
    "saturation_mode": "noise_penalty_v1",
    "decoder_eta": 0.9,
    "margin_act": 0.002,
    "noise_sigma": 0.05,
    "penalty_weight": 1.0,
}

STAGE_A_PASS_RULE = (
    "deterministic, all fifteen scenario rollouts succeed without invalidation "
    "and with zero clip/limit/nonfinite/unsafe frames"
)
STAGE_B_PASS_RULE = "all eight anchor handoffs succeed with zero student safety counts"
BRANCH_RULE = (
    "retrain once, only when every nominal rollout passes with zero safety counts, "
    "at least one shifted-start rollout fails, and no nonfinite frame appears anywhere"
)


@dataclass(frozen=True)
class BroaderEvaluationResult:
    directory: Path
    report_json: Path
    status: str


def memorization_signature(rollouts: list[Mapping[str, Any]]) -> bool:
    """Pre-registered branch trigger over Stage A rollout summaries."""
    nominal = [item for item in rollouts if item["scenario_id"] == "nominal"]
    shifted = [item for item in rollouts if item["scenario_id"] != "nominal"]
    if not nominal or not shifted:
        raise ValueError("memorization signature requires nominal and shifted rollouts")
    if any(
        int(item["safety_counts"]["nonfinite_frames"]) != 0
        for item in nominal + shifted
    ):
        return False
    nominal_clean = all(
        bool(item["success"])
        and not any(int(value) for value in item["safety_counts"].values())
        for item in nominal
    )
    return nominal_clean and any(not bool(item["success"]) for item in shifted)


def resolve_broader_evaluation_status(record: Mapping[str, Any]) -> str:
    """Validate the tranche branch trace and derive its terminal status."""
    if record.get("nonfinite_frames_observed"):
        if record.get("branch_taken"):
            raise ValueError("branch ran despite nonfinite frames")
        return "starts_not_resolved"
    promoted = record["promoted"]
    if promoted["stage_a_passed"]:
        if record.get("branch_taken"):
            raise ValueError("branch ran despite a promoted Stage A pass")
        return (
            "promoted_policy_robust"
            if promoted["stage_b_passed"]
            else "promoted_policy_passes_starts_fails_handoffs"
        )
    if record.get("branch_taken"):
        if not record.get("memorization_signature"):
            raise ValueError("branch ran without the memorization signature")
        retrained = record.get("retrained")
        if retrained is None:
            raise ValueError("branch taken without a retrained record")
        if not retrained.get("capture_ok", False):
            return "starts_not_resolved"
        if retrained["stage_a_passed"]:
            return (
                "retrained_policy_robust"
                if retrained["stage_b_passed"]
                else "retrained_policy_passes_starts_fails_handoffs"
            )
        return "starts_not_resolved"
    if record.get("memorization_signature"):
        raise ValueError("memorization signature present but the branch did not run")
    return "starts_not_resolved"


def _stage_a_passed(evaluation_report: Mapping[str, Any], expected_rollouts: int) -> bool:
    rollouts = evaluation_report.get("rollouts", [])
    return bool(
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


def _require_promoted_policy(
    gate_report: Mapping[str, Any], checkpoint_sha: str
) -> Mapping[str, Any]:
    status = str(gate_report.get("status", ""))
    if not status.startswith("promoted_"):
        raise ValueError("broader evaluation requires a promoted gate status")
    candidate_id = status.removeprefix("promoted_")
    candidate = next(
        (
            item for item in gate_report.get("candidates", [])
            if item.get("candidate_id") == candidate_id
        ),
        None,
    )
    if (
        candidate is None
        or candidate.get("state") != "nominal_passed"
        or candidate.get("checkpoint_sha256") != checkpoint_sha
    ):
        raise ValueError("checkpoint is not the gate's promoted candidate")
    return candidate


def run_broader_evaluation(
    model_path: str | Path,
    checkpoint_path: str | Path,
    gate_report_path: str | Path,
    oracle_manifest_path: str | Path,
    recovery_manifest_path: str | Path,
    preflight_report_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
    workers: int | None = None,
    numerics: NumericsSpec | None | str = AUTO,
) -> BroaderEvaluationResult:
    """Run the pre-registered broader-evaluation tranche end to end."""
    from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
    from .rollout import evaluate_closed_loop

    if isinstance(numerics, str):
        if numerics != AUTO:
            raise ValueError(f"unknown numerics request {numerics!r}")
        numerics = resolve_default_numerics()

    model_path = Path(model_path).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    output_dir = Path(output_dir)
    checkpoint_sha = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    gate_report = _load_hashed_json(gate_report_path, label="promotion gate report")
    _require_promoted_policy(gate_report, checkpoint_sha)
    probe = ChunkedClonePolicy(checkpoint_path)
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    if (
        oracle_manifest.get("scenario_ids") != ["nominal"]
        or int(oracle_manifest.get("teacher_horizon", 0)) not in (450, 480)
    ):
        raise ValueError("broader evaluation requires the canonical nominal oracle capture")
    recovery_manifest, _ = load_oracle_recovery_examples(recovery_manifest_path)
    if (
        recovery_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
    ):
        raise ValueError("broader evaluation manifests disagree")
    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("broader evaluation requires the passing deterministic v3 preflight")

    suite = load_simulation_suite("fixed_pick_place_v3")
    expected_rollouts = len(suite.scenarios) * suite.repeats
    identity = {
        "schema_version": 1,
        "experiment": "broader_evaluation_tranche_v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_report_content_sha256": probe.report_content_sha256,
        "gate_report_content_sha256": gate_report["content_sha256"],
        "gate_status": gate_report["status"],
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "suite_id": suite.suite_id,
        "suite_repeats": suite.repeats,
        "expected_rollouts": expected_rollouts,
        "stage_a_pass_rule": STAGE_A_PASS_RULE,
        "stage_b_pass_rule": STAGE_B_PASS_RULE,
        "branch_rule": BRANCH_RULE,
        "frozen_retrain_config": dict(FROZEN_RETRAIN_CONFIG),
        "margin_analyzer": ANALYZER_VERSION,
        "retry_budget": 1,
    }
    if numerics is not None:
        identity["numerics"] = numerics_identity(numerics)
    tranche_digest = content_sha256(identity)
    directory = output_dir / "broader_evaluations" / tranche_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="broader evaluation report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable broader evaluation differs: {directory}")
        return BroaderEvaluationResult(directory, report_path, str(existing["status"]))

    def _evaluate_policy(label: str, policy_checkpoint: Path) -> dict[str, Any]:
        policy_probe = ChunkedClonePolicy(policy_checkpoint)
        policy_sha = hashlib.sha256(policy_checkpoint.read_bytes()).hexdigest()
        broad_suite = replace(suite, suite_id=f"{suite.suite_id}.broader_{label}")
        evaluation_identity = content_sha256({
            "tranche_digest": tranche_digest,
            "stage": f"stage_a_{label}",
            "checkpoint_sha256": policy_sha,
            "checkpoint_report_content_sha256": policy_probe.report_content_sha256,
            "preflight_report_content_sha256": preflight["content_sha256"],
            "suite_id": broad_suite.suite_id,
        })
        evaluation = evaluate_closed_loop(
            model_path,
            broad_suite,
            {policy_probe.policy_id: PolicySpec(kind="chunked_clone", checkpoint=str(policy_checkpoint.resolve()))},
            output_dir / "broader_stage_a_evaluations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=record_video,
            workers=workers,
            provenance={
                "tranche_digest": tranche_digest,
                "stage": f"stage_a_{label}",
                "checkpoint_sha256": policy_sha,
                "offline_report_content_sha256": policy_probe.report_content_sha256,
                "preflight_report_content_sha256": preflight["content_sha256"],
            },
        )
        evaluation_report = _load_hashed_json(
            evaluation.report_json, label="stage A evaluation report"
        )
        margins_path = analyze_pick_place_margins(
            evaluation.report_json, output_dir, mujoco_model_path=model_path,
        )
        margins_report = _load_hashed_json(margins_path, label="margin analysis report")
        anchor_path = evaluate_policy_anchor_starts(
            model_path, oracle_manifest_path, recovery_manifest_path,
            policy_checkpoint, output_dir,
        )
        anchor_report = _load_hashed_json(anchor_path, label="anchor handoff report")
        summaries = [_rollout_summary(item) for item in evaluation_report["rollouts"]]
        return {
            "label": label,
            "checkpoint_sha256": policy_sha,
            "checkpoint_report_content_sha256": policy_probe.report_content_sha256,
            "policy_id": policy_probe.policy_id,
            "stage_a_evaluation": str(evaluation.report_json.resolve()),
            "stage_a_evaluation_content_sha256": evaluation_report["content_sha256"],
            "stage_a_passed": _stage_a_passed(evaluation_report, expected_rollouts),
            "stage_a_rollouts": summaries,
            "margin_report": str(margins_path.resolve()),
            "margin_report_content_sha256": margins_report["content_sha256"],
            "stage_b_report": str(anchor_path.resolve()),
            "stage_b_report_content_sha256": anchor_report["content_sha256"],
            "stage_b_passed": bool(anchor_report["passed"]),
            "stage_b_results": anchor_report["results"],
        }

    promoted_record = _evaluate_policy("promoted", checkpoint_path)
    all_summaries = list(promoted_record["stage_a_rollouts"])
    signature = (
        not promoted_record["stage_a_passed"]
        and memorization_signature(promoted_record["stage_a_rollouts"])
    )

    retrained_record: dict[str, Any] | None = None
    branch_taken = False
    if signature:
        branch_taken = True
        try:
            capture = capture_oracle_demonstrations(
                model_path, "fixed_pick_place_v3", preflight_report_path, output_dir,
                scenario="all", record_video=record_video,
            )
        except (RuntimeError, ValueError) as exc:
            capture = None
            retrained_record = {"capture_ok": False, "capture_error": str(exc)}
        if capture is not None:
            capture_manifest, _ = load_oracle_demonstrations(capture.manifest)
            capture_horizon = int(capture_manifest.get("teacher_horizon", 0))
            capture_ok = (
                capture_horizon in (450, 480)
                and len(capture_manifest.get("episodes", [])) == 5
                and capture.rows == 5 * capture_horizon
            )
            retrained_record = {
                "capture_manifest": str(capture.manifest.resolve()),
                "capture_manifest_content_sha256": capture_manifest["content_sha256"],
                "capture_rows": int(capture.rows),
                "capture_ok": capture_ok,
            }
        if capture is not None and retrained_record["capture_ok"]:
            training = train_chunked_clone(
                capture.manifest, output_dir,
                config=ChunkedCloneConfig(**FROZEN_RETRAIN_CONFIG),
                numerics=numerics,
            )
            retrained_record["training_report"] = str(training.report_json.resolve())
            retrained_record["training_report_content_sha256"] = _load_hashed_json(
                training.report_json, label="retrained training report"
            )["content_sha256"]
            retrained_record.update(_evaluate_policy("retrained", training.checkpoint))
            all_summaries.extend(retrained_record["stage_a_rollouts"])

    nonfinite_observed = any(
        int(item["safety_counts"]["nonfinite_frames"]) != 0 for item in all_summaries
    )
    record = {
        "promoted": promoted_record,
        "memorization_signature": signature,
        "branch_taken": branch_taken,
        "retrained": retrained_record,
        "nonfinite_frames_observed": nonfinite_observed,
    }
    status = resolve_broader_evaluation_status(record)
    next_actions = {
        "promoted_policy_robust": "write_next_tranche_proposal_vision_rung",
        "retrained_policy_robust": "write_next_tranche_proposal_vision_rung",
        "promoted_policy_passes_starts_fails_handoffs": (
            "stop_and_write_new_proposal_targeting_mid_trajectory_recovery"
        ),
        "retrained_policy_passes_starts_fails_handoffs": (
            "stop_and_write_new_proposal_targeting_mid_trajectory_recovery"
        ),
        "starts_not_resolved": "stop_and_write_new_proposal",
    }
    report = {
        **identity,
        "tranche_digest": tranche_digest,
        "status": status,
        "next_action": next_actions[status],
        **record,
        "margins_are_telemetry_only": True,
        "handoffs_never_trigger_retraining": True,
        "claim": "broader_robustness_decision_not_deployment_success",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return BroaderEvaluationResult(directory, report_path, status)


__all__ = [
    "BroaderEvaluationResult",
    "FROZEN_RETRAIN_CONFIG",
    "memorization_signature",
    "resolve_broader_evaluation_status",
    "run_broader_evaluation",
]
