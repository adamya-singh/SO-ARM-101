"""Precision tranche: staged pre-registered follow-up to the scaling gate.

Three stages executed as separate gate invocations, chained by report hash
(`notes/precision-tranche-proposal.md`): "schedule" tests a cosine LR decay
against the scaling gate's immutable width512_steps90k control; "budget"
scales steps/width under the selected schedule; "seeds" replicates the best
cell. Sibling of ``scaling.py`` (which is the immutable record of a completed
experiment and is never edited); Stage A/B pass rules are imported from it so
the gates stay byte-identical across tranches.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
import hashlib
import multiprocessing
from pathlib import Path
from typing import Any, Mapping, Sequence

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
from .recovery import evaluate_policy_anchor_starts, load_oracle_recovery_examples
from .scaling import FROZEN_RECIPE, STAGE_A_PASS_RULE, STAGE_B_PASS_RULE
from .suites import load_simulation_suite

PRECISION_EXPERIMENT = "precision_tranche_v1"
PRECISION_STAGES = ("schedule", "budget", "seeds")

# Immutable control: the scaling gate's width512_steps90k arm (referenced,
# never re-run).
CONTROL_CELL_ID = "control_fixed_w512_90k"
CONTROL_CANDIDATE_ID = "width512_steps90k"

SCHEDULE_CANDIDATES = (
    ("cosine_w512_90k",
     {"hidden_width": 512, "max_steps": 90_000, "lr_schedule": "cosine_floor_v1"}),
)
BUDGET_CANDIDATES = (
    ("steps150k", {"hidden_width": 512, "max_steps": 150_000}),
    ("steps300k", {"hidden_width": 512, "max_steps": 300_000}),
    ("width1024", {"hidden_width": 1024, "max_steps": 90_000}),
)
SEED_CANDIDATES = (202, 303)

# Registered tie-break order = increasing change from the control.
PRECISION_CELL_ORDER = (
    CONTROL_CELL_ID, "cosine_w512_90k", "steps150k", "steps300k", "width1024",
)

SCHEDULE_SELECTION_RULE = (
    "the budget stage inherits cosine_floor_v1 iff the schedule cell strictly "
    "improves on the control by (more Stage-A clean successes, else equal "
    "successes with fewer total safety frames); any tie keeps fixed"
)
BEST_CELL_RULE = (
    "most Stage-A clean successes, then fewer total safety frames "
    "(clip+limit+nonfinite+unsafe summed over 15 rollouts), then earliest in "
    "the registered cell order (smallest change from the control)"
)


@dataclass(frozen=True)
class PrecisionStageResult:
    directory: Path
    report_json: Path
    status: str


def cell_metrics(cell: Mapping[str, Any]) -> tuple[int, int]:
    """(stage_a_successes, total_safety_frames) from a cell record.

    Reads the ``_rollout_summary`` shape: ``success``/``invalidated`` at the
    top level, the four frame counters nested under ``safety_counts``.
    """
    rollouts = cell["stage_a_rollouts"]
    successes = sum(
        1 for item in rollouts
        if item.get("success") is True and item.get("invalidated") is False
    )
    safety_frames = sum(
        int(item["safety_counts"][name])
        for item in rollouts
        for name in (
            "clipping_frames", "limiting_frames",
            "nonfinite_frames", "unsafe_contact_frames",
        )
    )
    return successes, safety_frames


def resolve_schedule_selection(
    control: Mapping[str, Any], candidate: Mapping[str, Any]
) -> str:
    """'cosine_floor_v1' or 'fixed' per SCHEDULE_SELECTION_RULE."""
    control_successes, control_frames = cell_metrics(control)
    successes, frames = cell_metrics(candidate)
    if successes > control_successes:
        return "cosine_floor_v1"
    if successes == control_successes and frames < control_frames:
        return "cosine_floor_v1"
    return "fixed"


def select_best_cell(cells: Sequence[Mapping[str, Any]]) -> str:
    """cell_id of the best cell per BEST_CELL_RULE."""
    for cell in cells:
        if cell["cell_id"] not in PRECISION_CELL_ORDER:
            raise ValueError(f"unregistered cell {cell['cell_id']!r}")
    def key(cell: Mapping[str, Any]) -> tuple[int, int, int]:
        successes, frames = cell_metrics(cell)
        return (-successes, frames, PRECISION_CELL_ORDER.index(cell["cell_id"]))
    return sorted(cells, key=key)[0]["cell_id"]


def resolve_precision_stage_status(
    stage: str,
    cells: list[Mapping[str, Any]],
    *,
    control: Mapping[str, Any] | None = None,
) -> str:
    if stage not in PRECISION_STAGES:
        raise ValueError(f"unknown precision stage {stage!r}")
    for cell in cells:
        if "stage_a_passed" not in cell or cell.get("stage_a_evaluation_content_sha256") is None:
            raise ValueError("precision cell lacks evaluation evidence")
    if stage == "schedule":
        if control is None or len(cells) != 1:
            raise ValueError("schedule stage requires the control and exactly one cell")
        selection = resolve_schedule_selection(control, cells[0])
        if selection == "fixed":
            return "schedule_selected_fixed"
        if cells[0]["stage_a_passed"]:
            return "schedule_selected_cosine_floor_v1_stage_a_passed"
        return "schedule_selected_cosine_floor_v1"
    if stage == "budget":
        expected = [cell_id for cell_id, _ in BUDGET_CANDIDATES]
        if [cell["cell_id"] for cell in cells] != expected:
            raise ValueError("budget cells do not match the registered order")
        for cell in cells:
            if cell["stage_a_passed"]:
                return f"budget_promoted_{cell['cell_id']}"
        return "budget_not_resolved"
    expected_seeds = [f"seed{seed}" for seed in SEED_CANDIDATES]
    if [cell["cell_id"] for cell in cells] != expected_seeds:
        raise ValueError("seed cells do not match the registered order")
    if control is None:
        raise ValueError("seeds stage requires the best-cell control record")
    best_passed = bool(control.get("stage_a_passed"))
    if best_passed and all(cell["stage_a_passed"] for cell in cells):
        return "seeds_robust"
    return "seeds_not_resolved"


def _find_stage_report(output_dir: Path, content_sha: str) -> dict[str, Any]:
    """Locate a precision stage report by its content hash (hash-verified)."""
    for candidate in sorted((output_dir / "precision_gates").glob("*/report.json")):
        report = _load_hashed_json(candidate, label="precision stage report")
        if report["content_sha256"] == content_sha:
            return report
    raise FileNotFoundError(
        f"no precision stage report with content_sha256 {content_sha[:16]}… under {output_dir}"
    )


def _train_precision_cell(
    manifest_path: str, output_dir: str, overrides: dict[str, Any],
    *, legacy_numerics: bool = False,
) -> str:
    """Train one cell; safe in-process (GPU) or as a spawn child (legacy)."""
    from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
    from so_arm101_v2.learning.numerics import AUTO as NUMERICS_AUTO

    result = train_chunked_clone(
        manifest_path, output_dir,
        config=ChunkedCloneConfig(**{**FROZEN_RECIPE, **overrides}),
        numerics=None if legacy_numerics else NUMERICS_AUTO,
    )
    return str(result.report_json)


def run_precision_stage(
    stage: str,
    model_path: str | Path,
    capture_manifest_path: str | Path,
    oracle_manifest_path: str | Path,
    recovery_manifest_path: str | Path,
    preflight_report_path: str | Path,
    scaling_gate_report_path: str | Path,
    output_dir: str | Path,
    *,
    prior_stage_report_path: str | Path | None = None,
    record_video: bool = True,
    workers: int | None = None,
    numerics: NumericsSpec | None | str = AUTO,
) -> PrecisionStageResult:
    """Run one pre-registered precision stage end to end."""
    from .rollout import evaluate_closed_loop

    if isinstance(numerics, str):
        if numerics != AUTO:
            raise ValueError(f"unknown numerics request {numerics!r}")
        numerics = resolve_default_numerics()
    if stage not in PRECISION_STAGES:
        raise ValueError(f"unknown precision stage {stage!r}")

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    capture_manifest, _ = load_oracle_demonstrations(capture_manifest_path)
    episodes = capture_manifest.get("episodes", [])
    if (
        len(episodes) != 5
        or sum(int(item["rows"]) for item in episodes) != 2250
        or int(capture_manifest.get("teacher_horizon", 0)) != 450
    ):
        raise ValueError("precision stage requires the five-scenario 2,250-row capture")
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    if oracle_manifest.get("scenario_ids") != ["nominal"]:
        raise ValueError("precision stage requires the canonical nominal oracle capture")
    recovery_manifest, _ = load_oracle_recovery_examples(recovery_manifest_path)
    if (
        recovery_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
    ):
        raise ValueError("precision stage manifests disagree")
    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("precision stage requires the passing deterministic v3 preflight")
    scaling_report = _load_hashed_json(
        scaling_gate_report_path, label="scaling gate report"
    )
    if (
        scaling_report.get("experiment") != "optimization_scaling_tranche_v1"
        or scaling_report.get("status") != "scaling_not_resolved"
    ):
        raise ValueError("precision stage requires the terminal scaling gate report")
    control_candidate = next(
        (
            item for item in scaling_report.get("candidates", [])
            if item.get("candidate_id") == CONTROL_CANDIDATE_ID
        ),
        None,
    )
    if control_candidate is None:
        raise ValueError("scaling gate report lacks the width512_steps90k control cell")
    control_successes, control_frames = cell_metrics(control_candidate)
    control_record = {
        "cell_id": CONTROL_CELL_ID,
        "hidden_width": 512,
        "max_steps": 90_000,
        "lr_schedule": "fixed",
        "checkpoint_sha256": control_candidate["checkpoint_sha256"],
        "stage_a_passed": bool(control_candidate["stage_a_passed"]),
        "stage_a_rollouts": control_candidate["stage_a_rollouts"],
        "stage_a_successes": control_successes,
        "total_safety_frames": control_frames,
    }

    prior_report: dict[str, Any] | None = None
    if stage in ("budget", "seeds"):
        if prior_stage_report_path is None:
            raise ValueError(f"{stage} stage requires --prior-stage-report")
        prior_report = _load_hashed_json(
            prior_stage_report_path, label="prior precision stage report"
        )
        expected_prior = "schedule" if stage == "budget" else "budget"
        if (
            prior_report.get("experiment") != PRECISION_EXPERIMENT
            or prior_report.get("stage") != expected_prior
        ):
            raise ValueError(
                f"{stage} stage requires a terminal {expected_prior} stage report"
            )
    elif prior_stage_report_path is not None:
        raise ValueError("the schedule stage takes no prior stage report")

    selected_schedule = "fixed"
    best_cell: dict[str, Any] | None = None
    if stage == "schedule":
        candidates_registered: list[dict[str, Any]] = [
            {"cell_id": cell_id, **overrides} for cell_id, overrides in SCHEDULE_CANDIDATES
        ]
    elif stage == "budget":
        selected_schedule = str(prior_report["selected_lr_schedule"])
        candidates_registered = [
            {
                "cell_id": cell_id,
                **overrides,
                **({"lr_schedule": selected_schedule} if selected_schedule != "fixed" else {}),
            }
            for cell_id, overrides in BUDGET_CANDIDATES
        ]
    else:
        # Recompute the best cell over the FULL registered cell set (control +
        # schedule cells + budget cells) per BEST_CELL_RULE.  The budget
        # report also records a best_cell field, but the rule string in every
        # stage identity is authoritative; recomputing here makes the seeds
        # stage robust to any defect in an earlier stage's derived field.
        schedule_report = _find_stage_report(
            output_dir, prior_report["prior_stage_report_content_sha256"]
        )
        selection_pool = [
            control_record,
            *schedule_report["cells"],
            *prior_report["cells"],
        ]
        best_id = select_best_cell(selection_pool)
        chosen = next(cell for cell in selection_pool if cell["cell_id"] == best_id)
        best_cell = {
            "cell_id": best_id,
            "overrides": (
                chosen.get("overrides")
                or {"hidden_width": 512, "max_steps": 90_000}
            ),
            "checkpoint_sha256": chosen["checkpoint_sha256"],
            "stage_a_passed": bool(chosen.get("stage_a_passed")),
            "stage_a_rollouts": chosen["stage_a_rollouts"],
        }
        candidates_registered = [
            {
                "cell_id": f"seed{seed}",
                **best_cell["overrides"],
                "seed": seed,
            }
            for seed in SEED_CANDIDATES
        ]

    suite = load_simulation_suite("fixed_pick_place_v3")
    expected_rollouts = len(suite.scenarios) * suite.repeats
    if expected_rollouts != 15:
        raise ValueError("precision stage expects the fifteen-rollout v3 suite")

    identity: dict[str, Any] = {
        "schema_version": 1,
        "experiment": PRECISION_EXPERIMENT,
        "stage": stage,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "capture_manifest_content_sha256": capture_manifest["content_sha256"],
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "scaling_gate_report_content_sha256": scaling_report["content_sha256"],
        "control_cell": {
            "cell_id": CONTROL_CELL_ID,
            "hidden_width": 512,
            "max_steps": 90_000,
            "lr_schedule": "fixed",
            "checkpoint_sha256": control_record["checkpoint_sha256"],
            "stage_a_successes": control_successes,
            "total_safety_frames": control_frames,
        },
        "frozen_recipe": dict(FROZEN_RECIPE),
        "candidates_registered": candidates_registered,
        "cell_order": list(PRECISION_CELL_ORDER),
        "schedule_selection_rule": SCHEDULE_SELECTION_RULE,
        "best_cell_rule": BEST_CELL_RULE,
        "suite_id": suite.suite_id,
        "suite_repeats": suite.repeats,
        "expected_rollouts": expected_rollouts,
        "stage_a_pass_rule": STAGE_A_PASS_RULE,
        "stage_b_pass_rule": STAGE_B_PASS_RULE,
        "margin_analyzer": ANALYZER_VERSION,
        "retry_budget": 0,
    }
    if prior_report is not None:
        identity["prior_stage_report_content_sha256"] = prior_report["content_sha256"]
    if stage == "budget":
        identity["selected_lr_schedule"] = selected_schedule
    if stage == "seeds":
        identity["best_cell"] = {
            "cell_id": best_cell["cell_id"],
            "overrides": best_cell["overrides"],
            "checkpoint_sha256": best_cell["checkpoint_sha256"],
        }
    if numerics is not None:
        identity["numerics"] = numerics_identity(numerics)
    stage_digest = content_sha256(identity)
    directory = output_dir / "precision_gates" / stage_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="precision stage report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable precision stage differs: {directory}")
        return PrecisionStageResult(directory, report_path, str(existing["status"]))

    manifest_argument = str(Path(capture_manifest_path).resolve())
    cell_overrides = {
        record["cell_id"]: {k: v for k, v in record.items() if k != "cell_id"}
        for record in candidates_registered
    }
    if numerics is not None:
        # One GPU: sequential in-process trainings share one CUDA context and
        # a warm inductor cache; each cell is minutes.
        training_reports = {
            cell_id: Path(_train_precision_cell(
                manifest_argument, str(output_dir), overrides, legacy_numerics=False,
            ))
            for cell_id, overrides in cell_overrides.items()
        }
    else:
        with ProcessPoolExecutor(
            max_workers=len(cell_overrides),
            mp_context=multiprocessing.get_context("spawn"),
        ) as pool:
            futures = {
                cell_id: pool.submit(
                    _train_precision_cell, manifest_argument, str(output_dir),
                    overrides, legacy_numerics=True,
                )
                for cell_id, overrides in cell_overrides.items()
            }
            training_reports = {
                cell_id: Path(future.result()) for cell_id, future in futures.items()
            }

    def _evaluate_cell(cell_id: str, overrides: Mapping[str, Any]) -> dict[str, Any]:
        training_report_path = training_reports[cell_id]
        training_report = _load_hashed_json(
            training_report_path, label="cell training report"
        )
        checkpoint = training_report_path.parent / "model.pt"
        probe = ChunkedClonePolicy(checkpoint)
        checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        cell_suite = replace(
            suite, suite_id=f"{suite.suite_id}.precision_{stage}_{cell_id}"
        )
        evaluation_identity = content_sha256({
            "stage_digest": stage_digest,
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
            output_dir / "precision_stage_a_evaluations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=record_video,
            workers=workers,
            provenance={
                "stage_digest": stage_digest,
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
            "overrides": dict(overrides),
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

    cells = [
        _evaluate_cell(record["cell_id"], cell_overrides[record["cell_id"]])
        for record in candidates_registered
    ]
    control_for_resolution = control_record if stage == "schedule" else (
        best_cell if stage == "seeds" else None
    )
    status = resolve_precision_stage_status(stage, cells, control=control_for_resolution)

    report: dict[str, Any] = {
        **identity,
        "stage_digest": stage_digest,
        "status": status,
        "cells": cells,
        "margins_are_telemetry_only": True,
        "handoffs_never_gate": True,
        "offline_metrics_are_telemetry_only": True,
        "claim": "precision_tranche_decision_not_deployment_success",
    }
    if stage == "schedule":
        report["selected_lr_schedule"] = (
            "cosine_floor_v1" if status.startswith("schedule_selected_cosine") else "fixed"
        )
        report["next_action"] = "run_budget_stage_under_selected_schedule"
    elif stage == "budget":
        # Selection spans the FULL registered order: control + the schedule
        # stage's cells + this stage's cells (BEST_CELL_RULE).
        selection_pool = [control_record, *prior_report["cells"], *cells]
        chosen = (
            select_best_cell(selection_pool)
            if status == "budget_not_resolved"
            else status.removeprefix("budget_promoted_")
        )
        chosen_record = next(
            (cell for cell in selection_pool if cell["cell_id"] == chosen),
            control_record,
        )
        report["best_cell"] = {
            "cell_id": chosen,
            "overrides": (
                chosen_record.get("overrides")
                or {"hidden_width": 512, "max_steps": 90_000}
            ),
            "checkpoint_sha256": chosen_record["checkpoint_sha256"],
            "stage_a_passed": bool(chosen_record.get("stage_a_passed")),
            "stage_a_rollouts": chosen_record["stage_a_rollouts"],
        }
        report["next_action"] = "run_seeds_stage_on_best_cell"
    else:
        report["next_action"] = (
            "write_next_tranche_proposal_vision_rung" if status == "seeds_robust"
            else "stop_and_write_teacher_horizon_alignment_proposal"
        )
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return PrecisionStageResult(directory, report_path, status)


__all__ = [
    "BEST_CELL_RULE",
    "BUDGET_CANDIDATES",
    "PRECISION_CELL_ORDER",
    "PRECISION_STAGES",
    "PrecisionStageResult",
    "SCHEDULE_CANDIDATES",
    "SCHEDULE_SELECTION_RULE",
    "SEED_CANDIDATES",
    "cell_metrics",
    "resolve_precision_stage_status",
    "resolve_schedule_selection",
    "run_precision_stage",
    "select_best_cell",
]
