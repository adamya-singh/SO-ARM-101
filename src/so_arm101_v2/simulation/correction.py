"""DAgger-style oracle correction trajectories from policy-induced states."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from so_arm101_v2.contracts import (
    PickPlaceEvaluationState,
    evaluate_pick_place_step,
    load_pick_place_contract,
)
from so_arm101_v2.data._serialization import (
    content_sha256,
    write_immutable_bytes,
    write_immutable_json,
)

from .adapter import MujocoTaskAdapter
from .observability import STATE_FIELDS, _load_hashed_json
from .oracle import load_oracle_demonstrations
from .policy_specs import PolicySpec
from .privileged import PrivilegedStagedController
from .suites import load_simulation_suite


@dataclass(frozen=True)
class CorrectionSite:
    name: str
    action_index: int
    rationale: str

    def __post_init__(self) -> None:
        if not self.name or not 1 <= self.action_index < 450:
            raise ValueError("invalid correction site")


# The eight established recovery-anchor phase indices plus the three
# documented divergence onsets of the nominal-only clone (first drift above
# 0.005 ACT, above 0.01 ACT, and the shared-first-contact cube split).
DAGGER_CORRECTION_SITES = (
    CorrectionSite("drift_005", 31, "documented first clone drift above 0.005 ACT"),
    CorrectionSite("approach", 70, "end of high approach"),
    CorrectionSite("drift_010", 74, "documented first clone drift above 0.01 ACT"),
    CorrectionSite("contact_split", 194, "documented cube trajectory split at shared first contact"),
    CorrectionSite("first_contact", 195, "first post-contact state"),
    CorrectionSite("seating", 205, "start of final centering"),
    CorrectionSite("closure", 255, "start of jaw closure"),
    CorrectionSite("lift", 315, "start of lift motion"),
    CorrectionSite("transport", 365, "start of carry to napkin"),
    CorrectionSite("placement", 386, "start of set-down"),
    CorrectionSite("release", 419, "end of controlled release"),
)

_ROW_FIELDS = (
    "action_index", "progress", "current_act", "robot_qvel", "cube_position",
    "cube_quaternion_wxyz", "cube_linear_velocity", "cube_angular_velocity",
    "executed_act", "executed_delta_act",
)
_ARRAY_FIELDS = _ROW_FIELDS + ("site_index", "scenario_index")


@dataclass(frozen=True)
class OracleCorrectionCollection:
    directory: Path
    manifest: Path
    arrays: Path
    collection_digest: str
    rows: int
    accepted_sites: tuple[str, ...]
    rejected_sites: tuple[str, ...]


@dataclass(frozen=True)
class CorrectionGateResult:
    directory: Path
    report_json: Path
    status: str


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    write_immutable_bytes(
        path, data, conflict_message=f"immutable correction artifact differs: {path}"
    )


def _nominal_scenario_and_contract() -> tuple[Any, Any, Any]:
    suite = load_simulation_suite("fixed_pick_place_v3")
    scenario = next(item for item in suite.scenarios if item.scenario_id == "nominal")
    contract = load_pick_place_contract(suite.task_contract)
    return suite, scenario, contract


def _run_clone_nominal_rollout(
    model_path: Path, checkpoint: Path
) -> tuple[np.ndarray, dict[str, int]]:
    """Drive the inducing clone for 450 nominal actions and record its commands."""
    from .clone_policy import OracleCloneCheckpointPolicy

    _, scenario, contract = _nominal_scenario_and_contract()
    adapter = MujocoTaskAdapter(model_path)
    policy = OracleCloneCheckpointPolicy(checkpoint)
    counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
    executed: list[np.ndarray] = []
    try:
        adapter.reset(scenario)
        policy.reset(adapter)
        for _ in range(450):
            requested = policy.predict(
                np.empty((0,), dtype=np.uint8), adapter.current_act(), adapter,
            )
            command = adapter.apply_policy_command(requested)
            executed.append(command.executed_act.copy())
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(
                command,
                footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
            )
            counts["clip"] += int(measurement.pickup.command_bound_violation)
            counts["limit"] += int(measurement.pickup.delta_limiter_activated)
            counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
            counts["unsafe"] += int(measurement.pickup.unsafe_contact)
    finally:
        adapter.close()
    return np.stack(executed), counts


def _capture_site(
    model_path: Path,
    source: Mapping[str, np.ndarray],
    executed: np.ndarray,
    site: CorrectionSite,
) -> tuple[list[dict[str, np.ndarray]] | None, dict[str, Any]]:
    """Replay the clone prefix, then demonstrate a nominal-speed sub-episode."""
    _, scenario, contract = _nominal_scenario_and_contract()
    adapter = MujocoTaskAdapter(model_path)
    evaluation = None
    prefix_counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
    correction_counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
    record: dict[str, Any] = {**asdict(site)}
    rows: list[dict[str, np.ndarray]] = []
    try:
        adapter.reset(scenario)
        for action_index in range(site.action_index):
            requested = np.asarray(executed[action_index], dtype=np.float32)
            command = adapter.apply_policy_command(requested)
            if not np.array_equal(command.executed_act, requested):
                raise RuntimeError(
                    f"clone prefix replay diverged at {site.name} action {action_index}"
                )
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(
                command,
                footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
            )
            prefix_counts["clip"] += int(measurement.pickup.command_bound_violation)
            prefix_counts["limit"] += int(measurement.pickup.delta_limiter_activated)
            prefix_counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
            prefix_counts["unsafe"] += int(measurement.pickup.unsafe_contact)
        record["prefix_safety_counts"] = dict(prefix_counts)
        if any(prefix_counts.values()):
            record.update({"accepted": False, "rejection_reason": "unsafe_prefix"})
            return None, record

        induced = adapter.privileged_state()
        induced_qpos = np.asarray(adapter.data.qpos, dtype=np.float64).copy()
        induced_qvel = np.asarray(adapter.data.qvel, dtype=np.float64).copy()
        record["induced_observable_delta_from_nominal"] = {
            "current_act": (
                induced.current_act
                - np.asarray(source["current_act"][site.action_index], dtype=np.float32)
            ).tolist(),
            "cube_position_m": (
                induced.cube_position
                - np.asarray(source["cube_position"][site.action_index], dtype=np.float32)
            ).tolist(),
        }
        controller = PrivilegedStagedController()
        try:
            # Nominal-speed full replan: the correction is validated as its own
            # fresh sub-episode under the unchanged v3 contract, so it keeps
            # the proven stage schedule instead of compressing it into the
            # remaining global budget (which the task's grasp physics and
            # fixed pickup deadline make infeasible for k >~ 130).
            controller.reset(adapter)
        except RuntimeError as exc:
            record.update({
                "accepted": False,
                "rejection_reason": "planning_failed",
                "planning_error": str(exc),
            })
            return None, record
        adapter.rebase_observation_history()
        # State purity is checked on the raw integrator state: the planner must
        # leave qpos untouched up to float32 arm rounding and qvel bitwise.
        # Derived quantities (body xpos) legitimately refresh under the
        # planner's mj_forward because mj_step leaves them one substep stale;
        # that refresh is recorded as telemetry, not gated.
        after_qpos = np.asarray(adapter.data.qpos, dtype=np.float64)
        after_qvel = np.asarray(adapter.data.qvel, dtype=np.float64)
        arm_addresses = np.asarray(adapter._joint_qpos, dtype=np.int64)
        non_arm_mask = np.ones(after_qpos.shape[0], dtype=bool)
        non_arm_mask[arm_addresses] = False
        mutation = {
            "arm_qpos": float(np.max(np.abs(
                after_qpos[arm_addresses] - induced_qpos[arm_addresses]
            ))),
            "non_arm_qpos": float(np.max(np.abs(
                (after_qpos - induced_qpos)[non_arm_mask]
            ))) if bool(non_arm_mask.any()) else 0.0,
            "qvel": float(np.max(np.abs(after_qvel - induced_qvel))),
        }
        row_zero = adapter.privileged_state()
        record["reset_state_mutation_linf"] = mutation
        record["derived_state_refresh_linf"] = {
            name: float(np.max(np.abs(
                np.asarray(getattr(row_zero, name), dtype=np.float32)
                - np.asarray(getattr(induced, name), dtype=np.float32)
            )))
            for name in STATE_FIELDS
        }
        record["waypoint_boundaries"] = list(controller.boundaries)
        record["solve_diagnostics"] = controller.solve_diagnostics
        if (
            mutation["arm_qpos"] > 1e-6
            or mutation["non_arm_qpos"] != 0.0
            or mutation["qvel"] != 0.0
        ):
            record.update({"accepted": False, "rejection_reason": "reset_state_mutation"})
            return None, record

        # The correction is evaluated as a fresh sub-episode: its own contract
        # clock starts here, so pickup and placement get the same deadlines the
        # nominal demonstration had.  Rows keep the deployment clock: the
        # deployed policy computes progress = min(index, 449) / 449, so rows
        # past the global horizon saturate at 1.0 exactly as it would.
        state = PickPlaceEvaluationState()
        for sub_index in range(contract.max_actions):
            global_index = site.action_index + sub_index
            snapshot = adapter.privileged_state()
            requested = controller.predict(
                np.empty((0,), dtype=np.uint8), snapshot.current_act, adapter,
            )
            command = adapter.apply_policy_command(requested)
            # Same rule as the immutable oracle capture (oracle.py): the safety
            # path's act->physical->act round trip carries float32 rounding, so
            # unchanged means within 1e-6 ACT; real interventions are counted
            # by the clip/limit/nonfinite flags below.
            if float(np.max(np.abs(command.requested_act - command.executed_act))) > 1e-6:
                record.update({
                    "accepted": False,
                    "rejection_reason": "safety_layer_altered_label",
                    "failed_sub_action_index": sub_index,
                    "correction_safety_counts": dict(correction_counts),
                })
                return None, record
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(
                command,
                footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
            )
            correction_counts["clip"] += int(measurement.pickup.command_bound_violation)
            correction_counts["limit"] += int(measurement.pickup.delta_limiter_activated)
            correction_counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
            correction_counts["unsafe"] += int(measurement.pickup.unsafe_contact)
            rows.append({
                "action_index": np.asarray(global_index, dtype=np.int32),
                "progress": np.asarray(min(global_index, 449) / 449.0, dtype=np.float32),
                "current_act": snapshot.current_act,
                "robot_qvel": snapshot.robot_qvel,
                "cube_position": snapshot.cube_position,
                "cube_quaternion_wxyz": snapshot.cube_quaternion_wxyz,
                "cube_linear_velocity": snapshot.cube_linear_velocity,
                "cube_angular_velocity": snapshot.cube_angular_velocity,
                "executed_act": command.executed_act,
                "executed_delta_act": command.executed_act - snapshot.current_act,
            })
            if not state.completed:
                state, evaluation = evaluate_pick_place_step(contract, measurement, state)
            if state.completed:
                break
    finally:
        adapter.close()
    record["correction_safety_counts"] = dict(correction_counts)
    if evaluation is None:
        raise RuntimeError(f"correction at {site.name} was not evaluated")
    if not evaluation.success or any(correction_counts.values()):
        record.update({
            "accepted": False,
            "rejection_reason": (
                "unsafe_correction" if any(correction_counts.values())
                else "correction_failed_task"
            ),
            "final_evaluation": asdict(evaluation),
        })
        return None, record
    record.update({
        "accepted": True,
        "rejection_reason": None,
        "rows": len(rows),
        "final_evaluation": asdict(evaluation),
    })
    return rows, record


def capture_dagger_corrections(
    model_path: str | Path,
    oracle_manifest: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> OracleCorrectionCollection:
    """Capture nominal-speed sub-episode oracle corrections from clone-induced states."""
    from .clone_policy import OracleCloneCheckpointPolicy

    source_manifest, source = load_oracle_demonstrations(oracle_manifest)
    if (
        source_manifest.get("scenario_ids") != ["nominal"]
        or int(source_manifest.get("teacher_horizon", 0)) != 450
    ):
        raise ValueError("dagger corrections require the canonical nominal 450-row oracle capture")
    model_path = Path(model_path).resolve()
    checkpoint = Path(checkpoint).resolve()
    probe = OracleCloneCheckpointPolicy(checkpoint)
    inducing_report = _load_hashed_json(
        checkpoint.with_name("report.json"), label="inducing clone report"
    )
    if (
        probe.kind.value != "phase_state"
        or probe.input_dim != 10
        or probe.hidden_width != 256
        or probe.manifest_content_sha256 != source_manifest["content_sha256"]
        or inducing_report.get("recovery_augmentation") is not None
        or inducing_report.get("correction_augmentation") is not None
    ):
        raise ValueError(
            "dagger corrections require the nominal-only width-256 phase_state clone"
        )

    first_executed, first_counts = _run_clone_nominal_rollout(model_path, checkpoint)
    second_executed, second_counts = _run_clone_nominal_rollout(model_path, checkpoint)
    if not np.array_equal(first_executed, second_executed) or first_counts != second_counts:
        raise RuntimeError("inducing clone rollout is not deterministic")

    all_rows: list[dict[str, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for site_position, site in enumerate(DAGGER_CORRECTION_SITES):
        first_rows, first_record = _capture_site(model_path, source, first_executed, site)
        second_rows, second_record = _capture_site(model_path, source, first_executed, site)
        if (first_rows is None) != (second_rows is None) or first_record != second_record:
            raise RuntimeError(f"correction capture at {site.name} is not deterministic")
        if first_rows is not None:
            assert second_rows is not None
            for first_row, second_row in zip(first_rows, second_rows, strict=True):
                for field in first_row:
                    if not np.array_equal(first_row[field], second_row[field]):
                        raise RuntimeError(
                            f"correction rows at {site.name} are not deterministic"
                        )
            for row in first_rows:
                row["site_index"] = np.asarray(site_position, dtype=np.int32)
            all_rows.extend(first_rows)
        records.append({**first_record, "deterministic_repeats": 2})

    accepted = [item for item in records if item["accepted"]]
    if len(accepted) < 6:
        raise RuntimeError(
            f"only {len(accepted)} of {len(DAGGER_CORRECTION_SITES)} correction sites "
            "were accepted; capture invalidated"
        )
    if not any(int(item["action_index"]) >= 194 for item in accepted):
        raise RuntimeError(
            "no accepted post-contact correction site (action index >= 194); capture invalidated"
        )

    arrays = {name: np.stack([row[name] for row in all_rows]) for name in all_rows[0]}
    arrays["scenario_index"] = np.zeros(len(all_rows), dtype=np.int32)
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError("correction arrays contain nonfinite values")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    arrays_bytes = stream.getvalue()
    arrays_sha256 = hashlib.sha256(arrays_bytes).hexdigest()
    identity = {
        "schema_version": 1,
        "experiment": "dagger_oracle_correction_trajectories_v1",
        "source_manifest_content_sha256": source_manifest["content_sha256"],
        "source_collection_digest": source_manifest["collection_digest"],
        "inducing_checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "inducing_report_content_sha256": inducing_report["content_sha256"],
        "inducing_rollout_executed_sha256": hashlib.sha256(first_executed.tobytes()).hexdigest(),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "sites": [asdict(site) for site in DAGGER_CORRECTION_SITES],
        "planner": {
            "method": "full_replan_nominal_speed_sub_episode",
            "sub_episode_contract": "fixed_cube_pick_place_v3",
            "sub_episode_max_actions": 480,
        },
        "arrays_sha256": arrays_sha256,
    }
    collection_digest = content_sha256(identity)
    directory = Path(output_dir) / "corrections" / collection_digest[:16]
    arrays_path = directory / "correction_trajectories.npz"
    _write_immutable_bytes(arrays_path, arrays_bytes)
    manifest = {
        **identity,
        "collection_digest": collection_digest,
        "arrays": {"path": arrays_path.name, "sha256": arrays_sha256, "rows": len(all_rows)},
        "episodes": records,
        "inducing_rollout_safety_counts": first_counts,
        "label_rule": (
            "nominal-speed privileged replan executed open-loop from the "
            "policy-induced state"
        ),
        "validation_rule": (
            "two exact captures; labels within 1e-6 ACT through the safety layer "
            "with zero clip/limit/nonfinite flags; each correction passes the "
            "unchanged v3 contract evaluated as a fresh sub-episode with zero "
            "safety events"
        ),
        "progress_rule": "deployment_clock_saturating",
        "feature_schema": ["current_act[6]", "cube_position[3]", "progress[1]"],
        "known_omissions": [
            "robot_qvel", "cube_quaternion_wxyz", "cube_linear_velocity",
            "cube_angular_velocity", "contact flags", "grasp flags",
        ],
        "claim": "policy_induced_oracle_correction_trajectories_not_deployment_data",
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = directory / "manifest.json"
    write_immutable_json(manifest_path, manifest)
    return OracleCorrectionCollection(
        directory=directory,
        manifest=manifest_path,
        arrays=arrays_path,
        collection_digest=collection_digest,
        rows=len(all_rows),
        accepted_sites=tuple(item["name"] for item in accepted),
        rejected_sites=tuple(item["name"] for item in records if not item["accepted"]),
    )


def load_oracle_corrections(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load, hash-check, and structurally validate a correction collection."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stated = manifest.get("content_sha256")
    body = dict(manifest)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError("correction manifest content hash mismatch")
    arrays_path = path.parent / manifest["arrays"]["path"]
    raw = arrays_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["arrays"]["sha256"]:
        raise ValueError("correction arrays hash mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if set(arrays) != set(_ARRAY_FIELDS):
        raise ValueError("correction array field schema mismatch")
    rows = int(manifest["arrays"]["rows"])
    if any(value.shape[0] != rows for value in arrays.values()):
        raise ValueError("correction array row count mismatch")
    accepted = [item for item in manifest.get("episodes", []) if item.get("accepted")]
    if not accepted:
        raise ValueError("correction manifest contains no accepted episodes")
    if rows != sum(int(item["rows"]) for item in accepted):
        raise ValueError("correction rows disagree with accepted episode lengths")
    action_index = np.asarray(arrays["action_index"], dtype=np.int64)
    offset = 0
    for item in accepted:
        start = int(item["action_index"])
        length = int(item["rows"])
        if not 1 <= length <= 480:
            raise ValueError("correction episode length exceeds the sub-episode budget")
        segment = action_index[offset:offset + length]
        if not np.array_equal(segment, np.arange(start, start + length, dtype=np.int64)):
            raise ValueError("correction episode rows are not a contiguous sub-episode")
        offset += length
    expected_progress = (
        np.minimum(action_index, 449).astype(np.float64) / 449.0
    ).astype(np.float32)
    if not np.array_equal(np.asarray(arrays["progress"], dtype=np.float32), expected_progress):
        raise ValueError("correction rows do not use the deployment progress clock")
    return manifest, arrays


def resolve_correction_gate_status(candidate: Mapping[str, Any]) -> str:
    """Validate the single-candidate branch trace and derive the terminal status."""
    state = candidate.get("state")
    if state == "offline_failed":
        if candidate.get("evaluation") is not None:
            raise ValueError("offline-failing correction candidate reached closed-loop evaluation")
        return "blocked_offline"
    if state == "nominal_failed":
        if candidate.get("evaluation") is None:
            raise ValueError("offline-passing correction candidate lacks nominal evaluation")
        return "closed_loop_not_resolved"
    if state == "nominal_passed":
        if candidate.get("evaluation") is None:
            raise ValueError("nominal pass lacks evaluation evidence")
        return "passed"
    raise ValueError(f"invalid correction candidate state {state!r}")


def run_correction_gate(
    model_path: str | Path,
    oracle_manifest_path: str | Path,
    correction_manifest_path: str | Path,
    baseline_checkpoint: str | Path,
    baseline_evaluation_path: str | Path,
    prefix_parity_report_path: str | Path,
    preflight_report_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
    workers: int | None = None,
) -> CorrectionGateResult:
    """Train the one authorized correction-augmented control and gate it."""
    from so_arm101_v2.learning.oracle_distillation import (
        OracleCloneKind,
        OracleDistillationConfig,
        distill_oracle_policy,
    )
    from .clone_policy import OracleCloneCheckpointPolicy
    from .observability import _evaluation_passed, _rollout_summary
    from .rollout import evaluate_closed_loop

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    correction_manifest, _ = load_oracle_corrections(correction_manifest_path)
    if (
        correction_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
    ):
        raise ValueError("correction gate manifests disagree")

    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("correction gate requires the passing deterministic v3 preflight")

    baseline_checkpoint = Path(baseline_checkpoint).resolve()
    baseline_probe = OracleCloneCheckpointPolicy(baseline_checkpoint)
    baseline_report = _load_hashed_json(
        baseline_checkpoint.with_name("report.json"), label="baseline training report"
    )
    if (
        baseline_probe.kind.value != "phase_state"
        or baseline_probe.input_dim != 10
        or baseline_probe.hidden_width != 256
        or baseline_probe.seed != 101
        or baseline_probe.manifest_content_sha256 != oracle_manifest["content_sha256"]
        or baseline_report.get("recovery_augmentation") is not None
        or baseline_report.get("correction_augmentation") is not None
        or baseline_report.get("learning_rate_schedule") != "decay_10k_20k"
        or baseline_report.get("prefix_parity", {}).get("required") is not True
        or baseline_report.get("prefix_parity", {}).get("passed") is not True
    ):
        raise ValueError(
            "baseline is not the established nominal-only width-256 scheduled phase-state clone"
        )
    baseline_sha = hashlib.sha256(baseline_checkpoint.read_bytes()).hexdigest()
    if correction_manifest.get("inducing_checkpoint_sha256") != baseline_sha:
        raise ValueError("corrections were not induced by the baseline inducing checkpoint")
    baseline_evaluation = _load_hashed_json(
        baseline_evaluation_path, label="baseline evaluation"
    )
    if (
        baseline_evaluation.get("provenance", {}).get("checkpoint_sha256") != baseline_sha
        or baseline_evaluation.get("environment_proven") is not True
        or baseline_evaluation.get("deterministic") is not True
    ):
        raise ValueError("baseline evaluation provenance mismatch")
    baseline_nominal = [
        item for item in baseline_evaluation.get("rollouts", [])
        if item.get("scenario_id") == "nominal"
    ]
    if len(baseline_nominal) != 3 or any(
        item.get("success") is not False
        or item.get("invalidated") is not False
        or int(item.get("clipping_frames", -1)) != 0
        or int(item.get("limiting_frames", -1)) != 0
        or int(item.get("nonfinite_frames", -1)) != 0
        or int(item.get("unsafe_contact_frames", -1)) != 0
        for item in baseline_nominal
    ):
        raise ValueError(
            "correction gate requires the documented baseline gap: three deterministic "
            "nominal failures with zero safety frames"
        )
    prefix_parity_report = _load_hashed_json(
        prefix_parity_report_path, label="prefix parity reference"
    )

    identity = {
        "schema_version": 1,
        "experiment": "dagger_correction_decision_gate_v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "correction_manifest_content_sha256": correction_manifest["content_sha256"],
        "baseline_checkpoint_sha256": baseline_sha,
        "baseline_training_report_content_sha256": baseline_report["content_sha256"],
        "baseline_evaluation_content_sha256": baseline_evaluation["content_sha256"],
        "prefix_parity_report_content_sha256": prefix_parity_report["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "candidate_order": ["phase_state_with_corrections"],
        "fixed_training_config": {
            "model_kind": "phase_state",
            "seed": 101,
            "hidden_width": 256,
            "max_steps": 30_000,
            "learning_rate_schedule": "decay_10k_20k",
            "correction_loss_weight": 1.0,
            "nominal_normalized_mse_threshold": 1e-6,
            "maximum_act_error_threshold": 0.01,
            "baseline_improvement_factor": 100.0,
        },
        "pass_rule": "three deterministic nominal successes with zero safety counts",
        "blocked_offline_policy": "terminal_stop_record_loss_floor",
    }
    gate_digest = content_sha256(identity)
    directory = output_dir / "correction_gates" / gate_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="correction gate report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable correction gate differs: {directory}")
        return CorrectionGateResult(directory, report_path, str(existing["status"]))

    training = distill_oracle_policy(
        oracle_manifest_path,
        output_dir,
        kind=OracleCloneKind.PHASE_STATE,
        config=OracleDistillationConfig(
            seed=101,
            hidden_width=256,
            max_steps=30_000,
            lr_schedule="decay_10k_20k",
        ),
        prefix_parity_report=prefix_parity_report_path,
        correction_manifest_path=correction_manifest_path,
    )
    training_report = _load_hashed_json(training.report_json, label="candidate training report")
    candidate: dict[str, Any] = {
        "model_kind": "phase_state_with_corrections",
        "state": "offline_failed" if not training.passed else "offline_passed",
        "checkpoint_sha256": hashlib.sha256(training.checkpoint.read_bytes()).hexdigest(),
        "training_report": str(training.report_json.resolve()),
        "training_report_content_sha256": training_report["content_sha256"],
        "offline": {
            "passed": bool(training.passed),
            "steps": int(training.steps),
            "normalized_mse": float(training.normalized_mse),
            "maximum_act_error": float(training.max_act_error),
            "training_safety_violations": int(training_report["training_safety_violations"]),
            "best_observed": training_report["best_observed"],
        },
        "evaluation": None,
    }
    if training.passed:
        suite = load_simulation_suite("fixed_pick_place_v3")
        nominal_suite = replace(
            suite,
            suite_id="fixed_pick_place_v3.nominal_correction_gate",
            scenarios=tuple(item for item in suite.scenarios if item.scenario_id == "nominal"),
            repeats=3,
        )
        probe = OracleCloneCheckpointPolicy(training.checkpoint)
        evaluation_identity = content_sha256({
            "gate_digest": gate_digest,
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "training_report_content_sha256": training_report["content_sha256"],
            "preflight_report_content_sha256": preflight["content_sha256"],
        })
        evaluation = evaluate_closed_loop(
            model_path,
            nominal_suite,
            {probe.policy_id: PolicySpec(kind="oracle_clone", checkpoint=str(training.checkpoint.resolve()))},
            output_dir / "correction_gate_evaluations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=record_video,
            workers=workers,
            provenance={
                "gate_digest": gate_digest,
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "offline_report_content_sha256": training_report["content_sha256"],
                "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
                "correction_manifest_content_sha256": correction_manifest["content_sha256"],
                "preflight_report_content_sha256": preflight["content_sha256"],
            },
        )
        evaluation_report = _load_hashed_json(
            evaluation.report_json, label="candidate evaluation report"
        )
        passed = _evaluation_passed(evaluation_report)
        candidate["state"] = "nominal_passed" if passed else "nominal_failed"
        candidate["evaluation"] = {
            "report": str(evaluation.report_json.resolve()),
            "content_sha256": evaluation_report["content_sha256"],
            "passed": passed,
            "deterministic": bool(evaluation_report["deterministic"]),
            "rollouts": [_rollout_summary(item) for item in evaluation_report["rollouts"]],
        }

    status = resolve_correction_gate_status(candidate)
    next_actions = {
        "passed": "broader_evaluation_and_anchor_handoffs_of_correction_policy_only",
        "blocked_offline": "stop_tranche_documented_loss_floor_no_gate_or_capacity_changes",
        "closed_loop_not_resolved": "stop_single_dagger_round_and_reassess_before_any_new_proposal",
    }
    accepted_sites = [
        item["name"] for item in correction_manifest["episodes"] if item["accepted"]
    ]
    rejected_sites = [
        item["name"] for item in correction_manifest["episodes"] if not item["accepted"]
    ]
    report = {
        **identity,
        "gate_digest": gate_digest,
        "status": status,
        "next_action": next_actions[status],
        "correction_coverage": {
            "accepted_sites": accepted_sites,
            "rejected_sites": rejected_sites,
            "correction_rows": int(correction_manifest["arrays"]["rows"]),
        },
        "baseline_nominal_rollouts": [_rollout_summary(item) for item in baseline_nominal],
        "candidate": candidate,
        "training_runs_started": 1,
        "nominal_evaluations_started": int(candidate["evaluation"] is not None),
        "partial_milestones_are_non_promoting": True,
        "broader_evaluation_run": False,
        "anchor_handoff_evaluation_run": False,
        "claim": "single_candidate_dagger_correction_decision_not_deployment_success",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return CorrectionGateResult(directory, report_path, status)


__all__ = [
    "CorrectionGateResult", "CorrectionSite", "DAGGER_CORRECTION_SITES",
    "OracleCorrectionCollection",
    "capture_dagger_corrections", "load_oracle_corrections",
    "resolve_correction_gate_status", "run_correction_gate",
]
