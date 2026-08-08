"""Deterministic, phase-wide recovery examples for oracle distillation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any

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
from .oracle import load_oracle_demonstrations
from .suites import load_simulation_suite


@dataclass(frozen=True)
class RecoveryAnchor:
    name: str
    action_index: int
    perturbation_act: tuple[float, ...]
    rationale: str

    def __post_init__(self) -> None:
        if not self.name or not 1 <= self.action_index < 450:
            raise ValueError("invalid recovery anchor")
        if len(self.perturbation_act) != 6 or max(abs(value) for value in self.perturbation_act) > 0.0100001:
            raise ValueError("recovery perturbations must be six-dimensional and at most 0.01 ACT")
        if not any(self.perturbation_act):
            raise ValueError("recovery perturbation must be nonzero")


# One deliberately small, structured perturbation per major phase. Each is
# injected into the preceding command, so MuJoCo—not array arithmetic—creates
# the recovery state and any resulting contact/cube displacement.
PHASE_WIDE_RECOVERY_ANCHORS = (
    RecoveryAnchor("approach", 70, (0, .01, -.01, 0, 0, 0), "end of high approach"),
    RecoveryAnchor("first_contact", 195, (.01, 0, 0, -.01, 0, 0), "first post-contact state"),
    RecoveryAnchor("seating", 205, (-.01, 0, .01, 0, 0, 0), "start of final centering"),
    RecoveryAnchor("closure", 255, (0, 0, 0, 0, 0, .01), "start of jaw closure"),
    RecoveryAnchor("lift", 315, (0, .01, -.01, 0, 0, 0), "start of lift motion"),
    RecoveryAnchor("transport", 365, (.01, 0, 0, -.01, 0, 0), "start of carry to napkin"),
    RecoveryAnchor("placement", 386, (0, -.01, .01, 0, 0, 0), "start of set-down"),
    RecoveryAnchor("release", 419, (0, 0, 0, 0, 0, .01), "end of controlled release"),
)


@dataclass(frozen=True)
class OracleRecoveryCollection:
    directory: Path
    manifest: Path
    arrays: Path
    collection_digest: str
    rows: int


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    write_immutable_bytes(
        path, data, conflict_message=f"immutable recovery artifact differs: {path}"
    )


def _run_anchor(
    model_path: Path,
    source: dict[str, np.ndarray],
    anchor: RecoveryAnchor,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    suite = load_simulation_suite("fixed_pick_place_v3")
    scenario = next(item for item in suite.scenarios if item.scenario_id == "nominal")
    contract = load_pick_place_contract(suite.task_contract)
    adapter = MujocoTaskAdapter(model_path)
    state = PickPlaceEvaluationState()
    evaluation = None
    captured = None
    label_command = None
    counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
    try:
        adapter.reset(scenario)
        for action_index in range(450):
            requested = np.asarray(source["executed_act"][action_index], dtype=np.float32).copy()
            if action_index == anchor.action_index - 1:
                requested += np.asarray(anchor.perturbation_act, dtype=np.float32)
            if action_index == anchor.action_index:
                captured = adapter.privileged_state()
            command = adapter.apply_policy_command(requested)
            if action_index == anchor.action_index:
                label_command = command
            adapter.advance_control_period()
            measurement, _ = adapter.pick_place_measurement(
                command,
                footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
            )
            if not state.completed:
                state, evaluation = evaluate_pick_place_step(contract, measurement, state)
            counts["clip"] += int(measurement.pickup.command_bound_violation)
            counts["limit"] += int(measurement.pickup.delta_limiter_activated)
            counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
            counts["unsafe"] += int(measurement.pickup.unsafe_contact)
    finally:
        adapter.close()
    if captured is None or label_command is None or evaluation is None:
        raise RuntimeError(f"recovery anchor {anchor.name} was not fully evaluated")
    if not evaluation.success or any(counts.values()):
        raise RuntimeError(
            f"oracle suffix is not an unambiguous safe recovery at {anchor.name}: "
            f"success={evaluation.success}, safety={counts}"
        )
    intended = np.asarray(source["executed_act"][anchor.action_index], dtype=np.float32)
    if not np.array_equal(label_command.executed_act, intended):
        raise RuntimeError(f"recovery label at {anchor.name} was changed by the safety layer")
    nominal_current = np.asarray(source["current_act"][anchor.action_index], dtype=np.float32)
    nominal_cube = np.asarray(source["cube_position"][anchor.action_index], dtype=np.float32)
    observable_delta = np.concatenate((
        captured.current_act - nominal_current,
        captured.cube_position - nominal_cube,
    ))
    if float(np.max(np.abs(observable_delta))) <= 1e-5:
        raise RuntimeError(f"recovery perturbation at {anchor.name} did not reach the observed state")
    row = {
        "action_index": np.asarray(anchor.action_index, dtype=np.int32),
        "progress": np.asarray(anchor.action_index / 449.0, dtype=np.float32),
        "current_act": captured.current_act,
        "robot_qvel": captured.robot_qvel,
        "cube_position": captured.cube_position,
        "cube_quaternion_wxyz": captured.cube_quaternion_wxyz,
        "cube_linear_velocity": captured.cube_linear_velocity,
        "cube_angular_velocity": captured.cube_angular_velocity,
        "executed_act": label_command.executed_act,
        "executed_delta_act": label_command.executed_act - captured.current_act,
    }
    validation = {
        "oracle_label": label_command.executed_act.tolist(),
        "oracle_suffix_success": True,
        "oracle_suffix_safety_counts": counts,
        "safety_layer_changed_label": False,
        "observable_delta_from_nominal": {
            "current_act": (captured.current_act - nominal_current).tolist(),
            "cube_position_m": (captured.cube_position - nominal_cube).tolist(),
        },
        "omitted_input_delta_from_nominal": {
            "robot_qvel": (captured.robot_qvel - source["robot_qvel"][anchor.action_index]).tolist(),
            "cube_quaternion_wxyz": (
                captured.cube_quaternion_wxyz - source["cube_quaternion_wxyz"][anchor.action_index]
            ).tolist(),
            "cube_linear_velocity": (
                captured.cube_linear_velocity - source["cube_linear_velocity"][anchor.action_index]
            ).tolist(),
            "cube_angular_velocity": (
                captured.cube_angular_velocity - source["cube_angular_velocity"][anchor.action_index]
            ).tolist(),
        },
    }
    return row, validation


def capture_phase_wide_recovery_examples(
    model_path: str | Path,
    oracle_manifest: str | Path,
    output_dir: str | Path,
) -> OracleRecoveryCollection:
    """Create eight validated examples without changing the nominal capture."""
    source_manifest, source = load_oracle_demonstrations(oracle_manifest)
    if source_manifest.get("scenario_ids") != ["nominal"] or int(source_manifest.get("teacher_horizon", 0)) != 450:
        raise ValueError("phase-wide recovery requires the canonical nominal 450-row oracle capture")
    model_path = Path(model_path).resolve()
    rows: list[dict[str, np.ndarray]] = []
    records: list[dict[str, Any]] = []
    for anchor in PHASE_WIDE_RECOVERY_ANCHORS:
        first_row, first_validation = _run_anchor(model_path, source, anchor)
        second_row, second_validation = _run_anchor(model_path, source, anchor)
        for field in first_row:
            if not np.array_equal(first_row[field], second_row[field]):
                raise RuntimeError(f"recovery state at {anchor.name} is not deterministic")
        if first_validation != second_validation:
            raise RuntimeError(f"recovery validation at {anchor.name} is not deterministic")
        rows.append(first_row)
        records.append({**asdict(anchor), **first_validation, "deterministic_repeats": 2})

    arrays = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
    arrays["scenario_index"] = np.zeros(len(rows), dtype=np.int32)
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError("recovery arrays contain nonfinite values")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    arrays_bytes = stream.getvalue()
    arrays_sha256 = hashlib.sha256(arrays_bytes).hexdigest()
    identity = {
        "schema_version": 1,
        "experiment": "phase_wide_oracle_recovery_v1",
        "source_manifest_content_sha256": source_manifest["content_sha256"],
        "source_collection_digest": source_manifest["collection_digest"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "anchors": [asdict(anchor) for anchor in PHASE_WIDE_RECOVERY_ANCHORS],
        "arrays_sha256": arrays_sha256,
    }
    collection_digest = content_sha256(identity)
    directory = Path(output_dir) / "recovery" / collection_digest[:16]
    arrays_path = directory / "recovery_examples.npz"
    _write_immutable_bytes(arrays_path, arrays_bytes)
    manifest = {
        **identity,
        "collection_digest": collection_digest,
        "arrays": {"path": arrays_path.name, "sha256": arrays_sha256, "rows": len(rows)},
        "records": records,
        "label_rule": "unchanged oracle absolute action at the same phase index",
        "validation_rule": "two exact captures; unchanged label through safety layer; full fixed-oracle suffix succeeds without safety events",
        "feature_schema": ["current_act[6]", "cube_position[3]", "progress[1]"],
        "known_omissions": [
            "robot_qvel", "cube_quaternion_wxyz", "cube_linear_velocity", "cube_angular_velocity",
            "contact flags", "grasp flags",
        ],
        "claim": "eight_local_validated_recovery_labels_not_a_broad_robustness_distribution",
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = directory / "manifest.json"
    write_immutable_json(manifest_path, manifest)
    return OracleRecoveryCollection(directory, manifest_path, arrays_path, collection_digest, len(rows))


def load_oracle_recovery_examples(path: str | Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stated = manifest.get("content_sha256")
    body = dict(manifest)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError("recovery manifest content hash mismatch")
    arrays_path = path.parent / manifest["arrays"]["path"]
    raw = arrays_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["arrays"]["sha256"]:
        raise ValueError("recovery arrays hash mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if any(value.shape[0] != manifest["arrays"]["rows"] for value in arrays.values()):
        raise ValueError("recovery array row count mismatch")
    return manifest, arrays


def evaluate_recovery_anchor_starts(
    model_path: str | Path,
    oracle_manifest: str | Path,
    recovery_manifest_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> Path:
    """Replay each validated prefix, then hand control to the student at its anchor."""
    from .clone_policy import OracleCloneCheckpointPolicy

    source_manifest, source = load_oracle_demonstrations(oracle_manifest)
    recovery_manifest, _ = load_oracle_recovery_examples(recovery_manifest_path)
    if recovery_manifest.get("source_manifest_content_sha256") != source_manifest["content_sha256"]:
        raise ValueError("recovery and oracle manifests disagree")
    checkpoint = Path(checkpoint)
    model_path = Path(model_path).resolve()
    policy_probe = OracleCloneCheckpointPolicy(checkpoint)
    identity = {
        "schema_version": 1,
        "experiment": "phase_wide_recovery_anchor_evaluation_v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_report_content_sha256": policy_probe.report_content_sha256,
        "oracle_manifest_content_sha256": source_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    digest = content_sha256(identity)
    directory = Path(output_dir) / "recovery_evaluations" / digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        stated = existing.get("content_sha256")
        body = dict(existing)
        body.pop("content_sha256", None)
        if stated != content_sha256(body) or any(existing.get(k) != v for k, v in identity.items()):
            raise FileExistsError(f"immutable recovery evaluation differs: {directory}")
        return report_path

    suite = load_simulation_suite("fixed_pick_place_v3")
    scenario = next(item for item in suite.scenarios if item.scenario_id == "nominal")
    contract = load_pick_place_contract(suite.task_contract)
    results: list[dict[str, Any]] = []
    for anchor in PHASE_WIDE_RECOVERY_ANCHORS:
        adapter = MujocoTaskAdapter(model_path)
        policy = OracleCloneCheckpointPolicy(checkpoint)
        state = PickPlaceEvaluationState()
        evaluation = None
        counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
        events: set[str] = set()
        maximum_height = 0.0
        first_student_action = None
        try:
            adapter.reset(scenario)
            policy.action_index = anchor.action_index
            for action_index in range(450):
                if action_index < anchor.action_index:
                    requested = np.asarray(source["executed_act"][action_index], dtype=np.float32).copy()
                    if action_index == anchor.action_index - 1:
                        requested += np.asarray(anchor.perturbation_act, dtype=np.float32)
                else:
                    requested = policy.predict(
                        np.empty((0,), dtype=np.uint8), adapter.current_act(), adapter,
                    )
                    if first_student_action is None:
                        first_student_action = requested.tolist()
                command = adapter.apply_policy_command(requested)
                adapter.advance_control_period()
                measurement, _ = adapter.pick_place_measurement(
                    command,
                    footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
                )
                maximum_height = max(maximum_height, measurement.pickup.cube_height_gain_m)
                if not state.completed:
                    state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                    events.update(event.value for event in evaluation.pickup_events)
                    events.update(event.value for event in evaluation.events)
                if action_index >= anchor.action_index:
                    counts["clip"] += int(measurement.pickup.command_bound_violation)
                    counts["limit"] += int(measurement.pickup.delta_limiter_activated)
                    counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
                    counts["unsafe"] += int(measurement.pickup.unsafe_contact)
        finally:
            adapter.close()
        assert evaluation is not None and first_student_action is not None
        results.append({
            "anchor": anchor.name,
            "action_index": anchor.action_index,
            "student_actions": 450 - anchor.action_index,
            "success": bool(evaluation.success),
            "pickup_completed": "success" in events,
            "events": sorted(events),
            "maximum_cube_height_gain_m": maximum_height,
            "student_safety_counts": counts,
            "first_student_action": first_student_action,
        })
    report = {
        **identity,
        "evaluation_digest": digest,
        "results": results,
        "passed": all(item["success"] and not any(item["student_safety_counts"].values()) for item in results),
        "setup": "validated oracle prefix with one physical perturbation, autonomous student from anchor through action 449",
        "claim": "exact_anchor_handoff_diagnostic_not_random_perturbation_robustness",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return report_path


def evaluate_policy_anchor_starts(
    model_path: str | Path,
    oracle_manifest: str | Path,
    recovery_manifest_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> Path:
    """Anchor-handoff evaluation for a chunked clone checkpoint.

    Same protocol as :func:`evaluate_recovery_anchor_starts` (validated oracle
    prefix with one physical perturbation, autonomous student from the anchor
    onward), generalized to the chunked policy family and with the nominal
    episode selected explicitly by ``scenario_index`` rather than row order.
    """
    from .chunked import ChunkedClonePolicy

    source_manifest, source = load_oracle_demonstrations(oracle_manifest)
    recovery_manifest, _ = load_oracle_recovery_examples(recovery_manifest_path)
    if recovery_manifest.get("source_manifest_content_sha256") != source_manifest["content_sha256"]:
        raise ValueError("recovery and oracle manifests disagree")
    nominal_mask = np.asarray(source["scenario_index"], dtype=np.int64) == 0
    nominal_executed = np.asarray(source["executed_act"], dtype=np.float32)[nominal_mask]
    if nominal_executed.shape[0] != 450:
        raise ValueError("anchor handoffs require a 450-row nominal episode")
    checkpoint = Path(checkpoint)
    model_path = Path(model_path).resolve()
    policy_probe = ChunkedClonePolicy(checkpoint)
    identity = {
        "schema_version": 1,
        "experiment": "policy_anchor_handoff_evaluation_v1",
        "policy_kind": "chunked_clone",
        "policy_id": policy_probe.policy_id,
        "chunk_horizon": int(policy_probe.chunk_horizon),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_report_content_sha256": policy_probe.report_content_sha256,
        "oracle_manifest_content_sha256": source_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    digest = content_sha256(identity)
    directory = Path(output_dir) / "policy_anchor_evaluations" / digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        stated = existing.get("content_sha256")
        body = dict(existing)
        body.pop("content_sha256", None)
        if stated != content_sha256(body) or any(existing.get(k) != v for k, v in identity.items()):
            raise FileExistsError(f"immutable policy anchor evaluation differs: {directory}")
        return report_path

    suite = load_simulation_suite("fixed_pick_place_v3")
    scenario = next(item for item in suite.scenarios if item.scenario_id == "nominal")
    contract = load_pick_place_contract(suite.task_contract)
    results: list[dict[str, Any]] = []
    for anchor in PHASE_WIDE_RECOVERY_ANCHORS:
        adapter = MujocoTaskAdapter(model_path)
        policy = ChunkedClonePolicy(checkpoint)
        state = PickPlaceEvaluationState()
        evaluation = None
        counts = {"clip": 0, "limit": 0, "nonfinite": 0, "unsafe": 0}
        events: set[str] = set()
        maximum_height = 0.0
        first_student_action = None
        try:
            adapter.reset(scenario)
            policy.action_index = anchor.action_index
            for action_index in range(450):
                if action_index < anchor.action_index:
                    requested = np.asarray(nominal_executed[action_index], dtype=np.float32).copy()
                    if action_index == anchor.action_index - 1:
                        requested += np.asarray(anchor.perturbation_act, dtype=np.float32)
                else:
                    requested = policy.predict(
                        np.empty((0,), dtype=np.uint8), adapter.current_act(), adapter,
                    )
                    if first_student_action is None:
                        first_student_action = requested.tolist()
                command = adapter.apply_policy_command(requested)
                adapter.advance_control_period()
                measurement, _ = adapter.pick_place_measurement(
                    command,
                    footprint_edge_margin_m=contract.placement.footprint_edge_margin_m,
                )
                maximum_height = max(maximum_height, measurement.pickup.cube_height_gain_m)
                if not state.completed:
                    state, evaluation = evaluate_pick_place_step(contract, measurement, state)
                    events.update(event.value for event in evaluation.pickup_events)
                    events.update(event.value for event in evaluation.events)
                if action_index >= anchor.action_index:
                    counts["clip"] += int(measurement.pickup.command_bound_violation)
                    counts["limit"] += int(measurement.pickup.delta_limiter_activated)
                    counts["nonfinite"] += int(measurement.pickup.nonfinite_command)
                    counts["unsafe"] += int(measurement.pickup.unsafe_contact)
        finally:
            adapter.close()
        assert evaluation is not None and first_student_action is not None
        results.append({
            "anchor": anchor.name,
            "action_index": anchor.action_index,
            "student_actions": 450 - anchor.action_index,
            "success": bool(evaluation.success),
            "pickup_completed": "success" in events,
            "events": sorted(events),
            "maximum_cube_height_gain_m": maximum_height,
            "student_safety_counts": counts,
            "first_student_action": first_student_action,
        })
    report = {
        **identity,
        "evaluation_digest": digest,
        "results": results,
        "passed": all(item["success"] and not any(item["student_safety_counts"].values()) for item in results),
        "setup": (
            "validated nominal oracle prefix with one physical perturbation, "
            "autonomous chunked student from anchor through action 449"
        ),
        "claim": "exact_anchor_handoff_diagnostic_not_random_perturbation_robustness",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return report_path


def scan_oracle_clone_commands(
    oracle_manifest: str | Path,
    recovery_manifest_path: str | Path,
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    samples_per_path: int = 101,
) -> Path:
    """Densely screen commands on local feature paths without invoking MuJoCo."""
    from so_arm101_v2.contracts import evaluate_physical_command
    from so_arm101_v2.learning.oracle_distillation import build_oracle_features
    from .clone_policy import OracleCloneCheckpointPolicy

    if samples_per_path < 3:
        raise ValueError("static scan requires at least three samples per path")
    source_manifest, source = load_oracle_demonstrations(oracle_manifest)
    recovery_manifest, recovery = load_oracle_recovery_examples(recovery_manifest_path)
    if recovery_manifest.get("source_manifest_content_sha256") != source_manifest["content_sha256"]:
        raise ValueError("recovery and oracle manifests disagree")
    checkpoint = Path(checkpoint)
    policy = OracleCloneCheckpointPolicy(checkpoint)
    if policy.kind.value != "phase_state" or policy.input_dim != 10:
        raise ValueError("static scan requires the 10-input phase_state clone")
    nominal_features, _, _ = build_oracle_features(
        policy.kind, source, extras_mean=policy.extras_mean, extras_std=policy.extras_std,
    )
    recovery_features, _, _ = build_oracle_features(
        policy.kind, recovery, extras_mean=policy.extras_mean, extras_std=policy.extras_std,
    )
    nominal_current = np.asarray(source["current_act"], dtype=np.float32)
    recovery_current = np.asarray(recovery["current_act"], dtype=np.float32)
    recovery_indices = np.asarray(recovery["action_index"], dtype=np.int64)
    if nominal_features.shape[0] != 450 or recovery_features.shape[0] != 8:
        raise ValueError("static scan requires the immutable 450 nominal and 8 recovery rows")

    alphas = np.linspace(0.0, 1.0, samples_per_path, dtype=np.float32)
    features: list[np.ndarray] = []
    currents: list[np.ndarray] = []
    families: list[str] = []
    paths: list[str] = []
    for index in range(449):
        features.append(
            nominal_features[index][None] * (1.0 - alphas[:, None])
            + nominal_features[index + 1][None] * alphas[:, None]
        )
        currents.append(
            nominal_current[index][None] * (1.0 - alphas[:, None])
            + nominal_current[index + 1][None] * alphas[:, None]
        )
        families.extend(["consecutive_nominal_segment"] * samples_per_path)
        paths.extend([f"nominal:{index}->{index + 1}"] * samples_per_path)
    for recovery_position, nominal_index in enumerate(recovery_indices.tolist()):
        features.append(
            nominal_features[nominal_index][None] * (1.0 - alphas[:, None])
            + recovery_features[recovery_position][None] * alphas[:, None]
        )
        currents.append(
            nominal_current[nominal_index][None] * (1.0 - alphas[:, None])
            + recovery_current[recovery_position][None] * alphas[:, None]
        )
        families.extend(["same_phase_nominal_to_recovery_chord"] * samples_per_path)
        paths.extend([
            f"phase:{nominal_index}:{recovery_manifest['records'][recovery_position]['name']}"
        ] * samples_per_path)
    feature_batch = np.concatenate(features).astype(np.float32)
    current_batch = np.concatenate(currents).astype(np.float32)
    with policy.torch.inference_mode():
        normalized_delta = policy.model(policy.torch.from_numpy(feature_batch)).numpy().astype(np.float32)
    predicted_batch = current_batch + normalized_delta * policy.maximum_delta

    counts = {
        "samples_with_any_violation": 0,
        "act_clip_components": 0,
        "mujoco_clip_components": 0,
        "physical_clip_components": 0,
        "relative_limit_components": 0,
        "nonfinite_samples": int(np.count_nonzero(~np.all(np.isfinite(predicted_batch), axis=1))),
    }
    family_counts: dict[str, dict[str, int]] = {}
    first_violation: dict[str, Any] | None = None
    for sample_index, (current, predicted) in enumerate(zip(current_batch, predicted_batch, strict=True)):
        family = families[sample_index]
        summary = family_counts.setdefault(family, {"samples": 0, "samples_with_any_violation": 0})
        summary["samples"] += 1
        evaluation = evaluate_physical_command(current, predicted)
        components = {
            "act_clip_components": int(np.count_nonzero(evaluation.act_clip_mask)),
            "mujoco_clip_components": int(np.count_nonzero(evaluation.mujoco_clip_mask)),
            "physical_clip_components": int(np.count_nonzero(evaluation.physical_clip_mask)),
            "relative_limit_components": int(np.count_nonzero(evaluation.relative_limit_mask)),
        }
        violated = any(components.values()) or not np.all(np.isfinite(predicted))
        counts["samples_with_any_violation"] += int(violated)
        summary["samples_with_any_violation"] += int(violated)
        for key, value in components.items():
            counts[key] += value
        if violated and first_violation is None:
            first_violation = {
                "sample_index": sample_index,
                "family": family,
                "path": paths[sample_index],
                "current_act": current.tolist(),
                "predicted_act": predicted.tolist(),
                **components,
            }

    identity = {
        "schema_version": 1,
        "experiment": "local_phase_consistent_static_command_scan_v1",
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "checkpoint_report_content_sha256": policy.report_content_sha256,
        "oracle_manifest_content_sha256": source_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "samples_per_path": samples_per_path,
    }
    digest = content_sha256(identity)
    report = {
        **identity,
        "scan_digest": digest,
        "passed": counts["samples_with_any_violation"] == 0 and counts["nonfinite_samples"] == 0,
        "paths": {"consecutive_nominal_segments": 449, "same_phase_recovery_chords": 8},
        "total_samples": int(feature_batch.shape[0]),
        "counts": counts,
        "family_counts": family_counts,
        "first_violation": first_violation,
        "maximum_absolute_normalized_delta": float(np.max(np.abs(normalized_delta))),
        "scope": {
            "arbitrary_cross_phase_mixtures": False,
            "mujoco_used": False,
            "interpretation": "cheap necessary static screen, not proof of physical safety",
        },
    }
    report["content_sha256"] = content_sha256(report)
    report_path = Path(output_dir) / "static_command_scans" / digest[:16] / "report.json"
    write_immutable_json(report_path, report)
    return report_path


__all__ = [
    "OracleRecoveryCollection", "PHASE_WIDE_RECOVERY_ANCHORS", "RecoveryAnchor",
    "capture_phase_wide_recovery_examples", "load_oracle_recovery_examples",
    "evaluate_recovery_anchor_starts", "evaluate_policy_anchor_starts",
    "scan_oracle_clone_commands",
]
