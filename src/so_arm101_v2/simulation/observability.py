"""Causal privileged-observation artifacts and the bounded decision gate."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from so_arm101_v2.data._serialization import (
    content_sha256,
    write_immutable_bytes,
    write_immutable_json,
)

from .adapter import MujocoTaskAdapter, PrivilegedStateSnapshot
from .oracle import load_oracle_demonstrations
from .policy_specs import PolicySpec
from .privileged import PrivilegedStagedController
from .recovery import PHASE_WIDE_RECOVERY_ANCHORS, load_oracle_recovery_examples
from .suites import load_simulation_suite


STATE_FIELDS = (
    "current_act",
    "robot_qvel",
    "cube_position",
    "cube_quaternion_wxyz",
    "cube_linear_velocity",
    "cube_angular_velocity",
)
CONTACT_FIELDS = (
    "any_contact",
    "bilateral_interior_contact",
    "strict_bilateral_grasp",
)
REPLAY_ABSOLUTE_TOLERANCES = {
    "current_act": 5e-6,
    "robot_qvel": 5e-4,
    "cube_position": 1e-6,
    "cube_quaternion_wxyz": 2e-6,
    "cube_linear_velocity": 1e-5,
    "cube_angular_velocity": 5e-4,
}


@dataclass(frozen=True)
class ObservabilityCollection:
    directory: Path
    manifest: Path
    arrays: Path
    collection_digest: str


@dataclass(frozen=True)
class ObservabilityGateResult:
    directory: Path
    report_json: Path
    status: str


def _write_immutable_bytes(path: Path, data: bytes) -> None:
    write_immutable_bytes(
        path, data, conflict_message=f"immutable observability artifact differs: {path}"
    )


def _snapshot_arrays(snapshot: PrivilegedStateSnapshot) -> dict[str, np.ndarray]:
    return {
        name: np.asarray(getattr(snapshot, name), dtype=np.float32).copy()
        for name in STATE_FIELDS
    }


def _assert_snapshot_equal(
    snapshot: PrivilegedStateSnapshot,
    arrays: Mapping[str, np.ndarray],
    row: int,
    *,
    label: str,
    maxima: dict[str, float] | None = None,
) -> None:
    for name in STATE_FIELDS:
        actual = np.asarray(getattr(snapshot, name), dtype=np.float32)
        expected = np.asarray(arrays[name][row], dtype=np.float32)
        difference = float(np.max(np.abs(actual - expected)))
        if maxima is not None:
            maxima[name] = max(maxima.get(name, 0.0), difference)
        if difference > REPLAY_ABSOLUTE_TOLERANCES[name]:
            raise RuntimeError(
                f"observability replay state mismatch at {label} field {name}: {difference}"
            )


def _empty_rows() -> dict[str, list[np.ndarray]]:
    return {
        "contact_flags": [],
        "previous_contact_flags": [],
        **{f"previous_{name}": [] for name in STATE_FIELDS},
    }


def _append_observation(adapter: MujocoTaskAdapter, rows: dict[str, list[np.ndarray]]) -> None:
    previous = adapter.previous_privileged_state()
    rows["contact_flags"].append(adapter.privileged_contact_state().as_array())
    rows["previous_contact_flags"].append(
        adapter.previous_privileged_contact_state().as_array()
    )
    for name, value in _snapshot_arrays(previous).items():
        rows[f"previous_{name}"].append(value)


def _materialize(prefix: str, rows: Mapping[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    return {
        f"{prefix}__{name}": np.stack(values).astype(np.float32)
        for name, values in rows.items()
    }


def capture_observability_annotations(
    model_path: str | Path,
    oracle_manifest_path: str | Path,
    recovery_manifest_path: str | Path,
    output_dir: str | Path,
) -> ObservabilityCollection:
    """Replay canonical data and record causal contact plus two-frame history."""
    oracle_manifest, source = load_oracle_demonstrations(oracle_manifest_path)
    recovery_manifest, recovery = load_oracle_recovery_examples(recovery_manifest_path)
    if (
        oracle_manifest.get("scenario_ids") != ["nominal"]
        or int(oracle_manifest.get("teacher_horizon", 0)) != 450
        or int(oracle_manifest["arrays"]["rows"]) != 450
    ):
        raise ValueError("observability capture requires the canonical nominal 450-row oracle")
    if (
        recovery_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
        or int(recovery_manifest["arrays"]["rows"]) != len(PHASE_WIDE_RECOVERY_ANCHORS)
    ):
        raise ValueError("observability capture requires the canonical eight-row recovery set")

    model_path = Path(model_path).resolve()
    suite = load_simulation_suite("fixed_pick_place_v3")
    scenario = next(item for item in suite.scenarios if item.scenario_id == "nominal")

    nominal_rows = _empty_rows()
    replay_maxima = {name: 0.0 for name in STATE_FIELDS}
    adapter = MujocoTaskAdapter(model_path)
    try:
        adapter.reset(scenario)
        # The immutable oracle capture initialized its IK controller before
        # reading row zero.  That setup restores qpos/qvel but refreshes derived
        # MuJoCo state, so reproduce it exactly and then apply the row-zero
        # repeat-current history rule.
        PrivilegedStagedController().reset(adapter)
        adapter.rebase_observation_history()
        for action_index in range(450):
            snapshot = adapter.privileged_state()
            _assert_snapshot_equal(
                snapshot, source, action_index, label=f"nominal:{action_index}",
                maxima=replay_maxima,
            )
            previous = adapter.previous_privileged_state()
            expected_previous = action_index - 1 if action_index else 0
            _assert_snapshot_equal(
                previous, source, expected_previous,
                label=f"nominal:{action_index}:previous",
                maxima=replay_maxima,
            )
            _append_observation(adapter, nominal_rows)
            # Rendering is part of the immutable oracle capture's exact replay
            # path even though pixels are not observability inputs.
            adapter.render_wrist_observation()
            adapter.render("camera_side")
            requested = np.asarray(source["executed_act"][action_index], dtype=np.float32)
            command = adapter.apply_policy_command(requested)
            if not np.array_equal(command.executed_act, requested):
                raise RuntimeError(f"nominal replay label changed at action {action_index}")
            adapter.advance_control_period()
    finally:
        adapter.close()

    recovery_rows = _empty_rows()
    for recovery_position, anchor in enumerate(PHASE_WIDE_RECOVERY_ANCHORS):
        if int(recovery["action_index"][recovery_position]) != anchor.action_index:
            raise ValueError("recovery rows do not match the fixed anchor order")
        adapter = MujocoTaskAdapter(model_path)
        try:
            adapter.reset(scenario)
            for action_index in range(anchor.action_index + 1):
                if action_index == anchor.action_index:
                    snapshot = adapter.privileged_state()
                    _assert_snapshot_equal(
                        snapshot, recovery, recovery_position, label=f"recovery:{anchor.name}",
                        maxima=replay_maxima,
                    )
                    _append_observation(adapter, recovery_rows)
                    requested = np.asarray(
                        recovery["executed_act"][recovery_position], dtype=np.float32
                    )
                    command = adapter.apply_policy_command(requested)
                    if not np.array_equal(command.executed_act, requested):
                        raise RuntimeError(f"recovery replay label changed at {anchor.name}")
                    break
                requested = np.asarray(source["executed_act"][action_index], dtype=np.float32).copy()
                if action_index == anchor.action_index - 1:
                    requested += np.asarray(anchor.perturbation_act, dtype=np.float32)
                adapter.apply_policy_command(requested)
                adapter.advance_control_period()
        finally:
            adapter.close()

    arrays = {
        **_materialize("nominal", nominal_rows),
        **_materialize("recovery", recovery_rows),
    }
    if not all(np.all(np.isfinite(value)) for value in arrays.values()):
        raise RuntimeError("observability arrays contain nonfinite values")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    arrays_bytes = stream.getvalue()
    arrays_sha256 = hashlib.sha256(arrays_bytes).hexdigest()
    identity = {
        "schema_version": 1,
        "experiment": "bounded_observability_annotations_v1",
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "arrays_sha256": arrays_sha256,
        "history_length": 2,
        "history_padding": "repeat_current_at_episode_reset",
        "contact_timing": "pre_action_before_current_command_selection",
        "contact_fields": list(CONTACT_FIELDS),
        "state_fields": list(STATE_FIELDS),
        "replay_absolute_tolerances": REPLAY_ABSOLUTE_TOLERANCES,
    }
    collection_digest = content_sha256(identity)
    directory = Path(output_dir) / "observability" / collection_digest[:16]
    arrays_path = directory / "annotations.npz"
    _write_immutable_bytes(arrays_path, arrays_bytes)
    manifest = {
        **identity,
        "collection_digest": collection_digest,
        "arrays": {
            "path": arrays_path.name,
            "sha256": arrays_sha256,
            "fields": {name: list(value.shape) for name, value in arrays.items()},
        },
        "validation": {
            "nominal_state_rows_replayed": 450,
            "recovery_state_rows_replayed": len(PHASE_WIDE_RECOVERY_ANCHORS),
            "maximum_absolute_replay_error": replay_maxima,
            "state_replay_within_fixed_tolerance": True,
            "labels_unchanged": True,
            "future_state_used": False,
        },
        "claim": "causal_privileged_observation_annotations_not_deployment_inputs",
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = directory / "manifest.json"
    write_immutable_json(manifest_path, manifest)
    return ObservabilityCollection(directory, manifest_path, arrays_path, collection_digest)


def load_observability_annotations(
    path: str | Path,
) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Load, hash-check, and split aligned nominal/recovery annotations."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    stated = manifest.get("content_sha256")
    body = dict(manifest)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError("observability manifest content hash mismatch")
    arrays_path = path.parent / manifest["arrays"]["path"]
    raw = arrays_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != manifest["arrays"]["sha256"]:
        raise ValueError("observability arrays hash mismatch")
    with np.load(io.BytesIO(raw), allow_pickle=False) as loaded:
        raw_arrays = {name: loaded[name].copy() for name in loaded.files}
    nominal = {
        name.removeprefix("nominal__"): value
        for name, value in raw_arrays.items() if name.startswith("nominal__")
    }
    recovery = {
        name.removeprefix("recovery__"): value
        for name, value in raw_arrays.items() if name.startswith("recovery__")
    }
    if any(value.shape[0] != 450 for value in nominal.values()):
        raise ValueError("observability nominal row count mismatch")
    if any(value.shape[0] != len(PHASE_WIDE_RECOVERY_ANCHORS) for value in recovery.values()):
        raise ValueError("observability recovery row count mismatch")
    required = {"contact_flags", "previous_contact_flags"} | {
        f"previous_{name}" for name in STATE_FIELDS
    }
    if set(nominal) != required or set(recovery) != required:
        raise ValueError("observability annotation field schema mismatch")
    return manifest, nominal, recovery


def _load_hashed_json(path: str | Path, *, label: str) -> dict[str, Any]:
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    stated = payload.get("content_sha256")
    body = dict(payload)
    body.pop("content_sha256", None)
    if stated != content_sha256(body):
        raise ValueError(f"{label} content hash mismatch")
    return payload


def build_observability_feature_report(
    oracle_manifest_path: str | Path,
    recovery_manifest_path: str | Path,
    observability_manifest_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Describe how the fixed anchors separate under each candidate schema."""
    from so_arm101_v2.learning.oracle_distillation import (
        OracleCloneKind,
        build_oracle_features,
        oracle_feature_schema,
    )

    oracle_manifest, nominal = load_oracle_demonstrations(oracle_manifest_path)
    recovery_manifest, recovery = load_oracle_recovery_examples(recovery_manifest_path)
    observability_manifest, nominal_obs, recovery_obs = load_observability_annotations(
        observability_manifest_path
    )
    if (
        recovery_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
        or observability_manifest.get("oracle_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
        or observability_manifest.get("recovery_manifest_content_sha256")
        != recovery_manifest["content_sha256"]
    ):
        raise ValueError("observability feature-report inputs disagree")

    kinds = (
        OracleCloneKind.PHASE_DYNAMICS,
        OracleCloneKind.PHASE_DYNAMICS_CONTACT,
        OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
    )
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    schemas: dict[str, list[str]] = {}
    for kind in kinds:
        obs_nominal = None if kind is OracleCloneKind.PHASE_DYNAMICS else nominal_obs
        obs_recovery = None if kind is OracleCloneKind.PHASE_DYNAMICS else recovery_obs
        nominal_features, mean, std = build_oracle_features(
            kind, nominal, observability=obs_nominal,
        )
        recovery_features, _, _ = build_oracle_features(
            kind, recovery, observability=obs_recovery,
            extras_mean=mean, extras_std=std,
        )
        feature_sets[kind.value] = (nominal_features, recovery_features)
        schemas[kind.value] = list(oracle_feature_schema(kind))

    records: list[dict[str, Any]] = []
    for position, anchor in enumerate(PHASE_WIDE_RECOVERY_ANCHORS):
        if int(recovery["action_index"][position]) != anchor.action_index:
            raise ValueError("recovery anchor order changed")
        separations: dict[str, Any] = {}
        for name, (nominal_features, recovery_features) in feature_sets.items():
            same_phase = float(np.linalg.norm(
                recovery_features[position] - nominal_features[anchor.action_index]
            ))
            distances = np.linalg.norm(
                nominal_features - recovery_features[position][None], axis=1
            )
            nearest = int(np.argmin(distances))
            separations[name] = {
                "same_phase_normalized_l2": same_phase,
                "nearest_nominal_row": nearest,
                "nearest_nominal_normalized_l2": float(distances[nearest]),
            }
        records.append({
            "anchor": anchor.name,
            "action_index": anchor.action_index,
            "nominal_contact_flags": nominal_obs["contact_flags"][anchor.action_index].tolist(),
            "recovery_contact_flags": recovery_obs["contact_flags"][position].tolist(),
            "separation": separations,
        })

    identity = {
        "schema_version": 1,
        "experiment": "bounded_observability_feature_telemetry_v1",
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "observability_manifest_content_sha256": observability_manifest["content_sha256"],
    }
    digest = content_sha256(identity)
    report = {
        **identity,
        "analysis_digest": digest,
        "feature_schemas": schemas,
        "anchors": records,
        "interpretation": (
            "descriptive normalized feature separation only; not proof that feature aliasing "
            "caused closed-loop failure"
        ),
        "blocking_gate": False,
    }
    report["content_sha256"] = content_sha256(report)
    path = Path(output_dir) / "observability_analyses" / digest[:16] / "report.json"
    write_immutable_json(path, report)
    return path


def _rollout_summary(item: Mapping[str, Any]) -> dict[str, Any]:
    telemetry = _load_hashed_json(item["telemetry_path"], label="rollout telemetry")
    events: set[str] = set()
    for row in telemetry.get("rows", []):
        events.update(row.get("pickup_events", []))
        events.update(row.get("placement_events", []))
    safety = {
        "clipping_frames": int(item["clipping_frames"]),
        "limiting_frames": int(item["limiting_frames"]),
        "nonfinite_frames": int(item["nonfinite_frames"]),
        "unsafe_contact_frames": int(item["unsafe_contact_frames"]),
    }
    milestone_order = (
        "reach", "first_contact", "bilateral_interior_contact", "strict_grasp_acquired",
        "lift_5mm", "lift_10mm", "lift_20mm", "success", "entered_placement_region",
        "cube_supported", "released", "settled", "retreated",
    )
    achieved = [name for name in milestone_order if name in events]
    return {
        "scenario_id": item["scenario_id"],
        "repeat": int(item["repeat"]),
        "success": bool(item["success"]),
        "invalidated": bool(item["invalidated"]),
        "failure_category": item["failure_category"],
        "milestones": achieved,
        "maximum_cube_height_gain_m": float(item["maximum_height_gain_m"]),
        "safety_counts": safety,
        "telemetry_content_sha256": telemetry["content_sha256"],
    }


def _evaluation_passed(payload: Mapping[str, Any]) -> bool:
    rollouts = payload.get("rollouts", [])
    return bool(
        payload.get("deterministic") is True
        and len(rollouts) == 3
        and all(
            item.get("scenario_id") == "nominal"
            and item.get("success") is True
            and item.get("invalidated") is False
            and int(item.get("clipping_frames", -1)) == 0
            and int(item.get("limiting_frames", -1)) == 0
            and int(item.get("nonfinite_frames", -1)) == 0
            and int(item.get("unsafe_contact_frames", -1)) == 0
            for item in rollouts
        )
    )


def resolve_bounded_observability_status(
    candidates: list[Mapping[str, Any]],
) -> str:
    """Validate the bounded branch trace and derive its one terminal status."""
    if len(candidates) != 3:
        raise ValueError("bounded observability gate requires exactly three candidate records")
    passed_kind: str | None = None
    passed_offline = False
    for candidate in candidates:
        state = candidate.get("state")
        kind = str(candidate.get("model_kind"))
        if passed_kind is not None:
            if state != "skipped_after_first_pass":
                raise ValueError("candidate ran after the first nominal pass")
            continue
        if state == "offline_failed":
            if candidate.get("evaluation") is not None:
                raise ValueError("offline-failing candidate reached closed-loop evaluation")
            continue
        if state == "nominal_failed":
            passed_offline = True
            if candidate.get("evaluation") is None:
                raise ValueError("offline-passing candidate lacks nominal evaluation")
            continue
        if state == "nominal_passed":
            passed_offline = True
            if candidate.get("evaluation") is None:
                raise ValueError("nominal pass lacks evaluation evidence")
            passed_kind = kind
            continue
        raise ValueError(f"invalid bounded candidate state {state!r}")
    if passed_kind is not None:
        return f"passed_{passed_kind}"
    return "closed_loop_not_resolved" if passed_offline else "blocked_offline"


def run_bounded_observability_gate(
    model_path: str | Path,
    oracle_manifest_path: str | Path,
    recovery_manifest_path: str | Path,
    observability_manifest_path: str | Path,
    baseline_checkpoint: str | Path,
    baseline_evaluation_path: str | Path,
    preflight_report_path: str | Path,
    output_dir: str | Path,
    *,
    record_video: bool = True,
    workers: int | None = None,
) -> ObservabilityGateResult:
    """Run at most three fixed candidates and stop at the first safe nominal 3/3."""
    from so_arm101_v2.learning.oracle_distillation import (
        OracleCloneKind,
        OracleDistillationConfig,
        distill_oracle_policy,
    )
    from .clone_policy import OracleCloneCheckpointPolicy
    from .rollout import evaluate_closed_loop

    model_path = Path(model_path).resolve()
    output_dir = Path(output_dir)
    oracle_manifest, _ = load_oracle_demonstrations(oracle_manifest_path)
    recovery_manifest, _ = load_oracle_recovery_examples(recovery_manifest_path)
    observability_manifest, _, _ = load_observability_annotations(observability_manifest_path)
    if (
        recovery_manifest.get("source_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
        or observability_manifest.get("oracle_manifest_content_sha256")
        != oracle_manifest["content_sha256"]
        or observability_manifest.get("recovery_manifest_content_sha256")
        != recovery_manifest["content_sha256"]
    ):
        raise ValueError("bounded gate manifests disagree")

    preflight = _load_hashed_json(preflight_report_path, label="preflight report")
    if (
        preflight.get("environment_proven") is not True
        or preflight.get("deterministic") is not True
        or preflight.get("suite", {}).get("suite_id") != "fixed_pick_place_v3"
    ):
        raise ValueError("bounded gate requires the passing deterministic v3 preflight")

    baseline_checkpoint = Path(baseline_checkpoint).resolve()
    baseline_probe = OracleCloneCheckpointPolicy(baseline_checkpoint)
    baseline_report = _load_hashed_json(
        baseline_checkpoint.with_name("report.json"), label="baseline training report"
    )
    baseline_recovery_weight = float(
        (baseline_report.get("recovery_metrics") or {}).get(
            "loss_weight", baseline_report.get("config", {}).get("recovery_loss_weight", 1.0)
        )
    )
    if (
        baseline_probe.kind.value != "phase_state"
        or baseline_probe.input_dim != 10
        or baseline_probe.hidden_width != 256
        or baseline_probe.seed != 101
        or baseline_probe.manifest_content_sha256 != oracle_manifest["content_sha256"]
        or baseline_report.get("recovery_augmentation", {}).get("manifest_content_sha256")
        != recovery_manifest["content_sha256"]
        or baseline_report.get("recovery_augmentation", {}).get("rows") != 8
        or baseline_recovery_weight != 1.0
    ):
        raise ValueError("baseline is not the established 450+8 equal-weight phase-state control")
    baseline_sha = hashlib.sha256(baseline_checkpoint.read_bytes()).hexdigest()
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
    if len(baseline_nominal) != 3:
        raise ValueError("baseline evaluation must contain exactly three nominal repeats")

    feature_report_path = build_observability_feature_report(
        oracle_manifest_path, recovery_manifest_path, observability_manifest_path,
        output_dir,
    )
    feature_report = _load_hashed_json(feature_report_path, label="feature telemetry report")
    identity = {
        "schema_version": 1,
        "experiment": "bounded_observability_decision_gate_v1",
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
        "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
        "observability_manifest_content_sha256": observability_manifest["content_sha256"],
        "baseline_checkpoint_sha256": baseline_sha,
        "baseline_training_report_content_sha256": baseline_report["content_sha256"],
        "baseline_evaluation_content_sha256": baseline_evaluation["content_sha256"],
        "preflight_report_content_sha256": preflight["content_sha256"],
        "feature_report_content_sha256": feature_report["content_sha256"],
        "candidate_order": [
            "phase_dynamics", "phase_dynamics_contact",
            "phase_dynamics_contact_history2",
        ],
        "fixed_training_config": {
            "seed": 101,
            "hidden_width": 256,
            "max_steps": 30_000,
            "learning_rate_schedule": "decay_10k_20k",
            "recovery_loss_weight": 1.0,
            "nominal_normalized_mse_threshold": 1e-6,
            "maximum_act_error_threshold": 0.01,
            "baseline_improvement_factor": 100.0,
        },
        "pass_rule": "three deterministic nominal successes with zero safety counts",
    }
    gate_digest = content_sha256(identity)
    directory = output_dir / "observability_gates" / gate_digest[:16]
    report_path = directory / "report.json"
    if report_path.exists():
        existing = _load_hashed_json(report_path, label="observability gate report")
        if any(existing.get(name) != value for name, value in identity.items()):
            raise FileExistsError(f"immutable observability gate differs: {directory}")
        return ObservabilityGateResult(directory, report_path, str(existing["status"]))

    suite = load_simulation_suite("fixed_pick_place_v3")
    nominal_suite = replace(
        suite,
        suite_id="fixed_pick_place_v3.nominal_observability_gate",
        scenarios=tuple(item for item in suite.scenarios if item.scenario_id == "nominal"),
        repeats=3,
    )
    candidates: list[dict[str, Any]] = []
    terminal_status: str | None = None
    candidate_kinds = (
        OracleCloneKind.PHASE_DYNAMICS,
        OracleCloneKind.PHASE_DYNAMICS_CONTACT,
        OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
    )
    for kind in candidate_kinds:
        if terminal_status is not None:
            candidates.append({"model_kind": kind.value, "state": "skipped_after_first_pass"})
            continue
        training = distill_oracle_policy(
            oracle_manifest_path,
            output_dir,
            kind=kind,
            config=OracleDistillationConfig(
                seed=101,
                hidden_width=256,
                max_steps=30_000,
                lr_schedule="decay_10k_20k",
                recovery_loss_weight=1.0,
            ),
            recovery_manifest_path=recovery_manifest_path,
            observability_manifest_path=(
                observability_manifest_path
                if kind in (
                    OracleCloneKind.PHASE_DYNAMICS_CONTACT,
                    OracleCloneKind.PHASE_DYNAMICS_CONTACT_HISTORY2,
                ) else None
            ),
        )
        training_report = _load_hashed_json(training.report_json, label="candidate training report")
        candidate: dict[str, Any] = {
            "model_kind": kind.value,
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
            },
            "evaluation": None,
        }
        if not training.passed:
            candidates.append(candidate)
            continue
        probe = OracleCloneCheckpointPolicy(training.checkpoint)
        evaluation_identity = content_sha256({
            "gate_digest": gate_digest,
            "model_kind": kind.value,
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "training_report_content_sha256": training_report["content_sha256"],
            "preflight_report_content_sha256": preflight["content_sha256"],
        })
        evaluation = evaluate_closed_loop(
            model_path,
            nominal_suite,
            {probe.policy_id: PolicySpec(kind="oracle_clone", checkpoint=str(training.checkpoint.resolve()))},
            output_dir / "observability_gate_evaluations" / evaluation_identity[:16],
            environment_proven=True,
            record_video=record_video,
            workers=workers,
            provenance={
                "gate_digest": gate_digest,
                "checkpoint_sha256": candidate["checkpoint_sha256"],
                "offline_report_content_sha256": training_report["content_sha256"],
                "oracle_manifest_content_sha256": oracle_manifest["content_sha256"],
                "recovery_manifest_content_sha256": recovery_manifest["content_sha256"],
                "observability_manifest_content_sha256": observability_manifest["content_sha256"],
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
        candidates.append(candidate)
        if passed:
            terminal_status = f"passed_{kind.value}"

    resolved_status = resolve_bounded_observability_status(candidates)
    if terminal_status is not None and terminal_status != resolved_status:
        raise RuntimeError("bounded gate branch trace disagrees with terminal status")
    terminal_status = resolved_status
    next_actions = {
        "passed_phase_dynamics": "broader_evaluation_of_phase_dynamics_only",
        "passed_phase_dynamics_contact": "broader_evaluation_and_deployable_contact_signal_design",
        "passed_phase_dynamics_contact_history2": "broader_evaluation_of_two_frame_policy_only",
        "blocked_offline": "end_tiny_policy_work_and_use_a_stronger_supervised_model",
        "closed_loop_not_resolved": "collect_complete_oracle_correction_trajectories_dagger_style",
    }
    report = {
        **identity,
        "gate_digest": gate_digest,
        "status": terminal_status,
        "next_action": next_actions[terminal_status],
        "baseline_nominal_rollouts": [_rollout_summary(item) for item in baseline_nominal],
        "candidates": candidates,
        "training_runs_started": sum(item["state"] != "skipped_after_first_pass" for item in candidates),
        "nominal_evaluations_started": sum(item.get("evaluation") is not None for item in candidates),
        "partial_milestones_are_non_promoting": True,
        "broader_evaluation_run": False,
        "anchor_handoff_evaluation_run": False,
        "claim": "bounded_privileged_observability_decision_not_deployment_success",
    }
    report["content_sha256"] = content_sha256(report)
    write_immutable_json(report_path, report)
    return ObservabilityGateResult(directory, report_path, terminal_status)


__all__ = [
    "CONTACT_FIELDS", "STATE_FIELDS", "ObservabilityCollection",
    "ObservabilityGateResult", "capture_observability_annotations",
    "load_observability_annotations", "build_observability_feature_report",
    "run_bounded_observability_gate", "resolve_bounded_observability_status",
]
