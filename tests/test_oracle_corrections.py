from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.learning.oracle_distillation import (
    OracleCloneKind,
    OracleDistillationConfig,
    build_oracle_clone_model,
    distill_oracle_policy,
    oracle_feature_schema,
)
from so_arm101_v2.simulation import MujocoTaskAdapter, load_simulation_suite
from so_arm101_v2.simulation.clone_policy import OracleCloneCheckpointPolicy
from so_arm101_v2.simulation.correction import (
    DAGGER_CORRECTION_SITES,
    CorrectionSite,
    NonPromotableDiagnosticClonePolicy,
    load_oracle_corrections,
    resolve_correction_gate_status,
    resolve_correction_probe_status,
    run_correction_gate,
)
from so_arm101_v2.simulation.privileged import (
    PrivilegedStagedController,
    compress_boundaries,
)

from conftest import REPOSITORY_ROOT


PICKUP_BOUNDARIES = (0, 70, 110, 145, 175, 205, 255, 285, 315, 355, 450)
PICK_PLACE_BOUNDARIES = (
    0, 70, 110, 145, 175, 205, 255, 285, 315, 350, 365, 386, 403, 419, 424, 450,
)


def _write_tiny_oracle_manifest(directory: Path, rows: int = 4) -> tuple[Path, dict]:
    arrays = {
        "scenario_index": np.zeros(rows, dtype=np.int32),
        "action_index": np.arange(rows, dtype=np.int32),
        "progress": np.linspace(0, 1, rows, dtype=np.float32),
        "current_act": np.zeros((rows, 6), dtype=np.float32),
        "robot_qvel": np.zeros((rows, 6), dtype=np.float32),
        "cube_position": np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) * 0.01,
        "cube_quaternion_wxyz": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (rows, 1)),
        "cube_linear_velocity": np.zeros((rows, 3), dtype=np.float32),
        "cube_angular_velocity": np.zeros((rows, 3), dtype=np.float32),
        "executed_act": np.full((rows, 6), 0.002, dtype=np.float32),
        "executed_delta_act": np.full((rows, 6), 0.002, dtype=np.float32),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = stream.getvalue()
    arrays_path = directory / "demonstrations.npz"
    arrays_path.write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "collection_digest": "c" * 64,
        "teacher_horizon": rows,
        "arrays": {
            "path": arrays_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": rows,
        },
        "episodes": [{
            "scenario_id": "nominal", "rows": rows,
            "waypoint_boundaries": [0, 2, rows],
        }],
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def _write_correction_manifest(
    directory: Path,
    *,
    source_sha: str,
    start: int = 448,
    length: int = 3,
    inducing_checkpoint_sha256: str = "f" * 64,
    tamper_progress: bool = False,
    tamper_contiguity: bool = False,
) -> tuple[Path, dict]:
    action_index = np.arange(start, start + length, dtype=np.int32)
    if tamper_contiguity:
        action_index = action_index[::-1].copy()
    progress = (
        np.minimum(action_index, 449).astype(np.float64) / 449.0
    ).astype(np.float32)
    if tamper_progress:
        progress = np.zeros(length, dtype=np.float32)
    arrays = {
        "action_index": action_index,
        "progress": progress,
        "current_act": np.zeros((length, 6), dtype=np.float32),
        "robot_qvel": np.zeros((length, 6), dtype=np.float32),
        "cube_position": np.full((length, 3), 0.5, dtype=np.float32),
        "cube_quaternion_wxyz": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (length, 1)),
        "cube_linear_velocity": np.zeros((length, 3), dtype=np.float32),
        "cube_angular_velocity": np.zeros((length, 3), dtype=np.float32),
        "executed_act": np.zeros((length, 6), dtype=np.float32),
        "executed_delta_act": np.zeros((length, 6), dtype=np.float32),
        "site_index": np.full(length, 10, dtype=np.int32),
        "scenario_index": np.zeros(length, dtype=np.int32),
    }
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = stream.getvalue()
    arrays_path = directory / "correction_trajectories.npz"
    arrays_path.write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "experiment": "dagger_oracle_correction_trajectories_v1",
        "source_manifest_content_sha256": source_sha,
        "collection_digest": "d" * 64,
        "inducing_checkpoint_sha256": inducing_checkpoint_sha256,
        "progress_rule": "deployment_clock",
        "arrays": {
            "path": arrays_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": length,
        },
        "episodes": [
            {"name": "release", "action_index": start, "rows": length, "accepted": True},
            {"name": "placement", "action_index": 386, "accepted": False},
        ],
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def _write_scheduled_checkpoint_pair(
    directory: Path,
    *,
    hidden_width: int = 256,
    manifest_sha: str = "a" * 64,
    correction_augmentation: dict | None = None,
    checkpoint_correction_augmentation: dict | str | None = "same",
    prefix_parity: dict | None = None,
    passed: bool = True,
    eligible: bool = True,
) -> Path:
    model = build_oracle_clone_model(10, hidden_width)
    rows = list(range(450))
    report = {
        "schema_version": 1,
        "model_kind": "phase_state",
        "passed": passed,
        "closed_loop_eligible": eligible,
        "training_row_mode": "full",
        "training_row_indices": rows,
        "hidden_width": hidden_width,
        "manifest_content_sha256": manifest_sha,
        "learning_rate_schedule": "decay_10k_20k",
        "prefix_parity": prefix_parity or {
            "required": True, "applicable": True, "passed": True,
            "reference_content_sha256": "b" * 64,
        },
    }
    if correction_augmentation is not None:
        report["correction_augmentation"] = correction_augmentation
    report["content_sha256"] = content_sha256(report)
    payload = {
        "schema_version": 1,
        "model_kind": "phase_state",
        "input_dim": 10,
        "hidden_width": hidden_width,
        "extras_mean": torch.zeros(3),
        "extras_std": torch.ones(3),
        "maximum_act_delta_per_step": torch.ones(6),
        "teacher_horizon": 450,
        "manifest_content_sha256": manifest_sha,
        "report_content_sha256": report["content_sha256"],
        "training_row_mode": "full",
        "training_row_indices": rows,
        "closed_loop_eligible": True,
        "config": {"seed": 101},
        "state_dict": model.state_dict(),
    }
    if correction_augmentation is not None:
        if checkpoint_correction_augmentation == "same":
            payload["correction_augmentation"] = correction_augmentation
        else:
            payload["correction_augmentation"] = checkpoint_correction_augmentation
    checkpoint = directory / "model.pt"
    torch.save(payload, checkpoint)
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return checkpoint


def test_correction_sites_are_fixed_ordered_and_span_task() -> None:
    names = [site.name for site in DAGGER_CORRECTION_SITES]
    indices = [site.action_index for site in DAGGER_CORRECTION_SITES]
    assert len(DAGGER_CORRECTION_SITES) == 11
    assert indices == sorted(indices)
    assert set(indices) == {31, 70, 74, 194, 195, 205, 255, 315, 365, 386, 419}
    assert {"approach", "first_contact", "seating", "closure", "lift",
            "transport", "placement", "release"} < set(names)
    assert all(1 <= index < 450 for index in indices)
    with pytest.raises(ValueError, match="invalid correction site"):
        CorrectionSite("bad", 450, "outside horizon")


def test_compress_boundaries_identity_monotonic_and_budget_rejection() -> None:
    assert compress_boundaries(PICKUP_BOUNDARIES, 0) == PICKUP_BOUNDARIES
    assert compress_boundaries(PICK_PLACE_BOUNDARIES, 0) == PICK_PLACE_BOUNDARIES
    compressed = compress_boundaries(PICK_PLACE_BOUNDARIES, 70)
    assert compressed[0] == 70 and compressed[-1] == 450
    assert all(later > earlier for earlier, later in zip(compressed, compressed[1:]))
    with pytest.raises(RuntimeError, match="insufficient remaining correction budget"):
        compress_boundaries(PICK_PLACE_BOUNDARIES, 419)
    with pytest.raises(ValueError, match="inside the horizon"):
        compress_boundaries(PICK_PLACE_BOUNDARIES, 450)
    with pytest.raises(ValueError, match="start at 0"):
        compress_boundaries((10, 450), 0)


def test_reset_from_state_preserves_deployment_clock() -> None:
    model = REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"
    scenario = next(
        item for item in load_simulation_suite("fixed_pick_place_v3").scenarios
        if item.scenario_id == "nominal"
    )
    adapter = MujocoTaskAdapter(model)
    try:
        adapter.reset(scenario)
        reference = PrivilegedStagedController()
        reference.reset(adapter)
        adapter.reset(scenario)
        controller = PrivilegedStagedController()
        controller.reset_from_state(adapter, start_index=70)
        assert controller.action_index == 70
        assert controller.boundaries == compress_boundaries(reference.boundaries, 70)
        assert controller.boundaries[0] == 70 and controller.boundaries[-1] == 450
        command = controller.predict(np.empty((0,), dtype=np.uint8), adapter.current_act(), adapter)
        assert command.shape == (6,) and controller.action_index == 71
    finally:
        adapter.close()


def test_correction_manifest_and_arrays_are_hash_validated(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    manifest_path, manifest = _write_correction_manifest(valid, source_sha="a" * 64)
    loaded_manifest, arrays = load_oracle_corrections(manifest_path)
    assert loaded_manifest["arrays"]["rows"] == 3
    np.testing.assert_array_equal(arrays["action_index"], [448, 449, 450])
    # Rows past the global horizon saturate at 1.0 exactly as the deployed
    # policy's progress clock does.
    np.testing.assert_array_equal(
        arrays["progress"],
        np.asarray([448 / 449.0, 1.0, 1.0], dtype=np.float32),
    )
    (valid / "correction_trajectories.npz").write_bytes(
        (valid / "correction_trajectories.npz").read_bytes() + b"tamper"
    )
    with pytest.raises(ValueError, match="arrays hash mismatch"):
        load_oracle_corrections(manifest_path)

    contiguity = tmp_path / "contiguity"
    contiguity.mkdir()
    broken_path, _ = _write_correction_manifest(
        contiguity, source_sha="a" * 64, tamper_contiguity=True,
    )
    with pytest.raises(ValueError, match="contiguous sub-episode"):
        load_oracle_corrections(broken_path)


def test_correction_rows_preserve_deployment_progress_clock(tmp_path: Path) -> None:
    manifest_path, _ = _write_correction_manifest(
        tmp_path, source_sha="a" * 64, tamper_progress=True,
    )
    with pytest.raises(ValueError, match="deployment progress clock"):
        load_oracle_corrections(manifest_path)


def test_correction_and_recovery_manifests_are_mutually_exclusive(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        distill_oracle_policy(
            tmp_path / "manifest.json",
            tmp_path / "output",
            kind=OracleCloneKind.PHASE_STATE,
            recovery_manifest_path=tmp_path / "recovery.json",
            correction_manifest_path=tmp_path / "corrections.json",
        )


def test_correction_identity_keys_absent_preserve_legacy_digests(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = distill_oracle_policy(
        manifest_path,
        tmp_path / "output",
        kind=OracleCloneKind.PHASE_STATE,
        config=OracleDistillationConfig(max_steps=2),
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert "correction_manifest_content_sha256" not in report
    assert "correction_augmentation" not in report
    assert "correction_metrics" not in report
    identity_keys = (
        "manifest_content_sha256", "collection_digest", "model_kind", "optimizer",
        "source_rows", "training_row_indices", "config",
        "prefix_reference_content_sha256", "recovery_manifest_content_sha256",
        "recovery_collection_digest", "recovery_rows",
        "observability_manifest_content_sha256", "observability_collection_digest",
    )
    identity = {name: report[name] for name in identity_keys}
    assert content_sha256(identity) == report["run_digest"]


def test_correction_augmented_training_uses_nominal_statistics_and_new_identity(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_tiny_oracle_manifest(tmp_path)
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    correction_path, correction_manifest = _write_correction_manifest(
        corrections, source_sha=manifest["content_sha256"],
    )
    result = distill_oracle_policy(
        manifest_path,
        tmp_path / "output",
        kind=OracleCloneKind.PHASE_STATE,
        config=OracleDistillationConfig(max_steps=2),
        correction_manifest_path=correction_path,
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert report["rows"] == 7
    assert report["correction_rows"] == 3
    assert report["recovery_metrics"] is None
    assert report["correction_metrics"]["rows"] == 3
    assert report["correction_metrics"]["loss_weight"] == 1.0
    assert report["correction_augmentation"]["manifest_content_sha256"] == (
        correction_manifest["content_sha256"]
    )
    assert report["correction_augmentation"]["sites"] == [
        {"name": "release", "action_index": 448, "rows": 3}
    ]
    assert report["prefix_parity"]["applicable"] is False
    assert "correction rows" in report["prefix_parity"]["reason"]
    assert report["weighted_objective"]["denominator"] == 7
    assert "correction_rows" in report["weighted_objective"]["formula"]
    assert report["row_scenarios"][-3:] == [
        "correction:release:448", "correction:release:449", "correction:release:450",
    ]
    # Normalization must come from the nominal rows alone, not the concatenation.
    nominal_cube = np.arange(4 * 3, dtype=np.float32).reshape(4, 3) * 0.01
    np.testing.assert_allclose(
        np.asarray(report["normalization"]["extras_mean"], dtype=np.float32),
        nominal_cube.mean(axis=0, dtype=np.float64).astype(np.float32),
    )
    identity_keys = (
        "manifest_content_sha256", "collection_digest", "model_kind", "optimizer",
        "source_rows", "training_row_indices", "config",
        "prefix_reference_content_sha256", "recovery_manifest_content_sha256",
        "recovery_collection_digest", "recovery_rows",
        "observability_manifest_content_sha256", "observability_collection_digest",
        "correction_manifest_content_sha256", "correction_collection_digest",
        "correction_rows",
    )
    identity = {name: report[name] for name in identity_keys}
    assert content_sha256(identity) == report["run_digest"]
    with pytest.raises(ValueError, match="loss weight 1.0"):
        distill_oracle_policy(
            manifest_path,
            tmp_path / "other",
            kind=OracleCloneKind.PHASE_STATE,
            config=OracleDistillationConfig(max_steps=2, recovery_loss_weight=0.5),
            correction_manifest_path=correction_path,
        )


def test_correction_gate_branch_trace_resolution() -> None:
    assert resolve_correction_gate_status(
        {"state": "offline_failed", "evaluation": None}
    ) == "blocked_offline"
    assert resolve_correction_gate_status(
        {"state": "nominal_failed", "evaluation": {"passed": False}}
    ) == "closed_loop_not_resolved"
    assert resolve_correction_gate_status(
        {"state": "nominal_passed", "evaluation": {"passed": True}}
    ) == "passed"
    with pytest.raises(ValueError, match="reached closed-loop"):
        resolve_correction_gate_status(
            {"state": "offline_failed", "evaluation": {"passed": False}}
        )
    with pytest.raises(ValueError, match="lacks nominal evaluation"):
        resolve_correction_gate_status({"state": "nominal_failed", "evaluation": None})
    with pytest.raises(ValueError, match="invalid correction candidate state"):
        resolve_correction_gate_status({"state": "skipped_after_first_pass"})


def test_correction_gate_rejects_mismatched_baseline_or_inducing_checkpoint(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_tiny_oracle_manifest(tmp_path)
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    correction_path, _ = _write_correction_manifest(
        corrections, source_sha=manifest["content_sha256"],
        inducing_checkpoint_sha256="f" * 64,
    )
    preflight = {
        "environment_proven": True,
        "deterministic": True,
        "suite": {"suite_id": "fixed_pick_place_v3"},
    }
    preflight["content_sha256"] = content_sha256(preflight)
    preflight_path = tmp_path / "preflight.json"
    preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    model_path = tmp_path / "scene.xml"
    model_path.write_text("<mujoco/>", encoding="utf-8")

    narrow = tmp_path / "narrow"
    narrow.mkdir()
    narrow_checkpoint = _write_scheduled_checkpoint_pair(
        narrow, hidden_width=128, manifest_sha=manifest["content_sha256"],
    )
    with pytest.raises(ValueError, match="nominal-only width-256"):
        run_correction_gate(
            model_path, manifest_path, correction_path, narrow_checkpoint,
            tmp_path / "missing-evaluation.json", tmp_path / "missing-parity.json",
            preflight_path, tmp_path / "gate",
        )

    wide = tmp_path / "wide"
    wide.mkdir()
    wide_checkpoint = _write_scheduled_checkpoint_pair(
        wide, hidden_width=256, manifest_sha=manifest["content_sha256"],
    )
    with pytest.raises(ValueError, match="not induced by the baseline"):
        run_correction_gate(
            model_path, manifest_path, correction_path, wide_checkpoint,
            tmp_path / "missing-evaluation.json", tmp_path / "missing-parity.json",
            preflight_path, tmp_path / "gate",
        )


def test_correction_probe_resolver_applies_preregistered_rule() -> None:
    baseline = [
        {"milestones": ["reach", "first_contact", "released", "retreated"],
         "maximum_cube_height_gain_m": 0.000159,
         "safety_counts": {"clipping_frames": 0, "limiting_frames": 0,
                           "nonfinite_frames": 0, "unsafe_contact_frames": 0}},
    ] * 3
    unchanged = [
        {"milestones": ["reach", "first_contact"],
         "maximum_cube_height_gain_m": 0.002,
         "safety_counts": {"clipping_frames": 0, "limiting_frames": 0,
                           "nonfinite_frames": 0, "unsafe_contact_frames": 0}},
    ] * 3
    resolution = resolve_correction_probe_status(unchanged, baseline)
    assert resolution["status"] == "behavior_unchanged"
    assert resolution["new_milestones"] == []
    assert resolution["safety_regressed"] is False

    moved_by_milestone = [dict(unchanged[0], milestones=["reach", "lift_5mm"])] * 3
    resolution = resolve_correction_probe_status(moved_by_milestone, baseline)
    assert resolution["status"] == "behavior_moved"
    assert resolution["new_milestones"] == ["lift_5mm"]

    moved_by_height = [dict(unchanged[0], maximum_cube_height_gain_m=0.0062)] * 3
    assert resolve_correction_probe_status(moved_by_height, baseline)["status"] == "behavior_moved"

    regressed = [dict(
        unchanged[0],
        safety_counts={"clipping_frames": 2, "limiting_frames": 0,
                       "nonfinite_frames": 0, "unsafe_contact_frames": 0},
    )] * 3
    assert resolve_correction_probe_status(regressed, baseline)["safety_regressed"] is True
    with pytest.raises(ValueError, match="requires probe and baseline"):
        resolve_correction_probe_status([], baseline)


def test_diagnostic_probe_policy_loads_blocked_correction_checkpoint_only(
    tmp_path: Path,
) -> None:
    augmentation = {
        "manifest_content_sha256": "d" * 64,
        "collection_digest": "e" * 64,
        "rows": 4818,
        "sites": [{"name": "release", "action_index": 419, "rows": 438}],
        "progress_rule": "deployment_clock_saturating",
        "normalization": "unchanged_nominal_statistics",
    }
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    checkpoint = _write_scheduled_checkpoint_pair(
        blocked, correction_augmentation=augmentation,
        passed=False, eligible=False,
    )
    with pytest.raises(ValueError, match="eligible"):
        OracleCloneCheckpointPolicy(checkpoint)
    probe = NonPromotableDiagnosticClonePolicy(checkpoint)
    assert probe.policy_id == "phase_state_corrections.diagnostic"
    assert probe.input_dim == 10 and probe.teacher_horizon == 450

    plain = tmp_path / "plain"
    plain.mkdir()
    uncorrected = _write_scheduled_checkpoint_pair(plain, passed=False, eligible=False)
    with pytest.raises(ValueError, match="correction-augmented"):
        NonPromotableDiagnosticClonePolicy(uncorrected)

    report_path = blocked / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["passed"] = True
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        NonPromotableDiagnosticClonePolicy(checkpoint)


def test_checkpoint_promotion_gate_accepts_correction_augmented_scheduled_run(
    tmp_path: Path,
) -> None:
    augmentation = {
        "manifest_content_sha256": "d" * 64,
        "collection_digest": "e" * 64,
        "rows": 31,
        "sites": [{"name": "release", "action_index": 419, "rows": 31}],
        "progress_rule": "deployment_clock",
        "normalization": "unchanged_nominal_statistics",
    }
    parity = {
        "required": False, "applicable": False, "passed": None,
        "reference_content_sha256": "b" * 64,
    }
    valid = tmp_path / "valid"
    valid.mkdir()
    checkpoint = _write_scheduled_checkpoint_pair(
        valid, correction_augmentation=augmentation, prefix_parity=parity,
    )
    policy = OracleCloneCheckpointPolicy(checkpoint)
    assert policy.kind is OracleCloneKind.PHASE_STATE
    assert policy.feature_schema == oracle_feature_schema(OracleCloneKind.PHASE_STATE)

    disagree = tmp_path / "disagree"
    disagree.mkdir()
    broken = _write_scheduled_checkpoint_pair(
        disagree, correction_augmentation=augmentation,
        checkpoint_correction_augmentation={**augmentation, "rows": 30},
        prefix_parity=parity,
    )
    with pytest.raises(ValueError, match="correction metadata"):
        OracleCloneCheckpointPolicy(broken)

    missing_reference = tmp_path / "missing-reference"
    missing_reference.mkdir()
    ineligible = _write_scheduled_checkpoint_pair(
        missing_reference, correction_augmentation=augmentation,
        prefix_parity={**parity, "reference_content_sha256": None},
    )
    with pytest.raises(ValueError, match="eligible"):
        OracleCloneCheckpointPolicy(ineligible)
