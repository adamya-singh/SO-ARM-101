from __future__ import annotations

from pathlib import Path
import json
import hashlib
import io

import numpy as np
import pytest
import torch

from so_arm101_v2.learning import (
    ORACLE_MEMORIZATION32_ROWS,
    OracleCloneKind,
    OracleDistillationConfig,
    analyze_oracle_residuals,
    build_oracle_clone_model,
    build_oracle_features,
    count_oracle_command_safety_violations,
    oracle_training_row_indices,
    oracle_learning_rate,
)
from so_arm101_v2.simulation import MujocoTaskAdapter, PrivilegedStateSnapshot, load_simulation_suite
from so_arm101_v2.simulation.oracle import load_oracle_demonstrations
from so_arm101_v2.simulation.recovery import (
    PHASE_WIDE_RECOVERY_ANCHORS,
    load_oracle_recovery_examples,
)
from so_arm101_v2.simulation.clone_policy import OracleCloneCheckpointPolicy
from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.contracts import mujoco_qpos_to_act
from so_arm101_v2.learning.oracle_distillation import _validate_prefix_reference

from conftest import REPOSITORY_ROOT


def _arrays(rows: int = 4) -> dict[str, np.ndarray]:
    return {
        "current_act": np.zeros((rows, 6), dtype=np.float32),
        "robot_qvel": np.arange(rows * 6, dtype=np.float32).reshape(rows, 6),
        "cube_position": np.arange(rows * 3, dtype=np.float32).reshape(rows, 3) * 0.01,
        "cube_quaternion_wxyz": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (rows, 1)),
        "cube_linear_velocity": np.zeros((rows, 3), dtype=np.float32),
        "cube_angular_velocity": np.zeros((rows, 3), dtype=np.float32),
        "progress": np.linspace(0, 1, rows, dtype=np.float32),
    }


def test_oracle_models_initialize_to_hold_and_keep_samples_independent() -> None:
    torch.manual_seed(101)
    for input_dim, hidden_width in ((10, 128), (25, 256)):
        model = build_oracle_clone_model(input_dim, hidden_width)
        features = torch.rand(3, input_dim)
        with torch.inference_mode():
            batch = model(features)
            alone = model(features[:1])
        torch.testing.assert_close(batch, torch.zeros_like(batch))
        torch.testing.assert_close(batch[:1], alone)
        assert batch.shape == (3, 6)
        assert model.network[0].out_features == hidden_width


def test_oracle_config_restricts_capacity_and_row_modes() -> None:
    assert OracleDistillationConfig(hidden_width=128, training_rows="full").hidden_width == 128
    assert OracleDistillationConfig(hidden_width=256, training_rows="memorization32").hidden_width == 256
    with pytest.raises(ValueError, match="hidden_width"):
        OracleDistillationConfig(hidden_width=129)
    with pytest.raises(ValueError, match="training_rows"):
        OracleDistillationConfig(training_rows="random")
    with pytest.raises(ValueError, match="hidden_width"):
        build_oracle_clone_model(10, 512)
    scheduled = OracleDistillationConfig(
        hidden_width=256, max_steps=30_000, lr_schedule="decay_10k_20k"
    )
    assert scheduled.lr_schedule == "decay_10k_20k"
    with pytest.raises(ValueError, match="requires seed 101"):
        OracleDistillationConfig(
            hidden_width=128, max_steps=30_000, lr_schedule="decay_10k_20k"
        )
    with pytest.raises(ValueError, match="schedule"):
        OracleDistillationConfig(lr_schedule="cosine")


def test_fixed_oracle_memorization_rows_and_full_selection() -> None:
    selected = oracle_training_row_indices("memorization32", 450)
    np.testing.assert_array_equal(selected, np.asarray(ORACLE_MEMORIZATION32_ROWS))
    assert selected.shape == (32,)
    np.testing.assert_array_equal(oracle_training_row_indices("full", 4), np.arange(4))
    with pytest.raises(ValueError, match="450-row"):
        oracle_training_row_indices("memorization32", 449)


def test_nested_oracle_memorization_rungs_are_exact_and_deterministic() -> None:
    expected_hashes = {
        64: "5e307fc0a02296e1d460850266477039957ce2450e72005fa254c4871e2f2d86",
        128: "e36851042ac2a44bd9bc4d386f76eb5f5d35fc025f2ca5c5487fe739f3696412",
        256: "a26ee9f0ce8bedafed78a383b75bab4623bbda2907aa204b04e490e424442d81",
    }
    previous = set(ORACLE_MEMORIZATION32_ROWS)
    for count, expected_hash in expected_hashes.items():
        first = oracle_training_row_indices(f"memorization{count}", 450)
        second = oracle_training_row_indices(f"memorization{count}", 450)
        assert len(first) == count
        assert previous < set(first.tolist())
        np.testing.assert_array_equal(first, second)
        encoded = json.dumps(first.tolist(), separators=(",", ":")).encode()
        assert hashlib.sha256(encoded).hexdigest() == expected_hash
        previous = set(first.tolist())


def test_oracle_learning_rate_boundaries_are_one_based_and_exact() -> None:
    assert oracle_learning_rate("fixed", 1) == 1e-3
    assert oracle_learning_rate("fixed", 30_000) == 1e-3
    assert oracle_learning_rate("decay_10k_20k", 10_000) == 1e-3
    assert oracle_learning_rate("decay_10k_20k", 10_001) == 1e-4
    assert oracle_learning_rate("decay_10k_20k", 20_000) == 1e-4
    assert oracle_learning_rate("decay_10k_20k", 20_001) == 1e-5
    with pytest.raises(ValueError, match="positive"):
        oracle_learning_rate("fixed", 0)


def test_prefix_reference_validation_rejects_changed_or_tampered_reports(
    tmp_path: Path,
) -> None:
    manifest = {"content_sha256": "a" * 64, "collection_digest": "b" * 64}
    config = OracleDistillationConfig(
        hidden_width=256, max_steps=30_000, lr_schedule="decay_10k_20k"
    )
    report = {
        "manifest_content_sha256": manifest["content_sha256"],
        "collection_digest": manifest["collection_digest"],
        "model_kind": "phase_state",
        "source_rows": 4,
        "training_row_indices": [0, 1, 2, 3],
        "hidden_width": 256,
        "steps": 10_000,
        "loss_trace": [{"step": 1, "normalized_mse": 0.1}],
        "config": {
            "seed": 101,
            "learning_rate": 1e-3,
            "max_steps": 10_000,
            "normalized_mse_threshold": 1e-6,
            "max_act_error_threshold": 0.01,
            "baseline_improvement_factor": 100.0,
            "hidden_width": 256,
            "training_rows": "full",
            "lr_schedule": "fixed",
        },
    }
    report["content_sha256"] = content_sha256(report)
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    validated = _validate_prefix_reference(
        path,
        manifest=manifest,
        kind=OracleCloneKind.PHASE_STATE,
        config=config,
        source_rows=4,
        row_indices=np.arange(4),
    )
    assert validated["content_sha256"] == report["content_sha256"]

    changed = dict(report)
    changed["config"] = {**report["config"], "seed": 202}
    changed.pop("content_sha256")
    changed["content_sha256"] = content_sha256(changed)
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="seed mismatch"):
        _validate_prefix_reference(
            path,
            manifest=manifest,
            kind=OracleCloneKind.PHASE_STATE,
            config=config,
            source_rows=4,
            row_indices=np.arange(4),
        )
    changed["content_sha256"] = "0" * 64
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        _validate_prefix_reference(
            path,
            manifest=manifest,
            kind=OracleCloneKind.PHASE_STATE,
            config=config,
            source_rows=4,
            row_indices=np.arange(4),
        )


def test_residual_analysis_validates_hashes_stages_and_conflicts(tmp_path: Path) -> None:
    rows = 4
    arrays = _arrays(rows)
    arrays["scenario_index"] = np.zeros(rows, dtype=np.int32)
    arrays["action_index"] = np.arange(rows, dtype=np.int32)
    arrays["progress"] = np.array([0.0, 0.0, 0.75, 1.0], dtype=np.float32)
    arrays["cube_position"][1] = arrays["cube_position"][0]
    arrays["executed_delta_act"] = np.zeros((rows, 6), dtype=np.float32)
    arrays["executed_delta_act"][1, 0] = 0.02
    arrays["executed_act"] = arrays["current_act"] + arrays["executed_delta_act"]
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    arrays_bytes = stream.getvalue()
    arrays_path = tmp_path / "demonstrations.npz"
    arrays_path.write_bytes(arrays_bytes)
    manifest = {
        "arrays": {
            "path": arrays_path.name,
            "sha256": hashlib.sha256(arrays_bytes).hexdigest(),
            "rows": rows,
        },
        "collection_digest": "c" * 64,
        "teacher_horizon": rows,
        "episodes": [{
            "scenario_id": "nominal", "rows": rows,
            "waypoint_boundaries": [0, 2, 4],
        }],
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    model = build_oracle_clone_model(10, 256)
    training_report = {
        "schema_version": 1,
        "model_kind": "phase_state",
        "manifest_content_sha256": manifest["content_sha256"],
        "hidden_width": 256,
        "training_row_mode": "full",
        "training_row_indices": list(range(rows)),
        "source_rows": rows,
    }
    training_report["content_sha256"] = content_sha256(training_report)
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(training_report), encoding="utf-8")
    checkpoint_path = tmp_path / "model.pt"
    torch.save({
        "schema_version": 1,
        "model_kind": "phase_state",
        "manifest_content_sha256": manifest["content_sha256"],
        "report_content_sha256": training_report["content_sha256"],
        "hidden_width": 256,
        "training_row_mode": "full",
        "training_row_indices": list(range(rows)),
        "source_rows": rows,
        "input_dim": 10,
        "extras_mean": torch.from_numpy(arrays["cube_position"].mean(axis=0)),
        "extras_std": torch.from_numpy(
            np.maximum(arrays["cube_position"].std(axis=0), 1e-6)
        ),
        "maximum_act_delta_per_step": torch.ones(6),
        "state_dict": model.state_dict(),
    }, checkpoint_path)
    result = analyze_oracle_residuals(manifest_path, checkpoint_path, tmp_path / "output")
    analysis = json.loads(result.report_json.read_text())
    assert [item["rows"] for item in analysis["per_stage"]] == [2, 2]
    assert analysis["possible_label_conflicts"] == [{
        "feature_l2_distance": 0.0,
        "source_row_a": 0,
        "source_row_b": 1,
        "target_delta_linf_difference": pytest.approx(0.02),
    }]
    assert analyze_oracle_residuals(
        manifest_path, checkpoint_path, tmp_path / "output"
    ).report_json == result.report_json
    training_report["content_sha256"] = "0" * 64
    report_path.write_text(json.dumps(training_report), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        analyze_oracle_residuals(manifest_path, checkpoint_path, tmp_path / "other")


def test_oracle_feature_schemas_and_normalization_replay() -> None:
    arrays = _arrays()
    phase, phase_mean, phase_std = build_oracle_features(OracleCloneKind.PHASE_STATE, arrays)
    feedback, mean, std = build_oracle_features(OracleCloneKind.FEEDBACK_STATE, arrays)
    assert phase.shape == (4, 10)
    assert feedback.shape == (4, 25)
    replay, replay_mean, replay_std = build_oracle_features(
        OracleCloneKind.FEEDBACK_STATE, arrays, extras_mean=mean, extras_std=std
    )
    np.testing.assert_array_equal(replay, feedback)
    np.testing.assert_array_equal(replay_mean, mean)
    np.testing.assert_array_equal(replay_std, std)
    assert np.all(phase_std >= 1e-6) and np.all(std >= 1e-6)
    assert phase_mean.shape == (3,)

    # Selection happens after normalization: subset rows retain full-source statistics.
    larger = _arrays(450)
    full, full_mean, full_std = build_oracle_features(OracleCloneKind.PHASE_STATE, larger)
    selected = oracle_training_row_indices("memorization32", len(full))
    subset_direct, subset_mean, subset_std = build_oracle_features(
        OracleCloneKind.PHASE_STATE,
        {name: value[selected] for name, value in larger.items()},
    )
    assert not np.array_equal(full_mean, subset_mean) or not np.array_equal(full_std, subset_std)
    replay, _, _ = build_oracle_features(
        OracleCloneKind.PHASE_STATE,
        {name: value[selected] for name, value in larger.items()},
        extras_mean=full_mean,
        extras_std=full_std,
    )
    np.testing.assert_array_equal(replay, full[selected])
    assert not np.array_equal(subset_direct, replay)


def test_phase_wide_recovery_anchor_contract_is_small_and_spans_task() -> None:
    assert [item.name for item in PHASE_WIDE_RECOVERY_ANCHORS] == [
        "approach", "first_contact", "seating", "closure",
        "lift", "transport", "placement", "release",
    ]
    assert len(PHASE_WIDE_RECOVERY_ANCHORS) == 8
    assert [item.action_index for item in PHASE_WIDE_RECOVERY_ANCHORS] == sorted(
        item.action_index for item in PHASE_WIDE_RECOVERY_ANCHORS
    )
    assert all(max(abs(value) for value in item.perturbation_act) == pytest.approx(0.01)
               for item in PHASE_WIDE_RECOVERY_ANCHORS)


def test_recovery_manifest_and_arrays_are_hash_validated(tmp_path: Path) -> None:
    arrays = _arrays(2)
    arrays.update({
        "action_index": np.array([70, 195], dtype=np.int32),
        "executed_act": np.zeros((2, 6), dtype=np.float32),
        "executed_delta_act": np.zeros((2, 6), dtype=np.float32),
        "scenario_index": np.zeros(2, dtype=np.int32),
    })
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    raw = stream.getvalue()
    arrays_path = tmp_path / "recovery_examples.npz"
    arrays_path.write_bytes(raw)
    manifest = {
        "schema_version": 1,
        "arrays": {
            "path": arrays_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": 2,
        },
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded_manifest, loaded = load_oracle_recovery_examples(manifest_path)
    assert loaded_manifest["arrays"]["rows"] == 2
    np.testing.assert_array_equal(loaded["action_index"], [70, 195])
    arrays_path.write_bytes(raw + b"tamper")
    with pytest.raises(ValueError, match="arrays hash mismatch"):
        load_oracle_recovery_examples(manifest_path)


class _FakeAdapter:
    def __init__(self) -> None:
        self.snapshot = PrivilegedStateSnapshot(
            current_act=np.zeros(6, dtype=np.float32),
            robot_qvel=np.zeros(6, dtype=np.float32),
            cube_position=np.zeros(3, dtype=np.float32),
            cube_quaternion_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            cube_linear_velocity=np.zeros(3, dtype=np.float32),
            cube_angular_velocity=np.zeros(3, dtype=np.float32),
        )

    def privileged_state(self) -> PrivilegedStateSnapshot:
        return self.snapshot


def _write_checkpoint_pair(
    directory: Path,
    *,
    hidden_width: int = 128,
    passed: bool = True,
    eligible: bool = True,
    training_row_mode: str = "full",
    checkpoint_manifest: str = "a" * 64,
    report_manifest: str | None = None,
    learning_rate_schedule: str = "fixed",
    prefix_parity_passed: bool | None = None,
) -> Path:
    model = build_oracle_clone_model(10, hidden_width)
    checkpoint = directory / "model.pt"
    rows = list(range(450)) if training_row_mode == "full" else list(ORACLE_MEMORIZATION32_ROWS)
    report = {
        "schema_version": 1,
        "model_kind": "phase_state",
        "passed": passed,
        "closed_loop_eligible": eligible,
        "training_row_mode": training_row_mode,
        "training_row_indices": rows,
        "hidden_width": hidden_width,
        "manifest_content_sha256": report_manifest or checkpoint_manifest,
        "learning_rate_schedule": learning_rate_schedule,
        "prefix_parity": {
            "required": learning_rate_schedule == "decay_10k_20k",
            "passed": prefix_parity_passed,
        },
    }
    report["content_sha256"] = content_sha256(report)
    torch.save({
        "schema_version": 1,
        "model_kind": "phase_state",
        "input_dim": 10,
        "hidden_width": hidden_width,
        "extras_mean": torch.zeros(3),
        "extras_std": torch.ones(3),
        "maximum_act_delta_per_step": torch.ones(6),
        "teacher_horizon": 450,
        "manifest_content_sha256": checkpoint_manifest,
        "report_content_sha256": report["content_sha256"],
        "training_row_mode": training_row_mode,
        "training_row_indices": rows,
        "closed_loop_eligible": eligible,
        "config": {"seed": 101},
        "state_dict": model.state_dict(),
    }, checkpoint)
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    return checkpoint


@pytest.mark.parametrize("hidden_width", [128, 256])
def test_phase_checkpoint_reset_and_off_by_one(tmp_path: Path, hidden_width: int) -> None:
    checkpoint = _write_checkpoint_pair(tmp_path, hidden_width=hidden_width)
    policy = OracleCloneCheckpointPolicy(checkpoint)
    assert policy.hidden_width == hidden_width
    adapter = _FakeAdapter()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    np.testing.assert_array_equal(policy.predict(image, np.zeros(6, dtype=np.float32), adapter), np.zeros(6))
    assert policy.action_index == 1
    policy.reset(adapter)
    assert policy.action_index == 0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"passed": False, "eligible": False},
        {"eligible": False},
        {"training_row_mode": "memorization32", "eligible": False},
        {"report_manifest": "b" * 64},
        {
            "learning_rate_schedule": "decay_10k_20k",
            "prefix_parity_passed": False,
        },
    ],
)
def test_checkpoint_promotion_gate_rejects_ineligible_metadata(
    tmp_path: Path, kwargs: dict[str, object]
) -> None:
    checkpoint = _write_checkpoint_pair(tmp_path, **kwargs)
    with pytest.raises(ValueError, match="eligible|metadata disagree"):
        OracleCloneCheckpointPolicy(checkpoint)


def test_checkpoint_rejects_tampered_report(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint_pair(tmp_path)
    report = json.loads((tmp_path / "report.json").read_text())
    report["passed"] = False
    (tmp_path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash mismatch"):
        OracleCloneCheckpointPolicy(checkpoint)


def test_oracle_safety_count_covers_safe_and_failed_predictions() -> None:
    reset = load_simulation_suite("fixed_pick_place_v3").scenarios[0].robot_qpos_mujoco
    current = np.repeat(mujoco_qpos_to_act(reset)[None], 2, axis=0)
    safe = current.copy()
    unsafe = current.copy()
    unsafe[1, 0] = 100.0
    assert count_oracle_command_safety_violations(current, safe) == 0
    assert count_oracle_command_safety_violations(current, unsafe) == 1
    with pytest.raises(ValueError, match="matching"):
        count_oracle_command_safety_violations(current, unsafe[:, :5])


def test_privileged_snapshot_is_finite_named_state() -> None:
    model = REPOSITORY_ROOT / "simulation_code/model/menagerie_so_arm100/scene_v2.xml"
    adapter = MujocoTaskAdapter(model)
    try:
        adapter.reset(load_simulation_suite("fixed_pick_place_v3").scenarios[0])
        snapshot = adapter.privileged_state()
        assert snapshot.current_act.shape == (6,)
        assert snapshot.robot_qvel.shape == (6,)
        assert snapshot.cube_position.shape == (3,)
        assert snapshot.cube_quaternion_wxyz.shape == (4,)
        assert snapshot.cube_quaternion_wxyz[0] >= 0
        for value in snapshot.__dict__.values():
            assert np.all(np.isfinite(value))
    finally:
        adapter.close()


def test_oracle_manifest_and_array_hashes_are_enforced(tmp_path: Path) -> None:
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        action_index=np.arange(3, dtype=np.int32),
        current_act=np.zeros((3, 6), dtype=np.float32),
        executed_act=np.ones((3, 6), dtype=np.float32),
        executed_delta_act=np.ones((3, 6), dtype=np.float32),
    )
    arrays = stream.getvalue()
    arrays_path = tmp_path / "demonstrations.npz"
    arrays_path.write_bytes(arrays)
    manifest = {
        "arrays": {
            "path": arrays_path.name,
            "sha256": hashlib.sha256(arrays).hexdigest(),
            "rows": 3,
        }
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded, loaded_arrays = load_oracle_demonstrations(manifest_path)
    assert loaded["content_sha256"] == manifest["content_sha256"]
    np.testing.assert_array_equal(
        loaded_arrays["executed_act"] - loaded_arrays["current_act"],
        loaded_arrays["executed_delta_act"],
    )
    arrays_path.write_bytes(arrays + b"tamper")
    try:
        load_oracle_demonstrations(manifest_path)
    except ValueError as exc:
        assert "array hash mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("tampered oracle arrays were accepted")
