from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.learning.chunked import (
    ChunkedCloneConfig,
    build_chunked_clone_model,
    build_chunked_targets,
    train_chunked_clone,
)
from so_arm101_v2.learning.tiny_model import normalize_act
from so_arm101_v2.simulation import PrivilegedStateSnapshot
from so_arm101_v2.simulation.chunked import (
    CHUNK_HORIZON_LADDER,
    ChunkedClonePolicy,
    resolve_chunked_gate_status,
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
        "teacher_horizon": 450,
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


class _FakeAdapter:
    def __init__(self) -> None:
        self.privileged_calls = 0
        self.snapshot = PrivilegedStateSnapshot(
            current_act=np.zeros(6, dtype=np.float32),
            robot_qvel=np.zeros(6, dtype=np.float32),
            cube_position=np.zeros(3, dtype=np.float32),
            cube_quaternion_wxyz=np.array([1, 0, 0, 0], dtype=np.float32),
            cube_linear_velocity=np.zeros(3, dtype=np.float32),
            cube_angular_velocity=np.zeros(3, dtype=np.float32),
        )

    def privileged_state(self) -> PrivilegedStateSnapshot:
        self.privileged_calls += 1
        return self.snapshot


def test_chunked_config_and_model_initialize_to_hold() -> None:
    config = ChunkedCloneConfig(chunk_horizon=10)
    assert config.hidden_width == 256 and config.max_steps == 30_000
    with pytest.raises(ValueError, match="chunk_horizon"):
        ChunkedCloneConfig(chunk_horizon=0)
    with pytest.raises(ValueError, match="chunk_horizon"):
        ChunkedCloneConfig(chunk_horizon=481)
    assert ChunkedCloneConfig(chunk_horizon=10, hidden_width=512).hidden_width == 512
    with pytest.raises(ValueError, match="hidden_width"):
        ChunkedCloneConfig(chunk_horizon=10, hidden_width=384)
    torch.manual_seed(101)
    model = build_chunked_clone_model(10, 256, 30)
    with torch.inference_mode():
        output = model(torch.rand(5, 10))
    torch.testing.assert_close(output, torch.zeros(5, 180))


def test_chunked_targets_pad_by_repeating_final_command() -> None:
    arrays = {
        "current_act": np.zeros((4, 6), dtype=np.float32),
        "executed_act": np.stack([
            np.full(6, value, dtype=np.float32) for value in (0.01, 0.02, 0.03, 0.04)
        ]),
    }
    targets = build_chunked_targets(arrays, chunk_horizon=3)
    assert targets.shape == (4, 3, 6)
    expected_last = normalize_act(arrays["executed_act"][3]) - normalize_act(np.zeros(6, dtype=np.float32))
    # Row 3's whole chunk repeats the final command; row 2 pads offsets 1 and 2.
    np.testing.assert_array_equal(targets[3, 0], expected_last)
    np.testing.assert_array_equal(targets[3, 2], expected_last)
    np.testing.assert_array_equal(targets[2, 1], expected_last)
    constant = np.full((4, 6), 0.02, dtype=np.float32)
    zero_residual = build_chunked_targets(
        {"current_act": constant, "executed_act": constant}, chunk_horizon=2,
    )
    np.testing.assert_array_equal(zero_residual, np.zeros_like(zero_residual))


def test_chunked_training_is_content_addressed_and_idempotent(tmp_path: Path) -> None:
    manifest_path, manifest = _write_tiny_oracle_manifest(tmp_path)
    config = ChunkedCloneConfig(chunk_horizon=2, max_steps=2)
    result = train_chunked_clone(manifest_path, tmp_path / "output", config=config)
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert report["model_kind"] == "chunked_h2"
    assert report["offline_role"] == "telemetry_only_promotion_by_closed_loop"
    assert report["manifest_content_sha256"] == manifest["content_sha256"]
    identity_keys = (
        "manifest_content_sha256", "collection_digest", "model_kind", "optimizer",
        "source_rows", "config", "target", "chunk_padding", "offline_role",
    )
    identity = {name: report[name] for name in identity_keys}
    assert content_sha256(identity) == report["run_digest"]
    replay = train_chunked_clone(manifest_path, tmp_path / "output", config=config)
    assert replay.report_json == result.report_json
    assert replay.normalized_mse == result.normalized_mse


def test_chunked_policy_buffers_one_observation_per_chunk(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    config = ChunkedCloneConfig(chunk_horizon=3, max_steps=2)
    result = train_chunked_clone(manifest_path, tmp_path / "output", config=config)
    policy = ChunkedClonePolicy(result.checkpoint)
    adapter = _FakeAdapter()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    current = np.zeros(6, dtype=np.float32)
    commands = [policy.predict(image, current, adapter) for _ in range(7)]
    # Seven actions require ceil(7 / 3) = 3 privileged observations.
    assert adapter.privileged_calls == 3
    assert policy.action_index == 7
    assert all(command.shape == (6,) for command in commands)
    policy.reset(adapter)
    assert policy.action_index == 0 and adapter.privileged_calls == 3
    assert all(np.all(np.isfinite(command)) for command in commands)


def test_chunked_policy_rejects_tampered_report(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
    )
    report_path = result.report_json
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["normalized_mse"] = 0.0
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        ChunkedClonePolicy(result.checkpoint)


def test_chunked_gate_branch_trace_resolution() -> None:
    failed = lambda horizon: {
        "chunk_horizon": horizon, "state": "nominal_failed",
        "evaluation": {"passed": False},
    }
    passed = lambda horizon: {
        "chunk_horizon": horizon, "state": "nominal_passed",
        "evaluation": {"passed": True},
    }
    skipped = lambda horizon: {
        "chunk_horizon": horizon, "state": "skipped_after_first_pass",
    }
    assert resolve_chunked_gate_status(
        [failed(10), passed(30), skipped(90)]
    ) == "passed_h30"
    assert resolve_chunked_gate_status(
        [failed(10), failed(30), failed(90)]
    ) == "closed_loop_not_resolved"
    with pytest.raises(ValueError, match="after the first"):
        resolve_chunked_gate_status([passed(10), failed(30), skipped(90)])
    with pytest.raises(ValueError, match="ladder"):
        resolve_chunked_gate_status([failed(30), failed(10), failed(90)])
    with pytest.raises(ValueError, match="evaluation evidence"):
        resolve_chunked_gate_status([
            {"chunk_horizon": 10, "state": "nominal_failed", "evaluation": None},
            failed(30), failed(90),
        ])
    assert CHUNK_HORIZON_LADDER == (10, 30, 90)
