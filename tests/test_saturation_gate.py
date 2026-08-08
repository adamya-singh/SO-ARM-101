from __future__ import annotations

from pathlib import Path
import hashlib
import io
import json

import numpy as np
import pytest
import torch

from so_arm101_v2.contracts import (
    ACT_DATASET_HIGH,
    ACT_DATASET_LOW,
    effective_safe_act_bounds,
)
from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.learning.chunked import (
    ChunkedCloneConfig,
    build_chunked_targets,
    decode_feasible_chunk,
    train_chunked_clone,
)
from so_arm101_v2.learning.tiny_model import normalize_act
from so_arm101_v2.simulation.chunked import (
    SATURATION_CANDIDATE_ORDER,
    ChunkedClonePolicy,
    resolve_saturation_gate_status,
)

from test_chunked_promotion import _FakeAdapter, _write_tiny_oracle_manifest


def _write_tiny_correction_manifest(
    directory: Path, *, source_sha: str, start: int = 448, length: int = 3,
) -> tuple[Path, dict]:
    action_index = np.arange(start, start + length, dtype=np.int32)
    progress = (np.minimum(action_index, 449).astype(np.float64) / 449.0).astype(np.float32)
    arrays = {
        "action_index": action_index,
        "progress": progress,
        "current_act": np.full((length, 6), 0.01, dtype=np.float32),
        "robot_qvel": np.zeros((length, 6), dtype=np.float32),
        "cube_position": np.full((length, 3), 0.5, dtype=np.float32),
        "cube_quaternion_wxyz": np.tile(np.array([1, 0, 0, 0], dtype=np.float32), (length, 1)),
        "cube_linear_velocity": np.zeros((length, 3), dtype=np.float32),
        "cube_angular_velocity": np.zeros((length, 3), dtype=np.float32),
        "executed_act": np.full((length, 6), 0.012, dtype=np.float32),
        "executed_delta_act": np.full((length, 6), 0.002, dtype=np.float32),
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
        "inducing_checkpoint_sha256": "f" * 64,
        "progress_rule": "deployment_clock",
        "arrays": {
            "path": arrays_path.name,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "rows": length,
        },
        "episodes": [
            {"name": "release", "action_index": start, "rows": length, "accepted": True},
        ],
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, manifest


def test_effective_safe_act_bounds_intersect_and_handle_sign_swap() -> None:
    low, high = effective_safe_act_bounds()
    assert np.all(low < high)
    assert np.all(low >= ACT_DATASET_LOW - 1e-6)
    assert np.all(high <= ACT_DATASET_HIGH + 1e-6)
    # Gripper floor sits strictly above ACT zero (Menagerie ctrlrange overhang).
    assert 0 < low[5] < 1e-3
    # Non-trivial pulled-back bounds from the mujoco envelope.
    assert high[1] == pytest.approx(3.14063, abs=1e-4)
    assert high[2] == pytest.approx(2.91704, abs=1e-4)
    assert low[4] == pytest.approx(-3.13876, abs=1e-4)
    assert high[4] == pytest.approx(3.13874, abs=1e-4)
    # Joints 0 and 3 keep the plain ACT box despite the joint-0 sign swap.
    assert low[0] == pytest.approx(float(ACT_DATASET_LOW[0]))
    assert high[0] == pytest.approx(float(ACT_DATASET_HIGH[0]))


def test_saturation_config_validation() -> None:
    ChunkedCloneConfig(chunk_horizon=90)
    with pytest.raises(ValueError, match="saturation parameters"):
        ChunkedCloneConfig(chunk_horizon=90, decoder_eta=0.9)
    with pytest.raises(ValueError, match="saturation_mode"):
        ChunkedCloneConfig(chunk_horizon=90, saturation_mode="other")
    with pytest.raises(ValueError, match="decoder_eta"):
        ChunkedCloneConfig(chunk_horizon=90, saturation_mode="feasible_chain_v1")
    with pytest.raises(ValueError, match="margin_act"):
        ChunkedCloneConfig(
            chunk_horizon=90, saturation_mode="feasible_chain_v1",
            decoder_eta=0.9, margin_act=0.004,
        )
    chain = ChunkedCloneConfig(
        chunk_horizon=90, saturation_mode="feasible_chain_v1",
        decoder_eta=0.9, margin_act=0.002,
    )
    assert chain.noise_sigma is None
    with pytest.raises(ValueError, match="noise_penalty_v1"):
        ChunkedCloneConfig(
            chunk_horizon=90, saturation_mode="noise_penalty_v1",
            decoder_eta=0.9, margin_act=0.002,
        )
    with pytest.raises(ValueError, match="only valid for noise_penalty_v1"):
        ChunkedCloneConfig(
            chunk_horizon=90, saturation_mode="feasible_chain_v1",
            decoder_eta=0.9, margin_act=0.002, noise_sigma=0.05,
        )
    ChunkedCloneConfig(
        chunk_horizon=90, saturation_mode="noise_penalty_v1",
        decoder_eta=0.9, margin_act=0.002, noise_sigma=0.05, penalty_weight=1.0,
    )


def test_feasible_decoder_zero_raw_holds_current_pose() -> None:
    current = normalize_act(np.array([0.1, -0.2, 0.3, 0.0, 0.1, 0.5], dtype=np.float32))
    raw = torch.zeros(1, 4, 6)
    decoded = decode_feasible_chunk(
        raw, torch.from_numpy(current[None, :]), eta=0.9, margin_act=0.002,
    ).numpy()[0]
    np.testing.assert_array_equal(decoded, np.tile(current, (4, 1)))
    # An out-of-box current pose is pulled inside the shrunk box immediately.
    outside = current.copy()
    outside[5] = -1.5
    decoded = decode_feasible_chunk(
        raw, torch.from_numpy(outside[None, :]), eta=0.9, margin_act=0.002,
    ).numpy()[0]
    low, high = effective_safe_act_bounds()
    low_norm = normalize_act(low + np.float32(0.002))
    high_norm = normalize_act(high - np.float32(0.002))
    assert np.all(decoded >= low_norm - 1e-6) and np.all(decoded <= high_norm + 1e-6)


def test_feasible_decoder_is_near_identity_on_oracle_scale_deltas() -> None:
    current = normalize_act(np.zeros((1, 6), dtype=np.float32) + 0.3)
    desired = np.full((1, 5, 6), 0.03, dtype=np.float32)  # ~0.1 ACT per step
    decoded = decode_feasible_chunk(
        torch.from_numpy(desired), torch.from_numpy(current), eta=0.9, margin_act=0.002,
    ).numpy()
    steps = np.diff(
        np.concatenate([current[:, None, :], decoded], axis=1), axis=1
    )
    np.testing.assert_allclose(steps, desired, rtol=0.02)


def test_feasible_decoder_bounds_adversarial_raw_within_box_and_step_caps() -> None:
    from so_arm101_v2.learning.chunked import _feasible_decode_constants

    low_norm, high_norm, delta_norm = _feasible_decode_constants(0.002)
    current = normalize_act(np.zeros((1, 6), dtype=np.float32))
    raw = torch.full((1, 8, 6), 1e6)
    raw[0, 3:] = -1e6
    decoded = decode_feasible_chunk(
        raw, torch.from_numpy(current), eta=0.9, margin_act=0.002,
    ).numpy()[0]
    assert np.all(decoded >= low_norm - 1e-6) and np.all(decoded <= high_norm + 1e-6)
    chain = np.concatenate([np.clip(current, low_norm, high_norm), decoded])
    steps = np.abs(np.diff(chain.reshape(9, 6), axis=0))
    assert np.all(steps <= delta_norm * 0.9 + 1e-6)


def test_chunked_targets_respect_episode_boundaries() -> None:
    executed = np.stack([
        np.full(6, value, dtype=np.float32)
        for value in (0.01, 0.02, 0.03, 0.11, 0.12)
    ])
    arrays = {"current_act": np.zeros((5, 6), dtype=np.float32), "executed_act": executed}
    targets = build_chunked_targets(arrays, chunk_horizon=3, episode_lengths=[3, 2])
    hold = normalize_act(np.zeros(6, dtype=np.float32))
    # Row 2 (last of episode one) repeats its own final command, never row 3's.
    np.testing.assert_array_equal(
        targets[2, 2], normalize_act(executed[2]) - hold,
    )
    np.testing.assert_array_equal(
        targets[1, 2], normalize_act(executed[2]) - hold,
    )
    # Episode two pads with its own final command.
    np.testing.assert_array_equal(
        targets[3, 2], normalize_act(executed[4]) - hold,
    )
    with pytest.raises(ValueError, match="sum to the row count"):
        build_chunked_targets(arrays, chunk_horizon=2, episode_lengths=[3, 3])


def test_chunked_identity_new_keys_absent_preserve_legacy_digests(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert "saturation" not in report
    assert "correction_manifest_content_sha256" not in report
    assert set(report["config"]) == {
        "chunk_horizon", "seed", "hidden_width", "learning_rate", "max_steps",
    }
    identity_keys = (
        "manifest_content_sha256", "collection_digest", "model_kind", "optimizer",
        "source_rows", "config", "target", "chunk_padding", "offline_role",
    )
    identity = {name: report[name] for name in identity_keys}
    assert content_sha256(identity) == report["run_digest"]


def test_chunked_correction_training_uses_nominal_statistics_and_new_identity(
    tmp_path: Path,
) -> None:
    manifest_path, manifest = _write_tiny_oracle_manifest(tmp_path)
    corrections = tmp_path / "corrections"
    corrections.mkdir()
    correction_path, correction_manifest = _write_tiny_correction_manifest(
        corrections, source_sha=manifest["content_sha256"],
    )
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
        correction_manifest_path=correction_path,
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert report["rows"] == 7
    assert report["source_rows"] == 4
    assert report["correction_rows"] == 3
    assert report["correction_augmentation"]["episode_lengths"] == [4, 3]
    assert report["correction_augmentation"]["manifest_content_sha256"] == (
        correction_manifest["content_sha256"]
    )
    nominal_cube = np.arange(4 * 3, dtype=np.float32).reshape(4, 3) * 0.01
    np.testing.assert_allclose(
        np.asarray(report["normalization"]["extras_mean"], dtype=np.float32),
        nominal_cube.mean(axis=0, dtype=np.float64).astype(np.float32),
    )
    identity_keys = (
        "manifest_content_sha256", "collection_digest", "model_kind", "optimizer",
        "source_rows", "config", "target", "chunk_padding", "offline_role",
        "correction_manifest_content_sha256", "correction_collection_digest",
        "correction_rows",
    )
    identity = {name: report[name] for name in identity_keys}
    assert content_sha256(identity) == report["run_digest"]

    mismatched = tmp_path / "mismatched"
    mismatched.mkdir()
    bad_path, _ = _write_tiny_correction_manifest(mismatched, source_sha="0" * 64)
    with pytest.raises(ValueError, match="do not derive"):
        train_chunked_clone(
            manifest_path, tmp_path / "other",
            config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
            correction_manifest_path=bad_path,
        )


def test_noise_penalty_training_records_saturation_telemetry(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(
            chunk_horizon=2, max_steps=2, saturation_mode="noise_penalty_v1",
            decoder_eta=0.9, margin_act=0.002, noise_sigma=0.05, penalty_weight=1.0,
        ),
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    saturation = report["saturation"]
    assert saturation["mode"] == "noise_penalty_v1"
    assert saturation["final_penalty_value"] >= 0
    assert np.isfinite(saturation["final_penalty_value"])
    assert saturation["decoder_telemetry"] is None


def test_chunked_policy_applies_feasible_decoder_from_checkpoint(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    chain = train_chunked_clone(
        manifest_path, tmp_path / "chain",
        config=ChunkedCloneConfig(
            chunk_horizon=3, max_steps=2, saturation_mode="feasible_chain_v1",
            decoder_eta=0.9, margin_act=0.002,
        ),
    )
    report = json.loads(chain.report_json.read_text(encoding="utf-8"))
    telemetry = report["saturation"]["decoder_telemetry"]
    assert 0 <= telemetry["clamp_active_fraction"] <= 1
    assert 0 <= telemetry["tanh_activation_max_abs"] <= 1
    policy = ChunkedClonePolicy(chain.checkpoint)
    assert policy.saturation_mode == "feasible_chain_v1"
    assert policy.policy_id.endswith(".feasible_chain_v1")
    adapter = _FakeAdapter()
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    commands = [
        policy.predict(image, np.zeros(6, dtype=np.float32), adapter) for _ in range(3)
    ]
    low, high = effective_safe_act_bounds()
    for command in commands:
        assert np.all(command >= low - 1e-5) and np.all(command <= high + 1e-5)

    legacy = train_chunked_clone(
        manifest_path, tmp_path / "legacy",
        config=ChunkedCloneConfig(chunk_horizon=3, max_steps=2),
    )
    legacy_policy = ChunkedClonePolicy(legacy.checkpoint)
    assert legacy_policy.saturation_mode == "none"
    assert legacy_policy.policy_id == "chunked_h3.seed101"


def test_saturation_gate_resolver_applies_promotion_order_and_requires_all_candidates() -> None:
    def record(candidate_id: str, passed: bool) -> dict:
        return {
            "candidate_id": candidate_id,
            "state": "nominal_passed" if passed else "nominal_failed",
            "evaluation": {"passed": passed},
        }

    all_fail = [record(name, False) for name in SATURATION_CANDIDATE_ORDER]
    assert resolve_saturation_gate_status(all_fail) == "closed_loop_not_resolved"
    corrections_pass = [
        record(name, name == "corrections_only") for name in SATURATION_CANDIDATE_ORDER
    ]
    assert resolve_saturation_gate_status(corrections_pass) == "promoted_corrections_only"
    # The decoder family outranks later passers even when several pass.
    several = [
        record(name, name in ("corrections_and_feasible_decoder", "corrections_only"))
        for name in SATURATION_CANDIDATE_ORDER
    ]
    assert resolve_saturation_gate_status(several) == (
        "promoted_corrections_and_feasible_decoder"
    )
    with pytest.raises(ValueError, match="order"):
        resolve_saturation_gate_status(list(reversed(all_fail)))
    with pytest.raises(ValueError, match="evaluation evidence"):
        resolve_saturation_gate_status(
            [{**all_fail[0], "evaluation": None}] + all_fail[1:]
        )
    with pytest.raises(ValueError, match="one record per candidate"):
        resolve_saturation_gate_status(all_fail[:3])


# --- parallel gate orchestration -------------------------------------------
#
# The candidate worker itself is exercised end-to-end by the parity runs; here
# we verify the parent's orchestration contract: submission-order collection,
# byte-identical reports regardless of completion order, and resume behavior.
# A thread pool stands in for the spawn pool because monkeypatched fakes
# cannot cross a spawn boundary.


def _install_fake_gate_inputs(monkeypatch, model_path: Path) -> None:
    from so_arm101_v2.simulation import chunked as chunked_module
    from so_arm101_v2.simulation import correction as correction_module

    model_path.write_bytes(b"<mujoco/>")
    oracle_manifest = {
        "content_sha256": "oracle-sha",
        "scenario_ids": ["nominal"],
        "teacher_horizon": 450,
    }
    correction_manifest = {
        "content_sha256": "correction-sha",
        "source_manifest_content_sha256": "oracle-sha",
    }
    preflight = {
        "content_sha256": "preflight-sha",
        "environment_proven": True,
        "deterministic": True,
        "suite": {"suite_id": "fixed_pick_place_v3"},
    }
    monkeypatch.setattr(
        chunked_module, "load_oracle_demonstrations",
        lambda path: (oracle_manifest, None),
    )
    monkeypatch.setattr(
        correction_module, "load_oracle_corrections",
        lambda path: (correction_manifest, None),
    )
    original_load = chunked_module._load_hashed_json

    def fake_load(path, label=""):
        if label == "preflight report":
            return preflight
        return original_load(path, label=label)

    monkeypatch.setattr(chunked_module, "_load_hashed_json", fake_load)


def _install_fake_candidate(monkeypatch, calls: list, delays=None) -> None:
    import time

    from so_arm101_v2.simulation import chunked as chunked_module

    def fake_candidate(task):
        index = SATURATION_CANDIDATE_ORDER.index(task.candidate_id)
        if delays is not None:
            time.sleep(delays[index])
        calls.append(task.candidate_id)
        return {
            "candidate_id": task.candidate_id,
            "saturation_mode": "fake_mode",
            "corrections": False,
            "state": "nominal_failed",
            "checkpoint_sha256": f"checkpoint-{task.candidate_id}",
            "training_report": f"training-{task.candidate_id}",
            "training_report_content_sha256": f"training-sha-{task.candidate_id}",
            "offline_telemetry": {"normalized_mse": 0.5, "max_act_error": 0.5, "steps": 1},
            "evaluation": {
                "report": f"evaluation-{task.candidate_id}",
                "content_sha256": f"evaluation-sha-{task.candidate_id}",
                "passed": False,
                "deterministic": True,
                "rollouts": [],
            },
        }

    monkeypatch.setattr(chunked_module, "_run_saturation_candidate", fake_candidate)


def _install_thread_pool(monkeypatch) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from so_arm101_v2.simulation import chunked as chunked_module

    class _ThreadStandIn(ThreadPoolExecutor):
        def __init__(self, max_workers=None, mp_context=None, initializer=None):
            del mp_context, initializer
            super().__init__(max_workers=max_workers)

    monkeypatch.setattr(chunked_module, "ProcessPoolExecutor", _ThreadStandIn)


def test_saturation_gate_parallel_report_matches_sequential(tmp_path, monkeypatch) -> None:
    from so_arm101_v2.simulation.chunked import run_saturation_gate

    model_path = tmp_path / "scene.xml"
    _install_fake_gate_inputs(monkeypatch, model_path)
    _install_thread_pool(monkeypatch)

    sequential_calls: list = []
    _install_fake_candidate(monkeypatch, sequential_calls)
    sequential = run_saturation_gate(
        model_path, "oracle.json", "corrections.json", tmp_path / "preflight.json",
        tmp_path / "sequential", workers=1,
    )
    assert sequential_calls == list(SATURATION_CANDIDATE_ORDER)

    # Completion order scrambled: earliest-submitted candidate finishes last.
    parallel_calls: list = []
    _install_fake_candidate(
        monkeypatch, parallel_calls,
        delays=[0.2, 0.15, 0.1, 0.05, 0.0],
    )
    parallel = run_saturation_gate(
        model_path, "oracle.json", "corrections.json", tmp_path / "preflight.json",
        tmp_path / "parallel", workers=5,
    )
    assert parallel_calls != list(SATURATION_CANDIDATE_ORDER)
    assert sorted(parallel_calls) == sorted(SATURATION_CANDIDATE_ORDER)

    report_parallel = json.loads(parallel.report_json.read_text())
    assert [
        item["candidate_id"] for item in report_parallel["candidates"]
    ] == list(SATURATION_CANDIDATE_ORDER)
    assert parallel.report_json.read_bytes() == sequential.report_json.read_bytes()
    assert parallel.status == "closed_loop_not_resolved"


def test_saturation_gate_resume_skips_candidate_execution(tmp_path, monkeypatch) -> None:
    from so_arm101_v2.simulation import chunked as chunked_module
    from so_arm101_v2.simulation.chunked import run_saturation_gate

    model_path = tmp_path / "scene.xml"
    _install_fake_gate_inputs(monkeypatch, model_path)
    calls: list = []
    _install_fake_candidate(monkeypatch, calls)
    first = run_saturation_gate(
        model_path, "oracle.json", "corrections.json", tmp_path / "preflight.json",
        tmp_path / "out", workers=1,
    )
    assert len(calls) == len(SATURATION_CANDIDATE_ORDER)

    def exploding_candidate(task):
        raise AssertionError("resume must not re-run candidates")

    monkeypatch.setattr(chunked_module, "_run_saturation_candidate", exploding_candidate)
    resumed = run_saturation_gate(
        model_path, "oracle.json", "corrections.json", tmp_path / "preflight.json",
        tmp_path / "out", workers=5,
    )
    assert resumed.status == first.status
    assert resumed.report_json == first.report_json
