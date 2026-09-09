"""Vision lane: model pins, frames-sidecar training, and the eval policy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.learning.vision import (
    VISION_STATE_DIM,
    VisionChunkedConfig,
    build_vision_chunked_model,
    load_vision_frames,
    train_vision_chunked,
)
from so_arm101_v2.simulation.policy_specs import PolicySpec, build_policy

from test_chunked_promotion import _write_tiny_oracle_manifest


def _write_tiny_frames_manifest(directory: Path, rows: int = 4) -> Path:
    """Extend the tiny oracle fixture with a frames sidecar."""
    manifest_path, _ = _write_tiny_oracle_manifest(directory, rows=rows)
    frames = np.zeros((rows, 256, 256, 3), dtype=np.uint8)
    frames[:, 0, 0, 0] = np.arange(rows, dtype=np.uint8)  # non-degenerate pixels
    frames_path = directory / "images.npy"
    np.save(frames_path, frames)
    body = json.loads(manifest_path.read_text(encoding="utf-8"))
    body.pop("content_sha256")
    body["frames"] = {
        "path": "images.npy",
        "sha256": hashlib.sha256(frames_path.read_bytes()).hexdigest(),
        "rows": rows,
        "dtype": "uint8",
        "frame_shape": [256, 256, 3],
        "convention": "raw_wrist_hwc_uint8_preprocess_with_preprocess_wrist_image",
    }
    body["content_sha256"] = content_sha256(body)
    manifest_path.write_text(json.dumps(body), encoding="utf-8")
    return manifest_path


def test_vision_model_shapes_and_zero_init_head() -> None:
    model = build_vision_chunked_model(256, 2)
    images = torch.rand(3, 3, 256, 256)
    state = torch.rand(3, VISION_STATE_DIM)
    with torch.inference_mode():
        output = model(images, state)
    assert output.shape == (3, 12)
    assert torch.equal(output, torch.zeros_like(output))
    output.requires_grad
    with pytest.raises(ValueError, match="hidden_width"):
        build_vision_chunked_model(384, 2)


def test_vision_training_and_policy_round_trip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "legacy")
    manifest_path = _write_tiny_frames_manifest(tmp_path)
    config = VisionChunkedConfig(chunk_horizon=2, max_steps=3, batch_size=2)
    first = train_vision_chunked(
        manifest_path, tmp_path / "output", config=config, numerics=None,
    )
    report = json.loads(first.report_json.read_text(encoding="utf-8"))
    assert report["model_kind"] == "vision_h2"
    assert report["frames_sha256"]
    assert "cube_position" not in " ".join(report["input_schema"])
    # Idempotent resume.
    again = train_vision_chunked(
        manifest_path, tmp_path / "output", config=config, numerics=None,
    )
    assert again.report_json == first.report_json

    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy

    policy = VisionChunkedPolicy(first.checkpoint)
    assert policy.requires_pixels is True
    assert policy.policy_id == "vision_h2.seed101"
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    current = np.zeros(6, dtype=np.float32)
    command = policy.predict(frame, current)
    assert command.shape == (6,)
    # Zero-init head + zero current -> hold (denormalized midpoint of residual 0).
    second = policy.predict(frame, current)
    assert second.shape == (6,)

    spec = PolicySpec(kind="vision_chunked", checkpoint=str(first.checkpoint.resolve()))
    built = build_policy(spec)
    assert built.policy_id == "vision_h2.seed101"


def test_load_vision_frames_requires_sidecar(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    with pytest.raises(ValueError, match="frames-sidecar"):
        load_vision_frames(manifest_path)

def test_vision_policy_gripper_clamp(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "legacy")
    manifest_path = _write_tiny_frames_manifest(tmp_path)
    config = VisionChunkedConfig(chunk_horizon=2, max_steps=3, batch_size=2)
    result = train_vision_chunked(
        manifest_path, tmp_path / "output", config=config, numerics=None,
    )

    from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds
    from so_arm101_v2.simulation.vision_policy import VisionChunkedPolicy

    policy = VisionChunkedPolicy(result.checkpoint, clamp_channels=(5,))
    assert policy.policy_id == "vision_h2.seed101.gripper_clamp_v1"
    frame = np.zeros((256, 256, 3), dtype=np.uint8)
    command = policy.predict(frame, np.zeros(6, dtype=np.float32))
    low, high = effective_safe_act_bounds()
    assert low[5] <= command[5] <= high[5]

    with pytest.raises(ValueError, match="clamp_channels"):
        VisionChunkedPolicy(result.checkpoint, clamp_channels=(6,))

    spec = PolicySpec(
        kind="vision_chunked",
        checkpoint=str(result.checkpoint.resolve()),
        options=(("clamp_channels", (5,)),),
    )
    built = build_policy(spec)
    assert built.policy_id == "vision_h2.seed101.gripper_clamp_v1"
    # Legacy id untouched when no clamp is requested.
    assert VisionChunkedPolicy(result.checkpoint).policy_id == "vision_h2.seed101"


def test_on_loss_observer_is_digest_neutral(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "legacy")
    manifest_path = _write_tiny_frames_manifest(tmp_path)
    config = VisionChunkedConfig(chunk_horizon=2, max_steps=3, batch_size=2)
    calls: list[tuple[int, float]] = []
    observed = train_vision_chunked(
        manifest_path, tmp_path / "with_observer", config=config, numerics=None,
        on_loss=lambda step, mse: calls.append((step, mse)),
    )
    # Cadence: step 1, then %100 (none within 3 steps), then max_steps.
    assert [step for step, _ in calls] == [1, 3]
    assert all(isinstance(value, float) for _, value in calls)
    silent = train_vision_chunked(
        manifest_path, tmp_path / "without_observer", config=config, numerics=None,
    )
    assert observed.directory.name == silent.directory.name  # same run digest
    assert observed.normalized_mse == silent.normalized_mse


def test_sorted_gather_read_matches_fancy_index() -> None:
    from so_arm101_v2.learning.vision import _read_frame_rows

    rng = np.random.default_rng(0)
    frames = rng.integers(0, 256, size=(20, 4, 4, 3), dtype=np.uint8)
    for index_array in (
        np.array([3, 0, 19, 7], dtype=np.int64),
        np.array([5, 5, 1, 5], dtype=np.int64),  # duplicates
        np.arange(20, dtype=np.int64)[::-1].copy(),
        np.array([0], dtype=np.int64),
    ):
        np.testing.assert_array_equal(
            _read_frame_rows(frames, index_array), frames[index_array]
        )


def test_device_image_upload_is_bitwise_identical_to_cpu_path(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "legacy")
    manifest_path = _write_tiny_frames_manifest(tmp_path, rows=12)
    config = VisionChunkedConfig(chunk_horizon=2, max_steps=25, batch_size=4)
    monkeypatch.setenv("SO_ARM101_V2_IMAGE_UPLOAD", "device")
    device_path = train_vision_chunked(manifest_path, tmp_path / "device", config=config, numerics=None)
    monkeypatch.setenv("SO_ARM101_V2_IMAGE_UPLOAD", "cpu")
    cpu_path = train_vision_chunked(manifest_path, tmp_path / "cpu", config=config, numerics=None)
    assert device_path.directory.name == cpu_path.directory.name
    first = json.loads(device_path.report_json.read_text(encoding="utf-8"))
    second = json.loads(cpu_path.report_json.read_text(encoding="utf-8"))
    assert first["loss_trace"] == second["loss_trace"]
    assert hashlib.sha256(device_path.checkpoint.read_bytes()).hexdigest() == hashlib.sha256(cpu_path.checkpoint.read_bytes()).hexdigest()


def test_device_image_upload_is_bitwise_identical_on_cuda(tmp_path: Path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    monkeypatch.delenv("SO_ARM101_V2_NUMERICS", raising=False)  # conftest pins legacy; this test opts into the GPU regime
    from so_arm101_v2.learning.numerics import require_numerics, resolve_default_numerics
    try:
        spec = resolve_default_numerics()
        if spec is None:
            pytest.skip("no pinned GPU numerics regime resolved")
        require_numerics(spec)
    except Exception as exc:  # the pinned regime is machine-specific
        pytest.skip(f"pinned numerics unavailable here: {exc}")
    manifest_path = _write_tiny_frames_manifest(tmp_path, rows=12)
    config = VisionChunkedConfig(chunk_horizon=2, max_steps=25, batch_size=4)
    monkeypatch.setenv("SO_ARM101_V2_IMAGE_UPLOAD", "device")
    device_path = train_vision_chunked(manifest_path, tmp_path / "device", config=config)
    monkeypatch.setenv("SO_ARM101_V2_IMAGE_UPLOAD", "cpu")
    cpu_path = train_vision_chunked(manifest_path, tmp_path / "cpu", config=config)
    assert device_path.directory.name == cpu_path.directory.name
    first = json.loads(device_path.report_json.read_text(encoding="utf-8"))
    second = json.loads(cpu_path.report_json.read_text(encoding="utf-8"))
    assert first["loss_trace"] == second["loss_trace"]
    assert hashlib.sha256(device_path.checkpoint.read_bytes()).hexdigest() == hashlib.sha256(cpu_path.checkpoint.read_bytes()).hexdigest()


def test_frame_cache_is_bitwise_identical_to_the_memmap_and_is_reused(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "legacy")
    manifest_path = _write_tiny_frames_manifest(tmp_path, rows=12)
    config = VisionChunkedConfig(chunk_horizon=2, max_steps=25, batch_size=4)
    monkeypatch.setenv("SO_ARM101_V2_FRAME_CACHE", "off")
    memmap = train_vision_chunked(manifest_path, tmp_path / "memmap", config=config, numerics=None)
    monkeypatch.setenv("SO_ARM101_V2_FRAME_CACHE", "zlib")
    cached = train_vision_chunked(manifest_path, tmp_path / "cached", config=config, numerics=None)
    caches = list((tmp_path / "cached" / "frame_cache").glob("frames_cache_*.npz"))
    assert len(caches) == 1
    from so_arm101_v2.learning.vision import CompressedFrames, load_vision_frames
    _manifest, frames, _arrays = load_vision_frames(manifest_path)
    loaded = CompressedFrames.load(caches[0])
    assert loaded.shape == frames.shape
    assert np.array_equal(loaded[np.array([3, 0, 11])], np.asarray(frames[np.array([3, 0, 11])]))
    assert np.array_equal(loaded[2:5], np.asarray(frames[2:5]))
    # A run under a root that already holds the cache loads it instead of rebuilding (bitwise identical again).
    (tmp_path / "cached_again" / "frame_cache").mkdir(parents=True)
    (tmp_path / "cached_again" / "frame_cache" / caches[0].name).write_bytes(caches[0].read_bytes())
    reused = train_vision_chunked(manifest_path, tmp_path / "cached_again", config=config, numerics=None)
    for a, b in ((memmap, cached), (memmap, reused)):
        assert a.directory.name == b.directory.name
        assert json.loads(a.report_json.read_text())["loss_trace"] == json.loads(b.report_json.read_text())["loss_trace"]
        assert hashlib.sha256(a.checkpoint.read_bytes()).hexdigest() == hashlib.sha256(b.checkpoint.read_bytes()).hexdigest()


def test_prefetch_is_bitwise_identical_to_synchronous(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "legacy")
    manifest_path = _write_tiny_frames_manifest(tmp_path, rows=12)
    config = VisionChunkedConfig(chunk_horizon=2, max_steps=25, batch_size=4)

    monkeypatch.setenv("SO_ARM101_V2_PREFETCH", "1")
    prefetched = train_vision_chunked(
        manifest_path, tmp_path / "prefetched", config=config, numerics=None,
    )
    monkeypatch.setenv("SO_ARM101_V2_PREFETCH", "0")
    synchronous = train_vision_chunked(
        manifest_path, tmp_path / "synchronous", config=config, numerics=None,
    )
    assert prefetched.directory.name == synchronous.directory.name  # same run digest
    assert prefetched.normalized_mse == synchronous.normalized_mse
    first = json.loads(prefetched.report_json.read_text(encoding="utf-8"))
    second = json.loads(synchronous.report_json.read_text(encoding="utf-8"))
    assert first["loss_trace"] == second["loss_trace"]  # whole stream matched
    checkpoint_shas = [
        hashlib.sha256(result.checkpoint.read_bytes()).hexdigest()
        for result in (prefetched, synchronous)
    ]
    assert checkpoint_shas[0] == checkpoint_shas[1]
