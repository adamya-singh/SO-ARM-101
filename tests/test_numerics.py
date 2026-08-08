"""Numerics regime v2: identity conditionality, guards, and GPU determinism."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from so_arm101_v2.data._serialization import content_sha256
from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
from so_arm101_v2.learning.numerics import (
    PINNED_NUMERICS_V2,
    NumericsSpec,
    cpu_state_dict,
    noise_generator,
    numerics_identity,
    require_numerics,
    resolve_default_numerics,
    resolve_numerics,
)

from test_chunked_promotion import _write_tiny_oracle_manifest

_CUDA = torch.cuda.is_available()


def test_numerics_identity_key_set_is_pinned() -> None:
    identity = numerics_identity(PINNED_NUMERICS_V2)
    assert set(identity) == {
        "regime", "device", "gpu", "torch", "cuda", "driver", "compile",
        "compile_mode", "tf32", "deterministic_algorithms",
        "cublas_workspace", "noise_stream",
    }
    assert identity["regime"] == "cuda_inductor_v2"
    assert identity["tf32"] is False
    assert identity["compile_mode"] == "default"


def test_spec_validation_rejects_bad_regimes() -> None:
    with pytest.raises(ValueError, match="max-autotune|compile_mode"):
        replace(PINNED_NUMERICS_V2, compile_mode="max-autotune")
    with pytest.raises(ValueError, match="tf32"):
        replace(PINNED_NUMERICS_V2, tf32=True)
    with pytest.raises(ValueError, match="cuda numerics require"):
        replace(PINNED_NUMERICS_V2, driver_version=None)


def test_resolver_semantics(monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "legacy")
    assert resolve_default_numerics() is None
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "auto")
    assert resolve_numerics("cpu", compile=False) is None
    probe = resolve_numerics("cpu", compile=True)
    assert probe is not None and probe.regime == "cpu_inductor_v2"


def test_require_numerics_raises_on_mismatch() -> None:
    wrong_torch = replace(PINNED_NUMERICS_V2, torch_version="0.0.0+fake")
    with pytest.raises(RuntimeError, match="torch"):
        require_numerics(wrong_torch)
    if _CUDA:
        wrong_gpu = replace(PINNED_NUMERICS_V2, gpu="NVIDIA Fake GPU 9000")
        with pytest.raises(RuntimeError, match="gpu"):
            require_numerics(wrong_gpu)


def test_legacy_training_report_has_no_numerics_key(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    config = ChunkedCloneConfig(chunk_horizon=2, max_steps=2)
    result = train_chunked_clone(
        manifest_path, tmp_path / "legacy", config=config, numerics=None,
    )
    report = json.loads(result.report_json.read_text(encoding="utf-8"))
    assert "numerics" not in report
    identity_keys = (
        "manifest_content_sha256", "collection_digest", "model_kind", "optimizer",
        "source_rows", "config", "target", "chunk_padding", "offline_role",
    )
    identity = {name: report[name] for name in identity_keys}
    assert content_sha256(identity) == report["run_digest"]


def test_noise_stream_is_seeded_and_insulated() -> None:
    first = torch.randn((4, 6), generator=noise_generator(torch, 101))
    torch.manual_seed(999)          # perturb the global stream
    torch.randn(1000)               # consume from it
    second = torch.randn((4, 6), generator=noise_generator(torch, 101))
    assert torch.equal(first, second)
    assert not torch.equal(first, torch.randn((4, 6), generator=noise_generator(torch, 102)))


def test_cpu_state_dict_unwraps_compiled_models() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    compiled = torch.compile(model)
    state = cpu_state_dict(compiled)
    assert all(not key.startswith("_orig_mod.") for key in state)
    assert set(state) == set(model.state_dict())
    fresh = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
    fresh.load_state_dict(state)


@pytest.mark.skipif(not _CUDA, reason="regime v2 requires CUDA")
def test_v2_training_forks_digest_and_is_deterministic(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "auto")
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    config = ChunkedCloneConfig(chunk_horizon=2, max_steps=2)
    legacy = train_chunked_clone(
        manifest_path, tmp_path / "legacy", config=config, numerics=None,
    )
    spec = resolve_default_numerics()
    assert spec is not None
    first = train_chunked_clone(
        manifest_path, tmp_path / "v2_a", config=config, numerics=spec,
    )
    second = train_chunked_clone(
        manifest_path, tmp_path / "v2_b", config=config, numerics=spec,
    )
    report = json.loads(first.report_json.read_text(encoding="utf-8"))
    assert report["numerics"] == numerics_identity(spec)
    # v2 forks the digest space; the two v2 runs are bitwise identical.
    assert first.directory.name != legacy.directory.name
    assert first.directory.name == second.directory.name
    assert (
        hashlib.sha256(first.checkpoint.read_bytes()).hexdigest()
        == hashlib.sha256(second.checkpoint.read_bytes()).hexdigest()
    )
    # Idempotent resume within the regime: a re-run replays the cached report.
    resumed = train_chunked_clone(
        manifest_path, tmp_path / "v2_a", config=config, numerics=spec,
    )
    assert resumed.report_json == first.report_json
    # State dict is device-clean and loads into a CPU model.
    payload = torch.load(first.checkpoint, map_location="cpu", weights_only=True)
    assert all(not key.startswith("_orig_mod.") for key in payload["state_dict"])
    assert all(value.device.type == "cpu" for value in payload["state_dict"].values())


@pytest.mark.skipif(not _CUDA, reason="regime v2 requires CUDA")
def test_v2_eager_and_compiled_fork_digests(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SO_ARM101_V2_NUMERICS", "auto")
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    config = ChunkedCloneConfig(chunk_horizon=2, max_steps=2)
    spec = resolve_default_numerics()
    assert spec is not None
    eager_spec = replace(spec, compile=None)
    compiled = train_chunked_clone(
        manifest_path, tmp_path / "compiled", config=config, numerics=spec,
    )
    eager = train_chunked_clone(
        manifest_path, tmp_path / "eager", config=config, numerics=eager_spec,
    )
    assert compiled.directory.name != eager.directory.name
    # Same device, same seed: results must agree numerically even if the
    # digests are (deliberately) separate identities.
    a = torch.load(compiled.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    b = torch.load(eager.checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    for key in a:
        assert torch.allclose(a[key], b[key], rtol=1e-5, atol=1e-6), key


def test_eval_lane_policy_defaults_to_cpu() -> None:
    from so_arm101_v2.simulation.rollout import TorchCheckpointPolicy
    import inspect

    signature = inspect.signature(TorchCheckpointPolicy.__init__)
    assert signature.parameters["device"].default == "cpu"
