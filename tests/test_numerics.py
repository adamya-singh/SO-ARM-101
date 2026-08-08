"""Numerics regime v2: identity conditionality, guards, and GPU determinism."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

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


def test_eval_lane_policy_defaults_to_cpu() -> None:
    from so_arm101_v2.simulation.rollout import TorchCheckpointPolicy
    import inspect

    signature = inspect.signature(TorchCheckpointPolicy.__init__)
    assert signature.parameters["device"].default == "cpu"
