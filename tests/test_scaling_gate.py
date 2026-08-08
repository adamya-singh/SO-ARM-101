from __future__ import annotations

import numpy as np
import pytest
import torch

from so_arm101_v2.learning.chunked import ChunkedCloneConfig, build_chunked_clone_model
from so_arm101_v2.simulation.broader import FROZEN_RETRAIN_CONFIG
from so_arm101_v2.simulation.scaling import (
    FROZEN_RECIPE,
    SCALING_CANDIDATES,
    resolve_scaling_status,
)


def _candidate(candidate_id: str, stage_a: bool, stage_b: bool) -> dict:
    return {
        "candidate_id": candidate_id,
        "stage_a_passed": stage_a,
        "stage_b_passed": stage_b,
    }


def _records(**outcomes: tuple[bool, bool]) -> list[dict]:
    return [
        _candidate(candidate_id, *outcomes.get(candidate_id, (False, False)))
        for candidate_id, _ in SCALING_CANDIDATES
    ]


def test_registered_candidates_match_the_preregistered_factorial() -> None:
    assert SCALING_CANDIDATES == (
        ("steps90k", {"hidden_width": 256, "max_steps": 90_000}),
        ("width512", {"hidden_width": 512, "max_steps": 30_000}),
        ("width512_steps90k", {"hidden_width": 512, "max_steps": 90_000}),
    )
    # Candidates change nothing but the two scaled factors relative to the
    # frozen recipe promoted by the saturation gate.
    frozen_without_scaled = {
        name: value
        for name, value in FROZEN_RETRAIN_CONFIG.items()
        if name not in ("hidden_width", "max_steps")
    }
    assert FROZEN_RECIPE == frozen_without_scaled
    for _, overrides in SCALING_CANDIDATES:
        ChunkedCloneConfig(**FROZEN_RECIPE, **overrides)


def test_scaling_status_promotes_first_passing_candidate_in_order() -> None:
    assert resolve_scaling_status(_records()) == "scaling_not_resolved"
    assert (
        resolve_scaling_status(_records(steps90k=(True, True)))
        == "promoted_steps90k_robust"
    )
    assert (
        resolve_scaling_status(_records(steps90k=(True, False)))
        == "promoted_steps90k_starts_only"
    )
    assert (
        resolve_scaling_status(_records(width512=(True, True)))
        == "promoted_width512_robust"
    )
    assert (
        resolve_scaling_status(_records(width512_steps90k=(True, False)))
        == "promoted_width512_steps90k_starts_only"
    )
    # Minimal change wins even when a larger-change candidate also passes.
    assert (
        resolve_scaling_status(
            _records(steps90k=(True, False), width512_steps90k=(True, True))
        )
        == "promoted_steps90k_starts_only"
    )


def test_scaling_status_requires_all_candidates_in_registered_order() -> None:
    complete = _records()
    with pytest.raises(ValueError, match="registered order"):
        resolve_scaling_status(complete[:2])
    with pytest.raises(ValueError, match="registered order"):
        resolve_scaling_status(list(reversed(complete)))


def test_width_512_chunked_model_builds_with_zero_initialized_head() -> None:
    torch.manual_seed(101)
    model = build_chunked_clone_model(10, 512, 90)
    assert model.network[0].out_features == 512
    with torch.inference_mode():
        output = model(torch.randn(4, 10)).numpy()
    assert np.array_equal(output, np.zeros_like(output))
    with pytest.raises(ValueError, match="hidden_width"):
        build_chunked_clone_model(10, 384, 90)
