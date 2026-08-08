"""Gripper-clamp gate: legality pins, policy variant, and resolver."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from so_arm101_v2.contracts import evaluate_physical_command
from so_arm101_v2.contracts.coordinates import effective_safe_act_bounds
from so_arm101_v2.learning.chunked import ChunkedCloneConfig, train_chunked_clone
from so_arm101_v2.simulation.clamp import (
    CLAMP_CHANNELS,
    PROMOTION_RULE,
    resolve_clamp_status,
)
from so_arm101_v2.simulation.policy_specs import PolicySpec, build_policy

from test_chunked_promotion import _write_tiny_oracle_manifest
from test_precision_gate import _cell


def test_gripper_exact_bound_clamp_is_mask_legal() -> None:
    """The pre-registered legality proof: requesting exactly the gripper's
    effective bounds produces no clip masks (strict-inequality tests)."""
    low, high = effective_safe_act_bounds()
    current = np.zeros(6, dtype=np.float32)
    current[5] = 0.5
    for bound in (low[5], high[5]):
        target = current.copy()
        target[5] = bound
        evaluation = evaluate_physical_command(current, target)
        assert not evaluation.act_clip_mask.any()
        assert not evaluation.mujoco_clip_mask.any()
        assert not evaluation.physical_clip_mask.any()
    # And the failure regime being targeted: a sub-floor request clips.
    target = current.copy()
    target[5] = low[5] - 0.002
    evaluation = evaluate_physical_command(current, target)
    assert evaluation.mujoco_clip_mask[5] or evaluation.act_clip_mask[5]


def test_clamp_variant_policy_id_and_output(tmp_path: Path) -> None:
    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
        numerics=None,
    )
    from so_arm101_v2.simulation.chunked import ChunkedClonePolicy

    plain = ChunkedClonePolicy(result.checkpoint)
    clamped = ChunkedClonePolicy(result.checkpoint, clamp_channels=CLAMP_CHANNELS)
    assert plain.policy_id == "chunked_h2.seed101"
    assert clamped.policy_id == "chunked_h2.seed101.gripper_clamp_v1"
    with pytest.raises(ValueError, match="joint indices"):
        ChunkedClonePolicy(result.checkpoint, clamp_channels=(6,))
    # The clamp itself: force a sub-floor command through the buffer.
    low, high = effective_safe_act_bounds()
    clamped._buffer = [np.asarray([0, 0, 0, 0, 0, low[5] - 0.01], dtype=np.float32)]
    command = clamped.predict(np.empty(0, dtype=np.uint8), np.zeros(6, dtype=np.float32))
    assert command[5] == np.float32(low[5])
    plain._buffer = [np.asarray([0, 0, 0, 0, 0, low[5] - 0.01], dtype=np.float32)]
    unclamped = plain.predict(np.empty(0, dtype=np.uint8), np.zeros(6, dtype=np.float32))
    assert unclamped[5] < low[5]


def test_clamp_spec_option_round_trip(tmp_path: Path) -> None:
    import pickle

    manifest_path, _ = _write_tiny_oracle_manifest(tmp_path)
    result = train_chunked_clone(
        manifest_path, tmp_path / "output",
        config=ChunkedCloneConfig(chunk_horizon=2, max_steps=2),
        numerics=None,
    )
    spec = PolicySpec(
        kind="chunked_clone",
        checkpoint=str(result.checkpoint.resolve()),
        options=(("clamp_channels", CLAMP_CHANNELS),),
    )
    assert pickle.loads(pickle.dumps(spec)) == spec
    policy = build_policy(spec)
    assert policy.clamp_channels == CLAMP_CHANNELS
    assert policy.policy_id.endswith(".gripper_clamp_v1")


def test_resolve_clamp_status() -> None:
    assert "clamp_promoted_robust" in PROMOTION_RULE
    all_pass = [_cell(f"seed{seed}", 15, 0) for seed in (101, 202, 303)]
    assert resolve_clamp_status(all_pass) == "clamp_promoted_robust"
    one_fails = [*all_pass[:2], _cell("seed303", 14, 3)]
    assert resolve_clamp_status(one_fails) == "clamp_not_resolved"
    with pytest.raises(ValueError, match="at least one cell"):
        resolve_clamp_status([])
    with pytest.raises(ValueError, match="evaluation evidence"):
        broken = [dict(all_pass[0])]
        broken[0]["stage_a_evaluation_content_sha256"] = None
        resolve_clamp_status(broken)
