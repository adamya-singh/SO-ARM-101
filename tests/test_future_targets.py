from __future__ import annotations

import numpy as np
import pytest

from so_arm101_v2.data import (
    FutureTargetSpec,
    PaddingPolicy,
    build_future_target_index,
    gather_future_state_targets,
)


def _sequence(lengths: tuple[int, ...]):
    episodes = np.concatenate([
        np.full(length, episode, dtype=np.int64)
        for episode, length in enumerate(lengths)
    ])
    frames = np.concatenate([
        np.arange(length, dtype=np.int64) for length in lengths
    ])
    timestamps = frames.astype(np.float64) / 30.0
    states = np.stack([
        np.arange(len(frames), dtype=np.float32) + joint * 1000
        for joint in range(6)
    ], axis=1)
    return episodes, frames, timestamps, states


@pytest.mark.parametrize("lead", [1, 3, 5])
def test_future_targets_are_exact_future_states_and_drop_tails(lead: int) -> None:
    episodes, frames, timestamps, states = _sequence((8, 7))
    index = build_future_target_index(
        episodes, frames, timestamps, FutureTargetSpec(lead), fps=30
    )
    targets = gather_future_state_targets(states, index)

    np.testing.assert_array_equal(targets, states[index.target_indices])
    np.testing.assert_array_equal(
        index.target_frame_indices, index.anchor_frame_indices + lead
    )
    assert index.retained_count == sum(max(0, length - lead) for length in (8, 7))
    assert index.dropped_count == lead * 2
    assert not index.padding_mask.any()
    assert index.padding_mask.dtype == np.bool_


def test_sentinel_episodes_never_cross_boundaries() -> None:
    episodes, frames, timestamps, states = _sequence((4, 4))
    states[:4] = 10
    states[4:] = 20
    index = build_future_target_index(
        episodes, frames, timestamps, FutureTargetSpec(3), fps=30
    )
    targets = gather_future_state_targets(states, index)

    np.testing.assert_array_equal(index.anchor_indices, [0, 4])
    np.testing.assert_array_equal(index.target_indices, [3, 7])
    np.testing.assert_array_equal(targets[:, 0], [10, 20])
    np.testing.assert_array_equal(
        episodes[index.anchor_indices], episodes[index.target_indices]
    )


def test_short_episodes_produce_zero_targets_safely() -> None:
    episodes, frames, timestamps, states = _sequence((1, 2))
    index = build_future_target_index(
        episodes, frames, timestamps, FutureTargetSpec(3), fps=30
    )
    targets = gather_future_state_targets(states, index)
    assert index.retained_count == 0
    assert index.dropped_count == 3
    assert targets.shape == (0, 6)
    assert targets.dtype == np.float32


@pytest.mark.parametrize("lead", [0, -1])
def test_lead_must_be_positive(lead: int) -> None:
    with pytest.raises(ValueError, match="positive"):
        FutureTargetSpec(lead)


def test_lead_must_be_an_integer() -> None:
    with pytest.raises(TypeError, match="integer"):
        FutureTargetSpec(1.0)  # type: ignore[arg-type]
    assert FutureTargetSpec(1).padding is PaddingPolicy.DROP


@pytest.mark.parametrize(
    ("frames", "timestamps", "message"),
    [
        (np.array([0, 2], dtype=np.int64), np.array([0.0, 2 / 30]), "contiguous"),
        (np.array([0, 0], dtype=np.int64), np.array([0.0, 0.0]), "contiguous"),
        (np.array([0, 1], dtype=np.int64), np.array([0.0, 0.04]), "frame_index / fps"),
    ],
)
def test_invalid_episode_timing_fails(frames, timestamps, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_future_target_index(
            np.array([0, 0], dtype=np.int64),
            frames,
            timestamps.astype(np.float64),
            FutureTargetSpec(1),
            fps=30,
        )


def test_noncontiguous_episode_runs_fail() -> None:
    with pytest.raises(ValueError, match="one contiguous row run"):
        build_future_target_index(
            np.array([0, 1, 0], dtype=np.int64),
            np.array([0, 0, 1], dtype=np.int64),
            np.array([0.0, 0.0, 1 / 30], dtype=np.float64),
            FutureTargetSpec(1),
            fps=30,
        )


def test_index_and_state_validation_is_strict() -> None:
    with pytest.raises(ValueError, match="integer array"):
        build_future_target_index(
            np.array([0.0]), np.array([0]), np.array([0.0]),
            FutureTargetSpec(1), fps=30,
        )
    episodes, frames, timestamps, states = _sequence((3,))
    index = build_future_target_index(
        episodes, frames, timestamps, FutureTargetSpec(1), fps=30
    )
    with pytest.raises(ValueError, match="dtype float32"):
        gather_future_state_targets(states.astype(np.float64), index)
    with pytest.raises(ValueError, match="shape"):
        gather_future_state_targets(states[:, :5], index)
    states[0, 0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        gather_future_state_targets(states, index)
