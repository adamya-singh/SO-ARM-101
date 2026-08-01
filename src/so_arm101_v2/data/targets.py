"""Explicit episode-local future-state target construction."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Mapping

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .inventory import DatasetInventory
    from .splits import SplitManifest


class PaddingPolicy(str, Enum):
    """Supported terminal behavior for future targets."""

    DROP = "drop"


@dataclass(frozen=True)
class FutureTargetSpec:
    """A future absolute-state target definition with an explicit lead."""

    lead_steps: int
    padding: PaddingPolicy = PaddingPolicy.DROP
    source_key: str = "observation.state"
    representation: str = "absolute_act_coordinates"

    def __post_init__(self) -> None:
        if isinstance(self.lead_steps, bool) or not isinstance(self.lead_steps, int):
            raise TypeError("lead_steps must be a positive integer")
        if self.lead_steps <= 0:
            raise ValueError("lead_steps must be a positive integer")
        if self.padding is not PaddingPolicy.DROP:
            raise ValueError("only PaddingPolicy.DROP is supported")
        if self.source_key != "observation.state":
            raise ValueError("future targets must come from observation.state")
        if self.representation != "absolute_act_coordinates":
            raise ValueError("only absolute ACT-coordinate targets are supported")


def _readonly(array: NDArray[np.generic]) -> NDArray[np.generic]:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class FutureTargetIndex:
    """Rows retained by a future-target specification."""

    spec: FutureTargetSpec
    anchor_indices: NDArray[np.int64]
    target_indices: NDArray[np.int64]
    episode_indices: NDArray[np.int64]
    anchor_frame_indices: NDArray[np.int64]
    target_frame_indices: NDArray[np.int64]
    anchor_timestamps: NDArray[np.float64]
    target_timestamps: NDArray[np.float64]
    padding_mask: NDArray[np.bool_]
    total_rows: int
    episode_count: int

    def __post_init__(self) -> None:
        arrays = (
            self.anchor_indices,
            self.target_indices,
            self.episode_indices,
            self.anchor_frame_indices,
            self.target_frame_indices,
            self.anchor_timestamps,
            self.target_timestamps,
            self.padding_mask,
        )
        lengths = {len(value) for value in arrays}
        if len(lengths) != 1:
            raise ValueError("future-target index arrays must have equal lengths")
        names = (
            "anchor_indices",
            "target_indices",
            "episode_indices",
            "anchor_frame_indices",
            "target_frame_indices",
            "anchor_timestamps",
            "target_timestamps",
            "padding_mask",
        )
        for name, value in zip(names, arrays):
            object.__setattr__(self, name, _readonly(value))

    @property
    def retained_count(self) -> int:
        return len(self.anchor_indices)

    @property
    def dropped_count(self) -> int:
        return self.total_rows - self.retained_count


def _require_integer_vector(name: str, value: object) -> NDArray[np.int64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise ValueError(f"{name} must be a one-dimensional integer array")
    return np.asarray(array, dtype=np.int64)


def _require_timestamp_vector(value: object) -> NDArray[np.float64]:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind != "f":
        raise ValueError("timestamps must be a one-dimensional floating-point array")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError("timestamps contain nonfinite values")
    return result


def build_future_target_index(
    episode_indices: object,
    frame_indices: object,
    timestamps: object,
    spec: FutureTargetSpec,
    *,
    fps: float,
    timestamp_tolerance_s: float = 1e-5,
) -> FutureTargetIndex:
    """Build future targets from contiguous, episode-local frame indices."""
    if not isinstance(spec, FutureTargetSpec):
        raise TypeError("spec must be a FutureTargetSpec")
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be positive and finite")
    if not np.isfinite(timestamp_tolerance_s) or timestamp_tolerance_s < 0:
        raise ValueError("timestamp_tolerance_s must be finite and nonnegative")

    episodes = _require_integer_vector("episode_indices", episode_indices)
    frames = _require_integer_vector("frame_indices", frame_indices)
    times = _require_timestamp_vector(timestamps)
    if not (len(episodes) == len(frames) == len(times)):
        raise ValueError("episode_indices, frame_indices, and timestamps must align")
    if len(episodes) == 0:
        empty_i = _readonly(np.empty(0, dtype=np.int64))
        empty_f = _readonly(np.empty(0, dtype=np.float64))
        empty_b = _readonly(np.empty(0, dtype=np.bool_))
        return FutureTargetIndex(
            spec, empty_i, empty_i, empty_i, empty_i, empty_i, empty_f, empty_f,
            empty_b, 0, 0,
        )

    starts = np.r_[0, np.flatnonzero(episodes[1:] != episodes[:-1]) + 1]
    ends = np.r_[starts[1:], len(episodes)]
    if len(np.unique(episodes)) != len(starts):
        raise ValueError("each episode must occupy exactly one contiguous row run")

    anchors: list[NDArray[np.int64]] = []
    targets: list[NDArray[np.int64]] = []
    for start, end in zip(starts, ends):
        run_frames = frames[start:end]
        run_times = times[start:end]
        expected_frames = np.arange(end - start, dtype=np.int64)
        if not np.array_equal(run_frames, expected_frames):
            raise ValueError(
                f"episode {episodes[start]} frame indices must be contiguous from zero"
            )
        expected_times = run_frames.astype(np.float64) / float(fps)
        if np.max(np.abs(run_times - expected_times), initial=0.0) > timestamp_tolerance_s:
            raise ValueError(
                f"episode {episodes[start]} timestamps do not match frame_index / fps"
            )
        if len(run_times) > 1 and np.any(np.diff(run_times) <= 0):
            raise ValueError(f"episode {episodes[start]} timestamps are not increasing")
        retained = max(0, (end - start) - spec.lead_steps)
        if retained:
            anchor = np.arange(start, start + retained, dtype=np.int64)
            anchors.append(anchor)
            targets.append(anchor + spec.lead_steps)

    anchor_rows = np.concatenate(anchors) if anchors else np.empty(0, dtype=np.int64)
    target_rows = np.concatenate(targets) if targets else np.empty(0, dtype=np.int64)
    return FutureTargetIndex(
        spec=spec,
        anchor_indices=_readonly(anchor_rows),
        target_indices=_readonly(target_rows),
        episode_indices=_readonly(episodes[anchor_rows]),
        anchor_frame_indices=_readonly(frames[anchor_rows]),
        target_frame_indices=_readonly(frames[target_rows]),
        anchor_timestamps=_readonly(times[anchor_rows]),
        target_timestamps=_readonly(times[target_rows]),
        padding_mask=_readonly(np.zeros(len(anchor_rows), dtype=np.bool_)),
        total_rows=len(episodes),
        episode_count=len(starts),
    )


def gather_future_state_targets(
    states: object, index: FutureTargetIndex
) -> NDArray[np.float32]:
    """Gather absolute ``observation.state[t + lead]`` values."""
    values = np.asarray(states)
    if values.dtype != np.float32:
        raise ValueError("states must have dtype float32")
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError("states must have shape (N, 6)")
    if values.shape[0] != index.total_rows:
        raise ValueError("states row count does not match the future-target index")
    if not np.all(np.isfinite(values)):
        raise ValueError("states contain nonfinite values")
    return _readonly(np.asarray(values[index.target_indices], dtype=np.float32))


def _stats(values: NDArray[np.float64]) -> dict[str, float]:
    if len(values) == 0:
        return {key: 0.0 for key in ("min", "max", "mean", "median", "p90", "std")}
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.9)),
        "std": float(np.std(values)),
    }


def _audit_subset(
    inventory: "DatasetInventory", lead: int, episode_ids: set[int] | None
) -> dict[str, object]:
    mask = (
        np.ones(inventory.total_frames, dtype=np.bool_)
        if episode_ids is None
        else np.isin(inventory.episode_indices, tuple(sorted(episode_ids)))
    )
    episodes = inventory.episode_indices[mask]
    frames = inventory.frame_indices[mask]
    timestamps = inventory.timestamps[mask]
    states = inventory.states[mask]
    index = build_future_target_index(
        episodes, frames, timestamps, FutureTargetSpec(lead), fps=inventory.fps
    )
    targets = gather_future_state_targets(states, index)
    deltas = targets.astype(np.float64) - states[index.anchor_indices].astype(np.float64)
    l2 = np.linalg.norm(deltas, axis=1)
    timestamp_errors = np.abs(
        (index.target_timestamps - index.anchor_timestamps) - lead / inventory.fps
    )
    return {
        "episodes": index.episode_count,
        "frames": index.total_rows,
        "retained_samples": index.retained_count,
        "dropped_terminal_anchors": index.dropped_count,
        "padding_count": int(np.count_nonzero(index.padding_mask)),
        "max_target_interval_error_s": float(np.max(timestamp_errors, initial=0.0)),
        "per_joint_delta": {
            key: [float(value) for value in operation(deltas, axis=0)]
            for key, operation in (
                ("min", np.min), ("max", np.max), ("mean", np.mean), ("std", np.std)
            )
        } if len(deltas) else {key: [0.0] * 6 for key in ("min", "max", "mean", "std")},
        "l2_delta": _stats(l2),
        "zero_delta_fraction": float(np.mean(l2 == 0.0)) if len(l2) else 0.0,
    }


def audit_future_target_candidates(
    inventory: "DatasetInventory",
    manifests: Mapping[str, "SplitManifest"] | Iterable["SplitManifest"] | None = None,
    *,
    leads: tuple[int, ...] = (1, 3, 5),
) -> dict[str, object]:
    """Audit explicit future-state candidates for the full data and each split."""
    if leads != (1, 3, 5):
        raise ValueError("the version-1 candidate audit is fixed to leads (1, 3, 5)")
    if manifests is None:
        split_items: list[tuple[str, SplitManifest]] = []
    elif isinstance(manifests, Mapping):
        split_items = sorted(manifests.items())
    else:
        split_items = sorted((item.split, item) for item in manifests)
    return {
        "schema_version": 1,
        "dataset_id": inventory.dataset_id,
        "dataset_digest": inventory.dataset_digest,
        "target_source": "observation.state[t + lead]",
        "action_usage": "validation_only_for_action_equals_observation_state",
        "representation": "absolute_act_coordinates",
        "padding_policy": PaddingPolicy.DROP.value,
        "leads": {
            str(lead): {
                "full_dataset": _audit_subset(inventory, lead, None),
                "splits": {
                    name: _audit_subset(inventory, lead, set(manifest.episode_ids))
                    for name, manifest in split_items
                },
            }
            for lead in leads
        },
    }


__all__ = [
    "FutureTargetIndex",
    "FutureTargetSpec",
    "PaddingPolicy",
    "audit_future_target_candidates",
    "build_future_target_index",
    "gather_future_state_targets",
]
