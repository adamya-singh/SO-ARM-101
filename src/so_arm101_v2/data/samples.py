"""Traceable physical-dataset sample loading and model preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from .inventory import DatasetInventory


def _readonly(values: Any, dtype: Any) -> NDArray[np.generic]:
    result = np.array(values, dtype=dtype, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True, order=True)
class SampleReference:
    episode_id: int
    frame_index: int
    lead_steps: int

    def __post_init__(self) -> None:
        for name in ("episode_id", "frame_index", "lead_steps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if self.episode_id < 0 or self.frame_index < 0:
            raise ValueError("episode_id and frame_index must be nonnegative")
        if self.lead_steps not in (1, 3, 5):
            raise ValueError("lead_steps must be one of 1, 3, or 5")


@dataclass(frozen=True)
class FutureStateSample:
    reference: SampleReference
    raw_image: NDArray[np.uint8]
    model_image: NDArray[np.float32]
    current_state: NDArray[np.float32]
    future_target: NDArray[np.float32]
    observed_states: NDArray[np.float32]
    observed_timestamps: NDArray[np.float64]
    source_data_file: str
    source_video_file: str
    source_video_frame: int
    dataset_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_image", _readonly(self.raw_image, np.uint8))
        object.__setattr__(self, "model_image", _readonly(self.model_image, np.float32))
        object.__setattr__(self, "current_state", _readonly(self.current_state, np.float32))
        object.__setattr__(self, "future_target", _readonly(self.future_target, np.float32))
        object.__setattr__(self, "observed_states", _readonly(self.observed_states, np.float32))
        object.__setattr__(
            self, "observed_timestamps", _readonly(self.observed_timestamps, np.float64)
        )
        if self.raw_image.shape != (256, 256, 3) or self.raw_image.dtype != np.uint8:
            raise ValueError("raw_image must be RGB uint8 with shape (256, 256, 3)")
        if self.model_image.shape != (3, 256, 256) or self.model_image.dtype != np.float32:
            raise ValueError("model_image must be float32 with shape (3, 256, 256)")
        expected = self.reference.lead_steps + 1
        if self.observed_states.shape != (expected, 6):
            raise ValueError("observed_states do not cover current through target frames")
        if self.observed_timestamps.shape != (expected,):
            raise ValueError("observed_timestamps do not align with observed_states")
        if not np.array_equal(self.current_state, self.observed_states[0]):
            raise ValueError("current_state differs from the first observed state")
        if not np.array_equal(self.future_target, self.observed_states[-1]):
            raise ValueError("future_target differs from the last observed state")


def preprocess_wrist_image(rgb: Any) -> NDArray[np.float32]:
    """Convert exact dataset RGB uint8 HWC to float32 CHW in [0, 1]."""
    image = np.asarray(rgb)
    if image.shape != (256, 256, 3) or image.dtype != np.uint8:
        raise ValueError("wrist image must be RGB uint8 with shape (256, 256, 3)")
    result = np.transpose(image.astype(np.float32) / np.float32(255.0), (2, 0, 1))
    return _readonly(result, np.float32)


def _optional_dependencies() -> tuple[Any, Any]:
    try:
        import av
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("sample loading requires the 'data' extra") from exc
    return av, pq


def _episode_video_offsets(root: Path, pq: Any, fps: float) -> dict[int, tuple[str, int]]:
    result: dict[int, tuple[str, int]] = {}
    columns = [
        "episode_index",
        "videos/observation.images.wrist/chunk_index",
        "videos/observation.images.wrist/file_index",
        "videos/observation.images.wrist/from_timestamp",
    ]
    for path in sorted(root.glob("meta/episodes/**/*.parquet")):
        for row in pq.read_table(path, columns=columns).to_pylist():
            video = (
                "videos/observation.images.wrist/"
                f"chunk-{int(row['videos/observation.images.wrist/chunk_index']):03d}/"
                f"file-{int(row['videos/observation.images.wrist/file_index']):03d}.mp4"
            )
            start_frame = int(round(
                float(row["videos/observation.images.wrist/from_timestamp"]) * fps
            ))
            result[int(row["episode_index"])] = (video, start_frame)
    return result


def _decode_requested_frames(
    root: Path, av: Any, requests: dict[str, set[int]]
) -> dict[tuple[str, int], NDArray[np.uint8]]:
    decoded: dict[tuple[str, int], NDArray[np.uint8]] = {}
    for relative, indices in requests.items():
        remaining = set(indices)
        maximum = max(remaining)
        with av.open(str(root / relative)) as container:
            for index, frame in enumerate(container.decode(video=0)):
                if index in remaining:
                    decoded[(relative, index)] = frame.to_ndarray(format="rgb24")
                    remaining.remove(index)
                    if not remaining:
                        break
                if index > maximum:
                    break
        if remaining:
            raise ValueError(f"video {relative} is missing requested frames {sorted(remaining)}")
    return decoded


def load_future_state_samples(
    dataset_root: str | Path,
    references: Iterable[SampleReference],
    *,
    inventory: "DatasetInventory | None" = None,
) -> tuple[FutureStateSample, ...]:
    """Load exact episode-local frames and future-state evidence in input order."""
    from .inventory import inventory_physical_dataset

    refs = tuple(references)
    if not refs:
        return ()
    if any(not isinstance(ref, SampleReference) for ref in refs):
        raise TypeError("references must contain SampleReference objects")
    root = Path(dataset_root).resolve()
    inventory = inventory or inventory_physical_dataset(root)
    av, pq = _optional_dependencies()
    records = {record.episode_id: record for record in inventory.episodes}
    offsets = _episode_video_offsets(root, pq, inventory.fps)

    image_requests: dict[str, set[int]] = {}
    resolved: list[tuple[SampleReference, Any, str, int, int]] = []
    for ref in refs:
        if ref.episode_id not in records:
            raise ValueError(f"unknown episode {ref.episode_id}")
        record = records[ref.episode_id]
        target_frame = ref.frame_index + ref.lead_steps
        if target_frame >= record.frame_count:
            raise ValueError(
                f"sample {ref} crosses episode end at frame {record.frame_count - 1}"
            )
        video_file, episode_video_start = offsets[ref.episode_id]
        source_video_frame = episode_video_start + ref.frame_index
        image_requests.setdefault(video_file, set()).add(source_video_frame)
        global_row = record.global_from_index + ref.frame_index
        resolved.append((ref, record, video_file, source_video_frame, global_row))

    images = _decode_requested_frames(root, av, image_requests)
    samples = []
    for ref, record, video_file, source_video_frame, global_row in resolved:
        stop = global_row + ref.lead_steps + 1
        states = inventory.states[global_row:stop]
        timestamps = inventory.timestamps[global_row:stop]
        if not np.all(inventory.episode_indices[global_row:stop] == ref.episode_id):
            raise ValueError(f"sample {ref} crosses an episode boundary")
        raw = images[(video_file, source_video_frame)]
        samples.append(FutureStateSample(
            reference=ref,
            raw_image=raw,
            model_image=preprocess_wrist_image(raw),
            current_state=states[0],
            future_target=states[-1],
            observed_states=states,
            observed_timestamps=timestamps,
            source_data_file=record.source_data_file,
            source_video_file=video_file,
            source_video_frame=source_video_frame,
            dataset_digest=inventory.dataset_digest,
        ))
    return tuple(samples)


__all__ = [
    "FutureStateSample",
    "SampleReference",
    "load_future_state_samples",
    "preprocess_wrist_image",
]
