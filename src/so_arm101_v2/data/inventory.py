"""Strict inventory of the pinned SO-ARM-101 physical dataset."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._serialization import canonical_json_bytes, write_immutable_json
from .resources import read_resource_bytes
from .targets import FutureTargetSpec, build_future_target_index, gather_future_state_targets


DATASET_ID = "so101_pickplace_v1"
INVENTORY_VERSION = "inventory_v1"
TASK_TEXT = "pick up the cube and place it in the box"
TIMESTAMP_TOLERANCE_S = 1e-5
DATASET_JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass(frozen=True)
class FileRecord:
    path: str
    size_bytes: int
    sha256: str
    role: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "sha256": self.sha256,
            "role": self.role,
        }


@dataclass(frozen=True)
class ProvenanceBatch:
    batch_id: str
    episode_first: int
    episode_last: int
    repository_commit: str
    repository_commit_date: str
    provenance_kind: str = "repository_history_not_verified_capture_session"

    def to_dict(self) -> dict[str, object]:
        return {
            "batch_id": self.batch_id,
            "episode_first": self.episode_first,
            "episode_last": self.episode_last,
            "repository_commit": self.repository_commit,
            "repository_commit_date": self.repository_commit_date,
            "provenance_kind": self.provenance_kind,
        }


PROVENANCE_BATCHES = (
    ProvenanceBatch(
        "repo_batch_000_016", 0, 16,
        "c476a93de077b56d52025ee5620e0fb9bf7f74df",
        "2026-01-25T19:39:33-05:00",
    ),
    ProvenanceBatch(
        "repo_batch_017_049", 17, 49,
        "f624432e8cdf6ea0b9db69e5aa0d54613fdf38ba",
        "2026-01-25T19:59:01-05:00",
    ),
    ProvenanceBatch(
        "repo_batch_050_099", 50, 99,
        "0d2fa6ea7f1f2ddef8c60c32ae0937be345d71e1",
        "2026-03-21T20:37:02-04:00",
    ),
)


@dataclass(frozen=True)
class EpisodeRecord:
    episode_id: int
    frame_count: int
    global_from_index: int
    global_to_index: int
    duration_s: float
    final_frame_timestamp_s: float
    task: str
    source_data_file: str
    source_video_file: str
    provenance_batch: str
    capture_timestamp: None = None
    verified_physical_session: None = None
    outcome_label: None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "episode_id": self.episode_id,
            "frame_count": self.frame_count,
            "global_from_index": self.global_from_index,
            "global_to_index": self.global_to_index,
            "duration_s": self.duration_s,
            "final_frame_timestamp_s": self.final_frame_timestamp_s,
            "task": self.task,
            "source_data_file": self.source_data_file,
            "source_video_file": self.source_video_file,
            "provenance_batch": self.provenance_batch,
            "capture_timestamp": self.capture_timestamp,
            "verified_physical_session": self.verified_physical_session,
            "outcome_label": self.outcome_label,
        }


def _empty_i() -> NDArray[np.int64]:
    return np.empty(0, dtype=np.int64)


def _empty_f() -> NDArray[np.float64]:
    return np.empty(0, dtype=np.float64)


def _empty_states() -> NDArray[np.float32]:
    return np.empty((0, 6), dtype=np.float32)


def _readonly(array: NDArray[np.generic]) -> NDArray[np.generic]:
    result = np.array(array, copy=True, order="C")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class DatasetInventory:
    schema_version: int
    inventory_version: str
    dataset_id: str
    dataset_digest: str
    robot_type: str
    task: str
    fps: float
    total_episodes: int
    total_frames: int
    episode_length_statistics: dict[str, float]
    timing: dict[str, object]
    camera: dict[str, object]
    joints: dict[str, object]
    coordinate_contract: dict[str, object]
    calibration: dict[str, object]
    same_frame_invariant: dict[str, object]
    future_delta_statistics: dict[str, object]
    files: tuple[FileRecord, ...]
    provenance_batches: tuple[ProvenanceBatch, ...]
    episodes: tuple[EpisodeRecord, ...]
    warnings: tuple[str, ...]
    episode_indices: NDArray[np.int64] = field(default_factory=_empty_i, repr=False, compare=False)
    frame_indices: NDArray[np.int64] = field(default_factory=_empty_i, repr=False, compare=False)
    timestamps: NDArray[np.float64] = field(default_factory=_empty_f, repr=False, compare=False)
    states: NDArray[np.float32] = field(default_factory=_empty_states, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("episode_indices", "frame_indices", "timestamps", "states"):
            object.__setattr__(self, name, _readonly(getattr(self, name)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "inventory_version": self.inventory_version,
            "dataset_id": self.dataset_id,
            "dataset_digest": self.dataset_digest,
            "robot_type": self.robot_type,
            "task": self.task,
            "fps": self.fps,
            "total_episodes": self.total_episodes,
            "total_frames": self.total_frames,
            "episode_length_statistics": self.episode_length_statistics,
            "timing": self.timing,
            "camera": self.camera,
            "joints": self.joints,
            "coordinate_contract": self.coordinate_contract,
            "calibration": self.calibration,
            "same_frame_invariant": self.same_frame_invariant,
            "future_delta_statistics": self.future_delta_statistics,
            "files": [record.to_dict() for record in self.files],
            "provenance_batches": [batch.to_dict() for batch in self.provenance_batches],
            "episodes": [episode.to_dict() for episode in self.episodes],
            "warnings": list(self.warnings),
        }


def _optional_dependencies() -> tuple[Any, Any]:
    try:
        import av
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised without the data extra
        raise RuntimeError(
            "physical dataset inventory requires the 'data' extra: "
            "python -m pip install -e '.[data]'"
        ) from exc
    return av, pq


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_role(path: Path) -> str:
    text = path.as_posix()
    if text.startswith("data/"):
        return "frame_data"
    if text.startswith("videos/"):
        return "wrist_video"
    if text.startswith("meta/episodes/"):
        return "episode_metadata"
    return {
        "meta/info.json": "dataset_metadata",
        "meta/stats.json": "dataset_statistics",
        "meta/tasks.parquet": "task_metadata",
        "meta/act_coordinate_contract.json": "coordinate_contract",
        "meta/physical_inference_calibration_20260620.json": "physical_calibration",
    }.get(text, "metadata")


def _source_paths(root: Path) -> list[Path]:
    paths = [
        root / "meta/info.json",
        root / "meta/stats.json",
        root / "meta/tasks.parquet",
        root / "meta/act_coordinate_contract.json",
        root / "meta/physical_inference_calibration_20260620.json",
        *sorted(root.glob("meta/episodes/**/*.parquet")),
        *sorted(root.glob("data/**/*.parquet")),
        *sorted(root.glob("videos/**/*.mp4")),
    ]
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"dataset source file missing: {missing[0]}")
    if len(paths) != 17:
        raise ValueError(f"expected 17 version-1 source files, found {len(paths)}")
    return sorted(paths, key=lambda path: path.relative_to(root).as_posix())


def _file_records(root: Path) -> tuple[tuple[FileRecord, ...], str]:
    records = tuple(
        FileRecord(
            path=path.relative_to(root).as_posix(),
            size_bytes=path.stat().st_size,
            sha256=_sha256(path),
            role=_file_role(path.relative_to(root)),
        )
        for path in _source_paths(root)
    )
    digest = hashlib.sha256(
        canonical_json_bytes([record.to_dict() for record in records])
    ).hexdigest()
    return records, digest


def _fixed_matrix(column: Any, name: str) -> NDArray[np.float32]:
    combined = column.combine_chunks()
    if combined.type.list_size != 6 or str(combined.type.value_type) != "float":
        raise ValueError(f"{name} must be fixed_size_list<float32>[6]")
    values = combined.values.to_numpy(zero_copy_only=False).reshape(-1, 6)
    result = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains nonfinite values")
    return result


def _provenance_batch(episode_id: int) -> str:
    for batch in PROVENANCE_BATCHES:
        if batch.episode_first <= episode_id <= batch.episode_last:
            return batch.batch_id
    raise ValueError(f"episode {episode_id} is outside the provenance batches")


def _scalar_stats(values: NDArray[np.float64]) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
    }


def _future_delta_statistics(
    episodes: NDArray[np.int64], frames: NDArray[np.int64],
    timestamps: NDArray[np.float64], states: NDArray[np.float32], fps: float,
) -> dict[str, object]:
    result: dict[str, object] = {}
    for lead in (1, 3, 5):
        index = build_future_target_index(
            episodes, frames, timestamps, FutureTargetSpec(lead), fps=fps
        )
        target = gather_future_state_targets(states, index)
        deltas = target.astype(np.float64) - states[index.anchor_indices].astype(np.float64)
        l2 = np.linalg.norm(deltas, axis=1)
        result[str(lead)] = {
            "retained_samples": index.retained_count,
            "dropped_terminal_anchors": index.dropped_count,
            "per_joint_abs_delta_mean": [
                float(value) for value in np.mean(np.abs(deltas), axis=0)
            ],
            "l2_delta": {
                **_scalar_stats(l2),
                "p90": float(np.quantile(l2, 0.9)),
                "zero_fraction": float(np.mean(l2 == 0.0)),
            },
        }
    return result


def inventory_resource_name(inventory: DatasetInventory) -> str:
    return (
        f"{inventory.dataset_id}.inventory.v1."
        f"{inventory.dataset_digest[:12]}.json"
    )


def inventory_physical_dataset(
    dataset_root: str | Path, *, output_dir: str | Path | None = None
) -> DatasetInventory:
    """Validate and inventory the pinned physical cube pick-and-place dataset."""
    av, pq = _optional_dependencies()
    root = Path(dataset_root).resolve()
    files, dataset_digest = _file_records(root)
    info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
    statistics = json.loads((root / "meta/stats.json").read_text(encoding="utf-8"))
    fps = float(info["fps"])
    if (
        info.get("total_episodes") != 100
        or info.get("total_frames") != 41631
        or fps != 30.0
    ):
        raise ValueError("dataset metadata does not match the version-1 baseline")
    features = info.get("features", {})
    for key in ("action", "observation.state"):
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [6]:
            raise ValueError(f"invalid {key} metadata")
        if tuple(feature.get("names", ())) != DATASET_JOINT_NAMES:
            raise ValueError(f"invalid {key} joint order")
    camera_feature = features.get("observation.images.wrist", {})
    camera_info = camera_feature.get("info", {})
    if (
        camera_feature.get("dtype") != "video"
        or camera_feature.get("shape") != [256, 256, 3]
        or camera_info.get("video.codec") != "av1"
        or camera_info.get("video.pix_fmt") != "yuv420p"
        or camera_info.get("video.fps") != 30
        or camera_info.get("has_audio") is not False
    ):
        raise ValueError("invalid wrist-camera metadata")
    for key in ("action", "observation.state"):
        if statistics.get(key, {}).get("count") != [41631]:
            raise ValueError(f"invalid {key} statistics count")
    if statistics.get("action") != statistics.get("observation.state"):
        raise ValueError("action and observation.state summary statistics differ")

    packaged_coordinate = read_resource_bytes("act_coordinate_contract.json")
    packaged_calibration = read_resource_bytes("physical_inference_calibration_20260620.json")
    coordinate_path = root / "meta/act_coordinate_contract.json"
    calibration_path = root / "meta/physical_inference_calibration_20260620.json"
    if coordinate_path.read_bytes() != packaged_coordinate:
        raise ValueError("dataset coordinate contract differs from the pinned package resource")
    if calibration_path.read_bytes() != packaged_calibration:
        raise ValueError("dataset calibration differs from the pinned package resource")

    task_table = pq.read_table(root / "meta/tasks.parquet")
    if task_table.num_rows != 1 or task_table["task_index"][0].as_py() != 0:
        raise ValueError("expected exactly task_index 0")
    task_columns = [name for name in task_table.column_names if name != "task_index"]
    if len(task_columns) != 1 or task_table[task_columns[0]][0].as_py() != TASK_TEXT:
        raise ValueError("task metadata does not match the pinned task text")

    episode_tables = [pq.read_table(path) for path in sorted(root.glob("meta/episodes/**/*.parquet"))]
    episode_rows = [
        row
        for table in episode_tables
        for row in table.select([
            "episode_index", "tasks", "length", "data/chunk_index", "data/file_index",
            "dataset_from_index", "dataset_to_index",
            "videos/observation.images.wrist/chunk_index",
            "videos/observation.images.wrist/file_index",
        ]).to_pylist()
    ]
    episode_rows.sort(key=lambda row: row["episode_index"])
    if [row["episode_index"] for row in episode_rows] != list(range(100)):
        raise ValueError("episode metadata must cover episode IDs 0 through 99 exactly")

    all_actions: list[NDArray[np.float32]] = []
    all_states: list[NDArray[np.float32]] = []
    all_episodes: list[NDArray[np.int64]] = []
    all_frames: list[NDArray[np.int64]] = []
    all_times: list[NDArray[np.float64]] = []
    all_indices: list[NDArray[np.int64]] = []
    shard_rows: dict[str, int] = {}
    required_columns = {
        "action", "observation.state", "timestamp", "frame_index", "episode_index",
        "index", "task_index",
    }
    for path in sorted(root.glob("data/**/*.parquet")):
        table = pq.read_table(path)
        if set(table.column_names) != required_columns:
            raise ValueError(f"unexpected parquet schema in {path}")
        expected_scalar_types = {
            "timestamp": "float",
            "frame_index": "int64",
            "episode_index": "int64",
            "index": "int64",
            "task_index": "int64",
        }
        for column, expected_type in expected_scalar_types.items():
            if str(table.schema.field(column).type) != expected_type:
                raise ValueError(
                    f"{column} in {path} must have type {expected_type}"
                )
        actions = _fixed_matrix(table["action"], "action")
        states = _fixed_matrix(table["observation.state"], "observation.state")
        if not np.array_equal(actions, states):
            raise ValueError("action == observation.state invariant is violated")
        episodes = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
        frames = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
        indices = np.asarray(table["index"].to_numpy(), dtype=np.int64)
        task_indices = np.asarray(table["task_index"].to_numpy(), dtype=np.int64)
        timestamps = np.asarray(table["timestamp"].to_numpy(), dtype=np.float64)
        if np.any(task_indices != 0):
            raise ValueError("all data rows must reference task_index 0")
        if not np.all(np.isfinite(timestamps)):
            raise ValueError("timestamps contain nonfinite values")
        all_actions.append(actions)
        all_states.append(states)
        all_episodes.append(episodes)
        all_frames.append(frames)
        all_times.append(timestamps)
        all_indices.append(indices)
        shard_rows[path.relative_to(root).as_posix()] = table.num_rows

    actions = np.concatenate(all_actions)
    states = np.concatenate(all_states)
    episodes = np.concatenate(all_episodes)
    frames = np.concatenate(all_frames)
    timestamps = np.concatenate(all_times)
    indices = np.concatenate(all_indices)
    if len(states) != int(info["total_frames"]):
        raise ValueError("parquet row count does not match dataset metadata")
    if not np.array_equal(indices, np.arange(len(indices), dtype=np.int64)):
        raise ValueError("global index is not contiguous")
    build_future_target_index(
        episodes, frames, timestamps, FutureTargetSpec(1), fps=fps,
        timestamp_tolerance_s=TIMESTAMP_TOLERANCE_S,
    )

    episode_records: list[EpisodeRecord] = []
    for row in episode_rows:
        episode_id = int(row["episode_index"])
        length = int(row["length"])
        start = int(row["dataset_from_index"])
        end = int(row["dataset_to_index"])
        if end - start != length or not np.all(episodes[start:end] == episode_id):
            raise ValueError(f"episode {episode_id} metadata range is inconsistent")
        if row["tasks"] != [TASK_TEXT]:
            raise ValueError(f"episode {episode_id} has unexpected task metadata")
        data_file = (
            f"data/chunk-{int(row['data/chunk_index']):03d}/"
            f"file-{int(row['data/file_index']):03d}.parquet"
        )
        video_file = (
            "videos/observation.images.wrist/"
            f"chunk-{int(row['videos/observation.images.wrist/chunk_index']):03d}/"
            f"file-{int(row['videos/observation.images.wrist/file_index']):03d}.mp4"
        )
        episode_records.append(EpisodeRecord(
            episode_id=episode_id,
            frame_count=length,
            global_from_index=start,
            global_to_index=end,
            duration_s=length / fps,
            final_frame_timestamp_s=float(timestamps[end - 1]),
            task=TASK_TEXT,
            source_data_file=data_file,
            source_video_file=video_file,
            provenance_batch=_provenance_batch(episode_id),
        ))

    video_metadata: list[dict[str, object]] = []
    camera_feature = features["observation.images.wrist"]
    for path in sorted(root.glob("videos/**/*.mp4")):
        relative = path.relative_to(root).as_posix()
        expected_frames = shard_rows[relative.replace(
            "videos/observation.images.wrist", "data"
        ).replace(".mp4", ".parquet")]
        with av.open(str(path)) as container:
            if len(container.streams.video) != 1 or container.streams.audio:
                raise ValueError(f"{relative} must contain one silent video stream")
            stream = container.streams.video[0]
            decoded_frames = sum(1 for _ in container.decode(video=0))
            metadata = {
                "path": relative,
                "width": stream.codec_context.width,
                "height": stream.codec_context.height,
                "codec": camera_feature["info"]["video.codec"],
                "decoder": stream.codec_context.name,
                "pixel_format": stream.codec_context.format.name,
                "fps": float(stream.average_rate),
                "container_reported_frames": int(stream.frames),
                "decoded_frames": decoded_frames,
                "matching_parquet_rows": expected_frames,
            }
        if (
            metadata["width"] != 256 or metadata["height"] != 256
            or metadata["pixel_format"] != "yuv420p" or metadata["fps"] != fps
            or metadata["codec"] != "av1" or decoded_frames != expected_frames
            or int(metadata["container_reported_frames"]) != expected_frames
        ):
            raise ValueError(f"video metadata or frame count mismatch: {relative}")
        video_metadata.append(metadata)

    lengths = np.asarray([record.frame_count for record in episode_records], dtype=np.float64)
    within_episode = frames > 0
    timestamp_errors = np.abs(timestamps - frames.astype(np.float64) / fps)
    step_deltas = timestamps[within_episode] - timestamps[np.flatnonzero(within_episode) - 1]
    file_by_role = {record.role: record for record in files if record.role in {
        "coordinate_contract", "physical_calibration"
    }}
    inventory = DatasetInventory(
        schema_version=1,
        inventory_version=INVENTORY_VERSION,
        dataset_id=DATASET_ID,
        dataset_digest=dataset_digest,
        robot_type=str(info["robot_type"]),
        task=TASK_TEXT,
        fps=fps,
        total_episodes=len(episode_records),
        total_frames=len(states),
        episode_length_statistics=_scalar_stats(lengths),
        timing={
            "timestamp_dtype": "float32",
            "expected_period_s": 1.0 / fps,
            "timestamp_tolerance_s": TIMESTAMP_TOLERANCE_S,
            "max_abs_error_from_frame_index_over_fps_s": float(np.max(timestamp_errors)),
            "within_episode_step_s": _scalar_stats(step_deltas),
            "frame_indices_contiguous_from_zero": True,
            "global_indices_contiguous_from_zero": True,
            "timestamps_strictly_increasing_within_episode": True,
        },
        camera={
            "key": "observation.images.wrist",
            "shape": [256, 256, 3],
            "dtype": "video_decoded_as_rgb_uint8",
            "encoded_codec": "av1",
            "encoded_pixel_format": "yuv420p",
            "fps": fps,
            "has_audio": False,
            "files": video_metadata,
        },
        joints={
            "order": list(DATASET_JOINT_NAMES),
            "count": len(DATASET_JOINT_NAMES),
            "coordinate_space": "ACT_dataset_coordinates",
            "state_dtype": "float32",
            "action_dtype": "float32",
        },
        coordinate_contract={
            "path": file_by_role["coordinate_contract"].path,
            "sha256": file_by_role["coordinate_contract"].sha256,
            "matches_packaged_resource": True,
        },
        calibration={
            "path": file_by_role["physical_calibration"].path,
            "sha256": file_by_role["physical_calibration"].sha256,
            "matches_packaged_resource": True,
            "capture_time_calibration_record_available": False,
        },
        same_frame_invariant={
            "expression": "action[t] == observation.state[t]",
            "comparison": "exact_float32_elementwise",
            "rows_checked": len(states),
            "mismatched_rows": 0,
            "max_abs_difference": float(np.max(np.abs(actions - states))),
            "future_target_rule": "observation.state[t + lead]",
            "action_used_for_future_targets": False,
        },
        future_delta_statistics=_future_delta_statistics(
            episodes, frames, timestamps, states, fps
        ),
        files=files,
        provenance_batches=PROVENANCE_BATCHES,
        episodes=tuple(episode_records),
        warnings=(
            "Physical capture timestamps are absent; repository commit dates are not capture timestamps.",
            "Verified physical collection-session identifiers are absent; provenance batches describe repository history only.",
            "Episode outcome labels are absent and remain null.",
            "The calibration used at recording time is not recorded; the pinned inference calibration is provenance, not proof of capture calibration.",
            "The dataset task includes placement in a box, while fixed_cube_pickup_v1 evaluates pickup only.",
        ),
        episode_indices=episodes,
        frame_indices=frames,
        timestamps=timestamps,
        states=states,
    )
    if output_dir is not None:
        write_immutable_json(
            Path(output_dir) / inventory_resource_name(inventory), inventory.to_dict()
        )
    return inventory


__all__ = [
    "DATASET_ID",
    "DatasetInventory",
    "EpisodeRecord",
    "FileRecord",
    "INVENTORY_VERSION",
    "PROVENANCE_BATCHES",
    "ProvenanceBatch",
    "inventory_physical_dataset",
    "inventory_resource_name",
]
