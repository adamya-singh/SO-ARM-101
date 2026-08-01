from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from so_arm101_v2.data._serialization import write_immutable_json
from so_arm101_v2.data.inventory import (
    DATASET_ID,
    PROVENANCE_BATCHES,
    EpisodeRecord,
    _file_records,
)
from so_arm101_v2.data.splits import build_split_manifests


EXPECTED_VALIDATION = (13, 14, 22, 36, 45, 60, 69, 72, 82, 84)
EXPECTED_TEST = (10, 12, 40, 41, 46, 54, 57, 61, 74, 85)


def _fake_inventory(lengths: list[int]):
    episodes = []
    start = 0
    for episode_id, length in enumerate(lengths):
        batch = next(
            item for item in PROVENANCE_BATCHES
            if item.episode_first <= episode_id <= item.episode_last
        )
        episodes.append(EpisodeRecord(
            episode_id=episode_id,
            frame_count=length,
            global_from_index=start,
            global_to_index=start + length,
            duration_s=length / 30,
            final_frame_timestamp_s=(length - 1) / 30,
            task="task",
            source_data_file="data.parquet",
            source_video_file="video.mp4",
            provenance_batch=batch.batch_id,
        ))
        start += length
    return SimpleNamespace(
        dataset_id=DATASET_ID,
        dataset_digest="a" * 64,
        episodes=tuple(episodes),
        provenance_batches=PROVENANCE_BATCHES,
    )


def test_fixed_splits_are_disjoint_complete_and_stratified() -> None:
    manifests = build_split_manifests(_fake_inventory([1] * 100))
    train = set(manifests["train"].episode_ids)
    validation = set(manifests["validation"].episode_ids)
    test = set(manifests["test"].episode_ids)

    assert manifests["validation"].episode_ids == EXPECTED_VALIDATION
    assert manifests["test"].episode_ids == EXPECTED_TEST
    assert not (train & validation or train & test or validation & test)
    assert train | validation | test == set(range(100))
    assert dict(manifests["train"].provenance_batch_counts) == {
        "repo_batch_000_016": 13,
        "repo_batch_017_049": 27,
        "repo_batch_050_099": 40,
    }


def _minimal_source_tree(root: Path) -> None:
    relative_paths = [
        "meta/info.json", "meta/stats.json", "meta/tasks.parquet",
        "meta/act_coordinate_contract.json",
        "meta/physical_inference_calibration_20260620.json",
    ]
    relative_paths += [f"meta/episodes/chunk-000/file-{index:03d}.parquet" for index in range(4)]
    relative_paths += [f"data/chunk-000/file-{index:03d}.parquet" for index in range(4)]
    relative_paths += [
        f"videos/observation.images.wrist/chunk-000/file-{index:03d}.mp4"
        for index in range(4)
    ]
    for index, relative in enumerate(relative_paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{index}".encode())


def test_dataset_digest_changes_if_any_source_byte_changes(tmp_path: Path) -> None:
    _minimal_source_tree(tmp_path)
    _, before = _file_records(tmp_path)
    changed = tmp_path / "meta/info.json"
    changed.write_bytes(changed.read_bytes() + b"!")
    _, after = _file_records(tmp_path)
    assert before != after


def test_immutable_json_rejects_conflicting_regeneration(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    write_immutable_json(path, {"version": 1})
    write_immutable_json(path, {"version": 1})
    with pytest.raises(FileExistsError, match="different content"):
        write_immutable_json(path, {"version": 2})
