from __future__ import annotations

from importlib.resources import files

import numpy as np

from so_arm101_v2.data import audit_future_target_candidates, build_split_manifests
from so_arm101_v2.data._serialization import canonical_json_bytes
from so_arm101_v2.data.cli import target_audit_resource_name
from so_arm101_v2.data.inventory import inventory_resource_name
from so_arm101_v2.data.splits import split_resource_name


def _assert_packaged(name: str, value: object) -> None:
    resource = files("so_arm101_v2.data.resources").joinpath(name)
    assert resource.is_file()
    assert resource.read_bytes() == canonical_json_bytes(value, pretty=True)


def test_verified_physical_dataset_baseline(physical_inventory) -> None:
    inventory = physical_inventory
    assert inventory.total_episodes == 100
    assert inventory.total_frames == 41631
    assert inventory.fps == 30
    assert inventory.episode_length_statistics["min"] == 237
    assert inventory.episode_length_statistics["max"] == 816
    assert inventory.timing["max_abs_error_from_frame_index_over_fps_s"] < 1e-6
    assert inventory.same_frame_invariant["mismatched_rows"] == 0
    assert inventory.same_frame_invariant["max_abs_difference"] == 0
    assert inventory.joints["order"] == [
        "shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex",
        "wrist_roll", "gripper",
    ]
    assert all(item["decoded_frames"] == item["matching_parquet_rows"] for item in inventory.camera["files"])
    assert sum(item["decoded_frames"] for item in inventory.camera["files"]) == 41631
    assert all(episode.outcome_label is None for episode in inventory.episodes)
    assert all(episode.capture_timestamp is None for episode in inventory.episodes)
    assert np.array_equal(inventory.states, inventory.states.astype(np.float32))


def test_real_splits_and_candidate_counts(physical_inventory) -> None:
    manifests = build_split_manifests(physical_inventory)
    assert {name: manifest.frame_count for name, manifest in manifests.items()} == {
        "train": 33365,
        "validation": 3882,
        "test": 4384,
    }
    audit = audit_future_target_candidates(physical_inventory, manifests)
    expected = {
        "1": (41531, {"train": 33285, "validation": 3872, "test": 4374}),
        "3": (41331, {"train": 33125, "validation": 3852, "test": 4354}),
        "5": (41131, {"train": 32965, "validation": 3832, "test": 4334}),
    }
    for lead, (full_count, split_counts) in expected.items():
        assert audit["leads"][lead]["full_dataset"]["retained_samples"] == full_count
        for split, count in split_counts.items():
            assert audit["leads"][lead]["splits"][split]["retained_samples"] == count


def test_canonical_resources_regenerate_exactly(physical_inventory) -> None:
    inventory = physical_inventory
    manifests = build_split_manifests(inventory)
    audit = audit_future_target_candidates(inventory, manifests)
    _assert_packaged(inventory_resource_name(inventory), inventory.to_dict())
    for manifest in manifests.values():
        _assert_packaged(split_resource_name(manifest), manifest.to_dict())
    _assert_packaged(
        target_audit_resource_name(inventory.dataset_id, inventory.dataset_digest), audit
    )
